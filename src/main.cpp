// ---------------------------------------------------------------------
// main.cpp -- Entry point for the spectral photon + path tracer
// ---------------------------------------------------------------------
//
// Architecture (OptiX mandatory):
//   1. Load scene (OBJ + MTL)
//   2. OptiX init -> build_accel -> upload_scene_data -> upload_emitter_data
//   3. GPU photon trace (OptiX __raygen__photon_trace)
//   4. Interactive debug window (first-hit rendering via OptiX)
//   5. Press "R" to launch full path tracing -> PNG output
//
// Usage:
//   photon_tracer [scene.obj] [options]
//
// Options:
//   --width W          Image width  (default from config.h)
//   --height H         Image height (default from config.h)
//   --spp N            Samples per pixel (default from config.h)
//   --photons N        Number of photons (default from config.h)
//   --radius R         Gather radius (default from config.h)
//   --output FILE      Output file (default output/render.png)
//   --mode MODE        Render mode: full|direct|indirect|photon|normals|
//                                   material|depth
// ---------------------------------------------------------------------

#include "core/config.h"
#include "scene/obj_loader.h"
#include "scene/scene.h"
#include "renderer/renderer.h"
#include "renderer/camera.h"
#include "debug/debug.h"
#include "optix/optix_renderer.h"

#include <GLFW/glfw3.h>

// stb_image_write for PNG output
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4996) // sprintf deprecation in MSVC
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif

// stb_easy_font for debug overlay text
#include "stb_easy_font.h"

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <cstdio>
#include <iomanip>

namespace fs = std::filesystem;

// -- Image output -----------------------------------------------------

static bool write_png(const std::string& filename, const FrameBuffer& fb) {
    fs::path p(filename);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }

    // stbi_write_png expects rows top-to-bottom; flip vertically
    std::vector<uint8_t> flipped(fb.width * fb.height * 4);
    for (int y = 0; y < fb.height; ++y) {
        int src_y = fb.height - 1 - y;
        memcpy(&flipped[y * fb.width * 4],
               &fb.srgb[src_y * fb.width * 4],
               fb.width * 4);
    }

    int ok = stbi_write_png(filename.c_str(), fb.width, fb.height, 4,
                             flipped.data(), fb.width * 4);
    if (ok) {
        std::cout << "[Output] Wrote " << filename << " ("
                  << fb.width << "x" << fb.height << ")\n";
    } else {
        std::cerr << "[Output] Failed to write " << filename << "\n";
    }
    return ok != 0;
}

// -- Debug overlay (stb_easy_font) ------------------------------------

static void draw_overlay_text(float x, float y, const char* text,
                               float r, float g, float b, float a) {
    // stb_easy_font outputs quads as 4 vertices each (x,y,z,color)
    // Each vertex is 16 bytes: float x, float y, float z, uint8[4] color
    static char vertex_buffer[1024 * 64]; // 64 KB — enough for many lines
    unsigned char color[4] = {
        (unsigned char)(r * 255), (unsigned char)(g * 255),
        (unsigned char)(b * 255), (unsigned char)(a * 255)
    };
    int num_quads = stb_easy_font_print(x, y, const_cast<char*>(text),
                                         color, vertex_buffer,
                                         sizeof(vertex_buffer));
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 16, vertex_buffer);
    glDrawArrays(GL_QUADS, 0, num_quads * 4);
    glDisableClientState(GL_VERTEX_ARRAY);
}

static void render_help_overlay(int win_w, int win_h,
                                 const DebugState& debug) {
    // Set up 2D orthographic projection for overlay
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, win_w, win_h, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Semi-transparent background box
    float pad   = 10.f;
    float box_w = 280.f;
    float box_h = 290.f;
    float bx    = pad;
    float by    = pad;

    glColor4f(0.0f, 0.0f, 0.0f, 0.6f);
    glBegin(GL_QUADS);
    glVertex2f(bx,         by);
    glVertex2f(bx + box_w, by);
    glVertex2f(bx + box_w, by + box_h);
    glVertex2f(bx,         by + box_h);
    glEnd();

    // Build overlay text lines
    auto on_off = [](bool v) -> const char* { return v ? "ON " : "off"; };

    // Scale text 2x for readability
    float scale = 2.f;
    float tx = bx + 12.f;
    float ty = by + 12.f;
    float line_h = 14.f; // stb_easy_font is ~7px tall, scaled 2x = 14

    glPushMatrix();
    glTranslatef(tx, ty, 0.f);
    glScalef(scale, scale, 1.f);

    float ly = 0.f;
    // Title
    draw_overlay_text(0, ly, "=== Debug Controls ===", 1.f, 1.f, 0.3f, 1.f);
    ly += line_h;

    // Navigation
    draw_overlay_text(0, ly, "WASD   Move camera", 0.8f, 0.8f, 0.8f, 1.f);
    ly += line_h * 0.7f;
    draw_overlay_text(0, ly, "Mouse  Look around", 0.8f, 0.8f, 0.8f, 1.f);
    ly += line_h * 0.7f;
    draw_overlay_text(0, ly, "Shift  Fast move (3x)", 0.8f, 0.8f, 0.8f, 1.f);
    ly += line_h * 0.7f;
    draw_overlay_text(0, ly, "M      Toggle mouse capture", 0.8f, 0.8f, 0.8f, 1.f);
    ly += line_h * 0.7f;
    draw_overlay_text(0, ly, "R      Full path trace render", 0.8f, 0.8f, 0.8f, 1.f);
    ly += line_h * 0.7f;
    draw_overlay_text(0, ly, "ESC/Q  Quit", 0.8f, 0.8f, 0.8f, 1.f);
    ly += line_h;

    // Render modes
    char buf[128];
    snprintf(buf, sizeof(buf), "TAB  Mode: %s",
             DebugState::render_mode_name(debug.current_mode));
    draw_overlay_text(0, ly, buf, 0.3f, 1.f, 0.5f, 1.f);
    ly += line_h;

    // Debug toggles
    draw_overlay_text(0, ly, "--- Debug Toggles ---", 1.f, 1.f, 0.3f, 1.f);
    ly += line_h;

    struct Toggle { const char* key; const char* label; bool on; };
    Toggle toggles[] = {
        {"F1", "Photon points",    debug.show_photon_points},
        {"F2", "Global map",       debug.show_global_map},
        {"F3", "Caustic map",      debug.show_caustic_map},
        {"F4", "Hash grid",        debug.show_hash_grid},
        {"F5", "Photon dirs",      debug.show_photon_dirs},
        {"F6", "PDFs",             debug.show_pdfs},
        {"F7", "Radius sphere",    debug.show_radius_sphere},
        {"F8", "MIS weights",      debug.show_mis_weights},
        {"F9", "Spectral color",   debug.spectral_coloring},
    };
    for (auto& t : toggles) {
        snprintf(buf, sizeof(buf), "%s  %s [%s]", t.key, t.label, on_off(t.on));
        float cr = t.on ? 0.3f : 0.5f;
        float cg = t.on ? 1.0f : 0.5f;
        float cb = t.on ? 0.3f : 0.5f;
        draw_overlay_text(0, ly, buf, cr, cg, cb, 1.f);
        ly += line_h * 0.7f;
    }

    ly += line_h * 0.3f;
    draw_overlay_text(0, ly, "H  Toggle this overlay", 0.6f, 0.6f, 0.6f, 1.f);

    glPopMatrix();

    // Restore GL state
    glDisable(GL_BLEND);
    glEnable(GL_TEXTURE_2D);
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

// -- Parse command line -----------------------------------------------

struct Options {
    std::string scene_file;
    std::string output_file = "output/render.png";
    RenderConfig config;
};

static Options parse_args(int argc, char* argv[]) {
    Options opt;

    opt.scene_file = std::string(SCENES_DIR) + "/" + SCENE_OBJ_PATH;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--width" && i + 1 < argc) {
            opt.config.image_width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            opt.config.image_height = std::stoi(argv[++i]);
        } else if (arg == "--spp" && i + 1 < argc) {
            opt.config.samples_per_pixel = std::stoi(argv[++i]);
        } else if (arg == "--photons" && i + 1 < argc) {
            opt.config.num_photons = std::stoi(argv[++i]);
        } else if (arg == "--radius" && i + 1 < argc) {
            opt.config.gather_radius = std::stof(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            opt.output_file = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            std::string mode = argv[++i];
            if      (mode == "full")     opt.config.mode = RenderMode::Full;
            else if (mode == "direct")   opt.config.mode = RenderMode::DirectOnly;
            else if (mode == "indirect") opt.config.mode = RenderMode::IndirectOnly;
            else if (mode == "photon")   opt.config.mode = RenderMode::PhotonMap;
            else if (mode == "normals")  opt.config.mode = RenderMode::Normals;
            else if (mode == "material") opt.config.mode = RenderMode::MaterialID;
            else if (mode == "depth")    opt.config.mode = RenderMode::Depth;
        } else if (arg[0] != '-') {
            opt.scene_file = arg;
        }
    }

    return opt;
}

// -- GLFW callbacks ---------------------------------------------------

struct AppState {
    DebugState debug;
    bool       render_requested = false;  // R key pressed
    bool       rendering        = false;

    // Progressive final render state
    int        render_spp_done  = 0;      // samples completed so far
    int        render_spp_total = 0;      // target spp
    Camera     render_cam;                // frozen camera for render
    std::chrono::high_resolution_clock::time_point render_start;

    // Camera angles (yaw/pitch in radians)
    float      yaw   = 0.f;     // horizontal angle
    float      pitch = 0.f;     // vertical angle
    bool       mouse_captured = true;  // start with mouse captured
    double     last_mx = 0.0, last_my = 0.0;
    bool       first_mouse = true;
    bool       camera_moved = false;  // flag to reset accumulation
};

static AppState g_app;

static void key_callback(GLFWwindow* window, int key,
                          int /*scancode*/, int action, int /*mods*/) {
    if (action != GLFW_PRESS) return;

    if (key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q) {
        if (g_app.mouse_captured) {
            // Release mouse first; second press quits
            g_app.mouse_captured = false;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            return;
        }
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        return;
    }

    // "R" -> start final render
    if (key == GLFW_KEY_R) {
        g_app.render_requested = true;
        return;
    }

    // Left-click or M to toggle mouse capture
    if (key == GLFW_KEY_M) {
        g_app.mouse_captured = !g_app.mouse_captured;
        glfwSetInputMode(window, GLFW_CURSOR,
            g_app.mouse_captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
        g_app.first_mouse = true;
        return;
    }

    handle_debug_key(key, g_app.debug);
}

static void mouse_button_callback(GLFWwindow* window, int button,
                                   int action, int /*mods*/) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS
        && !g_app.mouse_captured) {
        g_app.mouse_captured = true;
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        g_app.first_mouse = true;
    }
}

// -- Interactive OptiX debug window -----------------------------------

static void run_interactive(
    OptixRenderer& optix_renderer,
    Camera& camera,
    const Options& opt,
    const Scene& scene)
{
    if (!glfwInit()) {
        std::cerr << "[GLFW] Failed to initialize\n";
        return;
    }

    int win_w = opt.config.image_width;
    int win_h = opt.config.image_height;

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(
        win_w, win_h,
        "Spectral Photon+Path Tracer [OptiX]", nullptr, nullptr);

    if (!window) {
        std::cerr << "[GLFW] Failed to create window\n";
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    // Start with mouse captured for FPS-style controls
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Compute initial yaw/pitch from camera look direction
    {
        float3 fwd = normalize(camera.look_at - camera.position);
        g_app.yaw   = atan2f(fwd.x, -fwd.z);
        g_app.pitch = asinf(fmaxf(-1.f, fminf(1.f, fwd.y)));
    }

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, win_w, win_h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glEnable(GL_TEXTURE_2D);

    std::cout << "[Window] OptiX debug viewer (first-hit rendering)\n";
    std::cout << "  WASD = move | Mouse = look | M = release/capture mouse\n";
    std::cout << "  ESC/Q = quit | F1-F9 = debug toggles | TAB = cycle mode\n";
    std::cout << "  R = full path tracing render -> " << opt.output_file << "\n";
    std::cout << "  H = toggle help overlay\n";

    FrameBuffer display_fb;
    display_fb.resize(win_w, win_h);

    int frame = 0;
    auto last_time = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Delta time
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        last_time = now;
        dt = fminf(dt, 0.1f); // clamp to avoid huge jumps

        // -- Mouse look -------------------------------------------------
        if (g_app.mouse_captured) {
            double mx, my;
            glfwGetCursorPos(window, &mx, &my);
            if (g_app.first_mouse) {
                g_app.last_mx = mx;
                g_app.last_my = my;
                g_app.first_mouse = false;
            }
            float dx = (float)(mx - g_app.last_mx);
            float dy = (float)(my - g_app.last_my);
            g_app.last_mx = mx;
            g_app.last_my = my;

            if (dx != 0.f || dy != 0.f) {
                g_app.yaw   += dx * DEFAULT_CAM_MOUSE_SENS;
                g_app.pitch -= dy * DEFAULT_CAM_MOUSE_SENS; // inverted Y
                // Clamp pitch to avoid gimbal lock
                constexpr float MAX_PITCH = 89.f * PI / 180.f;
                g_app.pitch = fmaxf(-MAX_PITCH, fminf(MAX_PITCH, g_app.pitch));
                g_app.camera_moved = true;
            }
        }

        // -- WASD movement ----------------------------------------------
        {
            // Forward direction (camera looks along -w in its frame)
            float3 forward = make_f3(
                sinf(g_app.yaw) * cosf(g_app.pitch),
                sinf(g_app.pitch),
                -cosf(g_app.yaw) * cosf(g_app.pitch));
            float3 right = normalize(cross(forward, make_f3(0, 1, 0)));
            float3 up_dir = make_f3(0, 1, 0);

            float speed = SCENE_CAM_SPEED * dt;
            // Shift for faster movement
            if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
                speed *= 3.f;

            float3 move = make_f3(0, 0, 0);
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) move = move + forward * speed;
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) move = move - forward * speed;
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) move = move - right   * speed;
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) move = move + right   * speed;
            if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)      move = move + up_dir * speed;
            if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) move = move - up_dir * speed;

            if (move.x != 0.f || move.y != 0.f || move.z != 0.f) {
                camera.position = camera.position + move;
                g_app.camera_moved = true;
            }

            // Update look_at from yaw/pitch
            camera.look_at = camera.position + forward;
            camera.width   = win_w;
            camera.height  = win_h;
            camera.update();
        }

        // Reset progressive accumulation if camera moved
        if (g_app.camera_moved) {
            optix_renderer.clear_buffers();
            frame = 0;
            g_app.camera_moved = false;
        }

        // Handle "R" key: start progressive final render
        if (g_app.render_requested && !g_app.rendering) {
            g_app.rendering = true;
            g_app.render_spp_done  = 0;
            g_app.render_spp_total = opt.config.samples_per_pixel;
            g_app.render_start = std::chrono::high_resolution_clock::now();

            // Freeze camera for the render
            g_app.render_cam = camera;
            g_app.render_cam.width  = opt.config.image_width;
            g_app.render_cam.height = opt.config.image_height;
            g_app.render_cam.update();

            optix_renderer.resize(opt.config.image_width, opt.config.image_height);

            std::cout << "\n========================================\n";
            std::cout << "  Progressive Render (R key)\n";
            std::cout << "  Camera pos: (" << camera.position.x << ", "
                      << camera.position.y << ", " << camera.position.z << ")\n";
            std::cout << "  " << opt.config.image_width << "x"
                      << opt.config.image_height << " @ "
                      << g_app.render_spp_total << " spp\n";
            std::cout << "  Output:     " << opt.output_file << "\n";
            std::cout << "========================================\n";

            // ── Photon debug visualization PNG (before path tracing) ─
            {
                const PhotonSoA& photons = optix_renderer.photons();
                if (photons.size() > 0) {
                    std::cout << "[Photon Debug] Rendering " << photons.size()
                              << " photons...\n";

                    const int pw = opt.config.image_width;
                    const int ph = opt.config.image_height;

                    FrameBuffer photon_fb;
                    photon_fb.resize(pw, ph);
                    // Black background
                    for (int i = 0; i < pw * ph; ++i) {
                        photon_fb.srgb[i * 4 + 0] = 0;
                        photon_fb.srgb[i * 4 + 1] = 0;
                        photon_fb.srgb[i * 4 + 2] = 0;
                        photon_fb.srgb[i * 4 + 3] = 255;
                    }

                    float aspect = (float)pw / (float)ph;
                    float theta  = g_app.render_cam.fov_deg * PI / 180.0f;
                    float half_h = tanf(theta * 0.5f);
                    float half_w = half_h * aspect;
                    const int DOT_RADIUS = 1;

                    for (size_t i = 0; i < photons.size(); i += DEBUG_PHOTON_STRIDE) {
                        float3 pos = make_f3(photons.pos_x[i],
                                             photons.pos_y[i],
                                             photons.pos_z[i]);
                        float3 to_p = pos - g_app.render_cam.position;
                        float depth = dot(to_p, g_app.render_cam.w * (-1.f));
                        if (depth <= 0.f) continue;

                        float u_coord = dot(to_p, g_app.render_cam.u);
                        float v_coord = dot(to_p, g_app.render_cam.v);
                        float sx = (u_coord / depth + half_w) / (2.f * half_w);
                        float sy = (v_coord / depth + half_h) / (2.f * half_h);
                        int cx = (int)(sx * pw);
                        int cy = (int)(sy * ph);
                        if (cx < 0 || cx >= pw || cy < 0 || cy >= ph) continue;

                        Spectrum s = Spectrum::zero();
                        s.value[photons.lambda_bin[i]] = 3.0f;
                        float3 rgb = spectrum_to_srgb(s);
                        uint8_t r = (uint8_t)(fminf(fmaxf(rgb.x, 0.f), 1.f) * 255.f);
                        uint8_t g = (uint8_t)(fminf(fmaxf(rgb.y, 0.f), 1.f) * 255.f);
                        uint8_t b = (uint8_t)(fminf(fmaxf(rgb.z, 0.f), 1.f) * 255.f);

                        for (int dy = -DOT_RADIUS; dy <= DOT_RADIUS; ++dy) {
                            for (int dx = -DOT_RADIUS; dx <= DOT_RADIUS; ++dx) {
                                int px = cx + dx;
                                int py = cy + dy;
                                if (px < 0 || px >= pw || py < 0 || py >= ph) continue;
                                int idx = py * pw + px;
                                photon_fb.srgb[idx * 4 + 0] = r;
                                photon_fb.srgb[idx * 4 + 1] = g;
                                photon_fb.srgb[idx * 4 + 2] = b;
                                photon_fb.srgb[idx * 4 + 3] = 255;
                            }
                        }
                    }

                    write_png("output/photon_debug.png", photon_fb);
                    std::cout << "[Photon Debug] Saved: output/photon_debug.png\n";
                }
            }

            // ── 1st-hit NEE debug PNG (no shadows, quick preview) ───
            {
                optix_renderer.render_debug_frame(
                    g_app.render_cam, 0, RenderMode::Full, 1);
                FrameBuffer nee_fb;
                optix_renderer.download_framebuffer(nee_fb);
                write_png("output/out_debug_nee.png", nee_fb);
                std::cout << "[Debug NEE] Saved: output/out_debug_nee.png\n";
                // Clear buffers so the progressive render starts clean
                optix_renderer.clear_buffers();
            }

            // ── OBJ export: scene mesh + photon point cloud ─────────
            {
                const PhotonSoA& photons = optix_renderer.photons();
                fs::create_directories("output");

                // Count how many photons we'll actually export
                size_t num_export = 0;
                for (size_t i = 0; i < photons.size(); i += DEBUG_OBJ_PHOTON_STRIDE)
                    num_export++;

                // Write MTL file with one material per spectral color band
                std::string mtl_file = "output/photon_debug.mtl";
                {
                    std::ofstream mtl(mtl_file);
                    mtl.imbue(std::locale::classic());
                    mtl << std::fixed << std::setprecision(6);
                    mtl << "# Photon debug materials\n\n";
                    mtl << "newmtl scene_grey\n";
                    mtl << "Kd 0.6 0.6 0.6\n\n";

                    for (int bin = 0; bin < NUM_LAMBDA; ++bin) {
                        Spectrum s = Spectrum::zero();
                        s.value[bin] = 3.0f;
                        float3 rgb = spectrum_to_srgb(s);
                        float r = fminf(fmaxf(rgb.x, 0.f), 1.f);
                        float g = fminf(fmaxf(rgb.y, 0.f), 1.f);
                        float b = fminf(fmaxf(rgb.z, 0.f), 1.f);
                        mtl << "newmtl photon_" << bin << "\n";
                        mtl << "Kd " << r << " " << g << " " << b << "\n\n";
                    }
                }

                // Write OBJ file (using stride to keep file manageable)
                std::string obj_file = "output/photon_debug.obj";
                std::ofstream ofs(obj_file);
                ofs.imbue(std::locale::classic());
                if (ofs.is_open()) {
                    ofs << std::fixed << std::setprecision(6);
                    ofs << "# Photon Path Tracer - scene + photon export\n";
                    ofs << "# Scene triangles: " << scene.triangles.size() << "\n";
                    ofs << "# Photons exported: " << num_export
                        << " (every " << DEBUG_OBJ_PHOTON_STRIDE << "th of "
                        << photons.size() << ")\n";
                    ofs << "mtllib photon_debug.mtl\n\n";

                    // Scene vertices
                    ofs << "# Scene geometry\n";
                    for (size_t t = 0; t < scene.triangles.size(); ++t) {
                        const auto& tri = scene.triangles[t];
                        ofs << "v " << tri.v0.x << " " << tri.v0.y << " " << tri.v0.z << "\n";
                        ofs << "v " << tri.v1.x << " " << tri.v1.y << " " << tri.v1.z << "\n";
                        ofs << "v " << tri.v2.x << " " << tri.v2.y << " " << tri.v2.z << "\n";
                    }

                    // Photon micro-triangles (strided)
                    constexpr float SZ = 0.002f;
                    ofs << "\n# Photon micro-triangles\n";
                    for (size_t i = 0; i < photons.size(); i += DEBUG_OBJ_PHOTON_STRIDE) {
                        float px = photons.pos_x[i];
                        float py = photons.pos_y[i];
                        float pz = photons.pos_z[i];
                        ofs << "v " << (px - SZ) << " " << (py - SZ * 0.577f) << " " << pz << "\n";
                        ofs << "v " << (px + SZ) << " " << (py - SZ * 0.577f) << " " << pz << "\n";
                        ofs << "v " << px        << " " << (py + SZ * 1.155f) << " " << pz << "\n";
                    }

                    // Scene faces
                    ofs << "\nusemtl scene_grey\n";
                    ofs << "g scene_mesh\n";
                    for (size_t t = 0; t < scene.triangles.size(); ++t) {
                        size_t base = t * 3 + 1;
                        ofs << "f " << base << " " << (base+1) << " " << (base+2) << "\n";
                    }

                    // Photon faces
                    ofs << "\ng photons\n";
                    size_t photon_v_base = scene.triangles.size() * 3;
                    for (int bin = 0; bin < NUM_LAMBDA; ++bin) {
                        bool header_written = false;
                        size_t export_idx = 0;
                        for (size_t i = 0; i < photons.size(); i += DEBUG_OBJ_PHOTON_STRIDE, ++export_idx) {
                            if (photons.lambda_bin[i] != bin) continue;
                            if (!header_written) {
                                ofs << "usemtl photon_" << bin << "\n";
                                header_written = true;
                            }
                            size_t base = photon_v_base + export_idx * 3 + 1;
                            ofs << "f " << base << " " << (base+1) << " " << (base+2) << "\n";
                        }
                    }

                    ofs.flush();
                    ofs.close();
                    std::cout << "[OBJ Export] Saved: " << obj_file
                              << " (" << scene.triangles.size() << " scene tris, "
                              << num_export << " photon tris)\n";
                } else {
                    std::cerr << "[OBJ Export] Failed to open " << obj_file << "\n";
                }
            }

            g_app.render_requested = false;
        }

        // Progressive rendering: one spp per main-loop iteration
        if (g_app.rendering) {
            optix_renderer.render_one_spp(
                g_app.render_cam, g_app.render_spp_done,
                opt.config.max_bounces);
            g_app.render_spp_done++;

            // Download and display the progressive result
            optix_renderer.download_framebuffer(display_fb);

            // Console progress
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - g_app.render_start).count();
            double eta = (g_app.render_spp_done < g_app.render_spp_total)
                ? elapsed * (g_app.render_spp_total - g_app.render_spp_done) / g_app.render_spp_done
                : 0.0;
            std::printf("\r  [Render] %d/%d spp  %.1fs  ETA %.1fs   ",
                        g_app.render_spp_done, g_app.render_spp_total,
                        elapsed, eta);
            std::fflush(stdout);

            // Check if done
            if (g_app.render_spp_done >= g_app.render_spp_total) {
                std::cout << "\n[Render] Done!\n";

                FrameBuffer final_fb;
                optix_renderer.download_framebuffer(final_fb);
                write_png(opt.output_file, final_fb);

                // Write component PNGs (NEE direct, Photon indirect, Combined)
                {
                    std::vector<float> nee_spec, photon_spec, samp_counts;
                    optix_renderer.download_component_buffers(
                        nee_spec, photon_spec, samp_counts);

                    int cw = final_fb.width;
                    int ch = final_fb.height;

                    // Helper: spectral buffer → sRGB FrameBuffer
                    auto spectral_to_fb = [&](const std::vector<float>& spec_buf,
                                              FrameBuffer& fb) {
                        fb.resize(cw, ch);
                        for (int y = 0; y < ch; ++y) {
                            for (int x = 0; x < cw; ++x) {
                                int px = y * cw + x;
                                float n = samp_counts[px];
                                Spectrum avg = Spectrum::zero();
                                if (n > 0.f) {
                                    for (int k = 0; k < NUM_LAMBDA; ++k)
                                        avg.value[k] = spec_buf[px * NUM_LAMBDA + k] / n;
                                }
                                float3 rgb = spectrum_to_srgb(avg);
                                rgb.x = fminf(fmaxf(rgb.x, 0.f), 1.f);
                                rgb.y = fminf(fmaxf(rgb.y, 0.f), 1.f);
                                rgb.z = fminf(fmaxf(rgb.z, 0.f), 1.f);
                                fb.srgb[px * 4 + 0] = (uint8_t)(rgb.x * 255.f);
                                fb.srgb[px * 4 + 1] = (uint8_t)(rgb.y * 255.f);
                                fb.srgb[px * 4 + 2] = (uint8_t)(rgb.z * 255.f);
                                fb.srgb[px * 4 + 3] = 255;
                            }
                        }
                    };

                    FrameBuffer nee_fb, photon_fb, combined_fb;

                    spectral_to_fb(nee_spec, nee_fb);
                    write_png("output/out_nee_direct.png", nee_fb);

                    spectral_to_fb(photon_spec, photon_fb);
                    write_png("output/out_photon_indirect.png", photon_fb);

                    // Combined = nee + photon (spectral sum, then convert)
                    std::vector<float> combined_spec(nee_spec.size());
                    for (size_t i = 0; i < nee_spec.size(); ++i)
                        combined_spec[i] = nee_spec[i] + photon_spec[i];
                    spectral_to_fb(combined_spec, combined_fb);
                    write_png("output/out_combined.png", combined_fb);
                }

                // Reset to debug view
                optix_renderer.resize(win_w, win_h);
                frame = 0;
                g_app.rendering = false;
                std::cout << "========================================\n";
                std::cout << "  Saved: " << opt.output_file << "\n";
                std::cout << "  Components: out_nee_direct.png, out_photon_indirect.png, out_combined.png\n";
                std::cout << "========================================\n\n";
            }
        } else {
            // Normal debug preview (first-hit, 1 spp per iteration)
            optix_renderer.render_debug_frame(
                camera, frame, g_app.debug.current_mode, 1);
            optix_renderer.download_framebuffer(display_fb);
            frame++;
        }

        // Blit to OpenGL texture
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, win_w, win_h,
                        GL_RGBA, GL_UNSIGNED_BYTE, display_fb.srgb.data());

        glClear(GL_COLOR_BUFFER_BIT);
        glColor4f(1.f, 1.f, 1.f, 1.f); // reset — overlay may have changed it
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(-1, -1);
        glTexCoord2f(1, 0); glVertex2f( 1, -1);
        glTexCoord2f(1, 1); glVertex2f( 1,  1);
        glTexCoord2f(0, 1); glVertex2f(-1,  1);
        glEnd();

        // Draw debug help overlay (if enabled)
        if (g_app.debug.show_help_overlay) {
            render_help_overlay(win_w, win_h, g_app.debug);
        }

        glfwSwapBuffers(window);
    }

    glDeleteTextures(1, &tex);
    glfwDestroyWindow(window);
    glfwTerminate();
}

// -- Main -------------------------------------------------------------

int main(int argc, char* argv[]) {
    std::cout << "== Spectral Photon + Path Tracing Renderer (OptiX) ==\n";
    std::cout << "   Scene: " << SCENE_DISPLAY_NAME << "\n\n";

    Options opt = parse_args(argc, argv);

    // -- Load scene ---------------------------------------------------
    Scene scene;
    auto t0 = std::chrono::high_resolution_clock::now();

    if (!load_obj(opt.scene_file, scene)) {
        std::cerr << "[Error] Failed to load scene: " << opt.scene_file << "\n";
        return 1;
    }

    scene.build_bvh();
    auto t1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[Scene] BVH built (" << scene.bvh_nodes.size()
              << " nodes) in " << load_ms << " ms\n";

    scene.build_emissive_distribution();
    std::cout << "[Scene] " << scene.num_emissive()
              << " emissive triangles, total power = "
              << scene.total_emissive_power << "\n\n";

    // -- Add light source if none exists ------------------------------
    if (scene.num_emissive() == 0) {
        std::cout << "[Scene] No emissive surfaces -- adding area light\n";

        Material light_mat;
        light_mat.name = "__area_light__";
        light_mat.type = MaterialType::Emissive;
        light_mat.Le = blackbody_spectrum(6500.f, 1e-8f);
        uint32_t light_mat_id = (uint32_t)scene.materials.size();
        scene.materials.push_back(light_mat);

        float3 v0 = make_f3(-0.15f,  0.499f, -0.15f);
        float3 v1 = make_f3( 0.15f,  0.499f, -0.15f);
        float3 v2 = make_f3( 0.15f,  0.499f,  0.15f);
        float3 v3 = make_f3(-0.15f,  0.499f,  0.15f);
        float3 n  = make_f3( 0.0f,  -1.0f,    0.0f);

        Triangle tri1;
        tri1.v0 = v0; tri1.v1 = v1; tri1.v2 = v2;
        tri1.n0 = tri1.n1 = tri1.n2 = n;
        tri1.uv0 = tri1.uv1 = tri1.uv2 = make_f2(0, 0);
        tri1.material_id = light_mat_id;

        Triangle tri2;
        tri2.v0 = v0; tri2.v1 = v2; tri2.v2 = v3;
        tri2.n0 = tri2.n1 = tri2.n2 = n;
        tri2.uv0 = tri2.uv1 = tri2.uv2 = make_f2(0, 0);
        tri2.material_id = light_mat_id;

        scene.triangles.push_back(tri1);
        scene.triangles.push_back(tri2);

        scene.build_bvh();
        scene.build_emissive_distribution();
        std::cout << "[Scene] Added area light ("
                  << scene.num_emissive() << " emissive triangles)\n\n";
    }

    // -- Setup camera (from scene profile) --------------------------
    Camera camera;
    camera.position = make_f3(SCENE_CAM_POS[0], SCENE_CAM_POS[1], SCENE_CAM_POS[2]);
    camera.look_at  = make_f3(SCENE_CAM_LOOKAT[0], SCENE_CAM_LOOKAT[1], SCENE_CAM_LOOKAT[2]);
    camera.up       = make_f3(0.0f, 1.0f, 0.0f);
    camera.fov_deg  = SCENE_CAM_FOV;
    camera.width    = opt.config.image_width;
    camera.height   = opt.config.image_height;
    camera.update();

    // -- OptiX pipeline -----------------------------------------------
    OptixRenderer optix_renderer;

    try {
        std::cout << "-- OptiX Initialization --\n";
        optix_renderer.init();
        optix_renderer.build_accel(scene);
        optix_renderer.upload_scene_data(scene);
        optix_renderer.upload_emitter_data(scene);
        std::cout << "\n";

        // -- GPU photon trace -----------------------------------------
        std::cout << "-- GPU Photon Trace --\n";
        auto tp0 = std::chrono::high_resolution_clock::now();
        optix_renderer.trace_photons(scene, opt.config);
        auto tp1 = std::chrono::high_resolution_clock::now();
        double photon_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();
        std::cout << "[Photon] GPU trace completed in " << photon_ms << " ms\n\n";

        // -- Interactive debug window (always) ------------------------
        run_interactive(optix_renderer, camera, opt, scene);

    } catch (const std::exception& e) {
        std::cerr << "[Fatal] " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n== Done ==\n";
    return 0;
}
