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
    const Options& opt)
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

        // Handle "R" key: full path tracing render (uses current camera)
        if (g_app.render_requested && !g_app.rendering) {
            g_app.rendering = true;
            std::cout << "\n========================================\n";
            std::cout << "  Final Render (R key)\n";
            std::cout << "  Camera pos: (" << camera.position.x << ", "
                      << camera.position.y << ", " << camera.position.z << ")\n";
            std::cout << "  Look dir:   yaw=" << g_app.yaw * 180.f / PI
                      << " pitch=" << g_app.pitch * 180.f / PI << " deg\n";
            std::cout << "  Output:     " << opt.output_file << "\n";
            std::cout << "========================================\n";

            optix_renderer.resize(opt.config.image_width, opt.config.image_height);

            // Update camera for render resolution
            Camera render_cam = camera;
            render_cam.width  = opt.config.image_width;
            render_cam.height = opt.config.image_height;
            render_cam.update();

            optix_renderer.render_final(render_cam, opt.config);

            std::cout << "[Render] Downloading framebuffer...\n";
            FrameBuffer final_fb;
            optix_renderer.download_framebuffer(final_fb);
            write_png(opt.output_file, final_fb);

            // Reset debug view
            optix_renderer.resize(win_w, win_h);
            frame = 0;

            g_app.render_requested = false;
            g_app.rendering = false;
            std::cout << "========================================\n";
            std::cout << "  Saved: " << opt.output_file << "\n";
            std::cout << "========================================\n\n";
        }

        // Progressive OptiX debug frame (first-hit, 1 spp per iteration)
        optix_renderer.render_debug_frame(
            camera, frame, g_app.debug.current_mode, 1);
        optix_renderer.download_framebuffer(display_fb);
        frame++;

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
        run_interactive(optix_renderer, camera, opt);

    } catch (const std::exception& e) {
        std::cerr << "[Fatal] " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n== Done ==\n";
    return 0;
}
