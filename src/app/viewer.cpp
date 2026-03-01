// ---------------------------------------------------------------------
// app/viewer.cpp -- GLFW window, event loop, input, overlay rendering
// ---------------------------------------------------------------------
//
// Extracted from main.cpp (§1.7).  Contains:
//   - Camera save/load (JSON persistence per scene folder)
//   - PNG output via stb_image_write
//   - Debug overlay text (stb_easy_font)
//   - Hover cell info overlay
//   - GLFW key/mouse callbacks
//   - run_interactive() event loop (debug preview + progressive render)
// ---------------------------------------------------------------------

#include "app/viewer.h"

#include "core/config.h"
#include "core/runtime_config.h"
#include "core/spectrum.h"
#include "debug/debug.h"
#include "debug/font_overlay.h"
#include "optix/optix_renderer.h"
#include "scene/scene_builder.h"
#include "scene/obj_loader.h"

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
#include <ctime>
#include <filesystem>
#include <cstdio>
#include <iomanip>

namespace fs = std::filesystem;

// -- Singleton app state ----------------------------------------------

static AppState s_app;

AppState& app_state() { return s_app; }

// -- Camera persistence -----------------------------------------------

std::string scene_folder_from_profile(const char* obj_path) {
    std::string p(obj_path);
    auto slash = p.find('/');
    if (slash == std::string::npos) slash = p.find('\\');
    std::string folder = (slash != std::string::npos) ? p.substr(0, slash) : ".";
    return std::string(SCENES_DIR) + "/" + folder;
}

static constexpr const char* SAVED_CAMERA_FILENAME = "saved_camera.json";

bool save_camera_to_file(const Camera& cam, float yaw, float pitch,
                         float light_scale,
                         const std::string& scene_folder) {
    std::string path = scene_folder + "/" + SAVED_CAMERA_FILENAME;
    std::ofstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr, "[Camera] Cannot write: %s\n", path.c_str());
        return false;
    }
    f << std::fixed << std::setprecision(6);
    f << "{\n";
    f << "  \"position\":      [" << cam.position.x << ", " << cam.position.y << ", " << cam.position.z << "],\n";
    f << "  \"look_at\":       [" << cam.look_at.x  << ", " << cam.look_at.y  << ", " << cam.look_at.z  << "],\n";
    f << "  \"fov_deg\":       " << cam.fov_deg << ",\n";
    f << "  \"yaw\":           " << yaw   << ",\n";
    f << "  \"pitch\":         " << pitch << ",\n";
    f << "  \"light_scale\":   " << light_scale << "\n";
    f << "}\n";
    f.close();
    std::printf("[Camera] Saved to %s\n", path.c_str());
    return true;
}

bool load_camera_from_file(Camera& cam, float& yaw, float& pitch,
                           float& light_scale,
                           const std::string& scene_folder) {
    std::string path = scene_folder + "/" + SAVED_CAMERA_FILENAME;
    std::ifstream f(path);
    if (!f.is_open()) return false;

    // Minimal JSON parser – same style as runtime_config.h
    auto trim_ws = [](const std::string& s) -> std::string {
        size_t l = 0, r = s.size();
        while (l < r && std::isspace((unsigned char)s[l])) ++l;
        while (r > l && std::isspace((unsigned char)s[r - 1])) --r;
        return s.substr(l, r - l);
    };

    bool got_pos = false, got_lookat = false;
    std::string line;
    while (std::getline(f, line)) {
        // Strip comments
        auto cpos = line.find("//");
        if (cpos != std::string::npos) line = line.substr(0, cpos);
        line = trim_ws(line);
        if (line.empty() || line[0] == '{' || line[0] == '}') continue;

        // Find "key": value
        auto q1 = line.find('"');
        if (q1 == std::string::npos) continue;
        auto q2 = line.find('"', q1 + 1);
        if (q2 == std::string::npos) continue;
        std::string key = line.substr(q1 + 1, q2 - q1 - 1);

        auto colon = line.find(':', q2 + 1);
        if (colon == std::string::npos) continue;
        std::string val = trim_ws(line.substr(colon + 1));
        // Remove trailing comma
        if (!val.empty() && val.back() == ',') val.pop_back();
        val = trim_ws(val);

        // Parse array [x, y, z]
        auto parse_float3 = [&](const std::string& s, float3& out) -> bool {
            auto lb = s.find('[');
            auto rb = s.find(']');
            if (lb == std::string::npos || rb == std::string::npos) return false;
            std::string inner = s.substr(lb + 1, rb - lb - 1);
            float xyz[3];
            int n = sscanf_s(inner.c_str(), "%f , %f , %f", &xyz[0], &xyz[1], &xyz[2]);
            if (n != 3) return false;
            out = make_f3(xyz[0], xyz[1], xyz[2]);
            return true;
        };

        if (key == "position")     { got_pos    = parse_float3(val, cam.position); }
        else if (key == "look_at")      { got_lookat = parse_float3(val, cam.look_at); }
        else if (key == "fov_deg")      { cam.fov_deg    = (float)std::atof(val.c_str()); }
        else if (key == "yaw")          { yaw            = (float)std::atof(val.c_str()); }
        else if (key == "pitch")        { pitch          = (float)std::atof(val.c_str()); }
        else if (key == "light_scale")  { light_scale    = (float)std::atof(val.c_str()); }
    }

    if (got_pos && got_lookat) {
        cam.update();
        std::printf("[Camera] Loaded saved position from %s\n", path.c_str());
        return true;
    }
    return false;
}

// -- PNG output -------------------------------------------------------

bool write_png(const std::string& filename, const FrameBuffer& fb) {
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

    // Stamp watermarks (top-left origin buffer, before writing)
    font_overlay::stampWatermarks(flipped,
        static_cast<uint32_t>(fb.width),
        static_cast<uint32_t>(fb.height));

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
                                 const DebugState& debug,
                                 const Camera& camera,
                                 bool volume_enabled,
                                 bool use_dense_grid,
                                 int active_scene_index = -1,
                                 float light_scale = 1.0f) {
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

    // Semi-transparent background box – right edge at window center minus gap
    float gap   = 10.f;
    float box_w = 196.f;
    float box_h = 402.f;
    float bx    = (float)win_w * 0.5f - gap - box_w;
    float by    = 10.f;

    glColor4f(0.0f, 0.0f, 0.0f, 0.6f);
    glBegin(GL_QUADS);
    glVertex2f(bx,         by);
    glVertex2f(bx + box_w, by);
    glVertex2f(bx + box_w, by + box_h);
    glVertex2f(bx,         by + box_h);
    glEnd();

    // Build overlay text lines
    auto on_off = [](bool v) -> const char* { return v ? "ON " : "off"; };

    // Scale text 1.12x for readability (70% of original 1.6x)
    float scale = 1.12f;
    float tx = bx + 8.4f;
    float ty = by + 8.4f;
    float line_h = 7.84f; // stb_easy_font is ~7px tall, scaled 1.12x ≈ 7.84

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
    draw_overlay_text(0, ly, "R      Load render_config.json + render", 0.8f, 0.8f, 0.8f, 1.f);
    ly += line_h * 0.7f;
    draw_overlay_text(0, ly, "ESC    Cancel render / release mouse / quit", 0.8f, 0.8f, 0.8f, 1.f);
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
        {"F1", "All photons",     debug.show_photon_points},
        {"F2", "Photon filter",    debug.photon_filter != PhotonFilterMode::Off},
        {"F3", "(reserved)",       false},
        {"F4", "Hash grid",       debug.show_hash_grid},
        {"F5", "Photon dirs",      debug.show_photon_dirs},
        {"F6", "PDFs",             debug.show_pdfs},
        {"F7", "Radius sphere",    debug.show_radius_sphere},
        {"F8", "MIS weights",      debug.show_mis_weights},
        {"F9", "Spectral color",   debug.spectral_coloring},
        {"F11","Photon heatmap",   debug.show_photon_heatmap},
    };
    for (auto& t : toggles) {
        snprintf(buf, sizeof(buf), "%s  %s [%s]", t.key, t.label, on_off(t.on));
        float cr = t.on ? 0.3f : 0.5f;
        float cg = t.on ? 1.0f : 0.5f;
        float cb = t.on ? 0.3f : 0.5f;
        draw_overlay_text(0, ly, buf, cr, cg, cb, 1.f);
        ly += line_h * 0.7f;
    }
    if (debug.photon_filter != PhotonFilterMode::Off) {
        snprintf(buf, sizeof(buf), "      Filter: %s", photon_filter_name(debug.photon_filter));
        draw_overlay_text(0, ly, buf, 0.2f, 0.8f, 1.0f, 1.f);
        ly += line_h * 0.7f;
    }

    ly += line_h * 0.3f;
    draw_overlay_text(0, ly, "--- Volume Scattering ---", 1.f, 1.f, 0.3f, 1.f);
    ly += line_h;
    {
        char vbuf[64];
        snprintf(vbuf, sizeof(vbuf), "V    Volume [%s]", volume_enabled ? "ON" : "OFF");
        float vr = volume_enabled ? 0.3f : 0.5f;
        float vg = volume_enabled ? 1.0f : 0.5f;
        float vb = volume_enabled ? 0.3f : 0.5f;
        draw_overlay_text(0, ly, vbuf, vr, vg, vb, 1.f);
        ly += line_h * 0.7f;

        char gbuf[64];
        snprintf(gbuf, sizeof(gbuf), "G    Dense Grid [%s]", use_dense_grid ? "ON" : "OFF");
        float gr = use_dense_grid ? 0.3f : 0.5f;
        float gg = use_dense_grid ? 1.0f : 0.5f;
        float gb = use_dense_grid ? 0.3f : 0.5f;
        draw_overlay_text(0, ly, gbuf, gr, gg, gb, 1.f);
        ly += line_h * 0.7f;
    }

    ly += line_h * 0.3f;
    draw_overlay_text(0, ly, "--- Scenes (1-9) ---", 1.f, 1.f, 0.3f, 1.f);
    ly += line_h;
    for (int i = 0; i < NUM_SCENE_PROFILES; ++i) {
        char sbuf[128];
        snprintf(sbuf, sizeof(sbuf), "%d    %s%s", i + 1,
                 SCENE_PROFILES[i].display_name,
                 (i == active_scene_index) ? " [*]" : "");
        float sr = (i == active_scene_index) ? 0.3f : 0.5f;
        float sg = (i == active_scene_index) ? 1.0f : 0.5f;
        float sb = (i == active_scene_index) ? 0.3f : 0.5f;
        draw_overlay_text(0, ly, sbuf, sr, sg, sb, 1.f);
        ly += line_h * 0.7f;
    }

    ly += line_h * 0.3f;
    draw_overlay_text(0, ly, "--- Light Brightness ---", 1.f, 1.f, 0.3f, 1.f);
    ly += line_h;
    {
        char lbuf[128];
        snprintf(lbuf, sizeof(lbuf), "+/-  Brightness: %.2fx", light_scale);
        draw_overlay_text(0, ly, lbuf, 0.8f, 0.8f, 0.8f, 1.f);
        ly += line_h * 0.7f;
    }

    ly += line_h * 0.3f;
    draw_overlay_text(0, ly, "--- Depth of Field ---", 1.f, 1.f, 0.3f, 1.f);
    ly += line_h;
    {
        char dof_buf[128];
        snprintf(dof_buf, sizeof(dof_buf), "O    DOF [%s]",
                 camera.dof_enabled ? "ON" : "OFF");
        float dr = camera.dof_enabled ? 0.3f : 0.5f;
        float dg = camera.dof_enabled ? 1.0f : 0.5f;
        float db = camera.dof_enabled ? 0.3f : 0.5f;
        draw_overlay_text(0, ly, dof_buf, dr, dg, db, 1.f);
        ly += line_h * 0.7f;
        snprintf(dof_buf, sizeof(dof_buf), "[/]  f-number: f/%.1f", camera.dof_f_number);
        draw_overlay_text(0, ly, dof_buf, 0.8f, 0.8f, 0.8f, 1.f);
        ly += line_h * 0.7f;
        snprintf(dof_buf, sizeof(dof_buf), ",/.  Focus dist: %.4f", camera.dof_focus_dist);
        draw_overlay_text(0, ly, dof_buf, 0.8f, 0.8f, 0.8f, 1.f);
        ly += line_h * 0.7f;
        draw_overlay_text(0, ly, "F    Auto-focus (center)", 0.8f, 0.8f, 0.8f, 1.f);
        ly += line_h * 0.7f;
        draw_overlay_text(0, ly, "MMB  Auto-focus (cursor)", 0.8f, 0.8f, 0.8f, 1.f);
        ly += line_h * 0.7f;
    }

    ly += line_h * 0.3f;
    draw_overlay_text(0, ly, "F10  Save camera position", 0.8f, 0.8f, 0.8f, 1.f);
    ly += line_h * 0.7f;

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

static void render_hover_cell_overlay(
    int win_w, int win_h,
    const Camera& camera,
    const DebugState& debug,
    const OptixRenderer& optix_renderer,
    bool mouse_captured)
{
    if (mouse_captured) return;
    if (!debug.photon_overlay_active()) return;
    if (debug.hover_x < 0 || debug.hover_x >= win_w ||
        debug.hover_y < 0 || debug.hover_y >= win_h) return;

    const PhotonSoA& photons = optix_renderer.photons();
    const HashGrid& grid = optix_renderer.grid();
    if (photons.size() == 0 || grid.sorted_indices.empty()) return;

    float s = ((float)debug.hover_x + 0.5f) / (float)win_w;
    float t = 1.0f - (((float)debug.hover_y + 0.5f) / (float)win_h);
    float3 ray_dir = normalize(
        camera.lower_left + camera.horizontal * s + camera.vertical * t
        - camera.position);

    HitRecord hit = optix_renderer.trace_single_ray(camera.position, ray_dir);
    if (!hit.hit) return;

    PhotonMapType map_type = PhotonMapType::Global;

    CellInfo info = query_cell_info(hit.position, photons, grid, map_type);

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

    float box_w = 340.f;
    float box_h = 150.f;
    float bx = (float)win_w * 0.5f + 10.f;   // left edge at window center + gap
    float by = 10.f;

    glColor4f(0.0f, 0.0f, 0.0f, 0.72f);
    glBegin(GL_QUADS);
    glVertex2f(bx,         by);
    glVertex2f(bx + box_w, by);
    glVertex2f(bx + box_w, by + box_h);
    glVertex2f(bx,         by + box_h);
    glEnd();

    char buf[256];
    float scale = 2.f;
    float tx = bx + 12.f;
    float ty = by + 12.f;
    float line_h = 14.f;

    glPushMatrix();
    glTranslatef(tx, ty, 0.f);
    glScalef(scale, scale, 1.f);

    float ly = 0.f;
    draw_overlay_text(0, ly, "=== Hover Cell Info ===", 1.f, 1.f, 0.3f, 1.f);
    ly += line_h;

    snprintf(buf, sizeof(buf), "Map: %s",
             (map_type == PhotonMapType::Caustic) ? "Caustic" : "Global");
    draw_overlay_text(0, ly, buf, 0.8f, 0.9f, 1.f, 1.f);
    ly += line_h * 0.75f;

    snprintf(buf, sizeof(buf), "Cell: (%d, %d, %d)",
             info.cell_index.x, info.cell_index.y, info.cell_index.z);
    draw_overlay_text(0, ly, buf, 0.8f, 0.8f, 0.8f, 1.f);
    ly += line_h * 0.75f;

    snprintf(buf, sizeof(buf), "Photons: %u", info.photon_count);
    draw_overlay_text(0, ly, buf, 0.8f, 0.8f, 0.8f, 1.f);
    ly += line_h * 0.75f;

    snprintf(buf, sizeof(buf), "Flux sum/avg: %.5g / %.5g",
             info.sum_flux, info.avg_flux);
    draw_overlay_text(0, ly, buf, 0.8f, 0.8f, 0.8f, 1.f);
    ly += line_h * 0.75f;

    snprintf(buf, sizeof(buf), "Dominant λ: %.1f nm (bin %d)",
             info.dominant_lambda_nm, info.dominant_lambda_bin);
    draw_overlay_text(0, ly, buf, 0.8f, 0.8f, 0.8f, 1.f);

    glPopMatrix();

    glDisable(GL_BLEND);
    glEnable(GL_TEXTURE_2D);
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

// -- GLFW callbacks ---------------------------------------------------

static Camera*         g_active_camera          = nullptr;
static OptixRenderer*  g_active_optix_renderer  = nullptr;
static Options*        g_active_options         = nullptr;

static void key_callback(GLFWwindow* window, int key,
                          int /*scancode*/, int action, int /*mods*/) {
    if (action != GLFW_PRESS) return;

    if (key == GLFW_KEY_ESCAPE) {
        // ESC when mouse captured -> release mouse
        if (s_app.mouse_captured) {
            s_app.mouse_captured = false;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            return;
        }
        // ESC otherwise -> quit
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        return;
    }

    if (key == GLFW_KEY_Q) {
        if (s_app.mouse_captured) {
            s_app.mouse_captured = false;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            return;
        }
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        return;
    }

    // "R" -> save timestamped snapshot (PNG + JSON statistics)
    if (key == GLFW_KEY_R) {
        s_app.snapshot_requested = true;
        return;
    }

    // Left-click or M to toggle mouse capture
    if (key == GLFW_KEY_M) {
        s_app.mouse_captured = !s_app.mouse_captured;
        glfwSetInputMode(window, GLFW_CURSOR,
            s_app.mouse_captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
        s_app.first_mouse = true;
        return;
    }

    // Volume scattering toggle – V key
    if (key == GLFW_KEY_V) {
        s_app.volume_enabled = !s_app.volume_enabled;
        if (g_active_optix_renderer)
            g_active_optix_renderer->set_volume_enabled(s_app.volume_enabled);
        printf("[Volume] Scattering: %s\n", s_app.volume_enabled ? "ON" : "OFF");
        s_app.camera_moved  = true;  // reset accumulation
        return;
    }

    // Gather mode toggle – G key (dense cell-bin grid vs per-photon hash walk)
    if (key == GLFW_KEY_G) {
        s_app.use_dense_grid = !s_app.use_dense_grid;
        if (g_active_optix_renderer)
            g_active_optix_renderer->set_use_dense_grid(s_app.use_dense_grid);
        printf("[Gather] Dense grid: %s\n", s_app.use_dense_grid ? "ON" : "OFF");
        s_app.camera_moved  = true;  // reset accumulation
        return;
    }

    // Scene switching – keys 1-9
    if (key >= GLFW_KEY_1 && key <= GLFW_KEY_9) {
        int idx = key - GLFW_KEY_1;  // 0-8
        if (idx < NUM_SCENE_PROFILES && idx != s_app.active_scene_index) {
            s_app.scene_switch_requested = idx;
            printf("[Scene] Switching to %s ...\n", SCENE_PROFILES[idx].display_name);
        } else if (idx < NUM_SCENE_PROFILES) {
            printf("[Scene] Already on %s\n", SCENE_PROFILES[idx].display_name);
        }
        return;
    }

    // Light brightness – +/= increase, -/_ decrease
    if ((key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD)) {
        s_app.light_scale = fminf(LIGHT_SCALE_MAX, s_app.light_scale * LIGHT_SCALE_STEP);
        s_app.light_scale_changed = true;
        printf("[Light] Brightness: %.2fx\n", s_app.light_scale);
        return;
    }
    if ((key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT)) {
        s_app.light_scale = fmaxf(LIGHT_SCALE_MIN, s_app.light_scale / LIGHT_SCALE_STEP);
        s_app.light_scale_changed = true;
        printf("[Light] Brightness: %.2fx\n", s_app.light_scale);
        return;
    }

    // DOF hotkeys – O toggle, [/] f-number, ,/. focus distance, F auto-focus center
    if (g_active_camera) {
        constexpr float kFStopFactor = 1.2599f; // 2^(1/3): one-third stop per key press
        constexpr float kFocusStep   = 1.10f;   // 10% focus distance per key press

        Camera& cam = *g_active_camera;
        bool dof_changed = false;

        if (key == GLFW_KEY_O) {
            cam.dof_enabled = !cam.dof_enabled;
            printf("[DOF] %s\n", cam.dof_enabled ? "ON" : "OFF");
            dof_changed = true;
        } else if (key == GLFW_KEY_LEFT_BRACKET) {
            // [ -> widen aperture (lower f-number = more blur)
            cam.dof_f_number = fmaxf(1.0f, cam.dof_f_number / kFStopFactor);
            printf("[DOF] f-number: f/%.2f (more blur)\n", cam.dof_f_number);
            dof_changed = true;
        } else if (key == GLFW_KEY_RIGHT_BRACKET) {
            // ] -> narrow aperture (higher f-number = less blur)
            cam.dof_f_number = fminf(64.0f, cam.dof_f_number * kFStopFactor);
            printf("[DOF] f-number: f/%.2f (less blur)\n", cam.dof_f_number);
            dof_changed = true;
        } else if (key == GLFW_KEY_COMMA) {
            // , -> focus closer
            cam.dof_focus_dist = fmaxf(0.01f, cam.dof_focus_dist / kFocusStep);
            printf("[DOF] Focus distance: %.4f\n", cam.dof_focus_dist);
            dof_changed = true;
        } else if (key == GLFW_KEY_PERIOD) {
            // . -> focus farther
            cam.dof_focus_dist = fminf(1000.0f, cam.dof_focus_dist * kFocusStep);
            printf("[DOF] Focus distance: %.4f\n", cam.dof_focus_dist);
            dof_changed = true;
        } else if (key == GLFW_KEY_F && g_active_optix_renderer) {
            // F -> auto-focus on screen center (cast a ray, set focus dist to hit)
            float s = 0.5f;
            float t = 0.5f;
            float3 ray_dir = normalize(
                cam.lower_left + cam.horizontal * s + cam.vertical * t
                - cam.position);
            HitRecord hit = g_active_optix_renderer->trace_single_ray(cam.position, ray_dir);
            if (hit.hit && hit.t > 0.01f && hit.t < 1000.0f) {
                cam.dof_focus_dist = hit.t;
                printf("[DOF] Auto-focus (center): %.4f\n", cam.dof_focus_dist);
                dof_changed = true;
            } else {
                printf("[DOF] Auto-focus: no hit\n");
            }
        }

        if (dof_changed) {
            cam.update();
            s_app.camera_moved   = true;
            return;
        }
    }

    // P -> retrace photons (rebuild photon map)
    if (key == GLFW_KEY_P) {
        s_app.photon_retrace_requested = true;
        s_app.camera_moved  = true;  // reset accumulation
        printf("[Photon] Retrace requested (rebuild photon maps)\n");
        return;
    }

    // F10 -> save current camera position to scene folder
    if (key == GLFW_KEY_F10 && g_active_camera) {
        int idx = s_app.active_scene_index;
        if (idx >= 0 && idx < NUM_SCENE_PROFILES) {
            std::string folder = scene_folder_from_profile(SCENE_PROFILES[idx].obj_path);
            save_camera_to_file(*g_active_camera, s_app.yaw, s_app.pitch,
                                s_app.light_scale, folder);
        } else {
            // Derive folder from the active options scene_file
            if (g_active_options) {
                fs::path p(g_active_options->scene_file);
                std::string folder = p.parent_path().string();
                save_camera_to_file(*g_active_camera, s_app.yaw, s_app.pitch,
                                    s_app.light_scale, folder);
            }
        }
        return;
    }

    // Handle debug key toggles
    handle_debug_key(key, s_app.debug);

    // Sync GPU-side debug state that lives on the renderer
    if (g_active_optix_renderer)
        g_active_optix_renderer->set_photon_heatmap(s_app.debug.show_photon_heatmap);
}

static void mouse_button_callback(GLFWwindow* window, int button,
                                   int action, int /*mods*/) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS
        && !s_app.mouse_captured) {
        s_app.mouse_captured = true;
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        s_app.first_mouse = true;
    }

    // Middle-click: auto-focus on the point under the cursor
    if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS
        && !s_app.mouse_captured && g_active_camera && g_active_optix_renderer) {
        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        int win_w, win_h;
        glfwGetWindowSize(window, &win_w, &win_h);
        float s = ((float)mx + 0.5f) / (float)win_w;
        float t = 1.0f - (((float)my + 0.5f) / (float)win_h);
        Camera& cam = *g_active_camera;
        float3 ray_dir = normalize(
            cam.lower_left + cam.horizontal * s + cam.vertical * t
            - cam.position);
        HitRecord hit = g_active_optix_renderer->trace_single_ray(cam.position, ray_dir);
        if (hit.hit && hit.t > 0.01f && hit.t < 1000.0f) {
            cam.dof_focus_dist = hit.t;
            cam.update();
            s_app.camera_moved  = true;
            printf("[DOF] Auto-focus (cursor): %.4f\n", cam.dof_focus_dist);
        } else {
            printf("[DOF] Auto-focus: no hit\n");
        }
    }
}

// -- Interactive OptiX debug window -----------------------------------

void run_interactive(
    OptixRenderer& optix_renderer,
    Camera& camera,
    Options& opt,
    Scene& scene)
{
    if (!glfwInit()) {
        std::cerr << "[GLFW] Failed to initialize\n";
        return;
    }

    int win_w = opt.config.image_width;
    int win_h = opt.config.image_height;

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_FALSE);
#if GLFW_VERSION_MAJOR >= 3 && GLFW_VERSION_MINOR >= 4
    glfwWindowHint(GLFW_SCALE_FRAMEBUFFER, GLFW_FALSE);
#endif
    GLFWwindow* window = glfwCreateWindow(
        win_w, win_h,
        "Spectral Photon+Path Tracer [OptiX]", nullptr, nullptr);

    if (!window) {
        std::cerr << "[GLFW] Failed to create window\n";
        glfwTerminate();
        return;
    }

    // Verify actual window and framebuffer sizes match the config
    {
        int actual_w, actual_h, fb_w, fb_h;
        glfwGetWindowSize(window, &actual_w, &actual_h);
        glfwGetFramebufferSize(window, &fb_w, &fb_h);
        std::printf("[Window] Requested: %dx%d  Window: %dx%d  Framebuffer: %dx%d\n",
                    win_w, win_h, actual_w, actual_h, fb_w, fb_h);
        if (fb_w != win_w || fb_h != win_h) {
            std::printf("[Window] WARNING: framebuffer size differs from config! "
                        "DPI scale: %.2fx%.2f\n",
                        (float)fb_w / win_w, (float)fb_h / win_h);
            // Use framebuffer dimensions so rendering matches the actual pixel count
            win_w = fb_w;
            win_h = fb_h;
        }
    }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    // Start with mouse captured for FPS-style controls
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Expose camera and renderer to key_callback for runtime editing
    g_active_camera         = &camera;
    g_active_optix_renderer = &optix_renderer;
    g_active_options        = &opt;

    // Sync initial volume state from AppState into the renderer
    optix_renderer.set_volume_enabled(s_app.volume_enabled);
    optix_renderer.set_use_dense_grid(s_app.use_dense_grid);

    // Compute initial yaw/pitch from camera look direction
    {
        float3 fwd = normalize(camera.look_at - camera.position);
        s_app.yaw   = atan2f(fwd.x, -fwd.z);
        s_app.pitch = asinf(fmaxf(-1.f, fminf(1.f, fwd.y)));
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
    std::cout << "  ESC = cancel render / release mouse / quit | Q = quit\n";
    std::cout << "  F1-F9 = debug toggles | TAB = cycle mode\n";
    std::cout << "  R = load render_config.json + full path tracing render -> " << opt.output_file << "\n";
    if (opt.config.sppm_enabled)
        std::cout << "      (SPPM mode: " << opt.config.sppm_iterations << " iterations, r="
                  << opt.config.sppm_initial_radius << ")\n";
    std::cout << "  H = toggle help overlay\n";
    std::cout << "  1-9 = switch scene\n";
    std::cout << "  +/- = adjust light brightness (re-traces photons)\n";
    std::cout << "  V = toggle volume scattering | O = toggle DOF | [/] = blur | ,/. = focus dist\n";
    std::cout << "  F10 = save camera position to scene folder\n";

    FrameBuffer display_fb;
    display_fb.resize(win_w, win_h);

    int frame = 0;
    auto last_time = std::chrono::high_resolution_clock::now();

    // Original emission spectra (for light brightness scaling)
    std::vector<Spectrum> original_Le;
    auto capture_original_Le = [&]() {
        original_Le.resize(scene.materials.size());
        for (size_t m = 0; m < scene.materials.size(); ++m)
            original_Le[m] = scene.materials[m].Le;
    };
    capture_original_Le();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // ── Runtime scene switch (keys 1-8) ────────────────────────────
        if (s_app.scene_switch_requested >= 0) {
            int idx = s_app.scene_switch_requested;
            s_app.scene_switch_requested = -1;

            const SceneProfile& prof = SCENE_PROFILES[idx];
            std::string obj_path = std::string(SCENES_DIR) + "/" + prof.obj_path;

            std::cout << "\n========================================\n";
            std::cout << "  Loading scene: " << prof.display_name << "\n";
            std::cout << "========================================\n";

            // Load new scene
            Scene new_scene;
            auto t0 = std::chrono::high_resolution_clock::now();
            if (!load_obj(obj_path, new_scene)) {
                std::cerr << "[Error] Failed to load: " << obj_path << "\n";
            } else {
                if (!prof.is_reference)
                    new_scene.normalize_to_reference();
                new_scene.build_bvh();
                new_scene.build_emissive_distribution();

                // Add lights based on scene lighting mode
                add_scene_lights(new_scene, prof.light_mode);

                // Replace current scene
                scene = std::move(new_scene);

                // Rebuild OptiX data
                optix_renderer.build_accel(scene);
                optix_renderer.upload_scene_data(scene);
                optix_renderer.upload_emitter_data(scene);

                // ── Trace photons on GPU ───────────────────────────
                auto tp0 = std::chrono::high_resolution_clock::now();
                optix_renderer.trace_photons(scene, opt.config);
                auto tp1 = std::chrono::high_resolution_clock::now();
                double photon_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

                // Reset camera to scene profile defaults
                camera.position = make_f3(prof.cam_pos[0], prof.cam_pos[1], prof.cam_pos[2]);
                camera.look_at  = make_f3(prof.cam_lookat[0], prof.cam_lookat[1], prof.cam_lookat[2]);
                camera.fov_deg  = prof.cam_fov;
                camera.width    = win_w;
                camera.height   = win_h;
                camera.update();

                // Override with saved camera position (if any)
                {
                    std::string folder = scene_folder_from_profile(prof.obj_path);
                    float saved_yaw = 0.f, saved_pitch = 0.f;
                    float saved_light = DEFAULT_LIGHT_SCALE;
                    if (load_camera_from_file(camera, saved_yaw, saved_pitch,
                                              saved_light, folder)) {
                        s_app.yaw   = saved_yaw;
                        s_app.pitch = saved_pitch;
                        s_app.light_scale         = saved_light;
                        s_app.light_scale_changed = true;
                    } else {
                        // Sync yaw/pitch from new camera direction
                        float3 fwd = normalize(camera.look_at - camera.position);
                        s_app.yaw   = atan2f(fwd.x, -fwd.z);
                        s_app.pitch = asinf(fmaxf(-1.f, fminf(1.f, fwd.y)));
                        s_app.light_scale = DEFAULT_LIGHT_SCALE;
                    }
                }

                s_app.active_cam_speed    = prof.cam_speed;
                s_app.active_scene_index  = idx;
                s_app.camera_moved        = true;
                opt.scene_file            = obj_path;

                // Re-capture original emission for brightness scaling
                capture_original_Le();

                auto t1 = std::chrono::high_resolution_clock::now();
                double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                std::cout << "[Scene] " << prof.display_name << " loaded in "
                          << total_ms << " ms"
                          << " (photons: " << (int)photon_ms << " ms)"
                          << "\n";
                std::cout << "[Scene] " << scene.num_triangles() << " tris, "
                          << scene.num_emissive() << " emissive\n";

                // Update window title
                std::string title = std::string("Spectral Photon+Path Tracer [OptiX] – ")
                                    + prof.display_name;
                glfwSetWindowTitle(window, title.c_str());
            }
        }

        // ── Photon retrace request (P key) ────────────────────────────
        if (s_app.photon_retrace_requested) {
            s_app.photon_retrace_requested = false;

            std::cout << "[Photon] Re-tracing photon maps...\n";
            auto tp0 = std::chrono::high_resolution_clock::now();
            optix_renderer.trace_photons(scene, opt.config);
            auto tp1 = std::chrono::high_resolution_clock::now();
            double photon_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();
            std::cout << "[Photon] Retrace done in " << photon_ms << " ms\n";

            frame = 0;
            optix_renderer.clear_buffers();
            s_app.camera_moved  = true;
        }

        // ── Light brightness change (+/- keys) ─────────────────────────
        if (s_app.light_scale_changed) {
            s_app.light_scale_changed = false;

            // Scale all emissive materials' Le by the new factor relative to 1.0
            // We store original Le values and always scale from those
            for (size_t m = 0; m < scene.materials.size(); ++m) {
                Spectrum scaled = original_Le[m];
                for (int k = 0; k < NUM_LAMBDA; ++k)
                    scaled.value[k] *= s_app.light_scale;
                scene.materials[m].Le = scaled;
            }

            // Rebuild emissive distribution with new powers
            scene.build_emissive_distribution();

            // Re-upload materials and emitter CDF
            optix_renderer.upload_scene_data(scene);
            optix_renderer.upload_emitter_data(scene);

            // Re-trace photons with updated emission
            std::cout << "[Light] Re-tracing photons at " << s_app.light_scale << "x brightness...\n";
            auto tp0 = std::chrono::high_resolution_clock::now();
            optix_renderer.trace_photons(scene, opt.config);
            auto tp1 = std::chrono::high_resolution_clock::now();
            double photon_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();
            std::cout << "[Light] Photon re-trace done in " << photon_ms << " ms\n";

            s_app.camera_moved  = true;
        }

        // Mouse cursor position for hover inspectors.
        {
            double mx, my;
            glfwGetCursorPos(window, &mx, &my);
            s_app.debug.hover_x = (int)mx;
            s_app.debug.hover_y = (int)my;
        }

        // Delta time
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        last_time = now;
        dt = fminf(dt, 0.1f); // clamp to avoid huge jumps

        // -- Mouse look -------------------------------------------------
        constexpr float kMouseSens = 0.0005f; // radians per pixel
        if (s_app.mouse_captured) {
            double mx, my;
            glfwGetCursorPos(window, &mx, &my);
            if (s_app.first_mouse) {
                s_app.last_mx = mx;
                s_app.last_my = my;
                s_app.first_mouse = false;
            }
            float dx = (float)(mx - s_app.last_mx);
            float dy = (float)(my - s_app.last_my);
            s_app.last_mx = mx;
            s_app.last_my = my;

            if (dx != 0.f || dy != 0.f) {
                s_app.yaw   += dx * kMouseSens;
                s_app.pitch -= dy * kMouseSens; // inverted Y
                // Clamp pitch to avoid gimbal lock
                constexpr float MAX_PITCH = 89.f * PI / 180.f;
                s_app.pitch = fmaxf(-MAX_PITCH, fminf(MAX_PITCH, s_app.pitch));
                s_app.camera_moved = true;
            }
        }

        // -- WASD movement -----------------------------------------------
        {
            // Forward direction (camera looks along -w in its frame)
            float3 forward = make_f3(
                sinf(s_app.yaw) * cosf(s_app.pitch),
                sinf(s_app.pitch),
                -cosf(s_app.yaw) * cosf(s_app.pitch));
            float3 right = normalize(cross(forward, make_f3(0, 1, 0)));
            float3 up_dir = make_f3(0, 1, 0);

            float speed = s_app.active_cam_speed * dt;
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
                s_app.camera_moved = true;
            }

            // Update look_at from yaw/pitch
            camera.look_at = camera.position + forward;
            camera.width   = win_w;
            camera.height  = win_h;
            camera.update();
        }

        // Reset progressive accumulation if camera moved
        if (s_app.camera_moved) {
            optix_renderer.clear_buffers();
            frame = 0;
            s_app.camera_moved = false;
        }

        // ── Handle "R" key: save timestamped snapshot (PNG + JSON) ───
        if (s_app.snapshot_requested) {
            s_app.snapshot_requested = false;

            // Ensure output directory exists
            fs::create_directories("output");

            // Build timestamped prefix: output/snapshot_YYYYMMDD_HHMMSS
            auto now_tp = std::chrono::system_clock::now();
            std::time_t now_t = std::chrono::system_clock::to_time_t(now_tp);
            std::tm tm_buf;
            localtime_s(&tm_buf, &now_t);
            char ts[64];
            std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tm_buf);
            std::string prefix = std::string("output/snapshot_") + ts;

            // Save PNG
            std::string png_path = prefix + ".png";
            optix_renderer.download_framebuffer(display_fb);
            write_png(png_path, display_fb);

            // Save raw (un-denoised) version when denoiser is active
            std::string raw_path;
            if (opt.config.denoiser_enabled) {
                raw_path = prefix + "_raw.png";
                FrameBuffer raw_fb;
                optix_renderer.download_raw_framebuffer(raw_fb);
                write_png(raw_path, raw_fb);
            }

            // Gather statistics and write JSON
            const char* scene_name = (s_app.active_scene_index >= 0
                && s_app.active_scene_index < NUM_SCENE_PROFILES)
                ? SCENE_PROFILES[s_app.active_scene_index].display_name
                : SCENE_DISPLAY_NAME;
            auto stats = optix_renderer.gather_stats(scene_name);

            std::string json_path = prefix + ".json";
            {
                std::ofstream jf(json_path);
                jf << std::fixed << std::setprecision(6);
                jf << "{\n";
                jf << "  \"timestamp\": \"" << ts << "\",\n";
                jf << "  \"png_file\": \"" << png_path << "\",\n";

                // Image
                jf << "  \"image\": {\n";
                jf << "    \"width\": " << stats.image_width << ",\n";
                jf << "    \"height\": " << stats.image_height << ",\n";
                jf << "    \"accumulated_spp\": " << stats.accumulated_spp << "\n";
                jf << "  },\n";

                // Camera
                jf << "  \"camera\": {\n";
                jf << "    \"position\": [" << camera.position.x << ", "
                   << camera.position.y << ", " << camera.position.z << "],\n";
                jf << "    \"look_at\": [" << camera.look_at.x << ", "
                   << camera.look_at.y << ", " << camera.look_at.z << "],\n";
                jf << "    \"fov_deg\": " << camera.fov_deg << ",\n";
                jf << "    \"yaw\": " << s_app.yaw << ",\n";
                jf << "    \"pitch\": " << s_app.pitch << "\n";
                jf << "  },\n";

                // Photon map
                jf << "  \"photon_map\": {\n";
                jf << "    \"photons_emitted\": " << stats.photons_emitted << ",\n";
                jf << "    \"photons_stored\": " << stats.photons_stored << ",\n";
                jf << "    \"caustic_emitted\": " << stats.caustic_emitted << ",\n";
                jf << "    \"gather_radius\": " << stats.gather_radius << ",\n";
                jf << "    \"caustic_radius\": " << stats.caustic_radius << ",\n";
                jf << "    \"tag_distribution\": {\n";
                jf << "      \"noncaustic\": " << stats.noncaustic_stored << ",\n";
                jf << "      \"global_caustic\": " << stats.global_caustic_stored << ",\n";
                jf << "      \"targeted_caustic\": " << stats.caustic_stored << "\n";
                jf << "    }\n";
                jf << "  },\n";

                // Guidance / cell analysis
                jf << "  \"guidance\": {\n";
                jf << "    \"cell_analysis_cells\": " << stats.cell_analysis_cells << ",\n";
                jf << "    \"avg_guide_fraction\": " << stats.avg_guide_fraction << ",\n";
                jf << "    \"avg_caustic_fraction\": " << stats.avg_caustic_fraction << "\n";
                jf << "  },\n";

                // Render config
                jf << "  \"config\": {\n";
                jf << "    \"max_bounces_camera\": " << stats.max_bounces_camera << ",\n";
                jf << "    \"max_bounces_photon\": " << stats.max_bounces_photon << ",\n";
                jf << "    \"min_bounces_rr\": " << stats.min_bounces_rr << ",\n";
                jf << "    \"rr_threshold\": " << stats.rr_threshold << ",\n";
                jf << "    \"guide_fraction\": " << stats.guide_fraction << ",\n";
                jf << "    \"exposure\": " << stats.exposure << ",\n";
                jf << "    \"denoiser_enabled\": " << (stats.denoiser_enabled ? "true" : "false") << ",\n";
                jf << "    \"knn_k\": " << stats.knn_k << ",\n";
                jf << "    \"surface_tau\": " << stats.surface_tau << ",\n";
                jf << "    \"light_scale\": " << s_app.light_scale << "\n";
                jf << "  },\n";

                // Scene
                jf << "  \"scene\": {\n";
                jf << "    \"name\": \"" << stats.scene_name << "\",\n";
                jf << "    \"num_triangles\": " << stats.num_triangles << ",\n";
                jf << "    \"num_emissive_tris\": " << stats.num_emissive_tris << "\n";
                jf << "  }\n";

                jf << "}\n";
            }

            std::cout << "\n========================================\n";
            std::cout << "  [Snapshot] " << png_path << "\n";
            std::cout << "  [Snapshot] " << json_path << "\n";
            if (!raw_path.empty())
                std::cout << "  [Snapshot] " << raw_path << " (raw)\n";
            std::cout << "  SPP: " << stats.accumulated_spp
                      << "  Photons: " << stats.photons_stored
                      << "  Cells: " << stats.cell_analysis_cells << "\n";
            std::cout << "========================================\n\n";
        }

        // ── Preview path: continuous progressive path tracing ────────
        {
            // Photon overlay mode: GL_POINTS only, no ray tracing
            if (s_app.debug.photon_overlay_active()) {
                // Clear to black background
                memset(display_fb.srgb.data(), 0,
                       display_fb.srgb.size() * sizeof(display_fb.srgb[0]));
                // Set alpha to 255
                for (int p = 0; p < display_fb.width * display_fb.height; ++p)
                    display_fb.srgb[p * 4 + 3] = 255;

                const PhotonSoA& photons = optix_renderer.photons();
                if (photons.size() > 0) {
                    uint8_t flag = photon_filter_flag(s_app.debug.photon_filter);
                    overlay_photon_points(
                        display_fb,
                        camera,
                        photons,
                        s_app.debug.spectral_coloring,
                        flag,
                        2.0f);
                }
            } else {
                // Normal preview (1 spp per iteration, progressive accumulation)
                optix_renderer.render_debug_frame(
                    camera, frame, s_app.debug.current_mode, 1);
                optix_renderer.download_framebuffer(display_fb);
            }

            frame++;
        }

        // Blit to OpenGL texture (handle resolution changes between preview and render)
        int fb_w = display_fb.width;
        int fb_h = display_fb.height;
        glBindTexture(GL_TEXTURE_2D, tex);
        {
            // Query current texture dimensions
            int cur_tex_w = 0, cur_tex_h = 0;
            glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH,  &cur_tex_w);
            glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &cur_tex_h);
            if (cur_tex_w != fb_w || cur_tex_h != fb_h) {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, fb_w, fb_h, 0,
                             GL_RGBA, GL_UNSIGNED_BYTE, display_fb.srgb.data());
                glViewport(0, 0, fb_w, fb_h);
            } else {
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, fb_w, fb_h,
                                GL_RGBA, GL_UNSIGNED_BYTE, display_fb.srgb.data());
            }
        }

        glClear(GL_COLOR_BUFFER_BIT);
        glColor4f(1.f, 1.f, 1.f, 1.f); // reset — overlay may have changed it
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(-1, -1);
        glTexCoord2f(1, 0); glVertex2f( 1, -1);
        glTexCoord2f(1, 1); glVertex2f( 1,  1);
        glTexCoord2f(0, 1); glVertex2f(-1,  1);
        glEnd();

        // Draw debug help overlay (if enabled)
        if (s_app.debug.show_help_overlay) {
            render_help_overlay(win_w, win_h, s_app.debug, camera, s_app.volume_enabled,
                                s_app.use_dense_grid, s_app.active_scene_index, s_app.light_scale);
        }

        // Draw hover-cell overlay (when map mode toggles are active and
        // mouse is released for inspection).
        render_hover_cell_overlay(
            win_w, win_h,
            camera,
            s_app.debug,
            optix_renderer,
            s_app.mouse_captured);

        glfwSwapBuffers(window);
    }

    glDeleteTextures(1, &tex);
    glfwDestroyWindow(window);
    glfwTerminate();
}
