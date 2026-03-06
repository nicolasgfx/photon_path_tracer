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

// Prevent Windows min/max macros from interfering with std::numeric_limits
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "app/viewer.h"

#include "core/config.h"
#include "core/runtime_config.h"
#include "core/spectrum.h"
#include "debug/debug.h"
#include "debug/font_overlay.h"
#include "optix/optix_renderer.h"
#include "scene/scene_builder.h"
#include "scene/obj_loader.h"
#include "scene/pbrt/pbrt_loader.h"
#include "photon/photon_io.h"

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

// tinyexr for HDR EXR output — use stb zlib (already linked from stb_image)
#define TINYEXR_USE_MINIZ (0)
#define TINYEXR_USE_STB_ZLIB (1)
#define TINYEXR_IMPLEMENTATION
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4245 4702)  // tinyexr: signed/unsigned mismatch, unreachable code
#endif
#include "tinyexr.h"
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
    namespace fs = std::filesystem;
    fs::path p(obj_path);
    // For relative-to-scenes paths ("cornell_box/file.obj"), prepend SCENES_DIR.
    // For paths starting with ".." ("../tools/.../kroken/camera-1.pbrt"),
    // resolve relative to SCENES_DIR so we reach the actual scene folder.
    fs::path base = fs::path(SCENES_DIR) / p.parent_path();
    // Normalise (collapses ".." segments)
    return fs::weakly_canonical(base).string();
}

static constexpr const char* SAVED_CAMERA_FILENAME = "saved_camera.json";

bool save_camera_to_file(const Camera& cam, float yaw, float pitch, float roll,
                         float light_scale,
                         const std::string& scene_folder,
                         const PostFxParams* postfx) {
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
    f << "  \"roll\":          " << roll  << ",\n";
    f << "  \"light_scale\":   " << light_scale << ",\n";
    // DOF settings
    f << "  \"dof_enabled\":   " << (cam.dof_enabled ? "true" : "false") << ",\n";
    f << "  \"dof_focus_dist\": " << cam.dof_focus_dist << ",\n";
    f << "  \"dof_f_number\":  " << cam.dof_f_number << ",\n";
    f << "  \"sensor_height\": " << cam.sensor_height << ",\n";
    f << "  \"dof_focus_range\": " << cam.dof_focus_range;
    // Bloom / post-FX settings
    if (postfx) {
        f << ",\n";
        f << "  \"bloom_enabled\":   " << (postfx->bloom_enabled ? "true" : "false") << ",\n";
        f << "  \"bloom_intensity\": " << postfx->bloom_intensity << ",\n";
        f << "  \"bloom_radius_h\":  " << postfx->bloom_radius_h << ",\n";
        f << "  \"bloom_radius_v\":  " << postfx->bloom_radius_v << "\n";
    } else {
        f << "\n";
    }
    f << "}\n";
    f.close();
    std::printf("[Camera] Saved to %s\n", path.c_str());
    return true;
}

bool load_camera_from_file(Camera& cam, float& yaw, float& pitch, float& roll,
                           float& light_scale,
                           const std::string& scene_folder,
                           std::string* out_envmap_path,
                           float3* out_envmap_rotation,
                           float* out_envmap_scale,
                           PostFxParams* out_postfx,
                           float3* out_envmap_constant) {
    std::string path = scene_folder + "/" + SAVED_CAMERA_FILENAME;
    std::ifstream f(path);
    if (!f.is_open()) {
        // Fall back to camera.json (shipped scene defaults) if no user-saved file
        path = scene_folder + "/camera.json";
        f.open(path);
        if (!f.is_open()) return false;
    }

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
        else if (key == "roll")         { roll           = (float)std::atof(val.c_str()); }
        else if (key == "light_scale")  { light_scale    = (float)std::atof(val.c_str()); }
        // DOF settings
        else if (key == "dof_enabled")    { cam.dof_enabled    = (val == "true" || val == "1"); }
        else if (key == "dof_focus_dist") { cam.dof_focus_dist = (float)std::atof(val.c_str()); }
        else if (key == "dof_f_number")   { cam.dof_f_number   = (float)std::atof(val.c_str()); }
        else if (key == "sensor_height")  { cam.sensor_height  = (float)std::atof(val.c_str()); }
        else if (key == "dof_focus_range"){ cam.dof_focus_range= (float)std::atof(val.c_str()); }
        else if (key == "environment_map" && out_envmap_path) {
            // Strip quotes from string value
            std::string v = val;
            if (!v.empty() && v.front() == '"') v.erase(0, 1);
            if (!v.empty() && v.back()  == '"') v.pop_back();
            if (!v.empty()) {
                namespace fs = std::filesystem;
                if (fs::path(v).is_absolute())
                    *out_envmap_path = v;
                else
                    *out_envmap_path = scene_folder + "/" + v;
            }
        }
        else if (key == "environment_rotation_deg" && out_envmap_rotation) {
            parse_float3(val, *out_envmap_rotation);
        }
        else if (key == "environment_scale" && out_envmap_scale) {
            *out_envmap_scale = (float)std::atof(val.c_str());
        }
        else if (key == "envmap_constant" && out_envmap_constant) {
            parse_float3(val, *out_envmap_constant);
        }
        // Bloom / post-FX settings
        else if (key == "bloom_enabled" && out_postfx) {
            out_postfx->bloom_enabled = (val == "true" || val == "1");
        }
        else if (key == "bloom_intensity" && out_postfx) {
            out_postfx->bloom_intensity = (float)std::atof(val.c_str());
        }
        else if (key == "bloom_radius_h" && out_postfx) {
            out_postfx->bloom_radius_h = (float)std::atof(val.c_str());
        }
        else if (key == "bloom_radius_v" && out_postfx) {
            out_postfx->bloom_radius_v = (float)std::atof(val.c_str());
        }
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

// -- EXR (HDR) output ------------------------------------------------
bool write_exr(const std::string& filename,
               const std::vector<float>& hdr_rgba,
               int width, int height)
{
    namespace fs = std::filesystem;
    fs::path p(filename);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());

    // tinyexr expects separate R, G, B channels (bottom-to-top)
    std::vector<float> r(width * height), g(width * height), b(width * height);
    for (int y = 0; y < height; ++y) {
        int src_y = height - 1 - y;   // flip vertically (same as PNG path)
        for (int x = 0; x < width; ++x) {
            int si = (src_y * width + x) * 4;
            int di = y * width + x;
            r[di] = hdr_rgba[si + 0];
            g[di] = hdr_rgba[si + 1];
            b[di] = hdr_rgba[si + 2];
        }
    }

    const float* channels[] = { b.data(), g.data(), r.data() };   // EXR is BGR

    EXRHeader header;
    InitEXRHeader(&header);
    EXRImage image;
    InitEXRImage(&image);
    image.num_channels = 3;
    image.images       = (unsigned char**)channels;
    image.width        = width;
    image.height       = height;

    header.num_channels = 3;
    std::vector<EXRChannelInfo> ch(3);
    snprintf(ch[0].name, sizeof(ch[0].name), "B");
    snprintf(ch[1].name, sizeof(ch[1].name), "G");
    snprintf(ch[2].name, sizeof(ch[2].name), "R");
    header.channels = ch.data();

    std::vector<int> pixel_types(3, TINYEXR_PIXELTYPE_FLOAT);
    std::vector<int> requested(3, TINYEXR_PIXELTYPE_FLOAT);
    header.pixel_types           = pixel_types.data();
    header.requested_pixel_types = requested.data();

    const char* err = nullptr;
    int ret = SaveEXRImageToFile(&image, &header, filename.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        std::cerr << "[EXR] Failed to write " << filename;
        if (err) { std::cerr << ": " << err; FreeEXRErrorMessage(err); }
        std::cerr << "\n";
        return false;
    }
    std::cout << "[Output] Wrote " << filename << " (" << width << "x" << height << " HDR)\n";
    return true;
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
    // ---------- 2D orthographic projection for overlay ----------
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

    // ---------- Layout constants ----------
    // 2× scale gives ~14px glyphs – readable on 1080p+
    constexpr float scale  = 2.0f;
    constexpr float line_h = 14.0f;          // 7px glyph * 2.0 scale
    constexpr float gap_section = line_h * 0.45f;
    constexpr float col_w  = 280.f;          // width of one column (scaled px)
    constexpr float gutter = 30.f;           // gap between columns
    constexpr float pad    = 16.f;           // inner padding around content

    // Two-column box, centered on screen
    float box_w = pad + col_w + gutter + col_w + pad;
    float bx = ((float)win_w - box_w) * 0.5f;
    float by = floorf((float)win_h * 0.08f); // 8% from top

    // ---- Pre-compute content height so the background fits exactly ----
    constexpr float ln = line_h * 0.78f;     // normal line step
    // Left column: title + nav(4) + render(3) + stats(3) + dof(6) + light(1)
    float left_h  = line_h * 1.1f                       // title
                  + (gap_section + line_h + 4 * ln)      // navigation
                  + (gap_section + line_h + 3 * ln)      // rendering
                  + (gap_section + line_h + 3 * ln)      // statistics
                  + (gap_section + line_h + 6 * ln)      // dof
                  + (gap_section + line_h + 1 * ln);     // light
    // Right column: debug toggles(10+maybe filter) + volume(2) + scenes(N) + misc(1)
    int extra_filter = (debug.photon_filter != PhotonFilterMode::Off) ? 1 : 0;
    float right_h = (gap_section + line_h + (10 + extra_filter) * ln) // debug overlays
                  + (gap_section + line_h + 2 * ln)                   // volume
                  + (gap_section + line_h + NUM_SCENE_PROFILES * ln)  // scenes
                  + (gap_section + line_h + 1 * ln);                  // misc
    float content_h = (left_h > right_h) ? left_h : right_h;
    float box_h = pad + content_h + gap_section + line_h + pad; // +footer

    // ---------- Draw background ----------
    // Dark near-opaque background with subtle border
    auto draw_rect = [](float x, float y, float w, float h,
                        float r, float g, float b, float a) {
        glColor4f(r, g, b, a);
        glBegin(GL_QUADS);
        glVertex2f(x,     y);
        glVertex2f(x + w, y);
        glVertex2f(x + w, y + h);
        glVertex2f(x,     y + h);
        glEnd();
    };
    // Border (1 px visual) around the box
    draw_rect(bx - 1, by - 1, box_w + 2, box_h + 2,
              0.35f, 0.35f, 0.40f, 0.70f);
    // Solid dark fill
    draw_rect(bx, by, box_w, box_h,
              0.05f, 0.05f, 0.08f, 0.88f);

    // ---------- Text helpers ----------
    auto on_off = [](bool v) -> const char* { return v ? "ON " : "off"; };

    // Shadow-text: draw black offset copy first, then foreground
    auto draw_shadow_text = [&](float x, float y, const char* text,
                                float r, float g, float b, float a) {
        draw_overlay_text(x + 0.7f, y + 0.7f, text, 0.f, 0.f, 0.f, a * 0.6f);
        draw_overlay_text(x, y, text, r, g, b, a);
    };

    // Column cursors
    float lx  = bx + pad;              // left column x
    float rx  = bx + pad + col_w + gutter; // right column x
    float ly  = by + pad;              // left column y cursor
    float ry  = by + pad;              // right column y cursor

    glPushMatrix();
    glScalef(scale, scale, 1.f);
    // All coordinates are now in *scaled* space, so divide by scale
    float inv = 1.f / scale;

    // Handy lambdas (work in unscaled coords, emit at scaled positions)
    char buf[128];

    auto section = [&](float& cy, float cx, const char* title) {
        cy += gap_section;
        draw_shadow_text(cx * inv, cy * inv, title, 1.0f, 0.9f, 0.35f, 1.f);
        cy += line_h;
    };

    auto line = [&](float& cy, float cx, const char* text,
                    float r = 1.0f, float g = 1.0f, float b = 1.0f) {
        draw_shadow_text(cx * inv, cy * inv, text, r, g, b, 1.f);
        cy += line_h * 0.78f;
    };

    auto toggle_line = [&](float& cy, float cx,
                           const char* key, const char* label, bool on) {
        snprintf(buf, sizeof(buf), "%-5s %s [%s]", key, label, on_off(on));
        float cr = on ? 0.35f : 0.50f;
        float cg = on ? 1.00f : 0.50f;
        float cb = on ? 0.35f : 0.50f;
        draw_shadow_text(cx * inv, cy * inv, buf, cr, cg, cb, 1.f);
        cy += line_h * 0.78f;
    };

    // ===================== LEFT COLUMN =====================

    // Title
    draw_shadow_text(lx * inv, ly * inv,
                     "PHOTON PATH TRACER",
                     0.6f, 0.85f, 1.0f, 1.f);
    ly += line_h * 1.1f;

    // -- Navigation --
    section(ly, lx, "Navigation");
    line(ly, lx, "WASD    Move camera");
    line(ly, lx, "Mouse   Look around");
    line(ly, lx, "Shift   Fast move (3x)");
    line(ly, lx, "M       Toggle mouse capture");

    // -- Rendering --
    section(ly, lx, "Rendering");
    {
        snprintf(buf, sizeof(buf), "TAB   Mode: %s",
                 DebugState::render_mode_name(debug.current_mode));
        line(ly, lx, buf, 0.35f, 1.0f, 0.55f);
    }
    line(ly, lx, "R       Save snapshot (PNG + EXR)");
    line(ly, lx, "ESC     Cancel / release / quit");

    // -- Statistics & Guidance --
    section(ly, lx, "Statistics & Guidance");
    toggle_line(ly, lx, "T", "Guided path tracing", true);   // always shown
    line(ly, lx, "C       Toggle histogram-only");
    line(ly, lx, "S       Toggle stats overlay");

    // -- Depth of Field --
    section(ly, lx, "Depth of Field");
    toggle_line(ly, lx, "O", "Depth of Field", camera.dof_enabled);
    {
        snprintf(buf, sizeof(buf), "[/]   f-number: f/%.1f", camera.dof_f_number);
        line(ly, lx, buf);
        snprintf(buf, sizeof(buf), ",/.   Focus dist: %.4f", camera.dof_focus_dist);
        line(ly, lx, buf);
    }
    line(ly, lx, "F       Auto-focus (center)");
    line(ly, lx, "MMB     Auto-focus (cursor)");

    // -- Light Brightness --
    section(ly, lx, "Light Brightness");
    {
        snprintf(buf, sizeof(buf), "+/-   Brightness: %.2fx", light_scale);
        line(ly, lx, buf);
    }

    // ===================== RIGHT COLUMN =====================

    // -- Debug Overlays --
    section(ry, rx, "Debug Overlays");
    toggle_line(ry, rx, "F1",  "All photons",      debug.show_photon_points);
    toggle_line(ry, rx, "F2",  "Cycle photon filt", debug.photon_filter != PhotonFilterMode::Off);
    if (debug.photon_filter != PhotonFilterMode::Off) {
        snprintf(buf, sizeof(buf), "        Filter: %s",
                 photon_filter_name(debug.photon_filter));
        line(ry, rx, buf, 0.25f, 0.80f, 1.0f);
    }
    toggle_line(ry, rx, "F3",  "Guide map",        false);
    toggle_line(ry, rx, "F4",  "Hash grid",        debug.show_hash_grid);
    toggle_line(ry, rx, "F5",  "Photon dirs",      debug.show_photon_dirs);
    toggle_line(ry, rx, "F6",  "PDFs",             debug.show_pdfs);
    toggle_line(ry, rx, "F7",  "Radius sphere",    debug.show_radius_sphere);
    toggle_line(ry, rx, "F8",  "MIS weights",      debug.show_mis_weights);
    toggle_line(ry, rx, "F9",  "Spectral color",   debug.spectral_coloring);
    toggle_line(ry, rx, "F11", "Photon heatmap",   debug.show_photon_heatmap);

    // -- Volume Scattering --
    section(ry, rx, "Volume Scattering");
    toggle_line(ry, rx, "V", "Volume",     volume_enabled);
    toggle_line(ry, rx, "G", "Dense Grid", use_dense_grid);

    // -- Scenes --
    section(ry, rx, "Scenes (1-9, 0)");
    for (int i = 0; i < NUM_SCENE_PROFILES; ++i) {
        int display_key = (i < 9) ? (i + 1) : 0;  // keys 1-9, then 0
        snprintf(buf, sizeof(buf), "%d     %s", display_key,
                 SCENE_PROFILES[i].display_name);
        bool active = (i == active_scene_index);
        float cr = active ? 0.35f : 0.50f;
        float cg = active ? 1.00f : 0.50f;
        float cb = active ? 0.35f : 0.50f;
        draw_shadow_text(rx * inv, ry * inv, buf, cr, cg, cb, 1.f);
        ry += line_h * 0.78f;
    }

    // -- Misc --
    section(ry, rx, "Misc");
    line(ry, rx, "F10     Save camera position");

    // ---- Footer ----
    float footer_y = (ly > ry ? ly : ry) + gap_section;
    float center_x = bx + box_w * 0.5f - 60.f; // approximate centering
    draw_shadow_text(center_x * inv, footer_y * inv,
                     "H  Toggle this overlay",
                     0.50f, 0.50f, 0.55f, 0.8f);

    glPopMatrix();

    // ---------- Restore GL state ----------
    glDisable(GL_BLEND);
    glEnable(GL_TEXTURE_2D);
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

// ── Stats overlay (S key) ────────────────────────────────────────────
// Displays live renderer statistics in the top-right corner.
// Gated by ENABLE_STATS at compile time and show_stats_overlay at runtime.
static void render_stats_overlay(int win_w, int win_h,
                                 const AppState& app,
                                 const OptixRenderer* renderer)
{
    if constexpr (!ENABLE_STATS) return;
    if (!app.show_stats_overlay || !renderer) return;

    // Gather stats snapshot
    auto rs = renderer->gather_stats("");

    // Format lines
    char buf[256];
    std::vector<std::string> lines;

    lines.push_back("=== Renderer Stats ===");
    snprintf(buf, sizeof(buf), "SPP: %d", rs.accumulated_spp);
    lines.push_back(buf);

    snprintf(buf, sizeof(buf), "Guided: %s  Hist-only: %s",
             app.guided_enabled ? "ON" : "off",
             app.histogram_only ? "ON" : "off");
    lines.push_back(buf);

    snprintf(buf, sizeof(buf), "Guide frac: %.2f", renderer->get_guide_fraction());
    lines.push_back(buf);

    lines.push_back("");
    lines.push_back("--- Photon Map ---");
    snprintf(buf, sizeof(buf), "Emitted: %d  Stored: %d",
             rs.photons_emitted, rs.photons_stored);
    lines.push_back(buf);
    snprintf(buf, sizeof(buf), "Global: %d  Caustic: %d  Targeted: %d",
             rs.noncaustic_stored, rs.global_caustic_stored, rs.caustic_stored);
    lines.push_back(buf);
    snprintf(buf, sizeof(buf), "Radii  gather: %.5f  caustic: %.5f",
             rs.gather_radius, rs.caustic_radius);
    lines.push_back(buf);

    lines.push_back("");
    lines.push_back("--- Hardware ---");
    snprintf(buf, sizeof(buf), "GPU: %s", renderer->gpu_name().c_str());
    lines.push_back(buf);
    snprintf(buf, sizeof(buf), "VRAM: %zu MB  SMs: %d",
             renderer->gpu_vram_total() / (1024 * 1024),
             renderer->gpu_sm_count());
    lines.push_back(buf);
    snprintf(buf, sizeof(buf), "Compute: %d.%d",
             renderer->gpu_cc_major(), renderer->gpu_cc_minor());
    lines.push_back(buf);

    if (app.last_render_ms > 0.0) {
        lines.push_back("");
        lines.push_back("--- Timing ---");
        snprintf(buf, sizeof(buf), "Render: %.0f ms", app.last_render_ms);
        lines.push_back(buf);
    }

    // Compute box dimensions
    float scale   = 1.12f;
    float line_h  = 7.84f;
    float box_h   = (float)lines.size() * line_h + 16.f;
    float box_w   = 260.f;
    float bx      = (float)win_w - box_w - 10.f;
    float by      = 10.f;

    // Set up 2D projection
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

    // Background box
    glColor4f(0.0f, 0.0f, 0.0f, 0.7f);
    glBegin(GL_QUADS);
    glVertex2f(bx,         by);
    glVertex2f(bx + box_w, by);
    glVertex2f(bx + box_w, by + box_h);
    glVertex2f(bx,         by + box_h);
    glEnd();

    // Draw text lines
    glPushMatrix();
    glTranslatef(bx + 8.f, by + 8.f, 0.f);
    glScalef(scale, scale, 1.f);
    float ly = 0.f;
    for (const auto& line : lines) {
        if (line.empty()) {
            ly += line_h / scale * 0.5f;
            continue;
        }
        bool is_header = (line[0] == '=' || line[0] == '-');
        if (is_header)
            draw_overlay_text(0, ly, line.c_str(), 0.3f, 0.9f, 0.3f, 1.f);
        else
            draw_overlay_text(0, ly, line.c_str(), 0.9f, 0.9f, 0.9f, 1.f);
        ly += line_h / scale;
    }
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
    const HashGrid& grid = optix_renderer.dm_hash_grid();
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

    // Any key press counts as user interaction for idle tracking
    s_app.last_input_time = std::chrono::steady_clock::now();

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

    // Q is now used for camera roll (polled per-frame in WASD section)

    // "R" -> save debug pictures (direction map, snapshot, stats)
    if (key == GLFW_KEY_R) {
        s_app.snapshot_requested = true;
        printf("[Snapshot] R: saving debug pictures...\n");
        return;
    }

    // "T" -> screenshot sequence using current settings (same as idle rendering)
    if (key == GLFW_KEY_T) {
        s_app.render_key_mode = AppState::RenderKeyMode::T_OptScreenshot;
        s_app.render_key_next_screenshot_spp = 1;
        s_app.render_key_output_dir.clear();
        s_app.render_key_requested = true;
        // Do NOT override guided/clamp settings — use whatever the normal
        // idle rendering mode would use (matches default behaviour exactly).
        printf("[Render] T: screenshot sequence (current settings), restarting...\n");
        return;
    }

    // "Z" -> unoptimized screenshot sequence (no guide, no clamp), restarts from frame 0
    if (key == GLFW_KEY_Z) {
        s_app.render_key_mode = AppState::RenderKeyMode::Z_UnoptScreenshot;
        s_app.render_key_next_screenshot_spp = 1;
        s_app.render_key_output_dir.clear();
        s_app.render_key_requested = true;
        s_app.guided_enabled = false;
        s_app.spectral_clamp_enabled = false;
        if (g_active_optix_renderer) {
            g_active_optix_renderer->set_guide_fraction(0.0f);
            g_active_optix_renderer->set_spectral_clamp_enabled(false);
        }
        printf("[Render] Z: unoptimized screenshot sequence, restarting...\n");
        return;
    }

    // "F12" -> save snapshot (PNG + EXR) — manual screenshot
    if (key == GLFW_KEY_F12) {
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
        // Dense grid is always active now (no hash grid fallback)
        printf("[Gather] Dense grid: %s\n", s_app.use_dense_grid ? "ON" : "OFF");
        s_app.camera_moved  = true;  // reset accumulation
        return;
    }

    // Scene switching – keys 1-9, 0
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
    if (key == GLFW_KEY_0) {
        int idx = 9;  // key 0 = profile index 9 (Bathroom)
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

    // ── Bloom controls ──────────────────────────────────────────────
    // B -> toggle bloom on/off
    if (key == GLFW_KEY_B) {
        s_app.postfx.bloom_enabled = !s_app.postfx.bloom_enabled;
        if (g_active_optix_renderer)
            g_active_optix_renderer->set_postfx_params(s_app.postfx);
        printf("[Bloom] %s  (intensity=%.2f, radius H=%.1f V=%.1f)\n",
               s_app.postfx.bloom_enabled ? "ON" : "OFF",
               s_app.postfx.bloom_intensity,
               s_app.postfx.bloom_radius_h,
               s_app.postfx.bloom_radius_v);
        s_app.camera_moved = true;
        return;
    }
    // Numpad 4/6 -> adjust bloom horizontal radius
    if (key == GLFW_KEY_KP_4) {
        s_app.postfx.bloom_radius_h = fmaxf(1.f, s_app.postfx.bloom_radius_h - 5.f);
        if (g_active_optix_renderer) g_active_optix_renderer->set_postfx_params(s_app.postfx);
        printf("[Bloom] Radius H: %.1f\n", s_app.postfx.bloom_radius_h);
        s_app.camera_moved = true;
        return;
    }
    if (key == GLFW_KEY_KP_6) {
        s_app.postfx.bloom_radius_h = fminf(200.f, s_app.postfx.bloom_radius_h + 5.f);
        if (g_active_optix_renderer) g_active_optix_renderer->set_postfx_params(s_app.postfx);
        printf("[Bloom] Radius H: %.1f\n", s_app.postfx.bloom_radius_h);
        s_app.camera_moved = true;
        return;
    }
    // Numpad 2/8 -> adjust bloom vertical radius
    if (key == GLFW_KEY_KP_2) {
        s_app.postfx.bloom_radius_v = fmaxf(1.f, s_app.postfx.bloom_radius_v - 5.f);
        if (g_active_optix_renderer) g_active_optix_renderer->set_postfx_params(s_app.postfx);
        printf("[Bloom] Radius V: %.1f\n", s_app.postfx.bloom_radius_v);
        s_app.camera_moved = true;
        return;
    }
    if (key == GLFW_KEY_KP_8) {
        s_app.postfx.bloom_radius_v = fminf(200.f, s_app.postfx.bloom_radius_v + 5.f);
        if (g_active_optix_renderer) g_active_optix_renderer->set_postfx_params(s_app.postfx);
        printf("[Bloom] Radius V: %.1f\n", s_app.postfx.bloom_radius_v);
        s_app.camera_moved = true;
        return;
    }
    // Numpad 5/+ -> adjust bloom intensity
    if (key == GLFW_KEY_KP_5) {
        s_app.postfx.bloom_intensity = fmaxf(0.05f, s_app.postfx.bloom_intensity * 0.8f);
        if (g_active_optix_renderer) g_active_optix_renderer->set_postfx_params(s_app.postfx);
        printf("[Bloom] Intensity: %.2f\n", s_app.postfx.bloom_intensity);
        s_app.camera_moved = true;
        return;
    }
    if (key == GLFW_KEY_KP_DECIMAL) {
        s_app.postfx.bloom_intensity = fminf(5.0f, s_app.postfx.bloom_intensity * 1.25f);
        if (g_active_optix_renderer) g_active_optix_renderer->set_postfx_params(s_app.postfx);
        printf("[Bloom] Intensity: %.2f\n", s_app.postfx.bloom_intensity);
        s_app.camera_moved = true;
        return;
    }

    // P -> retrace photons (rebuild photon map)
    if (key == GLFW_KEY_P) {
        s_app.photon_retrace_requested = true;
        s_app.camera_moved  = true;  // reset accumulation
        printf("[Photon] Retrace requested (rebuild photon maps)\n");
        return;
    }

    // N -> render animation sequence
    if (key == GLFW_KEY_N) {
        s_app.animation_requested = true;
        printf("[Animation] Sequence requested\n");
        return;
    }

    // X -> toggle spectral outlier clamp
    if (key == GLFW_KEY_X) {
        s_app.spectral_clamp_enabled = !s_app.spectral_clamp_enabled;
        if (g_active_optix_renderer)
            g_active_optix_renderer->set_spectral_clamp_enabled(s_app.spectral_clamp_enabled);
        // camera_moved resets accumulation via clear_buffers() in the main loop;
        // spectral_ref_buffer is NOT cleared by clear_buffers (it's a reference).
        s_app.camera_moved = true;
        printf("[Clamp] Spectral outlier clamp %s\n",
               s_app.spectral_clamp_enabled ? "ENABLED" : "DISABLED");
        return;
    }

    // C -> toggle histogram-only conclusions (skip expensive analysis)
    if (key == GLFW_KEY_C) {
        if (!s_app.guided_enabled) {
            printf("[Stats] Cannot toggle histogram-only: guided tracing is off\n");
            return;
        }
        s_app.histogram_only = !s_app.histogram_only;
        s_app.camera_moved = true;  // reset accumulation
        printf("[Stats] Histogram-only conclusions %s\n",
               s_app.histogram_only ? "ON" : "OFF");
        return;
    }

    // S -> toggle stats overlay
    if (key == GLFW_KEY_S) {
        s_app.show_stats_overlay = !s_app.show_stats_overlay;
        printf("[Stats] Overlay %s\n",
               s_app.show_stats_overlay ? "SHOWN" : "HIDDEN");
        return;
    }

    // F10 -> save current camera position to scene folder
    if (key == GLFW_KEY_F10 && g_active_camera) {
        int idx = s_app.active_scene_index;
        if (idx >= 0 && idx < NUM_SCENE_PROFILES) {
            std::string folder = scene_folder_from_profile(SCENE_PROFILES[idx].obj_path);
            save_camera_to_file(*g_active_camera, s_app.yaw, s_app.pitch,
                                s_app.roll, s_app.light_scale, folder, &s_app.postfx);
        } else {
            // Derive folder from the active options scene_file
            if (g_active_options) {
                fs::path p(g_active_options->scene_file);
                std::string folder = p.parent_path().string();
                save_camera_to_file(*g_active_camera, s_app.yaw, s_app.pitch,
                                    s_app.roll, s_app.light_scale, folder, &s_app.postfx);
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
    // Any click counts as user interaction for idle tracking
    if (action == GLFW_PRESS)
        s_app.last_input_time = std::chrono::steady_clock::now();

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

    // Sync initial AppState into the renderer
    optix_renderer.set_volume_enabled(s_app.volume_enabled);
    optix_renderer.set_guide_fraction(
        s_app.guided_enabled ? DEFAULT_GUIDE_FRACTION : 0.0f);
    // Dense grid is always active now (no toggle needed)

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
    std::cout << "  ESC = release mouse / quit | Q = quit\n";
    std::cout << "  F1-F9 = debug toggles | TAB = cycle mode\n";
    std::cout << "  R = save snapshot (PNG + EXR)\n";
    std::cout << "  H = toggle help overlay\n";
    std::cout << "  1-9 = switch scene\n";
    std::cout << "  +/- = adjust light brightness (re-traces photons)\n";
    std::cout << "  V = toggle volume scattering | O = toggle DOF | [/] = blur | ,/. = focus dist\n";
    std::cout << "  F10 = save camera position to scene folder\n";

    FrameBuffer display_fb;
    display_fb.resize(win_w, win_h);

    int frame = 0;
    auto last_time = std::chrono::high_resolution_clock::now();

    // Initialise idle tracking
    s_app.last_input_time = std::chrono::steady_clock::now();
    s_app.base_num_photons = opt.config.num_photons;

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

        // ── Runtime scene switch (keys 1-9, 0) ──────────────────────
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

            // Dispatch by file extension (.pbrt vs .obj)
            bool load_ok = false;
            {
                std::string ext;
                auto dot = obj_path.rfind('.');
                if (dot != std::string::npos) ext = obj_path.substr(dot);
                for (auto& c : ext) c = (char)std::tolower((unsigned char)c);
                if (ext == ".pbrt")
                    load_ok = load_pbrt(obj_path, new_scene);
                else
                    load_ok = load_obj(obj_path, new_scene);
            }

            if (!load_ok) {
                std::cerr << "[Error] Failed to load: " << obj_path << "\n";
            } else {
                // Ensure instancing metadata is populated
                if (new_scene.meshes.empty() && !new_scene.triangles.empty()) {
                    MeshDescriptor m0;
                    m0.tri_offset = 0;
                    m0.tri_count  = (uint32_t)new_scene.triangles.size();
                    new_scene.meshes.push_back(m0);
                    InstanceDescriptor inst0;
                    inst0.mesh_id = 0;
                    std::memset(inst0.transform, 0, sizeof(inst0.transform));
                    inst0.transform[0] = 1.f; inst0.transform[5] = 1.f; inst0.transform[10] = 1.f;
                    new_scene.instances.push_back(inst0);
                }
                if (!prof.is_reference)
                    new_scene.normalize_to_reference();
                if (prof.rotate_x_180)
                    new_scene.rotate_x_180();
                constexpr size_t BVH_TRI_LIMIT = 10'000'000;
                if (new_scene.triangles.size() <= BVH_TRI_LIMIT)
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

                // Scan emissive range for adaptive bloom
                scene.compute_emissive_radiance_range(
                    s_app.postfx.bloom_scene_min_Le,
                    s_app.postfx.bloom_scene_max_Le);

                // Reset camera to scene profile defaults (or PBRT camera)
                if (scene.pbrt_cam_valid) {
                    camera.position = scene.pbrt_cam_position;
                    camera.look_at  = scene.pbrt_cam_look_at;
                    camera.up       = scene.pbrt_cam_up;
                    camera.fov_deg  = scene.pbrt_cam_fov;
                } else {
                    camera.position = make_f3(prof.cam_pos[0], prof.cam_pos[1], prof.cam_pos[2]);
                    camera.look_at  = make_f3(prof.cam_lookat[0], prof.cam_lookat[1], prof.cam_lookat[2]);
                    camera.fov_deg  = prof.cam_fov;
                }
                camera.width    = win_w;
                camera.height   = win_h;
                camera.update();

                // Envmap data from PBRT loader (may be overridden by saved camera)
                std::string new_envmap_path      = scene.pbrt_envmap_path;
                float3 new_envmap_rotation       = scene.pbrt_envmap_rotation;
                float  new_envmap_scale          = scene.pbrt_envmap_scale;
                float3 new_envmap_constant       = scene.pbrt_envmap_constant;
                {
                    std::string folder = scene_folder_from_profile(prof.obj_path);
                    float saved_yaw = 0.f, saved_pitch = 0.f, saved_roll = 0.f;
                    float saved_light = DEFAULT_LIGHT_SCALE;
                    PostFxParams loaded_postfx;
                    if (load_camera_from_file(camera, saved_yaw, saved_pitch,
                                              saved_roll, saved_light, folder,
                                              &new_envmap_path, &new_envmap_rotation,
                                              &new_envmap_scale, &loaded_postfx,
                                              &new_envmap_constant)) {
                        s_app.yaw   = saved_yaw;
                        s_app.pitch = saved_pitch;
                        s_app.roll  = saved_roll;
                        s_app.light_scale         = saved_light;
                        s_app.light_scale_changed = true;
                        // Preserve the scene-scanned emissive range (not in JSON)
                        loaded_postfx.bloom_scene_min_Le = s_app.postfx.bloom_scene_min_Le;
                        loaded_postfx.bloom_scene_max_Le = s_app.postfx.bloom_scene_max_Le;
                        s_app.postfx = loaded_postfx;
                        optix_renderer.set_postfx_params(s_app.postfx);
                    } else {
                        // Sync yaw/pitch from new camera direction
                        float3 fwd = normalize(camera.look_at - camera.position);
                        s_app.yaw   = atan2f(fwd.x, -fwd.z);
                        s_app.pitch = asinf(fmaxf(-1.f, fminf(1.f, fwd.y)));
                        s_app.light_scale = DEFAULT_LIGHT_SCALE;
                        optix_renderer.set_postfx_params(s_app.postfx);
                    }
                }

                // ── Load environment map (if specified in camera config) ──
                scene.envmap.reset();  // clear old envmap
                scene.envmap_selection_prob = 0.f;
                if (!new_envmap_path.empty()) {
                    scene.envmap = std::make_shared<EnvironmentMap>();
                    if (load_environment_map(new_envmap_path, new_envmap_scale,
                                             new_envmap_rotation, *scene.envmap)) {
                        scene.envmap->scene_center = scene.scene_bounding_center();
                        scene.envmap->scene_radius = scene.scene_bounding_radius() * 1.1f;
                        scene.compute_envmap_selection_prob();
                        std::printf("[EnvMap] Loaded: %s (%dx%d, sel_p=%.3f)\n",
                                    new_envmap_path.c_str(),
                                    scene.envmap->width, scene.envmap->height,
                                    scene.envmap_selection_prob);
                    } else {
                        std::printf("[Warning] Failed to load envmap: %s\n",
                                    new_envmap_path.c_str());
                        scene.envmap.reset();
                    }
                } else if (new_envmap_constant.x > 0.f || new_envmap_constant.y > 0.f
                           || new_envmap_constant.z > 0.f) {
                    scene.envmap = std::make_shared<EnvironmentMap>();
                    if (create_constant_envmap(new_envmap_constant.x,
                                              new_envmap_constant.y,
                                              new_envmap_constant.z,
                                              new_envmap_scale,
                                              new_envmap_rotation, *scene.envmap)) {
                        scene.envmap->scene_center = scene.scene_bounding_center();
                        scene.envmap->scene_radius = scene.scene_bounding_radius() * 1.1f;
                        scene.compute_envmap_selection_prob();
                    } else {
                        scene.envmap.reset();
                    }
                }
                optix_renderer.upload_envmap_data(scene);

                // ── Wipe stale guidance data before rebuilding ──
                optix_renderer.clear_guidance_buffers();

                // ── Trace photons on GPU (after envmap is available) ──
                auto tp0 = std::chrono::high_resolution_clock::now();
                optix_renderer.trace_photons(scene, opt.config);
                optix_renderer.build_direction_map(camera, /*spp_seed=*/0);
                auto tp1 = std::chrono::high_resolution_clock::now();
                double photon_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

                s_app.active_cam_speed    = prof.cam_speed;
                s_app.active_scene_index  = idx;
                s_app.camera_moved        = true;
                s_app.last_input_time     = std::chrono::steady_clock::now();
                s_app.base_num_photons    = opt.config.num_photons;
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

            std::cout << "[Photon] Re-tracing photon maps (seed "
                      << s_app.idle_photon_seed << ")...\n";
            auto tp0 = std::chrono::high_resolution_clock::now();
            optix_renderer.trace_photons(
                scene, opt.config, 0.f, s_app.idle_photon_seed++);
            optix_renderer.build_direction_map(camera, /*spp_seed=*/0);
            auto tp1 = std::chrono::high_resolution_clock::now();
            double photon_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();
            std::cout << "[Photon] Retrace done in " << photon_ms << " ms\n";

            frame = 0;
            optix_renderer.clear_buffers();
            s_app.camera_moved  = true;
            s_app.last_input_time = std::chrono::steady_clock::now();
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

            // Update adaptive bloom range after Le rescale
            scene.compute_emissive_radiance_range(
                s_app.postfx.bloom_scene_min_Le,
                s_app.postfx.bloom_scene_max_Le);
            optix_renderer.set_postfx_params(s_app.postfx);

            // Re-upload materials and emitter CDF
            optix_renderer.upload_scene_data(scene);
            optix_renderer.upload_emitter_data(scene);

            // Re-trace photons with updated emission
            std::cout << "[Light] Re-tracing photons at " << s_app.light_scale << "x brightness...\n";
            auto tp0 = std::chrono::high_resolution_clock::now();
            optix_renderer.trace_photons(
                scene, opt.config, 0.f, s_app.idle_photon_seed++);
            auto tp1 = std::chrono::high_resolution_clock::now();
            double photon_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();
            std::cout << "[Light] Photon re-trace done in " << photon_ms << " ms\n";

            s_app.camera_moved  = true;
            s_app.last_input_time = std::chrono::steady_clock::now();
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
                s_app.last_input_time = std::chrono::steady_clock::now();
            }
        }

        // -- WASD movement -----------------------------------------------
        {
            // Forward direction (camera looks along -w in its frame)
            float3 forward = make_f3(
                sinf(s_app.yaw) * cosf(s_app.pitch),
                sinf(s_app.pitch),
                -cosf(s_app.yaw) * cosf(s_app.pitch));
            float3 right_unrolled = normalize(cross(forward, make_f3(0, 1, 0)));
            float3 up_perp = cross(right_unrolled, forward); // perpendicular up (no roll)
            // Apply roll: rotate right/up around forward axis
            float3 right = right_unrolled * cosf(s_app.roll) + up_perp * sinf(s_app.roll);
            float3 up_dir = up_perp * cosf(s_app.roll) - right_unrolled * sinf(s_app.roll);

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

            // Q/E -> camera roll (CCW / CW)
            constexpr float kRollSpeed = 1.0f; // radians per second
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
                s_app.roll += kRollSpeed * dt;
                s_app.camera_moved = true;
                s_app.last_input_time = std::chrono::steady_clock::now();
            }
            if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
                s_app.roll -= kRollSpeed * dt;
                s_app.camera_moved = true;
                s_app.last_input_time = std::chrono::steady_clock::now();
            }

            if (move.x != 0.f || move.y != 0.f || move.z != 0.f) {
                camera.position = camera.position + move;
                s_app.camera_moved = true;
                s_app.last_input_time = std::chrono::steady_clock::now();
            }

            // Update look_at from yaw/pitch, up from roll
            camera.look_at = camera.position + forward;
            camera.up = up_dir;
            camera.width   = win_w;
            camera.height  = win_h;
            camera.update();
        }

        // Reset progressive accumulation if camera moved
        if (s_app.camera_moved) {
            // Skip camera-moved reset if a render-key transition is pending
            // (R/T/Z key press on the same frame as mouse movement)
            if (s_app.render_key_requested) {
                s_app.camera_moved = false;
            } else {
                // ── Exit idle/full-quality mode on any user interaction ──
                if (s_app.idle_rendering_active) {
                    opt.config.num_photons = s_app.base_num_photons;
                    optix_renderer.set_preview_mode(true);
                    optix_renderer.trace_photons(
                        scene, opt.config, 0.f, s_app.idle_photon_seed++);
                    // Direction map + spectral ref are only built in quality mode
                    s_app.idle_rendering_active = false;
                }
                s_app.render_key_mode = AppState::RenderKeyMode::None;
                s_app.render_key_output_dir.clear();
                optix_renderer.clear_buffers();
                optix_renderer.clear_guidance_buffers();  // wipe stale dir map + spectral ref
                frame = 0;
                s_app.camera_moved = false;
            }
        }

        // ── Idle-to-full-quality transition ──────────────────────────
        {
            auto now_steady = std::chrono::steady_clock::now();
            float idle_sec = std::chrono::duration<float>(
                now_steady - s_app.last_input_time).count();

            if (idle_sec > IDLE_TIMEOUT_SEC && !s_app.idle_rendering_active) {
                // Boost photon budget and re-trace for higher quality
                opt.config.num_photons = s_app.base_num_photons;
                optix_renderer.trace_photons(
                    scene, opt.config, 0.f, s_app.idle_photon_seed++);
                optix_renderer.build_direction_map(camera, /*spp_seed=*/0);
                optix_renderer.set_preview_mode(false);
                optix_renderer.clear_buffers();

                frame = 0;
                s_app.idle_rendering_active = true;
                s_app.render_start_time = std::chrono::steady_clock::now();
                s_app.render_timing_active = true;
                printf("[Idle] Full-quality mode\n");
            }
        }

        // ── Handle R/T/Z render-key transition ──────────────────────
        if (s_app.render_key_requested) {
            s_app.render_key_requested = false;

            // Retrace photons and enter full-quality mode
            opt.config.num_photons = s_app.base_num_photons;
            optix_renderer.trace_photons(
                scene, opt.config, 0.f, s_app.idle_photon_seed++);

            // Build direction map only when guided (R/T) — skip for Z
            if (s_app.render_key_mode != AppState::RenderKeyMode::Z_UnoptScreenshot)
                optix_renderer.build_direction_map(camera, /*spp_seed=*/0);

            optix_renderer.set_preview_mode(false);
            optix_renderer.clear_buffers();
            frame = 0;
            s_app.idle_rendering_active = true;
            s_app.render_start_time = std::chrono::steady_clock::now();
            s_app.render_timing_active = true;

            const char* tag =
                s_app.render_key_mode == AppState::RenderKeyMode::T_OptScreenshot   ? "T (opt+screenshots)" :
                s_app.render_key_mode == AppState::RenderKeyMode::Z_UnoptScreenshot ? "Z (unopt+screenshots)" :
                "?";
            printf("[Render] %s: full-quality mode active\n", tag);
        }

        // ── Handle "N" key: render animation sequence ────────────────
        if (s_app.animation_requested) {
            s_app.animation_requested = false;

            constexpr int   ANIM_FPS           = 30;
            constexpr float ANIM_DURATION_SEC  = 5.0f;
            constexpr int   ANIM_TOTAL_FRAMES  = (int)(ANIM_DURATION_SEC * ANIM_FPS); // 150
            constexpr int   ANIM_SPP_PER_FRAME = 5000;
            constexpr float ANIM_TRAVEL_DIST   = 0.25f; // 25% of scene length (1.0)

            // Compute forward direction from current yaw/pitch
            float3 anim_forward = make_f3(
                sinf(s_app.yaw) * cosf(s_app.pitch),
                sinf(s_app.pitch),
                -cosf(s_app.yaw) * cosf(s_app.pitch));
            float3 anim_step = anim_forward * (ANIM_TRAVEL_DIST / (float)ANIM_TOTAL_FRAMES);

            // Save original camera position to restore after
            float3 anim_start_pos = camera.position;

            // Build timestamped output folder: output/animation_YYYYMMDD_HHMMSS/
            auto anim_now = std::chrono::system_clock::now();
            std::time_t anim_t = std::chrono::system_clock::to_time_t(anim_now);
            std::tm anim_tm;
            localtime_s(&anim_tm, &anim_t);
            char anim_ts[64];
            std::strftime(anim_ts, sizeof(anim_ts), "%Y%m%d_%H%M%S", &anim_tm);
            std::string anim_dir = std::string("output/animation_") + anim_ts;
            fs::create_directories(anim_dir);

            printf("[Animation] Rendering %d frames, %d spp each, travel=%.3f, output=%s\n",
                   ANIM_TOTAL_FRAMES, ANIM_SPP_PER_FRAME, ANIM_TRAVEL_DIST, anim_dir.c_str());

            auto anim_wall_start = std::chrono::steady_clock::now();

            for (int af = 0; af < ANIM_TOTAL_FRAMES; ++af) {
                // Position camera for this frame
                camera.position = anim_start_pos + anim_step * (float)af;
                camera.look_at  = camera.position + anim_forward;
                camera.update();

                // Clear accumulation and render ANIM_SPP_PER_FRAME passes
                optix_renderer.clear_buffers();
                for (int spp_i = 0; spp_i < ANIM_SPP_PER_FRAME; ++spp_i) {
                    // Periodic photon + direction map rebuild (matches render_final)
                    if (DEFAULT_GUIDE_REMAP_INTERVAL > 0 && spp_i > 0
                        && (spp_i % DEFAULT_GUIDE_REMAP_INTERVAL) == 0) {
                        optix_renderer.trace_photons(
                            scene, opt.config, 0.f, /*photon_map_seed=*/spp_i);
                        optix_renderer.build_direction_map(camera, /*spp_seed=*/spp_i);
                    }
                    optix_renderer.render_debug_frame(
                        camera, spp_i, s_app.debug.current_mode, 1);
                }

                // Download and save PNG
                optix_renderer.download_framebuffer(display_fb);
                char frame_name[128];
                std::snprintf(frame_name, sizeof(frame_name),
                              "%s/frame_%04d.png", anim_dir.c_str(), af);
                write_png(std::string(frame_name), display_fb);

                // Blit to screen so user can see progress
                glBindTexture(GL_TEXTURE_2D, tex);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                display_fb.width, display_fb.height,
                                GL_RGBA, GL_UNSIGNED_BYTE,
                                display_fb.srgb.data());
                glClear(GL_COLOR_BUFFER_BIT);
                glColor4f(1.f, 1.f, 1.f, 1.f);
                glBegin(GL_QUADS);
                glTexCoord2f(0, 0); glVertex2f(-1, -1);
                glTexCoord2f(1, 0); glVertex2f( 1, -1);
                glTexCoord2f(1, 1); glVertex2f( 1,  1);
                glTexCoord2f(0, 1); glVertex2f(-1,  1);
                glEnd();
                glfwSwapBuffers(window);
                glfwPollEvents();

                // Check if window should close (ESC during animation)
                if (glfwWindowShouldClose(window)) break;

                float elapsed = std::chrono::duration<float>(
                    std::chrono::steady_clock::now() - anim_wall_start).count();
                printf("[Animation] Frame %d/%d saved (%s) [%.1fs elapsed]\n",
                       af + 1, ANIM_TOTAL_FRAMES, frame_name, elapsed);
            }

            float total_sec = std::chrono::duration<float>(
                std::chrono::steady_clock::now() - anim_wall_start).count();
            printf("[Animation] Done — %d frames in %.1fs  (%s)\n",
                   ANIM_TOTAL_FRAMES, total_sec, anim_dir.c_str());

            // Restore camera to starting position
            camera.position = anim_start_pos;
            camera.look_at  = anim_start_pos + anim_forward;
            camera.update();
            optix_renderer.clear_buffers();
            frame = 0;
            s_app.camera_moved = false;
        }

        // ── Handle "F12" key: save timestamped snapshot (PNG + EXR) ─
        if (s_app.snapshot_requested) {
            s_app.snapshot_requested = false;

            // Build timestamped subfolder: output/snapshot_YYYYMMDD_HHMMSS/
            auto now_tp = std::chrono::system_clock::now();
            std::time_t now_t = std::chrono::system_clock::to_time_t(now_tp);
            std::tm tm_buf;
            localtime_s(&tm_buf, &now_t);
            char ts[64];
            std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tm_buf);
            std::string snap_dir = std::string("output/snapshot_") + ts;
            fs::create_directories(snap_dir);
            std::string prefix = snap_dir + "/snapshot";

            // Save PNG
            std::string png_path = prefix + ".png";
            optix_renderer.download_framebuffer(display_fb);
            write_png(png_path, display_fb);

            // Save HDR EXR (linear, no tone mapping)
            {
                std::vector<float> hdr_data;
                optix_renderer.download_hdr_buffer(hdr_data, /*convert_from_spectrum=*/true);
                std::string exr_path = prefix + ".exr";
                write_exr(exr_path, hdr_data, display_fb.width, display_fb.height);
            }

            // Save raw (un-denoised) version when denoiser is active
            std::string raw_path;
            if (opt.config.denoiser_enabled) {
                raw_path = prefix + "_raw.png";
                FrameBuffer raw_fb;
                optix_renderer.download_raw_framebuffer(raw_fb);
                write_png(raw_path, raw_fb);
            }

            // Basic snapshot log (always shown, even when ENABLE_STATS == false)
            std::cout << "\n========================================\n";
            std::cout << "  [Snapshot] " << png_path << "\n";
            if (!raw_path.empty())
                std::cout << "  [Snapshot] " << raw_path << " (raw)\n";

            // ── Direction map debug PNGs ─────────────────────────────
            {
                optix_renderer.build_direction_map(camera, /*spp_seed=*/0);
                optix_renderer.download_direction_map();
                const auto& dm = optix_renderer.direction_map();
                if (dm.base_width > 0 && dm.base_height > 0) {
                    std::string dm_debug = prefix + "_dirmap_debug.png";
                    std::string dm_str   = prefix + "_dirmap_strength.png";
                    dm.write_debug_png(dm_debug);
                    dm.write_strength_png(dm_str);
                    std::cout << "  [Snapshot] " << dm_debug << " (direction map)\n";
                    std::cout << "  [Snapshot] " << dm_str   << " (direction strength)\n";
                }
            }

            std::cout << "========================================\n\n";

            // ── Lightweight metadata JSON (always written, no GPU queries) ──
            {
                std::string meta_path = prefix + "_meta.json";
                std::ofstream mf(meta_path);
                mf << std::fixed << std::setprecision(6);
                mf << "{\n";
                mf << "  \"timestamp\": \"" << ts << "\",\n";
                mf << "  \"png_file\": \"" << png_path << "\",\n";

                // Hardware
                mf << "  \"hardware\": {\n";
                if (g_active_optix_renderer) {
                    mf << "    \"gpu\": \"" << g_active_optix_renderer->gpu_name() << "\",\n";
                    mf << "    \"vram_bytes\": " << g_active_optix_renderer->gpu_vram_total() << ",\n";
                    mf << "    \"sm_count\": " << g_active_optix_renderer->gpu_sm_count() << ",\n";
                    mf << "    \"compute_capability\": \"" << g_active_optix_renderer->gpu_cc_major()
                       << "." << g_active_optix_renderer->gpu_cc_minor() << "\"\n";
                } else {
                    mf << "    \"gpu\": \"unknown\"\n";
                }
                mf << "  },\n";

                // Rendering duration
                mf << "  \"duration\": {\n";
                if (s_app.render_timing_active) {
                    auto elapsed = std::chrono::steady_clock::now() - s_app.render_start_time;
                    double elapsed_sec = std::chrono::duration<double>(elapsed).count();
                    mf << "    \"seconds\": " << elapsed_sec << ",\n";
                    mf << "    \"accumulated_spp\": " << frame << "\n";
                } else {
                    mf << "    \"seconds\": 0,\n";
                    mf << "    \"accumulated_spp\": " << frame << "\n";
                }
                mf << "  },\n";

                // Scene geometry
                mf << "  \"scene\": {\n";
                const char* scene_disp = (s_app.active_scene_index >= 0
                    && s_app.active_scene_index < NUM_SCENE_PROFILES)
                    ? SCENE_PROFILES[s_app.active_scene_index].display_name
                    : SCENE_DISPLAY_NAME;
                mf << "    \"name\": \"" << scene_disp << "\",\n";
                mf << "    \"num_triangles\": " << scene.num_triangles() << ",\n";
                mf << "    \"num_emissive\": " << scene.num_emissive() << "\n";
                mf << "  },\n";

                // Photon map
                mf << "  \"photons\": {\n";
                mf << "    \"num_photons\": " << opt.config.num_photons << ",\n";
                mf << "    \"gather_radius\": " << opt.config.gather_radius << "\n";
                mf << "  },\n";

                // Camera
                mf << "  \"camera\": {\n";
                mf << "    \"position\": [" << camera.position.x << ", "
                   << camera.position.y << ", " << camera.position.z << "],\n";
                mf << "    \"look_at\": [" << camera.look_at.x << ", "
                   << camera.look_at.y << ", " << camera.look_at.z << "],\n";
                mf << "    \"fov_deg\": " << camera.fov_deg << ",\n";
                mf << "    \"dof_enabled\": " << (camera.dof_enabled ? "true" : "false") << ",\n";
                mf << "    \"dof_f_number\": " << camera.dof_f_number << ",\n";
                mf << "    \"dof_focus_dist\": " << camera.dof_focus_dist << "\n";
                mf << "  },\n";

                // Post-FX
                mf << "  \"postfx\": {\n";
                mf << "    \"bloom_enabled\": " << (s_app.postfx.bloom_enabled ? "true" : "false") << ",\n";
                mf << "    \"bloom_intensity\": " << s_app.postfx.bloom_intensity << "\n";
                mf << "  },\n";

                // Modes / toggles
                mf << "  \"modes\": {\n";
                mf << "    \"guided_pt\": " << (s_app.guided_enabled ? "true" : "false") << ",\n";
                mf << "    \"spectral_clamp\": " << (s_app.spectral_clamp_enabled ? "true" : "false") << ",\n";
                mf << "    \"denoiser\": " << (opt.config.denoiser_enabled ? "true" : "false") << ",\n";
                mf << "    \"exposure\": " << opt.config.exposure << ",\n";
                mf << "    \"light_scale\": " << s_app.light_scale << ",\n";
                const char* rk_tag =
                    s_app.render_key_mode == AppState::RenderKeyMode::T_OptScreenshot   ? "T_OptScreenshot" :
                    s_app.render_key_mode == AppState::RenderKeyMode::Z_UnoptScreenshot ? "Z_UnoptScreenshot" :
                    "None";
                mf << "    \"render_key_mode\": \"" << rk_tag << "\"\n";
                mf << "  }\n";

                mf << "}\n";
                std::cout << "  [Snapshot] " << meta_path << " (metadata)\n";
            }

            // ── Statistics gathering, JSON export, analysis report ────
            // Entire block gated by ENABLE_STATS: when false, snapshot
            // saves only the PNG(s) + lightweight metadata above.
            if constexpr (ENABLE_STATS) {
                const char* scene_name = (s_app.active_scene_index >= 0
                    && s_app.active_scene_index < NUM_SCENE_PROFILES)
                    ? SCENE_PROFILES[s_app.active_scene_index].display_name
                    : SCENE_DISPLAY_NAME;
                auto stats = optix_renderer.gather_stats(scene_name);

                // ── Snapshot JSON ────────────────────────────────────
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
                    jf << "    \"up\": [" << camera.up.x << ", "
                       << camera.up.y << ", " << camera.up.z << "],\n";
                    jf << "    \"fov_deg\": " << camera.fov_deg << ",\n";
                    jf << "    \"width\": " << camera.width << ",\n";
                    jf << "    \"height\": " << camera.height << ",\n";
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
                    jf << "    \"obj_path\": \"" << SCENE_OBJ_PATH << "\",\n";
                    jf << "    \"num_triangles\": " << stats.num_triangles << ",\n";
                    jf << "    \"num_emissive_tris\": " << stats.num_emissive_tris << ",\n";
                    jf << "    \"scene_aabb_min\": [" << scene.scene_bounds.mn.x << ", "
                       << scene.scene_bounds.mn.y << ", " << scene.scene_bounds.mn.z << "],\n";
                    jf << "    \"scene_aabb_max\": [" << scene.scene_bounds.mx.x << ", "
                       << scene.scene_bounds.mx.y << ", " << scene.scene_bounds.mx.z << "]\n";
                    jf << "  }\n";

                    jf << "}\n";
                }

                std::cout << "  [Snapshot] " << json_path << "\n";
                std::cout << "  SPP: " << stats.accumulated_spp
                          << "  Photons: " << stats.photons_stored
                          << "  Cells: " << stats.cell_analysis_cells << "\n";

                // ── Build RendererStats for console + analysis ───────
                RendererStats rs;
                // Path tracing
                rs.spp_min = rs.spp_max = stats.accumulated_spp;
                rs.spp_avg        = (float)stats.accumulated_spp;
                rs.guided_enabled = s_app.guided_enabled;
                rs.histogram_only = s_app.histogram_only;
                rs.guide_fraction = g_active_optix_renderer
                                  ? g_active_optix_renderer->get_guide_fraction()
                                  : 0.f;
                // Photon mapping
                rs.photons_emitted  = stats.photons_emitted;
                rs.photons_stored   = stats.photons_stored;
                rs.photons_global   = stats.noncaustic_stored;
                rs.photons_global_caustic = stats.global_caustic_stored;
                rs.photons_targeted = stats.caustic_stored;
                rs.gather_radius    = stats.gather_radius;
                rs.caustic_radius   = stats.caustic_radius;
                // Geometry
                rs.num_triangles    = stats.num_triangles;
                rs.num_emissive_tris= stats.num_emissive_tris;
                // Hardware
                if (g_active_optix_renderer) {
                    rs.gpu_name      = g_active_optix_renderer->gpu_name();
                    rs.gpu_vram_bytes= g_active_optix_renderer->gpu_vram_total();
                    rs.gpu_sm_count  = g_active_optix_renderer->gpu_sm_count();
                    rs.gpu_cc_major  = g_active_optix_renderer->gpu_cc_major();
                    rs.gpu_cc_minor  = g_active_optix_renderer->gpu_cc_minor();
                    // Photon flag tallies
                    rs.photon_flags  = tally_photon_flags(
                        g_active_optix_renderer->photons());
                    // Grid occupancy
                    rs.grid_occupancy = compute_grid_occupancy(
                        g_active_optix_renderer->dm_hash_grid());
                }
                // Timing
                rs.timing.total_render_ms = s_app.last_render_ms;
                // Cell analysis distribution + conclusions (from gather_stats)
                rs.guide_dist  = stats.guide_dist;
                rs.conclusions = stats.conclusions;
                // Hash histogram multi-resolution guide stats
                rs.hash_hist   = stats.hash_hist;
                print_stats_console(rs);

                // ── Analysis report JSON (for LLM / GPU expert) ──────
                AnalysisReport report;
                report.stats     = rs;
                report.timestamp = ts;
                report.png_path  = png_path;
                report.cam_pos[0] = camera.position.x;
                report.cam_pos[1] = camera.position.y;
                report.cam_pos[2] = camera.position.z;
                report.cam_lookat[0] = camera.look_at.x;
                report.cam_lookat[1] = camera.look_at.y;
                report.cam_lookat[2] = camera.look_at.z;
                report.cam_fov       = camera.fov_deg;
                report.light_scale   = s_app.light_scale;
                report.accumulated_spp = stats.accumulated_spp;
                // Cell analysis averages
                report.avg_guide_fraction   = stats.avg_guide_fraction;
                report.avg_guide_fraction_populated = stats.avg_guide_fraction_populated;
                report.guide_populated_cells = stats.guide_populated_cells;
                report.avg_caustic_fraction = stats.avg_caustic_fraction;
                report.cell_analysis_cells  = stats.cell_analysis_cells;

                std::string analysis_path = prefix + "_analysis.json";
                write_analysis_json(report, analysis_path);
                std::cout << "  [Snapshot] " << analysis_path << " (analysis)\n";

                // ── Save photon cache + launch analysis tool ─────────
                const auto& snap_photons = optix_renderer.photons();
                const auto& snap_grid    = optix_renderer.dm_hash_grid();
                if (snap_photons.size() > 0) {
                    std::string cache_path = snap_dir + "/photons.bin";
                    uint64_t snap_hash = compute_scene_hash(
                        opt.scene_file,
                        stats.num_triangles,
                        stats.photons_emitted,
                        stats.gather_radius);
                    if (save_photon_cache(cache_path, snap_photons,
                                          snap_grid, snap_hash,
                                          stats.gather_radius)) {
                        std::cout << "  [Snapshot] " << cache_path
                                  << " (photon cache, "
                                  << snap_photons.size() << " photons)\n";

                        // Launch photon_map_analysis asynchronously
                        std::string tool_exe = "build/photon_map_analysis.exe";
                        if (fs::exists(tool_exe)) {
                            std::string tool_out = snap_dir + "/hierarchy_report.json";
                            std::string cmd = "start /B \"\" \""
                                + fs::absolute(tool_exe).string() + "\" \""
                                + fs::absolute(cache_path).string() + "\" \""
                                + fs::absolute(opt.scene_file).string() + "\""
                                + " --snapshot \"" + fs::absolute(prefix + ".json").string() + "\""
                                + " --radius " + std::to_string(stats.gather_radius)
                                + " --output \"" + fs::absolute(tool_out).string() + "\"";
                            std::cout << "  [Snapshot] Launching analysis tool...\n";
                            std::system(cmd.c_str());
                        } else {
                            std::cout << "  [Snapshot] photon_map_analysis not found "
                                      << "(build with 'build.bat all')\n";
                        }
                    }
                }
            }
        }

        // ── Render path: preview (fast) or full-quality accumulation ──
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
                // Normal path tracing (1 spp per iteration, progressive accumulation)
                optix_renderer.render_debug_frame(
                    camera, frame, s_app.debug.current_mode, 1);
                optix_renderer.download_framebuffer(display_fb);


            }

            frame++;

            // ── Auto-screenshot at n² SPP milestones (T / Z keys) ───
            if ((s_app.render_key_mode == AppState::RenderKeyMode::T_OptScreenshot
              || s_app.render_key_mode == AppState::RenderKeyMode::Z_UnoptScreenshot)
                && frame >= s_app.render_key_next_screenshot_spp) {
                // Create session folder once per R/T/Z press
                if (s_app.render_key_output_dir.empty()) {
                    auto ss_now = std::chrono::system_clock::now();
                    std::time_t ss_t = std::chrono::system_clock::to_time_t(ss_now);
                    std::tm ss_tm;
                    localtime_s(&ss_tm, &ss_t);
                    char ss_ts[64];
                    std::strftime(ss_ts, sizeof(ss_ts), "%Y%m%d_%H%M%S", &ss_tm);
                    const char* mode_tag =
                        s_app.render_key_mode == AppState::RenderKeyMode::T_OptScreenshot
                        ? "opt" : "unopt";
                    s_app.render_key_output_dir = std::string("output/render_")
                        + mode_tag + "_" + ss_ts;
                    fs::create_directories(s_app.render_key_output_dir);
                }
                const char* mode_tag =
                    s_app.render_key_mode == AppState::RenderKeyMode::T_OptScreenshot
                    ? "opt" : "unopt";
                char spp_buf[64];
                std::snprintf(spp_buf, sizeof(spp_buf), "%s_spp%d", mode_tag, frame);
                std::string ss_prefix = s_app.render_key_output_dir + "/" + spp_buf;

                // Save PNG
                optix_renderer.download_framebuffer(display_fb);
                write_png(ss_prefix + ".png", display_fb);

                printf("[Render] Auto-screenshot at SPP %d -> %s\n",
                       frame, (ss_prefix + ".png").c_str());

                // Advance to next doubling milestone
                s_app.render_key_next_screenshot_spp =
                    next_screenshot_spp(s_app.render_key_next_screenshot_spp);
            }

            // Periodic photon + direction map rebuild (same logic as render_final)
            if (s_app.idle_rendering_active
                && DEFAULT_GUIDE_REMAP_INTERVAL > 0
                && frame > 0
                && (frame % DEFAULT_GUIDE_REMAP_INTERVAL) == 0) {
                std::printf("[Viewer] Periodic rebuild at frame %d ...\n", frame);
                optix_renderer.trace_photons(
                    scene, opt.config, 0.f, s_app.idle_photon_seed++);
                // Skip direction map in Z mode (unoptimized rendering)
                if (s_app.render_key_mode != AppState::RenderKeyMode::Z_UnoptScreenshot)
                    optix_renderer.build_direction_map(camera, /*spp_seed=*/frame);
            }
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

        // Draw stats overlay (S key)
        render_stats_overlay(win_w, win_h, s_app, &optix_renderer);

        // Draw hover-cell overlay (when map mode toggles are active and
        // mouse is released for inspection).
        render_hover_cell_overlay(
            win_w, win_h,
            camera,
            s_app.debug,
            optix_renderer,
            s_app.mouse_captured);

        // ── Title-bar status update ─────────────────────────────────
        {
            char title[256];
            if (s_app.idle_rendering_active)
                std::snprintf(title, sizeof(title),
                    "Photon Tracer  [Full SPP %d]", frame);
            else
                std::snprintf(title, sizeof(title),
                    "Photon Tracer  [Preview SPP %d]", frame);
            glfwSetWindowTitle(window, title);
        }

        glfwSwapBuffers(window);
    }

    glDeleteTextures(1, &tex);
    glfwDestroyWindow(window);
    glfwTerminate();
}
