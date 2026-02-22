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
//   --sppm             Enable SPPM rendering (R key triggers SPPM)
//   --sppm-iterations N  SPPM iterations (default 64, implies --sppm)
//   --sppm-radius R      SPPM initial radius (default 0.1, implies --sppm)
// ---------------------------------------------------------------------

#include "core/config.h"
#include "core/font_overlay.h"
#include "scene/obj_loader.h"
#include "scene/scene.h"
#include "renderer/renderer.h"
#include "renderer/camera.h"
#include "debug/debug.h"
#include "optix/optix_renderer.h"
#include "core/test_data_io.h"

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

// -- Scene lighting helpers -------------------------------------------

// Generate inward-facing emissive triangles on a (hemi)sphere.
// theta_min = 0 → north pole.  theta_max = PI   → full sphere.
//                               theta_max = PI/2 → upper hemisphere.
static void generate_sphere_lights(
    Scene& scene, uint32_t mat_id,
    float radius, float3 center,
    int n_slices, int n_stacks,
    float theta_min, float theta_max)
{
    auto sphere_pt = [&](int stack, int slice) -> float3 {
        float theta = theta_min + (theta_max - theta_min) * (float)stack / (float)n_stacks;
        float phi   = 2.0f * 3.14159265358979f * (float)slice / (float)n_slices;
        float st = sinf(theta);
        return make_f3(
            center.x + radius * st * cosf(phi),
            center.y + radius * cosf(theta),
            center.z + radius * st * sinf(phi));
    };

    float2 uv0 = make_f2(0, 0);
    int count = 0;

    for (int i = 0; i < n_stacks; ++i) {
        for (int j = 0; j < n_slices; ++j) {
            float3 p00 = sphere_pt(i,     j);
            float3 p10 = sphere_pt(i + 1, j);
            float3 p11 = sphere_pt(i + 1, j + 1);
            float3 p01 = sphere_pt(i,     j + 1);

            // Inward-facing normals (towards center)
            auto inward_n = [&](float3 p) { return normalize(center - p); };

            // Triangle 1: p00-p11-p10  (CW from outside = CCW from inside)
            {
                Triangle tri;
                tri.v0 = p00; tri.v1 = p11; tri.v2 = p10;
                tri.n0 = inward_n(p00); tri.n1 = inward_n(p11); tri.n2 = inward_n(p10);
                tri.uv0 = tri.uv1 = tri.uv2 = uv0;
                tri.material_id = mat_id;
                if (tri.area() > 1e-8f) { scene.triangles.push_back(tri); ++count; }
            }
            // Triangle 2: p00-p01-p11
            {
                Triangle tri;
                tri.v0 = p00; tri.v1 = p01; tri.v2 = p11;
                tri.n0 = inward_n(p00); tri.n1 = inward_n(p01); tri.n2 = inward_n(p11);
                tri.uv0 = tri.uv1 = tri.uv2 = uv0;
                tri.material_id = mat_id;
                if (tri.area() > 1e-8f) { scene.triangles.push_back(tri); ++count; }
            }
        }
    }
    std::cout << "[Scene]   Generated " << count << " emissive triangles "
              << "(r=" << radius << ", stacks=" << n_stacks
              << ", slices=" << n_slices << ")\n";
}

// Add appropriate light sources based on the scene's lighting mode.
// Called after scene load + normalize.  For MTL scenes the emissive
// surfaces are already in the triangle list.  For environment-lit
// scenes we generate tessellated (hemi)sphere geometry.
static void add_scene_lights(Scene& scene, SceneLightMode mode) {
    if (mode == SceneLightMode::FromMTL && scene.num_emissive() > 0)
        return;  // MTL already has lights

    const char* mode_name =
        mode == SceneLightMode::FromMTL            ? "FromMTL/fallback" :
        mode == SceneLightMode::DirectionalToFloor  ? "DirectionalToFloor" :
        mode == SceneLightMode::HemisphereEnv       ? "HemisphereEnv" :
                                                      "SphericalEnv";
    std::cout << "[Scene] Adding lights (mode: " << mode_name << ")\n";

    // ── HemisphereEnv (Sponza): sky dome above the scene ────────────
    if (mode == SceneLightMode::HemisphereEnv) {
        Material sky_mat;
        sky_mat.name = "__sky_dome__";
        sky_mat.type = MaterialType::Emissive;
        sky_mat.Le   = blackbody_spectrum(5800.f, 8e-9f);  // warm daylight
        uint32_t sky_id = (uint32_t)scene.materials.size();
        scene.materials.push_back(sky_mat);

        // Hemisphere: theta 0→PI/2, centered slightly above the scene
        float3 center = make_f3(0.0f, 0.0f, 0.0f);
        generate_sphere_lights(scene, sky_id,
            0.75f, center,       // radius just outside normalized [-0.5,0.5]
            24, 8,               // 24 slices × 8 stacks = 384 tris
            0.0f, 1.5707963f);   // theta 0→PI/2  (upper hemisphere)

        scene.build_bvh();
        scene.build_emissive_distribution();
        std::cout << "[Scene] Hemisphere sky dome: "
                  << scene.num_emissive() << " emissive triangles total\n";
        return;
    }

    // ── SphericalEnv (Hairball): full sphere surrounding the scene ──
    if (mode == SceneLightMode::SphericalEnv) {
        Material env_mat;
        env_mat.name = "__env_sphere__";
        env_mat.type = MaterialType::Emissive;
        env_mat.Le   = blackbody_spectrum(5500.f, 5e-9f);  // neutral daylight
        uint32_t env_id = (uint32_t)scene.materials.size();
        scene.materials.push_back(env_mat);

        // Full sphere: theta 0→PI
        float3 center = make_f3(0.0f, 0.0f, 0.0f);
        generate_sphere_lights(scene, env_id,
            0.80f, center,       // radius outside the model
            32, 16,              // 32 slices × 16 stacks = 1024 tris
            0.0f, 3.14159265f);  // theta 0→PI  (full sphere)

        scene.build_bvh();
        scene.build_emissive_distribution();
        std::cout << "[Scene] Full environment sphere: "
                  << scene.num_emissive() << " emissive triangles total\n";
        return;
    }

    // ── DirectionalToFloor / FromMTL fallback: ceiling area light ───
    Material light_mat;
    light_mat.name = "__area_light__";
    light_mat.type = MaterialType::Emissive;

    if (mode == SceneLightMode::DirectionalToFloor) {
        light_mat.Le = blackbody_spectrum(6500.f, 3e-8f);
    } else {
        light_mat.Le = blackbody_spectrum(6500.f, 1e-8f);
    }

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
              << scene.num_emissive() << " emissive triangles)\n";
}

// Apply complexity preset from a scene profile to a RenderConfig
static void apply_complexity_preset(RenderConfig& cfg, const SceneProfile& prof) {
    const ComplexityPreset& preset = prof.preset();
    cfg.global_photon_budget  = preset.global_photon_budget;
    cfg.caustic_photon_budget = preset.caustic_photon_budget;
    cfg.num_photons           = preset.global_photon_budget;
    cfg.gather_radius         = preset.gather_radius;
    cfg.caustic_radius        = preset.caustic_radius;
    cfg.samples_per_pixel     = preset.spp;
    cfg.max_bounces           = preset.photon_max_bounces;
    std::cout << "[Preset] " << prof.display_name << " ("
              << (prof.complexity == SceneComplexity::Low  ? "low" :
                  prof.complexity == SceneComplexity::Medium ? "medium" : "high")
              << "): " << preset.global_photon_budget << " global, "
              << preset.caustic_photon_budget << " caustic photons, "
              << "r=" << preset.gather_radius << ", "
              << preset.spp << " spp, "
              << preset.photon_max_bounces << " bounces\n";
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

    // Semi-transparent background box
    float pad   = 10.f;
    float box_w = 280.f;
    float box_h = 560.f;
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
    if (!debug.show_global_map && !debug.show_caustic_map) return;
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

    PhotonMapType map_type = debug.show_caustic_map
        ? PhotonMapType::Caustic : PhotonMapType::Global;

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
    float bx = (float)win_w - box_w - 10.f;
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

// -- Parse command line -----------------------------------------------

struct Options {
    std::string scene_file;
    std::string output_file = "output/render.png";
    std::string save_test_data_file;   // if non-empty, dump photons to this binary file

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
            if      (mode == "full")     opt.config.mode = RenderMode::Combined;
            else if (mode == "combined") opt.config.mode = RenderMode::Combined;
            else if (mode == "direct")   opt.config.mode = RenderMode::DirectOnly;
            else if (mode == "indirect") opt.config.mode = RenderMode::IndirectOnly;
            else if (mode == "photon")   opt.config.mode = RenderMode::PhotonMap;
            else if (mode == "normals")  opt.config.mode = RenderMode::Normals;
            else if (mode == "material") opt.config.mode = RenderMode::MaterialID;
            else if (mode == "depth")    opt.config.mode = RenderMode::Depth;
        } else if (arg == "--save-test-data" && i + 1 < argc) {
            opt.save_test_data_file = argv[++i];
        } else if (arg == "--sppm") {
            opt.config.sppm_enabled = true;
        } else if (arg == "--sppm-iterations" && i + 1 < argc) {
            opt.config.sppm_iterations = std::stoi(argv[++i]);
            opt.config.sppm_enabled = true;
        } else if (arg == "--sppm-radius" && i + 1 < argc) {
            opt.config.sppm_initial_radius = std::stof(argv[++i]);
            opt.config.sppm_enabled = true;
        // ── v2 CLI flags ────────────────────────────────────────────
        } else if (arg == "--global-photons" && i + 1 < argc) {
            opt.config.global_photon_budget = std::stoi(argv[++i]);
            opt.config.num_photons = opt.config.global_photon_budget;
        } else if (arg == "--caustic-photons" && i + 1 < argc) {
            opt.config.caustic_photon_budget = std::stoi(argv[++i]);
        } else if (arg == "--spatial" && i + 1 < argc) {
            std::string sp = argv[++i];
            if      (sp == "kdtree")   opt.config.use_kdtree = true;
            else if (sp == "hashgrid") opt.config.use_kdtree = false;
        } else if (arg == "--knn-k" && i + 1 < argc) {
            opt.config.knn_k = std::stoi(argv[++i]);
        } else if (arg == "--max-specular-chain" && i + 1 < argc) {
            opt.config.max_specular_chain = std::stoi(argv[++i]);
        // photon cache flags removed -- photon map is re-traced every MULTI_MAP_SPP_GROUP iters
        } else if (arg == "--adaptive-radius") {
            opt.config.use_knn_adaptive = true;
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
    bool       volume_enabled   = DEFAULT_VOLUME_ENABLED;  // V key toggle
    bool       use_dense_grid   = DEFAULT_USE_DENSE_GRID;  // G key toggle
    bool       showing_final    = false;  // keep final render on-screen

    // Scene switching (keys 1-4)
    int        scene_switch_requested = -1;  // -1 = none, 0-3 = profile index
    int        active_scene_index     = -1;  // currently loaded scene profile index
    float      active_cam_speed       = SCENE_CAM_SPEED; // runtime cam speed

    // Light brightness scaling (runtime, +/- keys)
    float      light_scale            = DEFAULT_LIGHT_SCALE;
    bool       light_scale_changed    = false;  // triggers photon re-trace

    // Render cancellation
    bool       render_cancel_requested = false;

    // Photon retrace request (P key)
    bool       photon_retrace_requested = false;

    // Progressive final render state
    int        render_spp_done  = 0;      // samples completed so far
    int        render_spp_total = 0;      // target spp
    Camera     render_cam;                // frozen camera for render
    std::chrono::high_resolution_clock::time_point render_start;

    // Camera angles (yaw/pitch in radians)
    float      yaw   = 0.f;     // horizontal angle
    float      pitch = 0.f;     // vertical angle
    bool       mouse_captured = true;  // start with mouse captured
    bool       mouse_was_captured = true; // saved state before render
    double     last_mx = 0.0, last_my = 0.0;
    bool       first_mouse = true;
    bool       camera_moved = false;  // flag to reset accumulation
};

static AppState        g_app;
static Camera*         g_active_camera          = nullptr;  // set by run_interactive for key_callback DOF editing
static OptixRenderer*  g_active_optix_renderer  = nullptr;  // set by run_interactive for key_callback volume toggle

static void key_callback(GLFWwindow* window, int key,
                          int /*scancode*/, int action, int /*mods*/) {
    if (action != GLFW_PRESS) return;

    if (key == GLFW_KEY_ESCAPE) {
        // ESC during rendering -> cancel the render
        if (g_app.rendering) {
            g_app.render_cancel_requested = true;
            printf("\n[Render] Cancelling...\n");
            return;
        }
        // ESC when mouse captured -> release mouse
        if (g_app.mouse_captured) {
            g_app.mouse_captured = false;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            return;
        }
        // ESC otherwise -> quit
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        return;
    }

    if (key == GLFW_KEY_Q) {
        if (g_app.mouse_captured) {
            g_app.mouse_captured = false;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            return;
        }
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        return;
    }

    // "R" -> start final render
    if (key == GLFW_KEY_R) {
        g_app.showing_final = false;
        g_app.render_requested = true;
        // Release mouse cursor during render so the camera stays frozen
        g_app.mouse_was_captured = g_app.mouse_captured;
        if (g_app.mouse_captured) {
            g_app.mouse_captured = false;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            g_app.first_mouse = true;
        }
        return;
    }

    // Left-click or M to toggle mouse capture
    if (key == GLFW_KEY_M) {
        g_app.showing_final = false;
        g_app.mouse_captured = !g_app.mouse_captured;
        glfwSetInputMode(window, GLFW_CURSOR,
            g_app.mouse_captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
        g_app.first_mouse = true;
        return;
    }

    // Volume scattering toggle – V key
    if (key == GLFW_KEY_V) {
        g_app.volume_enabled = !g_app.volume_enabled;
        if (g_active_optix_renderer)
            g_active_optix_renderer->set_volume_enabled(g_app.volume_enabled);
        printf("[Volume] Scattering: %s\n", g_app.volume_enabled ? "ON" : "OFF");
        g_app.camera_moved  = true;  // reset accumulation
        g_app.showing_final = false;
        return;
    }

    // Gather mode toggle – G key (dense cell-bin grid vs per-photon hash walk)
    if (key == GLFW_KEY_G) {
        g_app.use_dense_grid = !g_app.use_dense_grid;
        if (g_active_optix_renderer)
            g_active_optix_renderer->set_use_dense_grid(g_app.use_dense_grid);
        printf("[Gather] Dense grid: %s\n", g_app.use_dense_grid ? "ON" : "OFF");
        g_app.camera_moved  = true;  // reset accumulation
        g_app.showing_final = false;
        return;
    }

    // Scene switching – keys 1-9
    if (key >= GLFW_KEY_1 && key <= GLFW_KEY_9 && !g_app.rendering) {
        int idx = key - GLFW_KEY_1;  // 0-8
        if (idx < NUM_SCENE_PROFILES && idx != g_app.active_scene_index) {
            g_app.scene_switch_requested = idx;
            printf("[Scene] Switching to %s ...\n", SCENE_PROFILES[idx].display_name);
        } else if (idx < NUM_SCENE_PROFILES) {
            printf("[Scene] Already on %s\n", SCENE_PROFILES[idx].display_name);
        }
        return;
    }

    // Light brightness – +/= increase, -/_ decrease
    if ((key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD) && !g_app.rendering) {
        g_app.light_scale = fminf(LIGHT_SCALE_MAX, g_app.light_scale * LIGHT_SCALE_STEP);
        g_app.light_scale_changed = true;
        printf("[Light] Brightness: %.2fx\n", g_app.light_scale);
        return;
    }
    if ((key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT) && !g_app.rendering) {
        g_app.light_scale = fmaxf(LIGHT_SCALE_MIN, g_app.light_scale / LIGHT_SCALE_STEP);
        g_app.light_scale_changed = true;
        printf("[Light] Brightness: %.2fx\n", g_app.light_scale);
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
            g_app.camera_moved   = true;
            g_app.showing_final  = false;
            return;
        }
    }

    // P -> retrace photons (rebuild photon map)
    if (key == GLFW_KEY_P && !g_app.rendering) {
        g_app.photon_retrace_requested = true;
        g_app.camera_moved  = true;  // reset accumulation
        g_app.showing_final = false;
        printf("[Photon] Retrace requested (rebuild photon maps)\n");
        return;
    }

    // Any debug toggle should return to the interactive debug view.
    g_app.showing_final = false;
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

    // Middle-click: auto-focus on the point under the cursor
    if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS
        && !g_app.mouse_captured && g_active_camera && g_active_optix_renderer) {
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
            g_app.camera_moved  = true;
            g_app.showing_final = false;
            printf("[DOF] Auto-focus (cursor): %.4f\n", cam.dof_focus_dist);
        } else {
            printf("[DOF] Auto-focus: no hit\n");
        }
    }
}

// -- Interactive OptiX debug window -----------------------------------

static void run_interactive(
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

    // Expose camera and renderer to key_callback for runtime editing
    g_active_camera         = &camera;
    g_active_optix_renderer = &optix_renderer;

    // Sync initial volume state from AppState into the renderer
    optix_renderer.set_volume_enabled(g_app.volume_enabled);
    optix_renderer.set_use_dense_grid(g_app.use_dense_grid);

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
    std::cout << "  ESC = cancel render / release mouse / quit | Q = quit\n";
    std::cout << "  F1-F9 = debug toggles | TAB = cycle mode\n";
    std::cout << "  R = full path tracing render -> " << opt.output_file << "\n";
    if (opt.config.sppm_enabled)
        std::cout << "      (SPPM mode: " << opt.config.sppm_iterations << " iterations, r="
                  << opt.config.sppm_initial_radius << ")\n";
    std::cout << "  H = toggle help overlay\n";
    std::cout << "  1-9 = switch scene\n";
    std::cout << "  +/- = adjust light brightness (re-traces photons)\n";
    std::cout << "  V = toggle volume scattering | O = toggle DOF | [/] = blur | ,/. = focus dist\n";

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

        // ── Runtime scene switch (keys 1-4) ────────────────────────────
        if (g_app.scene_switch_requested >= 0) {
            int idx = g_app.scene_switch_requested;
            g_app.scene_switch_requested = -1;

            const SceneProfile& prof = SCENE_PROFILES[idx];
            std::string obj_path = std::string(SCENES_DIR) + "/" + prof.obj_path;

            std::cout << "\n========================================\n";
            std::cout << "  Loading scene: " << prof.display_name << "\n";
            std::cout << "========================================\n";

            // Apply complexity preset for new scene
            apply_complexity_preset(opt.config, prof);

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

                // Sync yaw/pitch from new camera direction
                float3 fwd = normalize(camera.look_at - camera.position);
                g_app.yaw   = atan2f(fwd.x, -fwd.z);
                g_app.pitch = asinf(fmaxf(-1.f, fminf(1.f, fwd.y)));

                g_app.active_cam_speed    = prof.cam_speed;
                g_app.active_scene_index  = idx;
                g_app.camera_moved        = true;
                g_app.showing_final       = false;
                g_app.light_scale         = DEFAULT_LIGHT_SCALE;  // reset brightness
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
        if (g_app.photon_retrace_requested && !g_app.rendering) {
            g_app.photon_retrace_requested = false;

            std::cout << "[Photon] Re-tracing photon maps...\n";
            auto tp0 = std::chrono::high_resolution_clock::now();
            optix_renderer.trace_photons(scene, opt.config);
            auto tp1 = std::chrono::high_resolution_clock::now();
            double photon_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();
            std::cout << "[Photon] Retrace done in " << photon_ms << " ms\n";

            frame = 0;
            optix_renderer.clear_buffers();
            g_app.camera_moved  = true;
            g_app.showing_final = false;
        }

        // ── Light brightness change (+/- keys) ─────────────────────────
        if (g_app.light_scale_changed) {
            g_app.light_scale_changed = false;

            // Scale all emissive materials' Le by the new factor relative to 1.0
            // We store original Le values and always scale from those
            for (size_t m = 0; m < scene.materials.size(); ++m) {
                Spectrum scaled = original_Le[m];
                for (int k = 0; k < NUM_LAMBDA; ++k)
                    scaled.value[k] *= g_app.light_scale;
                scene.materials[m].Le = scaled;
            }

            // Rebuild emissive distribution with new powers
            scene.build_emissive_distribution();

            // Re-upload materials and emitter CDF
            optix_renderer.upload_scene_data(scene);
            optix_renderer.upload_emitter_data(scene);

            // Re-trace photons with updated emission
            std::cout << "[Light] Re-tracing photons at " << g_app.light_scale << "x brightness...\n";
            auto tp0 = std::chrono::high_resolution_clock::now();
            optix_renderer.trace_photons(scene, opt.config);
            auto tp1 = std::chrono::high_resolution_clock::now();
            double photon_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();
            std::cout << "[Light] Photon re-trace done in " << photon_ms << " ms\n";

            g_app.camera_moved  = true;
            g_app.showing_final = false;
        }

        // Mouse cursor position for hover inspectors.
        {
            double mx, my;
            glfwGetCursorPos(window, &mx, &my);
            g_app.debug.hover_x = (int)mx;
            g_app.debug.hover_y = (int)my;
        }

        // Delta time
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        last_time = now;
        dt = fminf(dt, 0.1f); // clamp to avoid huge jumps

        // -- Mouse look -------------------------------------------------
        constexpr float kMouseSens = 0.0005f; // radians per pixel
        if (g_app.mouse_captured && !g_app.rendering) {
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
                g_app.yaw   += dx * kMouseSens;
                g_app.pitch -= dy * kMouseSens; // inverted Y
                // Clamp pitch to avoid gimbal lock
                constexpr float MAX_PITCH = 89.f * PI / 180.f;
                g_app.pitch = fmaxf(-MAX_PITCH, fminf(MAX_PITCH, g_app.pitch));
                g_app.camera_moved = true;
            }
        }

        // -- WASD movement (disabled during rendering) ------------------
        if (!g_app.rendering)
        {
            // Forward direction (camera looks along -w in its frame)
            float3 forward = make_f3(
                sinf(g_app.yaw) * cosf(g_app.pitch),
                sinf(g_app.pitch),
                -cosf(g_app.yaw) * cosf(g_app.pitch));
            float3 right = normalize(cross(forward, make_f3(0, 1, 0)));
            float3 up_dir = make_f3(0, 1, 0);

            float speed = g_app.active_cam_speed * dt;
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
            g_app.showing_final = false;
        }

        // Handle "R" key: start progressive final render (or SPPM)
        if (g_app.render_requested && !g_app.rendering) {
            // Freeze camera for the render
            g_app.render_cam = camera;
            g_app.render_cam.width  = opt.config.image_width;
            g_app.render_cam.height = opt.config.image_height;
            g_app.render_cam.update();

            // ── SPPM render path ────────────────────────────────────
            if (opt.config.sppm_enabled) {
                g_app.render_requested = false;
                g_app.showing_final = true;

                std::cout << "\n========================================\n";
                std::cout << "  SPPM Render (R key)\n";
                std::cout << "  Camera pos: (" << camera.position.x << ", "
                          << camera.position.y << ", " << camera.position.z << ")\n";
                std::cout << "  " << opt.config.image_width << "x"
                          << opt.config.image_height << " @ "
                          << opt.config.sppm_iterations << " SPPM iterations, "
                          << opt.config.num_photons << " photons/iter\n";
                std::cout << "  Initial radius: " << opt.config.sppm_initial_radius
                          << "  alpha: " << opt.config.sppm_alpha << "\n";
                std::cout << "========================================\n";

                // Build timestamp prefix once so all iteration PNGs sort together.
                auto sppm_start_tp = std::chrono::system_clock::now();
                std::time_t sppm_start_t = std::chrono::system_clock::to_time_t(sppm_start_tp);
                std::tm sppm_tm_buf;
                localtime_s(&sppm_tm_buf, &sppm_start_t);
                char sppm_ts[64];
                std::strftime(sppm_ts, sizeof(sppm_ts), "%Y%m%d_%H%M%S", &sppm_tm_buf);
                std::string sppm_run_prefix = std::string("output/sppm_") + sppm_ts;

                // Per-iteration callback: save a progress PNG after each full iteration.
                auto sppm_iter_cb = [&](int iter, const FrameBuffer& iter_fb) {
                    char iter_str[16];
                    std::snprintf(iter_str, sizeof(iter_str), "_iter%04d", iter + 1);
                    write_png(sppm_run_prefix + iter_str + ".png", iter_fb);
                };

                // Run blocking SPPM render
                optix_renderer.render_sppm(
                    g_app.render_cam, opt.config, scene, sppm_iter_cb);

                // Download and save final composite
                FrameBuffer final_fb;
                optix_renderer.download_framebuffer(final_fb);

                std::string sppm_path = sppm_run_prefix + "_final.png";

                write_png(sppm_path, final_fb);

                // Update display
                display_fb = final_fb;

                std::cout << "========================================\n";
                std::cout << "  Saved: " << sppm_path << "\n";
                std::cout << "========================================\n\n";

            // ── Normal progressive render path ──────────────────────
            } else {
            g_app.rendering = true;
            g_app.showing_final = false;
            g_app.render_spp_done  = 0;
            g_app.render_spp_total = opt.config.samples_per_pixel;
            g_app.render_start = std::chrono::high_resolution_clock::now();

            optix_renderer.resize(opt.config.image_width, opt.config.image_height);
            optix_renderer.clear_buffers();  // ensure clean accumulation

            std::cout << "\n========================================\n";
            std::cout << "  Progressive Render (R key)\n";
            std::cout << "  Camera pos: (" << camera.position.x << ", "
                      << camera.position.y << ", " << camera.position.z << ")\n";
            std::cout << "  " << opt.config.image_width << "x"
                      << opt.config.image_height << " @ "
                      << g_app.render_spp_total << " spp\n";
            std::cout << "  Output:     " << opt.output_file << "\n";
            std::cout << "========================================\n";

            // ── 1st-hit NEE debug PNG (quick preview) ───────────────
            {
                auto t_nee_start = std::chrono::high_resolution_clock::now();
                optix_renderer.render_debug_frame(
                    g_app.render_cam, 0, RenderMode::Full, 1,
                    true /* shadow_rays for NEE debug PNG */);
                FrameBuffer nee_fb;
                optix_renderer.download_framebuffer(nee_fb);
                write_png("output/out_debug_nee.png", nee_fb);
                auto t_nee_end = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(
                    t_nee_end - t_nee_start).count();
                std::printf("[Timing] NEE debug PNG:     %8.1f ms\n", ms);
                std::cout << "[Debug NEE] Saved: output/out_debug_nee.png\n";
                // Clear buffers so the progressive render starts clean
                optix_renderer.clear_buffers();
            }

            // ── Cell-bin grid is already built by trace_photons() ──
            // Nothing to do here — the grid is precomputed and uploaded
            // automatically during the photon tracing phase.

            g_app.render_requested = false;
            } // end else (normal progressive render)
        }

        // Progressive rendering: one spp per main-loop iteration
        if (g_app.rendering) {
            // Check for cancel request (ESC key)
            if (g_app.render_cancel_requested) {
                g_app.render_cancel_requested = false;
                g_app.rendering = false;
                g_app.showing_final = false;
                g_app.render_requested = false;

                // Restore mouse capture state
                if (g_app.mouse_was_captured) {
                    g_app.mouse_captured = true;
                    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                    g_app.first_mouse = true;
                }

                // Resize back to preview
                optix_renderer.resize(win_w, win_h);
                optix_renderer.clear_buffers();
                frame = 0;

                std::cout << "\n[Render] Cancelled at "
                          << g_app.render_spp_done << "/"
                          << g_app.render_spp_total << " spp\n";
            } else {
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

                // Print GPU kernel profiling summary
                optix_renderer.print_kernel_profiling();

                // Build timestamped prefix: output/render_YYYYMMDD_HHMMSS
                auto now_tp = std::chrono::system_clock::now();
                std::time_t now_t = std::chrono::system_clock::to_time_t(now_tp);
                std::tm tm_buf;
                localtime_s(&tm_buf, &now_t);
                char ts[64];
                std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tm_buf);
                std::string prefix = std::string("output/render_") + ts;

                std::string final_path   = prefix + ".png";
                std::string nee_path     = prefix + "_nee_direct.png";
                std::string photon_path  = prefix + "_photon_indirect.png";

                FrameBuffer final_fb;
                optix_renderer.download_framebuffer(final_fb);
                write_png(final_path, final_fb);

                // Write component PNGs (NEE direct, Photon indirect)
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

                    FrameBuffer nee_fb, photon_fb;

                    spectral_to_fb(nee_spec, nee_fb);
                    write_png(nee_path, nee_fb);

                    spectral_to_fb(photon_spec, photon_fb);
                    write_png(photon_path, photon_fb);
                }

                // Restore mouse capture state
                if (g_app.mouse_was_captured) {
                    g_app.mouse_captured = true;
                    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                    g_app.first_mouse = true;
                }

                // Reset to debug view
                g_app.rendering = false;
                g_app.showing_final = true;
                std::cout << "========================================\n";
                std::cout << "  Saved: " << final_path << "\n";
                std::cout << "  NEE:   " << nee_path << "\n";
                std::cout << "  Phot:  " << photon_path << "\n";
                std::cout << "========================================\n\n";
                }
            }
        } else {
            if (!g_app.showing_final) {
                // Normal debug preview (first-hit, 1 spp per iteration)
                optix_renderer.render_debug_frame(
                    camera, frame, g_app.debug.current_mode, 1);
                optix_renderer.download_framebuffer(display_fb);

                // Optional photon overlay in interactive debug mode.
                // Note: caustic map separation is not implemented yet, so
                // only photon points/global map use the current photon set.
                if (g_app.debug.show_photon_points || g_app.debug.show_global_map) {
                    const PhotonSoA& photons = optix_renderer.photons();
                    if (photons.size() > 0) {
                        overlay_photon_points(
                            display_fb,
                            camera,
                            photons,
                            g_app.debug.spectral_coloring,
                            2.0f);
                    }
                }

                frame++;
            }
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
            render_help_overlay(win_w, win_h, g_app.debug, camera, g_app.volume_enabled,
                                g_app.use_dense_grid, g_app.active_scene_index, g_app.light_scale);
        }

        // Draw hover-cell overlay (when map mode toggles are active and
        // mouse is released for inspection).
        render_hover_cell_overlay(
            win_w, win_h,
            camera,
            g_app.debug,
            optix_renderer,
            g_app.mouse_captured);

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
    {
        auto t_load0 = std::chrono::high_resolution_clock::now();

        if (!load_obj(opt.scene_file, scene)) {
            std::cerr << "[Error] Failed to load scene: " << opt.scene_file << "\n";
            return 1;
        }

        // Normalise non-reference scenes to Cornell-Box coordinate frame
        if (!SCENE_IS_REFERENCE) {
            scene.normalize_to_reference();
        }

        auto t_load1 = std::chrono::high_resolution_clock::now();
        double load_ms = std::chrono::duration<double, std::milli>(t_load1 - t_load0).count();
        std::printf("[Timing] OBJ load:          %8.1f ms  (%zu tris, %zu mats, %zu textures)\n",
                    load_ms, scene.triangles.size(), scene.materials.size(), scene.textures.size());

        scene.build_bvh();
        auto t_bvh = std::chrono::high_resolution_clock::now();
        double bvh_ms = std::chrono::duration<double, std::milli>(t_bvh - t_load1).count();
        std::printf("[Timing] BVH build:         %8.1f ms  (%zu nodes)\n",
                    bvh_ms, scene.bvh_nodes.size());

        scene.build_emissive_distribution();
        std::printf("[Scene]  Emissive tris: %d   total power = %.4f   total area = %.4f\n",
                    (int)scene.num_emissive(),
                    scene.total_emissive_power,
                    scene.total_emissive_area);
        std::printf("[Scene]  Render config: %dx%d  spp=%d  photons=%d  radius=%.5f  bounces=%d\n",
                    opt.config.image_width, opt.config.image_height,
                    opt.config.samples_per_pixel, opt.config.num_photons,
                    opt.config.gather_radius, opt.config.max_bounces);
    }

    // -- Add light source if none exists ------------------------------
    // Determine initial scene lighting mode from compile-time scene define
    SceneLightMode initial_light_mode = SceneLightMode::FromMTL;
    #if defined(SCENE_CORNELL_BOX)
        initial_light_mode = SCENE_PROFILES[0].light_mode;
    #elif defined(SCENE_CONFERENCE)
        initial_light_mode = SCENE_PROFILES[1].light_mode;
    #elif defined(SCENE_LIVING_ROOM)
        initial_light_mode = SCENE_PROFILES[2].light_mode;
    #elif defined(SCENE_SIBENIK)
        initial_light_mode = SCENE_PROFILES[5].light_mode;
    #endif
    add_scene_lights(scene, initial_light_mode);

    // -- Setup camera (from scene profile) --------------------------
    Camera camera;
    camera.position = make_f3(SCENE_CAM_POS[0], SCENE_CAM_POS[1], SCENE_CAM_POS[2]);
    camera.look_at  = make_f3(SCENE_CAM_LOOKAT[0], SCENE_CAM_LOOKAT[1], SCENE_CAM_LOOKAT[2]);
    camera.up       = make_f3(0.0f, 1.0f, 0.0f);
    camera.fov_deg  = SCENE_CAM_FOV;
    camera.width    = opt.config.image_width;
    camera.height   = opt.config.image_height;

    // Depth of field (thin-lens) – defaults from config.h
    camera.dof_enabled    = DEFAULT_DOF_ENABLED;
    camera.dof_focus_dist = DEFAULT_DOF_FOCUS_DISTANCE;
    camera.dof_f_number   = DEFAULT_DOF_F_NUMBER;
    camera.sensor_height  = DEFAULT_DOF_SENSOR_HEIGHT;

    camera.update();

    // -- OptiX pipeline -----------------------------------------------
    OptixRenderer optix_renderer;

    try {
        std::cout << "-- OptiX Initialization --\n";
        {
            auto t_init0 = std::chrono::high_resolution_clock::now();
            optix_renderer.init();
            auto t_init1 = std::chrono::high_resolution_clock::now();
            std::printf("[Timing] OptiX init:        %8.1f ms\n",
                std::chrono::duration<double, std::milli>(t_init1 - t_init0).count());

            auto t_accel0 = std::chrono::high_resolution_clock::now();
            optix_renderer.build_accel(scene);
            auto t_accel1 = std::chrono::high_resolution_clock::now();
            std::printf("[Timing] build_accel:       %8.1f ms\n",
                std::chrono::duration<double, std::milli>(t_accel1 - t_accel0).count());

            auto t_scene0 = std::chrono::high_resolution_clock::now();
            optix_renderer.upload_scene_data(scene);
            auto t_scene1 = std::chrono::high_resolution_clock::now();
            std::printf("[Timing] upload_scene_data: %8.1f ms\n",
                std::chrono::duration<double, std::milli>(t_scene1 - t_scene0).count());

            auto t_emit0 = std::chrono::high_resolution_clock::now();
            optix_renderer.upload_emitter_data(scene);
            auto t_emit1 = std::chrono::high_resolution_clock::now();
            std::printf("[Timing] upload_emitter:    %8.1f ms\n",
                std::chrono::duration<double, std::milli>(t_emit1 - t_emit0).count());
        }
        std::cout << "\n";

        // -- GPU photon trace ------------------------------------------
        std::cout << "-- GPU Photon Trace --\n";
        {
            auto tp0 = std::chrono::high_resolution_clock::now();
            optix_renderer.trace_photons(scene, opt.config);
            auto tp1 = std::chrono::high_resolution_clock::now();
            double photon_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();
            std::printf("[Photon] Initial trace completed in %.1f ms\n\n", photon_ms);
        }

        // -- Save test data (if requested) ----------------------------
        if (!opt.save_test_data_file.empty()) {
            TestDataHeader hdr;
            hdr.num_photons_cfg = (uint32_t)opt.config.num_photons;
            hdr.gather_radius   = opt.config.gather_radius;
            hdr.caustic_radius  = opt.config.caustic_radius;
            hdr.max_bounces     = (uint32_t)opt.config.max_bounces;
            hdr.min_bounces_rr  = (uint32_t)opt.config.min_bounces_rr;
            hdr.rr_threshold    = opt.config.rr_threshold;
            hdr.scene_path      = SCENE_OBJ_PATH;

            PhotonSoA empty_caustic;  // GPU path has no separate caustic map
            save_test_data(opt.save_test_data_file,
                           optix_renderer.photons(), empty_caustic, hdr);
        }

        // -- Interactive debug window (always) ------------------------
        // Determine initial scene index from compile-time define
        #if defined(SCENE_CORNELL_BOX)
            g_app.active_scene_index = 0;
        #elif defined(SCENE_CONFERENCE)
            g_app.active_scene_index = 1;
        #elif defined(SCENE_LIVING_ROOM)
            g_app.active_scene_index = 2;
        #elif defined(SCENE_SIBENIK)
            g_app.active_scene_index = 5;
        #endif
        g_app.active_cam_speed = SCENE_CAM_SPEED;

        run_interactive(optix_renderer, camera, opt, scene);

    } catch (const std::exception& e) {
        std::cerr << "[Fatal] " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n== Done ==\n";
    return 0;
}
