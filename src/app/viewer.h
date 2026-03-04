// ---------------------------------------------------------------------
// app/viewer.h -- GLFW window, event loop, input, overlay rendering
// ---------------------------------------------------------------------
//
// Extracted from main.cpp (§1.7).  The viewer owns the interactive
// debug window, all GLFW callbacks, the progressive render path,
// debug overlays, and PNG output.
// ---------------------------------------------------------------------
#pragma once

#include "core/config.h"
#include "debug/debug.h"            // DebugState
#include "debug/stats_collector.h"  // RendererStats, ENABLE_STATS
#include "renderer/camera.h"
#include "renderer/renderer.h"      // FrameBuffer
#include "postfx/postfx_params.h"   // PostFxParams
#include "scene/scene.h"
#include <string>
#include <chrono>

// Forward declarations
class OptixRenderer;

// -- Options parsed from the command line -----------------------------
struct Options {
    std::string scene_file;
    std::string output_file = "output/render.png";
    std::string save_test_data_file;   // if non-empty, dump photons to this binary file

    RenderConfig config;
};

// -- Application state accessible from main ---------------------------
struct AppState {
    DebugState debug;
    bool       snapshot_requested = false;  // R key: save PNG + EXR snapshot
    bool       volume_enabled   = DEFAULT_VOLUME_ENABLED;  // V key toggle
    bool       use_dense_grid   = true;                        // G key toggle

    // Scene switching (keys 1-9, 0)
    int        scene_switch_requested = -1;  // -1 = none, 0-9 = profile index
    int        active_scene_index     = -1;  // currently loaded scene profile index
    float      active_cam_speed       = SCENE_CAM_SPEED; // runtime cam speed

    // Light brightness scaling (runtime, +/- keys)
    float      light_scale            = DEFAULT_LIGHT_SCALE;
    bool       light_scale_changed    = false;  // triggers photon re-trace

    // Photon retrace request (P key)
    bool       photon_retrace_requested = false;

    // Guided path tracing toggle (T key)
    bool       guided_enabled    = DEFAULT_USE_GUIDE;   // true = photon-guided, false = brute force
    // Histogram-only conclusion mode (C key, only effective when guided)
    bool       histogram_only    = false;

    // Show statistics overlay (S key)
    bool       show_stats_overlay = false;

    // Animation sequence (N key)
    bool       animation_requested = false;  // N key: render animation sequence

    // Render timing: timestamp when progressive accumulation started
    std::chrono::steady_clock::time_point render_start_time;
    bool       render_timing_active = false;
    double     last_render_ms       = 0.0;  // elapsed ms at last snapshot

    // Collected statistics (populated on snapshot)
    RendererStats last_stats;

    // Camera angles (yaw/pitch/roll in radians)
    float      yaw   = 0.f;     // horizontal angle
    float      pitch = 0.f;     // vertical angle
    float      roll  = 0.f;     // roll angle (Q=CCW, E=CW)
    bool       mouse_captured = true;  // start with mouse captured
    double     last_mx = 0.0, last_my = 0.0;
    bool       first_mouse = true;
    bool       camera_moved = false;  // flag to reset accumulation

    // ── Idle tracking ───────────────────────────────────────────────
    std::chrono::steady_clock::time_point last_input_time;   // timestamp of last user interaction
    bool idle_rendering_active = false;  // true = full-quality accumulation after idle timeout
    int  idle_photon_seed      = 0;      // incrementing seed for multi-map decorrelation
    int  base_num_photons      = 0;      // original config.num_photons (before idle boost)

    // Post-FX params (bloom, etc.) — per-scene, saved in camera JSON
    PostFxParams postfx;
};

// Global application state — shared between main() and viewer internals
AppState& app_state();

// -- Camera persistence -----------------------------------------------
// Derives the scene folder from a scene profile's obj_path
// (e.g. "cornell_box/cornellbox.obj" -> "scenes/cornell_box")
std::string scene_folder_from_profile(const char* obj_path);

bool save_camera_to_file(const Camera& cam, float yaw, float pitch, float roll,
                         float light_scale,
                         const std::string& scene_folder,
                         const PostFxParams* postfx = nullptr);

bool load_camera_from_file(Camera& cam, float& yaw, float& pitch, float& roll,
                           float& light_scale,
                           const std::string& scene_folder,
                           std::string* out_envmap_path = nullptr,
                           float3* out_envmap_rotation = nullptr,
                           float* out_envmap_scale = nullptr,
                           PostFxParams* out_postfx = nullptr);

// -- PNG output -------------------------------------------------------
bool write_png(const std::string& filename, const FrameBuffer& fb);

// -- Interactive debug window -----------------------------------------
void run_interactive(
    OptixRenderer& optix_renderer,
    Camera& camera,
    Options& opt,
    Scene& scene);
