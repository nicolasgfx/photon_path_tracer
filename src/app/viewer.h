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
#include "renderer/camera.h"
#include "renderer/renderer.h"      // FrameBuffer
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
    bool       snapshot_requested = false;  // R key: save PNG + JSON snapshot
    bool       volume_enabled   = DEFAULT_VOLUME_ENABLED;  // V key toggle
    bool       use_dense_grid   = DEFAULT_USE_DENSE_GRID;  // G key toggle

    // Scene switching (keys 1-8)
    int        scene_switch_requested = -1;  // -1 = none, 0-7 = profile index
    int        active_scene_index     = -1;  // currently loaded scene profile index
    float      active_cam_speed       = SCENE_CAM_SPEED; // runtime cam speed

    // Light brightness scaling (runtime, +/- keys)
    float      light_scale            = DEFAULT_LIGHT_SCALE;
    bool       light_scale_changed    = false;  // triggers photon re-trace

    // Photon retrace request (P key)
    bool       photon_retrace_requested = false;

    // Camera angles (yaw/pitch in radians)
    float      yaw   = 0.f;     // horizontal angle
    float      pitch = 0.f;     // vertical angle
    bool       mouse_captured = true;  // start with mouse captured
    double     last_mx = 0.0, last_my = 0.0;
    bool       first_mouse = true;
    bool       camera_moved = false;  // flag to reset accumulation
};

// Global application state — shared between main() and viewer internals
AppState& app_state();

// -- Camera persistence -----------------------------------------------
// Derives the scene folder from a scene profile's obj_path
// (e.g. "cornell_box/cornellbox.obj" -> "scenes/cornell_box")
std::string scene_folder_from_profile(const char* obj_path);

bool save_camera_to_file(const Camera& cam, float yaw, float pitch,
                         float light_scale,
                         const std::string& scene_folder);

bool load_camera_from_file(Camera& cam, float& yaw, float& pitch,
                           float& light_scale,
                           const std::string& scene_folder);

// -- PNG output -------------------------------------------------------
bool write_png(const std::string& filename, const FrameBuffer& fb);

// -- Interactive debug window -----------------------------------------
void run_interactive(
    OptixRenderer& optix_renderer,
    Camera& camera,
    Options& opt,
    Scene& scene);
