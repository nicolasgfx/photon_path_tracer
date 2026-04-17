#pragma once
// ─────────────────────────────────────────────────────────────────────
// app/viewer.h – Interactive viewer API (v5, RGB)
//
// Owns the GLFW window, input handling, progressive render loop,
// debug overlays, and PNG/EXR output.  Integrates all v5 pipeline
// stages: scene → accel → lighting → integrator → postfx → display.
//
// The Viewer class replaces the old run_interactive() free function.
// ─────────────────────────────────────────────────────────────────────
#include "app/render_config.h"
#include "app/render_session.h"
#include "app/cli_args.h"
#include "core/config.h"
#include "core/types.h"
#include "core/stage_metrics.h"
#include "postfx/postfx_params.h"

#include <string>
#include <chrono>
#include <memory>

// Forward declarations
struct Scene;
struct GLFWwindow;

// ── Debug visualization state ───────────────────────────────────────

struct DebugState {
    // Render mode cycling (TAB)
    bool  show_normals    = false;
    bool  show_depth      = false;
    bool  show_material   = false;

    // Visualization toggles (F-keys)
    bool  show_caustic_only       = false;  // F1: isolate caustic buffer
    bool  show_stats_overlay       = false;  // S: performance stats

    // Overlay info
    bool  show_noise_map     = false;  // N: per-pixel noise heatmap
    bool  show_convergence   = false;  // C: convergence rate display
};

// ── Application state ───────────────────────────────────────────────

struct AppState {
    DebugState debug;

    // Snapshot
    bool  snapshot_requested = false;  // R: save PNG + EXR

    // Camera control
    float yaw   = 0.f;
    float pitch = 0.f;
    float roll  = 0.f;
    float cam_speed = 0.1f;
    bool  mouse_captured = true;
    double last_mx = 0.0, last_my = 0.0;
    bool  first_mouse = true;
    bool  camera_moved = false;  // flag to reset accumulation

    // Scene switching (keys 1-9, 0)
    int   scene_switch_requested = -1;
    int   active_scene_index     = -1;

    // Light intensity (runtime +/- keys)
    float light_scale          = DEFAULT_LIGHT_SCALE;
    bool  light_scale_changed  = false;

    // Depth of field controls
    bool  dof_toggle_requested  = false;
    bool  dof_focus_requested   = false;  // F: auto-focus at look direction
    int   dof_aperture_dir      = 0;      // [/]: widen/narrow aperture

    // Idle tracking (progressive rendering)
    std::chrono::steady_clock::time_point last_input_time;
    std::chrono::steady_clock::time_point render_start_time;
    bool  idle_rendering_active = false;
    bool  render_timing_active  = false;
    double last_render_ms       = 0.0;

    // Post-processing params (per-scene, saved in camera JSON)
    PostFxParams postfx;

    // Accumulated SPP
    int   current_spp = 0;
};

// Global accessor
AppState& app_state();

// ── Viewer class ────────────────────────────────────────────────────

class Viewer {
public:
    // Initialize window + OpenGL context.
    // Returns false on failure.
    bool init(int width, int height, const char* title);

    // Initialize GPU pipeline.  Call after init() and before run().
    // ptx_path is the path to the compiled .optix_ir file.
    bool init_gpu(const Scene& scene, const RenderConfig& config,
                  const std::string& ptx_path);

    // Run the interactive render loop until the user closes the window.
    // The viewer takes ownership of the render loop.
    void run(Scene& scene, const Options& opt);

    // Shut down (destroy window, release GL resources).
    void shutdown();

    // Accessors
    GLFWwindow* window() const { return window_; }
    const FrameBuffer& framebuffer() const { return fb_; }
    const RenderConfig& config() const { return config_; }

private:
    GLFWwindow*  window_ = nullptr;
    FrameBuffer  fb_;
    RenderConfig config_;
    std::unique_ptr<RenderSession> gpu_;
    Camera       cam_;  // persistent camera (FPS-style)
    bool         cam_flip_x_ = false;  // scene loader handedness flip
    std::string  ptx_path_;       // stored for scene reloading
    std::string  scene_folder_;   // current scene's folder (for camera save)

    // OpenGL display resources (compat-profile, fixed-function quad)
    unsigned int gl_tex_     = 0;
    bool         gl_inited_  = false;
    void init_gl_display();

    // Frame timing
    double frame_time_ms_ = 0.0;
    int    frame_count_    = 0;

    // ── Internal methods ────────────────────────────────────────────
    void process_input(float dt);
    void render_frame(Scene& scene);
    void display_frame();
    void handle_snapshot(const std::string& output_path);
    bool switch_scene(Scene& scene, int preset_idx);

    // Camera
    void update_camera_from_input(float dt);
    void apply_cam_flip();  // apply handedness flip after cam_.update()
    void reset_accumulation();
};

// ── Camera persistence ──────────────────────────────────────────────

std::string scene_folder_from_path(const std::string& scene_path);

bool save_camera_json(const std::string& scene_folder,
                      float3 pos, float yaw, float pitch, float roll,
                      float light_scale,
                      const PostFxParams* postfx = nullptr,
                      float fov_deg = 0.f,
                      const Camera* cam = nullptr);

bool load_camera_json(const std::string& scene_folder,
                      float3& pos, float& yaw, float& pitch, float& roll,
                      float& light_scale,
                      PostFxParams* postfx = nullptr,
                      float* fov_deg = nullptr,
                      Camera* cam = nullptr);

// ── PNG output ──────────────────────────────────────────────────────

bool write_png(const std::string& filename, const FrameBuffer& fb);

// ── EXR (HDR) output ────────────────────────────────────────────────

bool write_exr(const std::string& filename,
               const std::vector<float>& hdr_rgba,
               int width, int height);
