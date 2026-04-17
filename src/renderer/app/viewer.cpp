// ─────────────────────────────────────────────────────────────────────
// app/viewer.cpp – Interactive viewer implementation (v5, RGB)
//
// GLFW window, progressive render loop, input handling, overlays.
// Integrates all v5 pipeline stages via RenderSession.
// ─────────────────────────────────────────────────────────────────────
#ifdef _WIN32
#define NOMINMAX
#endif

#include "app/viewer.h"
#include "core/config.h"
#include "core/camera.h"
#include "scene/scene.h"
#include "scene/pbrt/pbrt_loader.h"
#include "scene/obj_loader.h"
#include "analyze/prepass_metrics.h"
#include "analyze/scene_profile_applicator.h"

// GLFW + OpenGL
#include <GLFW/glfw3.h>

// stb_image_write for PNG output
#include "stb_image_write.h"
#include "app/font_overlay.h"

#include "tinyexr.h"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <filesystem>

// ── Global application state ────────────────────────────────────────

static AppState s_app_state;

AppState& app_state() { return s_app_state; }

// ── GLFW callbacks ──────────────────────────────────────────────────

static void key_callback(GLFWwindow* window, int key, int /*scancode*/,
                          int action, int mods) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    auto& st = app_state();

    st.last_input_time = std::chrono::steady_clock::now();

    switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;

        // Snapshot
        case GLFW_KEY_R:
            st.snapshot_requested = true;
            break;

        // Caustic-only isolation
        case GLFW_KEY_F1:
            st.debug.show_caustic_only = !st.debug.show_caustic_only;
            std::printf("[Viewer] Caustic-only: %s\n",
                       st.debug.show_caustic_only ? "ON" : "OFF");
            break;

        // Stats overlay
        case GLFW_KEY_S:
            if (!(mods & GLFW_MOD_CONTROL))
                st.debug.show_stats_overlay = !st.debug.show_stats_overlay;
            break;

        // Noise map
        case GLFW_KEY_N:
            st.debug.show_noise_map = !st.debug.show_noise_map;
            break;

        // Convergence display
        case GLFW_KEY_C:
            st.debug.show_convergence = !st.debug.show_convergence;
            break;

        // ACES tonemapping toggle
        case GLFW_KEY_F2:
            st.postfx.use_aces = !st.postfx.use_aces;
            std::printf("[Viewer] ACES tonemap: %s\n",
                       st.postfx.use_aces ? "ON" : "OFF");
            break;

        // Light scale
        case GLFW_KEY_EQUAL:  // + key
            st.light_scale = std::min(st.light_scale * LIGHT_SCALE_STEP,
                                       LIGHT_SCALE_MAX);
            st.light_scale_changed = true;
            std::printf("[Viewer] Light scale: %.3f\n", st.light_scale);
            break;
        case GLFW_KEY_MINUS:
            st.light_scale = std::max(st.light_scale / LIGHT_SCALE_STEP,
                                       LIGHT_SCALE_MIN);
            st.light_scale_changed = true;
            std::printf("[Viewer] Light scale: %.3f\n", st.light_scale);
            break;

        // Depth of field toggle
        case GLFW_KEY_O:
            st.dof_toggle_requested = true;
            break;

        // Auto-focus at screen center
        case GLFW_KEY_F:
            st.dof_focus_requested = true;
            break;

        // Aperture: [ = wider (lower f-number), ] = narrower (higher f-number)
        case GLFW_KEY_LEFT_BRACKET:
            st.dof_aperture_dir = -1;
            break;
        case GLFW_KEY_RIGHT_BRACKET:
            st.dof_aperture_dir = +1;
            break;

        // Mouse capture toggle
        case GLFW_KEY_LEFT_ALT:
            st.mouse_captured = !st.mouse_captured;
            glfwSetInputMode(window,
                             GLFW_CURSOR,
                             st.mouse_captured ? GLFW_CURSOR_DISABLED
                                               : GLFW_CURSOR_NORMAL);
            st.first_mouse = true;
            break;

        // Scene switching: 1-9,0 for presets 0-9; Shift+1.. for presets 10+
        default:
            if (key >= GLFW_KEY_1 && key <= GLFW_KEY_9) {
                if (mods & GLFW_MOD_SHIFT)
                    st.scene_switch_requested = 10 + (key - GLFW_KEY_1);
                else
                    st.scene_switch_requested = key - GLFW_KEY_1;
            } else if (key == GLFW_KEY_0) {
                st.scene_switch_requested = 9;
            }

            break;
    }
}

static void mouse_button_callback(GLFWwindow* /*window*/, int /*button*/,
                                   int /*action*/, int /*mods*/) {
    app_state().last_input_time = std::chrono::steady_clock::now();
}

static void cursor_pos_callback(GLFWwindow* /*window*/, double xpos, double ypos) {
    auto& st = app_state();
    st.last_input_time = std::chrono::steady_clock::now();

    if (!st.mouse_captured) return;

    if (st.first_mouse) {
        st.last_mx = xpos;
        st.last_my = ypos;
        st.first_mouse = false;
        return;
    }

    float sensitivity = 0.002f;
    float dx = (float)(xpos - st.last_mx) * sensitivity;
    float dy = (float)(ypos - st.last_my) * sensitivity;
    st.last_mx = xpos;
    st.last_my = ypos;

    st.yaw   += dx;
    st.pitch -= dy;
    st.pitch = std::max(-1.5f, std::min(1.5f, st.pitch));  // clamp ±~86°
    st.camera_moved = true;
}

static void scroll_callback(GLFWwindow* /*window*/, double /*xoff*/, double yoff) {
    auto& st = app_state();
    st.last_input_time = std::chrono::steady_clock::now();
    st.cam_speed *= (yoff > 0) ? 1.1f : 0.9f;
    st.cam_speed = std::max(0.001f, std::min(10.f, st.cam_speed));
}

static void window_focus_callback(GLFWwindow* /*window*/, int focused) {
    if (focused == GLFW_TRUE) {
        auto& st = app_state();
        st.first_mouse = true;  // re-latch cursor pos on next move
        st.last_input_time = std::chrono::steady_clock::now();
    }
}

// ── Viewer implementation ───────────────────────────────────────────

bool Viewer::init(int width, int height, const char* title) {
    if (!glfwInit()) {
        std::fprintf(stderr, "[Viewer] Failed to initialize GLFW\n");
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window_) {
        std::fprintf(stderr, "[Viewer] Failed to create GLFW window\n");
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(0);  // no vsync — let GPU run flat-out

    // Set callbacks
    glfwSetKeyCallback(window_, key_callback);
    glfwSetMouseButtonCallback(window_, mouse_button_callback);
    glfwSetCursorPosCallback(window_, cursor_pos_callback);
    glfwSetScrollCallback(window_, scroll_callback);
    glfwSetWindowFocusCallback(window_, window_focus_callback);

    // Capture mouse initially
    glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Init framebuffer
    fb_.resize(width, height);
    config_.image_width  = width;
    config_.image_height = height;

    std::printf("[Viewer] Window created: %d×%d\n", width, height);
    return true;
}

bool Viewer::init_gpu(const Scene& scene, const RenderConfig& config,
                      const std::string& ptx_path) {
    ptx_path_ = ptx_path;
    gpu_ = std::make_unique<RenderSession>();
    if (!gpu_->init(scene, config, ptx_path)) {
        std::fprintf(stderr, "[Viewer] GPU pipeline init failed\n");
        return false;
    }
    return true;
}

void Viewer::run(Scene& scene, const Options& opt) {
    config_ = opt.config;
    auto& st = app_state();
    st.last_input_time  = std::chrono::steady_clock::now();
    st.render_start_time = st.last_input_time;
    st.postfx = config_.postfx;
    st.light_scale = config_.light_scale;
    st.active_scene_index = opt.initial_preset_index;

    // Init GL display resources for texture blitting
    init_gl_display();

    std::printf("[Viewer] Entering render loop  GPU ready: %s\n",
               gpu_ && gpu_->is_ready() ? "yes" : "no");

    // Set up camera from scene data or preset defaults
    if (scene.scene_cam_valid) {
        cam_.position = scene.scene_cam_position;
        cam_.look_at  = scene.scene_cam_look_at;
        cam_.up       = scene.scene_cam_up;
        cam_.fov_deg  = scene.scene_cam_fov;
    } else if (opt.initial_preset_index >= 0) {
        const auto& preset = SCENE_PRESETS[opt.initial_preset_index];
        cam_.position = make_f3(preset.cam_pos[0], preset.cam_pos[1], preset.cam_pos[2]);
        cam_.look_at  = make_f3(preset.cam_lookat[0], preset.cam_lookat[1], preset.cam_lookat[2]);
        cam_.up       = make_f3(0.f, 1.f, 0.f);
        cam_.fov_deg  = preset.cam_fov;
    } else {
        cam_.position = scene.scene_cam_position;
        cam_.look_at  = scene.scene_cam_look_at;
        cam_.up       = make_f3(0.f, 1.f, 0.f);
        cam_.fov_deg  = scene.scene_cam_fov;
    }
    cam_.width    = config_.image_width;
    cam_.height   = config_.image_height;

    // Derive initial yaw/pitch from scene camera direction
    float3 dir = normalize(cam_.look_at - cam_.position);
    st.yaw   = atan2f(dir.x, -dir.z);
    st.pitch = asinf(std::max(-1.f, std::min(1.f, dir.y)));
    st.roll  = 0.f;

    // Restore persisted viewer camera (overrides embedded camera if present).
    scene_folder_ = scene_folder_from_path(opt.scene_file);
    if (load_camera_json(scene_folder_,
                         cam_.position, st.yaw, st.pitch, st.roll,
                         st.light_scale, &st.postfx,
                         &cam_.fov_deg, &cam_)) {
        std::printf("[Viewer] Restored camera from %s/saved_camera.json\n",
                   scene_folder_.c_str());
    }

    cam_.update();
    cam_flip_x_ = scene.scene_cam_flip_x;
    apply_cam_flip();

    std::printf("[Viewer] Camera: pos=(%.2f,%.2f,%.2f) look=(%.2f,%.2f,%.2f) fov=%.1f flip=%d\n",
               cam_.position.x, cam_.position.y, cam_.position.z,
               cam_.look_at.x, cam_.look_at.y, cam_.look_at.z,
               cam_.fov_deg, cam_flip_x_ ? 1 : 0);

    // ── View-dependent pre-pass (before render loop) ────────────────
    if (gpu_ && gpu_->is_ready() && config_.prepass_spp > 0) {
        PrePassMetrics pm = gpu_->run_prepass(cam_, config_);
        SceneProfile sp = opt.scene_profile;
        refine_scene_profile(sp, config_, pm);
    }

    auto last_frame = std::chrono::steady_clock::now();

    while (!glfwWindowShouldClose(window_)) {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_frame).count();
        last_frame = now;

        glfwPollEvents();

        // Process keyboard movement (WASD + QE roll)
        process_input(dt);

        // Check if camera moved — reset accumulation and exit idle mode
        if (st.camera_moved) {
            reset_accumulation();
            st.idle_rendering_active = false;
            st.render_timing_active = false;
            st.camera_moved = false;
        }

        // Depth of field toggle
        if (st.dof_toggle_requested) {
            cam_.dof_enabled = !cam_.dof_enabled;
            cam_.update();
            reset_accumulation();
            std::printf("[Viewer] Depth of field: %s\n",
                       cam_.dof_enabled ? "ON" : "OFF");
            st.dof_toggle_requested = false;
        }

        // Auto-focus: set focus distance to camera→look_at distance
        if (st.dof_focus_requested) {
            float3 diff = cam_.look_at - cam_.position;
            float dist = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
            if (dist > 1e-4f) {
                cam_.dof_focus_dist = dist;
                cam_.update();
                reset_accumulation();
                std::printf("[Viewer] Focus distance: %.4f\n", cam_.dof_focus_dist);
            }
            st.dof_focus_requested = false;
        }

        // Aperture: [ = wider (lower f-number), ] = narrower (higher f-number)
        if (st.dof_aperture_dir != 0) {
            constexpr float FNUM_STEP = 1.4142f;  // √2 — one full f-stop
            if (st.dof_aperture_dir > 0)
                cam_.dof_f_number = std::min(cam_.dof_f_number * FNUM_STEP, 64.f);
            else
                cam_.dof_f_number = std::max(cam_.dof_f_number / FNUM_STEP, 1.0f);
            cam_.update();
            reset_accumulation();
            std::printf("[Viewer] f-number: f/%.1f\n", cam_.dof_f_number);
            st.dof_aperture_dir = 0;
        }

        // Idle timeout → switch from preview to full-quality rendering
        float idle_sec = std::chrono::duration<float>(
            now - st.last_input_time).count();
        if (idle_sec > IDLE_TIMEOUT_SEC && !st.idle_rendering_active) {
            st.idle_rendering_active = true;
            st.render_timing_active = true;
            st.render_start_time = now;
            reset_accumulation();

            std::printf("[Idle] Full-quality mode\n");
        }

        // Render one frame / accumulate
        render_frame(scene);

        // Tonemap + display
        display_frame();

        // ── Window title: scene name + [Preview/Full SPP N] ────────
        {
            char title[256];
            const char* sname = (st.active_scene_index >= 0 &&
                                 st.active_scene_index < NUM_SCENE_PRESETS)
                ? get_scene_preset(st.active_scene_index).display_name
                : "Scene";
            char hk = (st.active_scene_index >= 0 &&
                       st.active_scene_index < NUM_SCENE_PRESETS)
                ? scene_hotkey_char(st.active_scene_index) : '?';
            if (st.idle_rendering_active)
                std::snprintf(title, sizeof(title),
                    "%c \xe2\x80\x93 %s  [Full SPP %d]", hk, sname, st.current_spp);
            else
                std::snprintf(title, sizeof(title),
                    "%c \xe2\x80\x93 %s  [Preview SPP %d]", hk, sname, st.current_spp);
            glfwSetWindowTitle(window_, title);
        }

        // Handle snapshot request
        if (st.snapshot_requested) {
            handle_snapshot(opt.output_file);
            st.snapshot_requested = false;
        }

        // Scene switching
        if (st.scene_switch_requested >= 0) {
            int idx = st.scene_switch_requested;
            st.scene_switch_requested = -1;
            switch_scene(scene, idx);
        }

        glfwSwapBuffers(window_);
        ++frame_count_;
    }

    std::printf("[Viewer] Render loop exited after %d frames\n", frame_count_);

    if (!scene_folder_.empty()) {
        save_camera_json(scene_folder_, cam_.position,
                         st.yaw, st.pitch, st.roll,
                         st.light_scale, &st.postfx,
                         cam_.fov_deg, &cam_);
        std::printf("[Viewer] Camera saved to %s/saved_camera.json\n",
                   scene_folder_.c_str());
    }
}

void Viewer::shutdown() {
    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    glfwTerminate();
    std::printf("[Viewer] Shutdown complete\n");
}

// ── Input processing ────────────────────────────────────────────────

void Viewer::process_input(float dt) {
    if (!window_) return;
    auto& st = app_state();

    // Compute forward/right/up from yaw/pitch/roll
    float3 forward = make_f3(
        sinf(st.yaw) * cosf(st.pitch),
        sinf(st.pitch),
        -cosf(st.yaw) * cosf(st.pitch));
    float3 right_unrolled = normalize(cross(forward, make_f3(0.f, 1.f, 0.f)));
    float3 up_perp = cross(right_unrolled, forward);
    // Apply roll: rotate right/up around forward axis
    float3 right  = right_unrolled * cosf(st.roll) + up_perp * sinf(st.roll);
    float3 up_dir = up_perp * cosf(st.roll) - right_unrolled * sinf(st.roll);

    float speed = st.cam_speed * dt;
    if (glfwGetKey(window_, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        speed *= 3.f;

    float3 move = make_f3(0.f, 0.f, 0.f);
    if (glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS) move = move + forward * speed;
    if (glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS &&
        !(glfwGetKey(window_, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS))
        move = move - forward * speed;
    if (glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS) move = move - right * speed;
    if (glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS) move = move + right * speed;
    if (glfwGetKey(window_, GLFW_KEY_SPACE) == GLFW_PRESS)        move = move + up_dir * speed;
    if (glfwGetKey(window_, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) move = move - up_dir * speed;

    // Q/E roll
    constexpr float kRollSpeed = 1.0f;
    if (glfwGetKey(window_, GLFW_KEY_Q) == GLFW_PRESS) {
        st.roll += kRollSpeed * dt;
        st.camera_moved = true;
        st.last_input_time = std::chrono::steady_clock::now();
    }
    if (glfwGetKey(window_, GLFW_KEY_E) == GLFW_PRESS) {
        st.roll -= kRollSpeed * dt;
        st.camera_moved = true;
        st.last_input_time = std::chrono::steady_clock::now();
    }

    if (move.x != 0.f || move.y != 0.f || move.z != 0.f) {
        cam_.position = cam_.position + move;
        st.camera_moved = true;
        st.last_input_time = std::chrono::steady_clock::now();
        st.idle_rendering_active = false;
    }

    // Always update look_at / up from yaw/pitch/roll
    cam_.look_at = cam_.position + forward;
    cam_.up = up_dir;
    cam_.update();
    apply_cam_flip();
}

void Viewer::render_frame(Scene& /*scene*/) {
    auto& st = app_state();
    if (!gpu_ || !gpu_->is_ready()) return;

    // During navigation (non-idle), cap bounces to PREVIEW_MAX_BOUNCES for
    // responsive feedback — matches the old renderer's preview_mode behavior.
    RenderConfig rc = config_;
    if (!st.idle_rendering_active) {
        rc.max_bounces = std::min(rc.max_bounces, PREVIEW_MAX_BOUNCES);
    }

    gpu_->render_frame(cam_, st.current_spp, rc);
    st.current_spp = gpu_->accumulated_spp();

    // Caustic light tracing (idle mode only — skip during interactive preview)
    if (st.idle_rendering_active && config_.caustic_enabled) {
        gpu_->launch_caustics(cam_, st.current_spp, rc);
    }

    // Apply GPU post-processing (firefly filter → bloom → tonemap)
    PostFxParams pfx = st.postfx;
    pfx.exposure = st.light_scale * config_.exposure;
    pfx.denoiser_blend = config_.denoiser_blend;

    // Suppress firefly filter during early accumulation — at low SPP the
    // median-based filter treats valid bright indirect pixels as outliers.
    // Old code had no firefly filter at all; this is a conservative gate.
    if (st.current_spp < 8)
        pfx.firefly_enabled = false;

    pfx.caustic_only = st.debug.show_caustic_only;

    gpu_->apply_postfx(pfx);

    // Download sRGB result directly (skip CPU tonemap)
    gpu_->download_srgb(fb_.srgb);

    for (int i = 0; i < fb_.num_pixels(); ++i)
        fb_.sample_count[i] = (float)st.current_spp;
}

void Viewer::display_frame() {
    // sRGB data is already produced by GPU PostFx in render_frame().
    // Upload to GL texture and draw fullscreen quad (compat-profile)
    if (gl_inited_ && !fb_.srgb.empty()) {
        glBindTexture(GL_TEXTURE_2D, gl_tex_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                        fb_.width, fb_.height,
                        GL_RGBA, GL_UNSIGNED_BYTE,
                        fb_.srgb.data());

        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw fullscreen textured quad (fixed-function pipeline)
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, gl_tex_);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glBegin(GL_QUADS);
        // V texcoords are flipped (1→0 top-to-bottom) because:
        //   - Our sRGB buffer row 0 = scene top (cam_v negated)
        //   - GL stores row 0 as texture bottom (y=0 in texcoord)
        //   - So screen-top must sample texcoord.y=0, screen-bottom texcoord.y=1
        glTexCoord2f(0.f, 1.f); glVertex2f(-1.f, -1.f);
        glTexCoord2f(1.f, 1.f); glVertex2f( 1.f, -1.f);
        glTexCoord2f(1.f, 0.f); glVertex2f( 1.f,  1.f);
        glTexCoord2f(0.f, 0.f); glVertex2f(-1.f,  1.f);
        glEnd();

        glDisable(GL_TEXTURE_2D);
    } else {
        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }
}

void Viewer::handle_snapshot(const std::string& /*output_path*/) {
    namespace fs = std::filesystem;
    auto& st = app_state();

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

    if (gpu_) {
        PostFxParams pfx = st.postfx;
        pfx.exposure = st.light_scale * config_.exposure;

        // 1. Save raw PNG (current framebuffer, no denoiser)
        std::string raw_path = prefix + "_raw.png";
        write_png(raw_path, fb_);

        // 2. Save raw HDR EXR (linear float, no denoiser)
        std::string exr_path = prefix + ".exr";
        gpu_->apply_postfx(pfx);  // non-denoised postfx for HDR
        std::vector<float> hdr;
        gpu_->download_hdr(hdr);
        if (!hdr.empty())
            write_exr(exr_path, hdr, fb_.width, fb_.height);

        // 3. Save denoised PNG (re-apply postfx with OptiX denoiser)
        std::string denoised_path = prefix + "_denoised.png";
        gpu_->apply_snapshot_postfx(pfx);
        {
            std::vector<uint8_t> denoised_srgb;
            gpu_->download_srgb(denoised_srgb);
            FrameBuffer denoised_fb;
            denoised_fb.width  = fb_.width;
            denoised_fb.height = fb_.height;
            denoised_fb.srgb   = std::move(denoised_srgb);
            write_png(denoised_path, denoised_fb);
        }

        // 4. Save caustic map as separate PNG
        std::string caustic_path = prefix + "_caustics.png";
        {
            PostFxParams cpfx = pfx;
            cpfx.caustic_only = true;
            gpu_->apply_postfx(cpfx);
            std::vector<uint8_t> caustic_srgb;
            gpu_->download_srgb(caustic_srgb);
            FrameBuffer caustic_fb;
            caustic_fb.width  = fb_.width;
            caustic_fb.height = fb_.height;
            caustic_fb.srgb   = std::move(caustic_srgb);
            write_png(caustic_path, caustic_fb);
        }

        // 5. Restore display framebuffer to non-denoised state
        gpu_->apply_postfx(pfx);
        gpu_->download_srgb(fb_.srgb);

        std::printf("\n========================================\n");
        std::printf("  [Snapshot] %s (raw)\n", raw_path.c_str());
        std::printf("  [Snapshot] %s (denoised)\n", denoised_path.c_str());
        std::printf("  [Snapshot] %s (caustics)\n", caustic_path.c_str());
        std::printf("  [Snapshot] %s (HDR)\n", exr_path.c_str());
        std::printf("  [Snapshot] %d SPP\n", st.current_spp);
        std::printf("========================================\n\n");
    }
}

void Viewer::reset_accumulation() {
    fb_.clear();
    if (gpu_) gpu_->reset_accumulation();
    app_state().current_spp = 0;
}

void Viewer::update_camera_from_input(float /*dt*/) {
    // All camera movement is handled in process_input() which updates cam_ directly.
}

void Viewer::apply_cam_flip() {
    if (!cam_flip_x_) return;
    cam_.u          = cam_.u * -1.0f;
    cam_.horizontal = cam_.horizontal * -1.0f;
    cam_.lower_left = cam_.position
                    - cam_.horizontal * 0.5f
                    - cam_.vertical   * 0.5f
                    - cam_.w * ((cam_.lens_radius > 0.f)
                                ? cam_.dof_focus_dist : 1.0f);
}

// ── Scene switching ─────────────────────────────────────────────────

static std::string get_extension_lower(const std::string& path) {
    auto dot = path.rfind('.');
    if (dot == std::string::npos) return "";
    std::string ext = path.substr(dot);
    for (auto& c : ext) c = (char)std::tolower((unsigned char)c);
    return ext;
}

bool Viewer::switch_scene(Scene& scene, int preset_idx) {
    auto& st = app_state();

    if (preset_idx < 0 || preset_idx >= NUM_SCENE_PRESETS) {
        std::printf("[Viewer] Invalid scene index: %d\n", preset_idx);
        return false;
    }
    if (preset_idx == st.active_scene_index) {
        std::printf("[Viewer] Already on scene %d\n", preset_idx);
        return false;
    }

    const ScenePreset& preset = SCENE_PRESETS[preset_idx];
    std::string scene_path = std::string(SCENES_DIR) + "/" + preset.obj_path;
    std::printf("[Viewer] Switching to scene %c: %s (%s)\n",
               scene_hotkey_char(preset_idx), preset.display_name, scene_path.c_str());

    // ── Save camera for current scene before switching ──────────────
    if (!scene_folder_.empty()) {
        save_camera_json(scene_folder_, cam_.position,
                         st.yaw, st.pitch, st.roll,
                         st.light_scale, &st.postfx,
                         cam_.fov_deg, &cam_);
    }

    // ── Load new scene ──────────────────────────────────────────────
    Scene new_scene;
    bool load_ok = false;
    auto t0 = std::chrono::high_resolution_clock::now();

    std::string ext = get_extension_lower(scene_path);
    if (ext == ".pbrt") {
        load_ok = load_pbrt(scene_path, new_scene);
    } else if (ext == ".obj") {
        load_ok = load_obj(scene_path, new_scene);
    } else {
        std::fprintf(stderr, "[Viewer] Unsupported scene format: %s\n", ext.c_str());
        return false;
    }

    if (!load_ok) {
        std::fprintf(stderr, "[Viewer] Failed to load scene: %s\n", scene_path.c_str());
        return false;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::printf("[Viewer] Scene loaded: %.1f ms  (%zu tris, %zu mats)\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                new_scene.triangles.size(), new_scene.materials.size());

    // ── Normalize & rotate ──────────────────────────────────────────
    if (!preset.is_reference)
        new_scene.normalize_to_reference();
    if (preset.rotate_x_180)
        new_scene.rotate_x_180();

    // ── Finalize scene ──────────────────────────────────────────────
    new_scene.compute_bounds();
    new_scene.build_emissive_distribution();

    // ── Replace scene ───────────────────────────────────────────────
    scene = std::move(new_scene);

    // ── Rebuild GPU pipeline ────────────────────────────────────────
    gpu_ = std::make_unique<RenderSession>();
    if (!gpu_->init(scene, config_, ptx_path_)) {
        std::fprintf(stderr, "[Viewer] GPU pipeline reinit failed for scene %d\n",
                     preset_idx);
        return false;
    }

    // ── Reset camera ────────────────────────────────────────────────
    if (scene.scene_cam_valid) {
        cam_.position = scene.scene_cam_position;
        cam_.look_at  = scene.scene_cam_look_at;
        cam_.up       = scene.scene_cam_up;
        cam_.fov_deg  = scene.scene_cam_fov;
    } else {
        cam_.position = make_f3(preset.cam_pos[0], preset.cam_pos[1], preset.cam_pos[2]);
        cam_.look_at  = make_f3(preset.cam_lookat[0], preset.cam_lookat[1], preset.cam_lookat[2]);
        cam_.up       = make_f3(0.f, 1.f, 0.f);
        cam_.fov_deg  = preset.cam_fov;
    }
    cam_.width  = config_.image_width;
    cam_.height = config_.image_height;

    // Derive yaw/pitch from camera direction
    float3 dir = normalize(cam_.look_at - cam_.position);
    st.yaw   = atan2f(dir.x, -dir.z);
    st.pitch = asinf(std::max(-1.f, std::min(1.f, dir.y)));
    st.roll  = 0.f;
    st.cam_speed = preset.cam_speed;
    st.light_scale = DEFAULT_LIGHT_SCALE;
    st.postfx = config_.postfx;

    // Restore persisted viewer camera (overrides embedded camera if present).
    scene_folder_ = scene_folder_from_path(scene_path);
    if (load_camera_json(scene_folder_,
                         cam_.position, st.yaw, st.pitch, st.roll,
                         st.light_scale, &st.postfx,
                         &cam_.fov_deg, &cam_)) {
        std::printf("[Viewer] Restored camera from %s/saved_camera.json\n",
                   scene_folder_.c_str());
    }

    cam_.update();

    // Apply handedness flip if the scene loader flagged it
    cam_flip_x_ = scene.scene_cam_flip_x;
    apply_cam_flip();

    // ── Reset accumulation + state ──────────────────────────────────
    reset_accumulation();
    st.last_input_time = std::chrono::steady_clock::now();
    st.idle_rendering_active = false;
    st.render_timing_active = false;
    st.camera_moved = false;
    st.active_scene_index = preset_idx;
    st.light_scale_changed = true;

    std::printf("[Viewer] Scene switch complete: %s (key %c)\n",
               preset.display_name, scene_hotkey_char(preset_idx));
    return true;
}

// ── GL display initialisation ───────────────────────────────────────

void Viewer::init_gl_display() {
    if (gl_inited_) return;

    glGenTextures(1, &gl_tex_);
    glBindTexture(GL_TEXTURE_2D, gl_tex_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, 0x812F);  // GL_CLAMP_TO_EDGE
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, 0x812F);  // GL_CLAMP_TO_EDGE
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                 fb_.width, fb_.height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    gl_inited_ = true;
    std::printf("[Viewer] GL display initialised (%dx%d texture)\n",
               fb_.width, fb_.height);
}

// ── Camera persistence ──────────────────────────────────────────────

std::string scene_folder_from_path(const std::string& scene_path) {
    // Extract directory from scene file path
    auto slash = scene_path.find_last_of("/\\");
    if (slash == std::string::npos) return ".";
    return scene_path.substr(0, slash);
}

bool save_camera_json(const std::string& scene_folder,
                      float3 pos, float yaw, float pitch, float roll,
                      float light_scale,
                      const PostFxParams* postfx,
                      float fov_deg,
                      const Camera* cam) {
    std::string path = scene_folder + "/saved_camera.json";
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return false;

    std::fprintf(f, "{\n");
    std::fprintf(f, "  \"position\":      [%.6f, %.6f, %.6f],\n", pos.x, pos.y, pos.z);

    if (cam) {
        std::fprintf(f, "  \"look_at\":       [%.6f, %.6f, %.6f],\n",
                     cam->look_at.x, cam->look_at.y, cam->look_at.z);
    }

    if (fov_deg > 0.f)
        std::fprintf(f, "  \"fov_deg\":       %.6f,\n", fov_deg);

    std::fprintf(f, "  \"yaw\":           %.6f,\n", yaw);
    std::fprintf(f, "  \"pitch\":         %.6f,\n", pitch);
    std::fprintf(f, "  \"roll\":          %.6f,\n", roll);
    std::fprintf(f, "  \"light_scale\":   %.6f", light_scale);

    if (cam) {
        std::fprintf(f, ",\n  \"dof_enabled\":   %s", cam->dof_enabled ? "true" : "false");
        std::fprintf(f, ",\n  \"dof_focus_dist\": %.6f", cam->dof_focus_dist);
        std::fprintf(f, ",\n  \"dof_f_number\":  %.6f", cam->dof_f_number);
        std::fprintf(f, ",\n  \"sensor_height\": %.6f", cam->sensor_height);
        std::fprintf(f, ",\n  \"dof_focus_range\": %.6f", cam->dof_focus_range);
    }

    if (postfx) {
        std::fprintf(f, ",\n  \"bloom_enabled\":   %s", postfx->bloom_enabled ? "true" : "false");
        std::fprintf(f, ",\n  \"bloom_intensity\": %.6f", postfx->bloom_intensity);
        std::fprintf(f, ",\n  \"bloom_radius_h\":  %.6f", postfx->bloom_radius_h);
        std::fprintf(f, ",\n  \"bloom_radius_v\":  %.6f", postfx->bloom_radius_v);
        std::fprintf(f, ",\n  \"firefly_enabled\": %s", postfx->firefly_enabled ? "true" : "false");
        std::fprintf(f, ",\n  \"firefly_threshold\": %.4f", postfx->firefly_threshold);
    }

    std::fprintf(f, "\n}\n");
    fclose(f);
    return true;
}

bool load_camera_json(const std::string& scene_folder,
                      float3& pos, float& yaw, float& pitch, float& roll,
                      float& light_scale,
                      PostFxParams* postfx,
                      float* fov_deg,
                      Camera* cam) {
    // Only restore the viewer-managed sidecar.
    std::string path = scene_folder + "/saved_camera.json";
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return false;

    // Simple JSON-like parsing (not a full parser)
    char buf[4096];
    size_t len = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[len] = '\0';

    // Parse position: try "position" first, then legacy "pos"
    float px, py, pz;
    const char* pp = strstr(buf, "\"position\"");
    if (pp && sscanf(pp, "\"position\"%*[^[][%f, %f, %f]", &px, &py, &pz) == 3) {
        pos = make_f3(px, py, pz);
    } else if ((pp = strstr(buf, "\"pos\"")) != nullptr &&
               sscanf(pp, "\"pos\": [%f, %f, %f]", &px, &py, &pz) == 3) {
        pos = make_f3(px, py, pz);
    }

    const char* p;
    if ((p = strstr(buf, "\"yaw\"")) != nullptr)
        sscanf(p, "\"yaw\": %f", &yaw);
    if ((p = strstr(buf, "\"pitch\"")) != nullptr)
        sscanf(p, "\"pitch\": %f", &pitch);
    if ((p = strstr(buf, "\"roll\"")) != nullptr)
        sscanf(p, "\"roll\": %f", &roll);
    if ((p = strstr(buf, "\"light_scale\"")) != nullptr)
        sscanf(p, "\"light_scale\": %f", &light_scale);

    // FOV
    if (fov_deg) {
        if ((p = strstr(buf, "\"fov_deg\"")) != nullptr)
            sscanf(p, "\"fov_deg\": %f", fov_deg);
    }

    // DOF fields
    if (cam) {
        float val;
        if ((p = strstr(buf, "\"dof_enabled\"")) != nullptr)
            cam->dof_enabled = (strstr(p, "true") != nullptr &&
                               strstr(p, "true") < p + 30);
        if ((p = strstr(buf, "\"dof_focus_dist\"")) != nullptr &&
            sscanf(p, "\"dof_focus_dist\": %f", &val) == 1)
            cam->dof_focus_dist = val;
        if ((p = strstr(buf, "\"dof_f_number\"")) != nullptr &&
            sscanf(p, "\"dof_f_number\": %f", &val) == 1)
            cam->dof_f_number = val;
        if ((p = strstr(buf, "\"sensor_height\"")) != nullptr &&
            sscanf(p, "\"sensor_height\": %f", &val) == 1)
            cam->sensor_height = val;
        if ((p = strstr(buf, "\"dof_focus_range\"")) != nullptr &&
            sscanf(p, "\"dof_focus_range\": %f", &val) == 1)
            cam->dof_focus_range = val;
    }

    // PostFx fields
    if (postfx) {
        float val;
        if ((p = strstr(buf, "\"bloom_enabled\"")) != nullptr)
            postfx->bloom_enabled = (strstr(p, "true") != nullptr &&
                                    strstr(p, "true") < p + 30);
        if ((p = strstr(buf, "\"bloom_intensity\"")) != nullptr &&
            sscanf(p, "\"bloom_intensity\": %f", &val) == 1)
            postfx->bloom_intensity = val;
        if ((p = strstr(buf, "\"bloom_radius_h\"")) != nullptr &&
            sscanf(p, "\"bloom_radius_h\": %f", &val) == 1)
            postfx->bloom_radius_h = val;
        if ((p = strstr(buf, "\"bloom_radius_v\"")) != nullptr &&
            sscanf(p, "\"bloom_radius_v\": %f", &val) == 1)
            postfx->bloom_radius_v = val;
        if ((p = strstr(buf, "\"firefly_enabled\"")) != nullptr)
            postfx->firefly_enabled = (strstr(p, "true") != nullptr &&
                                      strstr(p, "true") < p + 30);
        if ((p = strstr(buf, "\"firefly_threshold\"")) != nullptr &&
            sscanf(p, "\"firefly_threshold\": %f", &val) == 1)
            postfx->firefly_threshold = val;
    }

    return true;
}

// ── PNG output ──────────────────────────────────────────────────────

bool write_png(const std::string& filename, const FrameBuffer& fb) {
    namespace fs = std::filesystem;
    fs::path p(filename);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    if (fb.srgb.empty()) return false;

    // Stamp watermark on a copy so the display buffer stays clean
    std::vector<uint8_t> stamped = fb.srgb;
    font_overlay::stampWatermarks(stamped,
        static_cast<uint32_t>(fb.width),
        static_cast<uint32_t>(fb.height));

    int ok = stbi_write_png(filename.c_str(),
                            fb.width, fb.height, 4,
                            stamped.data(), fb.width * 4);
    if (ok)
        std::printf("[Output] Wrote %s (%dx%d)\n",
                   filename.c_str(), fb.width, fb.height);
    return ok != 0;
}

// ── EXR (HDR) output ────────────────────────────────────────────────

bool write_exr(const std::string& filename,
               const std::vector<float>& hdr_rgba,
               int width, int height)
{
    namespace fs = std::filesystem;
    fs::path p(filename);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());

    // tinyexr expects separate B, G, R channels
    std::vector<float> r(width * height), g(width * height), b(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int i = (y * width + x) * 4;
            int di = y * width + x;
            r[di] = hdr_rgba[i + 0];
            g[di] = hdr_rgba[i + 1];
            b[di] = hdr_rgba[i + 2];
        }
    }

    const float* channels[] = { b.data(), g.data(), r.data() };

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
        std::fprintf(stderr, "[EXR] Failed to write %s", filename.c_str());
        if (err) { std::fprintf(stderr, ": %s", err); FreeEXRErrorMessage(err); }
        std::fprintf(stderr, "\n");
        return false;
    }
    std::printf("[Output] Wrote %s (%dx%d HDR)\n", filename.c_str(), width, height);
    return true;
}
