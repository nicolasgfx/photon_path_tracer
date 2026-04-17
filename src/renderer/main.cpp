// ─────────────────────────────────────────────────────────────────────
// main.cpp – Entry point for the photon + path tracing renderer (v5)
//
// v5: RGB pipeline, runtime scene selection, no compile-time scene macros.
// Loads scene from CLI or config, sets up camera, launches viewer.
// ─────────────────────────────────────────────────────────────────────
#include "app/cli_args.h"
#include "app/viewer.h"
#include "app/render_config.h"
#include "app/render_session.h"
#include "core/config.h"
#include "core/types.h"
#include "scene/scene.h"
#include "scene/pbrt/pbrt_loader.h"
#include "scene/obj_loader.h"
#include "analyze/scene_analyzer.h"
#include "analyze/scene_profile_applicator.h"
#include "analyze/analyze_json_output.h"
#include "analyze/prepass_metrics.h"
#include "diagnose/noise_analyzer.h"
#include "core/camera.h"

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <cmath>
#include <iterator>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellscalingapi.h>
#pragma comment(lib, "Shcore.lib")
#endif

// ── File extension helper ───────────────────────────────────────────

static std::string get_extension(const std::string& path) {
    auto dot = path.rfind('.');
    if (dot == std::string::npos) return "";
    std::string ext = path.substr(dot);
    for (auto& c : ext) c = (char)std::tolower((unsigned char)c);
    return ext;
}

// ── Main ────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
#ifdef _WIN32
    {
        typedef BOOL (WINAPI *PFN_SetProcessDpiAwarenessContext)(HANDLE);
        auto fn = (PFN_SetProcessDpiAwarenessContext)GetProcAddress(
            GetModuleHandleW(L"user32.dll"), "SetProcessDpiAwarenessContext");
        if (fn)
            fn((HANDLE)-4);  // DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        else
            SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);
    }
#endif

    setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("== Photon + Path Tracing Renderer v5 (OptiX, RGB) ==\n\n");

    // ── Parse CLI arguments ─────────────────────────────────────────

    Options opt = parse_args(argc, argv);

    if (opt.help) {
        print_usage(argv[0]);
        return 0;
    }

    // ── Load JSON config (before CLI overrides take effect) ─────────
    // Merge order: config.h defaults → JSON file → CLI args
    // parse_args already applied CLI overrides to opt.config, so we
    // load JSON first and then re-apply CLI args on top.
    if (!opt.config_file.empty()) {
        RenderConfig json_cfg;
        if (json_cfg.load_json(opt.config_file)) {
            // Re-parse: start from JSON as base, let CLI override
            RenderConfig cli_cfg = opt.config;
            opt.config = json_cfg;
            // CLI args that were explicitly set override JSON values.
            // Since parse_args writes to opt.config starting from defaults,
            // we detect CLI overrides by comparing against fresh defaults.
            RenderConfig defaults;
            if (cli_cfg.image_width       != defaults.image_width)       opt.config.image_width       = cli_cfg.image_width;
            if (cli_cfg.image_height      != defaults.image_height)      opt.config.image_height      = cli_cfg.image_height;
            if (cli_cfg.samples_per_pixel != defaults.samples_per_pixel) opt.config.samples_per_pixel = cli_cfg.samples_per_pixel;
            if (cli_cfg.max_bounces       != defaults.max_bounces)       opt.config.max_bounces       = cli_cfg.max_bounces;
            if (cli_cfg.exposure          != defaults.exposure)          opt.config.exposure          = cli_cfg.exposure;
            if (cli_cfg.denoiser_enabled  != defaults.denoiser_enabled)  opt.config.denoiser_enabled  = cli_cfg.denoiser_enabled;
            if (cli_cfg.adaptive_sampling != defaults.adaptive_sampling) opt.config.adaptive_sampling = cli_cfg.adaptive_sampling;
            if (cli_cfg.mode              != defaults.mode)              opt.config.mode              = cli_cfg.mode;
            if (cli_cfg.light_scale       != defaults.light_scale)       opt.config.light_scale       = cli_cfg.light_scale;
            if (cli_cfg.rr_threshold      != defaults.rr_threshold)      opt.config.rr_threshold      = cli_cfg.rr_threshold;
            if (cli_cfg.max_sample_luminance != defaults.max_sample_luminance) opt.config.max_sample_luminance = cli_cfg.max_sample_luminance;
            if (cli_cfg.postfx.use_aces   != defaults.postfx.use_aces)   opt.config.postfx.use_aces   = cli_cfg.postfx.use_aces;
            if (cli_cfg.postfx.bloom_enabled != defaults.postfx.bloom_enabled) opt.config.postfx.bloom_enabled = cli_cfg.postfx.bloom_enabled;
            if (cli_cfg.postfx.firefly_enabled != defaults.postfx.firefly_enabled) opt.config.postfx.firefly_enabled = cli_cfg.postfx.firefly_enabled;
        }
    }

    // If no scene file specified, default to Cornell Box
    if (opt.scene_file.empty()) {
        opt.initial_preset_index = 0;
        opt.scene_file = std::string(SCENES_DIR) + "/" + SCENE_PRESETS[0].obj_path;
        std::printf("[Scene] No scene file specified, defaulting to %s\n",
                    opt.scene_file.c_str());
    }

    // ── Load scene ──────────────────────────────────────────────────

    Scene scene;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        bool load_ok = false;

        {
            std::string ext = get_extension(opt.scene_file);
            if (ext == ".pbrt") {
                load_ok = load_pbrt(opt.scene_file, scene);
            } else if (ext == ".obj") {
                load_ok = load_obj(opt.scene_file, scene);
            } else {
                std::fprintf(stderr, "[Error] Unsupported scene format: %s\n",
                             ext.c_str());
                return 1;
            }
        }

        if (!load_ok) {
            std::fprintf(stderr, "[Error] Failed to load scene: %s\n",
                        opt.scene_file.c_str());
            return 1;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        std::printf("[Timing] Scene load: %.1f ms  (%zu tris, %zu mats)\n",
                    std::chrono::duration<double, std::milli>(t1 - t0).count(),
                    scene.triangles.size(), scene.materials.size());

        // Normalize non-reference scenes (rescale to unit cube + transform camera)
        if (opt.initial_preset_index >= 0) {
            const auto& preset = SCENE_PRESETS[opt.initial_preset_index];
            if (!preset.is_reference)
                scene.normalize_to_reference();
            if (preset.rotate_x_180)
                scene.rotate_x_180();
        } else {
            scene.normalize_to_reference();
        }

        // Finalize scene
        scene.compute_bounds();
        scene.build_emissive_distribution();

        std::printf("[Scene] Triangles: %zu  Emissive: %d  Power: %.4f\n",
                    scene.triangles.size(),
                    (int)scene.num_emissive(),
                    scene.total_emissive_power);
        std::printf("[Scene] Bounds: (%.2f,%.2f,%.2f)-(%.2f,%.2f,%.2f)\n",
                    scene.scene_bounds.lo.x, scene.scene_bounds.lo.y,
                    scene.scene_bounds.lo.z,
                    scene.scene_bounds.hi.x, scene.scene_bounds.hi.y,
                    scene.scene_bounds.hi.z);
    }

    // ── Scene analysis → auto-tune config ────────────────────────────
    SceneProfile scene_profile = analyze_scene(scene);
    apply_scene_profile(scene_profile, opt.config);
    opt.scene_profile = scene_profile;
    std::printf("[SceneProfile] lighting=%s  geo=%s  caustics=%s\n",
                to_string(scene_profile.dominant_lighting),
                to_string(scene_profile.geometry_complexity),
                scene_profile.has_caustic_paths ? "yes" : "no");
    std::printf("[SceneProfile] emitter_visibility=%.1f%%  indirect_emitters=%s\n",
                scene_profile.emitter_direct_visibility * 100.f,
                scene_profile.mostly_indirect_emitters ? "yes" : "no");
    std::printf("[SceneProfile] bounces=%d  spp=%d  exposure=%.2f\n",
                opt.config.max_bounces,
                opt.config.samples_per_pixel,
                opt.config.exposure);
    std::printf("[SceneProfile] caustics=%s  delta_tris=%d  delta_area=%.2f%%  budget=%d/frame\n",
                opt.config.caustic_enabled ? "ENABLED" : "disabled",
                scene_profile.num_delta_triangles,
                scene_profile.delta_area_fraction * 100.f,
                opt.config.caustic_photons_per_frame);

    // ── Setup camera ────────────────────────────────────────────────

    Camera camera;
    if (scene.scene_cam_valid) {
        camera.position = scene.scene_cam_position;
        camera.look_at  = scene.scene_cam_look_at;
        camera.up       = scene.scene_cam_up;
        camera.fov_deg  = scene.scene_cam_fov;
    } else if (opt.initial_preset_index >= 0) {
        const auto& preset = SCENE_PRESETS[opt.initial_preset_index];
        camera.position = make_f3(preset.cam_pos[0], preset.cam_pos[1], preset.cam_pos[2]);
        camera.look_at  = make_f3(preset.cam_lookat[0], preset.cam_lookat[1], preset.cam_lookat[2]);
        camera.up       = make_f3(0.f, 1.f, 0.f);
        camera.fov_deg  = preset.cam_fov;
    } else {
        camera.position = make_f3(0.f, 0.f, 1.2f);
        camera.look_at  = make_f3(0.f, 0.f, 0.f);
        camera.up       = make_f3(0.f, 1.f, 0.f);
        camera.fov_deg  = 60.f;
    }
    camera.width  = opt.config.image_width;
    camera.height = opt.config.image_height;
    camera.dof_enabled    = opt.config.dof_enabled;
    camera.dof_focus_dist = opt.config.dof_focus_distance;
    camera.dof_f_number   = opt.config.dof_f_number;
    camera.update();

    // Apply handedness flip if the scene loader flagged it
    if (scene.scene_cam_flip_x) {
        camera.u          = camera.u * -1.0f;
        camera.horizontal = camera.horizontal * -1.0f;
        camera.lower_left = camera.position
                          - camera.horizontal * 0.5f
                          - camera.vertical   * 0.5f
                          - camera.w * ((camera.lens_radius > 0.f)
                                        ? camera.dof_focus_dist : 1.0f);
    }

    std::printf("[Camera] pos=(%.2f,%.2f,%.2f) look=(%.2f,%.2f,%.2f) fov=%.1f\n",
                camera.position.x, camera.position.y, camera.position.z,
                camera.look_at.x, camera.look_at.y, camera.look_at.z,
                camera.fov_deg);

    // ── Load saved camera ───────────────────────────────────────────

    if (!opt.scene_file.empty()) {
        std::string folder = scene_folder_from_path(opt.scene_file);
        float yaw = 0.f, pitch = 0.f, roll = 0.f;
        float light_scale = DEFAULT_LIGHT_SCALE;
        float saved_fov = 0.f;
        PostFxParams postfx;
        if (load_camera_json(folder, camera.position, yaw, pitch, roll,
                              light_scale, &postfx, &saved_fov, &camera)) {
            auto& st = app_state();
            st.yaw   = yaw;
            st.pitch = pitch;
            st.roll  = roll;
            st.light_scale = light_scale;
            st.postfx = postfx;
            if (saved_fov > 0.f) camera.fov_deg = saved_fov;

            // Reconstruct look_at and up from yaw/pitch/roll
            // (viewer does this every frame in process_input)
            float3 forward = make_f3(
                sinf(yaw) * cosf(pitch),
                sinf(pitch),
                -cosf(yaw) * cosf(pitch));
            float3 right_unrolled = normalize(cross(forward, make_f3(0.f, 1.f, 0.f)));
            float3 up_perp = cross(right_unrolled, forward);
            camera.look_at = camera.position + forward;
            camera.up = up_perp * cosf(roll) - right_unrolled * sinf(roll);
            camera.update();

            // Re-apply handedness flip (camera.update() resets u/horizontal)
            if (scene.scene_cam_flip_x) {
                camera.u          = camera.u * -1.0f;
                camera.horizontal = camera.horizontal * -1.0f;
                camera.lower_left = camera.position
                                  - camera.horizontal * 0.5f
                                  - camera.vertical   * 0.5f
                                  - camera.w * ((camera.lens_radius > 0.f)
                                                ? camera.dof_focus_dist : 1.0f);
            }
            std::printf("[Camera] Loaded saved camera from %s/saved_camera.json\n",
                       folder.c_str());
        }
    }

    // ── Print render config ─────────────────────────────────────────

    std::printf("[Config] %dx%d  spp=%d  bounces=%d  denoiser=%s\n",
                opt.config.image_width, opt.config.image_height,
                opt.config.samples_per_pixel, opt.config.max_bounces,
                opt.config.denoiser_enabled ? "ON" : "OFF");

    // ── Launch viewer or batch render ───────────────────────────────

    if (opt.sweep) {
        // ── Convergence sweep mode ──────────────────────────────────
        // Render at multiple SPP checkpoints, saving an image and
        // collecting noise metrics at each level.

        static const int DEFAULT_SWEEP[] = { 16, 64, 256, 512, 1024, 2048 };
        const std::vector<int>& levels = opt.sweep_spp.empty()
            ? std::vector<int>(std::begin(DEFAULT_SWEEP), std::end(DEFAULT_SWEEP))
            : opt.sweep_spp;

        std::printf("[Sweep] Convergence sweep with %d levels:", (int)levels.size());
        for (int l : levels) std::printf(" %d", l);
        std::printf("\n");

        RenderSession gpu;
        if (!gpu.init(scene, opt.config, PTX_FILE_PATH)) {
            std::fprintf(stderr, "[Error] GPU pipeline init failed\n");
            return 1;
        }

        FrameBuffer fb;
        fb.resize(opt.config.image_width, opt.config.image_height);

        // View-dependent pre-pass
        if (opt.config.prepass_spp > 0) {
            PrePassMetrics pm = gpu.run_prepass(camera, opt.config);
            refine_scene_profile(scene_profile, opt.config, pm);
        }

        // Strip extension from output path for checkpoint naming
        std::string base = opt.output_file;
        {
            auto dot = base.rfind('.');
            if (dot != std::string::npos) base = base.substr(0, dot);
        }

        // Sweep report data
        struct SweepPoint { int spp; float noise_level; float mean_lum; };
        std::vector<SweepPoint> sweep_data;

        int rendered_spp = 0;  // cumulative SPP rendered so far
        for (int level : levels) {
            if (level <= rendered_spp) continue;  // skip if already past this level

            // Render frames from current position up to this checkpoint
            for (int spp = rendered_spp; spp < level; ++spp) {
                gpu.render_frame(camera, spp, opt.config);
                if ((spp + 1) % 100 == 0)
                    std::printf("[Sweep] %d / %d SPP\n", spp + 1, level);
            }
            rendered_spp = level;

            // Download raw HDR for noise measurement
            gpu.download_color(fb.rgb);
            int np = fb.num_pixels();

            // Compute luminance + noise estimate (Laplacian variance)
            std::vector<float> lum(np);
            double lum_sum = 0.0;
            for (int i = 0; i < np; ++i) {
                float r = fb.rgb[i * 3 + 0];
                float g = fb.rgb[i * 3 + 1];
                float b = fb.rgb[i * 3 + 2];
                lum[i] = r * 0.2126f + g * 0.7152f + b * 0.0722f;
                lum_sum += lum[i];
            }
            float mean_lum = (float)(lum_sum / np);

            float noise = noise_analyzer::estimate_noise_laplacian(
                lum.data(), fb.width, fb.height);

            sweep_data.push_back({ level, noise, mean_lum });
            std::printf("[Sweep] SPP=%5d  noise=%.6f  mean_lum=%.6f\n",
                        level, noise, mean_lum);

            // Save checkpoint image (apply postfx, then save PNG)
            PostFxParams pfx = opt.config.postfx;
            pfx.exposure = opt.config.exposure * opt.config.light_scale;
            pfx.denoiser_blend = opt.config.denoiser_blend;
            gpu.apply_postfx(pfx);
            gpu.download_srgb(fb.srgb);
            for (int i = 0; i < np; ++i)
                fb.sample_count[i] = (float)level;

            char checkpoint_file[512];
            std::snprintf(checkpoint_file, sizeof(checkpoint_file),
                          "%s_spp%04d.png", base.c_str(), level);
            if (write_png(checkpoint_file, fb))
                std::printf("[Sweep] Saved: %s\n", checkpoint_file);
        }

        // ── Convergence rate (log-log fit) ──────────────────────────
        if (sweep_data.size() >= 2) {
            std::vector<int>   spp_vals;
            std::vector<float> noise_vals;
            for (const auto& p : sweep_data) {
                spp_vals.push_back(p.spp);
                noise_vals.push_back(p.noise_level);
            }

            // Simple log-log fit: noise ∝ SPP^(-α)
            double sx = 0, sy = 0, sxx = 0, sxy = 0;
            int n = 0;
            for (size_t i = 0; i < spp_vals.size(); ++i) {
                if (noise_vals[i] <= 0.f) continue;
                double x = std::log((double)spp_vals[i]);
                double y = std::log((double)noise_vals[i]);
                sx += x; sy += y; sxx += x * x; sxy += x * y;
                ++n;
            }
            if (n >= 2) {
                double denom = n * sxx - sx * sx;
                float alpha = (std::abs(denom) > 1e-30)
                    ? (float)-((n * sxy - sx * sy) / denom)  // negate: rate is positive
                    : 0.f;
                std::printf("[Sweep] Convergence rate α = %.3f", alpha);
                if (alpha > 0.8f && alpha < 1.2f)
                    std::printf(" (normal MC convergence)\n");
                else if (alpha < 0.3f)
                    std::printf(" (STALLED — check guide/photon settings)\n");
                else if (alpha < 0.8f)
                    std::printf(" (slow — noise reduction technique may help)\n");
                else
                    std::printf(" (fast — possible bias, verify energy conservation)\n");
            }
        }

        // ── Save sweep report JSON ──────────────────────────────────
        {
            char report_path[512];
            std::snprintf(report_path, sizeof(report_path),
                          "%s_sweep.json", base.c_str());
            FILE* fp = std::fopen(report_path, "w");
            if (fp) {
                std::fprintf(fp, "{\n  \"sweep_points\": [\n");
                for (size_t i = 0; i < sweep_data.size(); ++i) {
                    std::fprintf(fp, "    { \"spp\": %d, \"noise\": %.8f, \"mean_lum\": %.6f }%s\n",
                                 sweep_data[i].spp, sweep_data[i].noise_level,
                                 sweep_data[i].mean_lum,
                                 (i + 1 < sweep_data.size()) ? "," : "");
                }
                std::fprintf(fp, "  ]\n}\n");
                std::fclose(fp);
                std::printf("[Sweep] Report saved: %s\n", report_path);
            }
        }

    } else if (opt.headless) {
        std::printf("[Batch] Headless mode — rendering %d SPP...\n",
                    opt.config.samples_per_pixel);

        RenderSession gpu;
        if (!gpu.init(scene, opt.config, PTX_FILE_PATH)) {
            std::fprintf(stderr, "[Error] GPU pipeline init failed\n");
            return 1;
        }

        FrameBuffer fb;
        fb.resize(opt.config.image_width, opt.config.image_height);

        // View-dependent pre-pass
        if (opt.config.prepass_spp > 0) {
            PrePassMetrics pm = gpu.run_prepass(camera, opt.config);
            refine_scene_profile(scene_profile, opt.config, pm);
        }

        for (int spp = 0; spp < opt.config.samples_per_pixel; ++spp) {
            gpu.render_frame(camera, spp, opt.config);
            if (opt.config.caustic_enabled)
                gpu.launch_caustics(camera, spp, opt.config);
            if ((spp + 1) % 100 == 0 || spp + 1 == opt.config.samples_per_pixel)
                std::printf("[Batch] %d / %d SPP\n", spp + 1,
                            opt.config.samples_per_pixel);
        }

        // GPU PostFx: firefly filter → bloom → tonemap → sRGB
        PostFxParams pfx = opt.config.postfx;
        pfx.exposure = opt.config.exposure * opt.config.light_scale;
        pfx.denoiser_blend = opt.config.denoiser_blend;
        gpu.apply_postfx(pfx);
        gpu.download_srgb(fb.srgb);

        // Keep raw color for diagnostics
        gpu.download_color(fb.rgb);
        for (int i = 0; i < fb.num_pixels(); ++i)
            fb.sample_count[i] = (float)opt.config.samples_per_pixel;

        // Print HDR stats for diagnosis
        {
            int np = fb.num_pixels();
            double lum_sum = 0.0;
            float lum_max = 0.f;
            int zero_count = 0;
            for (int i = 0; i < np; ++i) {
                float r = fb.rgb[i * 3 + 0];
                float g = fb.rgb[i * 3 + 1];
                float b = fb.rgb[i * 3 + 2];
                float lum = r * 0.2126f + g * 0.7152f + b * 0.0722f;
                lum_sum += lum;
                if (lum > lum_max) lum_max = lum;
                if (lum <= 0.f) ++zero_count;
            }
            std::printf("[HDR Stats] mean_lum=%.6f  max_lum=%.2f  zero=%.1f%%\n",
                        lum_sum / np, lum_max,
                        100.0 * zero_count / np);
        }

        if (write_png(opt.output_file, fb))
            std::printf("[Batch] Saved: %s\n", opt.output_file.c_str());
        else
            std::fprintf(stderr, "[Error] Failed to save: %s\n",
                         opt.output_file.c_str());
    } else {
        Viewer viewer;
        if (!viewer.init(opt.config.image_width, opt.config.image_height,
                         "Photon Path Tracer v5")) {
            std::fprintf(stderr, "[Error] Failed to create viewer window\n");
            return 1;
        }
        if (!viewer.init_gpu(scene, opt.config, PTX_FILE_PATH)) {
            std::fprintf(stderr, "[Warning] GPU init failed — running without GPU\n");
        }
        viewer.run(scene, opt);
        viewer.shutdown();
    }

    return 0;
}
