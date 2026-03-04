// ---------------------------------------------------------------------
// main.cpp -- Entry point for the spectral photon + path tracer
// ---------------------------------------------------------------------

#include "app/viewer.h"
#include "app/cli_args.h"
#include "core/config.h"
#include "scene/obj_loader.h"
#include "scene/scene.h"
#include "scene/scene_builder.h"
#include "scene/envmap.h"
#include "renderer/camera.h"
#include "optix/optix_renderer.h"
#include "tests/test_data_io.h"

#include <iostream>
#include <string>
#include <chrono>
#include <cstdio>

#ifdef _WIN32
#include <windows.h>
#include <shellscalingapi.h>
#pragma comment(lib, "Shcore.lib")
#endif

// -- Compile-time scene → profile index mapping -----------------------

static constexpr int scene_profile_index() {
    #if defined(SCENE_CORNELL_BOX)
        return 0;
    #elif defined(SCENE_FIREPLACE_ROOM)
        return 1;
    #elif defined(SCENE_STAIRCASE)
        return 2;
    #elif defined(SCENE_STAIRCASE_2)
        return 3;
    #elif defined(SCENE_BATHROOM)
        return 4;
    #elif defined(SCENE_LIVING_ROOM_2)
        return 5;
    #elif defined(SCENE_BEDROOM)
        return 6;
    #elif defined(SCENE_VILLA)
        return 7;
    #elif defined(SCENE_WATERCOLOR)
        return 8;
    #elif defined(SCENE_ZERO_DAY)
        return 9;
    #elif defined(SCENE_KROKEN)
        return 10;
    #else
        return -1;
    #endif
}

// -- Main -------------------------------------------------------------

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
    std::cout << "== Spectral Photon + Path Tracing Renderer (OptiX) ==\n";
    std::cout << "   Scene: " << SCENE_DISPLAY_NAME << "\n\n";

    Options opt = parse_args(argc, argv);

    // -- Load scene ---------------------------------------------------
    Scene scene;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        if (!load_obj(opt.scene_file, scene)) {
            std::cerr << "[Error] Failed to load scene: " << opt.scene_file << "\n";
            return 1;
        }
        if (!SCENE_IS_REFERENCE)
            scene.normalize_to_reference();
        {
            constexpr int pidx = scene_profile_index();
            if (pidx >= 0 && SCENE_PROFILES[pidx].rotate_x_180)
                scene.rotate_x_180();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        std::printf("[Timing] OBJ load:          %8.1f ms  (%zu tris, %zu mats, %zu textures)\n",
                    std::chrono::duration<double, std::milli>(t1 - t0).count(),
                    scene.triangles.size(), scene.materials.size(), scene.textures.size());

        scene.build_bvh();
        auto t2 = std::chrono::high_resolution_clock::now();
        std::printf("[Timing] BVH build:         %8.1f ms  (%zu nodes)\n",
                    std::chrono::duration<double, std::milli>(t2 - t1).count(),
                    scene.bvh_nodes.size());

        scene.build_emissive_distribution();
        std::printf("[Scene]  Emissive tris: %d   total power = %.4f\n",
                    (int)scene.num_emissive(), scene.total_emissive_power);
        std::printf("[Scene]  Render config: %dx%d  spp=%d  photons=%d  radius=%.5f  bounces=%d\n",
                    opt.config.image_width, opt.config.image_height,
                    opt.config.samples_per_pixel, opt.config.num_photons,
                    opt.config.gather_radius, opt.config.max_bounces);
    }

    // -- Add light source if none exists ------------------------------
    {
        constexpr int idx = scene_profile_index();
        SceneLightMode lm = (idx >= 0) ? SCENE_PROFILES[idx].light_mode
                                        : SceneLightMode::FromMTL;
        add_scene_lights(scene, lm);
    }

    // -- Setup camera -------------------------------------------------
    Camera camera;
    camera.position = make_f3(SCENE_CAM_POS[0], SCENE_CAM_POS[1], SCENE_CAM_POS[2]);
    camera.look_at  = make_f3(SCENE_CAM_LOOKAT[0], SCENE_CAM_LOOKAT[1], SCENE_CAM_LOOKAT[2]);
    camera.up       = make_f3(0.0f, 1.0f, 0.0f);
    camera.fov_deg  = SCENE_CAM_FOV;
    camera.width    = opt.config.image_width;
    camera.height   = opt.config.image_height;
    camera.dof_enabled    = DEFAULT_DOF_ENABLED;
    camera.dof_focus_dist = DEFAULT_DOF_FOCUS_DISTANCE;
    camera.dof_f_number   = DEFAULT_DOF_F_NUMBER;
    camera.sensor_height  = DEFAULT_DOF_SENSOR_HEIGHT;
    camera.update();

    // -- Load saved camera position (if any) --------------------------
    std::string envmap_path;
    float3 envmap_rotation = make_f3(0, 0, 0);
    float envmap_scale_val = 1.0f;
    {
        AppState& app = app_state();
        std::string folder = scene_folder_from_profile(SCENE_OBJ_PATH);
        float yaw_init = 0.f, pitch_init = 0.f, roll_init = 0.f, light_init = DEFAULT_LIGHT_SCALE;
        PostFxParams loaded_postfx;
        if (load_camera_from_file(camera, yaw_init, pitch_init, roll_init, light_init, folder,
                                  &envmap_path, &envmap_rotation, &envmap_scale_val, &loaded_postfx)) {
            app.yaw   = yaw_init;   app.pitch = pitch_init;  app.roll = roll_init;
            app.light_scale = light_init;  app.light_scale_changed = true;
            app.postfx = loaded_postfx;
        }
    }

    // -- Load environment map (if specified in camera.json) -----------
    if (!envmap_path.empty()) {
        auto t_env0 = std::chrono::high_resolution_clock::now();
        scene.envmap = std::make_shared<EnvironmentMap>();
        if (load_environment_map(envmap_path, envmap_scale_val,
                                 envmap_rotation, *scene.envmap)) {
            // Set bounding sphere from scene geometry
            scene.envmap->scene_center = scene.scene_bounding_center();
            scene.envmap->scene_radius = scene.scene_bounding_radius() * 1.1f;

            // Compute selection probability for one-sample MIS
            scene.compute_envmap_selection_prob();

            auto t_env1 = std::chrono::high_resolution_clock::now();
            std::printf("[Timing] EnvMap load:       %8.1f ms  (%dx%d, sel_p=%.3f)\n",
                        std::chrono::duration<double, std::milli>(t_env1 - t_env0).count(),
                        scene.envmap->width, scene.envmap->height,
                        scene.envmap_selection_prob);
        } else {
            std::printf("[Warning] Failed to load envmap: %s\n", envmap_path.c_str());
            scene.envmap.reset();
        }
    }

    // -- OptiX pipeline -----------------------------------------------
    OptixRenderer optix_renderer;
    try {
        std::cout << "-- OptiX Initialization --\n";
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            optix_renderer.init();
            auto t1 = std::chrono::high_resolution_clock::now();
            std::printf("[Timing] OptiX init:        %8.1f ms\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count());
            optix_renderer.build_accel(scene);
            auto t2 = std::chrono::high_resolution_clock::now();
            std::printf("[Timing] build_accel:       %8.1f ms\n",
                std::chrono::duration<double, std::milli>(t2 - t1).count());
            optix_renderer.upload_scene_data(scene);
            auto t3 = std::chrono::high_resolution_clock::now();
            std::printf("[Timing] upload_scene_data: %8.1f ms\n",
                std::chrono::duration<double, std::milli>(t3 - t2).count());
            optix_renderer.upload_emitter_data(scene);
            auto t4 = std::chrono::high_resolution_clock::now();
            std::printf("[Timing] upload_emitter:    %8.1f ms\n",
                std::chrono::duration<double, std::milli>(t4 - t3).count());
            optix_renderer.upload_envmap_data(scene);
            auto t5 = std::chrono::high_resolution_clock::now();
            std::printf("[Timing] upload_envmap:     %8.1f ms\n",
                std::chrono::duration<double, std::milli>(t5 - t4).count());
        }
        std::cout << "\n";

        // -- GPU photon trace -----------------------------------------
        auto tp0 = std::chrono::high_resolution_clock::now();
        optix_renderer.trace_photons(scene, opt.config);
        std::printf("[Photon] Initial trace completed in %.1f ms\n\n",
            std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - tp0).count());

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
            PhotonSoA empty_caustic;
            save_test_data(opt.save_test_data_file,
                           optix_renderer.photons(), empty_caustic, hdr);
        }

        // -- Determine initial scene index ----------------------------
        AppState& app2 = app_state();
        app2.active_scene_index = scene_profile_index();
        app2.active_cam_speed = SCENE_CAM_SPEED;

        // Sync loaded postfx params to renderer
        // Set adaptive bloom range from scene emissive triangles
        scene.compute_emissive_radiance_range(
            app2.postfx.bloom_scene_min_Le,
            app2.postfx.bloom_scene_max_Le);
        optix_renderer.set_postfx_params(app2.postfx);

        // -- Interactive debug window ---------------------------------
        run_interactive(optix_renderer, camera, opt, scene);

    } catch (const std::exception& e) {
        std::cerr << "[Fatal] " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n== Done ==\n";
    return 0;
}
