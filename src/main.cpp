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

#include "app/viewer.h"
#include "core/config.h"
#include "scene/obj_loader.h"
#include "scene/scene.h"
#include "scene/scene_builder.h"
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

// -- Parse command line -----------------------------------------------

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
        } else if (arg == "--adaptive-radius") {
            opt.config.use_knn_adaptive = true;
        } else if (arg[0] != '-') {
            opt.scene_file = arg;
        }
    }

    return opt;
}

// -- Main -------------------------------------------------------------

int main(int argc, char* argv[]) {
#ifdef _WIN32
    // Declare DPI awareness BEFORE any other init (CUDA, GLFW, etc.)
    {
        typedef BOOL (WINAPI *PFN_SetProcessDpiAwarenessContext)(HANDLE);
        auto fn = (PFN_SetProcessDpiAwarenessContext)GetProcAddress(
            GetModuleHandleW(L"user32.dll"), "SetProcessDpiAwarenessContext");
        if (fn) {
            fn((HANDLE)-4);  // DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        } else {
            SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);
        }
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
    SceneLightMode initial_light_mode = SceneLightMode::FromMTL;
    #if defined(SCENE_CORNELL_BOX)
        initial_light_mode = SCENE_PROFILES[0].light_mode;
    #elif defined(SCENE_CORNELL_SPHERE)
        initial_light_mode = SCENE_PROFILES[1].light_mode;
    #elif defined(SCENE_CORNELL_MIRROR)
        initial_light_mode = SCENE_PROFILES[2].light_mode;
    #elif defined(SCENE_CORNELL_WATER)
        initial_light_mode = SCENE_PROFILES[3].light_mode;
    #elif defined(SCENE_LIVING_ROOM)
        initial_light_mode = SCENE_PROFILES[4].light_mode;
    #elif defined(SCENE_CONFERENCE)
        initial_light_mode = SCENE_PROFILES[5].light_mode;
    #elif defined(SCENE_SALLE_DE_BAIN)
        initial_light_mode = SCENE_PROFILES[6].light_mode;
    #elif defined(SCENE_MORI_KNOB)
        initial_light_mode = SCENE_PROFILES[7].light_mode;
    #endif
    add_scene_lights(scene, initial_light_mode);

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
    {
        AppState& app = app_state();
        std::string folder = scene_folder_from_profile(SCENE_OBJ_PATH);
        float yaw_init = 0.f, pitch_init = 0.f;
        float light_init = DEFAULT_LIGHT_SCALE;
        if (load_camera_from_file(camera, yaw_init, pitch_init, light_init, folder)) {
            app.yaw         = yaw_init;
            app.pitch       = pitch_init;
            app.light_scale = light_init;
            app.light_scale_changed = true;
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

            auto t2 = std::chrono::high_resolution_clock::now();
            optix_renderer.build_accel(scene);
            auto t3 = std::chrono::high_resolution_clock::now();
            std::printf("[Timing] build_accel:       %8.1f ms\n",
                std::chrono::duration<double, std::milli>(t3 - t2).count());

            auto t4 = std::chrono::high_resolution_clock::now();
            optix_renderer.upload_scene_data(scene);
            auto t5 = std::chrono::high_resolution_clock::now();
            std::printf("[Timing] upload_scene_data: %8.1f ms\n",
                std::chrono::duration<double, std::milli>(t5 - t4).count());

            auto t6 = std::chrono::high_resolution_clock::now();
            optix_renderer.upload_emitter_data(scene);
            auto t7 = std::chrono::high_resolution_clock::now();
            std::printf("[Timing] upload_emitter:    %8.1f ms\n",
                std::chrono::duration<double, std::milli>(t7 - t6).count());
        }
        std::cout << "\n";

        // -- GPU photon trace ------------------------------------------
        std::cout << "-- GPU Photon Trace --\n";
        {
            auto tp0 = std::chrono::high_resolution_clock::now();
            optix_renderer.trace_photons(scene, opt.config);
            auto tp1 = std::chrono::high_resolution_clock::now();
            std::printf("[Photon] Initial trace completed in %.1f ms\n\n",
                std::chrono::duration<double, std::milli>(tp1 - tp0).count());
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

            PhotonSoA empty_caustic;
            save_test_data(opt.save_test_data_file,
                           optix_renderer.photons(), empty_caustic, hdr);
        }

        // -- Determine initial scene index ----------------------------
        {
            AppState& app = app_state();
            #if defined(SCENE_CORNELL_BOX)
                app.active_scene_index = 0;
            #elif defined(SCENE_CONFERENCE)
                app.active_scene_index = 1;
            #elif defined(SCENE_LIVING_ROOM)
                app.active_scene_index = 2;
            #elif defined(SCENE_SIBENIK)
                app.active_scene_index = 5;
            #endif
            app.active_cam_speed = SCENE_CAM_SPEED;
        }

        // -- Interactive debug window ---------------------------------
        run_interactive(optix_renderer, camera, opt, scene);

    } catch (const std::exception& e) {
        std::cerr << "[Fatal] " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n== Done ==\n";
    return 0;
}
