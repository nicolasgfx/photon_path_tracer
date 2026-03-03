#pragma once
// ─────────────────────────────────────────────────────────────────────
// cli_args.h — Command-line argument parsing (extracted from main.cpp)
// ─────────────────────────────────────────────────────────────────────
#include "viewer.h"
#include "core/config.h"
#include "core/types.h"
#include <string>

inline Options parse_args(int argc, char* argv[]) {
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
