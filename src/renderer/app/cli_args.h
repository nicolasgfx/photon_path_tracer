#pragma once
// ─────────────────────────────────────────────────────────────────────
// app/cli_args.h – Command-line argument parsing (v5)
//
// Parses common rendering options. Returns an Options struct that
// feeds into viewer initialization. Header-only.
// ─────────────────────────────────────────────────────────────────────
#include "app/render_config.h"
#include "core/scene_profile.h"
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>

// ── Options produced by CLI parsing ─────────────────────────────────

struct Options {
    std::string scene_file;                             // input scene path
    std::string output_file   = "output/render.png";    // output PNG path
    std::string config_file;                            // JSON config path (--config)
    bool        headless      = false;                  // no window (batch mode)
    bool        help          = false;
    int         initial_preset_index = -1;              // scene preset index (-1 = custom)

    // ── Convergence sweep ───────────────────────────────────────────
    bool              sweep          = false;           // --sweep mode
    std::vector<int>  sweep_spp;                        // custom SPP levels (empty = default)

    RenderConfig config;
    SceneProfile scene_profile;              // filled by analyze_scene()
};

// ── Parse argv ──────────────────────────────────────────────────────

inline Options parse_args(int argc, char* argv[]) {
    Options opt;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            opt.help = true;
        }
        else if (arg == "--config" && i + 1 < argc) {
            opt.config_file = argv[++i];
        }
        else if (arg == "--save-config" && i + 1 < argc) {
            // Deferred: save_json called after all args are parsed
            opt.config.save_json(argv[++i]);
        }
        else if (arg == "--width" && i + 1 < argc) {
            opt.config.image_width = std::atoi(argv[++i]);
        }
        else if (arg == "--height" && i + 1 < argc) {
            opt.config.image_height = std::atoi(argv[++i]);
        }
        else if (arg == "--spp" && i + 1 < argc) {
            opt.config.samples_per_pixel = std::atoi(argv[++i]);
        }
        else if (arg == "--bounces" && i + 1 < argc) {
            opt.config.max_bounces = std::atoi(argv[++i]);
        }
        else if (arg == "--output" && i + 1 < argc) {
            opt.output_file = argv[++i];
        }
        else if (arg == "--exposure" && i + 1 < argc) {
            opt.config.exposure = (float)std::atof(argv[++i]);
        }
        else if (arg == "--no-denoiser") {
            opt.config.denoiser_enabled = false;
        }
        else if (arg == "--no-aces") {
            opt.config.postfx.use_aces = false;
        }
        else if (arg == "--headless") {
            opt.headless = true;
        }
        else if (arg == "--mode" && i + 1 < argc) {
            std::string mode = argv[++i];
            if      (mode == "full" || mode == "combined")
                opt.config.mode = RenderMode::Combined;
            else if (mode == "direct")
                opt.config.mode = RenderMode::DirectOnly;
            else if (mode == "indirect")
                opt.config.mode = RenderMode::IndirectOnly;
            else if (mode == "normals")
                opt.config.mode = RenderMode::Normals;
            else if (mode == "material")
                opt.config.mode = RenderMode::MaterialID;
            else if (mode == "depth")
                opt.config.mode = RenderMode::Depth;
            else
                std::fprintf(stderr, "Warning: unknown render mode '%s'\n",
                             mode.c_str());
        }
        else if (arg == "--adaptive") {
            opt.config.adaptive_sampling = true;
        }
        else if (arg == "--light-scale" && i + 1 < argc) {
            opt.config.light_scale = (float)std::atof(argv[++i]);
        }
        else if (arg == "--no-bloom") {
            opt.config.postfx.bloom_enabled = false;
        }
        else if (arg == "--no-firefly") {
            opt.config.postfx.firefly_enabled = false;
        }
        else if (arg == "--no-path-clamp") {
            opt.config.max_path_throughput = 1e30f;
        }
        else if (arg == "--rr-threshold" && i + 1 < argc) {
            opt.config.rr_threshold = (float)std::atof(argv[++i]);
        }
        else if (arg == "--max-sample-luminance" && i + 1 < argc) {
            opt.config.max_sample_luminance = (float)std::atof(argv[++i]);
        }
        else if (arg == "--caustics") {
            opt.config.caustic_enabled = true;
            if (opt.config.caustic_photons_per_frame == 0)
                opt.config.caustic_photons_per_frame = DEFAULT_CAUSTIC_PHOTONS_PER_FRAME;
        }
        else if (arg == "--no-caustics") {
            opt.config.caustic_enabled = false;
            opt.config.caustic_photons_per_frame = 0;
        }
        else if (arg == "--caustic-budget" && i + 1 < argc) {
            opt.config.caustic_photons_per_frame = std::atoi(argv[++i]);
            if (opt.config.caustic_photons_per_frame > 0)
                opt.config.caustic_enabled = true;
        }
        else if (arg == "--no-light-tree") {
            opt.config.light_tree_enabled = false;
        }
        else if (arg == "--light-tree-leaf" && i + 1 < argc) {
            opt.config.light_tree_max_leaf_size = std::atoi(argv[++i]);
        }
        else if (arg == "--sweep") {
            opt.sweep    = true;
            opt.headless = true;  // sweep implies headless
            // Optional: parse comma-separated SPP levels (e.g., --sweep 16,64,256,1024,2048)
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                std::string levels = argv[++i];
                size_t pos = 0;
                while (pos < levels.size()) {
                    size_t comma = levels.find(',', pos);
                    if (comma == std::string::npos) comma = levels.size();
                    int val = std::atoi(levels.substr(pos, comma - pos).c_str());
                    if (val > 0) opt.sweep_spp.push_back(val);
                    pos = comma + 1;
                }
            }
        }
        else if (arg[0] != '-') {
            // Positional argument: scene file
            opt.scene_file = arg;
        }
        else {
            std::fprintf(stderr, "Warning: unknown argument '%s'\n", arg.c_str());
        }
    }

    return opt;
}

// ── Print usage ─────────────────────────────────────────────────────

inline void print_usage(const char* prog) {
    std::printf(
        "Usage: %s [options] [scene_file.pbrt]\n"
        "\n"
        "Configuration:\n"
        "  --config FILE        Load render config from JSON file\n"
        "  --save-config FILE   Save current config to JSON (after applying other args)\n"
        "\n"
        "Rendering options:\n"
        "  --width N            Image width  (default %d)\n"
        "  --height N           Image height (default %d)\n"
        "  --spp N              Samples per pixel (default %d)\n"
        "  --bounces N          Max camera bounces (default %d)\n"
        "  --exposure F         Tone map exposure (default %.1f)\n"
        "  --light-scale F      Light intensity multiplier (default %.1f)\n"
        "  --output FILE        Output PNG path (default output/render.png)\n"
        "\n"
        "Display:\n"
        "  --mode MODE          Render mode: full|direct|indirect|\n"
        "                       normals|material|depth\n"
        "  --no-denoiser        Disable OptiX AI denoiser\n"
        "  --no-aces            Disable ACES tonemapping\n"
        "  --no-bloom           Disable bloom post-FX\n"
        "  --no-firefly         Disable firefly filter\n"
        "  --no-path-clamp      Disable per-path throughput clamping\n"
        "  --headless           No window (batch render + save)\n"
        "  --adaptive           Enable adaptive sampling\n"
        "\n"
        "Quality tuning:\n"
        "  --rr-threshold F     Russian roulette threshold (default %.2f)\n"
        "  --max-sample-luminance F\n"
        "                       Per-sample luminance clamp (default %.0f)\n"
        "\n"
        "Convergence sweep:\n"
        "  --sweep [SPP,SPP,...] Render at multiple SPP levels, save intermediate\n"
        "                       images and convergence report (implies --headless).\n"
        "                       Default levels: 16,64,256,512,1024,2048\n"
        "\n"
        "Caustic light tracing:\n"
        "  --caustics           Enable caustic splatting (default: auto from scene)\n"
        "  --no-caustics        Disable caustic splatting\n"
        "  --caustic-budget N   Photons per frame (default: from scene analysis)\n"
        "\n"
        "Light tree (importance-driven emitter sampling):\n"
        "  --no-light-tree      Disable light tree (use flat power-weighted CDF)\n"
        "  --light-tree-leaf N  Max triangles per leaf node (default: %d)\n"
        "\n"
        "  -h, --help           Show this help\n",
        prog,
        DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, DEFAULT_SPP,
        DEFAULT_MAX_BOUNCES_CAMERA, DEFAULT_EXPOSURE,
        DEFAULT_LIGHT_SCALE,
        DEFAULT_RR_THRESHOLD, DEFAULT_MAX_SAMPLE_LUMINANCE,
        DEFAULT_LIGHT_TREE_MAX_LEAF_SIZE
    );
}
