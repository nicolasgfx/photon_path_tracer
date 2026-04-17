// ─────────────────────────────────────────────────────────────────────
// analyze_main.cpp – ppt_analyze entry point
//
// Usage: ppt_analyze <scene.pbrt|scene.obj>
//
// Loads a scene, runs analysis, prints SceneProfile as JSON to stdout.
// Exit code 0 on success, 1 on error.  No GPU required.
// ─────────────────────────────────────────────────────────────────────
#include "analyze/scene_analyzer.h"
#include "analyze/analyze_json_output.h"
#include "scene/scene.h"
#include "scene/pbrt/pbrt_loader.h"
#include "scene/obj_loader.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <chrono>

static std::string get_extension(const std::string& path) {
    auto dot = path.rfind('.');
    if (dot == std::string::npos) return "";
    std::string ext = path.substr(dot);
    for (auto& c : ext) c = (char)std::tolower((unsigned char)c);
    return ext;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: ppt_analyze <scene.pbrt|scene.obj>\n");
        return 1;
    }

    std::string scene_file = argv[1];

    // ── Load scene ──────────────────────────────────────────────────
    Scene scene;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        bool load_ok = false;
        std::string ext = get_extension(scene_file);

        if (ext == ".pbrt") {
            load_ok = load_pbrt(scene_file, scene);
        } else if (ext == ".obj") {
            load_ok = load_obj(scene_file, scene);
        } else {
            std::fprintf(stderr, "[Error] Unsupported format: %s\n", ext.c_str());
            return 1;
        }

        if (!load_ok) {
            std::fprintf(stderr, "[Error] Failed to load: %s\n", scene_file.c_str());
            return 1;
        }

        scene.compute_bounds();
        scene.build_emissive_distribution();

        auto t1 = std::chrono::high_resolution_clock::now();
        std::fprintf(stderr, "[ppt_analyze] Loaded %s (%.1f ms, %zu tris)\n",
                     scene_file.c_str(),
                     std::chrono::duration<double, std::milli>(t1 - t0).count(),
                     scene.triangles.size());
    }

    // ── Analyze ─────────────────────────────────────────────────────
    SceneProfile sp = analyze_scene(scene);

    // ── Output ──────────────────────────────────────────────────────
    print_scene_profile_json(sp);
    return 0;
}
