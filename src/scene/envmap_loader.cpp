// ─────────────────────────────────────────────────────────────────────
// envmap_loader.cpp – Load .exr environment maps via tinyexr
// ─────────────────────────────────────────────────────────────────────
#include "scene/envmap.h"
#include "tinyexr.h"

#include <cstdio>
#include <cstring>

bool load_environment_map(const std::string& exr_path,
                          float scale_factor,
                          float3 rotation_deg,
                          EnvironmentMap& envmap) {
    float* rgba = nullptr;
    int w = 0, h = 0;
    const char* err = nullptr;

    int ret = LoadEXR(&rgba, &w, &h, exr_path.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        std::fprintf(stderr, "[EnvMap] Failed to load '%s': %s\n",
                     exr_path.c_str(), err ? err : "unknown error");
        if (err) FreeEXRErrorMessage(err);
        return false;
    }

    envmap.width  = w;
    envmap.height = h;
    envmap.scale  = scale_factor;

    // Convert RGBA → RGB (drop alpha)
    envmap.pixels.resize(w * h * 3);
    for (int i = 0; i < w * h; ++i) {
        envmap.pixels[i * 3 + 0] = fmaxf(rgba[i * 4 + 0], 0.f);
        envmap.pixels[i * 3 + 1] = fmaxf(rgba[i * 4 + 1], 0.f);
        envmap.pixels[i * 3 + 2] = fmaxf(rgba[i * 4 + 2], 0.f);
    }
    free(rgba);

    // Build rotation matrices
    envmap.rotation     = rotation_from_euler_deg(rotation_deg.x,
                                                   rotation_deg.y,
                                                   rotation_deg.z);
    envmap.inv_rotation = rotation_transpose(envmap.rotation);

    // Build importance-sampling distribution
    envmap.build_distribution();

    std::printf("[EnvMap] Loaded '%s': %dx%d  scale=%.2f  total_power=%.4f\n",
                exr_path.c_str(), w, h, scale_factor, envmap.total_power);

    return true;
}
