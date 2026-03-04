#include "photon/direction_map.h"
#include <cmath>
#include <cstdio>
#include <algorithm>

// ── stb_image_write (header-only, implementation in viewer.cpp) ─────
// We only need the declarations here.
#ifndef STBI_WRITE_NO_STDIO
extern "C" int stbi_write_png(const char*, int, int, int, const void*, int);
#endif

// ── write_debug_png: normal-map encoding of sampled directions ──────
bool DirectionMap::write_debug_png(const std::string& path) const {
    if (entries.empty()) return false;

    int w = sub_width();
    int h = sub_height();
    std::vector<uint8_t> rgba(w * h * 4);

    for (int sy = 0; sy < h; ++sy) {
        for (int sx = 0; sx < w; ++sx) {
            int src = index(sx, sy);
            // PNG is top-to-bottom; flip Y
            int dst = ((h - 1 - sy) * w + sx) * 4;
            const DirMapEntry& e = entries[src];
            if (e.num_eligible == 0) {
                // No guidance — black
                rgba[dst + 0] = 0;
                rgba[dst + 1] = 0;
                rgba[dst + 2] = 0;
                rgba[dst + 3] = 255;
            } else {
                // Normal-map encoding: R = 0.5+0.5*x, etc.
                rgba[dst + 0] = (uint8_t)std::clamp(int(128.f + 127.f * e.dir_x), 0, 255);
                rgba[dst + 1] = (uint8_t)std::clamp(int(128.f + 127.f * e.dir_y), 0, 255);
                rgba[dst + 2] = (uint8_t)std::clamp(int(128.f + 127.f * e.dir_z), 0, 255);
                rgba[dst + 3] = 255;
            }
        }
    }

    int ok = stbi_write_png(path.c_str(), w, h, 4, rgba.data(), w * 4);
    if (ok) std::printf("[DirMap] Debug PNG: %s  (%d x %d)\n", path.c_str(), w, h);
    return ok != 0;
}

// ── write_strength_png: num_eligible → viridis-like heatmap ─────────
bool DirectionMap::write_strength_png(const std::string& path) const {
    if (entries.empty()) return false;

    int w = sub_width();
    int h = sub_height();

    // Find max eligible for normalization
    uint16_t max_elig = 1;
    for (auto& e : entries)
        if (e.num_eligible > max_elig) max_elig = e.num_eligible;

    std::vector<uint8_t> rgba(w * h * 4);

    for (int sy = 0; sy < h; ++sy) {
        for (int sx = 0; sx < w; ++sx) {
            int src = index(sx, sy);
            int dst = ((h - 1 - sy) * w + sx) * 4;
            const DirMapEntry& e = entries[src];

            float t = (float)e.num_eligible / (float)max_elig;
            // Simple viridis-like: black → blue → green → yellow
            float r, g, b;
            if (t < 0.33f) {
                float s = t / 0.33f;
                r = 0.f; g = 0.f; b = s;
            } else if (t < 0.66f) {
                float s = (t - 0.33f) / 0.33f;
                r = 0.f; g = s; b = 1.f - s;
            } else {
                float s = (t - 0.66f) / 0.34f;
                r = s; g = 1.f; b = 0.f;
            }

            rgba[dst + 0] = (uint8_t)(r * 255.f);
            rgba[dst + 1] = (uint8_t)(g * 255.f);
            rgba[dst + 2] = (uint8_t)(b * 255.f);
            rgba[dst + 3] = 255;
        }
    }

    int ok = stbi_write_png(path.c_str(), w, h, 4, rgba.data(), w * 4);
    if (ok) std::printf("[DirMap] Strength PNG: %s  (%d x %d)\n", path.c_str(), w, h);
    return ok != 0;
}
