#pragma once
// ─────────────────────────────────────────────────────────────────────
// texture.h – CPU-side texture storage and sampling
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include <vector>
#include <string>

struct Texture {
    std::vector<float> data;  // RGBA float
    int width    = 0;
    int height   = 0;
    int channels = 4;
    std::string path;         // source file path (for dedup)

    float3 sample(float2 uv) const {
        if (width == 0 || height == 0) return make_f3(1, 1, 1);
        // Wrap UVs to [0,1)
        float u = uv.x - floorf(uv.x);
        float v = uv.y - floorf(uv.y);
        // Flip V (OBJ convention: V=0 at bottom, image: row 0 at top)
        v = 1.f - v;
        int ix = (int)(u * width)  % width;
        int iy = (int)(v * height) % height;
        if (ix < 0) ix += width;
        if (iy < 0) iy += height;
        int idx = (iy * width + ix) * channels;
        return make_f3(data[idx], data[idx + 1], data[idx + 2]);
    }
};
