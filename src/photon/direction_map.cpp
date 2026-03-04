#include "photon/direction_map.h"
#include <cmath>
#include <cstdio>
#include <algorithm>

// ── stb_image_write (header-only, implementation in viewer.cpp) ─────
// We only need the declarations here.
#ifndef STBI_WRITE_NO_STDIO
extern "C" int stbi_write_png(const char*, int, int, int, const void*, int);
#endif

// ── write_debug_png: dominant direction → RGB ───────────────────────
// Maps abs(dir_x) → R, abs(dir_y) → G, abs(dir_z) → B so every
// guided direction produces a visible colour.  Black = no guidance.
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
                rgba[dst + 0] = 0;
                rgba[dst + 1] = 0;
                rgba[dst + 2] = 0;
                rgba[dst + 3] = 255;
            } else {
                // Signed direction → RGB:  component [-1..+1] maps to [0..255]
                // so opposite directions produce distinct colours.
                rgba[dst + 0] = (uint8_t)std::clamp(int((e.dir_x * 0.5f + 0.5f) * 255.f), 0, 255);
                rgba[dst + 1] = (uint8_t)std::clamp(int((e.dir_y * 0.5f + 0.5f) * 255.f), 0, 255);
                rgba[dst + 2] = (uint8_t)std::clamp(int((e.dir_z * 0.5f + 0.5f) * 255.f), 0, 255);
                rgba[dst + 3] = 255;
            }
        }
    }

    int ok = stbi_write_png(path.c_str(), w, h, 4, rgba.data(), w * 4);
    if (ok) std::printf("[DirMap] Debug PNG: %s  (%d x %d)\n", path.c_str(), w, h);
    return ok != 0;
}

// ── write_strength_png: guidance PDF → normalised grayscale ──────────
// Brightness = pdf / max_pdf (auto-normalised per frame).
// Black = no guidance, white = strongest guidance direction.
bool DirectionMap::write_strength_png(const std::string& path) const {
    if (entries.empty()) return false;

    int w = sub_width();
    int h = sub_height();

    // Find actual max PDF for auto-normalisation
    float max_pdf = 0.f;
    uint16_t max_elig = 0;
    for (auto& e : entries) {
        if (e.pdf > max_pdf) max_pdf = e.pdf;
        if (e.num_eligible > max_elig) max_elig = e.num_eligible;
    }

    float denom = (max_pdf > 1e-10f) ? max_pdf : 1.f;

    std::vector<uint8_t> rgba(w * h * 4);

    for (int sy = 0; sy < h; ++sy) {
        for (int sx = 0; sx < w; ++sx) {
            int src = index(sx, sy);
            int dst = ((h - 1 - sy) * w + sx) * 4;
            const DirMapEntry& e = entries[src];

            float t = e.pdf / denom;  // [0..1]
            uint8_t v = (uint8_t)std::clamp(int(t * 255.f), 0, 255);
            rgba[dst + 0] = v;
            rgba[dst + 1] = v;
            rgba[dst + 2] = v;
            rgba[dst + 3] = 255;
        }
    }

    int ok = stbi_write_png(path.c_str(), w, h, 4, rgba.data(), w * 4);
    if (ok) std::printf("[DirMap] Strength PNG: %s  (%d x %d, max_pdf=%.4f, max_eligible=%d/%d)\n",
                        path.c_str(), w, h, max_pdf, (int)max_elig, (int)MAX_GUIDE_PDF_PHOTONS);
    return ok != 0;
}
