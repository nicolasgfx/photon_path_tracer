#pragma once
// ─────────────────────────────────────────────────────────────────────
// direction_map.h — Directional SPP framebuffer ("direction map")
// ─────────────────────────────────────────────────────────────────────
// Per-pixel precomputed guidance for first-hit direction sampling.
//
// Layout:
//   - Grid: W × H (1:1 with framebuffer, DIR_MAP_SUBPIXEL_FACTOR = 1)
//   - Each pixel stores a compact guidance record computed from
//     nearby photon flux via a Fibonacci sphere histogram, filtered
//     by shadow rays to the kNN photon candidates.
//   - The direction map is built ONCE after photon tracing, then
//     consumed per-SPP in the render kernel (re-sampled with varying
//     RNG seed to stochastically vary the guided direction).
//
// GPU device structure:
//   DirMapEntry is uploaded as a flat device buffer and read by the
//   __raygen__render kernel at bounce 0.
// ─────────────────────────────────────────────────────────────────────

#include "core/config.h"
#include <cstdint>

// Per-pixel compact guidance record (GPU-resident)
// Each entry stores a single sampled direction + PDF.
// Shadow rays from hitpoint to kNN photons determine acceptance.
//
// Memory: ~48 bytes per pixel.  At 1:1 with framebuffer,
// 1024×768 → 786,432 pixels → ~36 MB.

struct DirMapEntry {
    // Sampled guided direction (world space, normalised)
    float dir_x, dir_y, dir_z;

    // Marginal guide PDF at this direction (for one-sample MIS)
    float pdf;

    // First-hit position (world space) — for debug visualisation
    float hit_x, hit_y, hit_z;

    // First-hit shading normal (world space) — for debug visualisation
    float norm_x, norm_y, norm_z;

    // Number of eligible photons found (0 = no guidance available)
    uint16_t num_eligible;

    // Material type at hit (for delta-boost debug overlay)
    uint8_t  mat_type;

    // Padding to 48 bytes for GPU alignment
    uint8_t  _pad[5];
};

static_assert(sizeof(DirMapEntry) == 48, "DirMapEntry must be 48 bytes for GPU alignment");

// ── Host-side direction map container ───────────────────────────────
// Only available in host code (not compiled by nvcc for device PTX).
#ifndef __CUDACC__

#include <vector>
#include <string>

struct DirectionMap {
    int base_width  = 0;    // framebuffer width
    int base_height = 0;    // framebuffer height
    int factor      = DIR_MAP_SUBPIXEL_FACTOR;  // subpixel multiplier

    int sub_width()  const { return base_width  * factor; }
    int sub_height() const { return base_height * factor; }
    int total_subpixels() const { return sub_width() * sub_height(); }

    // Host-side copy (downloaded from GPU for debug PNG output)
    std::vector<DirMapEntry> entries;

    void resize(int w, int h) {
        base_width  = w;
        base_height = h;
        entries.resize(total_subpixels());
    }

    void clear() {
        for (auto& e : entries) {
            e = DirMapEntry{};
        }
    }

    /// Map subpixel (sx, sy) to flat index
    int index(int sx, int sy) const {
        return sy * sub_width() + sx;
    }

    /// Map pixel (px, py) + subpixel offset (dx, dy) to flat index
    /// dx, dy in [0, factor)
    int index(int px, int py, int dx, int dy) const {
        int sx = px * factor + dx;
        int sy = py * factor + dy;
        return index(sx, sy);
    }

    /// Write direction map as debug PNG (dominant direction → RGB)
    /// R = |dir_x|, G = |dir_y|, B = |dir_z|
    /// Subpixels with no guidance (num_eligible == 0) are rendered as black.
    bool write_debug_png(const std::string& path) const;

    /// Write direction map strength PNG (grayscale)
    /// Brightness = num_eligible / max_eligible.  Black = no guidance.
    bool write_strength_png(const std::string& path) const;

    // Convenience accessors
    int width()  const { return sub_width(); }
    int height() const { return sub_height(); }
};

#endif // __CUDACC__
