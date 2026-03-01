#pragma once
// ─────────────────────────────────────────────────────────────────────
// adaptive_sampling.h  –  Host-callable CUDA helpers for per-pixel
//                         noise estimation and active-mask update.
//
// Usage (per render pass after a 1-spp launch):
//
//   AdaptiveParams ap;
//   ap.sample_counts = d_sample_counts_ptr;
//   ap.lum_sum       = d_lum_sum_ptr;
//   ap.lum_sum2      = d_lum_sum2_ptr;
//   ap.active_mask   = d_active_mask_ptr;
//   ap.width         = W;
//   ap.height        = H;
//   ap.min_spp       = adaptive_min_spp;
//   ap.max_spp       = effective_max_spp;
//   ap.threshold     = adaptive_threshold;
//   ap.radius        = adaptive_radius;
//
//   int active = adaptive_update_mask(ap);  // returns # active pixels
// ─────────────────────────────────────────────────────────────────────

#pragma once
#include <cstdint>

struct AdaptiveParams {
    const float*   sample_counts; ///< [W*H] read-only
    const float*   lum_sum;       ///< [W*H] read-only Σ Y
    const float*   lum_sum2;      ///< [W*H] read-only Σ Y²
    uint8_t*       active_mask;   ///< [W*H] write output 0/1
    const uint16_t* pixel_max_spp;///< [W*H] per-pixel budget or nullptr (AS-02)
    int            width;
    int            height;
    int            min_spp;       ///< pixels sampled < min_spp are always active
    int            max_spp;       ///< pixels sampled >= max_spp are never active
    float          threshold;     ///< relative-noise threshold (e.g. 0.02)
    int            radius;        ///< neighbourhood half-width (e.g. 1 → 3×3)
};

/// Compute the active mask from luminance moments and return the count
/// of pixels still needing more samples.  Launch threads = W*H.
int adaptive_update_mask(const AdaptiveParams& p);

// ── AS-02: Per-pixel cost map (§7) ──────────────────────────────────

struct CostMapParams {
    const float*   lum_sum;             ///< [W*H] pilot Σ Y
    const float*   lum_sum2;            ///< [W*H] pilot Σ Y²
    const float*   sample_counts;       ///< [W*H] samples so far
    const float*   cell_guide_fraction; ///< [CELL_CACHE_TABLE_SIZE] or nullptr
    const float*   cell_caustic_fraction; ///< [CELL_CACHE_TABLE_SIZE] or nullptr
    const float*   cell_flux_density;   ///< [CELL_CACHE_TABLE_SIZE] or nullptr
    const float*   spectrum_buffer;     ///< [W*H*NUM_LAMBDA]
    int            width;
    int            height;
    int            base_spp;            ///< target SPP (uniform default)
    int            min_spp_clamp;       ///< per-pixel min budget
    int            max_spp_clamp;       ///< per-pixel max budget
    uint16_t*      pixel_max_spp;       ///< [W*H] output per-pixel budget
};

/// Compute per-pixel SPP budgets from pilot-pass variance + photon analysis.
void compute_pixel_cost_map(const CostMapParams& p);
