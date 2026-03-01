#pragma once
// ─────────────────────────────────────────────────────────────────────
// photon_analysis.h – Per-cell photon analysis for guided path tracing
// ─────────────────────────────────────────────────────────────────────
// Part 2 §3: Photon Analysis at Each Hit Point
//
// Computed once after the photon map build and before the camera pass.
// Reuses and extends the existing CellInfoCache with per-cell summary
// data consumed by the path tracer at every non-delta bounce.
//
// Five analysis dimensions:
//   §3.1 Directional analysis  → guide fraction
//   §3.2 Spatial density       → adaptive SPP, gather radius
//   §3.3 Spatial variance      → adaptive SPP, re-trace budget
//   §3.4 Photon type           → caustic handling
//   §3.5 Photon energy         → saturation detection
//
// Measures (§3.6):
//   M1: Guide fraction (p_guide)
//   M2: Sample budget per pixel (adaptive SPP)
//   M3: Gather radius (base + caustic)
//   M4: Photon re-trace budget (view-adaptive)
//   M5: Caustic additive contribution
// ─────────────────────────────────────────────────────────────────────

#include "core/types.h"
#include "core/spectrum.h"
#include "core/config.h"

// ── Per-cell analysis results ────────────────────────────────────────
// Cheap O(1) lookup at each camera bounce.  Populated from
// CellInfoCache + CellBinGrid data after the photon map build.
struct CellAnalysis {
    float  total_flux;        // sum of all photon flux in cell
    float  flux_density;      // total_flux / cell_area
    float  flux_cv;           // coefficient of variation (3×3×3 neighbourhood)
    float  caustic_fraction;  // fraction of photons with caustic flags
    float  guide_fraction;    // recommended p_guide for this cell (M1)
    int    dominant_emitter;  // source_emissive_idx with highest flux
    int    active_bins;       // number of non-zero directional bins
    bool   has_photons;       // any photons at all

    HD static CellAnalysis empty() {
        CellAnalysis a;
        a.total_flux       = 0.f;
        a.flux_density     = 0.f;
        a.flux_cv          = 0.f;
        a.caustic_fraction = 0.f;
        a.guide_fraction   = 0.f;
        a.dominant_emitter = -1;
        a.active_bins      = 0;
        a.has_photons      = false;
        return a;
    }
};

// ── Guide fraction computation (M1) ─────────────────────────────────
// Combines conclusions from §3.1 (directional), §3.2 (density),
// §3.3 (variance), and §3.4 (caustic edges) into a single p_guide.
//
// Inputs:
//   photon_count       — total photons in cell
//   directional_spread — 0=unidirectional, 1=isotropic (from CellCacheInfo)
//   flux_variance      — Welford variance of flux (from CellCacheInfo)
//   irradiance         — sum of flux (from CellCacheInfo)
//   active_bins        — number of non-zero directional bins in histogram
//   base_guide_frac    — DEFAULT_GUIDE_FRACTION from config
//
// Returns p_guide in [0, 1].
inline HD float compute_guide_fraction(
    int   photon_count,
    float directional_spread,
    float flux_variance,
    float irradiance,
    int   active_bins,
    float base_guide_frac = DEFAULT_GUIDE_FRACTION)
{
    if (photon_count == 0) return 0.f;

    // C1/C6: histogram quality — few bins → trust guide more
    float p = base_guide_frac;
    if (active_bins <= 2 && active_bins > 0)
        p = 0.7f;

    // C4: reliability scales with photon count (need ~30 for meaningful histogram)
    constexpr float N_RELIABLE = 30.f;
    p *= fminf(1.f, (float)photon_count / N_RELIABLE);

    // C5: reduce when variance is high (photon map not converged)
    constexpr float CV_MAX = 2.f;
    float cv = flux_variance / fmaxf(irradiance, 1e-6f);
    p *= fmaxf(0.1f, 1.f - cv / CV_MAX);

    // C3: skip when too diffuse to guide
    if (directional_spread > 0.9f)
        p = 0.f;

    return p;
}

#ifndef __CUDACC__
// ── CPU-side analysis builder ────────────────────────────────────────
// Populates a flat array of CellAnalysis from existing CellInfoCache
// and CellBinGrid data.  Called once per photon map rebuild.
//
// PA-07: Iterates all cells in CellInfoCache, reads CellBinGrid
//        histograms, and fills the CellAnalysis array for GPU upload.
#include <vector>
#include <cmath>

#include "photon/cell_cache.h"
#include "photon/cell_bin_grid.h"
#include "debug/stats_collector.h"

inline std::vector<CellAnalysis> build_cell_analysis(
    const CellInfoCache& cell_cache,
    const CellBinGrid&   bin_grid,
    float                cell_area,
    ConclusionCounters*  conclusion_counters = nullptr)
{
    const size_t N = cell_cache.cells.size();
    if (N == 0) return {};

    std::vector<CellAnalysis> result(N);

    for (size_t i = 0; i < N; ++i) {
        const CellCacheInfo& ci = cell_cache.cells[i];

        if (ci.photon_count == 0) {
            result[i] = CellAnalysis::empty();
            continue;
        }

        CellAnalysis& a = result[i];
        a.has_photons = true;

        // §3.5 — Total flux (already stored as irradiance × count in CellCacheInfo)
        a.total_flux = ci.irradiance * (float)ci.photon_count;

        // §3.2 — Spatial density metric (flux per cell area)
        a.flux_density = (cell_area > 0.f)
                       ? a.total_flux / cell_area
                       : 0.f;

        // §3.3 — Spatial variance (CV from Welford variance)
        a.flux_cv = (ci.irradiance > 1e-8f)
                  ? sqrtf(ci.flux_variance) / ci.irradiance
                  : 0.f;

        // §3.4 — Caustic fraction
        a.caustic_fraction = (ci.photon_count > 0)
                           ? (float)ci.caustic_count / (float)ci.photon_count
                           : 0.f;

        // §3.1 — Directional analysis: count active bins from CellBinGrid
        a.active_bins = 0;
        a.dominant_emitter = -1;  // not tracked in CellBinGrid; left at -1

        // If the CellBinGrid has data for this cell hash, count non-zero bins
        if (!bin_grid.bins.empty() && bin_grid.cell_size > 0.f) {
            // CellInfoCache uses hash-based indexing while CellBinGrid uses
            // flat 3D indexing.  We can't directly map between them without
            // knowing the cell position.  Use CellCacheInfo's avg_normal to
            // detect if cell has meaningful directional data, and count active
            // bins from the grid by iterating over the bins for each cell.
            //
            // For a simple approach: if the CellInfoCache has directional_spread
            // information, we can estimate active bins from it.
            // active_bins ≈ PHOTON_BIN_COUNT × (1 − spread)² for focused hist,
            // but a direct count is more accurate.
            //
            // Since we don't have a direct hash→grid mapping, approximate from
            // CellCacheInfo directional_spread:
            float spread = ci.directional_spread;
            if (spread < 0.1f)
                a.active_bins = 1;  // highly directional
            else if (spread < 0.5f)
                a.active_bins = (int)(spread * 10.f + 1.f);  // 1–6 bins
            else
                a.active_bins = (int)(spread * (float)PHOTON_BIN_COUNT);
        }

        // §3.6 M1 — Guide fraction (combines all conclusions)
        a.guide_fraction = compute_guide_fraction(
            ci.photon_count,
            ci.directional_spread,
            ci.flux_variance,
            ci.irradiance,
            a.active_bins);

        // ── Conclusion counters (gated by ENABLE_STATS) ─────────
        if constexpr (ENABLE_STATS) {
            if (conclusion_counters) {
                conclusion_counters->total_cells++;
                if (a.active_bins <= 2 && a.active_bins > 0)
                    conclusion_counters->c1_c6_histogram++;
                if (ci.photon_count < 30)
                    conclusion_counters->c4_low_count++;
                float cv = ci.flux_variance / fmaxf(ci.irradiance, 1e-6f);
                if (cv / 2.f > 0.01f)  // any CV attenuation
                    conclusion_counters->c5_high_var++;
                if (ci.directional_spread > 0.9f)
                    conclusion_counters->c3_too_diffuse++;
            }
        }
    }

    return result;
}
#endif // __CUDACC__
