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
//   concentration      — f_max / f_total from histogram (C2 metric)
//   caustic_cv         — coefficient of variation of caustic flux (C6)
//   caustic_count      — number of caustic photons in cell (C6)
//   base_guide_frac    — DEFAULT_GUIDE_FRACTION from config
//
// Returns p_guide in [0, 1].
inline HD float compute_guide_fraction(
    int   photon_count,
    float directional_spread,
    float flux_variance,
    float irradiance,
    int   active_bins,
    float concentration,
    float caustic_cv,
    int   caustic_count,
    float base_guide_frac = DEFAULT_GUIDE_FRACTION)
{
    if (photon_count == 0) return 0.f;

    // C1: histogram quality — few bins → trust guide more
    float p = base_guide_frac;
    if (active_bins <= 2 && active_bins > 0)
        p = 0.7f;

    // C2: concentration — high f_max/f_total means a strong directional peak
    // Boost guide fraction for concentrated distributions
    if (concentration > 0.5f)
        p = fmaxf(p, 0.6f + 0.3f * concentration);  // up to 0.9

    // C4: reliability scales with photon count (need ~30 for meaningful histogram)
    constexpr float N_RELIABLE = 30.f;
    p *= fminf(1.f, (float)photon_count / N_RELIABLE);

    // C5: reduce when variance is high (photon map not converged)
    constexpr float CV_MAX = 2.f;
    float cv = flux_variance / fmaxf(irradiance, 1e-6f);
    p *= fmaxf(0.1f, 1.f - cv / CV_MAX);

    // C3: For very diffuse/isotropic cells, scale down the guide fraction
    // rather than killing it entirely.  The mixture PDF already degrades
    // gracefully — pure BSDF sampling still benefits from even a small
    // guide contribution in multi-bounce indirect lighting.
    if (directional_spread > 0.9f)
        p *= 0.2f;  // heavily attenuate but keep non-zero

    // C6: Caustic-edge cells — high caustic CV with enough caustic
    // photons indicates a sharp caustic boundary that needs careful
    // guided sampling to reduce noise.
    constexpr float CAUSTIC_CV_EDGE_THRESH = 0.5f;
    constexpr int   CAUSTIC_EDGE_MIN_COUNT = 10;
    if (caustic_count >= CAUSTIC_EDGE_MIN_COUNT &&
        caustic_cv > CAUSTIC_CV_EDGE_THRESH) {
        // Boost guide toward 0.7..0.8 at caustic edges
        float edge_boost = fminf(1.f, (caustic_cv - CAUSTIC_CV_EDGE_THRESH) * 2.f);
        p = fmaxf(p, 0.7f * edge_boost);
    }

    return p;
}

#ifndef __CUDACC__
// ── CPU-side analysis builder ────────────────────────────────────────
// Populates a flat array of CellAnalysis from existing CellInfoCache.
// Called once per photon map rebuild.
//
// Two passes:
//   Pass 1: Per-cell analysis (density, caustic fraction, active bins,
//           concentration, dominant emitter, guide fraction).
//   Pass 2: 3×3×3 neighbourhood CV (§3.3) — requires cell_pos from
//           CellCacheInfo.  Overwrites flux_cv with neighbourhood metric.
#include <vector>
#include <cmath>

#include "photon/cell_cache.h"
#include "debug/stats_collector.h"

inline std::vector<CellAnalysis> build_cell_analysis(
    const CellInfoCache&  cell_cache,
    float                 cell_area,
    ConclusionCounters*   conclusion_counters = nullptr)
{
    const size_t N = cell_cache.cells.size();
    if (N == 0) return {};

    std::vector<CellAnalysis> result(N);

    // ── Pass 1: Per-cell analysis ───────────────────────────────────
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

        // §3.3 — Single-cell CV (will be overwritten by neighbourhood CV in Pass 2)
        a.flux_cv = (ci.irradiance > 1e-8f)
                  ? sqrtf(ci.flux_variance) / ci.irradiance
                  : 0.f;

        // §3.4 — Caustic fraction
        a.caustic_fraction = (ci.photon_count > 0)
                           ? (float)ci.caustic_count / (float)ci.photon_count
                           : 0.f;

        // §3.1 — Directional analysis: derive active_bins estimate
        // from directional_spread (0=unidirectional, 1=isotropic).
        // With kNN guide replacing HashHistogram, active_bins is estimated
        // as spread * bin_count — high spread means many active bins.
        a.active_bins = (int)(ci.directional_spread * 32.f + 0.5f);
        if (a.active_bins < 1 && ci.photon_count > 0) a.active_bins = 1;
        float concentration = (ci.directional_spread < 0.5f)
                            ? 1.f - ci.directional_spread
                            : 0.f;

        // §3.4 — Dominant emitter (Gap 4 fix)
        a.dominant_emitter = ci.dominant_emitter;

        // §3.6 M1 — Guide fraction (combines all conclusions)
        a.guide_fraction = compute_guide_fraction(
            ci.photon_count,
            ci.directional_spread,
            ci.flux_variance,
            ci.irradiance,
            a.active_bins,
            concentration,
            ci.caustic_cv,
            ci.caustic_count);

        // ── Conclusion counters (gated by ENABLE_STATS) ─────────
        if constexpr (ENABLE_STATS) {
            if (conclusion_counters) {
                conclusion_counters->total_cells++;
                if (a.active_bins <= 2 && a.active_bins > 0)
                    conclusion_counters->c1_histogram++;
                if (concentration > 0.5f)
                    conclusion_counters->c2_concentrated++;
                if (ci.photon_count < 30)
                    conclusion_counters->c4_low_count++;
                float cv = ci.flux_variance / fmaxf(ci.irradiance, 1e-6f);
                if (cv / 2.f > 0.01f)  // any CV attenuation
                    conclusion_counters->c5_high_var++;
                if (ci.directional_spread > 0.9f)
                    conclusion_counters->c3_too_diffuse++;
                if (ci.caustic_count >= 10 && ci.caustic_cv > 0.5f)
                    conclusion_counters->c6_caustic_edge++;
            }
        }
    }

    // ── Pass 2: 3×3×3 neighbourhood CV (§3.3 — Gap 3 fix) ──────────
    // For each occupied cell, enumerate the 27 neighbour cells
    // (using cell_pos stored in CellCacheInfo), compute CV of their
    // irradiance values, and overwrite flux_cv.
    for (size_t i = 0; i < N; ++i) {
        const CellCacheInfo& ci = cell_cache.cells[i];
        if (ci.photon_count == 0) continue;

        double sum = 0.0, sum2 = 0.0;
        int    count = 0;
        int3   cp = ci.cell_pos;

        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int3 nc = make_i3(cp.x + dx, cp.y + dy, cp.z + dz);
            uint32_t nk = CellInfoCache::cell_hash(nc);
            const CellCacheInfo& nci = cell_cache.cells[nk];
            if (nci.photon_count == 0) continue;
            double v = (double)nci.irradiance;
            sum  += v;
            sum2 += v * v;
            count++;
        }

        if (count >= 2) {
            double mean  = sum / count;
            double var   = (sum2 / count) - mean * mean;
            if (var < 0.0) var = 0.0; // numerical guard
            double stddev = sqrt(var);
            result[i].flux_cv = (mean > 1e-12) ? (float)(stddev / mean) : 0.f;
        }
        // If count < 2, keep the single-cell CV from Pass 1
    }

    return result;
}
#endif // __CUDACC__
