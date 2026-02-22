#pragma once
// ─────────────────────────────────────────────────────────────────────
// cell_cache.h – Precomputed per-cell photon statistics (§10c)
// ─────────────────────────────────────────────────────────────────────
// Built once after the photon maps are traced and the hash grids are
// constructed.  Provides O(1) aggregate lookup per cell for:
//   • Adaptive gather radius (density-proportional sizing)
//   • Empty-region early exit
//   • Caustic hotspot detection (CV filter)
//   • Glass-path fraction for dispersion-aware rendering
//
// Uses the same spatial hash as HashGrid / LightCache (Teschner primes).
//
// Memory: CELL_CACHE_TABLE_SIZE × sizeof(CellCacheInfo)
//         = 65 536 × 56 ≈ 3.6 MB (independent of photon count)
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/config.h"
#include "core/spectrum.h"
#include "photon/photon.h"

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <chrono>

#ifdef _OPENMP
#  include <omp.h>
#endif

// ── Per-cell aggregate statistics ───────────────────────────────────

struct CellCacheInfo {
    // Irradiance & density
    float   irradiance      = 0.f;  // Mean scalar irradiance (Σflux / count)
    float   flux_variance   = 0.f;  // Welford running variance of scalar flux
    int     photon_count    = 0;    // Total photons deposited in this cell
    float   density         = 0.f;  // photon_count / cell_volume

    // Directional statistics
    float3  avg_wi          = {0.f, 0.f, 0.f};   // Mean incoming direction
    float   directional_spread = 0.f;             // 1 − |avg_wi| ∈ [0,1]

    // Caustic statistics
    int     caustic_count   = 0;    // Photons with CAUSTIC_GLASS flag
    float   caustic_flux    = 0.f;  // Sum of caustic photon flux
    bool    is_caustic_hotspot = false; // CV > threshold

    // Glass / dispersion path fraction
    float   glass_fraction  = 0.f;  // Fraction with TRAVERSED_GLASS

    // Normal statistics (for surface consistency)
    float3  avg_normal      = {0.f, 0.f, 0.f};   // Mean geometric normal
    float   normal_variance = 0.f;                // 1 − |avg_normal|

    // Adaptive gather radius
    float   adaptive_radius = 0.f;  // Precomputed optimal gather radius

    // Coefficient of variation for caustic flux
    float   caustic_cv      = 0.f;  // stddev / mean of caustic flux
};

// ── CellInfoCache ───────────────────────────────────────────────────

struct CellInfoCache {
    std::vector<CellCacheInfo> cells;
    float   cell_size = 0.f;
    float   base_radius = 0.f; // Reference gather radius used for adaptive sizing

    // ── Spatial hash (same algorithm as HashGrid / LightCache) ──────
    static HD uint32_t cell_hash(int3 cell) {
        uint32_t h = (uint32_t)(cell.x * 73856093u)
                   ^ (uint32_t)(cell.y * 19349663u)
                   ^ (uint32_t)(cell.z * 83492791u);
        return h % CELL_CACHE_TABLE_SIZE;
    }

    HD int3 cell_coord(float3 pos) const {
        return make_i3(
            (int)floorf(pos.x / cell_size),
            (int)floorf(pos.y / cell_size),
            (int)floorf(pos.z / cell_size)
        );
    }

    HD uint32_t cell_key(float3 pos) const {
        return cell_hash(cell_coord(pos));
    }

    // ── Query: look up cell info for a world position ───────────────
    const CellCacheInfo& query(float3 pos) const {
        uint32_t key = cell_key(pos);
        return cells[key];
    }

    // ── Is cell empty? (fast early-exit check) ──────────────────────
    bool is_empty(float3 pos) const {
        return cells[cell_key(pos)].photon_count == 0;
    }

    // ── Adaptive radius for a world position ────────────────────────
    float get_adaptive_radius(float3 pos) const {
        const CellCacheInfo& ci = cells[cell_key(pos)];
        if (ci.adaptive_radius > 0.f) return ci.adaptive_radius;
        return base_radius;
    }

    // ── Build from photon maps ──────────────────────────────────────
    //
    // Accumulates statistics from both global and caustic photon maps
    // using Welford's online algorithm for variance.
    //
    // Complexity: O(N_global + N_caustic) + O(CELL_CACHE_TABLE_SIZE)
    void build(const PhotonSoA& global_photons,
               const PhotonSoA& caustic_photons,
               float grid_cell_size,
               float gather_radius)
    {
        auto t_start = std::chrono::high_resolution_clock::now();

        cell_size   = grid_cell_size;
        base_radius = gather_radius;

        cells.clear();
        cells.resize(CELL_CACHE_TABLE_SIZE);

        // ── Welford accumulators per cell ───────────────────────────
        // For running mean/variance of scalar flux and caustic flux
        struct WelfordState {
            double mean      = 0.0;
            double M2        = 0.0;
            int    n         = 0;
            // Caustic Welford
            double c_mean    = 0.0;
            double c_M2     = 0.0;
            int    c_n       = 0;
            // Direction accumulator
            double wi_x = 0.0, wi_y = 0.0, wi_z = 0.0;
            // Normal accumulator
            double nx = 0.0, ny = 0.0, nz = 0.0;
            // Path flag counts
            int glass_count  = 0;
            int caustic_photon_count = 0;
            float caustic_flux_sum   = 0.f;
        };

        std::vector<WelfordState> accum(CELL_CACHE_TABLE_SIZE);

        // Helper lambda: accumulate one photon map
        auto accumulate_map = [&](const PhotonSoA& photons, bool is_caustic_map) {
            size_t n = photons.size();
            for (size_t i = 0; i < n; ++i) {
                float3 pos = make_f3(photons.pos_x[i],
                                     photons.pos_y[i],
                                     photons.pos_z[i]);
                uint32_t key = cell_key(pos);
                WelfordState& ws = accum[key];

                // Scalar flux (sum over spectral bins)
                float scalar_flux = 0.f;
                for (int b = 0; b < NUM_LAMBDA; ++b) {
                    size_t fi = i * NUM_LAMBDA + b;
                    if (fi < photons.spectral_flux.size())
                        scalar_flux += photons.spectral_flux[fi];
                }

                // Welford online variance update
                ws.n++;
                double delta = (double)scalar_flux - ws.mean;
                ws.mean += delta / (double)ws.n;
                double delta2 = (double)scalar_flux - ws.mean;
                ws.M2 += delta * delta2;

                // Direction accumulation
                if (i < photons.wi_x.size()) {
                    ws.wi_x += photons.wi_x[i];
                    ws.wi_y += photons.wi_y[i];
                    ws.wi_z += photons.wi_z[i];
                }

                // Normal accumulation
                if (i < photons.norm_x.size()) {
                    ws.nx += photons.norm_x[i];
                    ws.ny += photons.norm_y[i];
                    ws.nz += photons.norm_z[i];
                }

                // Path flag analysis
                if (i < photons.path_flags.size()) {
                    uint8_t flags = photons.path_flags[i];
                    if (flags & PHOTON_FLAG_TRAVERSED_GLASS)
                        ws.glass_count++;
                    if (flags & PHOTON_FLAG_CAUSTIC_GLASS) {
                        ws.caustic_photon_count++;
                        ws.caustic_flux_sum += scalar_flux;

                        // Welford for caustic flux
                        ws.c_n++;
                        double cd = (double)scalar_flux - ws.c_mean;
                        ws.c_mean += cd / (double)ws.c_n;
                        double cd2 = (double)scalar_flux - ws.c_mean;
                        ws.c_M2 += cd * cd2;
                    }
                }

                // Also count caustic map photons as caustic
                if (is_caustic_map && (photons.path_flags.empty() ||
                    i >= photons.path_flags.size())) {
                    ws.caustic_photon_count++;
                    ws.caustic_flux_sum += scalar_flux;
                    ws.c_n++;
                    double cd = (double)scalar_flux - ws.c_mean;
                    ws.c_mean += cd / (double)ws.c_n;
                    double cd2 = (double)scalar_flux - ws.c_mean;
                    ws.c_M2 += cd * cd2;
                }
            }
        };

        accumulate_map(global_photons,  false);
        accumulate_map(caustic_photons, true);

        // ── Finalize per-cell statistics ────────────────────────────
        float cell_vol = cell_size * cell_size * cell_size;
        int cells_with_photons = 0;
        int caustic_hotspots = 0;

        for (uint32_t k = 0; k < CELL_CACHE_TABLE_SIZE; ++k) {
            CellCacheInfo& ci  = cells[k];
            const WelfordState& ws = accum[k];

            ci.photon_count = ws.n;
            if (ws.n == 0) continue;
            cells_with_photons++;

            // Irradiance and variance
            ci.irradiance    = (float)ws.mean;
            ci.flux_variance = (ws.n >= 2)
                ? (float)(ws.M2 / (double)(ws.n - 1))
                : 0.f;
            ci.density = (float)ws.n / fmaxf(cell_vol, 1e-20f);

            // Directional spread: 1 − |mean_direction|
            float3 sum_wi = make_f3((float)ws.wi_x,
                                    (float)ws.wi_y,
                                    (float)ws.wi_z);
            float inv_n = 1.f / (float)ws.n;
            ci.avg_wi = sum_wi * inv_n;
            float len_wi = sqrtf(dot(ci.avg_wi, ci.avg_wi));
            ci.directional_spread = 1.f - fminf(len_wi, 1.f);

            // Normal statistics
            float3 sum_n = make_f3((float)ws.nx, (float)ws.ny, (float)ws.nz);
            ci.avg_normal = sum_n * inv_n;
            float len_n = sqrtf(dot(ci.avg_normal, ci.avg_normal));
            ci.normal_variance = 1.f - fminf(len_n, 1.f);

            // Glass fraction
            ci.glass_fraction = (float)ws.glass_count * inv_n;

            // Caustic statistics
            ci.caustic_count = ws.caustic_photon_count;
            ci.caustic_flux  = ws.caustic_flux_sum;

            // Caustic CV (coefficient of variation)
            if (ws.c_n >= CAUSTIC_MIN_FOR_ANALYSIS && ws.c_mean > 1e-12) {
                double c_var = (ws.c_n >= 2)
                    ? ws.c_M2 / (double)(ws.c_n - 1)
                    : 0.0;
                ci.caustic_cv = (float)(sqrt(c_var) / ws.c_mean);
                ci.is_caustic_hotspot = (ci.caustic_cv > CAUSTIC_CV_THRESHOLD);
                if (ci.is_caustic_hotspot) caustic_hotspots++;
            }

            // ── Adaptive gather radius ──────────────────────────────
            // Scale radius so that the expected number of photons in
            // the gather disk ≈ ADAPTIVE_RADIUS_TARGET_K.
            //
            // Expected photons in disk = density × π × r²
            // → r_opt = sqrt(K / (π × density))
            //
            // Clamped to [MIN_FACTOR × base, MAX_FACTOR × base].
            if (ci.density > 0.f) {
                float r_opt = sqrtf(ADAPTIVE_RADIUS_TARGET_K /
                                    (PI * ci.density));
                float r_min = base_radius * ADAPTIVE_RADIUS_MIN_FACTOR;
                float r_max = base_radius * ADAPTIVE_RADIUS_MAX_FACTOR;
                ci.adaptive_radius = fminf(fmaxf(r_opt, r_min), r_max);
            } else {
                ci.adaptive_radius = base_radius;
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        std::cout << "[CellCache] Built in " << ms << " ms  ("
                  << cells_with_photons << " occupied cells, "
                  << caustic_hotspots << " caustic hotspots)\n";
    }

    // ── Collect caustic hotspot cell keys ────────────────────────────
    // Returns cell indices where is_caustic_hotspot == true.
    // Used by adaptive caustic shooting to target emission.
    std::vector<uint32_t> get_caustic_hotspot_keys() const {
        std::vector<uint32_t> keys;
        for (uint32_t k = 0; k < CELL_CACHE_TABLE_SIZE; ++k) {
            if (cells[k].is_caustic_hotspot)
                keys.push_back(k);
        }
        return keys;
    }

    // ── World-space center of a cell key ────────────────────────────
    // NOTE: Because the hash is many-to-one, this uses the cell
    //       coordinate from the first photon that hashed to this key.
    //       For adaptive shooting, the actual photon positions within
    //       the cell are more useful — this is a rough centroid.
    // Use the irradiance-weighted centroid from the photon map instead.
};
