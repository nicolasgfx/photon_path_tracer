#pragma once
// ─────────────────────────────────────────────────────────────────────
// light_cache.h – Per-cell light importance cache for NEE (§7.2.2)
// ─────────────────────────────────────────────────────────────────────
// During photon map build, each deposited photon records which emissive
// triangle it originated from (source_emissive_idx).  The light cache
// aggregates this information per spatial cell: for each hash-grid cell,
// we store the top-K most important light sources ranked by accumulated
// photon flux deposited in that cell.
//
// At render time, NEE queries the light cache for the shading point's
// cell and samples shadow rays from the cached distribution — reducing
// variance by steering rays toward lights that actually contribute
// indirect illumination at that location.
//
// Memory:  LIGHT_CACHE_TABLE_SIZE × NEE_CELL_TOP_K × sizeof(CellLightEntry)
//          + 2 × LIGHT_CACHE_TABLE_SIZE × sizeof(float/int)
//          ≈ 64K × 16 × 6 + 64K × 8 = 6.5 MB (independent of photon count)
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/config.h"
#include "photon/photon.h"

#include <vector>
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <cmath>
#include <iostream>
#include <chrono>

#ifdef _OPENMP
#  include <omp.h>
#endif

// ── Configuration constants ─────────────────────────────────────────

// Number of top light sources stored per cell
constexpr int NEE_CELL_TOP_K = 16;

// Hash table size for the light cache (independent of photon hash grid)
constexpr uint32_t LIGHT_CACHE_TABLE_SIZE = 65536u;  // 64K cells

// Probability of falling back to the global power CDF instead of using
// the cache.  Ensures unseen lights are not completely starved.
//   0.0 = pure cache  |  0.05 = recommended  |  1.0 = no cache
constexpr float NEE_CACHE_FALLBACK_PROB = 0.05f;

// Default number of NEE shadow rays when using the light cache.
// Can be lower than DEFAULT_NEE_LIGHT_SAMPLES because the cache
// concentrates rays on the most relevant lights.
//   4–8 = preview  |  16 = recommended  |  64 = same as uncached
constexpr int DEFAULT_NEE_CACHED_LIGHT_SAMPLES = 16;

// Master switch for the light importance cache.
//   true = build cache and use dev_nee_cached (default)
//   false = use standard dev_nee_direct everywhere
constexpr bool DEFAULT_USE_LIGHT_CACHE = true;

// ── Data structures ─────────────────────────────────────────────────

// A single entry in the per-cell light importance list.
// Stored on both host and device.
struct CellLightEntry {
    uint16_t emissive_idx;   // local index into emissive_tri_indices[]
    float    importance;     // accumulated scalar flux from this light
};

// Per-cell light importance cache.
// Built on CPU after photon trace, uploaded to GPU for NEE.
struct LightCache {
    // Flat storage: entries[cell_key * NEE_CELL_TOP_K + k]
    // Only the first count[cell_key] entries are valid.
    std::vector<CellLightEntry> entries;          // [TABLE_SIZE * TOP_K]
    std::vector<int>            count;            // [TABLE_SIZE] valid entries per cell
    std::vector<float>          total_importance; // [TABLE_SIZE] sum of importance per cell

    float cell_size = 0.f;   // spatial cell size (same as hash grid: 2 × gather_radius)

    // ── Spatial hash (same algorithm as HashGrid, different table size) ──
    static HD uint32_t cache_cell_key(int3 cell) {
        uint32_t h = (uint32_t)(cell.x * 73856093u)
                   ^ (uint32_t)(cell.y * 19349663u)
                   ^ (uint32_t)(cell.z * 83492791u);
        return h % LIGHT_CACHE_TABLE_SIZE;
    }

    HD int3 cell_coord(float3 pos) const {
        return make_i3(
            (int)floorf(pos.x / cell_size),
            (int)floorf(pos.y / cell_size),
            (int)floorf(pos.z / cell_size)
        );
    }

    // ── Build the light cache from photon deposition statistics ─────
    //
    // For each photon with a valid source_emissive_idx, we accumulate
    // its scalar flux into a per-cell histogram keyed by emissive_idx.
    // After processing all photons, each cell's histogram is sorted by
    // importance and the top-K entries are stored.
    //
    // Complexity: O(N_photons) + O(TABLE_SIZE × K × log K)
    void build(const PhotonSoA& photons, float grid_cell_size) {
        auto t_start = std::chrono::high_resolution_clock::now();

        cell_size = grid_cell_size;

        // Allocate flat storage
        entries.resize((size_t)LIGHT_CACHE_TABLE_SIZE * NEE_CELL_TOP_K);
        count.resize(LIGHT_CACHE_TABLE_SIZE, 0);
        total_importance.resize(LIGHT_CACHE_TABLE_SIZE, 0.f);

        // Zero entries
        for (auto& e : entries) { e.emissive_idx = 0xFFFFu; e.importance = 0.f; }

        if (photons.size() == 0 || photons.source_emissive_idx.empty()) {
            std::cout << "[LightCache] No photons with source info — cache empty\n";
            return;
        }

        // Per-cell histogram: cell_key → {emissive_idx → total_flux}
        // Use a vector of unordered_maps (one per cell bucket)
        std::vector<std::unordered_map<uint16_t, float>> histograms(LIGHT_CACHE_TABLE_SIZE);

        size_t n = photons.size();
        size_t photons_with_source = 0;

        for (size_t i = 0; i < n; ++i) {
            uint16_t src = photons.source_emissive_idx[i];
            if (src == 0xFFFFu) continue;
            ++photons_with_source;

            float3 pos = make_f3(photons.pos_x[i], photons.pos_y[i], photons.pos_z[i]);
            int3 cc = cell_coord(pos);
            uint32_t key = cache_cell_key(cc);

            // Scalar flux from hero wavelengths
            float flux = 0.f;
            for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
                size_t fi = i * HERO_WAVELENGTHS + h;
                if (fi < photons.flux.size())
                    flux += photons.flux[fi];
            }
            if (flux <= 0.f) continue;

            histograms[key][src] += flux;
        }

        // For each cell, sort by importance and keep top-K
        int occupied_cells = 0;

        #pragma omp parallel for schedule(dynamic, 64) reduction(+:occupied_cells)
        for (int k = 0; k < (int)LIGHT_CACHE_TABLE_SIZE; ++k) {
            auto& hist = histograms[k];
            if (hist.empty()) continue;
            ++occupied_cells;

            // Collect entries
            std::vector<CellLightEntry> sorted;
            sorted.reserve(hist.size());
            for (auto& [idx, imp] : hist) {
                sorted.push_back({idx, imp});
            }

            // Sort descending by importance
            std::sort(sorted.begin(), sorted.end(),
                [](const CellLightEntry& a, const CellLightEntry& b) {
                    return a.importance > b.importance;
                });

            // Store top-K
            int nc = (std::min)((int)sorted.size(), NEE_CELL_TOP_K);
            float total = 0.f;
            for (int j = 0; j < nc; ++j) {
                entries[(size_t)k * NEE_CELL_TOP_K + j] = sorted[j];
                total += sorted[j].importance;
            }
            count[k] = nc;
            total_importance[k] = total;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        std::printf("[LightCache] Built: %d occupied cells  "
                    "%zu/%zu photons with source  top_k=%d  %.1f ms\n",
                    occupied_cells, photons_with_source, n, NEE_CELL_TOP_K, ms);
    }

    // ── Query: get the top-K lights for a world position ────────────
    // Returns pointer to the first entry; out_count/out_total are set.
    // Returns nullptr if no cache data for this cell.
    const CellLightEntry* query(float3 pos, int& out_count, float& out_total) const {
        if (entries.empty()) { out_count = 0; out_total = 0.f; return nullptr; }
        int3 cc = cell_coord(pos);
        uint32_t key = cache_cell_key(cc);
        out_count = count[key];
        out_total = total_importance[key];
        if (out_count <= 0) return nullptr;
        return &entries[(size_t)key * NEE_CELL_TOP_K];
    }

    bool valid() const { return !entries.empty() && cell_size > 0.f; }
};
