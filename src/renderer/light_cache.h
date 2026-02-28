#pragma once
// ─────────────────────────────────────────────────────────────────────
// light_cache.h – Per-cell directional-coverage shadow ray cache (§7.2.2)
// ─────────────────────────────────────────────────────────────────────
// For each spatial cell, we precompute a set of shadow ray targets
// that provide optimal angular coverage of the unit hemisphere as seen
// from the cell centre.  At render time, NEE traces one shadow ray to
// each cached target — no random selection, no importance weighting.
// The number of shadow rays varies per cell (coverage-driven, not
// capped at a fixed K).
//
// Algorithm per cell (CPU preprocessing):
//   1. Compute direction from cell centre to every emitter point.
//   2. Project onto an octahedral unit-hemisphere with B×B bins.
//   3. Greedy set-cover: accept the point that fills the most empty bins.
//      Stop when no more empty bins can be covered.
//   4. Store the variable-length list of accepted point indices.
#include "core/types.h"
#include "core/hash.h"
//
// Memory layout (variable-length):
//   cell_offset[TABLE_SIZE]    — start index into flat point_indices[]
//   cell_count[TABLE_SIZE]     — number of entries for this cell
//   point_indices[total]       — indices into shadow ray targets
//
// The old fixed-stride CellLightEntry is retained for backward compat
// (GPU struct field alignment) but the importance field is set to 1.0.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/config.h"

#ifndef __CUDACC__
#include "photon/photon.h"
#include <vector>
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <cmath>
#include <iostream>
#include <chrono>
#include <numeric>
#ifdef _OPENMP
#  include <omp.h>
#endif
#else
#include <cstdint>
#endif

// ── Configuration constants ─────────────────────────────────────────

// Number of top light sources stored per cell (legacy fixed-stride cap).
// The new coverage algorithm may store fewer or more, but GPU fallback
// code still references this for the old path.
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

// ── Hemisphere coverage grid resolution ─────────────────────────────
// B × B octahedral bins on the upper hemisphere for directional coverage.
// Higher = finer angular resolution, more shadow rays accepted.
//   4 = 16 bins (coarse)  |  8 = 64 bins (recommended)  |  12 = 144 bins
constexpr int HEMI_COVERAGE_BINS = 8;
constexpr int HEMI_TOTAL_BINS = HEMI_COVERAGE_BINS * HEMI_COVERAGE_BINS;

// Maximum shadow ray targets per cell (safety cap).
constexpr int MAX_SHADOW_TARGETS_PER_CELL = 128;

// ── Data structures ─────────────────────────────────────────────────

// A single entry in the per-cell light list.
// Stored on both host and device.
// In the new system, emissive_idx is replaced by point_index (into the
// global EmitterPointSet), but the struct name is kept for compat.
struct CellLightEntry {
    uint16_t emissive_idx;   // local index into emissive_tri_indices[]
    float    importance;     // 1.0 for all (coverage-driven, not flux-weighted)
};

// ── GPU-side shadow ray target (uploaded to device) ─────────────────
// Compact struct for GPU: just position + normal + material data needed
// for shadow ray evaluation.
struct ShadowRayTarget {
    float3   position;       // world-space position on emitter surface
    float3   normal;         // geometric normal of the emitter triangle
    uint16_t emissive_local_idx; // index into emissive_tri_indices[]
    uint32_t global_tri_idx; // global triangle index
};

#ifndef __CUDACC__
// Per-cell directional-coverage light cache.
// Built on CPU from EmitterPointSet, uploaded to GPU for NEE.
struct LightCache {
    // ── Legacy fixed-stride storage (for backward compat) ───────────
    // Still populated so that dev_nee_cached GPU code can work unchanged
    // during transitional period.  Filled from coverage results.
    std::vector<CellLightEntry> entries;          // [TABLE_SIZE * TOP_K]
    std::vector<int>            count;            // [TABLE_SIZE] valid entries per cell
    std::vector<float>          total_importance; // [TABLE_SIZE] sum of importance per cell

    // ── New variable-length coverage storage ────────────────────────
    std::vector<ShadowRayTarget> shadow_targets;  // flattened, all cells
    std::vector<int>             cell_offset;     // [TABLE_SIZE] start into shadow_targets
    std::vector<int>             cell_count;      // [TABLE_SIZE] number of targets per cell

    float cell_size = 0.f;   // spatial cell size (same as hash grid: 2 × gather_radius)

    // ── Spatial hash (same algorithm as HashGrid, different table size) ──
    static HD uint32_t cache_cell_key(int3 cell) {
        return teschner_hash(cell, LIGHT_CACHE_TABLE_SIZE);
    }

    HD int3 cell_coord(float3 pos) const {
        return make_i3(
            (int)floorf(pos.x / cell_size),
            (int)floorf(pos.y / cell_size),
            (int)floorf(pos.z / cell_size)
        );
    }

    // ── Octahedral hemisphere mapping ───────────────────────────────
    // Maps a unit direction (z >= 0) to a 2D bin in [0, B) × [0, B).
    // Uses equal-area octahedral mapping for uniform bin sizes.
    static inline int hemi_bin(float3 dir, int B) {
        // Normalize to upper hemisphere
        float z = fabsf(dir.z);  // mirror to upper hemi if needed
        float ax = fabsf(dir.x);
        float ay = fabsf(dir.y);
        float sum = ax + ay + z;
        if (sum < 1e-8f) return 0;

        // Octahedral projection: (x,y,z) → (u,v) in [-1,1]²
        float u = dir.x / sum;
        float v = dir.y / sum;

        // Map from [-1,1] to [0,1]
        float uf = (u + 1.0f) * 0.5f;
        float vf = (v + 1.0f) * 0.5f;

        int bi = (int)(uf * B);
        int bj = (int)(vf * B);
        bi = (bi < 0) ? 0 : (bi >= B ? B - 1 : bi);
        bj = (bj < 0) ? 0 : (bj >= B ? B - 1 : bj);

        return bj * B + bi;
    }

    // ── Build from photon flux ──────────────────────────────────────
    void build(const PhotonSoA& photons, float grid_cell_size) {
        auto t_start = std::chrono::high_resolution_clock::now();

        cell_size = grid_cell_size;

        entries.resize((size_t)LIGHT_CACHE_TABLE_SIZE * NEE_CELL_TOP_K);
        count.resize(LIGHT_CACHE_TABLE_SIZE, 0);
        total_importance.resize(LIGHT_CACHE_TABLE_SIZE, 0.f);
        for (auto& e : entries) { e.emissive_idx = 0xFFFFu; e.importance = 0.f; }

        if (photons.size() == 0 || photons.source_emissive_idx.empty()) {
            std::cout << "[LightCache] No photons with source info — cache empty\n";
            return;
        }

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

            float flux = 0.f;
            for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
                size_t fi = i * HERO_WAVELENGTHS + h;
                if (fi < photons.flux.size())
                    flux += photons.flux[fi];
            }
            if (flux <= 0.f) continue;

            histograms[key][src] += flux;
        }

        int occupied_cells = 0;

        #pragma omp parallel for schedule(dynamic, 64) reduction(+:occupied_cells)
        for (int k = 0; k < (int)LIGHT_CACHE_TABLE_SIZE; ++k) {
            auto& hist = histograms[k];
            if (hist.empty()) continue;
            ++occupied_cells;

            std::vector<CellLightEntry> sorted;
            sorted.reserve(hist.size());
            for (auto& [idx, imp] : hist)
                sorted.push_back({idx, imp});

            std::sort(sorted.begin(), sorted.end(),
                [](const CellLightEntry& a, const CellLightEntry& b) {
                    return a.importance > b.importance;
                });

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

        std::printf("[LightCache] Legacy flux build: %d occupied cells  "
                    "%zu/%zu photons  top_k=%d  %.1f ms\n",
                    occupied_cells, photons_with_source, n, NEE_CELL_TOP_K, ms);
    }

    // ── Query: get the top-K lights for a world position (legacy) ───
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

    // ── Check if new coverage data is available ─────────────────────
    bool has_coverage_data() const { return !shadow_targets.empty(); }
};
#endif // !__CUDACC__
