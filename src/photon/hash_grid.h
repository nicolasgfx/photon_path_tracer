#pragma once
// ─────────────────────────────────────────────────────────────────────
// hash_grid.h – Spatial hash grid for photon lookup
// ─────────────────────────────────────────────────────────────────────
// Implements Section 5 of the specification:
//   - Compute spatial hash keys
//   - Sort photons by key
//   - Build cell start/end arrays
//   - 3×3×3 neighbor query
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "photon/photon.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <functional>

struct HashGrid {
    float    cell_size;
    uint32_t table_size;   // Number of hash buckets

    std::vector<uint32_t> sorted_indices;   // Photon indices sorted by cell key
    std::vector<uint32_t> cell_start;       // First photon in each cell
    std::vector<uint32_t> cell_end;         // One past last photon in each cell

    // ── Spatial hash function ───────────────────────────────────────
    static HD uint32_t hash_cell(int3 cell, uint32_t table_size) {
        // Large primes for spatial hashing (Teschner et al.)
        uint32_t h = (uint32_t)(cell.x * 73856093u)
                   ^ (uint32_t)(cell.y * 19349663u)
                   ^ (uint32_t)(cell.z * 83492791u);
        return h % table_size;
    }

    HD int3 cell_coord(float3 pos) const {
        return make_i3(
            (int)floorf(pos.x / cell_size),
            (int)floorf(pos.y / cell_size),
            (int)floorf(pos.z / cell_size)
        );
    }

    HD uint32_t cell_key(float3 pos) const {
        return hash_cell(cell_coord(pos), table_size);
    }

    // ── Build grid from photon positions ────────────────────────────
    void build(const PhotonSoA& photons, float radius) {
        size_t n = photons.size();
        if (n == 0) return;

        cell_size  = radius * 2.0f;  // Cell size = photon gather diameter
        table_size = (uint32_t)std::max(n, (size_t)1024);

        // 1. Compute keys for each photon
        std::vector<uint32_t> keys(n);
        for (size_t i = 0; i < n; ++i) {
            float3 pos = make_f3(photons.pos_x[i], photons.pos_y[i], photons.pos_z[i]);
            keys[i] = cell_key(pos);
        }

        // 2. Create index array and sort by key
        sorted_indices.resize(n);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0u);
        std::sort(sorted_indices.begin(), sorted_indices.end(),
                  [&keys](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });

        // 3. Build cell_start and cell_end arrays
        cell_start.assign(table_size, 0xFFFFFFFFu); // sentinel: no photons
        cell_end.assign(table_size, 0);

        // Reorder keys by sorted order
        std::vector<uint32_t> sorted_keys(n);
        for (size_t i = 0; i < n; ++i) {
            sorted_keys[i] = keys[sorted_indices[i]];
        }

        for (size_t i = 0; i < n; ++i) {
            uint32_t k = sorted_keys[i];
            if (i == 0 || sorted_keys[i] != sorted_keys[i-1]) {
                cell_start[k] = (uint32_t)i;
            }
            cell_end[k] = (uint32_t)(i + 1);
        }
    }

    // ── Query: find all photon indices within radius of a point ─────
    // Scans the 3×3×3 neighboring cells and filters by distance.
    // Calls `callback(photon_index, distance_squared)` for each neighbor.
    template<typename Callback>
    void query(float3 pos, float radius, const PhotonSoA& photons,
               Callback callback) const {
        if (sorted_indices.empty()) return;

        int3 center_cell = cell_coord(pos);
        float r2 = radius * radius;

        // Track visited bucket keys to avoid duplicate processing
        // (different cell coordinates can hash to the same bucket)
        uint32_t visited_keys[27];
        int num_visited = 0;

        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int3 nc = make_i3(center_cell.x + dx,
                                       center_cell.y + dy,
                                       center_cell.z + dz);
                    uint32_t key = hash_cell(nc, table_size);

                    // Skip if we already visited this bucket
                    bool already_visited = false;
                    for (int v = 0; v < num_visited; ++v) {
                        if (visited_keys[v] == key) {
                            already_visited = true;
                            break;
                        }
                    }
                    if (already_visited) continue;
                    visited_keys[num_visited++] = key;

                    if (cell_start[key] == 0xFFFFFFFFu) continue;

                    for (uint32_t i = cell_start[key]; i < cell_end[key]; ++i) {
                        uint32_t idx = sorted_indices[i];
                        float3 p = make_f3(photons.pos_x[idx],
                                            photons.pos_y[idx],
                                            photons.pos_z[idx]);
                        float3 diff = pos - p;
                        float dist2 = dot(diff, diff);
                        if (dist2 <= r2) {
                            callback(idx, dist2);
                        }
                    }
                }
            }
        }
    }

    // ── Debug: count photons in a cell ──────────────────────────────
    uint32_t count_in_cell(int3 cell) const {
        uint32_t key = hash_cell(cell, table_size);
        if (cell_start[key] == 0xFFFFFFFFu) return 0;
        return cell_end[key] - cell_start[key];
    }
};
