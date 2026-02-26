#pragma once
// ─────────────────────────────────────────────────────────────────────
// hash_grid.h – Spatial hash grid for photon lookup (§6.2)
// ─────────────────────────────────────────────────────────────────────
// GPU-primary spatial index.  O(1) build via atomic counting.
//
// v2.1 updates:
//   - query()           : 3D Euclidean (legacy, build validation)
//   - query_tangential(): tangential distance + plane filter (§6.3)
//   - knn_shell_expansion(): GPU-friendly k-NN via expanding grid
//     shell layers (§6.5) — no recursion, no priority queues.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "photon/photon.h"
#include "photon/surface_filter.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <functional>
#include <execution>   // std::execution::par_unseq (MSVC PPL, C++17)
#ifdef _OPENMP
#  include <omp.h>
#endif

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

        // 1. Compute keys for each photon  (OMP parallel: embarrassingly independent)
        std::vector<uint32_t> keys(n);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)n; ++i) {
            float3 pos = make_f3(photons.pos_x[i], photons.pos_y[i], photons.pos_z[i]);
            keys[i] = cell_key(pos);
        }

        // 2. Create index array and sort by key.
        // std::execution::par_unseq uses the MSVC PPL thread pool — typically
        // 2-4x faster than std::sort for N > 500k.
        sorted_indices.resize(n);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0u);
        std::sort(std::execution::par_unseq,
                  sorted_indices.begin(), sorted_indices.end(),
                  [&keys](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });

        // 3. Build cell_start and cell_end arrays
        // Init is trivially parallel; the scan over sorted keys is sequential
        // (each element writes to cell_start[k] which can alias neighbors).
        cell_start.resize(table_size);
        cell_end.resize(table_size);
        #pragma omp parallel for schedule(static)
        for (int ci = 0; ci < (int)table_size; ++ci) {
            cell_start[ci] = 0xFFFFFFFFu;
            cell_end[ci]   = 0;
        }

        // Reorder keys by sorted order
        std::vector<uint32_t> sorted_keys(n);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)n; ++i) {
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
    // 3D Euclidean — legacy, used by build validation / tests.
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

    // ── Tangential range query (§6.3 — v2.1 mandatory) ──────────────
    // Uses tangential disk distance on the tangent plane of `normal`.
    // Scans 3×3×3 neighbors just like query(), but uses tangential
    // distance + plane distance filter instead of 3D Euclidean.
    // Calls callback(photon_index, tangential_distance_squared).
    template<typename Callback>
    void query_tangential(float3 pos, float3 normal, float radius,
                          float tau, const PhotonSoA& photons,
                          Callback callback) const
    {
        if (sorted_indices.empty()) return;

        int3 center_cell = cell_coord(pos);
        float r2 = radius * radius;

        uint32_t visited_keys[27];
        int num_visited = 0;

        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int3 nc = make_i3(center_cell.x + dx,
                                       center_cell.y + dy,
                                       center_cell.z + dz);
                    uint32_t key = hash_cell(nc, table_size);

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
                        float3 ppos = make_f3(photons.pos_x[idx],
                                              photons.pos_y[idx],
                                              photons.pos_z[idx]);
                        TangentialResult tr = compute_tangential(
                            pos, normal, ppos);
                        if (fabsf(tr.d_plane) > tau) continue;
                        if (tr.d_tan2 <= r2) {
                            callback(idx, tr.d_tan2);
                        }
                    }
                }
            }
        }
    }

    // ── Shell expansion k-NN (§6.5 — GPU-friendly k-NN) ────────────
    // Finds the k nearest photons using tangential distance, expanding
    // the grid search radius in concentric shells (layers 0, 1, 2, ...)
    // until k photons are found and any potential closer ones are ruled
    // out.
    //
    // No recursion, no priority queues — purely iterative, GPU-mappable.
    //
    // Algorithm:
    //   layer 0: center cell only (1 cell)
    //   layer 1: 3×3×3 = 27 cells
    //   layer 2: 5×5×5 = 125 cells
    //   ...
    //   Each layer only processes the NEW cells (shell), not interior.
    //   Stops when: heap.size >= k AND
    //               max_candidate_dist < (layer * cell_size)^2
    //     (no closer photons can exist in the next shell)
    //
    // out_indices: k nearest photon indices (sorted farthest-first)
    // out_max_dist2: tangential distance² to the k-th neighbor
    //
    // max_layers: safety limit (default 4 = 9×9×9 = 729 cells)
    void knn_shell_expansion(
        float3 pos, float3 normal, int k, float tau,
        const PhotonSoA& photons,
        std::vector<uint32_t>& out_indices,
        float& out_max_dist2,
        int max_layers = 4) const
    {
        out_indices.clear();
        out_max_dist2 = std::numeric_limits<float>::max();

        if (sorted_indices.empty() || k <= 0) return;

        int3 center_cell = cell_coord(pos);

        // Max-heap via std::vector + push_heap/pop_heap
        // (avoids #include <queue> which conflicts with CUDA headers)
        using HeapEntry = std::pair<float, uint32_t>;
        std::vector<HeapEntry> heap;
        heap.reserve(k + 1);

        // Track visited hash keys across all layers
        std::vector<uint32_t> all_visited;
        all_visited.reserve(128);

        for (int layer = 0; layer <= max_layers; ++layer) {
            int extent = layer;  // layer 0 = 0 offset, layer 1 = ±1, etc.

            // Scan the shell: all cells at max-distance = extent
            // (cells with ANY coordinate at ±extent that haven't been visited)
            for (int dz = -extent; dz <= extent; ++dz) {
                for (int dy = -extent; dy <= extent; ++dy) {
                    for (int dx = -extent; dx <= extent; ++dx) {
                        // Only process cells on the outer shell
                        // (skip interior cells from previous layers)
                        if (layer > 0) {
                            int max_abs = std::max({std::abs(dx),
                                                     std::abs(dy),
                                                     std::abs(dz)});
                            if (max_abs < extent) continue;
                        }

                        int3 nc = make_i3(center_cell.x + dx,
                                           center_cell.y + dy,
                                           center_cell.z + dz);
                        uint32_t key = hash_cell(nc, table_size);

                        // Skip already-visited buckets
                        bool visited = false;
                        for (size_t v = 0; v < all_visited.size(); ++v) {
                            if (all_visited[v] == key) {
                                visited = true;
                                break;
                            }
                        }
                        if (visited) continue;
                        all_visited.push_back(key);

                        if (cell_start[key] == 0xFFFFFFFFu) continue;

                        for (uint32_t i = cell_start[key];
                             i < cell_end[key]; ++i) {
                            uint32_t idx = sorted_indices[i];
                            float3 ppos = make_f3(photons.pos_x[idx],
                                                  photons.pos_y[idx],
                                                  photons.pos_z[idx]);
                            TangentialResult tr = compute_tangential(
                                pos, normal, ppos);
                            if (fabsf(tr.d_plane) > tau) continue;

                            float dist2 = tr.d_tan2;
                            if ((int)heap.size() < k) {
                                heap.push_back({dist2, idx});
                                std::push_heap(heap.begin(), heap.end());
                            } else if (dist2 < heap.front().first) {
                                std::pop_heap(heap.begin(), heap.end());
                                heap.back() = {dist2, idx};
                                std::push_heap(heap.begin(), heap.end());
                            }
                        }
                    }
                }
            }

            // Check termination: if we have k photons and the farthest
            // is closer than the minimum possible tangential distance
            // from any point outside the currently-expanded box, we're done.
            //
            // Correct bound: compute the minimum 3D distance from the query
            // to the boundary of the (2*extent+1)^3 box of cells, then
            // subtract tau^2 (since plane distance <= tau for accepted
            // photons => d_tan^2 >= d_3D^2 - tau^2).
            if ((int)heap.size() >= k) {
                float box_min_x = (float)(center_cell.x - extent) * cell_size;
                float box_min_y = (float)(center_cell.y - extent) * cell_size;
                float box_min_z = (float)(center_cell.z - extent) * cell_size;
                float box_max_x = (float)(center_cell.x + extent + 1) * cell_size;
                float box_max_y = (float)(center_cell.y + extent + 1) * cell_size;
                float box_max_z = (float)(center_cell.z + extent + 1) * cell_size;

                float dx = fminf(pos.x - box_min_x, box_max_x - pos.x);
                float dy = fminf(pos.y - box_min_y, box_max_y - pos.y);
                float dz = fminf(pos.z - box_min_z, box_max_z - pos.z);
                float min_to_boundary = fminf(dx, fminf(dy, dz));
                float min_tan_dist2 = fmaxf(
                    min_to_boundary * min_to_boundary - tau * tau, 0.0f);

                if (heap.front().first < min_tan_dist2) {
                    break;  // No closer photons outside this box
                }
            }
        }

        // Extract results (sort by distance ascending)
        std::sort_heap(heap.begin(), heap.end());
        out_max_dist2 = heap.empty() ? 0.0f : heap.back().first;
        out_indices.resize(heap.size());
        for (size_t i = 0; i < heap.size(); ++i) {
            out_indices[i] = heap[i].second;
        }
    }

    // ── Debug: count photons in a cell ──────────────────────────────
    uint32_t count_in_cell(int3 cell) const {
        uint32_t key = hash_cell(cell, table_size);
        if (cell_start[key] == 0xFFFFFFFFu) return 0;
        return cell_end[key] - cell_start[key];
    }
};

// ─────────────────────────────────────────────────────────────────────
// GPU-side hash grid construction (hash_grid.cu)
// ─────────────────────────────────────────────────────────────────────
// Builds the spatial hash grid entirely on the GPU using CUB radix sort.
// Call once with d_temp_storage==nullptr to query temp_storage_bytes,
// then allocate and call again.

void gpu_build_hash_grid(
    const float* d_pos_x, const float* d_pos_y, const float* d_pos_z,
    int n, float cell_size, uint32_t table_size,
    uint32_t* d_keys_in, uint32_t* d_keys_out,
    uint32_t* d_indices_in, uint32_t* d_sorted_indices,
    uint32_t* d_cell_start, uint32_t* d_cell_end,
    void* d_temp_storage, size_t& temp_storage_bytes);

// Optional: reorder photon SoA into sorted order for CPU download
void gpu_scatter_photon_soa(
    const float* d_src_pos_x, const float* d_src_pos_y, const float* d_src_pos_z,
    const float* d_src_wi_x,  const float* d_src_wi_y,  const float* d_src_wi_z,
    const float* d_src_norm_x, const float* d_src_norm_y, const float* d_src_norm_z,
    const uint16_t* d_src_lambda, const float* d_src_flux,
    const uint8_t* d_src_num_hero,
    const uint32_t* d_sorted_indices,
    float* d_dst_pos_x, float* d_dst_pos_y, float* d_dst_pos_z,
    float* d_dst_wi_x,  float* d_dst_wi_y,  float* d_dst_wi_z,
    float* d_dst_norm_x, float* d_dst_norm_y, float* d_dst_norm_z,
    uint16_t* d_dst_lambda, float* d_dst_flux,
    uint8_t* d_dst_num_hero,
    int n);

// Build 3-valued caustic pass tags on GPU
void gpu_build_caustic_tags(
    const uint8_t* d_is_caustic, uint8_t* d_tags_out,
    int global_count, int total_count);

// Tonemap post-process kernel (Optimization #5)
void launch_tonemap_kernel(
    const float* d_spectrum_buffer,
    const float* d_sample_counts,
    uint8_t* d_srgb_buffer,
    int width, int height,
    float exposure);
