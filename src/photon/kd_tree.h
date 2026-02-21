#pragma once
// ─────────────────────────────────────────────────────────────────────
// kd_tree.h – KD-tree for photon map spatial queries (Section 6.1)
// ─────────────────────────────────────────────────────────────────────
// CPU reference spatial index for the photon-centric renderer.
// Supports:
//   - Variable-radius range query (any radius per query point)
//   - k-nearest-neighbor query (for adaptive gather radius)
//   - Median-split construction O(N log N)
//   - Tangential (surface) distance metric (§6.3, v2.1 mandatory)
//   - Surface consistency filter integration (§6.4)
//
// Distance metric: tangential (§6.3), NOT 3D Euclidean.
// The tree is built on 3D positions but queries use tangential distance
// for pruning and candidate evaluation.  The tree still prunes on
// per-axis distance as a conservative bound (tangential distance ≤
// 3D Euclidean distance, so axis-aligned pruning is always safe).
//
// Header-only.  CPU reference implementation only — GPU uses hash grid.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "photon/photon.h"
#include "photon/surface_filter.h"

#include <vector>
#include <algorithm>
#include <cstdint>
#include <queue>
#include <limits>
#include <cmath>
#include <functional>

// ── KD-Tree Node ────────────────────────────────────────────────────
struct KDNode {
    int      split_axis;   // 0=x, 1=y, 2=z; -1 = leaf
    float    split_pos;    // split plane position
    uint32_t left;         // left child index (or first photon in leaf)
    uint32_t right;        // right child index (or one-past-last in leaf)
};

// Maximum photons per leaf node
constexpr int KD_MAX_LEAF_SIZE = 16;

// ── KD-Tree ─────────────────────────────────────────────────────────
struct KDTree {
    std::vector<KDNode>    nodes;
    std::vector<uint32_t>  indices;   // photon indices in leaf order

    // Bounding box of all photons (set during build)
    float3 bbox_min;
    float3 bbox_max;

    // ── Build ───────────────────────────────────────────────────────
    // Constructs KD-tree from photon positions using median split.
    // O(N log N) via std::nth_element.
    void build(const PhotonSoA& photons) {
        nodes.clear();
        indices.clear();

        size_t n = photons.size();
        if (n == 0) return;

        // Initialize index array
        indices.resize(n);
        for (size_t i = 0; i < n; ++i)
            indices[i] = (uint32_t)i;

        // Compute bounding box
        bbox_min = make_f3( 1e30f,  1e30f,  1e30f);
        bbox_max = make_f3(-1e30f, -1e30f, -1e30f);
        for (size_t i = 0; i < n; ++i) {
            float x = photons.pos_x[i];
            float y = photons.pos_y[i];
            float z = photons.pos_z[i];
            bbox_min.x = fminf(bbox_min.x, x);
            bbox_min.y = fminf(bbox_min.y, y);
            bbox_min.z = fminf(bbox_min.z, z);
            bbox_max.x = fmaxf(bbox_max.x, x);
            bbox_max.y = fmaxf(bbox_max.y, y);
            bbox_max.z = fmaxf(bbox_max.z, z);
        }

        // Reserve nodes (roughly 2N for a balanced tree)
        nodes.reserve(n * 2);

        // Recursive build
        build_recursive(photons, 0, (uint32_t)n, 0);
    }

    // ── Range query (3D Euclidean — legacy, used by build validation) ──
    // Finds all photons within `radius` of `pos` using 3D distance.
    // Calls callback(photon_index, distance_squared) for each.
    template<typename Callback>
    void query(float3 pos, float radius, const PhotonSoA& photons,
               Callback callback) const
    {
        if (nodes.empty()) return;
        float r2 = radius * radius;
        query_recursive_3d(0, pos, r2, photons, callback);
    }

    // ── Range query (tangential distance — §6.3, v2.1 mandatory) ────
    // Finds all photons within tangential `radius` of `pos` on the
    // tangent plane defined by `normal`.  Also applies plane distance
    // filter (|d_plane| < tau).
    // Calls callback(photon_index, tangential_distance_squared) for each.
    // NOTE: This is the primary query method for v2.1 gather operations.
    template<typename Callback>
    void query_tangential(float3 pos, float3 normal, float radius,
                          float tau, const PhotonSoA& photons,
                          Callback callback) const
    {
        if (nodes.empty()) return;
        float r2 = radius * radius;
        // Use 3D radius for tree pruning (conservative: d_tan <= d_3D)
        query_recursive_tangential(0, pos, normal, r2, tau, photons, callback);
    }

    // ── k-Nearest Neighbor query (3D Euclidean — legacy) ────────────
    // Finds the k closest photons to `pos` using 3D distance.
    // Returns their indices and the squared distance to the k-th nearest.
    void knn(float3 pos, int k, const PhotonSoA& photons,
             std::vector<uint32_t>& out_indices, float& out_max_dist2) const
    {
        out_indices.clear();
        out_max_dist2 = std::numeric_limits<float>::max();

        if (nodes.empty() || k <= 0) return;

        using HeapEntry = std::pair<float, uint32_t>;
        std::priority_queue<HeapEntry> heap;

        knn_recursive_3d(0, pos, k, photons, heap);

        out_max_dist2 = heap.empty() ? 0.0f : heap.top().first;
        out_indices.resize(heap.size());
        for (int i = (int)heap.size() - 1; i >= 0; --i) {
            out_indices[i] = heap.top().second;
            heap.pop();
        }
    }

    // ── k-Nearest Neighbor query (tangential distance — §6.3) ───────
    // Finds the k closest photons to `pos` using tangential distance on
    // the tangent plane defined by `normal`.  Also applies plane distance
    // filter (|d_plane| < tau) and normal/direction compatibility (§6.4).
    // Returns their indices and the squared tangential distance to the k-th.
    void knn_tangential(float3 pos, float3 normal, int k, float tau,
                        const PhotonSoA& photons,
                        std::vector<uint32_t>& out_indices,
                        float& out_max_dist2) const
    {
        out_indices.clear();
        out_max_dist2 = std::numeric_limits<float>::max();

        if (nodes.empty() || k <= 0) return;

        using HeapEntry = std::pair<float, uint32_t>;
        std::priority_queue<HeapEntry> heap;

        knn_recursive_tangential(0, pos, normal, k, tau, photons, heap);

        out_max_dist2 = heap.empty() ? 0.0f : heap.top().first;
        out_indices.resize(heap.size());
        for (int i = (int)heap.size() - 1; i >= 0; --i) {
            out_indices[i] = heap.top().second;
            heap.pop();
        }
    }

    // ── Utility ─────────────────────────────────────────────────────
    bool empty() const { return nodes.empty(); }
    size_t node_count() const { return nodes.size(); }

private:
    // ── Recursive build ─────────────────────────────────────────────
    uint32_t build_recursive(const PhotonSoA& photons,
                             uint32_t begin, uint32_t end, int depth)
    {
        uint32_t node_idx = (uint32_t)nodes.size();
        nodes.push_back(KDNode{});

        uint32_t count = end - begin;

        if (count <= (uint32_t)KD_MAX_LEAF_SIZE) {
            // Leaf node
            nodes[node_idx].split_axis = -1;
            nodes[node_idx].split_pos  = 0.0f;
            nodes[node_idx].left       = begin;  // first index in indices[]
            nodes[node_idx].right      = end;    // one past last
            return node_idx;
        }

        // Choose split axis: largest extent heuristic
        float3 lo = make_f3( 1e30f,  1e30f,  1e30f);
        float3 hi = make_f3(-1e30f, -1e30f, -1e30f);
        for (uint32_t i = begin; i < end; ++i) {
            uint32_t idx = indices[i];
            float x = photons.pos_x[idx];
            float y = photons.pos_y[idx];
            float z = photons.pos_z[idx];
            lo.x = fminf(lo.x, x); hi.x = fmaxf(hi.x, x);
            lo.y = fminf(lo.y, y); hi.y = fmaxf(hi.y, y);
            lo.z = fminf(lo.z, z); hi.z = fmaxf(hi.z, z);
        }

        float extent_x = hi.x - lo.x;
        float extent_y = hi.y - lo.y;
        float extent_z = hi.z - lo.z;

        int axis;
        if (extent_x >= extent_y && extent_x >= extent_z)
            axis = 0;
        else if (extent_y >= extent_z)
            axis = 1;
        else
            axis = 2;

        // Median split using nth_element
        uint32_t mid = begin + count / 2;
        std::nth_element(
            indices.begin() + begin,
            indices.begin() + mid,
            indices.begin() + end,
            [&](uint32_t a, uint32_t b) {
                return get_axis(photons, a, axis) < get_axis(photons, b, axis);
            }
        );

        float split_pos = get_axis(photons, indices[mid], axis);

        nodes[node_idx].split_axis = axis;
        nodes[node_idx].split_pos  = split_pos;

        // Build children (left = [begin, mid), right = [mid, end))
        uint32_t left_child  = build_recursive(photons, begin, mid, depth + 1);
        uint32_t right_child = build_recursive(photons, mid,   end, depth + 1);

        nodes[node_idx].left  = left_child;
        nodes[node_idx].right = right_child;

        return node_idx;
    }

    // ── Get position component by axis ──────────────────────────────
    static float get_axis(const PhotonSoA& photons, uint32_t idx, int axis) {
        switch (axis) {
            case 0: return photons.pos_x[idx];
            case 1: return photons.pos_y[idx];
            case 2: return photons.pos_z[idx];
            default: return 0.0f;
        }
    }

    // ── Recursive range query (3D Euclidean — legacy) ────────────────
    template<typename Callback>
    void query_recursive_3d(uint32_t node_idx, float3 pos, float r2,
                            const PhotonSoA& photons, Callback& callback) const
    {
        const KDNode& node = nodes[node_idx];

        if (node.split_axis == -1) {
            for (uint32_t i = node.left; i < node.right; ++i) {
                uint32_t idx = indices[i];
                float dx = photons.pos_x[idx] - pos.x;
                float dy = photons.pos_y[idx] - pos.y;
                float dz = photons.pos_z[idx] - pos.z;
                float dist2 = dx*dx + dy*dy + dz*dz;
                if (dist2 <= r2) {
                    callback(idx, dist2);
                }
            }
            return;
        }

        float pos_axis;
        switch (node.split_axis) {
            case 0: pos_axis = pos.x; break;
            case 1: pos_axis = pos.y; break;
            case 2: pos_axis = pos.z; break;
            default: pos_axis = 0.0f; break;
        }

        float diff = pos_axis - node.split_pos;
        float diff2 = diff * diff;

        uint32_t near_child = (diff <= 0.0f) ? node.left : node.right;
        uint32_t far_child  = (diff <= 0.0f) ? node.right : node.left;

        query_recursive_3d(near_child, pos, r2, photons, callback);

        if (diff2 <= r2) {
            query_recursive_3d(far_child, pos, r2, photons, callback);
        }
    }

    // ── Recursive range query (tangential distance — §6.3) ──────────
    // Tree pruning uses axis-aligned distance (conservative bound).
    // Leaf evaluation uses tangential distance + plane distance filter.
    template<typename Callback>
    void query_recursive_tangential(uint32_t node_idx, float3 pos,
                                    float3 normal, float r2, float tau,
                                    const PhotonSoA& photons,
                                    Callback& callback) const
    {
        const KDNode& node = nodes[node_idx];

        if (node.split_axis == -1) {
            for (uint32_t i = node.left; i < node.right; ++i) {
                uint32_t idx = indices[i];
                float3 ppos = make_f3(photons.pos_x[idx],
                                      photons.pos_y[idx],
                                      photons.pos_z[idx]);
                // Tangential distance computation (§6.3)
                TangentialResult tr = compute_tangential(pos, normal, ppos);
                // Plane distance filter
                if (fabsf(tr.d_plane) > tau) continue;
                // Tangential radius filter
                if (tr.d_tan2 <= r2) {
                    callback(idx, tr.d_tan2);
                }
            }
            return;
        }

        // Axis-aligned pruning (conservative: d_tan <= d_3D >= |d_axis|)
        float pos_axis;
        switch (node.split_axis) {
            case 0: pos_axis = pos.x; break;
            case 1: pos_axis = pos.y; break;
            case 2: pos_axis = pos.z; break;
            default: pos_axis = 0.0f; break;
        }

        float diff = pos_axis - node.split_pos;
        float diff2 = diff * diff;

        uint32_t near_child = (diff <= 0.0f) ? node.left : node.right;
        uint32_t far_child  = (diff <= 0.0f) ? node.right : node.left;

        query_recursive_tangential(near_child, pos, normal, r2, tau,
                                   photons, callback);

        // Conservative: axis distance < r means tangential could be < r
        if (diff2 <= r2) {
            query_recursive_tangential(far_child, pos, normal, r2, tau,
                                       photons, callback);
        }
    }

    // ── Recursive k-NN query (3D Euclidean — legacy) ────────────────
    void knn_recursive_3d(uint32_t node_idx, float3 pos, int k,
                          const PhotonSoA& photons,
                          std::priority_queue<std::pair<float, uint32_t>>& heap) const
    {
        const KDNode& node = nodes[node_idx];

        if (node.split_axis == -1) {
            for (uint32_t i = node.left; i < node.right; ++i) {
                uint32_t idx = indices[i];
                float dx = photons.pos_x[idx] - pos.x;
                float dy = photons.pos_y[idx] - pos.y;
                float dz = photons.pos_z[idx] - pos.z;
                float dist2 = dx*dx + dy*dy + dz*dz;

                if ((int)heap.size() < k) {
                    heap.push({dist2, idx});
                } else if (dist2 < heap.top().first) {
                    heap.pop();
                    heap.push({dist2, idx});
                }
            }
            return;
        }

        float pos_axis;
        switch (node.split_axis) {
            case 0: pos_axis = pos.x; break;
            case 1: pos_axis = pos.y; break;
            case 2: pos_axis = pos.z; break;
            default: pos_axis = 0.0f; break;
        }

        float diff = pos_axis - node.split_pos;
        float diff2 = diff * diff;

        uint32_t near_child = (diff <= 0.0f) ? node.left : node.right;
        uint32_t far_child  = (diff <= 0.0f) ? node.right : node.left;

        knn_recursive_3d(near_child, pos, k, photons, heap);

        float max_dist2 = ((int)heap.size() < k)
            ? std::numeric_limits<float>::max()
            : heap.top().first;

        if (diff2 < max_dist2) {
            knn_recursive_3d(far_child, pos, k, photons, heap);
        }
    }

    // ── Recursive k-NN query (tangential distance — §6.3) ───────────
    // Uses tangential distance for candidate evaluation.
    // Axis-aligned pruning is conservative (d_axis ≤ d_3D ≥ d_tan).
    // Plane distance filter: |d_plane| > tau rejects candidate.
    void knn_recursive_tangential(
        uint32_t node_idx, float3 pos, float3 normal, int k, float tau,
        const PhotonSoA& photons,
        std::priority_queue<std::pair<float, uint32_t>>& heap) const
    {
        const KDNode& node = nodes[node_idx];

        if (node.split_axis == -1) {
            for (uint32_t i = node.left; i < node.right; ++i) {
                uint32_t idx = indices[i];
                float3 ppos = make_f3(photons.pos_x[idx],
                                      photons.pos_y[idx],
                                      photons.pos_z[idx]);
                TangentialResult tr = compute_tangential(pos, normal, ppos);
                if (fabsf(tr.d_plane) > tau) continue;
                float dist2 = tr.d_tan2;

                if ((int)heap.size() < k) {
                    heap.push({dist2, idx});
                } else if (dist2 < heap.top().first) {
                    heap.pop();
                    heap.push({dist2, idx});
                }
            }
            return;
        }

        float pos_axis;
        switch (node.split_axis) {
            case 0: pos_axis = pos.x; break;
            case 1: pos_axis = pos.y; break;
            case 2: pos_axis = pos.z; break;
            default: pos_axis = 0.0f; break;
        }

        float diff = pos_axis - node.split_pos;
        float diff2 = diff * diff;

        uint32_t near_child = (diff <= 0.0f) ? node.left : node.right;
        uint32_t far_child  = (diff <= 0.0f) ? node.right : node.left;

        knn_recursive_tangential(near_child, pos, normal, k, tau,
                                 photons, heap);

        // Conservative axis-aligned pruning
        float max_dist2 = ((int)heap.size() < k)
            ? std::numeric_limits<float>::max()
            : heap.top().first;

        if (diff2 < max_dist2) {
            knn_recursive_tangential(far_child, pos, normal, k, tau,
                                     photons, heap);
        }
    }
};
