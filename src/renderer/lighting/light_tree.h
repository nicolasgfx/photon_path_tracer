#pragma once
// light_tree.h — CPU-side light BVH builder for importance-driven NEE
//
// Builds a binary tree over emissive triangles.  Each node stores a
// bounding box, aggregate flux (power), and a representative normal
// cone.  The GPU traverses top-down, probabilistically choosing the
// child whose importance bound is higher for the current shading point.
//
// Reference: Estevez & Kulla, "Importance Sampling of Many Lights with
// Adaptive Tree Splitting", HPG 2018.

#include "core/types.h"
#include "lighting/light_tree_node.h"
#include "scene/scene.h"
#include <algorithm>
#include <cstdio>
#include <vector>

// ── CPU builder ─────────────────────────────────────────────────────

struct LightTree {
    std::vector<LightTreeNode>  nodes;      // flat array (GPU upload target)
    std::vector<uint32_t>       tri_order;  // reordered emissive_tri_indices
    int root = 0;

    // Build from scene emissive data.  Call after build_emissive_distribution().
    void build(const Scene& scene, int max_leaf_size = 4) {
        max_leaf_size_ = (max_leaf_size >= 1) ? max_leaf_size : 4;
        int n = (int)scene.emissive_tri_indices.size();
        if (n == 0) return;

        // collect per-triangle info
        std::vector<TriInfo> infos(n);
        for (int i = 0; i < n; ++i) {
            uint32_t ti = scene.emissive_tri_indices[i];
            const auto& tri = scene.triangles[ti];
            const auto& mat = scene.materials[tri.material_id];
            AABB box = AABB::empty();
            box.expand(tri.v0);
            box.expand(tri.v1);
            box.expand(tri.v2);

            infos[i].global_tri = ti;
            infos[i].box = box;
            infos[i].normal = tri.geometric_normal();
            infos[i].centroid = box.center();
            infos[i].power = tri.area() * mat.mean_emission();
        }

        nodes.clear();
        nodes.reserve(2 * n);
        tri_order.clear();
        tri_order.reserve(n);
        tri_order_count_ = 0;

        root = build_recursive(infos.data(), 0, n);

        std::printf("[LightTree] Built: %d emissive tris, %d nodes\n",
                    n, (int)nodes.size());
    }

private:
    struct TriInfo {
        uint32_t global_tri;
        AABB     box;
        float3   normal;
        float3   centroid;
        float    power;
    };

    // --- recursive SAH-like builder
    int build_recursive(TriInfo* infos, int begin, int end) {
        int count = end - begin;

        // leaf: 1-N triangles (N = max_leaf_size_)
        if (count <= max_leaf_size_) return make_leaf(infos, begin, end);

        // compute centroid bounds for split axis
        AABB centroid_box = AABB::empty();
        for (int i = begin; i < end; ++i)
            centroid_box.expand(infos[i].centroid);

        float3 ext = centroid_box.extent();
        int axis = 0;
        if (ext.y > ext.x) axis = 1;
        if (((axis == 0) ? ext.x : ext.y) < ext.z) axis = 2;

        float extent_on_axis = (axis == 0) ? ext.x : (axis == 1) ? ext.y : ext.z;

        // degenerate: all centroids coincide
        if (extent_on_axis < 1e-6f)
            return make_leaf(infos, begin, end);

        // SAH binning (12 bins)
        constexpr int NUM_BINS = 12;
        struct Bin { AABB box = AABB::empty(); float flux = 0.f; int count = 0; };
        Bin bins[NUM_BINS];

        float lo = (axis == 0) ? centroid_box.lo.x : (axis == 1) ? centroid_box.lo.y : centroid_box.lo.z;
        float hi = (axis == 0) ? centroid_box.hi.x : (axis == 1) ? centroid_box.hi.y : centroid_box.hi.z;
        float inv_range = (hi - lo > 0.f) ? (float)NUM_BINS / (hi - lo) : 0.f;

        for (int i = begin; i < end; ++i) {
            float c = (axis == 0) ? infos[i].centroid.x :
                      (axis == 1) ? infos[i].centroid.y : infos[i].centroid.z;
            int b = (int)((c - lo) * inv_range);
            if (b >= NUM_BINS) b = NUM_BINS - 1;
            if (b < 0) b = 0;
            bins[b].box.expand(infos[i].box);
            bins[b].flux += infos[i].power;
            bins[b].count++;
        }

        // evaluate splits
        float best_cost = 1e30f;
        int best_split = -1;
        for (int s = 1; s < NUM_BINS; ++s) {
            AABB left_box = AABB::empty(); float left_flux = 0.f; int left_n = 0;
            AABB right_box = AABB::empty(); float right_flux = 0.f; int right_n = 0;
            for (int j = 0; j < s; ++j) {
                left_box.expand(bins[j].box); left_flux += bins[j].flux; left_n += bins[j].count;
            }
            for (int j = s; j < NUM_BINS; ++j) {
                right_box.expand(bins[j].box); right_flux += bins[j].flux; right_n += bins[j].count;
            }
            if (left_n == 0 || right_n == 0) continue;
            // cost ∝ flux × surface area (oriented cost)
            float cost = left_flux * box_surface_area(left_box) +
                         right_flux * box_surface_area(right_box);
            if (cost < best_cost) {
                best_cost = cost;
                best_split = s;
            }
        }

        // failed to find good split — make leaf
        if (best_split < 0)
            return make_leaf(infos, begin, end);

        // partition
        float split_val = lo + (float)best_split / inv_range;
        auto* mid_ptr = std::partition(infos + begin, infos + end,
            [axis, split_val](const TriInfo& t) {
                float c = (axis == 0) ? t.centroid.x :
                          (axis == 1) ? t.centroid.y : t.centroid.z;
                return c < split_val;
            });
        int mid = (int)(mid_ptr - infos);
        if (mid <= begin || mid >= end) {
            // partition failed, split in half
            mid = begin + count / 2;
            std::nth_element(infos + begin, infos + mid, infos + end,
                [axis](const TriInfo& a, const TriInfo& b) {
                    float ca = (axis == 0) ? a.centroid.x : (axis == 1) ? a.centroid.y : a.centroid.z;
                    float cb = (axis == 0) ? b.centroid.x : (axis == 1) ? b.centroid.y : b.centroid.z;
                    return ca < cb;
                });
        }

        int left  = build_recursive(infos, begin, mid);
        int right = build_recursive(infos, mid, end);

        // make interior node
        LightTreeNode node;
        node.bbox_lo = fminf3(nodes[left].bbox_lo, nodes[right].bbox_lo);
        node.bbox_hi = fmaxf3(nodes[left].bbox_hi, nodes[right].bbox_hi);
        node.flux    = nodes[left].flux + nodes[right].flux;
        node.child_left  = left;
        node.child_right = right;
        node.tri_count   = 0;

        // compute orientation cone from children
        combine_cones(nodes[left], nodes[right], node);

        int idx = (int)nodes.size();
        nodes.push_back(node);
        return idx;
    }

    int make_leaf(TriInfo* infos, int begin, int end) {
        int count = end - begin;
        LightTreeNode node;
        node.bbox_lo = make_f3(1e30f, 1e30f, 1e30f);
        node.bbox_hi = make_f3(-1e30f, -1e30f, -1e30f);
        node.flux = 0.f;

        float3 avg_normal = make_f3(0, 0, 0);
        int first_ordered = (int)tri_order.size();

        for (int i = begin; i < end; ++i) {
            node.bbox_lo = fminf3(node.bbox_lo, infos[i].box.lo);
            node.bbox_hi = fmaxf3(node.bbox_hi, infos[i].box.hi);
            node.flux += infos[i].power;
            avg_normal = avg_normal + infos[i].normal * infos[i].power;
            tri_order.push_back(infos[i].global_tri);
        }

        // orientation cone
        float nlen = length(avg_normal);
        if (nlen > 1e-8f) {
            node.axis = avg_normal * (1.f / nlen);
            // cos of cone half-angle: worst-case deviation among leaf tris
            float min_cos = 1.f;
            for (int i = begin; i < end; ++i) {
                float c = dot(node.axis, infos[i].normal);
                if (c < min_cos) min_cos = c;
            }
            node.cos_theta_o = min_cos;
        } else {
            node.axis = make_f3(0, 1, 0);
            node.cos_theta_o = -1.f;  // omni
        }

        node.child_left  = first_ordered;  // first index into tri_order
        node.child_right = -1;
        node.tri_count   = count;

        int idx = (int)nodes.size();
        nodes.push_back(node);
        return idx;
    }

    int tri_order_count_ = 0;
    int max_leaf_size_   = 4;

    // --- helpers
    static float box_surface_area(const AABB& b) {
        float3 e = b.extent();
        return 2.f * (e.x * e.y + e.y * e.z + e.z * e.x);
    }

    static void combine_cones(const LightTreeNode& a, const LightTreeNode& b,
                               LightTreeNode& out) {
        // flux-weighted average axis
        float3 avg = a.axis * a.flux + b.axis * b.flux;
        float nlen = length(avg);
        if (nlen < 1e-8f) {
            out.axis = make_f3(0, 1, 0);
            out.cos_theta_o = -1.f;
            return;
        }
        out.axis = avg * (1.f / nlen);

        // cone must contain both children's cones
        float cos_a = dot(out.axis, a.axis);
        float cos_b = dot(out.axis, b.axis);
        // half-angle of child a relative to new axis + child a's own half-angle
        float angle_a = acosf(fminf(fmaxf(cos_a, -1.f), 1.f)) +
                         acosf(fminf(fmaxf(a.cos_theta_o, -1.f), 1.f));
        float angle_b = acosf(fminf(fmaxf(cos_b, -1.f), 1.f)) +
                         acosf(fminf(fmaxf(b.cos_theta_o, -1.f), 1.f));
        float max_angle = fmaxf(angle_a, angle_b);
        if (max_angle >= PI) {
            out.cos_theta_o = -1.f;  // omni
        } else {
            out.cos_theta_o = cosf(max_angle);
        }
    }
};
