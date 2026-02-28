#pragma once
// ─────────────────────────────────────────────────────────────────────
// scene.h – Scene representation with BVH for CPU ray tracing
// ─────────────────────────────────────────────────────────────────────
#include "scene/triangle.h"
#include "scene/material.h"
#include "core/spectrum.h"
#include "core/config.h"
#include "core/alias_table.h"
#include "volume/medium.h"
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>

// ── BVH Node ────────────────────────────────────────────────────────
struct BVHNode {
    AABB     bounds;
    uint32_t left;        // child or first triangle index
    uint32_t right;       // child or triangle count (if leaf)
    bool     is_leaf;
};

// ── Texture ─────────────────────────────────────────────────────────
struct Texture {
    std::vector<float> data; // RGBA float
    int width  = 0;
    int height = 0;
    int channels = 4;
    std::string path;        // source file path (for dedup)

    float3 sample(float2 uv) const {
        if (width == 0 || height == 0) return make_f3(1, 1, 1);
        // Wrap UVs to [0,1)
        float u = uv.x - floorf(uv.x);
        float v = uv.y - floorf(uv.y);
        // Flip V (OBJ convention: V=0 at bottom, image: row 0 at top)
        v = 1.f - v;
        int ix = (int)(u * width)  % width;
        int iy = (int)(v * height) % height;
        if (ix < 0) ix += width;
        if (iy < 0) iy += height;
        int idx = (iy * width + ix) * channels;
        return make_f3(data[idx], data[idx+1], data[idx+2]);
    }
};

// ── Scene ───────────────────────────────────────────────────────────
struct Scene {
    std::vector<Triangle>  triangles;
    std::vector<Material>  materials;
    std::vector<Texture>   textures;
    std::vector<BVHNode>   bvh_nodes;
    std::vector<HomogeneousMedium> media;  // participating media (indexed by Material::medium_id)

    // Emissive triangle indices and alias tables
    std::vector<uint32_t>  emissive_tri_indices;
    AliasTable             emissive_alias_table;       // power-weighted
    AliasTable             emissive_area_alias_table;  // area-weighted (§7.2.1)
    float                  total_emissive_power = 0.f;
    float                  total_emissive_area  = 0.f;

    AABB                   scene_bounds;

    // ── Build acceleration structure ────────────────────────────────
    void build_bvh();

    // ── Build emissive triangle distribution ────────────────────────
    void build_emissive_distribution();

    // ── Normalize geometry to reference frame ───────────────────────
    // Translates scene centre to SCENE_REF_CENTER and scales the
    // longest axis to SCENE_REF_EXTENT.  Call AFTER load, BEFORE
    // build_bvh().  Skipped when SCENE_IS_REFERENCE == true.
    void normalize_to_reference();

    // ── CPU ray intersection (BVH traversal) ────────────────────────
    HitRecord intersect(const Ray& ray) const;

    // ── Scene statistics ────────────────────────────────────────────
    size_t num_triangles()  const { return triangles.size(); }
    size_t num_materials()  const { return materials.size(); }
    size_t num_emissive()   const { return emissive_tri_indices.size(); }

private:
    uint32_t build_bvh_recursive(std::vector<uint32_t>& indices,
                                  int start, int end, int depth);
};

// ── Implementation ──────────────────────────────────────────────────

// Normalise the scene so its bounding box matches the Cornell-Box
// reference frame: centred at SCENE_REF_CENTER, longest axis =
// SCENE_REF_EXTENT (1.0).  Vertex positions AND normals (normals are
// direction-only, so only positions are transformed).
inline void Scene::normalize_to_reference() {
    if (triangles.empty()) return;

    // 1. Compute current AABB
    AABB bb;
    for (const auto& t : triangles) {
        bb.expand(t.v0);
        bb.expand(t.v1);
        bb.expand(t.v2);
    }

    float3 cur_center = bb.center();
    float3 ext        = bb.extent();
    float  longest    = fmaxf(fmaxf(ext.x, ext.y), ext.z);

    if (longest < 1e-12f) return;  // degenerate

    float  scale = SCENE_REF_EXTENT / longest;
    float3 ref_c = make_f3(0.f, 0.f, 0.f);  // SCENE_REF_CENTER

    std::cout << "[Scene] Normalising: centre ("
              << cur_center.x << ", " << cur_center.y << ", "
              << cur_center.z << ")  extent ("
              << ext.x << ", " << ext.y << ", " << ext.z
              << ")  scale " << scale << "\n";

    // 2. Apply: p' = (p - cur_center) * scale + ref_center
    for (auto& t : triangles) {
        t.v0 = (t.v0 - cur_center) * scale + ref_c;
        t.v1 = (t.v1 - cur_center) * scale + ref_c;
        t.v2 = (t.v2 - cur_center) * scale + ref_c;
        // Normals are pure directions — no transformation needed.
    }
}

inline void Scene::build_emissive_distribution() {
    emissive_tri_indices.clear();
    std::vector<float> power_weights;
    std::vector<float> area_weights;

    for (uint32_t i = 0; i < (uint32_t)triangles.size(); ++i) {
        const auto& tri = triangles[i];
        const auto& mat = materials[tri.material_id];
        if (mat.is_emissive()) {
            float a = tri.area();
            float w = a * mat.mean_emission();
            emissive_tri_indices.push_back(i);
            power_weights.push_back(w);
            area_weights.push_back(a);
        }
    }

    if (!power_weights.empty()) {
        emissive_alias_table      = AliasTable::build(power_weights);
        total_emissive_power      = emissive_alias_table.total_weight;
        emissive_area_alias_table = AliasTable::build(area_weights);
        total_emissive_area       = emissive_area_alias_table.total_weight;

        const int   num_lights     = (int)emissive_tri_indices.size();
        const int   nee_samples    = DEFAULT_NEE_LIGHT_SAMPLES;
        const float coverage_pct   = std::min(1.0f, (float)nee_samples / (float)num_lights) * 100.0f;
        std::printf("[Scene] Emissive alias table: %d light triangles  "
                    "(NEE samples=%d  =>  %.1f%% coverage per shading point)\n",
                    num_lights, nee_samples, coverage_pct);
    }
}

inline uint32_t Scene::build_bvh_recursive(std::vector<uint32_t>& indices,
                                            int start, int end, int depth) {
    BVHNode node;
    node.bounds = AABB{};
    for (int i = start; i < end; ++i) {
        const auto& t = triangles[indices[i]];
        node.bounds.expand(t.v0);
        node.bounds.expand(t.v1);
        node.bounds.expand(t.v2);
    }

    int count = end - start;

    // Leaf node
    if (count <= 4 || depth > 30) {
        node.is_leaf = true;
        node.left    = start;
        node.right   = count;
        uint32_t idx = (uint32_t)bvh_nodes.size();
        bvh_nodes.push_back(node);
        return idx;
    }

    // Find split axis and midpoint
    int axis = node.bounds.longest_axis();
    float mid = 0.f;
    switch (axis) {
        case 0: mid = node.bounds.center().x; break;
        case 1: mid = node.bounds.center().y; break;
        case 2: mid = node.bounds.center().z; break;
    }

    // Partition
    int split = start;
    for (int i = start; i < end; ++i) {
        const auto& t = triangles[indices[i]];
        float3 c = (t.v0 + t.v1 + t.v2) / 3.f;
        float cv = (axis == 0) ? c.x : (axis == 1) ? c.y : c.z;
        if (cv < mid) {
            std::swap(indices[i], indices[split]);
            split++;
        }
    }

    // Fallback: split in half if partition failed
    if (split == start || split == end) {
        split = (start + end) / 2;
    }

    node.is_leaf = false;
    uint32_t idx = (uint32_t)bvh_nodes.size();
    bvh_nodes.push_back(node); // placeholder

    uint32_t left  = build_bvh_recursive(indices, start, split, depth + 1);
    uint32_t right = build_bvh_recursive(indices, split, end,   depth + 1);

    bvh_nodes[idx].left  = left;
    bvh_nodes[idx].right = right;
    return idx;
}

inline void Scene::build_bvh() {
    bvh_nodes.clear();
    if (triangles.empty()) return;

    std::vector<uint32_t> indices(triangles.size());
    for (uint32_t i = 0; i < (uint32_t)triangles.size(); ++i) indices[i] = i;

    // Compute scene bounds
    scene_bounds = AABB{};
    for (const auto& t : triangles) {
        scene_bounds.expand(t.v0);
        scene_bounds.expand(t.v1);
        scene_bounds.expand(t.v2);
    }

    build_bvh_recursive(indices, 0, (int)indices.size(), 0);

    // Reorder triangles to match BVH leaf order
    std::vector<Triangle> reordered(triangles.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        reordered[i] = triangles[indices[i]];
    }
    triangles = std::move(reordered);
}

inline HitRecord Scene::intersect(const Ray& ray) const {
    HitRecord result{};
    result.hit = false;
    result.t   = ray.tmax;

    if (bvh_nodes.empty()) return result;

    // Iterative BVH traversal with stack
    uint32_t stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0; // root

    while (stack_ptr > 0) {
        uint32_t node_idx = stack[--stack_ptr];
        const BVHNode& node = bvh_nodes[node_idx];

        float tmin_box, tmax_box;
        if (!node.bounds.intersect(ray, tmin_box, tmax_box)) continue;
        if (tmin_box > result.t) continue;

        if (node.is_leaf) {
            for (uint32_t i = node.left; i < node.left + node.right; ++i) {
                float t, u, v;
                Ray test_ray = ray;
                test_ray.tmax = result.t;
                if (triangles[i].intersect(test_ray, t, u, v)) {
                    if (t < result.t) {
                        result.t = t;
                        result.hit = true;
                        result.triangle_id = i;
                        result.material_id = triangles[i].material_id;
                        float alpha = 1.f - u - v;
                        result.position = triangles[i].interpolate_position(alpha, u, v);
                        result.normal = triangles[i].geometric_normal();
                        result.shading_normal = triangles[i].interpolate_normal(alpha, u, v);
                        result.uv = triangles[i].interpolate_uv(alpha, u, v);
                    }
                }
            }
        } else {
            stack[stack_ptr++] = node.left;
            stack[stack_ptr++] = node.right;
        }
    }

    return result;
}
