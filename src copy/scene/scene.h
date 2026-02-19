#pragma once
// ─────────────────────────────────────────────────────────────────────
// scene.h – Scene representation with BVH for CPU ray tracing
// ─────────────────────────────────────────────────────────────────────
#include "scene/triangle.h"
#include "scene/material.h"
#include "core/spectrum.h"
#include "core/alias_table.h"
#include <vector>
#include <string>

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

    float3 sample(float2 uv) const {
        if (width == 0 || height == 0) return make_f3(1, 1, 1);
        int x = (int)(uv.x * width)  % width;
        int y = (int)(uv.y * height) % height;
        if (x < 0) x += width;
        if (y < 0) y += height;
        int idx = (y * width + x) * channels;
        return make_f3(data[idx], data[idx+1], data[idx+2]);
    }
};

// ── Scene ───────────────────────────────────────────────────────────
struct Scene {
    std::vector<Triangle>  triangles;
    std::vector<Material>  materials;
    std::vector<Texture>   textures;
    std::vector<BVHNode>   bvh_nodes;

    // Emissive triangle indices and alias table
    std::vector<uint32_t>  emissive_tri_indices;
    AliasTable             emissive_alias_table;
    float                  total_emissive_power = 0.f;

    AABB                   scene_bounds;

    // ── Build acceleration structure ────────────────────────────────
    void build_bvh();

    // ── Build emissive triangle distribution ────────────────────────
    void build_emissive_distribution();

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

inline void Scene::build_emissive_distribution() {
    emissive_tri_indices.clear();
    std::vector<float> weights;

    for (uint32_t i = 0; i < (uint32_t)triangles.size(); ++i) {
        const auto& tri = triangles[i];
        const auto& mat = materials[tri.material_id];
        if (mat.is_emissive()) {
            float w = tri.area() * mat.mean_emission();
            emissive_tri_indices.push_back(i);
            weights.push_back(w);
        }
    }

    if (!weights.empty()) {
        emissive_alias_table = AliasTable::build(weights);
        total_emissive_power = emissive_alias_table.total_weight;
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
                        result.normal   = triangles[i].geometric_normal();
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
