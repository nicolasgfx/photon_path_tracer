#pragma once
// ─────────────────────────────────────────────────────────────────────
// scene.h – Scene representation with BVH for CPU ray tracing
// ─────────────────────────────────────────────────────────────────────
#include "scene/triangle.h"
#include "scene/material.h"
#include "scene/envmap.h"
#include "core/spectrum.h"
#include "core/config.h"
#include "core/alias_table.h"
#include "volume/medium.h"
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <memory>
#include <cfloat>

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

// ── Instancing descriptors ───────────────────────────────────────────
struct MeshDescriptor {
    uint32_t tri_offset;   // start index in scene.triangles[]
    uint32_t tri_count;    // number of triangles
};
struct InstanceDescriptor {
    uint32_t mesh_id;      // index into scene.meshes[]
    float transform[12];   // 3×4 row-major object→world
};

// ── Scene ───────────────────────────────────────────────────────────
struct Scene {
    std::vector<Triangle>  triangles;
    std::vector<Material>  materials;
    std::vector<Texture>   textures;
    std::vector<BVHNode>   bvh_nodes;
    std::vector<HomogeneousMedium> media;  // participating media (indexed by Material::medium_id)

    // Instancing (populated by PBRT loader; OBJ loader wraps as single mesh)
    std::vector<MeshDescriptor>    meshes;
    std::vector<InstanceDescriptor> instances;
    bool has_instances() const { return instances.size() > 1; }

    // Emissive triangle indices and alias table (power-weighted)
    std::vector<uint32_t>  emissive_tri_indices;
    AliasTable             emissive_alias_table;       // power-weighted
    float                  total_emissive_power = 0.f;

    AABB                   scene_bounds;

    // Normalization transform applied by normalize_to_reference().
    // Camera positions from PBRT files need the same transform:
    //   p_norm = (p_orig - norm_center) * norm_scale
    float3 norm_center = {0, 0, 0};
    float  norm_scale  = 1.0f;

    // ── Environment map (infinite light) ──────────────────────────
    std::shared_ptr<EnvironmentMap> envmap;  // nullptr if no envmap
    float envmap_selection_prob = 0.f;       // probability of choosing envmap in NEE/photon

    // Source data from PBRT loader (avoids JSON round-trip)
    bool        pbrt_has_envmap      = false;
    std::string pbrt_envmap_path;              // EXR/HDR path (empty for constant)
    float       pbrt_envmap_scale    = 1.0f;
    float3      pbrt_envmap_rotation = {0, 0, 0};
    float3      pbrt_envmap_constant = {0, 0, 0}; // unscaled RGB

    // PBRT camera (transformed by normalize_to_reference)
    bool   pbrt_cam_valid    = false;
    float3 pbrt_cam_position = {0, 0, 0};
    float3 pbrt_cam_look_at  = {0, 0, -1};
    float3 pbrt_cam_up       = {0, 1, 0};
    float  pbrt_cam_fov      = 90.0f;

    bool has_envmap() const { return envmap && envmap->width > 0; }

    // Compute the envmap vs emissive triangle selection probability
    // based on relative power.  Call after build_emissive_distribution()
    // and envmap->build_distribution().
    void compute_envmap_selection_prob() {
        if (!has_envmap()) { envmap_selection_prob = 0.f; return; }
        float env_power = envmap->total_power;
        float tri_power = total_emissive_power;
        if (env_power + tri_power <= 0.f) { envmap_selection_prob = 0.5f; return; }
        envmap_selection_prob = env_power / (env_power + tri_power);
        // Clamp to [0.1, 0.9] to avoid starving either strategy
        envmap_selection_prob = fmaxf(0.1f, fminf(0.9f, envmap_selection_prob));
        std::printf("[Scene] Envmap selection prob = %.4f  (env=%.2f  tri=%.2f)\n",
                    envmap_selection_prob, env_power, tri_power);
    }

    // Bounding sphere for photon emission from infinity
    float3 scene_bounding_center() const {
        return scene_bounds.center();
    }
    float scene_bounding_radius() const {
        float3 ext = scene_bounds.extent();
        return length(ext) * 0.5f * 1.01f;  // slight expansion
    }

    // ── Build acceleration structure ────────────────────────────────
    void build_bvh();

    // ── Build emissive triangle distribution ────────────────────────
    void build_emissive_distribution();

    // ── Compute min / max emissive radiance across all emittters ────
    // Scans every emissive triangle and returns the smallest and largest
    // material mean_emission() values.  Used for adaptive bloom.
    void compute_emissive_radiance_range(float& out_min_Le,
                                         float& out_max_Le) const;

    // ── Normalize geometry to reference frame ───────────────────────
    // Translates scene centre to origin and scales the
    // longest axis to 1.0 (reference frame).  Call AFTER load, BEFORE
    // build_bvh().  Skipped when SCENE_IS_REFERENCE == true.
    void normalize_to_reference();

    // ── Rotate geometry 180° around X axis ──────────────────────────
    // Negates Y and Z on positions and normals.  A rotation (not a
    // reflection) so winding order is preserved automatically.
    // Call AFTER normalize, BEFORE build_bvh().
    void rotate_x_180();

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
// reference frame: centred at origin, longest axis = 1.0.
// Vertex positions AND normals (normals are
// direction-only, so only positions are transformed).
//
// For instanced scenes: only meshes[0] (world-space geometry) has its
// vertices transformed directly.  Instanced template vertices stay in
// object space; their instance transforms are updated to
//   T_inst' = M_norm * T_inst
// so OptiX maps object→world→normalised in one step.
inline void Scene::normalize_to_reference() {
    if (triangles.empty()) return;

    // 1. Compute current AABB from ALL triangles (world-space +
    //    object-space) — the bounding box of the whole triangle buffer.
    //    For a correct AABB we'd need to transform instanced verts
    //    through their instance transforms, but meshes[0] (world-space
    //    non-instanced geometry) dominates the scene extents, and any
    //    instanced geometry that falls outside is still captured because
    //    both original world positions AND template positions were loaded
    //    in the same file — so the BB is a reasonable superset.
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

    float  scale = 1.0f / longest;

    // Store normalization params so the camera can be transformed to match.
    norm_center = cur_center;
    norm_scale  = scale;

    std::cout << "[Scene] Normalising: centre ("
              << cur_center.x << ", " << cur_center.y << ", "
              << cur_center.z << ")  extent ("
              << ext.x << ", " << ext.y << ", " << ext.z
              << ")  scale " << scale << "\n";

    // 2a-pre. Transform PBRT camera with the same mapping.
    if (pbrt_cam_valid) {
        pbrt_cam_position = (pbrt_cam_position - cur_center) * scale;
        pbrt_cam_look_at  = (pbrt_cam_look_at  - cur_center) * scale;
    }

    // 2a-media. Scale participating-media coefficients so that optical
    //   depth is preserved after geometry shrinks by `scale`.
    //   Original: tau = sigma * d_orig
    //   After:    d_norm = d_orig * scale  →  sigma_norm = sigma / scale
    {
        float inv_scale = 1.0f / scale;
        for (auto& m : media) {
            m.sigma_a = m.sigma_a * inv_scale;
            m.sigma_s = m.sigma_s * inv_scale;
            m.sigma_t = m.sigma_t * inv_scale;
        }
    }

    // 2a. Transform world-space (non-instanced) vertices in-place.
    //     If there's instancing metadata, only meshes[0] is world-space;
    //     meshes[1+] are object-space templates and must NOT be touched.
    uint32_t world_tri_end = 0;
    if (!meshes.empty())
        world_tri_end = meshes[0].tri_offset + meshes[0].tri_count;
    else
        world_tri_end = (uint32_t)triangles.size();  // no metadata yet: all tris

    for (uint32_t i = 0; i < world_tri_end; ++i) {
        auto& t = triangles[i];
        t.v0 = (t.v0 - cur_center) * scale;
        t.v1 = (t.v1 - cur_center) * scale;
        t.v2 = (t.v2 - cur_center) * scale;
    }

    // 2b. Update instance transforms: T' = M_norm * T
    //     M_norm is: p' = (p - center) * scale  →  3×4 row-major:
    //       [ s  0  0  -cx*s ]
    //       [ 0  s  0  -cy*s ]
    //       [ 0  0  s  -cz*s ]
    //     Multiply M_norm * T (3×4 * 4×4 → 3×4).
    if (has_instances()) {
        float cx = cur_center.x, cy = cur_center.y, cz = cur_center.z;
        float s  = scale;
        // Skip instances[0] — it's the identity for meshes[0] (already
        // transformed in-place above).  Update instances[1+].
        for (size_t i = 1; i < instances.size(); ++i) {
            float old[12];
            std::memcpy(old, instances[i].transform, sizeof(old));

            // M_norm * T  where M_norm = diag(s,s,s) with translation (-cx*s, -cy*s, -cz*s)
            // Row r of result: sum over k of M_norm[r][k] * T_col[k]
            // M_norm is diagonal+translation, so:
            //   result[r][c] = s * old[r][c]  +  (r==0 ? -cx*s : r==1 ? -cy*s : -cz*s) * (c==3 part from 4th row of T)
            // The 4th row of the 4×4 version of T (3×4 affine) is [0,0,0,1].
            // So result[r][c] = s * old[r*4+c]  for c<3
            //    result[r][3] = s * old[r*4+3]  + (-cr*s)   where cr = {cx,cy,cz}[r]
            float off[3] = { -cx * s, -cy * s, -cz * s };
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c)
                    instances[i].transform[r * 4 + c] = s * old[r * 4 + c];
                instances[i].transform[r * 4 + 3] = s * old[r * 4 + 3] + off[r];
            }
        }
    }
}

inline void Scene::rotate_x_180() {
    if (triangles.empty()) return;

    // Only transform world-space (non-instanced) vertices directly.
    uint32_t world_tri_end = 0;
    if (!meshes.empty())
        world_tri_end = meshes[0].tri_offset + meshes[0].tri_count;
    else
        world_tri_end = (uint32_t)triangles.size();

    for (uint32_t i = 0; i < world_tri_end; ++i) {
        auto& t = triangles[i];
        // Negate Y and Z on positions
        t.v0.y = -t.v0.y;  t.v0.z = -t.v0.z;
        t.v1.y = -t.v1.y;  t.v1.z = -t.v1.z;
        t.v2.y = -t.v2.y;  t.v2.z = -t.v2.z;
        // Negate Y and Z on shading normals
        t.n0.y = -t.n0.y;  t.n0.z = -t.n0.z;
        t.n1.y = -t.n1.y;  t.n1.z = -t.n1.z;
        t.n2.y = -t.n2.y;  t.n2.z = -t.n2.z;
    }

    // Update instance transforms: T' = M_rot * T
    // M_rot (X-180°): [ 1 0 0 0 ]
    //                  [ 0 -1 0 0 ]
    //                  [ 0 0 -1 0 ]
    // Result: row 0 unchanged, rows 1+2 negated.
    if (has_instances()) {
        for (size_t i = 1; i < instances.size(); ++i) {
            float* tf = instances[i].transform;
            for (int c = 0; c < 4; ++c) {
                tf[1 * 4 + c] = -tf[1 * 4 + c];  // negate row 1
                tf[2 * 4 + c] = -tf[2 * 4 + c];  // negate row 2
            }
        }
    }

    std::printf("[Scene] Rotated geometry 180 deg around X axis\n");
}

inline void Scene::build_emissive_distribution() {
    emissive_tri_indices.clear();
    std::vector<float> power_weights;

    for (uint32_t i = 0; i < (uint32_t)triangles.size(); ++i) {
        const auto& tri = triangles[i];
        const auto& mat = materials[tri.material_id];
        if (mat.is_emissive()) {
            float a = tri.area();
            float w = a * mat.mean_emission();
            emissive_tri_indices.push_back(i);
            power_weights.push_back(w);
        }
    }

    if (!power_weights.empty()) {
        emissive_alias_table = AliasTable::build(power_weights);
        total_emissive_power = emissive_alias_table.total_weight;

    }
}

inline void Scene::compute_emissive_radiance_range(float& out_min_Le,
                                                    float& out_max_Le) const {
    out_min_Le = 0.f;
    out_max_Le = 0.f;
    if (emissive_tri_indices.empty()) return;

    float lo = FLT_MAX;
    float hi = 0.f;
    for (uint32_t idx : emissive_tri_indices) {
        const auto& mat = materials[triangles[idx].material_id];
        float le = mat.mean_emission();
        if (le > 0.f) {
            lo = std::min(lo, le);
            hi = std::max(hi, le);
        }
    }
    if (lo > hi) lo = hi;  // single-value case
    out_min_Le = lo;
    out_max_Le = hi;
    std::printf("[Scene] Emissive radiance range: min=%.4f  max=%.4f  (ratio=%.1fx)\n",
                lo, hi, (lo > 0.f) ? hi / lo : 0.f);
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
