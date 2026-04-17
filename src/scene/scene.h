#pragma once
// ─────────────────────────────────────────────────────────────────────
// scene.h – Scene representation (v5 RGB)
//
// Ported from v4: Spectrum→Color3.  Identical layout/methods otherwise.
// ─────────────────────────────────────────────────────────────────────
#include "scene/triangle.h"
#include "scene/material.h"
#include "scene/texture.h"
#include "scene/medium.h"
#include "core/config.h"
#include "core/alias_table.h"
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <cfloat>
#include <cstring>

// ── Instancing descriptors ───────────────────────────────────────────
struct MeshDescriptor {
    uint32_t tri_offset;   // start index in scene.triangles[]
    uint32_t tri_count;
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
    std::vector<HomogeneousMedium> media;

    // Instancing
    std::vector<MeshDescriptor>     meshes;
    std::vector<InstanceDescriptor>  instances;
    bool has_instances() const { return instances.size() > 1; }

    // Emissive triangle distribution (power-weighted alias table)
    std::vector<uint32_t>  emissive_tri_indices;
    AliasTable             emissive_alias_table;
    float                  total_emissive_power = 0.f;

    AABB                   scene_bounds;

    // Normalization transform (applied by normalize_to_reference())
    float3 norm_center = {0, 0, 0};
    float  norm_scale  = 1.0f;

    // Scene loader metadata
    bool        has_portals     = false;

    // Loaded scene camera (set by the scene file loader)
    bool   scene_cam_valid    = false;
    float3 scene_cam_position = {0, 0, 0};
    float3 scene_cam_look_at  = {0, 0, -1};
    float3 scene_cam_up       = {0, 1, 0};
    float  scene_cam_fov      = 90.0f;
    bool   scene_cam_flip_x   = false;

    // ── Methods (implemented inline below) ──────────────────────────
    void compute_bounds();
    void build_emissive_distribution();
    void compute_emissive_radiance_range(float& out_min_Le, float& out_max_Le) const;
    void normalize_to_reference();
    void rotate_x_180();
    size_t num_triangles()  const { return triangles.size(); }
    size_t num_materials()  const { return materials.size(); }
    size_t num_emissive()   const { return emissive_tri_indices.size(); }
    float3 scene_bounding_center() const;
    float  scene_bounding_radius() const;
};

// ── Inline implementations ──────────────────────────────────────────

inline float3 Scene::scene_bounding_center() const {
    return scene_bounds.center();
}

inline float Scene::scene_bounding_radius() const {
    float3 ext = scene_bounds.extent();
    return length(ext) * 0.5f * 1.01f;
}

inline void Scene::compute_bounds() {
    scene_bounds = AABB::empty();
    for (const auto& t : triangles) {
        scene_bounds.expand(t.v0);
        scene_bounds.expand(t.v1);
        scene_bounds.expand(t.v2);
    }
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

    if (!emissive_tri_indices.empty()) {
        std::unordered_map<uint32_t, int> emit_mat_count;
        for (uint32_t idx : emissive_tri_indices)
            ++emit_mat_count[triangles[idx].material_id];
        std::printf("[Scene] Emissive distribution: %zu triangles, %zu distinct materials\n",
                    emissive_tri_indices.size(), emit_mat_count.size());
        for (auto& [mid, cnt] : emit_mat_count) {
            const auto& m = materials[mid];
            std::printf("  mat #%u '%s'  Le_max=%.4f  tris=%d\n",
                        mid, m.name.c_str(), m.Le.max_component(), cnt);
        }

    } else {
        std::printf("[Scene] Emissive distribution: 0 emissive triangles found\n");
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
    if (lo > hi) lo = hi;
    out_min_Le = lo;
    out_max_Le = hi;
    std::printf("[Scene] Emissive radiance range: min=%.4f  max=%.4f  (ratio=%.1fx)\n",
                lo, hi, (lo > 0.f) ? hi / lo : 0.f);
}

inline void Scene::normalize_to_reference() {
    if (triangles.empty()) return;

    AABB bb = AABB::empty();
    for (const auto& t : triangles) {
        bb.expand(t.v0);
        bb.expand(t.v1);
        bb.expand(t.v2);
    }

    float3 cur_center = bb.center();
    float3 ext        = bb.extent();
    float  longest    = fmaxf(fmaxf(ext.x, ext.y), ext.z);

    if (longest < 1e-12f) return;

    float scale = 1.0f / longest;
    norm_center = cur_center;
    norm_scale  = scale;

    std::cout << "[Scene] Normalising: centre ("
              << cur_center.x << ", " << cur_center.y << ", "
              << cur_center.z << ")  extent ("
              << ext.x << ", " << ext.y << ", " << ext.z
              << ")  scale " << scale << "\n";

    // Transform loaded scene camera
    if (scene_cam_valid) {
        scene_cam_position = (scene_cam_position - cur_center) * scale;
        scene_cam_look_at  = (scene_cam_look_at  - cur_center) * scale;
    }

    // Scale media coefficients (preserve optical depth)
    {
        float inv_scale = 1.0f / scale;
        for (auto& m : media) {
            m.sigma_a = m.sigma_a * inv_scale;
            m.sigma_s = m.sigma_s * inv_scale;
            m.sigma_t = m.sigma_t * inv_scale;
        }
    }

    // Transform world-space vertices
    uint32_t world_tri_end = 0;
    if (!meshes.empty())
        world_tri_end = meshes[0].tri_offset + meshes[0].tri_count;
    else
        world_tri_end = (uint32_t)triangles.size();

    for (uint32_t i = 0; i < world_tri_end; ++i) {
        auto& t = triangles[i];
        t.v0 = (t.v0 - cur_center) * scale;
        t.v1 = (t.v1 - cur_center) * scale;
        t.v2 = (t.v2 - cur_center) * scale;
    }

    // Update instance transforms
    if (has_instances()) {
        float cx = cur_center.x, cy = cur_center.y, cz = cur_center.z;
        float s  = scale;
        for (size_t i = 1; i < instances.size(); ++i) {
            float old[12];
            std::memcpy(old, instances[i].transform, sizeof(old));

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

    uint32_t world_tri_end = 0;
    if (!meshes.empty())
        world_tri_end = meshes[0].tri_offset + meshes[0].tri_count;
    else
        world_tri_end = (uint32_t)triangles.size();

    for (uint32_t i = 0; i < world_tri_end; ++i) {
        auto& t = triangles[i];
        t.v0.y = -t.v0.y;  t.v0.z = -t.v0.z;
        t.v1.y = -t.v1.y;  t.v1.z = -t.v1.z;
        t.v2.y = -t.v2.y;  t.v2.z = -t.v2.z;
        t.n0.y = -t.n0.y;  t.n0.z = -t.n0.z;
        t.n1.y = -t.n1.y;  t.n1.z = -t.n1.z;
        t.n2.y = -t.n2.y;  t.n2.z = -t.n2.z;
    }

    if (has_instances()) {
        for (size_t i = 1; i < instances.size(); ++i) {
            float* tf = instances[i].transform;
            for (int c = 0; c < 4; ++c) {
                tf[1 * 4 + c] = -tf[1 * 4 + c];
                tf[2 * 4 + c] = -tf[2 * 4 + c];
            }
        }
    }

    std::printf("[Scene] Rotated geometry 180 deg around X axis\n");
}
