#pragma once
// ─────────────────────────────────────────────────────────────────────
// specular_target.h – Specular triangle set for targeted caustic emission
// ─────────────────────────────────────────────────────────────────────
#include "scene/scene.h"
#include "scene/material.h"
#include "core/alias_table.h"
#include "core/types.h"
#include "core/spectrum.h"
#include "core/random.h"
#include "core/config.h"
#include "core/nee_sampling.h"
#include <vector>
#include <cstdint>
#include <cmath>

struct SpecularTargetSet {
    bool valid = false;
    std::vector<uint32_t> specular_tri_indices;
    AliasTable            area_alias_table;
    std::vector<float>    tri_areas;

    /// Build the target set by scanning the scene for specular triangles.
    static SpecularTargetSet build(const Scene& scene) {
        SpecularTargetSet s;

        // Collect triangles whose material is specular (glass/mirror)
        for (size_t i = 0; i < scene.triangles.size(); ++i) {
            uint32_t mat_id = scene.triangles[i].material_id;
            if (mat_id >= scene.materials.size()) continue;
            const Material& mat = scene.materials[mat_id];
            if (mat.type == MaterialType::Glass ||
                mat.type == MaterialType::Mirror) {
                s.specular_tri_indices.push_back((uint32_t)i);
            }
        }

        if (s.specular_tri_indices.empty()) {
            s.valid = false;
            return s;
        }

        // Compute per-triangle areas
        s.tri_areas.resize(s.specular_tri_indices.size());
        std::vector<float> weights(s.specular_tri_indices.size());
        for (size_t k = 0; k < s.specular_tri_indices.size(); ++k) {
            const Triangle& tri = scene.triangles[s.specular_tri_indices[k]];
            float3 e1 = tri.v1 - tri.v0;
            float3 e2 = tri.v2 - tri.v0;
            float3 c  = cross(e1, e2);
            float area = 0.5f * std::sqrt(c.x*c.x + c.y*c.y + c.z*c.z);
            s.tri_areas[k] = area;
            weights[k]     = area;
        }

        // Build alias table from area weights
        s.area_alias_table = AliasTable::build(weights);
        s.valid = true;
        return s;
    }
};

// ── Targeted caustic photon (returned by sample_targeted_caustic_photon) ──
struct TargetedCausticPhoton {
    bool     valid = false;
    Ray      ray;
    Spectrum spectral_flux;
    uint16_t source_emissive_idx = 0xFFFF;
};

/// Sample a photon aimed at a specular triangle from the nearest emitter.
/// 1. Pick a specular triangle (area-weighted alias table).
/// 2. Sample a point on the specular triangle.
/// 3. Pick the nearest (or power-weighted) emitter and generate direction.
/// 4. Visibility check from emitter → specular point.
/// Returns an invalid photon if visibility fails.
inline TargetedCausticPhoton sample_targeted_caustic_photon(
    const Scene& scene,
    const SpecularTargetSet& target_set,
    PCGRng& rng)
{
    TargetedCausticPhoton tcp;
    tcp.valid = false;

    if (!target_set.valid || target_set.specular_tri_indices.empty())
        return tcp;
    if (scene.emissive_tri_indices.empty())
        return tcp;

    // 1. Pick a specular triangle (area-weighted)
    int spec_local = target_set.area_alias_table.sample(
        rng.next_float(), rng.next_float());
    if (spec_local < 0 || spec_local >= (int)target_set.specular_tri_indices.size())
        return tcp;

    uint32_t spec_tri_idx = target_set.specular_tri_indices[spec_local];
    const Triangle& spec_tri = scene.triangles[spec_tri_idx];

    // 2. Sample a point on the specular triangle (uniform barycentric)
    float u1 = rng.next_float();
    float u2 = rng.next_float();
    float su = std::sqrt(u1);
    float b0 = 1.f - su;
    float b1 = u2 * su;
    float3 target_pt = spec_tri.v0 * b0 + spec_tri.v1 * b1
                     + spec_tri.v2 * (1.f - b0 - b1);

    // 3. Pick an emitter and create a photon aimed at the specular point
    //    Use the emitter point set if available, otherwise pick from CDF
    if (!scene.emitter_points.points.empty()) {
        int ept_idx = (int)(rng.next_float() * (float)scene.emitter_points.points.size());
        if (ept_idx >= (int)scene.emitter_points.points.size())
            ept_idx = (int)scene.emitter_points.points.size() - 1;
        const auto& ept = scene.emitter_points.points[ept_idx];

        float3 origin = ept.position;
        float3 dir    = normalize(target_pt - origin);
        float  dist   = length(target_pt - origin);

        // Check that direction faces outward from emitter
        if (dot(dir, ept.normal) <= 0.f) return tcp;

        // Visibility check (shadow ray)
        Ray shadow_ray;
        shadow_ray.origin    = origin + ept.normal * EPSILON;
        shadow_ray.direction = dir;
        shadow_ray.tmin      = EPSILON;
        shadow_ray.tmax      = dist - EPSILON;

        HitRecord shadow_hit = scene.intersect(shadow_ray);
        if (shadow_hit.hit) return tcp;  // occluded

        // Compute spectral flux
        uint32_t emitter_tri_idx = ept.global_tri_idx;
        const Triangle& emitter_tri = scene.triangles[emitter_tri_idx];
        const Material& emitter_mat = scene.materials[emitter_tri.material_id];

        float cos_theta = dot(dir, ept.normal);
        float emitter_area = emitter_tri.area();

        // flux = Le × area × cos(θ) / pdf_targeted
        //      approximate: scale Le by geometric factors
        float geom = cos_theta * emitter_area;
        for (int b = 0; b < NUM_LAMBDA; ++b)
            tcp.spectral_flux.value[b] = emitter_mat.Le.value[b] * geom;

        tcp.ray.origin    = origin + ept.normal * EPSILON;
        tcp.ray.direction = dir;
        tcp.ray.tmin      = EPSILON;
        tcp.ray.tmax      = 1e20f;
        tcp.source_emissive_idx = ept.emissive_local_idx;
        tcp.valid = true;
    } else {
        // Fallback: pick an emissive triangle from the CDF
        int emissive_idx = 0;
        for (int i = 0; i < (int)scene.emissive_tri_indices.size(); ++i) {
            // Simple linear search (could use binary search on CDF)
            emissive_idx = i;
            break;
        }
        uint32_t eidx = scene.emissive_tri_indices[emissive_idx];
        const Triangle& etri = scene.triangles[eidx];
        const Material& emat = scene.materials[etri.material_id];

        // Sample point on emitter
        float eu1 = rng.next_float();
        float eu2 = rng.next_float();
        float esu = std::sqrt(eu1);
        float eb0 = 1.f - esu;
        float eb1 = eu2 * esu;
        float3 origin = etri.v0 * eb0 + etri.v1 * eb1
                      + etri.v2 * (1.f - eb0 - eb1);
        float3 enrm = etri.geometric_normal();

        float3 dir  = normalize(target_pt - origin);
        float  dist = length(target_pt - origin);

        if (dot(dir, enrm) <= 0.f) return tcp;

        Ray shadow_ray;
        shadow_ray.origin    = origin + enrm * EPSILON;
        shadow_ray.direction = dir;
        shadow_ray.tmin      = EPSILON;
        shadow_ray.tmax      = dist - EPSILON;

        HitRecord shadow_hit = scene.intersect(shadow_ray);
        if (shadow_hit.hit) return tcp;

        float cos_theta = dot(dir, enrm);
        float emitter_area = etri.area();
        float geom = cos_theta * emitter_area;
        for (int b = 0; b < NUM_LAMBDA; ++b)
            tcp.spectral_flux.value[b] = emat.Le.value[b] * geom;

        tcp.ray.origin    = origin + enrm * EPSILON;
        tcp.ray.direction = dir;
        tcp.ray.tmin      = EPSILON;
        tcp.ray.tmax      = 1e20f;
        tcp.source_emissive_idx = (uint16_t)emissive_idx;
        tcp.valid = true;
    }

    return tcp;
}
