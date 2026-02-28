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
#include "renderer/nee_shared.h"
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
                mat.type == MaterialType::Mirror ||
                mat.type == MaterialType::Translucent) {
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

    // Compute target triangle normal for backface culling
    float3 spec_normal = spec_tri.geometric_normal();

    // 3. Pick an emitter and create a photon aimed at the specular point
    {
        // Power-weighted emitter selection via alias table (matches GPU).
        int emissive_idx = scene.emissive_alias_table.sample(
            rng.next_float(), rng.next_float());
        if (emissive_idx < 0 || emissive_idx >= (int)scene.emissive_tri_indices.size())
            emissive_idx = 0;
        float pdf_emitter = scene.emissive_alias_table.pdf(emissive_idx);
        if (pdf_emitter <= 0.f) return tcp;

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

        if (dot(dir, enrm) <= 0.f) return tcp;

        // Backface culling: reject if photon hits target from behind
        if (dot(dir, spec_normal) >= 0.f) return tcp;

        // Visibility: transparent-passthrough logic.
        {
            float dist = length(target_pt - origin);
            Ray vis_ray;
            vis_ray.origin    = origin + enrm * EPSILON;
            vis_ray.direction = dir;
            vis_ray.tmin      = EPSILON;
            vis_ray.tmax      = dist - EPSILON;
            HitRecord vis_hit = scene.intersect(vis_ray);
            if (vis_hit.hit) {
                uint32_t bmat = vis_hit.material_id;
                if (bmat < scene.materials.size()) {
                    const Material& blocker = scene.materials[bmat];
                    if (!blocker.is_specular()) return tcp;  // opaque blocker
                }
            }
        }

        float cos_theta = dot(dir, enrm);
        float emitter_area = etri.area();

        // Area-to-solid-angle PDF for the target direction (§3.4):
        //   p_ω = p_spec · (1/A_spec) · d² / cos_target
        float spec_pdf  = target_set.area_alias_table.pdf(spec_local);
        float spec_area = target_set.tri_areas[spec_local];
        float dist      = length(target_pt - origin);
        float dist_sq   = dist * dist;
        float cos_target = -dot(dir, spec_normal);  // photon approaches from front
        float pdf_target_sa = nee_pdf_area_to_solid_angle(
            spec_pdf, 1.f / fmaxf(spec_area, 1e-20f), dist_sq, cos_target);
        if (pdf_target_sa <= 0.f) return tcp;

        // Φ(λ) = Le(λ) · cos_light · A_light / (p_emitter · p_ω)
        float denom = pdf_emitter * pdf_target_sa;
        for (int b = 0; b < NUM_LAMBDA; ++b)
            tcp.spectral_flux.value[b] = fminf(
                emat.Le.value[b] * cos_theta * emitter_area / denom,
                1e6f);  // firefly clamp (matches GPU TC-11)

        tcp.ray.origin    = origin + enrm * EPSILON;
        tcp.ray.direction = dir;
        tcp.ray.tmin      = EPSILON;
        tcp.ray.tmax      = 1e20f;
        tcp.source_emissive_idx = (uint16_t)emissive_idx;
        tcp.valid = true;
    }

    return tcp;
}
