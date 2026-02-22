#pragma once
// ─────────────────────────────────────────────────────────────────────
// direct_light.h – Coverage-aware NEE (§7.2.1, v2.1)
// ─────────────────────────────────────────────────────────────────────
// Mixture sampling: p_select = (1-c) * p_power + c * p_area
//   c = DEFAULT_NEE_COVERAGE_FRACTION (default 0.3)
//   p_power = power-weighted alias table (existing)
//   p_area  = area-weighted (uniform over emissive surface area)
//
// This prevents dark spots under large dim emitters (coverage problem)
// while preserving low variance for small bright lights.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"
#include "core/random.h"
#include "core/config.h"
#include "scene/scene.h"
#include "bsdf/bsdf.h"

struct DirectLightSample {
    Spectrum Li;           // Incident radiance (spectral)
    float3   wi;           // Direction to light (world)
    float    pdf_light;    // PDF of light sampling (area measure → solid angle)
    float    distance;     // Distance to light point
    bool     visible;      // Shadow ray result
};

// ── Sample direct illumination (coverage-aware mixture, §7.2.1) ─────
//
// Mixture of power-weighted + area-weighted light sampling.
// coverage_fraction: 0 = pure importance, 1 = pure area, 0.3 = default
inline DirectLightSample sample_direct_light(
    float3 hit_pos,
    float3 hit_normal,
    const Scene& scene,
    PCGRng& rng,
    float coverage_fraction = DEFAULT_NEE_COVERAGE_FRACTION)
{
    DirectLightSample result;
    result.Li      = Spectrum::zero();
    result.visible = false;

    if (scene.emissive_tri_indices.empty()) return result;

    // ── Select emissive triangle via mixture ────────────────────────
    float u_select = rng.next_float();
    int local_idx;
    float pdf_tri;

    if (u_select > coverage_fraction) {
        // Power-weighted branch (probability 1-c)
        float u1 = rng.next_float();
        float u2 = rng.next_float();
        local_idx = scene.emissive_alias_table.sample(u1, u2);

        // Mixture PDF: p = (1-c)*p_power + c*p_area
        float p_power = scene.emissive_alias_table.pdf(local_idx);
        float p_area  = scene.emissive_area_alias_table.pdf(local_idx);
        pdf_tri = (1.0f - coverage_fraction) * p_power
                +          coverage_fraction  * p_area;
    } else {
        // Area-weighted branch (probability c): sample proportional to area
        float u1 = rng.next_float();
        float u2 = rng.next_float();
        local_idx = scene.emissive_area_alias_table.sample(u1, u2);

        float p_power = scene.emissive_alias_table.pdf(local_idx);
        float p_area  = scene.emissive_area_alias_table.pdf(local_idx);
        pdf_tri = (1.0f - coverage_fraction) * p_power
                +          coverage_fraction  * p_area;
    }

    uint32_t tri_idx = scene.emissive_tri_indices[local_idx];
    const Triangle& light_tri = scene.triangles[tri_idx];
    const Material& light_mat = scene.materials[light_tri.material_id];

    // 2. Sample point on the light triangle
    float3 bary = sample_triangle(rng.next_float(), rng.next_float());
    float3 light_pos = light_tri.interpolate_position(bary.x, bary.y, bary.z);
    float3 light_normal = light_tri.geometric_normal();
    float  light_area = light_tri.area();

    float pdf_pos = 1.0f / light_area;

    // 3. Compute direction and distance
    float3 to_light = light_pos - hit_pos;
    float  dist2    = dot(to_light, to_light);
    float  dist     = sqrtf(dist2);
    float3 wi       = to_light / dist;

    float cos_theta_emitter  = dot(wi * (-1.f), light_normal);

    // Geometry checks
    float cos_theta_receiver = dot(wi, hit_normal);
    if (cos_theta_receiver <= 0.f || cos_theta_emitter <= 0.f) {
        return result;
    }

    // 4. Convert area PDF to solid angle PDF
    // p_omega = p_area * dist^2 / cos_theta_emitter
    float pdf_solid_angle = pdf_tri * pdf_pos * dist2 / cos_theta_emitter;

    // 5. Visibility test (shadow ray)
    Ray shadow_ray;
    shadow_ray.origin    = hit_pos + hit_normal * EPSILON;
    shadow_ray.direction = wi;
    shadow_ray.tmin      = 1e-4f;
    shadow_ray.tmax      = dist - 2e-4f;

    HitRecord shadow_hit = scene.intersect(shadow_ray);
    if (shadow_hit.hit) {
        // Occluded
        return result;
    }

    // 6. Compute incident radiance
    result.Li        = light_mat.Le;
    result.wi        = wi;
    result.pdf_light = pdf_solid_angle;
    result.distance  = dist;
    result.visible   = true;

    return result;
}

// ── PDF of direct light sampling for a given direction (§7.2.1) ─────
// Uses coverage-aware mixture PDF: p = (1-c)*p_power + c*p_area
inline float direct_light_pdf(
    float3 hit_pos,
    float3 wi,
    const Scene& scene,
    float coverage_fraction = DEFAULT_NEE_COVERAGE_FRACTION)
{
    // Trace ray in direction wi and check if it hits an emissive surface
    Ray ray;
    ray.origin    = hit_pos;
    ray.direction = wi;

    HitRecord hit = scene.intersect(ray);
    if (!hit.hit) return 0.f;

    const Material& mat = scene.materials[hit.material_id];
    if (!mat.is_emissive()) return 0.f;

    // Find which emissive triangle was hit and compute light PDF
    float3 light_normal = hit.normal;
    float cos_theta = fabsf(dot(wi * (-1.f), light_normal));
    if (cos_theta <= 0.f) return 0.f;

    // Area of the hit triangle
    const Triangle& tri = scene.triangles[hit.triangle_id];
    float area = tri.area();

    // Find which emissive triangle was hit and compute mixture PDF
    float p_power = 0.f;
    float p_area  = 0.f;
    for (size_t i = 0; i < scene.emissive_tri_indices.size(); ++i) {
        if (scene.emissive_tri_indices[i] == hit.triangle_id) {
            p_power = scene.emissive_alias_table.pdf((int)i);
            p_area  = scene.emissive_area_alias_table.pdf((int)i);
            break;
        }
    }

    // Coverage-aware mixture PDF (§7.2.1)
    float pdf_tri_mix = (1.0f - coverage_fraction) * p_power
                      +          coverage_fraction  * p_area;

    float pdf_pos = 1.0f / area;
    float dist2 = hit.t * hit.t;

    return pdf_tri_mix * pdf_pos * dist2 / cos_theta;
}
