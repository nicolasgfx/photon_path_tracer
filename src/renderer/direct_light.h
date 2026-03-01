#pragma once
// ─────────────────────────────────────────────────────────────────────
// direct_light.h – Single-sample NEE direct light sampling (v3)
// ─────────────────────────────────────────────────────────────────────
// v3 simplification: pure power-weighted alias table sampling.
// Coverage-aware mixture removed (Part 2 §1.1, CL-08).
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"
#include "core/random.h"
#include "core/config.h"
#include "scene/scene.h"
#include "bsdf/bsdf.h"
#include "renderer/nee_shared.h"

struct DirectLightSample {
    Spectrum Li;           // Incident radiance (spectral)
    float3   wi;           // Direction to light (world)
    float    pdf_light;    // PDF of light sampling (area measure → solid angle)
    float    distance;     // Distance to light point
    bool     visible;      // Shadow ray result
};

// ── Sample direct illumination (pure power-weighted, v3) ─────────────
inline DirectLightSample sample_direct_light(
    float3 hit_pos,
    float3 hit_normal,
    const Scene& scene,
    PCGRng& rng,
    float /*coverage_fraction*/ = 0.f)  // kept for API compat, ignored
{
    DirectLightSample result;
    result.Li      = Spectrum::zero();
    result.visible = false;

    if (scene.emissive_tri_indices.empty()) return result;

    // ── Select emissive triangle via power-weighted alias table ──────
    float u1 = rng.next_float();
    float u2 = rng.next_float();
    int local_idx = scene.emissive_alias_table.sample(u1, u2);
    float pdf_tri = scene.emissive_alias_table.pdf(local_idx);

    uint32_t tri_idx = scene.emissive_tri_indices[local_idx];
    const Triangle& light_tri = scene.triangles[tri_idx];
    const Material& light_mat = scene.materials[light_tri.material_id];

    // 2. Sample point on the light triangle
    float3 bary = sample_triangle(rng.next_float(), rng.next_float());
    float3 light_pos = light_tri.interpolate_position(bary.x, bary.y, bary.z);
    float3 light_normal = light_tri.geometric_normal();
    float  light_area = light_tri.area();

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

    // 4. Convert area PDF to solid angle PDF (shared helper)
    float pdf_solid_angle = nee_pdf_area_to_solid_angle(
        pdf_tri, 1.0f / light_area, dist2, cos_theta_emitter);

    // 5. Visibility test (shadow ray)
    Ray shadow_ray;
    shadow_ray.origin    = nee_shadow_ray_origin(hit_pos, hit_normal);
    shadow_ray.direction = wi;
    shadow_ray.tmin      = NEE_RAY_EPSILON;
    shadow_ray.tmax      = nee_shadow_ray_tmax(dist);

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

// ── PDF of direct light sampling for a given direction (v3) ──────────
// Pure power-weighted PDF.
inline float direct_light_pdf(
    float3 hit_pos,
    float3 wi,
    const Scene& scene,
    float /*coverage_fraction*/ = 0.f)  // kept for API compat, ignored
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

    // Find power-weighted PDF for this emissive triangle
    float p_power = 0.f;
    for (size_t i = 0; i < scene.emissive_tri_indices.size(); ++i) {
        if (scene.emissive_tri_indices[i] == hit.triangle_id) {
            p_power = scene.emissive_alias_table.pdf((int)i);
            break;
        }
    }

    float dist2 = hit.t * hit.t;
    return nee_pdf_area_to_solid_angle(p_power, 1.0f / area, dist2, cos_theta);
}
