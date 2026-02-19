#pragma once
// ─────────────────────────────────────────────────────────────────────
// direct_light.h – Next-event estimation (direct light sampling)
// ─────────────────────────────────────────────────────────────────────
// Implements Section 7.2: always-on direct light sampling.
// Samples a point on an emissive surface and computes the direct
// illumination contribution with visibility testing.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"
#include "core/random.h"
#include "scene/scene.h"
#include "bsdf/bsdf.h"

struct DirectLightSample {
    Spectrum Li;           // Incident radiance (spectral)
    float3   wi;           // Direction to light (world)
    float    pdf_light;    // PDF of light sampling (area measure → solid angle)
    float    distance;     // Distance to light point
    bool     visible;      // Shadow ray result
};

// ── Sample direct illumination at a surface point ───────────────────
inline DirectLightSample sample_direct_light(
    float3 hit_pos,
    float3 hit_normal,
    const Scene& scene,
    PCGRng& rng)
{
    DirectLightSample result;
    result.Li      = Spectrum::zero();
    result.visible = false;

    if (scene.emissive_tri_indices.empty()) return result;

    // 1. Sample emissive triangle via alias table
    float u1 = rng.next_float();
    float u2 = rng.next_float();
    int local_idx = scene.emissive_alias_table.sample(u1, u2);
    uint32_t tri_idx = scene.emissive_tri_indices[local_idx];
    const Triangle& light_tri = scene.triangles[tri_idx];
    const Material& light_mat = scene.materials[light_tri.material_id];

    float pdf_tri = scene.emissive_alias_table.pdf(local_idx);

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

    // Geometry checks
    float cos_theta_receiver = dot(wi, hit_normal);
    float cos_theta_emitter  = dot(wi * (-1.f), light_normal);

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

// ── PDF of direct light sampling for a given direction ──────────────
// (needed for MIS with BSDF sampling)
inline float direct_light_pdf(
    float3 hit_pos,
    float3 wi,
    const Scene& scene)
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

    // Find the local index in emissive_tri_indices
    float pdf_tri = 0.f;
    for (size_t i = 0; i < scene.emissive_tri_indices.size(); ++i) {
        if (scene.emissive_tri_indices[i] == hit.triangle_id) {
            pdf_tri = scene.emissive_alias_table.pdf((int)i);
            break;
        }
    }

    float pdf_pos = 1.0f / area;
    float dist2 = hit.t * hit.t;

    return pdf_tri * pdf_pos * dist2 / cos_theta;
}
