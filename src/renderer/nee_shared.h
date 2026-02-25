#pragma once
// ─────────────────────────────────────────────────────────────────────
// nee_shared.h – Shared NEE math helpers for CPU↔GPU consistency
// ─────────────────────────────────────────────────────────────────────
// v2.2 consistency reset: CPU sample_direct_light() and GPU
// dev_nee_direct() must both use these exact helpers for:
//   - Triangle point sampling
//   - Area-to-solid-angle PDF conversion
//   - Balance heuristic MIS weights
//   - Backfacing/validity checks
//
// Canonical NEE v2.2 algorithm:
//   1. Select emissive triangle via mixture:
//        p_tri = (1-c)*p_power + c*p_area
//   2. Sample point on triangle uniformly (barycentric):
//        p_A = 1/Area(tri)
//   3. Convert to solid angle at shading point:
//        p_ω = p_A · d² / |n_L · (-ω_i)|
//   4. Cast shadow ray, evaluate BSDF, use that p_ω.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"

// ── PDF conversion: area measure → solid angle measure ──────────────
// This is the single most important NEE formula.  Both CPU and GPU
// must use this exact computation.
//
// p_ω = p_tri · p_pos · d² / |n_L · (-ω_i)|
//
// where:
//   p_tri = probability of selecting this triangle
//   p_pos = 1/area (uniform point on triangle)
//   d     = distance from shading point to light point
//   n_L   = geometric normal of the light triangle
//   ω_i   = direction from shading point to light (normalized)
//
// Returns 0 if the light is backfacing (cos_emitter <= 0).
inline HD float nee_pdf_area_to_solid_angle(
    float pdf_tri,        // Triangle selection probability
    float pdf_pos,        // Point sampling PDF = 1 / area
    float dist_squared,   // ||light_pos - hit_pos||²
    float cos_emitter)    // |dot(n_L, -ω_i)| — MUST be positive
{
    if (cos_emitter <= 0.f || pdf_pos <= 0.f) return 0.f;
    return pdf_tri * pdf_pos * dist_squared / cos_emitter;
}

// ── Coverage-aware mixture triangle selection PDF ───────────────────
// p = (1-c)*p_power + c*p_area
inline HD float nee_mixture_pdf(
    float p_power,            // Power-weighted alias table PDF
    float p_area,             // Area-weighted alias table PDF
    float coverage_fraction)  // Mixture weight [0,1]
{
    return (1.f - coverage_fraction) * p_power
         +        coverage_fraction  * p_area;
}

// ── Balance heuristic MIS weight (2-way) ────────────────────────────
// w_a = p_a² / (p_a² + p_b²)
inline HD float nee_mis_weight_2(float pdf_a, float pdf_b) {
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    return a2 / fmaxf(a2 + b2, 1e-30f);
}

// ── Balance heuristic MIS weight (3-way) ────────────────────────────
inline HD float nee_mis_weight_3(float pdf_a, float pdf_b, float pdf_c) {
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    float c2 = pdf_c * pdf_c;
    return a2 / fmaxf(a2 + b2 + c2, 1e-30f);
}

// ── Geometry validity check ─────────────────────────────────────────
// Returns true if the light sample is valid (both sides facing correctly).
// Uses GEOMETRIC normals for correctness.
struct NEEGeometry {
    float3 wi;            // Direction from shading point to light (normalized)
    float  distance;      // Distance to light point
    float  dist_squared;  // distance²
    float  cos_receiver;  // dot(wi, receiver_normal) — must be > 0
    float  cos_emitter;   // dot(-wi, light_normal)  — must be > 0
    bool   valid;         // Both cosines positive
};

inline HD NEEGeometry nee_compute_geometry(
    float3 hit_pos,        // Shading point
    float3 hit_normal,     // Geometric normal at shading point
    float3 light_pos,      // Sampled point on light
    float3 light_normal)   // Geometric normal of light triangle
{
    NEEGeometry g;
    float3 to_light = light_pos - hit_pos;
    g.dist_squared  = dot(to_light, to_light);
    g.distance      = sqrtf(g.dist_squared);
    g.wi            = to_light / fmaxf(g.distance, 1e-20f);
    g.cos_receiver  = dot(g.wi, hit_normal);
    g.cos_emitter   = dot(g.wi * (-1.f), light_normal);
    g.valid         = (g.cos_receiver > 0.f && g.cos_emitter > 0.f);
    return g;
}

// ── Shadow ray origin offset ────────────────────────────────────────
// Both CPU and GPU must use the same epsilon strategy for shadow rays.
constexpr float NEE_RAY_EPSILON = 1e-4f;

inline HD float3 nee_shadow_ray_origin(float3 hit_pos, float3 hit_normal) {
    return hit_pos + hit_normal * NEE_RAY_EPSILON;
}

inline HD float nee_shadow_ray_tmax(float distance) {
    return distance - 2.f * NEE_RAY_EPSILON;
}
