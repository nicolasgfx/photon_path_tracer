#pragma once
// ─────────────────────────────────────────────────────────────────────
// nee_shared.h – Shared NEE math helpers for CPU↔GPU consistency (v5)
//
// PDF conversion (area→solid angle), geometry validity, MIS weights,
// shadow ray offset.  Both CPU and GPU must use these exact helpers.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"

// ── PDF conversion: area measure → solid angle measure ──────────────
//
// p_ω = p_tri · p_pos · d² / |n_L · (-ω_i)|
//
// where:
//   p_tri = probability of selecting this triangle (alias table)
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

// ── Power heuristic MIS weight (3-way) ──────────────────────────────
inline HD float nee_mis_weight_3(float pdf_a, float pdf_b, float pdf_c) {
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    float c2 = pdf_c * pdf_c;
    return a2 / fmaxf(a2 + b2 + c2, 1e-30f);
}

// ── Geometry validity check ─────────────────────────────────────────
struct NEEGeometry {
    float3 wi;            // Direction from shading point to light (normalized)
    float  distance;      // Distance to light point
    float  dist_squared;  // distance²
    float  cos_receiver;  // dot(wi, receiver_normal) — must be > 0
    float  cos_emitter;   // dot(-wi, light_normal)  — must be > 0
    bool   valid;         // Both cosines positive
};

inline HD NEEGeometry nee_compute_geometry(
    float3 hit_pos,
    float3 hit_normal,
    float3 light_pos,
    float3 light_normal)
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
constexpr float NEE_RAY_EPSILON = 1e-4f;

inline HD float3 nee_shadow_ray_origin(float3 hit_pos, float3 hit_normal) {
    return hit_pos + hit_normal * NEE_RAY_EPSILON;
}

inline HD float nee_shadow_ray_tmax(float distance) {
    return distance - 2.f * NEE_RAY_EPSILON;
}
