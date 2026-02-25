#pragma once
// ─────────────────────────────────────────────────────────────────────
// surface_filter.h – Tangential distance metric & surface consistency (§6.3, §6.4)
// ─────────────────────────────────────────────────────────────────────
// Shared by CPU (KD-tree) and GPU (hash grid) gather paths.
//
// Root cause of planar blocking artifacts: using 3D spherical distance
// for surface irradiance estimation.  Photon mapping estimates irradiance
// ON SURFACES, not in volume.  The fix is the distance metric, not the
// data structure.
//
// This header provides:
//   1. tangential_distance2()  – 2D distance on the tangent plane
//   2. surface_consistency()   – 4-condition accept/reject filter
//   3. plane_distance()        – signed distance to the query tangent plane
//   4. effective_tau()         – robust τ with epsilon factor
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/config.h"

// ── Tangential (surface) distance computation (§6.3) ────────────────
//
// Given query point x with geometric normal n_x and photon position x_i:
//   v         = x_i - x
//   d_plane   = dot(n_x, v)            (signed distance along normal)
//   v_tan     = v - n_x * d_plane      (tangential component)
//   d_tan^2   = dot(v_tan, v_tan)      (tangential distance squared)
//
// d_tan replaces ||v|| (3D Euclidean) in all gather operations.

struct TangentialResult {
    float d_tan2;       // tangential distance squared
    float d_plane;      // signed plane distance (for filter)
    float3 v_tan;       // tangential vector (for debugging)
};

inline HD TangentialResult compute_tangential(
    float3 query_pos, float3 query_normal, float3 photon_pos)
{
    TangentialResult r;
    float3 v = photon_pos - query_pos;
    r.d_plane = dot(query_normal, v);
    r.v_tan   = v - query_normal * r.d_plane;
    r.d_tan2  = dot(r.v_tan, r.v_tan);
    return r;
}

// Convenience: tangential distance squared only
inline HD float tangential_distance2(
    float3 query_pos, float3 query_normal, float3 photon_pos)
{
    float3 v = photon_pos - query_pos;
    float d_plane = dot(query_normal, v);
    float3 v_tan = v - query_normal * d_plane;
    return dot(v_tan, v_tan);
}

// ── Plane distance (signed) ─────────────────────────────────────────
inline HD float plane_distance(
    float3 query_pos, float3 query_normal, float3 photon_pos)
{
    return dot(query_normal, photon_pos - query_pos);
}

// ── Robust τ computation (§6.3 config) ──────────────────────────────
//
// tau = max(user_tau, PLANE_TAU_EPSILON_FACTOR * ray_epsilon)
// Ensures τ is never smaller than the ray offset, which would reject
// photons deposited at the same surface due to floating-point offset.
inline HD float effective_tau(float user_tau, float ray_epsilon = EPSILON) {
    return fmaxf(user_tau, PLANE_TAU_EPSILON_FACTOR * ray_epsilon);
}

// ── Surface consistency filter (§6.4) ───────────────────────────────
//
// Reject photon i unless ALL conditions pass:
//   1. Tangential distance: d_tan^2 < r^2
//   2. Plane distance:      |d_plane| < τ
//   3. Normal compatibility: dot(n_photon, n_query) > 0
//   4. Direction consistency: dot(wi, n_query) > 0
//
// Uses geometric normals for stability (not shading normals, §15.1.2).

struct SurfaceFilterConfig {
    float radius2;      // r^2 — gather radius squared
    float tau;          // plane distance threshold
};

// Full filter: returns true if photon passes all 4 conditions
inline HD bool surface_consistency(
    float3 query_pos,
    float3 query_normal,    // geometric normal of query point
    float3 photon_pos,
    float3 photon_normal,   // geometric normal of photon deposit surface
    float3 photon_wi,       // incoming direction at photon deposit
    const SurfaceFilterConfig& cfg,
    float* out_d_tan2 = nullptr)
{
    // Compute tangential and plane distances
    TangentialResult tr = compute_tangential(query_pos, query_normal, photon_pos);

    if (out_d_tan2) *out_d_tan2 = tr.d_tan2;

    // Condition 1: tangential distance within radius
    if (tr.d_tan2 >= cfg.radius2)
        return false;

    // Condition 2: plane distance within τ
    if (fabsf(tr.d_plane) >= cfg.tau)
        return false;

    // Condition 3: normal compatibility (same surface orientation)
    if (dot(photon_normal, query_normal) <= 0.0f)
        return false;

    // Condition 4: photon incoming direction from correct hemisphere
    // (light arrives from the same side as the normal)
    if (dot(photon_wi, query_normal) <= 0.0f)
        return false;

    return true;
}

// Lightweight filter: conditions 1+2 only (for spatial queries that
// need fast pre-filtering before full BSDF evaluation)
inline HD bool surface_prefilter(
    float3 query_pos,
    float3 query_normal,
    float3 photon_pos,
    float radius2,
    float tau,
    float* out_d_tan2 = nullptr)
{
    TangentialResult tr = compute_tangential(query_pos, query_normal, photon_pos);
    if (out_d_tan2) *out_d_tan2 = tr.d_tan2;
    return (tr.d_tan2 < radius2) && (fabsf(tr.d_plane) < tau);
}


