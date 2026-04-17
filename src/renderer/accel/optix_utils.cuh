#pragma once
// ─────────────────────────────────────────────────────────────────────
// optix_utils.cuh – Device-side ray trace helpers
//
// Float/uint reinterpret, instance transforms, TraceResult struct,
// and optixTrace wrappers for radiance and shadow rays.
// ─────────────────────────────────────────────────────────────────────
#include "accel/launch_params.h"

// Forward-declare the extern params (defined in optix_programs.cu)
extern "C" {
    extern __constant__ LaunchParams params;
}

// ── Float / uint reinterpret helpers ─────────────────────────────────
__forceinline__ __device__ unsigned int f2u(float f) {
    return __float_as_uint(f);
}
__forceinline__ __device__ float u2f(unsigned int u) {
    return __uint_as_float(u);
}

// ── Instance transform helpers (3×4 row-major) ──────────────────────
__forceinline__ __device__
float3 transform_point_3x4(const float m[12], float3 p) {
    return make_f3(
        m[0]*p.x + m[1]*p.y + m[ 2]*p.z + m[ 3],
        m[4]*p.x + m[5]*p.y + m[ 6]*p.z + m[ 7],
        m[8]*p.x + m[9]*p.y + m[10]*p.z + m[11]);
}

__forceinline__ __device__
float3 transform_normal_3x4(const float m[12], float3 n) {
    return normalize(make_f3(
        m[0]*n.x + m[1]*n.y + m[ 2]*n.z,
        m[4]*n.x + m[5]*n.y + m[ 6]*n.z,
        m[8]*n.x + m[9]*n.y + m[10]*n.z));
}

// ── Trace result struct ──────────────────────────────────────────────
struct TraceResult {
    float3   position;
    float3   shading_normal;
    float3   geo_normal;
    float3   tangent;
    float2   uv;
    float    t;
    uint32_t material_id;
    uint32_t triangle_id;
    bool     hit;
};

// ── Trace radiance ray and unpack payload ────────────────────────────
__forceinline__ __device__
TraceResult trace_radiance(float3 origin, float3 direction,
                           float tmin = 1e-4f,
                           float tmax = 1e20f) {
    unsigned int p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17;
    p0=p1=p2=p3=p4=p5=p6=p7=p8=p9=p10=p11=p12=p13=p14=p15=p16=p17=0;

    optixTrace(
        params.traversable,
        origin, direction,
        tmin, tmax, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,   // SBT offset=0 (radiance), stride=1, miss index=0
        p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17);

    TraceResult r;
    r.position       = make_f3(u2f(p0), u2f(p1), u2f(p2));
    r.shading_normal = make_f3(u2f(p3), u2f(p4), u2f(p5));
    r.t              = u2f(p6);
    r.material_id    = p7;
    r.triangle_id    = p8;
    r.hit            = (p9 != 0);
    r.geo_normal     = make_f3(u2f(p10), u2f(p11), u2f(p12));
    r.uv             = make_f2(u2f(p13), u2f(p14));
    r.tangent        = make_f3(u2f(p15), u2f(p16), u2f(p17));
    return r;
}

// ── Trace shadow ray (visibility test) ───────────────────────────────
// Returns true if the path is VISIBLE (not occluded).
__forceinline__ __device__
bool trace_shadow(float3 origin, float3 direction, float max_dist) {
    unsigned int occluded = 1;
    optixTrace(
        params.traversable,
        origin, direction,
        1e-4f, max_dist - 1e-4f, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
        OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        1, 1, 1,   // SBT offset=1 (shadow), stride=1, miss index=1
        occluded);
    return (occluded == 0); // 0 = visible
}
