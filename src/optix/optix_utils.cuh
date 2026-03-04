#pragma once

// optix_utils.cuh – Float/uint helpers, TraceResult, ray trace wrappers

// == Float / uint reinterpret helpers =================================
__forceinline__ __device__ unsigned int f2u(float f) {
    return __float_as_uint(f);
}
__forceinline__ __device__ float u2f(unsigned int u) {
    return __uint_as_float(u);
}

// == Trace result struct ==============================================
struct TraceResult {
    float3   position;
    float3   shading_normal;
    float3   geo_normal;
    float2   uv;
    float    t;
    uint32_t material_id;
    uint32_t triangle_id;
    bool     hit;
};

// == Trace radiance ray and unpack payload ============================
__forceinline__ __device__
TraceResult trace_radiance(float3 origin, float3 direction,
                           float tmin = OPTIX_SCENE_EPSILON,
                           float tmax = DEFAULT_RAY_TMAX) {
    unsigned int p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14;
    p0=p1=p2=p3=p4=p5=p6=p7=p8=p9=p10=p11=p12=p13=p14=0;

    optixTrace(
        params.traversable,
        origin, direction,
        tmin, tmax, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,
        p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14);

    TraceResult r;
    r.position       = make_f3(u2f(p0), u2f(p1), u2f(p2));
    r.shading_normal = make_f3(u2f(p3), u2f(p4), u2f(p5));
    r.t              = u2f(p6);
    r.material_id    = p7;
    r.triangle_id    = p8;
    r.hit            = (p9 != 0);
    r.geo_normal     = make_f3(u2f(p10), u2f(p11), u2f(p12));
    r.uv             = make_f2(u2f(p13), u2f(p14));
    return r;
}

// == Trace shadow ray =================================================
__forceinline__ __device__
bool trace_shadow(float3 origin, float3 direction, float max_dist) {
    // Start occluded=1; closesthit is disabled so payload stays 1 on hit.
    // Only __miss__shadow sets it to 0 (visible).
    unsigned int occluded = 1;
    optixTrace(
        params.traversable,
        origin, direction,
        OPTIX_SCENE_EPSILON, max_dist - OPTIX_SCENE_EPSILON, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
        OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        1, 1, 1,
        occluded);
    return (occluded == 0); // 0 = visible
}

// == Shadow ray with material info (ray type 2) =======================
// Traces from origin towards a photon position and returns the material
// of the first intersected triangle.  The origin triangle is skipped
// via triangle-ID check in __anyhit__shadow_material.
struct ShadowMaterialResult {
    bool     hit;           // true = geometry found before target distance
    uint32_t material_id;   // valid only when hit == true
};

__forceinline__ __device__
ShadowMaterialResult trace_shadow_material(
    float3 origin, float3 direction, float max_dist,
    uint32_t origin_tri_id)
{
    // p0 = origin_tri_id (input: read by any-hit for self-intersection skip)
    //      overwritten to hit flag (output: 1 = hit, 0 = miss)
    // p1 = material_id   (output: written by closest-hit)
    unsigned int p0 = origin_tri_id;
    unsigned int p1 = 0;

    optixTrace(
        params.traversable,
        origin, direction,
        OPTIX_SCENE_EPSILON, max_dist, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        2, 1, 2,   // SBT offset=2 (shadow_material), stride=1, missSBTIndex=2
        p0, p1);

    ShadowMaterialResult r;
    r.hit         = (p0 != 0);
    r.material_id = p1;
    return r;
}
