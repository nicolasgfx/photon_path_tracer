// ─────────────────────────────────────────────────────────────────────
// optix_programs.cu – OptiX device programs (v5 RGB pipeline)
//
// Phase 3: core hit/miss programs for ray-scene intersection.
// Later stages add raygen programs (path tracer, photon tracer, etc.)
// [rebuild trigger: nearest-boundary blend]
// by #including their .cuh files into this compilation unit.
//
// Payload layout (18 values, p0–p17):
//   p0-p2  : hit position (float3)
//   p3-p5  : shading normal (float3)
//   p6     : hit distance t (float)
//   p7     : material ID (uint32_t)
//   p8     : triangle ID (uint32_t)
//   p9     : hit flag (0 = miss, 1 = hit)
//   p10-p12: geometric normal (float3)
//   p13-p14: texture UV (float2)
//   p15-p17: tangent (float3)
// ─────────────────────────────────────────────────────────────────────
#include <optix.h>
#include <cstdint>
#include "accel/launch_params.h"
#include "core/types.h"
#include "core/random.h"
#include "core/hash.h"

extern "C" {
    __constant__ LaunchParams params;
}

// ── Include device helpers ───────────────────────────────────────────
#include "accel/optix_utils.cuh"
#include "integrator/optix_nee.cuh"
#include "integrator/path_tracer.cuh"
#include "integrator/caustic_tracer.cuh"

// =====================================================================
// TEA hash for cheap per-intersection RNG (stochastic opacity)
// =====================================================================
__forceinline__ __device__
uint32_t tea4(uint32_t v0, uint32_t v1) {
    uint32_t s0 = 0;
    for (int n = 0; n < 4; ++n) {
        s0 += 0x9e3779b9u;
        v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4u);
        v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761eu);
    }
    return v0;
}

// =====================================================================
// __anyhit__radiance – stochastic alpha / opacity test
// =====================================================================
extern "C" __global__ void __anyhit__radiance() {
    const int      prim_idx = optixGetPrimitiveIndex();
    const uint32_t mat_id   = params.material_ids[prim_idx];

    // Per-material scalar opacity
    if (!params.opacity) return;
    const float opac = params.opacity[mat_id];

    if (opac >= 1.f) return;        // fully opaque
    if (opac <= 0.f) { optixIgnoreIntersection(); return; }

    // Cheap per-intersection random (TEA hash)
    const uint3    idx        = optixGetLaunchIndex();
    const uint32_t pixel_seed = idx.x + idx.y * 65537u;
    const uint32_t h  = tea4(pixel_seed, prim_idx ^ __float_as_uint(optixGetRayTmax()));
    const float    xi = (float)(h & 0x00FFFFFFu) / (float)0x01000000u;

    if (xi >= opac) optixIgnoreIntersection();
}

// =====================================================================
// __anyhit__shadow – delta-opaque + stochastic opacity
// =====================================================================
extern "C" __global__ void __anyhit__shadow() {
    const int      prim_idx = optixGetPrimitiveIndex();
    const uint32_t mat_id   = params.material_ids[prim_idx];

    // Delta surfaces (glass/translucent) block NEE shadow rays.
    // Light through dielectrics is handled by BSDF continuation
    // (delta bounce → emission at full weight), avoiding double-counting.

    // DiffuseTransmission is a diffuse scatterer — transmitted light is
    // cosine-scattered into the opposite hemisphere, NOT passed straight
    // through.  Shadow rays must be blocked; the transmission lobe is
    // already accounted for by BSDF continuation (multi-bounce).

    // Per-material scalar opacity
    if (!params.opacity) return;
    const float opac = params.opacity[mat_id];

    if (opac >= 1.f) return;
    if (opac <= 0.f) { optixIgnoreIntersection(); return; }

    const uint3    idx        = optixGetLaunchIndex();
    const uint32_t pixel_seed = idx.x + idx.y * 65537u;
    const uint32_t h  = tea4(pixel_seed, prim_idx ^ __float_as_uint(optixGetRayTmax()));
    const float    xi = (float)(h & 0x00FFFFFFu) / (float)0x01000000u;

    if (xi >= opac) optixIgnoreIntersection();
}

// =====================================================================
// __closesthit__radiance
// =====================================================================
extern "C" __global__ void __closesthit__radiance() {
    const int    prim_idx = optixGetPrimitiveIndex();
    const float2 bary     = optixGetTriangleBarycentrics();

    float alpha = 1.f - bary.x - bary.y;
    float beta  = bary.x;
    float gamma = bary.y;

    // Interpolate position
    float3 v0 = params.vertices[prim_idx * 3 + 0];
    float3 v1 = params.vertices[prim_idx * 3 + 1];
    float3 v2 = params.vertices[prim_idx * 3 + 2];
    float3 pos = v0 * alpha + v1 * beta + v2 * gamma;

    // Geometric (face) normal
    float3 geo_normal = normalize(cross(v1 - v0, v2 - v0));

    // Interpolate shading normal
    float3 n0 = params.normals[prim_idx * 3 + 0];
    float3 n1 = params.normals[prim_idx * 3 + 1];
    float3 n2 = params.normals[prim_idx * 3 + 2];
    float3 shading_normal = normalize(n0 * alpha + n1 * beta + n2 * gamma);

    // Instance transform (IAS path)
    float3 tangent_w = make_f3(0.f, 0.f, 0.f);
    if (params.has_instances) {
        float xform[12];
        optixGetObjectToWorldTransformMatrix(xform);
        pos            = transform_point_3x4(xform, pos);
        geo_normal     = transform_normal_3x4(xform, geo_normal);
        shading_normal = transform_normal_3x4(xform, shading_normal);
        if (params.tangents) {
            float3 T_obj = params.tangents[prim_idx * 3];
            tangent_w = normalize(make_f3(
                xform[0]*T_obj.x + xform[1]*T_obj.y + xform[ 2]*T_obj.z,
                xform[4]*T_obj.x + xform[5]*T_obj.y + xform[ 6]*T_obj.z,
                xform[8]*T_obj.x + xform[9]*T_obj.y + xform[10]*T_obj.z));
        }
    } else {
        if (params.tangents)
            tangent_w = params.tangents[prim_idx * 3];
    }

    // Interpolate UVs
    float2 uv0 = params.texcoords[prim_idx * 3 + 0];
    float2 uv1 = params.texcoords[prim_idx * 3 + 1];
    float2 uv2 = params.texcoords[prim_idx * 3 + 2];
    float2 uv  = make_f2(
        uv0.x * alpha + uv1.x * beta + uv2.x * gamma,
        uv0.y * alpha + uv1.y * beta + uv2.y * gamma);

    float t = optixGetRayTmax();
    uint32_t mat_id = params.material_ids[prim_idx];

    // Pack payload
    optixSetPayload_0(__float_as_uint(pos.x));
    optixSetPayload_1(__float_as_uint(pos.y));
    optixSetPayload_2(__float_as_uint(pos.z));
    optixSetPayload_3(__float_as_uint(shading_normal.x));
    optixSetPayload_4(__float_as_uint(shading_normal.y));
    optixSetPayload_5(__float_as_uint(shading_normal.z));
    optixSetPayload_6(__float_as_uint(t));
    optixSetPayload_7(mat_id);
    optixSetPayload_8((uint32_t)prim_idx);
    optixSetPayload_9(1u);   // hit = true
    optixSetPayload_10(__float_as_uint(geo_normal.x));
    optixSetPayload_11(__float_as_uint(geo_normal.y));
    optixSetPayload_12(__float_as_uint(geo_normal.z));
    optixSetPayload_13(__float_as_uint(uv.x));
    optixSetPayload_14(__float_as_uint(uv.y));
    optixSetPayload_15(__float_as_uint(tangent_w.x));
    optixSetPayload_16(__float_as_uint(tangent_w.y));
    optixSetPayload_17(__float_as_uint(tangent_w.z));
}

// =====================================================================
// __closesthit__shadow
// =====================================================================
extern "C" __global__ void __closesthit__shadow() {
    optixSetPayload_0(1u); // 1 = occluded
}

// =====================================================================
// __miss__radiance
// =====================================================================
extern "C" __global__ void __miss__radiance() {
    optixSetPayload_9(0u); // hit = false
}

// =====================================================================
// __miss__shadow
// =====================================================================
extern "C" __global__ void __miss__shadow() {
    optixSetPayload_0(0u); // 0 = not occluded (visible)
}

// =====================================================================
// __raygen__test_normals – diagnostic raygen for acceleration tests
//
// Traces one primary ray per pixel and writes the absolute shading
// normal as RGB to the output buffer.  Miss pixels get black.
// =====================================================================
extern "C" __global__ void __raygen__test_normals() {
    const uint3 idx = optixGetLaunchIndex();
    int px = idx.x;
    int py = idx.y;
    int pixel_idx = py * params.width + px;

    // Generate a simple pinhole camera ray
    float u_ndc = ((float)px + 0.5f) / (float)params.width;
    float v_ndc = ((float)py + 0.5f) / (float)params.height;

    // Map NDC [0,1] to camera-space direction
    float3 direction = normalize(
        params.cam_w +
        params.cam_u * (2.f * u_ndc - 1.f) +
        params.cam_v * (2.f * v_ndc - 1.f));

    TraceResult hit = trace_radiance(params.cam_pos, direction);

    float3 color;
    if (hit.hit) {
        // Absolute value of shading normal as RGB
        color = make_f3(
            fabsf(hit.shading_normal.x),
            fabsf(hit.shading_normal.y),
            fabsf(hit.shading_normal.z));
    } else {
        color = make_f3(0.f, 0.f, 0.f);
    }

    params.color_r[pixel_idx] = color.x;
    params.color_g[pixel_idx] = color.y;
    params.color_b[pixel_idx] = color.z;
    params.sample_counts[pixel_idx] = 1.f;
}

// =====================================================================
// __raygen__test_nee – diagnostic raygen for NEE / direct lighting tests
//
// Traces a primary ray.  At the first non-emissive hit, performs
// N_SAMPLES NEE evaluations and accumulates the average direct
// illumination into the output buffer.  Emissive surfaces get their
// emission directly.
// =====================================================================
extern "C" __global__ void __raygen__test_nee() {
    const uint3 idx = optixGetLaunchIndex();
    int px = idx.x;
    int py = idx.y;
    int pixel_idx = py * params.width + px;

    float u_ndc = ((float)px + 0.5f) / (float)params.width;
    float v_ndc = ((float)py + 0.5f) / (float)params.height;

    float3 direction = normalize(
        params.cam_w +
        params.cam_u * (2.f * u_ndc - 1.f) +
        params.cam_v * (2.f * v_ndc - 1.f));

    TraceResult hit = trace_radiance(params.cam_pos, direction);

    float3 color = make_f3(0.f, 0.f, 0.f);

    if (!hit.hit) {
        // Miss: no contribution
    } else {
        uint32_t mat_id = hit.material_id;
        uint8_t mt = params.mat_type ? params.mat_type[mat_id] : 0;

        // If we hit an emissive surface, return its emission
        if (mt == Emissive) {
            float3 Le = dev_get_Le(mat_id, hit.uv);
            color = Le;
        } else {
            // Non-emissive hit: evaluate NEE direct lighting
            // Multiple samples for lower noise in tests
            constexpr int N_SAMPLES = 16;
            float3 accum = make_f3(0.f, 0.f, 0.f);

            PCGRng rng = PCGRng::seed(
                (uint64_t)params.frame_number * 65537ull + 42ull,
                (uint64_t)pixel_idx + 1ull);

            float3 wo_world = direction * (-1.f);
            ONB frame = orient_frame_to_outgoing(hit.shading_normal, wo_world);
            float3 wo_local = frame.world_to_local(wo_world);

            for (int s = 0; s < N_SAMPLES; ++s) {
                NeeResult nee = dev_nee_dispatch(
                    hit.position, hit.shading_normal, hit.geo_normal,
                    frame, wo_local, mat_id, hit.uv, rng);

                accum.x += nee.L.x;
                accum.y += nee.L.y;
                accum.z += nee.L.z;
            }

            float inv_n = 1.f / (float)N_SAMPLES;
            color = make_f3(accum.x * inv_n, accum.y * inv_n, accum.z * inv_n);
        }
    }

    params.color_r[pixel_idx] = color.x;
    params.color_g[pixel_idx] = color.y;
    params.color_b[pixel_idx] = color.z;
    params.sample_counts[pixel_idx] = 1.f;
}
