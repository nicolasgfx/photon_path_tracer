// ---------------------------------------------------------------------
// optix_device.cu - All OptiX device programs (single compilation unit)
// ---------------------------------------------------------------------
//
// Programs:
//   __raygen__render                  - v3 first-hit guided brute-force PT
//   __raygen__photon_trace            - GPU photon emission + tracing
//   __raygen__targeted_photon_trace   - GPU targeted caustic emission (Jensen §9.2)
//   __raygen__direction_map           - Build direction map (per-subpixel)
//   __raygen__photon_gather           - Photon density estimation at first camera hit
//   __closesthit__radiance            - closest-hit for radiance rays
//   __closesthit__shadow    - closest-hit for shadow rays
//   __miss__radiance        - miss for radiance rays
//   __miss__shadow          - miss for shadow rays
//
// Payload layout (14 values):
//   p0-p2  : hit position (float3)
//   p3-p5  : shading normal (float3)
//   p6     : hit distance t (float)
//   p7     : material ID (uint32_t)
//   p8     : triangle ID (uint32_t)
//   p9     : hit flag (0 = miss, 1 = hit)
//   p10-p12: geometric normal (float3)
//   p13    : (reserved)
// ---------------------------------------------------------------------
#include <optix.h>
#include <cstdint>
#include "optix/launch_params.h"
#include "core/random.h"
#include "core/spectrum.h"
#include "core/config.h"
#include "core/ior_stack.h"
#include "volume/medium.h"
#include "bsdf/bsdf_shared.h"
#include "core/material_flags.h"
#include "core/hash.h"
#include "renderer/camera.h"
#include "renderer/nee_shared.h"

// ---------------------------------------------------------------------
// Module-local implementation constants
// (kept out of core/config.h to avoid clutter)
// ---------------------------------------------------------------------
namespace {
    // Ray epsilon to avoid self-intersections.
    constexpr float OPTIX_SCENE_EPSILON = 1e-4f;

    // Large tmax avoids clipping long rays in normalized scenes.
    constexpr float DEFAULT_RAY_TMAX = 1e20f;
}

// CIE 1931 colour matching now uses the shared HD analytic functions
// in core/spectrum.h (cie_x, cie_y, cie_z, spectrum_to_xyz).

extern "C" {
    __constant__ LaunchParams params;
}

// ── Textual includes: helper layers (order matters) ──────────────────
#include "optix/optix_utils.cuh"
#include "optix/optix_material.cuh"
#include "optix/optix_bsdf.cuh"
#include "optix/optix_nee.cuh"
#include "optix/optix_specular.cuh"
#include "optix/optix_guided.cuh"
#include "optix/optix_direction_map.cuh"
#include "optix/optix_path_trace_v3.cuh"
#include "optix/optix_camera.cuh"

// =====================================================================
extern "C" __global__ void __raygen__render() {
    const uint3 idx = optixGetLaunchIndex();
    int px = idx.x;
    int py = idx.y;
    int pixel_idx = py * params.width + px;

    // ── Adaptive sampling: skip inactive pixels ──────────────────────
    // active_mask is nullptr when adaptive sampling is disabled.
    if (params.active_mask && params.active_mask[pixel_idx] == 0)
        return;

    // ── NORMAL RENDER PATH ──────────────────────────────────────

    Spectrum L_accum = Spectrum::zero();
    Spectrum L_nee_accum = Spectrum::zero();
    Spectrum L_photon_accum = Spectrum::zero();
    Spectrum L_bounce_accum[MAX_AOV_BOUNCES];
    for (int b = 0; b < MAX_AOV_BOUNCES; ++b)
        L_bounce_accum[b] = Spectrum::zero();

    long long prof_total_start = clock64();
    long long prof_ray  = 0, prof_nee  = 0;
    long long prof_pg   = 0, prof_bsdf = 0;
    (void)prof_ray; (void)prof_nee; (void)prof_pg; (void)prof_bsdf;

    for (int s = 0; s < params.samples_per_pixel; ++s) {
        PCGRng rng = PCGRng::seed(
            (uint64_t)pixel_idx * 1000
                + (uint64_t)params.frame_number * 100000 + s,
            (uint64_t)pixel_idx + 1);

        // Stratified sub-pixel sampling (when SPP = STRATA_X * STRATA_Y)
        int sample_index = params.frame_number * params.samples_per_pixel + s;

        float3 origin, direction;
        generate_camera_ray_from_params(px, py, rng, origin, direction, sample_index);

        // v3: photon-guided iterative path tracer (Part 2 §4)
        {
            PathTraceResult ptr = full_path_trace_v3(origin, direction, rng, pixel_idx, s, params.samples_per_pixel);
            L_accum        += ptr.combined;
            L_nee_accum    += ptr.nee_direct;
            L_photon_accum += ptr.photon_indirect;
            if (params.bounce_aov_enabled) {
                for (int b = 0; b < MAX_AOV_BOUNCES; ++b)
                    L_bounce_accum[b] += ptr.bounce_contrib[b];
            }
            if constexpr (ENABLE_STATS) {
                prof_ray  += ptr.clk_ray_trace;
                prof_nee  += ptr.clk_nee;
                prof_pg   += ptr.clk_photon_gather;
                prof_bsdf += ptr.clk_bsdf;
            }

            // AOV: write denoiser guide layers on first sample of first frame
            if (s == 0 && params.frame_number == 0) {
                if (params.albedo_buffer) {
                    params.albedo_buffer[pixel_idx * 4 + 0] = ptr.first_hit_albedo.x;
                    params.albedo_buffer[pixel_idx * 4 + 1] = ptr.first_hit_albedo.y;
                    params.albedo_buffer[pixel_idx * 4 + 2] = ptr.first_hit_albedo.z;
                    params.albedo_buffer[pixel_idx * 4 + 3] = 1.0f;
                }
                if (params.normal_buffer) {
                    // World-space shading normal (OPTIX_DENOISER_MODEL_KIND_AOV expects world space)
                    float3 n = ptr.first_hit_normal;
                    params.normal_buffer[pixel_idx * 4 + 0] = n.x;
                    params.normal_buffer[pixel_idx * 4 + 1] = n.y;
                    params.normal_buffer[pixel_idx * 4 + 2] = n.z;
                    params.normal_buffer[pixel_idx * 4 + 3] = 0.0f;
                }
            }
        }
    }

    long long prof_total_clk = clock64() - prof_total_start;

    // Progressive accumulation (combined)
    for (int i = 0; i < NUM_LAMBDA; ++i)
        params.spectrum_buffer[pixel_idx * NUM_LAMBDA + i] += L_accum.value[i];
    params.sample_counts[pixel_idx] += (float)params.samples_per_pixel;

    // Adaptive sampling: accumulate luminance moments (if enabled)
    if (params.lum_sum && params.lum_sum2) {
        float Y = spectrum_to_xyz(L_accum).y;
        Y = fmaxf(Y, 0.f);
        params.lum_sum[pixel_idx]  += Y;
        params.lum_sum2[pixel_idx] += Y * Y;
    }

    // Component accumulation
    if (params.nee_direct_buffer) {
        for (int i = 0; i < NUM_LAMBDA; ++i)
            params.nee_direct_buffer[pixel_idx * NUM_LAMBDA + i] += L_nee_accum.value[i];
    }
    if (params.photon_indirect_buffer) {
        for (int i = 0; i < NUM_LAMBDA; ++i)
            params.photon_indirect_buffer[pixel_idx * NUM_LAMBDA + i] += L_photon_accum.value[i];
    }

    // Per-bounce AOV accumulation (DB-04, §10.3)
    if (params.bounce_aov_enabled) {
        for (int b = 0; b < MAX_AOV_BOUNCES; ++b) {
            if (params.bounce_aov[b]) {
                for (int i = 0; i < NUM_LAMBDA; ++i)
                    params.bounce_aov[b][pixel_idx * NUM_LAMBDA + i] += L_bounce_accum[b].value[i];
            }
        }
    }

    // Profiling accumulation (if buffers exist)
    if (params.prof_total) {
        params.prof_total[pixel_idx]         += prof_total_clk;
        params.prof_ray_trace[pixel_idx]     += prof_ray;
        params.prof_nee[pixel_idx]           += prof_nee;
        params.prof_photon_gather[pixel_idx] += prof_pg;
        params.prof_bsdf[pixel_idx]          += prof_bsdf;
    }

    // Tonemap to sRGB (skipped when skip_tonemap is set — post-process kernel will handle it)
    if (!params.skip_tonemap) {
        float n_samples = params.sample_counts[pixel_idx];
        Spectrum avg;
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            float val = (n_samples > 0.f)
                ? params.spectrum_buffer[pixel_idx * NUM_LAMBDA + i] / n_samples
                : 0.f;
            avg.value[i] = val;
        }

        float3 rgb = dev_spectrum_to_srgb(avg);
        rgb.x = fminf(fmaxf(rgb.x, 0.f), 1.f);
        rgb.y = fminf(fmaxf(rgb.y, 0.f), 1.f);
        rgb.z = fminf(fmaxf(rgb.z, 0.f), 1.f);

        params.srgb_buffer[pixel_idx * 4 + 0] = (uint8_t)(rgb.x * 255.f);
        params.srgb_buffer[pixel_idx * 4 + 1] = (uint8_t)(rgb.y * 255.f);
        params.srgb_buffer[pixel_idx * 4 + 2] = (uint8_t)(rgb.z * 255.f);
        params.srgb_buffer[pixel_idx * 4 + 3] = 255;
    }
}

// ── Textual includes: raygen entry points ────────────────────────────
#include "optix/optix_photon_trace.cuh"
#include "optix/optix_targeted_photon.cuh"

// =====================================================================
// __raygen__direction_map  –  Build direction map (one subpixel per thread)
// =====================================================================
// Launched at (dir_map_width × dir_map_height, 1, 1).
// Each thread builds one DirMapEntry by tracing a primary ray, following
// specular chains, gathering photons from the dense grid, and sampling
// a guided direction from the Fibonacci histogram.
extern "C" __global__ void __raygen__direction_map() {
    dev_build_direction_map_pixel(
        params.dir_map_buffer,
        params.dir_map_width,
        params.dir_map_height,
        params.dir_map_spp_seed);
}

// =====================================================================
// Stochastic opacity helpers (TEA hash for cheap per-intersection RNG)
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
// __anyhit__radiance  –  stochastic alpha test for translucent surfaces
// =====================================================================
extern "C" __global__ void __anyhit__radiance() {
    if (!params.opacity) return;   // no opacity data uploaded

    const int      prim_idx = optixGetPrimitiveIndex();
    const uint32_t mat_id   = params.material_ids[prim_idx];
    const float    opac     = params.opacity[mat_id];

    if (opac >= 1.f) return;       // fully opaque — accept hit
    if (opac <= 0.f) { optixIgnoreIntersection(); return; } // fully transparent

    // Cheap per-intersection random number via TEA hash
    const uint3  idx  = optixGetLaunchIndex();
    const uint32_t pixel_seed = idx.x + idx.y * 65537u;
    const uint32_t h   = tea4(pixel_seed, prim_idx ^ __float_as_uint(optixGetRayTmax()));
    const float    xi  = (float)(h & 0x00FFFFFFu) / (float)0x01000000u;  // [0,1)

    if (xi >= opac) optixIgnoreIntersection();  // stochastic transparency
}

// =====================================================================
// __anyhit__shadow  –  stochastic alpha test for shadow rays
// =====================================================================
extern "C" __global__ void __anyhit__shadow() {
    if (!params.opacity) return;

    const int      prim_idx = optixGetPrimitiveIndex();
    const uint32_t mat_id   = params.material_ids[prim_idx];
    const float    opac     = params.opacity[mat_id];

    if (opac >= 1.f) return;
    if (opac <= 0.f) { optixIgnoreIntersection(); return; }

    const uint3  idx  = optixGetLaunchIndex();
    const uint32_t pixel_seed = idx.x + idx.y * 65537u;
    const uint32_t h   = tea4(pixel_seed, prim_idx ^ __float_as_uint(optixGetRayTmax()));
    const float    xi  = (float)(h & 0x00FFFFFFu) / (float)0x01000000u;

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

    float3 v0 = params.vertices[prim_idx * 3 + 0];
    float3 v1 = params.vertices[prim_idx * 3 + 1];
    float3 v2 = params.vertices[prim_idx * 3 + 2];
    float3 pos = v0 * alpha + v1 * beta + v2 * gamma;
    float3 geo_normal = normalize(cross(v1 - v0, v2 - v0));

    float3 n0 = params.normals[prim_idx * 3 + 0];
    float3 n1 = params.normals[prim_idx * 3 + 1];
    float3 n2 = params.normals[prim_idx * 3 + 2];
    float3 shading_normal = normalize(n0 * alpha + n1 * beta + n2 * gamma);

    // Interpolate texture coordinates
    float2 uv0 = params.texcoords[prim_idx * 3 + 0];
    float2 uv1 = params.texcoords[prim_idx * 3 + 1];
    float2 uv2 = params.texcoords[prim_idx * 3 + 2];
    float2 uv  = make_f2(
        uv0.x * alpha + uv1.x * beta + uv2.x * gamma,
        uv0.y * alpha + uv1.y * beta + uv2.y * gamma);

    float t = optixGetRayTmax();
    uint32_t mat_id = params.material_ids[prim_idx];

    optixSetPayload_0(__float_as_uint(pos.x));
    optixSetPayload_1(__float_as_uint(pos.y));
    optixSetPayload_2(__float_as_uint(pos.z));
    optixSetPayload_3(__float_as_uint(shading_normal.x));
    optixSetPayload_4(__float_as_uint(shading_normal.y));
    optixSetPayload_5(__float_as_uint(shading_normal.z));
    optixSetPayload_6(__float_as_uint(t));
    optixSetPayload_7(mat_id);
    optixSetPayload_8((uint32_t)prim_idx);
    optixSetPayload_9(1u);
    optixSetPayload_10(__float_as_uint(geo_normal.x));
    optixSetPayload_11(__float_as_uint(geo_normal.y));
    optixSetPayload_12(__float_as_uint(geo_normal.z));
    optixSetPayload_13(__float_as_uint(uv.x));
    optixSetPayload_14(__float_as_uint(uv.y));
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
    optixSetPayload_0(0u);
    optixSetPayload_1(0u);
}

// =====================================================================
// __miss__shadow
// =====================================================================
extern "C" __global__ void __miss__shadow() {
    optixSetPayload_0(0u); // 0 = not occluded
}
