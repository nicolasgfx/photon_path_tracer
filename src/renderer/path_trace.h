#pragma once
// ─────────────────────────────────────────────────────────────────────
// path_trace.h — CPU Photon-Guided Path Tracer (v3)
// ─────────────────────────────────────────────────────────────────────
// Part 2 §6: CPU reference that mirrors the GPU v3 algorithm exactly.
//
// Single iterative bounce loop:
//   1. Emission MIS with pdf_combined_prev carry-forward
//   2. Delta surfaces (mirror/glass) — continue loop, no shading
//   3. NEE — 1 shadow ray, MIS-weighted against BSDF
//   4. Photon final gather at terminal bounce
//   5. BSDF direction sampling (no guided sampling on CPU)
//   6. Russian roulette after min_bounces_rr guaranteed bounces
//
// The CPU version does NOT use photon-guided direction sampling
// (CellBinGrid is not queried for mixture PDF).  This is intentional:
// the CPU reference is for unit tests and debugging, not production
// throughput.  BSDF-only sampling with NEE + emission MIS converges
// to the same result — just with higher variance per sample.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"
#include "core/config.h"
#include "core/random.h"
#include "core/ior_stack.h"
#include "bsdf/bsdf.h"
#include "bsdf/bsdf_shared.h"
#include "renderer/direct_light.h"
#include "renderer/nee_shared.h"
#include "scene/scene.h"
#include "photon/photon.h"
#include "photon/hash_grid.h"
#include "photon/density_estimator.h"

// ── Result struct (mirrors GPU PathTraceResult, minus profiling) ─────
struct CpuPathTraceResult {
    Spectrum combined;
    Spectrum nee_direct;
    Spectrum photon_indirect;
};

// ─────────────────────────────────────────────────────────────────────
// path_trace_cpu — CPU reference path tracer (v3 algorithm)
// ─────────────────────────────────────────────────────────────────────
inline CpuPathTraceResult path_trace_cpu(
    Ray ray,
    PCGRng& rng,
    const Scene& scene,
    const RenderConfig& cfg,
    const PhotonSoA& global_photons,
    const HashGrid& global_grid,
    const PhotonSoA& caustic_photons,
    const HashGrid& caustic_grid)
{
    // Caustic gather not yet implemented on CPU (future work)
    (void)caustic_photons;
    (void)caustic_grid;

    CpuPathTraceResult result;
    result.combined        = Spectrum::zero();
    result.nee_direct      = Spectrum::zero();
    result.photon_indirect = Spectrum::zero();

    Spectrum throughput = Spectrum::constant(1.0f);
    IORStack ior_stack;

    // Previous-bounce combined PDF for emission MIS (§4.3)
    float pdf_combined_prev = 0.f;

    const int max_bounces = cfg.max_bounces > 0
        ? cfg.max_bounces : DEFAULT_MAX_BOUNCES_CAMERA;
    const int min_bounces_rr = cfg.min_bounces_rr;
    const float rr_threshold = cfg.rr_threshold;
    const bool final_gather  = DEFAULT_PHOTON_FINAL_GATHER;

    float3 origin    = ray.origin;
    float3 direction = ray.direction;

    DensityEstimatorConfig de_cfg;
    de_cfg.radius            = cfg.gather_radius;
    de_cfg.caustic_radius    = cfg.caustic_radius;
    de_cfg.num_photons_total = cfg.num_photons;
    de_cfg.use_kernel        = true;  // Epanechnikov

    for (int bounce = 0; bounce < max_bounces; ++bounce) {

        // ── Trace ray ───────────────────────────────────────────────
        Ray cur;
        cur.origin    = origin;
        cur.direction = direction;
        HitRecord hit = scene.intersect(cur);

        if (!hit.hit) break;

        // Resolve material (apply diffuse/specular textures)
        Material mat = scene.materials[hit.material_id];
        if (mat.diffuse_tex >= 0 &&
            mat.diffuse_tex < (int)scene.textures.size()) {
            float3 rgb = scene.textures[mat.diffuse_tex].sample(hit.uv);
            mat.Kd = rgb_to_spectrum_reflectance(rgb.x, rgb.y, rgb.z);
        }
        if (mat.specular_tex >= 0 &&
            mat.specular_tex < (int)scene.textures.size()) {
            float3 rgb = scene.textures[mat.specular_tex].sample(hit.uv);
            mat.Ks = rgb_to_spectrum_reflectance(rgb.x, rgb.y, rgb.z);
        }

        // ── Emission (MIS-weighted) ─────────────────────────────────
        if (mat.is_emissive()) {
            Spectrum Le = mat.Le;
            if (bounce == 0) {
                // Camera sees light directly — no MIS needed
                result.combined  += throughput * Le;
                result.nee_direct += throughput * Le;
            } else {
                // MIS: BSDF direction hit a light.
                // When previous bounce was delta (specular/glass),
                // NEE cannot sample through it → full weight to BSDF.
                float w_bsdf;
                if (pdf_combined_prev <= 0.f) {
                    w_bsdf = 1.0f;
                } else {
                    float p_nee = direct_light_pdf(
                        origin, direction, scene);
                    w_bsdf = mis_weight_2(pdf_combined_prev, p_nee);
                }
                result.combined  += throughput * Le * w_bsdf;
                result.nee_direct += throughput * Le * w_bsdf;
            }
            break;
        }

        // ── Delta surfaces: mirror, glass, translucent ──────────────
        if (mat.is_specular() || mat.type == MaterialType::Translucent) {
            ONB frame = ONB::from_normal(hit.shading_normal);
            float3 wo_local = frame.world_to_local(direction * (-1.f));

            BSDFSample bs = bsdf::sample(mat, wo_local, rng);
            if (bs.pdf <= 0.f) break;

            float cos_theta = fabsf(bs.wi.z);
            for (int i = 0; i < NUM_LAMBDA; ++i)
                throughput.value[i] *= bs.f.value[i] * cos_theta / bs.pdf;

            float side = (bs.wi.z > 0.f) ? 1.f : -1.f;
            origin    = hit.position + hit.shading_normal * EPSILON * side;
            direction = frame.local_to_world(bs.wi);
            pdf_combined_prev = 0.f;  // delta → skip emission MIS
            continue;
        }

        // ── Non-delta surface: shading ──────────────────────────────
        ONB frame = ONB::from_normal(hit.shading_normal);
        float3 wo_local = frame.world_to_local(direction * (-1.f));
        if (wo_local.z <= 0.f) break;

        // ── NEE: 1 shadow ray ───────────────────────────────────────
        if (cfg.mode != RenderMode::IndirectOnly) {
            DirectLightSample dls = sample_direct_light(
                hit.position, hit.shading_normal, scene, rng);

            if (dls.visible && dls.pdf_light > 0.f) {
                float3 wi_local = frame.world_to_local(dls.wi);
                float cos_theta = fmaxf(0.f, wi_local.z);
                if (cos_theta > 0.f) {
                    Spectrum f = bsdf::evaluate(mat, wo_local, wi_local);

                    // MIS: weight NEE against BSDF
                    float pdf_bsdf = bsdf::pdf(mat, wo_local, wi_local);
                    float w_nee = mis_weight_2(dls.pdf_light, pdf_bsdf);

                    Spectrum contrib = dls.Li * f * (cos_theta * w_nee / dls.pdf_light);
                    result.combined   += throughput * contrib;
                    result.nee_direct += throughput * contrib;
                }
            }
        }

        // ── Photon final gather at terminal bounce ──────────────────
        if (bounce == max_bounces - 1 && final_gather) {
            if (cfg.mode != RenderMode::DirectOnly) {
                Spectrum L_photon = estimate_photon_density(
                    hit.position, hit.normal, wo_local, mat,
                    global_photons, global_grid, de_cfg,
                    cfg.gather_radius);
                Spectrum photon_contrib = throughput * L_photon;
                result.combined        += photon_contrib;
                result.photon_indirect += photon_contrib;
            }
            break;
        }

        // ── Next direction: BSDF sample ─────────────────────────────
        // (CPU does not use photon-guided mixture sampling)
        BSDFSample bs = bsdf::sample(mat, wo_local, rng);
        if (bs.pdf < 1e-8f || bs.wi.z <= 0.f) break;

        float3 wi_world = frame.local_to_world(bs.wi);
        float cos_theta = bs.wi.z;
        float combined_pdf = bs.pdf;

        for (int i = 0; i < NUM_LAMBDA; ++i)
            throughput.value[i] *= bs.f.value[i] * cos_theta / combined_pdf;

        // ── Russian roulette (§4.1) ─────────────────────────────────
        if (bounce >= min_bounces_rr) {
            float max_tp = throughput.max_component();
            float p_survive = fminf(rr_threshold, max_tp);
            if (p_survive < 1e-4f) break;
            if (rng.next_float() >= p_survive) break;
            float inv_survive = 1.f / p_survive;
            for (int i = 0; i < NUM_LAMBDA; ++i)
                throughput.value[i] *= inv_survive;
        }

        // ── Prepare next ray ────────────────────────────────────────
        origin    = hit.position + hit.shading_normal * EPSILON;
        direction = wi_world;
        pdf_combined_prev = combined_pdf;
    }

    return result;
}

