#pragma once
// ─────────────────────────────────────────────────────────────────────
// emitter.h – Photon emission from emissive surfaces
// ─────────────────────────────────────────────────────────────────────
// Implements Section 4 of the specification:
//   4.1 Build emissive triangle distribution
//   4.2 Sample emission position
//   4.3 Sample wavelength
//   4.4 Photon flux definition
//   4.5 Photon tracing with Russian roulette
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"
#include "core/random.h"
#include "core/alias_table.h"
#include "scene/scene.h"
#include "bsdf/bsdf.h"
#include "photon/photon.h"

#include <vector>

struct EmitterConfig {
    int    num_photons       = 1000000;  // Total photons to emit
    int    max_bounces       = 10;       // Max photon bounces
    float  rr_threshold      = 0.95f;    // Russian roulette survival cap
    int    min_bounces_rr    = 3;        // Min bounces before RR
};

// ── Photon emission from a single emissive triangle ─────────────────

struct EmittedPhoton {
    Ray      ray;           // Origin + direction of emitted photon
    uint16_t lambda_bin;    // Sampled wavelength bin
    float    flux;          // Photon flux = Le * cos(theta) / (p_tri * p_pos * p_dir * p_lambda)
};

// Sample an emitted photon given the scene's emissive distribution
inline EmittedPhoton sample_emitted_photon(const Scene& scene, PCGRng& rng) {
    EmittedPhoton ep;

    // 1. Sample emissive triangle via alias table
    float u1 = rng.next_float();
    float u2 = rng.next_float();
    int local_idx = scene.emissive_alias_table.sample(u1, u2);
    uint32_t tri_idx = scene.emissive_tri_indices[local_idx];
    const Triangle& tri = scene.triangles[tri_idx];
    const Material& mat = scene.materials[tri.material_id];

    float pdf_tri = scene.emissive_alias_table.pdf(local_idx);

    // 2. Sample position on triangle (uniform barycentric)
    float3 bary = sample_triangle(rng.next_float(), rng.next_float());
    float3 pos  = tri.interpolate_position(bary.x, bary.y, bary.z);
    float3 n    = tri.geometric_normal();
    float  area = tri.area();
    float  pdf_pos = 1.0f / area;

    // 3. Sample wavelength bin proportional to Le(lambda)
    //    p(lambda_i | x) = Le(lambda_i) / sum_j Le(lambda_j)
    float Le_sum = mat.Le.sum();
    float xi_lambda = rng.next_float() * Le_sum;
    float cumulative = 0.f;
    ep.lambda_bin = NUM_LAMBDA - 1;
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        cumulative += mat.Le.value[i];
        if (xi_lambda <= cumulative) {
            ep.lambda_bin = (uint16_t)i;
            break;
        }
    }
    float Le_lambda = mat.Le.value[ep.lambda_bin];
    float pdf_lambda = (Le_sum > 0.f) ? Le_lambda / Le_sum : 1.f / NUM_LAMBDA;

    // 4. Sample cosine-weighted direction (hemisphere above triangle)
    float3 local_dir = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
    ONB frame = ONB::from_normal(n);
    float3 world_dir = frame.local_to_world(local_dir);
    float cos_theta = local_dir.z;
    float pdf_dir = cosine_hemisphere_pdf(cos_theta);

    // 5. Compute photon flux
    // Phi = Le(x, omega, lambda) * cos_theta / (p_tri * p_pos * p_dir * p_lambda)
    //
    // Note: total_emissive_power is already factored into the alias table's
    // total_weight. The unnormalized triangle PDF includes area * mean_Le.
    // We need the actual normalized PDF.
    float denom = pdf_tri * pdf_pos * pdf_dir * pdf_lambda;
    ep.flux = (denom > 0.f) ? (Le_lambda * cos_theta) / denom : 0.f;

    // Scale by 1/N (N = total photons) – applied later in density estimator
    // Keep flux as total emitted power per photon here.

    ep.ray.origin    = pos + n * EPSILON; // Offset to avoid self-intersection
    ep.ray.direction = world_dir;

    return ep;
}

// ── Trace photons through the scene ─────────────────────────────────
// Fills global_map (and optionally caustic_map) with photon hits.

inline void trace_photons(const Scene& scene,
                           const EmitterConfig& config,
                           PhotonSoA& global_map,
                           PhotonSoA& caustic_map) {
    global_map.clear();
    caustic_map.clear();
    global_map.reserve(config.num_photons);
    caustic_map.reserve(config.num_photons / 4);

    if (scene.emissive_tri_indices.empty()) {
        return;
    }

    for (int photon_idx = 0; photon_idx < config.num_photons; ++photon_idx) {
        PCGRng rng = PCGRng::seed(photon_idx * 7 + 42, photon_idx + 1);

        EmittedPhoton ep = sample_emitted_photon(scene, rng);
        float flux = ep.flux;
        Ray ray = ep.ray;

        bool on_caustic_path = true; // True until first diffuse bounce

        for (int bounce = 0; bounce < config.max_bounces; ++bounce) {
            HitRecord hit = scene.intersect(ray);
            if (!hit.hit) break;

            const Material& mat = scene.materials[hit.material_id];

            // Store photon at diffuse surfaces
            if (!mat.is_specular()) {
                Photon p;
                p.position   = hit.position;
                p.wi         = ray.direction * (-1.f); // Flip: stored as incoming
                p.lambda_bin = ep.lambda_bin;
                p.flux       = flux;

                if (on_caustic_path && bounce > 0) {
                    // Caustic photon: came through specular path
                    caustic_map.push_back(p);
                }
                // No-direct-deposits rule (§4.3): skip first diffuse hit from light
                // (direct illumination is handled by NEE in the camera pass)
                if (bounce > 0) {
                    global_map.push_back(p);
                }

                on_caustic_path = false;
            }

            // Sample next direction via BSDF
            ONB frame = ONB::from_normal(hit.shading_normal);
            float3 wo_local = frame.world_to_local(ray.direction * (-1.f));

            BSDFSample bsdf_sample = bsdf::sample(mat, wo_local, rng);
            if (bsdf_sample.pdf <= 0.f) break;

            // Update flux: throughput = f * cos_theta / pdf
            float cos_theta = fabsf(bsdf_sample.wi.z);
            float throughput_lambda = bsdf_sample.f.value[ep.lambda_bin] * cos_theta / bsdf_sample.pdf;
            flux *= throughput_lambda;

            // Russian roulette
            if (bounce >= config.min_bounces_rr) {
                float p_rr = fminf(config.rr_threshold,
                                    bsdf_sample.f.max_component());
                if (rng.next_float() >= p_rr) break;
                flux /= p_rr;
            }

            // Update ray for next bounce
            float3 wi_world = frame.local_to_world(bsdf_sample.wi);
            ray.origin    = hit.position + hit.shading_normal * EPSILON;
            ray.direction = wi_world;
            ray.tmin      = 1e-4f;
            ray.tmax      = 1e20f;
        }
    }
}
