#pragma once
// ─────────────────────────────────────────────────────────────────────
// density_estimator.h – Photon density estimation (§6, v2.2 kNN)
// ─────────────────────────────────────────────────────────────────────
//
// v2.2: Adaptive-radius kNN gather with tangential distance metric.
//
// Diffuse estimate at query point x, outgoing direction ω_o, wavelength λ:
//
//   L_o(x, ω_o, λ) = (1 / A_k) Σ_{i=1..K} W(d²_tan,i / r²_k)
//                       · f_s(x, ω_i→ω_o, λ) · Φ_i(λ) / N
//
// where:
//   K     = DEFAULT_KNN_K nearest photons (tangential distance)
//   r_k   = distance to the K-th nearest photon (adaptive radius)
//   A_k   = π r_k²  (box) or (π/2) r_k²  (Epanechnikov)
//   W     = 1 (box) or 1 − d²/r²_k (Epanechnikov)
//
// Surface consistency filter (§6.4) is applied BEFORE accumulation.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"
#include "scene/material.h"
#include "bsdf/bsdf.h"
#include "photon/photon.h"
#include "photon/hash_grid.h"
#include "photon/surface_filter.h"

struct DensityEstimatorConfig {
    float  radius         = 0.05f;   // Gather radius (max kNN search radius)
    float  caustic_radius = 0.02f;   // Smaller radius for caustics
    float  surface_tau    = 0.02f;   // Surface consistency threshold (§6.4)
    int    num_photons_total = 1;    // For flux normalization (1/N)
    bool   use_kernel     = false;   // true = Epanechnikov, false = box weight
};

// ── Density estimate at a surface point ─────────────────────────────
//
// Returns spectral radiance estimate L_o(x, wo, lambda)
//
// ── Density estimate at a surface point (§6.3 tangential kernel) ────
//
// Returns spectral radiance estimate L_o(x, wo, lambda).
//
// v2.1: uses tangential disk distance on the query's tangent plane
// and the surface consistency filter (§6.4).
//
// Full spectral: each photon contributes to ALL wavelength bins.

inline Spectrum estimate_photon_density(
    float3 hit_pos,
    float3 hit_normal,
    float3 wo_local,
    const Material& mat,
    const PhotonSoA& photons,
    const HashGrid& grid,
    const DensityEstimatorConfig& config,
    float gather_radius)
{
    Spectrum L_estimate = Spectrum::zero();
    float r2 = gather_radius * gather_radius;
    float inv_N = 1.0f / (float)config.num_photons_total;
    float tau   = effective_tau(config.surface_tau);

    ONB frame = ONB::from_normal(hit_normal);

    // ── Phase 1: Collect kNN candidates ─────────────────────────────
    struct Candidate { float d_tan2; uint32_t idx; };
    std::vector<Candidate> candidates;
    candidates.reserve(256);

    grid.query_tangential(hit_pos, hit_normal, gather_radius, tau, photons,
        [&](uint32_t idx, float d_tan2) {
            // Normal compatibility
            if (!photons.norm_x.empty()) {
                float3 photon_n = make_f3(photons.norm_x[idx],
                                          photons.norm_y[idx],
                                          photons.norm_z[idx]);
                if (dot(photon_n, hit_normal) <= 0.0f) return;
            }

            // Direction consistency
            float3 photon_wi = make_f3(photons.wi_x[idx],
                                       photons.wi_y[idx],
                                       photons.wi_z[idx]);
            if (dot(photon_wi, hit_normal) <= 0.f) return;

            candidates.push_back({d_tan2, idx});
        });

    if (candidates.empty()) return L_estimate;

    // ── Phase 2: Determine adaptive radius ──────────────────────────
    const int K = DEFAULT_KNN_K;
    float r_k2;
    if ((int)candidates.size() > K) {
        // Partial sort: find the K-th smallest d_tan2
        std::nth_element(candidates.begin(), candidates.begin() + K,
                         candidates.end(),
                         [](const Candidate& a, const Candidate& b) {
                             return a.d_tan2 < b.d_tan2;
                         });
        r_k2 = candidates[K - 1].d_tan2;  // K-th nearest (0-indexed: K-1)
        r_k2 = std::max(r_k2, 1e-12f);
        candidates.resize(K);  // keep only the K nearest
    } else {
        r_k2 = r2;  // fallback: fixed radius
    }

    // Use tangential disk normalization (§6.3, §15.1.3):
    //   Epanechnikov: 2 / (π r_k²)
    //   Box:          1 / (π r_k²)
    float inv_area_knn = config.use_kernel
                       ? 2.0f / (PI * r_k2)
                       : 1.0f / (PI * r_k2);

    // ── Phase 3: Accumulate contributions ───────────────────────────
    for (const auto& c : candidates) {
        if (c.d_tan2 >= r_k2) continue;

        float w = config.use_kernel
                ? (1.0f - c.d_tan2 / r_k2)      // Epanechnikov
                : 1.0f;                           // Box

        float3 photon_wi = make_f3(photons.wi_x[c.idx],
                                   photons.wi_y[c.idx],
                                   photons.wi_z[c.idx]);
        float3 wi_local = frame.world_to_local(photon_wi);
        Spectrum f = bsdf::evaluate_diffuse(mat, wo_local, wi_local);

        Spectrum photon_flux = photons.get_flux(c.idx);
        for (int b = 0; b < NUM_LAMBDA; ++b)
            L_estimate.value[b] += w * photon_flux.value[b] * inv_N
                                 * f.value[b] * inv_area_knn;
    }

    return L_estimate;
}