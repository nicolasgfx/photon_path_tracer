#pragma once
// ─────────────────────────────────────────────────────────────────────
// density_estimator.h – Photon density estimation (Section 6)
// ─────────────────────────────────────────────────────────────────────
//
// Diffuse estimate (Equation from spec):
//   L_o(x, wo, lambda) = (1 / (pi * r^2)) * sum_i Phi_i(lambda) * f_s(x, wi, wo, lambda)
//
// With optional radial kernel weight W(||x - x_i||)
// and surface consistency filter.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"
#include "scene/material.h"
#include "bsdf/bsdf.h"
#include "photon/photon.h"
#include "photon/hash_grid.h"

struct DensityEstimatorConfig {
    float  radius         = 0.05f;   // Gather radius
    float  caustic_radius = 0.02f;   // Smaller radius for caustics
    float  surface_tau    = 0.01f;   // Surface consistency threshold
    int    num_photons_total = 1;    // For flux normalization (1/N)
    // NOTE: Epanechnikov kernel removed — incompatible with the dense 3D
    // cell-bin grid, which pre-aggregates photons without per-query distances.
    // The estimator always uses a box kernel (uniform weight within radius).
};

// ── Density estimate at a surface point ─────────────────────────────
//
// Returns spectral radiance estimate L_o(x, wo, lambda)
//
// Parameters:
//   hit_pos      – Surface point x
//   hit_normal   – Surface normal n at x
//   wo_local     – Outgoing direction in local frame
//   mat          – Material at x
//   photons      – Photon map (SoA)
//   grid         – Hash grid for the photon map
//   config       – Estimator parameters

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
    float inv_area = 1.0f / (PI * r2);  // 1 / (pi * r^2)
    float inv_N = 1.0f / (float)config.num_photons_total;

    ONB frame = ONB::from_normal(hit_normal);

    grid.query(hit_pos, gather_radius, photons,
        [&](uint32_t idx, float dist2) {
            // Surface consistency filter
            float3 photon_pos = make_f3(photons.pos_x[idx], photons.pos_y[idx], photons.pos_z[idx]);
            float3 diff = photon_pos - hit_pos;

            // Check: photon should be near the surface plane
            float plane_dist = fabsf(dot(hit_normal, diff));
            if (plane_dist > config.surface_tau) return;

            // Check: photon incoming direction should point into the surface
            float3 photon_wi = make_f3(photons.wi_x[idx], photons.wi_y[idx], photons.wi_z[idx]);
            if (dot(photon_wi, hit_normal) <= 0.f) return;

            // BSDF evaluation: f_s(x, wi, wo, lambda)
            float3 wi_local = frame.world_to_local(photon_wi);
            Spectrum f = bsdf::evaluate(mat, wo_local, wi_local);

            // Accumulate: Phi_i * f_s / (pi * r^2 * N)  — box kernel (weight = 1)
            uint16_t bin = photons.lambda_bin[idx];
            float photon_flux = photons.flux[idx] * inv_N;

            // Add contribution only for the photon's wavelength bin
            L_estimate.value[bin] += photon_flux * f.value[bin] * inv_area;
        });

    return L_estimate;
}

// ── Photon-guided directional sampling (Section 7.3) ────────────────
// Sample a direction from nearby photons for MIS

struct PhotonGuidedSample {
    float3 wi_world;
    float  pdf;
    bool   valid;
};

inline PhotonGuidedSample sample_photon_guided(
    float3 hit_pos,
    float3 hit_normal,
    const PhotonSoA& photons,
    const HashGrid& grid,
    float gather_radius,
    PCGRng& rng)
{
    PhotonGuidedSample result;
    result.valid = false;

    // Collect nearby photons and their fluxes
    struct NeighborPhoton {
        float3 wi;
        float  flux;
    };

    std::vector<NeighborPhoton> neighbors;
    float total_flux = 0.f;

    grid.query(hit_pos, gather_radius, photons,
        [&](uint32_t idx, float /*dist2*/) {
            float3 wi = make_f3(photons.wi_x[idx], photons.wi_y[idx], photons.wi_z[idx]);
            if (dot(wi, hit_normal) <= 0.f) return;

            NeighborPhoton np;
            np.wi   = wi;
            np.flux = photons.flux[idx];
            neighbors.push_back(np);
            total_flux += np.flux;
        });

    if (neighbors.empty() || total_flux <= 0.f) return result;

    // Discrete proposal: q(wi) proportional to Phi_i
    float xi = rng.next_float() * total_flux;
    float cumulative = 0.f;
    int chosen = (int)neighbors.size() - 1;
    for (int i = 0; i < (int)neighbors.size(); ++i) {
        cumulative += neighbors[i].flux;
        if (xi <= cumulative) {
            chosen = i;
            break;
        }
    }

    result.wi_world = neighbors[chosen].wi;
    result.pdf      = neighbors[chosen].flux / total_flux;
    result.valid    = true;
    return result;
}

// ── Photon-guided PDF for a given direction ─────────────────────────
inline float photon_guided_pdf(
    float3 wi_world,
    float3 hit_pos,
    float3 hit_normal,
    const PhotonSoA& photons,
    const HashGrid& grid,
    float gather_radius)
{
    float total_flux = 0.f;
    float matching_flux = 0.f;
    float cos_thresh = 0.95f; // Angular matching threshold

    grid.query(hit_pos, gather_radius, photons,
        [&](uint32_t idx, float /*dist2*/) {
            float3 wi = make_f3(photons.wi_x[idx], photons.wi_y[idx], photons.wi_z[idx]);
            if (dot(wi, hit_normal) <= 0.f) return;

            float f = photons.flux[idx];
            total_flux += f;

            // Check if this photon's direction matches wi_world
            if (dot(wi, wi_world) > cos_thresh) {
                matching_flux += f;
            }
        });

    if (total_flux <= 0.f) return 0.f;
    return matching_flux / total_flux;
}
