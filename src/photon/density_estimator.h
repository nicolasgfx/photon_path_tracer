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
    bool   use_kernel     = true;    // Use Epanechnikov kernel
    int    num_photons_total = 1;    // For flux normalization (1/N)
};

// ── Kernel functions ────────────────────────────────────────────────

// Epanechnikov kernel (optimal for MSE)
inline HD float epanechnikov_kernel(float dist2, float r2) {
    float u = dist2 / r2;
    return (u <= 1.f) ? (1.f - u) : 0.f;
}

// Box kernel (simplest)
inline HD float box_kernel(float dist2, float r2) {
    return (dist2 <= r2) ? 1.f : 0.f;
}

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

            // Kernel weight
            float kernel_w = config.use_kernel
                ? epanechnikov_kernel(dist2, r2)
                : box_kernel(dist2, r2);

            // BSDF evaluation: f_s(x, wi, wo, lambda)
            float3 wi_local = frame.world_to_local(photon_wi);
            Spectrum f = bsdf::evaluate(mat, wo_local, wi_local);

            // Accumulate: Phi_i * f_s * kernel / (pi * r^2)
            uint16_t bin = photons.lambda_bin[idx];
            float photon_flux = photons.flux[idx] * inv_N;

            // Add contribution only for the photon's wavelength bin
            L_estimate.value[bin] += photon_flux * f.value[bin] * kernel_w * inv_area;
        });

    // Normalize kernel if using Epanechnikov (integral = 2/3 over disk)
    if (config.use_kernel) {
        L_estimate *= 1.5f; // Correction for Epanechnikov normalization
    }

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
