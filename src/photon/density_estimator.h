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
    bool   use_kernel     = false;   // true = Epanechnikov, false = box kernel
};

// ── Kernel weight functions ─────────────────────────────────────────

// Box kernel: uniform weight 1 within radius, 0 outside
inline float box_kernel(float dist2, float r2) {
    return (dist2 <= r2) ? 1.0f : 0.0f;
}

// Epanechnikov kernel: W(t) = 1 - t  where t = dist2 / r2, clamped to [0,1]
// Gives higher weight to nearby photons for a smoother estimate.
inline float epanechnikov_kernel(float dist2, float r2) {
    if (dist2 >= r2) return 0.0f;
    float t = dist2 / r2;
    return 1.0f - t;
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
        [&](uint32_t idx, float /*dist2*/) {
            // Surface consistency filter
            float3 photon_pos = make_f3(photons.pos_x[idx], photons.pos_y[idx], photons.pos_z[idx]);
            float3 diff = photon_pos - hit_pos;

            // Check: photon should be near the surface plane
            float plane_dist = fabsf(dot(hit_normal, diff));
            if (plane_dist > config.surface_tau) return;

            // Check: photon must be on the same side of the surface as the
            // query point — reject photons deposited on opposite faces (e.g.
            // the back of a wall) to prevent irradiance leaking through walls.
            // Guard: only apply if norm arrays are populated (old binary files may lack them).
            if (!photons.norm_x.empty()) {
                float3 photon_n = make_f3(photons.norm_x[idx], photons.norm_y[idx], photons.norm_z[idx]);
                if (dot(photon_n, hit_normal) <= 0.f) return;
            }

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
// ── SPPM photon gather ──────────────────────────────────────────────
//
// Gathers photons within a per-pixel radius at a visible point and
// returns the raw flux contribution (before progressive update).
//
// Unlike estimate_photon_density() which computes final radiance,
// this returns:
//   phi(λ) = Σ_j f_s(x, ω_o, ω_j, λ) · Φ_j(λ)
//
// along with the photon count M.  The caller applies the SPPM
// progressive update using these values.
//
// @param hit_pos        Visible point position
// @param hit_normal     Shading normal at visible point
// @param wo_local       Outgoing direction in local frame
// @param mat            Material at visible point
// @param photons        Photon map (SoA)
// @param grid           Hash grid for the photon map
// @param gather_radius  Per-pixel gather radius (shrinks over iterations)
// @param surface_tau    Plane-distance filter threshold
// @param[out] M_out     Number of valid photons found
// @return               Raw spectral flux contribution Φ_new

inline Spectrum sppm_gather(
    float3 hit_pos,
    float3 hit_normal,
    float3 wo_local,
    const Material& mat,
    const PhotonSoA& photons,
    const HashGrid& grid,
    float gather_radius,
    float surface_tau,
    int& M_out)
{
    Spectrum phi = Spectrum::zero();
    int M = 0;

    ONB frame = ONB::from_normal(hit_normal);

    grid.query(hit_pos, gather_radius, photons,
        [&](uint32_t idx, float dist2) {
            float3 photon_pos = make_f3(photons.pos_x[idx],
                                        photons.pos_y[idx],
                                        photons.pos_z[idx]);
            float3 diff = photon_pos - hit_pos;

            // Surface consistency: plane distance
            float plane_dist = fabsf(dot(hit_normal, diff));
            if (plane_dist > surface_tau) return;

            // Normal consistency: reject opposite-facing photons
            if (!photons.norm_x.empty()) {
                float3 photon_n = make_f3(photons.norm_x[idx],
                                          photons.norm_y[idx],
                                          photons.norm_z[idx]);
                if (dot(photon_n, hit_normal) <= 0.f) return;
            }

            // Direction consistency: photon must enter from above the surface
            float3 photon_wi = make_f3(photons.wi_x[idx],
                                       photons.wi_y[idx],
                                       photons.wi_z[idx]);
            if (dot(photon_wi, hit_normal) <= 0.f) return;

            // BSDF evaluation
            float3 wi_local = frame.world_to_local(photon_wi);
            Spectrum f = bsdf::evaluate(mat, wo_local, wi_local);

            // Epanechnikov kernel: w = 1 - d²/r²
            float r2 = gather_radius * gather_radius;
            float w = 1.0f - dist2 / r2;
            if (w <= 0.f) return;

            // Accumulate kernel-weighted flux: w · f_s · Φ per wavelength bin
            uint16_t bin = photons.lambda_bin[idx];
            phi.value[bin] += w * f.value[bin] * photons.flux[idx];

            ++M;
        });

    M_out = M;
    return phi;
}