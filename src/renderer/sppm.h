#pragma once
// ─────────────────────────────────────────────────────────────────────
// sppm.h – Stochastic Progressive Photon Mapping (SPPM) data types
// ─────────────────────────────────────────────────────────────────────
//
// SPPM (Hachisuka & Jensen, 2009) progressively refines per-pixel
// photon density estimates by shrinking the gather radius iteration
// by iteration.  This file defines the per-pixel state and the
// progressive update logic.
//
// Per-iteration algorithm:
//   1. Camera pass  – trace eye paths to first diffuse hit ("visible
//      point"), record position, normal, material, throughput.
//      Direct lighting (NEE) is evaluated here.
//   2. Photon pass  – emit N_p photons from lights, trace them.
//   3. Gather pass  – for each visible point, query the photon hash
//      grid within radius r_i.  Count M_i photons, accumulate
//      new flux Φ_new via the BSDF.
//   4. Progressive update –
//        N_new  = N_i + α · M_i
//        r_new  = r_i · √(N_new / (N_i + M_i))
//        τ_new  = (τ_i + Φ_new) · (r_new / r_i)²
//      The radius shrinks monotonically, converging the estimate.
//   5. Reconstruction –
//        L(x, ω_o, λ) = τ(λ) / (π · r² · k · N_p)
//
// ─────────────────────────────────────────────────────────────────────

#include "core/types.h"
#include "core/spectrum.h"
#include "core/config.h"

#include <cmath>
#include <vector>

// ── SPPM configuration constants ────────────────────────────────────

// Shrinkage parameter α ∈ (0,1).  Lower → faster radius reduction.
// The classic value 2/3 balances bias reduction with variance stability.
constexpr float DEFAULT_SPPM_ALPHA           = 0.666667f;  // 2/3

// Number of SPPM iterations (each = camera + photon + gather pass).
// More iterations → sharper caustics, lower noise.
constexpr int   DEFAULT_SPPM_ITERATIONS      = 64;

// Initial gather radius for SPPM pixels.  Typically the same as
// DEFAULT_GATHER_RADIUS or slightly larger; the algorithm shrinks it.
constexpr float DEFAULT_SPPM_INITIAL_RADIUS  = 0.1f;

// Minimum radius clamp to prevent numerical underflow.
constexpr float DEFAULT_SPPM_MIN_RADIUS      = 1e-5f;

// ── Per-pixel SPPM state ────────────────────────────────────────────

struct SPPMPixel {
    // ── Visible-point data (written by camera pass each iteration) ───
    float3   position;      ///< hit position on first diffuse surface
    float3   normal;        ///< shading normal at visible point
    float3   wo_local;      ///< outgoing direction in local frame
    uint32_t material_id;   ///< material index
    float2   uv;            ///< texture coordinates
    Spectrum throughput;    ///< camera-path throughput to this point
    Spectrum L_direct;      ///< accumulated direct lighting (NEE sum over iterations)
    bool     valid;         ///< true if camera ray hit a diffuse surface

    // ── Progressive state (persists across iterations) ───────────────
    float    radius;        ///< current gather radius r_i
    float    N;             ///< accumulated weighted photon count
    Spectrum tau;           ///< accumulated (unnormalized) spectral flux
    int      M_count;       ///< photons found in the latest gather pass

    // ── Initialisation ───────────────────────────────────────────────
    void init(float initial_radius) {
        position    = make_f3(0, 0, 0);
        normal      = make_f3(0, 1, 0);
        wo_local    = make_f3(0, 0, 1);
        material_id = 0;
        uv          = make_f2(0, 0);
        throughput  = Spectrum::zero();
        L_direct    = Spectrum::zero();
        valid       = false;
        radius      = initial_radius;
        N           = 0.f;
        tau         = Spectrum::zero();
        M_count     = 0;
    }
};

// ── SPPM progressive update ─────────────────────────────────────────
//
// Called once per pixel after each gather pass.
//
// @param pixel      Per-pixel SPPM state (modified in place)
// @param phi_new    New flux contribution from this iteration's photons:
//                   Σ_j f_s(x, ω_o, ω_j) · Φ_j                  (spectral)
// @param M          Number of photons found in this iteration's gather
// @param alpha      Shrinkage factor α ∈ (0,1)
// @param min_radius Floor to prevent radius collapsing to zero

inline void sppm_progressive_update(
    SPPMPixel& pixel,
    const Spectrum& phi_new,
    int   M,
    float alpha    = DEFAULT_SPPM_ALPHA,
    float min_radius = DEFAULT_SPPM_MIN_RADIUS)
{
    if (M == 0) return;  // no photons → no update

    float N_old = pixel.N;
    float N_new = N_old + alpha * (float)M;

    // Radius shrinkage: r_new = r_old · √(N_new / (N_old + M))
    float ratio = N_new / (N_old + (float)M);
    float r_old = pixel.radius;
    float r_new = r_old * sqrtf(ratio);
    if (r_new < min_radius) r_new = min_radius;

    // Area ratio for flux correction: (r_new / r_old)²
    float area_ratio = (r_new * r_new) / (r_old * r_old);

    // Flux update: τ_new = (τ_old + Φ_new) · area_ratio
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        pixel.tau.value[i] = (pixel.tau.value[i] + phi_new.value[i]) * area_ratio;
    }

    pixel.N      = N_new;
    pixel.radius = r_new;
    pixel.M_count = M;
}

// ── SPPM radiance reconstruction ────────────────────────────────────
//
// Reconstructs the final spectral radiance for a pixel after all
// iterations have completed.
//
// @param pixel             Per-pixel SPPM state
// @param total_iterations  Number of SPPM iterations completed (k)
// @param photons_per_iter  Number of emitted photons per iteration (N_p)
// @return                  Reconstructed spectral radiance L

inline Spectrum sppm_reconstruct(
    const SPPMPixel& pixel,
    int   total_iterations,
    int   photons_per_iter)
{
    if (!pixel.valid || total_iterations <= 0 || photons_per_iter <= 0)
        return Spectrum::zero();

    float r2 = pixel.radius * pixel.radius;
    // Denominator: (π/2) · r² · k · N_p   (Epanechnikov kernel normalisation)
    float denom = 0.5f * PI * r2 * (float)total_iterations * (float)photons_per_iter;
    if (denom <= 0.f) return Spectrum::zero();

    float inv_denom = 1.f / denom;

    // L_indirect = τ / ((π/2) · r² · k · N_p)  (throughput already baked into τ)
    Spectrum L;
    for (int i = 0; i < NUM_LAMBDA; ++i)
        L.value[i] = pixel.tau.value[i] * inv_denom;

    // Add averaged direct lighting: L_direct was summed over k iterations
    Spectrum L_direct_avg;
    for (int i = 0; i < NUM_LAMBDA; ++i)
        L_direct_avg.value[i] = pixel.L_direct.value[i] / (float)total_iterations;

    return L + L_direct_avg;
}

// ── SPPM pixel buffer (CPU side) ────────────────────────────────────

struct SPPMBuffer {
    std::vector<SPPMPixel> pixels;
    int width  = 0;
    int height = 0;

    void resize(int w, int h, float initial_radius) {
        width  = w;
        height = h;
        pixels.resize((size_t)w * h);
        for (auto& p : pixels)
            p.init(initial_radius);
    }

    SPPMPixel& at(int x, int y) { return pixels[y * width + x]; }
    const SPPMPixel& at(int x, int y) const { return pixels[y * width + x]; }
};
