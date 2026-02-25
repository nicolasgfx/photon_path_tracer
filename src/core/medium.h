#pragma once
// ─────────────────────────────────────────────────────────────────────
// medium.h – Homogeneous participating medium with Rayleigh scattering
// ─────────────────────────────────────────────────────────────────────
// Provides Beer–Lambert transmittance and single-scattering utilities.
// All functions are HD (host + device) for use from both CPU renderer
// and OptiX device code.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"
#include "core/config.h"

// ── Homogeneous medium description ──────────────────────────────────
struct HomogeneousMedium {
    Spectrum sigma_s;   // scattering coefficient (per wavelength)
    Spectrum sigma_a;   // absorption coefficient (per wavelength)
    Spectrum sigma_t;   // extinction = sigma_s + sigma_a (per wavelength)
    float    g = 0.0f;  // Henyey-Greenstein asymmetry parameter
};

// ── Build a Rayleigh-like medium from user knobs ────────────────────
// density  : overall optical‐thickness scale
// albedo   : σ_s / σ_t  (probability of scatter vs absorb)
// falloff  : exponential height decay coefficient (0 = homogeneous)
// y        : world-space y position of the sample point
//
// Spectral shape:  σ_s(λ) ∝ 1/λ⁴   (Rayleigh)
// Reference wavelength:  λ_ref = 550 nm  (chosen so density ≈ σ_t
// at the peak of visible light).
inline HD HomogeneousMedium make_rayleigh_medium(float density,
                                                  float albedo,
                                                  float falloff = 0.f,
                                                  float y       = 0.f) {
    HomogeneousMedium m;

    // Height falloff: ρ(y) = exp(-k * y),  y=0 is ground
    float height_factor = (falloff > 0.f) ? expf(-falloff * y) : 1.f;
    float effective_density = density * height_factor;

    constexpr float lambda_ref = 550.f; // nm
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        float lam   = lambda_of_bin(i);
        float ratio = lambda_ref / lam;
        float r4    = ratio * ratio * ratio * ratio; // (λ_ref / λ)⁴

        float sig_t = effective_density * r4;
        m.sigma_t.value[i] = sig_t;
        m.sigma_s.value[i] = sig_t * albedo;
        m.sigma_a.value[i] = sig_t * (1.f - albedo);
    }
    return m;
}

// ── Beer–Lambert transmittance ──────────────────────────────────────
// T(d) = exp(-σ_t · d)   per wavelength
inline HD Spectrum transmittance(const HomogeneousMedium& m, float distance) {
    Spectrum T;
    for (int i = 0; i < NUM_LAMBDA; ++i)
        T.value[i] = expf(-m.sigma_t.value[i] * distance);
    return T;
}

// ── Scalar extinction at a specific wavelength bin ──────────────────
inline HD float sigma_t_at(const HomogeneousMedium& m, int lambda_bin) {
    return m.sigma_t.value[lambda_bin];
}

// ── Average (scalar) extinction ─────────────────────────────────────
// Useful for distance sampling / Russian roulette.
inline HD float sigma_t_avg(const HomogeneousMedium& m) {
    float s = 0.f;
    for (int i = 0; i < NUM_LAMBDA; ++i) s += m.sigma_t.value[i];
    return s / (float)NUM_LAMBDA;
}
