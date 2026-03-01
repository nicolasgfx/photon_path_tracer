#pragma once
// ─────────────────────────────────────────────────────────────────────
// volume/medium.h – Participating media and phase functions
// ─────────────────────────────────────────────────────────────────────
// Merged from core/medium.h + core/phase_function.h.
// Provides:
//   - HomogeneousMedium struct + Beer–Lambert transmittance
//   - Rayleigh and Henyey–Greenstein phase functions
// All functions are HD (host + device).
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"
#include "core/config.h"

// =====================================================================
// Phase Functions
// =====================================================================

// ── Rayleigh phase function ─────────────────────────────────────────
// p_R(cosθ) = 3 / (16π) · (1 + cos²θ)
//
// Normalised over the full sphere:  ∫₄π p_R dω = 1.
inline HD float rayleigh_phase(float cos_theta) {
    constexpr float k = 3.f / (16.f * PI);
    return k * (1.f + cos_theta * cos_theta);
}

// ── Henyey–Greenstein phase function ────────────────────────────────
// p_HG(cosθ; g) = (1 - g²) / (4π · (1 + g² - 2g·cosθ)^{3/2})
//
// g ∈ (-1, 1):  g > 0 → forward scattering, g = 0 → isotropic.
inline HD float henyey_greenstein_phase(float cos_theta, float g) {
    float denom = 1.f + g * g - 2.f * g * cos_theta;
    denom = fmaxf(denom, 1e-10f);
    float denom32 = denom * sqrtf(denom);
    return (1.f - g * g) / (4.f * PI * denom32);
}

// ── Henyey–Greenstein importance sampling ────────────────────────────
// Samples a direction in LOCAL coordinates (z = forward) from the
// Henyey–Greenstein phase function.
// u1, u2 are uniform [0,1).
// Returns a unit vector in local frame where z = incident direction.
inline HD float3 sample_henyey_greenstein(float g, float u1, float u2) {
    float cos_theta;
    if (fabsf(g) < 1e-5f) {
        // Isotropic: uniform sphere sampling
        cos_theta = 1.f - 2.f * u1;
    } else {
        float sqr = (1.f - g * g) / (1.f - g + 2.f * g * u1);
        cos_theta = (1.f + g * g - sqr * sqr) / (2.f * g);
    }
    float sin_theta = sqrtf(fmaxf(0.f, 1.f - cos_theta * cos_theta));
    float phi = 2.f * PI * u2;
    return make_f3(sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta);
}

// =====================================================================
// Homogeneous Participating Medium
// =====================================================================

// ── Homogeneous medium description ──────────────────────────────────
struct HomogeneousMedium {
    Spectrum sigma_s;   // scattering coefficient (per wavelength)
    Spectrum sigma_a;   // absorption coefficient (per wavelength)
    Spectrum sigma_t;   // extinction = sigma_s + sigma_a (per wavelength)
    float    g = 0.0f;  // Henyey-Greenstein asymmetry parameter
};

// ── Medium stack (nested-object interior tracking) ──────────────────
// Parallel to IORStack (core/ior_stack.h): push medium_id on entering
// a Translucent surface, pop on exiting.  Empty stack → no interior
// medium (-1).  Shared HD for CPU emitter and GPU kernels (§7.10).
struct MediumStack {
    static constexpr int MAX_DEPTH = 4;
    int  stack[MAX_DEPTH] = {-1, -1, -1, -1};
    int  depth            = 0;

    HD int  current_medium_id() const { return depth > 0 ? stack[depth - 1] : -1; }
    HD void push(int medium_id) { if (depth < MAX_DEPTH) stack[depth++] = medium_id; }
    HD void pop()               { if (depth > 0) --depth; }
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
