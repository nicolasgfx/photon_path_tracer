#pragma once
// ─────────────────────────────────────────────────────────────────────
// phase_function.h – Phase functions for volumetric scattering
// ─────────────────────────────────────────────────────────────────────
// All functions are HD (host + device).
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"

// ── Rayleigh phase function ─────────────────────────────────────────
// p_R(cosθ) = 3 / (16π) · (1 + cos²θ)
//
// Normalised over the full sphere:  ∫₄π p_R dω = 1.
inline HD float rayleigh_phase(float cos_theta) {
    constexpr float k = 3.f / (16.f * PI);
    return k * (1.f + cos_theta * cos_theta);
}

// ── Henyey–Greenstein phase function (optional, for future use) ─────
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
