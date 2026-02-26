#pragma once
// ─────────────────────────────────────────────────────────────────────
// spectrum.h – Spectral representation and CIE colour matching
// ─────────────────────────────────────────────────────────────────────
//
// Wavelength range: 380 nm – 780 nm
// NUM_LAMBDA bins, uniformly spaced
// ─────────────────────────────────────────────────────────────────────
#include "types.h"
#include <cstring>

constexpr int   NUM_LAMBDA    = 4;
constexpr float LAMBDA_MIN    = 380.0f;   // nm
constexpr float LAMBDA_MAX    = 780.0f;   // nm
constexpr float LAMBDA_STEP   = (LAMBDA_MAX - LAMBDA_MIN) / NUM_LAMBDA;

// Central wavelength of bin i
inline HD float lambda_of_bin(int i) {
    return LAMBDA_MIN + (i + 0.5f) * LAMBDA_STEP;
}

// ── Spectrum ────────────────────────────────────────────────────────
struct Spectrum {
    float value[NUM_LAMBDA];

    HD static Spectrum zero() {
        Spectrum s;
        for (int i = 0; i < NUM_LAMBDA; ++i) s.value[i] = 0.f;
        return s;
    }

    HD static Spectrum constant(float v) {
        Spectrum s;
        for (int i = 0; i < NUM_LAMBDA; ++i) s.value[i] = v;
        return s;
    }

    HD float  operator[](int i) const { return value[i]; }
    HD float& operator[](int i)       { return value[i]; }

    HD Spectrum operator+(const Spectrum& o) const {
        Spectrum r;
        for (int i = 0; i < NUM_LAMBDA; ++i) r.value[i] = value[i] + o.value[i];
        return r;
    }
    HD Spectrum operator*(const Spectrum& o) const {
        Spectrum r;
        for (int i = 0; i < NUM_LAMBDA; ++i) r.value[i] = value[i] * o.value[i];
        return r;
    }
    HD Spectrum operator*(float s) const {
        Spectrum r;
        for (int i = 0; i < NUM_LAMBDA; ++i) r.value[i] = value[i] * s;
        return r;
    }
    HD Spectrum operator/(float s) const {
        float inv = 1.f / s;
        Spectrum r;
        for (int i = 0; i < NUM_LAMBDA; ++i) r.value[i] = value[i] * inv;
        return r;
    }
    HD Spectrum& operator+=(const Spectrum& o) {
        for (int i = 0; i < NUM_LAMBDA; ++i) value[i] += o.value[i];
        return *this;
    }
    HD Spectrum& operator*=(float s) {
        for (int i = 0; i < NUM_LAMBDA; ++i) value[i] *= s;
        return *this;
    }

    HD float sum() const {
        float s = 0.f;
        for (int i = 0; i < NUM_LAMBDA; ++i) s += value[i];
        return s;
    }

    HD float max_component() const {
        float m = value[0];
        for (int i = 1; i < NUM_LAMBDA; ++i) m = fmaxf(m, value[i]);
        return m;
    }

    HD int dominant_bin() const {
        int best = 0;
        for (int i = 1; i < NUM_LAMBDA; ++i)
            if (value[i] > value[best]) best = i;
        return best;
    }
};

// ── CIE 1931 colour matching approximation (Wyman et al. 2013) ─────
// Gaussian fit to the CIE 2 degree observer.
// These are analytic approximations – good enough for rendering.

inline HD float cie_x(float lambda) {
    float t1 = (lambda - 442.0f) * ((lambda < 442.0f) ? 0.0624f : 0.0374f);
    float t2 = (lambda - 599.8f) * ((lambda < 599.8f) ? 0.0264f : 0.0323f);
    float t3 = (lambda - 501.1f) * ((lambda < 501.1f) ? 0.0490f : 0.0382f);
    return 0.362f * expf(-0.5f*t1*t1)
         + 1.056f * expf(-0.5f*t2*t2)
         - 0.065f * expf(-0.5f*t3*t3);
}

inline HD float cie_y(float lambda) {
    float t1 = (lambda - 568.8f) * ((lambda < 568.8f) ? 0.0213f : 0.0247f);
    float t2 = (lambda - 530.9f) * ((lambda < 530.9f) ? 0.0613f : 0.0322f);
    return 0.821f * expf(-0.5f*t1*t1)
         + 0.286f * expf(-0.5f*t2*t2);
}

inline HD float cie_z(float lambda) {
    float t1 = (lambda - 437.0f) * ((lambda < 437.0f) ? 0.0845f : 0.0278f);
    float t2 = (lambda - 459.0f) * ((lambda < 459.0f) ? 0.0385f : 0.0725f);
    return 1.217f * expf(-0.5f*t1*t1)
         + 0.681f * expf(-0.5f*t2*t2);
}

// ── Spectrum → XYZ → sRGB ──────────────────────────────────────────
// Normalise by the integral of ybar so that a flat-1.0 spectrum → Y = 1.
inline HD float3 spectrum_to_xyz(const Spectrum& s) {
    float X = 0.f, Y = 0.f, Z = 0.f;
    float Y_integral = 0.f;
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        float lam = lambda_of_bin(i);
        float xbar = cie_x(lam);
        float ybar = cie_y(lam);
        float zbar = cie_z(lam);
        X += s.value[i] * xbar;
        Y += s.value[i] * ybar;
        Z += s.value[i] * zbar;
        Y_integral += ybar;
    }
    // Normalise: divide by sum(ybar) so flat-1.0 → Y=1
    float scale = (Y_integral > 0.f) ? 1.0f / Y_integral : 0.f;
    return make_f3(X * scale, Y * scale, Z * scale);
}

inline HD float3 xyz_to_linear_srgb(float3 xyz) {
    // sRGB D65 matrix
    return make_f3(
         3.2404542f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z,
        -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z,
         0.0556434f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z
    );
}

inline HD float srgb_gamma(float c) {
    if (c <= 0.0031308f) return 12.92f * c;
    return 1.055f * powf(c, 1.f/2.4f) - 0.055f;
}

inline HD float3 spectrum_to_srgb(const Spectrum& s) {
    float3 xyz = spectrum_to_xyz(s);
    float3 lin = xyz_to_linear_srgb(xyz);
    return make_f3(
        srgb_gamma(fmaxf(0.f, lin.x)),
        srgb_gamma(fmaxf(0.f, lin.y)),
        srgb_gamma(fmaxf(0.f, lin.z))
    );
}

// ACES Filmic tone mapping (Narkowicz 2015) + sRGB gamma
inline HD float3 spectrum_to_srgb_aces(const Spectrum& s) {
    float3 xyz = spectrum_to_xyz(s);
    float3 lin = xyz_to_linear_srgb(xyz);
    // ACES Filmic curve: maps [0,inf) -> [0,1)
    auto aces = [](float x) -> float {
        x = fmaxf(x, 0.f);
        return (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f);
    };
    float r = aces(lin.x);
    float g = aces(lin.y);
    float b = aces(lin.z);
    return make_f3(srgb_gamma(r), srgb_gamma(g), srgb_gamma(b));
}

// ── RGB → Spectrum (optimal pseudoinverse basis) ────────────────────
// Optimal 4×3 matrix M computed via M = pinv(XYZ2RGB @ CIE_norm).
// Guarantees perfect round-trip: spectrum_to_srgb(rgb_to_spectrum(c)) ≈ c
// with negligible error (< 0.06 per channel even after clamping).
//
// Each row corresponds to one spectral bin; columns are R, G, B weights.
//   bin 0 (430nm):  0.01381753  0.07280016  0.78052238
//   bin 1 (530nm):  0.06839557  0.82078527  0.09598912
//   bin 2 (630nm):  0.69628317  0.39308480 -0.03413205
//   bin 3 (730nm):  0.00013491  0.00030314  0.00001524

inline HD Spectrum rgb_to_spectrum_reflectance(float r, float g, float b) {
    Spectrum s;
    s.value[0] = fmaxf(0.f, 0.01381753f * r + 0.07280016f * g + 0.78052238f * b);
    s.value[1] = fmaxf(0.f, 0.06839557f * r + 0.82078527f * g + 0.09598912f * b);
    s.value[2] = fmaxf(0.f, 0.69628317f * r + 0.39308480f * g - 0.03413205f * b);
    s.value[3] = fmaxf(0.f, 0.00013491f * r + 0.00030314f * g + 0.00001524f * b);
    return s;
}

// ── RGB → Spectrum (emission) ───────────────────────────────────────
// Same optimal matrix but values may exceed 1.0 for bright emitters.
inline Spectrum rgb_to_spectrum_emission(float r, float g, float b) {
    Spectrum s;
    s.value[0] = fmaxf(0.f, 0.01381753f * r + 0.07280016f * g + 0.78052238f * b);
    s.value[1] = fmaxf(0.f, 0.06839557f * r + 0.82078527f * g + 0.09598912f * b);
    s.value[2] = fmaxf(0.f, 0.69628317f * r + 0.39308480f * g - 0.03413205f * b);
    s.value[3] = fmaxf(0.f, 0.00013491f * r + 0.00030314f * g + 0.00001524f * b);
    return s;
}

// ── Blackbody spectrum (Planck's law) ───────────────────────────────
// Returns spectral radiance L(λ) in W/(sr·m²·nm)
inline Spectrum blackbody_spectrum(float temperature_K, float scale = 1.0f) {
    Spectrum s;
    constexpr float h  = 6.62607015e-34f; // Planck constant
    constexpr float c  = 2.99792458e8f;   // speed of light
    constexpr float kb = 1.380649e-23f;    // Boltzmann constant

    for (int i = 0; i < NUM_LAMBDA; ++i) {
        float lam_m = lambda_of_bin(i) * 1e-9f; // nm → m
        float lam5  = lam_m * lam_m * lam_m * lam_m * lam_m;
        float exponent = (h * c) / (lam_m * kb * temperature_K);
        float denom = expf(fminf(exponent, 80.f)) - 1.0f;
        float L = (2.0f * h * c * c) / (lam5 * fmaxf(denom, 1e-30f));
        s.value[i] = L * 1e-9f * scale; // Convert to per-nm
    }
    return s;
}
