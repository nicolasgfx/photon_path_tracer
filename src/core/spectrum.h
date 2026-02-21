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

constexpr int   NUM_LAMBDA    = 32;
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

// ── ACES Filmic Tone Mapping (§Q8) ─────────────────────────────────
// Attempt to replace Reinhard with better highlight handling.
// Narkowicz 2015 ACES fit.
inline HD float aces_filmic(float x) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

inline HD float3 aces_tonemap(float3 color) {
    return make_f3(
        aces_filmic(fmaxf(0.f, color.x)),
        aces_filmic(fmaxf(0.f, color.y)),
        aces_filmic(fmaxf(0.f, color.z))
    );
}

// Spectrum → sRGB with ACES tone mapping
inline HD float3 spectrum_to_srgb_aces(const Spectrum& s) {
    float3 xyz = spectrum_to_xyz(s);
    float3 lin = xyz_to_linear_srgb(xyz);
    float3 mapped = aces_tonemap(lin);
    return make_f3(
        srgb_gamma(fmaxf(0.f, fminf(1.f, mapped.x))),
        srgb_gamma(fmaxf(0.f, fminf(1.f, mapped.y))),
        srgb_gamma(fmaxf(0.f, fminf(1.f, mapped.z)))
    );
}

// ── RGB → Spectrum (Smits-style Gaussian basis) ────────────────────
// Reconstruct a plausible spectral reflectance from RGB.
// Uses narrow Gaussians to minimise inter-channel crosstalk.
inline HD Spectrum rgb_to_spectrum_reflectance(float r, float g, float b) {
    Spectrum s = Spectrum::zero();
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        float lam = lambda_of_bin(i);
        // Narrow Gaussians to reduce yellow crosstalk between red & green
        float dr = (lam - 630.f) / 28.f;
        float dg = (lam - 532.f) / 28.f;
        float db = (lam - 460.f) / 24.f;
        float fr = expf(-0.5f * dr * dr);
        float fg = expf(-0.5f * dg * dg);
        float fb = expf(-0.5f * db * db);

        // Normalize so that white (1,1,1) → flat spectrum
        float sum = fr + fg + fb;
        if (sum < 0.01f) sum = 0.01f;

        s.value[i] = fmaxf(0.f, (r * fr + g * fg + b * fb) / sum);
    }
    return s;
}

// ── RGB → Spectrum (emission) ───────────────────────────────────────
// For emitters: use the SAME white-normalised spectral shape as
// reflectance, then scale by the apparent luminosity.  This keeps the
// spectral profile smooth (no narrow peaks from raw Gaussians) while
// preserving absolute intensity — values can exceed 1.0.
inline Spectrum rgb_to_spectrum_emission(float r, float g, float b) {
    // Apparent luminosity (Rec.709 luminance)
    float Y = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    if (Y <= 0.f) return Spectrum::zero();

    // Use the normalised reflectance basis for shape …
    Spectrum shape = rgb_to_spectrum_reflectance(
        r / Y, g / Y, b / Y);      // chromaticity (sums to ~constant)

    // … then scale by luminosity so brightness is correct.
    for (int i = 0; i < NUM_LAMBDA; ++i)
        shape.value[i] *= Y;

    return shape;
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
