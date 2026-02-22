#pragma once
// ─────────────────────────────────────────────────────────────────────
// material.h – Spectral material definition
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"
#include <string>

enum class MaterialType : uint8_t {
    Lambertian,         // Pure diffuse
    Mirror,             // Perfect specular reflection
    Glass,              // Specular transmission + reflection (dielectric)
    GlossyMetal,        // Rough specular (Cook-Torrance, metallic Fresnel: F0 = Ks)
    Emissive,           // Area light
    GlossyDielectric    // Cook-Torrance + Lambertian (dielectric Fresnel: F0 from IOR)
};

struct Material {
    std::string   name;
    MaterialType  type         = MaterialType::Lambertian;

    // Spectral reflectance / albedo
    Spectrum      Kd           = Spectrum::constant(0.5f);
    Spectrum      Ks           = Spectrum::zero();       // Specular reflectance
    Spectrum      Le           = Spectrum::zero();       // Emission (spectral radiance)
    Spectrum      Tf           = Spectrum::constant(1.0f);  // Transmittance filter (glass colour)

    float         roughness    = 1.0f;   // 0 = mirror, 1 = diffuse
    float         ior          = 1.5f;   // Index of refraction (glass)

    // ── Chromatic dispersion (Cauchy equation) ──────────────────────
    // n(λ) = cauchy_A + cauchy_B / λ²   (λ in nm)
    // When dispersion == false, the constant `ior` is used.
    float         cauchy_A     = 1.5046f;  // base refractive index (crown glass)
    float         cauchy_B     = 4200.0f;  // dispersion coefficient (nm²)
    bool          dispersion   = false;    // enable wavelength-dependent IOR

    // Opacity: 1.0 = fully opaque, 0.0 = fully transparent
    // Maps to MTL "d" (dissolve).  "Tr" = 1 - d.
    float         opacity      = 1.0f;

    // Texture IDs (−1 = none)
    int           diffuse_tex  = -1;
    int           specular_tex = -1;   // map_Ks: per-texel specular
    int           alpha_tex    = -1;   // map_d:  alpha mask texture
    int           emission_tex = -1;   // map_Ke: emission texture
    int           bump_tex     = -1;   // map_bump / bump: bump map

    bool is_emissive() const { return Le.max_component() > 0.f || emission_tex >= 0; }
    bool is_specular() const {
        return type == MaterialType::Mirror || type == MaterialType::Glass;
    }

    // Mean emission (average over wavelengths) for weighting
    float mean_emission() const { return Le.sum() / NUM_LAMBDA; }

    // Per-wavelength IOR via Cauchy equation
    HD float ior_at_lambda(float lambda_nm) const {
        if (!dispersion) return ior;
        return cauchy_A + cauchy_B / (lambda_nm * lambda_nm);
    }
};
