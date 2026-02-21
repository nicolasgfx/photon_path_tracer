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

    float         roughness    = 1.0f;   // 0 = mirror, 1 = diffuse
    float         ior          = 1.5f;   // Index of refraction (glass)

    // Texture IDs (−1 = none)
    int           diffuse_tex  = -1;
    int           specular_tex = -1;

    bool is_emissive() const { return Le.max_component() > 0.f; }
    bool is_specular() const {
        return type == MaterialType::Mirror || type == MaterialType::Glass;
    }

    // Mean emission (average over wavelengths) for weighting
    float mean_emission() const { return Le.sum() / NUM_LAMBDA; }
};
