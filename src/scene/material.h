#pragma once
// ─────────────────────────────────────────────────────────────────────
// material.h – Spectral material definition
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"
#include <string>
#include <cstdint>

enum class MaterialType : uint8_t {
    Lambertian,         // Pure diffuse
    Mirror,             // Perfect specular reflection
    Glass,              // Specular transmission + reflection (dielectric)
    GlossyMetal,        // Rough specular (Cook-Torrance, metallic Fresnel: F0 = Ks)
    Emissive,           // Area light
    GlossyDielectric,   // Cook-Torrance + Lambertian (dielectric Fresnel: F0 from IOR)
    Translucent,        // Surface BSDF + interior participating medium
    Clearcoat,          // Layered: dielectric coat over a base BRDF
    Fabric              // Diffuse + sheen lobe (cloth)
};

// ── pb_brdf model tag (parsed from MTL) ─────────────────────────────
enum class PbBrdf : uint8_t {
    None,         // not specified in MTL
    Lambert,
    Dielectric,
    Conductor,
    Clearcoat,
    Emissive,
    Fabric
};

// ── pb_semantic hint tag ────────────────────────────────────────────
enum class PbSemantic : uint8_t {
    None,
    Subsurface,
    Glass,
    Metal,
    Fabric,
    Leather,
    WoodNatural,
    WoodPainted,
    Wallpaper,
    Stone,
    Plastic
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

    // Interior participating medium (Translucent materials)
    // medium_id >= 0 references a HomogeneousMedium in the scene's medium table.
    // -1 = no interior medium.
    int           medium_id    = -1;

    // ── Photon-Beam Material Extensions (pb_*) ──────────────────────
    // Parsed from MTL; applied as overrides during post-processing.
    // Fields with _set == true were explicitly specified in the MTL.

    PbBrdf        pb_brdf         = PbBrdf::None;
    PbSemantic    pb_semantic     = PbSemantic::None;

    // Surface roughness / anisotropy
    float         pb_roughness    = -1.f;   // <0 = not set
    float         pb_anisotropy   = 0.f;
    float         pb_roughness_x  = -1.f;   // <0 = not set
    float         pb_roughness_y  = -1.f;   // <0 = not set
    bool          pb_roughness_set    = false;
    bool          pb_anisotropy_set   = false;
    bool          pb_roughness_xy_set = false;

    // IOR
    float         pb_eta          = -1.f;
    bool          pb_eta_set      = false;

    // Conductor complex IOR (RGB → spectral)
    float         pb_conductor_eta_rgb[3] = {0.f, 0.f, 0.f};
    float         pb_conductor_k_rgb[3]   = {0.f, 0.f, 0.f};
    Spectrum      pb_conductor_eta_spec   = Spectrum::zero();
    Spectrum      pb_conductor_k_spec     = Spectrum::zero();
    bool          pb_conductor_set  = false;

    // Transmission / thin materials
    float         pb_transmission   = -1.f;   // <0 = not set
    bool          pb_transmission_set = false;
    bool          pb_thin           = false;
    float         pb_thickness      = 0.001f;  // metres, default 1mm

    // Clearcoat layering
    float         pb_clearcoat            = 1.0f;
    float         pb_clearcoat_roughness  = -1.f;  // <0 = not set
    PbBrdf        pb_base_brdf            = PbBrdf::Lambert;
    float         pb_base_roughness       = -1.f;
    bool          pb_clearcoat_set        = false;

    // Fabric sheen
    float         pb_sheen         = 0.f;
    float         pb_sheen_tint    = 0.f;
    bool          pb_sheen_set     = false;

    // Volumetric / participating medium (pb_medium homogeneous)
    bool          pb_medium_enabled = false;
    float         pb_density        = 1.0f;
    float         pb_sigma_a_rgb[3] = {0.f, 0.f, 0.f};
    float         pb_sigma_s_rgb[3] = {0.f, 0.f, 0.f};
    float         pb_g              = 0.0f;
    bool          pb_sigma_a_set    = false;
    bool          pb_sigma_s_set    = false;

    // Chromatic dispersion (Cauchy B coefficient, nm²)
    // pb_dispersion <cauchy_b> enables dispersion and sets cauchy_B.
    // cauchy_A is auto-derived so that n(589nm) == ior.
    float         pb_dispersion_B    = -1.f;   // <0 = not set
    bool          pb_dispersion_set  = false;

    // Scene scale hint
    float         pb_meters_per_unit = 1.0f;

    // ── Helper queries ──────────────────────────────────────────────

    bool is_emissive() const { return Le.max_component() > 0.f || emission_tex >= 0; }
    bool is_specular() const {
        return type == MaterialType::Mirror || type == MaterialType::Glass
            || type == MaterialType::Translucent;
    }

    /// True if this material has an interior participating medium
    bool has_medium() const { return medium_id >= 0; }

    // Mean emission (average over wavelengths) for weighting
    float mean_emission() const { return Le.sum() / NUM_LAMBDA; }

    // Per-wavelength IOR via Cauchy equation
    HD float ior_at_lambda(float lambda_nm) const {
        if (!dispersion) return ior;
        return cauchy_A + cauchy_B / (lambda_nm * lambda_nm);
    }
};
