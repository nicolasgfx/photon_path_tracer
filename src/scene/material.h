#pragma once
// ─────────────────────────────────────────────────────────────────────
// material.h – RGB material definition (v5)
//
// Ported from v4 spectral pipeline: all Spectrum fields replaced with
// Color3 (linear RGB).  Dispersion IOR is kept scalar (no spectral).
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/color.h"
#include <string>
#include <cstdint>

// MaterialType is a shared plain enum defined in core/types.h.

// ── pb_brdf model tag (parsed from MTL / scene files) ──────────────────────
enum class PbBrdf : uint8_t {
    None,         // not specified
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

    // ── RGB reflectance / albedo ────────────────────────────────────
    Color3        Kd           = Color3::constant(0.5f);    // diffuse albedo
    Color3        Ks           = Color3::zero();             // specular reflectance
    Color3        Le           = Color3::zero();             // emission (linear RGB radiance)
    Color3        Tf           = Color3::one();              // transmittance filter (glass colour)

    float         roughness    = 1.0f;   // 0 = mirror, 1 = diffuse
    float         ior          = 1.5f;   // index of refraction (glass)

    // ── Chromatic dispersion (Cauchy equation) ──────────────────────
    // n(λ) = cauchy_A + cauchy_B / λ²   (λ in nm)
    float         cauchy_A     = 1.5046f;
    float         cauchy_B     = 4200.0f;
    bool          dispersion   = false;

    float         opacity      = 1.0f;   // 1 = opaque, 0 = transparent

    // Texture IDs (-1 = none)
    int           diffuse_tex  = -1;
    int           specular_tex = -1;
    int           alpha_tex    = -1;
    int           emission_tex = -1;
    int           bump_tex     = -1;     // height-map bump
    int           normal_tex   = -1;     // tangent-space normal map

    // Displacement (stored for future tessellation/ray marching)
    int           displacement_tex   = -1;
    float         displacement_scale = 1.0f;

    // Interior participating medium index (-1 = none)
    int           medium_id    = -1;

    // ── Photon-Beam Material Extensions (pb_*) ──────────────────────
    PbBrdf        pb_brdf         = PbBrdf::None;
    PbSemantic    pb_semantic     = PbSemantic::None;

    // Surface roughness / anisotropy
    float         pb_roughness    = -1.f;
    float         pb_anisotropy   = 0.f;
    float         pb_roughness_x  = -1.f;
    float         pb_roughness_y  = -1.f;
    bool          pb_roughness_set    = false;
    bool          pb_anisotropy_set   = false;
    bool          pb_roughness_xy_set = false;

    // IOR
    float         pb_eta          = -1.f;
    bool          pb_eta_set      = false;

    // Conductor complex IOR (RGB)
    float         pb_conductor_eta_rgb[3] = {0.f, 0.f, 0.f};
    float         pb_conductor_k_rgb[3]   = {0.f, 0.f, 0.f};
    Color3        pb_conductor_eta_spec   = Color3::zero();
    Color3        pb_conductor_k_spec     = Color3::zero();
    bool          pb_conductor_set  = false;

    // Transmission / thin materials
    float         pb_transmission   = -1.f;
    bool          pb_transmission_set = false;
    bool          pb_thin           = false;
    float         pb_thickness      = 0.001f;  // metres

    // Clearcoat layering
    float         pb_clearcoat            = 1.0f;
    float         pb_clearcoat_roughness  = -1.f;
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

    // Direct transmittance override (RGB, replaces spectral pb_tf_spectrum)
    Color3        pb_tf_color       = Color3::one();
    bool          pb_tf_color_set   = false;

    // Chromatic dispersion (Cauchy B coefficient, nm²)
    float         pb_dispersion_B    = -1.f;
    bool          pb_dispersion_set  = false;

    // Scene scale hint
    float         pb_meters_per_unit = 1.0f;

    // ── Helper queries ──────────────────────────────────────────────

    bool is_emissive() const { return Le.max_component() > 0.f || emission_tex >= 0; }

    bool is_specular() const {
        return type == MaterialType::Mirror
            || type == MaterialType::Glass
            || type == MaterialType::Translucent;
    }

    bool has_medium() const { return medium_id >= 0; }

    // Mean emission (average of RGB channels) for weighting
    float mean_emission() const { return Le.sum() / 3.0f; }

    // Per-wavelength IOR via Cauchy equation (kept for refractive paths)
    HD float ior_at_lambda(float lambda_nm) const {
        if (!dispersion) return ior;
        return cauchy_A + cauchy_B / (lambda_nm * lambda_nm);
    }
};
