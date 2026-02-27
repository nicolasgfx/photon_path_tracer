#pragma once
// ─────────────────────────────────────────────────────────────────────
// material_flags.h – Canonical material classification for photon mapping
// ─────────────────────────────────────────────────────────────────────
// Shared between CPU and GPU.  Provides a single source of truth for:
//   - Is this material a delta/specular transport?
//   - Is it a caustic caster (Glass, Translucent)?
//   - Is it emissive?
//
// v2.2 consistency reset: both CPU photon tracer and GPU shader must
// use these classifications identically.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"

// Only include full Material on CPU (CUDA device compilation can't use std::string)
#ifndef __CUDACC__
#include "scene/material.h"
#endif

// ── Material classification enum ────────────────────────────────────
enum class MaterialClass : uint8_t {
    Diffuse,        // Lambertian, Emissive (non-delta)
    Glossy,         // Non-delta microfacet (GlossyMetal, GlossyDielectric, Clearcoat, Fabric)
    Specular,       // Delta reflection (Mirror)
    Transparent,    // Delta transmission/reflection (Glass)
    Translucent,    // Surface BSDF + interior medium
    Emissive        // Area light (also classified as Diffuse for bouncing)
};

// ── Material flags for photon mapping decisions ─────────────────────
struct MaterialFlags {
    bool is_emissive;      // Surface emits light
    bool is_delta;         // Delta BSDF (mirror, glass, translucent boundary)
    bool caustic_caster;   // Photons through this material create caustics
};

// ── Classify by MaterialType enum value (works on both CPU and GPU) ──
// Uses uint8_t to match the serialized mat_type[] buffer.
// MaterialType enum values (must match material.h order):
//   0=Lambertian, 1=Mirror, 2=Glass, 3=GlossyMetal, 4=Emissive,
//   5=GlossyDielectric, 6=Translucent, 7=Clearcoat, 8=Fabric
inline HD MaterialFlags classify_for_photons_by_type(uint8_t mat_type_val) {
    MaterialFlags f{};
    f.is_emissive   = (mat_type_val == 4);
    f.is_delta      = (mat_type_val == 1 || mat_type_val == 2 || mat_type_val == 6);
    f.caustic_caster = (mat_type_val == 1 || mat_type_val == 2 || mat_type_val == 6);  // Mirror, Glass, Translucent
    return f;
}

// ── CPU-only functions that use the full Material struct ─────────────
#ifndef __CUDACC__

// Canonical classification function (CPU).
// This is THE source of truth.  GPU code uses classify_for_photons_by_type
// which must produce identical results.
inline MaterialFlags classify_for_photons(const Material& m) {
    return classify_for_photons_by_type(static_cast<uint8_t>(m.type));
}

// Convenience: classify by MaterialClass
inline MaterialClass get_material_class(const Material& m) {
    switch (m.type) {
        case MaterialType::Lambertian:       return MaterialClass::Diffuse;
        case MaterialType::Mirror:           return MaterialClass::Specular;
        case MaterialType::Glass:            return MaterialClass::Transparent;
        case MaterialType::GlossyMetal:      return MaterialClass::Glossy;
        case MaterialType::Emissive:         return MaterialClass::Emissive;
        case MaterialType::GlossyDielectric: return MaterialClass::Glossy;
        case MaterialType::Translucent:      return MaterialClass::Translucent;
        case MaterialType::Clearcoat:        return MaterialClass::Glossy;
        case MaterialType::Fabric:           return MaterialClass::Glossy;
        default:                             return MaterialClass::Diffuse;
    }
}

#endif // __CUDACC__
