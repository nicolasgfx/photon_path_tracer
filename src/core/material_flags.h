#pragma once
// ─────────────────────────────────────────────────────────────────────
// material_flags.h – Canonical material classification for photon mapping
//
// Shared between CPU and GPU.  Single source of truth for delta/
// caustic/emissive classification.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"

enum class MaterialClass : uint8_t {
    Diffuse,
    Glossy,
    Specular,
    Transparent,
    Translucent,
    Emissive
};

struct MaterialFlags {
    bool is_emissive;
    bool is_delta;
    bool caustic_caster;
};

// Classify by MaterialType enum value (works on both CPU and GPU).
inline HD MaterialFlags classify_for_photons_by_type(uint8_t mat_type_val) {
    MaterialFlags f{};
    f.is_emissive    = (mat_type_val == 4);
    f.is_delta       = (mat_type_val == 1 || mat_type_val == 2 || mat_type_val == 6);
    f.caustic_caster = (mat_type_val == 1 || mat_type_val == 2 || mat_type_val == 6);
    return f;
}
