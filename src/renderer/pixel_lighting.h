#pragma once
// ─────────────────────────────────────────────────────────────────────
// pixel_lighting.h – Central per-pixel lighting composition
// ─────────────────────────────────────────────────────────────────────
// v2.2 consistency reset: every rendering backend (CPU and GPU) fills
// this struct at the single "shade" call site.  No hidden framebuffer
// writes in helpers — all contributions are computed in one place.
//
// Debug AOVs: each field can be written separately for diagnostics.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"

// ── Per-pixel lighting decomposition ────────────────────────────────
struct PixelLighting {
    Spectrum emission;          // Hit emissive surface directly
    Spectrum direct_nee;        // Direct lighting via NEE shadow rays
    Spectrum indirect_global;   // Gather from global photon map
    Spectrum indirect_caustic;  // Gather from caustic photon map
    Spectrum glossy_indirect;   // Glossy continuation bounces (optional)
    Spectrum translucency;      // Photon beams / BSSRDF term (optional)

    // ── Combined radiance (the final pixel value) ───────────────────
    HD Spectrum combined() const {
        return emission + direct_nee + indirect_global + indirect_caustic
             + glossy_indirect + translucency;
    }

    // ── Zero-initialize all fields ──────────────────────────────────
    HD static PixelLighting zero() {
        PixelLighting pl;
        pl.emission         = Spectrum::zero();
        pl.direct_nee       = Spectrum::zero();
        pl.indirect_global  = Spectrum::zero();
        pl.indirect_caustic = Spectrum::zero();
        pl.glossy_indirect  = Spectrum::zero();
        pl.translucency     = Spectrum::zero();
        return pl;
    }

    // ── Accumulate another PixelLighting (for multi-sample averaging) ─
    HD PixelLighting& operator+=(const PixelLighting& o) {
        emission         += o.emission;
        direct_nee       += o.direct_nee;
        indirect_global  += o.indirect_global;
        indirect_caustic += o.indirect_caustic;
        glossy_indirect  += o.glossy_indirect;
        translucency     += o.translucency;
        return *this;
    }
};

// ── Debug AOV component selector ────────────────────────────────────
enum class PixelLightingAOV {
    Combined,
    Emission,
    DirectNEE,
    IndirectGlobal,
    IndirectCaustic,
    GlossyIndirect,
    Translucency
};

inline HD Spectrum pixel_lighting_get_aov(const PixelLighting& pl, PixelLightingAOV aov) {
    switch (aov) {
        case PixelLightingAOV::Combined:        return pl.combined();
        case PixelLightingAOV::Emission:        return pl.emission;
        case PixelLightingAOV::DirectNEE:       return pl.direct_nee;
        case PixelLightingAOV::IndirectGlobal:  return pl.indirect_global;
        case PixelLightingAOV::IndirectCaustic: return pl.indirect_caustic;
        case PixelLightingAOV::GlossyIndirect:  return pl.glossy_indirect;
        case PixelLightingAOV::Translucency:    return pl.translucency;
        default:                                return pl.combined();
    }
}
