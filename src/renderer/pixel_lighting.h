#pragma once
// ─────────────────────────────────────────────────────────────────────
// pixel_lighting.h – Per-pixel lighting decomposition (v3)
// ─────────────────────────────────────────────────────────────────────
// v3: three-channel decomposition aligned with the path tracer output.
//   combined  = full pixel radiance (emission + direct + indirect)
//   direct    = NEE direct lighting component
//   indirect  = photon-gathered indirect lighting component
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"

struct PixelLighting {
    Spectrum combined;   // Total radiance at this pixel
    Spectrum direct;     // Direct lighting (NEE shadow rays)
    Spectrum indirect;   // Indirect lighting (photon gather)

    // ── Zero-initialize all fields ──────────────────────────────────
    HD static PixelLighting zero() {
        PixelLighting pl;
        pl.combined = Spectrum::zero();
        pl.direct   = Spectrum::zero();
        pl.indirect = Spectrum::zero();
        return pl;
    }

    // ── Accumulate another PixelLighting (for multi-sample averaging) ─
    HD PixelLighting& operator+=(const PixelLighting& o) {
        combined += o.combined;
        direct   += o.direct;
        indirect += o.indirect;
        return *this;
    }
};
