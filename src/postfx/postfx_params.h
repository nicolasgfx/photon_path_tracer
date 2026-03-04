#pragma once
// ─────────────────────────────────────────────────────────────────────
// postfx/postfx_params.h – Post-processing effect parameters
// ─────────────────────────────────────────────────────────────────────
// Lightweight struct holding all post-FX tunables, stored per-scene
// in saved_camera.json.  New effects add fields here.
// ─────────────────────────────────────────────────────────────────────
#include "core/config.h"

struct PostFxParams {
    // ── Bloom / glow ────────────────────────────────────────────────
    bool  bloom_enabled    = DEFAULT_BLOOM_ENABLED;
    float bloom_intensity  = DEFAULT_BLOOM_INTENSITY;   // additive strength
    float bloom_radius_h   = DEFAULT_BLOOM_RADIUS_H;    // horizontal kernel radius (pixels at full res)
    float bloom_radius_v   = DEFAULT_BLOOM_RADIUS_V;    // vertical kernel radius (pixels at full res)

    // Future effects (vignette, chromatic aberration, grain, …)
    // would add fields here.
};
