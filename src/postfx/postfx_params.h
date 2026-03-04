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

    // Adaptive bloom: scene emissive luminance range (set automatically
    // by scanning all emissive triangles at scene load — NOT user-tunable).
    // min_Le > 0 activates the adaptive ramp; otherwise falls back to
    // the simple 25%-of-peak threshold.
    float bloom_scene_min_Le = 0.f;  // dimmest emissive material radiance
    float bloom_scene_max_Le = 0.f;  // brightest emissive material radiance

    // Future effects (vignette, chromatic aberration, grain, …)
    // would add fields here.
};
