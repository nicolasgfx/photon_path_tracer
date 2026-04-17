#pragma once
// ─────────────────────────────────────────────────────────────────────
// postfx/postfx_params.h – Post-processing parameters (v5 RGB pipeline)
// ─────────────────────────────────────────────────────────────────────
#include "core/config.h"

struct PostFxParams {
    // ── Bloom ────────────────────────────────────────────────────────
    bool  bloom_enabled     = DEFAULT_BLOOM_ENABLED;
    float bloom_intensity   = DEFAULT_BLOOM_INTENSITY;
    float bloom_radius_h    = DEFAULT_BLOOM_RADIUS_H;
    float bloom_radius_v    = DEFAULT_BLOOM_RADIUS_V;
    float bloom_scene_min_Le = 0.f;   // min emissive luminance (from scene scan)
    float bloom_scene_max_Le = 0.f;   // max emissive luminance (from scene scan)

    // ── Firefly filter (pre-denoiser outlier suppression) ────────────
    bool  firefly_enabled   = DEFAULT_FIREFLY_FILTER_ENABLED;
    int   firefly_radius    = FIREFLY_FILTER_RADIUS;
    float firefly_threshold = FIREFLY_FILTER_THRESHOLD;

    // ── Tonemap / exposure ───────────────────────────────────────────
    float exposure          = DEFAULT_EXPOSURE;
    bool  use_aces          = USE_ACES_TONEMAPPING;

    // -- Denoiser blend (0=full denoise, 1=passthrough)
    float denoiser_blend    = DEFAULT_DENOISER_BLEND;

    // -- Caustic composition (set by RenderSession before apply_postfx) --
    const float* caustic_r  = nullptr;  // device SoA [w*h] (nullable)
    const float* caustic_g  = nullptr;
    const float* caustic_b  = nullptr;
    int   caustic_frames    = 0;        // frames accumulated so far
    bool  caustic_only      = false;    // debug: show caustic buffer only
};
