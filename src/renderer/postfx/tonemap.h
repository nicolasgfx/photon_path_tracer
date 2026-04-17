#pragma once
// ─────────────────────────────────────────────────────────────────────
// postfx/tonemap.h – Tonemapping kernels (v5 RGB pipeline)
//
// Two-stage pipeline:
//   1. rgb_to_hdr: color SoA channels (float accum / sample_count) → HDR float4
//   2. tonemap_hdr: HDR float4 → sRGB uint8 (ACES + gamma)
// ─────────────────────────────────────────────────────────────────────

// Convert RGB SoA accumulator buffers to HDR float4 with exposure.
//   color_r/g/b: [width*height] per-channel accumulators (SoA)
//   sample_cnt:  [width*height] per-pixel sample counts (nullable → use 1)
//   d_hdr:       [width*height*4] output float4 HDR
void launch_rgb_to_hdr(
    const float* d_color_r,
    const float* d_color_g,
    const float* d_color_b,
    const float* d_sample_cnt,
    float* d_hdr,
    int width, int height,
    float exposure,
    const float* d_caustic_r = nullptr,
    const float* d_caustic_g = nullptr,
    const float* d_caustic_b = nullptr,
    int caustic_frames = 0,
    bool caustic_only = false);

// Tonemap HDR float4 → sRGB uint8 output.
//   d_hdr:  [width*height*4] input HDR (post-bloom, post-firefly)
//   d_srgb: [width*height*4] output RGBA8
void launch_tonemap_hdr(
    const float* d_hdr,
    uint8_t* d_srgb,
    int width, int height,
    bool use_aces);
