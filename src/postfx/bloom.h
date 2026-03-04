#pragma once
// ─────────────────────────────────────────────────────────────────────
// postfx/bloom.h – GPU bloom kernel declarations
// ─────────────────────────────────────────────────────────────────────
#include <cstdint>

/// Find the maximum luminance in the HDR buffer via parallel reduction.
/// Writes the single float result to d_max_lum[0].
void launch_bloom_find_max_luminance(
    const float* d_hdr,         // [W*H*4] float4
    float*       d_max_lum,     // [1] output
    int width, int height);

/// Extract bright pixels (luminance > threshold) into half-resolution
/// scratch buffer.  Excess brightness (lum - threshold) drives the
/// bloom intensity so dimmer lights get less glow.
void launch_bloom_bright_extract(
    const float* d_hdr,         // [W*H*4] full-res input
    float*       d_mip0,        // [W/2 * H/2 * 4] half-res output
    int src_w, int src_h,
    float threshold);

/// Bilinear 2× downsample: src → dst (each half the previous size).
void launch_bloom_downsample(
    const float* d_src, float* d_dst,
    int src_w, int src_h);

/// Separable Gaussian blur – horizontal pass.
void launch_bloom_blur_h(
    const float* d_src, float* d_dst,
    int w, int h, float radius);

/// Separable Gaussian blur – vertical pass.
void launch_bloom_blur_v(
    const float* d_src, float* d_dst,
    int w, int h, float radius);

/// Bilinear 2× upsample and additively accumulate into dst.
void launch_bloom_upsample_accumulate(
    const float* d_src, float* d_dst,
    int dst_w, int dst_h);

/// Additive composite: hdr[i] += bloom[i] * intensity.
/// bloom is at half resolution and is upsampled bilinearly on the fly.
void launch_bloom_composite(
    float*       d_hdr,         // [W*H*4] modified in place
    const float* d_bloom,       // [W/2 * H/2 * 4]
    int width, int height,
    float intensity);
