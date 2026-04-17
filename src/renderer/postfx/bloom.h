#pragma once
// ─────────────────────────────────────────────────────────────────────
// postfx/bloom.h – GPU bloom kernel declarations (v5)
//
// Multi-scale mip-chain bloom: bright-extract → downsample chain →
// separable Gaussian blur per mip → upsample-accumulate → composite.
// All kernels operate on float4 linear HDR data.
// ─────────────────────────────────────────────────────────────────────

// 1. Parallel reduction to find max luminance in the HDR buffer.
void launch_bloom_find_max_luminance(
    const float* d_hdr, float* d_max_lum,
    int width, int height);

// 2. Bright-pass extract with adaptive ramp (full-res → half-res).
void launch_bloom_bright_extract(
    const float* d_hdr, float* d_mip0,
    int src_w, int src_h,
    float lo_threshold, float hi_threshold);

// 2b. Adaptive bright-extract: reads max_lum from device pointer,
//     computes thresholds on-GPU (avoids host round-trip sync).
void launch_bloom_bright_extract_adaptive(
    const float* d_hdr, float* d_mip0,
    const float* d_max_lum,
    int src_w, int src_h,
    float scene_min_Le, float scene_max_Le);

// 3. Bilinear 2× downsample.
void launch_bloom_downsample(
    const float* d_src, float* d_dst,
    int src_w, int src_h);

// 4. Separable Gaussian blur (horizontal / vertical).
void launch_bloom_blur_h(
    const float* d_src, float* d_dst,
    int w, int h, float radius);

void launch_bloom_blur_v(
    const float* d_src, float* d_dst,
    int w, int h, float radius);

// 5. Bilinear 2× upsample + additive accumulate.
void launch_bloom_upsample_accumulate(
    const float* d_src, float* d_dst,
    int dst_w, int dst_h);

// 6. Final composite: add upsampled bloom onto full-res HDR.
void launch_bloom_composite(
    float* d_hdr, const float* d_bloom,
    int width, int height, float intensity);
