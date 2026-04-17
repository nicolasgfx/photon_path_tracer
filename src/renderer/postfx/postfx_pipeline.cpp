// ─────────────────────────────────────────────────────────────────────
// postfx/postfx_pipeline.cpp – GPU post-processing pipeline (v5)
// ─────────────────────────────────────────────────────────────────────
#include "postfx/postfx_pipeline.h"
#include "postfx/bloom.h"
#include "postfx/firefly_filter.h"
#include "postfx/tonemap.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>

// ── Lifecycle ───────────────────────────────────────────────────────

PostFxPipeline::~PostFxPipeline() { cleanup(); }

void PostFxPipeline::init(int width, int height) {
    if (width == alloc_w_ && height == alloc_h_ && d_hdr_) return;
    cleanup();

    alloc_w_ = width;
    alloc_h_ = height;

    // Working HDR float4 buffer
    size_t hdr_bytes = (size_t)width * height * 4 * sizeof(float);
    cudaMalloc(&d_hdr_, hdr_bytes);

    // Firefly filter scratch (same size as HDR)
    cudaMalloc(&d_firefly_temp_, hdr_bytes);

    // Bloom mip chain
    int w = std::max(width  / 2, 1);
    int h = std::max(height / 2, 1);
    for (int i = 0; i < NUM_MIP_LEVELS; ++i) {
        mip_w_[i] = w;
        mip_h_[i] = h;
        size_t bytes = (size_t)w * h * 4 * sizeof(float);
        cudaMalloc(&d_mip_[i],     bytes);
        cudaMalloc(&d_mip_tmp_[i], bytes);
        cudaMemset(d_mip_[i],     0, bytes);
        cudaMemset(d_mip_tmp_[i], 0, bytes);
        w = std::max(w / 2, 1);
        h = std::max(h / 2, 1);
    }

    cudaMalloc(&d_max_lum_, sizeof(float));
}

void PostFxPipeline::cleanup() {
    if (d_hdr_)          { cudaFree(d_hdr_);          d_hdr_ = nullptr; }
    if (d_firefly_temp_) { cudaFree(d_firefly_temp_); d_firefly_temp_ = nullptr; }
    if (d_max_lum_)      { cudaFree(d_max_lum_);      d_max_lum_ = nullptr; }

    for (int i = 0; i < NUM_MIP_LEVELS; ++i) {
        if (d_mip_[i])     { cudaFree(d_mip_[i]);     d_mip_[i] = nullptr; }
        if (d_mip_tmp_[i]) { cudaFree(d_mip_tmp_[i]); d_mip_tmp_[i] = nullptr; }
        mip_w_[i] = mip_h_[i] = 0;
    }

    alloc_w_ = alloc_h_ = 0;
}

// ── Main entry point ────────────────────────────────────────────────

void PostFxPipeline::apply(
    const float* d_color_r,
    const float* d_color_g,
    const float* d_color_b,
    const float* d_sample_cnt,
    uint8_t* d_srgb_out,
    float* d_hdr_out,
    int width, int height,
    const PostFxParams& params)
{
    init(width, height);

    // 1. RGB SoA accumulators → HDR float4
    launch_rgb_to_hdr(d_color_r, d_color_g, d_color_b,
                      d_sample_cnt, d_hdr_,
                      width, height, params.exposure,
                      params.caustic_r, params.caustic_g, params.caustic_b,
                      params.caustic_frames, params.caustic_only);

    // 2. Firefly filter (pre-denoiser outlier suppression)
    if (params.firefly_enabled)
        apply_firefly_(d_hdr_, width, height, params);

    // 3. Bloom
    if (params.bloom_enabled)
        apply_bloom_(d_hdr_, width, height, params);

    // 4. Copy HDR out for denoiser if requested
    if (d_hdr_out) {
        cudaMemcpy(d_hdr_out, d_hdr_,
                   (size_t)width * height * 4 * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }

    // 5. Tonemap → sRGB
    launch_tonemap_hdr(d_hdr_, d_srgb_out, width, height, params.use_aces);

    // No explicit sync needed — downstream cudaMemcpy(D2H) serializes
    // on the default stream.
}

void PostFxPipeline::apply_pre_denoise(
    float* d_hdr, int width, int height,
    const PostFxParams& params)
{
    if (params.firefly_enabled)
        apply_firefly_(d_hdr, width, height, params);
}

    // ── Two-phase pipeline (denoiser-aware) ────────────────────────────

    float* PostFxPipeline::apply_phase1(
        const float* d_color_r, const float* d_color_g,
        const float* d_color_b, const float* d_sample_cnt,
        int width, int height, const PostFxParams& params)
    {
        init(width, height);

        // RGB SoA accum → HDR float4
        launch_rgb_to_hdr(d_color_r, d_color_g, d_color_b,
                          d_sample_cnt, d_hdr_,
                          width, height, params.exposure,
                          params.caustic_r, params.caustic_g, params.caustic_b,
                          params.caustic_frames, params.caustic_only);

        // Firefly filter (pre-denoiser outlier suppression)
        if (params.firefly_enabled)
            apply_firefly_(d_hdr_, width, height, params);

        return d_hdr_;  // caller will run denoiser on this buffer
    }

    void PostFxPipeline::apply_phase2(
        uint8_t* d_srgb_out, int width, int height,
        const PostFxParams& params)
    {
        // Bloom (post-denoiser)
        if (params.bloom_enabled)
            apply_bloom_(d_hdr_, width, height, params);

        // Tonemap → sRGB
        launch_tonemap_hdr(d_hdr_, d_srgb_out, width, height, params.use_aces);

        // No explicit sync — downstream cudaMemcpy(D2H) serializes.
    }

// ── Firefly filter ──────────────────────────────────────────────────

void PostFxPipeline::apply_firefly_(
    float* d_hdr, int width, int height,
    const PostFxParams& params)
{
    init(width, height);
    launch_firefly_filter(d_hdr, d_firefly_temp_, width, height,
                          params.firefly_radius, params.firefly_threshold);
}

// ── Bloom ────────────────────────────────────────────────────────────

void PostFxPipeline::apply_bloom_(
    float* d_hdr, int width, int height,
    const PostFxParams& params)
{
    init(width, height);

    // 1. Find max luminance (result stays on device — no host round-trip)
    launch_bloom_find_max_luminance(d_hdr, d_max_lum_, width, height);

    // 2. Adaptive bright extract reads d_max_lum_ on-device
    launch_bloom_bright_extract_adaptive(d_hdr, d_mip_[0], d_max_lum_,
                                         width, height,
                                         params.bloom_scene_min_Le,
                                         params.bloom_scene_max_Le);

    // 3. Downsample chain
    for (int i = 1; i < NUM_MIP_LEVELS; ++i) {
        launch_bloom_downsample(d_mip_[i - 1], d_mip_[i],
                                mip_w_[i - 1], mip_h_[i - 1]);
    }

    // 4. Separable Gaussian blur at each mip level
    for (int i = 0; i < NUM_MIP_LEVELS; ++i) {
        int w = mip_w_[i];
        int h = mip_h_[i];

        float radius_h = params.bloom_radius_h / (float)(1 << (i + 1));
        float radius_v = params.bloom_radius_v / (float)(1 << (i + 1));
        radius_h = fmaxf(radius_h, 1.0f);
        radius_v = fmaxf(radius_v, 1.0f);

        launch_bloom_blur_h(d_mip_[i],     d_mip_tmp_[i], w, h, radius_h);
        launch_bloom_blur_v(d_mip_tmp_[i], d_mip_[i],     w, h, radius_v);
    }

    // 5. Upsample-accumulate
    for (int i = NUM_MIP_LEVELS - 1; i >= 1; --i) {
        launch_bloom_upsample_accumulate(d_mip_[i], d_mip_[i - 1],
                                         mip_w_[i - 1], mip_h_[i - 1]);
    }

    // 6. Composite bloom onto HDR
    launch_bloom_composite(d_hdr, d_mip_[0], width, height,
                           params.bloom_intensity);
}
