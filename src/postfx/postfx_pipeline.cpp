// ─────────────────────────────────────────────────────────────────────
// postfx/postfx_pipeline.cpp – GPU post-processing pipeline
// ─────────────────────────────────────────────────────────────────────
#include "postfx/postfx_pipeline.h"
#include "postfx/bloom.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>

// ── Lifecycle ───────────────────────────────────────────────────────

PostFxPipeline::~PostFxPipeline() { cleanup(); }

void PostFxPipeline::init(int width, int height) {
    if (width == alloc_w_ && height == alloc_h_ && d_mip_[0]) return;
    cleanup();

    alloc_w_ = width;
    alloc_h_ = height;

    // Build mip dimensions: each level is half the previous
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
    for (int i = 0; i < NUM_MIP_LEVELS; ++i) {
        if (d_mip_[i])     { cudaFree(d_mip_[i]);     d_mip_[i] = nullptr; }
        if (d_mip_tmp_[i]) { cudaFree(d_mip_tmp_[i]); d_mip_tmp_[i] = nullptr; }
        mip_w_[i] = mip_h_[i] = 0;
    }
    if (d_max_lum_) { cudaFree(d_max_lum_); d_max_lum_ = nullptr; }
    alloc_w_ = alloc_h_ = 0;
}

// ── Main entry point ────────────────────────────────────────────────

void PostFxPipeline::apply(float* d_hdr, int width, int height,
                           const PostFxParams& params) {
    if (params.bloom_enabled)
        apply_bloom_(d_hdr, width, height, params);

    // Future effects would be called here:
    // if (params.vignette_enabled) apply_vignette_(d_hdr, width, height, params);
}

// ── Bloom implementation ────────────────────────────────────────────

void PostFxPipeline::apply_bloom_(float* d_hdr, int width, int height,
                                  const PostFxParams& params) {
    // Ensure scratch buffers match current resolution
    init(width, height);

    // 1. Find max luminance for automatic threshold
    launch_bloom_find_max_luminance(d_hdr, d_max_lum_, width, height);
    cudaDeviceSynchronize();

    float max_lum = 0.f;
    cudaMemcpy(&max_lum, d_max_lum_, sizeof(float), cudaMemcpyDeviceToHost);

    // Auto-threshold: bloom starts at 25% of peak luminance (ensures only
    // the bright sources contribute, and the threshold scales with dynamic
    // range automatically).
    constexpr float BLOOM_THRESHOLD_FRACTION = 0.25f;
    float threshold = max_lum * BLOOM_THRESHOLD_FRACTION;
    if (threshold < 1e-6f) return;  // scene is black — nothing to bloom

    // 2. Bright-pass extract → mip[0] (half-res)
    launch_bloom_bright_extract(d_hdr, d_mip_[0], width, height, threshold);

    // 3. Downsample chain: mip[0] → mip[1] → … → mip[N-1]
    for (int i = 1; i < NUM_MIP_LEVELS; ++i) {
        launch_bloom_downsample(d_mip_[i - 1], d_mip_[i],
                                mip_w_[i - 1], mip_h_[i - 1]);
    }

    // 4. Separable Gaussian blur at each mip level.
    //    Radius scales with mip level: finer mips use the user radius,
    //    coarser mips use proportionally larger kernels (in mip-space pixels,
    //    which covers more screen area).
    for (int i = 0; i < NUM_MIP_LEVELS; ++i) {
        int w = mip_w_[i];
        int h = mip_h_[i];

        // Scale radius for this mip level:
        //   mip 0 = user radius / 2 (half-res),
        //   mip 1 = user radius / 4, etc.
        // But we want coarser mips to blur *more* (not less) in screen
        // space.  Using a constant kernel radius at each mip level
        // naturally covers 2× more screen space per mip level, which
        // is exactly the multi-scale behaviour we want.
        float radius_h = params.bloom_radius_h / (float)(1 << (i + 1));
        float radius_v = params.bloom_radius_v / (float)(1 << (i + 1));

        // Ensure at least a small blur at each level
        radius_h = fmaxf(radius_h, 1.0f);
        radius_v = fmaxf(radius_v, 1.0f);

        // Horizontal blur: mip[i] → tmp[i]
        launch_bloom_blur_h(d_mip_[i],     d_mip_tmp_[i], w, h, radius_h);
        // Vertical blur:   tmp[i] → mip[i]
        launch_bloom_blur_v(d_mip_tmp_[i], d_mip_[i],     w, h, radius_v);
    }

    // 5. Upsample-accumulate: bottom up (coarsest → finest)
    //    mip[N-1] → upsample → add to mip[N-2] → … → mip[0]
    for (int i = NUM_MIP_LEVELS - 1; i >= 1; --i) {
        launch_bloom_upsample_accumulate(d_mip_[i], d_mip_[i - 1],
                                         mip_w_[i - 1], mip_h_[i - 1]);
    }

    // 6. Composite: add mip[0] (half-res accumulated bloom) onto the
    //    full-resolution HDR buffer with user intensity.
    launch_bloom_composite(d_hdr, d_mip_[0], width, height,
                           params.bloom_intensity);

    cudaDeviceSynchronize();
}
