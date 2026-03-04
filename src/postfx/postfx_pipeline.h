#pragma once
// ─────────────────────────────────────────────────────────────────────
// postfx/postfx_pipeline.h – GPU post-processing pipeline
// ─────────────────────────────────────────────────────────────────────
// Owns scratch GPU buffers, orchestrates all 2D post-processing effects
// on the linear HDR image *before* tone mapping.
//
// Usage:
//   1. pipeline.init(w, h)                — allocate scratch buffers
//   2. pipeline.apply(d_hdr, w, h, params)— run enabled effects in-place
//   3. pipeline.cleanup()                 — free buffers (or destructor)
//
// Adding a new effect:
//   - Add params to PostFxParams
//   - Add a private method (e.g. apply_vignette_())
//   - Call it inside apply()
// ─────────────────────────────────────────────────────────────────────
#include "postfx/postfx_params.h"
#include <cstddef>

class PostFxPipeline {
public:
    PostFxPipeline() = default;
    ~PostFxPipeline();

    /// Allocate / reallocate scratch buffers for the given resolution.
    void init(int width, int height);

    /// Apply all enabled post-FX to the HDR buffer in-place.
    /// @param d_hdr  device pointer to float4 [width*height*4].
    void apply(float* d_hdr, int width, int height,
               const PostFxParams& params);

    /// Free all device memory.
    void cleanup();

    // ── Bloom mip-chain configuration ───────────────────────────────
    static constexpr int NUM_MIP_LEVELS = 5;  // 1/2, 1/4, 1/8, 1/16, 1/32

private:
    void apply_bloom_(float* d_hdr, int width, int height,
                      const PostFxParams& params);

    // Mip chain: each stores float4 [mip_w * mip_h * 4]
    float* d_mip_[NUM_MIP_LEVELS]     = {};   // bloom mip buffers
    float* d_mip_tmp_[NUM_MIP_LEVELS] = {};   // temp for separable blur ping-pong
    int    mip_w_[NUM_MIP_LEVELS]     = {};
    int    mip_h_[NUM_MIP_LEVELS]     = {};

    // Single float for max-luminance reduction result
    float* d_max_lum_ = nullptr;

    int alloc_w_ = 0;
    int alloc_h_ = 0;
};
