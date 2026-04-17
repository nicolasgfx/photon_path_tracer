#pragma once
// ─────────────────────────────────────────────────────────────────────
// postfx/postfx_pipeline.h – GPU post-processing pipeline (v5)
//
// Orchestrates: firefly filter → bloom → tonemap
// All GPU-side, float4 HDR intermediate.
// ─────────────────────────────────────────────────────────────────────
#include "postfx/postfx_params.h"
#include <cstdint>

class PostFxPipeline {
public:
    ~PostFxPipeline();

    // Allocate/reallocate scratch buffers for the given resolution.
    void init(int width, int height);

    // Run full pipeline: RGB SoA accum → HDR float4 → firefly → bloom → tonemap → sRGB
    //   color_r/g/b: [w*h] per-channel accumulators (device, SoA)
    //   sample_cnt:  [w*h] per-pixel counts (device, nullable)
    //   srgb_out:    [w*h*4] RGBA8 output (device)
    //   hdr_out:     [w*h*4] HDR float4 output (device, nullable — for denoiser)
    void apply(const float* d_color_r,
               const float* d_color_g,
               const float* d_color_b,
               const float* d_sample_cnt,
               uint8_t* d_srgb_out,
               float* d_hdr_out,
               int width, int height,
               const PostFxParams& params);

    // Fire only the pre-denoiser pass (firefly filter) on an existing HDR buffer.
    void apply_pre_denoise(float* d_hdr, int width, int height,
                           const PostFxParams& params);

    // Get the internal HDR buffer (valid after apply()).
    float* hdr_buffer() const { return d_hdr_; }

        // Two-phase pipeline for use with external denoiser:
        //   Phase 1 — RGB SoA accum → HDR float4 → firefly filter
        //             Returns internal HDR buffer ready for denoising.
        float* apply_phase1(const float* d_color_r, const float* d_color_g,
                            const float* d_color_b, const float* d_sample_cnt,
                            int width, int height, const PostFxParams& params);
        //   Phase 2 — bloom → tonemap → sRGB (call after denoiser writes d_hdr_)
        void   apply_phase2(uint8_t* d_srgb_out, int width, int height,
                            const PostFxParams& params);

    void cleanup();

private:
    static constexpr int NUM_MIP_LEVELS = 5;

    void apply_firefly_(float* d_hdr, int width, int height,
                        const PostFxParams& params);
    void apply_bloom_(float* d_hdr, int width, int height,
                      const PostFxParams& params);

    // ── Allocated buffers ────────────────────────────────────────────
    float* d_hdr_          = nullptr;   // [w*h*4] working HDR buffer
    float* d_firefly_temp_ = nullptr;   // [w*h*4] firefly scratch
    float* d_max_lum_      = nullptr;   // [1] bloom max luminance

    // Bloom mip chain
    float* d_mip_[NUM_MIP_LEVELS]     = {};
    float* d_mip_tmp_[NUM_MIP_LEVELS] = {};
    int    mip_w_[NUM_MIP_LEVELS]     = {};
    int    mip_h_[NUM_MIP_LEVELS]     = {};

    int alloc_w_ = 0;
    int alloc_h_ = 0;
};
