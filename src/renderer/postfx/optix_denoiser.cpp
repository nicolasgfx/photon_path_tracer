// ─────────────────────────────────────────────────────────────────────
// postfx/optix_denoiser.cpp – OptiX AI denoiser wrapper (v5, RGB)
// ─────────────────────────────────────────────────────────────────────
#include "postfx/optix_denoiser.h"

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

// ── Helpers ──────────────────────────────────────────────────────────

#define OPTIX_DENOISER_CHECK(expr)                                      \
    do {                                                                 \
        OptixResult _r = (expr);                                         \
        if (_r != OPTIX_SUCCESS) {                                       \
            std::fprintf(stderr, "[Denoiser] OptiX error in %s: %s\n",  \
                        #expr, optixGetErrorString(_r));                  \
            return;                                                      \
        }                                                                \
    } while (0)

#define OPTIX_DENOISER_CHECK_BOOL(expr)                                 \
    do {                                                                 \
        OptixResult _r = (expr);                                         \
        if (_r != OPTIX_SUCCESS) {                                       \
            std::fprintf(stderr, "[Denoiser] OptiX error in %s: %s\n",  \
                        #expr, optixGetErrorString(_r));                  \
            return false;                                                \
        }                                                                \
    } while (0)

// ── Init ─────────────────────────────────────────────────────────────

bool DenoiserSession::init(OptixDeviceContext ctx, int width, int height,
                           bool use_guides) {
    if (!ctx) { std::fprintf(stderr, "[Denoiser] Null OptiX context\n"); return false; }

    // Destroy old denoiser if re-initialising at new resolution
    cleanup();
    use_guides_   = use_guides;
    alloc_w_ = width;
    alloc_h_ = height;

    OptixDenoiserOptions options = {};
    if (use_guides) {
        options.guideAlbedo = 1;
        options.guideNormal = 1;
    }
#if OPTIX_VERSION >= 80000
    options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
#endif

    OPTIX_DENOISER_CHECK_BOOL(
        optixDenoiserCreate(ctx, OPTIX_DENOISER_MODEL_KIND_HDR,
                            &options, &denoiser_));

    // Compute memory requirements
    OptixDenoiserSizes sizes = {};
    OPTIX_DENOISER_CHECK_BOOL(
        optixDenoiserComputeMemoryResources(denoiser_, width, height, &sizes));

    state_size_   = sizes.stateSizeInBytes;
    scratch_size_ = sizes.withoutOverlapScratchSizeInBytes;

    cudaMalloc(&d_state_,     state_size_);
    cudaMalloc(&d_scratch_,   scratch_size_);
    cudaMalloc(&d_intensity_, sizeof(float));
    cudaMalloc(&d_output_,    (size_t)width * height * 4 * sizeof(float));

    OPTIX_DENOISER_CHECK_BOOL(
        optixDenoiserSetup(denoiser_, /*stream=*/0,
                           width, height,
                           (CUdeviceptr)d_state_,   state_size_,
                           (CUdeviceptr)d_scratch_, scratch_size_));

    std::printf("[Denoiser] Ready  %dx%d  guides=%s  state=%.1f KB\n",
                width, height, use_guides ? "yes" : "no",
                (double)state_size_ / 1024.0);
    return true;
}

void DenoiserSession::cleanup() {
    if (denoiser_)    { optixDenoiserDestroy(denoiser_); denoiser_ = nullptr; }
    if (d_state_)     { cudaFree(d_state_);     d_state_     = nullptr; }
    if (d_scratch_)   { cudaFree(d_scratch_);   d_scratch_   = nullptr; }
    if (d_intensity_) { cudaFree(d_intensity_); d_intensity_ = nullptr; }
    if (d_output_)    { cudaFree(d_output_);    d_output_    = nullptr; }
    alloc_w_ = alloc_h_ = 0;
    state_size_ = scratch_size_ = 0;
}

// ── Per-frame denoise ─────────────────────────────────────────────────

void DenoiserSession::denoise(float* d_hdr,
                              const float* d_albedo,
                              const float* d_normal,
                              int width, int height,
                              float blend)
{
    if (!denoiser_) return;

    // Compute average scene intensity for HDR normalisation
    OptixImage2D input_layer = {};
    input_layer.data               = (CUdeviceptr)d_hdr;
    input_layer.width              = (unsigned int)width;
    input_layer.height             = (unsigned int)height;
    input_layer.rowStrideInBytes   = (unsigned int)(width * 4 * sizeof(float));
    input_layer.pixelStrideInBytes = 0;
    input_layer.format             = OPTIX_PIXEL_FORMAT_FLOAT4;

    OPTIX_DENOISER_CHECK(
        optixDenoiserComputeIntensity(denoiser_, /*stream=*/0,
                                      &input_layer,
                                      (CUdeviceptr)d_intensity_,
                                      (CUdeviceptr)d_scratch_, scratch_size_));

    // Guide layers
    OptixDenoiserGuideLayer guide_layer = {};
    if (use_guides_ && d_albedo && d_normal) {
        guide_layer.albedo.data               = (CUdeviceptr)d_albedo;
        guide_layer.albedo.width              = (unsigned int)width;
        guide_layer.albedo.height             = (unsigned int)height;
        guide_layer.albedo.rowStrideInBytes   = (unsigned int)(width * 4 * sizeof(float));
        guide_layer.albedo.pixelStrideInBytes = 0;
        guide_layer.albedo.format             = OPTIX_PIXEL_FORMAT_FLOAT4;

        guide_layer.normal.data               = (CUdeviceptr)d_normal;
        guide_layer.normal.width              = (unsigned int)width;
        guide_layer.normal.height             = (unsigned int)height;
        guide_layer.normal.rowStrideInBytes   = (unsigned int)(width * 4 * sizeof(float));
        guide_layer.normal.pixelStrideInBytes = 0;
        guide_layer.normal.format             = OPTIX_PIXEL_FORMAT_FLOAT4;
    }

    // Output image
    OptixImage2D output_layer = {};
    output_layer.data               = (CUdeviceptr)d_output_;
    output_layer.width              = (unsigned int)width;
    output_layer.height             = (unsigned int)height;
    output_layer.rowStrideInBytes   = (unsigned int)(width * 4 * sizeof(float));
    output_layer.pixelStrideInBytes = 0;
    output_layer.format             = OPTIX_PIXEL_FORMAT_FLOAT4;

    OptixDenoiserLayer layers = {};
    layers.input  = input_layer;
    layers.output = output_layer;

    OptixDenoiserParams params = {};
#if OPTIX_VERSION < 80000
#  if OPTIX_VERSION >= 70500
    params.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
#  else
    params.denoiseAlpha = 0;
#  endif
#endif
    params.hdrIntensity = (CUdeviceptr)d_intensity_;
    params.blendFactor  = blend;

    OPTIX_DENOISER_CHECK(
        optixDenoiserInvoke(denoiser_, /*stream=*/0,
                            &params,
                            (CUdeviceptr)d_state_, state_size_,
                            &guide_layer,
                            &layers, 1,
                            /*offsetX=*/0, /*offsetY=*/0,
                            (CUdeviceptr)d_scratch_, scratch_size_));

    // Copy denoised output back over the input (in-place semantics)
    cudaMemcpy(d_hdr, d_output_,
               (size_t)width * height * 4 * sizeof(float),
               cudaMemcpyDeviceToDevice);
}
