// ---------------------------------------------------------------------
// optix_denoiser.cpp -- OptiX AI Denoiser management
// ---------------------------------------------------------------------
// Extracted from optix_renderer.cpp (§1.8):
//   setup_denoiser(), run_denoiser(), cleanup_denoiser()
// ---------------------------------------------------------------------
#include "optix/optix_renderer.h"

#include <cuda_runtime.h>
#include <optix.h>

// =====================================================================
// OptiX AI Denoiser — setup, invoke, cleanup
// =====================================================================
void OptixRenderer::setup_denoiser(int w, int h,
                                    bool guide_albedo, bool guide_normal)
{
    // Destroy previous denoiser if dimensions or guide config changed
    if (denoiser_ && (denoiser_width_ != w || denoiser_height_ != h ||
                      denoiser_guide_albedo_ != guide_albedo ||
                      denoiser_guide_normal_ != guide_normal)) {
        cleanup_denoiser();
    }

    if (denoiser_) return;  // already set up with matching dimensions

    denoiser_width_  = w;
    denoiser_height_ = h;
    denoiser_guide_albedo_ = guide_albedo;
    denoiser_guide_normal_ = guide_normal;

    OptixDenoiserOptions options = {};
    options.guideAlbedo = guide_albedo ? 1u : 0u;
    options.guideNormal = guide_normal ? 1u : 0u;

    OPTIX_CHECK(optixDenoiserCreate(
        context_,
        OPTIX_DENOISER_MODEL_KIND_AOV,
        &options,
        &denoiser_));

    // Query memory requirements
    OptixDenoiserSizes sizes = {};
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(
        denoiser_, (unsigned int)w, (unsigned int)h, &sizes));

    d_denoiser_state_.alloc(sizes.stateSizeInBytes);
    // Use recommendedScratchSizeInBytes (larger = better quality)
    size_t scratch_size = sizes.withoutOverlapScratchSizeInBytes;
    if (sizes.withOverlapScratchSizeInBytes > scratch_size)
        scratch_size = sizes.withOverlapScratchSizeInBytes;
    d_denoiser_scratch_.alloc(scratch_size);

    OPTIX_CHECK(optixDenoiserSetup(
        denoiser_,
        nullptr,  // CUDA stream
        (unsigned int)w, (unsigned int)h,
        reinterpret_cast<CUdeviceptr>(d_denoiser_state_.d_ptr),
        d_denoiser_state_.bytes,
        reinterpret_cast<CUdeviceptr>(d_denoiser_scratch_.d_ptr),
        d_denoiser_scratch_.bytes));

    CUDA_CHECK(cudaDeviceSynchronize());
}

void OptixRenderer::run_denoiser(float blend_factor)
{
    if (!denoiser_) return;

    const unsigned int row_stride = (unsigned int)(denoiser_width_ * 4 * sizeof(float));

    // Guide layer: albedo + normal
    OptixDenoiserGuideLayer guide = {};
    if (denoiser_guide_albedo_ && d_albedo_buffer_.d_ptr) {
        guide.albedo.data               = reinterpret_cast<CUdeviceptr>(d_albedo_buffer_.d_ptr);
        guide.albedo.width              = (unsigned int)denoiser_width_;
        guide.albedo.height             = (unsigned int)denoiser_height_;
        guide.albedo.rowStrideInBytes   = row_stride;
        guide.albedo.pixelStrideInBytes = 4 * sizeof(float);
        guide.albedo.format             = OPTIX_PIXEL_FORMAT_FLOAT4;
    }
    if (denoiser_guide_normal_ && d_normal_buffer_.d_ptr) {
        guide.normal.data               = reinterpret_cast<CUdeviceptr>(d_normal_buffer_.d_ptr);
        guide.normal.width              = (unsigned int)denoiser_width_;
        guide.normal.height             = (unsigned int)denoiser_height_;
        guide.normal.rowStrideInBytes   = row_stride;
        guide.normal.pixelStrideInBytes = 4 * sizeof(float);
        guide.normal.format             = OPTIX_PIXEL_FORMAT_FLOAT4;
    }

    // Input layer: linear HDR color
    OptixDenoiserLayer layer = {};
    layer.input.data               = reinterpret_cast<CUdeviceptr>(d_hdr_buffer_.d_ptr);
    layer.input.width              = (unsigned int)denoiser_width_;
    layer.input.height             = (unsigned int)denoiser_height_;
    layer.input.rowStrideInBytes   = row_stride;
    layer.input.pixelStrideInBytes = 4 * sizeof(float);
    layer.input.format             = OPTIX_PIXEL_FORMAT_FLOAT4;

    // Output layer: denoised HDR color
    layer.output.data               = reinterpret_cast<CUdeviceptr>(d_hdr_denoised_.d_ptr);
    layer.output.width              = (unsigned int)denoiser_width_;
    layer.output.height             = (unsigned int)denoiser_height_;
    layer.output.rowStrideInBytes   = row_stride;
    layer.output.pixelStrideInBytes = 4 * sizeof(float);
    layer.output.format             = OPTIX_PIXEL_FORMAT_FLOAT4;

    // Denoiser parameters
    OptixDenoiserParams params = {};
    params.blendFactor  = blend_factor;  // 0 = fully denoised, 1 = original

    OPTIX_CHECK(optixDenoiserInvoke(
        denoiser_,
        nullptr,  // CUDA stream
        &params,
        reinterpret_cast<CUdeviceptr>(d_denoiser_state_.d_ptr),
        d_denoiser_state_.bytes,
        &guide,
        &layer,
        1,         // numLayers
        0, 0,      // inputOffsetX, inputOffsetY
        reinterpret_cast<CUdeviceptr>(d_denoiser_scratch_.d_ptr),
        d_denoiser_scratch_.bytes));

    CUDA_CHECK(cudaDeviceSynchronize());
}

void OptixRenderer::cleanup_denoiser()
{
    if (denoiser_) {
        optixDenoiserDestroy(denoiser_);
        denoiser_ = nullptr;
    }
    d_denoiser_state_.free();
    d_denoiser_scratch_.free();
    denoiser_width_  = 0;
    denoiser_height_ = 0;
}
