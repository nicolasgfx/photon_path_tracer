#pragma once
// ─────────────────────────────────────────────────────────────────────
// tonemap.h – CUDA tonemap / colour-space conversion kernels
// ─────────────────────────────────────────────────────────────────────
// Extracted from photon/hash_grid.h/cu (§1.6): these kernels are
// unrelated to the spatial hash grid.
// ─────────────────────────────────────────────────────────────────────
#include <cstdint>

// Spectrum → sRGB post-process kernel (Optimization #5)
// Converts the accumulated spectral buffer to sRGB in a single GPU pass.
void launch_tonemap_kernel(
    const float* d_spectrum_buffer,
    const float* d_sample_counts,
    uint8_t* d_srgb_buffer,
    int width, int height,
    float exposure,
    const float* d_photon_gather_buffer = nullptr);

// Spectrum → linear HDR float4 conversion kernel (for denoiser input)
// Converts the accumulated spectral buffer to linear HDR RGB (no tone mapping).
void launch_spectrum_to_hdr_kernel(
    const float* d_spectrum_buffer,
    const float* d_sample_counts,
    float*       d_hdr_buffer,      // [W*H*4] float4 output
    int width, int height,
    float exposure,
    const float* d_photon_gather_buffer = nullptr);

// Tone map from linear HDR float4 → sRGB uint8 (after denoiser)
void launch_tonemap_hdr_kernel(
    const float* d_hdr_buffer,      // [W*H*4] float4 input
    uint8_t*     d_srgb_buffer,     // [W*H*4] uint8 output
    int width, int height);
