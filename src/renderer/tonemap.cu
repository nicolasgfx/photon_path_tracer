// ─────────────────────────────────────────────────────────────────────
// tonemap.cu – CUDA tonemap / colour-space conversion kernels
// ─────────────────────────────────────────────────────────────────────
// Extracted from photon/hash_grid.cu (§1.6): these kernels are
// unrelated to the spatial hash grid.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"
#include "core/config.h"

// CIE tables needed for spectrum→sRGB (duplicated from optix_device.cu
// since this .cu is compiled separately and doesn't see __device__ constants).
// These are the Wyman et al. 2013 CIE 1931 colour-matching functions
// evaluated at bin centres: λ_i = 380 + (i + 0.5) × 100 nm, i = 0..3.
__device__ static const float CIE_XBAR[NUM_LAMBDA] = {
    0.27339309f, 0.15960659f, 0.65621063f, 0.00015248f,
};
__device__ static const float CIE_YBAR[NUM_LAMBDA] = {
    0.01038388f, 0.86905173f, 0.26366744f, 0.00029635f,
};
__device__ static const float CIE_ZBAR[NUM_LAMBDA] = {
    1.38682275f, 0.04303550f, 0.00000068f, 0.00000000f,
};
__device__ static const float CIE_YBAR_SUM_INV = 1.0f / 1.14339940f;

// =====================================================================
// Spectrum → sRGB tonemap kernel
// =====================================================================
__global__ void tonemap_kernel(
    const float* __restrict__ spectrum_buffer,  // [W*H*NUM_LAMBDA]
    const float* __restrict__ sample_counts,    // [W*H]
    uint8_t*     __restrict__ srgb_buffer,      // [W*H*4]
    int width, int height,
    float exposure,
    const float* __restrict__ photon_gather_buffer)  // [W*H*NUM_LAMBDA] or nullptr
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    int pixel_idx = py * width + px;
    float n_samples = sample_counts[pixel_idx];

    // Compute average spectrum + additive photon gather contribution
    float X = 0.f, Y = 0.f, Z = 0.f;
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        float val = (n_samples > 0.f)
            ? spectrum_buffer[pixel_idx * NUM_LAMBDA + i] / n_samples
            : 0.f;
        if (photon_gather_buffer)
            val += photon_gather_buffer[pixel_idx * NUM_LAMBDA + i];
        X += val * CIE_XBAR[i];
        Y += val * CIE_YBAR[i];
        Z += val * CIE_ZBAR[i];
    }

    X *= CIE_YBAR_SUM_INV; Y *= CIE_YBAR_SUM_INV; Z *= CIE_YBAR_SUM_INV;
    X *= exposure; Y *= exposure; Z *= exposure;

    float r =  3.2406f*X - 1.5372f*Y - 0.4986f*Z;
    float g = -0.9689f*X + 1.8758f*Y + 0.0415f*Z;
    float b =  0.0557f*X - 0.2040f*Y + 1.0570f*Z;

    // ACES Filmic tone mapping (Narkowicz 2015)
    if (USE_ACES_TONEMAPPING) {
        auto aces = [](float x) -> float {
            x = fmaxf(x, 0.f);
            return (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f);
        };
        r = aces(r);
        g = aces(g);
        b = aces(b);
    } else {
        r = fmaxf(r, 0.f);
        g = fmaxf(g, 0.f);
        b = fmaxf(b, 0.f);
    }

    // sRGB gamma
    auto gamma = [](float c) -> float {
        c = fmaxf(c, 0.f);
        return (c <= 0.0031308f) ? 12.92f*c : 1.055f*powf(c, 1.f/2.4f) - 0.055f;
    };

    r = fminf(gamma(r), 1.f);
    g = fminf(gamma(g), 1.f);
    b = fminf(gamma(b), 1.f);

    srgb_buffer[pixel_idx * 4 + 0] = (uint8_t)(r * 255.f);
    srgb_buffer[pixel_idx * 4 + 1] = (uint8_t)(g * 255.f);
    srgb_buffer[pixel_idx * 4 + 2] = (uint8_t)(b * 255.f);
    srgb_buffer[pixel_idx * 4 + 3] = 255;
}

void launch_tonemap_kernel(
    const float* d_spectrum_buffer,
    const float* d_sample_counts,
    uint8_t*     d_srgb_buffer,
    int width, int height,
    float exposure,
    const float* d_photon_gather_buffer)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    tonemap_kernel<<<grid, block>>>(
        d_spectrum_buffer, d_sample_counts, d_srgb_buffer,
        width, height, exposure, d_photon_gather_buffer);
}

// =====================================================================
// Spectrum → linear HDR float4 kernel (for OptiX denoiser input)
// No tone mapping: just spectrum → XYZ → linear sRGB primaries × exposure.
// =====================================================================
__global__ void spectrum_to_hdr_kernel(
    const float* __restrict__ spectrum_buffer,
    const float* __restrict__ sample_counts,
    float*       __restrict__ hdr_buffer,
    int width, int height,
    float exposure,
    const float* __restrict__ photon_gather_buffer)  // [W*H*NUM_LAMBDA] or nullptr
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    int pixel_idx = py * width + px;
    float n_samples = sample_counts[pixel_idx];

    float X = 0.f, Y = 0.f, Z = 0.f;
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        float val = (n_samples > 0.f)
            ? spectrum_buffer[pixel_idx * NUM_LAMBDA + i] / n_samples
            : 0.f;
        if (photon_gather_buffer)
            val += photon_gather_buffer[pixel_idx * NUM_LAMBDA + i];
        X += val * CIE_XBAR[i];
        Y += val * CIE_YBAR[i];
        Z += val * CIE_ZBAR[i];
    }

    X *= CIE_YBAR_SUM_INV; Y *= CIE_YBAR_SUM_INV; Z *= CIE_YBAR_SUM_INV;
    X *= exposure; Y *= exposure; Z *= exposure;

    // XYZ → linear sRGB (no tone mapping, no gamma)
    float r = fmaxf( 3.2406f*X - 1.5372f*Y - 0.4986f*Z, 0.f);
    float g = fmaxf(-0.9689f*X + 1.8758f*Y + 0.0415f*Z, 0.f);
    float b = fmaxf( 0.0557f*X - 0.2040f*Y + 1.0570f*Z, 0.f);

    hdr_buffer[pixel_idx * 4 + 0] = r;
    hdr_buffer[pixel_idx * 4 + 1] = g;
    hdr_buffer[pixel_idx * 4 + 2] = b;
    hdr_buffer[pixel_idx * 4 + 3] = 1.f;
}

void launch_spectrum_to_hdr_kernel(
    const float* d_spectrum_buffer,
    const float* d_sample_counts,
    float*       d_hdr_buffer,
    int width, int height,
    float exposure,
    const float* d_photon_gather_buffer)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    spectrum_to_hdr_kernel<<<grid, block>>>(
        d_spectrum_buffer, d_sample_counts, d_hdr_buffer,
        width, height, exposure, d_photon_gather_buffer);
}

// =====================================================================
// Linear HDR float4 → sRGB uint8 kernel (after denoising)
// Applies ACES tone mapping + sRGB gamma.
// =====================================================================
__global__ void tonemap_hdr_kernel(
    const float* __restrict__ hdr_buffer,
    uint8_t*     __restrict__ srgb_buffer,
    int width, int height)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    int pixel_idx = py * width + px;

    float r = hdr_buffer[pixel_idx * 4 + 0];
    float g = hdr_buffer[pixel_idx * 4 + 1];
    float b = hdr_buffer[pixel_idx * 4 + 2];

    // ACES Filmic tone mapping (Narkowicz 2015)
    if (USE_ACES_TONEMAPPING) {
        auto aces = [](float x) -> float {
            x = fmaxf(x, 0.f);
            return (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f);
        };
        r = aces(r);
        g = aces(g);
        b = aces(b);
    } else {
        r = fmaxf(r, 0.f);
        g = fmaxf(g, 0.f);
        b = fmaxf(b, 0.f);
    }

    // sRGB gamma
    auto gamma = [](float c) -> float {
        c = fmaxf(c, 0.f);
        return (c <= 0.0031308f) ? 12.92f*c : 1.055f*powf(c, 1.f/2.4f) - 0.055f;
    };

    r = fminf(gamma(r), 1.f);
    g = fminf(gamma(g), 1.f);
    b = fminf(gamma(b), 1.f);

    srgb_buffer[pixel_idx * 4 + 0] = (uint8_t)(r * 255.f);
    srgb_buffer[pixel_idx * 4 + 1] = (uint8_t)(g * 255.f);
    srgb_buffer[pixel_idx * 4 + 2] = (uint8_t)(b * 255.f);
    srgb_buffer[pixel_idx * 4 + 3] = 255;
}

void launch_tonemap_hdr_kernel(
    const float* d_hdr_buffer,
    uint8_t*     d_srgb_buffer,
    int width, int height)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    tonemap_hdr_kernel<<<grid, block>>>(
        d_hdr_buffer, d_srgb_buffer, width, height);
}
