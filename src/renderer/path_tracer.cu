// ─────────────────────────────────────────────────────────────────────
// path_tracer.cu – CUDA kernels for camera path tracing
// ─────────────────────────────────────────────────────────────────────
// Stub: the CPU path in renderer.cpp handles path tracing.
// This file provides CUDA kernels for GPU-parallel rendering.
// ─────────────────────────────────────────────────────────────────────

#include "core/types.h"
#include "core/spectrum.h"
#include "core/random.h"

// GPU path tracing kernel (placeholder for OptiX integration)
__global__ void path_trace_kernel(
    float* __restrict__ output_spectrum,  // [width * height * NUM_LAMBDA]
    float* __restrict__ sample_counts,    // [width * height]
    int width,
    int height,
    int samples_per_pixel,
    int max_bounces)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    int pixel_idx = py * width + px;

    for (int s = 0; s < samples_per_pixel; ++s) {
        PCGRng rng = PCGRng::seed(
            (uint64_t)pixel_idx * 1000 + s,
            (uint64_t)pixel_idx + 1);

        // Camera ray generation would go here
        // OptiX trace call would go here
        // Path tracing loop would go here

        // Placeholder: accumulate zero
        sample_counts[pixel_idx] += 1.f;
    }
}

// Tonemap kernel
__global__ void tonemap_kernel(
    const float* __restrict__ spectrum_buffer,  // [width * height * NUM_LAMBDA]
    const float* __restrict__ sample_counts,
    uint8_t* __restrict__ srgb_output,          // [width * height * 4]
    int width,
    int height,
    float exposure)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    int pixel_idx = py * width + px;
    float count = sample_counts[pixel_idx];
    if (count <= 0.f) count = 1.f;
    float inv_count = exposure / count;

    // Accumulate XYZ from spectrum
    float X = 0.f, Y = 0.f, Z = 0.f;
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        float val = spectrum_buffer[pixel_idx * NUM_LAMBDA + i] * inv_count;
        float lam = LAMBDA_MIN + (i + 0.5f) * LAMBDA_STEP;
        X += val * cie_x(lam);
        Y += val * cie_y(lam);
        Z += val * cie_z(lam);
    }

    float scale = LAMBDA_STEP;
    X *= scale; Y *= scale; Z *= scale;

    // XYZ to linear sRGB
    float r =  3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z;
    float g = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
    float b =  0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z;

    // Clamp and gamma
    auto gamma = [](float c) -> float {
        c = fmaxf(0.f, c);
        if (c <= 0.0031308f) return 12.92f * c;
        return 1.055f * powf(c, 1.f/2.4f) - 0.055f;
    };

    srgb_output[pixel_idx * 4 + 0] = (uint8_t)(fminf(gamma(r) * 255.f, 255.f));
    srgb_output[pixel_idx * 4 + 1] = (uint8_t)(fminf(gamma(g) * 255.f, 255.f));
    srgb_output[pixel_idx * 4 + 2] = (uint8_t)(fminf(gamma(b) * 255.f, 255.f));
    srgb_output[pixel_idx * 4 + 3] = 255;
}
