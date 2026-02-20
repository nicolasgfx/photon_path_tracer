// ─────────────────────────────────────────────────────────────────────
// adaptive_sampling.cu  –  CUDA kernel for screen-noise adaptive
//                          sampling mask update.
// ─────────────────────────────────────────────────────────────────────
#include "optix/adaptive_sampling.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// ── Device helpers ────────────────────────────────────────────────────

/// Relative-noise estimate for a single pixel.
/// Returns the standard error of the mean divided by the mean luminance.
__device__ __forceinline__
float pixel_relative_noise(float sum_Y, float sum_Y2, float n)
{
    if (n < 2.f) return 1.f;   // not enough samples → treat as high noise
    float mu      = sum_Y / n;
    float var     = fmaxf(sum_Y2 / n - mu * mu, 0.f);  // Bessel-corrected clamped
    float se      = sqrtf(var / n);                      // std error of the mean
    float rel     = se / (fabsf(mu) + 1e-4f);           // relative noise
    return rel;
}

// ── Kernel ────────────────────────────────────────────────────────────

/// One thread per pixel.
/// Computes localNoise = max relative noise in (2R+1)×(2R+1) neighbourhood
/// and writes active_mask[p] accordingly.
__global__
void k_update_mask(
    const float*   sample_counts,
    const float*   lum_sum,
    const float*   lum_sum2,
    uint8_t*       active_mask,
    int            width,
    int            height,
    int            min_spp,
    int            max_spp,
    float          threshold,
    int            radius,
    int*           out_active_count)   // device-side atomic counter
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    int idx = py * width + px;
    float n = sample_counts[idx];

    // --- Warmup: not yet at min_spp, always mark active ---------------
    if (n < (float)min_spp) {
        active_mask[idx] = 1;
        atomicAdd(out_active_count, 1);
        return;
    }

    // --- Reached the per-pixel cap ------------------------------------
    if (n >= (float)max_spp) {
        active_mask[idx] = 0;
        return;
    }

    // --- Compute neighbourhood max relative noise --------------------
    float local_noise = 0.f;
    for (int dy = -radius; dy <= radius; ++dy) {
        int ny = py + dy;
        if (ny < 0 || ny >= height) continue;
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = px + dx;
            if (nx < 0 || nx >= width) continue;
            int nidx = ny * width + nx;
            float nn  = sample_counts[nidx];
            float r   = pixel_relative_noise(lum_sum[nidx], lum_sum2[nidx], nn);
            local_noise = fmaxf(local_noise, r);
        }
    }

    uint8_t active = (local_noise > threshold) ? 1u : 0u;
    active_mask[idx] = active;
    if (active) atomicAdd(out_active_count, 1);
}

// ── Host entry point ──────────────────────────────────────────────────

int adaptive_update_mask(const AdaptiveParams& p)
{
    // Allocate a 1-int device counter
    int* d_count = nullptr;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    dim3 block(16, 16);
    dim3 grid((p.width  + block.x - 1) / block.x,
              (p.height + block.y - 1) / block.y);

    k_update_mask<<<grid, block>>>(
        p.sample_counts,
        p.lum_sum,
        p.lum_sum2,
        p.active_mask,
        p.width,
        p.height,
        p.min_spp,
        p.max_spp,
        p.threshold,
        p.radius,
        d_count);

    cudaDeviceSynchronize();

    int h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_count);

    return h_count;
}
