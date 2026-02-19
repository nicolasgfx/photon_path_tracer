// ─────────────────────────────────────────────────────────────────────
// direct_light.cu – CUDA kernels for direct light sampling
// ─────────────────────────────────────────────────────────────────────
// Stub: the CPU path in direct_light.h handles direct illumination.
// This file provides CUDA kernels for shadow ray batching.
// ─────────────────────────────────────────────────────────────────────

#include "core/types.h"
#include "core/random.h"

// Placeholder kernel for GPU shadow ray testing
__global__ void shadow_ray_kernel(
    const float* __restrict__ ray_ox, const float* __restrict__ ray_oy, const float* __restrict__ ray_oz,
    const float* __restrict__ ray_dx, const float* __restrict__ ray_dy, const float* __restrict__ ray_dz,
    const float* __restrict__ max_dist,
    int* __restrict__ visible, // 1 = visible, 0 = occluded
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // OptiX shadow ray would go here
    // For now, mark all as visible (placeholder)
    visible[idx] = 1;
}
