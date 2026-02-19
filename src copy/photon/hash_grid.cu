// ─────────────────────────────────────────────────────────────────────
// hash_grid.cu – CUDA implementation of spatial hash grid
// ─────────────────────────────────────────────────────────────────────
// Stub: the CPU path in hash_grid.h handles grid construction.
// This file provides CUDA kernels for GPU-accelerated grid build and
// photon lookup when running on GPU.
// ─────────────────────────────────────────────────────────────────────

#include "core/types.h"
#include "photon/photon.h"

// ── CUDA kernel: compute hash keys ─────────────────────────────────
__global__ void compute_hash_keys_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    uint32_t* __restrict__ keys,
    int n,
    float cell_size,
    uint32_t table_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int cx = (int)floorf(pos_x[idx] / cell_size);
    int cy = (int)floorf(pos_y[idx] / cell_size);
    int cz = (int)floorf(pos_z[idx] / cell_size);

    uint32_t h = (uint32_t)(cx * 73856093u)
               ^ (uint32_t)(cy * 19349663u)
               ^ (uint32_t)(cz * 83492791u);
    keys[idx] = h % table_size;
}

// ── CUDA kernel: find cell boundaries after sort ────────────────────
__global__ void find_cell_bounds_kernel(
    const uint32_t* __restrict__ sorted_keys,
    uint32_t* __restrict__ cell_start,
    uint32_t* __restrict__ cell_end,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t key = sorted_keys[idx];

    if (idx == 0 || sorted_keys[idx - 1] != key) {
        cell_start[key] = idx;
    }
    if (idx == n - 1 || sorted_keys[idx + 1] != key) {
        cell_end[key] = idx + 1;
    }
}

// Placeholder for device-side grid build orchestration
// (Radix sort would use CUB or Thrust)
void gpu_build_hash_grid(/* parameters */) {
    // TODO: implement with thrust::sort_by_key
}
