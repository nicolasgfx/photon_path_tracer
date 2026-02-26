// ─────────────────────────────────────────────────────────────────────
// hash_grid.cu – CUDA implementation of spatial hash grid
// ─────────────────────────────────────────────────────────────────────
// GPU-accelerated grid build using CUB radix sort.
// Eliminates the GPU→CPU→GPU round-trip that was the primary pipeline
// bottleneck in the photon tracing phase.
//
// Pipeline:
//   1. compute_hash_keys_kernel   – O(N) key computation
//   2. CUB DeviceRadixSort        – O(N) radix sort (key, index pairs)
//   3. find_cell_bounds_kernel    – O(N) cell boundary detection
//   4. build_caustic_tags_kernel  – O(N) 3-valued tag assignment
//   5. scatter_photon_soa_kernel  – O(N) SoA reorder by sorted indices
//
// All steps run on the GPU.  No host memcpy except for the photon
// count (4 bytes) and optional diagnostic downloads.
// ─────────────────────────────────────────────────────────────────────

#include "core/types.h"
#include "core/config.h"
#include "photon/photon.h"
#include "photon/hash_grid.h"

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <cstdio>

// ── CUDA kernel: compute hash keys ─────────────────────────────────
__global__ void compute_hash_keys_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    uint32_t* __restrict__ keys,
    uint32_t* __restrict__ indices,
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
    keys[idx]    = h % table_size;
    indices[idx] = (uint32_t)idx;
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

// ── CUDA kernel: build 3-valued caustic tags on GPU ─────────────────
// tag = 0  global-pass, non-caustic
// tag = 1  global-pass, caustic path
// tag = 2  targeted-pass caustic
__global__ void build_caustic_tags_kernel(
    const uint8_t* __restrict__ is_caustic,   // per-photon caustic flag (from photon trace)
    uint8_t*       __restrict__ tags_out,      // [n] output tags
    int global_count,
    int total_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_count) return;

    if (idx < global_count) {
        tags_out[idx] = (is_caustic && is_caustic[idx]) ? (uint8_t)1 : (uint8_t)0;
    } else {
        tags_out[idx] = (uint8_t)2;
    }
}

// ── CUDA kernel: scatter a float SoA channel by sorted indices ──────
__global__ void scatter_float_kernel(
    const float*    __restrict__ src,
    const uint32_t* __restrict__ sorted_indices,
    float*          __restrict__ dst,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = src[sorted_indices[idx]];
}

// ── CUDA kernel: scatter uint16 SoA channel by sorted indices ───────
__global__ void scatter_uint16_kernel(
    const uint16_t* __restrict__ src,
    const uint32_t* __restrict__ sorted_indices,
    uint16_t*       __restrict__ dst,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = src[sorted_indices[idx]];
}

// ── CUDA kernel: scatter uint8 SoA channel by sorted indices ────────
__global__ void scatter_uint8_kernel(
    const uint8_t*  __restrict__ src,
    const uint32_t* __restrict__ sorted_indices,
    uint8_t*        __restrict__ dst,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = src[sorted_indices[idx]];
}

// ── CUDA kernel: scatter float SoA with hero-wavelength stride ──────
// Each photon has HERO_WAVELENGTHS values; scatter all of them.
__global__ void scatter_float_hero_kernel(
    const float*    __restrict__ src,
    const uint32_t* __restrict__ sorted_indices,
    float*          __restrict__ dst,
    int n,
    int hero_wavelengths)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t src_idx = sorted_indices[idx];
    for (int h = 0; h < hero_wavelengths; ++h)
        dst[idx * hero_wavelengths + h] = src[src_idx * hero_wavelengths + h];
}

// ── CUDA kernel: scatter uint16 SoA with hero-wavelength stride ─────
__global__ void scatter_uint16_hero_kernel(
    const uint16_t* __restrict__ src,
    const uint32_t* __restrict__ sorted_indices,
    uint16_t*       __restrict__ dst,
    int n,
    int hero_wavelengths)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t src_idx = sorted_indices[idx];
    for (int h = 0; h < hero_wavelengths; ++h)
        dst[idx * hero_wavelengths + h] = src[src_idx * hero_wavelengths + h];
}

// =====================================================================
// gpu_build_hash_grid() – Full GPU-side hash grid construction
// =====================================================================
// Inputs:  photon SoA pointers already on GPU (d_pos_x/y/z)
// Outputs: sorted indices, cell_start, cell_end arrays on GPU
//
// The caller provides pre-allocated device buffers for keys, indices,
// sorted outputs, cell_start, cell_end, and a temp buffer for CUB.
// =====================================================================

void gpu_build_hash_grid(
    // Photon positions (already on GPU)
    const float* d_pos_x,
    const float* d_pos_y,
    const float* d_pos_z,
    int n,
    float cell_size,
    uint32_t table_size,
    // Pre-allocated GPU buffers (managed by caller)
    uint32_t* d_keys_in,
    uint32_t* d_keys_out,
    uint32_t* d_indices_in,
    uint32_t* d_sorted_indices,  // output: sorted photon indices
    uint32_t* d_cell_start,      // output: [table_size]
    uint32_t* d_cell_end,        // output: [table_size]
    void*     d_temp_storage,
    size_t&   temp_storage_bytes)
{
    if (n <= 0) return;

    constexpr int BLOCK = 256;
    int grid = (n + BLOCK - 1) / BLOCK;

    // Step 1: Compute hash keys and initial indices
    compute_hash_keys_kernel<<<grid, BLOCK>>>(
        d_pos_x, d_pos_y, d_pos_z,
        d_keys_in, d_indices_in,
        n, cell_size, table_size);

    // Step 2: CUB radix sort (key, index) pairs
    // First call with nullptr to get required temp storage size
    if (d_temp_storage == nullptr) {
        cub::DeviceRadixSort::SortPairs(
            nullptr, temp_storage_bytes,
            d_keys_in, d_keys_out,
            d_indices_in, d_sorted_indices,
            n);
        return;  // caller allocates temp_storage and calls again
    }

    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        d_indices_in, d_sorted_indices,
        n);

    // Step 3: Initialize cell_start/cell_end to sentinel
    cudaMemset(d_cell_start, 0xFF, table_size * sizeof(uint32_t));  // 0xFFFFFFFF
    cudaMemset(d_cell_end,   0x00, table_size * sizeof(uint32_t));

    // Step 4: Find cell boundaries
    find_cell_bounds_kernel<<<grid, BLOCK>>>(
        d_keys_out, d_cell_start, d_cell_end, n);
}

// =====================================================================
// gpu_scatter_photon_soa() – Reorder photon SoA by sorted indices
// =====================================================================
// After radix sort, photon data is still in original order.  The gather
// kernel uses sorted_indices for indirection, so the SoA doesn't
// *need* to be reordered.  However, for CPU-side downloads (diagnostics,
// kNN adaptive radius, irradiance heatmap) we may want the data in
// sorted order.
//
// This function is optional — only call when you need sorted SoA
// (e.g. for download to CPU).
// =====================================================================

void gpu_scatter_photon_soa(
    // Source SoA (unsorted, on GPU)
    const float*    d_src_pos_x,  const float*    d_src_pos_y,  const float*    d_src_pos_z,
    const float*    d_src_wi_x,   const float*    d_src_wi_y,   const float*    d_src_wi_z,
    const float*    d_src_norm_x, const float*    d_src_norm_y, const float*    d_src_norm_z,
    const uint16_t* d_src_lambda, const float*    d_src_flux,
    const uint8_t*  d_src_num_hero,
    // Sorted indices from gpu_build_hash_grid
    const uint32_t* d_sorted_indices,
    // Destination SoA (sorted, on GPU)
    float*    d_dst_pos_x,  float*    d_dst_pos_y,  float*    d_dst_pos_z,
    float*    d_dst_wi_x,   float*    d_dst_wi_y,   float*    d_dst_wi_z,
    float*    d_dst_norm_x, float*    d_dst_norm_y, float*    d_dst_norm_z,
    uint16_t* d_dst_lambda, float*    d_dst_flux,
    uint8_t*  d_dst_num_hero,
    int n)
{
    if (n <= 0) return;

    constexpr int BLOCK = 256;
    int grid = (n + BLOCK - 1) / BLOCK;

    scatter_float_kernel<<<grid, BLOCK>>>(d_src_pos_x,  d_sorted_indices, d_dst_pos_x,  n);
    scatter_float_kernel<<<grid, BLOCK>>>(d_src_pos_y,  d_sorted_indices, d_dst_pos_y,  n);
    scatter_float_kernel<<<grid, BLOCK>>>(d_src_pos_z,  d_sorted_indices, d_dst_pos_z,  n);
    scatter_float_kernel<<<grid, BLOCK>>>(d_src_wi_x,   d_sorted_indices, d_dst_wi_x,   n);
    scatter_float_kernel<<<grid, BLOCK>>>(d_src_wi_y,   d_sorted_indices, d_dst_wi_y,   n);
    scatter_float_kernel<<<grid, BLOCK>>>(d_src_wi_z,   d_sorted_indices, d_dst_wi_z,   n);
    scatter_float_kernel<<<grid, BLOCK>>>(d_src_norm_x, d_sorted_indices, d_dst_norm_x, n);
    scatter_float_kernel<<<grid, BLOCK>>>(d_src_norm_y, d_sorted_indices, d_dst_norm_y, n);
    scatter_float_kernel<<<grid, BLOCK>>>(d_src_norm_z, d_sorted_indices, d_dst_norm_z, n);
    scatter_float_hero_kernel<<<grid, BLOCK>>>(d_src_flux,   d_sorted_indices, d_dst_flux,   n, HERO_WAVELENGTHS);
    scatter_uint16_hero_kernel<<<grid, BLOCK>>>(d_src_lambda, d_sorted_indices, d_dst_lambda, n, HERO_WAVELENGTHS);
    scatter_uint8_kernel<<<grid, BLOCK>>>(d_src_num_hero, d_sorted_indices, d_dst_num_hero, n);
}

// =====================================================================
// gpu_build_caustic_tags() – Compute 3-valued tags on GPU
// =====================================================================

void gpu_build_caustic_tags(
    const uint8_t* d_is_caustic,  // [total_count] from photon trace (nullptr OK)
    uint8_t*       d_tags_out,    // [total_count] output tags
    int global_count,
    int total_count)
{
    if (total_count <= 0) return;
    constexpr int BLOCK = 256;
    int grid = (total_count + BLOCK - 1) / BLOCK;
    build_caustic_tags_kernel<<<grid, BLOCK>>>(
        d_is_caustic, d_tags_out, global_count, total_count);
}

// =====================================================================
// Tonemap post-process kernel (Optimization #5)
// =====================================================================
// Converts the accumulated spectral buffer to sRGB in a single pass
// after all SPP are complete.  Avoids redundant per-SPP tonemapping
// in __raygen__render.
// =====================================================================

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

__global__ void tonemap_kernel(
    const float* __restrict__ spectrum_buffer,  // [W*H*NUM_LAMBDA]
    const float* __restrict__ sample_counts,    // [W*H]
    uint8_t*     __restrict__ srgb_buffer,      // [W*H*4]
    int width, int height,
    float exposure)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    int pixel_idx = py * width + px;
    float n_samples = sample_counts[pixel_idx];

    // Compute average spectrum
    float X = 0.f, Y = 0.f, Z = 0.f;
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        float val = (n_samples > 0.f)
            ? spectrum_buffer[pixel_idx * NUM_LAMBDA + i] / n_samples
            : 0.f;
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
    float exposure)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    tonemap_kernel<<<grid, block>>>(
        d_spectrum_buffer, d_sample_counts, d_srgb_buffer,
        width, height, exposure);
}
