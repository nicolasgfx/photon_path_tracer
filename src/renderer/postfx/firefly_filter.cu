// ─────────────────────────────────────────────────────────────────────
// postfx/firefly_filter.cu – Median+MAD outlier suppression (v5)
// ─────────────────────────────────────────────────────────────────────
// Detects outlier pixels via median luminance + K×MAD in a local window.
// Outliers are clamped to the median luminance while preserving
// chromaticity.  All operations on float4 linear HDR data.
// ─────────────────────────────────────────────────────────────────────
#include <cuda_runtime.h>
#include <cstdio>

// ── Helpers ─────────────────────────────────────────────────────────

static __device__ __forceinline__ float luminance_rgb(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// Insertion sort for small arrays (N ≤ 25).
static __device__ void insertion_sort(float* arr, int n) {
    for (int i = 1; i < n; ++i) {
        float key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            --j;
        }
        arr[j + 1] = key;
    }
}

// ── Firefly detection + clamp kernel ────────────────────────────────

__global__ void firefly_filter_kernel(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    int width, int height,
    int radius, float K)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    // Collect luminances in the window
    float lum_buf[25];
    int count = 0;

    for (int dy = -radius; dy <= radius; ++dy) {
        int sy = min(max(py + dy, 0), height - 1);
        for (int dx = -radius; dx <= radius; ++dx) {
            int sx = min(max(px + dx, 0), width - 1);
            int idx = (sy * width + sx) * 4;
            float lum = luminance_rgb(src[idx], src[idx + 1], src[idx + 2]);
            lum_buf[count++] = lum;
        }
    }

    // Sort to find median
    insertion_sort(lum_buf, count);
    float median = lum_buf[count / 2];

    // Compute MAD (Median Absolute Deviation)
    float abs_dev[25];
    for (int i = 0; i < count; ++i)
        abs_dev[i] = fabsf(lum_buf[i] - median);
    insertion_sort(abs_dev, count);
    float mad = abs_dev[count / 2];

    // Robust sigma estimate: sigma ≈ 1.4826 × MAD
    float sigma = 1.4826f * mad;
    float threshold_lum = median + K * sigma;

    // Read center pixel
    int center_idx = (py * width + px) * 4;
    float r = src[center_idx + 0];
    float g = src[center_idx + 1];
    float b = src[center_idx + 2];
    float a = src[center_idx + 3];
    float center_lum = luminance_rgb(r, g, b);

    // Clamp if outlier — preserve chromaticity
    if (center_lum > threshold_lum && center_lum > 1e-6f) {
        float scale = threshold_lum / center_lum;
        r *= scale;
        g *= scale;
        b *= scale;
    }

    dst[center_idx + 0] = r;
    dst[center_idx + 1] = g;
    dst[center_idx + 2] = b;
    dst[center_idx + 3] = a;
}

// ── Copy back kernel ────────────────────────────────────────────────

__global__ void firefly_copy_back_kernel(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    int total_float4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_float4) return;
    int base = idx * 4;
    dst[base + 0] = src[base + 0];
    dst[base + 1] = src[base + 1];
    dst[base + 2] = src[base + 2];
    dst[base + 3] = src[base + 3];
}

// ── Host launch wrapper ─────────────────────────────────────────────

void launch_firefly_filter(float* d_hdr, float* d_temp,
                           int width, int height,
                           int radius, float threshold)
{
    // Clamp radius to [1, 2]
    radius = (radius < 1) ? 1 : (radius > 2) ? 2 : radius;

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    // Pass 1: detect + clamp outliers → d_temp
    firefly_filter_kernel<<<grid, block>>>(
        d_hdr, d_temp, width, height, radius, threshold);

    // Pass 2: copy cleaned result back to d_hdr
    int total = width * height;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    firefly_copy_back_kernel<<<blocks, threads>>>(d_temp, d_hdr, total);
}
