// ─────────────────────────────────────────────────────────────────────
// postfx/bloom.cu – GPU bloom kernels (mip-chain approach)
// ─────────────────────────────────────────────────────────────────────
// Multi-scale bloom: bright-extract → downsample chain → separable
// Gaussian blur per mip → upsample-accumulate → composite onto HDR.
//
// All kernels operate on float4 (RGBA) linear HDR data.
// ─────────────────────────────────────────────────────────────────────
#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>

// ── Helpers ─────────────────────────────────────────────────────────

static __device__ __forceinline__ float luminance_rgb(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// Bilinear sample from a float4 image (clamped addressing).
static __device__ __forceinline__ void sample_bilinear(
    const float* __restrict__ img, int w, int h,
    float u, float v,
    float& out_r, float& out_g, float& out_b)
{
    float fx = fmaxf(0.f, fminf(u, (float)(w - 1)));
    float fy = fmaxf(0.f, fminf(v, (float)(h - 1)));
    int x0 = (int)fx;
    int y0 = (int)fy;
    int x1 = min(x0 + 1, w - 1);
    int y1 = min(y0 + 1, h - 1);
    float sx = fx - x0;
    float sy = fy - y0;

    auto fetch = [&](int x, int y, float& r, float& g, float& b) {
        int idx = (y * w + x) * 4;
        r = img[idx + 0];
        g = img[idx + 1];
        b = img[idx + 2];
    };

    float r00, g00, b00, r10, g10, b10, r01, g01, b01, r11, g11, b11;
    fetch(x0, y0, r00, g00, b00);
    fetch(x1, y0, r10, g10, b10);
    fetch(x0, y1, r01, g01, b01);
    fetch(x1, y1, r11, g11, b11);

    out_r = (r00*(1-sx) + r10*sx)*(1-sy) + (r01*(1-sx) + r11*sx)*sy;
    out_g = (g00*(1-sx) + g10*sx)*(1-sy) + (g01*(1-sx) + g11*sx)*sy;
    out_b = (b00*(1-sx) + b10*sx)*(1-sy) + (b01*(1-sx) + b11*sx)*sy;
}

// =====================================================================
// 1. Parallel reduction: find max luminance
// =====================================================================

__global__ void bloom_max_lum_kernel(
    const float* __restrict__ hdr,
    float*       __restrict__ partial_max,
    int total_pixels)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float my_max = 0.f;
    // Grid-stride loop
    for (int i = gid; i < total_pixels; i += blockDim.x * gridDim.x) {
        float r = hdr[i * 4 + 0];
        float g = hdr[i * 4 + 1];
        float b = hdr[i * 4 + 2];
        float lum = luminance_rgb(r, g, b);
        my_max = fmaxf(my_max, lum);
    }
    sdata[tid] = my_max;
    __syncthreads();

    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0) partial_max[blockIdx.x] = sdata[0];
}

__global__ void bloom_reduce_max_kernel(
    const float* __restrict__ partial,
    float*       __restrict__ result,
    int n)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    sdata[tid] = (tid < n) ? partial[tid] : 0.f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) result[0] = sdata[0];
}

void launch_bloom_find_max_luminance(
    const float* d_hdr, float* d_max_lum,
    int width, int height)
{
    int total = width * height;
    constexpr int BLOCK = 256;
    int num_blocks = min((total + BLOCK - 1) / BLOCK, 1024);

    // Temporary partial-max buffer
    float* d_partial = nullptr;
    cudaMalloc(&d_partial, num_blocks * sizeof(float));

    bloom_max_lum_kernel<<<num_blocks, BLOCK, BLOCK * sizeof(float)>>>(
        d_hdr, d_partial, total);

    // Second pass: reduce partial results
    int reduce_block = 1;
    while (reduce_block < num_blocks) reduce_block <<= 1;
    reduce_block = min(reduce_block, 1024);
    bloom_reduce_max_kernel<<<1, reduce_block, reduce_block * sizeof(float)>>>(
        d_partial, d_max_lum, num_blocks);

    cudaFree(d_partial);
}

// =====================================================================
// 2. Bright-pass extract (full-res → half-res)
// =====================================================================

__global__ void bloom_bright_extract_kernel(
    const float* __restrict__ hdr,
    float*       __restrict__ mip0,
    int src_w, int src_h,
    int dst_w, int dst_h,
    float threshold)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    // Bilinear sample from source at 2× coordinate
    float u = (float)dx * 2.f + 0.5f;
    float v = (float)dy * 2.f + 0.5f;
    float r, g, b;
    sample_bilinear(hdr, src_w, src_h, u, v, r, g, b);

    float lum = luminance_rgb(r, g, b);
    if (lum > threshold && threshold > 0.f) {
        // Scale by (lum - threshold) / lum to keep colour proportional
        float scale = (lum - threshold) / lum;
        r *= scale;
        g *= scale;
        b *= scale;
    } else {
        r = g = b = 0.f;
    }

    int idx = (dy * dst_w + dx) * 4;
    mip0[idx + 0] = r;
    mip0[idx + 1] = g;
    mip0[idx + 2] = b;
    mip0[idx + 3] = 1.f;
}

void launch_bloom_bright_extract(
    const float* d_hdr, float* d_mip0,
    int src_w, int src_h, float threshold)
{
    int dst_w = max(src_w / 2, 1);
    int dst_h = max(src_h / 2, 1);
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    bloom_bright_extract_kernel<<<grid, block>>>(
        d_hdr, d_mip0, src_w, src_h, dst_w, dst_h, threshold);
}

// =====================================================================
// 3. Bilinear 2× downsample
// =====================================================================

__global__ void bloom_downsample_kernel(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    int src_w, int src_h,
    int dst_w, int dst_h)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    float u = (float)dx * 2.f + 0.5f;
    float v = (float)dy * 2.f + 0.5f;
    float r, g, b;
    sample_bilinear(src, src_w, src_h, u, v, r, g, b);

    int idx = (dy * dst_w + dx) * 4;
    dst[idx + 0] = r;
    dst[idx + 1] = g;
    dst[idx + 2] = b;
    dst[idx + 3] = 1.f;
}

void launch_bloom_downsample(
    const float* d_src, float* d_dst,
    int src_w, int src_h)
{
    int dst_w = max(src_w / 2, 1);
    int dst_h = max(src_h / 2, 1);
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    bloom_downsample_kernel<<<grid, block>>>(
        d_src, d_dst, src_w, src_h, dst_w, dst_h);
}

// =====================================================================
// 4. Separable Gaussian blur (horizontal + vertical)
// =====================================================================

// Gaussian weight (un-normalised).
static __device__ __forceinline__ float gauss_weight(float x, float sigma) {
    return expf(-0.5f * (x * x) / (sigma * sigma));
}

__global__ void bloom_blur_h_kernel(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    int w, int h,
    float sigma, int kernel_radius)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    float sum_r = 0.f, sum_g = 0.f, sum_b = 0.f, sum_w = 0.f;
    for (int dx = -kernel_radius; dx <= kernel_radius; ++dx) {
        int sx = min(max(px + dx, 0), w - 1);
        float wt = gauss_weight((float)dx, sigma);
        int idx = (py * w + sx) * 4;
        sum_r += src[idx + 0] * wt;
        sum_g += src[idx + 1] * wt;
        sum_b += src[idx + 2] * wt;
        sum_w += wt;
    }

    float inv = (sum_w > 0.f) ? 1.f / sum_w : 0.f;
    int out_idx = (py * w + px) * 4;
    dst[out_idx + 0] = sum_r * inv;
    dst[out_idx + 1] = sum_g * inv;
    dst[out_idx + 2] = sum_b * inv;
    dst[out_idx + 3] = 1.f;
}

__global__ void bloom_blur_v_kernel(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    int w, int h,
    float sigma, int kernel_radius)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    float sum_r = 0.f, sum_g = 0.f, sum_b = 0.f, sum_w = 0.f;
    for (int dy = -kernel_radius; dy <= kernel_radius; ++dy) {
        int sy = min(max(py + dy, 0), h - 1);
        float wt = gauss_weight((float)dy, sigma);
        int idx = (sy * w + px) * 4;
        sum_r += src[idx + 0] * wt;
        sum_g += src[idx + 1] * wt;
        sum_b += src[idx + 2] * wt;
        sum_w += wt;
    }

    float inv = (sum_w > 0.f) ? 1.f / sum_w : 0.f;
    int out_idx = (py * w + px) * 4;
    dst[out_idx + 0] = sum_r * inv;
    dst[out_idx + 1] = sum_g * inv;
    dst[out_idx + 2] = sum_b * inv;
    dst[out_idx + 3] = 1.f;
}

void launch_bloom_blur_h(
    const float* d_src, float* d_dst,
    int w, int h, float radius)
{
    if (radius < 0.5f) {
        // No blur needed — just copy
        cudaMemcpy(d_dst, d_src, (size_t)w * h * 4 * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        return;
    }
    float sigma = radius / 2.5f;  // 2.5σ ≈ radius for practical Gaussian cutoff
    int kernel_radius = (int)ceilf(radius);
    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);
    bloom_blur_h_kernel<<<grid, block>>>(d_src, d_dst, w, h, sigma, kernel_radius);
}

void launch_bloom_blur_v(
    const float* d_src, float* d_dst,
    int w, int h, float radius)
{
    if (radius < 0.5f) {
        cudaMemcpy(d_dst, d_src, (size_t)w * h * 4 * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        return;
    }
    float sigma = radius / 2.5f;
    int kernel_radius = (int)ceilf(radius);
    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);
    bloom_blur_v_kernel<<<grid, block>>>(d_src, d_dst, w, h, sigma, kernel_radius);
}

// =====================================================================
// 5. Bilinear 2× upsample + additive accumulate
// =====================================================================

__global__ void bloom_upsample_acc_kernel(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    int dst_w, int dst_h,
    int src_w, int src_h)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    // Map dst pixel to src coordinates (half resolution)
    float u = ((float)dx + 0.5f) * 0.5f - 0.5f;
    float v = ((float)dy + 0.5f) * 0.5f - 0.5f;
    float r, g, b;
    sample_bilinear(src, src_w, src_h, u, v, r, g, b);

    int idx = (dy * dst_w + dx) * 4;
    dst[idx + 0] += r;
    dst[idx + 1] += g;
    dst[idx + 2] += b;
}

void launch_bloom_upsample_accumulate(
    const float* d_src, float* d_dst,
    int dst_w, int dst_h)
{
    int src_w = max(dst_w / 2, 1);
    int src_h = max(dst_h / 2, 1);
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    bloom_upsample_acc_kernel<<<grid, block>>>(
        d_src, d_dst, dst_w, dst_h, src_w, src_h);
}

// =====================================================================
// 6. Final composite: add bloom onto HDR (half-res bloom → full-res HDR)
// =====================================================================

__global__ void bloom_composite_kernel(
    float*       __restrict__ hdr,
    const float* __restrict__ bloom,
    int width, int height,
    int bloom_w, int bloom_h,
    float intensity)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    // Bilinear upsample bloom from half-res
    float u = ((float)px + 0.5f) * 0.5f - 0.5f;
    float v = ((float)py + 0.5f) * 0.5f - 0.5f;
    float br, bg, bb;
    sample_bilinear(bloom, bloom_w, bloom_h, u, v, br, bg, bb);

    int idx = (py * width + px) * 4;
    hdr[idx + 0] += br * intensity;
    hdr[idx + 1] += bg * intensity;
    hdr[idx + 2] += bb * intensity;
}

void launch_bloom_composite(
    float* d_hdr, const float* d_bloom,
    int width, int height, float intensity)
{
    int bloom_w = max(width / 2, 1);
    int bloom_h = max(height / 2, 1);
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    bloom_composite_kernel<<<grid, block>>>(
        d_hdr, d_bloom, width, height, bloom_w, bloom_h, intensity);
}
