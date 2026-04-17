// ─────────────────────────────────────────────────────────────────────
// postfx/tonemap.cu – Tonemapping kernels (v5 RGB pipeline)
// ─────────────────────────────────────────────────────────────────────
// Stage 1: rgb_to_hdr — accumulator / sample_count × exposure → float4
// Stage 2: tonemap_hdr — float4 HDR → ACES → gamma → sRGB uint8
// ─────────────────────────────────────────────────────────────────────
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// ── ACES filmic tonemap (Narkowicz 2015 fit) ────────────────────────

static __device__ __forceinline__ float aces_filmic(float x) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return fminf(fmaxf((x * (a * x + b)) / (x * (c * x + d) + e), 0.f), 1.f);
}

// ── Linear → sRGB gamma ─────────────────────────────────────────────

static __device__ __forceinline__ float linear_to_srgb(float c) {
    if (c <= 0.0031308f) return 12.92f * c;
    return 1.055f * powf(c, 1.f / 2.4f) - 0.055f;
}

// =====================================================================
// 1. RGB accumulator → HDR float4
// =====================================================================

__global__ void rgb_to_hdr_kernel(
    const float* __restrict__ color_r,
    const float* __restrict__ color_g,
    const float* __restrict__ color_b,
    const float* __restrict__ sample_cnt,
    float*       __restrict__ hdr,
    int width, int height,
    float exposure,
    const float* __restrict__ caustic_r,
    const float* __restrict__ caustic_g,
    const float* __restrict__ caustic_b,
    float caustic_inv_frames,
    int caustic_only)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    int pixel = py * width + px;

    float r, g, b;
    if (caustic_only) {
        // Debug isolation: show only the caustic contribution
        if (caustic_r && caustic_inv_frames > 0.f) {
            r = caustic_r[pixel] * caustic_inv_frames * exposure;
            g = caustic_g[pixel] * caustic_inv_frames * exposure;
            b = caustic_b[pixel] * caustic_inv_frames * exposure;
        } else {
            r = g = b = 0.f;
        }
    } else {
        // Normal path: path tracer + caustic composite
        r = color_r[pixel] * exposure;
        g = color_g[pixel] * exposure;
        b = color_b[pixel] * exposure;

        if (caustic_r && caustic_inv_frames > 0.f) {
            r += caustic_r[pixel] * caustic_inv_frames * exposure;
            g += caustic_g[pixel] * caustic_inv_frames * exposure;
            b += caustic_b[pixel] * caustic_inv_frames * exposure;
        }
    }

    int dst = pixel * 4;
    hdr[dst + 0] = r;
    hdr[dst + 1] = g;
    hdr[dst + 2] = b;
    hdr[dst + 3] = 1.f;
}

void launch_rgb_to_hdr(
    const float* d_color_r,
    const float* d_color_g,
    const float* d_color_b,
    const float* d_sample_cnt,
    float* d_hdr,
    int width, int height,
    float exposure,
    const float* d_caustic_r,
    const float* d_caustic_g,
    const float* d_caustic_b,
    int caustic_frames,
    bool caustic_only)
{
    float inv_frames = (caustic_frames > 0) ? 1.f / (float)caustic_frames : 0.f;
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    rgb_to_hdr_kernel<<<grid, block>>>(
        d_color_r, d_color_g, d_color_b,
        d_sample_cnt, d_hdr, width, height, exposure,
        d_caustic_r, d_caustic_g, d_caustic_b, inv_frames,
        (int)caustic_only);
}

// =====================================================================
// 2. HDR float4 → sRGB uint8
// =====================================================================

__global__ void tonemap_hdr_kernel(
    const float*   __restrict__ hdr,
    uint8_t*       __restrict__ srgb,
    int width, int height,
    bool use_aces)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    int idx4 = (py * width + px) * 4;
    float r = fmaxf(hdr[idx4 + 0], 0.f);
    float g = fmaxf(hdr[idx4 + 1], 0.f);
    float b = fmaxf(hdr[idx4 + 2], 0.f);

    if (use_aces) {
        r = aces_filmic(r);
        g = aces_filmic(g);
        b = aces_filmic(b);
    }
    // When use_aces == false: skip tonemapping entirely.
    // Values pass through to sRGB gamma as-is (linear clamp to [0,1]).
    // This is the "barebone" display path for Phase 0 diagnostics.
    r = fminf(r, 1.f);
    g = fminf(g, 1.f);
    b = fminf(b, 1.f);

    // Linear → sRGB gamma
    r = linear_to_srgb(r);
    g = linear_to_srgb(g);
    b = linear_to_srgb(b);

    srgb[idx4 + 0] = (uint8_t)(fminf(r * 255.f + 0.5f, 255.f));
    srgb[idx4 + 1] = (uint8_t)(fminf(g * 255.f + 0.5f, 255.f));
    srgb[idx4 + 2] = (uint8_t)(fminf(b * 255.f + 0.5f, 255.f));
    srgb[idx4 + 3] = 255;
}

void launch_tonemap_hdr(
    const float* d_hdr,
    uint8_t* d_srgb,
    int width, int height,
    bool use_aces)
{
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    tonemap_hdr_kernel<<<grid, block>>>(
        d_hdr, d_srgb, width, height, use_aces);
}
