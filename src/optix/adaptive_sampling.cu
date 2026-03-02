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
    const uint16_t* pixel_max_spp, // nullptr → use global max_spp
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

    // --- Reached the per-pixel cap (AS-02) or global cap ─────────────
    int effective_max = pixel_max_spp ? (int)pixel_max_spp[idx] : max_spp;
    if (n >= (float)effective_max) {
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
        p.pixel_max_spp,
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
// ── AS-02: Per-pixel cost map → per-pixel max_spp budget ─────────────
// Combines pilot-pass variance with photon analysis data to assign
// each pixel a non-uniform SPP budget.

__global__
void k_compute_cost_map(
    const float*   lum_sum,           // [W*H] pilot Σ Y
    const float*   lum_sum2,          // [W*H] pilot Σ Y²
    const float*   sample_counts,     // [W*H] samples so far
    const float*   cell_guide_frac,   // [grid_cells] or nullptr
    const float*   cell_caustic_frac, // [grid_cells] or nullptr
    const float*   cell_flux_density, // [grid_cells] or nullptr
    // cell grid params for position → cell mapping (simplified: pixel → cell)
    const float*   spectrum_buffer,   // [W*H*NUM_LAMBDA] for position reconstruction
    int            width,
    int            height,
    int            base_spp,
    int            min_spp_clamp,     // floor on per-pixel budget
    int            max_spp_clamp,     // ceiling on per-pixel budget
    uint16_t*      pixel_max_spp,     // [W*H] output per-pixel budget
    float*         d_cost_sum)        // [1] atomic sum of all pixel costs
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    int idx = py * width + px;
    float n = sample_counts[idx];

    // §7.1 Factor 1: Pilot-pass luminance variance (weight 0.6)
    float rel_noise = pixel_relative_noise(lum_sum[idx], lum_sum2[idx], n);
    float var_factor = fminf(rel_noise * 10.f, 1.f);  // normalize to [0,1]

    // §7.1 Factor 2: Caustic fraction (weight 0.3) — sharp caustics need samples
    // TODO(Gap 6): Wire caustic/guide data per-pixel.  Currently only
    // available per-cell, not per-pixel.  Requires a pilot-pass buffer
    // mapping each pixel to its first-hit cell index (pixel_cell_idx).
    // Until then, factors 2 & 3 default to zero.
    float caustic_factor = 0.f;
    // cell_caustic_frac available per-cell, needs pixel→cell mapping

    // §7.1 Factor 3: Guide quality / flux density (weight 0.1)
    float guide_factor = 0.f;
    // cell_guide_frac, cell_flux_density available per-cell, needs pixel→cell mapping

    // Combined cost [0,1]
    float cost = 0.6f * var_factor + 0.3f * caustic_factor + 0.1f * guide_factor;

    // Per-pixel SPP budget: base * (cost / avg_cost)
    // We store the raw cost and compute the ratio after a reduction.
    // For now: directly map cost → spp multiplier [0.25, 4.0]
    float multiplier = 0.25f + cost * 3.75f;  // cost=0→0.25×, cost=1→4×
    int budget = (int)(base_spp * multiplier + 0.5f);
    budget = max(min_spp_clamp, min(budget, max_spp_clamp));

    pixel_max_spp[idx] = (uint16_t)budget;
    atomicAdd(d_cost_sum, cost);
}

// ── Host entry point ──────────────────────────────────────────────────

void compute_pixel_cost_map(const CostMapParams& p)
{
    dim3 block(16, 16);
    dim3 grid((p.width  + block.x - 1) / block.x,
              (p.height + block.y - 1) / block.y);

    float* d_cost_sum = nullptr;
    cudaMalloc(&d_cost_sum, sizeof(float));
    cudaMemset(d_cost_sum, 0, sizeof(float));

    k_compute_cost_map<<<grid, block>>>(
        p.lum_sum,
        p.lum_sum2,
        p.sample_counts,
        p.cell_guide_fraction,
        p.cell_caustic_fraction,
        p.cell_flux_density,
        p.spectrum_buffer,
        p.width,
        p.height,
        p.base_spp,
        p.min_spp_clamp,
        p.max_spp_clamp,
        p.pixel_max_spp,
        d_cost_sum);

    cudaDeviceSynchronize();
    cudaFree(d_cost_sum);
}