// ─────────────────────────────────────────────────────────────────────
// emitter.cu – CUDA kernels for photon emission
// ─────────────────────────────────────────────────────────────────────
// Stub: the CPU path in emitter.h handles photon tracing.
// This file provides CUDA kernels for GPU-parallel photon emission.
// ─────────────────────────────────────────────────────────────────────

#include "core/types.h"
#include "core/spectrum.h"
#include "core/random.h"

// ── CUDA kernel: emit photons in parallel ───────────────────────────
__global__ void emit_photons_kernel(
    // Emissive triangle data (alias table)
    const float* __restrict__ alias_prob,
    const uint32_t* __restrict__ alias_redirect,
    const float* __restrict__ alias_pdf,
    int num_emissive,
    // Triangle geometry (SoA)
    const float* __restrict__ tri_v0x, const float* __restrict__ tri_v0y, const float* __restrict__ tri_v0z,
    const float* __restrict__ tri_v1x, const float* __restrict__ tri_v1y, const float* __restrict__ tri_v1z,
    const float* __restrict__ tri_v2x, const float* __restrict__ tri_v2y, const float* __restrict__ tri_v2z,
    const float* __restrict__ tri_nx,  const float* __restrict__ tri_ny,  const float* __restrict__ tri_nz,
    // Emission spectra (per emissive triangle, flattened)
    const float* __restrict__ Le_spectra, // [num_emissive * NUM_LAMBDA]
    // Output: emitted photon rays
    float* __restrict__ ray_ox, float* __restrict__ ray_oy, float* __restrict__ ray_oz,
    float* __restrict__ ray_dx, float* __restrict__ ray_dy, float* __restrict__ ray_dz,
    uint16_t* __restrict__ lambda_bins,
    float* __restrict__ fluxes,
    int num_photons)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_photons) return;

    PCGRng rng = PCGRng::seed(idx * 7 + 42, idx + 1);

    // 1. Sample emissive triangle
    float u1 = rng.next_float();
    float u2 = rng.next_float();
    int local_idx = (int)(u1 * num_emissive);
    if (local_idx >= num_emissive) local_idx = num_emissive - 1;
    if (u2 >= alias_prob[local_idx]) {
        local_idx = alias_redirect[local_idx];
    }

    float pdf_tri = alias_pdf[local_idx];

    // 2. Sample position on triangle
    float su = sqrtf(rng.next_float());
    float alpha = 1.f - su;
    float beta  = rng.next_float() * su;
    float gamma = 1.f - alpha - beta;

    float px = alpha * tri_v0x[local_idx] + beta * tri_v1x[local_idx] + gamma * tri_v2x[local_idx];
    float py = alpha * tri_v0y[local_idx] + beta * tri_v1y[local_idx] + gamma * tri_v2y[local_idx];
    float pz = alpha * tri_v0z[local_idx] + beta * tri_v1z[local_idx] + gamma * tri_v2z[local_idx];

    // 3. Sample wavelength
    const float* Le = &Le_spectra[local_idx * NUM_LAMBDA];
    float Le_sum = 0.f;
    for (int i = 0; i < NUM_LAMBDA; ++i) Le_sum += Le[i];

    float xi = rng.next_float() * Le_sum;
    float cumulative = 0.f;
    int bin = NUM_LAMBDA - 1;
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        cumulative += Le[i];
        if (xi <= cumulative) { bin = i; break; }
    }

    float Le_lambda = Le[bin];
    float pdf_lambda = (Le_sum > 0.f) ? Le_lambda / Le_sum : 1.f / NUM_LAMBDA;

    // 4. Sample direction (cosine hemisphere)
    float r = sqrtf(rng.next_float());
    float phi = 6.28318530718f * rng.next_float();
    float dx_local = r * cosf(phi);
    float dy_local = r * sinf(phi);
    float dz_local = sqrtf(fmaxf(0.f, 1.f - dx_local*dx_local - dy_local*dy_local));

    // Build ONB from normal
    float nx = tri_nx[local_idx], ny = tri_ny[local_idx], nz = tri_nz[local_idx];
    // ... (simplified – full ONB build would go here)

    // 5. Compute flux
    float cos_theta = dz_local;
    float pdf_dir = cos_theta * 0.31830988618f; // cos/pi

    // Compute triangle area
    float e1x = tri_v1x[local_idx] - tri_v0x[local_idx];
    float e1y = tri_v1y[local_idx] - tri_v0y[local_idx];
    float e1z = tri_v1z[local_idx] - tri_v0z[local_idx];
    float e2x = tri_v2x[local_idx] - tri_v0x[local_idx];
    float e2y = tri_v2y[local_idx] - tri_v0y[local_idx];
    float e2z = tri_v2z[local_idx] - tri_v0z[local_idx];
    float cx = e1y*e2z - e1z*e2y;
    float cy = e1z*e2x - e1x*e2z;
    float cz = e1x*e2y - e1y*e2x;
    float area = 0.5f * sqrtf(cx*cx + cy*cy + cz*cz);

    float pdf_pos = 1.f / area;
    float denom = pdf_tri * pdf_pos * pdf_dir * pdf_lambda;
    float flux = (denom > 0.f) ? (Le_lambda * cos_theta) / denom : 0.f;

    // Store results
    ray_ox[idx] = px + nx * 1e-4f;
    ray_oy[idx] = py + ny * 1e-4f;
    ray_oz[idx] = pz + nz * 1e-4f;
    ray_dx[idx] = dx_local; // NOTE: needs ONB transform in production
    ray_dy[idx] = dy_local;
    ray_dz[idx] = dz_local;
    lambda_bins[idx] = (uint16_t)bin;
    fluxes[idx] = flux;
}
