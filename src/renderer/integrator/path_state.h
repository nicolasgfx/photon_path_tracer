#pragma once
// ─────────────────────────────────────────────────────────────────────
// path_state.h – Per-path state for the GPU path integrator (v5 RGB)
//
// Lightweight struct carried through the bounce loop.
// No IOR stack / medium stack yet (added in later phases).
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"

// ── Per-path result ─────────────────────────────────────────────────
struct PathResult {
    float3 radiance;     // accumulated radiance (RGB)
    float3 albedo;       // AOV: first non-specular hit diffuse albedo
    float3 normal;       // AOV: first non-specular hit shading normal
    int    num_bounces;  // actual bounces taken (for pre-pass diagnostics)
};

// ── DevBSDFSample — device-side BSDF sample (mirrors CPU BSDFSample) ─
struct DevBSDFSample {
    float3 wi;           // sampled direction (local frame)
    float  pdf;          // PDF of sample
    float3 f;            // BSDF value f(wo, wi) (RGB)
    bool   is_specular;  // true for delta distributions (mirror/glass)
};


