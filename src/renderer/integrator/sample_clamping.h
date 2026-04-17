#pragma once
// ─────────────────────────────────────────────────────────────────────
// sample_clamping.h – Throughput and luminance clamping (v5 RGB)
//
// Three-level safety net against fireflies:
//   1. Per-bounce contribution clamp
//   2. Per-path throughput clamp
//   3. Per-sample luminance clamp (applied after accumulation)
//
// All functions take explicit limit parameters so they can be driven
// from LaunchParams (GPU) or RenderConfig (CPU) without compile-time
// global constants.
//
// Host/device shared (HD qualifier).
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"

// ── Clamp a float3 so no component exceeds max_val ──────────────────
inline HD float3 clamp_f3(float3 v, float max_val) {
    return make_f3(
        fminf(v.x, max_val),
        fminf(v.y, max_val),
        fminf(v.z, max_val));
}

// ── Per-bounce contribution clamp ───────────────────────────────────
// Clamp f*cos/pdf per channel to prevent single-bounce spikes.
inline HD float3 clamp_bounce_contribution(float3 f_over_pdf, float limit) {
    return make_f3(
        fminf(f_over_pdf.x, limit),
        fminf(f_over_pdf.y, limit),
        fminf(f_over_pdf.z, limit));
}

// ── Per-path throughput clamp ───────────────────────────────────────
// Scale throughput so its max component does not exceed limit.
inline HD float3 clamp_path_throughput(float3 throughput, float limit) {
    float mx = max_component(throughput);
    if (mx > limit) {
        float s = limit / mx;
        return throughput * s;
    }
    return throughput;
}

// ── Per-sample luminance clamp ──────────────────────────────────────
// Scale radiance so luminance does not exceed limit. Preserves hue.
inline HD float3 clamp_sample_luminance(float3 L, float limit) {
    float lum = luminance(L);
    if (lum > limit) {
        float s = limit / lum;
        return L * s;
    }
    return L;
}
