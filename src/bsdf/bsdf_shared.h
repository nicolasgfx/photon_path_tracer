#pragma once
// ─────────────────────────────────────────────────────────────────────
// bsdf_shared.h – Shared BSDF helpers for CPU↔GPU consistency
// ─────────────────────────────────────────────────────────────────────
// v2.2 consistency reset: all roughness clamps, F0 computations, and
// lobe probability calculations MUST use these shared helpers so that
// CPU and GPU produce identical results.
//
// Rule: if GPU clamps roughness, CPU must use the same clamp — here.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"

// ── Minimum alpha (roughness²) clamp ────────────────────────────────
// Both CPU and GPU must use this.  Prevents numerical singularities
// in GGX microfacet distribution for near-mirror surfaces.
constexpr float BSDF_MIN_ALPHA = 0.001f;

// ── Clamp roughness to alpha ────────────────────────────────────────
inline HD float bsdf_roughness_to_alpha(float roughness) {
    return fmaxf(roughness * roughness, BSDF_MIN_ALPHA);
}

// ── Dielectric F0 from IOR ──────────────────────────────────────────
inline HD float bsdf_f0_from_ior(float ior) {
    float t = (ior - 1.f) / (ior + 1.f);
    return t * t;
}

// ── Schlick Fresnel (shared) ────────────────────────────────────────
inline HD float fresnel_schlick(float cos_theta, float f0) {
    float t = 1.f - cos_theta;
    float t2 = t * t;
    return f0 + (1.f - f0) * t2 * t2 * t;
}

// ── Exact dielectric Fresnel (shared) ───────────────────────────────
inline HD float fresnel_dielectric(float cos_i, float eta) {
    float sin2_t = eta * eta * (1.f - cos_i * cos_i);
    if (sin2_t >= 1.f) return 1.f; // Total internal reflection
    float cos_t = sqrtf(fmaxf(0.f, 1.f - sin2_t));
    cos_i = fabsf(cos_i);
    float rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    float rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    return 0.5f * (rs * rs + rp * rp);
}

// ── Luminance of a spectrum (for lobe probability weighting) ────────
inline HD float bsdf_spectrum_luminance(const Spectrum& s) {
    // Approximate: average of all bins (fast, consistent CPU/GPU)
    return s.sum() / (float)NUM_LAMBDA;
}

// ── Lobe sampling probabilities (diffuse + specular mixture) ────────
// Canonical v2.2: probability based on energy, clamped to avoid
// zero-probability issues.
//
// p_spec = clamp(spec_energy / (spec_energy + diff_energy), 0.05, 0.95)
// p_diff = 1 - p_spec
//
// For dielectric materials, spec_energy = Ks.max * F0 (to account
// for the fact that dielectrics have low specular at normal incidence).
struct LobeProbabilities {
    float p_spec;
    float p_diff;
};

inline HD LobeProbabilities bsdf_lobe_probabilities(
    float spec_weight, float diff_weight)
{
    LobeProbabilities lp;
    float total = spec_weight + diff_weight;
    if (total > 0.f) {
        lp.p_spec = spec_weight / total;
    } else {
        lp.p_spec = 0.5f;
    }
    // Clamp to ensure both lobes get some samples
    lp.p_spec = fmaxf(0.05f, fminf(0.95f, lp.p_spec));
    lp.p_diff = 1.f - lp.p_spec;
    return lp;
}

// ── Metal lobe probabilities ────────────────────────────────────────
inline HD LobeProbabilities bsdf_metal_lobe_probs(const Spectrum& Kd, const Spectrum& Ks) {
    return bsdf_lobe_probabilities(Ks.max_component(), Kd.max_component());
}

// ── Dielectric lobe probabilities ───────────────────────────────────
inline HD LobeProbabilities bsdf_dielectric_lobe_probs(
    const Spectrum& Kd, const Spectrum& Ks, float ior)
{
    float F0 = bsdf_f0_from_ior(ior);
    return bsdf_lobe_probabilities(Ks.max_component() * F0, Kd.max_component());
}

// ── GGX NDF (shared, float3 local-frame half-vector) ────────────────
inline HD float ggx_D(float3 h, float alpha) {
    float NdotH = h.z; // In local frame, N = (0,0,1)
    if (NdotH <= 0.f) return 0.f;
    float a2 = alpha * alpha;
    float d  = NdotH * NdotH * (a2 - 1.f) + 1.f;
    return a2 / (PI * d * d);
}

// ── GGX Smith G1 (shared, float3 local-frame direction) ─────────────
inline HD float ggx_G1(float3 v, float alpha) {
    float NdotV = fabsf(v.z);
    float a2 = alpha * alpha;
    return 2.f * NdotV / (NdotV + sqrtf(a2 + (1.f - a2) * NdotV * NdotV));
}

// ── GGX Smith G (shared) ───────────────────────────────────────────
inline HD float ggx_G(float3 wo, float3 wi, float alpha) {
    return ggx_G1(wo, alpha) * ggx_G1(wi, alpha);
}

// ── GGX Visible Normal Distribution sampling (VNDF) ─────────────────
inline HD float3 ggx_sample_halfvector(float3 wo, float alpha, float u1, float u2) {
    // Stretch
    float3 wh = normalize(make_f3(alpha * wo.x, alpha * wo.y, wo.z));

    // Orthonormal basis
    float3 t1 = (wh.z < 0.9999f) ? normalize(cross(make_f3(0,0,1), wh))
                                   : make_f3(1,0,0);
    float3 t2 = cross(wh, t1);

    // Uniform disk sample
    float r   = sqrtf(u1);
    float phi = TWO_PI * u2;
    float p1  = r * cosf(phi);
    float p2  = r * sinf(phi);
    float s   = 0.5f * (1.f + wh.z);
    p2 = (1.f - s) * sqrtf(fmaxf(0.f, 1.f - p1*p1)) + s * p2;

    // Project onto hemisphere
    float3 nh = t1 * p1 + t2 * p2 + wh * sqrtf(fmaxf(0.f, 1.f - p1*p1 - p2*p2));

    // Unstretch
    return normalize(make_f3(alpha * nh.x, alpha * nh.y, fmaxf(0.f, nh.z)));
}

// ── Reflect / Refract (local shading frame) ─────────────────────────

inline HD float3 reflect_local(float3 wo) {
    return make_f3(-wo.x, -wo.y, wo.z);
}

inline HD bool refract_local(float3 wo, float eta, float3& wt) {
    float cos_i = wo.z;
    float sin2_i = fmaxf(0.f, 1.f - cos_i * cos_i);
    float sin2_t = eta * eta * sin2_i;
    if (sin2_t >= 1.f) return false;

    float cos_t = sqrtf(1.f - sin2_t);
    wt = make_f3(-eta * wo.x, -eta * wo.y, -cos_t);
    return true;
}

// ── MIS weight (2-way power heuristic) ──────────────────────────────
inline HD float mis_weight_2(float pdf_a, float pdf_b) {
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    return a2 / fmaxf(a2 + b2, 1e-30f);
}

// ── Combined PDF for mixture sampling ───────────────────────────────
// p = p_diff * pdf_diff(wi) + p_spec * pdf_spec(wi)
inline HD float bsdf_combined_pdf(float p_diff, float pdf_diff,
                                   float p_spec, float pdf_spec) {
    return p_diff * pdf_diff + p_spec * pdf_spec;
}
