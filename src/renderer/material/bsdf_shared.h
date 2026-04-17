#pragma once
// ─────────────────────────────────────────────────────────────────────
// bsdf_shared.h – Shared BSDF helpers for CPU↔GPU consistency (v5 RGB)
//
// All roughness clamps, F0 computations, Fresnel, GGX, VNDF sampling,
// and lobe probability calculations.  Used by both bsdf.h (CPU) and
// future optix_bsdf.cuh (GPU).
//
// Ported from v4: Spectrum→Color3.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/color.h"

// ── Minimum alpha (roughness²) clamp ────────────────────────────────
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

// Per-channel Schlick Fresnel for metallic reflectance (F0 per channel).
inline HD float3 fresnel_schlick3(float cos_theta, float3 f0) {
    float t = 1.f - cos_theta;
    float t2 = t * t;
    float t5 = t2 * t2 * t;
    return make_f3(
        f0.x + (1.f - f0.x) * t5,
        f0.y + (1.f - f0.y) * t5,
        f0.z + (1.f - f0.z) * t5);
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

// ── Luminance of a Color3 (for lobe probability weighting) ──────────
inline HD float bsdf_color_luminance(const Color3& c) {
    return c.luminance();
}

// ── Lobe sampling probabilities (diffuse + specular mixture) ────────
struct LobeProbabilities {
    float p_spec;
    float p_diff;
};

inline HD LobeProbabilities bsdf_lobe_probabilities(float spec_weight, float diff_weight) {
    LobeProbabilities lp;
    float total = spec_weight + diff_weight;
    if (total > 0.f) {
        lp.p_spec = spec_weight / total;
    } else {
        lp.p_spec = 0.5f;
    }
    lp.p_spec = fmaxf(0.05f, fminf(0.95f, lp.p_spec));
    lp.p_diff = 1.f - lp.p_spec;
    return lp;
}

inline HD LobeProbabilities bsdf_metal_lobe_probs(const Color3& Kd, const Color3& Ks) {
    return bsdf_lobe_probabilities(Ks.max_component(), Kd.max_component());
}

inline HD LobeProbabilities bsdf_dielectric_lobe_probs(const Color3& Kd, const Color3& Ks, float ior) {
    float F0 = bsdf_f0_from_ior(ior);
    return bsdf_lobe_probabilities(Ks.max_component() * F0, Kd.max_component());
}

// ── GGX NDF (local-frame half-vector, N = (0,0,1)) ─────────────────
inline HD float ggx_D(float3 h, float alpha) {
    float NdotH = h.z;
    if (NdotH <= 0.f) return 0.f;
    float a2 = alpha * alpha;
    float d  = NdotH * NdotH * (a2 - 1.f) + 1.f;
    return a2 / (PI * d * d);
}

// ── GGX Smith G1 (local-frame direction) ────────────────────────────
inline HD float ggx_G1(float3 v, float alpha) {
    float NdotV = fabsf(v.z);
    float a2 = alpha * alpha;
    return 2.f * NdotV / (NdotV + sqrtf(a2 + (1.f - a2) * NdotV * NdotV));
}

// ── GGX Smith G (separable masking-shadowing) ───────────────────────
inline HD float ggx_G(float3 wo, float3 wi, float alpha) {
    float a2 = alpha * alpha;
    float NdotO = fabsf(wo.z);
    float NdotI = fabsf(wi.z);
    float denom_o = NdotO + sqrtf(a2 + (1.f - a2) * NdotO * NdotO);
    float denom_i = NdotI + sqrtf(a2 + (1.f - a2) * NdotI * NdotI);
    return 4.f * NdotO * NdotI / (denom_o * denom_i);
}

// ── Cook-Torrance denominator (local frame, N = (0,0,1)) ───────────
inline HD float ggx_denom(float3 wo, float3 wi) {
    return 4.f * fabsf(wo.z) * fabsf(wi.z) + EPSILON;
}

// ── GGX Visible Normal Distribution sampling (VNDF) ─────────────────
inline HD float3 ggx_sample_halfvector(float3 wo, float alpha, float u1, float u2) {
    // Stretch
    float3 wh = normalize(make_f3(alpha * wo.x, alpha * wo.y, wo.z));

    // Orthonormal basis
    float3 t1 = (wh.z < 0.9999f) ? normalize(cross(make_f3(0, 0, 1), wh))
                                   : make_f3(1, 0, 0);
    float3 t2 = cross(wh, t1);

    // Uniform disk sample
    float r   = sqrtf(u1);
    float phi = TWO_PI * u2;
    float sp, cp;
#ifdef __CUDA_ARCH__
    sincosf(phi, &sp, &cp);
#else
    sp = sinf(phi);
    cp = cosf(phi);
#endif
    float p1 = r * cp;
    float p2 = r * sp;
    float s  = 0.5f * (1.f + wh.z);
    p2 = (1.f - s) * sqrtf(fmaxf(0.f, 1.f - p1 * p1)) + s * p2;

    // Project onto hemisphere
    float3 nh = t1 * p1 + t2 * p2 + wh * sqrtf(fmaxf(0.f, 1.f - p1 * p1 - p2 * p2));

    // Unstretch
    return normalize(make_f3(alpha * nh.x, alpha * nh.y, fmaxf(0.f, nh.z)));
}

// ── Reflect / Refract (local shading frame, N = (0,0,1)) ───────────

inline HD float3 reflect_local(float3 wo) {
    return make_f3(-wo.x, -wo.y, wo.z);
}

// Thin dielectric: pass straight through (negate wo entirely).
inline HD float3 transmit_thin_local(float3 wo) {
    return make_f3(-wo.x, -wo.y, -wo.z);
}

inline HD bool refract_local(float3 wo, float eta, float3& wt) {
    float cos_i = wo.z;
    float sin2_i = fmaxf(0.f, 1.f - cos_i * cos_i);
    float sin2_t = eta * eta * sin2_i;
    if (sin2_t >= 1.f) return false;
    float cos_t = sqrtf(1.f - sin2_t);
    // Refracted ray must cross the surface: opposite hemisphere from wo.
    wt = make_f3(-eta * wo.x, -eta * wo.y, copysignf(cos_t, -wo.z));
    return true;
}

// ── MIS weight (2-way power heuristic) ──────────────────────────────
inline HD float mis_weight_2(float pdf_a, float pdf_b) {
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    return a2 / fmaxf(a2 + b2, 1e-30f);
}

// ── Combined PDF for mixture sampling ───────────────────────────────
inline HD float bsdf_combined_pdf(float p_diff, float pdf_diff,
                                   float p_spec, float pdf_spec) {
    return p_diff * pdf_diff + p_spec * pdf_spec;
}
