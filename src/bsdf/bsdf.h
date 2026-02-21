#pragma once
// ─────────────────────────────────────────────────────────────────────
// bsdf.h – Bidirectional Scattering Distribution Functions
// ─────────────────────────────────────────────────────────────────────
// Implements:
//   - Lambertian diffuse
//   - Perfect mirror (specular reflection)
//   - Dielectric glass (specular refraction + reflection)
//   - Glossy metal (Cook-Torrance microfacet)
//
// All operations are per-wavelength-bin for spectral correctness.
// Directions are in local shading frame (z = normal).
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"
#include "core/random.h"
#include "scene/material.h"

// ── BSDF evaluation result ──────────────────────────────────────────
struct BSDFSample {
    float3   wi;          // sampled incoming direction (local frame)
    float    pdf;         // probability density
    Spectrum f;           // BSDF value f(wo, wi) per wavelength
    bool     is_specular; // true for delta distributions (mirror/glass)
};

// ── Fresnel ─────────────────────────────────────────────────────────

// Schlick approximation
inline HD float fresnel_schlick(float cos_theta, float f0) {
    float t = 1.f - cos_theta;
    float t2 = t * t;
    return f0 + (1.f - f0) * t2 * t2 * t;
}

// Exact dielectric Fresnel
inline HD float fresnel_dielectric(float cos_i, float eta) {
    float sin2_t = eta * eta * (1.f - cos_i * cos_i);
    if (sin2_t >= 1.f) return 1.f; // Total internal reflection

    float cos_t = sqrtf(fmaxf(0.f, 1.f - sin2_t));
    cos_i = fabsf(cos_i);

    float rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    float rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    return 0.5f * (rs * rs + rp * rp);
}

// ── GGX microfacet distribution ─────────────────────────────────────

inline HD float ggx_D(float3 h, float alpha) {
    float NdotH = h.z; // In local frame, N = (0,0,1)
    if (NdotH <= 0.f) return 0.f;
    float a2 = alpha * alpha;
    float d  = NdotH * NdotH * (a2 - 1.f) + 1.f;
    return a2 / (PI * d * d);
}

inline HD float ggx_G1(float3 v, float alpha) {
    float NdotV = fabsf(v.z);
    float a2 = alpha * alpha;
    return 2.f * NdotV / (NdotV + sqrtf(a2 + (1.f - a2) * NdotV * NdotV));
}

inline HD float ggx_G(float3 wo, float3 wi, float alpha) {
    return ggx_G1(wo, alpha) * ggx_G1(wi, alpha);
}

// Sample GGX visible normal (VNDF)
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

// ── Reflect / Refract ───────────────────────────────────────────────

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

// ── BSDF evaluation functions ───────────────────────────────────────

namespace bsdf {

// ── Lambertian diffuse ──────────────────────────────────────────────

inline HD Spectrum lambertian_f(const Spectrum& Kd) {
    return Kd * INV_PI;
}

inline HD BSDFSample lambertian_sample(const Spectrum& Kd, float3 /*wo*/, PCGRng& rng) {
    BSDFSample s;
    float u1 = rng.next_float();
    float u2 = rng.next_float();
    s.wi = sample_cosine_hemisphere(u1, u2);
    s.pdf = cosine_hemisphere_pdf(s.wi.z);
    s.f = Kd * INV_PI;
    s.is_specular = false;
    return s;
}

inline HD float lambertian_pdf(float3 wi) {
    return cosine_hemisphere_pdf(wi.z);
}

// ── Perfect mirror ──────────────────────────────────────────────────

inline HD BSDFSample mirror_sample(const Spectrum& Ks, float3 wo) {
    BSDFSample s;
    s.wi = reflect_local(wo);
    s.pdf = 1.f;
    s.f = Ks / fabsf(s.wi.z + EPSILON); // Delta BRDF: f = Ks * delta / cos_theta
    s.is_specular = true;
    return s;
}

// ── Dielectric glass ────────────────────────────────────────────────

inline HD BSDFSample glass_sample(float3 wo, float ior, PCGRng& rng) {
    BSDFSample s;

    bool entering = wo.z > 0.f;
    float eta = entering ? (1.f / ior) : ior;
    float cos_i = fabsf(wo.z);

    float F = fresnel_dielectric(cos_i, eta);

    if (rng.next_float() < F) {
        // Reflect
        s.wi = reflect_local(wo);
        s.pdf = F;
        s.f = Spectrum::constant(F / (fabsf(s.wi.z) + EPSILON));
        s.is_specular = true;
    } else {
        // Refract
        float3 wt;
        if (refract_local(wo, eta, wt)) {
            s.wi = wt;
            s.pdf = 1.f - F;
            float factor = (1.f - F) / (fabsf(s.wi.z) + EPSILON);
            s.f = Spectrum::constant(factor);
            s.is_specular = true;
        } else {
            // Total internal reflection fallback
            s.wi = reflect_local(wo);
            s.pdf = 1.f;
            s.f = Spectrum::constant(1.f / (fabsf(s.wi.z) + EPSILON));
            s.is_specular = true;
        }
    }
    return s;
}

// ── Cook-Torrance glossy metal ──────────────────────────────────────

inline HD BSDFSample glossy_sample(const Spectrum& Kd, const Spectrum& Ks,
                                    float roughness, float3 wo, PCGRng& rng) {
    BSDFSample s;
    float alpha = fmaxf(roughness * roughness, 0.001f);

    // Choose between diffuse and specular lobe
    float spec_weight = Ks.max_component();
    float diff_weight = Kd.max_component();
    float total = spec_weight + diff_weight;
    float p_spec = (total > 0.f) ? spec_weight / total : 0.5f;

    if (rng.next_float() < p_spec) {
        // Sample GGX specular
        float u1 = rng.next_float();
        float u2 = rng.next_float();
        float3 h = ggx_sample_halfvector(wo, alpha, u1, u2);

        s.wi = make_f3(2.f * dot(wo, h) * h.x - wo.x,
                        2.f * dot(wo, h) * h.y - wo.y,
                        2.f * dot(wo, h) * h.z - wo.z);

        if (s.wi.z <= 0.f) {
            s.pdf = 0.f;
            s.f = Spectrum::zero();
            s.is_specular = false;
            return s;
        }

        float ndf = ggx_D(h, alpha);
        float geo = ggx_G(wo, s.wi, alpha);
        float VdotH = fabsf(dot(wo, h));

        // Specular PDF (from GGX halfvector)
        float spec_pdf = ndf * fabsf(h.z) / (4.f * VdotH + EPSILON);
        float diff_pdf = cosine_hemisphere_pdf(s.wi.z);
        s.pdf = p_spec * spec_pdf + (1.f - p_spec) * diff_pdf;

        // f = D*G*F / (4*cos_o*cos_i) + Kd/pi
        float denom = 4.f * fabsf(wo.z) * fabsf(s.wi.z) + EPSILON;
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            float F_i = fresnel_schlick(VdotH, Ks.value[i]);
            s.f.value[i] = (ndf * geo * F_i) / denom + Kd.value[i] * INV_PI;
        }
    } else {
        // Sample diffuse
        float u1 = rng.next_float();
        float u2 = rng.next_float();
        s.wi = sample_cosine_hemisphere(u1, u2);

        if (s.wi.z <= 0.f) {
            s.pdf = 0.f;
            s.f = Spectrum::zero();
            s.is_specular = false;
            return s;
        }

        float diff_pdf = cosine_hemisphere_pdf(s.wi.z);

        // Compute specular PDF for this direction
        float3 h = normalize(wo + s.wi);
        float ndf = ggx_D(h, alpha);
        float VdotH = fabsf(dot(wo, h));
        float spec_pdf = ndf * fabsf(h.z) / (4.f * VdotH + EPSILON);

        s.pdf = p_spec * spec_pdf + (1.f - p_spec) * diff_pdf;

        float geo = ggx_G(wo, s.wi, alpha);
        float denom = 4.f * fabsf(wo.z) * fabsf(s.wi.z) + EPSILON;
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            float F_i = fresnel_schlick(VdotH, Ks.value[i]);
            s.f.value[i] = (ndf * geo * F_i) / denom + Kd.value[i] * INV_PI;
        }
    }

    s.is_specular = false;
    return s;
}

// ── Evaluate diffuse-only BSDF (for photon density estimation) ──────
// Standard photon mapping practice: gather uses only the diffuse
// component.  The peaked specular lobe creates unbounded variance in
// fixed-radius kernel estimators, producing visible coloured hotspots.

inline HD Spectrum evaluate_diffuse(const Material& mat, float3 wo, float3 wi) {
    if (wi.z <= 0.f || wo.z <= 0.f) return Spectrum::zero();
    switch (mat.type) {
        case MaterialType::Mirror:
        case MaterialType::Glass:
            return Spectrum::zero();   // delta distributions
        default:
            return lambertian_f(mat.Kd);
    }
}

// ── Evaluate BSDF for given directions ──────────────────────────────

inline HD Spectrum evaluate(const Material& mat, float3 wo, float3 wi) {
    if (wi.z <= 0.f || wo.z <= 0.f) return Spectrum::zero();

    switch (mat.type) {
        case MaterialType::Lambertian:
            return lambertian_f(mat.Kd);

        case MaterialType::GlossyMetal: {
            float alpha = fmaxf(mat.roughness * mat.roughness, 0.001f);
            float3 h = normalize(wo + wi);
            float ndf = ggx_D(h, alpha);
            float geo = ggx_G(wo, wi, alpha);
            float VdotH = fabsf(dot(wo, h));
            float denom = 4.f * fabsf(wo.z) * fabsf(wi.z) + EPSILON;
            Spectrum f;
            for (int i = 0; i < NUM_LAMBDA; ++i) {
                float Fr = fresnel_schlick(VdotH, mat.Ks.value[i]);
                f.value[i] = (ndf * geo * Fr) / denom + mat.Kd.value[i] * INV_PI;
            }
            return f;
        }

        case MaterialType::Mirror:
        case MaterialType::Glass:
            // Delta distributions: f is zero for non-delta directions
            return Spectrum::zero();

        case MaterialType::Emissive:
            return lambertian_f(mat.Kd);

        default:
            return Spectrum::zero();
    }
}

// ── PDF for given directions ────────────────────────────────────────

inline HD float pdf(const Material& mat, float3 wo, float3 wi) {
    if (wi.z <= 0.f || wo.z <= 0.f) return 0.f;

    switch (mat.type) {
        case MaterialType::Lambertian:
        case MaterialType::Emissive:
            return lambertian_pdf(wi);

        case MaterialType::GlossyMetal: {
            float alpha = fmaxf(mat.roughness * mat.roughness, 0.001f);
            float spec_weight = mat.Ks.max_component();
            float diff_weight = mat.Kd.max_component();
            float total = spec_weight + diff_weight;
            float p_spec = (total > 0.f) ? spec_weight / total : 0.5f;

            float diff_pdf = cosine_hemisphere_pdf(wi.z);
            float3 h = normalize(wo + wi);
            float ndf = ggx_D(h, alpha);
            float VdotH = fabsf(dot(wo, h));
            float spec_pdf = ndf * fabsf(h.z) / (4.f * VdotH + EPSILON);

            return p_spec * spec_pdf + (1.f - p_spec) * diff_pdf;
        }

        case MaterialType::Mirror:
        case MaterialType::Glass:
            return 0.f; // Delta distribution

        default:
            return 0.f;
    }
}

// ── Sample a direction from the BSDF ───────────────────────────────

inline HD BSDFSample sample(const Material& mat, float3 wo, PCGRng& rng) {
    switch (mat.type) {
        case MaterialType::Lambertian:
        case MaterialType::Emissive:
            return lambertian_sample(mat.Kd, wo, rng);

        case MaterialType::Mirror:
            return mirror_sample(mat.Ks, wo);

        case MaterialType::Glass:
            return glass_sample(wo, mat.ior, rng);

        case MaterialType::GlossyMetal:
            return glossy_sample(mat.Kd, mat.Ks, mat.roughness, wo, rng);

        default:
            return lambertian_sample(mat.Kd, wo, rng);
    }
}

} // namespace bsdf
