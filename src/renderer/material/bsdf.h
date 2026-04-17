#pragma once
// ─────────────────────────────────────────────────────────────────────
// bsdf.h – Bidirectional Scattering Distribution Functions (v5 RGB)
//
// Implements:
//   - Lambertian diffuse
//   - Perfect mirror (specular reflection)
//   - Dielectric glass (specular refraction + reflection)
//   - Glossy metal (Cook-Torrance microfacet)
//   - Glossy dielectric (IOR-based Fresnel + diffuse)
//   - Clearcoat (layered GGX coat over diffuse base)
//   - Fabric (diffuse + Fresnel sheen)
//
// Ported from v4: Spectrum→Color3, per-bin loops→Color3 ops.
// Glass dispersion simplified: scalar IOR + Color3 Tf filter (no
// per-channel refraction in RGB pipeline).
// Directions are in local shading frame (z = normal).
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/color.h"
#include "core/random.h"
#include "scene/material.h"
#include "material/bsdf_shared.h"

// ── BSDF evaluation result ──────────────────────────────────────────
struct BSDFSample {
    float3   wi;          // sampled incoming direction (local frame)
    float    pdf;         // probability density
    Color3   f;           // BSDF value f(wo, wi)
    bool     is_specular; // true for delta distributions (mirror/glass)
};

namespace bsdf {

// ── Lambertian diffuse ──────────────────────────────────────────────

inline HD Color3 lambertian_f(const Color3& Kd) {
    return Kd * INV_PI;
}

inline HD BSDFSample lambertian_sample(const Color3& Kd, float3 /*wo*/, PCGRng& rng) {
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

inline HD BSDFSample mirror_sample(const Color3& Ks, float3 wo) {
    BSDFSample s;
    s.wi = reflect_local(wo);
    s.pdf = 1.f;
    s.f = Ks / fabsf(s.wi.z + EPSILON); // Delta BRDF: f = Ks * delta / cos_theta
    s.is_specular = true;
    return s;
}

// ── Dielectric glass (RGB: scalar IOR + Color3 Tf filter) ───────────
// No per-channel refraction in RGB pipeline. Dispersion flag is
// ignored — refraction always uses the scalar IOR.

inline HD BSDFSample glass_sample(float3 wo, const Material& mat, PCGRng& rng,
                                  TransportMode mode = TransportMode::Radiance) {
    BSDFSample s;

    bool entering = wo.z > 0.f;
    float cos_i = fabsf(wo.z);
    float eta = entering ? (1.f / mat.ior) : mat.ior;

    float F = fresnel_dielectric(cos_i, eta);

    bool do_reflect = (rng.next_float() < F);

    if (do_reflect) {
        s.wi = reflect_local(wo);
        s.pdf = F;
        float factor = F / (fabsf(s.wi.z) + EPSILON);
        s.f = Color3::constant(factor);
        s.is_specular = true;
    } else {
        float3 wt;
        if (refract_local(wo, eta, wt)) {
            s.wi = wt;
            s.pdf = 1.f - F;
            float factor = (1.f - F) / (fabsf(s.wi.z) + EPSILON);
            s.f = mat.Tf * factor;

            // Adjoint η² correction for importance (photon) transport
            if (mode == TransportMode::Importance) {
                s.f = s.f * (eta * eta);
            }
            s.is_specular = true;
        } else {
            // Total internal reflection fallback
            s.wi = reflect_local(wo);
            s.pdf = 1.f;
            float factor = 1.f / (fabsf(s.wi.z) + EPSILON);
            s.f = Color3::constant(factor);
            s.is_specular = true;
        }
    }
    return s;
}

// Legacy overload — scalar IOR, no material struct
inline HD BSDFSample glass_sample(float3 wo, float ior, PCGRng& rng) {
    Material m;
    m.ior = ior;
    m.type = MaterialType::Glass;
    m.Tf = Color3::one();
    m.dispersion = false;
    return glass_sample(wo, m, rng);
}

// ── Cook-Torrance glossy metal ──────────────────────────────────────

inline HD BSDFSample glossy_sample(const Color3& Kd, const Color3& Ks,
                                    float roughness, float3 wo, PCGRng& rng) {
    BSDFSample s;
    float alpha = bsdf_roughness_to_alpha(roughness);

    float spec_weight = Ks.max_component();
    float diff_weight = Kd.max_component();
    float total = spec_weight + diff_weight;
    float p_spec = (total > 0.f) ? spec_weight / total : 0.5f;
    p_spec = fmaxf(0.05f, fminf(0.95f, p_spec));

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
            s.f = Color3::zero();
            s.is_specular = false;
            return s;
        }

        float ndf = ggx_D(h, alpha);
        float geo = ggx_G(wo, s.wi, alpha);
        float VdotH = fabsf(dot(wo, h));

        float spec_pdf = ndf * ggx_G1(wo, alpha) / (4.f * fabsf(wo.z) + EPSILON);
        float diff_pdf = cosine_hemisphere_pdf(s.wi.z);
        s.pdf = p_spec * spec_pdf + (1.f - p_spec) * diff_pdf;

        float denom = 4.f * fabsf(wo.z) * fabsf(s.wi.z) + EPSILON;
        // Per-channel Schlick Fresnel for metallic reflectance
        Color3 F_c = {fresnel_schlick(VdotH, Ks.r),
                      fresnel_schlick(VdotH, Ks.g),
                      fresnel_schlick(VdotH, Ks.b)};
        s.f = F_c * (ndf * geo / denom) + Kd * INV_PI;
    } else {
        // Sample diffuse
        float u1 = rng.next_float();
        float u2 = rng.next_float();
        s.wi = sample_cosine_hemisphere(u1, u2);

        if (s.wi.z <= 0.f) {
            s.pdf = 0.f;
            s.f = Color3::zero();
            s.is_specular = false;
            return s;
        }

        float diff_pdf = cosine_hemisphere_pdf(s.wi.z);

        float3 h = normalize(wo + s.wi);
        float ndf = ggx_D(h, alpha);
        float VdotH = fabsf(dot(wo, h));
        float spec_pdf = ndf * ggx_G1(wo, alpha) / (4.f * fabsf(wo.z) + EPSILON);

        s.pdf = p_spec * spec_pdf + (1.f - p_spec) * diff_pdf;

        float geo = ggx_G(wo, s.wi, alpha);
        float denom = 4.f * fabsf(wo.z) * fabsf(s.wi.z) + EPSILON;
        Color3 F_c = {fresnel_schlick(VdotH, Ks.r),
                      fresnel_schlick(VdotH, Ks.g),
                      fresnel_schlick(VdotH, Ks.b)};
        s.f = F_c * (ndf * geo / denom) + Kd * INV_PI;
    }

    s.is_specular = false;
    return s;
}

// ── Cook-Torrance glossy dielectric ─────────────────────────────────

inline HD BSDFSample glossy_dielectric_sample(const Color3& Kd, const Color3& Ks,
                                               float roughness, float ior,
                                               float3 wo, PCGRng& rng) {
    BSDFSample s;
    float alpha = bsdf_roughness_to_alpha(roughness);

    float f0t = (ior - 1.f) / (ior + 1.f);
    float F0 = f0t * f0t;

    float spec_weight = Ks.max_component() * F0;
    float diff_weight = Kd.max_component();
    float total = spec_weight + diff_weight;
    float p_spec = (total > 0.f) ? spec_weight / total : 0.5f;
    p_spec = fmaxf(p_spec, 0.05f);

    if (rng.next_float() < p_spec) {
        float u1 = rng.next_float();
        float u2 = rng.next_float();
        float3 h = ggx_sample_halfvector(wo, alpha, u1, u2);

        s.wi = make_f3(2.f * dot(wo, h) * h.x - wo.x,
                        2.f * dot(wo, h) * h.y - wo.y,
                        2.f * dot(wo, h) * h.z - wo.z);

        if (s.wi.z <= 0.f) {
            s.pdf = 0.f;
            s.f = Color3::zero();
            s.is_specular = false;
            return s;
        }

        float ndf = ggx_D(h, alpha);
        float geo = ggx_G(wo, s.wi, alpha);
        float VdotH = fabsf(dot(wo, h));

        float spec_pdf = ndf * ggx_G1(wo, alpha) / (4.f * fabsf(wo.z) + EPSILON);
        float diff_pdf = cosine_hemisphere_pdf(s.wi.z);
        s.pdf = p_spec * spec_pdf + (1.f - p_spec) * diff_pdf;

        float denom = 4.f * fabsf(wo.z) * fabsf(s.wi.z) + EPSILON;
        float Fr = fresnel_schlick(VdotH, F0);
        s.f = Ks * (ndf * geo * Fr / denom) + Kd * ((1.f - Fr) * INV_PI);
    } else {
        float u1 = rng.next_float();
        float u2 = rng.next_float();
        s.wi = sample_cosine_hemisphere(u1, u2);

        if (s.wi.z <= 0.f) {
            s.pdf = 0.f;
            s.f = Color3::zero();
            s.is_specular = false;
            return s;
        }

        float diff_pdf = cosine_hemisphere_pdf(s.wi.z);

        float3 h = normalize(wo + s.wi);
        float ndf = ggx_D(h, alpha);
        float VdotH = fabsf(dot(wo, h));
        float spec_pdf = ndf * ggx_G1(wo, alpha) / (4.f * fabsf(wo.z) + EPSILON);

        s.pdf = p_spec * spec_pdf + (1.f - p_spec) * diff_pdf;

        float geo = ggx_G(wo, s.wi, alpha);
        float denom = 4.f * fabsf(wo.z) * fabsf(s.wi.z) + EPSILON;
        float Fr = fresnel_schlick(VdotH, F0);
        s.f = Ks * (ndf * geo * Fr / denom) + Kd * ((1.f - Fr) * INV_PI);
    }

    s.is_specular = false;
    return s;
}

// ── Clearcoat (GGX coat over diffuse base) ──────────────────────────
// NOTE: Simplified simultaneous blend of coat + base lobes, not
// physically-correct stochastic layer traversal.  The base diffuse is
// attenuated by (1 - coat_weight * Fr), which is a reasonable
// approximation but the integral is not guaranteed energy-conserving.

inline HD BSDFSample clearcoat_sample(const Material& mat, float3 wo, PCGRng& rng) {
    BSDFSample s;

    float coat_weight = mat.pb_clearcoat;
    float coat_alpha  = bsdf_roughness_to_alpha(mat.pb_clearcoat_roughness);

    float coat_f0t = (mat.ior - 1.f) / (mat.ior + 1.f);
    float coat_F0  = coat_f0t * coat_f0t;

    float p_coat = coat_weight * coat_F0;
    p_coat = fmaxf(p_coat, 0.05f);
    p_coat = fminf(p_coat, 0.95f);

    if (rng.next_float() < p_coat) {
        float3 h = ggx_sample_halfvector(wo, coat_alpha, rng.next_float(), rng.next_float());
        s.wi = make_f3(2.f * dot(wo, h) * h.x - wo.x,
                        2.f * dot(wo, h) * h.y - wo.y,
                        2.f * dot(wo, h) * h.z - wo.z);
        if (s.wi.z <= 0.f) {
            s.pdf = 0.f; s.f = Color3::zero(); s.is_specular = false;
            return s;
        }
        float VdotH = fabsf(dot(wo, h));
        float Fr = fresnel_schlick(VdotH, coat_F0);
        float ndf_c = ggx_D(h, coat_alpha);
        float geo_c = ggx_G(wo, s.wi, coat_alpha);
        float spec_pdf = ndf_c * ggx_G1(wo, coat_alpha) / (4.f * fabsf(wo.z) + EPSILON);
        float diff_pdf = cosine_hemisphere_pdf(s.wi.z);
        s.pdf = p_coat * spec_pdf + (1.f - p_coat) * diff_pdf;

        float denom = 4.f * fabsf(wo.z) * fabsf(s.wi.z) + EPSILON;
        float coat_spec = coat_weight * (ndf_c * geo_c * Fr) / denom;
        s.f = Color3::constant(coat_spec) + mat.Kd * ((1.f - coat_weight * Fr) * INV_PI);
    } else {
        s.wi = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
        if (s.wi.z <= 0.f) {
            s.pdf = 0.f; s.f = Color3::zero(); s.is_specular = false;
            return s;
        }
        float diff_pdf = cosine_hemisphere_pdf(s.wi.z);
        float3 h = normalize(wo + s.wi);
        float ndf_c = ggx_D(h, coat_alpha);
        float VdotH = fabsf(dot(wo, h));
        float spec_pdf = ndf_c * ggx_G1(wo, coat_alpha) / (4.f * fabsf(wo.z) + EPSILON);
        s.pdf = p_coat * spec_pdf + (1.f - p_coat) * diff_pdf;

        float Fr = fresnel_schlick(VdotH, coat_F0);
        float geo_c = ggx_G(wo, s.wi, coat_alpha);
        float denom = 4.f * fabsf(wo.z) * fabsf(s.wi.z) + EPSILON;
        float coat_spec = coat_weight * (ndf_c * geo_c * Fr) / denom;
        s.f = Color3::constant(coat_spec) + mat.Kd * ((1.f - coat_weight * Fr) * INV_PI);
    }
    s.is_specular = false;
    return s;
}

// ── Fabric (diffuse + sheen) ────────────────────────────────────────

inline HD BSDFSample fabric_sample(const Material& mat, float3 wo, PCGRng& rng) {
    BSDFSample s;

    s.wi = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
    if (s.wi.z <= 0.f) {
        s.pdf = 0.f; s.f = Color3::zero(); s.is_specular = false;
        return s;
    }
    s.pdf = cosine_hemisphere_pdf(s.wi.z);

    float3 h = normalize(wo + s.wi);
    float cos_theta_h = fabsf(dot(wo, h));
    float t = 1.f - cos_theta_h;
    float t5 = t * t * t * t * t;

    float sheen_w = mat.pb_sheen;
    float tint    = mat.pb_sheen_tint;

    // Sheen colour: blend between white and Kd-tinted
    Color3 sheen_col = Color3::one() * (1.f - tint) + mat.Kd * tint;
    Color3 sheen = sheen_col * (sheen_w * t5 * INV_PI);
    s.f = mat.Kd * INV_PI + sheen;
    s.is_specular = false;
    return s;
}

// ── Diffuse transmission (two-sided Lambert: R reflect, T transmit) ─

inline HD BSDFSample diffuse_transmission_sample(const Material& mat, float3 wo, PCGRng& rng) {
    BSDFSample s;
    (void)wo;  // cosine hemisphere is independent of view direction

    // Lobe selection: reflect (Kd) vs transmit (Tf)
    float w_r = mat.Kd.max_component();
    float w_t = mat.Tf.max_component();
    float total = w_r + w_t;
    float p_r = (total > 0.f) ? w_r / total : 0.5f;
    p_r = fmaxf(0.1f, fminf(0.9f, p_r));
    float p_t = 1.f - p_r;

    float u1 = rng.next_float(), u2 = rng.next_float();
    if (rng.next_float() < p_r) {
        // Reflection: cosine hemisphere, same side as wo
        s.wi = sample_cosine_hemisphere(u1, u2);
        s.pdf = p_r * cosine_hemisphere_pdf(s.wi.z);
        s.f = mat.Kd * INV_PI;
    } else {
        // Transmission: cosine hemisphere, opposite side
        s.wi = sample_cosine_hemisphere(u1, u2);
        s.wi.z = -s.wi.z;
        s.pdf = p_t * cosine_hemisphere_pdf(fabsf(s.wi.z));
        s.f = mat.Tf * INV_PI;
    }
    s.is_specular = false;
    return s;
}

inline HD Color3 diffuse_transmission_evaluate(const Material& mat, float3 wo, float3 wi) {
    if (wo.z <= 0.f) return Color3::zero();
    if (wi.z > 0.f) return mat.Kd * INV_PI;   // same hemisphere → reflect
    if (wi.z < 0.f) return mat.Tf * INV_PI;    // opposite hemisphere → transmit
    return Color3::zero();
}

inline HD float diffuse_transmission_pdf(const Material& mat, float3 wo, float3 wi) {
    if (wo.z <= 0.f) return 0.f;
    float w_r = mat.Kd.max_component();
    float w_t = mat.Tf.max_component();
    float total = w_r + w_t;
    float p_r = (total > 0.f) ? w_r / total : 0.5f;
    p_r = fmaxf(0.1f, fminf(0.9f, p_r));

    if (wi.z > 0.f) return p_r * cosine_hemisphere_pdf(wi.z);
    if (wi.z < 0.f) return (1.f - p_r) * cosine_hemisphere_pdf(fabsf(wi.z));
    return 0.f;
}

// ── Evaluate diffuse-only BSDF (for photon density estimation) ──────

inline HD Color3 evaluate_diffuse(const Material& mat, float3 wo, float3 wi) {
    if (wi.z <= 0.f || wo.z <= 0.f) return Color3::zero();
    switch (mat.type) {
        case MaterialType::Mirror:
        case MaterialType::Glass:
        case MaterialType::Translucent:
            return Color3::zero();
        case MaterialType::Clearcoat: {
            float coat_f0t = (mat.ior - 1.f) / (mat.ior + 1.f);
            float coat_F0  = coat_f0t * coat_f0t;
            float cos_o = fabsf(wo.z);
            float Fr = fresnel_schlick(cos_o, coat_F0);
            return mat.Kd * ((1.f - mat.pb_clearcoat * Fr) * INV_PI);
        }
        case MaterialType::Fabric:
            return lambertian_f(mat.Kd);
        default:
            return lambertian_f(mat.Kd);
    }
}

// ── Evaluate BSDF for given directions ──────────────────────────────

inline HD Color3 evaluate(const Material& mat, float3 wo, float3 wi) {
    // DiffuseTransmission allows wi.z < 0 (transmission lobe)
    if (mat.type == MaterialType::DiffuseTransmission)
        return diffuse_transmission_evaluate(mat, wo, wi);
    if (wi.z <= 0.f || wo.z <= 0.f) return Color3::zero();

    switch (mat.type) {
        case MaterialType::Lambertian:
            return lambertian_f(mat.Kd);

        case MaterialType::GlossyMetal: {
            float alpha = bsdf_roughness_to_alpha(mat.roughness);
            float3 h = normalize(wo + wi);
            float ndf = ggx_D(h, alpha);
            float geo = ggx_G(wo, wi, alpha);
            float VdotH = fabsf(dot(wo, h));
            float denom = 4.f * fabsf(wo.z) * fabsf(wi.z) + EPSILON;
            Color3 F_c = {fresnel_schlick(VdotH, mat.Ks.r),
                          fresnel_schlick(VdotH, mat.Ks.g),
                          fresnel_schlick(VdotH, mat.Ks.b)};
            return F_c * (ndf * geo / denom) + mat.Kd * INV_PI;
        }

        case MaterialType::GlossyDielectric: {
            float alpha = bsdf_roughness_to_alpha(mat.roughness);
            float3 h = normalize(wo + wi);
            float ndf = ggx_D(h, alpha);
            float geo = ggx_G(wo, wi, alpha);
            float VdotH = fabsf(dot(wo, h));
            float denom = 4.f * fabsf(wo.z) * fabsf(wi.z) + EPSILON;
            float f0t = (mat.ior - 1.f) / (mat.ior + 1.f);
            float F0 = f0t * f0t;
            float Fr = fresnel_schlick(VdotH, F0);
            return mat.Ks * (ndf * geo * Fr / denom) + mat.Kd * ((1.f - Fr) * INV_PI);
        }

        case MaterialType::Mirror:
        case MaterialType::Glass:
        case MaterialType::Translucent:
            return Color3::zero(); // Delta distributions

        case MaterialType::Emissive:
            return lambertian_f(mat.Kd);

        case MaterialType::Clearcoat: {
            float coat_alpha = bsdf_roughness_to_alpha(mat.pb_clearcoat_roughness);
            float coat_f0t = (mat.ior - 1.f) / (mat.ior + 1.f);
            float coat_F0  = coat_f0t * coat_f0t;
            float3 h = normalize(wo + wi);
            float ndf_c = ggx_D(h, coat_alpha);
            float geo_c = ggx_G(wo, wi, coat_alpha);
            float VdotH = fabsf(dot(wo, h));
            float Fr = fresnel_schlick(VdotH, coat_F0);
            float denom = 4.f * fabsf(wo.z) * fabsf(wi.z) + EPSILON;
            float coat_spec = mat.pb_clearcoat * (ndf_c * geo_c * Fr) / denom;
            return Color3::constant(coat_spec) + mat.Kd * ((1.f - mat.pb_clearcoat * Fr) * INV_PI);
        }

        case MaterialType::Fabric: {
            float3 h = normalize(wo + wi);
            float cos_theta_h = fabsf(dot(wo, h));
            float t = 1.f - cos_theta_h;
            float t5 = t * t * t * t * t;
            Color3 sheen_col = Color3::one() * (1.f - mat.pb_sheen_tint) + mat.Kd * mat.pb_sheen_tint;
            return mat.Kd * INV_PI + sheen_col * (mat.pb_sheen * t5 * INV_PI);
        }

        default:
            return Color3::zero();
    }
}

// ── PDF for given directions ────────────────────────────────────────

inline HD float pdf(const Material& mat, float3 wo, float3 wi) {
    // DiffuseTransmission allows wi.z < 0 (transmission lobe)
    if (mat.type == MaterialType::DiffuseTransmission)
        return diffuse_transmission_pdf(mat, wo, wi);
    if (wi.z <= 0.f || wo.z <= 0.f) return 0.f;

    switch (mat.type) {
        case MaterialType::Lambertian:
        case MaterialType::Emissive:
            return lambertian_pdf(wi);

        case MaterialType::GlossyMetal: {
            float alpha = bsdf_roughness_to_alpha(mat.roughness);
            float spec_weight = mat.Ks.max_component();
            float diff_weight = mat.Kd.max_component();
            float total = spec_weight + diff_weight;
            float p_spec = (total > 0.f) ? spec_weight / total : 0.5f;

            float diff_pdf = cosine_hemisphere_pdf(wi.z);
            float3 h = normalize(wo + wi);
            float ndf = ggx_D(h, alpha);
            float spec_pdf = ndf * ggx_G1(wo, alpha) / (4.f * fabsf(wo.z) + EPSILON);

            return p_spec * spec_pdf + (1.f - p_spec) * diff_pdf;
        }

        case MaterialType::GlossyDielectric: {
            float alpha = bsdf_roughness_to_alpha(mat.roughness);
            float f0t = (mat.ior - 1.f) / (mat.ior + 1.f);
            float F0 = f0t * f0t;
            float spec_weight = mat.Ks.max_component() * F0;
            float diff_weight = mat.Kd.max_component();
            float total = spec_weight + diff_weight;
            float p_spec = (total > 0.f) ? spec_weight / total : 0.5f;
            p_spec = fmaxf(p_spec, 0.05f);

            float diff_pdf = cosine_hemisphere_pdf(wi.z);
            float3 h = normalize(wo + wi);
            float ndf = ggx_D(h, alpha);
            float spec_pdf = ndf * ggx_G1(wo, alpha) / (4.f * fabsf(wo.z) + EPSILON);

            return p_spec * spec_pdf + (1.f - p_spec) * diff_pdf;
        }

        case MaterialType::Mirror:
        case MaterialType::Glass:
        case MaterialType::Translucent:
            return 0.f;

        case MaterialType::Clearcoat: {
            float coat_alpha = bsdf_roughness_to_alpha(mat.pb_clearcoat_roughness);
            float coat_f0t = (mat.ior - 1.f) / (mat.ior + 1.f);
            float coat_F0  = coat_f0t * coat_f0t;
            float p_coat = fmaxf(fminf(mat.pb_clearcoat * coat_F0, 0.95f), 0.05f);
            float diff_pdf = cosine_hemisphere_pdf(wi.z);
            float3 h = normalize(wo + wi);
            float ndf_c = ggx_D(h, coat_alpha);
            float spec_pdf = ndf_c * ggx_G1(wo, coat_alpha) / (4.f * fabsf(wo.z) + EPSILON);
            return p_coat * spec_pdf + (1.f - p_coat) * diff_pdf;
        }

        case MaterialType::Fabric:
            return cosine_hemisphere_pdf(wi.z);

        default:
            return 0.f;
    }
}

// ── Sample a direction from the BSDF ───────────────────────────────

inline HD BSDFSample sample(const Material& mat, float3 wo, PCGRng& rng,
                            TransportMode mode = TransportMode::Radiance) {
    switch (mat.type) {
        case MaterialType::Lambertian:
        case MaterialType::Emissive:
            return lambertian_sample(mat.Kd, wo, rng);

        case MaterialType::Mirror:
            return mirror_sample(mat.Ks, wo);

        case MaterialType::Glass:
        case MaterialType::Translucent:
            return glass_sample(wo, mat, rng, mode);

        case MaterialType::GlossyMetal:
            return glossy_sample(mat.Kd, mat.Ks, mat.roughness, wo, rng);

        case MaterialType::GlossyDielectric:
            return glossy_dielectric_sample(mat.Kd, mat.Ks, mat.roughness, mat.ior, wo, rng);

        case MaterialType::Clearcoat:
            return clearcoat_sample(mat, wo, rng);

        case MaterialType::Fabric:
            return fabric_sample(mat, wo, rng);

        case MaterialType::DiffuseTransmission:
            return diffuse_transmission_sample(mat, wo, rng);

        default:
            return lambertian_sample(mat.Kd, wo, rng);
    }
}

} // namespace bsdf
