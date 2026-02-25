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
#include "bsdf/bsdf_shared.h"

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

// ── Dielectric glass (with spectral Tf and chromatic dispersion) ────

inline HD BSDFSample glass_sample(float3 wo, const Material& mat, PCGRng& rng,
                                  int hero_bin = -1) {
    BSDFSample s;

    bool entering = wo.z > 0.f;
    float cos_i = fabsf(wo.z);

    // Hero wavelength determines the refraction direction when dispersion is
    // enabled; all other bins get per-wavelength Fresnel weights but share
    // the same ray direction (spectral MIS à la PBRT v4).
    //
    // hero_bin < 0 → use D-line bin (~589 nm, nominal IOR). This is the
    //   correct default for camera paths: the refraction direction should
    //   match the material's stated IOR (pb_eta).
    // hero_bin >= 0 → use that specific bin. Photon tracers pass the
    //   photon's primary hero bin so each photon refracts at its own
    //   wavelength, producing physically correct chromatic dispersion.
    constexpr int DLINE_BIN = (int)((589.0f - LAMBDA_MIN) / LAMBDA_STEP);
    int effective_hero = (hero_bin >= 0 && hero_bin < NUM_LAMBDA)
                             ? hero_bin : DLINE_BIN;
    float hero_ior = mat.dispersion ? mat.ior_at_lambda(lambda_of_bin(effective_hero))
                                    : mat.ior;
    float eta = entering ? (1.f / hero_ior) : hero_ior;

    float F = fresnel_dielectric(cos_i, eta);

    bool do_reflect = (rng.next_float() < F);

    if (do_reflect) {
        // Reflect — Tf is NOT applied to reflection.  Fresnel reflection
        // off a dielectric surface is colour-neutral; only transmitted
        // light is filtered by the glass body colour (Tf).
        s.wi = reflect_local(wo);
        s.pdf = F;
        if (mat.dispersion) {
            // Per-wavelength Fresnel weighting on reflection
            for (int b = 0; b < NUM_LAMBDA; ++b) {
                float n_b  = mat.ior_at_lambda(lambda_of_bin(b));
                float eta_b = entering ? (1.f / n_b) : n_b;
                float F_b  = fresnel_dielectric(cos_i, eta_b);
                s.f.value[b] = F_b / (fabsf(s.wi.z) + EPSILON);
            }
        } else {
            float base_factor = F / (fabsf(s.wi.z) + EPSILON);
            s.f = Spectrum::constant(base_factor);
        }
        s.is_specular = true;
    } else {
        // Refract — direction from hero wavelength IOR
        float3 wt;
        if (refract_local(wo, eta, wt)) {
            s.wi = wt;
            s.pdf = 1.f - F;

            if (mat.dispersion) {
                // Per-wavelength Fresnel for chromatic dispersion.
                // Each bin uses its own IOR for the Fresnel coefficient but
                // shares the hero-wavelength refraction direction.  Bins that
                // would undergo TIR at their own IOR get zero weight.
                for (int b = 0; b < NUM_LAMBDA; ++b) {
                    float lam  = lambda_of_bin(b);
                    float n_b  = mat.ior_at_lambda(lam);
                    float eta_b = entering ? (1.f / n_b) : n_b;
                    // Check per-bin TIR
                    float sin2_t_b = eta_b * eta_b * (1.f - cos_i * cos_i);
                    if (sin2_t_b >= 1.f) {
                        s.f.value[b] = 0.f; // TIR for this wavelength
                        continue;
                    }
                    float F_b = fresnel_dielectric(cos_i, eta_b);
                    float factor_b = (1.f - F_b) / (fabsf(s.wi.z) + EPSILON);
                    s.f.value[b] = mat.Tf.value[b] * factor_b;
                }
            } else {
                float base_factor = (1.f - F) / (fabsf(s.wi.z) + EPSILON);
                s.f = mat.Tf * base_factor;
            }
            s.is_specular = true;
        } else {
            // Total internal reflection fallback — no Tf (pure reflection)
            s.wi = reflect_local(wo);
            s.pdf = 1.f;
            float factor = 1.f / (fabsf(s.wi.z) + EPSILON);
            s.f = Spectrum::constant(factor);
            s.is_specular = true;
        }
    }
    return s;
}

// Legacy overload — DEPRECATED.  Use glass_sample(wo, mat, rng) instead.
// Exists only for backward compatibility; creates a dummy Material with
// neutral Tf and no dispersion.  External callers should migrate.
inline HD BSDFSample glass_sample(float3 wo, float ior, PCGRng& rng) {
    Material m;
    m.ior = ior;
    m.type = MaterialType::Glass;
    m.Tf = Spectrum::constant(1.0f);
    m.dispersion = false;
    return glass_sample(wo, m, rng);
}

// ── Cook-Torrance glossy metal ──────────────────────────────────────

inline HD BSDFSample glossy_sample(const Spectrum& Kd, const Spectrum& Ks,
                                    float roughness, float3 wo, PCGRng& rng) {
    BSDFSample s;
    float alpha = bsdf_roughness_to_alpha(roughness);

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

// ── Cook-Torrance glossy dielectric ─────────────────────────────────
// Uses IOR-based F0 for Fresnel (not Ks). Ks scales the specular lobe
// colour/intensity. Diffuse is energy-conserving: weighted by (1 - Fr).

inline HD BSDFSample glossy_dielectric_sample(const Spectrum& Kd, const Spectrum& Ks,
                                               float roughness, float ior,
                                               float3 wo, PCGRng& rng) {
    BSDFSample s;
    float alpha = bsdf_roughness_to_alpha(roughness);

    // Dielectric F0 from IOR
    float f0t = (ior - 1.f) / (ior + 1.f);
    float F0 = f0t * f0t;

    // Sampling weights — scale specular by F0 so mostly-diffuse surfaces
    // don't waste samples on the tiny specular peak
    float spec_weight = Ks.max_component() * F0;
    float diff_weight = Kd.max_component();
    float total = spec_weight + diff_weight;
    float p_spec = (total > 0.f) ? spec_weight / total : 0.5f;
    p_spec = fmaxf(p_spec, 0.05f); // ensure highlights get some samples

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

        float spec_pdf = ndf * fabsf(h.z) / (4.f * VdotH + EPSILON);
        float diff_pdf = cosine_hemisphere_pdf(s.wi.z);
        s.pdf = p_spec * spec_pdf + (1.f - p_spec) * diff_pdf;

        float denom = 4.f * fabsf(wo.z) * fabsf(s.wi.z) + EPSILON;
        float Fr = fresnel_schlick(VdotH, F0);
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            s.f.value[i] = Ks.value[i] * (ndf * geo * Fr) / denom
                          + (1.f - Fr) * Kd.value[i] * INV_PI;
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

        float3 h = normalize(wo + s.wi);
        float ndf = ggx_D(h, alpha);
        float VdotH = fabsf(dot(wo, h));
        float spec_pdf = ndf * fabsf(h.z) / (4.f * VdotH + EPSILON);

        s.pdf = p_spec * spec_pdf + (1.f - p_spec) * diff_pdf;

        float geo = ggx_G(wo, s.wi, alpha);
        float denom = 4.f * fabsf(wo.z) * fabsf(s.wi.z) + EPSILON;
        float Fr = fresnel_schlick(VdotH, F0);
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            s.f.value[i] = Ks.value[i] * (ndf * geo * Fr) / denom
                          + (1.f - Fr) * Kd.value[i] * INV_PI;
        }
    }

    s.is_specular = false;
    return s;
}

// ── Clearcoat (layered: dielectric coat over base BRDF) ─────────────
// Coat: GGX microfacet dielectric lobe with its own roughness.
// Base: Lambert by default (or GlossyDielectric if pb_base_brdf).
// Energy: coat reflection is removed from base energy (1 - Fr_coat).

inline HD BSDFSample clearcoat_sample(const Material& mat, float3 wo, PCGRng& rng) {
    BSDFSample s;

    float coat_weight = mat.pb_clearcoat;
    float coat_alpha  = bsdf_roughness_to_alpha(mat.pb_clearcoat_roughness);

    // Coat IOR → F0
    float coat_f0t = (mat.ior - 1.f) / (mat.ior + 1.f);
    float coat_F0  = coat_f0t * coat_f0t;

    // Decide: sample coat vs base
    // Rough estimate of coat contribution at normal incidence
    float p_coat = coat_weight * coat_F0;
    p_coat = fmaxf(p_coat, 0.05f);
    p_coat = fminf(p_coat, 0.95f);

    if (rng.next_float() < p_coat) {
        // Sample coat GGX
        float3 h = ggx_sample_halfvector(wo, coat_alpha, rng.next_float(), rng.next_float());
        s.wi = make_f3(2.f * dot(wo, h) * h.x - wo.x,
                        2.f * dot(wo, h) * h.y - wo.y,
                        2.f * dot(wo, h) * h.z - wo.z);
        if (s.wi.z <= 0.f) {
            s.pdf = 0.f; s.f = Spectrum::zero(); s.is_specular = false;
            return s;
        }
        float VdotH = fabsf(dot(wo, h));
        float Fr = fresnel_schlick(VdotH, coat_F0);
        float ndf_c = ggx_D(h, coat_alpha);
        float geo_c = ggx_G(wo, s.wi, coat_alpha);
        float spec_pdf = ndf_c * fabsf(h.z) / (4.f * VdotH + EPSILON);
        float diff_pdf = cosine_hemisphere_pdf(s.wi.z);
        s.pdf = p_coat * spec_pdf + (1.f - p_coat) * diff_pdf;

        float denom = 4.f * fabsf(wo.z) * fabsf(s.wi.z) + EPSILON;
        float coat_spec = coat_weight * (ndf_c * geo_c * Fr) / denom;
        // Base = diffuse weighted by (1 - coat Fresnel)
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            s.f.value[i] = coat_spec + (1.f - coat_weight * Fr) * mat.Kd.value[i] * INV_PI;
        }
    } else {
        // Sample base (Lambert) 
        s.wi = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
        if (s.wi.z <= 0.f) {
            s.pdf = 0.f; s.f = Spectrum::zero(); s.is_specular = false;
            return s;
        }
        float diff_pdf = cosine_hemisphere_pdf(s.wi.z);
        // Compute coat specular pdf for this direction
        float3 h = normalize(wo + s.wi);
        float ndf_c = ggx_D(h, coat_alpha);
        float VdotH = fabsf(dot(wo, h));
        float spec_pdf = ndf_c * fabsf(h.z) / (4.f * VdotH + EPSILON);
        s.pdf = p_coat * spec_pdf + (1.f - p_coat) * diff_pdf;

        float Fr = fresnel_schlick(VdotH, coat_F0);
        float geo_c = ggx_G(wo, s.wi, coat_alpha);
        float denom = 4.f * fabsf(wo.z) * fabsf(s.wi.z) + EPSILON;
        float coat_spec = coat_weight * (ndf_c * geo_c * Fr) / denom;
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            s.f.value[i] = coat_spec + (1.f - coat_weight * Fr) * mat.Kd.value[i] * INV_PI;
        }
    }
    s.is_specular = false;
    return s;
}

// ── Fabric (diffuse + sheen lobe) ───────────────────────────────────
// Sheen: Fresnel-like grazing highlight for cloth.
// f_sheen = sheen_weight * (1 - cos_theta_h)^5 / (4 * pi)
// Optionally tinted toward material colour via sheen_tint.

inline HD BSDFSample fabric_sample(const Material& mat, float3 wo, PCGRng& rng) {
    BSDFSample s;

    // Always sample cosine hemisphere (both lobes are diffuse-like)
    s.wi = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
    if (s.wi.z <= 0.f) {
        s.pdf = 0.f; s.f = Spectrum::zero(); s.is_specular = false;
        return s;
    }
    s.pdf = cosine_hemisphere_pdf(s.wi.z);

    float3 h = normalize(wo + s.wi);
    float cos_theta_h = fabsf(dot(wo, h));
    float t = 1.f - cos_theta_h;
    float t5 = t * t * t * t * t;  // (1 - cos_h)^5

    float sheen_w = mat.pb_sheen;
    float tint    = mat.pb_sheen_tint;

    // Diffuse base + sheen
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        float base = mat.Kd.value[i] * INV_PI;
        // Sheen colour: blend between white and Kd-tinted
        float sheen_col = (1.f - tint) * 1.0f + tint * mat.Kd.value[i];
        float sheen = sheen_w * sheen_col * t5 * INV_PI;
        s.f.value[i] = base + sheen;
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
        case MaterialType::Translucent:
            return Spectrum::zero();   // delta distributions
        case MaterialType::Clearcoat: {
            // Diffuse portion attenuated by coat energy loss
            float coat_f0t = (mat.ior - 1.f) / (mat.ior + 1.f);
            float coat_F0  = coat_f0t * coat_f0t;
            float cos_o = fabsf(wo.z);
            float Fr = fresnel_schlick(cos_o, coat_F0);
            Spectrum f;
            for (int i = 0; i < NUM_LAMBDA; ++i)
                f.value[i] = (1.f - mat.pb_clearcoat * Fr) * mat.Kd.value[i] * INV_PI;
            return f;
        }
        case MaterialType::Fabric:
            return lambertian_f(mat.Kd);  // diffuse base only
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
            float alpha = bsdf_roughness_to_alpha(mat.roughness);
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
            Spectrum f;
            for (int i = 0; i < NUM_LAMBDA; ++i) {
                f.value[i] = mat.Ks.value[i] * (ndf * geo * Fr) / denom
                           + (1.f - Fr) * mat.Kd.value[i] * INV_PI;
            }
            return f;
        }

        case MaterialType::Mirror:
        case MaterialType::Glass:
        case MaterialType::Translucent:
            // Delta distributions: f is zero for non-delta directions
            return Spectrum::zero();

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
            Spectrum f;
            for (int i = 0; i < NUM_LAMBDA; ++i)
                f.value[i] = coat_spec + (1.f - mat.pb_clearcoat * Fr) * mat.Kd.value[i] * INV_PI;
            return f;
        }

        case MaterialType::Fabric: {
            float3 h = normalize(wo + wi);
            float cos_theta_h = fabsf(dot(wo, h));
            float t = 1.f - cos_theta_h;
            float t5 = t * t * t * t * t;
            Spectrum f;
            for (int i = 0; i < NUM_LAMBDA; ++i) {
                float sheen_col = (1.f - mat.pb_sheen_tint) + mat.pb_sheen_tint * mat.Kd.value[i];
                f.value[i] = mat.Kd.value[i] * INV_PI + mat.pb_sheen * sheen_col * t5 * INV_PI;
            }
            return f;
        }

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
            float alpha = bsdf_roughness_to_alpha(mat.roughness);
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
            float VdotH = fabsf(dot(wo, h));
            float spec_pdf = ndf * fabsf(h.z) / (4.f * VdotH + EPSILON);

            return p_spec * spec_pdf + (1.f - p_spec) * diff_pdf;
        }

        case MaterialType::Mirror:
        case MaterialType::Glass:
        case MaterialType::Translucent:
            return 0.f; // Delta distribution

        case MaterialType::Clearcoat: {
            float coat_alpha = bsdf_roughness_to_alpha(mat.pb_clearcoat_roughness);
            float coat_f0t = (mat.ior - 1.f) / (mat.ior + 1.f);
            float coat_F0  = coat_f0t * coat_f0t;
            float p_coat = fmaxf(fminf(mat.pb_clearcoat * coat_F0, 0.95f), 0.05f);
            float diff_pdf = cosine_hemisphere_pdf(wi.z);
            float3 h = normalize(wo + wi);
            float ndf_c = ggx_D(h, coat_alpha);
            float VdotH = fabsf(dot(wo, h));
            float spec_pdf = ndf_c * fabsf(h.z) / (4.f * VdotH + EPSILON);
            return p_coat * spec_pdf + (1.f - p_coat) * diff_pdf;
        }

        case MaterialType::Fabric:
            return cosine_hemisphere_pdf(wi.z);

        default:
            return 0.f;
    }
}

// ── Sample a direction from the BSDF ───────────────────────────────
// hero_bin: optional hero wavelength bin for glass dispersion.
//   hero_bin < 0 → D-line default  (camera path / non-dispersive)
//   hero_bin >= 0 → use that bin   (photon tracer with spectral hero)

inline HD BSDFSample sample(const Material& mat, float3 wo, PCGRng& rng,
                            int hero_bin = -1) {
    switch (mat.type) {
        case MaterialType::Lambertian:
        case MaterialType::Emissive:
            return lambertian_sample(mat.Kd, wo, rng);

        case MaterialType::Mirror:
            return mirror_sample(mat.Ks, wo);

        case MaterialType::Glass:
        case MaterialType::Translucent:
            return glass_sample(wo, mat, rng, hero_bin);

        case MaterialType::GlossyMetal:
            return glossy_sample(mat.Kd, mat.Ks, mat.roughness, wo, rng);

        case MaterialType::GlossyDielectric:
            return glossy_dielectric_sample(mat.Kd, mat.Ks, mat.roughness, mat.ior, wo, rng);

        case MaterialType::Clearcoat:
            return clearcoat_sample(mat, wo, rng);

        case MaterialType::Fabric:
            return fabric_sample(mat, wo, rng);

        default:
            return lambertian_sample(mat.Kd, wo, rng);
    }
}

} // namespace bsdf
