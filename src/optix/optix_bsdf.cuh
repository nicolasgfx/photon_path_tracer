#pragma once

// optix_bsdf.cuh – BSDF evaluate/sample/pdf for OptiX device code.
//
// Core math (GGX, Fresnel, VNDF, MIS weight) now lives in
// bsdf/bsdf_shared.h as inline HD functions shared with the CPU.

// == BSDF evaluate / pdf / sample (Lambertian + Cook-Torrance glossy) =

// Diffuse-only BSDF for photon density estimation (§6 standard practice).
// The full Cook-Torrance specular lobe produces unbounded variance
// in fixed-radius kernel estimators, creating coloured hotspots.
// Use the Lambertian component only for density estimation;
// NEE (direct lighting) still uses the full BSDF.
__forceinline__ __device__
Spectrum bsdf_evaluate_diffuse(uint32_t mat_id, float3 wo, float3 wi, float2 uv) {
    if (wi.z <= 0.f || wo.z <= 0.f) return Spectrum::zero();
    Spectrum Kd = dev_get_Kd(mat_id, uv);

    // Clearcoat: diffuse portion attenuated by coat energy loss
    if (dev_is_clearcoat(mat_id)) {
        float coat_w = dev_get_clearcoat_weight(mat_id);
        float ior = dev_get_ior(mat_id);
        float coat_f0t = (ior - 1.f) / (ior + 1.f);
        float coat_F0  = coat_f0t * coat_f0t;
        float cos_o = fabsf(wo.z);
        float Fr = fresnel_schlick(cos_o, coat_F0);
        Spectrum f;
        for (int i = 0; i < NUM_LAMBDA; ++i)
            f.value[i] = (1.f - coat_w * Fr) * Kd.value[i] * INV_PI;
        return f;
    }

    // Fabric: diffuse base only (no sheen for photon gather)
    return Kd * INV_PI;
}

__forceinline__ __device__
Spectrum bsdf_evaluate(uint32_t mat_id, float3 wo, float3 wi, float2 uv) {
    if (wi.z <= 0.f || wo.z <= 0.f) return Spectrum::zero();

    Spectrum Kd = dev_get_Kd(mat_id, uv);

    // ── Clearcoat: dielectric coat GGX + attenuated Lambert base ────
    if (dev_is_clearcoat(mat_id)) {
        float coat_w = dev_get_clearcoat_weight(mat_id);
        float coat_r = dev_get_clearcoat_roughness(mat_id);
        float coat_alpha = fmaxf(coat_r * coat_r, 0.001f);
        float ior = dev_get_ior(mat_id);
        float coat_f0t = (ior - 1.f) / (ior + 1.f);
        float coat_F0  = coat_f0t * coat_f0t;

        float3 h = normalize(wo + wi);
        float ndf_c = ggx_D(h, coat_alpha);
        float geo_c = ggx_G(wo, wi, coat_alpha);
        float VdotH = fabsf(dot(wo, h));
        float Fr = fresnel_schlick(VdotH, coat_F0);
        float denom = 4.f * fabsf(wo.z) * fabsf(wi.z) + EPSILON;
        float coat_spec = coat_w * (ndf_c * geo_c * Fr) / denom;
        Spectrum f;
        for (int i = 0; i < NUM_LAMBDA; ++i)
            f.value[i] = coat_spec + (1.f - coat_w * Fr) * Kd.value[i] * INV_PI;
        return f;
    }

    // ── Fabric: diffuse + sheen lobe ────────────────────────────────
    if (dev_is_fabric(mat_id)) {
        float sheen_w = dev_get_sheen(mat_id);
        float tint    = dev_get_sheen_tint(mat_id);
        float3 h = normalize(wo + wi);
        float cos_theta_h = fabsf(dot(wo, h));
        float t = 1.f - cos_theta_h;
        float t5 = t * t * t * t * t;
        Spectrum f;
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            float sheen_col = (1.f - tint) * 1.0f + tint * Kd.value[i];
            f.value[i] = Kd.value[i] * INV_PI + sheen_w * sheen_col * t5 * INV_PI;
        }
        return f;
    }

    if (!dev_is_glossy(mat_id) && !dev_is_dielectric_glossy(mat_id)) {
        // Pure Lambertian
        return Kd * INV_PI;
    }

    // Cook-Torrance glossy: specular lobe + diffuse
    Spectrum Ks = dev_get_Ks(mat_id);
    float roughness = dev_get_roughness(mat_id);
    float alpha = bsdf_roughness_to_alpha(roughness);

    float3 h = normalize(wo + wi);
    float ndf = ggx_D(h, alpha);
    float geo = ggx_G(wo, wi, alpha);
    float VdotH = fabsf(dot(wo, h));
    float denom = 4.f * fabsf(wo.z) * fabsf(wi.z) + EPSILON;

    Spectrum f;
    if (dev_is_dielectric_glossy(mat_id)) {
        // Dielectric Fresnel: F0 from IOR, Ks scales specular color
        float ior = dev_get_ior(mat_id);
        float F0 = ((ior - 1.f) / (ior + 1.f)) * ((ior - 1.f) / (ior + 1.f));
        float Fr = fresnel_schlick(VdotH, F0);
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            float spec = (ndf * geo * Fr * Ks.value[i]) / denom;
            float diff = (1.f - Fr) * Kd.value[i] * INV_PI;
            f.value[i] = spec + diff;
        }
    } else {
        // Metallic: Ks is the Fresnel F0 per wavelength
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            float Fr = fresnel_schlick(VdotH, Ks.value[i]);
            f.value[i] = (ndf * geo * Fr) / denom + Kd.value[i] * INV_PI;
        }
    }
    return f;
}

__forceinline__ __device__
float dev_bsdf_pdf(uint32_t mat_id, float3 wo, float3 wi) {
    if (wi.z <= 0.f || wo.z <= 0.f) return 0.f;

    float diff_pdf = fmaxf(0.f, wi.z) * INV_PI;

    // ── Clearcoat: mixed coat-GGX + cosine pdf ─────────────────────
    if (dev_is_clearcoat(mat_id)) {
        float coat_w = dev_get_clearcoat_weight(mat_id);
        float coat_r = dev_get_clearcoat_roughness(mat_id);
        float coat_alpha = fmaxf(coat_r * coat_r, 0.001f);
        float ior = dev_get_ior(mat_id);
        float coat_f0t = (ior - 1.f) / (ior + 1.f);
        float coat_F0  = coat_f0t * coat_f0t;
        float p_coat = fmaxf(fminf(coat_w * coat_F0, 0.95f), 0.05f);

        float3 h = normalize(wo + wi);
        float ndf_c = ggx_D(h, coat_alpha);
        float VdotH = fabsf(dot(wo, h));
        float spec_pdf = ndf_c * fabsf(h.z) / (4.f * VdotH + EPSILON);
        return p_coat * spec_pdf + (1.f - p_coat) * diff_pdf;
    }

    // ── Fabric: cosine hemisphere only ──────────────────────────────
    if (dev_is_fabric(mat_id)) {
        return diff_pdf;
    }

    if (!dev_is_glossy(mat_id) && !dev_is_dielectric_glossy(mat_id)) {
        return diff_pdf;
    }

    // Glossy: mixed PDF = p_spec * spec_pdf + (1-p_spec) * diff_pdf
    Spectrum Ks = dev_get_Ks(mat_id);
    Spectrum Kd = dev_get_Kd(mat_id, make_float2(0.f, 0.f));
    float roughness = dev_get_roughness(mat_id);
    float alpha = bsdf_roughness_to_alpha(roughness);

    float spec_weight, diff_weight;
    if (dev_is_dielectric_glossy(mat_id)) {
        float ior = dev_get_ior(mat_id);
        float F0 = ((ior - 1.f) / (ior + 1.f)) * ((ior - 1.f) / (ior + 1.f));
        spec_weight = fmaxf(Ks.max_component() * F0, 0.05f);
        diff_weight = Kd.max_component();
    } else {
        spec_weight = Ks.max_component();
        diff_weight = Kd.max_component();
    }
    // Must match the roughness boost in dev_bsdf_sample()
    float roughness_boost = 1.f / (1.f + 10.f * alpha);
    spec_weight = fmaxf(spec_weight, roughness_boost);

    float total = spec_weight + diff_weight;
    float p_spec = (total > 0.f) ? spec_weight / total : 0.5f;

    float3 h = normalize(wo + wi);
    float ndf = ggx_D(h, alpha);
    float VdotH = fabsf(dot(wo, h));
    float spec_pdf = ndf * fabsf(h.z) / (4.f * VdotH + EPSILON);

    return p_spec * spec_pdf + (1.f - p_spec) * diff_pdf;
}

// Overload for backward compatibility (Lambertian-only call sites)
__forceinline__ __device__
float dev_bsdf_pdf(float3 wi) {
    return fmaxf(0.f, wi.z) * INV_PI;
}

// Result struct for device BSDF sampling
struct DevBSDFSample {
    float3   wi;          // sampled direction (local frame)
    float    pdf;         // probability density
    Spectrum f;           // BSDF value f(wo, wi) per wavelength
    bool     is_specular; // true for delta distributions
};

// Sample BSDF: handles Lambertian, GlossyMetal (Cook-Torrance + diffuse)
// pixel_idx >= 0 enables the per-pixel Bresenham lobe balance accumulator
// (pass -1 to fall back to a random coin flip, e.g. for photon tracing).
__forceinline__ __device__
DevBSDFSample dev_bsdf_sample(uint32_t mat_id, float3 wo, float2 uv,
                              PCGRng& rng, int pixel_idx = -1) {
    DevBSDFSample s;
    s.is_specular = false;

    Spectrum Kd = dev_get_Kd(mat_id, uv);

    // ── Clearcoat: sample coat GGX or base Lambert ──────────────────
    if (dev_is_clearcoat(mat_id)) {
        float coat_w = dev_get_clearcoat_weight(mat_id);
        float coat_r = dev_get_clearcoat_roughness(mat_id);
        float coat_alpha = fmaxf(coat_r * coat_r, 0.001f);
        float ior = dev_get_ior(mat_id);
        float coat_f0t = (ior - 1.f) / (ior + 1.f);
        float coat_F0  = coat_f0t * coat_f0t;
        float p_coat = fmaxf(fminf(coat_w * coat_F0, 0.95f), 0.05f);

        if (rng.next_float() < p_coat) {
            // Sample coat GGX lobe
            float3 h = ggx_sample_halfvector(wo, coat_alpha,
                                                  rng.next_float(), rng.next_float());
            s.wi = make_f3(2.f * dot(wo, h) * h.x - wo.x,
                           2.f * dot(wo, h) * h.y - wo.y,
                           2.f * dot(wo, h) * h.z - wo.z);
            if (s.wi.z <= 0.f) { s.pdf = 0.f; s.f = Spectrum::zero(); return s; }

            float VdotH = fabsf(dot(wo, h));
            float Fr = fresnel_schlick(VdotH, coat_F0);
            float ndf_c = ggx_D(h, coat_alpha);
            float geo_c = ggx_G(wo, s.wi, coat_alpha);
            float spec_pdf = ndf_c * fabsf(h.z) / (4.f * VdotH + EPSILON);
            float diff_pdf = fmaxf(0.f, s.wi.z) * INV_PI;
            s.pdf = p_coat * spec_pdf + (1.f - p_coat) * diff_pdf;

            float denom = 4.f * fabsf(wo.z) * fabsf(s.wi.z) + EPSILON;
            float coat_spec = coat_w * (ndf_c * geo_c * Fr) / denom;
            for (int i = 0; i < NUM_LAMBDA; ++i)
                s.f.value[i] = coat_spec + (1.f - coat_w * Fr) * Kd.value[i] * INV_PI;
        } else {
            // Sample base Lambert
            s.wi = sample_cosine_hemisphere_dev(rng);
            if (s.wi.z <= 0.f) { s.pdf = 0.f; s.f = Spectrum::zero(); return s; }

            float diff_pdf = fmaxf(0.f, s.wi.z) * INV_PI;
            float3 h = normalize(wo + s.wi);
            float ndf_c = ggx_D(h, coat_alpha);
            float VdotH = fabsf(dot(wo, h));
            float spec_pdf = ndf_c * fabsf(h.z) / (4.f * VdotH + EPSILON);
            s.pdf = p_coat * spec_pdf + (1.f - p_coat) * diff_pdf;

            float Fr = fresnel_schlick(VdotH, coat_F0);
            float geo_c = ggx_G(wo, s.wi, coat_alpha);
            float denom = 4.f * fabsf(wo.z) * fabsf(s.wi.z) + EPSILON;
            float coat_spec = coat_w * (ndf_c * geo_c * Fr) / denom;
            for (int i = 0; i < NUM_LAMBDA; ++i)
                s.f.value[i] = coat_spec + (1.f - coat_w * Fr) * Kd.value[i] * INV_PI;
        }
        return s;
    }

    // ── Fabric: cosine hemisphere + sheen ───────────────────────────
    if (dev_is_fabric(mat_id)) {
        s.wi = sample_cosine_hemisphere_dev(rng);
        if (s.wi.z <= 0.f) { s.pdf = 0.f; s.f = Spectrum::zero(); return s; }
        s.pdf = fmaxf(0.f, s.wi.z) * INV_PI;

        float sheen_w = dev_get_sheen(mat_id);
        float tint    = dev_get_sheen_tint(mat_id);
        float3 h = normalize(wo + s.wi);
        float cos_theta_h = fabsf(dot(wo, h));
        float t = 1.f - cos_theta_h;
        float t5 = t * t * t * t * t;
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            float sheen_col = (1.f - tint) * 1.0f + tint * Kd.value[i];
            s.f.value[i] = Kd.value[i] * INV_PI + sheen_w * sheen_col * t5 * INV_PI;
        }
        return s;
    }

    if (!dev_is_glossy(mat_id) && !dev_is_dielectric_glossy(mat_id)) {
        // Pure Lambertian: cosine hemisphere
        s.wi = sample_cosine_hemisphere_dev(rng);
        s.pdf = fmaxf(0.f, s.wi.z) * INV_PI;
        s.f = Kd * INV_PI;
        return s;
    }

    // Cook-Torrance glossy with diffuse+specular lobe selection
    Spectrum Ks = dev_get_Ks(mat_id);
    float roughness = dev_get_roughness(mat_id);
    float alpha = bsdf_roughness_to_alpha(roughness);
    bool is_diel = dev_is_dielectric_glossy(mat_id);
    float ior_val = is_diel ? dev_get_ior(mat_id) : 1.5f;
    float F0_diel = ((ior_val - 1.f) / (ior_val + 1.f)) * ((ior_val - 1.f) / (ior_val + 1.f));

    float spec_weight, diff_weight;
    if (is_diel) {
        spec_weight = fmaxf(Ks.max_component() * F0_diel, 0.05f);
        diff_weight = Kd.max_component();
    } else {
        spec_weight = Ks.max_component();
        diff_weight = Kd.max_component();
    }

    // Boost specular sampling probability for low-roughness (near-mirror)
    // surfaces.  Without this, a shiny dielectric with large Kd and small
    // Ks*F0 sends ~94 % of samples to the diffuse cosine lobe, wasting
    // almost all of them — they evaluate to near-zero specular BSDF at
    // random directions far from the narrow GGX peak, adding pure noise.
    // The boost smoothly fades to zero for rough surfaces (alpha → 1).
    float roughness_boost = 1.f / (1.f + 10.f * alpha);
    spec_weight = fmaxf(spec_weight, roughness_boost);

    float total_w = spec_weight + diff_weight;
    float p_spec = (total_w > 0.f) ? spec_weight / total_w : 0.5f;

    // ── Lobe selection via Bresenham accumulator ─────────────────────
    // v2.2: Gated behind DEFAULT_ENABLE_BRESENHAM_BSDF (off by default
    // to ensure CPU↔GPU consistency).
    // When pixel_idx >= 0 and the lobe_balance buffer is available we
    // use a Bresenham error accumulator instead of a random coin flip.
    // Positive balance = specular deficit → choose specular this sample.
    // This guarantees that over K samples the specular count is within
    // 1 of K * p_spec, eliminating binomial variance in lobe counts.
    bool choose_specular;
    if constexpr (DEFAULT_ENABLE_BRESENHAM_BSDF) {
        if (pixel_idx >= 0 && params.lobe_balance) {
            float balance = params.lobe_balance[pixel_idx];
            choose_specular = (balance >= 0.f);
            if (choose_specular)
                balance -= (1.f - p_spec);
            else
                balance += p_spec;
            params.lobe_balance[pixel_idx] = balance;
        } else {
            choose_specular = (rng.next_float() < p_spec);
        }
    } else {
        choose_specular = (rng.next_float() < p_spec);
    }

    if (choose_specular) {
        // Sample GGX specular lobe
        float u1 = rng.next_float();
        float u2 = rng.next_float();
        float3 h = ggx_sample_halfvector(wo, alpha, u1, u2);

        s.wi = make_f3(2.f * dot(wo, h) * h.x - wo.x,
                       2.f * dot(wo, h) * h.y - wo.y,
                       2.f * dot(wo, h) * h.z - wo.z);

        if (s.wi.z <= 0.f) {
            s.pdf = 0.f;
            s.f = Spectrum::zero();
            return s;
        }

        float ndf = ggx_D(h, alpha);
        float geo = ggx_G(wo, s.wi, alpha);
        float VdotH = fabsf(dot(wo, h));

        float spec_pdf = ndf * fabsf(h.z) / (4.f * VdotH + EPSILON);
        float diff_pdf = fmaxf(0.f, s.wi.z) * INV_PI;
        s.pdf = p_spec * spec_pdf + (1.f - p_spec) * diff_pdf;

        float denom = 4.f * fabsf(wo.z) * fabsf(s.wi.z) + EPSILON;
        if (is_diel) {
            float Fr = fresnel_schlick(VdotH, F0_diel);
            for (int i = 0; i < NUM_LAMBDA; ++i) {
                float spec = (ndf * geo * Fr * Ks.value[i]) / denom;
                float diff = (1.f - Fr) * Kd.value[i] * INV_PI;
                s.f.value[i] = spec + diff;
            }
        } else {
            for (int i = 0; i < NUM_LAMBDA; ++i) {
                float Fr = fresnel_schlick(VdotH, Ks.value[i]);
                s.f.value[i] = (ndf * geo * Fr) / denom + Kd.value[i] * INV_PI;
            }
        }
    } else {
        // Sample diffuse lobe
        s.wi = sample_cosine_hemisphere_dev(rng);

        if (s.wi.z <= 0.f) {
            s.pdf = 0.f;
            s.f = Spectrum::zero();
            return s;
        }

        float diff_pdf = fmaxf(0.f, s.wi.z) * INV_PI;

        float3 h = normalize(wo + s.wi);
        float ndf = ggx_D(h, alpha);
        float VdotH = fabsf(dot(wo, h));
        float spec_pdf = ndf * fabsf(h.z) / (4.f * VdotH + EPSILON);

        s.pdf = p_spec * spec_pdf + (1.f - p_spec) * diff_pdf;

        float geo = ggx_G(wo, s.wi, alpha);
        float denom = 4.f * fabsf(wo.z) * fabsf(s.wi.z) + EPSILON;
        if (is_diel) {
            float Fr = fresnel_schlick(VdotH, F0_diel);
            for (int i = 0; i < NUM_LAMBDA; ++i) {
                float spec = (ndf * geo * Fr * Ks.value[i]) / denom;
                float diff = (1.f - Fr) * Kd.value[i] * INV_PI;
                s.f.value[i] = spec + diff;
            }
        } else {
            for (int i = 0; i < NUM_LAMBDA; ++i) {
                float Fr = fresnel_schlick(VdotH, Ks.value[i]);
                s.f.value[i] = (ndf * geo * Fr) / denom + Kd.value[i] * INV_PI;
            }
        }
    }

    return s;
}
