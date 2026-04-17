#ifndef PPT_INTEGRATOR_PATH_TRACER_CUH
#define PPT_INTEGRATOR_PATH_TRACER_CUH
// ─────────────────────────────────────────────────────────────────────
// path_tracer.cuh – GPU path integrator (v5 RGB)
//
// Core render loop with bouncing, NEE, MIS, Russian roulette.
//
// Provides:
//   dev_bsdf_sample()    — device-side BSDF importance sampling
//   trace_path()         — iterative bounce loop
//   __raygen__render     — main rendering raygen program
// ─────────────────────────────────────────────────────────────────────
#include "integrator/path_state.h"
#include "integrator/russian_roulette.h"
#include "integrator/sample_clamping.h"
#include "material/bsdf_shared.h"

// Forward: LaunchParams params, trace_radiance, trace_shadow,
//          dev_bsdf_evaluate, dev_bsdf_pdf, dev_nee_dispatch,
//          dev_get_* material accessors
//          — all from optix_nee.cuh (already included in optix_programs.cu)

constexpr float PATH_EPSILON = 1e-4f;

// =====================================================================
// Normal map perturbation — applies tangent-space normal from texture
// =====================================================================

__forceinline__ __device__
float3 dev_apply_normal_map(uint32_t mat_id, float3 shading_normal,
                            float3 geo_normal, float3 tangent, float2 uv) {
    if (!params.normal_tex || params.normal_tex[mat_id] < 0)
        return shading_normal;

    float3 tex_val = dev_sample_texture_rgb(params.normal_tex[mat_id], uv);

    // Decode from [0,1] to [-1,1]
    float3 n_tan = make_f3(
        tex_val.x * 2.f - 1.f,
        tex_val.y * 2.f - 1.f,
        tex_val.z * 2.f - 1.f);

    // Build TBN frame from shading normal and tangent
    ONB tbn = onb_from_normal_and_tangent(shading_normal, tangent);
    float3 perturbed = normalize(tbn.local_to_world(n_tan));

    // Guard: perturbed normal must stay on same side as geometric normal
    if (dot(perturbed, geo_normal) <= 0.f)
        return shading_normal;

    return perturbed;
}

__forceinline__ __device__
float3 dev_apply_bump_map(uint32_t mat_id, float3 shading_normal,
                          float3 geo_normal, float3 tangent, float2 uv) {
    if (!params.bump_tex || params.bump_tex[mat_id] < 0)
        return shading_normal;
    // Also skip if we already have a normal map for this material
    if (params.normal_tex && params.normal_tex[mat_id] >= 0)
        return shading_normal;

    int tex_id = params.bump_tex[mat_id];
    constexpr float BUMP_SCALE = 1.f;
    constexpr float INV_DELTA = 512.f;  // 1/delta for finite differences

    // Central height
    float h0 = dev_sample_texture_rgb(tex_id, uv).x;

    // Finite differences in UV space
    float du = 1.f / INV_DELTA;
    float hu = dev_sample_texture_rgb(tex_id, make_f2(uv.x + du, uv.y)).x;
    float hv = dev_sample_texture_rgb(tex_id, make_f2(uv.x, uv.y + du)).x;

    float dh_du = (hu - h0) * INV_DELTA * BUMP_SCALE;
    float dh_dv = (hv - h0) * INV_DELTA * BUMP_SCALE;

    // Build TBN and perturb
    ONB tbn = onb_from_normal_and_tangent(shading_normal, tangent);
    float3 perturbed = normalize(
        shading_normal - tbn.u * dh_du - tbn.v * dh_dv);

    if (dot(perturbed, geo_normal) <= 0.f)
        return shading_normal;

    return perturbed;
}

// =====================================================================
// Device-side BSDF sampling (all 9 material types)
// =====================================================================

__forceinline__ __device__
DevBSDFSample dev_bsdf_sample(uint32_t mat_id, float3 wo, bool entering,
                              float2 uv, PCGRng& rng) {
    DevBSDFSample s;
    s.wi = make_f3(0, 0, 1);
    s.pdf = 0.f;
    s.f = make_f3(0, 0, 0);
    s.is_specular = false;

    uint8_t mt = dev_get_mat_type(mat_id);
    float3 Kd = dev_get_Kd(mat_id, uv);

    switch (mt) {
        case Lambertian:
        case Emissive: {
            float u1 = rng.next_float(), u2 = rng.next_float();
            s.wi = sample_cosine_hemisphere(u1, u2);
            s.pdf = cosine_hemisphere_pdf(s.wi.z);
            s.f = Kd * INV_PI;
            return s;
        }

        case Mirror: {
            float3 Ks = dev_get_Ks(mat_id, uv);
            s.wi = reflect_local(wo);
            s.pdf = 1.f;
            float factor = 1.f / (fabsf(s.wi.z) + EPSILON);
            s.f = Ks * factor;
            s.is_specular = true;
            return s;
        }

        case Glass:
        case Translucent: {
            float ior = dev_get_ior(mat_id);
            float3 Tf = dev_get_Tf(mat_id);
            float cos_i = fabsf(wo.z);

            // Thin dielectric: no refraction bend, straight-through
            // transmission.  Always treat as "entering" for Fresnel.
            if (dev_is_thin(mat_id)) {
                float eta = 1.f / ior;
                float F = fresnel_dielectric(cos_i, eta);

                if (rng.next_float() < F) {
                    s.wi = reflect_local(wo);
                    s.pdf = F;
                    float factor = F / (fabsf(s.wi.z) + EPSILON);
                    s.f = make_f3(factor, factor, factor);
                } else {
                    s.wi = transmit_thin_local(wo);
                    s.pdf = 1.f - F;
                    float factor = (1.f - F) / (fabsf(s.wi.z) + EPSILON);
                    s.f = Tf * factor;
                }
                s.is_specular = true;
                return s;
            }

            float eta = entering ? (1.f / ior) : ior;
            float F = fresnel_dielectric(cos_i, eta);

            if (rng.next_float() < F) {
                s.wi = reflect_local(wo);
                s.pdf = F;
                float factor = F / (fabsf(s.wi.z) + EPSILON);
                s.f = make_f3(factor, factor, factor);
            } else {
                float3 wt;
                if (refract_local(wo, eta, wt)) {
                    s.wi = wt;
                    s.pdf = 1.f - F;
                    float factor = (1.f - F) / (fabsf(s.wi.z) + EPSILON);
                    s.f = Tf * factor;
                } else {
                    // TIR fallback
                    s.wi = reflect_local(wo);
                    s.pdf = 1.f;
                    float factor = 1.f / (fabsf(s.wi.z) + EPSILON);
                    s.f = make_f3(factor, factor, factor);
                }
            }
            s.is_specular = true;
            return s;
        }

        case GlossyMetal: {
            float3 Ks = dev_get_Ks(mat_id, uv);
            float alpha = bsdf_roughness_to_alpha(dev_get_roughness(mat_id));

            LobeProbabilities lp = bsdf_lobe_probabilities(
                max_component(Ks), max_component(Kd));
            float p_spec = lp.p_spec;

            if (rng.next_float() < p_spec) {
                // GGX specular
                float u1 = rng.next_float(), u2 = rng.next_float();
                float3 h = ggx_sample_halfvector(wo, alpha, u1, u2);
                float wdh = dot(wo, h);
                s.wi = make_f3(2.f*wdh*h.x - wo.x, 2.f*wdh*h.y - wo.y, 2.f*wdh*h.z - wo.z);
                if (s.wi.z <= 0.f) { s.pdf = 0.f; return s; }

                float ndf = ggx_D(h, alpha);
                float geo = ggx_G(wo, s.wi, alpha);
                float VdotH = fabsf(wdh);
                float spec_pdf = ndf * ggx_G1(wo, alpha) / (4.f * fabsf(wo.z) + EPSILON);
                float diff_pdf = cosine_hemisphere_pdf(s.wi.z);
                s.pdf = p_spec * spec_pdf + (1.f - p_spec) * diff_pdf;

                float denom = ggx_denom(wo, s.wi);
                float3 F_c = fresnel_schlick3(VdotH, Ks);
                float spec_term = ndf * geo / denom;
                s.f = F_c * spec_term + Kd * INV_PI;
            } else {
                // Cosine-weighted diffuse
                float u1 = rng.next_float(), u2 = rng.next_float();
                s.wi = sample_cosine_hemisphere(u1, u2);
                if (s.wi.z <= 0.f) { s.pdf = 0.f; return s; }

                float diff_pdf = cosine_hemisphere_pdf(s.wi.z);
                float3 h = normalize(wo + s.wi);
                float ndf = ggx_D(h, alpha);
                float VdotH = fabsf(dot(wo, h));
                float spec_pdf = ndf * ggx_G1(wo, alpha) / (4.f * fabsf(wo.z) + EPSILON);
                s.pdf = p_spec * spec_pdf + (1.f - p_spec) * diff_pdf;

                float geo = ggx_G(wo, s.wi, alpha);
                float denom = ggx_denom(wo, s.wi);
                float3 F_c = fresnel_schlick3(VdotH, Ks);
                float spec_term = ndf * geo / denom;
                s.f = F_c * spec_term + Kd * INV_PI;
            }
            return s;
        }

        case GlossyDielectric: {
            float3 Ks = dev_get_Ks(mat_id, uv);
            float alpha = bsdf_roughness_to_alpha(dev_get_roughness(mat_id));
            float ior = dev_get_ior(mat_id);
            float F0 = bsdf_f0_from_ior(ior);

            LobeProbabilities lp = bsdf_lobe_probabilities(
                max_component(Ks) * F0, max_component(Kd));
            float p_spec = lp.p_spec;

            if (rng.next_float() < p_spec) {
                float u1 = rng.next_float(), u2 = rng.next_float();
                float3 h = ggx_sample_halfvector(wo, alpha, u1, u2);
                float wdh = dot(wo, h);
                s.wi = make_f3(2.f*wdh*h.x - wo.x, 2.f*wdh*h.y - wo.y, 2.f*wdh*h.z - wo.z);
                if (s.wi.z <= 0.f) { s.pdf = 0.f; return s; }

                float ndf = ggx_D(h, alpha);
                float geo = ggx_G(wo, s.wi, alpha);
                float VdotH = fabsf(wdh);
                float spec_pdf = ndf * ggx_G1(wo, alpha) / (4.f * fabsf(wo.z) + EPSILON);
                float diff_pdf = cosine_hemisphere_pdf(s.wi.z);
                s.pdf = p_spec * spec_pdf + (1.f - p_spec) * diff_pdf;

                float denom = ggx_denom(wo, s.wi);
                float Fr = fresnel_schlick(VdotH, F0);
                float spec_term = ndf * geo * Fr / denom;
                s.f = Ks * spec_term + Kd * ((1.f - Fr) * INV_PI);
            } else {
                float u1 = rng.next_float(), u2 = rng.next_float();
                s.wi = sample_cosine_hemisphere(u1, u2);
                if (s.wi.z <= 0.f) { s.pdf = 0.f; return s; }

                float diff_pdf = cosine_hemisphere_pdf(s.wi.z);
                float3 h = normalize(wo + s.wi);
                float ndf = ggx_D(h, alpha);
                float VdotH = fabsf(dot(wo, h));
                float spec_pdf = ndf * ggx_G1(wo, alpha) / (4.f * fabsf(wo.z) + EPSILON);
                s.pdf = p_spec * spec_pdf + (1.f - p_spec) * diff_pdf;

                float geo = ggx_G(wo, s.wi, alpha);
                float denom = ggx_denom(wo, s.wi);
                float Fr = fresnel_schlick(VdotH, F0);
                float spec_term = ndf * geo * Fr / denom;
                s.f = Ks * spec_term + Kd * ((1.f - Fr) * INV_PI);
            }
            return s;
        }

        case Clearcoat: {
            float coat_w = params.clearcoat_weight ? params.clearcoat_weight[mat_id] : 1.0f;
            float coat_r = params.clearcoat_roughness ? params.clearcoat_roughness[mat_id] : 0.1f;
            float coat_alpha = fmaxf(coat_r * coat_r, 0.001f);
            float ior = dev_get_ior(mat_id);
            float coat_F0 = bsdf_f0_from_ior(ior);

            // MIS weight: coat specular vs base diffuse (must match dev_bsdf_pdf)
            float p_coat = fmaxf(0.05f, fminf(0.95f, coat_w * coat_F0));
            float p_base = 1.f - p_coat;

            if (rng.next_float() < p_coat) {
                // Sample clearcoat GGX
                float u1 = rng.next_float(), u2 = rng.next_float();
                float3 h = ggx_sample_halfvector(wo, coat_alpha, u1, u2);
                float wdh = dot(wo, h);
                s.wi = make_f3(2.f*wdh*h.x - wo.x, 2.f*wdh*h.y - wo.y, 2.f*wdh*h.z - wo.z);
                if (s.wi.z <= 0.f) { s.pdf = 0.f; return s; }
            } else {
                // Sample diffuse base
                float u1 = rng.next_float(), u2 = rng.next_float();
                s.wi = sample_cosine_hemisphere(u1, u2);
                if (s.wi.z <= 0.f) { s.pdf = 0.f; return s; }
            }

            // Evaluate combined PDF and BSDF
            float3 h = normalize(wo + s.wi);
            float ndf_c = ggx_D(h, coat_alpha);
            float geo_c = ggx_G(wo, s.wi, coat_alpha);
            float VdotH = fabsf(dot(wo, h));
            float Fr = fresnel_schlick(VdotH, coat_F0);
            float denom = ggx_denom(wo, s.wi);
            float coat_spec = coat_w * (ndf_c * geo_c * Fr) / denom;
            float base_diff = (1.f - coat_w * Fr) * INV_PI;

            float coat_pdf = ndf_c * ggx_G1(wo, coat_alpha) / (4.f * fabsf(wo.z) + EPSILON);
            float diff_pdf = cosine_hemisphere_pdf(s.wi.z);
            s.pdf = p_coat * coat_pdf + p_base * diff_pdf;

            s.f = make_f3(
                coat_spec + Kd.x * base_diff,
                coat_spec + Kd.y * base_diff,
                coat_spec + Kd.z * base_diff);
            return s;
        }

        case Fabric: {
            // Sample cosine-weighted (sheen is view-dependent, not easily importance-sampled)
            float u1 = rng.next_float(), u2 = rng.next_float();
            s.wi = sample_cosine_hemisphere(u1, u2);
            s.pdf = cosine_hemisphere_pdf(s.wi.z);
            if (s.wi.z <= 0.f) { s.pdf = 0.f; return s; }

            float sheen_w = params.sheen ? params.sheen[mat_id] : 0.f;
            float tint = params.sheen_tint ? params.sheen_tint[mat_id] : 0.f;
            float3 h = normalize(wo + s.wi);
            float ct = fabsf(dot(wo, h));
            float t = 1.f - ct;
            float t5 = t * t * t * t * t;
            s.f = make_f3(
                Kd.x * INV_PI + sheen_w * ((1.f-tint) + tint*Kd.x) * t5 * INV_PI,
                Kd.y * INV_PI + sheen_w * ((1.f-tint) + tint*Kd.y) * t5 * INV_PI,
                Kd.z * INV_PI + sheen_w * ((1.f-tint) + tint*Kd.z) * t5 * INV_PI);
            return s;
        }

        case DiffuseTransmission: {
            float3 Tf = dev_get_Tf(mat_id);
            float w_r = max_component(Kd);
            float w_t = max_component(Tf);
            float total_w = w_r + w_t;
            float p_r = (total_w > 0.f) ? w_r / total_w : 0.5f;
            p_r = fmaxf(0.1f, fminf(0.9f, p_r));

            float u1 = rng.next_float(), u2 = rng.next_float();
            if (rng.next_float() < p_r) {
                s.wi = sample_cosine_hemisphere(u1, u2);
                s.pdf = p_r * cosine_hemisphere_pdf(s.wi.z);
                s.f = Kd * INV_PI;
            } else {
                s.wi = sample_cosine_hemisphere(u1, u2);
                s.wi.z = -s.wi.z;
                s.pdf = (1.f - p_r) * cosine_hemisphere_pdf(fabsf(s.wi.z));
                s.f = Tf * INV_PI;
            }
            return s;
        }

        default: {
            float u1 = rng.next_float(), u2 = rng.next_float();
            s.wi = sample_cosine_hemisphere(u1, u2);
            s.pdf = cosine_hemisphere_pdf(s.wi.z);
            s.f = Kd * INV_PI;
            return s;
        }
    }
}

__forceinline__ __device__
DevBSDFSample dev_bsdf_sample(uint32_t mat_id, float3 wo, float2 uv, PCGRng& rng) {
    return dev_bsdf_sample(mat_id, wo, wo.z > 0.f, uv, rng);
}

// =====================================================================
// Device helper: is this material a delta (mirror/glass/translucent)?
// =====================================================================
__forceinline__ __device__
bool dev_is_delta(uint32_t mat_id) {
    uint8_t mt = dev_get_mat_type(mat_id);
    return mt == Mirror || mt == Glass || mt == Translucent;
}

// =====================================================================
// trace_path() — iterative bounce loop (v5 RGB, BSDF + NEE)
// =====================================================================

__forceinline__ __device__
PathResult trace_path(float3 origin, float3 direction, PCGRng& rng) {
    PathResult result;
    result.radiance    = make_f3(0, 0, 0);
    result.albedo      = make_f3(0, 0, 0);
    result.normal      = make_f3(0, 0, 1);
    result.num_bounces = 0;

    float3 throughput = make_f3(1, 1, 1);
    bool aov_written = false;

    // pdf_prev: combined PDF of previous bounce direction.
    // Used for emission MIS (BSDF hit light → weight vs NEE).
    // 0 means previous was delta → full weight to BSDF.
    float pdf_prev = 0.f;

    int max_bounces = params.max_bounces;

    for (int bounce = 0; bounce < max_bounces; ++bounce) {
        result.num_bounces = bounce + 1;
        TraceResult hit = trace_radiance(origin, direction);

        if (!hit.hit) {
            break;
        }

        uint32_t mat_id = hit.material_id;
        uint8_t mt = params.mat_type ? params.mat_type[mat_id] : 0;

        // ── Normal/bump map perturbation ────────────────────────────
        hit.shading_normal = dev_apply_normal_map(
            mat_id, hit.shading_normal, hit.geo_normal, hit.tangent, hit.uv);
        hit.shading_normal = dev_apply_bump_map(
            mat_id, hit.shading_normal, hit.geo_normal, hit.tangent, hit.uv);

        // ── Emission MIS ────────────────────────────────────────────
        if (mt == Emissive) {
            // Orient emitter geo_normal to agree with its shading
            // normal — PLY winding order can disagree.
            ShadingFrame esf = build_shading_frame(
                hit.shading_normal, hit.geo_normal, direction * (-1.f));

            float cos_light = dot(esf.geo_n, direction * (-1.f));
            if (cos_light <= 0.f) break;

            float3 Le = dev_get_Le(mat_id, hit.uv);
            float3 emitted = Le;
            if (bounce == 0) {
                result.radiance = result.radiance + throughput * emitted;
            } else {
                float w_bsdf = 1.f;
                if (pdf_prev > 0.f) {
                    float p_nee = dev_light_pdf(
                        hit.triangle_id, esf.geo_n, direction, hit.t,
                        origin);
                    w_bsdf = mis_weight_2(pdf_prev, p_nee);
                }
                emitted = emitted * w_bsdf;
                result.radiance = result.radiance + throughput * emitted;
            }
            break;
        }

        // ── Delta surfaces (mirror/glass/translucent) ───────────────
        if (dev_is_delta(mat_id)) {
            float3 wo_world = direction * (-1.f);
            ShadingFrame sf = build_shading_frame(
                hit.shading_normal, hit.geo_normal, wo_world);
            if (!sf.valid) break;
            float3 wo_local = sf.wo_local(wo_world);

            DevBSDFSample bs = dev_bsdf_sample(
                mat_id, wo_local, sf.entering, hit.uv, rng);

            if (bs.pdf <= 0.f) break;

            // Delta: throughput *= f * |cos| / pdf.
            // For mirror pdf=1 so this gives Ks.  For glass the
            // stochastic Fresnel pdf (F or 1-F) cancels the Fresnel
            // factor baked into f, yielding energy-conserving transport.
            float cos_theta = fabsf(bs.wi.z);
            float inv_pdf = 1.f / bs.pdf;
            float3 delta_transfer = bs.f * (cos_theta * inv_pdf);
            float3 wi_world = sf.frame.local_to_world(bs.wi);

            throughput = throughput * delta_transfer;
            if (params.clamping_enabled)
                throughput = clamp_path_throughput(throughput, params.max_path_throughput);

            // Offset along geometric normal; for refraction flip to
            // travel direction so the ray starts on the correct side.
            float3 offset_n = sf.geo_n;
            if (dot(wi_world, sf.geo_n) < 0.f)
                offset_n = -offset_n;
            origin    = hit.position + offset_n * PATH_EPSILON;
            direction = wi_world;
            pdf_prev  = 0.f;  // delta → no emission MIS weighting
            continue;
        }

        // ── Non-delta surface shading ───────────────────────────────

        float3 wo_world = direction * (-1.f);
        ShadingFrame sf = build_shading_frame(
            hit.shading_normal, hit.geo_normal, wo_world);
        if (!sf.valid) break;
        float3 wo_local = sf.wo_local(wo_world);

        // AOV: first non-specular hit
        if (!aov_written) {
            result.albedo = dev_get_Kd(mat_id, hit.uv);
            result.normal = hit.shading_normal;
            aov_written = true;
        }

        // ── NEE: 1 shadow ray ─────────────────────────────────────────────
        float3 nee_radiance = make_f3(0, 0, 0);
        if (params.render_mode != RenderMode::IndirectOnly) {
            NeeResult nee = dev_nee_dispatch(
                hit.position, sf.frame.w, sf.geo_n,
                sf.frame, wo_local, mat_id, hit.uv, rng);

            nee_radiance = nee.L;
            float3 nee_contrib = throughput * nee_radiance;
            if (params.clamping_enabled)
                nee_contrib = clamp_f3(nee_contrib, params.max_nee_contribution);
            result.radiance = result.radiance + nee_contrib;
        }

        // ── BSDF sample (next direction) ─────────────────────────────

        float3 wi_world = make_f3(0, 0, 0);
        float3 bsdf_val = make_f3(0, 0, 0);
        float  cos_theta_i = 0.f;
        float  bsdf_pdf = 0.f;
        bool   sample_valid = false;
        bool   transmitted  = false;  // DiffuseTransmission: wi crosses surface

        {
            DevBSDFSample bs = dev_bsdf_sample(mat_id, wo_local, hit.uv, rng);
            if (bs.pdf >= 1e-8f && bs.wi.z > 0.f) {
                wi_world    = sf.frame.local_to_world(bs.wi);
                bsdf_val    = bs.f;
                cos_theta_i = bs.wi.z;
                bsdf_pdf    = bs.pdf;
                sample_valid = true;
            }
            // DiffuseTransmission: accept wi.z < 0 (transmission lobe)
            else if (bs.pdf >= 1e-8f && bs.wi.z < 0.f
                     && dev_get_mat_type(mat_id) == DiffuseTransmission) {
                wi_world    = sf.frame.local_to_world(bs.wi);
                bsdf_val    = bs.f;
                cos_theta_i = fabsf(bs.wi.z);
                bsdf_pdf    = bs.pdf;
                sample_valid = true;
                transmitted  = true;
            }
        }

        if (sample_valid && !transmitted && dot(wi_world, sf.geo_n) <= 0.f)
            sample_valid = false;

        if (!sample_valid) break;

        // f_over_pdf per channel, clamped per-bounce
        float3 f_over_pdf = make_f3(
            bsdf_val.x * cos_theta_i / bsdf_pdf,
            bsdf_val.y * cos_theta_i / bsdf_pdf,
            bsdf_val.z * cos_theta_i / bsdf_pdf);
        if (params.clamping_enabled)
            f_over_pdf = clamp_f3(f_over_pdf, params.max_bounce_contribution);

        throughput = throughput * f_over_pdf;
        if (params.clamping_enabled)
            throughput = clamp_path_throughput(throughput, params.max_path_throughput);

        // ── Russian roulette ────────────────────────────────────────
        if (bounce >= params.min_bounces_rr) {
            float max_tp = max_component(throughput);
            RRResult rr = russian_roulette(max_tp, params.rr_threshold,
                                           rng.next_float());
            if (rr.terminate) break;
            throughput = throughput * rr.inv_survival;
            if (params.clamping_enabled)
                throughput = clamp_path_throughput(throughput, params.max_path_throughput);
        }

        // ── Prepare next ray ────────────────────────────────────────
        // DiffuseTransmission: transmitted ray starts on opposite side
        float3 offset_n = sf.geo_n;
        if (transmitted) offset_n = offset_n * (-1.f);
        origin    = hit.position + offset_n * PATH_EPSILON;
        direction = wi_world;
        pdf_prev  = bsdf_pdf;
    }

    // Per-sample luminance clamp
    if (params.clamping_enabled)
        result.radiance = clamp_sample_luminance(result.radiance, params.max_sample_luminance);

    return result;
}

// =====================================================================
// __raygen__render — main rendering program
//
// Runs samples_per_pixel paths per pixel and accumulates. Supports
// progressive rendering (frame_number > 0 accumulates into buffer).
// =====================================================================

extern "C" __global__ void __raygen__render() {
    const uint3 idx = optixGetLaunchIndex();
    int px = idx.x;
    int py = idx.y;
    int pixel_idx = py * params.width + px;

    int spp = params.samples_per_pixel;
    if (spp <= 0) spp = 1;

    float3 accum = make_f3(0, 0, 0);
    float3 albedo_accum = make_f3(0, 0, 0);
    float3 normal_accum = make_f3(0, 0, 0);

    for (int s = 0; s < spp; ++s) {
        PCGRng rng = PCGRng::seed(
            (uint64_t)params.frame_number * 65537ull +
            (uint64_t)s * 104729ull + 42ull,
            (uint64_t)pixel_idx + 1ull);

        // Sub-pixel jitter
        float jx = rng.next_float();
        float jy = rng.next_float();
        float u_ndc = ((float)px + jx) / (float)params.width;
        float v_ndc = ((float)py + jy) / (float)params.height;

        float3 direction = normalize(
            params.cam_w +
            params.cam_u * (2.f * u_ndc - 1.f) +
            params.cam_v * (2.f * v_ndc - 1.f));

        PathResult pr = trace_path(params.cam_pos, direction, rng);

        // NaN/inf guard
        if (!is_finite_f3(pr.radiance))
            pr.radiance = make_f3(0, 0, 0);

        accum = accum + pr.radiance;
        albedo_accum = albedo_accum + pr.albedo;
        normal_accum = normal_accum + pr.normal;

        // Pre-pass diagnostics: per-sample atomic counters
        if (params.prepass_active) {
            atomicAdd(params.prepass_total_paths, 1u);
            atomicAdd(params.prepass_bounce_sum, (unsigned int)pr.num_bounces);
            float lum_sample = luminance(pr.radiance);
            if (lum_sample <= 0.f)
                atomicAdd(params.prepass_zero_paths, 1u);
        }
    }

    float inv_spp = 1.f / (float)spp;
    float3 color = accum * inv_spp;

    // Write to output (accumulate for progressive rendering, SoA layout)
    if (params.frame_number == 0) {
        params.color_r[pixel_idx] = color.x;
        params.color_g[pixel_idx] = color.y;
        params.color_b[pixel_idx] = color.z;
        params.sample_counts[pixel_idx] = (float)spp;
    } else {
        // Progressive: weighted average with existing
        float old_count = params.sample_counts[pixel_idx];
        float new_count = old_count + (float)spp;
        float w_old = old_count / new_count;
        float w_new = (float)spp / new_count;
        params.color_r[pixel_idx] =
            params.color_r[pixel_idx] * w_old + color.x * w_new;
        params.color_g[pixel_idx] =
            params.color_g[pixel_idx] * w_old + color.y * w_new;
        params.color_b[pixel_idx] =
            params.color_b[pixel_idx] * w_old + color.z * w_new;
        params.sample_counts[pixel_idx] = new_count;
    }

    // AOV buffers (first sample only)
    if (params.frame_number == 0 && params.albedo_buffer) {
        float3 alb = albedo_accum * inv_spp;
        params.albedo_buffer[pixel_idx * 4 + 0] = alb.x;
        params.albedo_buffer[pixel_idx * 4 + 1] = alb.y;
        params.albedo_buffer[pixel_idx * 4 + 2] = alb.z;
        params.albedo_buffer[pixel_idx * 4 + 3] = 1.f;
    }
    if (params.frame_number == 0 && params.normal_buffer) {
        float3 nrm = normal_accum * inv_spp;
        params.normal_buffer[pixel_idx * 4 + 0] = nrm.x;
        params.normal_buffer[pixel_idx * 4 + 1] = nrm.y;
        params.normal_buffer[pixel_idx * 4 + 2] = nrm.z;
        params.normal_buffer[pixel_idx * 4 + 3] = 0.f;
    }

    // Adaptive variance tracking
    if (params.lum_sum) {
        float lum = color.x * 0.2126f + color.y * 0.7152f + color.z * 0.0722f;
        params.lum_sum[pixel_idx]  += lum;
        params.lum_sum2[pixel_idx] += lum * lum;
    }
}

#endif // PPT_INTEGRATOR_PATH_TRACER_CUH
