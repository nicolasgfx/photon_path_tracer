#ifndef PPT_INTEGRATOR_OPTIX_NEE_CUH
#define PPT_INTEGRATOR_OPTIX_NEE_CUH
// ─────────────────────────────────────────────────────────────────────
// optix_nee.cuh – Device-side Next Event Estimation (v5 RGB)
//
// Triangle-only NEE with power-weighted CDF for emitter selection and
// power-heuristic MIS (β = 2).
//
// No CPU fallback — pure OptiX device code.
// ─────────────────────────────────────────────────────────────────────
#include "accel/launch_params.h"
#include "accel/optix_utils.cuh"
#include "integrator/nee_shared.h"
#include "lighting/light_tree_device.cuh"
#include "material/bsdf_shared.h"

// Forward: extern params already declared in optix_utils.cuh

// ── Device-side texture sampling ─────────────────────────────────────
// Hardware-accelerated bilinear sampling via CUDA texture objects.
// Returns float4 (RGBA).  tex_id == -1 returns default (1,1,1,1).

__forceinline__ __device__
float4 dev_sample_texture(int tex_id, float2 uv) {
    if (tex_id < 0 || !params.textures || tex_id >= params.num_textures)
        return make_float4(1.f, 1.f, 1.f, 1.f);
    return tex2D<float4>(params.textures[tex_id], uv.x, 1.0f - uv.y);
}

__forceinline__ __device__
float3 dev_sample_texture_rgb(int tex_id, float2 uv) {
    float4 c = dev_sample_texture(tex_id, uv);
    return make_f3(c.x, c.y, c.z);
}

// ── Device-side material helpers (v5 RGB, simplified) ────────────────
// Access material arrays from LaunchParams for BSDF evaluate/pdf.
// Full GPU BSDF with all 9 types — mirrors CPU bsdf.h logic.

__forceinline__ __device__
float3 dev_get_Kd(uint32_t mat_id, float2 uv) {
    float3 base = make_f3(params.Kd[mat_id*3+0], params.Kd[mat_id*3+1], params.Kd[mat_id*3+2]);
    if (params.diffuse_tex && params.diffuse_tex[mat_id] >= 0) {
        float3 tex_col = dev_sample_texture_rgb(params.diffuse_tex[mat_id], uv);
        return base * tex_col;
    }
    return base;
}

__forceinline__ __device__
float3 dev_get_Ks(uint32_t mat_id, float2 uv) {
    float3 base = make_f3(params.Ks[mat_id*3+0], params.Ks[mat_id*3+1], params.Ks[mat_id*3+2]);
    if (params.specular_tex && params.specular_tex[mat_id] >= 0) {
        float3 tex_col = dev_sample_texture_rgb(params.specular_tex[mat_id], uv);
        return base * tex_col;
    }
    return base;
}

__forceinline__ __device__
float3 dev_get_Le(uint32_t mat_id, float2 uv) {
    float3 base = make_f3(params.Le[mat_id*3+0], params.Le[mat_id*3+1], params.Le[mat_id*3+2]);
    if (params.emission_tex && params.emission_tex[mat_id] >= 0) {
        float3 tex_col = dev_sample_texture_rgb(params.emission_tex[mat_id], uv);
        return base * tex_col;
    }
    return base;
}

__forceinline__ __device__
float dev_get_alpha_tex(uint32_t mat_id, float2 uv) {
    if (params.alpha_tex && params.alpha_tex[mat_id] >= 0) {
        float4 c = dev_sample_texture(params.alpha_tex[mat_id], uv);
        return c.w;  // use alpha channel
    }
    return 1.f;
}

__forceinline__ __device__
float3 dev_get_Tf(uint32_t mat_id) {
    if (!params.Tf) return make_f3(1.f, 1.f, 1.f);
    return make_f3(params.Tf[mat_id*3+0], params.Tf[mat_id*3+1], params.Tf[mat_id*3+2]);
}

__forceinline__ __device__
float dev_get_roughness(uint32_t mat_id) {
    return params.roughness[mat_id];
}

__forceinline__ __device__
float dev_get_ior(uint32_t mat_id) {
    return params.ior[mat_id];
}

__forceinline__ __device__
uint8_t dev_get_mat_type(uint32_t mat_id) {
    return params.mat_type[mat_id];
}

__forceinline__ __device__
bool dev_is_clearcoat(uint32_t mat_id) {
    return dev_get_mat_type(mat_id) == Clearcoat;
}

__forceinline__ __device__
bool dev_is_fabric(uint32_t mat_id) {
    return dev_get_mat_type(mat_id) == Fabric;
}

__forceinline__ __device__
bool dev_is_thin(uint32_t mat_id) {
    return params.mat_thin && params.mat_thin[mat_id];
}

// ── Simplified GPU BSDF evaluate (non-delta lobes only) ──────────────
// This evaluates the BSDF for NEE MIS. Delta lobes (mirror/glass)
// return zero since they can't be sampled by NEE.

__forceinline__ __device__
float3 dev_bsdf_evaluate(uint32_t mat_id, float3 wo, float3 wi, float2 uv) {
    // DiffuseTransmission: allows wi.z < 0 (transmission lobe)
    if (dev_get_mat_type(mat_id) == DiffuseTransmission && wo.z > 0.f) {
        float3 Kd = dev_get_Kd(mat_id, uv);
        if (wi.z > 0.f) return Kd * INV_PI;
        if (wi.z < 0.f) return dev_get_Tf(mat_id) * INV_PI;
        return make_f3(0,0,0);
    }

    if (wi.z <= 0.f || wo.z <= 0.f) return make_f3(0,0,0);

    uint8_t mt = dev_get_mat_type(mat_id);
    float3 Kd = dev_get_Kd(mat_id, uv);

    switch (mt) {
        case Lambertian:
        case Emissive:
            return Kd * INV_PI;

        case GlossyMetal: {
            float alpha = bsdf_roughness_to_alpha(dev_get_roughness(mat_id));
            float3 Ks = dev_get_Ks(mat_id, uv);
            float3 h = normalize(wo + wi);
            float ndf = ggx_D(h, alpha);
            float geo = ggx_G(wo, wi, alpha);
            float VdotH = fabsf(dot(wo, h));
            float denom = ggx_denom(wo, wi);
            float3 F_c = fresnel_schlick3(VdotH, Ks);
            float spec_term = ndf * geo / denom;
            return F_c * spec_term + Kd * INV_PI;
        }

        case GlossyDielectric: {
            float alpha = bsdf_roughness_to_alpha(dev_get_roughness(mat_id));
            float3 Ks = dev_get_Ks(mat_id, uv);
            float3 h = normalize(wo + wi);
            float ndf = ggx_D(h, alpha);
            float geo = ggx_G(wo, wi, alpha);
            float VdotH = fabsf(dot(wo, h));
            float denom = ggx_denom(wo, wi);
            float F0 = bsdf_f0_from_ior(dev_get_ior(mat_id));
            float Fr = fresnel_schlick(VdotH, F0);
            float spec_term = ndf * geo * Fr / denom;
            return Ks * spec_term + Kd * ((1.f - Fr) * INV_PI);
        }

        case Clearcoat: {
            float coat_w = params.clearcoat_weight ? params.clearcoat_weight[mat_id] : 1.0f;
            float coat_r = params.clearcoat_roughness ? params.clearcoat_roughness[mat_id] : 0.1f;
            float coat_alpha = fmaxf(coat_r * coat_r, 0.001f);
            float ior = dev_get_ior(mat_id);
            float coat_F0 = bsdf_f0_from_ior(ior);
            float3 h = normalize(wo + wi);
            float ndf_c = ggx_D(h, coat_alpha);
            float geo_c = ggx_G(wo, wi, coat_alpha);
            float VdotH = fabsf(dot(wo, h));
            float Fr = fresnel_schlick(VdotH, coat_F0);
            float denom = ggx_denom(wo, wi);
            float cs = coat_w * (ndf_c * geo_c * Fr) / denom;
            float base = (1.f - coat_w * Fr) * INV_PI;
            return make_f3(cs + Kd.x * base, cs + Kd.y * base, cs + Kd.z * base);
        }

        case Fabric: {
            float sheen_w = params.sheen ? params.sheen[mat_id] : 0.f;
            float tint = params.sheen_tint ? params.sheen_tint[mat_id] : 0.f;
            float3 h = normalize(wo + wi);
            float ct = fabsf(dot(wo, h));
            float t = 1.f - ct;
            float t5 = t * t * t * t * t;
            return make_f3(
                Kd.x * INV_PI + sheen_w * ((1.f-tint) + tint*Kd.x) * t5 * INV_PI,
                Kd.y * INV_PI + sheen_w * ((1.f-tint) + tint*Kd.y) * t5 * INV_PI,
                Kd.z * INV_PI + sheen_w * ((1.f-tint) + tint*Kd.z) * t5 * INV_PI);
        }

        case Mirror:
        case Glass:
        case Translucent:
            return make_f3(0,0,0); // delta distributions

        default:
            return make_f3(0,0,0);
    }
}

// ── Simplified GPU BSDF PDF (non-delta lobes only) ───────────────────

__forceinline__ __device__
float dev_bsdf_pdf(uint32_t mat_id, float3 wo, float3 wi, float2 uv) {
    // DiffuseTransmission: allows wi.z < 0 (transmission lobe)
    if (dev_get_mat_type(mat_id) == DiffuseTransmission && wo.z > 0.f) {
        float3 Kd = dev_get_Kd(mat_id, uv);
        float3 Tf = dev_get_Tf(mat_id);
        float w_r = max_component(Kd);
        float w_t = max_component(Tf);
        float total_w = w_r + w_t;
        float p_r = (total_w > 0.f) ? w_r / total_w : 0.5f;
        p_r = fmaxf(0.1f, fminf(0.9f, p_r));
        if (wi.z > 0.f) return p_r * cosine_hemisphere_pdf(wi.z);
        if (wi.z < 0.f) return (1.f - p_r) * cosine_hemisphere_pdf(fabsf(wi.z));
        return 0.f;
    }

    if (wi.z <= 0.f || wo.z <= 0.f) return 0.f;

    uint8_t mt = dev_get_mat_type(mat_id);

    switch (mt) {
        case Lambertian:
        case Emissive:
            return cosine_hemisphere_pdf(wi.z);

        case GlossyMetal: {
            float alpha = bsdf_roughness_to_alpha(dev_get_roughness(mat_id));
            float3 Ks = dev_get_Ks(mat_id, uv);
            float3 Kd = dev_get_Kd(mat_id, uv);
            LobeProbabilities lp = bsdf_lobe_probabilities(
                max_component(Ks), max_component(Kd));
            float diff_pdf = cosine_hemisphere_pdf(wi.z);
            float3 h = normalize(wo + wi);
            float ndf = ggx_D(h, alpha);
            float spec_pdf = ndf * ggx_G1(wo, alpha) / (4.f * fabsf(wo.z) + EPSILON);
            return lp.p_spec * spec_pdf + lp.p_diff * diff_pdf;
        }

        case GlossyDielectric: {
            float alpha = bsdf_roughness_to_alpha(dev_get_roughness(mat_id));
            float F0 = bsdf_f0_from_ior(dev_get_ior(mat_id));
            float3 Ks = dev_get_Ks(mat_id, uv);
            float3 Kd = dev_get_Kd(mat_id, uv);
            LobeProbabilities lp = bsdf_lobe_probabilities(
                max_component(Ks) * F0, max_component(Kd));
            float diff_pdf = cosine_hemisphere_pdf(wi.z);
            float3 h = normalize(wo + wi);
            float ndf = ggx_D(h, alpha);
            float spec_pdf = ndf * ggx_G1(wo, alpha) / (4.f * fabsf(wo.z) + EPSILON);
            return lp.p_spec * spec_pdf + lp.p_diff * diff_pdf;
        }

        case Clearcoat: {
            float coat_r = params.clearcoat_roughness ? params.clearcoat_roughness[mat_id] : 0.1f;
            float coat_alpha = fmaxf(coat_r * coat_r, 0.001f);
            float ior = dev_get_ior(mat_id);
            float coat_F0 = bsdf_f0_from_ior(ior);
            float coat_w = params.clearcoat_weight ? params.clearcoat_weight[mat_id] : 1.0f;
            float p_coat = fmaxf(fminf(coat_w * coat_F0, 0.95f), 0.05f);
            float diff_pdf = cosine_hemisphere_pdf(wi.z);
            float3 h = normalize(wo + wi);
            float ndf_c = ggx_D(h, coat_alpha);
            float spec_pdf = ndf_c * ggx_G1(wo, coat_alpha) / (4.f * fabsf(wo.z) + EPSILON);
            return p_coat * spec_pdf + (1.f - p_coat) * diff_pdf;
        }

        case Fabric:
            return cosine_hemisphere_pdf(wi.z);

        case Mirror:
        case Glass:
        case Translucent:
            return 0.f;

        default:
            return 0.f;
    }
}

// ── Light PDF for a direction that hit an emissive triangle ──────────
// pos = shading point (needed for light tree importance evaluation)

__forceinline__ __device__
float dev_light_pdf(uint32_t tri_id, float3 geo_normal, float3 wi, float t,
                    float3 pos) {
    if (params.num_emissive == 0) return 0.f;

    float3 v0 = params.vertices[tri_id * 3 + 0];
    float3 v1 = params.vertices[tri_id * 3 + 1];
    float3 v2 = params.vertices[tri_id * 3 + 2];
    float area = length(cross(v1 - v0, v2 - v0)) * 0.5f;
    if (area <= 0.f) return 0.f;

    float cos_o = dot(wi * (-1.f), geo_normal);
    if (cos_o <= 0.f) return 0.f;

    float dist2 = t * t;

    // triangle selection PDF depends on light tree vs CDF
    float pdf_tri = 0.f;
    if (params.light_tree_enabled && params.light_tree_nodes) {
        pdf_tri = dev_light_tree_pdf(pos, tri_id);
    } else {
        if (params.emissive_local_idx) {
            int i = params.emissive_local_idx[tri_id];
            if (i < 0) return 0.f;
            pdf_tri = (i == 0) ? params.emissive_cdf[0]
                               : params.emissive_cdf[i] - params.emissive_cdf[i - 1];
        }
    }
    if (pdf_tri <= 0.f) return 0.f;

    return nee_pdf_area_to_solid_angle(pdf_tri, 1.f / area, dist2, cos_o);
}

// ── Triangle barycentric sampling (device) ───────────────────────────

__forceinline__ __device__
float3 sample_triangle_dev(float u1, float u2) {
    float su = sqrtf(u1);
    float alpha = 1.f - su;
    float beta  = u2 * su;
    float gamma = 1.f - alpha - beta;
    return make_f3(alpha, beta, gamma);
}

// ── Emitter selection: power-weighted CDF ────────────────────────────

__forceinline__ __device__
int dev_nee_select_global(PCGRng& rng, float& p_tri_out) {
    float xi = rng.next_float();
    int local_idx = binary_search_cdf(
        params.emissive_cdf, params.num_emissive, xi);
    if (local_idx >= params.num_emissive) local_idx = params.num_emissive - 1;

    p_tri_out = (local_idx == 0)
        ? params.emissive_cdf[0]
        : params.emissive_cdf[local_idx] - params.emissive_cdf[local_idx - 1];
    return local_idx;
}

// ── NeeResult: returned by all NEE variants ──────────────────────────

struct NeeResult {
    float3 L;           // direct lighting contribution (RGB)
    float  visibility;  // 0 or 1
    float3 wi;          // direction toward light
    float  pdf_light;   // light sampling PDF (solid angle)
    float  pdf_bsdf;    // BSDF PDF for this direction
    float  mis_weight;  // balance heuristic weight
    int    light_type;  // 0=triangle, -1=N/A
};

// ── Evaluate a single triangle NEE sample ────────────────────────────

__forceinline__ __device__
NeeResult dev_nee_evaluate_triangle(
    int local_idx, float p_tri,
    float3 pos, float3 normal, float3 geo_normal, float3 wo_local,
    uint32_t mat_id, const ONB& frame, float2 receiver_uv, PCGRng& rng)
{
    NeeResult r;
    r.L = make_f3(0,0,0);
    r.visibility = 0.f;
    r.wi = make_f3(0,0,0);
    r.pdf_light = 0.f;
    r.pdf_bsdf = 0.f;
    r.mis_weight = 0.f;
    r.light_type = 0;

    uint32_t light_tri = params.emissive_tri_indices[local_idx];

    // Sample point on triangle
    float3 bary = sample_triangle_dev(rng.next_float(), rng.next_float());
    float3 lv0 = params.vertices[light_tri * 3 + 0];
    float3 lv1 = params.vertices[light_tri * 3 + 1];
    float3 lv2 = params.vertices[light_tri * 3 + 2];
    float3 light_pos = lv0 * bary.x + lv1 * bary.y + lv2 * bary.z;

    float3 le1 = lv1 - lv0;
    float3 le2 = lv2 - lv0;
    float3 cross_e = cross(le1, le2);
    float  cross_len = length(cross_e);
    float3 light_normal = cross_e / fmaxf(cross_len, 1e-30f);
    float  light_area   = cross_len * 0.5f;

    // Orient light_normal to agree with stored shading normal —
    // PLY winding order can disagree with per-vertex normals.
    float3 ln0 = params.normals[light_tri * 3];
    if (dot(light_normal, ln0) < 0.f)
        light_normal = light_normal * (-1.f);

    // Direction and distance
    float3 to_light = light_pos - pos;
    float dist2    = dot(to_light, to_light);
    float inv_dist = rsqrtf(dist2 + 1e-30f);
    float dist     = dist2 * inv_dist;
    float3 wi      = to_light * inv_dist;

    float cos_x = dot(wi, normal);
    float cos_y = -dot(wi, light_normal);

    // DiffuseTransmission receivers accept lights on either side
    bool is_diff_trans = (dev_get_mat_type(mat_id) == DiffuseTransmission);
    if (is_diff_trans) {
        // Still require light to face us (cos_y > 0)
        if (cos_y <= 0.f) return r;
    } else {
        if (cos_x <= 0.f || cos_y <= 0.f) return r;
    }

    // Geometric backface cull: reject when light is behind the actual
    // triangle face, even if the smooth shading normal says otherwise.
    // Prevents light leaking through thin geometry (gaps, seams).
    // DiffuseTransmission: light can be on either side.
    if (!is_diff_trans && dot(wi, geo_normal) <= 0.f) return r;

    // Shadow ray — no origin offset.  trace_shadow's built-in tmin
    // (1e-4) handles self-intersection with the originating triangle.
    // A normal-based offset would push the origin off the surface plane,
    // causing false occlusion when the receiver and light are coplanar
    // (e.g. ceiling adjacent to an area light at the same height).
    if (!trace_shadow(pos, wi, dist))
        return r;
    r.visibility = 1.f;
    r.wi = wi;

    if (p_tri <= 0.f) return r;

    // Emission (sample UV on the light triangle for textured emitters)
    uint32_t light_mat = params.material_ids[light_tri];
    float2 light_uv = make_f2(0.f, 0.f);
    {
        float2 luv0 = params.texcoords[light_tri * 3 + 0];
        float2 luv1 = params.texcoords[light_tri * 3 + 1];
        float2 luv2 = params.texcoords[light_tri * 3 + 2];
        light_uv = make_f2(
            luv0.x * bary.x + luv1.x * bary.y + luv2.x * bary.z,
            luv0.y * bary.x + luv1.y * bary.y + luv2.y * bary.z);
    }
    float3 Le = dev_get_Le(light_mat, light_uv);

    float3 wi_local = frame.world_to_local(wi);
    float3 f = dev_bsdf_evaluate(mat_id, wo_local, wi_local, receiver_uv);

    // PDF conversion: area → solid angle
    float p_wi = nee_pdf_area_to_solid_angle(p_tri, 1.f / light_area, dist2, cos_y);

    float pdf_bsdf_val = dev_bsdf_pdf(mat_id, wo_local, wi_local, receiver_uv);
    float w_mis = mis_weight_2(p_wi, pdf_bsdf_val);

    r.pdf_light = p_wi;
    r.pdf_bsdf = pdf_bsdf_val;
    r.mis_weight = w_mis;

    // Use shading-normal cosine so smooth meshes don't show per-triangle
    // faceting in direct lighting.  The geometric normal is still used for
    // the backface cull above (line 338) to prevent light leaks.
    // DiffuseTransmission: use |cos| since light can be on either side.
    float cos_g = is_diff_trans ? fabsf(dot(wi, normal)) : dot(wi, normal);
    float inv_pdf = 1.f / fmaxf(p_wi, 1e-8f);
    r.L = make_f3(
        w_mis * f.x * Le.x * cos_g * inv_pdf,
        w_mis * f.y * Le.y * cos_g * inv_pdf,
        w_mis * f.z * Le.z * cos_g * inv_pdf);

    return r;
}

// ── Triangle-only NEE dispatch ───────────────────────────────────────

__forceinline__ __device__
NeeResult dev_nee_direct(float3 pos, float3 normal, float3 geo_normal,
                         const ONB& frame, float3 wo_local,
                         uint32_t mat_id, float2 receiver_uv, PCGRng& rng)
{
    NeeResult result;
    result.L = make_f3(0,0,0);
    result.visibility = 0.f;
    result.wi = make_f3(0,0,0);
    result.pdf_light = 0.f;
    result.pdf_bsdf = 0.f;
    result.mis_weight = 0.f;
    result.light_type = 0;

    if (params.num_emissive <= 0) return result;

    if (params.light_tree_enabled && params.light_tree_nodes) {
        // light tree path: select triangle via importance-driven traversal
        float pdf_tree;
        int global_tri = dev_light_tree_sample(pos, rng, pdf_tree);
        // find local emissive index for evaluate_triangle (which uses it
        // only to look up emissive_tri_indices — we pass the global tri
        // directly by finding its local index)
        int local_idx = -1;
        if (params.emissive_local_idx)
            local_idx = params.emissive_local_idx[global_tri];
        if (local_idx < 0) return result;
        return dev_nee_evaluate_triangle(local_idx, pdf_tree, pos, normal,
            geo_normal, wo_local, mat_id, frame, receiver_uv, rng);
    }

    // fallback: power-weighted CDF
    float p_tri;
    int local_idx = dev_nee_select_global(rng, p_tri);

    return dev_nee_evaluate_triangle(local_idx, p_tri, pos, normal, geo_normal,
                                      wo_local, mat_id, frame, receiver_uv, rng);
}

// ── NEE dispatch (triangle emitters only) ────────────────────────────

__forceinline__ __device__
NeeResult dev_nee_dispatch(float3 pos, float3 normal, float3 geo_normal,
                           const ONB& frame, float3 wo_local,
                           uint32_t mat_id, float2 receiver_uv, PCGRng& rng)
{
    NeeResult result;
    result.L = make_f3(0,0,0);
    result.visibility = 0.f;
    result.wi = make_f3(0,0,0);
    result.pdf_light = 0.f;
    result.pdf_bsdf = 0.f;
    result.mis_weight = 0.f;
    result.light_type = -1;

    if (params.num_emissive <= 0) return result;

    // Pre-pass: count NEE attempt
    if (params.prepass_active)
        atomicAdd(params.prepass_nee_attempts, 1u);

    result = dev_nee_direct(pos, normal, geo_normal, frame, wo_local, mat_id, receiver_uv, rng);

    // Pre-pass: count NEE hit (shadow ray unoccluded)
    if (params.prepass_active && result.visibility > 0.f)
        atomicAdd(params.prepass_nee_hits, 1u);

    return result;
}

#endif // PPT_LIGHTING_OPTIX_NEE_CUH
