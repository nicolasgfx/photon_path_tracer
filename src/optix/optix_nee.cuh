#pragma once

// optix_nee.cuh – Light PDF, debug render modes, hash-grid photon gather,
//                  triangle sampling, NEE evaluate sample

// == Compute light sampling PDF for a direction that hit a light =======
__forceinline__ __device__
float dev_light_pdf(uint32_t tri_id, float3 geo_normal, float3 wi, float t) {
    if (params.num_emissive == 0) return 0.f;

    // O(1) lookup via inverse-index table (tri_id → local emissive index)
    float pdf_tri = 0.f;
    if (params.emissive_local_idx) {
        int i = params.emissive_local_idx[tri_id];
        if (i < 0) return 0.f;  // not an emissive triangle
        pdf_tri = (i == 0) ? params.emissive_cdf[0]
                           : params.emissive_cdf[i] - params.emissive_cdf[i - 1];
    } else {
        // Fallback: linear scan (shouldn't happen with current upload code)
        for (int i = 0; i < params.num_emissive; ++i) {
            if (params.emissive_tri_indices[i] == tri_id) {
                if (i == 0) pdf_tri = params.emissive_cdf[0];
                else pdf_tri = params.emissive_cdf[i] - params.emissive_cdf[i - 1];
                break;
            }
        }
    }
    if (pdf_tri <= 0.f) return 0.f;

    float3 v0 = params.vertices[tri_id * 3 + 0];
    float3 v1 = params.vertices[tri_id * 3 + 1];
    float3 v2 = params.vertices[tri_id * 3 + 2];
    float area = length(cross(v1 - v0, v2 - v0)) * 0.5f;
    if (area <= 0.f) return 0.f;

    float cos_o = fabsf(dot(wi * (-1.f), geo_normal));
    if (cos_o <= 0.f) return 0.f;

    float dist2 = t * t;
    return nee_pdf_area_to_solid_angle(pdf_tri, 1.f / area, dist2, cos_o);
}

// NOTE: CDF binary search lives in core/cdf.h (host+device).

// == Sample triangle barycentric (device) =============================
__forceinline__ __device__
float3 sample_triangle_dev(float u1, float u2) {
    float su = sqrtf(u1);
    float alpha = 1.f - su;
    float beta  = u2 * su;
    float gamma = 1.f - alpha - beta;
    return make_f3(alpha, beta, gamma);
}

// NeeResult: returned by all NEE variants
struct NeeResult {
    Spectrum L;                // direct lighting contribution
    float    visibility;       // fraction of unoccluded shadow samples [0,1]
};

// =====================================================================
// dev_nee_evaluate_sample -- shared inner loop for all NEE variants
//
// Given a selected emissive triangle (local_idx) and its selection
// probability (p_tri), samples a point on the triangle, casts a shadow
// ray, evaluates emission × BSDF × MIS, and returns the single-sample
// contribution.  Each NEE variant only needs to implement its own
// emitter selection strategy and PDF computation, then call this helper.
// =====================================================================
struct NeeSampleResult {
    Spectrum L;       // MIS-weighted contribution (zero if occluded/backfacing)
    bool     visible; // shadow ray unoccluded?
};

__forceinline__ __device__
NeeSampleResult dev_nee_evaluate_sample(
    int local_idx, float p_tri,
    float3 pos, float3 normal, float3 wo_local,
    uint32_t mat_id, const ONB& frame, float2 uv, PCGRng& rng)
{
    NeeSampleResult r;
    r.L = Spectrum::zero();
    r.visible = false;

    uint32_t light_tri = params.emissive_tri_indices[local_idx];

    // Sample uniform point on triangle
    float3 bary = sample_triangle_dev(rng.next_float(), rng.next_float());
    float3 lv0 = params.vertices[light_tri * 3 + 0];
    float3 lv1 = params.vertices[light_tri * 3 + 1];
    float3 lv2 = params.vertices[light_tri * 3 + 2];
    float3 light_pos = lv0 * bary.x + lv1 * bary.y + lv2 * bary.z;

    float3 le1 = lv1 - lv0;
    float3 le2 = lv2 - lv0;
    float3 cross_e = cross(le1, le2);
    float  cross_inv_len = rsqrtf(dot(cross_e, cross_e) + 1e-30f);
    float3 light_normal  = cross_e * cross_inv_len;
    float  light_area    = (1.f / cross_inv_len) * 0.5f;

    // Direction, distance, cosines
    float3 to_light = light_pos - pos;
    float dist2    = dot(to_light, to_light);
    float inv_dist = rsqrtf(dist2 + 1e-30f);
    float dist     = dist2 * inv_dist;
    float3 wi      = to_light * inv_dist;

    float cos_x = dot(wi, normal);
    float cos_y = -dot(wi, light_normal);
    if (cos_x <= 0.f || cos_y <= 0.f) return r;

    // Shadow ray
    if (!trace_shadow(pos + normal * OPTIX_SCENE_EPSILON, wi, dist))
        return r;
    r.visible = true;

    if (p_tri <= 0.f) return r;

    // Emission and BSDF
    uint32_t light_mat = params.material_ids[light_tri];
    float2 luv0 = params.texcoords[light_tri * 3 + 0];
    float2 luv1 = params.texcoords[light_tri * 3 + 1];
    float2 luv2 = params.texcoords[light_tri * 3 + 2];
    float2 light_uv = make_float2(
        luv0.x * bary.x + luv1.x * bary.y + luv2.x * bary.z,
        luv0.y * bary.x + luv1.y * bary.y + luv2.y * bary.z);
    Spectrum Le = dev_get_Le(light_mat, light_uv);

    float3 wi_local = frame.world_to_local(wi);
    Spectrum f = bsdf_evaluate(mat_id, wo_local, wi_local, uv);

    // PDF conversion: area → solid angle (shared helper)
    float p_wi = nee_pdf_area_to_solid_angle(p_tri, 1.f / light_area, dist2, cos_y);

    // MIS vs BSDF sampling
    float w_mis = 1.0f;
    {  // MIS always enabled
        float pdf_bsdf = bsdf_pdf(mat_id, wo_local, wi_local);
        w_mis = mis_weight_2(p_wi, pdf_bsdf);
    }

    // MIS-weighted: f * Le * cos_x / p_wi
    for (int i = 0; i < NUM_LAMBDA; ++i)
        r.L.value[i] = w_mis * f.value[i] * Le.value[i]
                      * cos_x / fmaxf(p_wi, 1e-8f);
    return r;
}

// =====================================================================
// Emitter selection: pure power-weighted CDF
// =====================================================================

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

// PDF for a given emissive index under the power-weighted CDF
__forceinline__ __device__
float dev_nee_global_pdf(int local_idx) {
    return (local_idx == 0)
        ? params.emissive_cdf[0]
        : params.emissive_cdf[local_idx] - params.emissive_cdf[local_idx - 1];
}

// =====================================================================
// Single-sample NEE direct lighting
// =====================================================================

__forceinline__ __device__
NeeResult dev_nee_direct(float3 pos, float3 normal, float3 wo_local,
                         uint32_t mat_id, PCGRng& rng, int bounce,
                         float2 uv)
{
    NeeResult result;
    result.L = Spectrum::zero();
    result.visibility = 0.f;
    if (params.num_emissive <= 0) return result;

    ONB frame = ONB::from_normal(normal);
    float p_tri;
    int local_idx = dev_nee_select_global(rng, p_tri);

    NeeSampleResult sr = dev_nee_evaluate_sample(
        local_idx, p_tri, pos, normal, wo_local, mat_id, frame, uv, rng);
    result.L = sr.L;
    result.visibility = sr.visible ? 1.f : 0.f;
    return result;
}

// ── MT-07: Volume photon kNN density estimate ────────────────────────
// kNN gather in 3D (Euclidean distance, no normal gate) for radiance
// estimation at terminal medium scatter events.  Uses the HG phase
// function instead of surface BSDF.  Sphere normalisation 3/(4π r³).
__forceinline__ __device__
Spectrum dev_estimate_volume_photon_density(
    float3 pos, float3 wo_world, float hg_g,
    const HomogeneousMedium& med)
{
    Spectrum L = Spectrum::zero();
    if (params.num_vol_photons == 0 || params.vol_grid_table_size == 0)
        return L;

    const float r_search = params.vol_gather_radius;
    const float r2_search = r_search * r_search;

    constexpr int KNN_K = DEFAULT_KNN_K;
    float    knn_d2[KNN_K];
    uint32_t knn_idx[KNN_K];
    int      knn_count = 0;

    const float cell_size = params.vol_grid_cell_size;
    int cx0 = (int)floorf((pos.x - r_search) / cell_size);
    int cy0 = (int)floorf((pos.y - r_search) / cell_size);
    int cz0 = (int)floorf((pos.z - r_search) / cell_size);
    int cx1 = (int)floorf((pos.x + r_search) / cell_size);
    int cy1 = (int)floorf((pos.y + r_search) / cell_size);
    int cz1 = (int)floorf((pos.z + r_search) / cell_size);

    uint32_t visited_keys[27];
    int num_visited = 0;

    for (int iz = cz0; iz <= cz1; ++iz)
    for (int iy = cy0; iy <= cy1; ++iy)
    for (int ix = cx0; ix <= cx1; ++ix) {
        uint32_t key = teschner_hash(make_i3(ix, iy, iz),
                                     params.vol_grid_table_size);
        bool already = false;
        for (int v = 0; v < num_visited; ++v)
            if (visited_keys[v] == key) { already = true; break; }
        if (already) continue;
        visited_keys[num_visited++] = key;

        uint32_t start = params.vol_grid_cell_start[key];
        uint32_t end   = params.vol_grid_cell_end[key];
        if (start == 0xFFFFFFFF) continue;

        for (uint32_t j = start; j < end; ++j) {
            uint32_t idx = params.vol_grid_sorted_indices[j];
            float3 pp = make_f3(
                params.vol_photon_pos_x[idx],
                params.vol_photon_pos_y[idx],
                params.vol_photon_pos_z[idx]);
            float3 diff = pos - pp;
            float d2 = dot(diff, diff);
            if (d2 > r2_search) continue;
            if (knn_count >= KNN_K && d2 >= knn_d2[0]) continue;

            // ── Insert into max-heap ────────────────────────────────
            if (knn_count < KNN_K) {
                knn_d2[knn_count]  = d2;
                knn_idx[knn_count] = idx;
                knn_count++;
                int ci = knn_count - 1;
                while (ci > 0) {
                    int pi = (ci - 1) / 2;
                    if (knn_d2[ci] <= knn_d2[pi]) break;
                    float td = knn_d2[ci]; knn_d2[ci] = knn_d2[pi]; knn_d2[pi] = td;
                    uint32_t ti = knn_idx[ci]; knn_idx[ci] = knn_idx[pi]; knn_idx[pi] = ti;
                    ci = pi;
                }
            } else {
                knn_d2[0]  = d2;
                knn_idx[0] = idx;
                int ci = 0;
                while (true) {
                    int left = 2*ci+1, right = 2*ci+2, largest = ci;
                    if (left  < KNN_K && knn_d2[left]  > knn_d2[largest]) largest = left;
                    if (right < KNN_K && knn_d2[right] > knn_d2[largest]) largest = right;
                    if (largest == ci) break;
                    float td = knn_d2[ci]; knn_d2[ci] = knn_d2[largest]; knn_d2[largest] = td;
                    uint32_t ti = knn_idx[ci]; knn_idx[ci] = knn_idx[largest]; knn_idx[largest] = ti;
                    ci = largest;
                }
            }
        }
    }

    if (knn_count == 0) return L;

    // Adaptive radius: k-th nearest or fallback to search radius
    float r_k2 = (knn_count >= KNN_K) ? knn_d2[0] : r2_search;
    r_k2 = fmaxf(r_k2, 1e-12f);
    float r_k = sqrtf(r_k2);

    // 3D sphere normalisation: 3/(4π r³)
    float inv_vol = 3.f / (4.f * PI * r_k * r_k2);
    float inv_emitted = 1.f / (float)params.num_vol_photons_emitted;

    for (int i = 0; i < knn_count; ++i) {
        uint32_t idx = knn_idx[i];
        float d2 = knn_d2[i];
        if (d2 >= r_k2) continue;

        // 3D Epanechnikov kernel: (1 - d²/r²)
        float w = 1.f - d2 / r_k2;

        // HG phase function evaluation
        float3 wi_world = make_f3(
            params.vol_photon_wi_x[idx],
            params.vol_photon_wi_y[idx],
            params.vol_photon_wi_z[idx]);
        float cos_theta = dot(wo_world * (-1.f), wi_world);
        float p_hg = henyey_greenstein_phase(cos_theta, hg_g);

        // Volume photons are single-hero (flat index)
        float p_flux = params.vol_photon_flux[idx];
        int   bin    = (int)params.vol_photon_lambda[idx];
        if (bin >= 0 && bin < NUM_LAMBDA)
            L.value[bin] += p_hg * p_flux * w * inv_vol * inv_emitted;
    }

    return L;
}
// =====================================================================
// dev_nee_dispatch — entry point for all NEE calls
// =====================================================================

__forceinline__ __device__
NeeResult dev_nee_dispatch(float3 pos, float3 normal, float3 wo_local,
                           uint32_t mat_id, PCGRng& rng, int bounce,
                           float2 uv)
{
    return dev_nee_direct(pos, normal, wo_local, mat_id, rng, bounce, uv);
}

// =====================================================================
// Volume NEE at medium scatter events (MT-06)
// =====================================================================
// Evaluates HG phase function instead of surface BSDF.
// Applies Beer-Lambert transmittance along the shadow ray for each λ.

__forceinline__ __device__
NeeResult dev_nee_volume_scatter(float3 pos, float3 wo_world,
                                  float hg_g, const HomogeneousMedium& med,
                                  PCGRng& rng)
{
    NeeResult result;
    result.L = Spectrum::zero();
    result.visibility = 0.f;
    if (params.num_emissive <= 0) return result;

    // Select a light from the power-weighted CDF
    float p_tri;
    int local_idx = dev_nee_select_global(rng, p_tri);
    uint32_t light_tri = params.emissive_tri_indices[local_idx];

    // Sample a point on the light triangle
    float3 bary = sample_triangle_dev(rng.next_float(), rng.next_float());
    float3 lv0 = params.vertices[light_tri * 3 + 0];
    float3 lv1 = params.vertices[light_tri * 3 + 1];
    float3 lv2 = params.vertices[light_tri * 3 + 2];
    float3 light_pos = lv0 * bary.x + lv1 * bary.y + lv2 * bary.z;

    float3 le1 = lv1 - lv0;
    float3 le2 = lv2 - lv0;
    float3 cross_e = cross(le1, le2);
    float  cross_inv_len = rsqrtf(dot(cross_e, cross_e) + 1e-30f);
    float3 light_normal  = cross_e * cross_inv_len;
    float  light_area    = (1.f / cross_inv_len) * 0.5f;

    // Direction and distance to light
    float3 to_light = light_pos - pos;
    float dist2    = dot(to_light, to_light);
    float inv_dist = rsqrtf(dist2 + 1e-30f);
    float dist     = dist2 * inv_dist;
    float3 wi      = to_light * inv_dist;

    // Light must face towards the scatter point
    float cos_y = -dot(wi, light_normal);
    if (cos_y <= 0.f) return result;

    // Shadow ray (no normal offset — we're inside a volume)
    if (!trace_shadow(pos, wi, dist))
        return result;
    result.visibility = 1.f;

    if (p_tri <= 0.f) return result;

    // Emission at light surface
    uint32_t light_mat = params.material_ids[light_tri];
    float2 luv0 = params.texcoords[light_tri * 3 + 0];
    float2 luv1 = params.texcoords[light_tri * 3 + 1];
    float2 luv2 = params.texcoords[light_tri * 3 + 2];
    float2 light_uv = make_float2(
        luv0.x * bary.x + luv1.x * bary.y + luv2.x * bary.z,
        luv0.y * bary.x + luv1.y * bary.y + luv2.y * bary.z);
    Spectrum Le = dev_get_Le(light_mat, light_uv);

    // Phase function value (HG) using cos(theta) between wo and wi
    float cos_theta = dot(wo_world * (-1.f), wi);
    float p_hg = henyey_greenstein_phase(cos_theta, hg_g);

    // PDF conversion: area → solid angle
    float p_wi = nee_pdf_area_to_solid_angle(p_tri, 1.f / light_area, dist2, cos_y);

    // MIS: NEE vs phase function sampling
    float w_mis = 1.0f;
    {  // MIS always enabled
        // Phase function PDF at this direction (normalised over full sphere)
        float pdf_phase = p_hg;  // HG is already normalised over 4π
        w_mis = mis_weight_2(p_wi, pdf_phase);
    }

    // Beer-Lambert transmittance along shadow ray (per wavelength)
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        float T_shadow = expf(-med.sigma_t.value[i] * dist);
        result.L.value[i] = w_mis * p_hg * Le.value[i] * T_shadow
                           / fmaxf(p_wi, 1e-8f);
    }

    return result;
}
