#pragma once

// optix_sppm.cuh – SPPM camera pass + gather pass (GPU)
//   Stochastic Progressive Photon Mapping per-pixel operations.

// =====================================================================
// SPPM camera pass – traces eye paths to first diffuse hit and stores
// the visible-point data per pixel.  Also evaluates NEE at the visible
// point for the direct-lighting component.
// =====================================================================
static __forceinline__ __device__ void sppm_camera_pass(
    int px, int py, int pixel_idx,
    float3 origin, float3 direction, PCGRng& rng)
{
    Spectrum throughput = Spectrum::constant(1.0f);
    Spectrum L_direct   = Spectrum::zero();
    IORStack ior_stack;  // track nested dielectrics across camera bounces

    for (int bounce = 0; bounce <= DEFAULT_MAX_SPECULAR_CHAIN; ++bounce) {
        TraceResult hit = trace_radiance(origin, direction);
        if (!hit.hit) break;

        uint32_t mat_id = hit.material_id;

        // Emission seen directly or via specular chain
        if (dev_is_emissive(mat_id) && bounce == 0) {
            Spectrum Le = dev_get_Le(mat_id, hit.uv);
            L_direct += throughput * Le;
        }

        // Specular (glass/mirror/translucent): bounce through
        if (dev_is_specular(mat_id) || dev_is_translucent(mat_id)) {
            SpecularBounceResult sb = dev_specular_bounce(
                direction, hit.position, hit.shading_normal, hit.geo_normal,
                mat_id, hit.uv, rng, nullptr, 0, &ior_stack);
            for (int i = 0; i < NUM_LAMBDA; ++i)
                throughput.value[i] *= sb.filter.value[i];
            direction = sb.new_dir;
            origin    = sb.new_pos;
            continue;
        }

        // ── Diffuse hit: store visible point ────────────────────
        ONB frame = ONB::from_normal(hit.shading_normal);
        float3 wo_local = frame.world_to_local(direction * (-1.f));
        if (wo_local.z <= 0.f) break;

        // Store visible-point data
        params.sppm_vp_pos_x[pixel_idx]  = hit.position.x;
        params.sppm_vp_pos_y[pixel_idx]  = hit.position.y;
        params.sppm_vp_pos_z[pixel_idx]  = hit.position.z;
        params.sppm_vp_norm_x[pixel_idx] = hit.geo_normal.x;
        params.sppm_vp_norm_y[pixel_idx] = hit.geo_normal.y;
        params.sppm_vp_norm_z[pixel_idx] = hit.geo_normal.z;
        params.sppm_vp_wo_x[pixel_idx]   = wo_local.x;
        params.sppm_vp_wo_y[pixel_idx]   = wo_local.y;
        params.sppm_vp_wo_z[pixel_idx]   = wo_local.z;
        params.sppm_vp_mat_id[pixel_idx] = mat_id;
        params.sppm_vp_uv_u[pixel_idx]   = hit.uv.x;
        params.sppm_vp_uv_v[pixel_idx]   = hit.uv.y;
        for (int i = 0; i < NUM_LAMBDA; ++i)
            params.sppm_vp_throughput[pixel_idx * NUM_LAMBDA + i] = throughput.value[i];
        params.sppm_vp_valid[pixel_idx] = 1;

        // ── NEE at visible point (via shared dev_nee_dispatch) ────
        NeeResult sppm_nee = dev_nee_dispatch(
            hit.position, hit.shading_normal, wo_local,
            mat_id, rng, bounce, hit.uv);
        L_direct += throughput * sppm_nee.L;

        break;  // stop at first diffuse hit
    }

    // Accumulate direct lighting into persistent buffer
    for (int i = 0; i < NUM_LAMBDA; ++i)
        params.sppm_L_direct[pixel_idx * NUM_LAMBDA + i] += L_direct.value[i];
}

// =====================================================================
// SPPM gather pass – for each pixel with a valid visible point, query
// the hash grid within the pixel's current radius, accumulate BSDF-
// weighted flux, and perform the progressive radius/flux update.
// =====================================================================
static __forceinline__ __device__ void sppm_gather_pass(int px, int py, int pixel_idx) {
    if (params.sppm_vp_valid[pixel_idx] == 0) return;

    // Read visible-point data
    float3 pos     = make_f3(params.sppm_vp_pos_x[pixel_idx],
                             params.sppm_vp_pos_y[pixel_idx],
                             params.sppm_vp_pos_z[pixel_idx]);
    float3 normal  = make_f3(params.sppm_vp_norm_x[pixel_idx],
                             params.sppm_vp_norm_y[pixel_idx],
                             params.sppm_vp_norm_z[pixel_idx]);
    float3 wo_local = make_f3(params.sppm_vp_wo_x[pixel_idx],
                              params.sppm_vp_wo_y[pixel_idx],
                              params.sppm_vp_wo_z[pixel_idx]);
    uint32_t mat_id = params.sppm_vp_mat_id[pixel_idx];
    float2 uv       = make_f2(params.sppm_vp_uv_u[pixel_idx],
                               params.sppm_vp_uv_v[pixel_idx]);

    float radius = params.sppm_radius[pixel_idx];
    float r2     = radius * radius;

    // Build ONB for BSDF evaluation
    ONB frame = ONB::from_normal(normal);

    // Read camera throughput
    Spectrum tp;
    for (int i = 0; i < NUM_LAMBDA; ++i)
        tp.value[i] = params.sppm_vp_throughput[pixel_idx * NUM_LAMBDA + i];

    // Hash grid query — gather photons within radius
    Spectrum phi = Spectrum::zero();
    int M = 0;

    float cell_size = params.grid_cell_size;
    int cx0 = (int)floorf((pos.x - radius) / cell_size);
    int cy0 = (int)floorf((pos.y - radius) / cell_size);
    int cz0 = (int)floorf((pos.z - radius) / cell_size);
    int cx1 = (int)floorf((pos.x + radius) / cell_size);
    int cy1 = (int)floorf((pos.y + radius) / cell_size);
    int cz1 = (int)floorf((pos.z + radius) / cell_size);

    uint32_t visited_keys[64];
    int num_visited = 0;

    for (int iz = cz0; iz <= cz1; ++iz)
    for (int iy = cy0; iy <= cy1; ++iy)
    for (int ix = cx0; ix <= cx1; ++ix) {
        uint32_t key = teschner_hash(make_i3(ix, iy, iz), params.grid_table_size);

        bool already = false;
        for (int v = 0; v < num_visited; ++v)
            if (visited_keys[v] == key) { already = true; break; }
        if (already) continue;
        if (num_visited >= 64) break;  // safety: prevent buffer overflow
        visited_keys[num_visited++] = key;

        uint32_t start = params.grid_cell_start[key];
        uint32_t end   = params.grid_cell_end[key];
        if (start == 0xFFFFFFFF) continue;

        for (uint32_t j = start; j < end; ++j) {
            uint32_t idx = params.grid_sorted_indices[j];
            float3 pp = make_f3(params.photon_pos_x[idx],
                                params.photon_pos_y[idx],
                                params.photon_pos_z[idx]);
            float3 diff = pos - pp;

            // Tangential-disk distance metric (§7.1 guideline)
            float d_plane = dot(diff, normal);
            float3 v_tan = diff - normal * d_plane;
            float d_tan2 = dot(v_tan, v_tan);
            if (d_tan2 > r2) continue;

            // Surface consistency
            float plane_dist = fabsf(d_plane);
            if (plane_dist > DEFAULT_SURFACE_TAU) continue;

            // Normal consistency (§15.1.2: threshold = 0)
            float3 photon_n = make_f3(params.photon_norm_x[idx],
                                      params.photon_norm_y[idx],
                                      params.photon_norm_z[idx]);
            if (dot(photon_n, normal) <= 0.0f) continue;

            // Direction consistency
            float3 wi_world = make_f3(params.photon_wi_x[idx],
                                      params.photon_wi_y[idx],
                                      params.photon_wi_z[idx]);
            if (dot(wi_world, normal) <= 0.f) continue;

            // Diffuse-only BSDF for density estimation (§6 standard practice).
            // Full Cook-Torrance creates 50x+ variance on glossy surfaces.
            float3 wi_local = frame.world_to_local(wi_world);
            Spectrum f = bsdf_evaluate_diffuse(mat_id, wo_local, wi_local, uv);

            // Epanechnikov kernel: w = 1 - d_tan²/r² (smooth falloff)
            float w = 1.0f - d_tan2 / r2;

            // Accumulate HERO_WAVELENGTHS bins per photon (multi-hero transport)
            int n_hero = params.photon_num_hero ? (int)params.photon_num_hero[idx] : 1;
            for (int h = 0; h < n_hero; ++h) {
                uint16_t bin = params.photon_lambda[idx * HERO_WAVELENGTHS + h];
                if (bin < NUM_LAMBDA) {
                    phi.value[bin] += w * f.value[bin] * params.photon_flux[idx * HERO_WAVELENGTHS + h];
                }
            }
            ++M;
        }
    }

    // Apply camera throughput
    for (int i = 0; i < NUM_LAMBDA; ++i)
        phi.value[i] *= tp.value[i];

    // ── Progressive update ──────────────────────────────────────
    if (M > 0) {
        float N_old = params.sppm_N[pixel_idx];
        float N_new = N_old + params.sppm_alpha * (float)M;
        float ratio = N_new / (N_old + (float)M);
        float r_old = radius;
        float r_new = r_old * sqrtf(ratio);
        if (r_new < params.sppm_min_radius) r_new = params.sppm_min_radius;

        float area_ratio = (r_new * r_new) / (r_old * r_old);

        for (int i = 0; i < NUM_LAMBDA; ++i) {
            params.sppm_tau[pixel_idx * NUM_LAMBDA + i] =
                (params.sppm_tau[pixel_idx * NUM_LAMBDA + i] + phi.value[i]) * area_ratio;
        }

        params.sppm_N[pixel_idx]      = N_new;
        params.sppm_radius[pixel_idx] = r_new;
    }

    // ── Reconstruct and tonemap ─────────────────────────────────
    // L_indirect = tau / (A_kernel * k * N_p)
    // With Epanechnikov kernel, A_kernel = pi*r^2/2  (not pi*r^2)
    float r_final = params.sppm_radius[pixel_idx];
    float denom = 0.5f * PI * r_final * r_final
                  * (float)(params.sppm_iteration + 1)
                  * (float)params.sppm_photons_per_iter;
    float inv_denom = (denom > 0.f) ? (1.f / denom) : 0.f;
    float inv_k = 1.f / (float)(params.sppm_iteration + 1);

    Spectrum L;
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        float tau_val = params.sppm_tau[pixel_idx * NUM_LAMBDA + i];
        float direct  = params.sppm_L_direct[pixel_idx * NUM_LAMBDA + i];
        L.value[i] = tau_val * inv_denom + direct * inv_k;
    }

    // Write to spectrum buffer for tonemap
    for (int i = 0; i < NUM_LAMBDA; ++i)
        params.spectrum_buffer[pixel_idx * NUM_LAMBDA + i] = L.value[i];
    params.sample_counts[pixel_idx] = 1.f;

    // Tonemap to sRGB
    float3 rgb = dev_spectrum_to_srgb(L);
    rgb.x = fminf(fmaxf(rgb.x, 0.f), 1.f);
    rgb.y = fminf(fmaxf(rgb.y, 0.f), 1.f);
    rgb.z = fminf(fmaxf(rgb.z, 0.f), 1.f);
    params.srgb_buffer[pixel_idx * 4 + 0] = (uint8_t)(rgb.x * 255.f);
    params.srgb_buffer[pixel_idx * 4 + 1] = (uint8_t)(rgb.y * 255.f);
    params.srgb_buffer[pixel_idx * 4 + 2] = (uint8_t)(rgb.z * 255.f);
    params.srgb_buffer[pixel_idx * 4 + 3] = 255;
}
