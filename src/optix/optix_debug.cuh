#pragma once

// optix_debug.cuh – debug_first_hit: simple direct-lighting preview
//                    (one ray, one shadow test, one frame)

// =====================================================================
// DEBUG FIRST-HIT RENDERING
// Simple direct-lighting: one ray, one shadow test, one frame
// =====================================================================
__forceinline__ __device__
Spectrum debug_first_hit(float3 origin, float3 direction, PCGRng& rng) {
    TraceResult hit = trace_radiance(origin, direction);
    if (!hit.hit) return Spectrum::zero();

    uint32_t mat_id = hit.material_id;

    // Debug visualisation modes
    if (params.render_mode == RenderMode::Normals)
        return render_normals_dev(hit);
    if (params.render_mode == RenderMode::MaterialID)
        return render_material_id_dev(mat_id);
    if (params.render_mode == RenderMode::Depth)
        return render_depth_dev(hit.t, 5.0f);

    // Per-triangle photon irradiance heatmap (§ preview photon map)
    if (params.show_photon_heatmap && params.tri_photon_irradiance
        && hit.triangle_id < (uint32_t)params.num_triangles) {
        float density = params.tri_photon_irradiance[hit.triangle_id];
        // Log-scale heatmap: blue(cold) → green → red(hot)
        float t = fminf(log2f(density * 1e4f + 1.f) / 15.f, 1.f);
        // Simple 3-stop gradient: blue → green → red
        float r = fminf(fmaxf(t * 2.f - 1.f, 0.f), 1.f);
        float g = 1.f - fabsf(t * 2.f - 1.f);
        float b = fminf(fmaxf(1.f - t * 2.f, 0.f), 1.f);
        Spectrum s;
        for (int i = 0; i < NUM_LAMBDA; ++i) s.value[i] = 0.f;
        for (int i = 0; i < NUM_LAMBDA/3; ++i) s.value[i] = r;
        for (int i = NUM_LAMBDA/3; i < 2*NUM_LAMBDA/3; ++i) s.value[i] = g;
        for (int i = 2*NUM_LAMBDA/3; i < NUM_LAMBDA; ++i) s.value[i] = b;
        return s;
    }

    // Emission
    if (dev_is_emissive(mat_id))
        return dev_get_Le(mat_id, hit.uv);

    // Specular: one bounce then direct lighting
    float3 cur_pos = hit.position;
    float3 cur_dir = direction;
    float3 cur_normal = hit.shading_normal;
    float3 cur_geo_normal = hit.geo_normal;
    uint32_t cur_mat = mat_id;
    float2 cur_uv = hit.uv;
    Spectrum throughput_s = Spectrum::constant(1.0f);

    for (int bounce = 0; bounce < DEFAULT_MAX_SPECULAR_CHAIN; ++bounce) {
        if (!dev_is_specular(cur_mat) && !dev_is_translucent(cur_mat)) break;

        SpecularBounceResult sb = dev_specular_bounce(
            cur_dir, cur_pos, cur_normal, cur_geo_normal, cur_mat, cur_uv, rng);
        for (int i = 0; i < NUM_LAMBDA; ++i)
            throughput_s.value[i] *= sb.filter.value[i];
        cur_dir = sb.new_dir;
        cur_pos = sb.new_pos;

        TraceResult hit2 = trace_radiance(cur_pos, cur_dir);
        if (!hit2.hit) return Spectrum::zero();
        if (dev_is_emissive(hit2.material_id))
            return throughput_s * dev_get_Le(hit2.material_id, hit2.uv);
        cur_pos = hit2.position;
        cur_normal = hit2.shading_normal;
        cur_geo_normal = hit2.geo_normal;
        cur_mat = hit2.material_id;
        cur_uv = hit2.uv;
    }

    // Diffuse/glossy hit: fast single-sample unshadowed direct lighting.
    // When params.debug_shadow_rays is on, __raygen__render routes to
    // full_path_trace instead, so this path is always the fast preview.
    Spectrum L = Spectrum::zero();

    {
        // Real-time debug: fast single-sample, no shadow ray
        // Supports both diffuse and glossy materials with reflection bounces
        Spectrum glossy_tp = Spectrum::constant(1.0f);

        for (int gb = 0; gb <= DEFAULT_MAX_GLOSSY_BOUNCES; ++gb) {
            ONB frame = ONB::from_normal(cur_normal);
            float3 wo_local = frame.world_to_local(normalize(-cur_dir));
            if (wo_local.z <= 0.f) break;

            // ── Fast unshadowed direct lighting at this hit ──
            if (params.num_emissive > 0) {
                float xi = rng.next_float();
                int local_idx = binary_search_cdf(
                    params.emissive_cdf, params.num_emissive, xi);
                if (local_idx >= params.num_emissive) local_idx = params.num_emissive - 1;
                uint32_t light_tri = params.emissive_tri_indices[local_idx];

                float3 bary = sample_triangle_dev(rng.next_float(), rng.next_float());
                float3 lv0 = params.vertices[light_tri * 3 + 0];
                float3 lv1 = params.vertices[light_tri * 3 + 1];
                float3 lv2 = params.vertices[light_tri * 3 + 2];
                float3 light_pos = lv0 * bary.x + lv1 * bary.y + lv2 * bary.z;

                float3 le1 = lv1 - lv0;
                float3 le2 = lv2 - lv0;
                float3 light_normal = normalize(cross(le1, le2));
                float  light_area   = length(cross(le1, le2)) * 0.5f;

                float3 to_light = light_pos - cur_pos;
                float dist2 = dot(to_light, to_light);
                float dist  = sqrtf(dist2);
                float3 wi   = to_light * (1.f / dist);

                float cos_i = dot(wi, cur_normal);
                float cos_o = -dot(wi, light_normal);

                if (cos_i > 0.f && cos_o > 0.f) {
                    float pdf_tri;
                    if (local_idx == 0)
                        pdf_tri = params.emissive_cdf[0];
                    else
                        pdf_tri = params.emissive_cdf[local_idx] - params.emissive_cdf[local_idx - 1];

                    float pdf_area = 1.f / light_area;
                    float geom = cos_o / dist2;
                    float pdf_solid_angle = pdf_tri * pdf_area / geom;

                    uint32_t light_mat = params.material_ids[light_tri];
                    // Interpolate UV at sampled light point for emission texture
                    float2 luv0 = params.texcoords[light_tri * 3 + 0];
                    float2 luv1 = params.texcoords[light_tri * 3 + 1];
                    float2 luv2 = params.texcoords[light_tri * 3 + 2];
                    float2 light_uv = make_float2(
                        luv0.x * bary.x + luv1.x * bary.y + luv2.x * bary.z,
                        luv0.y * bary.x + luv1.y * bary.y + luv2.y * bary.z);
                    Spectrum Le = dev_get_Le(light_mat, light_uv);

                    // Use full BSDF for glossy surfaces, Lambertian otherwise
                    float3 wi_local = frame.world_to_local(wi);
                    Spectrum bsdf_val = bsdf_evaluate(cur_mat, wo_local, wi_local, cur_uv);

                    for (int i = 0; i < NUM_LAMBDA; ++i) {
                        L.value[i] += glossy_tp.value[i] * Le.value[i]
                                     * bsdf_val.value[i]
                                     * cos_i / fmaxf(pdf_solid_angle, 1e-8f);
                    }
                }
            }

            // ── Glossy continuation: specular reflection for glossy surfaces ──
            // Trace a deterministic mirror-direction ray weighted by the correct
            // Monte Carlo weight G(wo,wi) × F(cosθ).  The GGX NDF D cancels
            // with the importance-sampling PDF, so we never evaluate D directly.
            // Only for near-mirror surfaces (roughness < 0.1) where one ray
            // at the specular peak is a reasonable approximation of the lobe.
            if (!dev_is_any_glossy(cur_mat)) break;

            // For clearcoat, use the coat roughness for continuation check
            float cont_roughness = dev_is_clearcoat(cur_mat)
                ? dev_get_clearcoat_roughness(cur_mat)
                : dev_get_roughness(cur_mat);
            if (cont_roughness >= 0.1f) break;

            // Mirror reflection direction (used as the half-vector = normal)
            float3 refl_dir = cur_dir - cur_normal * (2.f * dot(cur_dir, cur_normal));
            refl_dir = normalize(refl_dir);

            // Glossy continuation weight:
            // We trace one ray at the mirror direction.  The correct Monte
            // Carlo weight for a GGX-importance-sampled mirror direction is
            //   weight = G(wo,wi) · F(cosθ)
            // (the NDF D in the BRDF numerator cancels with the GGX sampling
            // PDF).  This avoids the old D·G·F/(4cos²) blowout for smooth
            // surfaces where D(n) → ∞.
            float cos_view = fabsf(dot(normalize(-cur_dir), cur_normal));
            float alpha_r = bsdf_roughness_to_alpha(cont_roughness);

            // Smith G for both wo and wi (same angle for mirror direction)
            float G_val = ggx_G1(wo_local, alpha_r);
            G_val *= G_val;  // G(wo,wi) ≈ G1(wo)·G1(wi), same angle

            if (dev_is_clearcoat(cur_mat)) {
                // Clearcoat: dielectric coat Fresnel, weight = coat_w × G × Fr
                float coat_w = dev_get_clearcoat_weight(cur_mat);
                float ior_r = dev_get_ior(cur_mat);
                float F0_r = ((ior_r - 1.f) / (ior_r + 1.f)) * ((ior_r - 1.f) / (ior_r + 1.f));
                float Fr_r = fresnel_schlick(cos_view, F0_r);
                for (int i = 0; i < NUM_LAMBDA; ++i)
                    glossy_tp.value[i] *= G_val * coat_w * Fr_r;
            } else if (dev_is_dielectric_glossy(cur_mat)) {
                Spectrum Ks_r = dev_get_Ks(cur_mat);
                float ior_r = dev_get_ior(cur_mat);
                float F0_r = ((ior_r - 1.f) / (ior_r + 1.f)) * ((ior_r - 1.f) / (ior_r + 1.f));
                float Fr_r = fresnel_schlick(cos_view, F0_r);
                for (int i = 0; i < NUM_LAMBDA; ++i)
                    glossy_tp.value[i] *= G_val * Fr_r * Ks_r.value[i];
            } else {
                // Metallic: per-channel Fresnel
                Spectrum Ks_r = dev_get_Ks(cur_mat);
                for (int i = 0; i < NUM_LAMBDA; ++i) {
                    float Fr_r = fresnel_schlick(cos_view, Ks_r.value[i]);
                    glossy_tp.value[i] *= G_val * Fr_r;
                }
            }

            float max_tp = glossy_tp.max_component();
            if (max_tp < 0.001f) break;

            cur_pos = cur_pos + cur_normal * OPTIX_SCENE_EPSILON;
            TraceResult hit_g = trace_radiance(cur_pos, refl_dir);
            if (!hit_g.hit) break;

            cur_mat = hit_g.material_id;
            cur_dir = refl_dir;

            if (dev_is_emissive(cur_mat)) {
                L += glossy_tp * dev_get_Le(cur_mat, hit_g.uv);
                break;
            }

            // Follow specular chain if we hit mirror/glass/translucent
            if (dev_is_specular(cur_mat) || dev_is_translucent(cur_mat)) {
                for (int s = 0; s < DEFAULT_MAX_SPECULAR_CHAIN; ++s) {
                    SpecularBounceResult sb = dev_specular_bounce(
                        cur_dir, hit_g.position, hit_g.shading_normal, hit_g.geo_normal,
                        cur_mat, hit_g.uv, rng);
                    for (int i = 0; i < NUM_LAMBDA; ++i)
                        glossy_tp.value[i] *= sb.filter.value[i];
                    cur_dir = sb.new_dir;
                    cur_pos = sb.new_pos;

                    hit_g = trace_radiance(cur_pos, cur_dir);
                    if (!hit_g.hit) goto debug_done;
                    cur_mat = hit_g.material_id;
                    if (dev_is_emissive(cur_mat)) {
                        L += glossy_tp * dev_get_Le(cur_mat, hit_g.uv);
                        goto debug_done;
                    }
                    if (!dev_is_specular(cur_mat) && !dev_is_translucent(cur_mat)) break;
                }
            }

            cur_pos    = hit_g.position;
            cur_normal = hit_g.shading_normal;
            cur_uv     = hit_g.uv;
        }
    }

debug_done:
    return throughput_s * L;
}
