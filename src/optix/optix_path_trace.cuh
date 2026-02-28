#pragma once

// optix_path_trace.cuh – PathTraceResult struct and full_path_trace()
//                         Hybrid: NEE (direct) + Photon density (indirect)

// =====================================================================
// FULL PATH TRACING (final render only)
// Hybrid: NEE (direct) + Photon density estimation (indirect)
// Returns: combined, nee_direct, photon_indirect components separately
//
// =====================================================================
// full_path_trace — v2 Architecture (first-hit + specular chain + glossy continuation)
//
// Camera rays are first-hit probes.  Specular surfaces (mirror/glass)
// are followed through a chain of up to DEFAULT_MAX_SPECULAR_CHAIN bounces.
// At the first non-delta hit:
//   1. NEE captures direct illumination
//   2. Photon hash-grid gather captures indirect illumination
//   3. If glossy: BSDF-sample a continuation ray and repeat (up to
//      DEFAULT_MAX_GLOSSY_BOUNCES) — this produces scene reflections
//   4. If diffuse: stop (photon map has the rest)
// Volume integration is disabled in v2 (§Q9).
// =====================================================================
struct PathTraceResult {
    Spectrum combined;
    Spectrum nee_direct;
    Spectrum photon_indirect;
    // AOV for denoiser guide layers
    float3 first_hit_albedo;   // linear diffuse albedo at first non-specular hit
    float3 first_hit_normal;   // world-space shading normal at first non-specular hit
    // Kernel profiling clocks (accumulated across bounces)
    long long clk_ray_trace;
    long long clk_nee;
    long long clk_photon_gather;
    long long clk_bsdf;
};

__forceinline__ __device__
PathTraceResult full_path_trace(float3 origin, float3 direction, PCGRng& rng,
                                int pixel_idx,
                                int sample_index, int total_spp) {
    PathTraceResult result;
    result.combined        = Spectrum::zero();
    result.nee_direct      = Spectrum::zero();
    result.photon_indirect = Spectrum::zero();
    result.first_hit_albedo = make_f3(0.f, 0.f, 0.f);
    result.first_hit_normal = make_f3(0.f, 0.f, 1.f);
    result.clk_ray_trace     = 0;
    result.clk_nee           = 0;
    result.clk_photon_gather = 0;
    result.clk_bsdf          = 0;

    Spectrum throughput = Spectrum::constant(1.0f);
    IORStack ior_stack;  // track nested dielectrics across camera bounces
    bool aov_written = false;  // AOV captured at first non-specular hit

    const int max_spec = DEFAULT_MAX_SPECULAR_CHAIN;

    for (int bounce = 0; bounce <= max_spec; ++bounce) {
        long long t0 = clock64();
        TraceResult hit = trace_radiance(origin, direction);
        result.clk_ray_trace += clock64() - t0;

        if (!hit.hit) break;

        uint32_t mat_id = hit.material_id;

        // Emission: only on first bounce (camera sees a light)
        if (dev_is_emissive(mat_id) && bounce == 0) {
            result.combined  += throughput * dev_get_Le(mat_id, hit.uv);
            result.nee_direct += throughput * dev_get_Le(mat_id, hit.uv);
            break;
        }
        if (dev_is_emissive(mat_id)) break;  // specular chain hit a light

        // Specular bounce: follow the chain
        if (dev_is_specular(mat_id)) {
            SpecularBounceResult sb = dev_specular_bounce(
                direction, hit.position, hit.shading_normal, hit.geo_normal,
                mat_id, hit.uv, rng, nullptr, 0, &ior_stack);
            for (int i = 0; i < NUM_LAMBDA; ++i)
                throughput.value[i] *= sb.filter.value[i];
            direction = sb.new_dir;
            origin    = sb.new_pos;
            continue;
        }

        // Translucent: Fresnel boundary bounce, then continue.
        // Currently treated as a pure dielectric boundary (like glass).
        // TODO(D1): add NEE + photon gather at translucent surfaces —
        // requires dual-hemisphere gather and diffuse+specular BSDF split.
        if (dev_is_translucent(mat_id)) {
            SpecularBounceResult sb = dev_specular_bounce(
                direction, hit.position, hit.shading_normal, hit.geo_normal,
                mat_id, hit.uv, rng, nullptr, 0, &ior_stack);
            for (int i = 0; i < NUM_LAMBDA; ++i)
                throughput.value[i] *= sb.filter.value[i];
            direction = sb.new_dir;
            origin    = sb.new_pos;
            continue;
        }
        ONB frame = ONB::from_normal(hit.shading_normal);
        float3 wo_local = frame.world_to_local(direction * (-1.f));
        if (wo_local.z <= 0.f) break;

        // AOV: capture albedo + normal at first non-specular hit (for denoiser)
        if (!aov_written) {
            // Diffuse albedo in linear RGB
            Spectrum Kd_aov = dev_get_Kd(mat_id, hit.uv);
            float3 xyz = spectrum_to_xyz(Kd_aov);
            float3 lin = xyz_to_linear_srgb(xyz);
            result.first_hit_albedo = make_f3(
                fmaxf(lin.x, 0.f), fmaxf(lin.y, 0.f), fmaxf(lin.z, 0.f));
            result.first_hit_normal = hit.shading_normal;
            aov_written = true;
        }

        // 1. NEE: direct lighting via shadow ray
        if (params.render_mode != RenderMode::IndirectOnly) {
            long long t_nee = clock64();
            NeeResult nee = dev_nee_dispatch(
                hit.position, hit.shading_normal, wo_local,
                mat_id, rng, bounce, hit.uv);
            result.clk_nee += clock64() - t_nee;

            Spectrum nee_contrib = throughput * nee.L;
            result.combined   += nee_contrib;
            result.nee_direct += nee_contrib;
        }

        // 2. Photon hash-grid gather: indirect lighting
        if (params.render_mode != RenderMode::DirectOnly) {
            long long t_pg = clock64();
            Spectrum L_photon = dev_estimate_photon_density(
                hit.position, hit.shading_normal, hit.geo_normal,
                wo_local, mat_id, params.gather_radius, hit.uv);
            result.clk_photon_gather += clock64() - t_pg;

            Spectrum photon_contrib = throughput * L_photon;
            result.combined         += photon_contrib;
            result.photon_indirect  += photon_contrib;
        }

        // ── Glossy BSDF continuation (§7.1.1) ──────────────────────
        // If the surface is glossy, sample the BSDF to trace a
        // reflection ray and continue gathering at subsequent hits.
        // This is what produces visible scene reflections on glossy
        // surfaces (not just specular highlights of light sources).
        //
        // §4: When the cell-bin grid is available, use mixture sampling
        // (BSDF + photon-guided direction) with MIS to reduce variance.
        if (!dev_is_any_glossy(mat_id)) break;  // pure diffuse: stop

        // §4: Read the per-cell directional histogram once (O(1))
        DevPhotonBinDirs fib;
        fib.init(params.photon_bin_count);
        GuidedHistogram guide_hist = dev_read_cell_histogram(
            hit.position, hit.shading_normal);

        // BSDF continuation loop for glossy surfaces
        for (int g_bounce = 0; g_bounce < DEFAULT_MAX_GLOSSY_BOUNCES; ++g_bounce) {
            // §4 Mixture sampling: stochastically select BSDF or guided direction
            float3 wi_world;
            Spectrum f_for_throughput;
            float combined_pdf;
            bool sample_valid = false;

            // Guide fraction: how often to use guided direction vs BSDF.
            // Use guide when available and we have reasonable histogram data.
            // At glossy surfaces the BSDF is broad so guide helps significantly.
            float p_guide = (guide_hist.valid) ? 0.5f : 0.0f;

            if (p_guide > 0.f && rng.next_float() < p_guide) {
                // ── Guided sample (§4.4) ────────────────────────────
                GuidedSample gs = dev_sample_guided_direction(
                    guide_hist, fib, hit.shading_normal, rng);
                if (gs.valid) {
                    // Transform to local frame for BSDF evaluation
                    float3 wi_local = frame.world_to_local(gs.wi_world);
                    if (wi_local.z > 0.f) {
                        // Evaluate BSDF at the guided direction
                        Spectrum f_eval = bsdf_evaluate(mat_id, wo_local, wi_local, hit.uv);
                        float pdf_bsdf = dev_bsdf_pdf(mat_id, wo_local, wi_local);

                        // Combined mixture PDF (§4.4 MIS)
                        combined_pdf = p_guide * gs.pdf + (1.f - p_guide) * pdf_bsdf;
                        if (combined_pdf > 1e-8f) {
                            float cos_theta = wi_local.z;
                            for (int i = 0; i < NUM_LAMBDA; ++i)
                                f_for_throughput.value[i] = f_eval.value[i] * cos_theta / combined_pdf;
                            wi_world = gs.wi_world;
                            sample_valid = true;
                        }
                    }
                }
            }

            if (!sample_valid) {
                // ── BSDF sample (standard path) ─────────────────────
                long long t_bsdf = clock64();
                DevBSDFSample bs = dev_bsdf_sample(mat_id, wo_local, hit.uv, rng, pixel_idx);
                result.clk_bsdf += clock64() - t_bsdf;

                if (bs.pdf < 1e-8f || bs.wi.z <= 0.f) break;

                wi_world = frame.local_to_world(bs.wi);
                float cos_theta = bs.wi.z;

                // If guide is available, apply MIS to the BSDF sample
                if (p_guide > 0.f) {
                    float pdf_guide = dev_guided_pdf(guide_hist, fib, wi_world);
                    combined_pdf = (1.f - p_guide) * bs.pdf + p_guide * pdf_guide;
                } else {
                    combined_pdf = bs.pdf;
                }
                for (int i = 0; i < NUM_LAMBDA; ++i)
                    f_for_throughput.value[i] = bs.f.value[i] * cos_theta / combined_pdf;
                sample_valid = true;
            }

            if (!sample_valid) break;

            // Update throughput
            for (int i = 0; i < NUM_LAMBDA; ++i)
                throughput.value[i] *= f_for_throughput.value[i];

            // Russian roulette on throughput to avoid tracing negligible paths
            float max_tp = throughput.max_component();
            if (max_tp < 0.01f) break;
            if (g_bounce >= 1) {
                float survive = fminf(max_tp, 0.95f);
                if (rng.next_float() > survive) break;
                float inv_survive = 1.f / survive;
                for (int i = 0; i < NUM_LAMBDA; ++i)
                    throughput.value[i] *= inv_survive;
            }

            // Transform sampled direction to world space and trace
            origin = hit.position + hit.shading_normal * OPTIX_SCENE_EPSILON;

            long long t_rt = clock64();
            hit = trace_radiance(origin, wi_world);
            result.clk_ray_trace += clock64() - t_rt;

            if (!hit.hit) break;

            mat_id = hit.material_id;
            direction = wi_world;

            // If we hit an emissive surface, add its contribution.
            // This stochastic BSDF sample competes with the NEE shadow rays
            // fired at the previous diffuse/glossy hit.  Apply 2-way power-
            // heuristic MIS so both estimators contribute without double
            // counting.  NEE's matching weight is applied in dev_nee_direct().
            //
            // Note: throughput already contains (f * cosθ / pdf_bsdf) from
            // the BSDF sample above, so the raw contribution is
            //   throughput * Le  =  (f cosθ / pdf_bsdf) * Le.
            // The MIS weight scales this down when NEE would have had high
            // probability to reach this same emitter (p_nee >> pdf_bsdf).
            if (dev_is_emissive(mat_id)) {
                Spectrum Le = dev_get_Le(mat_id, hit.uv);
                float w_bsdf = 1.0f;
                if (DEFAULT_USE_MIS) {
                    // p_nee: solid-angle PDF that the NEE strategy would assign
                    // to this direction (power-weighted CDF × area → solid-angle).
                    float p_nee = dev_light_pdf(
                        hit.triangle_id, hit.geo_normal, direction, hit.t);
                    // Use the combined_pdf from the mixture sampling (§4 MIS)
                    w_bsdf = mis_weight_2(combined_pdf, p_nee);
                }
                result.combined   += throughput * Le * w_bsdf;
                result.nee_direct += throughput * Le * w_bsdf;
                break;
            }

            // If we hit a specular or translucent surface, follow it
            if (dev_is_specular(mat_id) || dev_is_translucent(mat_id)) {
                int spec_remain = max_spec - bounce;
                for (int s = 0; s < spec_remain; ++s) {
                    SpecularBounceResult sb = dev_specular_bounce(
                        direction, hit.position, hit.shading_normal, hit.geo_normal,
                        mat_id, hit.uv, rng, nullptr, 0, &ior_stack);
                    for (int i = 0; i < NUM_LAMBDA; ++i)
                        throughput.value[i] *= sb.filter.value[i];
                    direction = sb.new_dir;
                    origin    = sb.new_pos;

                    long long t_rt2 = clock64();
                    hit = trace_radiance(origin, direction);
                    result.clk_ray_trace += clock64() - t_rt2;
                    if (!hit.hit) goto done;
                    mat_id = hit.material_id;
                    if (dev_is_emissive(mat_id)) {
                        result.combined   += throughput * dev_get_Le(mat_id, hit.uv);
                        result.nee_direct += throughput * dev_get_Le(mat_id, hit.uv);
                        goto done;
                    }
                    if (!dev_is_specular(mat_id) && !dev_is_translucent(mat_id)) break;
                }
            }

            // NEE + photon gather at this bounce's hit
            frame = ONB::from_normal(hit.shading_normal);
            wo_local = frame.world_to_local(direction * (-1.f));
            if (wo_local.z <= 0.f) break;

            if (params.render_mode != RenderMode::IndirectOnly) {
                long long t_nee2 = clock64();
                NeeResult nee = dev_nee_dispatch(
                    hit.position, hit.shading_normal, wo_local,
                    mat_id, rng, bounce + g_bounce + 1, hit.uv);
                result.clk_nee += clock64() - t_nee2;

                Spectrum nee_c = throughput * nee.L;
                result.combined   += nee_c;
                result.nee_direct += nee_c;
            }

            if (params.render_mode != RenderMode::DirectOnly) {
                long long t_pg2 = clock64();
                Spectrum L_ph = dev_estimate_photon_density(
                    hit.position, hit.shading_normal, hit.geo_normal,
                    wo_local, mat_id, params.gather_radius, hit.uv);
                result.clk_photon_gather += clock64() - t_pg2;

                Spectrum ph_c = throughput * L_ph;
                result.combined        += ph_c;
                result.photon_indirect += ph_c;
            }

            // If the continuation hit is not glossy, stop
            if (!dev_is_any_glossy(mat_id)) break;

            // §4: Update guide histogram for the next glossy bounce
            guide_hist = dev_read_cell_histogram(hit.position, hit.shading_normal);
        }

        break;  // exit outer specular-chain loop
    }

done:
    return result;
}
