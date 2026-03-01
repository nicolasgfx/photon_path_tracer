#pragma once

// ─────────────────────────────────────────────────────────────────────
// optix_path_trace_v3.cuh — Photon-Guided Path Tracer (v3)
// ─────────────────────────────────────────────────────────────────────
// Part 2 §4: Single iterative bounce loop with:
//   1. NEE — 1 shadow ray, MIS-weighted against BSDF (Veach 1997)
//   2. Photon-guided direction sampling — cell-bin histogram provides
//      directional PDF mixed with BSDF (§4.3 mixture PDF)
//   3. Russian roulette after MIN_BOUNCES_RR guaranteed bounces (§4.1)
//   4. Photon caustic additive contribution at caustic cells (§4.2)
//   5. Photon final gather at terminal bounce (§4.2)
// ─────────────────────────────────────────────────────────────────────

// ── PathTraceResult ─────────────────────────────────────────────────
struct PathTraceResult {
    Spectrum combined;
    Spectrum nee_direct;
    Spectrum photon_indirect;
    // Per-bounce radiance contributions (DB-04, §10.3)
    Spectrum bounce_contrib[MAX_AOV_BOUNCES];
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
PathTraceResult full_path_trace_v3(float3 origin, float3 direction, PCGRng& rng,
                                    int pixel_idx,
                                    int sample_index, int total_spp) {
    PathTraceResult result;
    result.combined        = Spectrum::zero();
    result.nee_direct      = Spectrum::zero();
    result.photon_indirect = Spectrum::zero();
    for (int b = 0; b < MAX_AOV_BOUNCES; ++b)
        result.bounce_contrib[b] = Spectrum::zero();
    result.first_hit_albedo = make_f3(0.f, 0.f, 0.f);
    result.first_hit_normal = make_f3(0.f, 0.f, 1.f);
    result.clk_ray_trace     = 0;
    result.clk_nee           = 0;
    result.clk_photon_gather = 0;
    result.clk_bsdf          = 0;

    Spectrum throughput = Spectrum::constant(1.0f);
    IORStack ior_stack;
    MediumStack medium_stack;  // per-material interior medium tracking (§7.10)
    bool aov_written = false;

    // Fibonacci sphere directions for cell-bin histogram sampling
    DevPhotonBinDirs fib;
    fib.init(params.photon_bin_count);

    // Previous-bounce combined PDF for emission MIS (§4.3)
    float pdf_combined_prev = 0.f;

    const int max_bounces = params.max_bounces_camera;

    for (int bounce = 0; bounce < max_bounces; ++bounce) {

        // ── Trace ray ───────────────────────────────────────────────
        long long t0 = clock64();
        TraceResult hit = trace_radiance(origin, direction);
        result.clk_ray_trace += clock64() - t0;

        if (!hit.hit) break;

        // ── Per-material interior medium transport (§7.7) ───────────
        // When the camera ray is inside a Translucent object's medium,
        // apply free-flight sampling / Beer-Lambert over the segment.
        // Uses the same spectral MIS scheme as the atmospheric volume.
        bool inside_object_medium = false;
        {
            int cur_mid = medium_stack.current_medium_id();
            if (cur_mid >= 0 && params.media) {
                inside_object_medium = true;
                HomogeneousMedium med = dev_get_medium(cur_mid);

                constexpr int REF_BIN = NUM_LAMBDA / 2;
                float sig_t_ref = med.sigma_t.value[REF_BIN];

                if (sig_t_ref > 0.f) {
                    float u_ff = rng.next_float();
                    float t_ff = -logf(fmaxf(u_ff, 1e-12f)) / sig_t_ref;

                    if (t_ff < hit.t) {
                        // ── Scatter event inside per-material medium ─
                        float3 scatter_pos = origin + direction * t_ff;

                        // Spectral MIS scatter weight (§9.4)
                        float inv_ref = 1.f / sig_t_ref;
                        for (int i = 0; i < NUM_LAMBDA; ++i) {
                            float w = med.sigma_s.value[i]
                                    * expf(-(med.sigma_t.value[i] - sig_t_ref) * t_ff)
                                    * inv_ref;
                            throughput.value[i] *= w;
                        }

                        // NEE at scatter event
                        if (params.render_mode != RenderMode::IndirectOnly) {
                            NeeResult vnee = dev_nee_volume_scatter(
                                scatter_pos, direction, med.g, med, rng);
                            Spectrum vnee_contrib = throughput * vnee.L;
                            result.combined   += vnee_contrib;
                            result.nee_direct += vnee_contrib;
                            if (params.bounce_aov_enabled && bounce < MAX_AOV_BOUNCES)
                                result.bounce_contrib[bounce] += vnee_contrib;
                        }

                        // Terminal bounce: photon final gather
                        if (bounce == max_bounces - 1 && params.photon_final_gather) {
                            if (params.render_mode != RenderMode::DirectOnly) {
                                Spectrum L_vol = dev_estimate_volume_photon_density(
                                    scatter_pos, direction, med.g, med);
                                Spectrum vol_contrib = throughput * L_vol;
                                result.combined        += vol_contrib;
                                result.photon_indirect += vol_contrib;
                                if (params.bounce_aov_enabled && bounce < MAX_AOV_BOUNCES)
                                    result.bounce_contrib[bounce] += vol_contrib;
                            }
                            break;
                        }

                        // HG phase function sampling for next direction
                        ONB scatter_frame = ONB::from_normal(direction * (-1.f));
                        float3 wi_local = sample_henyey_greenstein(
                            med.g, rng.next_float(), rng.next_float());
                        float3 wi_world = scatter_frame.local_to_world(wi_local);

                        // RR inside per-material media
                        if (bounce >= params.min_bounces_rr) {
                            float max_tp = throughput.max_component();
                            float max_albedo = med.sigma_s.value[REF_BIN]
                                             / fmaxf(med.sigma_t.value[REF_BIN], 1e-20f);
                            float p_survive = fminf(params.rr_threshold,
                                                    fmaxf(max_tp, max_albedo));
                            if (p_survive < 1e-4f) break;
                            if (rng.next_float() >= p_survive) break;
                            float inv_survive = 1.f / p_survive;
                            for (int i = 0; i < NUM_LAMBDA; ++i)
                                throughput.value[i] *= inv_survive;
                        }

                        origin    = scatter_pos;
                        direction = wi_world;
                        pdf_combined_prev = 0.f;
                        continue;
                    } else {
                        // No scatter: Beer-Lambert transmittance
                        for (int i = 0; i < NUM_LAMBDA; ++i) {
                            throughput.value[i] *= expf(
                                -(med.sigma_t.value[i] - sig_t_ref) * hit.t);
                        }
                    }
                }
            }
        }

        // ── MT-02: Free-flight sampling inside atmospheric medium ───
        // §7.10 Double-attenuation guard: skip atmospheric volume when
        // inside a per-material medium to avoid double-counting.
        if (params.volume_enabled && params.volume_density > 0.f
            && !inside_object_medium) {
            float mid_y = origin.y + direction.y * (hit.t * 0.5f);
            HomogeneousMedium med = make_rayleigh_medium(
                params.volume_density, params.volume_albedo,
                params.volume_falloff, mid_y);

            // Reference wavelength for free-flight sampling
            constexpr int REF_BIN = NUM_LAMBDA / 2;
            float sig_t_ref = med.sigma_t.value[REF_BIN];

            if (sig_t_ref > 0.f) {
                float u_ff = rng.next_float();
                float t_ff = -logf(fmaxf(u_ff, 1e-12f)) / sig_t_ref;

                if (t_ff < hit.t) {
                    // ── Medium scatter event ────────────────────────
                    float3 scatter_pos = origin + direction * t_ff;

                    // Spectral MIS scatter weight (§9.4):
                    // w[i] = σ_s[i] * exp(-(σ_t[i] - σ_t_ref) * t) / σ_t_ref
                    float inv_ref = 1.f / sig_t_ref;
                    for (int i = 0; i < NUM_LAMBDA; ++i) {
                        float w = med.sigma_s.value[i]
                                * expf(-(med.sigma_t.value[i] - sig_t_ref) * t_ff)
                                * inv_ref;
                        throughput.value[i] *= w;
                    }

                    // ── MT-06: NEE at scatter event ─────────────────
                    if (params.render_mode != RenderMode::IndirectOnly) {
                        NeeResult vnee = dev_nee_volume_scatter(
                            scatter_pos, direction, med.g, med, rng);
                        Spectrum vnee_contrib = throughput * vnee.L;
                        result.combined   += vnee_contrib;
                        result.nee_direct += vnee_contrib;
                        if (params.bounce_aov_enabled && bounce < MAX_AOV_BOUNCES)
                            result.bounce_contrib[bounce] += vnee_contrib;
                    }

                    // ── MT-07: Volume photon final gather ───────────
                    // At the terminal bounce inside media, query the
                    // volume photon map for a kNN density estimate of
                    // remaining in-medium radiance (analogous to
                    // surface photon final gather at §4.2).
                    if (bounce == max_bounces - 1 && params.photon_final_gather) {
                        if (params.render_mode != RenderMode::DirectOnly) {
                            Spectrum L_vol = dev_estimate_volume_photon_density(
                                scatter_pos, direction, med.g, med);
                            Spectrum vol_contrib = throughput * L_vol;
                            result.combined        += vol_contrib;
                            result.photon_indirect += vol_contrib;
                            if (params.bounce_aov_enabled && bounce < MAX_AOV_BOUNCES)
                                result.bounce_contrib[bounce] += vol_contrib;
                        }
                        break;
                    }

                    // ── MT-04: HG + volume photon guide mixture ─────
                    // Read volume cell-bin histogram for guided scatter
                    GuidedHistogram vol_hist = dev_read_vol_cell_histogram(scatter_pos);
                    float p_vol_guide = (vol_hist.valid) ? 0.5f : 0.f;

                    float3 wi_world;
                    bool scatter_valid = false;

                    if (p_vol_guide > 0.f && rng.next_float() < p_vol_guide) {
                        // Sample from volume photon guide
                        GuidedSample gs = dev_sample_guided_direction(
                            vol_hist, fib, direction * (-1.f), rng);
                        if (gs.valid) {
                            wi_world = gs.wi_world;
                            scatter_valid = true;
                        }
                    }

                    if (!scatter_valid) {
                        // ── MT-03: HG phase function sampling ───────
                        ONB scatter_frame = ONB::from_normal(direction * (-1.f));
                        float3 wi_local = sample_henyey_greenstein(
                            med.g, rng.next_float(), rng.next_float());
                        wi_world = scatter_frame.local_to_world(wi_local);
                        scatter_valid = true;
                    }

                    // ── MT-05: Russian roulette inside media ────────
                    if (bounce >= params.min_bounces_rr) {
                        float max_tp = throughput.max_component();
                        float max_albedo = med.sigma_s.value[REF_BIN]
                                         / fmaxf(med.sigma_t.value[REF_BIN], 1e-20f);
                        float p_survive = fminf(params.rr_threshold,
                                                fmaxf(max_tp, max_albedo));
                        if (p_survive < 1e-4f) break;
                        if (rng.next_float() >= p_survive) break;
                        float inv_survive = 1.f / p_survive;
                        for (int i = 0; i < NUM_LAMBDA; ++i)
                            throughput.value[i] *= inv_survive;
                    }

                    // Continue from scatter point
                    origin    = scatter_pos;
                    direction = wi_world;
                    pdf_combined_prev = 0.f;  // no emission MIS after scatter
                    continue;
                } else {
                    // No scatter: apply Beer-Lambert transmittance
                    // w_transmit[i] = exp(-(σ_t[i] - σ_t_ref) * d)
                    // Combined with the free-flight "survival" this gives
                    // the standard exp(-σ_t[i] * d) transmittance.
                    for (int i = 0; i < NUM_LAMBDA; ++i) {
                        throughput.value[i] *= expf(
                            -(med.sigma_t.value[i] - sig_t_ref) * hit.t);
                    }
                }
            }
        }

        uint32_t mat_id = hit.material_id;

        // ── Emission (MIS-weighted) ─────────────────────────────────
        // §4.1: If the ray hits an emissive surface, add Le weighted
        // by MIS(pdf_combined_prev, pdf_nee).
        if (dev_is_emissive(mat_id)) {
            Spectrum Le = dev_get_Le(mat_id, hit.uv);
            if (bounce == 0) {
                // Camera sees light directly — no MIS needed
                Spectrum Le_contrib = throughput * Le;
                result.combined  += Le_contrib;
                result.nee_direct += Le_contrib;
                if (params.bounce_aov_enabled && bounce < MAX_AOV_BOUNCES)
                    result.bounce_contrib[bounce] += Le_contrib;
            } else {
                // MIS: BSDF/guide direction hit a light.  Weight against
                // the NEE strategy's PDF for this same emitter.
                // When previous bounce was delta (specular/glass), pdf is
                // Dirac (encoded as 0) → NEE cannot sample this path,
                // so give full weight to the BSDF strategy.
                float w_bsdf;
                if (pdf_combined_prev <= 0.f) {
                    w_bsdf = 1.0f;
                } else {
                    float p_nee = dev_light_pdf(
                        hit.triangle_id, hit.geo_normal, direction, hit.t);
                    w_bsdf = mis_weight_2(pdf_combined_prev, p_nee);
                }
                Spectrum Le_contrib = throughput * Le * w_bsdf;
                result.combined  += Le_contrib;
                result.nee_direct += Le_contrib;
                if (params.bounce_aov_enabled && bounce < MAX_AOV_BOUNCES)
                    result.bounce_contrib[bounce] += Le_contrib;
            }
            break;
        }

        // ── Delta surfaces: mirror, glass ───────────────────────────
        // §4.4: No NEE, no photon gather, no guide at delta surfaces.
        // Just update throughput and continue the loop.
        if (dev_is_specular(mat_id) && !dev_is_translucent(mat_id)) {
            SpecularBounceResult sb = dev_specular_bounce(
                direction, hit.position, hit.shading_normal, hit.geo_normal,
                mat_id, hit.uv, rng, nullptr, 0, &ior_stack,
                TransportMode::Radiance, &medium_stack);
            for (int i = 0; i < NUM_LAMBDA; ++i)
                throughput.value[i] *= sb.filter.value[i];
            direction = sb.new_dir;
            origin    = sb.new_pos;
            pdf_combined_prev = 0.f;  // delta: pdf is a Dirac → skip emission MIS on next bounce
            continue;
        }

        // ── Translucent surface (delta + medium boundary) ───────────
        // §7.7: Dielectric boundary with interior participating medium.
        // MediumStack push/pop happens inside dev_specular_bounce.
        if (dev_is_translucent(mat_id)) {
            SpecularBounceResult sb = dev_specular_bounce(
                direction, hit.position, hit.shading_normal, hit.geo_normal,
                mat_id, hit.uv, rng, nullptr, 0, &ior_stack,
                TransportMode::Radiance, &medium_stack);
            for (int i = 0; i < NUM_LAMBDA; ++i)
                throughput.value[i] *= sb.filter.value[i];
            direction = sb.new_dir;
            origin    = sb.new_pos;
            pdf_combined_prev = 0.f;
            continue;
        }

        // ── Non-delta surface: shading ──────────────────────────────
        ONB frame = ONB::from_normal(hit.shading_normal);
        float3 wo_local = frame.world_to_local(direction * (-1.f));
        if (wo_local.z <= 0.f) break;

        // AOV: capture albedo + normal at first non-specular hit
        if (!aov_written) {
            Spectrum Kd_aov = dev_get_Kd(mat_id, hit.uv);
            float3 xyz = spectrum_to_xyz(Kd_aov);
            float3 lin = xyz_to_linear_srgb(xyz);
            result.first_hit_albedo = make_f3(
                fmaxf(lin.x, 0.f), fmaxf(lin.y, 0.f), fmaxf(lin.z, 0.f));
            result.first_hit_normal = hit.shading_normal;
            aov_written = true;
        }

        // ── Photon analysis: read cell histogram ────────────────────
        // §3.1: Directional analysis from cell-bin grid (O(1)).
        GuidedHistogram guide_hist = dev_read_cell_histogram(
            hit.position, hit.shading_normal);

        // §3.6 M1: Adaptive guide fraction.
        // Use precomputed per-cell guide fraction if available,
        // otherwise compute from histogram quality inline.
        float p_guide = 0.f;
        if (guide_hist.valid) {
            if (params.cell_guide_fraction) {
                uint32_t cell = dev_cell_cache_index(hit.position);
                p_guide = params.cell_guide_fraction[cell];
            } else {
                // Fallback: simple histogram quality check
                int active_bins = 0;
                for (int k = 0; k < guide_hist.num_bins; ++k)
                    if (guide_hist.bin_flux[k] > 0.f) active_bins++;
                p_guide = (active_bins <= 2) ? 0.7f : params.guide_fraction;
            }
        }

        // ── Caustic photon contribution (additive, §4.2 / M5) ──────
        // §3.4: Caustic photons carry L→S→D transport that the camera
        // path cannot efficiently reproduce.  Add directly rather than
        // relying on the path tracer to discover the caustic path.
        // This uses the existing dual-budget photon density estimator
        // which separates caustic from global photons.
        // Currently the caustic contribution is folded into the normal
        // photon gather — an isolated caustic-only gather will be added
        // in a follow-up iteration when CellAnalysis is fully wired.

        // ── NEE: 1 shadow ray ───────────────────────────────────────
        // §4.1: Standard Veach-style MIS: sample a light, weight against BSDF.
        if (params.render_mode != RenderMode::IndirectOnly) {
            long long t_nee = clock64();
            NeeResult nee = dev_nee_dispatch(
                hit.position, hit.shading_normal, wo_local,
                mat_id, rng, bounce, hit.uv);
            result.clk_nee += clock64() - t_nee;

            Spectrum nee_contrib = throughput * nee.L;
            result.combined   += nee_contrib;
            result.nee_direct += nee_contrib;
            if (params.bounce_aov_enabled && bounce < MAX_AOV_BOUNCES)
                result.bounce_contrib[bounce] += nee_contrib;
        }

        // ── Photon final gather at terminal bounce ──────────────────
        // §4.2: At the last allowed bounce, instead of returning black,
        // query the photon map for an estimate of remaining indirect.
        if (bounce == max_bounces - 1 && params.photon_final_gather) {
            if (params.render_mode != RenderMode::DirectOnly) {
                long long t_pg = clock64();
                Spectrum L_photon = dev_estimate_photon_density(
                    hit.position, hit.shading_normal, hit.geo_normal,
                    wo_local, mat_id, params.gather_radius, hit.uv);
                result.clk_photon_gather += clock64() - t_pg;

                Spectrum photon_contrib = throughput * L_photon;
                result.combined        += photon_contrib;
                result.photon_indirect += photon_contrib;
                if (params.bounce_aov_enabled && bounce < MAX_AOV_BOUNCES)
                    result.bounce_contrib[bounce] += photon_contrib;
            }
            break;
        }

        // ── Next direction: guided or BSDF (§4.3 mixture PDF) ──────
        float3 wi_world;
        float combined_pdf = 0.f;
        Spectrum f_over_pdf = Spectrum::zero();
        bool sample_valid = false;

        if (p_guide > 0.f && rng.next_float() < p_guide) {
            // ── Guided sample ───────────────────────────────────────
            GuidedSample gs = dev_sample_guided_direction(
                guide_hist, fib, hit.shading_normal, rng);
            if (gs.valid) {
                float3 wi_local = frame.world_to_local(gs.wi_world);
                if (wi_local.z > 0.f) {
                    Spectrum f_eval = bsdf_evaluate(mat_id, wo_local, wi_local, hit.uv);
                    float pdf_bsdf = bsdf_pdf(mat_id, wo_local, wi_local);

                    // Mixture PDF: p_guide * pdf_guide + (1 - p_guide) * pdf_bsdf
                    combined_pdf = p_guide * gs.pdf + (1.f - p_guide) * pdf_bsdf;
                    if (combined_pdf > 1e-8f) {
                        float cos_theta = wi_local.z;
                        for (int i = 0; i < NUM_LAMBDA; ++i)
                            f_over_pdf.value[i] = f_eval.value[i] * cos_theta / combined_pdf;
                        wi_world = gs.wi_world;
                        sample_valid = true;
                    }
                }
            }
        }

        if (!sample_valid) {
            // ── BSDF sample ─────────────────────────────────────────
            long long t_bsdf = clock64();
            BSDFSample bs = bsdf_sample(mat_id, wo_local, hit.uv, rng, pixel_idx);
            result.clk_bsdf += clock64() - t_bsdf;

            if (bs.pdf < 1e-8f || bs.wi.z <= 0.f) break;

            wi_world = frame.local_to_world(bs.wi);
            float cos_theta = bs.wi.z;

            // Mixture PDF: apply MIS when guide is available
            if (p_guide > 0.f) {
                float pdf_guide = dev_guided_pdf(guide_hist, fib, wi_world);
                combined_pdf = (1.f - p_guide) * bs.pdf + p_guide * pdf_guide;
            } else {
                combined_pdf = bs.pdf;
            }
            if (combined_pdf < 1e-8f) break;

            for (int i = 0; i < NUM_LAMBDA; ++i)
                f_over_pdf.value[i] = bs.f.value[i] * cos_theta / combined_pdf;
            sample_valid = true;
        }

        if (!sample_valid) break;

        // Update throughput
        for (int i = 0; i < NUM_LAMBDA; ++i)
            throughput.value[i] *= f_over_pdf.value[i];

        // ── Russian roulette (§4.1) ─────────────────────────────────
        // After min_bounces_rr guaranteed bounces, probabilistically
        // terminate paths with low throughput.  Unbiased: the surviving
        // path is boosted by 1/p_survive.
        if (bounce >= params.min_bounces_rr) {
            float max_tp = throughput.max_component();
            float p_survive = fminf(params.rr_threshold, max_tp);
            if (p_survive < 1e-4f) break;  // negligible — terminate
            if (rng.next_float() >= p_survive) break;
            float inv_survive = 1.f / p_survive;
            for (int i = 0; i < NUM_LAMBDA; ++i)
                throughput.value[i] *= inv_survive;
        }

        // ── Prepare next ray ────────────────────────────────────────
        origin    = hit.position + hit.shading_normal * OPTIX_SCENE_EPSILON;
        direction = wi_world;
        pdf_combined_prev = combined_pdf;
    }

    return result;
}
