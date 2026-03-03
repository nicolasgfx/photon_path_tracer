#pragma once

// optix_photon_trace.cuh – GPU photon emission and tracing
//   Multi-hero wavelength transport (PBRT v4 style, §1.1 config.h):
//   Each photon carries HERO_WAVELENGTHS bins with stratified offsets.

// =====================================================================
// __raygen__photon_trace  -  GPU photon emission and tracing
// =====================================================================
extern "C" __global__ void __raygen__photon_trace() {
    const uint3 idx = optixGetLaunchIndex();
    int photon_idx = idx.x;
    if (photon_idx >= params.num_photons) return;
    if (params.num_emissive <= 0) return;

    // Incorporate photon_map_seed for multi-map re-tracing (§1.2)
    PCGRng rng = PCGRng::seed(
        (uint64_t)photon_idx * 7 + 42 + (uint64_t)params.photon_map_seed * 0x100000007ULL,
        (uint64_t)photon_idx + 1);

    // 1. Sample emissive triangle (power CDF)
    int local_idx = 0;
    {
        float xi = rng.next_float();
        local_idx = binary_search_cdf(
            params.emissive_cdf, params.num_emissive, xi);
        if (local_idx >= params.num_emissive) local_idx = params.num_emissive - 1;
    }
    uint32_t tri_idx = params.emissive_tri_indices[local_idx];

    float pdf_tri;
    if (local_idx == 0) pdf_tri = params.emissive_cdf[0];
    else pdf_tri = params.emissive_cdf[local_idx] - params.emissive_cdf[local_idx - 1];

    // 2. Get triangle geometry
    float3 v0 = params.vertices[tri_idx * 3 + 0];
    float3 v1 = params.vertices[tri_idx * 3 + 1];
    float3 v2 = params.vertices[tri_idx * 3 + 2];
    uint32_t mat_id = params.material_ids[tri_idx];

    // 3. Sample point on triangle
    float3 bary = sample_triangle_dev(rng.next_float(), rng.next_float());
    float3 pos = v0 * bary.x + v1 * bary.y + v2 * bary.z;
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 geo_n = normalize(cross(e1, e2));
    float  area  = length(cross(e1, e2)) * 0.5f;
    float  pdf_pos = 1.f / area;

    // 4. Sample HERO_WAVELENGTHS stratified wavelength bins
    //    Hero = sampled from Le CDF; companions at stratified offsets
    //    (Wilkie et al. 2002 / PBRT v4 style)
    // Interpolate UV at emitter sample point for emission texture
    float2 euv0 = params.texcoords[tri_idx * 3 + 0];
    float2 euv1 = params.texcoords[tri_idx * 3 + 1];
    float2 euv2 = params.texcoords[tri_idx * 3 + 2];
    float2 emit_uv = make_float2(
        euv0.x * bary.x + euv1.x * bary.y + euv2.x * bary.z,
        euv0.y * bary.x + euv1.y * bary.y + euv2.y * bary.z);
    Spectrum Le = dev_get_Le(mat_id, emit_uv);
    float Le_sum = 0.f;
    for (int i = 0; i < NUM_LAMBDA; ++i) Le_sum += Le.value[i];
    if (Le_sum <= 0.f) return;

    // Sample primary hero wavelength from Le CDF
    float xi_lambda = rng.next_float() * Le_sum;
    int hero_bin = NUM_LAMBDA - 1;
    float cum = 0.f;
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        cum += Le.value[i];
        if (xi_lambda <= cum) { hero_bin = i; break; }
    }

    // Build stratified companion bins
    int   hero_bins[HERO_WAVELENGTHS];
    float hero_flux[HERO_WAVELENGTHS];
    int   num_hero = HERO_WAVELENGTHS;

    hero_bins[0] = hero_bin;
    for (int h = 1; h < HERO_WAVELENGTHS; ++h) {
        // Stratified offset: evenly spaced across the spectrum
        int offset = (h * NUM_LAMBDA) / HERO_WAVELENGTHS;
        int companion = (hero_bin + offset) % NUM_LAMBDA;
        hero_bins[h] = companion;
    }

    // 5. Sample cosine-weighted direction within emission cone
    constexpr float cone_half_rad = DEFAULT_LIGHT_CONE_HALF_ANGLE_DEG * (PI / 180.0f);
    const float cos_cone_max = cosf(cone_half_rad);
    float3 local_dir = sample_cosine_cone_dev(rng, cos_cone_max);
    ONB frame = ONB::from_normal(geo_n);
    float3 world_dir = frame.local_to_world(local_dir);
    float cos_theta = local_dir.z;
    float cone_denom = PI * (1.0f - cos_cone_max * cos_cone_max);
    float pdf_dir = (cone_denom > 0.f) ? cos_theta / cone_denom : cos_theta * INV_PI;

    // 6. Compute initial flux per hero wavelength
    //    Each hero channel: flux_h = Le(λ_h) * cos / (pdf_tri * pdf_pos * pdf_dir * pdf_lambda_h)
    //    pdf_lambda_h for companion channels uses the same Le CDF probability
    //    as if that bin had been directly sampled: pdf_lambda_h = Le(λ_h) / Le_sum
    //
    //    PBRT v4 §14.3 hero-wavelength normalization: divide by
    //    HERO_WAVELENGTHS because each physical photon contributes to
    //    HERO_WAVELENGTHS spectral bins, but the density estimator
    //    divides by N_emitted (the number of physical photons, not
    //    the number of per-bin contributions).  Without this factor
    //    the indirect component is HERO_WAVELENGTHS× too bright.
    float denom_common = pdf_tri * pdf_pos * pdf_dir;
    float inv_hero = 1.0f / (float)HERO_WAVELENGTHS;
    for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
        int bin = hero_bins[h];
        float Le_h = Le.value[bin];
        float pdf_lambda_h = Le_h / Le_sum;
        hero_flux[h] = (denom_common * pdf_lambda_h > 0.f)
                     ? (Le_h * cos_theta) / (denom_common * pdf_lambda_h) * inv_hero
                     : 0.f;
    }

    // 7. Trace through scene
    float3 origin    = pos + geo_n * OPTIX_SCENE_EPSILON;
    float3 direction = world_dir;
    // Caustic tracking (matches CPU emitter.h convention):
    // Starts false; set true on first specular/translucent hit; reset
    // to false on diffuse/glossy.  Only L→S→D (or L→D→S→D) paths
    // are tagged as caustic.
    bool on_caustic_path = false;
    uint8_t path_flags = 0;  // PHOTON_FLAG_* accumulator for F2 debug overlay
    IORStack ior_stack;      // track nested dielectrics across bounces
    MediumStack medium_stack; // track per-material interior media (§7.10)

    for (int bounce = 0; bounce < params.photon_max_bounces; ++bounce) {
        TraceResult hit = trace_radiance(origin, direction);
        if (!hit.hit) break;

        // RNG spatial decorrelation
        {
            uint32_t cell_key = teschner_hash(make_i3(
                (int)floorf(hit.position.x / params.dense_cell_size),
                (int)floorf(hit.position.y / params.dense_cell_size),
                (int)floorf(hit.position.z / params.dense_cell_size)),
                0x7FFFFFFFu);
            rng.advance(cell_key * 0x9E3779B9u);
        }

        uint32_t hit_mat = hit.material_id;

        // ── Per-material interior medium (§7.7 Translucent) ────────
        // If the photon is inside a per-material medium, apply
        // Beer-Lambert transmittance over the ray segment.
        bool inside_object_medium = false;
        {
            int cur_mid = medium_stack.current_medium_id();
            if (cur_mid >= 0 && params.media) {
                inside_object_medium = true;
                HomogeneousMedium med = dev_get_medium(cur_mid);
                float seg_t = hit.t;
                // Beer-Lambert attenuation per hero wavelength
                for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
                    float sig_t_h = med.sigma_t.value[hero_bins[h]];
                    hero_flux[h] *= expf(-sig_t_h * seg_t);
                }
            }
        }

        // ── Atmospheric volume photon deposit (Beer–Lambert free-flight) ───────
        // §7.10 Double-attenuation guard: skip atmospheric volume when
        // inside a per-material medium to avoid double-counting extinction.
        if (params.volume_enabled && params.volume_density > 0.f && hit.hit
            && !inside_object_medium) {
            float seg_t = hit.t;
            float mid_y = origin.y + direction.y * (seg_t * 0.5f);
            HomogeneousMedium med = make_rayleigh_medium(
                params.volume_density, params.volume_albedo,
                params.volume_falloff, mid_y);

            // Use hero bin 0 for volume scattering decision
            float sig_t_lam = med.sigma_t.value[hero_bins[0]];
            if (sig_t_lam > 0.f) {
                float u_ff = rng.next_float();
                float t_ff = -logf(fmaxf(1.f - u_ff, 1e-12f)) / sig_t_lam;

                if (t_ff < seg_t) {
                    path_flags |= 0x04;  // PHOTON_FLAG_VOLUME_SCATTER
                    float3 vol_pos = origin + direction * t_ff;
                    float sig_s_lam = med.sigma_s.value[hero_bins[0]];
                    float vol_flux = hero_flux[0] * (sig_s_lam / fmaxf(sig_t_lam, 1e-20f));

                    uint32_t vslot = atomicAdd(params.out_vol_photon_count, 1u);
                    if (vslot < (uint32_t)params.max_stored_vol_photons) {
                        params.out_vol_photon_pos_x[vslot]  = vol_pos.x;
                        params.out_vol_photon_pos_y[vslot]  = vol_pos.y;
                        params.out_vol_photon_pos_z[vslot]  = vol_pos.z;
                        params.out_vol_photon_wi_x[vslot]   = -direction.x;
                        params.out_vol_photon_wi_y[vslot]   = -direction.y;
                        params.out_vol_photon_wi_z[vslot]   = -direction.z;
                        params.out_vol_photon_lambda[vslot]  = (uint16_t)hero_bins[0];
                        params.out_vol_photon_flux[vslot]    = vol_flux;
                    }
                }

                // Attenuate all hero flux by transmittance over this segment
                for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
                    float sig_t_h = med.sigma_t.value[hero_bins[h]];
                    hero_flux[h] *= expf(-sig_t_h * seg_t);
                }
            }
        }

        // Skip emissive surfaces
        if (dev_is_emissive(hit_mat)) break;

        // Store photon at diffuse surfaces (skip bounce 0 = direct lighting)
        if (!dev_is_specular(hit_mat) && !dev_is_translucent(hit_mat) && bounce > 0) {
            // In caustic-only mode, skip non-caustic photons
            if (params.caustic_only_store && !on_caustic_path) {
                // Don't store non-caustic photon.  Note: on_caustic_path
                // CAN become true again if a subsequent specular hit occurs
                // (L→D→S→D path), so we continue bouncing rather than break.
            } else {
            uint32_t slot = atomicAdd(params.out_photon_count, 1u);
            if (slot < (uint32_t)params.max_stored_photons) {
                params.out_photon_pos_x[slot]   = hit.position.x;
                params.out_photon_pos_y[slot]   = hit.position.y;
                params.out_photon_pos_z[slot]   = hit.position.z;
                params.out_photon_wi_x[slot]    = -direction.x;
                params.out_photon_wi_y[slot]    = -direction.y;
                params.out_photon_wi_z[slot]    = -direction.z;
                params.out_photon_norm_x[slot]  = hit.geo_normal.x;
                params.out_photon_norm_y[slot]  = hit.geo_normal.y;
                params.out_photon_norm_z[slot]  = hit.geo_normal.z;
                // Write HERO_WAVELENGTHS bins per photon (interleaved)
                for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
                    params.out_photon_lambda[slot * HERO_WAVELENGTHS + h] = (uint16_t)hero_bins[h];
                    params.out_photon_flux[slot * HERO_WAVELENGTHS + h]   = hero_flux[h];
                }
                params.out_photon_num_hero[slot] = (uint8_t)num_hero;
                params.out_photon_source_emissive[slot] = (uint16_t)local_idx;
                if (params.out_photon_is_caustic)
                    params.out_photon_is_caustic[slot] = on_caustic_path ? (uint8_t)1 : (uint8_t)0;
                if (params.out_photon_path_flags)
                    params.out_photon_path_flags[slot] = path_flags;
                if (params.out_photon_tri_id)
                    params.out_photon_tri_id[slot] = hit.triangle_id;
            }
            }
        }

        // Bounce — track per-hero-channel throughput
        float rr_albedo = 1.0f;
        if (dev_is_specular(hit_mat) || dev_is_translucent(hit_mat)) {
            on_caustic_path = true;  // specular → mark caustic (matches CPU emitter.h)
            // Track detailed path flags for F2 debug overlay
            // Must distinguish Glass/Translucent (glass flags) from Mirror (specular flag).
            // dev_is_specular() returns true for BOTH Mirror AND Glass, so we need
            // explicit glass check.  Matches CPU emitter.h flag logic.
            if (dev_is_glass(hit_mat) || dev_is_translucent(hit_mat)) {
                path_flags |= 0x01;  // PHOTON_FLAG_TRAVERSED_GLASS
                if (bounce == 0)
                    path_flags |= 0x02;  // PHOTON_FLAG_CAUSTIC_GLASS (direct caustic only)
                if (dev_has_dispersion(hit_mat))
                    path_flags |= 0x08;  // PHOTON_FLAG_DISPERSION
            } else {
                path_flags |= 0x10;  // PHOTON_FLAG_CAUSTIC_SPECULAR (Mirror only)
            }
            SpecularBounceResult sb = dev_specular_bounce(
                direction, hit.position, hit.shading_normal, hit.geo_normal,
                hit_mat, hit.uv, rng, hero_bins, HERO_WAVELENGTHS, &ior_stack,
                TransportMode::Importance, &medium_stack);
            // Apply transmittance filter to hero channels
            for (int h = 0; h < HERO_WAVELENGTHS; ++h)
                hero_flux[h] *= sb.filter.value[hero_bins[h]];
            direction = sb.new_dir;
            origin    = sb.new_pos;
        } else if (dev_is_any_glossy(hit_mat)) {
            ONB bounce_frame = ONB::from_normal(hit.shading_normal);
            float3 wo_local = bounce_frame.world_to_local(-direction);
            if (wo_local.z <= 0.f) break;

            BSDFSample bs = bsdf_sample(hit_mat, wo_local, hit.uv, rng);
            if (bs.pdf <= 0.f || bs.wi.z <= 0.f) break;

            float cos_theta_b = bs.wi.z;
            // Per-hero-channel throughput
            float max_throughput = 0.f;
            for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
                float throughput_h = bs.f.value[hero_bins[h]] * cos_theta_b / bs.pdf;
                hero_flux[h] *= throughput_h;
                max_throughput = fmaxf(max_throughput, throughput_h);
            }
            rr_albedo = fminf(max_throughput, 1.0f);

            direction = bounce_frame.local_to_world(bs.wi);
            origin = hit.position + hit.shading_normal * OPTIX_SCENE_EPSILON;
            on_caustic_path = false;
        } else {
            // Diffuse: cosine hemisphere sampling, per-hero throughput
            ONB bounce_frame = ONB::from_normal(hit.shading_normal);
            float3 wi_local = sample_cosine_hemisphere_dev(rng);
            Spectrum Kd = dev_get_Kd(hit_mat, hit.uv);
            float max_albedo = 0.f;
            for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
                float albedo_h = Kd.value[hero_bins[h]];
                hero_flux[h] *= albedo_h;
                max_albedo = fmaxf(max_albedo, albedo_h);
            }
            rr_albedo = max_albedo;

            direction = bounce_frame.local_to_world(wi_local);
            origin = hit.position + hit.shading_normal * OPTIX_SCENE_EPSILON;
            on_caustic_path = false;
        }

        // Russian roulette — skip for specular (glass/mirror) bounces so
        // that caustic photons survive the full glass path unattenuated.
        if (bounce >= DEFAULT_MIN_BOUNCES_RR && !dev_is_specular(hit_mat) && !dev_is_translucent(hit_mat)) {
            float p_rr = fminf(DEFAULT_RR_THRESHOLD, rr_albedo);
            if (rng.next_float() >= p_rr) break;
            for (int h = 0; h < HERO_WAVELENGTHS; ++h)
                hero_flux[h] /= p_rr;
        }
    }
}
