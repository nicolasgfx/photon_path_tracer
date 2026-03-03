#pragma once

// optix_targeted_photon.cuh – Targeted caustic emission (GPU)
//   Jensen §9.2: importance-sample emission toward specular geometry.

// =====================================================================
// __raygen__targeted_photon_trace  -  Targeted caustic emission (GPU)
//   Jensen §9.2: importance-sample emission toward specular geometry.
//   1. Pick emitter (power CDF + uniform mix)
//   2. Pick specular triangle (area-weighted alias table)
//   3. Sample point on specular triangle
//   4. Compute direction from light → target, visibility check (shadow ray)
//   5. Compute hero-wavelength flux with PDF correction
//   6. Trace through scene (identical bounce loop to standard photon trace)
// =====================================================================
extern "C" __global__ void __raygen__targeted_photon_trace() {
    const uint3 idx = optixGetLaunchIndex();
    int photon_idx = idx.x;
    if (photon_idx >= params.num_photons) return;
    if (params.num_emissive <= 0) return;
    if (params.num_targeted_spec_tris <= 0) return;

    // Decorrelated RNG (different constant from standard trace)
    PCGRng rng = PCGRng::seed(
        (uint64_t)photon_idx * 13 + 0xCA051CULL + (uint64_t)params.photon_map_seed * 0x100000007ULL,
        (uint64_t)photon_idx + 7);

    // ── 1. Pick emitter triangle (power CDF) ──
    int local_idx = 0;
    {
        float xi = rng.next_float();
        local_idx = binary_search_cdf(params.emissive_cdf, params.num_emissive, xi);
        if (local_idx >= params.num_emissive) local_idx = params.num_emissive - 1;
    }
    uint32_t emit_tri = params.emissive_tri_indices[local_idx];

    float pdf_emitter;
    if (local_idx == 0) pdf_emitter = params.emissive_cdf[0];
    else pdf_emitter = params.emissive_cdf[local_idx] - params.emissive_cdf[local_idx - 1];

    // Get emitter triangle geometry
    float3 ev0 = params.vertices[emit_tri * 3 + 0];
    float3 ev1 = params.vertices[emit_tri * 3 + 1];
    float3 ev2 = params.vertices[emit_tri * 3 + 2];
    uint32_t emit_mat = params.material_ids[emit_tri];

    // Sample point on emitter
    float3 ebary = sample_triangle_dev(rng.next_float(), rng.next_float());
    float3 light_pos = ev0 * ebary.x + ev1 * ebary.y + ev2 * ebary.z;
    float3 ee1 = ev1 - ev0;
    float3 ee2 = ev2 - ev0;
    float3 light_normal = normalize(cross(ee1, ee2));
    float  light_area   = length(cross(ee1, ee2)) * 0.5f;

    // Get Le
    float2 euv0 = params.texcoords[emit_tri * 3 + 0];
    float2 euv1 = params.texcoords[emit_tri * 3 + 1];
    float2 euv2 = params.texcoords[emit_tri * 3 + 2];
    float2 emit_uv = make_float2(
        euv0.x * ebary.x + euv1.x * ebary.y + euv2.x * ebary.z,
        euv0.y * ebary.x + euv1.y * ebary.y + euv2.y * ebary.z);
    Spectrum Le = dev_get_Le(emit_mat, emit_uv);
    float Le_sum = 0.f;
    for (int i = 0; i < NUM_LAMBDA; ++i) Le_sum += Le.value[i];
    if (Le_sum <= 0.f) return;

    // ── 2. Pick specular triangle (area-weighted alias table) ────────
    float s1 = rng.next_float();
    float s2 = rng.next_float();
    int spec_n = params.num_targeted_spec_tris;
    int spec_local = (int)(s1 * spec_n);
    if (spec_local >= spec_n) spec_local = spec_n - 1;
    if (s2 >= params.targeted_spec_alias_prob[spec_local])
        spec_local = (int)params.targeted_spec_alias_idx[spec_local];
    float pdf_spec_tri = params.targeted_spec_pdf[spec_local];
    float spec_area    = params.targeted_spec_areas[spec_local];
    uint32_t spec_global = params.targeted_spec_tri_indices[spec_local];

    // ── 3. Sample point on specular triangle ─────────────────────────
    float3 sv0 = params.vertices[spec_global * 3 + 0];
    float3 sv1 = params.vertices[spec_global * 3 + 1];
    float3 sv2 = params.vertices[spec_global * 3 + 2];
    float3 sbary = sample_triangle_dev(rng.next_float(), rng.next_float());
    float3 target_pos = sv0 * sbary.x + sv1 * sbary.y + sv2 * sbary.z;
    float3 se1 = sv1 - sv0;
    float3 se2 = sv2 - sv0;
    float3 spec_normal = normalize(cross(se1, se2));
    float  pdf_on_tri  = 1.0f / fmaxf(spec_area, 1e-20f);

    // ── 4. Direction from light → target + visibility ────────────────
    float3 to_target = target_pos - light_pos;
    float  dist_sq   = dot(to_target, to_target);
    float  dist      = sqrtf(dist_sq);
    if (dist < OPTIX_SCENE_EPSILON) return;
    float3 dir = to_target / dist;

    // Light-side cosine: photon must leave from the front
    float cos_light = dot(dir, light_normal);
    if (cos_light <= 0.f) return;

    // Visibility: trace a radiance ray instead of a binary shadow test.
    // If the first hit is specular/translucent (e.g. a wave-peak on the
    // water mesh, or the front hemisphere of a glass sphere when targeting
    // the back), the photon is NOT rejected — the bounce loop will enter
    // the specular geometry at that hit point.  Only truly opaque
    // blockers cause rejection.
    float3 shadow_origin = light_pos + light_normal * OPTIX_SCENE_EPSILON;
    TraceResult vis_hit = trace_radiance(shadow_origin, dir);
    if (!vis_hit.hit) return;                                       // missed everything
    uint32_t vis_mat = vis_hit.material_id;
    if (dev_is_emissive(vis_mat)) return;                            // hit a light source
    if (!dev_is_specular(vis_mat) && !dev_is_translucent(vis_mat))   // opaque blocker
        return;

    // Target-side cosine (for area→solid angle Jacobian).
    // Reject back-facing sampled targets: the PDF is only valid when the
    // photon approaches the target triangle from the front.
    float cos_target = dot(dir * (-1.f), spec_normal);
    if (cos_target < 1e-6f) return;

    // ── 5. Hero wavelength setup + flux with PDF correction ──────────
    // Sample primary hero wavelength from Le CDF
    float xi_lambda = rng.next_float() * Le_sum;
    int hero_bin = NUM_LAMBDA - 1;
    float cum = 0.f;
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        cum += Le.value[i];
        if (xi_lambda <= cum) { hero_bin = i; break; }
    }

    int   hero_bins[HERO_WAVELENGTHS];
    float hero_flux[HERO_WAVELENGTHS];
    int   num_hero = HERO_WAVELENGTHS;

    hero_bins[0] = hero_bin;
    for (int h = 1; h < HERO_WAVELENGTHS; ++h) {
        int offset = (h * NUM_LAMBDA) / HERO_WAVELENGTHS;
        hero_bins[h] = (hero_bin + offset) % NUM_LAMBDA;
    }

    // PDF of target direction in solid angle (from the light, shared helper):
    float pdf_target_sa = nee_pdf_area_to_solid_angle(pdf_spec_tri, pdf_on_tri, dist_sq, cos_target);
    if (pdf_target_sa <= 0.f) return;

    // Flux: Φ(λ_h) = Le(λ_h) * cos_light * light_area / (pdf_emitter * pdf_target_sa * pdf_lambda_h)
    //   The light_area accounts for pdf_pos_on_emitter = 1/light_area (uniform
    //   point sampling on the chosen emitter triangle).  Must match the CPU
    //   formula in specular_target.h: scale = cos_light * A / (pdf_emitter * pdf_target_sa).
    //   HERO_WAVELENGTHS normalisation (PBRT v4 §14.3)
    float denom_common = pdf_emitter * pdf_target_sa;
    float inv_hero = 1.0f / (float)HERO_WAVELENGTHS;
    for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
        int bin = hero_bins[h];
        float Le_h = Le.value[bin];
        float pdf_lambda_h = Le_h / Le_sum;
        hero_flux[h] = (denom_common * pdf_lambda_h > 0.f)
                     ? (Le_h * cos_light * light_area) / (denom_common * pdf_lambda_h) * inv_hero
                     : 0.f;
    }

    // Clamp to prevent fireflies from near-miss geometry
    for (int h = 0; h < HERO_WAVELENGTHS; ++h)
        hero_flux[h] = fminf(hero_flux[h], 1e6f);

    // ── 6. Trace through scene (identical bounce loop) ───────────────
    float3 origin    = shadow_origin;
    float3 direction = dir;
    bool on_caustic_path = false;  // becomes true after first specular hit
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

        // ── Per-material interior medium Beer-Lambert (§7.7) ────────
        {
            int cur_mid = medium_stack.current_medium_id();
            if (cur_mid >= 0 && params.media) {
                HomogeneousMedium med = dev_get_medium(cur_mid);
                float seg_t = hit.t;
                for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
                    float sig_t_h = med.sigma_t.value[hero_bins[h]];
                    hero_flux[h] *= expf(-sig_t_h * seg_t);
                }
            }
        }

        // Skip emissive surfaces
        if (dev_is_emissive(hit_mat)) break;

        // Store photon at diffuse surfaces (only if on caustic path)
        if (!dev_is_specular(hit_mat) && !dev_is_translucent(hit_mat) && bounce > 0) {
            if (on_caustic_path) {
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
                    for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
                        params.out_photon_lambda[slot * HERO_WAVELENGTHS + h] = (uint16_t)hero_bins[h];
                        params.out_photon_flux[slot * HERO_WAVELENGTHS + h]   = hero_flux[h];
                    }
                    params.out_photon_num_hero[slot] = (uint8_t)num_hero;
                    params.out_photon_source_emissive[slot] = (uint16_t)local_idx;
                    if (params.out_photon_is_caustic)
                        params.out_photon_is_caustic[slot] = (uint8_t)1;
                    if (params.out_photon_path_flags)
                        params.out_photon_path_flags[slot] = path_flags;
                    if (params.out_photon_tri_id)
                        params.out_photon_tri_id[slot] = hit.triangle_id;
                }
            }
            // Caustic path ends at diffuse surface — done
            break;
        }

        // Specular/translucent hit: mark caustic path and bounce
        if (dev_is_specular(hit_mat) || dev_is_translucent(hit_mat)) {
            on_caustic_path = true;
            // Track detailed path flags for F2 debug overlay
            // Must distinguish Glass/Translucent (glass flags) from Mirror (specular flag).
            // Matches CPU emitter.h flag logic.
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
            for (int h = 0; h < HERO_WAVELENGTHS; ++h)
                hero_flux[h] *= sb.filter.value[hero_bins[h]];
            direction = sb.new_dir;
            origin    = sb.new_pos;
            continue;  // skip RR for specular bounces
        }

        // Glossy bounce (non-caustic after first diffuse interaction)
        if (dev_is_any_glossy(hit_mat)) {
            ONB bounce_frame = ONB::from_normal(hit.shading_normal);
            float3 wo_local = bounce_frame.world_to_local(-direction);
            if (wo_local.z <= 0.f) break;
            BSDFSample bs = bsdf_sample(hit_mat, wo_local, hit.uv, rng);
            if (bs.pdf <= 0.f || bs.wi.z <= 0.f) break;
            for (int h = 0; h < HERO_WAVELENGTHS; ++h)
                hero_flux[h] *= bs.f.value[hero_bins[h]] * bs.wi.z / bs.pdf;
            direction = bounce_frame.local_to_world(bs.wi);
            origin = hit.position + hit.shading_normal * OPTIX_SCENE_EPSILON;
            on_caustic_path = false;
        } else {
            // Diffuse bounce
            ONB bounce_frame = ONB::from_normal(hit.shading_normal);
            float3 wi_local = sample_cosine_hemisphere_dev(rng);
            Spectrum Kd = dev_get_Kd(hit_mat, hit.uv);
            float rr_albedo = 0.f;
            for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
                float albedo_h = Kd.value[hero_bins[h]];
                hero_flux[h] *= albedo_h;
                rr_albedo = fmaxf(rr_albedo, albedo_h);
            }
            direction = bounce_frame.local_to_world(wi_local);
            origin = hit.position + hit.shading_normal * OPTIX_SCENE_EPSILON;
            on_caustic_path = false;
        }

        // Russian roulette (skip for specular, already handled via continue)
        if (bounce >= DEFAULT_MIN_BOUNCES_RR) {
            float max_flux = 0.f;
            for (int h = 0; h < HERO_WAVELENGTHS; ++h)
                max_flux = fmaxf(max_flux, hero_flux[h]);
            float p_rr = fminf(DEFAULT_RR_THRESHOLD, fminf(max_flux, 1.0f));
            if (rng.next_float() >= p_rr) break;
            for (int h = 0; h < HERO_WAVELENGTHS; ++h)
                hero_flux[h] /= p_rr;
        }
    }
}
