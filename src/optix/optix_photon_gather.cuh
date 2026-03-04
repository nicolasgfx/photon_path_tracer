#pragma once

// ─────────────────────────────────────────────────────────────────────
// optix_photon_gather.cuh — Photon density estimation at first camera hit
// ─────────────────────────────────────────────────────────────────────
// Launched ONCE per photon map (not per SPP).  For each pixel:
//   1. Trace camera ray through specular chain to first non-delta hit
//   2. Evaluate NEE direct lighting at the hitpoint
//   3. Gather nearby photons (diffuse + caustic) via density estimation
//   4. Write result into photon_gather_buffer
//
// Uses the dense grid for spatial lookup.  Fixed-radius gather with
// box kernel (upgrade to kNN if banding appears).
//
// Filters applied per photon:
//   - Normal compatibility:  dot(photon_normal, hit_normal) > 0
//   - Plane tau:            |dot(pos_diff, hit_normal)| < tau
//   - Tangential distance:  d_tan² < r²
//   - Wi hemisphere:        dot(photon_wi, hit_normal) > 0
//   (relaxed for delta hitpoints — no positive halfspace constraint)
//
// Caustic photons (is_caustic > 0) use a tighter gather radius.
// Non-caustic photons (is_caustic == 0) use the standard gather radius.
// ─────────────────────────────────────────────────────────────────────

// ── Tangential distance (inline, matches surface_filter.h) ──────────
__forceinline__ __device__
float dev_tangential_distance2(float3 pos, float3 photon_pos, float3 normal) {
    float3 diff = pos - photon_pos;
    float d_plane = dot(diff, normal);
    float3 tangent = diff - normal * d_plane;
    return dot(tangent, tangent);
}

// ── Density estimation at a surface point ───────────────────────────
// Gathers photons from the dense grid neighbourhood and accumulates:
//   L = Σ W(d²/r²) · f_s(wo, wi) · Φ(λ) / (N · π · r²)
//
// @param is_caustic_pass  true = only caustic photons, false = only non-caustic
// @param gather_r         gather radius
// @param n_emitted        N_emitted for normalisation (global or caustic budget)
__forceinline__ __device__
Spectrum dev_photon_density_estimate(
    float3      pos,
    float3      hit_normal,
    float3      wo_local,
    uint32_t    mat_id,
    float2      uv,
    float       gather_r,
    int         n_emitted,
    bool        is_caustic_pass,
    const ONB&  frame)
{
    Spectrum L = Spectrum::zero();
    if (!params.dense_valid || params.num_photons == 0 || n_emitted <= 0)
        return L;
    if (!params.photon_spectral_flux || !params.photon_is_caustic)
        return L;

    const float r2      = gather_r * gather_r;
    const float inv_N   = 1.0f / (float)n_emitted;
    const float inv_area = 1.0f / (PI * r2);  // box kernel normalisation
    const float tau     = DEFAULT_SURFACE_TAU;

    // Compute dense grid neighbourhood radius
    const int R = (int)ceilf(gather_r / params.dense_cell_size);
    const int R_clamped = min(R, 10);  // safety cap

    int cx = (int)floorf((pos.x - params.dense_min_x) / params.dense_cell_size);
    int cy = (int)floorf((pos.y - params.dense_min_y) / params.dense_cell_size);
    int cz = (int)floorf((pos.z - params.dense_min_z) / params.dense_cell_size);

    for (int dz = -R_clamped; dz <= R_clamped; ++dz) {
        int iz = cz + dz;
        if (iz < 0 || iz >= params.dense_dim_z) continue;
        for (int dy = -R_clamped; dy <= R_clamped; ++dy) {
            int iy = cy + dy;
            if (iy < 0 || iy >= params.dense_dim_y) continue;
            for (int dx = -R_clamped; dx <= R_clamped; ++dx) {
                int ix = cx + dx;
                if (ix < 0 || ix >= params.dense_dim_x) continue;
                int cell = ix + iy * params.dense_dim_x
                             + iz * params.dense_dim_x * params.dense_dim_y;
                uint32_t cs = params.dense_cell_start[cell];
                uint32_t ce = params.dense_cell_end[cell];

                for (uint32_t j = cs; j < ce; ++j) {
                    uint32_t idx = params.dense_sorted_indices[j];

                    // Caustic filter: only accept matching photon type
                    uint8_t ptag = params.photon_is_caustic[idx];
                    if (is_caustic_pass && ptag == 0) continue;    // want caustic, got non-caustic
                    if (!is_caustic_pass && ptag != 0) continue;   // want non-caustic, got caustic

                    // Normal compatibility gate
                    float3 pn = make_f3(
                        params.photon_norm_x[idx],
                        params.photon_norm_y[idx],
                        params.photon_norm_z[idx]);
                    if (dot(pn, hit_normal) <= 0.f) continue;

                    // Plane-distance (tau) gate
                    float3 pp = make_f3(
                        params.photon_pos_x[idx],
                        params.photon_pos_y[idx],
                        params.photon_pos_z[idx]);
                    if (fabsf(dot(pos - pp, hit_normal)) > tau) continue;

                    // Tangential distance gate
                    float d_tan2 = dev_tangential_distance2(pos, pp, hit_normal);
                    if (d_tan2 >= r2) continue;

                    // Wi hemisphere gate
                    float3 pw = make_f3(
                        params.photon_wi_x[idx],
                        params.photon_wi_y[idx],
                        params.photon_wi_z[idx]);
                    if (dot(pw, hit_normal) <= 0.f) continue;

                    // Evaluate BSDF at this photon's incoming direction
                    float3 wi_local = frame.world_to_local(pw);
                    Spectrum f = bsdf_evaluate(mat_id, wo_local, wi_local, uv);

                    // Read spectral flux
                    const int flux_base = idx * NUM_LAMBDA;
                    for (int b = 0; b < NUM_LAMBDA; ++b) {
                        float phi = params.photon_spectral_flux[flux_base + b];
                        L.value[b] += phi * f.value[b] * inv_N * inv_area;
                    }
                }
            }
        }
    }
    return L;
}

// ─────────────────────────────────────────────────────────────────────
// __raygen__photon_gather — Entry point
// ─────────────────────────────────────────────────────────────────────
extern "C" __global__ void __raygen__photon_gather()
{
    const uint3 idx = optixGetLaunchIndex();
    const int px = idx.x;
    const int py = idx.y;
    const int pixel_idx = py * params.width + px;

    // Seed RNG (deterministic per pixel for this pass)
    PCGRng rng = PCGRng::seed((uint64_t)pixel_idx, 0x50686F74ULL);  // "Phot" seed

    // Generate camera ray
    float3 origin, direction;
    generate_camera_ray_from_params(px, py, rng, origin, direction, /*sample_index=*/0);

    // ── Trace through specular chain to first non-delta hit ─────────
    Spectrum throughput = Spectrum::constant(1.0f);
    IORStack ior_stack;
    MediumStack medium_stack;

    const int max_spec = DEFAULT_MAX_SPECULAR_CHAIN;
    bool found_diffuse = false;
    TraceResult hit;

    for (int spec = 0; spec < max_spec; ++spec) {
        hit = trace_radiance(origin, direction);
        if (!hit.hit) return;  // miss — nothing to gather

        uint32_t mat_id = hit.material_id;

        // Skip emissive (direct lighting handled differently)
        if (dev_is_emissive(mat_id)) return;

        // Delta surface: bounce through
        if (dev_is_specular(mat_id) || dev_is_translucent(mat_id)) {
            SpecularBounceResult sb = dev_specular_bounce(
                direction, hit.position, hit.shading_normal, hit.geo_normal,
                mat_id, hit.uv, rng, nullptr, 0, &ior_stack,
                TransportMode::Radiance, &medium_stack);
            for (int i = 0; i < NUM_LAMBDA; ++i)
                throughput.value[i] *= sb.filter.value[i];
            direction = sb.new_dir;
            origin    = sb.new_pos;
            continue;
        }

        // Non-delta surface found
        found_diffuse = true;
        break;
    }
    if (!found_diffuse) return;

    // ── Set up local frame ──────────────────────────────────────────
    ONB frame = ONB::from_normal(hit.shading_normal);
    float3 wo_local = frame.world_to_local(direction * (-1.f));
    if (wo_local.z < 0.f) {
        hit.shading_normal = hit.shading_normal * (-1.f);
        frame.u = frame.u * (-1.f);
        frame.v = frame.v * (-1.f);
        frame.w = hit.shading_normal;
        wo_local = make_f3(-wo_local.x, -wo_local.y, -wo_local.z);
    }
    if (wo_local.z <= 0.f) return;

    uint32_t mat_id = hit.material_id;

    // ── NEE direct lighting ─────────────────────────────────────────
    Spectrum L_direct = Spectrum::zero();
    if (params.render_mode != RenderMode::IndirectOnly) {
        NeeResult nee = dev_nee_dispatch(
            hit.position, hit.shading_normal, wo_local,
            mat_id, rng, /*bounce=*/0, hit.uv);
        L_direct = nee.L;
    }

    // ── Diffuse photon gather (non-caustic, bigger radius) ──────────
    Spectrum L_diffuse = dev_photon_density_estimate(
        hit.position, hit.shading_normal, wo_local, mat_id, hit.uv,
        params.gather_radius, params.num_photons_emitted,
        /*is_caustic_pass=*/false, frame);

    // ── Caustic photon gather (caustic only, tighter radius) ────────
    int caustic_n = (params.num_caustic_emitted > 0)
                        ? params.num_caustic_emitted
                        : params.num_photons_emitted;
    Spectrum L_caustic = dev_photon_density_estimate(
        hit.position, hit.shading_normal, wo_local, mat_id, hit.uv,
        params.caustic_gather_radius, caustic_n,
        /*is_caustic_pass=*/true, frame);

    // ── Combine and write ───────────────────────────────────────────
    Spectrum L_total = Spectrum::zero();
    for (int b = 0; b < NUM_LAMBDA; ++b) {
        L_total.value[b] = throughput.value[b]
            * (L_direct.value[b] + L_diffuse.value[b] + L_caustic.value[b]);
    }

    // Write to photon_gather_buffer (not progressive — single write)
    if (params.photon_gather_buffer) {
        for (int b = 0; b < NUM_LAMBDA; ++b)
            params.photon_gather_buffer[pixel_idx * NUM_LAMBDA + b] = L_total.value[b];
    }
}
