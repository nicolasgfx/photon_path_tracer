#pragma once
// ─────────────────────────────────────────────────────────────────────
// optix_direction_map.cuh — Direction map build kernel (v2: hash grid + shadow rays)
// ─────────────────────────────────────────────────────────────────────
// Launched as an OptiX raygen program after photon tracing to
// precompute the directional guidance framebuffer (1:1 with pixels).
//
// For each pixel:
//   1. Cast a primary ray → find 1st hitpoint.
//   2. If delta material, follow specular chain to first non-delta hit.
//   3. Gather nearby photons via hash grid kNN (Teschner spatial hash).
//   4. For each candidate photon, trace a shadow ray from the hitpoint.
//      Accept/reject based on material of the first intersection:
//        - miss               → accept  (w = DIR_MAP_DEFAULT_WEIGHT)
//        - delta (glass/mirror)→ accept  (w = DIR_MAP_DELTA_WEIGHT)
//        - translucent        → accept  (w = DIR_MAP_TRANSLUCENT_WEIGHT)
//        - emissive           → reject
//        - non-delta diffuse  → reject
//   5. Build a 128-bin Fibonacci sphere histogram from accepted photons.
//   6. Sample one direction from the histogram using RNG.
//   7. Compute the marginal guide PDF at that direction.
//   8. Write DirMapEntry to device buffer.
// ─────────────────────────────────────────────────────────────────────

#include "photon/direction_map.h"

// ── Fibonacci sphere for direction map (128 bins) ───────────────────
// Larger than DevPhotonBinDirs (which caps at MAX_PHOTON_BIN_COUNT=64).
// This one uses DIR_MAP_SPHERE_BINS from config.h (128).

struct DirMapFibSphere {
    float3 dirs[DIR_MAP_SPHERE_BINS];

    __forceinline__ __device__
    void init() {
        const float golden_angle = PI * (3.0f - sqrtf(5.0f));
        for (int k = 0; k < DIR_MAP_SPHERE_BINS; ++k) {
            float theta = acosf(1.0f - 2.0f * (k + 0.5f) / (float)DIR_MAP_SPHERE_BINS);
            float phi   = golden_angle * k;
            dirs[k] = make_f3(
                sinf(theta) * cosf(phi),
                sinf(theta) * sinf(phi),
                cosf(theta));
        }
    }

    __forceinline__ __device__
    int find_nearest(float3 wi) const {
        int   best     = 0;
        float best_dot = -2.0f;
        for (int k = 0; k < DIR_MAP_SPHERE_BINS; ++k) {
            float d = dot(wi, dirs[k]);
            if (d > best_dot) { best_dot = d; best = k; }
        }
        return best;
    }
};

// ── Direction map histogram bin (per-subpixel, in registers) ────────
struct DirMapBin {
    float   weight;         // accumulated Epanechnikov-weighted flux
    float3  centroid;       // flux-weighted direction centroid
    int     count;          // number of photons in this bin
};

// ── Build direction map for one subpixel ────────────────────────────
// Called from __raygen__render (or a separate CUDA kernel launch).
// Writes one DirMapEntry to the output buffer.

__forceinline__ __device__
DirMapEntry dev_build_direction_map_entry(
    int sub_x, int sub_y,
    int sub_width, int sub_height,
    int spp_seed,   // varies per SPP to get different sampled direction
    const DirMapFibSphere& fib)
{
    DirMapEntry entry = {};

    // Map subpixel to pixel + subpixel jitter
    int factor = DIR_MAP_SUBPIXEL_FACTOR;
    int px = sub_x / factor;
    int py = sub_y / factor;
    float dx = ((float)(sub_x % factor) + 0.5f) / (float)factor;
    float dy = ((float)(sub_y % factor) + 0.5f) / (float)factor;

    // RNG seeded per-subpixel + per-SPP
    PCGRng rng = PCGRng::seed(
        (uint64_t)(sub_y * sub_width + sub_x) * 1000ULL + (uint64_t)spp_seed * 100000ULL,
        (uint64_t)(sub_y * sub_width + sub_x) + 1ULL);

    // Generate camera ray with subpixel offset
    float u = ((float)px + dx) / (float)params.width;
    float v = ((float)py + dy) / (float)params.height;
    float3 origin = params.cam_pos;
    float3 direction = normalize(
        params.cam_lower_left
        + params.cam_horizontal * u
        + params.cam_vertical * v
        - params.cam_pos);

    // ── Trace primary ray ─────────────────────────────────────────
    TraceResult hit = trace_radiance(origin, direction);
    if (!hit.hit) return entry;

    // ── Follow specular chain for delta materials ─────────────────
    float3 cur_pos = origin;
    float3 cur_dir = direction;
    IORStack ior_stack;
    MediumStack medium_stack;

    for (int spec = 0; spec < DEFAULT_MAX_SPECULAR_CHAIN; ++spec) {
        if (!hit.hit) return entry;

        uint32_t mat_id = hit.material_id;

        // If non-delta, we found our hitpoint
        if (!dev_is_specular(mat_id) && !dev_is_translucent(mat_id))
            break;

        // Delta: follow the specular chain
        SpecularBounceResult sb = dev_specular_bounce(
            cur_dir, hit.position, hit.shading_normal, hit.geo_normal,
            mat_id, hit.uv, rng, nullptr, 0, &ior_stack,
            TransportMode::Radiance, &medium_stack);
        cur_dir = sb.new_dir;
        cur_pos = sb.new_pos;

        hit = trace_radiance(cur_pos, cur_dir);
    }

    if (!hit.hit) return entry;
    if (dev_is_emissive(hit.material_id)) return entry;  // hit a light, no guidance needed
    if (dev_is_specular(hit.material_id) || dev_is_translucent(hit.material_id))
        return entry;  // still on a delta surface after max chain

    // Store hitpoint info
    entry.hit_x  = hit.position.x;
    entry.hit_y  = hit.position.y;
    entry.hit_z  = hit.position.z;
    entry.norm_x = hit.shading_normal.x;
    entry.norm_y = hit.shading_normal.y;
    entry.norm_z = hit.shading_normal.z;
    entry.mat_type = params.mat_type[hit.material_id];

    // ── Hash grid photon gathering + shadow ray filtering ───────────
    if (!params.dm_hash_valid || params.num_photons == 0) return entry;

    float3 pos = hit.position;
    float3 N   = hit.shading_normal;

    // Faceforward
    float3 wo = cur_dir * (-1.f);
    if (dot(wo, N) < 0.f) N = N * (-1.f);

    const float r_search = params.guide_radius;
    const float r2_search = r_search * r_search;
    const float cell_size = params.dm_hash_cell_size;
    const uint32_t origin_tri = hit.triangle_id;

    // Compute cell range covering the search sphere
    int cx0 = (int)floorf((pos.x - r_search) / cell_size);
    int cy0 = (int)floorf((pos.y - r_search) / cell_size);
    int cz0 = (int)floorf((pos.z - r_search) / cell_size);
    int cx1 = (int)floorf((pos.x + r_search) / cell_size);
    int cy1 = (int)floorf((pos.y + r_search) / cell_size);
    int cz1 = (int)floorf((pos.z + r_search) / cell_size);

    // Visited-key deduplication (max 27 cells in 3×3×3)
    uint32_t visited_keys[27];
    int num_visited = 0;

    // Histogram bins (in registers / local memory)
    DirMapBin bins[DIR_MAP_SPHERE_BINS];
    for (int k = 0; k < DIR_MAP_SPHERE_BINS; ++k) {
        bins[k].weight   = 0.f;
        bins[k].centroid = make_f3(0.f, 0.f, 0.f);
        bins[k].count    = 0;
    }

    int n_eligible = 0;
    float total_weight = 0.f;

    for (int iz = cz0; iz <= cz1; ++iz)
    for (int iy = cy0; iy <= cy1; ++iy)
    for (int ix = cx0; ix <= cx1; ++ix) {
        uint32_t key = teschner_hash(make_i3(ix, iy, iz),
                                     params.dm_hash_table_size);
        // Deduplicate buckets (different cells can hash to same bucket)
        bool already = false;
        for (int v = 0; v < num_visited; ++v)
            if (visited_keys[v] == key) { already = true; break; }
        if (already) continue;
        if (num_visited < 27) visited_keys[num_visited++] = key;

        uint32_t cs = params.dm_hash_cell_start[key];
        uint32_t ce = params.dm_hash_cell_end[key];
        if (cs == 0xFFFFFFFFu) continue;

        for (uint32_t j = cs; j < ce; ++j) {
            uint32_t idx = params.dm_hash_sorted_indices[j];

            // ── Radius gate (3D Euclidean) ──
            float3 pp = make_f3(
                params.photon_pos_x[idx],
                params.photon_pos_y[idx],
                params.photon_pos_z[idx]);
            float3 diff = pos - pp;
            float d2 = dot(diff, diff);
            if (d2 > r2_search) continue;

            // ── Shadow ray from hitpoint to photon ──
            float3 to_photon = pp - pos;
            float  dist      = sqrtf(d2);
            if (dist < 1e-6f) continue;  // degenerate: photon at hitpoint
            float3 ray_dir   = to_photon * (1.f / dist);

            // Shorten max distance slightly so we don't hit the
            // photon's own surface triangle at t ≈ dist.
            float shadow_dist = dist * 0.999f;

            ShadowMaterialResult sr = trace_shadow_material(
                pos, ray_dir, shadow_dist, origin_tri);

            float w = 0.f;  // weight for this photon (0 = rejected)

            if (!sr.hit) {
                // Miss: no geometry between hitpoint and photon → accept
                w = DIR_MAP_DEFAULT_WEIGHT;
            } else {
                // Hit something — classify by material type
                uint32_t sr_mat = sr.material_id;

                if (dev_is_emissive(sr_mat)) {
                    // Light source → reject
                    continue;
                } else if (dev_is_specular(sr_mat)) {
                    // Delta material (glass/mirror) → accept with boost
                    w = DIR_MAP_DELTA_WEIGHT;
                } else if (dev_is_translucent(sr_mat)) {
                    // Translucent material → accept with boost
                    w = DIR_MAP_TRANSLUCENT_WEIGHT;
                } else {
                    // Non-delta diffuse surface → reject
                    continue;
                }
            }

            ++n_eligible;

            // Accumulate into Fibonacci bin using the photon's
            // incoming direction (wi)
            float3 pw = make_f3(
                params.photon_wi_x[idx],
                params.photon_wi_y[idx],
                params.photon_wi_z[idx]);

            int bin = fib.find_nearest(pw);
            bins[bin].weight += w;
            bins[bin].centroid = bins[bin].centroid + pw * w;
            bins[bin].count += 1;
            total_weight += w;

            // Cap eligible count
            if (n_eligible >= MAX_GUIDE_PDF_PHOTONS) goto gather_done;
        }
    }
gather_done:

    entry.num_eligible = (uint16_t)n_eligible;

    if (n_eligible == 0 || total_weight <= 0.f) return entry;

    // ── Sample one direction from the histogram ─────────────────────
    // Discrete CDF sampling over non-empty bins.
    float u_sample = rng.next_float() * total_weight;
    float cumulative = 0.f;
    int chosen_bin = 0;

    for (int k = 0; k < DIR_MAP_SPHERE_BINS; ++k) {
        cumulative += bins[k].weight;
        if (cumulative >= u_sample) {
            chosen_bin = k;
            break;
        }
    }

    // Direction: use the flux-weighted centroid of the chosen bin
    // (or the Fibonacci direction if the centroid is degenerate)
    float3 wi_dir;
    if (bins[chosen_bin].count > 0 && length(bins[chosen_bin].centroid) > 1e-6f) {
        wi_dir = normalize(bins[chosen_bin].centroid);
    } else {
        wi_dir = fib.dirs[chosen_bin];
    }

    // Apply cone jitter for smoothness
    float cos_half = cosf(DEFAULT_PHOTON_GUIDE_CONE_HALF_ANGLE);
    if (cos_half < 1.f - 1e-6f) {
        ONB cone_frame = ONB::from_normal(wi_dir);
        float3 cone_local = sample_cosine_cone(
            rng.next_float(), rng.next_float(), cos_half);
        wi_dir = cone_frame.local_to_world(cone_local);
    }

    // Ensure wi_dir is in the correct hemisphere
    if (dot(wi_dir, N) <= 0.f) {
        // Reflect across the surface plane
        wi_dir = wi_dir - N * (2.f * dot(wi_dir, N));
    }

    entry.dir_x = wi_dir.x;
    entry.dir_y = wi_dir.y;
    entry.dir_z = wi_dir.z;

    // ── Compute marginal guide PDF ──────────────────────────────────
    // pdf = weight_of_chosen_bin / total_weight × cone_pdf (or 1/(2π) if no jitter)
    float bin_prob = bins[chosen_bin].weight / total_weight;
    float cone_pdf_val;
    if (cos_half < 1.f - 1e-6f) {
        // Cosine-cone PDF at the jittered direction (matches sample_cosine_cone)
        float cos_theta = dot(wi_dir, fib.dirs[chosen_bin]);
        cone_pdf_val = cosine_cone_pdf(cos_theta, cos_half);
    } else {
        // No jitter: Dirac-like — use 1/(2π) as a stand-in
        cone_pdf_val = 1.f / (2.f * PI);
    }

    entry.pdf = bin_prob * cone_pdf_val;
    // Safety: clamp PDF to avoid division by zero later
    if (entry.pdf < 1e-10f) entry.pdf = 0.f;

    return entry;
}

// ── CUDA kernel: build the full direction map ───────────────────────
// Launched as a CUDA kernel (not OptiX raygen), using the same
// LaunchParams structure already on the device.
//
// This is actually called from within __raygen__render as a setup pass,
// or via a dedicated OptiX raygen program.  We implement it as an
// inline device function that the raygen program calls.

__forceinline__ __device__
void dev_build_direction_map_pixel(
    DirMapEntry* dir_map_buffer,
    int sub_width, int sub_height,
    int spp_seed)
{
    const uint3 idx = optixGetLaunchIndex();
    int sub_x = idx.x;
    int sub_y = idx.y;

    if (sub_x >= sub_width || sub_y >= sub_height) return;

    // Build Fibonacci sphere (in registers, once per thread)
    DirMapFibSphere fib;
    fib.init();

    DirMapEntry entry = dev_build_direction_map_entry(
        sub_x, sub_y, sub_width, sub_height, spp_seed, fib);

    int flat = sub_y * sub_width + sub_x;
    dir_map_buffer[flat] = entry;
}
