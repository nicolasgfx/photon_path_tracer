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
    float   weight;         // accumulated shadow-ray-classified weight
    float3  centroid;       // weight-averaged direction centroid
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
    const DirMapFibSphere& fib,
    int pixel_idx = -1)  // pixel index for spectral_ref_buffer write (-1 = skip)
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

    // ── Phase 1: Max-heap kNN — find K closest photons ─────────────
    if (!params.dm_hash_valid || params.num_photons == 0) return entry;

    float3 pos = hit.position;
    float3 N   = hit.shading_normal;

    // Faceforward
    float3 wo = cur_dir * (-1.f);
    if (dot(wo, N) < 0.f) N = N * (-1.f);

    const float r_search  = params.guide_radius;
    const float r2_search = r_search * r_search;
    const float cell_size = params.dm_hash_cell_size;
    const uint32_t origin_tri = hit.triangle_id;

    constexpr int KNN_K = MAX_GUIDE_PDF_PHOTONS;   // 64
    float    knn_d2[KNN_K];
    uint32_t knn_idx[KNN_K];
    int      knn_count = 0;

    // ── Shell-expansion kNN: expand outward from center cell ────────
    // Instead of iterating the full bounding box (11³ = 1331 cells for
    // r_search/cell_size ≈ 5), iterate in concentric shells and stop
    // early when the K-th nearest photon is closer than the next shell.
    int cc_x = (int)floorf(pos.x / cell_size);
    int cc_y = (int)floorf(pos.y / cell_size);
    int cc_z = (int)floorf(pos.z / cell_size);

    // Fractional offset of pos within center cell (world units, [0, cell_size))
    float frac_x = pos.x - (float)cc_x * cell_size;
    float frac_y = pos.y - (float)cc_y * cell_size;
    float frac_z = pos.z - (float)cc_z * cell_size;

    const int geom_max = (int)ceilf(r_search / cell_size);
    const int max_layer = (geom_max < DM_KNN_MAX_LAYERS) ? geom_max : DM_KNN_MAX_LAYERS;

    // Visited-key deduplication
    constexpr int MAX_VISITED = 256;
    uint32_t visited_keys[MAX_VISITED];
    int num_visited = 0;

    for (int layer = 0; layer <= max_layer; ++layer) {
        for (int dz = -layer; dz <= layer; ++dz)
        for (int dy = -layer; dy <= layer; ++dy)
        for (int dx = -layer; dx <= layer; ++dx) {
            // Only process cells on the outer shell (Chebyshev dist == layer)
            if (layer > 0) {
                int ax = dx < 0 ? -dx : dx;
                int ay = dy < 0 ? -dy : dy;
                int az = dz < 0 ? -dz : dz;
                int chebyshev = ax;
                if (ay > chebyshev) chebyshev = ay;
                if (az > chebyshev) chebyshev = az;
                if (chebyshev < layer) continue;
            }

            uint32_t key = teschner_hash(
                make_i3(cc_x + dx, cc_y + dy, cc_z + dz),
                params.dm_hash_table_size);

            bool already = false;
            for (int v = 0; v < num_visited; ++v)
                if (visited_keys[v] == key) { already = true; break; }
            if (already) continue;
            if (num_visited < MAX_VISITED) visited_keys[num_visited++] = key;

            uint32_t cs = params.dm_hash_cell_start[key];
            uint32_t ce = params.dm_hash_cell_end[key];
            if (cs == 0xFFFFFFFFu) continue;

            for (uint32_t j = cs; j < ce; ++j) {
                uint32_t idx = params.dm_hash_sorted_indices[j];

                float3 pp = make_f3(
                    params.photon_pos_x[idx],
                    params.photon_pos_y[idx],
                    params.photon_pos_z[idx]);
                float3 diff = pos - pp;
                float d2 = dot(diff, diff);
                if (d2 > r2_search) continue;
                if (knn_count >= KNN_K && d2 >= knn_d2[0]) continue;

                // ── Insert into max-heap (sorted by largest d² at root) ──
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

        // Early termination: K-th nearest is closer than nearest face
        // of the next shell → no closer photon can exist beyond here.
        if (knn_count >= KNN_K && layer < max_layer) {
            float bd_x = fminf((float)(layer + 1) * cell_size - frac_x,
                               (float)layer * cell_size + frac_x);
            float bd_y = fminf((float)(layer + 1) * cell_size - frac_y,
                               (float)layer * cell_size + frac_y);
            float bd_z = fminf((float)(layer + 1) * cell_size - frac_z,
                               (float)layer * cell_size + frac_z);
            float min_boundary_d = fminf(bd_x, fminf(bd_y, bd_z));
            if (knn_d2[0] < min_boundary_d * min_boundary_d) break;
        }
    }

    // ── Phase 2: Shadow-ray filter + Epanechnikov-weighted histogram ──

    // Adaptive radius = distance to the K-th nearest (heap root)
    float r_k2 = (knn_count >= KNN_K) ? knn_d2[0] : r2_search;
    r_k2 = fmaxf(r_k2, 1e-12f);

    // Sort kNN heap by distance ascending so the B3 shadow-ray budget
    // evaluates the CLOSEST photons first.  This ensures:
    //   (a) best spatial locality — most relevant for local irradiance
    //   (b) uniform wavelength coverage across the evaluated subset
    // Simple insertion sort — KNN_K <= 64, fully in registers.
    for (int i = 1; i < knn_count; ++i) {
        float    key_d2  = knn_d2[i];
        uint32_t key_idx = knn_idx[i];
        int j = i - 1;
        while (j >= 0 && knn_d2[j] > key_d2) {
            knn_d2[j + 1]  = knn_d2[j];
            knn_idx[j + 1] = knn_idx[j];
            --j;
        }
        knn_d2[j + 1]  = key_d2;
        knn_idx[j + 1] = key_idx;
    }

    // Histogram bins (in registers / local memory)
    DirMapBin bins[DIR_MAP_SPHERE_BINS];
    for (int k = 0; k < DIR_MAP_SPHERE_BINS; ++k) {
        bins[k].weight   = 0.f;
        bins[k].centroid = make_f3(0.f, 0.f, 0.f);
        bins[k].count    = 0;
    }

    int n_eligible = 0;
    float total_weight = 0.f;

    // Spectral flux accumulation for reference buffer (visibility-filtered)
    float spec_irrad[NUM_LAMBDA] = {0.f, 0.f, 0.f, 0.f};
    float spec_weight_sum = 0.f;

    // B3: Cap shadow rays — closest photons evaluated first (sorted above)
    const int shadow_budget = (knn_count < MAX_DM_SHADOW_RAYS)
                              ? knn_count : MAX_DM_SHADOW_RAYS;
    for (int i = 0; i < shadow_budget; ++i) {
        uint32_t idx = knn_idx[i];
        float d2     = knn_d2[i];

        float3 pp = make_f3(
            params.photon_pos_x[idx],
            params.photon_pos_y[idx],
            params.photon_pos_z[idx]);

        float3 to_photon = pp - pos;
        float  dist      = sqrtf(d2);
        if (dist < 1e-6f) continue;  // degenerate: photon at hitpoint
        float3 ray_dir   = to_photon * (1.f / dist);

        // Shadow ray from hitpoint to photon (shorten slightly to
        // avoid self-intersection at t ≈ dist)
        ShadowMaterialResult sr = trace_shadow_material(
            pos, ray_dir, dist * 0.999f, origin_tri);

        float w = 0.f;  // weight for this photon (0 = rejected)

        if (!sr.hit) {
            w = DIR_MAP_DEFAULT_WEIGHT;
        } else {
            uint32_t sr_mat = sr.material_id;
            if (dev_is_emissive(sr_mat)) {
                continue;      // light source → reject
            } else if (dev_is_specular(sr_mat)) {
                w = DIR_MAP_DELTA_WEIGHT;
            } else if (dev_is_translucent(sr_mat)) {
                w = DIR_MAP_TRANSLUCENT_WEIGHT;
            } else {
                continue;      // non-delta diffuse → reject
            }
        }

        // Epanechnikov kernel: weight falls off smoothly to zero at r_k
        float epan = fmaxf(0.f, 1.f - d2 / r_k2);
        w *= epan;

        ++n_eligible;

        // Accumulate into Fibonacci bin using the photon's
        // incoming direction (wi)
        float3 pw = make_f3(
            params.photon_wi_x[idx],
            params.photon_wi_y[idx],
            params.photon_wi_z[idx]);

        int bin = fib.find_nearest(pw);
        bins[bin].weight   += w;
        bins[bin].centroid  = bins[bin].centroid + pw * w;
        bins[bin].count    += 1;
        total_weight       += w;

        // Accumulate spectral flux for outlier-clamp reference buffer
        if (pixel_idx >= 0 && params.photon_flux && params.photon_lambda) {
            for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
                int fi = idx * HERO_WAVELENGTHS + h;
                int lambda_bin = (int)params.photon_lambda[fi];
                if (lambda_bin >= 0 && lambda_bin < NUM_LAMBDA) {
                    spec_irrad[lambda_bin] += params.photon_flux[fi] * w;
                }
            }
            spec_weight_sum += w;
        }
    }

    entry.num_eligible = (uint16_t)n_eligible;

    // Write spectral reference buffer: normalise by πr_k² × N_emitted
    if (pixel_idx >= 0 && params.spectral_ref_buffer &&
        spec_weight_sum > 0.f && params.num_photons_emitted > 0) {
        float r_k = sqrtf(r_k2);
        float area = PI * r_k * r_k;
        float norm = 1.0f / (area * (float)params.num_photons_emitted);
        for (int b = 0; b < NUM_LAMBDA; ++b) {
            params.spectral_ref_buffer[pixel_idx * NUM_LAMBDA + b] =
                spec_irrad[b] * norm;
        }
    }

    // B5: Skip histogram sampling for sparse regions
    // (Spectral reference is already written above.)
    if (knn_count < MIN_GUIDE_PHOTONS) return entry;

    if (n_eligible == 0 || total_weight <= 0.f) return entry;

    // ── Sample one direction from the histogram ─────────────────────
    // Discrete CDF sampling over non-empty bins.
    float u_sample = rng.next_float() * total_weight;
    float cumulative = 0.f;
    int chosen_bin = DIR_MAP_SPHERE_BINS - 1;  // default to last bin (float safety)

    for (int k = 0; k < DIR_MAP_SPHERE_BINS; ++k) {
        cumulative += bins[k].weight;
        if (cumulative >= u_sample) {
            chosen_bin = k;
            break;
        }
    }

    // Direction: use the flux-weighted centroid of the chosen bin
    // (or the Fibonacci direction if the centroid is degenerate).
    // This is the cone axis for the jitter step below.
    float3 cone_axis;
    if (bins[chosen_bin].count > 0 && length(bins[chosen_bin].centroid) > 1e-6f) {
        cone_axis = normalize(bins[chosen_bin].centroid);
    } else {
        cone_axis = fib.dirs[chosen_bin];
    }

    // Apply cone jitter for smoothness
    float3 wi_dir = cone_axis;
    float cos_half = cosf(DEFAULT_PHOTON_GUIDE_CONE_HALF_ANGLE);
    if (cos_half < 1.f - 1e-6f) {
        ONB cone_frame = ONB::from_normal(cone_axis);
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
    // pdf = weight_of_chosen_bin / total_weight × cone_pdf
    // The cone_pdf must be evaluated relative to the actual cone axis
    // (centroid-based), not the Fibonacci bin center.
    float bin_prob = bins[chosen_bin].weight / total_weight;
    float cone_pdf_val;
    if (cos_half < 1.f - 1e-6f) {
        // Cosine-cone PDF: angle between jittered direction and cone axis
        float cos_theta = dot(wi_dir, cone_axis);
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

    // Compute pixel index for spectral reference buffer
    int factor = DIR_MAP_SUBPIXEL_FACTOR;
    int px = sub_x / factor;
    int py = sub_y / factor;
    int pixel_idx = py * params.width + px;

    DirMapEntry entry = dev_build_direction_map_entry(
        sub_x, sub_y, sub_width, sub_height, spp_seed, fib, pixel_idx);

    int flat = sub_y * sub_width + sub_x;
    dir_map_buffer[flat] = entry;
}
