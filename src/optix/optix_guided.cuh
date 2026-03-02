#pragma once

// optix_guided.cuh — §4 Photon-guided direction sampling on GPU
//
// At each camera bounce, a per-hitpoint kNN query of the photon hash grid
// builds a local directional histogram (Fibonacci sphere, 32 bins).
// The guide direction is MIS-combined with the standard BSDF sample
// using the balance heuristic.
//
// Build phase:  Hash grid already built by photon trace (hash_grid.h).
// Query phase:  dev_knn_guide_sample() — per-hitpoint kNN from hash grid.
// Sample phase: dev_sample_guided_direction() — inverse-CDF over bins.

// ── Fibonacci sphere bin directions (device) ────────────────────────
// Pre-compute once and store in a small local array.
// Matches PhotonBinDirs::init(n) on CPU.

struct DevPhotonBinDirs {
    float3 dirs[MAX_PHOTON_BIN_COUNT];
    int    count;

    __forceinline__ __device__
    void init(int n) {
        count = (n > MAX_PHOTON_BIN_COUNT) ? MAX_PHOTON_BIN_COUNT : n;
        const float golden_angle = PI * (3.0f - sqrtf(5.0f));
        for (int k = 0; k < count; ++k) {
            float theta = acosf(1.0f - 2.0f * (k + 0.5f) / (float)count);
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
        for (int k = 0; k < count; ++k) {
            float d = dot(wi, dirs[k]);
            if (d > best_dot) { best_dot = d; best = k; }
        }
        return best;
    }
};

// ── Approximate solid angle per Fibonacci bin ───────────────────────
// For N quasi-uniform bins on S²: Ω_bin ≈ 4π/N.
// This is exact in the limit; sufficiently accurate for MIS with N=32.

__forceinline__ __device__
float bin_solid_angle(int num_bins) {
    return (4.0f * PI) / (float)num_bins;
}

// ── Cell lookup ─────────────────────────────────────────────────────
// Dense-grid lookup is retained only for the volume photon guide.

// ── Cell cache lookup: world position → hash-table index ────────────
// Mirrors CellInfoCache::cell_key() — Teschner spatial hash into a
// fixed 65K table (CELL_CACHE_TABLE_SIZE).  Used for cell_guide_fraction,
// cell_caustic_fraction, and cell_flux_density lookups which are indexed
// by hash bucket, not by dense grid index.

__forceinline__ __device__
uint32_t dev_cell_cache_index(float3 pos) {
    float cs = params.grid_cell_size;  // gather_radius * 2 (same as CellInfoCache)
    int3 cell = make_i3(
        (int)floorf(pos.x / cs),
        (int)floorf(pos.y / cs),
        (int)floorf(pos.z / cs));
    return teschner_hash(cell, CELL_CACHE_TABLE_SIZE);
}

// ── Guided histogram: directional bin data for guide sampling ────────

struct GuidedHistogram {
    float bin_flux[MAX_PHOTON_BIN_COUNT];   // scalar_flux per bin
    float total_flux;                        // sum over all bins
    int   num_bins;
    bool  valid;                             // true if any data found
};

// =====================================================================
// dev_knn_guide_sample — Per-hitpoint kNN guided direction histogram
//
// Replaces the pre-aggregated HashHistogram with an exact per-hitpoint
// kNN query.  Walks the hash grid, collects the GUIDE_KNN_K nearest
// photons (tangential-disk metric), and bins their incident directions
// into Fibonacci bins weighted by scalar flux.
//
// The result is a GuidedHistogram identical in layout to the old
// dev_read_cell_histogram() output — consumed by dev_sample_guided_direction()
// and dev_guided_pdf() without changes.
//
// Cost: ~same as dev_estimate_caustic_only() but with smaller K.
// =====================================================================

constexpr int GUIDE_KNN_K = 32;   // neighbours for guide histogram

__forceinline__ __device__
GuidedHistogram dev_knn_guide_sample(
    float3 pos, float3 normal,
    float3 filter_normal,
    const DevPhotonBinDirs& fib)
{
    GuidedHistogram h;
    h.num_bins   = fib.count;
    h.total_flux = 0.f;
    h.valid      = false;
    for (int k = 0; k < h.num_bins; ++k)
        h.bin_flux[k] = 0.f;

    if (params.num_photons == 0 || params.grid_table_size == 0)
        return h;

    // Search radius: use the main gather radius (kNN will tighten adaptively)
    float radius = params.gather_radius;
    float r2_max = radius * radius;

    // ── Phase 1: kNN candidate collection via hash grid ──────────────
    float    knn_d2[GUIDE_KNN_K];
    uint32_t knn_idx[GUIDE_KNN_K];
    int      knn_count = 0;

    float cell_size = params.grid_cell_size;
    int cx0 = (int)floorf((pos.x - radius) / cell_size);
    int cy0 = (int)floorf((pos.y - radius) / cell_size);
    int cz0 = (int)floorf((pos.z - radius) / cell_size);
    int cx1 = (int)floorf((pos.x + radius) / cell_size);
    int cy1 = (int)floorf((pos.y + radius) / cell_size);
    int cz1 = (int)floorf((pos.z + radius) / cell_size);

    uint32_t visited_keys[27];
    int num_visited = 0;

    for (int iz = cz0; iz <= cz1; ++iz)
    for (int iy = cy0; iy <= cy1; ++iy)
    for (int ix = cx0; ix <= cx1; ++ix) {
        uint32_t key = teschner_hash(make_i3(ix, iy, iz), params.grid_table_size);

        bool already_visited = false;
        for (int v = 0; v < num_visited; ++v) {
            if (visited_keys[v] == key) { already_visited = true; break; }
        }
        if (already_visited) continue;
        visited_keys[num_visited++] = key;

        uint32_t start = params.grid_cell_start[key];
        uint32_t end   = params.grid_cell_end[key];
        if (start == 0xFFFFFFFF) continue;

        for (uint32_t j = start; j < end; ++j) {
            uint32_t idx = params.grid_sorted_indices[j];
            float3 pp = make_f3(
                params.photon_pos_x[idx],
                params.photon_pos_y[idx],
                params.photon_pos_z[idx]);

            float3 diff = pos - pp;

            // Tangential-disk distance metric (§7.1)
            float d_plane = dot(diff, filter_normal);
            float3 v_tan = diff - filter_normal * d_plane;
            float d_tan2 = dot(v_tan, v_tan);

            if (d_tan2 > r2_max) continue;
            if (knn_count >= GUIDE_KNN_K && d_tan2 >= knn_d2[0]) continue;
            if (fabsf(d_plane) > DEFAULT_SURFACE_TAU) continue;

            // Normal gate
            float3 photon_n = make_f3(
                params.photon_norm_x[idx],
                params.photon_norm_y[idx],
                params.photon_norm_z[idx]);
            if (dot(photon_n, filter_normal) <= 0.0f) continue;

            // Wi direction gate
            float3 wi_world = make_f3(
                params.photon_wi_x[idx],
                params.photon_wi_y[idx],
                params.photon_wi_z[idx]);
            if (dot(wi_world, filter_normal) <= 0.f) continue;

            // ── Insert into max-heap ────────────────────────────────
            if (knn_count < GUIDE_KNN_K) {
                knn_d2[knn_count]  = d_tan2;
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
                knn_d2[0]  = d_tan2;
                knn_idx[0] = idx;
                int ci = 0;
                while (true) {
                    int left = 2 * ci + 1, right = 2 * ci + 2, largest = ci;
                    if (left  < GUIDE_KNN_K && knn_d2[left]  > knn_d2[largest]) largest = left;
                    if (right < GUIDE_KNN_K && knn_d2[right] > knn_d2[largest]) largest = right;
                    if (largest == ci) break;
                    float td = knn_d2[ci]; knn_d2[ci] = knn_d2[largest]; knn_d2[largest] = td;
                    uint32_t ti = knn_idx[ci]; knn_idx[ci] = knn_idx[largest]; knn_idx[largest] = ti;
                    ci = largest;
                }
            }
        }
    }

    if (knn_count == 0) return h;

    // ── Phase 2: Bin photon directions by Fibonacci bin ──────────────
    // Weight each photon's contribution by its scalar flux (sum of hero
    // wavelength fluxes) so the guide PDF is proportional to radiant flux.
    for (int i = 0; i < knn_count; ++i) {
        uint32_t idx = knn_idx[i];

        float3 wi_world = make_f3(
            params.photon_wi_x[idx],
            params.photon_wi_y[idx],
            params.photon_wi_z[idx]);

        int bin = fib.find_nearest(wi_world);

        // Scalar flux = sum of hero wavelength fluxes
        int n_hero = params.photon_num_hero ? (int)params.photon_num_hero[idx] : 1;
        float scalar_flux = 0.f;
        for (int hh = 0; hh < n_hero; ++hh)
            scalar_flux += params.photon_flux[idx * HERO_WAVELENGTHS + hh];

        if (scalar_flux > 0.f) {
            h.bin_flux[bin] += scalar_flux;
            h.total_flux    += scalar_flux;
        }
    }

    h.valid = (h.total_flux > 0.f);
    return h;
}

// ── Sample a guided direction from the histogram ────────────────────
// Returns: world-space direction, PDF in solid angle, sampled bin index.

struct GuidedSample {
    float3 wi_world;    // sampled world-space direction
    float  pdf;         // probability density in solid angle
    int    bin;         // sampled bin index (-1 if invalid)
    bool   valid;
};

__forceinline__ __device__
GuidedSample dev_sample_guided_direction(
    const GuidedHistogram& h,
    const DevPhotonBinDirs& fib,
    float3 normal,
    PCGRng& rng)
{
    GuidedSample s;
    s.valid = false;
    s.pdf   = 0.f;
    s.bin   = -1;

    if (!h.valid) return s;

    // Inverse-CDF discrete sampling over the bin fluxes
    float xi = rng.next_float() * h.total_flux;
    float cumulative = 0.f;
    int chosen = h.num_bins - 1;
    for (int k = 0; k < h.num_bins; ++k) {
        cumulative += h.bin_flux[k];
        if (xi <= cumulative) { chosen = k; break; }
    }

    s.bin = chosen;

    // Continuous direction: Fibonacci sphere direction for this bin,
    // with small within-bin jitter to avoid discrete aliasing.
    // Jitter within the bin's approximate angular extent (~sqrt(Ω_bin))
    float3 center = fib.dirs[chosen];
    float  jitter_scale = sqrtf(bin_solid_angle(h.num_bins) * INV_PI) * 0.3f;
    float  u1 = (rng.next_float() - 0.5f) * jitter_scale;
    float  u2 = (rng.next_float() - 0.5f) * jitter_scale;
    // Perturb in a tangent frame around the bin center
    float3 t1, t2;
    {
        float3 a = (fabsf(center.x) > 0.9f) ? make_f3(0,1,0) : make_f3(1,0,0);
        t1 = normalize(cross(center, a));
        t2 = cross(center, t1);
    }
    float3 perturbed = normalize(center + t1 * u1 + t2 * u2);

    // Ensure the guided direction is in the upper hemisphere of the surface
    if (dot(perturbed, normal) <= 0.f) {
        // Flip to same hemisphere as normal
        perturbed = perturbed - normal * (2.f * dot(perturbed, normal));
        perturbed = normalize(perturbed);
    }
    if (dot(perturbed, normal) <= 0.f) return s;  // degenerate

    s.wi_world = perturbed;

    // PDF = (bin_flux / total_flux) / Ω_bin
    float p_bin = h.bin_flux[chosen] / h.total_flux;
    float omega = bin_solid_angle(h.num_bins);
    s.pdf = p_bin / omega;
    s.valid = (s.pdf > 0.f);

    return s;
}

// ── Evaluate guide PDF for an arbitrary direction ───────────────────
// Used for MIS when the BSDF sample is chosen but we need the guide's
// PDF at that direction.

__forceinline__ __device__
float dev_guided_pdf(
    const GuidedHistogram& h,
    const DevPhotonBinDirs& fib,
    float3 wi_world)
{
    if (!h.valid) return 0.f;

    // Find which bin this direction belongs to
    int bin = fib.find_nearest(wi_world);
    float p_bin = h.bin_flux[bin] / h.total_flux;
    float omega = bin_solid_angle(h.num_bins);
    return p_bin / omega;
}

// =====================================================================
// Volume photon guide (§9.7 — VP-03/MT-04)
// =====================================================================
// Volume variant: reads from vol_cell_bin_grid, no normal gate.

__forceinline__ __device__
int dev_vol_cell_grid_index(float3 pos) {
    int ix = (int)floorf((pos.x - params.vol_cell_grid_min_x) / params.vol_cell_grid_cell_size);
    int iy = (int)floorf((pos.y - params.vol_cell_grid_min_y) / params.vol_cell_grid_cell_size);
    int iz = (int)floorf((pos.z - params.vol_cell_grid_min_z) / params.vol_cell_grid_cell_size);
    ix = max(0, min(ix, params.vol_cell_grid_dim_x - 1));
    iy = max(0, min(iy, params.vol_cell_grid_dim_y - 1));
    iz = max(0, min(iz, params.vol_cell_grid_dim_z - 1));
    return ix + iy * params.vol_cell_grid_dim_x
             + iz * params.vol_cell_grid_dim_x * params.vol_cell_grid_dim_y;
}

__forceinline__ __device__
GuidedHistogram dev_read_vol_cell_histogram(float3 pos) {
    GuidedHistogram h;
    h.total_flux = 0.f;
    h.num_bins   = params.photon_bin_count;
    h.valid      = false;

    if (!params.vol_cell_grid_valid || !params.vol_cell_bin_grid || h.num_bins <= 0)
        return h;

    int cell = dev_vol_cell_grid_index(pos);
    const PhotonBin* bins = &params.vol_cell_bin_grid[cell * h.num_bins];

    // No normal gate for volume photons — aggregate all bins
    for (int k = 0; k < h.num_bins; ++k) {
        float sf = bins[k].scalar_flux;
        h.bin_flux[k] = (sf > 0.f) ? sf : 0.f;
        h.total_flux += h.bin_flux[k];
    }

    h.valid = (h.total_flux > 0.f);
    return h;
}
