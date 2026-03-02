#pragma once

// optix_guided.cuh — §4 Photon-guided direction sampling on GPU
//
// At each camera bounce, the precomputed hash histogram (Fibonacci sphere
// histograms, 32 bins/cell, multi-resolution) provides an O(1) directional
// guide.  The guide direction is MIS-combined with the standard BSDF sample
// using the balance heuristic.
//
// Build phase:  HashHistogram::build() on CPU (hash_histogram.h).
// Query phase:  this file — called from full_path_trace().
//
// Multi-resolution: multiple hash tables at increasing cell sizes.
// Coarse-to-fine selection: iterate from coarsest to finest level,
// take the finest level that has photon data for the queried cell.

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

// ── Cell lookup: world position → dense grid index (volume only) ─────
// The surface histogram now uses hash-indexed guide_histogram[].
// This dense-grid lookup is retained only for the volume photon guide.

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

// ── Read histogram: multi-res hash → total_flux + scalar_flux array ─

struct GuidedHistogram {
    float bin_flux[MAX_PHOTON_BIN_COUNT];   // scalar_flux per bin
    float total_flux;                        // sum over all bins
    int   num_bins;
    bool  valid;                             // true if any level has data
};

// Hash a world-space position at a given cell_size to a table bucket.
__forceinline__ __device__
uint32_t dev_guide_hash(float3 pos, float cell_size) {
    int3 cell = make_i3(
        (int)floorf(pos.x / cell_size),
        (int)floorf(pos.y / cell_size),
        (int)floorf(pos.z / cell_size));
    return teschner_hash(cell, CELL_CACHE_TABLE_SIZE);
}

__forceinline__ __device__
GuidedHistogram dev_read_cell_histogram(float3 pos, float3 normal) {
    GuidedHistogram h;
    h.total_flux = 0.f;
    h.num_bins   = params.photon_bin_count;
    h.valid      = false;

    if (params.guide_num_levels <= 0 || h.num_bins <= 0)
        return h;

    // Coarse-to-fine level selection:
    // Start at the coarsest level, iterate towards finest.
    // Use the finest level that has any photon data (total_flux > 0).
    for (int level = params.guide_num_levels - 1; level >= 0; --level) {
        float cs = params.guide_cell_size[level];
        if (cs <= 0.f || !params.guide_histogram[level]) continue;

        uint32_t bucket = dev_guide_hash(pos, cs);
        const GpuGuideBin* bins =
            &params.guide_histogram[level][bucket * h.num_bins];

        float level_flux = 0.f;
        float level_bins[MAX_PHOTON_BIN_COUNT];

        for (int k = 0; k < h.num_bins; ++k) {
            float sf = bins[k].scalar_flux;
            // Normal gate: skip bins whose average normal faces away
            float3 avg_n = make_f3(bins[k].avg_nx, bins[k].avg_ny, bins[k].avg_nz);
            float n_dot = dot(avg_n, normal);
            if (n_dot < 0.1f || sf <= 0.f) {
                level_bins[k] = 0.f;
            } else {
                level_bins[k] = sf;
                level_flux += sf;
            }
        }

        if (level_flux > 0.f) {
            // This level has data — use it (finest reliable level)
            for (int k = 0; k < h.num_bins; ++k)
                h.bin_flux[k] = level_bins[k];
            h.total_flux = level_flux;
            h.valid = true;
            break;
        }
    }

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
