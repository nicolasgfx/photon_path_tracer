#pragma once

// optix_guided.cuh — §4 Photon-guided direction sampling on GPU
//
// At each camera bounce, the pre-built CellBinGrid (Fibonacci sphere
// histograms, 32 bins/cell) provides an O(1) directional guide.
// The guide direction is MIS-combined with the standard BSDF sample
// using the balance heuristic.
//
// Build phase:  CellBinGrid::build() on CPU (fully implemented).
// Query phase:  this file — called from full_path_trace().

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

// ── Cell lookup: world position → flat grid index ───────────────────

__forceinline__ __device__
int dev_cell_grid_index(float3 pos) {
    int ix = (int)floorf((pos.x - params.cell_grid_min_x) / params.cell_grid_cell_size);
    int iy = (int)floorf((pos.y - params.cell_grid_min_y) / params.cell_grid_cell_size);
    int iz = (int)floorf((pos.z - params.cell_grid_min_z) / params.cell_grid_cell_size);
    ix = max(0, min(ix, params.cell_grid_dim_x - 1));
    iy = max(0, min(iy, params.cell_grid_dim_y - 1));
    iz = max(0, min(iz, params.cell_grid_dim_z - 1));
    return ix + iy * params.cell_grid_dim_x
             + iz * params.cell_grid_dim_x * params.cell_grid_dim_y;
}

// ── Read histogram for a cell → total_flux + scalar_flux array ──────

struct GuidedHistogram {
    float bin_flux[MAX_PHOTON_BIN_COUNT];   // scalar_flux per bin
    float total_flux;                        // sum over all bins
    int   num_bins;
    bool  valid;                             // true if grid available and total > 0
};

__forceinline__ __device__
GuidedHistogram dev_read_cell_histogram(float3 pos, float3 normal) {
    GuidedHistogram h;
    h.total_flux = 0.f;
    h.num_bins   = params.photon_bin_count;
    h.valid      = false;

    if (!params.cell_grid_valid || !params.cell_bin_grid || h.num_bins <= 0)
        return h;

    int cell = dev_cell_grid_index(pos);
    const PhotonBin* bins = &params.cell_bin_grid[cell * h.num_bins];

    // Sum scalar_flux over all bins, applying a hemisphere gate:
    // only include bins whose average photon normal is compatible
    // with the query-point normal (prevents cross-surface leakage).
    for (int k = 0; k < h.num_bins; ++k) {
        float sf = bins[k].scalar_flux;
        // Normal gate: skip bins whose deposited normals face away
        float3 avg_n = make_f3(bins[k].avg_nx, bins[k].avg_ny, bins[k].avg_nz);
        float n_dot = dot(avg_n, normal);
        if (n_dot < 0.1f || sf <= 0.f) {
            h.bin_flux[k] = 0.f;
        } else {
            h.bin_flux[k] = sf;
            h.total_flux += sf;
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
