#pragma once

// optix_guided.cuh — Photon-guided direction sampling on GPU
//
// Surface path: Dense-grid cell lookup → random photon pick → bounce in wi.
// Volume path:  Cell-bin histogram → Fibonacci inverse-CDF sampling (unchanged).

// ── Fibonacci sphere bin directions (device) ────────────────────────
// Pre-compute once and store in a small local array.
// Matches PhotonBinDirs::init(n) on CPU.
// Used by volume guide path only (surface path uses dense grid random pick).

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
__forceinline__ __device__
float bin_solid_angle(int num_bins) {
    return (4.0f * PI) / (float)num_bins;
}

// =====================================================================
// Dense grid cell lookup (surface photon guide)
// =====================================================================

__forceinline__ __device__
int dev_dense_cell_index(float3 pos) {
    int ix = (int)floorf((pos.x - params.dense_min_x) / params.dense_cell_size);
    int iy = (int)floorf((pos.y - params.dense_min_y) / params.dense_cell_size);
    int iz = (int)floorf((pos.z - params.dense_min_z) / params.dense_cell_size);
    ix = max(0, min(ix, params.dense_dim_x - 1));
    iy = max(0, min(iy, params.dense_dim_y - 1));
    iz = max(0, min(iz, params.dense_dim_z - 1));
    return ix + iy * params.dense_dim_x
             + iz * params.dense_dim_x * params.dense_dim_y;
}

// ── Pick a random photon from the 3x3x3 dense cell neighbourhood ────
// Returns the photon index, or -1 if all cells are empty / filtered.
// Applies normal gate, surface tau filter, and wi hemisphere gate.
__forceinline__ __device__
int dev_random_photon_in_cell(float3 pos, float3 filter_normal, PCGRng& rng) {
    if (!params.dense_valid || params.num_photons == 0)
        return -1;

    // Centre cell coordinates
    int cx = (int)floorf((pos.x - params.dense_min_x) / params.dense_cell_size);
    int cy = (int)floorf((pos.y - params.dense_min_y) / params.dense_cell_size);
    int cz = (int)floorf((pos.z - params.dense_min_z) / params.dense_cell_size);

    // Count total photons in the 3x3x3 neighbourhood
    uint32_t total_count = 0;
    for (int dz = -1; dz <= 1; ++dz) {
        int iz = cz + dz;
        if (iz < 0 || iz >= params.dense_dim_z) continue;
        for (int dy = -1; dy <= 1; ++dy) {
            int iy = cy + dy;
            if (iy < 0 || iy >= params.dense_dim_y) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int ix = cx + dx;
                if (ix < 0 || ix >= params.dense_dim_x) continue;
                int cell = ix + iy * params.dense_dim_x
                             + iz * params.dense_dim_x * params.dense_dim_y;
                total_count += params.dense_cell_end[cell]
                             - params.dense_cell_start[cell];
            }
        }
    }
    if (total_count == 0) return -1;

    // Try up to 8 random picks from the neighbourhood pool
    for (int attempt = 0; attempt < 8; ++attempt) {
        uint32_t flat = (uint32_t)(rng.next_float() * (float)total_count);
        if (flat >= total_count) flat = total_count - 1;

        // Walk the 3x3x3 to locate which cell owns this flat index
        uint32_t acc = 0;
        uint32_t idx = 0;
        bool found = false;
        for (int dz = -1; dz <= 1 && !found; ++dz) {
            int iz = cz + dz;
            if (iz < 0 || iz >= params.dense_dim_z) continue;
            for (int dy = -1; dy <= 1 && !found; ++dy) {
                int iy = cy + dy;
                if (iy < 0 || iy >= params.dense_dim_y) continue;
                for (int dx = -1; dx <= 1 && !found; ++dx) {
                    int ix = cx + dx;
                    if (ix < 0 || ix >= params.dense_dim_x) continue;
                    int cell = ix + iy * params.dense_dim_x
                                 + iz * params.dense_dim_x * params.dense_dim_y;
                    uint32_t cs = params.dense_cell_start[cell];
                    uint32_t ce = params.dense_cell_end[cell];
                    uint32_t cnt = ce - cs;
                    if (flat < acc + cnt) {
                        idx = params.dense_sorted_indices[cs + (flat - acc)];
                        found = true;
                    }
                    acc += cnt;
                }
            }
        }
        if (!found) continue;

        // Normal gate
        float3 photon_n = make_f3(
            params.photon_norm_x[idx],
            params.photon_norm_y[idx],
            params.photon_norm_z[idx]);
        if (dot(photon_n, filter_normal) <= 0.0f) continue;

        // Surface tau (plane-distance) gate
        float3 pp = make_f3(
            params.photon_pos_x[idx],
            params.photon_pos_y[idx],
            params.photon_pos_z[idx]);
        float d_plane = fabsf(dot(pos - pp, filter_normal));
        if (d_plane > DEFAULT_SURFACE_TAU) continue;

        // Wi direction gate: photon must arrive from upper hemisphere
        float3 wi = make_f3(
            params.photon_wi_x[idx],
            params.photon_wi_y[idx],
            params.photon_wi_z[idx]);
        if (dot(wi, filter_normal) <= 0.f) continue;

        return (int)idx;
    }
    return -1;
}

// ── Evaluate the marginal guide PDF at a given world-space direction ─
// Loops over eligible photons in the dense-grid neighbourhood and
// computes:  pdf = (1/N) Σᵢ cosine_cone_pdf(cos∠(ω, wᵢ), cos_half)
// where N = number of eligible photons.  Returns 0 when no photons
// pass the normal / tau / hemisphere gates.
// Cap the loop at MAX_GUIDE_PDF_PHOTONS to keep cost bounded.

constexpr int MAX_GUIDE_PDF_PHOTONS = 64;

__forceinline__ __device__
float dev_guide_pdf_at_direction(
    float3 pos, float3 filter_normal, float3 wi_world, float cos_half)
{
    if (!params.dense_valid || params.num_photons == 0)
        return 0.f;

    const int R = params.guide_use_neighbourhood ? 1 : 0;

    int cx = (int)floorf((pos.x - params.dense_min_x) / params.dense_cell_size);
    int cy = (int)floorf((pos.y - params.dense_min_y) / params.dense_cell_size);
    int cz = (int)floorf((pos.z - params.dense_min_z) / params.dense_cell_size);

    float  pdf_sum   = 0.f;
    int    n_eligible = 0;

    for (int dz = -R; dz <= R; ++dz) {
        int iz = cz + dz;
        if (iz < 0 || iz >= params.dense_dim_z) continue;
        for (int dy = -R; dy <= R; ++dy) {
            int iy = cy + dy;
            if (iy < 0 || iy >= params.dense_dim_y) continue;
            for (int dx = -R; dx <= R; ++dx) {
                int ix = cx + dx;
                if (ix < 0 || ix >= params.dense_dim_x) continue;
                int cell = ix + iy * params.dense_dim_x
                             + iz * params.dense_dim_x * params.dense_dim_y;
                uint32_t cs = params.dense_cell_start[cell];
                uint32_t ce = params.dense_cell_end[cell];
                for (uint32_t j = cs; j < ce; ++j) {
                    if (n_eligible >= MAX_GUIDE_PDF_PHOTONS) goto done;
                    uint32_t idx = params.dense_sorted_indices[j];

                    // Normal gate
                    float3 pn = make_f3(
                        params.photon_norm_x[idx],
                        params.photon_norm_y[idx],
                        params.photon_norm_z[idx]);
                    if (dot(pn, filter_normal) <= 0.f) continue;

                    // Surface-tau gate
                    float3 pp = make_f3(
                        params.photon_pos_x[idx],
                        params.photon_pos_y[idx],
                        params.photon_pos_z[idx]);
                    if (fabsf(dot(pos - pp, filter_normal)) > DEFAULT_SURFACE_TAU)
                        continue;

                    // Wi hemisphere gate
                    float3 pw = make_f3(
                        params.photon_wi_x[idx],
                        params.photon_wi_y[idx],
                        params.photon_wi_z[idx]);
                    if (dot(pw, filter_normal) <= 0.f) continue;

                    ++n_eligible;
                    float cos_angle = dot(wi_world, pw);
                    pdf_sum += (cos_half < 1.f - 1e-6f)
                        ? cosine_cone_pdf(cos_angle, cos_half)
                        : INV_2PI;
                }
            }
        }
    }
done:
    return (n_eligible > 0) ? pdf_sum / (float)n_eligible : 0.f;
}

// ── Guided histogram: directional bin data for guide sampling ────────
// Used by volume guide path (and kept for API compatibility).

struct GuidedHistogram {
    float bin_flux[MAX_PHOTON_BIN_COUNT];   // scalar_flux per bin
    float total_flux;                        // sum over all bins
    int   num_bins;
    bool  valid;                             // true if any data found
};

// ── Sample a guided direction from the histogram (volume only) ──────
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
