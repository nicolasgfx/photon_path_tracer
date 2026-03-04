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

// ── Tangential distance² (projected onto surface plane) ─────────────
// Same metric as optix_photon_gather.cuh, duplicated here because the
// gather and guide PTX are compiled separately.
__forceinline__ __device__
float dev_guide_tangential_dist2(float3 pos, float3 photon_pos, float3 normal) {
    float3 diff = pos - photon_pos;
    float d_plane = dot(diff, normal);
    float3 tangent = diff - normal * d_plane;
    return dot(tangent, tangent);
}

// Forward declaration — defined below, called by dev_guide_sample_and_pdf_full.
__forceinline__ __device__
float dev_guide_pdf_at_direction(
    float3 pos, float3 filter_normal, float3 wi_world, float cos_half);

// ── Fused guide sample + PDF (complete version) ─────────────────────
// Two-phase approach:
//   Phase 1: Single pass with weighted reservoir-1 to pick one photon.
//            Each eligible photon is weighted by an Epanechnikov kernel
//            w = max(0, 1 − d_tan²/r²) based on tangential distance,
//            so nearby photons dominate the pick.  This, combined with
//            the continuous radius gate, eliminates hard cell-boundary
//            discontinuities that caused block artifacts.
//   Phase 2: Second pass to compute the EXACT marginal guide PDF at
//            the jittered direction — identical to dev_guide_pdf_at_direction.
//
// Both phases use the same MAX_GUIDE_PDF_PHOTONS cap and identical
// cell iteration order, ensuring PDF-sample consistency for MIS.

struct GuideSamplePdfResult {
    int    photon_idx;   // -1 if no eligible photon found
    float  pdf_guide;    // marginal guide PDF at wi_dir
    float3 wi_dir;       // direction after cone jitter (ready for tracing)
};

__forceinline__ __device__
GuideSamplePdfResult dev_guide_sample_and_pdf_full(
    float3 pos, float3 filter_normal, float cos_half, PCGRng& rng)
{
    GuideSamplePdfResult res;
    res.photon_idx = -1;
    res.pdf_guide  = 0.f;
    res.wi_dir     = make_f3(0.f, 0.f, 1.f);

    if (!params.dense_valid || params.num_photons == 0)
        return res;

    const int R = DIR_MAP_NEIGHBOURHOOD_EXTENT;  // 5×5×5 neighbourhood
    const float guide_r2 = params.guide_radius * params.guide_radius;

    int cx = (int)floorf((pos.x - params.dense_min_x) / params.dense_cell_size);
    int cy = (int)floorf((pos.y - params.dense_min_y) / params.dense_cell_size);
    int cz = (int)floorf((pos.z - params.dense_min_z) / params.dense_cell_size);

    // ── Phase 1: Weighted reservoir-1 pick ───────────────────────────
    // Epanechnikov kernel: w = max(0, 1 − d_tan²/r²).  Photons near
    // the query point contribute more weight than distant ones.
    int      picked_idx = -1;
    float3   picked_wi  = make_f3(0.f, 0.f, 1.f);
    float    w_sum      = 0.f;   // cumulative kernel weight
    int      n_eligible = 0;

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
                    uint32_t idx = params.dense_sorted_indices[j];

                    // Normal gate
                    float3 pn = make_f3(
                        params.photon_norm_x[idx],
                        params.photon_norm_y[idx],
                        params.photon_norm_z[idx]);
                    if (dot(pn, filter_normal) <= 0.f) continue;

                    // Surface-tau gate (plane distance)
                    float3 pp = make_f3(
                        params.photon_pos_x[idx],
                        params.photon_pos_y[idx],
                        params.photon_pos_z[idx]);
                    if (fabsf(dot(pos - pp, filter_normal)) > DEFAULT_SURFACE_TAU)
                        continue;

                    // Tangential distance gate (continuous radius)
                    float d_tan2 = dev_guide_tangential_dist2(pos, pp, filter_normal);
                    if (d_tan2 >= guide_r2) continue;

                    // Wi hemisphere gate
                    float3 pw = make_f3(
                        params.photon_wi_x[idx],
                        params.photon_wi_y[idx],
                        params.photon_wi_z[idx]);
                    if (dot(pw, filter_normal) <= 0.f) continue;

                    ++n_eligible;

                    // Epanechnikov kernel weight
                    float w = 1.f - d_tan2 / guide_r2;

                    // Weighted reservoir-1: replace with prob w / w_sum
                    w_sum += w;
                    if (rng.next_float() < w / w_sum) {
                        picked_idx = (int)idx;
                        picked_wi  = pw;
                    }

                    // Cap eligible count to bound worst-case iteration
                    if (n_eligible >= MAX_GUIDE_PDF_PHOTONS) goto reservoir_done;
                }
            }
        }
    }
reservoir_done:

    // ── Guide stats: track per-neighbourhood eligible photon counts ──
    if constexpr (ENABLE_GUIDE_STATS) {
        if (params.guide_stats_buf) {
            atomicMin(&params.guide_stats_buf[0], n_eligible);
            atomicMax(&params.guide_stats_buf[1], n_eligible);
            atomicAdd(&params.guide_stats_buf[2], n_eligible);
            atomicAdd(&params.guide_stats_buf[3], 1);
        }
    }

    if (n_eligible == 0 || picked_idx < 0) return res;

    res.photon_idx = picked_idx;

    // Apply cone jitter
    float3 wi_dir = picked_wi;
    if (cos_half < 1.f - 1e-6f) {
        ONB cone_frame = ONB::from_normal(picked_wi);
        float3 cone_local = sample_cosine_cone(
            rng.next_float(), rng.next_float(), cos_half);
        wi_dir = cone_frame.local_to_world(cone_local);
    }
    res.wi_dir = wi_dir;

    // ── Phase 2: Exact marginal PDF ──────────────────────────────────
    // Second pass over the same neighbourhood to compute the exact PDF
    // at wi_dir.  Uses identical gates, radius, cap, and iteration
    // order as Phase 1, ensuring PDF-sample consistency for MIS.
    res.pdf_guide = dev_guide_pdf_at_direction(pos, filter_normal, wi_dir, cos_half);

    return res;
}

// ── Evaluate the marginal guide PDF at a given world-space direction ─
// Standalone version: used when the BSDF branch wins the coin flip
// and we need the guide PDF at the BSDF-sampled direction.
// Loops over eligible photons in the dense-grid neighbourhood and
// computes:  pdf = Σᵢ wᵢ·cone_pdf(ω, wᵢ) / Σᵢ wᵢ
// where wᵢ = Epanechnikov kernel weight for photon i.
// Returns 0 when no photons pass the gates.

__forceinline__ __device__
float dev_guide_pdf_at_direction(
    float3 pos, float3 filter_normal, float3 wi_world, float cos_half)
{
    if (!params.dense_valid || params.num_photons == 0)
        return 0.f;

    const int R = DIR_MAP_NEIGHBOURHOOD_EXTENT;  // 5×5×5 neighbourhood
    const float guide_r2 = params.guide_radius * params.guide_radius;

    int cx = (int)floorf((pos.x - params.dense_min_x) / params.dense_cell_size);
    int cy = (int)floorf((pos.y - params.dense_min_y) / params.dense_cell_size);
    int cz = (int)floorf((pos.z - params.dense_min_z) / params.dense_cell_size);

    float  pdf_sum    = 0.f;
    float  w_sum      = 0.f;
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
                    uint32_t idx = params.dense_sorted_indices[j];

                    // Normal gate
                    float3 pn = make_f3(
                        params.photon_norm_x[idx],
                        params.photon_norm_y[idx],
                        params.photon_norm_z[idx]);
                    if (dot(pn, filter_normal) <= 0.f) continue;

                    // Surface-tau gate (plane distance)
                    float3 pp = make_f3(
                        params.photon_pos_x[idx],
                        params.photon_pos_y[idx],
                        params.photon_pos_z[idx]);
                    if (fabsf(dot(pos - pp, filter_normal)) > DEFAULT_SURFACE_TAU)
                        continue;

                    // Tangential distance gate (continuous radius)
                    float d_tan2 = dev_guide_tangential_dist2(pos, pp, filter_normal);
                    if (d_tan2 >= guide_r2) continue;

                    // Wi hemisphere gate
                    float3 pw = make_f3(
                        params.photon_wi_x[idx],
                        params.photon_wi_y[idx],
                        params.photon_wi_z[idx]);
                    if (dot(pw, filter_normal) <= 0.f) continue;

                    ++n_eligible;

                    // Epanechnikov kernel weight (matches reservoir)
                    float w = 1.f - d_tan2 / guide_r2;
                    w_sum += w;

                    float cos_angle = dot(wi_world, pw);
                    pdf_sum += w * ((cos_half < 1.f - 1e-6f)
                        ? cosine_cone_pdf(cos_angle, cos_half)
                        : INV_2PI);

                    // Same cap as reservoir — must match for PDF consistency
                    if (n_eligible >= MAX_GUIDE_PDF_PHOTONS) goto pdf_done;
                }
            }
        }
    }
pdf_done:
    return (w_sum > 0.f) ? pdf_sum / w_sum : 0.f;
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
