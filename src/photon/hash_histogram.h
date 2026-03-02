#pragma once
// ─────────────────────────────────────────────────────────────────────
// hash_histogram.h – Multi-resolution hash-indexed directional histograms
// ─────────────────────────────────────────────────────────────────────
// Replaces CellBinGrid for photon-guided direction sampling (§4).
//
// Instead of a dense 3D AABB grid, directional histograms are stored in
// a Teschner spatial hash table (same hash as CellInfoCache / HashGrid).
// This eliminates the mismatch between hash-indexed cell analysis data
// (guide_fraction, caustic_fraction) and dense-grid–indexed histograms.
//
// Multi-resolution: multiple hash tables at increasing cell sizes.
// At query time, the GPU uses coarse-to-fine selection: starting from
// the coarsest level, it takes the finest level that has photon data
// for the queried cell.
//
// Build algorithm per level (same as CellBinGrid, hash-indexed):
//   For each photon:
//     1. Compute integer cell coords at this level's cell_size.
//     2. Scatter to 3×3×3 neighbourhood (27 cells).
//     3. Tangential plane projection + Epanechnikov kernel.
//     4. Deposit flux-weighted hero wavelength into the photon's
//        precomputed Fibonacci bin at hash(neighbour_cell).
//
// Memory per level: TABLE_SIZE × BIN_COUNT × sizeof(GpuGuideBin)
//   = 65536 × 32 × 16 = 32 MB.
// ─────────────────────────────────────────────────────────────────────

#include "core/types.h"
#include "core/config.h"
#include "core/hash.h"
#include "core/spectrum.h"
#include "photon/photon.h"
#include "photon/photon_bins.h"

#include <vector>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>

// ── Configuration ────────────────────────────────────────────────────
// MAX_GUIDE_LEVELS is defined in core/config.h

// Default level multipliers relative to gather_radius.
// Level 0 = gather_radius × GUIDE_LEVEL_SCALES[0], etc.
// The current CellBinGrid uses cell_size = gather_radius × 2.0,
// which matches level 0 with scale 2.0.
static constexpr float GUIDE_LEVEL_SCALES[] = {
    2.0f,   // level 0: cell_size = gather_radius × 2  (current default)
    4.0f,   // level 1: cell_size = gather_radius × 4
    8.0f,   // level 2: cell_size = gather_radius × 8
    16.0f,  // level 3: cell_size = gather_radius × 16
};
static constexpr int DEFAULT_GUIDE_NUM_LEVELS = 1;  // start conservative

// ── Hash histogram (single resolution level) ────────────────────────
// Internal per-bin accumulator used during the CPU build pass.
// Larger than GpuGuideBin because it accumulates spectral flux and
// flux-weighted direction/normal before normalization.

struct HashHistLevel {
    static constexpr uint32_t TABLE_SIZE = CELL_CACHE_TABLE_SIZE;

    float cell_size = 0.0f;
    int   bin_count = PHOTON_BIN_COUNT;

    // Internal build accumulators — full PhotonBin (52 bytes) per slot.
    // Layout: accum[hash_bucket * bin_count + k]
    std::vector<PhotonBin> accum;

    // Compact GPU-ready data extracted after build.
    // Layout: gpu_bins[hash_bucket * bin_count + k]
    std::vector<GpuGuideBin> gpu_bins;

    // ── Allocate and zero accumulators ───────────────────────────────
    void allocate(float cs, int num_bins) {
        cell_size = cs;
        bin_count = num_bins;
        const size_t total = (size_t)TABLE_SIZE * bin_count;
        accum.resize(total);
        std::memset(accum.data(), 0, total * sizeof(PhotonBin));
    }

    // ── Hash a world-space position to a bucket index ────────────────
    uint32_t cell_hash(float px, float py, float pz) const {
        int3 cell = make_i3(
            (int)std::floor(px / cell_size),
            (int)std::floor(py / cell_size),
            (int)std::floor(pz / cell_size));
        return teschner_hash(cell, TABLE_SIZE);
    }

    // ── Scatter one photon into the 3×3×3 neighbourhood ─────────────
    // Same tangential-disk Epanechnikov kernel as CellBinGrid::build().
    void scatter_photon(
        float px, float py, float pz,
        float wi_x, float wi_y, float wi_z,
        float nx, float ny, float nz,
        int   bin_idx,
        const float* hero_flux,      // [HERO_WAVELENGTHS]
        const uint16_t* hero_lambda,  // [HERO_WAVELENGTHS]
        int   n_hero,
        float total_hero_flux)
    {
        // Integer cell coords of photon's cell at this level
        const int cx = (int)std::floor(px / cell_size);
        const int cy = (int)std::floor(py / cell_size);
        const int cz = (int)std::floor(pz / cell_size);

        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const int ncx = cx + dx;
            const int ncy = cy + dy;
            const int ncz = cz + dz;

            // Cell centre in world space
            const float cell_cx = (ncx + 0.5f) * cell_size;
            const float cell_cy = (ncy + 0.5f) * cell_size;
            const float cell_cz = (ncz + 0.5f) * cell_size;

            // Tangential plane projection (§7.1)
            const float dcx = px - cell_cx;
            const float dcy = py - cell_cy;
            const float dcz = pz - cell_cz;
            const float d_plane = nx * dcx + ny * dcy + nz * dcz;

            const float vx = dcx - nx * d_plane;
            const float vy = dcy - ny * d_plane;
            const float vz = dcz - nz * d_plane;
            const float d_tan2 = vx*vx + vy*vy + vz*vz;

            // Epanechnikov kernel, using this level's effective radius.
            // The kernel radius scales with cell_size / 2 (= effective
            // gather radius at this resolution level).
            const float level_r2 = (cell_size * 0.5f) * (cell_size * 0.5f);
            const float w = 1.f - d_tan2 / level_r2;
            if (w <= 0.f) continue;

            // Hash the neighbour cell
            const int3 nc = make_i3(ncx, ncy, ncz);
            const uint32_t bucket = teschner_hash(nc, TABLE_SIZE);

            PhotonBin& b = accum[(size_t)bucket * bin_count + bin_idx];

            // Deposit each hero wavelength's flux
            for (int h = 0; h < n_hero; ++h) {
                int lam_bin = (int)hero_lambda[h];
                float p_flux = hero_flux[h];
                if (lam_bin >= 0 && lam_bin < NUM_LAMBDA && p_flux > 0.f)
                    b.flux[lam_bin] += w * p_flux;
            }

            // Flux-weighted direction and normal
            const float wf = w * total_hero_flux;
            b.dir_x  += wi_x * wf;
            b.dir_y  += wi_y * wf;
            b.dir_z  += wi_z * wf;
            b.avg_nx += nx   * wf;
            b.avg_ny += ny   * wf;
            b.avg_nz += nz   * wf;
            b.weight += w;
            b.count  += 1;
        }
    }

    // ── Per-level statistics (computed during finalize()) ────────────
    int   populated_buckets  = 0;   // buckets where at least one bin has flux > 0
    int   populated_bins     = 0;   // individual bins with scalar_flux > 0
    float total_flux         = 0.f; // sum of scalar_flux across all bins
    float min_flux           = 0.f; // min scalar_flux among populated bins
    float max_flux           = 0.f; // max scalar_flux among populated bins
    float avg_flux           = 0.f; // mean scalar_flux of populated bins
    size_t gpu_memory_bytes  = 0;   // sizeof(GpuGuideBin) * gpu_bins.size()

    // ── Per-bucket statistics (for §3 analysis: Gaps 1 & 2) ─────────
    // Populated during finalize(); one entry per TABLE_SIZE bucket.
    std::vector<int>   bucket_active_bins;   // non-zero bins in bucket
    std::vector<float> bucket_total_flux;    // sum of scalar_flux
    std::vector<float> bucket_max_flux;      // max bin scalar_flux

    // Query helpers for photon_analysis.h
    int count_active_bins(uint32_t bucket) const {
        return (bucket < bucket_active_bins.size()) ? bucket_active_bins[bucket] : 0;
    }
    float get_bucket_total_flux(uint32_t bucket) const {
        return (bucket < bucket_total_flux.size()) ? bucket_total_flux[bucket] : 0.f;
    }
    float get_bucket_max_flux(uint32_t bucket) const {
        return (bucket < bucket_max_flux.size()) ? bucket_max_flux[bucket] : 0.f;
    }
    float get_concentration(uint32_t bucket) const {
        float tf = get_bucket_total_flux(bucket);
        return (tf > 0.f) ? get_bucket_max_flux(bucket) / tf : 0.f;
    }

    // ── Normalize accumulators and extract compact GPU bins ──────────
    void finalize() {
        const size_t total_bins = (size_t)TABLE_SIZE * bin_count;
        gpu_bins.resize(total_bins);

        // Reset stats
        populated_buckets = 0;
        populated_bins    = 0;
        total_flux        = 0.f;
        min_flux          = 1e30f;
        max_flux          = 0.f;

        // Allocate per-bucket stats
        bucket_active_bins.assign(TABLE_SIZE, 0);
        bucket_total_flux.assign(TABLE_SIZE, 0.f);
        bucket_max_flux.assign(TABLE_SIZE, 0.f);

        for (size_t bucket = 0; bucket < TABLE_SIZE; ++bucket) {
            bool bucket_has_data = false;
            int  b_active = 0;
            float b_total = 0.f;
            float b_max   = 0.f;

            for (int k = 0; k < bin_count; ++k) {
                size_t slot = bucket * bin_count + k;
                PhotonBin& b = accum[slot];

                // Compute scalar_flux = sum of spectral channels
                float sf = 0.f;
                for (int lam = 0; lam < NUM_LAMBDA; ++lam)
                    sf += b.flux[lam];
                b.scalar_flux = sf;

                // Normalize average normal
                float nlen = std::sqrt(b.avg_nx * b.avg_nx +
                                       b.avg_ny * b.avg_ny +
                                       b.avg_nz * b.avg_nz);
                if (nlen > 1e-8f) {
                    b.avg_nx /= nlen;
                    b.avg_ny /= nlen;
                    b.avg_nz /= nlen;
                }

                // Extract compact GPU data
                gpu_bins[slot].scalar_flux = sf;
                gpu_bins[slot].avg_nx      = b.avg_nx;
                gpu_bins[slot].avg_ny      = b.avg_ny;
                gpu_bins[slot].avg_nz      = b.avg_nz;

                // Accumulate stats
                if (sf > 0.f) {
                    bucket_has_data = true;
                    populated_bins++;
                    total_flux += sf;
                    if (sf < min_flux) min_flux = sf;
                    if (sf > max_flux) max_flux = sf;
                    // Per-bucket stats
                    b_active++;
                    b_total += sf;
                    if (sf > b_max) b_max = sf;
                }
            }
            bucket_active_bins[bucket] = b_active;
            bucket_total_flux[bucket]  = b_total;
            bucket_max_flux[bucket]    = b_max;
            if (bucket_has_data) populated_buckets++;
        }

        if (populated_bins == 0) min_flux = 0.f;
        avg_flux = (populated_bins > 0) ? total_flux / (float)populated_bins : 0.f;
        gpu_memory_bytes = gpu_bins.size() * sizeof(GpuGuideBin);

        // Free accumulators (no longer needed; GPU data is in gpu_bins)
        accum.clear();
        accum.shrink_to_fit();
    }
};

// ── Multi-resolution hash histogram ─────────────────────────────────
// Wraps N HashHistLevel instances.  Build all levels from the same
// photon data, extract GpuGuideBin arrays for GPU upload.

struct HashHistogram {
    int num_levels = 0;
    HashHistLevel levels[MAX_GUIDE_LEVELS];

    // ── Build multi-resolution histograms from photon data ───────────
    // `gather_radius` = base gather radius (e.g. 0.05).
    // `num_bins` = directional bins per cell (typically PHOTON_BIN_COUNT = 32).
    // `n_levels` = how many resolution levels (1..MAX_GUIDE_LEVELS).
    //
    // Photons must have bin_idx already precomputed (via PhotonBinDirs).
    void build(const PhotonSoA& photons,
               float gather_radius,
               int   num_bins,
               int   n_levels = DEFAULT_GUIDE_NUM_LEVELS)
    {
        num_levels = std::min(n_levels, (int)MAX_GUIDE_LEVELS);
        const size_t N = photons.size();

        if (N == 0 || gather_radius <= 0.f || num_levels <= 0) {
            num_levels = 0;
            return;
        }

        // ── 1. Initialize each level ────────────────────────────────
        for (int lv = 0; lv < num_levels; ++lv) {
            float scale = (lv < (int)(sizeof(GUIDE_LEVEL_SCALES)/sizeof(float)))
                        ? GUIDE_LEVEL_SCALES[lv]
                        : GUIDE_LEVEL_SCALES[0] * (float)(1 << lv);
            float cs = gather_radius * scale;
            levels[lv].allocate(cs, num_bins);

            std::printf("[HashHistogram] Level %d: cell_size=%.4f  "
                        "table=%u  bins=%d  %.1f MB\n",
                        lv, cs, HashHistLevel::TABLE_SIZE, num_bins,
                        (double)((size_t)HashHistLevel::TABLE_SIZE * num_bins *
                                 sizeof(PhotonBin)) / (1024.0 * 1024.0));
        }

        // ── 2. Single pass over photons, scatter to all levels ──────
        for (size_t i = 0; i < N; ++i) {
            const float px = photons.pos_x[i];
            const float py = photons.pos_y[i];
            const float pz = photons.pos_z[i];
            const float wi_x = photons.wi_x[i];
            const float wi_y = photons.wi_y[i];
            const float wi_z = photons.wi_z[i];
            const float nx   = photons.norm_x[i];
            const float ny   = photons.norm_y[i];
            const float nz   = photons.norm_z[i];

            int n_hero = photons.num_hero.empty() ? 1 : (int)photons.num_hero[i];
            float total_hero_flux = 0.f;
            for (int h = 0; h < n_hero; ++h)
                total_hero_flux += photons.flux[i * HERO_WAVELENGTHS + h];
            if (total_hero_flux <= 0.f) continue;

            int k = photons.bin_idx.empty() ? 0 : (int)photons.bin_idx[i];
            if (k < 0 || k >= num_bins) k = 0;

            const float* hero_flux  = &photons.flux[i * HERO_WAVELENGTHS];
            const uint16_t* hero_lam = &photons.lambda_bin[i * HERO_WAVELENGTHS];

            for (int lv = 0; lv < num_levels; ++lv) {
                levels[lv].scatter_photon(
                    px, py, pz,
                    wi_x, wi_y, wi_z,
                    nx, ny, nz,
                    k,
                    hero_flux, hero_lam, n_hero,
                    total_hero_flux);
            }
        }

        // ── 3. Normalize and extract compact GPU data ────────────────
        size_t total_gpu_bytes = 0;
        for (int lv = 0; lv < num_levels; ++lv) {
            levels[lv].finalize();
            total_gpu_bytes += levels[lv].gpu_memory_bytes;
        }

        // ── 4. Print detailed build summary ──────────────────────────
        std::printf("\n[HashHistogram] ═══════════════════════════════════════════\n");
        std::printf("[HashHistogram] Photons processed: %zu  |  Levels: %d  |  "
                    "Gather radius: %.5f\n", N, num_levels, gather_radius);
        for (int lv = 0; lv < num_levels; ++lv) {
            const auto& L = levels[lv];
            float occ_pct = 100.f * (float)L.populated_buckets
                          / (float)HashHistLevel::TABLE_SIZE;
            std::printf("[HashHistogram] Level %d  cell_size=%.4f  "
                        "scale=%.0fx\n", lv, L.cell_size,
                        L.cell_size / gather_radius);
            std::printf("  Buckets: %u total, %d populated (%.1f%% occupancy)\n",
                        HashHistLevel::TABLE_SIZE, L.populated_buckets, occ_pct);
            std::printf("  Bins:    %d populated / %u total\n",
                        L.populated_bins,
                        (uint32_t)((size_t)HashHistLevel::TABLE_SIZE * L.bin_count));
            std::printf("  Flux:    total=%.4f  min=%.6f  max=%.4f  avg=%.6f\n",
                        L.total_flux, L.min_flux, L.max_flux, L.avg_flux);
            std::printf("  GPU:     %.1f MB (%zu bytes)\n",
                        (double)L.gpu_memory_bytes / (1024.0 * 1024.0),
                        L.gpu_memory_bytes);
        }
        std::printf("[HashHistogram] Total GPU memory: %.1f MB\n",
                    (double)total_gpu_bytes / (1024.0 * 1024.0));
        std::printf("[HashHistogram] ═══════════════════════════════════════════\n\n");
    }

    // ── Cell sizes for GPU upload ────────────────────────────────────
    float cell_size(int level) const {
        return (level >= 0 && level < num_levels) ? levels[level].cell_size : 0.f;
    }

    // ── GPU bin data for one level ───────────────────────────────────
    const std::vector<GpuGuideBin>& gpu_bins(int level) const {
        return levels[level].gpu_bins;
    }
};
