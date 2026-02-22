#pragma once
// ─────────────────────────────────────────────────────────────────────
// cell_bin_grid.h – Dense 3D grid of precomputed photon directional bins
// ─────────────────────────────────────────────────────────────────────
// §3.5 / §6.7 — Fast O(PHOTON_BIN_COUNT) gather replace for the hash-grid
// per-photon walk.  Used by OptixRenderer when DEFAULT_USE_DENSE_GRID is
// true (runtime toggle: G key).
//
// Build algorithm (single-pass, CPU, run after hash-grid build):
//
//   For each photon:
//     1. Compute its integer cell (cx, cy, cz) in the same grid as the
//        hash grid (cell_size = 2 × gather_radius).
//     2. Project the photon onto its cell's tangent plane (§7.1):
//          d_plane = dot(photon_pos - cell_centre,  photon_normal)
//          v_tan   = (photon_pos - cell_centre) - photon_normal * d_plane
//          d_tan²  = |v_tan|²
//        Apply surface tau filter (§6.4): skip if |d_plane| > tau.
//        Apply Epanechnikov kernel:  w = max(0, 1 - d_tan²/r²).
//        Skip if w ≤ 0 (photon outside tangential disk).
//     3. Scatter flux w × hero_flux into the photon's precomputed bin
//        (bin_idx, already stored per-photon) in this cell.
//        Also accumulate the flux-weighted surface normal for the cell's
//        normal gate.
//
// Note: no neighbour scatter (pass 2).  Once Epanechnikov weight is baked
// relative to the cell centre, any photon in cell C has d_tan ≥ r to ALL
// adjacent cell centres (those are 2r away), so pass-2 weight = 0 for all
// combinations.  Dropping pass 2 is mathematically correct here.
//
// GPU query (O(PHOTON_BIN_COUNT)):
//   pos → cell index → bins[cell * N + k].
//   Per-bin: normal gate (avg_n · filter_normal > 0),
//            hemisphere gate (bin_dir · filter_normal > 0),
//            diffuse BSDF eval at bin centroid direction.
//   No extra spatial kernel weight at query time (kernel baked at build).
//   Normalization: same inv_area = 2/(π r²) and 1/N_emitted as hash grid.
//
// Memory:  cells × PHOTON_BIN_COUNT × sizeof(PhotonBin)
//          for r = 0.05, scene 1.0³: 1000 × 32 × 164 ≈ 5.1 MB
// ─────────────────────────────────────────────────────────────────────

#include "core/types.h"
#include "core/config.h"
#include "core/photon_bins.h"
#include "photon/photon.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdio>

struct CellBinGrid {
    // ── Grid geometry ────────────────────────────────────────────────
    float  cell_size = 0.0f;      // same as hash grid cell size (2 × radius)
    float  min_x = 0.0f, min_y = 0.0f, min_z = 0.0f;  // AABB min corner
    int    dim_x = 0, dim_y = 0, dim_z = 0;            // grid dimensions

    // ── Flattened bin data ───────────────────────────────────────────
    // Layout: bins[flat_cell_idx * bin_count + k]
    // flat_cell_idx = ix + iy * dim_x + iz * dim_x * dim_y
    std::vector<PhotonBin> bins;
    int bin_count = PHOTON_BIN_COUNT;

    // ── Per-cell dominant surface normal (flux-weighted, from pass 1) ─
    // Layout: cell_dominant_normal[flat_cell_idx]
    // Populated in pass 1, consumed in pass 2.
    std::vector<float3> cell_dominant_normal;

    // ── Total number of cells ────────────────────────────────────────
    int total_cells() const { return dim_x * dim_y * dim_z; }

    // ── World-position → cell index (clamped to grid) ────────────────
    int cell_index(float px, float py, float pz) const {
        int ix = (int)std::floor((px - min_x) / cell_size);
        int iy = (int)std::floor((py - min_y) / cell_size);
        int iz = (int)std::floor((pz - min_z) / cell_size);
        ix = std::max(0, std::min(ix, dim_x - 1));
        iy = std::max(0, std::min(iy, dim_y - 1));
        iz = std::max(0, std::min(iz, dim_z - 1));
        return ix + iy * dim_x + iz * dim_x * dim_y;
    }

    // ── Position → bins pointer (returns nullptr if grid is empty) ───
    const PhotonBin* lookup(float3 pos) const {
        if (bins.empty() || cell_size <= 0.f) return nullptr;
        int idx = cell_index(pos.x, pos.y, pos.z);
        return &bins[(size_t)idx * bin_count];
    }

    // ── Trilinear interpolation helper ───────────────────────────────
    // Returns up to 8 cell indices and trilinear weights for smooth
    // volume gather (eliminates hard cell-boundary artifacts).
    struct TrilinearResult {
        int   cell[8];
        float weight[8];
        int   count = 0;
    };

    TrilinearResult trilinear_cells(float3 pos) const {
        TrilinearResult r;
        r.count = 0;
        if (bins.empty() || cell_size <= 0.f) return r;

        // Position in cell-centre coordinates (cell centres at 0.5, 1.5, …)
        float fx = (pos.x - min_x) / cell_size - 0.5f;
        float fy = (pos.y - min_y) / cell_size - 0.5f;
        float fz = (pos.z - min_z) / cell_size - 0.5f;

        int ix0 = (int)std::floor(fx);  float tx = fx - ix0;
        int iy0 = (int)std::floor(fy);  float ty = fy - iy0;
        int iz0 = (int)std::floor(fz);  float tz = fz - iz0;

        for (int c = 0; c < 8; ++c) {
            int ix = (c & 1) ? ix0 + 1 : ix0;
            int iy = (c & 2) ? iy0 + 1 : iy0;
            int iz = (c & 4) ? iz0 + 1 : iz0;
            // Clamp to grid bounds
            ix = std::max(0, std::min(ix, dim_x - 1));
            iy = std::max(0, std::min(iy, dim_y - 1));
            iz = std::max(0, std::min(iz, dim_z - 1));

            float wx = (c & 1) ? tx : (1.f - tx);
            float wy = (c & 2) ? ty : (1.f - ty);
            float wz = (c & 4) ? tz : (1.f - tz);
            float w = wx * wy * wz;
            if (w <= 0.f) continue;

            r.cell[r.count]   = ix + iy * dim_x + iz * dim_x * dim_y;
            r.weight[r.count] = w;
            r.count++;
        }
        return r;
    }

    // ── Compute grid AABB and dimensions from photon positions ───────
    // Shared by build() and external reference implementations.
    void compute_grid_geometry(const PhotonSoA& photons) {
        float lo_x =  1e30f, lo_y =  1e30f, lo_z =  1e30f;
        float hi_x = -1e30f, hi_y = -1e30f, hi_z = -1e30f;
        const size_t N = photons.size();
        for (size_t i = 0; i < N; ++i) {
            lo_x = std::min(lo_x, photons.pos_x[i]);
            lo_y = std::min(lo_y, photons.pos_y[i]);
            lo_z = std::min(lo_z, photons.pos_z[i]);
            hi_x = std::max(hi_x, photons.pos_x[i]);
            hi_y = std::max(hi_y, photons.pos_y[i]);
            hi_z = std::max(hi_z, photons.pos_z[i]);
        }

        // Pad by one cell so boundary photons have room for neighbors
        lo_x -= cell_size;  lo_y -= cell_size;  lo_z -= cell_size;
        hi_x += cell_size;  hi_y += cell_size;  hi_z += cell_size;

        min_x = lo_x;  min_y = lo_y;  min_z = lo_z;
        dim_x = std::max(1, (int)std::ceil((hi_x - lo_x) / cell_size));
        dim_y = std::max(1, (int)std::ceil((hi_y - lo_y) / cell_size));
        dim_z = std::max(1, (int)std::ceil((hi_z - lo_z) / cell_size));
    }

    // ── Build the dense grid from photon data ────────────────────────
    // Single-pass: accumulate into photon's own cell only, with
    // tangential-disk kernel baked in (§7.1) and surface tau filter (§6.4).
    // `photons` must have bin_idx already precomputed.
    void build(const PhotonSoA& photons, float gather_radius, int num_bins) {
        bin_count = num_bins;
        cell_size = gather_radius * 2.0f;  // matches hash grid cell size

        if (photons.size() == 0 || cell_size <= 0.0f) {
            dim_x = dim_y = dim_z = 0;
            bins.clear();
            cell_dominant_normal.clear();
            return;
        }

        // ── 1. Compute AABB from photon positions ────────────────────
        compute_grid_geometry(photons);

        const size_t total = (size_t)dim_x * dim_y * dim_z;
        const size_t total_bins = total * bin_count;
        const size_t N = photons.size();
        const float  r2 = gather_radius * gather_radius;

        std::printf("[CellBinGrid] Dims %d\xc3\x97%d\xc3\x97%d = %zu cells, "
                    "%.2f MB (%d bins/cell)\n",
                    dim_x, dim_y, dim_z, total,
                    (double)(total_bins * sizeof(PhotonBin)) / (1024.0 * 1024.0),
                    bin_count);

        // ── 2. Allocate and zero bins and dominant normals ───────────
        bins.resize(total_bins);
        std::memset(bins.data(), 0, total_bins * sizeof(PhotonBin));
        cell_dominant_normal.resize(total);
        for (size_t c = 0; c < total; ++c)
            cell_dominant_normal[c] = make_f3(0.f, 0.f, 0.f);

        // ── 3. Single-pass accumulation with tangential disk kernel ──
        // For each photon:
        //   a. Find its integer cell (cx, cy, cz).
        //   b. Compute tangential distance from photon position to the
        //      cell centre, projected along the photon surface normal
        //      (§7.1 tangential plane projection).
        //   c. Skip if |plane_distance| > DEFAULT_SURFACE_TAU (§6.4).
        //   d. Apply Epanechnikov kernel: w = max(0, 1 - d_tan²/r²).
        //      Skip if w ≤ 0 (photon outside the disk).
        //   e. Scatter w-weighted hero flux into the bin.
        //
        // Note: neighbour scatter (pass 2) is intentionally omitted.
        // Once the Epanechnikov weight is computed relative to the cell
        // centre, the distance from any photon in cell C to any adjacent
        // cell centre is ≥ 2r (= cell_size), so the kernel weight for all
        // neighbours is ≤ 0.  Dropping pass 2 is therefore mathematically
        // exact here.
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

            // Per-hero-wavelength flux: each hero has a spectral bin index
            // and a scalar flux.  We deposit into the correct spectral bin
            // (matching the hash-grid path exactly).
            int n_hero = photons.num_hero.empty() ? 1 : (int)photons.num_hero[i];
            // Quick check: any nonzero flux?
            float total_hero_flux = 0.f;
            for (int h = 0; h < n_hero; ++h)
                total_hero_flux += photons.flux[i * HERO_WAVELENGTHS + h];
            if (total_hero_flux <= 0.f) continue;

            int k = photons.bin_idx.empty() ? 0 : (int)photons.bin_idx[i];
            if (k < 0 || k >= bin_count) k = 0;

            // Integer cell coords (unclamped; AABB computed above guarantees
            // all photons fall strictly inside [0, dim_*) after padding)
            int cx = (int)std::floor((px - min_x) / cell_size);
            int cy = (int)std::floor((py - min_y) / cell_size);
            int cz = (int)std::floor((pz - min_z) / cell_size);
            cx = std::max(0, std::min(cx, dim_x - 1));
            cy = std::max(0, std::min(cy, dim_y - 1));
            cz = std::max(0, std::min(cz, dim_z - 1));

            // ── Tangential plane projection (§7.1) ───────────────────
            // Project cell-centre-to-photon vector onto the photon's
            // surface normal to separate the plane component from the
            // lateral (tangential) component.
            const float cell_cx = min_x + (cx + 0.5f) * cell_size;
            const float cell_cy = min_y + (cy + 0.5f) * cell_size;
            const float cell_cz = min_z + (cz + 0.5f) * cell_size;
            const float dcx = px - cell_cx;
            const float dcy = py - cell_cy;
            const float dcz = pz - cell_cz;
            const float d_plane = nx * dcx + ny * dcy + nz * dcz;

            // Surface tau filter (§6.4): reject photon if too far from
            // the cell's tangent plane (cross-surface leakage prevention).
            if (std::fabsf(d_plane) > DEFAULT_SURFACE_TAU) continue;

            // Tangential distance from photon to cell centre
            const float vx = dcx - nx * d_plane;
            const float vy = dcy - ny * d_plane;
            const float vz = dcz - nz * d_plane;
            const float d_tan2 = vx*vx + vy*vy + vz*vz;

            // Epanechnikov kernel weight (§6.3)
            const float w = 1.f - d_tan2 / r2;
            if (w <= 0.f) continue;  // outside disk

            // ── Accumulate into this cell's bin ──────────────────────
            const int flat = cx + cy * dim_x + cz * dim_x * dim_y;

            PhotonBin& b = bins[(size_t)flat * bin_count + k];

            // Deposit each hero wavelength's flux into its spectral bin,
            // weighted by the Epanechnikov kernel.  This matches the
            // hash-grid path's per-wavelength accumulation exactly.
            for (int h = 0; h < n_hero; ++h) {
                int lam_bin = (int)photons.lambda_bin[i * HERO_WAVELENGTHS + h];
                float p_flux = photons.flux[i * HERO_WAVELENGTHS + h];
                if (lam_bin >= 0 && lam_bin < NUM_LAMBDA && p_flux > 0.f)
                    b.flux[lam_bin] += w * p_flux;
            }

            // Flux-weighted direction and normal
            // Use total hero flux for direction/normal weighting
            const float wf = w * total_hero_flux;
            b.dir_x  += wi_x * wf;
            b.dir_y  += wi_y * wf;
            b.dir_z  += wi_z * wf;
            b.avg_nx += nx   * wf;
            b.avg_ny += ny   * wf;
            b.avg_nz += nz   * wf;
            b.weight += w;     // Epanechnikov weight (for normalisation)
            b.count  += 1;

            // Flux-weighted dominant normal (for GPU normal gate)
            cell_dominant_normal[flat].x += nx * wf;
            cell_dominant_normal[flat].y += ny * wf;
            cell_dominant_normal[flat].z += nz * wf;
        }

        // ── 4. Normalize dominant normals for each cell ─────────────
        for (size_t c = 0; c < total; ++c) {
            float len = std::sqrt(cell_dominant_normal[c].x * cell_dominant_normal[c].x +
                                  cell_dominant_normal[c].y * cell_dominant_normal[c].y +
                                  cell_dominant_normal[c].z * cell_dominant_normal[c].z);
            if (len > 1e-8f) {
                cell_dominant_normal[c].x /= len;
                cell_dominant_normal[c].y /= len;
                cell_dominant_normal[c].z /= len;
            }
        }

        // ── 5. Compute scalar_flux and normalize directions/normals ──
        PhotonBinDirs fib;
        fib.init(bin_count);

        for (size_t c = 0; c < total; ++c) {
            for (int k = 0; k < bin_count; ++k) {
                PhotonBin& b = bins[c * bin_count + k];

                // Compute scalar_flux = sum of per-wavelength fluxes
                // (used by guided bounce/NEE as relative importance weight)
                float sf = 0.f;
                for (int lam = 0; lam < NUM_LAMBDA; ++lam)
                    sf += b.flux[lam];
                b.scalar_flux = sf;

                if (b.count > 0) {
                    float len = std::sqrt(b.dir_x * b.dir_x +
                                          b.dir_y * b.dir_y +
                                          b.dir_z * b.dir_z);
                    if (len > 1e-8f) {
                        b.dir_x /= len;
                        b.dir_y /= len;
                        b.dir_z /= len;
                    } else {
                        // Fallback to Fibonacci center
                        b.dir_x = fib.dirs[k].x;
                        b.dir_y = fib.dirs[k].y;
                        b.dir_z = fib.dirs[k].z;
                    }

                    // Normalize average surface normal
                    float nlen = std::sqrt(b.avg_nx * b.avg_nx +
                                            b.avg_ny * b.avg_ny +
                                            b.avg_nz * b.avg_nz);
                    if (nlen > 1e-8f) {
                        b.avg_nx /= nlen;
                        b.avg_ny /= nlen;
                        b.avg_nz /= nlen;
                    }
                    // (if nlen ≈ 0 the bin is degenerate; leave as zero)
                }
            }
        }
    }
};
