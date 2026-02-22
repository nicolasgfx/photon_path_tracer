#pragma once
// ═════════════════════════════ DEPRECATED ═══════════════════════════
// cell_bin_grid.h — v1 dense 3D photon-bin grid (superseded in v2.1)
// ────────────────────────────────────────────────────────────────────
// The v2.1 renderer uses the tangential-disk hash-grid (hash_grid.h)
// instead.  This file is retained only for the OptiX host-side path
// (build_cell_bin_grid in optix_renderer.cpp) and test hooks.  Do not
// use in new code.
// TODO: Remove once build_cell_bin_grid is deleted from OptixRenderer.
// ════════════════════════════════════════════════════════════════════
// ────────────────── Original documentation ────────────────────────
// cell_bin_grid.h – Dense 3D grid of precomputed photon directional bins
// ────────────────────────────────────────────────────────────────────
// Two-pass normal-gated scatter:
//
//   Pass 1 – Each photon is accumulated into its OWN cell only.
//            After this pass every cell has a flux-weighted dominant
//            surface normal derived from all photons that physically
//            fell in that cell.
//
//   Pass 2 – Each photon scatters into the 26 neighbours (3×3×3),
//            but is SKIPPED for any neighbour cell whose dominant
//            normal (from pass 1) has dot(photon_normal, cell_normal)
//            <= 0.  This prevents cross-surface contamination (e.g.
//            wall photons leaking into floor cells).
//
// At render time a hitpoint simply indexes into the flat grid array
// for O(1) directional-bin lookup — there is NO per-bounce photon
// gather.
//
// Memory: grid_x * grid_y * grid_z * PHOTON_BIN_COUNT * sizeof(PhotonBin)
//       + grid_x * grid_y * grid_z * sizeof(float3) for dominant normals
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
    // Two-pass normal-gated scatter.
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

        std::printf("[CellBinGrid] Dims %d×%d×%d = %zu cells, "
                    "%.1f MB (%d bins/cell)\n",
                    dim_x, dim_y, dim_z, total,
                    (double)(total_bins * sizeof(PhotonBin)) / (1024.0 * 1024.0),
                    bin_count);

        // ── 2. Allocate and zero bins ────────────────────────────────
        bins.resize(total_bins);
        std::memset(bins.data(), 0, total_bins * sizeof(PhotonBin));

        // ── 3. Pass 1: native-cell-only accumulation ─────────────────
        // Accumulate photons into ONLY their own cell.  Also build the
        // flux-weighted dominant normal per cell (unnormalised sum).
        cell_dominant_normal.resize(total);
        for (size_t c = 0; c < total; ++c)
            cell_dominant_normal[c] = make_f3(0.f, 0.f, 0.f);

        for (size_t i = 0; i < N; ++i) {
            float px = photons.pos_x[i];
            float py = photons.pos_y[i];
            float pz = photons.pos_z[i];
            float flux = photons.total_flux(i);
            float wi_x = photons.wi_x[i];
            float wi_y = photons.wi_y[i];
            float wi_z = photons.wi_z[i];
            float nx   = photons.norm_x[i];
            float ny   = photons.norm_y[i];
            float nz   = photons.norm_z[i];
            int k = (int)photons.bin_idx[i];
            if (k >= bin_count) k = 0;

            int cx = (int)std::floor((px - min_x) / cell_size);
            int cy = (int)std::floor((py - min_y) / cell_size);
            int cz = (int)std::floor((pz - min_z) / cell_size);
            cx = std::max(0, std::min(cx, dim_x - 1));
            cy = std::max(0, std::min(cy, dim_y - 1));
            cz = std::max(0, std::min(cz, dim_z - 1));

            int flat = cx + cy * dim_x + cz * dim_x * dim_y;

            // Accumulate into native cell
            PhotonBin& b = bins[(size_t)flat * bin_count + k];
            b.flux   += flux;
            b.dir_x  += wi_x * flux;
            b.dir_y  += wi_y * flux;
            b.dir_z  += wi_z * flux;
            b.avg_nx += nx * flux;
            b.avg_ny += ny * flux;
            b.avg_nz += nz * flux;
            b.weight += 1.0f;
            b.count  += 1;

            // Accumulate flux-weighted normal for this cell
            cell_dominant_normal[flat].x += nx * flux;
            cell_dominant_normal[flat].y += ny * flux;
            cell_dominant_normal[flat].z += nz * flux;
        }

        // Normalise dominant normals
        for (size_t c = 0; c < total; ++c) {
            float len = std::sqrt(cell_dominant_normal[c].x * cell_dominant_normal[c].x +
                                  cell_dominant_normal[c].y * cell_dominant_normal[c].y +
                                  cell_dominant_normal[c].z * cell_dominant_normal[c].z);
            if (len > 1e-8f) {
                cell_dominant_normal[c].x /= len;
                cell_dominant_normal[c].y /= len;
                cell_dominant_normal[c].z /= len;
            }
            // (zero-length stays zero → cell is empty, no gate needed)
        }

        // ── 4. Pass 2: normal-gated 3×3×3 neighbour scatter ─────────
        // Each photon scatters into the 26 surrounding cells, but is
        // SKIPPED for any cell whose dominant normal (from pass 1) is
        // incompatible with the photon's surface normal.
        for (size_t i = 0; i < N; ++i) {
            float px = photons.pos_x[i];
            float py = photons.pos_y[i];
            float pz = photons.pos_z[i];
            float flux = photons.total_flux(i);
            float wi_x = photons.wi_x[i];
            float wi_y = photons.wi_y[i];
            float wi_z = photons.wi_z[i];
            float nx   = photons.norm_x[i];
            float ny   = photons.norm_y[i];
            float nz   = photons.norm_z[i];
            int k = (int)photons.bin_idx[i];
            if (k >= bin_count) k = 0;

            int cx = (int)std::floor((px - min_x) / cell_size);
            int cy = (int)std::floor((py - min_y) / cell_size);
            int cz = (int)std::floor((pz - min_z) / cell_size);

            for (int dz = -1; dz <= 1; ++dz) {
                int nz_ = cz + dz;
                if (nz_ < 0 || nz_ >= dim_z) continue;
                for (int dy = -1; dy <= 1; ++dy) {
                    int ny_ = cy + dy;
                    if (ny_ < 0 || ny_ >= dim_y) continue;
                    for (int dx = -1; dx <= 1; ++dx) {
                        // Skip the native cell — already accumulated in pass 1
                        if (dx == 0 && dy == 0 && dz == 0) continue;

                        int nx_ = cx + dx;
                        if (nx_ < 0 || nx_ >= dim_x) continue;

                        int flat = nx_ + ny_ * dim_x + nz_ * dim_x * dim_y;

                        // ── Normal gate: skip if photon's surface normal is
                        //    incompatible with this cell's dominant normal.
                        //    dot <= 0 ⟹ photon is on a different surface
                        //    orientation (e.g. wall vs floor).
                        float3 cdn = cell_dominant_normal[flat];
                        float d = nx * cdn.x + ny * cdn.y + nz * cdn.z;
                        if (d <= 0.f) continue;

                        PhotonBin& b = bins[(size_t)flat * bin_count + k];
                        b.flux   += flux;
                        b.dir_x  += wi_x * flux;
                        b.dir_y  += wi_y * flux;
                        b.dir_z  += wi_z * flux;
                        b.avg_nx += nx * flux;
                        b.avg_ny += ny * flux;
                        b.avg_nz += nz * flux;
                        b.weight += 1.0f;
                        b.count  += 1;
                    }
                }
            }
        }

        // ── 5. Normalize centroid directions and surface normals ─────
        PhotonBinDirs fib;
        fib.init(bin_count);

        for (size_t c = 0; c < total; ++c) {
            for (int k = 0; k < bin_count; ++k) {
                PhotonBin& b = bins[c * bin_count + k];
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
