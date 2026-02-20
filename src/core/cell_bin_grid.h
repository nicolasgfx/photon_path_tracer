#pragma once
// ─────────────────────────────────────────────────────────────────────
// cell_bin_grid.h – Dense 3D grid of precomputed photon directional bins
// ─────────────────────────────────────────────────────────────────────
// After the hash grid is built and bin_idx is precomputed, this grid
// is populated by scattering each photon into its own cell AND the 26
// neighbor cells (3×3×3).  At render time a hitpoint simply indexes
// into the flat grid array for O(1) directional-bin lookup — there is
// NO per-bounce photon gather.
//
// Memory: grid_x * grid_y * grid_z * PHOTON_BIN_COUNT * sizeof(PhotonBin)
// For a normalised scene fitting in ~[-0.5,0.5]³ with cell_size=0.1:
//   10 × 10 × 10 × 32 × 24 = 768 KB  (trivial)
// For larger scenes or finer grids the cost grows cubically but stays
// well within 12 GB for any realistic indoor scene.
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

    // ── Build the dense grid from photon data ────────────────────────
    // `photons` must have bin_idx already precomputed.
    void build(const PhotonSoA& photons, float gather_radius, int num_bins) {
        bin_count = num_bins;
        cell_size = gather_radius * 2.0f;  // matches hash grid cell size

        if (photons.size() == 0 || cell_size <= 0.0f) {
            dim_x = dim_y = dim_z = 0;
            bins.clear();
            return;
        }

        // ── 1. Compute AABB from photon positions ────────────────────
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

        const size_t total = (size_t)dim_x * dim_y * dim_z;
        const size_t total_bins = total * bin_count;

        std::printf("[CellBinGrid] Dims %d×%d×%d = %zu cells, "
                    "%.1f MB (%d bins/cell)\n",
                    dim_x, dim_y, dim_z, total,
                    (double)(total_bins * sizeof(PhotonBin)) / (1024.0 * 1024.0),
                    bin_count);

        // ── 2. Allocate and zero bins ────────────────────────────────
        bins.resize(total_bins);
        std::memset(bins.data(), 0, total_bins * sizeof(PhotonBin));

        // ── 3. Scatter each photon into its 3×3×3 neighborhood ──────
        for (size_t i = 0; i < N; ++i) {
            float px = photons.pos_x[i];
            float py = photons.pos_y[i];
            float pz = photons.pos_z[i];
            float flux = photons.flux[i];
            float wi_x = photons.wi_x[i];
            float wi_y = photons.wi_y[i];
            float wi_z = photons.wi_z[i];
            int k = (int)photons.bin_idx[i];
            if (k >= bin_count) k = 0;

            // Centre cell of this photon
            int cx = (int)std::floor((px - min_x) / cell_size);
            int cy = (int)std::floor((py - min_y) / cell_size);
            int cz = (int)std::floor((pz - min_z) / cell_size);

            // Scatter into 3×3×3 neighbors
            for (int dz = -1; dz <= 1; ++dz) {
                int nz = cz + dz;
                if (nz < 0 || nz >= dim_z) continue;
                for (int dy = -1; dy <= 1; ++dy) {
                    int ny = cy + dy;
                    if (ny < 0 || ny >= dim_y) continue;
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = cx + dx;
                        if (nx < 0 || nx >= dim_x) continue;

                        int flat = nx + ny * dim_x + nz * dim_x * dim_y;
                        PhotonBin& b = bins[(size_t)flat * bin_count + k];
                        b.flux   += flux;
                        b.dir_x  += wi_x * flux;
                        b.dir_y  += wi_y * flux;
                        b.dir_z  += wi_z * flux;
                        b.weight += 1.0f;
                        b.count  += 1;
                    }
                }
            }
        }

        // ── 4. Normalize centroid directions ─────────────────────────
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
                }
            }
        }
    }
};
