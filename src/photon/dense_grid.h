#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>

struct PhotonSoA;  // forward declaration

struct DenseGridData {
    float min_x = 0.f, min_y = 0.f, min_z = 0.f;
    float cell_size = 0.025f;
    int dim_x = 0, dim_y = 0, dim_z = 0;
    std::vector<uint32_t> sorted_indices; // photon indices sorted by cell
    std::vector<uint32_t> cell_start;     // [total_cells] first photon index
    std::vector<uint32_t> cell_end;       // [total_cells] one-past-last

    int total_cells() const { return dim_x * dim_y * dim_z; }

    int cell_index(float x, float y, float z) const {
        int ix = (int)std::floor((x - min_x) / cell_size);
        int iy = (int)std::floor((y - min_y) / cell_size);
        int iz = (int)std::floor((z - min_z) / cell_size);
        ix = std::max(0, std::min(ix, dim_x - 1));
        iy = std::max(0, std::min(iy, dim_y - 1));
        iz = std::max(0, std::min(iz, dim_z - 1));
        return ix + iy * dim_x + iz * dim_x * dim_y;
    }
};

/// Build a dense grid from photon positions.
/// cell_size should be DENSE_GRID_CELL_SIZE (0.025 m).
inline DenseGridData build_dense_grid(
    const float* pos_x, const float* pos_y, const float* pos_z,
    int n, float cell_size)
{
    DenseGridData g;
    g.cell_size = cell_size;

    if (n <= 0) {
        g.min_x = g.min_y = g.min_z = 0.f;
        g.dim_x = g.dim_y = g.dim_z = 0;
        return g;
    }

    // Compute AABB with half-cell padding
    float pad = cell_size * 0.5f;
    g.min_x = g.min_y = g.min_z =  1e30f;
    float max_x = -1e30f, max_y = -1e30f, max_z = -1e30f;
    for (int i = 0; i < n; ++i) {
        if (pos_x[i] < g.min_x) g.min_x = pos_x[i];
        if (pos_y[i] < g.min_y) g.min_y = pos_y[i];
        if (pos_z[i] < g.min_z) g.min_z = pos_z[i];
        if (pos_x[i] > max_x) max_x = pos_x[i];
        if (pos_y[i] > max_y) max_y = pos_y[i];
        if (pos_z[i] > max_z) max_z = pos_z[i];
    }
    g.min_x -= pad; g.min_y -= pad; g.min_z -= pad;
    max_x += pad; max_y += pad; max_z += pad;

    g.dim_x = std::max(1, (int)std::ceil((max_x - g.min_x) / cell_size));
    g.dim_y = std::max(1, (int)std::ceil((max_y - g.min_y) / cell_size));
    g.dim_z = std::max(1, (int)std::ceil((max_z - g.min_z) / cell_size));

    int total = g.total_cells();

    // Assign each photon to a cell, count per cell
    std::vector<uint32_t> cell_ids(n);
    std::vector<uint32_t> counts(total, 0);
    for (int i = 0; i < n; ++i) {
        cell_ids[i] = (uint32_t)g.cell_index(pos_x[i], pos_y[i], pos_z[i]);
        counts[cell_ids[i]]++;
    }

    // Prefix sum -> cell_start
    g.cell_start.resize(total);
    g.cell_end.resize(total);
    uint32_t offset = 0;
    for (int c = 0; c < total; ++c) {
        g.cell_start[c] = offset;
        offset += counts[c];
        g.cell_end[c] = offset;
    }

    // Scatter photon indices into sorted order
    g.sorted_indices.resize(n);
    std::vector<uint32_t> write_pos = g.cell_start; // copy
    for (int i = 0; i < n; ++i) {
        uint32_t c = cell_ids[i];
        g.sorted_indices[write_pos[c]++] = (uint32_t)i;
    }

    return g;
}

// Convenience overload: build from PhotonSoA (defined in photon.h)
#include "photon/photon.h"
inline DenseGridData build_dense_grid(const PhotonSoA& photons, float cell_size) {
    return build_dense_grid(
        photons.pos_x.data(), photons.pos_y.data(), photons.pos_z.data(),
        (int)photons.size(), cell_size);
}
