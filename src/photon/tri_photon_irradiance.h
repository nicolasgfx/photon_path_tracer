#pragma once
// ─────────────────────────────────────────────────────────────────────
// tri_photon_irradiance.h – Per-triangle photon irradiance heatmap
// ─────────────────────────────────────────────────────────────────────
// Builds a scalar irradiance value per triangle from the photon map.
// Used for preview-mode heatmap overlays in OptiX rendering.
// ─────────────────────────────────────────────────────────────────────
#include "photon/photon.h"
#include <vector>
#include <algorithm>
#include <cstdint>

/// Build per-triangle accumulated scalar irradiance from photon data.
/// Returns a vector of length num_triangles (one float per triangle).
inline std::vector<float> build_tri_photon_irradiance(
    const PhotonSoA& photons, int num_triangles)
{
    if (num_triangles <= 0 || photons.size() == 0)
        return {};

    std::vector<float> irradiance((size_t)num_triangles, 0.f);

    for (size_t i = 0; i < photons.size(); ++i) {
        // Use tri_id if available
        if (i < photons.tri_id.size()) {
            uint32_t tid = photons.tri_id[i];
            if (tid < (uint32_t)num_triangles) {
                // Accumulate total flux across all wavelengths
                float total = 0.f;
                if (i * NUM_LAMBDA + NUM_LAMBDA <= photons.spectral_flux.size()) {
                    for (int b = 0; b < NUM_LAMBDA; ++b)
                        total += photons.spectral_flux[i * NUM_LAMBDA + b];
                } else if (i * HERO_WAVELENGTHS < photons.flux.size()) {
                    // Fall back to hero wavelength flux
                    for (int h = 0; h < HERO_WAVELENGTHS && i * HERO_WAVELENGTHS + h < photons.flux.size(); ++h)
                        total += photons.flux[i * HERO_WAVELENGTHS + h];
                }
                irradiance[tid] += total;
            }
        }
    }

    return irradiance;
}
