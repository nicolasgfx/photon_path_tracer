#pragma once
// ─────────────────────────────────────────────────────────────────────
// tri_photon_irradiance.h — Per-triangle photon irradiance accumulation
// ─────────────────────────────────────────────────────────────────────
// Build a flat float[num_triangles] buffer that stores the total
// accumulated scalar photon irradiance per scene triangle.  This is
// used by the preview renderer to display a cheap photon heatmap
// (one global-memory read per pixel) without iterating all photons.
//
// Two variants:
//   build_from_tri_ids()   — uses pre-stored per-photon triangle IDs
//   build_from_positions() — brute-force nearest-triangle lookup (slow)
//
// The result does NOT need to be precise — it is purely a debug
// visualisation aid (see proposal in the photon heatmap design doc).
// ─────────────────────────────────────────────────────────────────────

#include "photon/photon.h"
#include "core/config.h"
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────
// build_tri_photon_irradiance()
//
// Accumulate scalar photon flux into a per-triangle irradiance buffer.
//
// Inputs:
//   photons      — PhotonSoA with hero-wavelength flux
//   num_tris     — total scene triangles (size of output)
//
// Returns:
//   std::vector<float> of length num_tris, where [t] = sum of hero
//   flux of all photons deposited on triangle t.  Triangles with no
//   photons have value 0.
//
// If photons.tri_id is empty or undersized, falls back to all-zero.
// ─────────────────────────────────────────────────────────────────────
inline std::vector<float> build_tri_photon_irradiance(
    const PhotonSoA& photons,
    int num_tris)
{
    std::vector<float> irradiance(num_tris > 0 ? (size_t)num_tris : 0, 0.f);

    if (num_tris <= 0) return irradiance;

    const size_t n = photons.size();
    if (n == 0) return irradiance;

    // Require tri_id to be populated
    if (photons.tri_id.size() < n) return irradiance;

    const bool has_hero = (photons.flux.size() >= n * HERO_WAVELENGTHS
                           && photons.num_hero.size() >= n);

    for (size_t i = 0; i < n; ++i) {
        uint32_t t = photons.tri_id[i];
        if (t >= (uint32_t)num_tris) continue;  // out-of-range guard

        float total = 0.f;
        if (has_hero) {
            int nh = (int)photons.num_hero[i];
            if (nh > HERO_WAVELENGTHS) nh = HERO_WAVELENGTHS;
            for (int h = 0; h < nh; ++h)
                total += photons.flux[i * HERO_WAVELENGTHS + h];
        } else {
            // Fallback: sum spectral_flux
            if (photons.spectral_flux.size() >= (i + 1) * NUM_LAMBDA) {
                for (int b = 0; b < NUM_LAMBDA; ++b)
                    total += photons.spectral_flux[i * NUM_LAMBDA + b];
            }
        }

        irradiance[t] += total;
    }

    return irradiance;
}

// ─────────────────────────────────────────────────────────────────────
// build_tri_photon_count()
//
// Count photons per triangle (simpler than irradiance).
// Returns std::vector<uint32_t> of length num_tris.
// ─────────────────────────────────────────────────────────────────────
inline std::vector<uint32_t> build_tri_photon_count(
    const PhotonSoA& photons,
    int num_tris)
{
    std::vector<uint32_t> counts(num_tris > 0 ? (size_t)num_tris : 0, 0u);
    if (num_tris <= 0) return counts;

    const size_t n = photons.size();
    if (n == 0 || photons.tri_id.size() < n) return counts;

    for (size_t i = 0; i < n; ++i) {
        uint32_t t = photons.tri_id[i];
        if (t < (uint32_t)num_tris)
            ++counts[t];
    }

    return counts;
}
