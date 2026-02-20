#pragma once
// ─────────────────────────────────────────────────────────────────────
// photon_density_cache.h – Cache hit/write gating for photon density
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"

inline HD bool should_read_photon_density_cache(
    bool use_bins,
    int bounce,
    const float* photon_density_cache,
    int frame_number)
{
    return use_bins
        && (bounce == 0)
        && (photon_density_cache != nullptr)
        && (frame_number > 0);
}

inline HD bool should_write_photon_density_cache(
    bool use_bins,
    int bounce,
    float* photon_density_cache,
    int frame_number)
{
    return use_bins
        && (bounce == 0)
        && (photon_density_cache != nullptr)
        && (frame_number == 0);
}
