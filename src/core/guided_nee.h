#pragma once
// ─────────────────────────────────────────────────────────────────────
// guided_nee.h – Small helpers for guided NEE (host + device)
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/photon_bins.h"

inline HD bool guided_nee_should_fallback(
    int num_emissive,
    int max_emissive,
    float total_bin_flux)
{
    return (num_emissive <= 0)
        || (num_emissive > max_emissive)
        || !(total_bin_flux > 0.0f);
}

// Returns normalized bin boost in [0,1] for a direction wi.
// - Only boosts directions in the positive hemisphere of `normal`.
// - If total_bin_flux <= 0 or N <= 0, returns 0.
inline HD float guided_nee_bin_boost(
    float3 wi,
    float3 normal,
    const PhotonBin* bins,
    int N,
    const PhotonBinDirs& bin_dirs,
    float total_bin_flux)
{
    if (!(total_bin_flux > 0.0f) || N <= 0 || bins == nullptr) return 0.0f;
    if (dot(wi, normal) <= 0.0f) return 0.0f;
    int k = bin_dirs.find_nearest(wi);
    if (k < 0) k = 0;
    if (k >= N) k = N - 1;
    float b = bins[k].flux / total_bin_flux;
    if (b < 0.0f) b = 0.0f;
    if (b > 1.0f) b = 1.0f;
    return b;
}

inline HD float guided_nee_weight(float p_orig, float bin_boost, float alpha) {
    // Mirrors device: w = p_orig * (1 + alpha * boost)
    return p_orig * (1.0f + alpha * bin_boost);
}
