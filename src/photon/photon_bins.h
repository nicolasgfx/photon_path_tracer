#pragma once
// ─────────────────────────────────────────────────────────────────────
// photon_bins.h – Directional bin cache for photon density estimation
// ─────────────────────────────────────────────────────────────────────
// Fibonacci sphere binning: N quasi-uniform directions on S² used to
// cache photon flux distribution at each pixel's first diffuse hit.
// Used for guided BSDF bounce and cached density estimation.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/config.h"
#include "core/spectrum.h"    // NUM_LAMBDA

// ── Per-bin data (GPU cache) ────────────────────────────────────────
// Per-wavelength flux preserves spectral fidelity.  Each hero wavelength's
// Epanechnikov-weighted flux is deposited into its own spectral bin during
// the CPU build, matching the hash-grid path's per-wavelength accumulation.
// Size: (NUM_LAMBDA + 8) * 4 + 4 = 52 bytes per bin (NUM_LAMBDA=4).
// Memory: 1000 cells × 32 bins × 52 B ≈ 1.6 MB.
struct PhotonBin {
    float flux[NUM_LAMBDA]; // per-wavelength Epanechnikov-weighted flux
    float scalar_flux;// sum of flux[] — used by guided bounce/NEE as importance weight
    float dir_x;      // flux-weighted centroid direction x
    float dir_y;      // flux-weighted centroid direction y
    float dir_z;      // flux-weighted centroid direction z
    float weight;     // total Epanechnikov weight (for normalization)
    int   count;      // number of photons accumulated in this bin
    // Flux-weighted average surface normal of the photons in this bin.
    // Used as a visibility term: reject contributions whose deposited-surface
    // normal faces away from the query-point normal (photons through walls).
    float avg_nx;     // average surface normal x
    float avg_ny;     // average surface normal y
    float avg_nz;     // average surface normal z
};

// ── Compact GPU bin (16 bytes) ──────────────────────────────────────
// Subset of PhotonBin uploaded to the GPU for guided direction sampling.
// Contains only the fields read by dev_read_cell_histogram():
//   scalar_flux — importance weight for inverse-CDF sampling
//   avg_n{x,y,z} — per-bin average normal for the hemisphere gate
struct GpuGuideBin {
    float scalar_flux;
    float avg_nx, avg_ny, avg_nz;
};

// ── Fibonacci sphere bin directions ─────────────────────────────────
// Quasi-uniform distribution of N points on S².
// Stored in a fixed-size array (MAX_PHOTON_BIN_COUNT upper bound).

struct PhotonBinDirs {
    float3 dirs[MAX_PHOTON_BIN_COUNT];
    int    count;

    /// Compute Fibonacci sphere directions for n bins.
    HD void init(int n) {
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

    /// Find nearest bin for direction wi (brute-force dot product scan).
    /// O(N) for N ≤ MAX_PHOTON_BIN_COUNT — cheaper than any spatial index for this size.
    HD int find_nearest(float3 wi) const {
        int   best     = 0;
        float best_dot = -2.0f;
        for (int k = 0; k < count; ++k) {
            float d = dot(wi, dirs[k]);
            if (d > best_dot) { best_dot = d; best = k; }
        }
        return best;
    }
};
