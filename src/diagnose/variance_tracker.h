#pragma once
// ─────────────────────────────────────────────────────────────────────
// diagnostics/variance_tracker.h – Per-pixel variance estimation
//
// Tracks running mean and variance of pixel luminance using Welford's
// online algorithm.  Operates on GPU buffers (lum_sum, lum_sum2,
// sample_counts from LaunchParams §11 / §0).
//
// Provides:
//   - Per-pixel noise map (std-dev / mean)
//   - Image-wide variance statistics
//   - Convergence rate estimation (variance at N vs 2N SPP)
// ─────────────────────────────────────────────────────────────────────
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

struct VarianceStats {
    float mean_variance    = 0.f;   // mean of per-pixel variance
    float max_variance     = 0.f;   // max per-pixel variance
    float median_variance  = 0.f;   // median per-pixel variance
    float mean_relative_noise = 0.f; // mean of (std_dev / mean) across pixels
    int   num_pixels       = 0;
    int   num_noisy_pixels = 0;     // pixels with relative noise > threshold
    float noisy_fraction   = 0.f;   // num_noisy / num_pixels
};

struct ConvergencePoint {
    int   spp;
    float mean_variance;
    float mean_relative_noise;
};

class VarianceTracker {
public:
    // Download GPU buffers and compute statistics.
    //   lum_sum:       [w*h] cumulative luminance (device)
    //   lum_sum2:      [w*h] cumulative luminance² (device)
    //   sample_counts: [w*h] per-pixel sample counts (device)
    //   All pointers may be null — returns zero stats.
    VarianceStats compute_stats(
        const float* d_lum_sum,
        const float* d_lum_sum2,
        const float* d_sample_counts,
        int width, int height,
        float noisy_threshold = 0.1f);

    // Compute per-pixel noise map (relative noise = std_dev / mean).
    // Returns host-side vector [width * height].
    std::vector<float> compute_noise_map(
        const float* d_lum_sum,
        const float* d_lum_sum2,
        const float* d_sample_counts,
        int width, int height);

    // Record a convergence data point for rate estimation.
    void record_convergence_point(int spp, const VarianceStats& stats);

    // Estimate convergence rate.  For unbiased estimators, variance
    // should scale as 1/N → rate ≈ 1.0.  Returns the exponent α
    // in variance ∝ N^(-α).  Returns 0 if insufficient data.
    float estimate_convergence_rate() const;

    // Get all recorded convergence points.
    const std::vector<ConvergencePoint>& convergence_history() const {
        return history_;
    }

    void clear_history() { history_.clear(); }

private:
    std::vector<ConvergencePoint> history_;
};
