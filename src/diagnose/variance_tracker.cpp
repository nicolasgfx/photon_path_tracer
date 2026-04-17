// ─────────────────────────────────────────────────────────────────────
// diagnostics/variance_tracker.cpp – Per-pixel variance estimation
// ─────────────────────────────────────────────────────────────────────
#include "diagnose/variance_tracker.h"
#include <cuda_runtime.h>
#include <cmath>

VarianceStats VarianceTracker::compute_stats(
    const float* d_lum_sum,
    const float* d_lum_sum2,
    const float* d_sample_counts,
    int width, int height,
    float noisy_threshold)
{
    VarianceStats stats;
    int n = width * height;
    stats.num_pixels = n;

    if (!d_lum_sum || !d_lum_sum2 || !d_sample_counts || n == 0)
        return stats;

    // Download from GPU
    std::vector<float> lum_sum(n), lum_sum2(n), counts(n);
    cudaMemcpy(lum_sum.data(), d_lum_sum, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(lum_sum2.data(), d_lum_sum2, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(counts.data(), d_sample_counts, n * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> variances(n);
    double total_var = 0.0;
    double total_rel_noise = 0.0;
    int valid_count = 0;
    int noisy_count = 0;

    for (int i = 0; i < n; ++i) {
        float c = counts[i];
        if (c < 2.f) {
            variances[i] = 0.f;
            continue;
        }

        float mean = lum_sum[i] / c;
        float var = (lum_sum2[i] / c) - (mean * mean);
        var = std::max(var, 0.f);
        variances[i] = var;

        total_var += var;
        stats.max_variance = std::max(stats.max_variance, var);
        ++valid_count;

        if (mean > 1e-6f) {
            float rel_noise = std::sqrt(var) / mean;
            total_rel_noise += rel_noise;
            if (rel_noise > noisy_threshold)
                ++noisy_count;
        }
    }

    if (valid_count > 0) {
        stats.mean_variance = (float)(total_var / valid_count);
        stats.mean_relative_noise = (float)(total_rel_noise / valid_count);

        // Median variance
        std::vector<float> sorted_var(variances.begin(), variances.end());
        std::sort(sorted_var.begin(), sorted_var.end());
        stats.median_variance = sorted_var[n / 2];
    }

    stats.num_noisy_pixels = noisy_count;
    stats.noisy_fraction = (n > 0) ? (float)noisy_count / (float)n : 0.f;

    return stats;
}

std::vector<float> VarianceTracker::compute_noise_map(
    const float* d_lum_sum,
    const float* d_lum_sum2,
    const float* d_sample_counts,
    int width, int height)
{
    int n = width * height;
    std::vector<float> noise_map(n, 0.f);

    if (!d_lum_sum || !d_lum_sum2 || !d_sample_counts || n == 0)
        return noise_map;

    std::vector<float> lum_sum(n), lum_sum2(n), counts(n);
    cudaMemcpy(lum_sum.data(), d_lum_sum, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(lum_sum2.data(), d_lum_sum2, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(counts.data(), d_sample_counts, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) {
        float c = counts[i];
        if (c < 2.f) continue;

        float mean = lum_sum[i] / c;
        float var = (lum_sum2[i] / c) - (mean * mean);
        var = std::max(var, 0.f);

        if (mean > 1e-6f)
            noise_map[i] = std::sqrt(var) / mean;
    }

    return noise_map;
}

void VarianceTracker::record_convergence_point(int spp, const VarianceStats& stats) {
    history_.push_back({spp, stats.mean_variance, stats.mean_relative_noise});
}

float VarianceTracker::estimate_convergence_rate() const {
    if (history_.size() < 2) return 0.f;

    // Fit log(variance) vs log(spp) using least squares.
    // variance ∝ N^(-α) → log(var) = -α·log(N) + C
    // We want α.
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
    int valid = 0;

    for (const auto& pt : history_) {
        if (pt.mean_variance <= 0.f || pt.spp <= 0) continue;
        double x = std::log((double)pt.spp);
        double y = std::log((double)pt.mean_variance);
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
        ++valid;
    }

    if (valid < 2) return 0.f;

    double n = (double)valid;
    double denom = n * sum_x2 - sum_x * sum_x;
    if (std::abs(denom) < 1e-12) return 0.f;

    double slope = (n * sum_xy - sum_x * sum_y) / denom;
    return (float)(-slope);  // α = -slope (we expect positive α ≈ 1.0)
}
