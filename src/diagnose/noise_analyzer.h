#pragma once
// ─────────────────────────────────────────────────────────────────────
// oracle/noise_analyzer.h – No-reference noise estimation (v5)
//
// Estimates noise level from a single image (no ground truth needed).
// Uses:
//   1. Local variance in smooth regions (Laplacian-based)
//   2. Firefly detection (isolated bright pixels)
//   3. Spatial autocorrelation of noise (low-freq splotch detection)
// ─────────────────────────────────────────────────────────────────────
#include "diagnose/image_metrics.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

struct NoiseEstimate {
    float global_noise_level   = 0.f;  // [0,1] estimated RMS noise
    float laplacian_variance   = 0.f;  // Laplacian-based noise estimate
    int   num_firefly_pixels   = 0;    // isolated bright outliers
    float firefly_fraction     = 0.f;  // fraction of pixels that are fireflies
    float splotch_score        = 0.f;  // [0,1] low-frequency noise indicator
};

namespace noise_analyzer {

// ── Laplacian variance noise estimation ─────────────────────────────
// Immerkaer (1996): noise σ ≈ √(π/2) × (1/6(W-2)(H-2)) × Σ |L(x,y)|
// where L is the discrete Laplacian.  This estimates noise without
// needing a reference image.

inline float estimate_noise_laplacian(const float* luminance,
                                       int width, int height)
{
    if (width < 3 || height < 3) return 0.f;

    double sum = 0.0;
    int count = 0;

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            // 3×3 Laplacian kernel:  [1 -2 1; -2 4 -2; 1 -2 1]
            float lap =
                  1.f * luminance[(y-1) * width + (x-1)]
                - 2.f * luminance[(y-1) * width + x]
                + 1.f * luminance[(y-1) * width + (x+1)]
                - 2.f * luminance[y * width + (x-1)]
                + 4.f * luminance[y * width + x]
                - 2.f * luminance[y * width + (x+1)]
                + 1.f * luminance[(y+1) * width + (x-1)]
                - 2.f * luminance[(y+1) * width + x]
                + 1.f * luminance[(y+1) * width + (x+1)];

            sum += (double)(lap * lap);
            ++count;
        }
    }

    if (count == 0) return 0.f;

    // σ² ≈ (π/2) / (6(W-2)(H-2)) × Σ L²
    // Simplification: σ = √(sum / count) × correction_factor
    double variance = sum / count;
    float sigma = (float)std::sqrt(variance) * 0.20f;  // empirical scaling
    return sigma;
}

// ── Firefly detection ───────────────────────────────────────────────
// A pixel is a firefly if its luminance is > K × local_median.
// Returns per-pixel mask (1 = firefly, 0 = normal).

inline std::vector<uint8_t> detect_fireflies(
    const float* luminance,
    int width, int height,
    float K = 10.f,
    int radius = 2)
{
    int n = width * height;
    std::vector<uint8_t> mask(n, 0);

    int window_size = (2 * radius + 1) * (2 * radius + 1);
    std::vector<float> buf(window_size);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int count = 0;
            for (int dy = -radius; dy <= radius; ++dy) {
                int sy = (std::max)(0, (std::min)(y + dy, height - 1));
                for (int dx = -radius; dx <= radius; ++dx) {
                    int sx = (std::max)(0, (std::min)(x + dx, width - 1));
                    buf[count++] = luminance[sy * width + sx];
                }
            }
            std::sort(buf.begin(), buf.begin() + count);
            float median = buf[count / 2];

            float center = luminance[y * width + x];
            if (median > 1e-6f && center > K * median) {
                mask[y * width + x] = 1;
            }
        }
    }

    return mask;
}

// ── Splotch detection (low-frequency noise) ─────────────────────────
// Detects blocky/splotchy artifacts from under-trained path guides.
// Uses spatial autocorrelation: high autocorrelation at large offsets
// indicates structured (low-freq) noise rather than white noise.

inline float estimate_splotch_score(const float* luminance,
                                     int width, int height)
{
    if (width < 32 || height < 32) return 0.f;

    // Compute mean
    int n = width * height;
    double mean = 0.0;
    for (int i = 0; i < n; ++i) mean += luminance[i];
    mean /= n;

    // Compute variance and autocorrelation at lag = 8 pixels
    double variance = 0.0;
    double autocorr = 0.0;
    int ac_count = 0;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double d = luminance[y * width + x] - mean;
            variance += d * d;

            // Horizontal autocorrelation at lag 8
            if (x + 8 < width) {
                double d2 = luminance[y * width + x + 8] - mean;
                autocorr += d * d2;
                ++ac_count;
            }
        }
    }

    variance /= n;
    if (variance < 1e-12 || ac_count == 0) return 0.f;

    autocorr /= ac_count;
    float rho = (float)(autocorr / variance);

    // For white noise, ρ(8) ≈ 0.  For splotchy noise, ρ(8) is high.
    // Score: clamp to [0, 1]
    return (std::max)(0.f, (std::min)(rho, 1.f));
}

// ── Full noise analysis ─────────────────────────────────────────────

inline NoiseEstimate analyze(const float* rgb_image, int width, int height,
                             float firefly_K = 10.f)
{
    NoiseEstimate result;
    int n = width * height;
    if (n == 0) return result;

    // Extract luminance
    std::vector<float> lum(n);
    for (int i = 0; i < n; ++i) {
        lum[i] = image_metrics::luminance(
            rgb_image[i * 3], rgb_image[i * 3 + 1], rgb_image[i * 3 + 2]);
    }

    result.laplacian_variance = estimate_noise_laplacian(lum.data(), width, height);
    result.global_noise_level = (std::min)(result.laplacian_variance, 1.f);

    auto firefly_mask = detect_fireflies(lum.data(), width, height, firefly_K);
    result.num_firefly_pixels = 0;
    for (auto v : firefly_mask)
        if (v) ++result.num_firefly_pixels;
    result.firefly_fraction = (float)result.num_firefly_pixels / (float)n;

    result.splotch_score = estimate_splotch_score(lum.data(), width, height);

    return result;
}

} // namespace noise_analyzer
