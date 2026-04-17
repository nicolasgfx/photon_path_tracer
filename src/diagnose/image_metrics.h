#pragma once
// ─────────────────────────────────────────────────────────────────────
// oracle/image_metrics.h – Image quality metrics (v5)
//
// Pure CPU implementations of RMSE, PSNR, SSIM for HDR float and
// LDR uint8 images.  No external dependencies beyond <cmath>.
// ─────────────────────────────────────────────────────────────────────
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

struct ImageMetricsResult {
    float rmse  = 0.f;   // root-mean-square error
    float psnr  = 0.f;   // peak signal-to-noise ratio (dB)
    float ssim  = 0.f;   // structural similarity [0,1]
    int   width  = 0;
    int   height = 0;
};

namespace image_metrics {

// ── Luminance helper ────────────────────────────────────────────────

inline float luminance(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// ── RMSE (float3 buffers, 3 channels) ───────────────────────────────

inline float compute_rmse(const float* a, const float* b, int num_pixels) {
    double sum = 0.0;
    int count = num_pixels * 3;
    for (int i = 0; i < count; ++i) {
        double d = (double)a[i] - (double)b[i];
        sum += d * d;
    }
    return (float)std::sqrt(sum / count);
}

// ── PSNR (from RMSE, with given peak value) ─────────────────────────

inline float compute_psnr(float rmse, float peak = 1.0f) {
    if (rmse < 1e-10f) return 100.f;  // essentially identical
    return 20.f * std::log10(peak / rmse);
}

// ── SSIM (luminance-based, 8×8 windows) ─────────────────────────────
// Wang et al. 2004 with standard constants.
// Operates on single-channel luminance maps.

inline float compute_ssim(const float* lum_a, const float* lum_b,
                           int width, int height)
{
    constexpr int WINDOW = 8;
    constexpr float C1 = 0.01f * 0.01f;  // (K1*L)² with L=1 for normalized
    constexpr float C2 = 0.03f * 0.03f;

    if (width < WINDOW || height < WINDOW) return 1.f;

    double ssim_sum = 0.0;
    int count = 0;

    for (int y = 0; y <= height - WINDOW; y += WINDOW / 2) {
        for (int x = 0; x <= width - WINDOW; x += WINDOW / 2) {
            double mu_a = 0, mu_b = 0;
            int n = WINDOW * WINDOW;

            for (int dy = 0; dy < WINDOW; ++dy) {
                for (int dx = 0; dx < WINDOW; ++dx) {
                    int idx = (y + dy) * width + (x + dx);
                    mu_a += lum_a[idx];
                    mu_b += lum_b[idx];
                }
            }
            mu_a /= n;
            mu_b /= n;

            double sig_a2 = 0, sig_b2 = 0, sig_ab = 0;
            for (int dy = 0; dy < WINDOW; ++dy) {
                for (int dx = 0; dx < WINDOW; ++dx) {
                    int idx = (y + dy) * width + (x + dx);
                    double da = lum_a[idx] - mu_a;
                    double db = lum_b[idx] - mu_b;
                    sig_a2 += da * da;
                    sig_b2 += db * db;
                    sig_ab += da * db;
                }
            }
            sig_a2 /= (n - 1);
            sig_b2 /= (n - 1);
            sig_ab /= (n - 1);

            double num = (2.0 * mu_a * mu_b + C1) * (2.0 * sig_ab + C2);
            double den = (mu_a * mu_a + mu_b * mu_b + C1) * (sig_a2 + sig_b2 + C2);
            ssim_sum += num / den;
            ++count;
        }
    }

    return (count > 0) ? (float)(ssim_sum / count) : 1.f;
}

// ── Convenience: full metrics from RGB float buffers ────────────────

inline ImageMetricsResult compute_metrics(
    const float* img_a, const float* img_b,
    int width, int height)
{
    ImageMetricsResult result;
    result.width = width;
    result.height = height;

    int n = width * height;
    result.rmse = compute_rmse(img_a, img_b, n);
    result.psnr = compute_psnr(result.rmse);

    // Extract luminance for SSIM
    std::vector<float> lum_a(n), lum_b(n);
    for (int i = 0; i < n; ++i) {
        lum_a[i] = luminance(img_a[i * 3], img_a[i * 3 + 1], img_a[i * 3 + 2]);
        lum_b[i] = luminance(img_b[i * 3], img_b[i * 3 + 1], img_b[i * 3 + 2]);
    }
    result.ssim = compute_ssim(lum_a.data(), lum_b.data(), width, height);

    return result;
}

// ── Per-pixel error map (luminance RMSE per pixel) ──────────────────

inline std::vector<float> compute_error_map(
    const float* img_a, const float* img_b,
    int width, int height)
{
    int n = width * height;
    std::vector<float> error(n);
    for (int i = 0; i < n; ++i) {
        float dr = img_a[i * 3 + 0] - img_b[i * 3 + 0];
        float dg = img_a[i * 3 + 1] - img_b[i * 3 + 1];
        float db = img_a[i * 3 + 2] - img_b[i * 3 + 2];
        error[i] = std::sqrt((dr * dr + dg * dg + db * db) / 3.f);
    }
    return error;
}

} // namespace image_metrics
