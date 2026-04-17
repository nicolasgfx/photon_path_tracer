#pragma once
// ─────────────────────────────────────────────────────────────────────
// prepass_metrics.h – View-dependent pre-pass analysis results
//
// Computed from a short unguided brute-force path-tracing pre-pass
// at quarter resolution.  Feeds back into SceneProfile refinement
// and render parameter tuning.
// ─────────────────────────────────────────────────────────────────────
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "stb_image_write.h"

// ── Pre-pass metrics ────────────────────────────────────────────────

struct PrePassMetrics {
    // Per-pixel variance statistics
    float mean_pixel_variance = 0.f;
    float max_pixel_variance  = 0.f;
    float variance_p90        = 0.f;    // 90th percentile
    float variance_p99        = 0.f;    // 99th percentile

    // Path transport statistics (from GPU atomic counters)
    float nee_hit_rate        = 0.f;    // nee_hits / nee_attempts
    float zero_path_fraction  = 0.f;    // zero_paths / total_paths
    float avg_bounce_depth    = 0.f;    // bounce_sum / total_paths

    // Raw counters
    uint32_t nee_attempts     = 0;
    uint32_t nee_hits         = 0;
    uint32_t zero_paths       = 0;
    uint32_t total_paths      = 0;
    uint32_t bounce_sum       = 0;

    // Pre-pass parameters
    int prepass_spp    = 0;
    int prepass_width  = 0;
    int prepass_height = 0;
    float time_ms      = 0.f;
};

// ── Compute metrics from downloaded GPU buffers ─────────────────────

inline PrePassMetrics compute_prepass_metrics(
    const float* lum_sum,       // [num_pixels]
    const float* lum_sum2,      // [num_pixels]
    int          num_pixels,
    int          spp,
    uint32_t     nee_attempts,
    uint32_t     nee_hits,
    uint32_t     zero_paths,
    uint32_t     bounce_sum,
    uint32_t     total_paths)
{
    PrePassMetrics m;
    m.nee_attempts = nee_attempts;
    m.nee_hits     = nee_hits;
    m.zero_paths   = zero_paths;
    m.total_paths  = total_paths;
    m.bounce_sum   = bounce_sum;
    m.prepass_spp  = spp;

    // Derived rates
    m.nee_hit_rate       = nee_attempts > 0 ? (float)nee_hits / (float)nee_attempts : 0.f;
    m.zero_path_fraction = total_paths  > 0 ? (float)zero_paths / (float)total_paths : 0.f;
    m.avg_bounce_depth   = total_paths  > 0 ? (float)bounce_sum / (float)total_paths : 0.f;

    // Per-pixel variance: var = E[X²] - E[X]² = lum_sum2/spp - (lum_sum/spp)²
    if (num_pixels <= 0 || spp <= 0) return m;

    float inv_spp = 1.f / (float)spp;
    std::vector<float> var(num_pixels);
    double var_sum = 0.0;
    float var_max = 0.f;

    for (int i = 0; i < num_pixels; ++i) {
        float mean = lum_sum[i] * inv_spp;
        float mean_sq = lum_sum2[i] * inv_spp;
        float v = mean_sq - mean * mean;
        if (v < 0.f) v = 0.f;          // numerical safety
        var[i] = v;
        var_sum += (double)v;
        if (v > var_max) var_max = v;
    }

    m.mean_pixel_variance = (float)(var_sum / num_pixels);
    m.max_pixel_variance  = var_max;

    // Percentiles via nth_element (O(n) average)
    int idx_p90 = (int)((float)num_pixels * 0.90f);
    int idx_p99 = (int)((float)num_pixels * 0.99f);
    idx_p90 = (std::min)(idx_p90, num_pixels - 1);
    idx_p99 = (std::min)(idx_p99, num_pixels - 1);

    std::nth_element(var.begin(), var.begin() + idx_p90, var.end());
    m.variance_p90 = var[idx_p90];

    std::nth_element(var.begin() + idx_p90, var.begin() + idx_p99, var.end());
    m.variance_p99 = var[idx_p99];

    return m;
}

// ── Save variance heatmap as PNG ────────────────────────────────────

inline bool save_variance_heatmap(const std::string& path,
                                  const float* lum_sum,
                                  const float* lum_sum2,
                                  int width, int height, int spp) {
    int num_pixels = width * height;
    if (num_pixels <= 0 || spp <= 0) return false;

    float inv_spp = 1.f / (float)spp;

    // Compute per-pixel variance
    std::vector<float> var(num_pixels);
    float var_max = 0.f;
    for (int i = 0; i < num_pixels; ++i) {
        float mean = lum_sum[i] * inv_spp;
        float mean_sq = lum_sum2[i] * inv_spp;
        float v = mean_sq - mean * mean;
        if (v < 0.f) v = 0.f;
        var[i] = v;
        if (v > var_max) var_max = v;
    }

    // Blue→Red heatmap (RGBA8)
    std::vector<uint8_t> rgba(num_pixels * 4);
    float inv_max = var_max > 1e-8f ? 1.f / var_max : 0.f;
    for (int i = 0; i < num_pixels; ++i) {
        float t = std::sqrt(var[i] * inv_max);   // sqrt for perceptual scaling
        t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
        // Blue (0,0,1) → Red (1,0,0)
        uint8_t r = (uint8_t)(t * 255.f);
        uint8_t g = (uint8_t)((t < 0.5f ? t * 2.f : (1.f - t) * 2.f) * 255.f);
        uint8_t b = (uint8_t)((1.f - t) * 255.f);
        rgba[i * 4 + 0] = r;
        rgba[i * 4 + 1] = g;
        rgba[i * 4 + 2] = b;
        rgba[i * 4 + 3] = 255;
    }

    // stbi_write_png is provided by stb_image_write.h
    return stbi_write_png(path.c_str(), width, height, 4, rgba.data(),
                          width * 4) != 0;
}

// ── Save pre-pass metrics JSON ──────────────────────────────────────

inline bool save_prepass_json(const std::string& path,
                              const PrePassMetrics& m) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return false;

    std::fprintf(f,
        "{\n"
        "  \"prepass\": {\n"
        "    \"spp\": %d,\n"
        "    \"width\": %d,\n"
        "    \"height\": %d,\n"
        "    \"time_ms\": %.1f,\n"
        "    \"nee_hit_rate\": %.4f,\n"
        "    \"zero_path_fraction\": %.4f,\n"
        "    \"avg_bounce_depth\": %.2f,\n"
        "    \"mean_variance\": %.6f,\n"
        "    \"max_variance\": %.6f,\n"
        "    \"variance_p90\": %.6f,\n"
        "    \"variance_p99\": %.6f,\n"
        "    \"nee_attempts\": %u,\n"
        "    \"nee_hits\": %u,\n"
        "    \"zero_paths\": %u,\n"
        "    \"total_paths\": %u,\n"
        "    \"bounce_sum\": %u\n"
        "  }\n"
        "}\n",
        m.prepass_spp, m.prepass_width, m.prepass_height,
        m.time_ms,
        m.nee_hit_rate, m.zero_path_fraction, m.avg_bounce_depth,
        m.mean_pixel_variance, m.max_pixel_variance,
        m.variance_p90, m.variance_p99,
        m.nee_attempts, m.nee_hits, m.zero_paths, m.total_paths, m.bounce_sum);

    fclose(f);
    return true;
}
