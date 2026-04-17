#pragma once
// ─────────────────────────────────────────────────────────────────────
// testing/test_framework.h – Shared test utilities (v5)
//
// Provides:
//   - Test macros (BEGIN/PASS/EXPECT) with counters
//   - Image comparison helpers (float RGB → RMSE, PSNR, SSIM)
//   - Buffer statistics (mean, max, NaN/inf detection)
//   - Test result tracking and reporting
// ─────────────────────────────────────────────────────────────────────
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <functional>

// ── Test counters (global per compilation unit) ─────────────────────

struct TestCounters {
    int tests_run     = 0;
    int tests_passed  = 0;
    int tests_failed  = 0;

    void begin(const char* name) {
        ++tests_run;
        printf("\n[TEST %d] %s\n", tests_run, name);
    }
    void pass(const char* name) {
        ++tests_passed;
        printf("[PASS] %s\n", name);
    }
    void fail(const char* name, const char* cond) {
        ++tests_failed;
        printf("[FAIL] %s: %s\n", name, cond);
    }
    int report(const char* suite_name) const {
        printf("\n========================================\n");
        printf(" %s: %d / %d passed", suite_name, tests_passed, tests_run);
        if (tests_failed > 0) printf(" (%d FAILED)", tests_failed);
        printf("\n========================================\n");
        return (tests_failed == 0) ? 0 : 1;
    }
};

// ── Macros (use a TestCounters instance) ────────────────────────────

#define TF_BEGIN(counters, name) (counters).begin(name)
#define TF_PASS(counters, name) (counters).pass(name)

#define TF_EXPECT(counters, cond, msg)                   \
    do {                                                  \
        if (!(cond)) {                                    \
            (counters).fail(msg, #cond);                  \
            return;                                       \
        }                                                 \
    } while(0)

// ── Buffer statistics ───────────────────────────────────────────────

namespace test_utils {

struct BufferStats {
    float mean   = 0.f;
    float max_val = 0.f;
    float min_val = 0.f;
    int   nan_count = 0;
    int   inf_count = 0;
    int   zero_count = 0;
    int   total = 0;
    bool  has_nan() const { return nan_count > 0; }
    bool  has_inf() const { return inf_count > 0; }
};

inline BufferStats compute_buffer_stats(const float* data, int count) {
    BufferStats s;
    s.total = count;
    if (count == 0) return s;

    double sum = 0.0;
    s.min_val = data[0];
    s.max_val = data[0];

    for (int i = 0; i < count; ++i) {
        float v = data[i];
        if (std::isnan(v)) { ++s.nan_count; continue; }
        if (std::isinf(v)) { ++s.inf_count; continue; }
        if (v == 0.f) ++s.zero_count;
        sum += v;
        s.min_val = std::min(s.min_val, v);
        s.max_val = std::max(s.max_val, v);
    }

    int valid = count - s.nan_count - s.inf_count;
    s.mean = (valid > 0) ? (float)(sum / valid) : 0.f;
    return s;
}

// ── Luminance helper ────────────────────────────────────────────────

inline float luminance(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// ── RMSE between two float3 RGB buffers ─────────────────────────────

inline float compute_rmse(const float* a, const float* b, int num_pixels) {
    double sum = 0.0;
    int count = num_pixels * 3;
    for (int i = 0; i < count; ++i) {
        double d = (double)a[i] - (double)b[i];
        sum += d * d;
    }
    return (float)std::sqrt(sum / count);
}

// ── Mean luminance of a float3 RGB buffer ───────────────────────────

inline float mean_luminance(const float* rgb, int num_pixels) {
    double sum = 0.0;
    for (int i = 0; i < num_pixels; ++i) {
        sum += luminance(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
    }
    return (float)(sum / num_pixels);
}

// ── Check if buffer contains any NaN or Inf values ──────────────────

inline bool buffer_is_clean(const float* data, int count) {
    for (int i = 0; i < count; ++i) {
        if (std::isnan(data[i]) || std::isinf(data[i]))
            return false;
    }
    return true;
}

// ── Check if buffer has any non-zero values ─────────────────────────

inline bool buffer_has_nonzero(const float* data, int count,
                                float threshold = 1e-10f) {
    for (int i = 0; i < count; ++i) {
        if (std::abs(data[i]) > threshold) return true;
    }
    return false;
}

// ── Divide RGB accumulator by sample count ──────────────────────────
// Converts raw accumulator to average (like SPP normalization).

inline std::vector<float> normalize_by_spp(
    const std::vector<float>& rgb_accum, int spp)
{
    std::vector<float> result(rgb_accum.size());
    float inv = 1.f / (float)spp;
    for (size_t i = 0; i < rgb_accum.size(); ++i)
        result[i] = rgb_accum[i] * inv;
    return result;
}

} // namespace test_utils
