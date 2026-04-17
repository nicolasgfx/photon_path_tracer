#pragma once
// ─────────────────────────────────────────────────────────────────────
// testing/statistical_tests.h – Statistical validation tests (v5)
//
// Provides:
//   - chi_squared_test: validate PDF sampling against expected PDF
//   - furnace_test: energy conservation (white room → converges to 1)
//   - reciprocity_test: f(wi→wo) == f(wo→wi) for BSDFs
//   - ks_test_uniform: Kolmogorov-Smirnov test for uniform distribution
// ─────────────────────────────────────────────────────────────────────
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

namespace statistical_tests {

// ── Chi-squared PDF test ────────────────────────────────────────────
// Bins N samples from a sampler, compares against expected PDF.
// Returns p-value. Fail if p < significance_level (default 0.01).
//
// bins[i]        = observed count in bin i
// expected[i]    = expected count in bin i (sum should be N)
// num_bins       = number of bins
//
// Returns chi-squared statistic and p-value.

struct ChiSquaredResult {
    double chi2      = 0.0;
    double p_value   = 1.0;
    int    df        = 0;
    bool   passed    = true;
    double threshold = 0.01;  // significance level

    void report(const char* name) const {
        printf("  %s: chi2=%.2f, df=%d, p=%.4f → %s\n",
               name, chi2, df, p_value,
               passed ? "PASS" : "FAIL");
    }
};

// Incomplete gamma function approximation for chi-squared p-value.
// Uses series expansion for regularized lower incomplete gamma.
// Q(a, x) = 1 - P(a, x) where P is regularized lower incomplete gamma.

namespace detail {

inline double log_gamma(double x) {
    // Stirling's approximation + Lanczos coefficients
    static const double c[7] = {
        1.000000000190015, 76.18009172947146, -86.50532032941677,
        24.01409824083091, -1.231739572450155, 0.1208650973866179e-2,
        -0.5395239384953e-5
    };
    double y = x, tmp = x + 5.5;
    tmp -= (x + 0.5) * std::log(tmp);
    double ser = c[0];
    for (int j = 1; j <= 6; ++j) ser += c[j] / ++y;
    return -tmp + std::log(2.5066282746310005 * ser / x);
}

// Regularized lower incomplete gamma via series expansion
inline double gamma_inc_lower_series(double a, double x, int max_iter = 200) {
    if (x < 0.0) return 0.0;
    if (x == 0.0) return 0.0;

    double ap = a;
    double sum = 1.0 / a;
    double del = sum;
    for (int n = 1; n < max_iter; ++n) {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if (std::abs(del) < std::abs(sum) * 1e-10) break;
    }
    return sum * std::exp(-x + a * std::log(x) - log_gamma(a));
}

// Upper incomplete gamma via continued fraction (for large x)
inline double gamma_inc_upper_cf(double a, double x, int max_iter = 200) {
    double b = x + 1.0 - a;
    double c = 1.0 / 1e-30;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i < max_iter; ++i) {
        double an = -(double)i * ((double)i - a);
        b += 2.0;
        d = an * d + b;
        if (std::abs(d) < 1e-30) d = 1e-30;
        c = b + an / c;
        if (std::abs(c) < 1e-30) c = 1e-30;
        d = 1.0 / d;
        double del = d * c;
        h *= del;
        if (std::abs(del - 1.0) < 1e-10) break;
    }
    return std::exp(-x + a * std::log(x) - log_gamma(a)) * h;
}

// Q(a, x) = upper regularized incomplete gamma = 1 - P(a, x)
inline double regularized_gamma_q(double a, double x) {
    if (x < 0.0) return 1.0;
    if (x == 0.0) return 1.0;
    if (x < a + 1.0) {
        return 1.0 - gamma_inc_lower_series(a, x);
    } else {
        return gamma_inc_upper_cf(a, x);
    }
}

} // namespace detail

// Chi-squared p-value: Q(df/2, chi2/2)
inline double chi_squared_pvalue(double chi2, int df) {
    return detail::regularized_gamma_q(df / 2.0, chi2 / 2.0);
}

inline ChiSquaredResult chi_squared_test(
    const int* observed, const double* expected,
    int num_bins, double significance = 0.01)
{
    ChiSquaredResult r;
    r.df = num_bins - 1;
    r.threshold = significance;
    r.chi2 = 0.0;

    for (int i = 0; i < num_bins; ++i) {
        if (expected[i] < 1e-12) continue; // skip empty bins
        double d = (double)observed[i] - expected[i];
        r.chi2 += (d * d) / expected[i];
    }

    r.p_value = chi_squared_pvalue(r.chi2, r.df);
    r.passed = (r.p_value >= significance);
    return r;
}

// Convenience: pass float vectors
inline ChiSquaredResult chi_squared_test(
    const std::vector<int>& observed,
    const std::vector<double>& expected,
    double significance = 0.01)
{
    int n = (int)std::min(observed.size(), expected.size());
    return chi_squared_test(observed.data(), expected.data(), n, significance);
}

// ── Furnace test ────────────────────────────────────────────────────
// A perfectly white lambertian room should converge to radiance = 1.0
// given emittance = 1.0. Checks energy conservation.

struct FurnaceResult {
    float mean_luminance = 0.f;  // measured
    float expected       = 1.f;  // expected
    float tolerance      = 0.05f;
    bool  passed         = true;

    void report(const char* name) const {
        printf("  %s: mean_lum=%.4f, expected=%.4f, tol=%.4f → %s\n",
               name, mean_luminance, expected, tolerance,
               passed ? "PASS" : "FAIL");
    }
};

inline FurnaceResult furnace_test(
    const float* rgb_buffer, int num_pixels,
    float expected = 1.0f, float tolerance = 0.05f)
{
    FurnaceResult r;
    r.expected = expected;
    r.tolerance = tolerance;

    double sum = 0.0;
    for (int i = 0; i < num_pixels; ++i) {
        // Use luminance
        float lum = 0.2126f * rgb_buffer[i * 3]
                   + 0.7152f * rgb_buffer[i * 3 + 1]
                   + 0.0722f * rgb_buffer[i * 3 + 2];
        sum += lum;
    }
    r.mean_luminance = (float)(sum / num_pixels);
    r.passed = std::abs(r.mean_luminance - expected) <= tolerance;
    return r;
}

// ── Reciprocity test ────────────────────────────────────────────────
// Verifies f(wi→wo) ≈ f(wo→wi) for a BSDF evaluation function.
//
// The user provides a callable: float eval(wi, wo) → BSDF value.
// We test a grid of direction pairs.

struct ReciprocityResult {
    float max_relative_error = 0.f;
    float avg_relative_error = 0.f;
    float tolerance = 0.01f;
    int   num_tested = 0;
    int   num_failed = 0;
    bool  passed = true;

    void report(const char* name) const {
        printf("  %s: max_rel_err=%.6f, avg=%.6f, %d/%d passed → %s\n",
               name, max_relative_error, avg_relative_error,
               num_tested - num_failed, num_tested,
               passed ? "PASS" : "FAIL");
    }
};

// General reciprocity test with callable
template<typename EvalFunc>
ReciprocityResult reciprocity_test(
    EvalFunc eval_bsdf,  // float eval(float3 wi, float3 wo)
    int num_pairs = 100,
    float tolerance = 0.01f)
{
    ReciprocityResult r;
    r.tolerance = tolerance;
    r.num_tested = num_pairs;

    double total_err = 0.0;
    // Simple stratified direction pairs on hemisphere
    int side = (int)std::ceil(std::sqrt((double)num_pairs));
    int pair_idx = 0;

    for (int i = 0; i < side && pair_idx < num_pairs; ++i) {
        for (int j = 0; j < side && pair_idx < num_pairs; ++j) {
            // Generate two unit vectors on hemisphere
            float u1 = ((float)i + 0.5f) / side;
            float u2 = ((float)j + 0.5f) / side;

            // Cosine-weighted hemisphere directions
            float phi1 = 2.f * 3.14159265f * u1;
            float cos_theta1 = std::sqrt(u2);
            float sin_theta1 = std::sqrt(1.f - u2);

            float phi2 = 2.f * 3.14159265f * u2;
            float cos_theta2 = std::sqrt(u1);
            float sin_theta2 = std::sqrt(1.f - u1);

            float wi[3] = {
                sin_theta1 * std::cos(phi1),
                sin_theta1 * std::sin(phi1),
                cos_theta1
            };
            float wo[3] = {
                sin_theta2 * std::cos(phi2),
                sin_theta2 * std::sin(phi2),
                cos_theta2
            };

            float fwd = eval_bsdf(wi, wo);
            float rev = eval_bsdf(wo, wi);

            float max_val = std::max(std::abs(fwd), std::abs(rev));
            float err = (max_val > 1e-8f)
                ? std::abs(fwd - rev) / max_val
                : 0.f;

            total_err += err;
            r.max_relative_error = std::max(r.max_relative_error, err);
            if (err > tolerance) ++r.num_failed;

            ++pair_idx;
        }
    }

    r.avg_relative_error = (float)(total_err / r.num_tested);
    r.passed = (r.num_failed == 0);
    return r;
}

// ── Kolmogorov-Smirnov test for uniformity ──────────────────────────
// Tests if samples in [0,1] are uniformly distributed.

struct KSResult {
    double d_statistic = 0.0;
    double threshold   = 0.0;  // critical value
    int    n           = 0;
    bool   passed      = true;

    void report(const char* name) const {
        printf("  %s: D=%.6f, threshold=%.6f, n=%d → %s\n",
               name, d_statistic, threshold, n,
               passed ? "PASS" : "FAIL");
    }
};

inline KSResult ks_test_uniform(
    const float* samples, int count,
    double alpha = 0.05)
{
    KSResult r;
    r.n = count;

    // Sort samples
    std::vector<float> sorted(samples, samples + count);
    std::sort(sorted.begin(), sorted.end());

    // Compute D statistic
    double d_max = 0.0;
    for (int i = 0; i < count; ++i) {
        double fn = (double)(i + 1) / count;  // empirical CDF
        double d_plus  = std::abs(fn - (double)sorted[i]);
        double d_minus = std::abs((double)sorted[i] - (double)i / count);
        d_max = std::max(d_max, std::max(d_plus, d_minus));
    }
    r.d_statistic = d_max;

    // Critical value at alpha=0.05: ~1.36/sqrt(n)
    // For alpha=0.01: ~1.63/sqrt(n)
    double c_alpha = (alpha <= 0.01) ? 1.63 : (alpha <= 0.05) ? 1.36 : 1.22;
    r.threshold = c_alpha / std::sqrt((double)count);
    r.passed = (r.d_statistic <= r.threshold);
    return r;
}

} // namespace statistical_tests
