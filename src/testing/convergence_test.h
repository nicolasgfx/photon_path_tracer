#pragma once
// ─────────────────────────────────────────────────────────────────────
// testing/convergence_test.h – MSE-at-N-SPP regression tests (v5)
//
// Provides:
//   - ConvergenceBaseline: stored expected MSE at multiple SPP levels
//   - ConvergenceResult: measured convergence rate + pass/fail
//   - measure_convergence: renders at [16,64,256,1024] SPP, fits rate
//   - compare_to_baseline: checks measured vs stored baseline
//   - save/load_baseline: persistence for convergence baselines
// ─────────────────────────────────────────────────────────────────────
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace convergence_test {

// ── Standard SPP levels for convergence testing ─────────────────────

static const int DEFAULT_SPP_LEVELS[] = { 16, 64, 256, 1024 };
static const int DEFAULT_NUM_LEVELS   = 4;

// ── Convergence baseline ────────────────────────────────────────────

struct ConvergenceBaseline {
    std::string scene_name;
    std::vector<int>   spp_levels;   // e.g., [16, 64, 256, 1024]
    std::vector<float> mse_values;   // MSE at each level
    float convergence_rate = 0.f;    // slope of log-log MSE vs SPP

    bool valid() const {
        return !scene_name.empty()
            && spp_levels.size() == mse_values.size()
            && spp_levels.size() >= 2;
    }
};

// ── Log-log linear regression ───────────────────────────────────────
// Fits log(MSE) = a + b * log(SPP).  b is the convergence rate.
// Returns slope b (should be negative, typically around -1 for MC).

struct LinFitResult {
    float slope     = 0.f;
    float intercept = 0.f;
    float r_squared = 0.f;  // goodness of fit
};

inline LinFitResult log_log_fit(
    const int* spp, const float* mse, int n)
{
    LinFitResult r;
    if (n < 2) return r;

    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int valid = 0;

    for (int i = 0; i < n; ++i) {
        if (mse[i] <= 0.f) continue;  // skip zero MSE
        double x = std::log((double)spp[i]);
        double y = std::log((double)mse[i]);
        sx  += x;
        sy  += y;
        sxx += x * x;
        sxy += x * y;
        ++valid;
    }

    if (valid < 2) return r;

    double nf = (double)valid;
    double denom = nf * sxx - sx * sx;
    if (std::abs(denom) < 1e-30) return r;

    r.slope     = (float)((nf * sxy - sx * sy) / denom);
    r.intercept = (float)((sy - r.slope * sx) / nf);

    // R^2
    double mean_y = sy / nf;
    double ss_tot = 0, ss_res = 0;
    for (int i = 0; i < n; ++i) {
        if (mse[i] <= 0.f) continue;
        double x = std::log((double)spp[i]);
        double y = std::log((double)mse[i]);
        double y_pred = r.intercept + r.slope * x;
        ss_tot += (y - mean_y) * (y - mean_y);
        ss_res += (y - y_pred) * (y - y_pred);
    }
    r.r_squared = (ss_tot > 1e-30f)
        ? (float)(1.0 - ss_res / ss_tot)
        : 0.f;

    return r;
}

// ── Convergence test result ─────────────────────────────────────────

struct ConvergenceResult {
    std::string scene_name;
    std::vector<int>   spp_levels;
    std::vector<float> measured_mse;
    float measured_rate    = 0.f;  // slope from log-log fit
    float baseline_rate   = 0.f;  // expected rate
    float r_squared       = 0.f;
    float rate_tolerance   = 0.15f;  // allow 15% worse rate
    bool  rate_ok          = true;
    bool  all_mse_ok       = true;
    bool  passed           = true;

    void report() const {
        printf("  Convergence [%s]:\n", scene_name.c_str());
        for (size_t i = 0; i < spp_levels.size(); ++i) {
            printf("    SPP %5d: MSE=%.6f\n",
                   spp_levels[i], measured_mse[i]);
        }
        printf("    Rate: measured=%.3f, baseline=%.3f (R²=%.3f)\n",
               measured_rate, baseline_rate, r_squared);
        printf("    → %s\n", passed ? "PASS" : "FAIL");
    }
};

// ── Compare measured convergence against baseline ───────────────────

inline ConvergenceResult compare_to_baseline(
    const std::vector<int>&   spp_levels,
    const std::vector<float>& measured_mse,
    const ConvergenceBaseline& baseline,
    float rate_tolerance = 0.15f,
    float mse_scale_tolerance = 3.0f)
{
    ConvergenceResult r;
    r.scene_name    = baseline.scene_name;
    r.spp_levels    = spp_levels;
    r.measured_mse  = measured_mse;
    r.rate_tolerance = rate_tolerance;

    int n = (int)spp_levels.size();

    // Fit convergence rate from measured data
    LinFitResult fit = log_log_fit(
        spp_levels.data(), measured_mse.data(), n);
    r.measured_rate = fit.slope;
    r.r_squared     = fit.r_squared;

    // Baseline rate
    LinFitResult base_fit = log_log_fit(
        baseline.spp_levels.data(), baseline.mse_values.data(),
        (int)baseline.spp_levels.size());
    r.baseline_rate = base_fit.slope;

    // Rate check: measured rate should not be worse (less negative) by
    // more than rate_tolerance fraction of baseline rate.
    // e.g., baseline = -0.95, tolerance = 0.15
    //   measured must be <= -0.95 * (1 - 0.15) = -0.8075
    if (std::abs(r.baseline_rate) > 1e-6f) {
        float worst_allowed = r.baseline_rate * (1.f - rate_tolerance);
        // Rate is negative: "worse" means closer to 0
        r.rate_ok = (r.measured_rate <= worst_allowed);
    }

    // Per-level MSE check: each level shouldn't be much worse
    r.all_mse_ok = true;
    for (size_t i = 0; i < spp_levels.size(); ++i) {
        // Find matching baseline level
        for (size_t j = 0; j < baseline.spp_levels.size(); ++j) {
            if (baseline.spp_levels[j] == spp_levels[i]) {
                if (measured_mse[i] > baseline.mse_values[j] * mse_scale_tolerance) {
                    r.all_mse_ok = false;
                }
                break;
            }
        }
    }

    r.passed = r.rate_ok && r.all_mse_ok;
    return r;
}

// ── Save baseline to file ───────────────────────────────────────────
// Format: name\n spp1 mse1\n spp2 mse2\n ...

inline bool save_baseline(const ConvergenceBaseline& b,
                           const std::string& filepath)
{
    std::ofstream ofs(filepath);
    if (!ofs) return false;

    ofs << b.scene_name << "\n";
    for (size_t i = 0; i < b.spp_levels.size(); ++i) {
        ofs << b.spp_levels[i] << " " << b.mse_values[i] << "\n";
    }
    return ofs.good();
}

// ── Load baseline from file ─────────────────────────────────────────

inline bool load_baseline(const std::string& filepath,
                           ConvergenceBaseline& out)
{
    std::ifstream ifs(filepath);
    if (!ifs) return false;

    std::getline(ifs, out.scene_name);
    out.spp_levels.clear();
    out.mse_values.clear();

    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        int spp;
        float mse;
        if (iss >> spp >> mse) {
            out.spp_levels.push_back(spp);
            out.mse_values.push_back(mse);
        }
    }

    // Compute rate from loaded data
    LinFitResult fit = log_log_fit(
        out.spp_levels.data(), out.mse_values.data(),
        (int)out.spp_levels.size());
    out.convergence_rate = fit.slope;

    return out.valid();
}

// ── Create baseline from measurements ───────────────────────────────

inline ConvergenceBaseline create_baseline(
    const std::string& scene_name,
    const std::vector<int>& spp_levels,
    const std::vector<float>& mse_values)
{
    ConvergenceBaseline b;
    b.scene_name  = scene_name;
    b.spp_levels  = spp_levels;
    b.mse_values  = mse_values;

    LinFitResult fit = log_log_fit(
        spp_levels.data(), mse_values.data(), (int)spp_levels.size());
    b.convergence_rate = fit.slope;

    return b;
}

} // namespace convergence_test
