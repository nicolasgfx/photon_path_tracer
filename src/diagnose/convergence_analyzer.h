#pragma once
// ─────────────────────────────────────────────────────────────────────
// diagnostics/convergence_analyzer.h – MSE vs SPP analysis
//
// Analyzes how rendering error decreases with sample count.
// An unbiased Monte Carlo estimator converges at O(1/N):
//   variance(N) ∝ N^(-1)  →  convergence_rate α ≈ 1.0
//
// Deviations indicate bias (α > 1 = super-convergence ← suspicious)
// or inefficiency (α < 1 = slow convergence ← bottleneck).
// ─────────────────────────────────────────────────────────────────────
#include "diagnose/variance_tracker.h"
#include <cmath>
#include <string>
#include <vector>

struct ConvergenceAnalysis {
    float rate_alpha     = 0.f;    // exponent α in var ∝ N^(-α)
    bool  converging     = false;  // α > 0.5
    bool  normal_rate    = false;  // 0.8 ≤ α ≤ 1.2 (expected for unbiased)
    bool  stalled        = false;  // α < 0.3
    float latest_mse     = 0.f;
    int   latest_spp     = 0;
    std::string assessment;        // human-readable summary
};

class ConvergenceAnalyzer {
public:
    // Analyze the variance history collected by a VarianceTracker.
    ConvergenceAnalysis analyze(const VarianceTracker& tracker) const {
        ConvergenceAnalysis result;

        const auto& history = tracker.convergence_history();
        if (history.empty()) {
            result.assessment = "No convergence data available.";
            return result;
        }

        result.latest_mse = history.back().mean_variance;
        result.latest_spp = history.back().spp;
        result.rate_alpha = const_cast<VarianceTracker&>(tracker).estimate_convergence_rate();

        result.converging  = result.rate_alpha > 0.5f;
        result.normal_rate = result.rate_alpha >= 0.8f && result.rate_alpha <= 1.2f;
        result.stalled     = result.rate_alpha < 0.3f && history.size() >= 3;

        if (history.size() < 2) {
            result.assessment = "Insufficient data (need ≥2 SPP snapshots).";
        } else if (result.stalled) {
            result.assessment = "Convergence STALLED (α=" +
                std::to_string(result.rate_alpha) +
                "). Possible bias or systematic error.";
        } else if (result.normal_rate) {
            result.assessment = "Normal convergence (α=" +
                std::to_string(result.rate_alpha) +
                "). Estimator appears unbiased.";
        } else if (result.rate_alpha < 0.8f) {
            result.assessment = "Slow convergence (α=" +
                std::to_string(result.rate_alpha) +
                "). A pipeline stage may be a bottleneck.";
        } else {
            result.assessment = "Super-convergence (α=" +
                std::to_string(result.rate_alpha) +
                "). Check for bias-variance trade-offs (e.g. clamping).";
        }

        return result;
    }

    // Predict variance at a future SPP given the current rate.
    // Uses: var(N_target) = var(N_latest) × (N_latest / N_target)^α
    float predict_variance_at_spp(const ConvergenceAnalysis& analysis,
                                  int target_spp) const {
        if (analysis.latest_spp <= 0 || analysis.rate_alpha <= 0.f)
            return analysis.latest_mse;

        float ratio = (float)analysis.latest_spp / (float)target_spp;
        return analysis.latest_mse * std::pow(ratio, analysis.rate_alpha);
    }

    // Estimate SPP needed to reach a target relative noise level.
    // Returns -1 if not achievable (stalled convergence).
    int estimate_spp_for_quality(const ConvergenceAnalysis& analysis,
                                 float target_relative_noise) const {
        if (analysis.rate_alpha <= 0.f || analysis.latest_mse <= 0.f)
            return -1;

        const auto& h = analysis;
        float target_var = target_relative_noise * target_relative_noise;
        if (h.latest_mse <= target_var) return h.latest_spp;

        // var(N) = var(latest) × (latest/N)^α  → solve for N
        // N = latest × (var(latest) / target_var)^(1/α)
        float factor = std::pow(h.latest_mse / target_var, 1.f / h.rate_alpha);
        float needed = (float)h.latest_spp * factor;
        return (int)std::ceil(needed);
    }
};
