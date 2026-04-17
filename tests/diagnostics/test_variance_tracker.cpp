// ─────────────────────────────────────────────────────────────────────
// tests/diagnostics/test_variance_tracker.cpp
//
// Meta-skill A: Per-pixel variance estimation and noise maps.
// Uses CPU-side synthetic buffers (host data passed as if GPU).
// ─────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include "diagnose/variance_tracker.h"
#include <vector>
#include <cmath>

// Helper: create synthetic host buffers (VarianceTracker downloads from
// device pointers — for CPU tests we pass host pointers through the
// same interface using the diagnostics CPU fallback path if available,
// or we test just the ConvergencePoint/rate estimation path).

TEST(VarianceTracker, ConvergencePointRecording) {
    VarianceTracker tracker;
    tracker.clear_history();

    VarianceStats s1{};
    s1.mean_variance = 0.10f;
    s1.mean_relative_noise = 0.50f;
    tracker.record_convergence_point(16, s1);

    VarianceStats s2{};
    s2.mean_variance = 0.025f;
    s2.mean_relative_noise = 0.25f;
    tracker.record_convergence_point(64, s2);

    VarianceStats s3{};
    s3.mean_variance = 0.006f;
    s3.mean_relative_noise = 0.12f;
    tracker.record_convergence_point(256, s3);

    EXPECT_EQ((int)tracker.convergence_history().size(), 3);
}

TEST(VarianceTracker, ConvergenceRateEstimation) {
    VarianceTracker tracker;
    tracker.clear_history();

    // Synthetic 1/N convergence
    float base_var = 1.0f;
    int spps[] = {16, 32, 64, 128, 256};
    for (int spp : spps) {
        VarianceStats s{};
        s.mean_variance = base_var / spp;
        s.mean_relative_noise = std::sqrt(s.mean_variance);
        tracker.record_convergence_point(spp, s);
    }

    float rate = tracker.estimate_convergence_rate();
    // Should be ≈ 1.0 for perfect 1/N convergence
    EXPECT_NEAR(rate, 1.0f, 0.15f);
}

TEST(VarianceTracker, StalledConvergence) {
    VarianceTracker tracker;
    tracker.clear_history();

    // Flat variance (not converging)
    for (int spp : {16, 64, 256, 1024}) {
        VarianceStats s{};
        s.mean_variance = 0.1f;  // constant
        tracker.record_convergence_point(spp, s);
    }

    float rate = tracker.estimate_convergence_rate();
    // Should be ≈ 0 (not converging)
    EXPECT_LT(rate, 0.3f);
}

TEST(VarianceTracker, ClearHistory) {
    VarianceTracker tracker;
    VarianceStats s{};
    s.mean_variance = 0.1f;
    tracker.record_convergence_point(64, s);
    EXPECT_EQ((int)tracker.convergence_history().size(), 1);

    tracker.clear_history();
    EXPECT_EQ((int)tracker.convergence_history().size(), 0);
}

TEST(VarianceTracker, InsufficientData) {
    VarianceTracker tracker;
    tracker.clear_history();

    // Single point → rate = 0 (insufficient)
    VarianceStats s{};
    s.mean_variance = 0.1f;
    tracker.record_convergence_point(64, s);

    float rate = tracker.estimate_convergence_rate();
    EXPECT_NEAR(rate, 0.0f, 0.01f);
}
