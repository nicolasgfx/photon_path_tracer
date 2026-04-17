// ─────────────────────────────────────────────────────────────────────
// tests/convergence/test_convergence_direct.cpp
//
// Convergence regression: Direct-only rendering at [16,64,256] SPP.
// Verifies RMSE decreases monotonically against reference.
//
// NOTE: This test requires a fully wired GPU pipeline. Until the
// render pipeline is assembled, these tests validate the convergence
// testing infrastructure with synthetic data.
// ─────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include "testing/convergence_test.h"
#include "testing/test_framework.h"
#include <cmath>

// ── Synthetic convergence test (validates the harness) ──────────────

TEST(ConvergenceDirect, LogLogFitDecreasing) {
    // Simulate direct-only convergence: MSE ∝ 1/N
    int spp[] = {16, 64, 256};
    float mse[] = {0.15f, 0.04f, 0.01f};

    auto fit = convergence_test::log_log_fit(spp, mse, 3);

    // Slope should be negative (MSE decreasing with SPP)
    EXPECT_LT(fit.slope, -0.5f);
    // R² should be high for clean 1/N data
    EXPECT_GT(fit.r_squared, 0.95f);
}

TEST(ConvergenceDirect, BaselineComparison) {
    auto baseline = convergence_test::create_baseline(
        "direct_synthetic", {16, 64, 256}, {0.15f, 0.04f, 0.01f});

    // "Measured" data that's slightly better
    std::vector<int> spp = {16, 64, 256};
    std::vector<float> mse = {0.14f, 0.035f, 0.009f};

    auto result = convergence_test::compare_to_baseline(spp, mse, baseline);
    EXPECT_TRUE(result.passed);
    EXPECT_TRUE(result.rate_ok);
    EXPECT_TRUE(result.all_mse_ok);
}

TEST(ConvergenceDirect, RegressionDetected) {
    auto baseline = convergence_test::create_baseline(
        "direct_regressed", {16, 64, 256}, {0.15f, 0.04f, 0.01f});

    // "Measured" data that's much worse (3× at each level)
    std::vector<int> spp = {16, 64, 256};
    std::vector<float> mse = {0.50f, 0.20f, 0.08f};

    auto result = convergence_test::compare_to_baseline(spp, mse, baseline);
    EXPECT_FALSE(result.passed);
}

// ── TODO: GPU direct-only render test ───────────────────────────────
// When the full pipeline is wired, add:
// TEST(ConvergenceDirect, CornellBoxDirect) {
//     auto ref = reference_scenes::cornell_box();
//     // Render at 16,64,256 SPP in direct-only mode
//     // Measure RMSE at each level
//     // Compare against stored baseline
// }
