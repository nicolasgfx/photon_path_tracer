// ─────────────────────────────────────────────────────────────────────
// tests/convergence/test_convergence_full.cpp
//
// Convergence regression: Full path-traced render at [16,64,256,1024].
// Validates overall convergence rate and MSE regression.
//
// Until the full GPU pipeline is wired, tests validate the convergence
// testing infrastructure with synthetic data.
// ─────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include "testing/convergence_test.h"
#include <cmath>

TEST(ConvergenceFull, IdealMCConvergence) {
    // Perfect 1/N convergence: slope should be ≈ -1.0
    int spp[]   = {16, 64, 256, 1024};
    float mse[] = {1.0f / 16, 1.0f / 64, 1.0f / 256, 1.0f / 1024};

    auto fit = convergence_test::log_log_fit(spp, mse, 4);
    EXPECT_NEAR(fit.slope, -1.0f, 0.05f);
    EXPECT_GT(fit.r_squared, 0.999f);
}

TEST(ConvergenceFull, BaselineRoundtrip) {
    auto baseline = convergence_test::create_baseline(
        "full_pt", {16, 64, 256, 1024}, {0.15f, 0.04f, 0.01f, 0.003f});

    std::string path = "test_baseline_full.tmp";
    ASSERT_TRUE(convergence_test::save_baseline(baseline, path));

    convergence_test::ConvergenceBaseline loaded;
    ASSERT_TRUE(convergence_test::load_baseline(path, loaded));

    EXPECT_EQ(loaded.scene_name, "full_pt");
    EXPECT_EQ(loaded.spp_levels.size(), 4u);
    EXPECT_NEAR(loaded.mse_values[0], 0.15f, 1e-4f);
    EXPECT_LT(loaded.convergence_rate, -0.5f);

    std::remove(path.c_str());
}

TEST(ConvergenceFull, SlowerThanIdealAcceptable) {
    // Scene with caustics: slower convergence but still acceptable
    auto baseline = convergence_test::create_baseline(
        "complex_scene", {16, 64, 256, 1024}, {0.30f, 0.12f, 0.05f, 0.02f});

    // Rate is slower (≈ -0.65) but still converging
    EXPECT_LT(baseline.convergence_rate, -0.4f);
}

// ── TODO: GPU full path-traced render test ──────────────────────────
// TEST(ConvergenceFull, CornellBoxFullPT) {
//     // Render Cornell at [16,64,256,1024] SPP
//     // Compare against stored baseline
// }
