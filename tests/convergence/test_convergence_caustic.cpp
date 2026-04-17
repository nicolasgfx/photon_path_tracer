// ─────────────────────────────────────────────────────────────────────
// tests/convergence/test_convergence_caustic.cpp
//
// Convergence regression: Caustic convergence rate.
// Glass sphere scenes converge slower (≈ -0.5) vs diffuse (≈ -1.0).
//
// Until the full GPU pipeline is wired, tests validate the testing
// infrastructure with synthetic data.
// ─────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include "testing/convergence_test.h"
#include "testing/reference_scenes.h"
#include <cmath>

TEST(ConvergenceCaustic, SlowerThanDiffuse) {
    // Caustic scenes converge slower than pure diffuse
    auto diffuse = convergence_test::create_baseline(
        "diffuse", {16, 64, 256, 1024}, {0.15f, 0.04f, 0.01f, 0.003f});

    auto caustic = convergence_test::create_baseline(
        "caustic", {16, 64, 256, 1024}, {0.40f, 0.20f, 0.10f, 0.05f});

    // Caustic rate is less negative (slower convergence)
    EXPECT_GT(caustic.convergence_rate, diffuse.convergence_rate);
}

TEST(ConvergenceCaustic, CausticStillConverges) {
    auto caustic = convergence_test::create_baseline(
        "caustic_conv", {16, 64, 256, 1024}, {0.40f, 0.20f, 0.10f, 0.05f});

    // Must still converge (negative rate)
    EXPECT_LT(caustic.convergence_rate, 0.0f);
    // But slower than ideal: rate > -0.7
    EXPECT_GT(caustic.convergence_rate, -0.75f);
}

TEST(ConvergenceCaustic, PhotonsImproveCausticRate) {
    // With photon-guided caustics, rate should be better
    auto without_photons = convergence_test::create_baseline(
        "no_photons", {16, 64, 256}, {0.40f, 0.25f, 0.16f});

    auto with_photons = convergence_test::create_baseline(
        "with_photons", {16, 64, 256}, {0.35f, 0.12f, 0.04f});

    // Photon-guided should converge faster
    EXPECT_LT(with_photons.convergence_rate, without_photons.convergence_rate);
}

TEST(ConvergenceCaustic, ReferenceSceneHasCaustics) {
    auto ref = reference_scenes::glass_sphere();
    EXPECT_TRUE(ref.has_caustics);
    EXPECT_NEAR(ref.expected_convergence_rate, -0.5f, 0.2f);
}

// ── TODO: GPU caustic render test ───────────────────────────────────
// TEST(ConvergenceCaustic, GlassSphereCaustic) {
//     // Render glass sphere scene at 256 SPP
//     // Verify caustic region is illuminated
//     // Compare convergence rate against baseline
// }
