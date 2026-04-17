// ─────────────────────────────────────────────────────────────────────
// test_harness_test.cpp – Phase 12 test: Test Harness
//
// Tests:
//   1. BufferStats: clean buffer → correct mean/min/max
//   2. BufferStats: NaN/Inf detection
//   3. RMSE: identical buffers → 0
//   4. RMSE: known difference → expected value
//   5. Chi-squared: uniform bins → pass
//   6. Chi-squared: badly skewed bins → fail
//   7. KS test: uniform samples → pass
//   8. KS test: skewed samples → fail
//   9. Convergence: decreasing MSE → negative slope
//  10. Convergence: regression detected when MSE increases
//  11. Baseline save/load roundtrip
//  12. Reference scenes: Cornell box builds valid scene
//  13. Reference scenes: Furnace scene has correct setup
//  14. Furnace test: uniform buffer near 1.0 → pass
//  15. Furnace test: buffer too low → fail
//
// Pure CPU — no GPU required.
// ─────────────────────────────────────────────────────────────────────
#include "testing/test_framework.h"
#include "testing/statistical_tests.h"
#include "testing/convergence_test.h"
#include "testing/reference_scenes.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <random>

// ── Test helpers ─────────────────────────────────────────────────────
static int g_tests_run    = 0;
static int g_tests_passed = 0;

#define TEST_BEGIN(name)                                 \
    do {                                                 \
        g_tests_run++;                                   \
        printf("\n[TEST %d] %s\n", g_tests_run, name);  \
    } while(0)

#define TEST_PASS(name)                                  \
    do {                                                 \
        g_tests_passed++;                                \
        printf("[PASS] %s\n", name);                     \
    } while(0)

#define EXPECT_TRUE(cond, msg)                           \
    do {                                                 \
        if (!(cond)) {                                   \
            printf("[FAIL] %s: %s\n", msg, #cond);       \
            return;                                      \
        }                                                \
    } while(0)

// ─────────────────────────────────────────────────────────────────────
// Test 1: Buffer stats on clean data
// ─────────────────────────────────────────────────────────────────────
static void test_buffer_stats_clean() {
    TEST_BEGIN("BufferStats: clean buffer");

    float data[] = { 1.f, 2.f, 3.f, 4.f, 5.f };
    auto s = test_utils::compute_buffer_stats(data, 5);

    EXPECT_TRUE(std::abs(s.mean - 3.f) < 0.01f, "mean should be 3");
    EXPECT_TRUE(s.min_val == 1.f, "min should be 1");
    EXPECT_TRUE(s.max_val == 5.f, "max should be 5");
    EXPECT_TRUE(!s.has_nan(), "no NaN");
    EXPECT_TRUE(!s.has_inf(), "no Inf");
    EXPECT_TRUE(s.zero_count == 0, "no zeros");

    TEST_PASS("BufferStats: clean buffer");
}

// ─────────────────────────────────────────────────────────────────────
// Test 2: Buffer stats with NaN/Inf
// ─────────────────────────────────────────────────────────────────────
static void test_buffer_stats_bad() {
    TEST_BEGIN("BufferStats: NaN/Inf detection");

    float nan_val = std::nanf("");
    float inf_val = INFINITY;
    float data[] = { 1.f, nan_val, 3.f, inf_val, 5.f };
    auto s = test_utils::compute_buffer_stats(data, 5);

    EXPECT_TRUE(s.has_nan(), "should detect NaN");
    EXPECT_TRUE(s.has_inf(), "should detect Inf");
    EXPECT_TRUE(s.nan_count == 1, "one NaN");
    EXPECT_TRUE(s.inf_count == 1, "one Inf");
    // Mean of valid values: (1+3+5)/3 = 3
    EXPECT_TRUE(std::abs(s.mean - 3.f) < 0.01f, "mean of valid values");

    TEST_PASS("BufferStats: NaN/Inf detection");
}

// ─────────────────────────────────────────────────────────────────────
// Test 3: RMSE of identical buffers
// ─────────────────────────────────────────────────────────────────────
static void test_rmse_identical() {
    TEST_BEGIN("RMSE: identical buffers");

    std::vector<float> img(64 * 64 * 3, 0.5f);
    float rmse = test_utils::compute_rmse(img.data(), img.data(),
                                           64 * 64);
    EXPECT_TRUE(rmse == 0.f, "RMSE of identical should be 0");

    TEST_PASS("RMSE: identical buffers");
}

// ─────────────────────────────────────────────────────────────────────
// Test 4: RMSE of known difference
// ─────────────────────────────────────────────────────────────────────
static void test_rmse_known() {
    TEST_BEGIN("RMSE: known difference");

    int n = 100;
    std::vector<float> a(n * 3, 0.f);
    std::vector<float> b(n * 3, 0.1f);  // every element differs by 0.1

    float rmse = test_utils::compute_rmse(a.data(), b.data(), n);
    // RMSE = sqrt(mean of 0.01) = 0.1
    EXPECT_TRUE(std::abs(rmse - 0.1f) < 0.001f, "RMSE should be 0.1");

    TEST_PASS("RMSE: known difference");
}

// ─────────────────────────────────────────────────────────────────────
// Test 5: Chi-squared test on uniform bins
// ─────────────────────────────────────────────────────────────────────
static void test_chi2_uniform() {
    TEST_BEGIN("Chi-squared: uniform bins");

    // 10 bins, 1000 samples equally distributed
    int num_bins = 10;
    std::vector<int> observed(num_bins, 100);   // 100 each
    std::vector<double> expected(num_bins, 100.0);

    auto r = statistical_tests::chi_squared_test(observed, expected);
    r.report("uniform");

    EXPECT_TRUE(r.passed, "uniform distribution should pass chi2");
    EXPECT_TRUE(r.chi2 < 1e-6, "chi2 stat should be ~0 for perfect match");

    TEST_PASS("Chi-squared: uniform bins");
}

// ─────────────────────────────────────────────────────────────────────
// Test 6: Chi-squared test on skewed bins (should fail)
// ─────────────────────────────────────────────────────────────────────
static void test_chi2_skewed() {
    TEST_BEGIN("Chi-squared: skewed bins");

    int num_bins = 10;
    std::vector<int> observed(num_bins, 50);   // expect 100 each
    observed[0] = 500;  // heavily skewed first bin
    std::vector<double> expected(num_bins, 100.0);

    auto r = statistical_tests::chi_squared_test(observed, expected);
    r.report("skewed");

    EXPECT_TRUE(!r.passed, "skewed distribution should fail chi2");
    EXPECT_TRUE(r.chi2 > 100.0, "chi2 stat should be large");

    TEST_PASS("Chi-squared: skewed bins");
}

// ─────────────────────────────────────────────────────────────────────
// Test 7: KS test on uniform samples
// ─────────────────────────────────────────────────────────────────────
static void test_ks_uniform() {
    TEST_BEGIN("KS test: uniform samples");

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);

    int n = 10000;
    std::vector<float> samples(n);
    for (int i = 0; i < n; ++i) samples[i] = dist(rng);

    auto r = statistical_tests::ks_test_uniform(samples.data(), n);
    r.report("uniform_rng");

    EXPECT_TRUE(r.passed, "truly uniform samples should pass KS test");

    TEST_PASS("KS test: uniform samples");
}

// ─────────────────────────────────────────────────────────────────────
// Test 8: KS test on biased samples (should fail)
// ─────────────────────────────────────────────────────────────────────
static void test_ks_biased() {
    TEST_BEGIN("KS test: biased samples");

    int n = 10000;
    std::vector<float> samples(n);
    // All samples in [0, 0.5] — clearly not uniform on [0, 1]
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 0.5f);
    for (int i = 0; i < n; ++i) samples[i] = dist(rng);

    auto r = statistical_tests::ks_test_uniform(samples.data(), n);
    r.report("biased_rng");

    EXPECT_TRUE(!r.passed, "biased samples should fail KS test");

    TEST_PASS("KS test: biased samples");
}

// ─────────────────────────────────────────────────────────────────────
// Test 9: Convergence: decreasing MSE → negative slope
// ─────────────────────────────────────────────────────────────────────
static void test_convergence_decreasing() {
    TEST_BEGIN("Convergence: decreasing MSE");

    // Simulate 1/N convergence:  MSE ≈ C / SPP
    std::vector<int>   spp  = { 16,   64,   256,  1024 };
    std::vector<float> mse  = { 1.0f, 0.25f, 0.0625f, 0.015625f };

    auto fit = convergence_test::log_log_fit(
        spp.data(), mse.data(), (int)spp.size());

    printf("  slope=%.3f, R²=%.3f\n", fit.slope, fit.r_squared);

    EXPECT_TRUE(fit.slope < -0.5f, "slope should be negative");
    EXPECT_TRUE(fit.r_squared > 0.99f, "R² should be near 1");

    TEST_PASS("Convergence: decreasing MSE");
}

// ─────────────────────────────────────────────────────────────────────
// Test 10: Convergence regression detection
// ─────────────────────────────────────────────────────────────────────
static void test_convergence_regression() {
    TEST_BEGIN("Convergence: regression detected");

    // Baseline: good convergence (1/N)
    auto baseline = convergence_test::create_baseline(
        "test_scene",
        { 16, 64, 256, 1024 },
        { 1.0f, 0.25f, 0.0625f, 0.015625f });

    // Measured: poor convergence (barely decreasing)
    std::vector<int>   spp      = { 16,   64,  256,  1024 };
    std::vector<float> bad_mse  = { 1.0f, 0.9f, 0.85f, 0.8f };

    auto result = convergence_test::compare_to_baseline(
        spp, bad_mse, baseline, 0.15f, 3.0f);
    result.report();

    EXPECT_TRUE(!result.passed, "regression should be detected");
    EXPECT_TRUE(!result.rate_ok, "rate should be flagged");

    TEST_PASS("Convergence: regression detected");
}

// ─────────────────────────────────────────────────────────────────────
// Test 11: Baseline save/load roundtrip
// ─────────────────────────────────────────────────────────────────────
static void test_baseline_roundtrip() {
    TEST_BEGIN("Baseline save/load roundtrip");

    auto orig = convergence_test::create_baseline(
        "roundtrip_scene",
        { 16, 64, 256, 1024 },
        { 0.5f, 0.12f, 0.03f, 0.008f });

    std::string path = "test_baseline_tmp.txt";
    bool saved = convergence_test::save_baseline(orig, path);
    EXPECT_TRUE(saved, "save should succeed");

    convergence_test::ConvergenceBaseline loaded;
    bool ok = convergence_test::load_baseline(path, loaded);
    EXPECT_TRUE(ok, "load should succeed");
    EXPECT_TRUE(loaded.scene_name == "roundtrip_scene", "name matches");
    EXPECT_TRUE(loaded.spp_levels.size() == 4, "4 levels loaded");

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(loaded.spp_levels[i] == orig.spp_levels[i],
                     "spp level matches");
        EXPECT_TRUE(std::abs(loaded.mse_values[i] - orig.mse_values[i]) < 1e-4f,
                     "mse value matches");
    }

    // Clean up
    std::remove(path.c_str());

    TEST_PASS("Baseline save/load roundtrip");
}

// ─────────────────────────────────────────────────────────────────────
// Test 12: Reference scenes — Cornell box valid
// ─────────────────────────────────────────────────────────────────────
static void test_ref_cornell() {
    TEST_BEGIN("Reference: Cornell box valid");

    auto ref = reference_scenes::cornell_box();

    EXPECT_TRUE(!ref.scene.triangles.empty(), "has triangles");
    EXPECT_TRUE(ref.scene.triangles.size() >= 12, "at least 6 quads = 12 tris");
    EXPECT_TRUE(ref.scene.materials.size() >= 4, "4 materials");
    EXPECT_TRUE(ref.name == "cornell_box", "name is cornell_box");

    // Camera should be outside the box looking in
    EXPECT_TRUE(ref.camera.pos.z > 0.5f, "camera outside box");

    TEST_PASS("Reference: Cornell box valid");
}

// ─────────────────────────────────────────────────────────────────────
// Test 13: Reference scenes — Furnace correct setup
// ─────────────────────────────────────────────────────────────────────
static void test_ref_furnace() {
    TEST_BEGIN("Reference: Furnace scene setup");

    auto ref = reference_scenes::furnace();

    EXPECT_TRUE(ref.is_furnace, "marked as furnace");
    EXPECT_TRUE(ref.expected_mean_luminance > 0.f, "has expected luminance");
    EXPECT_TRUE(!ref.scene.triangles.empty(), "has geometry");
    EXPECT_TRUE(ref.scene.materials.size() >= 2, "white + emissive");

    // Check first material is white with albedo 1.0
    EXPECT_TRUE(ref.scene.materials[0].Kd.r >= 0.99f, "white mat r=1");
    EXPECT_TRUE(ref.scene.materials[0].Kd.g >= 0.99f, "white mat g=1");
    EXPECT_TRUE(ref.scene.materials[0].Kd.b >= 0.99f, "white mat b=1");

    TEST_PASS("Reference: Furnace scene setup");
}

// ─────────────────────────────────────────────────────────────────────
// Test 14: Furnace test — buffer near 1.0 passes
// ─────────────────────────────────────────────────────────────────────
static void test_furnace_pass() {
    TEST_BEGIN("Furnace: buffer near 1.0 passes");

    int n = 256;
    std::vector<float> rgb(n * 3);
    // Fill with luminance ~1.0 (R=G=B=1.0)
    for (int i = 0; i < n * 3; ++i) rgb[i] = 1.0f;

    auto r = statistical_tests::furnace_test(rgb.data(), n, 1.0f, 0.05f);
    r.report("furnace_pass");

    EXPECT_TRUE(r.passed, "buffer at 1.0 should pass furnace test");

    TEST_PASS("Furnace: buffer near 1.0 passes");
}

// ─────────────────────────────────────────────────────────────────────
// Test 15: Furnace test — buffer too low fails
// ─────────────────────────────────────────────────────────────────────
static void test_furnace_fail() {
    TEST_BEGIN("Furnace: buffer too low fails");

    int n = 256;
    std::vector<float> rgb(n * 3, 0.3f);  // all channels 0.3

    auto r = statistical_tests::furnace_test(rgb.data(), n, 1.0f, 0.05f);
    r.report("furnace_fail");

    EXPECT_TRUE(!r.passed, "buffer at 0.3 should fail furnace test");

    TEST_PASS("Furnace: buffer too low fails");
}

// ─────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────
int main() {
    printf("=== Test Harness Tests (Phase 12) ===\n");

    test_buffer_stats_clean();
    test_buffer_stats_bad();
    test_rmse_identical();
    test_rmse_known();
    test_chi2_uniform();
    test_chi2_skewed();
    test_ks_uniform();
    test_ks_biased();
    test_convergence_decreasing();
    test_convergence_regression();
    test_baseline_roundtrip();
    test_ref_cornell();
    test_ref_furnace();
    test_furnace_pass();
    test_furnace_fail();

    printf("\n========================================\n");
    printf(" Test Harness: %d / %d passed",
           g_tests_passed, g_tests_run);
    if (g_tests_run != g_tests_passed)
        printf(" (%d FAILED)", g_tests_run - g_tests_passed);
    printf("\n========================================\n");

    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
