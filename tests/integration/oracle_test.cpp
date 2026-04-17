// ─────────────────────────────────────────────────────────────────────
// oracle_test.cpp – Phase 11 test: Image Oracle
//
// Tests:
//   1. RMSE/PSNR: identical images → RMSE=0, PSNR=100
//   2. RMSE/PSNR: known difference → expected RMSE
//   3. SSIM: identical images → SSIM=1.0
//   4. SSIM: different images → SSIM < 1.0
//   5. Noise estimation: clean image → low noise
//   6. Noise estimation: noisy image → high noise
//   7. Firefly detection: synthetic fireflies detected
//   8. Artifact detector: firefly artifacts generate corrections
//   9. ImageOracle: no-reference analysis on clean image
//  10. ImageOracle: reference-based analysis with known error
//
// Pure CPU — no GPU required.
// ─────────────────────────────────────────────────────────────────────
#include "diagnose/image_oracle.h"

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

// ── Image generation helpers ────────────────────────────────────────

static std::vector<float> make_uniform_image(int w, int h, float r, float g, float b) {
    std::vector<float> img(w * h * 3);
    for (int i = 0; i < w * h; ++i) {
        img[i * 3 + 0] = r;
        img[i * 3 + 1] = g;
        img[i * 3 + 2] = b;
    }
    return img;
}

static std::vector<float> make_gradient_image(int w, int h) {
    std::vector<float> img(w * h * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float v = (float)x / (float)(w - 1);
            int i = (y * w + x) * 3;
            img[i + 0] = v;
            img[i + 1] = v;
            img[i + 2] = v;
        }
    }
    return img;
}

static std::vector<float> add_noise(const std::vector<float>& img,
                                     float sigma, unsigned seed = 42) {
    std::vector<float> noisy = img;
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.f, sigma);
    for (size_t i = 0; i < noisy.size(); ++i) {
        noisy[i] += dist(rng);
        if (noisy[i] < 0.f) noisy[i] = 0.f;
    }
    return noisy;
}

// ── Test 1: RMSE identical ──────────────────────────────────────────

static void test_rmse_identical() {
    TEST_BEGIN("RMSE/PSNR: identical images");

    auto img = make_uniform_image(16, 16, 0.5f, 0.5f, 0.5f);
    float rmse = image_metrics::compute_rmse(img.data(), img.data(), 16 * 16);
    float psnr = image_metrics::compute_psnr(rmse);

    printf("  RMSE=%.6f PSNR=%.1f\n", rmse, psnr);
    EXPECT_TRUE(rmse < 1e-6f, "RMSE of identical images should be ~0");
    EXPECT_TRUE(psnr >= 99.f, "PSNR should be very high");

    TEST_PASS("RMSE/PSNR: identical images");
}

// ── Test 2: RMSE known difference ───────────────────────────────────

static void test_rmse_known() {
    TEST_BEGIN("RMSE/PSNR: known difference");

    auto a = make_uniform_image(8, 8, 0.0f, 0.0f, 0.0f);
    auto b = make_uniform_image(8, 8, 0.1f, 0.1f, 0.1f);
    float rmse = image_metrics::compute_rmse(a.data(), b.data(), 8 * 8);

    printf("  RMSE=%.6f (expected 0.1)\n", rmse);
    EXPECT_TRUE(std::abs(rmse - 0.1f) < 0.001f, "RMSE should be 0.1");

    float psnr = image_metrics::compute_psnr(rmse);
    printf("  PSNR=%.1f dB\n", psnr);
    EXPECT_TRUE(psnr > 0.f && psnr < 100.f, "PSNR should be finite");

    TEST_PASS("RMSE/PSNR: known difference");
}

// ── Test 3: SSIM identical ──────────────────────────────────────────

static void test_ssim_identical() {
    TEST_BEGIN("SSIM: identical images");

    auto img = make_gradient_image(64, 64);
    int n = 64 * 64;
    std::vector<float> lum(n);
    for (int i = 0; i < n; ++i)
        lum[i] = image_metrics::luminance(img[i*3], img[i*3+1], img[i*3+2]);

    float ssim = image_metrics::compute_ssim(lum.data(), lum.data(), 64, 64);
    printf("  SSIM=%.6f\n", ssim);
    EXPECT_TRUE(std::abs(ssim - 1.0f) < 0.01f, "SSIM of identical images should be ~1.0");

    TEST_PASS("SSIM: identical images");
}

// ── Test 4: SSIM different ──────────────────────────────────────────

static void test_ssim_different() {
    TEST_BEGIN("SSIM: different images");

    auto a = make_gradient_image(64, 64);
    auto b = add_noise(a, 0.2f);
    int n = 64 * 64;
    std::vector<float> lum_a(n), lum_b(n);
    for (int i = 0; i < n; ++i) {
        lum_a[i] = image_metrics::luminance(a[i*3], a[i*3+1], a[i*3+2]);
        lum_b[i] = image_metrics::luminance(b[i*3], b[i*3+1], b[i*3+2]);
    }

    float ssim = image_metrics::compute_ssim(lum_a.data(), lum_b.data(), 64, 64);
    printf("  SSIM=%.4f (with noise σ=0.2)\n", ssim);
    EXPECT_TRUE(ssim < 0.95f, "SSIM with noise should be < 0.95");
    EXPECT_TRUE(ssim > 0.0f, "SSIM should be positive");

    TEST_PASS("SSIM: different images");
}

// ── Test 5: Noise estimation (clean) ────────────────────────────────

static void test_noise_clean() {
    TEST_BEGIN("Noise estimation: clean gradient");

    auto img = make_gradient_image(64, 64);
    auto noise = noise_analyzer::analyze(img.data(), 64, 64);

    printf("  noise_level=%.4f fireflies=%d splotch=%.4f\n",
           noise.global_noise_level, noise.num_firefly_pixels, noise.splotch_score);

    EXPECT_TRUE(noise.global_noise_level < 0.1f, "clean image should have low noise");
    EXPECT_TRUE(noise.num_firefly_pixels == 0, "clean image should have no fireflies");

    TEST_PASS("Noise estimation: clean gradient");
}

// ── Test 6: Noise estimation (noisy) ────────────────────────────────

static void test_noise_noisy() {
    TEST_BEGIN("Noise estimation: noisy image");

    auto clean = make_uniform_image(64, 64, 0.5f, 0.5f, 0.5f);
    auto noisy = add_noise(clean, 0.3f);
    auto noise = noise_analyzer::analyze(noisy.data(), 64, 64);

    printf("  noise_level=%.4f laplacian=%.4f\n",
           noise.global_noise_level, noise.laplacian_variance);

    EXPECT_TRUE(noise.global_noise_level > 0.05f, "noisy image should have detectable noise");

    TEST_PASS("Noise estimation: noisy image");
}

// ── Test 7: Firefly detection ───────────────────────────────────────

static void test_firefly_detection() {
    TEST_BEGIN("Firefly detection: synthetic");

    constexpr int W = 32, H = 32;
    auto img = make_uniform_image(W, H, 0.5f, 0.5f, 0.5f);

    // Inject 5 firefly pixels
    int firefly_positions[] = {100, 200, 300, 500, 700};
    for (int pos : firefly_positions) {
        if (pos < W * H) {
            img[pos * 3 + 0] = 100.f;
            img[pos * 3 + 1] = 100.f;
            img[pos * 3 + 2] = 100.f;
        }
    }

    auto noise = noise_analyzer::analyze(img.data(), W, H, 10.f);

    printf("  fireflies=%d fraction=%.4f\n",
           noise.num_firefly_pixels, noise.firefly_fraction);

    EXPECT_TRUE(noise.num_firefly_pixels >= 4, "should detect most fireflies");
    EXPECT_TRUE(noise.firefly_fraction > 0.001f, "firefly fraction should be non-zero");

    TEST_PASS("Firefly detection: synthetic");
}

// ── Test 8: Artifact corrections ────────────────────────────────────

static void test_artifact_corrections() {
    TEST_BEGIN("Artifact detector: corrections for fireflies");

    NoiseEstimate noise;
    noise.num_firefly_pixels = 50;
    noise.firefly_fraction = 0.005f;  // above threshold
    noise.global_noise_level = 0.05f;
    noise.splotch_score = 0.05f;

    auto artifacts = artifact_detector::detect(noise);
    printf("  artifacts detected: %zu\n", artifacts.size());

    bool has_firefly = false;
    for (const auto& a : artifacts) {
        if (a.type == ArtifactType::Firefly) has_firefly = true;
        printf("    %s severity=%.2f\n", artifact_type_name(a.type), a.severity);
    }
    EXPECT_TRUE(has_firefly, "should detect firefly artifact");

    auto corrections = artifact_detector::suggest_corrections(artifacts);
    printf("  corrections: %zu\n", corrections.size());
    EXPECT_TRUE(!corrections.empty(), "should generate corrections");

    bool has_path_integrator_fix = false;
    for (const auto& c : corrections) {
        if (c.target_skill == "path-integrator") has_path_integrator_fix = true;
        printf("    [%s] %s: %s\n", c.target_skill.c_str(), c.parameter.c_str(),
               c.rationale.c_str());
    }
    EXPECT_TRUE(has_path_integrator_fix, "should suggest path-integrator correction");

    TEST_PASS("Artifact detector: corrections for fireflies");
}

// ── Test 9: Oracle no-reference ─────────────────────────────────────

static void test_oracle_no_reference() {
    TEST_BEGIN("ImageOracle: no-reference on clean image");

    auto img = make_gradient_image(64, 64);
    ImageOracle oracle;
    auto verdict = oracle.analyze(img.data(), 64, 64);

    printf("  noise=%.4f psnr=%.1f ssim=%.3f\n",
           verdict.noise_level, verdict.psnr_vs_reference, verdict.ssim_vs_reference);
    printf("  artifacts: %zu  corrections: %zu\n",
           verdict.artifacts.size(), verdict.corrections.size());
    printf("  summary: %s\n", verdict.summary.c_str());

    EXPECT_TRUE(verdict.noise_level < 0.15f, "clean image noise should be low");
    EXPECT_TRUE(verdict.psnr_vs_reference < 0.f, "no reference → PSNR should be -1");

    TEST_PASS("ImageOracle: no-reference on clean image");
}

// ── Test 10: Oracle with reference ──────────────────────────────────

static void test_oracle_with_reference() {
    TEST_BEGIN("ImageOracle: reference-based with known error");

    auto reference = make_gradient_image(64, 64);
    auto rendered = add_noise(reference, 0.05f);

    ImageOracle oracle;
    auto verdict = oracle.analyze_with_reference(
        rendered.data(), reference.data(), 64, 64);

    printf("  noise=%.4f psnr=%.1fdB ssim=%.3f\n",
           verdict.noise_level, verdict.psnr_vs_reference, verdict.ssim_vs_reference);
    printf("  artifacts: %zu  corrections: %zu\n",
           verdict.artifacts.size(), verdict.corrections.size());

    EXPECT_TRUE(verdict.psnr_vs_reference > 0.f, "should have PSNR");
    EXPECT_TRUE(verdict.ssim_vs_reference > 0.f && verdict.ssim_vs_reference <= 1.f,
                "SSIM should be in (0, 1]");
    EXPECT_TRUE(verdict.psnr_vs_reference > 20.f, "small noise should give decent PSNR");

    TEST_PASS("ImageOracle: reference-based with known error");
}

// ═════════════════════════════════════════════════════════════════════

int main() {
    printf("========================================\n");
    printf(" Phase 11: Image Oracle Test\n");
    printf("========================================\n");

    test_rmse_identical();
    test_rmse_known();
    test_ssim_identical();
    test_ssim_different();
    test_noise_clean();
    test_noise_noisy();
    test_firefly_detection();
    test_artifact_corrections();
    test_oracle_no_reference();
    test_oracle_with_reference();

    printf("\n========================================\n");
    printf(" Results: %d / %d passed\n", g_tests_passed, g_tests_run);
    printf("========================================\n");

    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
