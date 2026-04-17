// ─────────────────────────────────────────────────────────────────────
// diagnostics_test.cpp – Phase 10 test: Render diagnostics meta-skill
//
// Tests:
//   1. VarianceTracker: compute stats from known GPU buffers
//   2. VarianceTracker: noise map values match expected
//   3. VarianceTracker: convergence rate estimation (synthetic 1/N data)
//   4. ConvergenceAnalyzer: normal convergence detection
//   5. ConvergenceAnalyzer: stalled convergence detection
//   6. BottleneckAnalyzer: identifies direct-lighting bottleneck
//   7. BottleneckAnalyzer: identifies indirect bottleneck
//   8. RenderDiagnostics: full pipeline integration
//
// Requires: CUDA GPU (for variance tracker GPU buffer tests).
// ─────────────────────────────────────────────────────────────────────
#include "diagnose/diagnostics.h"
#include "core/stage_metrics.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

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

// ── Test 1: VarianceTracker compute stats ────────────────────────────

static void test_variance_stats() {
    TEST_BEGIN("VarianceTracker: compute stats from GPU buffers");

    constexpr int W = 4, H = 4, N = W * H;

    // Simulate: each pixel has 100 samples, mean luminance = 0.5,
    // variance = 0.01 → lum_sum = 50, lum_sum2 = 50*0.5 + 100*0.01 = 26
    // Actually: mean = sum/n = 50/100 = 0.5
    //           var  = sum2/n - mean² = 26/100 - 0.25 = 0.01
    std::vector<float> lum_sum(N, 50.f);
    std::vector<float> lum_sum2(N, 26.f);
    std::vector<float> counts(N, 100.f);

    float* d_sum = nullptr, *d_sum2 = nullptr, *d_cnt = nullptr;
    cudaMalloc(&d_sum, N * sizeof(float));
    cudaMalloc(&d_sum2, N * sizeof(float));
    cudaMalloc(&d_cnt, N * sizeof(float));
    cudaMemcpy(d_sum, lum_sum.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum2, lum_sum2.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cnt, counts.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    VarianceTracker tracker;
    VarianceStats stats = tracker.compute_stats(d_sum, d_sum2, d_cnt, W, H);

    printf("  mean_var=%.4f max_var=%.4f mean_rel_noise=%.4f\n",
           stats.mean_variance, stats.max_variance, stats.mean_relative_noise);

    EXPECT_TRUE(std::abs(stats.mean_variance - 0.01f) < 0.001f,
                "mean variance should be ~0.01");
    EXPECT_TRUE(stats.num_pixels == N, "pixel count");
    // relative noise = sqrt(0.01) / 0.5 = 0.2
    EXPECT_TRUE(std::abs(stats.mean_relative_noise - 0.2f) < 0.05f,
                "relative noise should be ~0.2");

    cudaFree(d_sum);
    cudaFree(d_sum2);
    cudaFree(d_cnt);
    TEST_PASS("VarianceTracker: compute stats from GPU buffers");
}

// ── Test 2: VarianceTracker noise map ────────────────────────────────

static void test_noise_map() {
    TEST_BEGIN("VarianceTracker: noise map values");

    constexpr int W = 2, H = 2, N = W * H;

    // Pixel 0: mean=1.0, var=0.04 → rel_noise = 0.2
    // Pixel 1: mean=10.0, var=0.01 → rel_noise = 0.01
    // Pixel 2: mean=0.1, var=0.04 → rel_noise = 2.0
    // Pixel 3: mean=5.0, var=1.0 → rel_noise = 0.2
    float ls[] = {100.f, 1000.f, 10.f, 500.f};   // lum_sum = mean * count
    float ls2[] = {104.f, 10001.f, 5.f, 25100.f}; // lum_sum2 = (var + mean²) * count
    float cnt[] = {100.f, 100.f, 100.f, 100.f};

    float* d_sum, *d_sum2, *d_cnt;
    cudaMalloc(&d_sum, N * sizeof(float));
    cudaMalloc(&d_sum2, N * sizeof(float));
    cudaMalloc(&d_cnt, N * sizeof(float));
    cudaMemcpy(d_sum, ls, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum2, ls2, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cnt, cnt, N * sizeof(float), cudaMemcpyHostToDevice);

    VarianceTracker tracker;
    auto noise = tracker.compute_noise_map(d_sum, d_sum2, d_cnt, W, H);

    printf("  noise[0]=%.3f noise[1]=%.3f noise[2]=%.3f noise[3]=%.3f\n",
           noise[0], noise[1], noise[2], noise[3]);

    EXPECT_TRUE(std::abs(noise[0] - 0.2f) < 0.05f, "pixel 0 rel noise ~0.2");
    EXPECT_TRUE(noise[1] < 0.05f, "pixel 1 low noise");
    EXPECT_TRUE(noise[2] > 1.0f, "pixel 2 high noise");

    cudaFree(d_sum);
    cudaFree(d_sum2);
    cudaFree(d_cnt);
    TEST_PASS("VarianceTracker: noise map values");
}

// ── Test 3: Convergence rate estimation ──────────────────────────────

static void test_convergence_rate() {
    TEST_BEGIN("VarianceTracker: convergence rate (synthetic 1/N)");

    VarianceTracker tracker;

    // Simulate ideal 1/N convergence: var(N) = 1.0/N
    int spp_values[] = {16, 32, 64, 128, 256};
    for (int spp : spp_values) {
        VarianceStats stats;
        stats.mean_variance = 1.0f / (float)spp;
        stats.mean_relative_noise = std::sqrt(stats.mean_variance);
        tracker.record_convergence_point(spp, stats);
    }

    float rate = tracker.estimate_convergence_rate();
    printf("  Estimated rate α=%.3f (expected ~1.0)\n", rate);
    EXPECT_TRUE(std::abs(rate - 1.0f) < 0.1f, "rate should be ~1.0 for 1/N convergence");

    TEST_PASS("VarianceTracker: convergence rate (synthetic 1/N)");
}

// ── Test 4: Normal convergence detection ─────────────────────────────

static void test_normal_convergence() {
    TEST_BEGIN("ConvergenceAnalyzer: normal convergence");

    VarianceTracker tracker;
    int spp_values[] = {16, 32, 64, 128, 256};
    for (int spp : spp_values) {
        VarianceStats stats;
        stats.mean_variance = 1.0f / (float)spp;
        tracker.record_convergence_point(spp, stats);
    }

    ConvergenceAnalyzer analyzer;
    auto result = analyzer.analyze(tracker);

    printf("  α=%.3f converging=%d normal=%d stalled=%d\n",
           result.rate_alpha, result.converging, result.normal_rate, result.stalled);
    printf("  Assessment: %s\n", result.assessment.c_str());

    EXPECT_TRUE(result.converging, "should be converging");
    EXPECT_TRUE(result.normal_rate, "should be normal rate");
    EXPECT_TRUE(!result.stalled, "should not be stalled");

    TEST_PASS("ConvergenceAnalyzer: normal convergence");
}

// ── Test 5: Stalled convergence detection ────────────────────────────

static void test_stalled_convergence() {
    TEST_BEGIN("ConvergenceAnalyzer: stalled convergence");

    VarianceTracker tracker;

    // Simulate stalled: variance barely decreases with SPP
    int spp_values[] = {16, 32, 64, 128, 256};
    for (int spp : spp_values) {
        VarianceStats stats;
        stats.mean_variance = 0.1f * std::pow((float)spp, -0.1f); // α = 0.1
        tracker.record_convergence_point(spp, stats);
    }

    ConvergenceAnalyzer analyzer;
    auto result = analyzer.analyze(tracker);

    printf("  α=%.3f converging=%d normal=%d stalled=%d\n",
           result.rate_alpha, result.converging, result.normal_rate, result.stalled);
    printf("  Assessment: %s\n", result.assessment.c_str());

    EXPECT_TRUE(result.stalled, "should detect stalled convergence");
    EXPECT_TRUE(!result.normal_rate, "should not be normal rate");

    TEST_PASS("ConvergenceAnalyzer: stalled convergence");
}

// ── Test 6: Bottleneck — direct lighting ─────────────────────────────

static void test_bottleneck_direct() {
    TEST_BEGIN("BottleneckAnalyzer: direct-lighting bottleneck");

    FrameMetrics frame;
    frame.frame_number = 1;
    frame.spp = 64;

    StageMetrics nee;
    nee.stage_name = "direct-lighting";
    nee.time_ms = 50.f;
    nee.variance_contribution = 0.8f;  // 80% of variance
    nee.num_samples = 1000;
    frame.add(nee);

    StageMetrics indirect;
    indirect.stage_name = "path-integrator";
    indirect.time_ms = 30.f;
    indirect.variance_contribution = 0.15f;  // 15%
    indirect.num_samples = 1000;
    frame.add(indirect);

    StageMetrics photon;
    photon.stage_name = "photon-gather";
    photon.time_ms = 20.f;
    photon.variance_contribution = 0.05f;  // 5%
    photon.num_samples = 1000;
    frame.add(photon);

    VarianceStats var_stats;
    var_stats.mean_variance = 0.05f;
    var_stats.noisy_fraction = 0.3f;

    ConvergenceAnalysis conv;
    conv.rate_alpha = 0.95f;
    conv.normal_rate = true;

    BottleneckAnalyzer analyzer;
    auto report = analyzer.analyze(frame, var_stats, conv);

    printf("  Bottleneck: %s\n", report.bottleneck_stage.c_str());
    printf("  Direct frac: %.2f  Indirect frac: %.2f\n",
           report.direct_variance_fraction, report.indirect_variance_fraction);
    printf("  Summary: %s\n", report.summary.c_str());

    EXPECT_TRUE(report.bottleneck_stage == "direct-lighting",
                "bottleneck should be direct-lighting");
    EXPECT_TRUE(report.direct_variance_fraction > 0.7f,
                "direct variance fraction should be >70%");
    EXPECT_TRUE(!report.suggestions.empty(), "should have recommendations");

    TEST_PASS("BottleneckAnalyzer: direct-lighting bottleneck");
}

// ── Test 7: Bottleneck — indirect ────────────────────────────────────

static void test_bottleneck_indirect() {
    TEST_BEGIN("BottleneckAnalyzer: indirect bottleneck");

    FrameMetrics frame;
    StageMetrics nee;
    nee.stage_name = "direct-lighting";
    nee.variance_contribution = 0.1f;
    nee.time_ms = 20.f;
    frame.add(nee);

    StageMetrics indirect;
    indirect.stage_name = "path-integrator";
    indirect.variance_contribution = 0.85f;
    indirect.time_ms = 60.f;
    frame.add(indirect);

    VarianceStats var_stats;
    var_stats.mean_variance = 0.1f;

    ConvergenceAnalysis conv;
    conv.rate_alpha = 0.7f;

    BottleneckAnalyzer analyzer;
    auto report = analyzer.analyze(frame, var_stats, conv);

    printf("  Bottleneck: %s\n", report.bottleneck_stage.c_str());
    printf("  Indirect frac: %.2f\n", report.indirect_variance_fraction);

    EXPECT_TRUE(report.bottleneck_stage == "path-integrator",
                "bottleneck should be path-integrator");
    EXPECT_TRUE(report.indirect_variance_fraction > 0.8f,
                "indirect variance should dominate");

    bool has_integrator_suggestion = false;
    for (const auto& s : report.suggestions) {
        if (s.target_stage == \"path-integrator\") has_integrator_suggestion = true;
    }
    EXPECT_TRUE(has_integrator_suggestion, \"should recommend path-integrator improvement\");

    TEST_PASS("BottleneckAnalyzer: indirect bottleneck");
}

// ── Test 8: Full diagnostics integration ─────────────────────────────

static void test_full_diagnostics() {
    TEST_BEGIN("RenderDiagnostics: full pipeline");

    constexpr int W = 4, H = 4, N = W * H;

    RenderDiagnostics diag;

    // Simulate two snapshots at different SPP
    for (int pass = 0; pass < 2; ++pass) {
        int spp = (pass == 0) ? 32 : 128;
        float variance = 1.0f / (float)spp;

        // Create GPU buffers simulating the variance
        float mean = 0.5f;
        float sum_val = mean * (float)spp;
        float sum2_val = (variance + mean * mean) * (float)spp;

        std::vector<float> lum_sum(N, sum_val);
        std::vector<float> lum_sum2(N, sum2_val);
        std::vector<float> counts(N, (float)spp);

        float* d_sum, *d_sum2, *d_cnt;
        cudaMalloc(&d_sum, N * sizeof(float));
        cudaMalloc(&d_sum2, N * sizeof(float));
        cudaMalloc(&d_cnt, N * sizeof(float));
        cudaMemcpy(d_sum, lum_sum.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sum2, lum_sum2.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cnt, counts.data(), N * sizeof(float), cudaMemcpyHostToDevice);

        diag.submit_variance(d_sum, d_sum2, d_cnt, W, H, spp);

        // Also submit frame metrics
        FrameMetrics frame;
        frame.frame_number = pass;
        frame.spp = spp;
        StageMetrics nee;
        nee.stage_name = "direct-lighting";
        nee.variance_contribution = 0.4f;
        nee.time_ms = 20.f;
        frame.add(nee);
        StageMetrics indirect;
        indirect.stage_name = "path-integrator";
        indirect.variance_contribution = 0.5f;
        indirect.time_ms = 40.f;
        frame.add(indirect);
        diag.submit_frame(frame);

        cudaFree(d_sum);
        cudaFree(d_sum2);
        cudaFree(d_cnt);
    }

    auto report = diag.generate_report();

    printf("  Total MSE: %.6f\n", report.total_mse);
    printf("  Convergence rate: %.3f\n", report.convergence_rate);
    printf("  Bottleneck: %s\n", report.bottleneck_stage.c_str());
    printf("  Summary: %s\n", report.summary.c_str());

    EXPECT_TRUE(report.convergence_rate > 0.5f, "should detect convergence");
    EXPECT_TRUE(!report.bottleneck_stage.empty(), "should identify a bottleneck");
    EXPECT_TRUE(!report.summary.empty(), "should produce summary");

    diag.reset();

    TEST_PASS("RenderDiagnostics: full pipeline");
}

// ═════════════════════════════════════════════════════════════════════
//  main
// ═════════════════════════════════════════════════════════════════════

int main() {
    printf("========================================\n");
    printf(" Phase 10: Render Diagnostics Test\n");
    printf("========================================\n");

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found.\n");
        return 1;
    }
    cudaSetDevice(0);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    test_variance_stats();
    test_noise_map();
    test_convergence_rate();
    test_normal_convergence();
    test_stalled_convergence();
    test_bottleneck_direct();
    test_bottleneck_indirect();
    test_full_diagnostics();

    printf("\n========================================\n");
    printf(" Results: %d / %d passed\n", g_tests_passed, g_tests_run);
    printf("========================================\n");

    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
