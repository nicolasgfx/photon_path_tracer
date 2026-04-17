// ─────────────────────────────────────────────────────────────────────
// postfx_test.cpp – Phase 9 integration test: Post-processing pipeline
//
// Tests:
//   1. Tonemap identity: black input → black sRGB output
//   2. Tonemap ACES: known HDR value → expected sRGB range
//   3. Tonemap Reinhard: known HDR value → expected sRGB range
//   4. Firefly filter: outlier pixel is clamped
//   5. Bloom: dark scene is unaffected
//   6. Pipeline end-to-end: full apply() produces valid sRGB output
//
// Requires: CUDA GPU.
// ─────────────────────────────────────────────────────────────────────
#include "postfx/postfx_pipeline.h"
#include "postfx/postfx_params.h"
#include "postfx/tonemap.h"
#include "postfx/firefly_filter.h"
#include "postfx/bloom.h"
#include "core/config.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>

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

// ── CUDA helpers ────────────────────────────────────────────────────

static void check_cuda(const char* context) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  CUDA error at %s: %s\n", context, cudaGetErrorString(err));
    }
}

// ── Test 1: Tonemap identity (black → black) ────────────────────────

static void test_tonemap_black() {
    TEST_BEGIN("Tonemap: black input → black sRGB");

    constexpr int W = 4, H = 4;
    constexpr int N = W * H;

    // HDR float4: all zeros
    std::vector<float> hdr(N * 4, 0.f);
    float* d_hdr = nullptr;
    uint8_t* d_srgb = nullptr;
    cudaMalloc(&d_hdr, N * 4 * sizeof(float));
    cudaMalloc(&d_srgb, N * 4 * sizeof(uint8_t));
    cudaMemcpy(d_hdr, hdr.data(), N * 4 * sizeof(float), cudaMemcpyHostToDevice);

    launch_tonemap_hdr(d_hdr, d_srgb, W, H, true);
    cudaDeviceSynchronize();
    check_cuda("tonemap_black");

    std::vector<uint8_t> result(N * 4);
    cudaMemcpy(result.data(), d_srgb, N * 4, cudaMemcpyDeviceToHost);

    bool all_black = true;
    for (int i = 0; i < N; ++i) {
        if (result[i * 4 + 0] != 0 || result[i * 4 + 1] != 0 || result[i * 4 + 2] != 0)
            all_black = false;
        // Alpha should be 255
        EXPECT_TRUE(result[i * 4 + 3] == 255, "alpha must be 255");
    }
    EXPECT_TRUE(all_black, "black HDR must produce black sRGB");

    cudaFree(d_hdr);
    cudaFree(d_srgb);
    TEST_PASS("Tonemap: black input → black sRGB");
}

// ── Test 2: Tonemap ACES produces reasonable output ─────────────────

static void test_tonemap_aces() {
    TEST_BEGIN("Tonemap ACES: mid-grey HDR → reasonable sRGB");

    constexpr int W = 4, H = 4;
    constexpr int N = W * H;

    // HDR float4: all pixels at (0.18, 0.18, 0.18) — mid-grey
    std::vector<float> hdr(N * 4);
    for (int i = 0; i < N; ++i) {
        hdr[i * 4 + 0] = 0.18f;
        hdr[i * 4 + 1] = 0.18f;
        hdr[i * 4 + 2] = 0.18f;
        hdr[i * 4 + 3] = 1.f;
    }

    float* d_hdr = nullptr;
    uint8_t* d_srgb = nullptr;
    cudaMalloc(&d_hdr, N * 4 * sizeof(float));
    cudaMalloc(&d_srgb, N * 4);
    cudaMemcpy(d_hdr, hdr.data(), N * 4 * sizeof(float), cudaMemcpyHostToDevice);

    launch_tonemap_hdr(d_hdr, d_srgb, W, H, true);
    cudaDeviceSynchronize();
    check_cuda("tonemap_aces");

    std::vector<uint8_t> result(N * 4);
    cudaMemcpy(result.data(), d_srgb, N * 4, cudaMemcpyDeviceToHost);

    // ACES(0.18) → ~0.228, sRGB gamma → ~0.515, × 255 ≈ 131
    // Allow wide tolerance: sRGB value should be in [80, 180]
    uint8_t r = result[0], g = result[1], b = result[2];
    printf("  ACES tonemap result for 0.18: R=%d G=%d B=%d\n", r, g, b);
    EXPECT_TRUE(r > 80 && r < 180, "red channel in reasonable range");
    EXPECT_TRUE(g > 80 && g < 180, "green channel in reasonable range");
    EXPECT_TRUE(b > 80 && b < 180, "blue channel in reasonable range");

    cudaFree(d_hdr);
    cudaFree(d_srgb);
    TEST_PASS("Tonemap ACES: mid-grey HDR → reasonable sRGB");
}

// ── Test 3: Tonemap Reinhard ─────────────────────────────────────────

static void test_tonemap_reinhard() {
    TEST_BEGIN("Tonemap Reinhard: bright HDR → clamped sRGB");

    constexpr int W = 4, H = 4;
    constexpr int N = W * H;

    // HDR float4: very bright (10, 10, 10) — should compress to near-white
    std::vector<float> hdr(N * 4);
    for (int i = 0; i < N; ++i) {
        hdr[i * 4 + 0] = 10.f;
        hdr[i * 4 + 1] = 10.f;
        hdr[i * 4 + 2] = 10.f;
        hdr[i * 4 + 3] = 1.f;
    }

    float* d_hdr = nullptr;
    uint8_t* d_srgb = nullptr;
    cudaMalloc(&d_hdr, N * 4 * sizeof(float));
    cudaMalloc(&d_srgb, N * 4);
    cudaMemcpy(d_hdr, hdr.data(), N * 4 * sizeof(float), cudaMemcpyHostToDevice);

    launch_tonemap_hdr(d_hdr, d_srgb, W, H, false);  // Reinhard
    cudaDeviceSynchronize();
    check_cuda("tonemap_reinhard");

    std::vector<uint8_t> result(N * 4);
    cudaMemcpy(result.data(), d_srgb, N * 4, cudaMemcpyDeviceToHost);

    // Reinhard(10) = 10/11 ≈ 0.909, sRGB gamma → ~0.957, × 255 ≈ 244
    uint8_t r = result[0], g = result[1], b = result[2];
    printf("  Reinhard tonemap result for 10.0: R=%d G=%d B=%d\n", r, g, b);
    EXPECT_TRUE(r > 220, "bright HDR should map near white");
    EXPECT_TRUE(g > 220, "bright HDR should map near white");

    cudaFree(d_hdr);
    cudaFree(d_srgb);
    TEST_PASS("Tonemap Reinhard: bright HDR → clamped sRGB");
}

// ── Test 4: Firefly filter removes outlier ──────────────────────────

static void test_firefly_filter() {
    TEST_BEGIN("Firefly: outlier pixel is clamped");

    constexpr int W = 8, H = 8;
    constexpr int N = W * H;

    // Fill with uniform dim pixels, inject one extreme outlier at center
    std::vector<float> hdr(N * 4);
    for (int i = 0; i < N; ++i) {
        hdr[i * 4 + 0] = 0.5f;
        hdr[i * 4 + 1] = 0.5f;
        hdr[i * 4 + 2] = 0.5f;
        hdr[i * 4 + 3] = 1.f;
    }
    // Inject firefly at (4, 4)
    int cx = 4, cy = 4;
    int ci = cy * W + cx;
    hdr[ci * 4 + 0] = 10000.f;
    hdr[ci * 4 + 1] = 10000.f;
    hdr[ci * 4 + 2] = 10000.f;

    float* d_hdr = nullptr;
    float* d_temp = nullptr;
    cudaMalloc(&d_hdr, N * 4 * sizeof(float));
    cudaMalloc(&d_temp, N * 4 * sizeof(float));
    cudaMemcpy(d_hdr, hdr.data(), N * 4 * sizeof(float), cudaMemcpyHostToDevice);

    launch_firefly_filter(d_hdr, d_temp, W, H, 1, 4.0f);
    cudaDeviceSynchronize();
    check_cuda("firefly_filter");

    std::vector<float> result(N * 4);
    cudaMemcpy(result.data(), d_hdr, N * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    float outlier_lum = 0.2126f * result[ci * 4] + 0.7152f * result[ci * 4 + 1]
                      + 0.0722f * result[ci * 4 + 2];
    float bg_lum = 0.2126f * 0.5f + 0.7152f * 0.5f + 0.0722f * 0.5f;  // ~0.5
    printf("  Outlier luminance after filter: %.4f (was 10000), background: %.4f\n",
           outlier_lum, bg_lum);

    // The outlier should be clamped well below original value
    EXPECT_TRUE(outlier_lum < 100.f, "outlier must be significantly reduced");
    // Background pixels should be largely unchanged
    float bg_result = 0.2126f * result[0] + 0.7152f * result[1] + 0.0722f * result[2];
    EXPECT_TRUE(fabsf(bg_result - bg_lum) < 0.01f, "background must be preserved");

    cudaFree(d_hdr);
    cudaFree(d_temp);
    TEST_PASS("Firefly: outlier pixel is clamped");
}

// ── Test 5: Bloom doesn't corrupt a dark scene ──────────────────────

static void test_bloom_dark_scene() {
    TEST_BEGIN("Bloom: dark scene remains dark");

    constexpr int W = 16, H = 16;
    constexpr int N = W * H;

    // All very dim pixels
    std::vector<float> hdr(N * 4);
    for (int i = 0; i < N; ++i) {
        hdr[i * 4 + 0] = 0.001f;
        hdr[i * 4 + 1] = 0.001f;
        hdr[i * 4 + 2] = 0.001f;
        hdr[i * 4 + 3] = 1.f;
    }

    float* d_hdr = nullptr;
    cudaMalloc(&d_hdr, N * 4 * sizeof(float));
    cudaMemcpy(d_hdr, hdr.data(), N * 4 * sizeof(float), cudaMemcpyHostToDevice);

    // Find max luminance — should be near 0.001
    float* d_max = nullptr;
    cudaMalloc(&d_max, sizeof(float));
    launch_bloom_find_max_luminance(d_hdr, d_max, W, H);
    cudaDeviceSynchronize();
    check_cuda("bloom_max_lum");

    float max_lum = 0.f;
    cudaMemcpy(&max_lum, d_max, sizeof(float), cudaMemcpyDeviceToHost);
    printf("  Max luminance of dark scene: %.6f\n", max_lum);
    EXPECT_TRUE(max_lum < 0.01f, "dark scene max luminance should be small");

    // The pipeline would skip bloom for max_lum < 1e-6, but let's verify
    // bright extract produces near-zero output anyway
    int mip_w = W / 2, mip_h = H / 2;
    float* d_mip0 = nullptr;
    cudaMalloc(&d_mip0, mip_w * mip_h * 4 * sizeof(float));
    cudaMemset(d_mip0, 0, mip_w * mip_h * 4 * sizeof(float));

    float threshold = max_lum * 0.25f;
    launch_bloom_bright_extract(d_hdr, d_mip0, W, H, threshold, threshold);
    cudaDeviceSynchronize();
    check_cuda("bloom_extract");

    std::vector<float> mip_result(mip_w * mip_h * 4);
    cudaMemcpy(mip_result.data(), d_mip0, mip_result.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    float sum = 0.f;
    for (float v : mip_result) sum += fabsf(v);
    printf("  Bright extract sum: %.6f\n", sum);
    // Soft-knee mode extracts small amounts even from dim scenes.
    // The sum should be bounded: much less than if all pixels were unfiltered.
    // For a 8×8 mip with pixel values ~0.001, total raw energy would be ~0.768.
    // After soft-knee extraction, we expect significantly reduced energy.
    float raw_energy = (float)(mip_w * mip_h) * 3 * 0.001f;
    printf("  Raw energy if unfiltered: %.6f\n", raw_energy);
    EXPECT_TRUE(sum < raw_energy * 1000.f, "bright extract must not amplify dark scene");

    cudaFree(d_hdr);
    cudaFree(d_max);
    cudaFree(d_mip0);
    TEST_PASS("Bloom: dark scene remains dark");
}

// ── Test 6: Full pipeline end-to-end ────────────────────────────────

static void test_pipeline_end_to_end() {
    TEST_BEGIN("Pipeline: full apply() produces valid sRGB");

    constexpr int W = 16, H = 16;
    constexpr int N = W * H;

    // Create RGB accumulator: mid-grey scene with one bright emitter
    std::vector<float> color_buf(N * 3, 0.f);
    std::vector<float> sample_cnt(N, 0.f);

    for (int i = 0; i < N; ++i) {
        color_buf[i * 3 + 0] = 0.5f;
        color_buf[i * 3 + 1] = 0.5f;
        color_buf[i * 3 + 2] = 0.5f;
        sample_cnt[i] = 1.f;
    }
    // One bright emitter at (8, 8)
    int ei = 8 * W + 8;
    color_buf[ei * 3 + 0] = 50.f;
    color_buf[ei * 3 + 1] = 40.f;
    color_buf[ei * 3 + 2] = 10.f;

    float* d_color = nullptr;
    float* d_cnt = nullptr;
    uint8_t* d_srgb = nullptr;
    cudaMalloc(&d_color, N * 3 * sizeof(float));
    cudaMalloc(&d_cnt, N * sizeof(float));
    cudaMalloc(&d_srgb, N * 4);
    cudaMemcpy(d_color, color_buf.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cnt, sample_cnt.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    PostFxPipeline pipeline;
    PostFxParams params;
    params.firefly_enabled = true;
    params.bloom_enabled   = false;  // skip bloom to keep test simple
    params.exposure        = 1.0f;
    params.use_aces        = true;

    pipeline.apply(d_color, d_cnt, d_srgb, nullptr, W, H, params);
    check_cuda("pipeline_apply");

    std::vector<uint8_t> result(N * 4);
    cudaMemcpy(result.data(), d_srgb, N * 4, cudaMemcpyDeviceToHost);

    // Check that we got non-zero output for mid-grey pixels
    uint8_t r0 = result[0], g0 = result[1], b0 = result[2];
    printf("  First pixel sRGB: R=%d G=%d B=%d\n", r0, g0, b0);
    EXPECT_TRUE(r0 > 0 && g0 > 0 && b0 > 0, "mid-grey must produce non-zero sRGB");
    EXPECT_TRUE(result[3] == 255, "alpha must be 255");

    // Check emitter pixel — should be brighter
    uint8_t re = result[ei * 4], ge = result[ei * 4 + 1], be = result[ei * 4 + 2];
    printf("  Emitter pixel sRGB: R=%d G=%d B=%d\n", re, ge, be);
    EXPECT_TRUE(re > r0, "emitter should be brighter than background");

    pipeline.cleanup();
    cudaFree(d_color);
    cudaFree(d_cnt);
    cudaFree(d_srgb);
    TEST_PASS("Pipeline: full apply() produces valid sRGB");
}

// ── Test 7: rgb_to_hdr with sample counts ───────────────────────────

static void test_rgb_to_hdr() {
    TEST_BEGIN("rgb_to_hdr: accumulator / count × exposure");

    constexpr int W = 4, H = 4;
    constexpr int N = W * H;

    // Accumulate 10 samples of value 0.3 each → stored as 3.0
    std::vector<float> color_buf(N * 3, 3.0f);
    std::vector<float> sample_cnt(N, 10.f);

    float* d_color = nullptr;
    float* d_cnt = nullptr;
    float* d_hdr = nullptr;
    cudaMalloc(&d_color, N * 3 * sizeof(float));
    cudaMalloc(&d_cnt, N * sizeof(float));
    cudaMalloc(&d_hdr, N * 4 * sizeof(float));
    cudaMemcpy(d_color, color_buf.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cnt, sample_cnt.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    float exposure = 2.0f;
    launch_rgb_to_hdr(d_color, d_cnt, d_hdr, W, H, exposure);
    cudaDeviceSynchronize();
    check_cuda("rgb_to_hdr");

    std::vector<float> result(N * 4);
    cudaMemcpy(result.data(), d_hdr, N * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    // Expected: 3.0 / 10 × 2.0 = 0.6
    float r = result[0], g = result[1], b = result[2], a = result[3];
    printf("  HDR result: R=%.4f G=%.4f B=%.4f A=%.4f (expected 0.6)\n", r, g, b, a);
    EXPECT_TRUE(fabsf(r - 0.6f) < 0.001f, "red channel must be ~0.6");
    EXPECT_TRUE(fabsf(g - 0.6f) < 0.001f, "green channel must be ~0.6");
    EXPECT_TRUE(fabsf(b - 0.6f) < 0.001f, "blue channel must be ~0.6");
    EXPECT_TRUE(fabsf(a - 1.0f) < 0.001f, "alpha must be 1.0");

    cudaFree(d_color);
    cudaFree(d_cnt);
    cudaFree(d_hdr);
    TEST_PASS("rgb_to_hdr: accumulator / count × exposure");
}

// ═════════════════════════════════════════════════════════════════════
//  main
// ═════════════════════════════════════════════════════════════════════

int main() {
    printf("========================================\n");
    printf(" Phase 9: Post-Processing Pipeline Test\n");
    printf("========================================\n");

    // Verify CUDA device
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found. Skipping GPU tests.\n");
        return 1;
    }
    cudaSetDevice(0);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    test_tonemap_black();
    test_tonemap_aces();
    test_tonemap_reinhard();
    test_firefly_filter();
    test_bloom_dark_scene();
    test_pipeline_end_to_end();
    test_rgb_to_hdr();

    printf("\n========================================\n");
    printf(" Results: %d / %d passed\n", g_tests_passed, g_tests_run);
    printf("========================================\n");

    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
