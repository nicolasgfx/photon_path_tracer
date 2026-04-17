// ─────────────────────────────────────────────────────────────────────
// lighting_test.cpp – Phase 5 integration test: direct lighting / NEE
//
// Tests:
//   1. Build emissive distribution + upload to GPU
//   2. Cornell box NEE (ceiling light → direct illumination)
//   3. No-NaN stress test (multiple camera angles)
//   4. No NaN/inf in any output
//
// Requires: CUDA GPU + OptiX runtime.  PTX path passed via define.
// ─────────────────────────────────────────────────────────────────────
#include "accel/accel_builder.h"
#include "accel/lighting_upload.h"
#include "scene/scene_builder.h"
#include "core/types.h"
#include "core/color.h"

using namespace scene_builder;

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <numeric>

#ifndef V5_PTX_FILE_PATH
#error "V5_PTX_FILE_PATH must be defined (path to compiled optix_programs.ptx)"
#endif

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

#define EXPECT_NEAR(a, b, tol, msg)                      \
    do {                                                 \
        if (fabsf((a) - (b)) > (tol)) {                  \
            printf("[FAIL] %s: %.6f != %.6f (tol=%.6f)\n", msg, (float)(a), (float)(b), (float)(tol)); \
            return;                                      \
        }                                                \
    } while(0)

// ── Shared camera setup ──────────────────────────────────────────────
struct TestCamera {
    float3 pos, u, v, w;
};

static TestCamera cornell_camera() {
    float fov_rad = 60.f * PI / 180.f;
    float half_w  = tanf(fov_rad * 0.5f);
    return {
        make_f3(0.f, 0.f, 2.5f),            // pos
        make_f3(half_w, 0.f, 0.f),           // u (right)
        make_f3(0.f, half_w, 0.f),           // v (up)
        make_f3(0.f, 0.f, -1.f)              // w (forward)
    };
}

// ── Helper: check for NaN/inf ────────────────────────────────────────
static bool check_finite(const std::vector<float>& buf, const char* label) {
    for (size_t i = 0; i < buf.size(); ++i) {
        if (std::isnan(buf[i]) || std::isinf(buf[i])) {
            printf("  [FAIL] %s: NaN/inf at index %zu (value=%.6f)\n", label, i, buf[i]);
            return false;
        }
    }
    return true;
}

// ── Test 1: Emissive upload sanity ───────────────────────────────────
static void test_emissive_upload(AccelBuilder& /*builder*/) {
    TEST_BEGIN("Emissive distribution upload");

    Scene cornell = build_cornell_box();

    EXPECT_TRUE(!cornell.emissive_tri_indices.empty(),
                "Cornell box should have emissive triangles");
    EXPECT_TRUE(cornell.total_emissive_power > 0.f,
                "Total emissive power should be positive");

    printf("  Emissive tris: %zu  total_power: %.4f\n",
           cornell.emissive_tri_indices.size(), cornell.total_emissive_power);

    // Upload emissives
    LightingUploader uploader;
    uploader.upload_emissives(cornell);

    // Fill params and verify pointers are set
    LaunchParams lp = {};
    uploader.fill_params(lp);

    EXPECT_TRUE(lp.emissive_tri_indices != nullptr, "Emissive indices should be uploaded");
    EXPECT_TRUE(lp.emissive_cdf != nullptr, "Emissive CDF should be uploaded");
    EXPECT_TRUE(lp.num_emissive > 0, "num_emissive should be > 0");
    EXPECT_TRUE(lp.total_emissive_power > 0.f, "total_emissive_power should be > 0");

    printf("  LaunchParams: num_emissive=%d  power=%.4f\n",
           lp.num_emissive, lp.total_emissive_power);

    TEST_PASS("Emissive distribution upload");
}

// ── Test 2: Cornell box NEE (emissive triangle direct lighting) ──────
static void test_cornell_nee(AccelBuilder& /*builder*/) {
    TEST_BEGIN("Cornell box NEE direct lighting");

    Scene cornell = build_cornell_box();

    // Rebuild accel for this scene
    AccelBuilder nee_builder;
    nee_builder.init();
    nee_builder.build(cornell, V5_PTX_FILE_PATH);
    nee_builder.upload_geometry(cornell);
    nee_builder.upload_materials(cornell);

    // Upload lighting data
    LightingUploader uploader;
    uploader.upload_emissives(cornell);

    LaunchParams extra = {};
    uploader.fill_params(extra);

    // Camera
    TestCamera cam = cornell_camera();
    int width = 64, height = 64;
    std::vector<float> color_out;

    nee_builder.launch_test_nee(width, height, cam.pos, cam.u, cam.v, cam.w,
                                 extra, color_out);

    EXPECT_TRUE(color_out.size() == (size_t)(width * height * 3),
                "Output buffer size");

    // Check for NaN/inf
    EXPECT_TRUE(check_finite(color_out, "Cornell NEE"), "All values should be finite");

    // Center pixel should have visible illumination
    // (floor of Cornell box directly lit by ceiling light)
    int cx = width / 2, cy = height / 2;
    int ci = (cy * width + cx) * 3;
    float center_r = color_out[ci + 0];
    float center_g = color_out[ci + 1];
    float center_b = color_out[ci + 2];
    float center_lum = center_r * 0.2126f + center_g * 0.7152f + center_b * 0.0722f;

    printf("  Center pixel: (%.4f, %.4f, %.4f)  lum=%.4f\n",
           center_r, center_g, center_b, center_lum);

    // The center ray hits the back wall which should receive some direct light
    // from the ceiling emitter.  Expect non-trivial illumination.
    EXPECT_TRUE(center_lum > 0.001f, "Center pixel should have nonzero illumination");

    // Count illuminated vs dark pixels
    int lit_count = 0;
    int total = width * height;
    double total_energy = 0.0;
    for (int i = 0; i < total; ++i) {
        float r = color_out[i*3+0], g = color_out[i*3+1], b = color_out[i*3+2];
        float lum = r * 0.2126f + g * 0.7152f + b * 0.0722f;
        if (lum > 0.001f) lit_count++;
        total_energy += (double)(r + g + b);
    }
    float lit_ratio = (float)lit_count / (float)total;
    printf("  Lit pixels: %d / %d = %.1f%%\n", lit_count, total, lit_ratio * 100.f);
    printf("  Total energy: %.4f\n", total_energy);

    // Most visible surfaces should receive some direct light
    EXPECT_TRUE(lit_ratio > 0.05f, "At least 5% of pixels should be lit");

    // Emissive pixels (hitting the light directly) should be bright
    // Find max brightness
    float max_lum = 0.f;
    for (int i = 0; i < total; ++i) {
        float r = color_out[i*3+0], g = color_out[i*3+1], b = color_out[i*3+2];
        float lum = r * 0.2126f + g * 0.7152f + b * 0.0722f;
        if (lum > max_lum) max_lum = lum;
    }
    printf("  Max luminance: %.4f\n", max_lum);
    EXPECT_TRUE(max_lum > 1.0f, "Max luminance should exceed 1.0 (emissive light is 15)");

    // All values should be non-negative
    bool all_positive = true;
    for (size_t i = 0; i < color_out.size(); ++i) {
        if (color_out[i] < -0.001f) {
            printf("  Negative value at index %zu: %.6f\n", i, color_out[i]);
            all_positive = false;
            break;
        }
    }
    EXPECT_TRUE(all_positive, "All pixel values should be non-negative");

    TEST_PASS("Cornell box NEE direct lighting");
}

// ── Test 3: No NaN/inf stress test (multiple launches) ───────────────
static void test_no_nan_stress() {
    TEST_BEGIN("No NaN/inf stress (3 camera positions)");

    Scene cornell = build_cornell_box();

    AccelBuilder sb;
    sb.init();
    sb.build(cornell, V5_PTX_FILE_PATH);
    sb.upload_geometry(cornell);
    sb.upload_materials(cornell);

    LightingUploader uploader;
    uploader.upload_emissives(cornell);

    LaunchParams extra = {};
    uploader.fill_params(extra);

    // Test from 3 different camera positions
    float3 positions[] = {
        make_f3(0.f, 0.f, 2.5f),   // front
        make_f3(0.3f, 0.2f, 1.5f), // off-center
        make_f3(-0.2f, 0.4f, 0.5f) // inside-ish
    };

    float fov_rad = 60.f * PI / 180.f;
    float half_w  = tanf(fov_rad * 0.5f);

    for (int c = 0; c < 3; ++c) {
        float3 cam_pos = positions[c];
        float3 cam_u = make_f3(half_w, 0.f, 0.f);
        float3 cam_v = make_f3(0.f, half_w, 0.f);
        float3 cam_w = make_f3(0.f, 0.f, -1.f);

        std::vector<float> color_out;
        sb.launch_test_nee(32, 32, cam_pos, cam_u, cam_v, cam_w, extra, color_out);

        bool ok = check_finite(color_out, "stress");
        if (!ok) {
            printf("  [FAIL] NaN/inf from camera position %d\n", c);
            return;
        }
        printf("  Camera %d: OK (max=%.4f)\n", c,
               *std::max_element(color_out.begin(), color_out.end()));
    }

    TEST_PASS("No NaN/inf stress (3 camera positions)");
}

// =====================================================================
// main
// =====================================================================
int main() {
    printf("=== Phase 5: Direct Lighting / NEE Test ===\n");
    printf("PTX: %s\n", V5_PTX_FILE_PATH);

    try {
        AccelBuilder builder;
        builder.init();

        printf("\nGPU: %s  |  %.0f MB VRAM\n",
               builder.gpu_name().c_str(),
               (double)builder.gpu_vram_total() / (1024.0 * 1024.0));

        test_emissive_upload(builder);
        test_cornell_nee(builder);
        test_no_nan_stress();

    } catch (const std::exception& e) {
        printf("\n[FATAL] %s\n", e.what());
        return 1;
    }

    printf("\n=== Results: %d / %d tests passed ===\n", g_tests_passed, g_tests_run);
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
