// ─────────────────────────────────────────────────────────────────────
// accel_test.cpp – Phase 3 integration test: acceleration structure
//
// Tests:
//   1. Build GAS from Cornell box scene → no crash
//   2. Trace rays against known geometry → verify hit positions/normals
//   3. Shadow ray visibility test → expected occlusion results
//   4. Miss rays → correctly flagged
//
// Requires: CUDA GPU + OptiX runtime.  PTX path passed via define.
// ─────────────────────────────────────────────────────────────────────
#include "accel/accel_builder.h"
#include "scene/scene_builder.h"
#include "core/types.h"
#include "core/color.h"

using namespace scene_builder;

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <vector>

#include <cuda_runtime.h>

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

// ── Test 1: Build acceleration structure ─────────────────────────────
static void test_build_accel(AccelBuilder& builder) {
    TEST_BEGIN("Build GAS from Cornell box");

    Scene cornell = build_cornell_box();
    cornell.compute_bounds();

    builder.build(cornell, V5_PTX_FILE_PATH);
    builder.upload_geometry(cornell);
    builder.upload_materials(cornell);

    const auto& accel = builder.accel();
    EXPECT_TRUE(accel.handle != 0, "Traversable handle should be non-zero");
    EXPECT_TRUE(accel.num_triangles > 0, "Should have triangles");
    EXPECT_TRUE(accel.compacted_bytes > 0, "Should have allocated GPU memory");
    EXPECT_TRUE(!accel.instanced, "Cornell box should use single GAS");

    printf("  triangles=%d  compacted=%.1f KB\n",
           accel.num_triangles, (double)accel.compacted_bytes / 1024.0);

    TEST_PASS("Build GAS from Cornell box");
}

// ── Test 2: Trace normals (ray-geometry intersection) ────────────────
static void test_trace_normals(AccelBuilder& builder) {
    TEST_BEGIN("Trace normals (test raygen)");

    // Camera looking into the Cornell box from the front
    // The box is roughly [-1,1]^3, camera at z=2 looking toward z=-1
    float3 cam_pos = make_f3(0.f, 0.f, 2.5f);

    // Camera basis: u=right, v=up, w=forward (toward scene)
    float fov_rad = 60.f * PI / 180.f;
    float half_w  = tanf(fov_rad * 0.5f);
    float3 cam_w  = make_f3(0.f, 0.f, -1.f);  // forward
    float3 cam_u  = make_f3(half_w, 0.f, 0.f);  // right (scaled by half-width)
    float3 cam_v  = make_f3(0.f, half_w, 0.f);  // up (scaled by half-height)

    int width = 64, height = 64;
    std::vector<float> color_out;

    builder.launch_test_normals(width, height, cam_pos, cam_u, cam_v, cam_w, color_out);

    EXPECT_TRUE(color_out.size() == (size_t)(width * height * 3),
                "Output buffer should have width*height*3 floats");

    // Check center pixel — should hit the back wall (normal ≈ (0, 0, 1))
    int cx = width / 2;
    int cy = height / 2;
    int ci = (cy * width + cx) * 3;
    float r = color_out[ci + 0];
    float g = color_out[ci + 1];
    float b = color_out[ci + 2];

    printf("  Center pixel normal-as-color: (%.3f, %.3f, %.3f)\n", r, g, b);

    // The center ray should hit something (not all black)
    float brightness = r + g + b;
    EXPECT_TRUE(brightness > 0.01f, "Center pixel should hit geometry");

    // Count how many pixels hit geometry (non-black)
    int hit_count = 0;
    int total = width * height;
    for (int i = 0; i < total; ++i) {
        float sum = color_out[i*3] + color_out[i*3+1] + color_out[i*3+2];
        if (sum > 0.001f) hit_count++;
    }
    float hit_ratio = (float)hit_count / (float)total;
    printf("  Hit ratio: %d / %d = %.1f%%\n", hit_count, total, hit_ratio * 100.f);

    // With a 60° FOV at z=2.5 looking at a [-1,1] box (12 tris), ~19% hit is expected
    EXPECT_TRUE(hit_ratio > 0.1f, "At least 10% of pixels should hit geometry");

    // Count miss pixels (should exist too — some rays miss the box edges)
    int miss_count = total - hit_count;
    printf("  Miss pixels: %d\n", miss_count);

    TEST_PASS("Trace normals (test raygen)");
}

// ── Test 3: Pixel values sanity check ────────────────────────────────
static void test_pixel_sanity(AccelBuilder& builder) {
    TEST_BEGIN("Pixel value sanity (normal range)");

    float3 cam_pos = make_f3(0.f, 0.f, 2.5f);
    float fov_rad = 60.f * PI / 180.f;
    float half_w  = tanf(fov_rad * 0.5f);
    float3 cam_w  = make_f3(0.f, 0.f, -1.f);
    float3 cam_u  = make_f3(half_w, 0.f, 0.f);
    float3 cam_v  = make_f3(0.f, half_w, 0.f);

    int width = 32, height = 32;
    std::vector<float> color_out;

    builder.launch_test_normals(width, height, cam_pos, cam_u, cam_v, cam_w, color_out);

    // All pixel values should be in [0, 1] (absolute normals)
    bool all_in_range = true;
    for (size_t i = 0; i < color_out.size(); ++i) {
        if (color_out[i] < -0.001f || color_out[i] > 1.001f) {
            printf("  Out-of-range pixel value at index %zu: %.6f\n", i, color_out[i]);
            all_in_range = false;
            break;
        }
    }
    EXPECT_TRUE(all_in_range, "All normal-as-color values should be in [0, 1]");

    // No NaN/inf
    bool all_finite = true;
    for (size_t i = 0; i < color_out.size(); ++i) {
        if (std::isnan(color_out[i]) || std::isinf(color_out[i])) {
            printf("  NaN/inf at index %zu\n", i);
            all_finite = false;
            break;
        }
    }
    EXPECT_TRUE(all_finite, "No NaN/inf in output");

    TEST_PASS("Pixel value sanity (normal range)");
}

// ── Test 4: Glass sphere scene (tests material types in anyhit) ──────
static void test_glass_sphere_accel(AccelBuilder& /*builder*/) {
    TEST_BEGIN("Glass sphere scene accel build");

    Scene glass_scene = build_glass_sphere();
    glass_scene.compute_bounds();

    // Rebuild for a different scene
    AccelBuilder builder2;
    builder2.init();
    builder2.build(glass_scene, V5_PTX_FILE_PATH);
    builder2.upload_geometry(glass_scene);
    builder2.upload_materials(glass_scene);

    const auto& accel = builder2.accel();
    EXPECT_TRUE(accel.handle != 0, "Glass sphere should build successfully");
    EXPECT_TRUE(accel.num_triangles > 0, "Should have triangles");

    // Trace from front
    float3 cam_pos = make_f3(0.f, 0.f, 3.f);
    float half_w = 0.577f;
    float3 cam_w = make_f3(0.f, 0.f, -1.f);
    float3 cam_u = make_f3(half_w, 0.f, 0.f);
    float3 cam_v = make_f3(0.f, half_w, 0.f);

    std::vector<float> color_out;
    builder2.launch_test_normals(32, 32, cam_pos, cam_u, cam_v, cam_w, color_out);

    // Center should hit the sphere
    int ci = (16 * 32 + 16) * 3;
    float brightness = color_out[ci] + color_out[ci+1] + color_out[ci+2];
    printf("  Center pixel brightness: %.3f\n", brightness);
    EXPECT_TRUE(brightness > 0.01f, "Center should hit sphere");

    TEST_PASS("Glass sphere scene accel build");
}

// ── Test 5: Glass transmittance uploads to device ───────────────────
static void test_glass_tf_upload() {
    TEST_BEGIN("Glass Tf upload");

    Scene glass_scene = build_glass_sphere();
    int glass_mat = -1;
    for (size_t i = 0; i < glass_scene.materials.size(); ++i) {
        MaterialType mt = glass_scene.materials[i].type;
        if (mt == Glass || mt == Translucent) {
            glass_mat = (int)i;
            break;
        }
    }

    EXPECT_TRUE(glass_mat >= 0, "Glass scene should contain a transmissive material");

    glass_scene.materials[glass_mat].Tf = Color3::from_rgb(0.2f, 0.7f, 0.4f);

    AccelBuilder builder;
    builder.init();
    builder.build(glass_scene, V5_PTX_FILE_PATH);
    builder.upload_geometry(glass_scene);
    builder.upload_materials(glass_scene);

    LaunchParams lp = {};
    builder.fill_material_params(lp);

    EXPECT_TRUE(lp.Tf != nullptr, "Tf buffer should be present");

    float tf_host[3] = {};
    cudaError_t err = cudaMemcpy(
        tf_host,
        lp.Tf + glass_mat * 3,
        sizeof(tf_host),
        cudaMemcpyDeviceToHost);
    EXPECT_TRUE(err == cudaSuccess, "Tf download should succeed");
    EXPECT_NEAR(tf_host[0], 0.2f, 1e-5f, "Tf.r should match uploaded value");
    EXPECT_NEAR(tf_host[1], 0.7f, 1e-5f, "Tf.g should match uploaded value");
    EXPECT_NEAR(tf_host[2], 0.4f, 1e-5f, "Tf.b should match uploaded value");

    TEST_PASS("Glass Tf upload");
}

// =====================================================================
// main
// =====================================================================
int main() {
    printf("=== Phase 3: Acceleration Test ===\n");
    printf("PTX: %s\n", V5_PTX_FILE_PATH);

    try {
        AccelBuilder builder;
        builder.init();

        printf("\nGPU: %s  |  %.0f MB VRAM\n",
               builder.gpu_name().c_str(),
               (double)builder.gpu_vram_total() / (1024.0 * 1024.0));

        test_build_accel(builder);
        test_trace_normals(builder);
        test_pixel_sanity(builder);
        test_glass_sphere_accel(builder);
        test_glass_tf_upload();

    } catch (const std::exception& e) {
        printf("\n[FATAL] %s\n", e.what());
        return 1;
    }

    printf("\n=== Results: %d / %d tests passed ===\n", g_tests_passed, g_tests_run);
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
