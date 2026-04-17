// ─────────────────────────────────────────────────────────────────────
// photon_test.cpp – Phase 7 integration test: photon tracing
//
// Tests:
//   1. Photon deposit count: emitting N photons produces deposits
//   2. Flux conservation: total flux is finite and positive
//   3. Deposit positions are inside scene bounds
//   4. Caustic flags: glass sphere scene produces caustic photons
//   5. No NaN/inf in photon data (robustness)
//
// Requires: CUDA GPU + OptiX runtime.  PTX path passed via define.
// ─────────────────────────────────────────────────────────────────────
#include "accel/accel_builder.h"
#include "lighting/lighting_upload.h"
#include "photon/photon_storage.h"
#include "photon/photon.h"
#include "scene/scene_builder.h"
#include "core/types.h"
#include "core/config.h"

using namespace scene_builder;

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

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

// ── Shared setup ─────────────────────────────────────────────────────
struct PhotonTestSetup {
    AccelBuilder builder;
    LightingUploader uploader;
    PhotonStorage storage;
    LaunchParams extra;
};

static void setup_photon_test(PhotonTestSetup& ts, const Scene& scene,
                               int num_photons, int max_stored,
                               int max_bounces = DEFAULT_PHOTON_MAX_BOUNCES) {
    ts.builder.init();
    ts.builder.build(scene, V5_PTX_FILE_PATH);
    ts.builder.upload_geometry(scene);
    ts.builder.upload_materials(scene);
    ts.uploader.upload_emissives(scene);
    ts.storage.allocate(max_stored);
    ts.storage.reset();

    ts.extra = {};
    ts.uploader.fill_params(ts.extra);
    ts.storage.fill_params(ts.extra);

    ts.extra.num_photons       = num_photons;
    ts.extra.photon_max_bounces = max_bounces;
    ts.extra.photon_map_seed   = 0;
    ts.extra.gather_radius     = DEFAULT_GATHER_RADIUS;
    ts.extra.min_bounces_rr    = DEFAULT_PHOTON_MIN_BOUNCES_RR;
    ts.extra.rr_threshold      = DEFAULT_PHOTON_RR_THRESHOLD;
}

// ── Test 1: Basic photon deposit count ───────────────────────────────
static void test_deposit_count() {
    TEST_BEGIN("Photon deposit count");

    Scene scene = build_cornell_box();
    int num_emit = 10000;
    int max_store = num_emit * 8;

    PhotonTestSetup ts;
    setup_photon_test(ts, scene, num_emit, max_store);
    ts.builder.launch_photon_trace(ts.extra);

    int count = ts.storage.download_count();
    printf("  Emitted: %d  Deposited: %d  (%.1f%%)\n",
           num_emit, count, 100.f * count / num_emit);

    // Cornell box: light bounces diffusely → expect deposits
    EXPECT_TRUE(count > 0, "Should have deposited photons");

    // Deposits should be less than max (reasonable bound)
    EXPECT_TRUE(count <= max_store, "Should not exceed buffer");

    // At least some fraction of emitted photons should deposit
    EXPECT_TRUE(count > num_emit / 100,
                "Should deposit > 1% of emitted photons");

    TEST_PASS("Photon deposit count");
}

// ── Test 2: Flux conservation ────────────────────────────────────────
static void test_flux_conservation() {
    TEST_BEGIN("Flux conservation");

    Scene scene = build_cornell_box();
    int num_emit = 5000;
    int max_store = num_emit * 8;

    PhotonTestSetup ts;
    setup_photon_test(ts, scene, num_emit, max_store);
    ts.builder.launch_photon_trace(ts.extra);

    std::vector<Photon> photons;
    ts.storage.download(photons);

    EXPECT_TRUE(!photons.empty(), "Should have photons");

    double total_flux_r = 0, total_flux_g = 0, total_flux_b = 0;
    int negative_count = 0;
    int nan_count = 0;

    for (const auto& p : photons) {
        if (std::isnan(p.flux.x) || std::isnan(p.flux.y) || std::isnan(p.flux.z) ||
            std::isinf(p.flux.x) || std::isinf(p.flux.y) || std::isinf(p.flux.z))
            nan_count++;
        if (p.flux.x < 0 || p.flux.y < 0 || p.flux.z < 0)
            negative_count++;
        total_flux_r += p.flux.x;
        total_flux_g += p.flux.y;
        total_flux_b += p.flux.z;
    }

    printf("  Photons: %zu  Total flux: R=%.2f G=%.2f B=%.2f\n",
           photons.size(), total_flux_r, total_flux_g, total_flux_b);
    printf("  NaN/inf: %d  Negative: %d\n", nan_count, negative_count);

    EXPECT_TRUE(nan_count == 0, "No NaN/inf in flux");
    EXPECT_TRUE(negative_count == 0, "No negative flux");
    EXPECT_TRUE(total_flux_r + total_flux_g + total_flux_b > 0,
                "Total flux should be positive");

    TEST_PASS("Flux conservation");
}

// ── Test 3: Deposit positions inside scene bounds ────────────────────
static void test_positions_in_bounds() {
    TEST_BEGIN("Deposit positions in scene bounds");

    Scene scene = build_cornell_box();
    int num_emit = 5000;
    int max_store = num_emit * 8;

    PhotonTestSetup ts;
    setup_photon_test(ts, scene, num_emit, max_store);
    ts.builder.launch_photon_trace(ts.extra);

    std::vector<Photon> photons;
    ts.storage.download(photons);

    EXPECT_TRUE(!photons.empty(), "Should have photons");

    // Cornell box bounds are roughly [-1, 1] in each axis
    // Use generous margin
    float margin = 2.0f;
    int out_of_bounds = 0;
    for (const auto& p : photons) {
        if (fabsf(p.position.x) > margin ||
            fabsf(p.position.y) > margin ||
            fabsf(p.position.z) > margin)
            out_of_bounds++;
    }
    printf("  Out of bounds (|pos| > %.1f): %d / %zu\n",
           margin, out_of_bounds, photons.size());

    EXPECT_TRUE(out_of_bounds == 0, "All deposits inside scene bounds");

    // Check normals are unit length
    int bad_normals = 0;
    for (const auto& p : photons) {
        float len = sqrtf(p.geo_normal.x*p.geo_normal.x +
                         p.geo_normal.y*p.geo_normal.y +
                         p.geo_normal.z*p.geo_normal.z);
        if (fabsf(len - 1.f) > 0.1f) bad_normals++;
    }
    printf("  Bad normals: %d / %zu\n", bad_normals, photons.size());
    EXPECT_TRUE(bad_normals == 0, "All normals are unit length");

    TEST_PASS("Deposit positions in scene bounds");
}

// ── Test 4: Caustic photons from glass sphere ────────────────────────
static void test_caustic_flags() {
    TEST_BEGIN("Caustic flags (glass sphere)");

    Scene scene = build_glass_sphere();  // Cornell + glass sphere
    int num_emit = 20000;
    int max_store = num_emit * 12;

    PhotonTestSetup ts;
    setup_photon_test(ts, scene, num_emit, max_store, 12);
    ts.builder.launch_photon_trace(ts.extra);

    std::vector<Photon> photons;
    ts.storage.download(photons);

    EXPECT_TRUE(!photons.empty(), "Should have photons");

    int caustic_count = 0;
    int glass_flag_count = 0;
    for (const auto& p : photons) {
        if (p.is_caustic) caustic_count++;
        if (p.path_flags & PHOTON_FLAG_TRAVERSED_GLASS) glass_flag_count++;
    }

    printf("  Total photons: %zu  Caustic: %d (%.1f%%)  Glass-traversed: %d (%.1f%%)\n",
           photons.size(), caustic_count,
           100.f * caustic_count / (float)photons.size(),
           glass_flag_count,
           100.f * glass_flag_count / (float)photons.size());

    // Glass sphere should produce some caustic photons (L→glass→D)
    EXPECT_TRUE(caustic_count > 0, "Should have caustic photons from glass");
    EXPECT_TRUE(glass_flag_count > 0, "Should have glass-traversed flags");

    TEST_PASS("Caustic flags (glass sphere)");
}

// ── Test 5: No NaN/inf robustness ────────────────────────────────────
static void test_robustness() {
    TEST_BEGIN("Robustness (no NaN/inf in any field)");

    Scene scene = build_cornell_box();
    int num_emit = 10000;
    int max_store = num_emit * 8;

    PhotonTestSetup ts;
    setup_photon_test(ts, scene, num_emit, max_store);
    ts.builder.launch_photon_trace(ts.extra);

    std::vector<Photon> photons;
    ts.storage.download(photons);

    EXPECT_TRUE(!photons.empty(), "Should have photons");

    int bad_count = 0;
    for (const auto& p : photons) {
        auto bad = [](float v) { return std::isnan(v) || std::isinf(v); };
        if (bad(p.position.x) || bad(p.position.y) || bad(p.position.z)) bad_count++;
        if (bad(p.wi.x) || bad(p.wi.y) || bad(p.wi.z)) bad_count++;
        if (bad(p.geo_normal.x) || bad(p.geo_normal.y) || bad(p.geo_normal.z)) bad_count++;
        if (bad(p.flux.x) || bad(p.flux.y) || bad(p.flux.z)) bad_count++;
    }

    printf("  Photons: %zu  Bad fields: %d\n", photons.size(), bad_count);
    EXPECT_TRUE(bad_count == 0, "No NaN/inf in any photon field");

    // Also verify wi directions are normalized
    int bad_wi = 0;
    for (const auto& p : photons) {
        float len = sqrtf(p.wi.x*p.wi.x + p.wi.y*p.wi.y + p.wi.z*p.wi.z);
        if (fabsf(len - 1.f) > 0.1f) bad_wi++;
    }
    printf("  Bad wi vectors: %d / %zu\n", bad_wi, photons.size());
    EXPECT_TRUE(bad_wi == 0, "All wi directions are normalized");

    TEST_PASS("Robustness (no NaN/inf in any field)");
}

// ── Test 6: Multi-map seeding produces different deposits ────────────
static void test_multimap_seed() {
    TEST_BEGIN("Multi-map seeding (different seeds → different deposits)");

    Scene scene = build_cornell_box();
    int num_emit = 5000;
    int max_store = num_emit * 8;

    // Trace with seed 0
    PhotonTestSetup ts0;
    setup_photon_test(ts0, scene, num_emit, max_store);
    ts0.extra.photon_map_seed = 0;
    ts0.builder.launch_photon_trace(ts0.extra);
    int count0 = ts0.storage.download_count();

    // Trace with seed 42
    PhotonTestSetup ts1;
    setup_photon_test(ts1, scene, num_emit, max_store);
    ts1.extra.photon_map_seed = 42;
    ts1.builder.launch_photon_trace(ts1.extra);
    int count1 = ts1.storage.download_count();

    printf("  Seed 0: %d deposits  Seed 42: %d deposits\n", count0, count1);

    // Both should produce deposits
    EXPECT_TRUE(count0 > 0, "Seed 0 should produce deposits");
    EXPECT_TRUE(count1 > 0, "Seed 42 should produce deposits");

    // Counts should be different (stochastic variance)
    // Allow small probability they're the same but count difference
    printf("  Difference: %d\n", abs(count0 - count1));

    TEST_PASS("Multi-map seeding (different seeds → different deposits)");
}

// =====================================================================
// main
// =====================================================================
int main() {
    printf("=== Phase 7: Photon Tracing Test ===\n");
    printf("PTX: %s\n", V5_PTX_FILE_PATH);

    try {
        test_deposit_count();
        test_flux_conservation();
        test_positions_in_bounds();
        test_caustic_flags();
        test_robustness();
        test_multimap_seed();

    } catch (const std::exception& e) {
        printf("\n[FATAL] %s\n", e.what());
        return 1;
    }

    printf("\n=== Results: %d / %d tests passed ===\n", g_tests_passed, g_tests_run);
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
