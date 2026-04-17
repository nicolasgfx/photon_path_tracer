// ─────────────────────────────────────────────────────────────────────
// tests/unit/test_integrator.cpp – Path integrator unit tests
//
// Stage 6: PathResult, RRResult, sample clamping.
// CPU-only type and formula tests.
// ─────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include "integrator/path_state.h"
#include "integrator/russian_roulette.h"
#include "integrator/sample_clamping.h"
#include "core/types.h"
#include "core/config.h"
#include <cmath>

// ── PathResult / DevBSDFSample ──────────────────────────────────────

TEST(Integrator, PathResultInit) {
    PathResult pr{};
    pr.radiance = make_f3(0, 0, 0);
    pr.albedo   = make_f3(0.5f, 0.5f, 0.5f);
    pr.normal   = make_f3(0, 1, 0);

    EXPECT_FLOAT_EQ(pr.radiance.x, 0.0f);
    EXPECT_FLOAT_EQ(pr.albedo.x, 0.5f);
    EXPECT_FLOAT_EQ(pr.normal.y, 1.0f);
}

TEST(Integrator, DevBSDFSampleFields) {
    DevBSDFSample s{};
    s.wi = make_f3(0, 0, 1);
    s.pdf = 0.318f;
    s.f = make_f3(0.25f, 0.25f, 0.25f);
    s.is_specular = false;

    EXPECT_NEAR(s.pdf, 0.318f, 1e-5f);
    EXPECT_FALSE(s.is_specular);
}

// ── Russian roulette ────────────────────────────────────────────────

TEST(Integrator, RRBrightPathSurvives) {
    // max_tp = 0.9, threshold = 0.95, xi = 0.5
    // p_survive = min(0.95, 0.9) = 0.9; xi < p_survive → survives
    RRResult rr = russian_roulette(0.9f, 0.95f, 0.5f);
    EXPECT_FALSE(rr.terminate);
    EXPECT_NEAR(rr.inv_survival, 1.0f / 0.9f, 1e-4f);
}

TEST(Integrator, RRDimPathTerminates) {
    // max_tp = 0.1, threshold = 0.95, xi = 0.5
    // p_survive = 0.1; xi >= p_survive → terminates
    RRResult rr = russian_roulette(0.1f, 0.95f, 0.5f);
    EXPECT_TRUE(rr.terminate);
}

TEST(Integrator, RRZeroThroughputTerminates) {
    RRResult rr = russian_roulette(0.0f, 0.95f, 0.5f);
    EXPECT_TRUE(rr.terminate);
}

TEST(Integrator, RRUnbiased) {
    // For a surviving path, throughput * inv_survival >= original
    float max_tp = 0.6f;
    RRResult rr = russian_roulette(max_tp, 0.95f, 0.1f);
    EXPECT_FALSE(rr.terminate);
    float corrected = max_tp * rr.inv_survival;
    EXPECT_GE(corrected, max_tp - 1e-5f);
}

// ── Sample clamping ─────────────────────────────────────────────────

TEST(Integrator, ClampBounceNormal) {
    float3 val = make_f3(0.5f, 0.3f, 0.1f);
    float3 clamped = clamp_bounce_contribution(val, DEFAULT_MAX_BOUNCE_CONTRIBUTION);
    EXPECT_FLOAT_EQ(clamped.x, 0.5f);
    EXPECT_FLOAT_EQ(clamped.y, 0.3f);
}

TEST(Integrator, ClampBounceOutlier) {
    float3 val = make_f3(1e6f, 0.5f, 0.1f);
    float3 clamped = clamp_bounce_contribution(val, DEFAULT_MAX_BOUNCE_CONTRIBUTION);
    EXPECT_LE(clamped.x, DEFAULT_MAX_BOUNCE_CONTRIBUTION);
    EXPECT_FLOAT_EQ(clamped.y, 0.5f);
}

TEST(Integrator, ClampPathThroughput) {
    float3 high = make_f3(1e4f, 0.5f, 0.1f);
    float3 clamped = clamp_path_throughput(high, DEFAULT_MAX_PATH_THROUGHPUT);
    float mx = fmaxf(clamped.x, fmaxf(clamped.y, clamped.z));
    EXPECT_LE(mx, DEFAULT_MAX_PATH_THROUGHPUT + 1e-5f);
}

TEST(Integrator, ClampSampleLuminance) {
    float3 bright = make_f3(1e5f, 1e5f, 1e5f);
    float3 clamped = clamp_sample_luminance(bright, DEFAULT_MAX_SAMPLE_LUMINANCE);
    float lum = clamped.x * 0.2126f + clamped.y * 0.7152f + clamped.z * 0.0722f;
    EXPECT_LE(lum, DEFAULT_MAX_SAMPLE_LUMINANCE + 1e-3f);
}

TEST(Integrator, ClampZeroUnchanged) {
    float3 zero = make_f3(0, 0, 0);
    EXPECT_FLOAT_EQ(clamp_sample_luminance(zero, DEFAULT_MAX_SAMPLE_LUMINANCE).x, 0.0f);
    EXPECT_FLOAT_EQ(clamp_path_throughput(zero, DEFAULT_MAX_PATH_THROUGHPUT).x, 0.0f);
    EXPECT_FLOAT_EQ(clamp_bounce_contribution(zero, DEFAULT_MAX_BOUNCE_CONTRIBUTION).x, 0.0f);
}
