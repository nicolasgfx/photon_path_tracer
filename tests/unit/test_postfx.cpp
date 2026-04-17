// ─────────────────────────────────────────────────────────────────────
// tests/unit/test_postfx.cpp – Post-processing unit tests
//
// Stage 9: PostFxParams defaults, tonemap formula checks.
// CPU-only parameter and formula validation.
// (Full GPU pipeline tests are in the standalone postfx_test target.)
// ─────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include "postfx/postfx_params.h"
#include "core/config.h"
#include <cmath>

// ── PostFxParams defaults ───────────────────────────────────────────

TEST(PostFx, ParamsDefaults) {
    PostFxParams p{};
    EXPECT_FLOAT_EQ(p.exposure, DEFAULT_EXPOSURE);
    EXPECT_EQ(p.use_aces, USE_ACES_TONEMAPPING);
    EXPECT_GT(p.firefly_threshold, 0.0f);
}

TEST(PostFx, ParamsFieldAccess) {
    PostFxParams p{};
    p.exposure = 2.0f;
    p.bloom_intensity = 0.1f;
    p.bloom_radius_h = 5.0f;

    EXPECT_FLOAT_EQ(p.exposure, 2.0f);
    EXPECT_FLOAT_EQ(p.bloom_intensity, 0.1f);
    EXPECT_FLOAT_EQ(p.bloom_radius_h, 5.0f);
}

// ── ACES tonemap formula (CPU reference) ────────────────────────────
// ACES RRT+ODT simplified fit: (x(2.51x+0.03))/(x(2.43x+0.59)+0.14)

static float aces_cpu(float x) {
    float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

TEST(PostFx, ACESBlackIsBlack) {
    EXPECT_NEAR(aces_cpu(0.0f), 0.0f, 1e-6f);
}

TEST(PostFx, ACESMidGrey) {
    float mid = aces_cpu(0.18f);
    EXPECT_GT(mid, 0.0f);
    EXPECT_LT(mid, 1.0f);
}

TEST(PostFx, ACESHighValueClamps) {
    float bright = aces_cpu(100.0f);
    EXPECT_GT(bright, 0.9f);
    EXPECT_LE(bright, 1.05f);  // ACES approaches but doesn't exceed ~1.0
}

// ── sRGB gamma ──────────────────────────────────────────────────────

static float linear_to_srgb(float x) {
    if (x <= 0.0031308f ) return x * 12.92f;
    return 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
}

TEST(PostFx, sRGBGammaBlack) {
    EXPECT_NEAR(linear_to_srgb(0.0f), 0.0f, 1e-6f);
}

TEST(PostFx, sRGBGammaWhite) {
    EXPECT_NEAR(linear_to_srgb(1.0f), 1.0f, 1e-3f);
}

TEST(PostFx, sRGBGammaMidGrey) {
    // Linear 0.18 → sRGB ≈ 0.46
    float srgb = linear_to_srgb(0.18f);
    EXPECT_GT(srgb, 0.3f);
    EXPECT_LT(srgb, 0.6f);
}
