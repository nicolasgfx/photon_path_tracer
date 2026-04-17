// ─────────────────────────────────────────────────────────────────────
// tests/unit/test_acceleration.cpp – Acceleration structure unit tests
//
// Stage 2: Type sizes, LaunchParams layout, pipeline constants.
// CPU-only struct/type validation. Compiled without PPT_USE_OPTIX.
// (Full GPU build+trace tests are in the standalone accel_test target.)
// ─────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include "accel/accel_types.h"
#include "core/types.h"
#include "core/config.h"
#include <cstring>

// ── Pipeline constants ──────────────────────────────────────────────

TEST(Acceleration, RayTypeConstants) {
    EXPECT_EQ(accel::RAY_TYPE_RADIANCE, 0);
    EXPECT_EQ(accel::RAY_TYPE_SHADOW, 1);
    EXPECT_EQ(accel::NUM_RAY_TYPES, 2);
}

TEST(Acceleration, StackSizeReasonable) {
    EXPECT_GE(accel::STACK_SIZE, 4096);
    EXPECT_LE(accel::STACK_SIZE, 65536);
}

TEST(Acceleration, MaxTraceDepth) {
    EXPECT_GE(accel::MAX_TRACE_DEPTH, 1);
    EXPECT_LE(accel::MAX_TRACE_DEPTH, 16);
}

TEST(Acceleration, SceneEpsilon) {
    EXPECT_GT(accel::SCENE_EPSILON, 0.0f);
    EXPECT_LT(accel::SCENE_EPSILON, 0.1f);
}

TEST(Acceleration, PayloadValues) {
    // 18 payload values = enough for pos(3) + sn(3) + t(1) + mat(1) + tri(1)
    // + hit(1) + gn(3) + uv(2) + tangent(3) = 18
    EXPECT_EQ(accel::NUM_PAYLOAD_VALUES, 18);
}
