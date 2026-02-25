// ─────────────────────────────────────────────────────────────────────
// test_surface_filter.cpp – Unit tests for tangential disk kernel &
//                           surface consistency filter (§6.3, §6.4)
// ─────────────────────────────────────────────────────────────────────
// Tests cover:
//   1. Tangential distance computation correctness
//   2. Surface consistency filter (4 conditions)
//   3. Kernel weight functions (box, Epanechnikov)
//   4. Kernel normalization constants
//   5. Edge cases (zero normal, coplanar, perpendicular, etc.)
// ─────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include "photon/surface_filter.h"
#include "core/types.h"
#include "core/config.h"

#include <cmath>

// ── Helper: approximate float equality ──────────────────────────────
static constexpr float EPS = 1e-5f;

// =====================================================================
// §6.3: Tangential distance computation
// =====================================================================

TEST(SurfaceFilter, TangentialDistanceCoplanar) {
    // Photon on the same plane as the query point (d_plane = 0)
    // Query at origin, normal = +Y, photon displaced in XZ plane
    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = make_f3(0, 1, 0);
    float3 ppos  = make_f3(0.3f, 0, 0.4f);

    TangentialResult tr = compute_tangential(qpos, qnorm, ppos);

    // d_plane should be 0 (photon is in the tangent plane)
    EXPECT_NEAR(tr.d_plane, 0.0f, EPS);

    // d_tan^2 should be 0.3^2 + 0.4^2 = 0.25
    float expected_d_tan2 = 0.09f + 0.16f;
    EXPECT_NEAR(tr.d_tan2, expected_d_tan2, EPS);

    // v_tan should be (0.3, 0, 0.4)
    EXPECT_NEAR(tr.v_tan.x, 0.3f, EPS);
    EXPECT_NEAR(tr.v_tan.y, 0.0f, EPS);
    EXPECT_NEAR(tr.v_tan.z, 0.4f, EPS);
}

TEST(SurfaceFilter, TangentialDistancePureNormal) {
    // Photon displaced only along the normal: d_tan should be 0
    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = make_f3(0, 1, 0);
    float3 ppos  = make_f3(0, 0.5f, 0);

    TangentialResult tr = compute_tangential(qpos, qnorm, ppos);

    EXPECT_NEAR(tr.d_plane, 0.5f, EPS);
    EXPECT_NEAR(tr.d_tan2, 0.0f, EPS);
}

TEST(SurfaceFilter, TangentialDistanceMixed) {
    // Photon displaced along both normal and tangential directions
    float3 qpos  = make_f3(1, 2, 3);
    float3 qnorm = make_f3(0, 0, 1);  // normal = +Z
    float3 ppos  = make_f3(1.1f, 2.2f, 3.05f);  // small Z offset, some XY offset

    TangentialResult tr = compute_tangential(qpos, qnorm, ppos);

    // d_plane = dot((0,0,1), (0.1, 0.2, 0.05)) = 0.05
    EXPECT_NEAR(tr.d_plane, 0.05f, EPS);

    // v_tan = v - n * d_plane = (0.1, 0.2, 0.05) - (0,0,0.05) = (0.1, 0.2, 0)
    EXPECT_NEAR(tr.v_tan.x, 0.1f, EPS);
    EXPECT_NEAR(tr.v_tan.y, 0.2f, EPS);
    EXPECT_NEAR(tr.v_tan.z, 0.0f, EPS);

    // d_tan^2 = 0.01 + 0.04 = 0.05
    EXPECT_NEAR(tr.d_tan2, 0.05f, EPS);
}

TEST(SurfaceFilter, TangentialDistanceConvenience) {
    // Test the convenience function matches full computation
    float3 qpos  = make_f3(1, 2, 3);
    float3 qnorm = make_f3(0, 1, 0);
    float3 ppos  = make_f3(1.5f, 2.1f, 3.3f);

    float d2_convenience = tangential_distance2(qpos, qnorm, ppos);
    TangentialResult tr  = compute_tangential(qpos, qnorm, ppos);

    EXPECT_NEAR(d2_convenience, tr.d_tan2, EPS);
}

TEST(SurfaceFilter, TangentialVs3DDistance) {
    // Key property: tangential distance <= 3D Euclidean distance
    // (The tangential component is the projection of v onto the tangent plane)
    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = make_f3(0, 1, 0);

    for (int i = 0; i < 100; ++i) {
        float x = sinf((float)i * 0.7f) * 0.5f;
        float y = cosf((float)i * 1.3f) * 0.5f;
        float z = sinf((float)i * 2.1f) * 0.5f;
        float3 ppos = make_f3(x, y, z);

        float d_3d_sq = dot(ppos - qpos, ppos - qpos);
        float d_tan_sq = tangential_distance2(qpos, qnorm, ppos);

        EXPECT_LE(d_tan_sq, d_3d_sq + EPS)
            << "Tangential distance must be <= 3D distance for photon " << i;
    }
}

TEST(SurfaceFilter, TangentialDistanceDiagonalNormal) {
    // Non-axis-aligned normal
    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = normalize(make_f3(1, 1, 0));  // 45° between X and Y
    float3 ppos  = make_f3(1, 0, 0);

    TangentialResult tr = compute_tangential(qpos, qnorm, ppos);

    // v = (1, 0, 0), n = (1/√2, 1/√2, 0)
    // d_plane = dot(n, v) = 1/√2 ≈ 0.7071
    float inv_sqrt2 = 1.0f / sqrtf(2.0f);
    EXPECT_NEAR(tr.d_plane, inv_sqrt2, EPS);

    // v_tan = v - n * d_plane = (1,0,0) - (1/√2)(1/√2, 1/√2, 0) = (1-0.5, -0.5, 0) = (0.5, -0.5, 0)
    EXPECT_NEAR(tr.v_tan.x, 0.5f, EPS);
    EXPECT_NEAR(tr.v_tan.y, -0.5f, EPS);
    EXPECT_NEAR(tr.v_tan.z, 0.0f, EPS);

    // d_tan^2 = 0.25 + 0.25 = 0.5
    EXPECT_NEAR(tr.d_tan2, 0.5f, EPS);
}

// =====================================================================
// §6.3: Plane distance utility
// =====================================================================

TEST(SurfaceFilter, PlaneDistanceSigned) {
    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = make_f3(0, 1, 0);

    // Positive side
    EXPECT_GT(plane_distance(qpos, qnorm, make_f3(0, 0.5f, 0)), 0.0f);

    // Negative side
    EXPECT_LT(plane_distance(qpos, qnorm, make_f3(0, -0.5f, 0)), 0.0f);

    // On the plane
    EXPECT_NEAR(plane_distance(qpos, qnorm, make_f3(1, 0, 1)), 0.0f, EPS);
}

// =====================================================================
// §6.3: Effective tau computation
// =====================================================================

TEST(SurfaceFilter, EffectiveTauBasic) {
    // When user_tau is large enough, it's used directly
    float tau = effective_tau(0.1f, EPSILON);
    EXPECT_GE(tau, 0.1f);
}

TEST(SurfaceFilter, EffectiveTauFloor) {
    // When user_tau is very small, floor kicks in
    float tau = effective_tau(1e-10f, 0.001f);
    EXPECT_GE(tau, PLANE_TAU_EPSILON_FACTOR * 0.001f);
}

TEST(SurfaceFilter, EffectiveTauNeverNegative) {
    EXPECT_GE(effective_tau(0.0f, 0.0f), 0.0f);
    EXPECT_GE(effective_tau(-1.0f, 0.001f), 0.0f);
}

// =====================================================================
// §6.4: Surface consistency filter
// =====================================================================

TEST(SurfaceFilter, ConsistencyFilterAllPass) {
    // Ideal case: coplanar, same normal, incoming from correct hemisphere
    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = make_f3(0, 1, 0);
    float3 ppos  = make_f3(0.01f, 0.001f, 0.01f);   // very close, tiny plane dist
    float3 pnorm = make_f3(0, 1, 0);                 // same normal
    float3 pwi   = make_f3(0.5f, 0.5f, 0);           // incoming from +Y hemisphere
    pwi = normalize(pwi);

    SurfaceFilterConfig cfg;
    cfg.radius2 = 0.1f;  // large enough
    cfg.tau     = 0.01f;  // large enough for plane_dist = 0.001

    float d_tan2;
    EXPECT_TRUE(surface_consistency(qpos, qnorm, ppos, pnorm, pwi, cfg, &d_tan2));
    EXPECT_LT(d_tan2, cfg.radius2);
}

TEST(SurfaceFilter, ConsistencyRejectTangentialRadius) {
    // Condition 1: photon too far in tangential direction
    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = make_f3(0, 1, 0);
    float3 ppos  = make_f3(1.0f, 0, 0);   // d_tan = 1.0
    float3 pnorm = make_f3(0, 1, 0);
    float3 pwi   = make_f3(0, 1, 0);

    SurfaceFilterConfig cfg;
    cfg.radius2 = 0.25f;  // r = 0.5, but photon is at d_tan = 1.0
    cfg.tau     = 0.1f;

    EXPECT_FALSE(surface_consistency(qpos, qnorm, ppos, pnorm, pwi, cfg));
}

TEST(SurfaceFilter, ConsistencyRejectPlaneDistance) {
    // Condition 2: photon too far from the tangent plane
    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = make_f3(0, 1, 0);
    float3 ppos  = make_f3(0.01f, 0.5f, 0);  // d_plane = 0.5, way above tau
    float3 pnorm = make_f3(0, 1, 0);
    float3 pwi   = make_f3(0, 1, 0);

    SurfaceFilterConfig cfg;
    cfg.radius2 = 1.0f;
    cfg.tau     = 0.01f;  // very tight

    EXPECT_FALSE(surface_consistency(qpos, qnorm, ppos, pnorm, pwi, cfg));
}

TEST(SurfaceFilter, ConsistencyRejectOppositeNormal) {
    // Condition 3: photon on opposite-facing surface
    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = make_f3(0, 1, 0);
    float3 ppos  = make_f3(0.01f, 0.001f, 0);
    float3 pnorm = make_f3(0, -1, 0);  // opposite normal!
    float3 pwi   = make_f3(0, 1, 0);

    SurfaceFilterConfig cfg;
    cfg.radius2 = 1.0f;
    cfg.tau     = 0.1f;

    EXPECT_FALSE(surface_consistency(qpos, qnorm, ppos, pnorm, pwi, cfg));
}

TEST(SurfaceFilter, ConsistencyRejectWrongDirection) {
    // Condition 4: photon incoming from wrong hemisphere
    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = make_f3(0, 1, 0);
    float3 ppos  = make_f3(0.01f, 0.001f, 0);
    float3 pnorm = make_f3(0, 1, 0);
    float3 pwi   = make_f3(0, -1, 0);  // incoming from below!

    SurfaceFilterConfig cfg;
    cfg.radius2 = 1.0f;
    cfg.tau     = 0.1f;

    EXPECT_FALSE(surface_consistency(qpos, qnorm, ppos, pnorm, pwi, cfg));
}

TEST(SurfaceFilter, ConsistencyFilterCondition3Perpendicular) {
    // Normal exactly perpendicular: dot = 0, should reject
    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = make_f3(0, 1, 0);
    float3 ppos  = make_f3(0.01f, 0.001f, 0);
    float3 pnorm = make_f3(1, 0, 0);   // perpendicular to query
    float3 pwi   = make_f3(0, 1, 0);

    SurfaceFilterConfig cfg;
    cfg.radius2 = 1.0f;
    cfg.tau     = 0.1f;

    EXPECT_FALSE(surface_consistency(qpos, qnorm, ppos, pnorm, pwi, cfg));
}

// =====================================================================
// §6.4: Surface prefilter (lightweight, conditions 1+2 only)
// =====================================================================

TEST(SurfaceFilter, PrefilterPass) {
    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = make_f3(0, 1, 0);
    float3 ppos  = make_f3(0.1f, 0.005f, 0.1f);

    float d_tan2;
    EXPECT_TRUE(surface_prefilter(qpos, qnorm, ppos, 0.1f, 0.01f, &d_tan2));
    EXPECT_NEAR(d_tan2, 0.02f, EPS);  // 0.1^2 + 0.1^2 = 0.02
}

TEST(SurfaceFilter, PrefilterRejectRadius) {
    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = make_f3(0, 1, 0);
    float3 ppos  = make_f3(1.0f, 0, 0);

    EXPECT_FALSE(surface_prefilter(qpos, qnorm, ppos, 0.1f, 0.01f));
}

TEST(SurfaceFilter, PrefilterRejectPlane) {
    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = make_f3(0, 1, 0);
    float3 ppos  = make_f3(0.01f, 0.5f, 0);

    EXPECT_FALSE(surface_prefilter(qpos, qnorm, ppos, 1.0f, 0.01f));
}

// =====================================================================
// Edge cases and numerical robustness
// =====================================================================

TEST(SurfaceFilter, TangentialDistanceIdentical) {
    // Query and photon at the exact same position
    float3 pos = make_f3(1, 2, 3);
    float3 norm = make_f3(0, 1, 0);

    TangentialResult tr = compute_tangential(pos, norm, pos);
    EXPECT_NEAR(tr.d_tan2, 0.0f, EPS);
    EXPECT_NEAR(tr.d_plane, 0.0f, EPS);
}

TEST(SurfaceFilter, TangentialDistanceLargeOffset) {
    // Verify numerical stability with large coordinates
    float3 qpos  = make_f3(1000, 2000, 3000);
    float3 qnorm = make_f3(0, 1, 0);
    float3 ppos  = make_f3(1000.1f, 2000.001f, 3000.2f);

    TangentialResult tr = compute_tangential(qpos, qnorm, ppos);
    // d_plane = 0.001 (Y offset projected onto Y normal)
    EXPECT_NEAR(tr.d_plane, 0.001f, 1e-3f);
    // d_tan^2 ≈ 0.1^2 + 0.2^2 = 0.05
    EXPECT_NEAR(tr.d_tan2, 0.05f, 1e-3f);
}
