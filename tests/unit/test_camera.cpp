// ─────────────────────────────────────────────────────────────────────
// tests/unit/test_camera.cpp – Camera ray generation tests
//
// Stage 3: Pinhole camera, DOF, sub-pixel jitter.
// ─────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include "core/camera.h"
#include "core/types.h"
#include "core/random.h"
#include <cmath>

TEST(Camera, DefaultConstruction) {
    Camera cam;
    cam.position = make_f3(0, 0, 1.2f);
    cam.look_at  = make_f3(0, 0, 0);
    cam.up       = make_f3(0, 1, 0);
    cam.fov_deg  = 60.0f;
    cam.update();

    // Camera should be looking toward -Z from +Z
    EXPECT_NEAR(cam.position.z, 1.2f, 1e-5f);
}

TEST(Camera, CenterRayDirection) {
    Camera cam;
    cam.position = make_f3(0, 0, 1);
    cam.look_at  = make_f3(0, 0, 0);
    cam.up       = make_f3(0, 1, 0);
    cam.fov_deg  = 90.0f;
    cam.width    = 100;
    cam.height   = 100;
    cam.update();

    // Center pixel → should point toward (0,0,-1)
    PCGRng rng = PCGRng::seed(42);
    Ray ray = cam.generate_ray(50, 50, rng);
    float3 d = normalize(ray.direction);
    EXPECT_NEAR(d.x, 0.0f, 0.1f);
    EXPECT_NEAR(d.y, 0.0f, 0.1f);
    EXPECT_LT(d.z, 0.0f);  // pointing -Z
}

TEST(Camera, RayOriginMatchesPosition) {
    Camera cam;
    cam.position = make_f3(1, 2, 3);
    cam.look_at  = make_f3(0, 0, 0);
    cam.up       = make_f3(0, 1, 0);
    cam.fov_deg  = 60.0f;
    cam.width    = 100;
    cam.height   = 100;
    cam.update();

    PCGRng rng = PCGRng::seed(42);
    Ray ray = cam.generate_ray(50, 50, rng);
    // Without DOF, origin equals camera position
    EXPECT_NEAR(ray.origin.x, 1.0f, 1e-3f);
    EXPECT_NEAR(ray.origin.y, 2.0f, 1e-3f);
    EXPECT_NEAR(ray.origin.z, 3.0f, 1e-3f);
}

TEST(Camera, CornerRaysDiverge) {
    Camera cam;
    cam.position = make_f3(0, 0, 0);
    cam.look_at  = make_f3(0, 0, -1);
    cam.up       = make_f3(0, 1, 0);
    cam.fov_deg  = 90.0f;
    cam.width    = 100;
    cam.height   = 100;
    cam.update();

    PCGRng rng1 = PCGRng::seed(1);
    PCGRng rng2 = PCGRng::seed(2);
    Ray top_left = cam.generate_ray(0, 0, rng1);
    Ray bot_right = cam.generate_ray(99, 99, rng2);

    float3 d1 = normalize(top_left.direction);
    float3 d2 = normalize(bot_right.direction);

    // Corners should point in different directions
    float cos_angle = dot(d1, d2);
    EXPECT_LT(cos_angle, 0.9f);
}

TEST(Camera, NarrowFOVConverges) {
    Camera cam;
    cam.position = make_f3(0, 0, 0);
    cam.look_at  = make_f3(0, 0, -1);
    cam.up       = make_f3(0, 1, 0);
    cam.width    = 100;
    cam.height   = 100;
    cam.update();

    PCGRng rng1 = PCGRng::seed(10);
    PCGRng rng2 = PCGRng::seed(20);

    cam.fov_deg = 10.0f;
    cam.update();
    Ray narrow_corner = cam.generate_ray(0, 0, rng1);

    cam.fov_deg = 120.0f;
    cam.update();
    Ray wide_corner = cam.generate_ray(0, 0, rng2);

    float3 center = make_f3(0, 0, -1);
    float dot_narrow = dot(normalize(narrow_corner.direction), center);
    float dot_wide   = dot(normalize(wide_corner.direction), center);

    // Narrow FOV corner should be closer to center than wide FOV corner
    EXPECT_GT(dot_narrow, dot_wide);
}
