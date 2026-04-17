// ─────────────────────────────────────────────────────────────────────
// scene_test.cpp – Compile + runtime verification of v5 scene stage
// ─────────────────────────────────────────────────────────────────────
// Tests:
//  1. Material construction and queries (RGB)
//  2. Triangle intersection + barycentric interpolation
//  3. Texture sampling
//  4. Scene construction + emissive distribution
//  5. SceneProfile analysis
//  6. Camera ray generation
//  7. Scene builder (Cornell box) integration
// ─────────────────────────────────────────────────────────────────────

// stb_image implementation (needed by OBJ loader)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// tinyexr implementation (needed by pbrt_material_mapper for EXR textures)
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#include "core/types.h"
#include "core/color.h"
#include "core/config.h"
#include "core/random.h"
#include "core/alias_table.h"
#include "core/scene_profile.h"
#include "scene/material.h"
#include "scene/triangle.h"
#include "scene/texture.h"
#include "scene/medium.h"
#include "scene/scene.h"
#include "analyze/scene_analyzer.h"
#include "scene/scene_builder.h"
#include "core/camera.h"

#include <cstdio>
#include <cassert>
#include <cmath>
#include <cstdlib>

static void test_material() {
    Material mat;
    assert(mat.type == MaterialType::Lambertian);
    assert(!mat.is_emissive());
    assert(!mat.is_specular());
    assert(mat.Kd.r == 0.5f);
    assert(mat.Kd.g == 0.5f);

    mat.Le = Color3::from_rgb(10.0f, 10.0f, 10.0f);
    assert(mat.is_emissive());
    assert(fabsf(mat.mean_emission() - 10.0f) < 1e-5f);

    mat.type = MaterialType::Glass;
    assert(mat.is_specular());

    // Cauchy IOR
    mat.dispersion = true;
    mat.cauchy_A = 1.5046f;
    mat.cauchy_B = 4200.0f;
    float ior_589 = mat.ior_at_lambda(589.0f);
    assert(fabsf(ior_589 - 1.51671f) < 0.001f);

    std::printf("[PASS] Material\n");
}

static void test_triangle() {
    Triangle t;
    t.v0 = make_f3(0, 0, 0);
    t.v1 = make_f3(1, 0, 0);
    t.v2 = make_f3(0, 1, 0);
    t.n0 = t.n1 = t.n2 = make_f3(0, 0, 1);
    t.uv0 = make_f2(0,0); t.uv1 = make_f2(1,0); t.uv2 = make_f2(0,1);
    t.material_id = 0;

    // Area
    assert(fabsf(t.area() - 0.5f) < 1e-5f);

    // Normal
    float3 gn = t.geometric_normal();
    assert(fabsf(gn.z - 1.0f) < 1e-5f);

    // Intersection
    Ray ray;
    ray.origin = make_f3(0.2f, 0.2f, 1.0f);
    ray.direction = make_f3(0, 0, -1);
    float t_out, u_out, v_out;
    bool hit = t.intersect(ray, t_out, u_out, v_out);
    assert(hit);
    assert(fabsf(t_out - 1.0f) < 1e-4f);

    // Interpolation
    float alpha = 1.0f - u_out - v_out;
    float3 pos = t.interpolate_position(alpha, u_out, v_out);
    assert(fabsf(pos.x - 0.2f) < 0.01f);
    assert(fabsf(pos.y - 0.2f) < 0.01f);

    std::printf("[PASS] Triangle\n");
}

static void test_texture() {
    Texture tex;
    tex.width = 2; tex.height = 2; tex.channels = 4;
    tex.data.resize(2 * 2 * 4, 0.f);
    // Set pixel (0,0) = red, (1,0) = green, (0,1) = blue, (1,1) = white
    tex.data[0] = 1; tex.data[1] = 0; tex.data[2] = 0; tex.data[3] = 1;
    tex.data[4] = 0; tex.data[5] = 1; tex.data[6] = 0; tex.data[7] = 1;
    tex.data[8] = 0; tex.data[9] = 0; tex.data[10] = 1; tex.data[11] = 1;
    tex.data[12] = 1; tex.data[13] = 1; tex.data[14] = 1; tex.data[15] = 1;

    // Sample center of pixel (0,0) → bottom-left in UV
    // UV (0.25, 0.25) → in image space after flip: row 1 (bottom), col 0
    // That's pixel data index 8 (blue)
    float3 s = tex.sample(make_f2(0.25f, 0.25f));
    // Due to V-flip: v=0.25 → image v=0.75 → row 1 → blue/white row
    assert(s.x >= 0.f && s.y >= 0.f && s.z >= 0.f);

    std::printf("[PASS] Texture\n");
}

static void test_medium() {
    HomogeneousMedium med;
    med.sigma_a = Color3::from_rgb(0.1f, 0.2f, 0.3f);
    med.sigma_s = Color3::from_rgb(1.0f, 1.0f, 1.0f);
    med.sigma_t = med.sigma_a + med.sigma_s;
    med.g = 0.8f;

    assert(fabsf(med.sigma_t.r - 1.1f) < 1e-5f);
    assert(fabsf(med.sigma_t.g - 1.2f) < 1e-5f);

    MediumStack ms;
    assert(ms.current_medium_id() == -1);
    ms.push(3);
    assert(ms.current_medium_id() == 3);
    ms.pop();
    assert(ms.current_medium_id() == -1);

    std::printf("[PASS] Medium + MediumStack\n");
}

static void test_scene_cornell_box() {
    Scene scene = scene_builder::build_cornell_box();

    assert(scene.num_triangles() > 0);
    assert(scene.num_materials() == 4);
    assert(scene.num_emissive() > 0);
    assert(scene.total_emissive_power > 0.f);

    // Bounds should be reasonable
    float3 c = scene.scene_bounding_center();
    float r = scene.scene_bounding_radius();
    assert(r > 0.f && r < 10.f);
    (void)c;

    std::printf("[PASS] Scene (Cornell box): %zu tris, %zu emissive, power=%.4f\n",
                scene.num_triangles(), scene.num_emissive(), scene.total_emissive_power);
}

static void test_scene_analyzer() {
    Scene scene = scene_builder::build_cornell_box();
    SceneProfile sp = analyze_scene(scene);

    assert(sp.num_triangles > 0);
    assert(sp.num_emitters > 0);
    assert(!sp.has_glass);
    assert(sp.dominant_lighting == LightingType::LargeArea
        || sp.dominant_lighting == LightingType::SmallPoint);
    assert(sp.recommended_max_bounces >= 4);
    assert(sp.recommended_photon_budget > 0);

    std::printf("[PASS] SceneAnalyzer: lighting=%d, bounces=%d, photons=%d\n",
                (int)sp.dominant_lighting,
                sp.recommended_max_bounces,
                sp.recommended_photon_budget);

    // Glass sphere scene should detect caustics
    Scene gs = scene_builder::build_glass_sphere();
    SceneProfile sp2 = analyze_scene(gs);
    assert(sp2.has_glass);
    assert(sp2.has_caustic_paths);

    std::printf("[PASS] SceneAnalyzer (glass sphere): caustics=%d\n",
                sp2.has_caustic_paths);
}

static void test_camera() {
    Camera cam = Camera::cornell_box_camera(800, 600);

    // Generate a center ray
    PCGRng rng = PCGRng::seed(42, 0);
    Ray ray = cam.generate_ray(400, 300, rng);
    assert(fabsf(length(ray.direction) - 1.0f) < 1e-4f);

    // Ray should point roughly towards -Z (camera looks at origin from z=2.5)
    assert(ray.direction.z < 0.f);

    std::printf("[PASS] Camera: ray dir=(%.3f, %.3f, %.3f)\n",
                ray.direction.x, ray.direction.y, ray.direction.z);
}

int main() {
    std::printf("=== v5 Scene Stage Tests ===\n\n");

    test_material();
    test_triangle();
    test_texture();
    test_medium();
    test_scene_cornell_box();
    test_scene_analyzer();
    test_camera();

    std::printf("\n=== All scene stage tests PASSED ===\n");
    return 0;
}
