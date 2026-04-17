#pragma once
// ─────────────────────────────────────────────────────────────────────
// scene_builder.h – Programmatic scene construction (for tests)
//
// Provides helper functions to build simple scenes without loading
// files.  Used by the test harness to create reproducible test scenes.
// ─────────────────────────────────────────────────────────────────────
#include "scene/scene.h"
#include "analyze/scene_analyzer.h"
#include "core/camera.h"

namespace scene_builder {

// ── Add a quad (two triangles) to the scene ─────────────────────────
inline void add_quad(Scene& scene, uint32_t mat_id,
                     float3 a, float3 b, float3 c, float3 d,
                     float3 normal) {
    Triangle t0;
    t0.v0 = a; t0.v1 = b; t0.v2 = c;
    t0.n0 = t0.n1 = t0.n2 = normal;
    t0.uv0 = make_f2(0,0); t0.uv1 = make_f2(1,0); t0.uv2 = make_f2(1,1);
    t0.material_id = mat_id;

    Triangle t1;
    t1.v0 = a; t1.v1 = c; t1.v2 = d;
    t1.n0 = t1.n1 = t1.n2 = normal;
    t1.uv0 = make_f2(0,0); t1.uv1 = make_f2(1,1); t1.uv2 = make_f2(0,1);
    t1.material_id = mat_id;

    scene.triangles.push_back(t0);
    scene.triangles.push_back(t1);
}

// ── Build a simple Cornell-box-like scene ───────────────────────────
// 5 walls (no top) + 1 emissive quad on ceiling.
// Returns a fully finalized scene (bounds computed, emissive distribution built).
inline Scene build_cornell_box() {
    Scene scene;

    // Materials
    Material white;  white.name = "white";  white.Kd = Color3::constant(0.73f);
    Material red;    red.name = "red";      red.Kd = Color3::from_rgb(0.65f, 0.05f, 0.05f);
    Material green;  green.name = "green";  green.Kd = Color3::from_rgb(0.12f, 0.45f, 0.15f);
    Material light;  light.name = "light";  light.type = MaterialType::Emissive;
    light.Le = Color3::constant(15.0f);

    scene.materials.push_back(white);  // 0
    scene.materials.push_back(red);    // 1
    scene.materials.push_back(green);  // 2
    scene.materials.push_back(light);  // 3

    // Half-extent
    float h = 0.5f;

    // Floor (white)
    add_quad(scene, 0,
             make_f3(-h, -h, -h), make_f3(h, -h, -h),
             make_f3(h, -h, h),   make_f3(-h, -h, h),
             make_f3(0, 1, 0));

    // Ceiling (white)
    add_quad(scene, 0,
             make_f3(-h, h, h),  make_f3(h, h, h),
             make_f3(h, h, -h),  make_f3(-h, h, -h),
             make_f3(0, -1, 0));

    // Back wall (white)
    add_quad(scene, 0,
             make_f3(-h, -h, -h), make_f3(h, -h, -h),
             make_f3(h, h, -h),   make_f3(-h, h, -h),
             make_f3(0, 0, 1));

    // Left wall (red)
    add_quad(scene, 1,
             make_f3(-h, -h, h),  make_f3(-h, -h, -h),
             make_f3(-h, h, -h),  make_f3(-h, h, h),
             make_f3(1, 0, 0));

    // Right wall (green)
    add_quad(scene, 2,
             make_f3(h, -h, -h), make_f3(h, -h, h),
             make_f3(h, h, h),   make_f3(h, h, -h),
             make_f3(-1, 0, 0));

    // Ceiling light (emissive)
    float lh = 0.13f;
    add_quad(scene, 3,
             make_f3(-lh, h - 0.001f, -lh), make_f3(lh, h - 0.001f, -lh),
             make_f3(lh, h - 0.001f, lh),   make_f3(-lh, h - 0.001f, lh),
             make_f3(0, -1, 0));

    scene.compute_bounds();
    scene.build_emissive_distribution();

    return scene;
}

// ── Build a scene with a glass sphere for caustic testing ───────────
inline Scene build_glass_sphere() {
    Scene scene = build_cornell_box();

    // Add a glass material
    Material glass;
    glass.name = "glass";
    glass.type = MaterialType::Glass;
    glass.ior  = 1.5f;
    glass.Kd   = Color3::zero();
    glass.Ks   = Color3::one();
    glass.Tf   = Color3::one();
    uint32_t glass_id = (uint32_t)scene.materials.size();
    scene.materials.push_back(glass);

    // Tessellate a simple icosahedron-ish sphere (rough approximation)
    // For test purposes, 8 triangles forming an octahedron
    float r = 0.15f;
    float3 c = make_f3(0.0f, -0.35f + r, 0.0f);
    float3 px = make_f3(r, 0, 0) + c;
    float3 nx = make_f3(-r, 0, 0) + c;
    float3 py = make_f3(0, r, 0) + c;
    float3 ny = make_f3(0, -r, 0) + c;
    float3 pz = make_f3(0, 0, r) + c;
    float3 nz = make_f3(0, 0, -r) + c;

    auto add_tri = [&](float3 a, float3 b, float3 cv) {
        Triangle t;
        t.v0 = a; t.v1 = b; t.v2 = cv;
        t.n0 = normalize(a - c);
        t.n1 = normalize(b - c);
        t.n2 = normalize(cv - c);
        t.uv0 = t.uv1 = t.uv2 = make_f2(0, 0);
        t.material_id = glass_id;
        scene.triangles.push_back(t);
    };

    // Top 4
    add_tri(py, pz, px);
    add_tri(py, px, nz);
    add_tri(py, nz, nx);
    add_tri(py, nx, pz);
    // Bottom 4
    add_tri(ny, px, pz);
    add_tri(ny, nz, px);
    add_tri(ny, nx, nz);
    add_tri(ny, pz, nx);

    scene.compute_bounds();
    scene.build_emissive_distribution();

    return scene;
}

} // namespace scene_builder
