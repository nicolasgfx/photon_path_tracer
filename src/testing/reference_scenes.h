#pragma once
// ─────────────────────────────────────────────────────────────────────
// testing/reference_scenes.h – Canonical test scenes with cameras (v5)
//
// Wraps scene_builder scenes with camera configurations and metadata.
// Each reference scene is a complete test case setup: scene + camera +
// expected properties (convergence rate, energy level, etc.).
// ─────────────────────────────────────────────────────────────────────
#include "scene/scene_builder.h"
#include "core/types.h"

namespace reference_scenes {

// ── Test camera (simple pinhole) ────────────────────────────────────

struct TestCamera {
    float3 pos;
    float3 u, v, w;  // right, up, forward (ONB)
};

// ── Reference scene descriptor ──────────────────────────────────────

struct ReferenceScene {
    std::string    name;
    Scene          scene;
    TestCamera     camera;

    // Expected properties for validation
    float expected_mean_luminance = -1.f;  // -1 = don't check
    float expected_convergence_rate = -1.f; // log-log slope
    bool  has_caustics = false;
    bool  is_furnace   = false;             // energy conservation test
};

// ── Cornell Box ─────────────────────────────────────────────────────

inline ReferenceScene cornell_box() {
    ReferenceScene ref;
    ref.name  = "cornell_box";
    ref.scene = scene_builder::build_cornell_box();

    ref.camera.pos = make_f3(0.f, 0.f, 1.2f);
    ref.camera.u   = make_f3(1.f, 0.f, 0.f);
    ref.camera.v   = make_f3(0.f, 1.f, 0.f);
    ref.camera.w   = make_f3(0.f, 0.f, -1.f);

    ref.expected_convergence_rate = -0.9f;  // ~1/N for MC
    return ref;
}

// ── Glass sphere (caustics) ─────────────────────────────────────────

inline ReferenceScene glass_sphere() {
    ReferenceScene ref;
    ref.name  = "glass_sphere";
    ref.scene = scene_builder::build_glass_sphere();

    ref.camera.pos = make_f3(0.f, 0.f, 1.2f);
    ref.camera.u   = make_f3(1.f, 0.f, 0.f);
    ref.camera.v   = make_f3(0.f, 1.f, 0.f);
    ref.camera.w   = make_f3(0.f, 0.f, -1.f);

    ref.has_caustics = true;
    ref.expected_convergence_rate = -0.5f;  // slower due to caustics
    return ref;
}

// ── Furnace test scene (all-white room, emitter everywhere) ─────────
// Everything is white lambertian with albedo 1.0.  The scene should
// converge to uniform radiance.  Used for energy conservation testing.

inline ReferenceScene furnace() {
    ReferenceScene ref;
    ref.name = "furnace";

    Scene& s = ref.scene;

    // Single material: perfect white lambertian
    Material white;
    white.name = "white_furnace";
    white.Kd   = Color3::one();  // albedo = 1.0
    s.materials.push_back(white);

    // Emissive ceiling material
    Material emit;
    emit.name = "emit_furnace";
    emit.type = MaterialType::Emissive;
    emit.Le   = Color3::one();   // radiance = 1.0
    s.materials.push_back(emit);

    float h = 0.5f;

    // 5 white walls
    scene_builder::add_quad(s, 0,
        make_f3(-h,-h,-h), make_f3(h,-h,-h),
        make_f3(h,-h,h),   make_f3(-h,-h,h), make_f3(0,1,0));
    scene_builder::add_quad(s, 0,
        make_f3(-h,h,h),  make_f3(h,h,h),
        make_f3(h,h,-h),  make_f3(-h,h,-h), make_f3(0,-1,0));
    scene_builder::add_quad(s, 0,
        make_f3(-h,-h,-h), make_f3(h,-h,-h),
        make_f3(h,h,-h),   make_f3(-h,h,-h), make_f3(0,0,1));
    scene_builder::add_quad(s, 0,
        make_f3(-h,-h,h),  make_f3(-h,-h,-h),
        make_f3(-h,h,-h),  make_f3(-h,h,h), make_f3(1,0,0));
    scene_builder::add_quad(s, 0,
        make_f3(h,-h,-h), make_f3(h,-h,h),
        make_f3(h,h,h),   make_f3(h,h,-h), make_f3(-1,0,0));

    // Emissive ceiling (full coverage)
    scene_builder::add_quad(s, 1,
        make_f3(-h, h-0.001f, -h), make_f3(h, h-0.001f, -h),
        make_f3(h, h-0.001f, h),   make_f3(-h, h-0.001f, h),
        make_f3(0,-1,0));

    s.compute_bounds();
    s.build_emissive_distribution();

    ref.camera.pos = make_f3(0.f, 0.f, 0.f);
    ref.camera.u   = make_f3(1.f, 0.f, 0.f);
    ref.camera.v   = make_f3(0.f, 1.f, 0.f);
    ref.camera.w   = make_f3(0.f, 0.f, -1.f);

    ref.is_furnace = true;
    ref.expected_mean_luminance = 1.0f;
    return ref;
}

} // namespace reference_scenes
