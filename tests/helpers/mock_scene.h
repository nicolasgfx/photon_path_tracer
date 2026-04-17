#pragma once
// ─────────────────────────────────────────────────────────────────────
// tests/helpers/mock_scene.h – Minimal test scene constructors
//
// Wraps reference_scenes.h for use in GTest unit + convergence tests.
// ─────────────────────────────────────────────────────────────────────
#include "testing/reference_scenes.h"
#include "scene/scene_builder.h"

namespace mock_scene {

// Re-export reference_scenes for test convenience
using reference_scenes::ReferenceScene;
using reference_scenes::TestCamera;
using reference_scenes::cornell_box;
using reference_scenes::glass_sphere;
using reference_scenes::furnace;

// Quick single-triangle scene for trivial tests
inline Scene single_triangle() {
    Scene s;
    Material white;
    white.Kd = Color3::constant(0.8f);
    s.materials.push_back(white);

    scene_builder::add_quad(s, 0,
        make_f3(-1,-1,0), make_f3(1,-1,0),
        make_f3(1,1,0),   make_f3(-1,1,0),
        make_f3(0,0,1));

    s.compute_bounds();
    return s;
}

} // namespace mock_scene
