// ---------------------------------------------------------------------
// scene_builder.h -- Scene augmentation: procedural lights & geometry
// ---------------------------------------------------------------------
//
// Extracted from main.cpp (§1.7) to keep scene construction logic in
// the scene/ module where it belongs.
// ---------------------------------------------------------------------
#pragma once

#include "scene/scene.h"
#include "core/config.h"
#include "core/spectrum.h"
#include <cmath>
#include <iostream>

// Generate inward-facing emissive triangles on a (hemi)sphere.
// theta_min = 0 → north pole.  theta_max = PI   → full sphere.
//                               theta_max = PI/2 → upper hemisphere.
inline void generate_sphere_lights(
    Scene& scene, uint32_t mat_id,
    float radius, float3 center,
    int n_slices, int n_stacks,
    float theta_min, float theta_max)
{
    auto sphere_pt = [&](int stack, int slice) -> float3 {
        float theta = theta_min + (theta_max - theta_min) * (float)stack / (float)n_stacks;
        float phi   = 2.0f * 3.14159265358979f * (float)slice / (float)n_slices;
        float st = sinf(theta);
        return make_f3(
            center.x + radius * st * cosf(phi),
            center.y + radius * cosf(theta),
            center.z + radius * st * sinf(phi));
    };

    float2 uv0 = make_f2(0, 0);
    int count = 0;

    for (int i = 0; i < n_stacks; ++i) {
        for (int j = 0; j < n_slices; ++j) {
            float3 p00 = sphere_pt(i,     j);
            float3 p10 = sphere_pt(i + 1, j);
            float3 p11 = sphere_pt(i + 1, j + 1);
            float3 p01 = sphere_pt(i,     j + 1);

            // Inward-facing normals (towards center)
            auto inward_n = [&](float3 p) { return normalize(center - p); };

            // Triangle 1: p00-p11-p10  (CW from outside = CCW from inside)
            {
                Triangle tri;
                tri.v0 = p00; tri.v1 = p11; tri.v2 = p10;
                tri.n0 = inward_n(p00); tri.n1 = inward_n(p11); tri.n2 = inward_n(p10);
                tri.uv0 = tri.uv1 = tri.uv2 = uv0;
                tri.material_id = mat_id;
                if (tri.area() > 1e-8f) { scene.triangles.push_back(tri); ++count; }
            }
            // Triangle 2: p00-p01-p11
            {
                Triangle tri;
                tri.v0 = p00; tri.v1 = p01; tri.v2 = p11;
                tri.n0 = inward_n(p00); tri.n1 = inward_n(p01); tri.n2 = inward_n(p11);
                tri.uv0 = tri.uv1 = tri.uv2 = uv0;
                tri.material_id = mat_id;
                if (tri.area() > 1e-8f) { scene.triangles.push_back(tri); ++count; }
            }
        }
    }
    std::cout << "[Scene]   Generated " << count << " emissive triangles "
              << "(r=" << radius << ", stacks=" << n_stacks
              << ", slices=" << n_slices << ")\n";
}

// Add appropriate light sources based on the scene's lighting mode.
// Called after scene load + normalize.  For MTL scenes the emissive
// surfaces are already in the triangle list.  For environment-lit
// scenes we generate tessellated (hemi)sphere geometry.
inline void add_scene_lights(Scene& scene, SceneLightMode mode) {
    if (mode == SceneLightMode::FromMTL && scene.num_emissive() > 0)
        return;  // MTL already has lights

    // ── HemisphereEnv (Sponza): sky dome above the scene ────────────
    if (mode == SceneLightMode::HemisphereEnv) {
        Material sky_mat;
        sky_mat.name = "__sky_dome__";
        sky_mat.type = MaterialType::Emissive;
        sky_mat.Le   = blackbody_spectrum(5800.f, 8e-9f);  // warm daylight
        uint32_t sky_id = (uint32_t)scene.materials.size();
        scene.materials.push_back(sky_mat);

        // Hemisphere: theta 0→PI/2, centered slightly above the scene
        float3 center = make_f3(0.0f, 0.0f, 0.0f);
        generate_sphere_lights(scene, sky_id,
            0.75f, center,       // radius just outside normalized [-0.5,0.5]
            24, 8,               // 24 slices × 8 stacks = 384 tris
            0.0f, 1.5707963f);   // theta 0→PI/2  (upper hemisphere)

        scene.build_bvh();
        scene.build_emissive_distribution();
        std::cout << "[Scene] Hemisphere sky dome: "
                  << scene.num_emissive() << " emissive triangles total\n";
        return;
    }

    // ── SphericalEnv (Mori Knob): full sphere surrounding the scene ──
    if (mode == SceneLightMode::SphericalEnv) {
        Material env_mat;
        env_mat.name = "__env_sphere__";
        env_mat.type = MaterialType::Emissive;
        env_mat.Le   = blackbody_spectrum(5500.f, 5e-9f);  // neutral daylight
        uint32_t env_id = (uint32_t)scene.materials.size();
        scene.materials.push_back(env_mat);

        // Full sphere: theta 0→PI
        float3 center = make_f3(0.0f, 0.0f, 0.0f);
        generate_sphere_lights(scene, env_id,
            0.80f, center,       // radius outside the model
            32, 16,              // 32 slices × 16 stacks = 1024 tris
            0.0f, 3.14159265f);  // theta 0→PI  (full sphere)

        scene.build_bvh();
        scene.build_emissive_distribution();
        std::cout << "[Scene] Full environment sphere: "
                  << scene.num_emissive() << " emissive triangles total\n";
        return;
    }

    // ── DirectionalToFloor / FromMTL fallback: ceiling area light ───
    Material light_mat;
    light_mat.name = "__area_light__";
    light_mat.type = MaterialType::Emissive;

    if (mode == SceneLightMode::DirectionalToFloor) {
        light_mat.Le = blackbody_spectrum(6500.f, 3e-8f);
    } else {
        light_mat.Le = blackbody_spectrum(6500.f, 1e-8f);
    }

    uint32_t light_mat_id = (uint32_t)scene.materials.size();
    scene.materials.push_back(light_mat);

    float3 v0 = make_f3(-0.15f,  0.499f, -0.15f);
    float3 v1 = make_f3( 0.15f,  0.499f, -0.15f);
    float3 v2 = make_f3( 0.15f,  0.499f,  0.15f);
    float3 v3 = make_f3(-0.15f,  0.499f,  0.15f);
    float3 n  = make_f3( 0.0f,  -1.0f,    0.0f);

    Triangle tri1;
    tri1.v0 = v0; tri1.v1 = v1; tri1.v2 = v2;
    tri1.n0 = tri1.n1 = tri1.n2 = n;
    tri1.uv0 = tri1.uv1 = tri1.uv2 = make_f2(0, 0);
    tri1.material_id = light_mat_id;

    Triangle tri2;
    tri2.v0 = v0; tri2.v1 = v2; tri2.v2 = v3;
    tri2.n0 = tri2.n1 = tri2.n2 = n;
    tri2.uv0 = tri2.uv1 = tri2.uv2 = make_f2(0, 0);
    tri2.material_id = light_mat_id;

    scene.triangles.push_back(tri1);
    scene.triangles.push_back(tri2);

    scene.build_bvh();
    scene.build_emissive_distribution();
}
