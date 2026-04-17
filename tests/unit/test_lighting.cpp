// ─────────────────────────────────────────────────────────────────────
// tests/unit/test_lighting.cpp – Direct lighting / NEE unit tests
//
// Stage 5: NEE shared types, MIS weight computation, geometry helpers.
// CPU-only aspects of the lighting stage.
// ─────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include "integrator/nee_shared.h"
#include "lighting/light_tree.h"
#include "core/types.h"
#include "core/config.h"
#include "core/alias_table.h"
#include "scene/scene.h"
#include "scene/scene_builder.h"
#include <cmath>

// ── MIS power heuristic (3-way) ─────────────────────────────────────

TEST(Lighting, MIS3WayBalanced) {
    // Equal pdfs → each gets 1/3 weight
    float w = nee_mis_weight_3(1.0f, 1.0f, 1.0f);
    EXPECT_NEAR(w, 1.0f / 3.0f, 1e-5f);
}

TEST(Lighting, MIS3WayDominated) {
    // pdf_a >> others → weight ≈ 1.0
    float w = nee_mis_weight_3(100.0f, 0.01f, 0.01f);
    EXPECT_GT(w, 0.99f);
}

TEST(Lighting, MIS3WayZeroA) {
    float w = nee_mis_weight_3(0.0f, 1.0f, 1.0f);
    EXPECT_NEAR(w, 0.0f, 1e-5f);
}

// ── PDF area → solid angle conversion ───────────────────────────────

TEST(Lighting, PDFAreaToSolidAngle) {
    // pdf_tri=1, pdf_pos=2, d²=4, cos_emitter=0.5 → result=16
    float sa = nee_pdf_area_to_solid_angle(1.0f, 2.0f, 4.0f, 0.5f);
    EXPECT_NEAR(sa, 16.0f, 1e-4f);
}

TEST(Lighting, PDFAreaToSolidAngleBackfacing) {
    // Backfacing → 0
    float sa = nee_pdf_area_to_solid_angle(1.0f, 2.0f, 4.0f, -0.1f);
    EXPECT_FLOAT_EQ(sa, 0.0f);
}

// ── NEE geometry computation ────────────────────────────────────────

TEST(Lighting, GeometryValid) {
    float3 hit_pos    = make_f3(0, 0, 0);
    float3 hit_normal = make_f3(0, 1, 0);
    float3 light_pos  = make_f3(0, 2, 0);
    float3 light_norm = make_f3(0, -1, 0);

    NEEGeometry g = nee_compute_geometry(hit_pos, hit_normal, light_pos, light_norm);
    EXPECT_TRUE(g.valid);
    EXPECT_NEAR(g.distance, 2.0f, 1e-4f);
    EXPECT_NEAR(g.cos_receiver, 1.0f, 1e-4f);
    EXPECT_NEAR(g.cos_emitter, 1.0f, 1e-4f);
}

TEST(Lighting, GeometryBackfacing) {
    float3 hit_pos    = make_f3(0, 0, 0);
    float3 hit_normal = make_f3(0, -1, 0);  // pointing away from light
    float3 light_pos  = make_f3(0, 2, 0);
    float3 light_norm = make_f3(0, -1, 0);

    NEEGeometry g = nee_compute_geometry(hit_pos, hit_normal, light_pos, light_norm);
    EXPECT_FALSE(g.valid);  // receiver cos < 0
}

// ── Shadow ray helpers ──────────────────────────────────────────────

TEST(Lighting, ShadowRayOriginOffset) {
    float3 pos = make_f3(0, 0, 0);
    float3 n   = make_f3(0, 1, 0);
    float3 origin = nee_shadow_ray_origin(pos, n);
    EXPECT_GT(origin.y, 0.0f);
    EXPECT_NEAR(origin.y, NEE_RAY_EPSILON, 1e-6f);
}

TEST(Lighting, ShadowRayTMax) {
    float tmax = nee_shadow_ray_tmax(5.0f);
    EXPECT_LT(tmax, 5.0f);
    EXPECT_GT(tmax, 4.99f);
}

// ── Scene emissive data ─────────────────────────────────────────────

TEST(Lighting, CornellBoxEmissiveData) {
    Scene s = scene_builder::build_cornell_box();
    EXPECT_GE(s.num_emissive(), 1u);
    EXPECT_GT(s.total_emissive_power, 0.0f);
}

// ── Light tree CPU builder ──────────────────────────────────────────

TEST(LightTree, BuildCornellBox) {
    Scene s = scene_builder::build_cornell_box();
    LightTree tree;
    tree.build(s);
    EXPECT_GT((int)tree.nodes.size(), 0);
    EXPECT_EQ((int)tree.tri_order.size(), (int)s.num_emissive());
}

TEST(LightTree, RootContainsAllFlux) {
    Scene s = scene_builder::build_cornell_box();
    LightTree tree;
    tree.build(s);
    if (tree.nodes.empty()) GTEST_SKIP();
    const auto& root = tree.nodes[tree.root];
    // root flux should match total emissive power (within tolerance)
    EXPECT_NEAR(root.flux, s.total_emissive_power, s.total_emissive_power * 0.01f);
}

TEST(LightTree, LeavesContainAllTriangles) {
    Scene s = scene_builder::build_cornell_box();
    LightTree tree;
    tree.build(s);
    if (tree.nodes.empty()) GTEST_SKIP();
    // count total triangles in all leaves
    int leaf_tris = 0;
    for (const auto& node : tree.nodes) {
        if (node.tri_count > 0) leaf_tris += node.tri_count;
    }
    EXPECT_EQ(leaf_tris, (int)s.num_emissive());
}

TEST(LightTree, TriOrderContainsAllEmissives) {
    Scene s = scene_builder::build_cornell_box();
    LightTree tree;
    tree.build(s);
    if (tree.nodes.empty()) GTEST_SKIP();
    // tri_order should be a permutation of emissive_tri_indices
    std::vector<uint32_t> sorted_order = tree.tri_order;
    std::vector<uint32_t> sorted_orig = s.emissive_tri_indices;
    std::sort(sorted_order.begin(), sorted_order.end());
    std::sort(sorted_orig.begin(), sorted_orig.end());
    ASSERT_EQ(sorted_order.size(), sorted_orig.size());
    for (size_t i = 0; i < sorted_order.size(); ++i)
        EXPECT_EQ(sorted_order[i], sorted_orig[i]);
}

TEST(LightTree, RootBBoxContainsAllEmissives) {
    Scene s = scene_builder::build_cornell_box();
    LightTree tree;
    tree.build(s);
    if (tree.nodes.empty()) GTEST_SKIP();
    const auto& root = tree.nodes[tree.root];
    // every emissive vertex should be inside root AABB
    for (uint32_t ti : s.emissive_tri_indices) {
        const auto& tri = s.triangles[ti];
        EXPECT_GE(tri.v0.x, root.bbox_lo.x - 1e-4f);
        EXPECT_LE(tri.v0.x, root.bbox_hi.x + 1e-4f);
        EXPECT_GE(tri.v0.y, root.bbox_lo.y - 1e-4f);
        EXPECT_LE(tri.v0.y, root.bbox_hi.y + 1e-4f);
        EXPECT_GE(tri.v0.z, root.bbox_lo.z - 1e-4f);
        EXPECT_LE(tri.v0.z, root.bbox_hi.z + 1e-4f);
    }
}

TEST(LightTree, SingleEmitterSceneWorks) {
    // build a minimal scene with one emissive triangle
    Scene s;
    Triangle t;
    t.v0 = make_f3(0, 0, 0);
    t.v1 = make_f3(1, 0, 0);
    t.v2 = make_f3(0, 1, 0);
    t.n0 = t.n1 = t.n2 = make_f3(0, 0, 1);
    t.uv0 = t.uv1 = t.uv2 = make_f2(0, 0);
    t.material_id = 0;
    s.triangles.push_back(t);

    Material m;
    m.name = "emitter";
    m.Le = Color3{1.f, 1.f, 1.f};
    m.type = MaterialType::Emissive;
    s.materials.push_back(m);

    s.build_emissive_distribution();
    ASSERT_EQ(s.num_emissive(), 1u);

    LightTree tree;
    tree.build(s);
    ASSERT_EQ((int)tree.nodes.size(), 1);
    EXPECT_EQ(tree.nodes[0].tri_count, 1);
    EXPECT_GT(tree.nodes[0].flux, 0.f);
}

TEST(LightTree, InteriorNodesHaveZeroTriCount) {
    Scene s = scene_builder::build_cornell_box();
    LightTree tree;
    tree.build(s);
    if (tree.nodes.empty()) GTEST_SKIP();
    for (const auto& node : tree.nodes) {
        if (node.tri_count == 0) {
            // interior — children should be valid indices
            EXPECT_GE(node.child_left, 0);
            EXPECT_LT(node.child_left, (int)tree.nodes.size());
            EXPECT_GE(node.child_right, 0);
            EXPECT_LT(node.child_right, (int)tree.nodes.size());
        }
    }
}
