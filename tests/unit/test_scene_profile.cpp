// ─────────────────────────────────────────────────────────────────────
// tests/unit/test_scene_profile.cpp – SceneProfile classification tests
//
// Stage 1: Scene analysis and profile-driven parameter selection.
// ─────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include "core/types.h"
#include "core/config.h"
#include "core/scene_profile.h"
#include "scene/scene.h"
#include "scene/scene_builder.h"
#include "analyze/scene_analyzer.h"

TEST(SceneProfile, CornellBoxAnalysis) {
    Scene s = scene_builder::build_cornell_box();
    SceneProfile prof = analyze_scene(s);

    EXPECT_EQ(prof.dominant_lighting, LightingType::LargeArea);
    EXPECT_FALSE(prof.has_caustic_paths);
    EXPECT_GE(prof.recommended_max_bounces, 4);
}

TEST(SceneProfile, GlassSphereDetectsCaustics) {
    Scene s = scene_builder::build_glass_sphere();
    SceneProfile prof = analyze_scene(s);

    EXPECT_TRUE(prof.has_caustic_paths);
    EXPECT_GT(prof.recommended_photon_budget, 0);
}

TEST(SceneProfile, ProfileDefaults) {
    SceneProfile prof{};

    EXPECT_EQ(prof.recommended_max_bounces, 8);
    EXPECT_FALSE(prof.has_caustic_paths);
}

TEST(SceneProfile, RecommendedBouncesByLighting) {
    Scene cornell = scene_builder::build_cornell_box();
    SceneProfile p1 = analyze_scene(cornell);

    Scene glass = scene_builder::build_glass_sphere();
    SceneProfile p2 = analyze_scene(glass);

    // Glass scenes may need more bounces for caustics
    EXPECT_GE(p2.recommended_max_bounces, p1.recommended_max_bounces);
}
