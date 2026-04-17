// ─────────────────────────────────────────────────────────────────────
// tests/unit/test_scene_loader.cpp – Scene loading unit tests
//
// Stage 1: PBRT parsing, material construction, scene building.
// ─────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include "core/types.h"
#include "core/color.h"
#include "core/config.h"
#include "core/alias_table.h"
#include "scene/material.h"
#include "scene/triangle.h"
#include "scene/scene.h"
#include "scene/scene_builder.h"
#include "scene/pbrt/pbrt_parser.h"
#include "scene/pbrt/pbrt_loader.h"
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>

namespace {

namespace fs = std::filesystem;
constexpr float kPi = 3.14159265358979323846f;

fs::path make_temp_dir() {
    auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    fs::path dir = fs::temp_directory_path() / ("ppt_pbrt_test_" + std::to_string(stamp));
    fs::create_directories(dir);
    return dir;
}

void write_text(const fs::path& path, const char* text) {
    std::ofstream file(path, std::ios::binary);
    ASSERT_TRUE(file.is_open()) << path.string();
    file << text;
}

const Material* find_material(const Scene& scene, const char* name) {
    for (const auto& material : scene.materials) {
        if (material.name == name) return &material;
    }
    return nullptr;
}

} // namespace

// ── Material ────────────────────────────────────────────────────────

TEST(SceneLoader, MaterialDefaults) {
    Material m;
    EXPECT_EQ(m.type, MaterialType::Lambertian);
    EXPECT_FALSE(m.is_emissive());
    EXPECT_FALSE(m.is_specular());
    EXPECT_FLOAT_EQ(m.Kd.r, 0.5f);
}

TEST(SceneLoader, EmissiveMaterial) {
    Material m;
    m.type = MaterialType::Emissive;
    m.Le = Color3::constant(10.0f);
    EXPECT_TRUE(m.is_emissive());
    EXPECT_FLOAT_EQ(m.Le.r, 10.0f);
}

TEST(SceneLoader, GlassMaterial) {
    Material m;
    m.type = MaterialType::Glass;
    m.ior = 1.5f;
    EXPECT_TRUE(m.is_specular());
    EXPECT_FLOAT_EQ(m.ior, 1.5f);
}

// ── Triangle ────────────────────────────────────────────────────────

TEST(SceneLoader, TriangleArea) {
    Triangle tri;
    tri.v0 = make_f3(0, 0, 0);
    tri.v1 = make_f3(1, 0, 0);
    tri.v2 = make_f3(0, 1, 0);
    float area = tri.area();
    EXPECT_NEAR(area, 0.5f, 1e-5f);
}

TEST(SceneLoader, TriangleNormal) {
    Triangle tri;
    tri.v0 = make_f3(0, 0, 0);
    tri.v1 = make_f3(1, 0, 0);
    tri.v2 = make_f3(0, 1, 0);
    float3 n = tri.geometric_normal();
    EXPECT_NEAR(n.z, 1.0f, 1e-5f);
}

TEST(SceneLoader, TriangleRayIntersect) {
    Triangle tri;
    tri.v0 = make_f3(-1, -1, -2);
    tri.v1 = make_f3(1, -1, -2);
    tri.v2 = make_f3(0, 1, -2);

    Ray ray;
    ray.origin = make_f3(0, 0, 0);
    ray.direction = make_f3(0, 0, -1);
    ray.tmin = 0.001f;
    ray.tmax = 1e30f;

    float t_out, u_out, v_out;
    bool hit = tri.intersect(ray, t_out, u_out, v_out);
    EXPECT_TRUE(hit);
    EXPECT_NEAR(t_out, 2.0f, 1e-4f);
}


// ── Scene building ──────────────────────────────────────────────────

TEST(SceneLoader, CornellBoxTriangleCount) {
    Scene s = scene_builder::build_cornell_box();
    EXPECT_GE((int)s.triangles.size(), 12);
}

TEST(SceneLoader, CornellBoxEmissives) {
    Scene s = scene_builder::build_cornell_box();
    EXPECT_GE(s.num_emissive(), 1);
    EXPECT_GT(s.total_emissive_power, 0.0f);
}

TEST(SceneLoader, CornellBoxBounds) {
    Scene s = scene_builder::build_cornell_box();
    AABB b;
    b.lo = s.scene_bounds.lo;
    b.hi = s.scene_bounds.hi;
    EXPECT_GT(b.diagonal(), 0.0f);
}

TEST(SceneLoader, GlassSphereScene) {
    Scene s = scene_builder::build_glass_sphere();
    bool has_glass = false;
    for (auto& m : s.materials) {
        if (m.type == MaterialType::Glass) has_glass = true;
    }
    EXPECT_TRUE(has_glass);
}

TEST(SceneLoader, PbrtAttributeShapeInheritance) {
    fs::path dir = make_temp_dir();
    fs::path path = dir / "shape_attr.pbrt";

    write_text(path,
        "WorldBegin\n"
        "Attribute \"shape\" \"float radius\" 2\n"
        "Shape \"sphere\"\n");

    pbrt::PbrtParser parser;
    pbrt::PbrtScene scene = parser.parse_file(path.string());

    ASSERT_EQ(scene.shapes.size(), 1u);
    EXPECT_DOUBLE_EQ(pbrt::get_float(scene.shapes[0].params, "radius", 0.0), 2.0);

    fs::remove_all(dir);
}

TEST(SceneLoader, PbrtAttributeMaterialInheritance) {
    fs::path dir = make_temp_dir();
    fs::path path = dir / "material_attr.pbrt";

    write_text(path,
        "WorldBegin\n"
        "Attribute \"material\" \"rgb reflectance\" [0.2 0.3 0.4]\n"
        "Material \"diffuse\"\n"
        "Shape \"sphere\"\n");

    pbrt::PbrtParser parser;
    pbrt::PbrtScene scene = parser.parse_file(path.string());

    ASSERT_EQ(scene.shapes.size(), 1u);
    ASSERT_TRUE(scene.shapes[0].inline_mat);
    auto reflectance = pbrt::get_rgb(scene.shapes[0].inline_mat->params, "reflectance");
    ASSERT_EQ(reflectance.size(), 3u);
    EXPECT_DOUBLE_EQ(reflectance[0], 0.2);
    EXPECT_DOUBLE_EQ(reflectance[1], 0.3);
    EXPECT_DOUBLE_EQ(reflectance[2], 0.4);

    fs::remove_all(dir);
}

TEST(SceneLoader, PbrtImportAndCoordSysTransform) {
    fs::path dir = make_temp_dir();
    fs::path main_path = dir / "main.pbrt";
    fs::path child_path = dir / "child.pbrt";

    write_text(child_path,
        "Attribute \"shape\" \"float radius\" 3\n"
        "Shape \"sphere\"\n");

    write_text(main_path,
        "WorldBegin\n"
        "Import \"child.pbrt\"\n"
        "Translate 1 2 3\n"
        "CoordinateSystem \"saved\"\n"
        "Identity\n"
        "CoordSysTransform \"saved\"\n"
        "Shape \"sphere\"\n");

    pbrt::PbrtParser parser;
    pbrt::PbrtScene scene = parser.parse_file(main_path.string());

    ASSERT_EQ(scene.shapes.size(), 2u);
    EXPECT_DOUBLE_EQ(pbrt::get_float(scene.shapes[0].params, "radius", 0.0), 3.0);
    EXPECT_NEAR(scene.shapes[1].transform.m[0][3], 1.0, 1e-8);
    EXPECT_NEAR(scene.shapes[1].transform.m[1][3], 2.0, 1e-8);
    EXPECT_NEAR(scene.shapes[1].transform.m[2][3], 3.0, 1e-8);

    fs::remove_all(dir);
}

TEST(SceneLoader, PbrtRejectsAnimatedTransforms) {
    fs::path dir = make_temp_dir();
    fs::path path = dir / "animated.pbrt";

    write_text(path,
        "TransformTimes 0 1\n"
        "WorldBegin\n"
        "Shape \"sphere\"\n");

    pbrt::PbrtParser parser;
    EXPECT_THROW(parser.parse_file(path.string()), std::runtime_error);

    fs::remove_all(dir);
}

TEST(SceneLoader, PbrtLightProxyIntensityImport) {
    fs::path dir = make_temp_dir();
    fs::path path = dir / "lights.pbrt";

    write_text(path,
        "WorldBegin\n"
        "LightSource \"point\" \"rgb I\" [2 4 6]\n"
        "LightSource \"spot\" \"rgb I\" [1 1 1] \"float coneangle\" 20 \"float conedeltaangle\" 10\n"
        "LightSource \"distant\" \"rgb L\" [1 1 1] \"float illuminance\" 12\n"
        "LightSource \"infinite\" \"rgb L\" [1 1 1] \"float illuminance\" 9 \"point3 portal\" [-1 -1 0  1 -1 0  1 1 0  -1 1 0]\n"
        "Shape \"sphere\"\n");

    Scene scene;
    ASSERT_TRUE(load_pbrt(path.string(), scene));
    scene.compute_bounds();
    scene.build_emissive_distribution();

    EXPECT_TRUE(scene.has_portals);

    const Material* point = find_material(scene, "__point_light");
    const Material* spot = find_material(scene, "__spot_light");
    const Material* distant = find_material(scene, "__distant_light");
    const Material* portal = find_material(scene, "__portal_emitter");

    ASSERT_NE(point, nullptr);
    ASSERT_NE(spot, nullptr);
    ASSERT_NE(distant, nullptr);
    ASSERT_NE(portal, nullptr);

    float point_expected = 4.0f / (kPi * 0.01f * 0.01f);
    EXPECT_NEAR(point->mean_emission(), point_expected, point_expected * 0.02f);

    float outer = 30.0f * kPi / 180.0f;
    float inner = 20.0f * kPi / 180.0f;
    float spot_omega = 2.0f * kPi * (1.0f - std::cos(inner))
                     + kPi * (std::cos(inner) - std::cos(outer));
    float sphere_power_unit = 4.0f * kPi * kPi * 0.01f * 0.01f;
    float spot_expected = spot_omega / sphere_power_unit;
    EXPECT_NEAR(spot->mean_emission(), spot_expected, spot_expected * 0.02f);

    float irradiance_expected = 12.0f / kPi;
    EXPECT_NEAR(distant->mean_emission(), irradiance_expected, irradiance_expected * 0.02f);

    float portal_expected = 9.0f / kPi;
    EXPECT_NEAR(portal->mean_emission(), portal_expected, portal_expected * 0.02f);

    fs::remove_all(dir);
}

TEST(SceneLoader, PbrtAreaLightTwosidedDuplicatesGeometry) {
    fs::path dir = make_temp_dir();
    fs::path path = dir / "twosided_area.pbrt";

    write_text(path,
        "WorldBegin\n"
        "AreaLightSource \"diffuse\" \"rgb L\" [1 1 1] \"bool twosided\" true\n"
        "Shape \"disk\" \"float radius\" 1\n");

    Scene scene;
    ASSERT_TRUE(load_pbrt(path.string(), scene));
    EXPECT_EQ(scene.triangles.size(), 128u);

    fs::remove_all(dir);
}
