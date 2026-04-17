// ─────────────────────────────────────────────────────────────────────
// tests/unit/test_material.cpp – Material / BSDF unit tests (GTest)
//
// Stage 4: Furnace, reciprocity, PDF consistency for all material types.
// Subset of the 34 tests in src/material_test.cpp, restructured for
// the ppt_unit_tests aggregate target.
// ─────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include "material/bsdf_shared.h"
#include "material/bsdf.h"
#include "scene/material.h"
#include "core/random.h"
#include "core/color.h"
#include <cmath>

// ── Helpers ─────────────────────────────────────────────────────────

static Material make_lambertian(Color3 Kd) {
    Material m;
    m.type = MaterialType::Lambertian;
    m.Kd = Kd;
    return m;
}

static Material make_glossy_metal(Color3 Kd, Color3 Ks, float roughness) {
    Material m;
    m.type = MaterialType::GlossyMetal;
    m.Kd = Kd;
    m.Ks = Ks;
    m.roughness = roughness;
    return m;
}

static Material make_mirror(Color3 Ks) {
    Material m;
    m.type = MaterialType::Mirror;
    m.Ks = Ks;
    return m;
}

static Material make_glass(float ior) {
    Material m;
    m.type = MaterialType::Glass;
    m.ior = ior;
    m.Tf = Color3::one();
    m.dispersion = false;
    return m;
}

static float white_furnace_integral(const Material& mat, float3 wo, int N = 50000) {
    PCGRng rng = PCGRng::seed(42, 0);
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        BSDFSample samp = bsdf::sample(mat, wo, rng);
        if (samp.pdf > 1e-10f && samp.wi.z > 0.0f) {
            float cos_i = samp.wi.z;
            Color3 f = bsdf::evaluate(mat, wo, samp.wi);
            float lum = f.luminance();
            sum += (lum * cos_i) / samp.pdf;
        }
    }
    return (float)(sum / N);
}

// ── Fresnel helpers ─────────────────────────────────────────────────

TEST(MaterialUnit, FresnelSchlickNormal) {
    float F = fresnel_schlick(1.0f, 0.04f);
    EXPECT_NEAR(F, 0.04f, 1e-3f);
}

TEST(MaterialUnit, FresnelSchlickGrazing) {
    float F = fresnel_schlick(0.0f, 0.04f);
    EXPECT_NEAR(F, 1.0f, 1e-3f);
}

// ── Furnace tests ───────────────────────────────────────────────────

TEST(MaterialUnit, FurnaceLambertian) {
    auto mat = make_lambertian(Color3::one());
    float3 wo = make_f3(0, 0, 1);
    float integral = white_furnace_integral(mat, wo);
    EXPECT_NEAR(integral, 1.0f, 0.08f);
}

TEST(MaterialUnit, FurnaceGlossyMetal) {
    auto mat = make_glossy_metal(Color3::zero(), Color3::one(), 0.3f);
    float3 wo = make_f3(0, 0, 1);
    float integral = white_furnace_integral(mat, wo);
    EXPECT_LE(integral, 1.1f);
}

// ── Reciprocity ─────────────────────────────────────────────────────

TEST(MaterialUnit, ReciprocityLambertian) {
    auto mat = make_lambertian(Color3::constant(0.8f));
    float3 wo1 = normalize(make_f3(0.3f, 0.2f, 0.9f));
    float3 wi1 = normalize(make_f3(-0.2f, 0.4f, 0.85f));

    Color3 f_fwd = bsdf::evaluate(mat, wo1, wi1);
    Color3 f_rev = bsdf::evaluate(mat, wi1, wo1);

    EXPECT_NEAR(f_fwd.luminance(), f_rev.luminance(), 0.01f);
}

// ── PDF consistency ─────────────────────────────────────────────────

TEST(MaterialUnit, PDFConsistencyLambertian) {
    auto mat = make_lambertian(Color3::constant(0.8f));
    float3 wo = make_f3(0, 0, 1);
    PCGRng rng = PCGRng::seed(123, 0);

    int consistent = 0;
    int total = 1000;
    for (int i = 0; i < total; ++i) {
        BSDFSample samp = bsdf::sample(mat, wo, rng);
        if (samp.pdf < 1e-10f || samp.wi.z <= 0.0f) continue;

        float eval_pdf = bsdf::pdf(mat, wo, samp.wi);
        float ratio = samp.pdf / std::max(eval_pdf, 1e-10f);
        if (ratio > 0.5f && ratio < 2.0f) ++consistent;
    }
    EXPECT_GT(consistent, 900);
}

// ── Specular ────────────────────────────────────────────────────────

TEST(MaterialUnit, MirrorReflection) {
    auto mat = make_mirror(Color3::one());
    float3 wo = normalize(make_f3(0.3f, 0, 0.95f));
    PCGRng rng = PCGRng::seed(7, 0);

    BSDFSample samp = bsdf::sample(mat, wo, rng);
    EXPECT_TRUE(samp.is_specular);
    EXPECT_GT(samp.wi.z, 0.0f);              // reflected into upper hemisphere
    EXPECT_NEAR(samp.wi.x, -wo.x, 0.05f);   // x flips
    EXPECT_NEAR(samp.wi.z, wo.z, 0.05f);     // z preserved
}

TEST(MaterialUnit, GlassEnergyConservation) {
    auto mat = make_glass(1.5f);
    float3 wo = make_f3(0, 0, 1);
    PCGRng rng = PCGRng::seed(42, 0);

    double total_weight = 0.0;
    int N = 10000;
    for (int i = 0; i < N; ++i) {
        BSDFSample samp = bsdf::sample(mat, wo, rng);
        if (samp.pdf > 0.0f) {
            total_weight += samp.f.luminance() / samp.pdf;
        }
    }
    float avg = (float)(total_weight / N);
    EXPECT_NEAR(avg, 1.0f, 0.2f);
}

TEST(MaterialUnit, MirrorEvaluateZero) {
    auto mat = make_mirror(Color3::one());
    float3 wo = make_f3(0, 0, 1);
    float3 wi = normalize(make_f3(0.5f, 0, 0.866f));
    Color3 f = bsdf::evaluate(mat, wo, wi);
    EXPECT_FLOAT_EQ(f.luminance(), 0.0f);
}
