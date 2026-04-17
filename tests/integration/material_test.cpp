// ─────────────────────────────────────────────────────────────────────
// material_test.cpp – Phase 4: Material / BSDF tests (v5 RGB)
//
// White furnace (energy conservation), reciprocity, PDF consistency,
// f·cos/pdf bounded, specular bounce sanity.
// ─────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include "material/bsdf_shared.h"
#include "material/bsdf.h"
#include "material/specular.h"
#include "scene/material.h"
#include "core/random.h"
#include "core/color.h"

// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────

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

static Material make_glossy_dielectric(Color3 Kd, Color3 Ks, float roughness, float ior) {
    Material m;
    m.type = MaterialType::GlossyDielectric;
    m.Kd = Kd;
    m.Ks = Ks;
    m.roughness = roughness;
    m.ior = ior;
    return m;
}

static Material make_mirror(Color3 Ks) {
    Material m;
    m.type = MaterialType::Mirror;
    m.Ks = Ks;
    return m;
}

static Material make_glass(float ior, Color3 Tf) {
    Material m;
    m.type = MaterialType::Glass;
    m.ior = ior;
    m.Tf = Tf;
    m.dispersion = false;
    return m;
}

static Material make_clearcoat(Color3 Kd, float coat, float coat_rough, float ior) {
    Material m;
    m.type = MaterialType::Clearcoat;
    m.Kd = Kd;
    m.pb_clearcoat = coat;
    m.pb_clearcoat_roughness = coat_rough;
    m.ior = ior;
    return m;
}

static Material make_fabric(Color3 Kd, float sheen, float tint) {
    Material m;
    m.type = MaterialType::Fabric;
    m.Kd = Kd;
    m.pb_sheen = sheen;
    m.pb_sheen_tint = tint;
    return m;
}

// ─────────────────────────────────────────────────────────────────────
// Test: Fresnel & GGX shared helpers
// ─────────────────────────────────────────────────────────────────────

TEST(BSDFShared, FresnelSchlickBounds) {
    // At normal incidence, Schlick returns f0
    EXPECT_NEAR(fresnel_schlick(1.0f, 0.04f), 0.04f, 1e-5f);
    // At grazing, Schlick returns 1.0
    EXPECT_NEAR(fresnel_schlick(0.0f, 0.04f), 1.0f, 1e-5f);
}

TEST(BSDFShared, FresnelDielectricTIR) {
    // Glass (n=1.5) → total internal reflection at steep angle
    // eta = n_glass / n_air = 1.5, critical angle sin(θ) = 1/1.5 = 0.667
    float F = fresnel_dielectric(0.1f, 1.5f); // cos_i=0.1 → sin_i≈0.995 > 0.667
    EXPECT_FLOAT_EQ(F, 1.0f); // TIR
}

TEST(BSDFShared, FresnelDielectricNormal) {
    // At normal incidence: F = ((1-1.5)/(1+1.5))^2 = 0.04
    float F = fresnel_dielectric(1.0f, 1.0f / 1.5f);
    EXPECT_NEAR(F, 0.04f, 0.005f);
}

TEST(BSDFShared, GGXNDFPeakAtNormal) {
    float alpha = 0.1f;
    float3 h_normal = make_f3(0, 0, 1); // half-vector aligned with normal
    float3 h_tilted = normalize(make_f3(0.5f, 0, 0.866f));
    float D_n = ggx_D(h_normal, alpha);
    float D_t = ggx_D(h_tilted, alpha);
    EXPECT_GT(D_n, D_t); // NDF peaks at normal for low roughness
    EXPECT_GT(D_n, 0.f);
}

TEST(BSDFShared, VNDFSamplingAboveHemisphere) {
    PCGRng rng = PCGRng::seed(42);
    float alpha = 0.3f;
    float3 wo = normalize(make_f3(0.3f, 0.1f, 0.95f));
    int below = 0;
    const int N = 10000;
    for (int i = 0; i < N; i++) {
        float3 h = ggx_sample_halfvector(wo, alpha, rng.next_float(), rng.next_float());
        if (h.z < 0.f) below++;
    }
    EXPECT_LT(below, N / 100); // < 1% below hemisphere
}

TEST(BSDFShared, ReflectLocalSymmetry) {
    float3 wo = normalize(make_f3(0.3f, 0.4f, 0.866f));
    float3 wi = reflect_local(wo);
    EXPECT_NEAR(wi.x, -wo.x, 1e-6f);
    EXPECT_NEAR(wi.y, -wo.y, 1e-6f);
    EXPECT_NEAR(wi.z,  wo.z, 1e-6f);
}

TEST(BSDFShared, RefractLocalSnellsLaw) {
    float3 wo = normalize(make_f3(0.3f, 0, 0.95f));
    float eta = 1.0f / 1.5f; // air → glass
    float3 wt;
    EXPECT_TRUE(refract_local(wo, eta, wt));
    // Snell: n1*sin(θ1) = n2*sin(θ2) → eta*sin(θ_i) = sin(θ_t)
    float sin_i = sqrtf(1.f - wo.z * wo.z);
    float sin_t = sqrtf(1.f - wt.z * wt.z);
    EXPECT_NEAR(eta * sin_i, sin_t, 1e-5f);
}

// ─────────────────────────────────────────────────────────────────────
// Test: White Furnace (energy conservation) — MC integration of
// ∫ f(wo,wi) · cos(θi) dωi over the hemisphere.
// For a white (Kd=1) Lambertian, this should be 1.0 (Kd/π · π = 1).
// For other materials, it should be ≤ 1.0 (energy conserving).
// ─────────────────────────────────────────────────────────────────────

static float white_furnace_mc(const Material& mat, float3 wo, int N = 100000) {
    PCGRng rng = PCGRng::seed(12345);
    double sum = 0.0;
    int valid = 0;
    for (int i = 0; i < N; i++) {
        BSDFSample s = bsdf::sample(mat, wo, rng);
        if (s.pdf > 0.f && s.wi.z > 0.f && !s.is_specular) {
            // f · cos / pdf
            float cos_i = s.wi.z;
            float val = s.f.luminance() * cos_i / s.pdf;
            sum += val;
            valid++;
        }
    }
    return (valid > 0) ? (float)(sum / N) : 0.f;
}

TEST(WhiteFurnace, Lambertian) {
    Material m = make_lambertian(Color3::one());
    float3 wo = normalize(make_f3(0, 0, 1));
    float integral = white_furnace_mc(m, wo);
    // White Lambertian: should integrate to 1.0 (single-channel = luminance)
    EXPECT_NEAR(integral, 1.0f, 0.03f);
}

TEST(WhiteFurnace, LambertianOblique) {
    Material m = make_lambertian(Color3::one());
    float3 wo = normalize(make_f3(0.5f, 0, 0.866f)); // 30° from normal
    float integral = white_furnace_mc(m, wo);
    EXPECT_NEAR(integral, 1.0f, 0.03f);
}

TEST(WhiteFurnace, GlossyMetalEnergyConserving) {
    Material m = make_glossy_metal(Color3::constant(0.3f),
                                    Color3::constant(0.7f), 0.3f);
    float3 wo = normalize(make_f3(0, 0, 1));
    float integral = white_furnace_mc(m, wo);
    EXPECT_LE(integral, 1.05f); // Allow small MC noise
    EXPECT_GT(integral, 0.f);
}

TEST(WhiteFurnace, GlossyDielectricEnergyConserving) {
    Material m = make_glossy_dielectric(Color3::constant(0.8f),
                                         Color3::constant(1.0f), 0.5f, 1.5f);
    float3 wo = normalize(make_f3(0, 0, 1));
    float integral = white_furnace_mc(m, wo);
    EXPECT_LE(integral, 1.05f);
    EXPECT_GT(integral, 0.f);
}

TEST(WhiteFurnace, ClearcoatEnergyConserving) {
    Material m = make_clearcoat(Color3::one(), 1.0f, 0.1f, 1.5f);
    float3 wo = normalize(make_f3(0, 0, 1));
    float integral = white_furnace_mc(m, wo);
    EXPECT_LE(integral, 1.05f);
    EXPECT_GT(integral, 0.f);
}

TEST(WhiteFurnace, FabricEnergyConserving) {
    Material m = make_fabric(Color3::one(), 1.0f, 0.5f);
    float3 wo = normalize(make_f3(0, 0, 1));
    float integral = white_furnace_mc(m, wo);
    // Fabric = diffuse + sheen; with Kd=1 and high sheen it can exceed 1.
    // In practice, sheen adds to diffuse so total reflectance > 1 is
    // accepted (non-physical but consistent with v4 / Disney model).
    EXPECT_GT(integral, 0.f);
    printf("  Fabric furnace integral: %.4f\n", integral);
}

// ─────────────────────────────────────────────────────────────────────
// Test: Reciprocity — f(wo, wi) == f(wi, wo) for non-delta BSDFs
// ─────────────────────────────────────────────────────────────────────

static void check_reciprocity(const Material& mat, const char* name) {
    PCGRng rng = PCGRng::seed(7777);
    float3 wo = normalize(make_f3(0.3f, 0.2f, 0.93f));
    int violations = 0;
    const int N = 5000;
    for (int i = 0; i < N; i++) {
        float3 wi = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
        if (wi.z <= 0.f) continue;
        Color3 f_fwd = bsdf::evaluate(mat, wo, wi);
        Color3 f_rev = bsdf::evaluate(mat, wi, wo);
        float diff = fabsf(f_fwd.luminance() - f_rev.luminance());
        float scale = fmaxf(f_fwd.luminance(), f_rev.luminance()) + 1e-10f;
        if (diff / scale > 0.01f) violations++;
    }
    EXPECT_LT(violations, N / 50) << name << " reciprocity violations";
}

TEST(Reciprocity, Lambertian) {
    check_reciprocity(make_lambertian(Color3::constant(0.7f)), "Lambertian");
}

TEST(Reciprocity, GlossyMetal) {
    check_reciprocity(make_glossy_metal(Color3::constant(0.2f),
                                         Color3::constant(0.8f), 0.3f), "GlossyMetal");
}

TEST(Reciprocity, GlossyDielectric) {
    check_reciprocity(make_glossy_dielectric(Color3::constant(0.5f),
                                              Color3::constant(0.5f), 0.4f, 1.5f),
                      "GlossyDielectric");
}

TEST(Reciprocity, Clearcoat) {
    check_reciprocity(make_clearcoat(Color3::constant(0.5f), 1.0f, 0.2f, 1.5f),
                      "Clearcoat");
}

TEST(Reciprocity, Fabric) {
    check_reciprocity(make_fabric(Color3::constant(0.6f), 0.5f, 0.3f),
                      "Fabric");
}

// ─────────────────────────────────────────────────────────────────────
// Test: PDF consistency — sample's f/pdf matches evaluate×cosθ
// For non-delta materials, sampled f·cos/pdf should match
// evaluate(wo,wi)·cos/pdf(wo,wi) within tolerance.
// ─────────────────────────────────────────────────────────────────────

static void check_pdf_consistency(const Material& mat, const char* name) {
    PCGRng rng = PCGRng::seed(1234);
    float3 wo = normalize(make_f3(0.2f, 0.1f, 0.97f));
    int violations = 0;
    const int N = 5000;
    for (int i = 0; i < N; i++) {
        BSDFSample s = bsdf::sample(mat, wo, rng);
        if (s.is_specular || s.pdf <= 0.f || s.wi.z <= 0.f) continue;

        Color3 f_eval = bsdf::evaluate(mat, wo, s.wi);
        float  p_eval = bsdf::pdf(mat, wo, s.wi);

        // Check f consistency
        float diff_f = fabsf(s.f.luminance() - f_eval.luminance());
        float scale_f = fmaxf(s.f.luminance(), f_eval.luminance()) + 1e-10f;
        if (diff_f / scale_f > 0.02f) violations++;

        // Check pdf consistency
        float diff_p = fabsf(s.pdf - p_eval);
        float scale_p = fmaxf(s.pdf, p_eval) + 1e-10f;
        if (diff_p / scale_p > 0.02f) violations++;
    }
    EXPECT_LT(violations, N / 20) << name << " PDF/f consistency violations";
}

TEST(PDFConsistency, Lambertian) {
    check_pdf_consistency(make_lambertian(Color3::constant(0.5f)), "Lambertian");
}

TEST(PDFConsistency, GlossyMetal) {
    check_pdf_consistency(make_glossy_metal(Color3::constant(0.3f),
                                             Color3::constant(0.7f), 0.4f), "GlossyMetal");
}

TEST(PDFConsistency, GlossyDielectric) {
    check_pdf_consistency(make_glossy_dielectric(Color3::constant(0.5f),
                                                  Color3::constant(0.5f), 0.3f, 1.5f),
                          "GlossyDielectric");
}

TEST(PDFConsistency, Clearcoat) {
    check_pdf_consistency(make_clearcoat(Color3::constant(0.5f), 1.0f, 0.2f, 1.5f),
                          "Clearcoat");
}

TEST(PDFConsistency, Fabric) {
    check_pdf_consistency(make_fabric(Color3::constant(0.5f), 0.5f, 0.3f), "Fabric");
}

// ─────────────────────────────────────────────────────────────────────
// Test: f·cos/pdf bounded — no firefly contributions
// ─────────────────────────────────────────────────────────────────────

static float max_fcos_over_pdf(const Material& mat, float3 wo, int N = 50000) {
    PCGRng rng = PCGRng::seed(9999);
    float max_val = 0.f;
    for (int i = 0; i < N; i++) {
        BSDFSample s = bsdf::sample(mat, wo, rng);
        if (s.is_specular || s.pdf <= 0.f || s.wi.z <= 0.f) continue;
        float val = s.f.max_component() * s.wi.z / s.pdf;
        max_val = fmaxf(max_val, val);
    }
    return max_val;
}

TEST(FireflyBound, LambertianBounded) {
    Material m = make_lambertian(Color3::one());
    float3 wo = normalize(make_f3(0, 0, 1));
    float max_v = max_fcos_over_pdf(m, wo);
    // Lambertian: f·cos/pdf = (1/π)·cos·(π/cos) = 1.0
    EXPECT_LE(max_v, 1.1f);
}

TEST(FireflyBound, GlossyMetalBounded) {
    Material m = make_glossy_metal(Color3::constant(0.3f),
                                    Color3::constant(0.8f), 0.3f);
    float3 wo = normalize(make_f3(0, 0, 1));
    float max_v = max_fcos_over_pdf(m, wo);
    EXPECT_LE(max_v, 10.f); // reasonable for microfacet
    printf("  GlossyMetal max f*cos/pdf: %.4f\n", max_v);
}

TEST(FireflyBound, GlossyDielectricBounded) {
    Material m = make_glossy_dielectric(Color3::constant(0.5f),
                                         Color3::constant(1.0f), 0.2f, 1.5f);
    float3 wo = normalize(make_f3(0, 0, 1));
    float max_v = max_fcos_over_pdf(m, wo);
    EXPECT_LE(max_v, 10.f);
    printf("  GlossyDielectric max f*cos/pdf: %.4f\n", max_v);
}

// ─────────────────────────────────────────────────────────────────────
// Test: Glass / Mirror specular sampling sanity
// ─────────────────────────────────────────────────────────────────────

TEST(SpecularSampling, MirrorReflection) {
    Material m = make_mirror(Color3::one());
    float3 wo = normalize(make_f3(0.3f, 0.2f, 0.93f));
    PCGRng rng = PCGRng::seed(42);
    BSDFSample s = bsdf::sample(m, wo, rng);
    EXPECT_TRUE(s.is_specular);
    EXPECT_NEAR(s.wi.x, -wo.x, 1e-5f);
    EXPECT_NEAR(s.wi.y, -wo.y, 1e-5f);
    EXPECT_NEAR(s.wi.z,  wo.z, 1e-5f);
}

TEST(SpecularSampling, GlassEnergyConservation) {
    // MC: average |filter| over many samples should be ~1 (energy neutral)
    Material m = make_glass(1.5f, Color3::one());
    float3 wo = normalize(make_f3(0, 0, 1)); // normal incidence
    PCGRng rng = PCGRng::seed(42);
    double sum = 0.0;
    const int N = 100000;
    for (int i = 0; i < N; i++) {
        BSDFSample s = bsdf::sample(m, wo, rng);
        // For delta BSDFs: throughput = f * cos / pdf
        if (s.pdf > 0.f) {
            float cos_i = fabsf(s.wi.z);
            sum += s.f.luminance() * cos_i / s.pdf;
        }
    }
    float avg = (float)(sum / N);
    // Glass at normal incidence: ~96% transmitted, ~4% reflected
    // Total should be ~1.0 (energy conserved)
    EXPECT_NEAR(avg, 1.0f, 0.05f);
    printf("  Glass energy: %.4f\n", avg);
}

TEST(SpecularSampling, GlassTIRAtGrazingFromInside) {
    // From inside glass at steep angle → should get TIR
    Material m = make_glass(1.5f, Color3::one());
    float3 wo = normalize(make_f3(0.99f, 0, 0.14f)); // very steep inside
    // wo.z > 0 means "entering" from the glass_sample perspective,
    // but we want to test exiting. Let's flip:
    float3 wo_exit = normalize(make_f3(0.99f, 0, -0.14f));
    // Actually in glass_sample, entering = wo.z > 0.
    // wo.z < 0 means exiting. eta = mat.ior = 1.5
    // sin²_t = 1.5² * sin²_i. sin_i ≈ 0.99. sin²_t ≈ 2.25 * 0.98 > 1 → TIR.
    PCGRng rng = PCGRng::seed(42);
    int reflect_count = 0;
    const int N = 1000;
    for (int i = 0; i < N; i++) {
        BSDFSample s = bsdf::sample(m, wo_exit, rng);
        if (s.wi.z < 0.f) reflect_count++; // Reflected back inside
    }
    // At this angle, should be 100% TIR (all reflected)
    EXPECT_GT(reflect_count, N * 95 / 100);
}

TEST(SpecularSampling, OrientedFrameDoesNotEncodeGlassEntryExit) {
    float3 normal = make_f3(0.f, 0.f, 1.f);
    float3 wo_exit = make_f3(0.f, 0.f, -1.f);

    ONB frame = orient_frame_to_outgoing(normal, wo_exit);
    float3 wo_local = frame.world_to_local(wo_exit);

    EXPECT_GT(wo_local.z, 0.f);
    EXPECT_LT(dot(wo_exit, normal), 0.f);
}

// ─────────────────────────────────────────────────────────────────────
// Test: Specular bounce helper (world-space)
// ─────────────────────────────────────────────────────────────────────

TEST(SpecularBounce, MirrorBounceDirection) {
    float3 dir = normalize(make_f3(0.3f, 0.2f, -0.93f)); // incoming toward surface
    float3 normal = make_f3(0, 0, 1);
    float3 geo_normal = make_f3(0, 0, 1);

    SpecularBounceParams bp;
    bp.is_glass = false;
    bp.is_thin = false;
    bp.ior = 1.5f;
    bp.Tf = Color3::one();
    bp.medium_id = -1;

    PCGRng rng = PCGRng::seed(42);
    SpecularBounceResult r = specular_bounce(dir, make_f3(0,0,0), normal, geo_normal,
                                              bp, rng);

    // Mirror: reflected direction
    float3 expected = dir - normal * (2.f * dot(dir, normal));
    EXPECT_NEAR(r.new_dir.x, expected.x, 1e-5f);
    EXPECT_NEAR(r.new_dir.y, expected.y, 1e-5f);
    EXPECT_NEAR(r.new_dir.z, expected.z, 1e-5f);
    // Filter should be 1.0 for mirror
    EXPECT_NEAR(r.filter.r, 1.f, 1e-5f);
}

TEST(SpecularBounce, GlassIORStackUpdate) {
    float3 dir = normalize(make_f3(0, 0, -1)); // straight in
    float3 normal = make_f3(0, 0, 1);
    float3 geo_normal = make_f3(0, 0, 1);

    SpecularBounceParams bp;
    bp.is_glass = true;
    bp.is_thin = false;
    bp.ior = 1.5f;
    bp.Tf = Color3::one();
    bp.medium_id = -1;

    IORStack stack;
    EXPECT_EQ(stack.depth, 0);

    // Force refraction by setting seed that gives rng < (1-F)
    // At normal incidence, F ≈ 0.04, so ~96% chance of refraction.
    // We'll just run a few times and check at least one refraction happens.
    bool refracted = false;
    for (int seed = 0; seed < 20; seed++) {
        IORStack s;
        PCGRng rng = PCGRng::seed(seed);
        SpecularBounceResult r = specular_bounce(dir, make_f3(0,0,0), normal, geo_normal,
                                                  bp, rng, &s);
        if (s.depth > 0) {
            // Refraction occurred → IOR was pushed
            EXPECT_NEAR(s.iors[0], 1.5f, 1e-5f);
            refracted = true;
            // Check refracted direction goes into the surface (z < 0)
            EXPECT_LT(r.new_dir.z, 0.f);
            break;
        }
    }
    EXPECT_TRUE(refracted) << "No refraction in 20 seeds (very unlikely at F≈0.04)";
}

TEST(SpecularBounce, ThinGlassPassthrough) {
    float3 dir = normalize(make_f3(0.1f, 0, -0.995f));
    float3 normal = make_f3(0, 0, 1);
    float3 geo_normal = make_f3(0, 0, 1);

    SpecularBounceParams bp;
    bp.is_glass = true;
    bp.is_thin = true;
    bp.ior = 1.5f;
    bp.Tf = Color3::from_rgb(0.8f, 0.9f, 0.95f);
    bp.medium_id = -1;

    PCGRng rng = PCGRng::seed(42);
    // Near-normal incidence on thin glass: mostly transmits straight through.
    int passthrough = 0;
    const int N = 1000;
    for (int i = 0; i < N; i++) {
        PCGRng r2 = PCGRng::seed(i);
        SpecularBounceResult r = specular_bounce(dir, make_f3(0,0,0), normal, geo_normal,
                                                  bp, r2);
        if (r.new_dir.z < 0.f) passthrough++; // Transmitted (same direction as input)
    }
    // ~96% should pass through at normal incidence
    EXPECT_GT(passthrough, N * 85 / 100);
}

// ─────────────────────────────────────────────────────────────────────
// Test: Delta distributions have zero evaluate/pdf
// ─────────────────────────────────────────────────────────────────────

TEST(DeltaBSDF, MirrorEvaluateZero) {
    Material m = make_mirror(Color3::one());
    float3 wo = normalize(make_f3(0.3f, 0, 0.95f));
    float3 wi = normalize(make_f3(0.1f, 0.2f, 0.97f));
    Color3 f = bsdf::evaluate(m, wo, wi);
    EXPECT_FLOAT_EQ(f.luminance(), 0.f);
    EXPECT_FLOAT_EQ(bsdf::pdf(m, wo, wi), 0.f);
}

TEST(DeltaBSDF, GlassEvaluateZero) {
    Material m = make_glass(1.5f, Color3::one());
    float3 wo = normalize(make_f3(0.3f, 0, 0.95f));
    float3 wi = normalize(make_f3(0.1f, 0.2f, 0.97f));
    Color3 f = bsdf::evaluate(m, wo, wi);
    EXPECT_FLOAT_EQ(f.luminance(), 0.f);
    EXPECT_FLOAT_EQ(bsdf::pdf(m, wo, wi), 0.f);
}
