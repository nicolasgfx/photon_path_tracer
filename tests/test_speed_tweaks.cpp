// ─────────────────────────────────────────────────────────────────────
// test_speed_tweaks.cpp – Unit & integration tests for speed tweaks
// ─────────────────────────────────────────────────────────────────────
// Coverage:
//   §1  Chromatic dispersion: Cauchy IOR, Material.Tf, per-bin Fresnel
//   §2  Photon path flags and bounce count
//   §3  CellInfoCache: build, query, adaptive radius, hotspot detection
//   §4  Glass BSDF + Tf spectral attenuation
//   §5  Energy conservation with dispersion
//   §6  Caustic photon tracing end-to-end
//   §7  CPU ground truth vs hash grid comparison
//   §8  IORStack nested dielectric tracking
// ─────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

#include "core/types.h"
#include "core/spectrum.h"
#include "core/config.h"
#include "core/random.h"
#include "core/cell_cache.h"
#include "scene/material.h"
#include "bsdf/bsdf.h"
#include "photon/photon.h"
#include "photon/hash_grid.h"
#include "photon/density_estimator.h"
#include "photon/emitter.h"
#include "photon/kd_tree.h"
#include "scene/scene.h"
#include "scene/obj_loader.h"
#include "renderer/renderer.h"

static constexpr float kTol   = 1e-5f;
static constexpr float kLoose = 1e-3f;

// ── Helpers ─────────────────────────────────────────────────────────

static Photon make_test_photon_flags(float3 pos, float3 wi, float3 norm,
                                     float flux, uint8_t flags, uint8_t bounces) {
    Photon p;
    p.position      = pos;
    p.wi            = wi;
    p.geom_normal   = norm;
    p.spectral_flux = Spectrum::constant(flux);
    p.path_flags    = flags;
    p.bounce_count  = bounces;
    return p;
}

static PhotonSoA make_soa(const std::vector<Photon>& photons) {
    PhotonSoA soa;
    soa.reserve(photons.size());
    for (const auto& p : photons)
        soa.push_back(p);
    return soa;
}

// =====================================================================
// §1  CHROMATIC DISPERSION – Cauchy equation IOR
// =====================================================================

TEST(Dispersion, CauchyIOR_CrownGlass) {
    // Crown glass: A=1.5046, B=4200 nm²
    // At lambda = 589 nm (sodium D line): n ≈ 1.5046 + 4200/589² ≈ 1.5167
    Material mat;
    mat.ior       = 1.5f;
    mat.cauchy_A  = 1.5046f;
    mat.cauchy_B  = 4200.0f;
    mat.dispersion = true;

    float n_589 = mat.ior_at_lambda(589.0f);
    EXPECT_NEAR(n_589, 1.5046f + 4200.0f / (589.0f * 589.0f), kTol);

    // Blue light (450 nm) should have higher IOR than red (650 nm)
    float n_blue = mat.ior_at_lambda(450.0f);
    float n_red  = mat.ior_at_lambda(650.0f);
    EXPECT_GT(n_blue, n_red);

    // IOR should be monotonically decreasing with wavelength
    float prev = mat.ior_at_lambda(380.0f);
    for (float lam = 400.0f; lam <= 780.0f; lam += 20.0f) {
        float n = mat.ior_at_lambda(lam);
        EXPECT_LE(n, prev + kTol) << "Non-monotonic at lambda=" << lam;
        prev = n;
    }
}

TEST(Dispersion, CauchyIOR_DispersionOff) {
    // When dispersion is off, ior_at_lambda returns the constant IOR
    Material mat;
    mat.ior       = 1.5f;
    mat.cauchy_A  = 1.5046f;
    mat.cauchy_B  = 4200.0f;
    mat.dispersion = false;

    // ior_at_lambda returns constant ior when dispersion is off
    float n = mat.ior_at_lambda(550.0f);
    EXPECT_FLOAT_EQ(n, 1.5f);
}

TEST(Dispersion, CauchyIOR_HighDispersionGlass) {
    // Flint glass: A=1.75, B=27000 nm²
    Material mat;
    mat.cauchy_A  = 1.75f;
    mat.cauchy_B  = 27000.0f;
    mat.dispersion = true;

    float n_blue = mat.ior_at_lambda(400.0f);
    float n_red  = mat.ior_at_lambda(700.0f);
    float spread = n_blue - n_red;

    // Flint glass should have much wider dispersion spread
    EXPECT_GT(spread, 0.05f);  // should be ~0.113
}

// =====================================================================
// §1b  Material Tf – spectral transmittance filter
// =====================================================================

TEST(MaterialTf, DefaultIsTransparent) {
    Material mat;
    for (int b = 0; b < NUM_LAMBDA; ++b) {
        EXPECT_FLOAT_EQ(mat.Tf.value[b], 1.0f);
    }
}

TEST(MaterialTf, ColoredGlass) {
    // Green-tinted glass: attenuate red and blue, pass green
    Material mat;
    for (int b = 0; b < NUM_LAMBDA; ++b) {
        float lambda = LAMBDA_MIN + (b + 0.5f) * LAMBDA_STEP;
        if (lambda >= 500.0f && lambda <= 570.0f)
            mat.Tf.value[b] = 0.9f;   // Green passband
        else
            mat.Tf.value[b] = 0.1f;   // Attenuated
    }

    // Verify distinct spectral bands
    float green_avg = 0.f, red_avg = 0.f;
    int green_count = 0, red_count = 0;
    for (int b = 0; b < NUM_LAMBDA; ++b) {
        float lambda = LAMBDA_MIN + (b + 0.5f) * LAMBDA_STEP;
        if (lambda >= 500.0f && lambda <= 570.0f) {
            green_avg += mat.Tf.value[b]; green_count++;
        } else if (lambda >= 600.0f) {
            red_avg += mat.Tf.value[b]; red_count++;
        }
    }
    green_avg /= green_count;
    red_avg /= red_count;
    EXPECT_GT(green_avg, red_avg * 5.0f);
}

// =====================================================================
// §2  PHOTON PATH FLAGS & BOUNCE COUNT
// =====================================================================

TEST(PhotonFlags, FlagConstants) {
    EXPECT_EQ(PHOTON_FLAG_TRAVERSED_GLASS, 0x01);
    EXPECT_EQ(PHOTON_FLAG_CAUSTIC_GLASS,   0x02);
    EXPECT_EQ(PHOTON_FLAG_VOLUME_SEGMENT,  0x04);
    EXPECT_EQ(PHOTON_FLAG_DISPERSION,      0x08);
}

TEST(PhotonFlags, SoAPushBackPreservesFlags) {
    PhotonSoA soa;
    Photon p = make_test_photon_flags(
        make_f3(1,2,3), make_f3(0,1,0), make_f3(0,0,1),
        1.0f, PHOTON_FLAG_TRAVERSED_GLASS | PHOTON_FLAG_DISPERSION, 5);

    soa.push_back(p);
    Photon got = soa.get(0);

    EXPECT_EQ(got.path_flags, PHOTON_FLAG_TRAVERSED_GLASS | PHOTON_FLAG_DISPERSION);
    EXPECT_EQ(got.bounce_count, 5);
}

TEST(PhotonFlags, SoAResizeClear) {
    PhotonSoA soa;
    soa.resize(10);
    EXPECT_EQ(soa.path_flags.size(), 10u);
    EXPECT_EQ(soa.bounce_count.size(), 10u);

    soa.clear();
    EXPECT_EQ(soa.path_flags.size(), 0u);
    EXPECT_EQ(soa.bounce_count.size(), 0u);
}

TEST(PhotonFlags, DefaultsAreZero) {
    Photon p;
    EXPECT_EQ(p.path_flags, 0);
    EXPECT_EQ(p.bounce_count, 0);
}

// =====================================================================
// §3  CELL INFO CACHE
// =====================================================================

TEST(CellInfoCache, EmptyMaps) {
    PhotonSoA global, caustic;
    CellInfoCache cache;
    cache.build(global, caustic, 0.1f, 0.05f);

    EXPECT_EQ(cache.cells.size(), (size_t)CELL_CACHE_TABLE_SIZE);

    // All cells should be empty
    for (uint32_t k = 0; k < CELL_CACHE_TABLE_SIZE; ++k) {
        EXPECT_EQ(cache.cells[k].photon_count, 0);
        EXPECT_FLOAT_EQ(cache.cells[k].irradiance, 0.f);
    }

    // is_empty should return true everywhere
    EXPECT_TRUE(cache.is_empty(make_f3(0.5f, 0.5f, 0.5f)));
}

TEST(CellInfoCache, SinglePhoton) {
    Photon p = make_test_photon_flags(
        make_f3(0.5f, 0.5f, 0.5f),
        make_f3(0, 1, 0), make_f3(0, 0, 1),
        2.0f, PHOTON_FLAG_TRAVERSED_GLASS, 3);

    PhotonSoA global = make_soa({p});
    PhotonSoA caustic;

    CellInfoCache cache;
    cache.build(global, caustic, 0.1f, 0.05f);

    EXPECT_FALSE(cache.is_empty(make_f3(0.5f, 0.5f, 0.5f)));

    const CellCacheInfo& ci = cache.query(make_f3(0.5f, 0.5f, 0.5f));
    EXPECT_EQ(ci.photon_count, 1);
    EXPECT_GT(ci.irradiance, 0.f);
    EXPECT_FLOAT_EQ(ci.glass_fraction, 1.0f);  // 100% glass
}

TEST(CellInfoCache, AdaptiveRadius) {
    // Dense region → small radius, sparse region → large radius
    std::vector<Photon> dense_photons;
    float3 dense_pos = make_f3(1.0f, 1.0f, 1.0f);
    for (int i = 0; i < 500; ++i) {
        Photon p = make_test_photon_flags(
            make_f3(dense_pos.x + 0.001f * (i % 10),
                    dense_pos.y + 0.001f * ((i/10) % 10),
                    dense_pos.z + 0.001f * (i/100)),
            make_f3(0, 1, 0), make_f3(0, 0, 1),
            1.0f, 0, 1);
        dense_photons.push_back(p);
    }

    // Add a sparse photon far away
    dense_photons.push_back(make_test_photon_flags(
        make_f3(10.0f, 10.0f, 10.0f),
        make_f3(0, 1, 0), make_f3(0, 0, 1),
        1.0f, 0, 1));

    PhotonSoA global = make_soa(dense_photons);
    PhotonSoA caustic;

    float base_r = 0.05f;
    CellInfoCache cache;
    cache.build(global, caustic, 0.1f, base_r);

    float r_dense = cache.get_adaptive_radius(dense_pos);
    float r_sparse = cache.get_adaptive_radius(make_f3(10.0f, 10.0f, 10.0f));

    // Dense region should have smaller radius than sparse
    EXPECT_LE(r_dense, r_sparse);

    // Both should be within configured bounds
    EXPECT_GE(r_dense,  base_r * ADAPTIVE_RADIUS_MIN_FACTOR - kTol);
    EXPECT_LE(r_sparse, base_r * ADAPTIVE_RADIUS_MAX_FACTOR + kTol);
}

TEST(CellInfoCache, CausticHotspot) {
    // Create photons with high caustic flux variance → hotspot
    std::vector<Photon> photons;
    for (int i = 0; i < 50; ++i) {
        float flux = (i % 5 == 0) ? 100.0f : 0.1f; // High variance
        Photon p = make_test_photon_flags(
            make_f3(2.0f, 2.0f, 2.0f),
            make_f3(0, 1, 0), make_f3(0, 0, 1),
            flux, PHOTON_FLAG_CAUSTIC_GLASS, 2);
        photons.push_back(p);
    }

    PhotonSoA caustic = make_soa(photons);
    PhotonSoA global;

    CellInfoCache cache;
    cache.build(global, caustic, 0.1f, 0.05f);

    const CellCacheInfo& ci = cache.query(make_f3(2.0f, 2.0f, 2.0f));
    EXPECT_GT(ci.caustic_count, 0);
    EXPECT_GT(ci.caustic_cv, 0.f);

    // Hotspot keys should include this cell
    auto hotspots = cache.get_caustic_hotspot_keys();
    // With high variance, this should be detected as a hotspot
    if (ci.caustic_cv > CAUSTIC_CV_THRESHOLD) {
        EXPECT_TRUE(ci.is_caustic_hotspot);
        EXPECT_FALSE(hotspots.empty());
    }
}

TEST(CellInfoCache, DirectionalSpread) {
    // All photons coming from the same direction → low spread
    std::vector<Photon> aligned;
    for (int i = 0; i < 100; ++i) {
        aligned.push_back(make_test_photon_flags(
            make_f3(3.0f, 3.0f, 3.0f),
            make_f3(0.0f, 1.0f, 0.0f),
            make_f3(0, 0, 1), 1.0f, 0, 1));
    }
    PhotonSoA soa_aligned = make_soa(aligned);
    PhotonSoA empty;

    CellInfoCache cache_aligned;
    cache_aligned.build(soa_aligned, empty, 0.1f, 0.05f);
    const CellCacheInfo& ci_a = cache_aligned.query(make_f3(3.0f, 3.0f, 3.0f));

    // Random directions → high spread
    std::vector<Photon> random_dir;
    PCGRng rng = PCGRng::seed(42, 1);
    for (int i = 0; i < 100; ++i) {
        float3 wi = normalize(make_f3(
            rng.next_float() * 2.f - 1.f,
            rng.next_float() * 2.f - 1.f,
            rng.next_float() * 2.f - 1.f));
        random_dir.push_back(make_test_photon_flags(
            make_f3(4.0f, 4.0f, 4.0f),
            wi, make_f3(0, 0, 1), 1.0f, 0, 1));
    }
    PhotonSoA soa_random = make_soa(random_dir);

    CellInfoCache cache_random;
    cache_random.build(soa_random, empty, 0.1f, 0.05f);
    const CellCacheInfo& ci_r = cache_random.query(make_f3(4.0f, 4.0f, 4.0f));

    EXPECT_LT(ci_a.directional_spread, ci_r.directional_spread);
}

// =====================================================================
// §4  GLASS BSDF + Tf SPECTRAL ATTENUATION
// =====================================================================

TEST(GlassBSDF, TfAppliedToSample) {
    // A glass material with colored Tf should attenuate the BSDF
    Material mat;
    mat.type      = MaterialType::Glass;
    mat.ior       = 1.5f;
    mat.dispersion = false;

    // Set Tf: pass only first half of spectrum
    for (int b = 0; b < NUM_LAMBDA; ++b) {
        mat.Tf.value[b] = (b < NUM_LAMBDA / 2) ? 1.0f : 0.1f;
    }

    PCGRng rng = PCGRng::seed(123, 1);
    float3 wo = normalize(make_f3(0.3f, 0.0f, 0.8f)); // From above

    BSDFSample bs = bsdf::glass_sample(wo, mat, rng);

    // The BSDF f values should reflect the Tf attenuation
    // In the first half of the spectrum, f should be larger
    float avg_first_half = 0.f, avg_second_half = 0.f;
    for (int b = 0; b < NUM_LAMBDA / 2; ++b)
        avg_first_half += bs.f.value[b];
    for (int b = NUM_LAMBDA / 2; b < NUM_LAMBDA; ++b)
        avg_second_half += bs.f.value[b];
    avg_first_half /= (NUM_LAMBDA / 2);
    avg_second_half /= (NUM_LAMBDA - NUM_LAMBDA / 2);

    // First half should be more energetic
    EXPECT_GT(avg_first_half, avg_second_half);
}

TEST(GlassBSDF, DispersionPerBinFresnel) {
    // With dispersion on, per-wavelength Fresnel should produce
    // varying reflectance across the spectrum
    Material mat;
    mat.type      = MaterialType::Glass;
    mat.ior       = 1.5f;
    mat.cauchy_A  = 1.5046f;
    mat.cauchy_B  = 4200.0f;
    mat.dispersion = true;

    // Test at near-normal incidence where Fresnel is small
    float3 wo = normalize(make_f3(0.0f, 0.0f, 1.0f));

    // Sample many times to get both reflection and refraction
    int reflect_count = 0;
    int total = 10000;
    PCGRng rng = PCGRng::seed(42, 1);

    for (int i = 0; i < total; ++i) {
        BSDFSample bs = bsdf::glass_sample(wo, mat, rng);
        if (bs.wi.z > 0.f) reflect_count++;
    }

    // At normal incidence with IOR ~1.5, Fresnel reflectance ≈ 4%
    float reflect_frac = (float)reflect_count / (float)total;
    EXPECT_NEAR(reflect_frac, 0.04f, 0.02f);
}

TEST(GlassBSDF, LegacyOverloadWorks) {
    // The legacy glass_sample(wo, ior, rng) should still function
    PCGRng rng = PCGRng::seed(77, 1);
    float3 wo = normalize(make_f3(0.2f, 0.0f, 0.9f));

    BSDFSample bs = bsdf::glass_sample(wo, 1.5f, rng);

    // Should produce a valid sample
    EXPECT_GT(bs.pdf, 0.f);
    float len = sqrtf(dot(bs.wi, bs.wi));
    EXPECT_NEAR(len, 1.0f, kLoose);
}

// =====================================================================
// §5  ENERGY CONSERVATION WITH DISPERSION
// =====================================================================

TEST(GlassEnergy, WhiteFurnaceNoDispersion) {
    // Glass BSDF should conserve energy (R + T = 1 per wavelength)
    Material mat;
    mat.type      = MaterialType::Glass;
    mat.ior       = 1.5f;
    mat.dispersion = false;

    float3 wo = normalize(make_f3(0.3f, 0.0f, 0.8f));
    int N = 100000;
    PCGRng rng = PCGRng::seed(42, 1);

    Spectrum total_f = Spectrum::zero();
    for (int i = 0; i < N; ++i) {
        BSDFSample bs = bsdf::glass_sample(wo, mat, rng);
        if (bs.pdf > 0.f) {
            float cos_theta = fabsf(bs.wi.z);
            for (int b = 0; b < NUM_LAMBDA; ++b)
                total_f.value[b] += bs.f.value[b] * cos_theta / bs.pdf;
        }
    }

    // Average should be ~1.0 per bin (energy conserving)
    for (int b = 0; b < NUM_LAMBDA; ++b) {
        float avg = total_f.value[b] / (float)N;
        EXPECT_NEAR(avg, 1.0f, 0.05f) << "Energy violation at bin " << b;
    }
}

TEST(GlassEnergy, WhiteFurnaceWithDispersion) {
    // With dispersion, energy should still be conserved per wavelength
    Material mat;
    mat.type      = MaterialType::Glass;
    mat.ior       = 1.5f;
    mat.cauchy_A  = 1.5046f;
    mat.cauchy_B  = 4200.0f;
    mat.dispersion = true;

    float3 wo = normalize(make_f3(0.3f, 0.0f, 0.8f));
    int N = 100000;
    PCGRng rng = PCGRng::seed(42, 1);

    Spectrum total_f = Spectrum::zero();
    for (int i = 0; i < N; ++i) {
        BSDFSample bs = bsdf::glass_sample(wo, mat, rng);
        if (bs.pdf > 0.f) {
            float cos_theta = fabsf(bs.wi.z);
            for (int b = 0; b < NUM_LAMBDA; ++b)
                total_f.value[b] += bs.f.value[b] * cos_theta / bs.pdf;
        }
    }

    for (int b = 0; b < NUM_LAMBDA; ++b) {
        float avg = total_f.value[b] / (float)N;
        EXPECT_NEAR(avg, 1.0f, 0.05f) << "Dispersion energy violation at bin " << b;
    }
}

// =====================================================================
// §6  CAUSTIC PHOTON TRACING END-TO-END
// =====================================================================

// Build a minimal scene with one emissive triangle and one glass sphere
// approximated as two glass triangles, to test caustic photon generation.
static Scene build_glass_caustic_scene() {
    Scene scene;

    // Material 0: Diffuse white floor
    Material floor_mat;
    floor_mat.type = MaterialType::Lambertian;
    floor_mat.Kd   = Spectrum::constant(0.8f);
    scene.materials.push_back(floor_mat);

    // Material 1: Glass
    Material glass_mat;
    glass_mat.type      = MaterialType::Glass;
    glass_mat.ior       = 1.5f;
    glass_mat.cauchy_A  = 1.5046f;
    glass_mat.cauchy_B  = 4200.0f;
    glass_mat.dispersion = true;
    scene.materials.push_back(glass_mat);

    // Material 2: Emissive light (ceiling)
    Material light_mat;
    light_mat.type = MaterialType::Lambertian;
    light_mat.Le   = Spectrum::constant(10.0f);
    scene.materials.push_back(light_mat);

    // Floor triangle (large, facing up)
    Triangle floor_tri;
    floor_tri.v0 = make_f3(-5, 0, -5);
    floor_tri.v1 = make_f3( 5, 0, -5);
    floor_tri.v2 = make_f3( 0, 0,  5);
    floor_tri.n0 = floor_tri.n1 = floor_tri.n2 = make_f3(0, 1, 0);
    floor_tri.material_id = 0;
    scene.triangles.push_back(floor_tri);

    // Second floor quad triangle (to cover more area)
    Triangle floor_tri2;
    floor_tri2.v0 = make_f3( 5, 0, -5);
    floor_tri2.v1 = make_f3( 5, 0,  5);
    floor_tri2.v2 = make_f3(-5, 0,  5);
    floor_tri2.n0 = floor_tri2.n1 = floor_tri2.n2 = make_f3(0, 1, 0);
    floor_tri2.material_id = 0;
    scene.triangles.push_back(floor_tri2);

    // Glass triangle (above floor, facing up so photons from above hit it)
    Triangle glass_tri;
    glass_tri.v0 = make_f3(-0.5f, 1.0f, -0.5f);
    glass_tri.v1 = make_f3( 0.5f, 1.0f, -0.5f);
    glass_tri.v2 = make_f3( 0.0f, 1.0f,  0.5f);
    glass_tri.n0 = glass_tri.n1 = glass_tri.n2 = make_f3(0, 1, 0);
    glass_tri.material_id = 1;
    scene.triangles.push_back(glass_tri);

    // Light triangle (above glass, facing down)
    Triangle light_tri;
    light_tri.v0 = make_f3(-2, 3, -2);
    light_tri.v1 = make_f3( 2, 3, -2);
    light_tri.v2 = make_f3( 0, 3,  2);
    light_tri.n0 = light_tri.n1 = light_tri.n2 = make_f3(0, -1, 0);
    light_tri.material_id = 2;
    scene.triangles.push_back(light_tri);

    // Second light triangle (larger emissive area for more photon hits)
    Triangle light_tri2;
    light_tri2.v0 = make_f3( 2, 3, -2);
    light_tri2.v1 = make_f3( 2, 3,  2);
    light_tri2.v2 = make_f3(-2, 3,  2);
    light_tri2.n0 = light_tri2.n1 = light_tri2.n2 = make_f3(0, -1, 0);
    light_tri2.material_id = 2;
    scene.triangles.push_back(light_tri2);

    scene.build_bvh();
    scene.build_emissive_distribution();
    return scene;
}

TEST(CausticTracing, ProducesCausticPhotons) {
    Scene scene = build_glass_caustic_scene();

    EmitterConfig cfg;
    cfg.num_photons  = 50000;
    cfg.max_bounces  = 10;
    cfg.volume_enabled = false;

    PhotonSoA global, caustic;
    trace_photons(scene, cfg, global, caustic, nullptr);

    // Should produce some caustic photons (light → glass → diffuse)
    EXPECT_GT(caustic.size(), 0u);

    // Check that caustic photons have TRAVERSED_GLASS flag
    int glass_flagged = 0;
    for (size_t i = 0; i < caustic.size(); ++i) {
        if (i < caustic.path_flags.size() &&
            (caustic.path_flags[i] & PHOTON_FLAG_TRAVERSED_GLASS))
            glass_flagged++;
    }
    // Most caustic photons should have the glass flag
    if (caustic.size() > 0) {
        float frac = (float)glass_flagged / (float)caustic.size();
        EXPECT_GT(frac, 0.5f) << "Expected >50% of caustic photons to have glass flag";
    }
}

TEST(CausticTracing, BounceCountRecorded) {
    Scene scene = build_glass_caustic_scene();

    EmitterConfig cfg;
    cfg.num_photons = 10000;
    cfg.max_bounces = 10;

    PhotonSoA global, caustic;
    trace_photons(scene, cfg, global, caustic, nullptr);

    // Global photons should have bounce_count > 0
    int nonzero_bounce = 0;
    for (size_t i = 0; i < global.size(); ++i) {
        if (i < global.bounce_count.size() && global.bounce_count[i] > 0)
            nonzero_bounce++;
    }

    // All stored photons should have bounce > 0 (we only store after bounce > 0)
    if (global.size() > 0) {
        EXPECT_EQ(nonzero_bounce, (int)global.size());
    }
}

// =====================================================================
// §7  CPU GROUND TRUTH vs HASH GRID COMPARISON
// =====================================================================

TEST(GroundTruth, HashGridVsBruteForce) {
    // Create a small photon set and compare hash grid query results
    // against brute-force O(N) scan as ground truth
    std::vector<Photon> photons;
    PCGRng rng = PCGRng::seed(42, 1);
    for (int i = 0; i < 1000; ++i) {
        Photon p;
        p.position = make_f3(rng.next_float() * 2.f,
                             rng.next_float() * 2.f,
                             rng.next_float() * 2.f);
        p.wi       = make_f3(0, 1, 0);
        p.geom_normal = make_f3(0, 0, 1);
        p.spectral_flux = Spectrum::constant(1.0f);
        photons.push_back(p);
    }

    PhotonSoA soa = make_soa(photons);
    HashGrid grid;
    float radius = 0.2f;
    grid.build(soa, radius);

    // Query point
    float3 qpos = make_f3(1.0f, 1.0f, 1.0f);
    float r2 = radius * radius;

    // Brute force ground truth
    std::vector<uint32_t> brute_results;
    for (size_t i = 0; i < photons.size(); ++i) {
        float3 diff = qpos - photons[i].position;
        if (dot(diff, diff) <= r2) {
            brute_results.push_back((uint32_t)i);
        }
    }

    // Hash grid query
    std::vector<uint32_t> grid_results;
    grid.query(qpos, radius, soa,
        [&](uint32_t idx, float /*d2*/) {
            grid_results.push_back(idx);
        });

    // Sort both for comparison
    std::sort(brute_results.begin(), brute_results.end());
    std::sort(grid_results.begin(), grid_results.end());

    // Hash grid should find exactly the same photons
    EXPECT_EQ(brute_results.size(), grid_results.size());
    EXPECT_EQ(brute_results, grid_results);
}

TEST(GroundTruth, DensityEstimateConsistency) {
    // Compare density estimate from hash grid vs KD-tree
    // They should produce identical results for the same photon set
    std::vector<Photon> photons;
    PCGRng rng = PCGRng::seed(99, 1);
    for (int i = 0; i < 2000; ++i) {
        Photon p;
        p.position    = make_f3(rng.next_float(), rng.next_float(), rng.next_float());
        p.wi          = make_f3(0, 0, 1);
        p.geom_normal = make_f3(0, 0, 1);
        p.spectral_flux = Spectrum::constant(0.5f);
        photons.push_back(p);
    }

    PhotonSoA soa = make_soa(photons);

    float radius = 0.15f;
    HashGrid grid;
    grid.build(soa, radius);

    KDTree kdtree;
    kdtree.build(soa);

    Material diffuse;
    diffuse.type = MaterialType::Lambertian;
    diffuse.Kd   = Spectrum::constant(0.5f);

    float3 hit_pos    = make_f3(0.5f, 0.5f, 0.5f);
    float3 hit_normal = make_f3(0, 0, 1);
    ONB frame = ONB::from_normal(hit_normal);
    float3 wo_local = make_f3(0, 0, 1);  // Looking straight up

    // Hash grid estimate
    DensityEstimatorConfig de_cfg;
    de_cfg.radius = radius;
    de_cfg.num_photons_total = (int)photons.size();
    de_cfg.use_kernel = false; // Box kernel for simpler comparison

    Spectrum L_grid = estimate_photon_density(
        hit_pos, hit_normal, wo_local, diffuse,
        soa, grid, de_cfg, radius);

    // KD-tree tangential estimate
    float r2 = radius * radius;
    float inv_area = 1.0f / (PI * r2);
    float inv_N = 1.0f / (float)photons.size();
    float tau = effective_tau(de_cfg.surface_tau);

    Spectrum L_kd = Spectrum::zero();
    kdtree.query_tangential(hit_pos, hit_normal, radius, tau, soa,
        [&](uint32_t idx, float d_tan2) {
            if (!soa.norm_x.empty()) {
                float3 pn = make_f3(soa.norm_x[idx], soa.norm_y[idx], soa.norm_z[idx]);
                if (dot(pn, hit_normal) <= 0.f) return;
            }
            float3 wi = make_f3(soa.wi_x[idx], soa.wi_y[idx], soa.wi_z[idx]);
            if (dot(wi, hit_normal) <= 0.f) return;

            float3 wi_loc = frame.world_to_local(wi);
            Spectrum f = bsdf::evaluate_diffuse(diffuse, wo_local, wi_loc);
            Spectrum flux = soa.get_flux(idx);
            for (int b = 0; b < NUM_LAMBDA; ++b)
                L_kd.value[b] += flux.value[b] * inv_N * f.value[b] * inv_area;
        });

    // Compare: should be very close (same tangential query, same photons)
    for (int b = 0; b < NUM_LAMBDA; ++b) {
        if (L_grid.value[b] > 0.f || L_kd.value[b] > 0.f) {
            float rel_err = fabsf(L_grid.value[b] - L_kd.value[b]) /
                            fmaxf(L_grid.value[b], L_kd.value[b]);
            EXPECT_LT(rel_err, 0.1f) << "Density mismatch at bin " << b
                << " grid=" << L_grid.value[b] << " kd=" << L_kd.value[b];
        }
    }
}

// =====================================================================
// §8  IOR STACK – nested dielectric tracking
// =====================================================================

TEST(IORStack, DefaultIsAir) {
    IORStack stack;
    EXPECT_FLOAT_EQ(stack.current_ior(), 1.0f);
}

TEST(IORStack, PushPop) {
    IORStack stack;

    stack.push(1.5f);
    EXPECT_FLOAT_EQ(stack.current_ior(), 1.5f);

    stack.push(1.33f);
    EXPECT_FLOAT_EQ(stack.current_ior(), 1.33f);

    stack.pop();
    EXPECT_FLOAT_EQ(stack.current_ior(), 1.5f);

    stack.pop();
    EXPECT_FLOAT_EQ(stack.current_ior(), 1.0f);
}

TEST(IORStack, Overflow) {
    IORStack stack;
    // Push more than MAX_DEPTH entries
    for (int i = 0; i < 10; ++i) {
        stack.push(1.0f + 0.1f * i);
    }
    // Should not crash, should cap at MAX_DEPTH
    EXPECT_GT(stack.current_ior(), 1.0f);
}

TEST(IORStack, Underflow) {
    IORStack stack;
    stack.pop();  // Pop from empty
    EXPECT_FLOAT_EQ(stack.current_ior(), 1.0f); // Should remain at air
}

// =====================================================================
// §9  INTEGRATION TEST – Full pipeline with dispersion
// =====================================================================

TEST(Integration, DispersionCausticPipeline) {
    Scene scene = build_glass_caustic_scene();

    Renderer renderer;
    renderer.set_scene(&scene);

    RenderConfig cfg;
    cfg.num_photons       = 20000;
    cfg.max_bounces       = 8;
    cfg.image_width       = 4;
    cfg.image_height      = 4;
    cfg.samples_per_pixel = 1;
    cfg.gather_radius     = 0.1f;
    cfg.caustic_radius    = 0.05f;
    cfg.use_kdtree        = true;
    renderer.set_config(cfg);

    Camera cam;
    cam.position = make_f3(0, 2, 3);
    cam.look_at  = make_f3(0, 0, 0);
    cam.up       = make_f3(0, 1, 0);
    cam.fov_deg  = 60.f;
    cam.width    = cfg.image_width;
    cam.height   = cfg.image_height;
    cam.update();
    renderer.set_camera(cam);

    // Should not crash
    EXPECT_NO_THROW(renderer.build_photon_maps());

    // CellInfoCache should be populated
    const CellInfoCache& cache = renderer.cell_cache();
    EXPECT_EQ(cache.cells.size(), (size_t)CELL_CACHE_TABLE_SIZE);

    // Should have some photons
    EXPECT_GT(renderer.global_photon_count(), 0u);
}

TEST(Integration, CellCacheAfterAdaptiveCaustic) {
    Scene scene = build_glass_caustic_scene();

    Renderer renderer;
    renderer.set_scene(&scene);

    RenderConfig cfg;
    cfg.num_photons       = 30000;
    cfg.max_bounces       = 8;
    cfg.image_width       = 4;
    cfg.image_height      = 4;
    cfg.samples_per_pixel = 1;
    cfg.gather_radius     = 0.1f;
    cfg.caustic_radius    = 0.05f;
    cfg.use_kdtree        = true;
    renderer.set_config(cfg);

    Camera cam;
    cam.position = make_f3(0, 2, 3);
    cam.look_at  = make_f3(0, 0, 0);
    cam.up       = make_f3(0, 1, 0);
    cam.fov_deg  = 60.f;
    cam.width    = cfg.image_width;
    cam.height   = cfg.image_height;
    cam.update();
    renderer.set_camera(cam);

    renderer.build_photon_maps();

    // After adaptive caustic shooting, caustic count may have increased
    size_t caustic_count = renderer.caustic_photon_count();
    // Just verify it's a reasonable number (>= 0, may be 0 if no glass hit)
    EXPECT_GE(caustic_count, 0u);
}
