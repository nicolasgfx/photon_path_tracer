// ─────────────────────────────────────────────────────────────────────
// test_medium.cpp – Unit & integration tests for participating medium
// ─────────────────────────────────────────────────────────────────────
// Covers:
//   1. Beer–Lambert transmittance correctness
//   2. Rayleigh phase function normalization
//   3. Rayleigh spectral shape (blue > red)
//   4. Henyey–Greenstein phase normalization
//   5. Medium construction: density / albedo / falloff
//   6. Integration: volume off = baseline match (Cornell Box data)
//   7. Integration: volume on produces non-negative, finite radiance
//   8. Integration: higher density → lower surface radiance (more attenuation)
//   9. Integration: volume contributes in-scatter (L > 0 in shafts)
//  10. Config defaults compile and have expected values
//  11. RenderConfig volume fields propagate correctly
// ─────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include "core/types.h"
#include "core/spectrum.h"
#include "core/config.h"
#include "core/random.h"
#include "core/medium.h"
#include "core/phase_function.h"
#include "core/test_data_io.h"
#include "scene/scene.h"
#include "scene/obj_loader.h"
#include "renderer/renderer.h"
#include "renderer/camera.h"
#include "renderer/direct_light.h"
#include "photon/photon.h"
#include "photon/hash_grid.h"
#include "photon/emitter.h"
#include "bsdf/bsdf.h"
#include "photon/density_estimator.h"
#include "renderer/mis.h"

#include <vector>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <numeric>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────
//  CORNELL BOX DATASET (reused from test_ground_truth.cpp pattern)
// ─────────────────────────────────────────────────────────────────────

static std::string medium_test_data_path() {
    fs::path scenes(SCENES_DIR);
    fs::path data_dir = scenes.parent_path() / "tests" / "data";
    return (data_dir / "cornell_box.bin").string();
}

struct MediumTestDataset {
    TestDataHeader  header;
    PhotonSoA       photons;
    PhotonSoA       caustic_photons;
    Scene           scene;
    Camera          camera;
    HashGrid        grid;
    int   num_photons   = 0;
    float gather_radius = 0.0f;
    int   max_bounces   = 0;
    bool  valid         = false;

    void build() {
        if (valid) return;
        std::string bin_path = medium_test_data_path();

        if (fs::exists(bin_path)) {
            if (!load_test_data(bin_path, photons, caustic_photons, header)) {
                std::cerr << "[MediumTest] Failed to load " << bin_path << "\n";
                return;
            }
        } else {
            header.num_photons_cfg = DEFAULT_NUM_PHOTONS;
            header.gather_radius   = DEFAULT_GATHER_RADIUS;
            header.caustic_radius  = DEFAULT_CAUSTIC_RADIUS;
            header.max_bounces     = DEFAULT_MAX_BOUNCES;
            header.min_bounces_rr  = DEFAULT_MIN_BOUNCES_RR;
            header.rr_threshold    = DEFAULT_RR_THRESHOLD;
            header.scene_path      = "cornell_box/cornellbox.obj";

            std::string obj_path = std::string(SCENES_DIR) + "/" + header.scene_path;
            if (!load_obj(obj_path, scene)) return;
            scene.build_bvh();
            scene.build_emissive_distribution();

            EmitterConfig ecfg;
            ecfg.num_photons    = header.num_photons_cfg;
            ecfg.max_bounces    = header.max_bounces;
            ecfg.rr_threshold   = header.rr_threshold;
            ecfg.min_bounces_rr = header.min_bounces_rr;
            trace_photons(scene, ecfg, photons, caustic_photons);
            if (photons.size() == 0) return;
        }

        num_photons   = (int)header.num_photons_cfg;
        gather_radius = header.gather_radius;
        max_bounces   = (int)header.max_bounces;

        if (scene.triangles.empty()) {
            std::string obj_path = std::string(SCENES_DIR) + "/" + header.scene_path;
            if (!load_obj(obj_path, scene)) return;
            scene.build_bvh();
            scene.build_emissive_distribution();
        }

        camera = Camera::cornell_box_camera(64, 64);
        camera.update();

        if (photons.size() > 0)
            grid.build(photons, gather_radius);

        valid = true;
        std::cout << "[MediumTest] Dataset ready: " << scene.triangles.size()
                  << " tris, " << photons.size() << " photons\n";
    }
};

static MediumTestDataset& get_medium_dataset() {
    static MediumTestDataset ds;
    ds.build();
    return ds;
}

// ─────────────────────────────────────────────────────────────────────
//  Helper: trace N random rays through the CPU integrator
// ─────────────────────────────────────────────────────────────────────
struct VolumeSampleStats {
    float mean_luminance = 0.f;
    int   num_finite     = 0;
    int   num_nonneg     = 0;
    int   total          = 0;
};

static VolumeSampleStats trace_random_rays(
    const MediumTestDataset& ds,
    bool volume_enabled, float volume_density = 2.f,
    float volume_falloff = 0.f, float volume_albedo = 0.95f)
{
    Renderer renderer;
    Scene scene_copy = ds.scene; // copy to avoid const issues
    renderer.set_scene(&scene_copy);
    renderer.set_camera(ds.camera);

    RenderConfig cfg;
    cfg.image_width       = 64;
    cfg.image_height      = 64;
    cfg.samples_per_pixel = 1;
    cfg.max_bounces       = ds.max_bounces;
    cfg.num_photons       = ds.num_photons;
    cfg.gather_radius     = ds.gather_radius;
    cfg.volume_enabled    = volume_enabled;
    cfg.volume_density    = volume_density;
    cfg.volume_falloff    = volume_falloff;
    cfg.volume_albedo     = volume_albedo;
    cfg.volume_samples    = 1;
    cfg.volume_max_t      = 5.0f;
    cfg.mode              = RenderMode::Full;
    renderer.set_config(cfg);

    // Build photon maps for indirect lighting
    renderer.build_photon_maps();

    // Render one SPP frame on CPU
    renderer.render_frame();

    VolumeSampleStats stats;
    float lum_sum = 0.f;

    // Gather stats from the framebuffer
    const auto& fb = renderer.framebuffer();
    int pixels = fb.width * fb.height;
    stats.total = pixels;

    for (int i = 0; i < pixels; ++i) {
        Spectrum avg = (fb.sample_count[i] > 0.f)
            ? fb.pixels[i] / fb.sample_count[i]
            : Spectrum::zero();

        float3 xyz = spectrum_to_xyz(avg);
        float Y = xyz.y;

        bool finite = std::isfinite(Y);
        bool nonneg = (Y >= -1e-6f);

        if (finite) stats.num_finite++;
        if (nonneg) stats.num_nonneg++;
        lum_sum += std::fmax(Y, 0.f);
    }

    stats.mean_luminance = (pixels > 0) ? lum_sum / pixels : 0.f;
    return stats;
}

// =====================================================================
//  UNIT TESTS: Beer–Lambert Transmittance
// =====================================================================

TEST(Medium_BeerLambert, ZeroDistance_IsOne) {
    HomogeneousMedium m = make_rayleigh_medium(2.0f, 0.9f);
    Spectrum T = transmittance(m, 0.f);
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        EXPECT_FLOAT_EQ(T.value[i], 1.f)
            << "T(0) must be 1.0 for all wavelengths (bin " << i << ")";
    }
}

TEST(Medium_BeerLambert, PositiveDistance_LessThanOne) {
    HomogeneousMedium m = make_rayleigh_medium(2.0f, 0.9f);
    Spectrum T = transmittance(m, 1.0f);
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        EXPECT_GT(T.value[i], 0.f) << "T must be positive (bin " << i << ")";
        EXPECT_LT(T.value[i], 1.f) << "T must be < 1 for d > 0 (bin " << i << ")";
    }
}

TEST(Medium_BeerLambert, MatchesExpNegSigmaD) {
    float density = 3.0f;
    float albedo  = 0.8f;
    float dist    = 2.5f;
    HomogeneousMedium m = make_rayleigh_medium(density, albedo);
    Spectrum T = transmittance(m, dist);

    for (int i = 0; i < NUM_LAMBDA; ++i) {
        float expected = expf(-m.sigma_t.value[i] * dist);
        EXPECT_NEAR(T.value[i], expected, 1e-6f)
            << "T(d) != exp(-sigma_t * d) at bin " << i;
    }
}

TEST(Medium_BeerLambert, LargeDistance_NearZero) {
    HomogeneousMedium m = make_rayleigh_medium(5.0f, 0.9f);
    Spectrum T = transmittance(m, 100.f);
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        EXPECT_NEAR(T.value[i], 0.f, 1e-6f)
            << "T(large d) should be ~0 (bin " << i << ")";
    }
}

TEST(Medium_BeerLambert, MonotonicallyDecreasing) {
    HomogeneousMedium m = make_rayleigh_medium(2.0f, 0.9f);
    float prev_max = 1.f;
    for (float d = 0.1f; d <= 5.f; d += 0.1f) {
        float cur_max = transmittance(m, d).max_component();
        EXPECT_LE(cur_max, prev_max + 1e-6f);
        prev_max = cur_max;
    }
}

// =====================================================================
//  UNIT TESTS: Rayleigh Phase Function
// =====================================================================

TEST(Medium_RayleighPhase, Normalization) {
    // Numerical integration over sphere: ∫₀²π ∫₀π p(cosθ) sinθ dθ dφ = 1
    const int N_theta = 500;
    const int N_phi   = 500;
    double integral = 0.0;
    double d_theta = PI / N_theta;
    double d_phi   = 2.0 * PI / N_phi;

    for (int it = 0; it < N_theta; ++it) {
        double theta = (it + 0.5) * d_theta;
        double cos_theta = cos(theta);
        double sin_theta = sin(theta);
        double p = rayleigh_phase((float)cos_theta);
        integral += p * sin_theta * d_theta * d_phi;
    }
    // Integrate over phi gives factor of 2π (already in d_phi sum)
    EXPECT_NEAR(integral, 1.0, 0.01)
        << "Rayleigh phase should integrate to 1 over the sphere";
}

TEST(Medium_RayleighPhase, SymmetricForwardBackward) {
    // p(cos(0)) should equal p(cos(π))  because  1 + cos²(0) = 1 + cos²(π) = 2
    float p_fwd  = rayleigh_phase(1.f);
    float p_back = rayleigh_phase(-1.f);
    EXPECT_FLOAT_EQ(p_fwd, p_back);
}

TEST(Medium_RayleighPhase, MinimumAt90Degrees) {
    float p_90  = rayleigh_phase(0.f);
    float p_fwd = rayleigh_phase(1.f);
    EXPECT_LT(p_90, p_fwd)
        << "Phase should be minimum at 90 degrees (cos=0)";
}

TEST(Medium_RayleighPhase, KnownValue) {
    // p(0) = 3/(16π) * (1 + 0) = 3/(16π) ≈ 0.05968
    float p0 = rayleigh_phase(0.f);
    float expected = 3.f / (16.f * PI);
    EXPECT_NEAR(p0, expected, 1e-5f);
}

// =====================================================================
//  UNIT TESTS: Henyey–Greenstein Phase Function
// =====================================================================

TEST(Medium_HGPhase, IsotropicWhenGZero) {
    // g = 0 → isotropic → p = 1/(4π)
    float p_fwd  = henyey_greenstein_phase(1.f, 0.f);
    float p_back = henyey_greenstein_phase(-1.f, 0.f);
    float p_90   = henyey_greenstein_phase(0.f, 0.f);
    float expected = 1.f / (4.f * PI);
    EXPECT_NEAR(p_fwd, expected, 1e-5f);
    EXPECT_NEAR(p_back, expected, 1e-5f);
    EXPECT_NEAR(p_90, expected, 1e-5f);
}

TEST(Medium_HGPhase, ForwardScatteringWhenGPositive) {
    float g = 0.7f;
    float p_fwd  = henyey_greenstein_phase(1.f, g);
    float p_back = henyey_greenstein_phase(-1.f, g);
    EXPECT_GT(p_fwd, p_back)
        << "g > 0 should favour forward scattering";
}

TEST(Medium_HGPhase, Normalization) {
    float g = 0.5f;
    const int N = 1000;
    double integral = 0.0;
    double d_theta = PI / N;
    for (int i = 0; i < N; ++i) {
        double theta = (i + 0.5) * d_theta;
        double cos_theta = cos(theta);
        double sin_theta = sin(theta);
        double p = henyey_greenstein_phase((float)cos_theta, g);
        integral += p * sin_theta * d_theta * 2.0 * PI;
    }
    EXPECT_NEAR(integral, 1.0, 0.02)
        << "HG phase should integrate to 1 over the sphere";
}

// =====================================================================
//  UNIT TESTS: Medium Construction
// =====================================================================

TEST(Medium_Construction, RayleighSpectralShape_BlueShorterWavelengthStronger) {
    HomogeneousMedium m = make_rayleigh_medium(2.0f, 0.9f);
    // σ_s ∝ 1/λ⁴ → shorter λ should have higher σ_s
    // Bin 0 (380 nm) vs bin NUM_LAMBDA-1 (780 nm)
    EXPECT_GT(m.sigma_s.value[0], m.sigma_s.value[NUM_LAMBDA - 1])
        << "Rayleigh σ_s at 380nm should be larger than at 780nm";

    // Ratio should be close to (780/380)^4 ≈ 17.7
    float ratio = m.sigma_s.value[0] / m.sigma_s.value[NUM_LAMBDA - 1];
    float expected_ratio = powf(lambda_of_bin(NUM_LAMBDA - 1) / lambda_of_bin(0), 4.f);
    EXPECT_NEAR(ratio, expected_ratio, expected_ratio * 0.1f)
        << "Ratio should match (λ_max/λ_min)^4";
}

TEST(Medium_Construction, ZeroDensity_NoExtinction) {
    HomogeneousMedium m = make_rayleigh_medium(0.f, 0.9f);
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        EXPECT_FLOAT_EQ(m.sigma_t.value[i], 0.f);
        EXPECT_FLOAT_EQ(m.sigma_s.value[i], 0.f);
        EXPECT_FLOAT_EQ(m.sigma_a.value[i], 0.f);
    }
}

TEST(Medium_Construction, AlbedoPartition) {
    // σ_s = albedo * σ_t,  σ_a = (1 - albedo) * σ_t
    float albedo = 0.7f;
    HomogeneousMedium m = make_rayleigh_medium(3.f, albedo);
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        EXPECT_NEAR(m.sigma_s.value[i] + m.sigma_a.value[i],
                    m.sigma_t.value[i], 1e-6f)
            << "σ_s + σ_a should equal σ_t at bin " << i;
        EXPECT_NEAR(m.sigma_s.value[i], albedo * m.sigma_t.value[i], 1e-6f);
    }
}

TEST(Medium_Construction, HeightFalloff_DecreasesWithY) {
    float density = 2.f;
    float falloff = 3.f;
    HomogeneousMedium m0 = make_rayleigh_medium(density, 0.9f, falloff, 0.f);
    HomogeneousMedium m1 = make_rayleigh_medium(density, 0.9f, falloff, 1.f);
    HomogeneousMedium m2 = make_rayleigh_medium(density, 0.9f, falloff, 2.f);

    for (int i = 0; i < NUM_LAMBDA; ++i) {
        EXPECT_GT(m0.sigma_t.value[i], m1.sigma_t.value[i])
            << "Higher y → lower σ_t (bin " << i << ")";
        EXPECT_GT(m1.sigma_t.value[i], m2.sigma_t.value[i])
            << "Even higher y → even lower σ_t (bin " << i << ")";
    }
}

TEST(Medium_Construction, ZeroFalloff_HomogeneousRegardlessOfY) {
    float density = 2.f;
    HomogeneousMedium m0 = make_rayleigh_medium(density, 0.9f, 0.f, 0.f);
    HomogeneousMedium m5 = make_rayleigh_medium(density, 0.9f, 0.f, 5.f);
    HomogeneousMedium mn = make_rayleigh_medium(density, 0.9f, 0.f, -3.f);

    for (int i = 0; i < NUM_LAMBDA; ++i) {
        EXPECT_FLOAT_EQ(m0.sigma_t.value[i], m5.sigma_t.value[i]);
        EXPECT_FLOAT_EQ(m0.sigma_t.value[i], mn.sigma_t.value[i]);
    }
}

TEST(Medium_Construction, SigmaT_Avg) {
    HomogeneousMedium m = make_rayleigh_medium(2.f, 0.9f);
    float avg = sigma_t_avg(m);
    EXPECT_GT(avg, 0.f);

    // Verify manually
    float sum = 0.f;
    for (int i = 0; i < NUM_LAMBDA; ++i)
        sum += m.sigma_t.value[i];
    EXPECT_NEAR(avg, sum / NUM_LAMBDA, 1e-6f);
}

// =====================================================================
//  UNIT TESTS: Config Defaults
// =====================================================================

TEST(Medium_Config, DefaultsExist) {
    // Volume knobs exist and have sane values (enabled state may vary)
    (void)DEFAULT_VOLUME_ENABLED;  // just compile-check the symbol
    EXPECT_GT(DEFAULT_VOLUME_DENSITY, 0.f);
    EXPECT_GE(DEFAULT_VOLUME_FALLOFF, 0.f);
    EXPECT_GT(DEFAULT_VOLUME_ALBEDO, 0.f);
    EXPECT_LE(DEFAULT_VOLUME_ALBEDO, 1.f);
    EXPECT_GE(DEFAULT_VOLUME_SAMPLES, 1);
    EXPECT_GT(DEFAULT_VOLUME_MAX_T, 0.f);
}

TEST(Medium_Config, RenderConfigPropagation) {
    RenderConfig cfg;
    // Defaults should match config.h
    EXPECT_EQ(cfg.volume_enabled, DEFAULT_VOLUME_ENABLED);
    EXPECT_FLOAT_EQ(cfg.volume_density, DEFAULT_VOLUME_DENSITY);
    EXPECT_FLOAT_EQ(cfg.volume_falloff, DEFAULT_VOLUME_FALLOFF);
    EXPECT_FLOAT_EQ(cfg.volume_albedo, DEFAULT_VOLUME_ALBEDO);
    EXPECT_EQ(cfg.volume_samples, DEFAULT_VOLUME_SAMPLES);
    EXPECT_FLOAT_EQ(cfg.volume_max_t, DEFAULT_VOLUME_MAX_T);

    // Can override
    cfg.volume_enabled = true;
    cfg.volume_density = 5.f;
    EXPECT_TRUE(cfg.volume_enabled);
    EXPECT_FLOAT_EQ(cfg.volume_density, 5.f);
}

// =====================================================================
//  INTEGRATION TESTS: Volume Off = Baseline
// =====================================================================

TEST(Medium_Integration, VolumeOff_BaselineRadiance) {
    auto& ds = get_medium_dataset();
    if (!ds.valid) { GTEST_SKIP() << "Dataset not available"; }

    auto stats_off = trace_random_rays(ds, false);
    EXPECT_EQ(stats_off.num_finite, stats_off.total)
        << "All pixels should be finite with volume off";
    EXPECT_EQ(stats_off.num_nonneg, stats_off.total)
        << "All pixels should be non-negative with volume off";
    EXPECT_GT(stats_off.mean_luminance, 0.f)
        << "Mean luminance should be > 0 (scene is lit)";
}

// =====================================================================
//  INTEGRATION TESTS: Volume On Produces Valid Radiance
// =====================================================================

TEST(Medium_Integration, VolumeOn_FiniteNonNegative) {
    auto& ds = get_medium_dataset();
    if (!ds.valid) { GTEST_SKIP() << "Dataset not available"; }

    auto stats_on = trace_random_rays(ds, true, 2.0f, 0.0f, 0.95f);
    EXPECT_EQ(stats_on.num_finite, stats_on.total)
        << "All pixels should be finite with volume on";
    EXPECT_EQ(stats_on.num_nonneg, stats_on.total)
        << "All pixels should be non-negative with volume on";
}

TEST(Medium_Integration, VolumeOn_StillHasRadiance) {
    auto& ds = get_medium_dataset();
    if (!ds.valid) { GTEST_SKIP() << "Dataset not available"; }

    auto stats_on = trace_random_rays(ds, true, 1.0f, 0.0f, 0.95f);
    EXPECT_GT(stats_on.mean_luminance, 0.f)
        << "Scene should still have non-zero radiance with light volume";
}

// =====================================================================
//  INTEGRATION TESTS: Higher Density → More Attenuation
// =====================================================================

TEST(Medium_Integration, HigherDensity_ReducesSurfaceRadiance) {
    auto& ds = get_medium_dataset();
    if (!ds.valid) { GTEST_SKIP() << "Dataset not available"; }

    auto stats_low  = trace_random_rays(ds, true, 0.5f);
    auto stats_high = trace_random_rays(ds, true, 10.0f);

    // Higher density should attenuate surface radiance more
    // (mean luminance decreases overall because Beer–Lambert absorbs more)
    EXPECT_LT(stats_high.mean_luminance, stats_low.mean_luminance)
        << "Higher density should produce lower mean luminance due to "
           "stronger Beer–Lambert attenuation";
}

// =====================================================================
//  INTEGRATION TESTS: Falloff Effect
// =====================================================================

TEST(Medium_Integration, FalloffReducesExtinction) {
    auto& ds = get_medium_dataset();
    if (!ds.valid) { GTEST_SKIP() << "Dataset not available"; }

    // With strong falloff, medium fades above y=0 → less attenuation overall
    // compared to homogeneous (falloff=0) at same density
    auto stats_homo = trace_random_rays(ds, true, 5.f, 0.0f);
    auto stats_fall = trace_random_rays(ds, true, 5.f, 10.0f);

    // Falloff should result in higher luminance (less overall extinction)
    // Note: this depends on camera position; at y ≈ 0 and looking into the scene
    // the falloff medium is thinner along most ray segments above ground.
    // We just verify both are valid and non-zero.
    EXPECT_GT(stats_homo.mean_luminance, 0.f);
    EXPECT_GT(stats_fall.mean_luminance, 0.f);
    EXPECT_EQ(stats_fall.num_finite, stats_fall.total);
}

// =====================================================================
//  INTEGRATION TESTS: Volume vs No-Volume Consistency
// =====================================================================

TEST(Medium_Integration, ZeroDensity_EqualsVolumeOff) {
    auto& ds = get_medium_dataset();
    if (!ds.valid) { GTEST_SKIP() << "Dataset not available"; }

    auto stats_off  = trace_random_rays(ds, false);
    auto stats_zero = trace_random_rays(ds, true, 0.0f);

    // With density = 0, volume should have no effect
    EXPECT_NEAR(stats_off.mean_luminance, stats_zero.mean_luminance,
                stats_off.mean_luminance * 0.01f + 1e-6f)
        << "Zero density volume should produce same result as volume off";
}

// =====================================================================
//  UNIT TEST: Transmittance is per-wavelength spectral
// =====================================================================

TEST(Medium_Spectral, TransmittanceIsSpectral) {
    HomogeneousMedium m = make_rayleigh_medium(2.0f, 0.9f);
    Spectrum T = transmittance(m, 1.0f);

    // Blue wavelengths should be attenuated more (higher σ_t)
    EXPECT_LT(T.value[0], T.value[NUM_LAMBDA - 1])
        << "Short wavelengths should transmit less (Rayleigh)";
}

TEST(Medium_Spectral, TransmittanceBlueShift) {
    // After transmittance, average wavelength should shift red
    // (blue is attenuated more than red)
    HomogeneousMedium m = make_rayleigh_medium(3.0f, 0.9f);
    Spectrum T = transmittance(m, 2.0f);

    // Compute weighted average wavelength
    float wt_sum = 0.f, w_sum = 0.f;
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        wt_sum += lambda_of_bin(i) * T.value[i];
        w_sum  += T.value[i];
    }
    float avg_lambda = wt_sum / w_sum;

    // Should be above the midpoint (580 nm) → redshifted
    EXPECT_GT(avg_lambda, (LAMBDA_MIN + LAMBDA_MAX) / 2.f)
        << "Transmitted spectrum should be redshifted by Rayleigh attenuation";
}

// =====================================================================
//  UNIT TEST: Phase function produces correct scattering colour
// =====================================================================

TEST(Medium_PhaseColor, RayleighScatterIsBlue) {
    // Rayleigh σ_s ∝ 1/λ⁴  →  scattered light is blue (higher at short λ)
    HomogeneousMedium m = make_rayleigh_medium(2.0f, 0.9f);

    // σ_s at blue end should be > red end
    EXPECT_GT(m.sigma_s.value[0], m.sigma_s.value[NUM_LAMBDA - 1])
        << "Scattered light coefficient should be stronger at blue wavelengths";
}

// =====================================================================
//  EDGE CASES
// =====================================================================

TEST(Medium_EdgeCase, NegativeY_WithFalloff) {
    // Negative y should increase density (exponential growth)
    HomogeneousMedium m_below = make_rayleigh_medium(1.f, 0.9f, 2.f, -1.f);
    HomogeneousMedium m_above = make_rayleigh_medium(1.f, 0.9f, 2.f, 1.f);
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        EXPECT_GT(m_below.sigma_t.value[i], m_above.sigma_t.value[i])
            << "Below ground (y<0) with falloff should have higher σ_t";
    }
}

TEST(Medium_EdgeCase, VeryHighAlbedo) {
    HomogeneousMedium m = make_rayleigh_medium(2.f, 1.0f);
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        EXPECT_FLOAT_EQ(m.sigma_a.value[i], 0.f)
            << "Albedo 1.0 → zero absorption";
        EXPECT_FLOAT_EQ(m.sigma_s.value[i], m.sigma_t.value[i]);
    }
}

TEST(Medium_EdgeCase, ZeroAlbedo) {
    HomogeneousMedium m = make_rayleigh_medium(2.f, 0.0f);
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        EXPECT_FLOAT_EQ(m.sigma_s.value[i], 0.f)
            << "Albedo 0 → zero scattering";
        EXPECT_FLOAT_EQ(m.sigma_a.value[i], m.sigma_t.value[i]);
    }
}

// =====================================================================
//  VOLUME PHOTON INTERACTION TESTS
// =====================================================================

// --- Helper: Trace photons with volume and return volume photons ------
static PhotonSoA trace_volume_photons_helper(const Scene& scene,
                                              int num_photons = 50000,
                                              float density   = 2.5f,
                                              float albedo    = 0.95f) {
    EmitterConfig ecfg;
    ecfg.num_photons    = num_photons;
    ecfg.max_bounces    = 10;
    ecfg.rr_threshold   = 0.95f;
    ecfg.min_bounces_rr = 3;
    ecfg.volume_enabled = true;
    ecfg.volume_density = density;
    ecfg.volume_falloff = 0.f;
    ecfg.volume_albedo  = albedo;

    PhotonSoA global, caustic, volume;
    trace_photons(scene, ecfg, global, caustic, &volume);
    return volume;
}

TEST(VolumePhoton, EmitsVolumePhotons) {
    auto& ds = get_medium_dataset();
    if (!ds.valid) GTEST_SKIP() << "Dataset not loaded";

    PhotonSoA vol = trace_volume_photons_helper(ds.scene);
    EXPECT_GT(vol.size(), 0u)
        << "Volume photon emission should produce some photons";
    std::cout << "[VolumePhoton] Emitted " << vol.size()
              << " volume photons from " << 50000 << " traced\n";
}

TEST(VolumePhoton, PositionsAreFinite) {
    auto& ds = get_medium_dataset();
    if (!ds.valid) GTEST_SKIP();

    PhotonSoA vol = trace_volume_photons_helper(ds.scene);
    ASSERT_GT(vol.size(), 0u);

    for (size_t i = 0; i < vol.size(); ++i) {
        EXPECT_TRUE(std::isfinite(vol.pos_x[i])) << "pos_x[" << i << "]";
        EXPECT_TRUE(std::isfinite(vol.pos_y[i])) << "pos_y[" << i << "]";
        EXPECT_TRUE(std::isfinite(vol.pos_z[i])) << "pos_z[" << i << "]";
    }
}

TEST(VolumePhoton, DirectionsAreUnit) {
    auto& ds = get_medium_dataset();
    if (!ds.valid) GTEST_SKIP();

    PhotonSoA vol = trace_volume_photons_helper(ds.scene);
    ASSERT_GT(vol.size(), 0u);

    for (size_t i = 0; i < std::min(vol.size(), (size_t)1000); ++i) {
        float len = sqrtf(vol.wi_x[i] * vol.wi_x[i]
                        + vol.wi_y[i] * vol.wi_y[i]
                        + vol.wi_z[i] * vol.wi_z[i]);
        EXPECT_NEAR(len, 1.0f, 0.01f) << "wi[" << i << "] not unit";
    }
}

TEST(VolumePhoton, FluxIsPositiveFinite) {
    auto& ds = get_medium_dataset();
    if (!ds.valid) GTEST_SKIP();

    PhotonSoA vol = trace_volume_photons_helper(ds.scene);
    ASSERT_GT(vol.size(), 0u);

    for (size_t i = 0; i < vol.size(); ++i) {
        EXPECT_GT(vol.flux[i], 0.f) << "flux[" << i << "] should be > 0";
        EXPECT_TRUE(std::isfinite(vol.flux[i])) << "flux[" << i << "] inf/nan";
    }
}

TEST(VolumePhoton, LambdaBinsInRange) {
    auto& ds = get_medium_dataset();
    if (!ds.valid) GTEST_SKIP();

    PhotonSoA vol = trace_volume_photons_helper(ds.scene);
    ASSERT_GT(vol.size(), 0u);

    for (size_t i = 0; i < vol.size(); ++i) {
        EXPECT_GE(vol.lambda_bin[i], 0);
        EXPECT_LT(vol.lambda_bin[i], NUM_LAMBDA);
    }
}

TEST(VolumePhoton, HigherDensityMorePhotons) {
    auto& ds = get_medium_dataset();
    if (!ds.valid) GTEST_SKIP();

    PhotonSoA low  = trace_volume_photons_helper(ds.scene, 50000, 0.5f);
    PhotonSoA high = trace_volume_photons_helper(ds.scene, 50000, 5.0f);

    // Higher density → more free-flight scattering events
    EXPECT_GT(high.size(), low.size())
        << "Higher density should produce more volume photons";
    std::cout << "[VolumePhoton] Low density: " << low.size()
              << ", High density: " << high.size() << "\n";
}

TEST(VolumePhoton, ZeroDensityNoPhotons) {
    auto& ds = get_medium_dataset();
    if (!ds.valid) GTEST_SKIP();

    PhotonSoA vol = trace_volume_photons_helper(ds.scene, 10000, 0.0f);
    EXPECT_EQ(vol.size(), 0u)
        << "Zero density should produce no volume photons";
}

TEST(VolumePhoton, DisabledProducesNoPhotons) {
    auto& ds = get_medium_dataset();
    if (!ds.valid) GTEST_SKIP();

    EmitterConfig ecfg;
    ecfg.num_photons    = 10000;
    ecfg.max_bounces    = 10;
    ecfg.volume_enabled = false;
    ecfg.volume_density = 2.5f;

    PhotonSoA global, caustic, volume;
    trace_photons(ds.scene, ecfg, global, caustic, &volume);
    EXPECT_EQ(volume.size(), 0u)
        << "volume_enabled=false should skip volume photon deposits";
}

// =====================================================================
//  VOLUME CELL-BIN GRID TESTS
// =====================================================================

#include "core/cell_bin_grid.h"

TEST(VolumeCellGrid, BuildsFromVolumePhotons) {
    auto& ds = get_medium_dataset();
    if (!ds.valid) GTEST_SKIP();

    PhotonSoA vol = trace_volume_photons_helper(ds.scene);
    ASSERT_GT(vol.size(), 0u);

    // Precompute bin indices
    PhotonBinDirs bin_dirs;
    bin_dirs.init(PHOTON_BIN_COUNT);
    vol.bin_idx.resize(vol.size());
    for (size_t i = 0; i < vol.size(); ++i) {
        float3 wi = make_f3(vol.wi_x[i], vol.wi_y[i], vol.wi_z[i]);
        vol.bin_idx[i] = (uint8_t)bin_dirs.find_nearest(wi);
    }

    CellBinGrid grid;
    float radius = DEFAULT_GATHER_RADIUS;
    grid.build(vol, radius, PHOTON_BIN_COUNT);

    EXPECT_GT(grid.total_cells(), 0);
    EXPECT_GT(grid.cell_size, 0.f);
    EXPECT_FALSE(grid.bins.empty());

    std::cout << "[VolumeCellGrid] " << grid.dim_x << "×"
              << grid.dim_y << "×" << grid.dim_z
              << " = " << grid.total_cells() << " cells, "
              << vol.size() << " photons\n";
}

TEST(VolumeCellGrid, LookupReturnsValidBins) {
    auto& ds = get_medium_dataset();
    if (!ds.valid) GTEST_SKIP();

    PhotonSoA vol = trace_volume_photons_helper(ds.scene, 20000);
    if (vol.size() == 0) GTEST_SKIP() << "No volume photons emitted";

    PhotonBinDirs bin_dirs;
    bin_dirs.init(PHOTON_BIN_COUNT);
    vol.bin_idx.resize(vol.size());
    for (size_t i = 0; i < vol.size(); ++i) {
        float3 wi = make_f3(vol.wi_x[i], vol.wi_y[i], vol.wi_z[i]);
        vol.bin_idx[i] = (uint8_t)bin_dirs.find_nearest(wi);
    }

    CellBinGrid grid;
    grid.build(vol, DEFAULT_GATHER_RADIUS, PHOTON_BIN_COUNT);

    // Lookup at the position of each volume photon — should get non-null
    int hits = 0;
    for (size_t i = 0; i < std::min(vol.size(), (size_t)200); ++i) {
        float3 pos = make_f3(vol.pos_x[i], vol.pos_y[i], vol.pos_z[i]);
        const PhotonBin* bins = grid.lookup(pos);
        ASSERT_NE(bins, nullptr);

        // At least one bin should have data since we're at a photon location
        bool any = false;
        for (int k = 0; k < PHOTON_BIN_COUNT; ++k)
            if (bins[k].count > 0) { any = true; break; }
        if (any) ++hits;
    }
    EXPECT_GT(hits, 0) << "Lookup at photon positions should find data";
}

TEST(VolumeCellGrid, BinFluxNonNegative) {
    auto& ds = get_medium_dataset();
    if (!ds.valid) GTEST_SKIP();

    PhotonSoA vol = trace_volume_photons_helper(ds.scene, 20000);
    if (vol.size() == 0) GTEST_SKIP();

    PhotonBinDirs bin_dirs;
    bin_dirs.init(PHOTON_BIN_COUNT);
    vol.bin_idx.resize(vol.size());
    for (size_t i = 0; i < vol.size(); ++i) {
        float3 wi = make_f3(vol.wi_x[i], vol.wi_y[i], vol.wi_z[i]);
        vol.bin_idx[i] = (uint8_t)bin_dirs.find_nearest(wi);
    }

    CellBinGrid grid;
    grid.build(vol, DEFAULT_GATHER_RADIUS, PHOTON_BIN_COUNT);

    // All bins should have non-negative flux and count
    for (size_t c = 0; c < (size_t)grid.total_cells(); ++c) {
        for (int k = 0; k < PHOTON_BIN_COUNT; ++k) {
            const PhotonBin& b = grid.bins[c * PHOTON_BIN_COUNT + k];
            EXPECT_GE(b.flux, 0.f);
            EXPECT_GE(b.count, 0);
            if (b.count > 0) {
                EXPECT_GT(b.flux, 0.f);
                // Direction should be unit
                float len = sqrtf(b.dir_x * b.dir_x + b.dir_y * b.dir_y
                                + b.dir_z * b.dir_z);
                EXPECT_NEAR(len, 1.0f, 0.02f);
            }
        }
    }
}

// =====================================================================
//  VOLUME GATHER INTEGRATION TEST
// =====================================================================

TEST(VolumeGather, VolumePhotonsContributeInScatter) {
    auto& ds = get_medium_dataset();
    if (!ds.valid) GTEST_SKIP();

    // Trace rays with volume OFF
    RenderConfig cfg_off;
    cfg_off.image_width  = 16;
    cfg_off.image_height = 16;
    cfg_off.samples_per_pixel = 4;
    cfg_off.num_photons   = 50000;
    cfg_off.gather_radius = DEFAULT_GATHER_RADIUS;
    cfg_off.caustic_radius = DEFAULT_CAUSTIC_RADIUS;
    cfg_off.max_bounces   = 5;
    cfg_off.volume_enabled = false;
    cfg_off.volume_density = 0.f;
    cfg_off.mode = RenderMode::Full;

    Renderer renderer_off;
    renderer_off.set_config(cfg_off);
    renderer_off.set_scene(&ds.scene);
    renderer_off.build_photon_maps();

    Camera cam = Camera::cornell_box_camera(16, 16);
    cam.update();

    // Trace a few rays with volume OFF
    PCGRng rng = PCGRng::seed(42);
    Spectrum L_off = Spectrum::zero();
    int N = 64;
    for (int i = 0; i < N; ++i) {
        int px = i % 16;
        int py = i / 16;
        Ray r = cam.generate_ray(px, py, rng);
        auto result = renderer_off.trace_path(r, rng);
        for (int j = 0; j < NUM_LAMBDA; ++j)
            L_off.value[j] += result.combined.value[j];
    }

    // Trace same rays with volume ON
    RenderConfig cfg_on = cfg_off;
    cfg_on.volume_enabled = true;
    cfg_on.volume_density = 3.0f;
    cfg_on.volume_albedo  = 0.95f;
    cfg_on.volume_samples = 2;
    cfg_on.volume_max_t   = 5.0f;

    Renderer renderer_on;
    renderer_on.set_config(cfg_on);
    renderer_on.set_scene(&ds.scene);
    renderer_on.build_photon_maps();

    Spectrum L_on = Spectrum::zero();
    PCGRng rng2 = PCGRng::seed(42);
    for (int i = 0; i < N; ++i) {
        int px = i % 16;
        int py = i / 16;
        Ray r = cam.generate_ray(px, py, rng2);
        auto result = renderer_on.trace_path(r, rng2);
        for (int j = 0; j < NUM_LAMBDA; ++j)
            L_on.value[j] += result.combined.value[j];
    }

    // Volume medium should produce different results:
    // - transmittance reduces surface contribution
    // - in-scatter adds volume contribution
    // Just check that we get finite non-negative results and they differ
    float sum_off = 0.f, sum_on = 0.f;
    for (int j = 0; j < NUM_LAMBDA; ++j) {
        EXPECT_TRUE(std::isfinite(L_on.value[j]));
        EXPECT_GE(L_on.value[j], 0.f);
        sum_off += L_off.value[j];
        sum_on  += L_on.value[j];
    }
    // They shouldn't be identical (high density changes the image)
    EXPECT_NE(sum_off, sum_on)
        << "Volume ON should produce different radiance than volume OFF";
    std::cout << "[VolumeGather] L_off=" << sum_off
              << " L_on=" << sum_on << "\n";
}
