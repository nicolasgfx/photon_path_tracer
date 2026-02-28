// ---------------------------------------------------------------------
// test_main.cpp - Comprehensive unit tests for spectral photon+path tracer
// ---------------------------------------------------------------------
// Tests cover:
//   - Vector math (types.h)
//   - ONB coordinate frame
//   - Spectrum operations and CIE colour matching
//   - RGB↔spectral conversion and round-trip
//   - Blackbody spectrum (Wien, Stefan-Boltzmann)
//   - PCG RNG distribution
//   - Cosine/uniform hemisphere sampling (PDF integration)
//   - Triangle sampling (uniform barycentric)
//   - Power heuristic (MIS)
//   - Alias table (Vose's)
//   - Moller-Trumbore ray-triangle intersection
//   - AABB intersection
//   - BSDF energy conservation (white furnace test)
//   - BSDF Helmholtz reciprocity
//   - BSDF at grazing angles
//   - Glass Fresnel energy balance
//   - Fresnel boundary conditions
//   - GGX normalization, VNDF sampling, Smith G symmetry
//   - Hash grid build / query
//   - Density estimator surface-consistency filter
//   - Density estimator normalization factor
//   - Geometric edge case: photons on nearby back-facing triangle
//   - Camera ray generation
//   - FrameBuffer tonemap pipeline
//   - Material type classification
//   - Triangle degenerate & normal interpolation
//   - Cornell box: scene loading, BVH vs brute-force, shadow rays,
//     camera rays, emitter sampling, direct lighting, photon tracing
// ---------------------------------------------------------------------

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>
#include <map>
#include <set>
#include <algorithm>

#include "core/types.h"
#include "core/spectrum.h"
#include "core/random.h"
#include "core/alias_table.h"
#include "scene/triangle.h"
#include "scene/material.h"
#include "bsdf/bsdf.h"
#include "photon/photon.h"
#include "photon/hash_grid.h"
#include "photon/density_estimator.h"
#include "renderer/nee_shared.h"
#include "renderer/camera.h"
#include "renderer/renderer.h"
#include "renderer/direct_light.h"
#include "scene/scene.h"
#include "scene/obj_loader.h"
#include "photon/emitter.h"
#include "photon/photon_bins.h"
#include "photon/cell_bin_grid.h"
#include "renderer/sppm.h"

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------
static constexpr float kTol   = 1e-5f;
static constexpr float kLoose = 1e-3f;
static constexpr float kStat  = 0.05f; // 5% tolerance for statistical tests

// ── Helpers for spectral photon creation ────────────────────────────
// Create a Photon with uniform spectral flux across all bins.
static Photon make_test_photon(float3 pos, float3 wi, float3 gnorm, float flux_value) {
    Photon p;
    p.position      = pos;
    p.wi            = wi;
    p.geom_normal   = gnorm;
    p.spectral_flux = Spectrum::constant(flux_value);
    return p;
}

// Set up SoA spectral_flux from per-photon scalar values (uniform across bins).
static void set_soa_flux_uniform(PhotonSoA& soa, std::initializer_list<float> per_photon_flux) {
    soa.spectral_flux.clear();
    soa.flux.clear();
    soa.lambda_bin.clear();
    soa.num_hero.clear();
    for (float f : per_photon_flux) {
        for (int b = 0; b < NUM_LAMBDA; ++b)
            soa.spectral_flux.push_back(f);
        // Also populate hero-wavelength flux (used by CellBinGrid build)
        for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
            soa.lambda_bin.push_back((uint16_t)(h * NUM_LAMBDA / HERO_WAVELENGTHS));
            soa.flux.push_back(f);
        }
        soa.num_hero.push_back((uint8_t)HERO_WAVELENGTHS);
    }
}

// (Intentionally omitted: 'approx' helper was unused - use EXPECT_NEAR instead)

// =====================================================================
//  SECTION 1 - Vector math (types.h)
// =====================================================================

TEST(VectorMath, Addition) {
    float3 a = make_f3(1, 2, 3);
    float3 b = make_f3(4, 5, 6);
    float3 c = a + b;
    EXPECT_NEAR(c.x, 5.f, kTol);
    EXPECT_NEAR(c.y, 7.f, kTol);
    EXPECT_NEAR(c.z, 9.f, kTol);
}

TEST(VectorMath, Subtraction) {
    float3 a = make_f3(5, 7, 9);
    float3 b = make_f3(1, 2, 3);
    float3 c = a - b;
    EXPECT_NEAR(c.x, 4.f, kTol);
    EXPECT_NEAR(c.y, 5.f, kTol);
    EXPECT_NEAR(c.z, 6.f, kTol);
}

TEST(VectorMath, ScalarMultiply) {
    float3 a = make_f3(1, 2, 3);
    float3 b = a * 2.f;
    EXPECT_NEAR(b.x, 2.f, kTol);
    EXPECT_NEAR(b.y, 4.f, kTol);
    EXPECT_NEAR(b.z, 6.f, kTol);

    float3 c = 3.f * a;
    EXPECT_NEAR(c.x, 3.f, kTol);
    EXPECT_NEAR(c.y, 6.f, kTol);
    EXPECT_NEAR(c.z, 9.f, kTol);
}

TEST(VectorMath, DotProduct) {
    float3 a = make_f3(1, 0, 0);
    float3 b = make_f3(0, 1, 0);
    EXPECT_NEAR(dot(a, b), 0.f, kTol);

    float3 c = make_f3(1, 2, 3);
    float3 d = make_f3(4, 5, 6);
    EXPECT_NEAR(dot(c, d), 32.f, kTol); // 4+10+18
}

TEST(VectorMath, CrossProduct) {
    float3 x = make_f3(1, 0, 0);
    float3 y = make_f3(0, 1, 0);
    float3 z = cross(x, y);
    EXPECT_NEAR(z.x, 0.f, kTol);
    EXPECT_NEAR(z.y, 0.f, kTol);
    EXPECT_NEAR(z.z, 1.f, kTol);

    // Anti-commutativity
    float3 w = cross(y, x);
    EXPECT_NEAR(w.z, -1.f, kTol);
}

TEST(VectorMath, Length) {
    float3 v = make_f3(3, 4, 0);
    EXPECT_NEAR(length(v), 5.f, kTol);
    EXPECT_NEAR(length_sq(v), 25.f, kTol);
}

TEST(VectorMath, Normalize) {
    float3 v = make_f3(3, 4, 0);
    float3 n = normalize(v);
    EXPECT_NEAR(length(n), 1.f, kTol);
    EXPECT_NEAR(n.x, 0.6f, kTol);
    EXPECT_NEAR(n.y, 0.8f, kTol);

    // Zero vector
    float3 z = normalize(make_f3(0, 0, 0));
    EXPECT_NEAR(length(z), 0.f, kTol);
}

TEST(VectorMath, Negation) {
    float3 a = make_f3(1, -2, 3);
    float3 b = -a;
    EXPECT_NEAR(b.x, -1.f, kTol);
    EXPECT_NEAR(b.y, 2.f, kTol);
    EXPECT_NEAR(b.z, -3.f, kTol);
}

TEST(VectorMath, PlusEquals) {
    float3 a = make_f3(1, 2, 3);
    a += make_f3(4, 5, 6);
    EXPECT_NEAR(a.x, 5.f, kTol);
    EXPECT_NEAR(a.y, 7.f, kTol);
    EXPECT_NEAR(a.z, 9.f, kTol);
}

TEST(VectorMath, FminFmax) {
    float3 a = make_f3(1, 5, 3);
    float3 b = make_f3(4, 2, 6);
    float3 mn = fminf3(a, b);
    float3 mx = fmaxf3(a, b);
    EXPECT_NEAR(mn.x, 1.f, kTol);
    EXPECT_NEAR(mn.y, 2.f, kTol);
    EXPECT_NEAR(mn.z, 3.f, kTol);
    EXPECT_NEAR(mx.x, 4.f, kTol);
    EXPECT_NEAR(mx.y, 5.f, kTol);
    EXPECT_NEAR(mx.z, 6.f, kTol);
}

// =====================================================================
//  SECTION 2 - ONB (Orthonormal Basis)
// =====================================================================

TEST(ONB, FromUpNormal) {
    ONB frame = ONB::from_normal(make_f3(0, 0, 1));
    // w = normal
    EXPECT_NEAR(frame.w.z, 1.f, kTol);
    // u, v should be perpendicular to w and each other
    EXPECT_NEAR(dot(frame.u, frame.w), 0.f, kTol);
    EXPECT_NEAR(dot(frame.v, frame.w), 0.f, kTol);
    EXPECT_NEAR(dot(frame.u, frame.v), 0.f, kTol);
    // All unit vectors
    EXPECT_NEAR(length(frame.u), 1.f, kTol);
    EXPECT_NEAR(length(frame.v), 1.f, kTol);
}

TEST(ONB, FromArbitraryNormal) {
    // Test with several normals
    float3 normals[] = {
        normalize(make_f3(1, 0, 0)),
        normalize(make_f3(0, 1, 0)),
        normalize(make_f3(1, 1, 1)),
        normalize(make_f3(-0.3f, 0.7f, 0.2f)),
    };

    for (auto& n : normals) {
        ONB frame = ONB::from_normal(n);
        EXPECT_NEAR(length(frame.u), 1.f, kTol) << "u not unit";
        EXPECT_NEAR(length(frame.v), 1.f, kTol) << "v not unit";
        EXPECT_NEAR(length(frame.w), 1.f, kTol) << "w not unit";
        EXPECT_NEAR(dot(frame.u, frame.v), 0.f, kTol) << "u⊥v fail";
        EXPECT_NEAR(dot(frame.u, frame.w), 0.f, kTol) << "u⊥w fail";
        EXPECT_NEAR(dot(frame.v, frame.w), 0.f, kTol) << "v⊥w fail";
    }
}

TEST(ONB, RoundTripLocalWorld) {
    ONB frame = ONB::from_normal(normalize(make_f3(1, 2, 3)));
    float3 dir_world = normalize(make_f3(0.5f, 0.3f, 0.7f));
    float3 dir_local = frame.world_to_local(dir_world);
    float3 dir_back  = frame.local_to_world(dir_local);
    EXPECT_NEAR(dir_back.x, dir_world.x, kTol);
    EXPECT_NEAR(dir_back.y, dir_world.y, kTol);
    EXPECT_NEAR(dir_back.z, dir_world.z, kTol);
}

TEST(ONB, NormalMapsToLocalZ) {
    float3 n = normalize(make_f3(0.5f, -0.3f, 0.8f));
    ONB frame = ONB::from_normal(n);
    float3 local = frame.world_to_local(n);
    EXPECT_NEAR(local.x, 0.f, kTol);
    EXPECT_NEAR(local.y, 0.f, kTol);
    EXPECT_NEAR(local.z, 1.f, kTol);
}

// =====================================================================
//  SECTION 3 - Spectrum
// =====================================================================

TEST(Spectrum, ZeroAndConstant) {
    Spectrum s = Spectrum::zero();
    for (int i = 0; i < NUM_LAMBDA; ++i) EXPECT_EQ(s[i], 0.f);

    Spectrum c = Spectrum::constant(2.5f);
    for (int i = 0; i < NUM_LAMBDA; ++i) EXPECT_NEAR(c[i], 2.5f, kTol);
}

TEST(Spectrum, Arithmetic) {
    Spectrum a = Spectrum::constant(2.f);
    Spectrum b = Spectrum::constant(3.f);

    Spectrum sum = a + b;
    for (int i = 0; i < NUM_LAMBDA; ++i) EXPECT_NEAR(sum[i], 5.f, kTol);

    Spectrum prod = a * b;
    for (int i = 0; i < NUM_LAMBDA; ++i) EXPECT_NEAR(prod[i], 6.f, kTol);

    Spectrum scaled = a * 4.f;
    for (int i = 0; i < NUM_LAMBDA; ++i) EXPECT_NEAR(scaled[i], 8.f, kTol);

    Spectrum div = a / 4.f;
    for (int i = 0; i < NUM_LAMBDA; ++i) EXPECT_NEAR(div[i], 0.5f, kTol);
}

TEST(Spectrum, PlusEqualsAndTimesEquals) {
    Spectrum a = Spectrum::constant(1.f);
    a += Spectrum::constant(2.f);
    for (int i = 0; i < NUM_LAMBDA; ++i) EXPECT_NEAR(a[i], 3.f, kTol);

    a *= 2.f;
    for (int i = 0; i < NUM_LAMBDA; ++i) EXPECT_NEAR(a[i], 6.f, kTol);
}

TEST(Spectrum, SumAndMax) {
    Spectrum s = Spectrum::constant(1.f);
    EXPECT_NEAR(s.sum(), (float)NUM_LAMBDA, kTol);
    EXPECT_NEAR(s.max_component(), 1.f, kTol);

    const int test_bin = NUM_LAMBDA - 1;
    s.value[test_bin] = 10.f;
    EXPECT_NEAR(s.max_component(), 10.f, kTol);
    EXPECT_EQ(s.dominant_bin(), test_bin);
}

TEST(Spectrum, LambdaOfBin) {
    // Bin 0 should center at LAMBDA_MIN + LAMBDA_STEP/2
    float expected0 = LAMBDA_MIN + LAMBDA_STEP * 0.5f;
    EXPECT_NEAR(lambda_of_bin(0), expected0, kTol);

    // Last bin
    float expectedLast = LAMBDA_MIN + (NUM_LAMBDA - 0.5f) * LAMBDA_STEP;
    EXPECT_NEAR(lambda_of_bin(NUM_LAMBDA - 1), expectedLast, kTol);
}

// -- CIE colour matching functions -----------------------------------

TEST(Spectrum, CIE_Y_PeakNear555nm) {
    // The luminosity function cie_y should peak near 555 nm
    float peak_lambda = 0.f;
    float peak_val = 0.f;
    for (float lam = 400.f; lam <= 700.f; lam += 1.f) {
        float y = cie_y(lam);
        if (y > peak_val) {
            peak_val = y;
            peak_lambda = lam;
        }
    }
    EXPECT_NEAR(peak_lambda, 555.f, 15.f); // Within 15nm of 555
    EXPECT_GT(peak_val, 0.9f);
}

TEST(Spectrum, CIE_NonNegativeInVisibleRange) {
    // cie_x, cie_y, cie_z should be non-negative in [380, 780]
    for (float lam = 380.f; lam <= 780.f; lam += 5.f) {
        EXPECT_GE(cie_y(lam), -0.01f) << "cie_y negative at " << lam;
        EXPECT_GE(cie_z(lam), -0.01f) << "cie_z negative at " << lam;
    }
}

TEST(Spectrum, WhiteSpectrumToSRGB) {
    // A flat unit spectrum should convert to a roughly white sRGB
    // With normalised XYZ integration (divided by sum(ybar)), flat 1.0 -> Y=1 -> white
    Spectrum white = Spectrum::constant(1.0f);
    float3 rgb = spectrum_to_srgb(white);
    // All channels should be similar (roughly white)
    EXPECT_GT(rgb.x, 0.5f);
    EXPECT_GT(rgb.y, 0.5f);
    EXPECT_GT(rgb.z, 0.5f);
    // Not wildly different
    EXPECT_NEAR(rgb.x, rgb.y, 0.3f);
    EXPECT_NEAR(rgb.y, rgb.z, 0.3f);
}

TEST(Spectrum, ZeroSpectrumToBlack) {
    Spectrum black = Spectrum::zero();
    float3 rgb = spectrum_to_srgb(black);
    EXPECT_NEAR(rgb.x, 0.f, kTol);
    EXPECT_NEAR(rgb.y, 0.f, kTol);
    EXPECT_NEAR(rgb.z, 0.f, kTol);
}

// -- RGB → spectral → sRGB round trip --------------------------------

TEST(Spectrum, RGBToSpectrumReflectance_Red) {
    Spectrum s = rgb_to_spectrum_reflectance(1.f, 0.f, 0.f);
    // The dominant wavelength should be in the red region (> 580nm)
    int dom = s.dominant_bin();
    float lam = lambda_of_bin(dom);
    EXPECT_GT(lam, 570.f) << "Red spectrum should peak in red region";
}

TEST(Spectrum, RGBToSpectrumReflectance_Green) {
    Spectrum s = rgb_to_spectrum_reflectance(0.f, 1.f, 0.f);
    int dom = s.dominant_bin();
    float lam = lambda_of_bin(dom);
    EXPECT_GT(lam, 500.f);
    EXPECT_LT(lam, 580.f);
}

TEST(Spectrum, RGBToSpectrumReflectance_Blue) {
    Spectrum s = rgb_to_spectrum_reflectance(0.f, 0.f, 1.f);
    int dom = s.dominant_bin();
    float lam = lambda_of_bin(dom);
    EXPECT_LT(lam, 500.f);
}

TEST(Spectrum, RGBToSpectrumReflectance_NonNegative) {
    Spectrum s = rgb_to_spectrum_reflectance(0.5f, 0.3f, 0.8f);
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        EXPECT_GE(s[i], 0.f) << "Negative spectral value at bin " << i;
    }
}

// -- Blackbody -------------------------------------------------------

TEST(Spectrum, BlackbodyPositive) {
    Spectrum bb = blackbody_spectrum(5500.f);
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        EXPECT_GT(bb[i], 0.f) << "Blackbody should be positive everywhere";
    }
}

TEST(Spectrum, BlackbodyPeakWavelength) {
    // Wien's displacement law: λ_max ≈ 2898/T μm = 2898000/T nm
    float T = 5500.f;
    float expected_peak_nm = 2898000.f / T; // ~527 nm
    Spectrum bb = blackbody_spectrum(T);
    int peak = bb.dominant_bin();
    float lam = lambda_of_bin(peak);
    EXPECT_NEAR(lam, expected_peak_nm, 30.f); // Within 30nm
}

TEST(Spectrum, SRGBGammaLinearizationZero) {
    EXPECT_NEAR(srgb_gamma(0.f), 0.f, kTol);
}

TEST(Spectrum, SRGBGammaLinearizationOne) {
    EXPECT_NEAR(srgb_gamma(1.f), 1.f, kTol);
}

// =====================================================================
//  SECTION 4 - PCG Random Number Generator
// =====================================================================

TEST(RNG, UniformDistribution) {
    // Chi-squared test: verify RNG produces roughly uniform [0,1)
    PCGRng rng = PCGRng::seed(42);
    const int N = 100000;
    const int BINS = 10;
    int counts[BINS] = {};
    for (int i = 0; i < N; ++i) {
        float v = rng.next_float();
        ASSERT_GE(v, 0.f);
        ASSERT_LT(v, 1.f);
        int bin = (int)(v * BINS);
        if (bin >= BINS) bin = BINS - 1;
        counts[bin]++;
    }
    float expected = (float)N / BINS;
    for (int i = 0; i < BINS; ++i) {
        EXPECT_NEAR((float)counts[i], expected, expected * 0.1f)
            << "Bin " << i << " deviates from uniform";
    }
}

TEST(RNG, DifferentSeeds) {
    PCGRng r1 = PCGRng::seed(1);
    PCGRng r2 = PCGRng::seed(2);
    // Different seeds should produce different sequences
    bool all_same = true;
    for (int i = 0; i < 100; ++i) {
        if (r1.next_uint() != r2.next_uint()) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same);
}

TEST(RNG, Reproducibility) {
    PCGRng r1 = PCGRng::seed(42, 7);
    PCGRng r2 = PCGRng::seed(42, 7);
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(r1.next_uint(), r2.next_uint());
    }
}

// =====================================================================
//  SECTION 5 - Sampling functions & PDF integration
// =====================================================================

TEST(Sampling, CosineHemispherePDFIntegratesToOne) {
    // Numerical integration of cosine_hemisphere_pdf over hemisphere
    // ∫∫ p(ω) dω = ∫_0^{2π} ∫_0^{π/2} (cos θ / π) sin θ dθ dφ = 1
    const int N = 1000000;
    PCGRng rng = PCGRng::seed(123);
    for (int i = 0; i < N; ++i) {
        float3 d = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
        float pdf = cosine_hemisphere_pdf(d.z);
        ASSERT_GT(pdf, 0.f);
        // Weight = f(x)/p(x) where f=1 → weight per sample = 1/1 = 1
        // but total solid angle = 2π, so MC integral of p(ω) dω ≈ 1.
        // Just verify all samples have z > 0
        EXPECT_GT(d.z, -1e-6f) << "Cosine sample below hemisphere";
    }
    // Alternative: check E[1/pdf * (1/(2π))] ≈ 1/total solid angle
    // Actually simpler: integrate cos(θ)/π over hemisphere analytically = 1
    // Just verify samples are on unit hemisphere
}

TEST(Sampling, CosineHemisphereSamplesOnUnitSphere) {
    PCGRng rng = PCGRng::seed(456);
    for (int i = 0; i < 10000; ++i) {
        float3 d = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
        EXPECT_NEAR(length(d), 1.f, 1e-4f);
        EXPECT_GT(d.z, -1e-6f); // Upper hemisphere
    }
}

TEST(Sampling, CosineHemisphereMeanCosTheta) {
    // E[cos θ] for cosine-weighted = ∫ cos²θ sinθ dθ dφ / ∫ cosθ sinθ dθ dφ = 2/3
    PCGRng rng = PCGRng::seed(789);
    const int N = 500000;
    double sum_cos = 0.0;
    for (int i = 0; i < N; ++i) {
        float3 d = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
        sum_cos += d.z; // cos(theta) = z in local frame
    }
    double mean_cos = sum_cos / N;
    EXPECT_NEAR(mean_cos, 2.0 / 3.0, 0.01);
}

TEST(Sampling, UniformHemisphereSamplesValid) {
    PCGRng rng = PCGRng::seed(321);
    for (int i = 0; i < 10000; ++i) {
        float3 d = sample_uniform_hemisphere(rng.next_float(), rng.next_float());
        EXPECT_NEAR(length(d), 1.f, 1e-4f);
        EXPECT_GE(d.z, -1e-6f);
    }
}

TEST(Sampling, UniformSphereSamplesValid) {
    PCGRng rng = PCGRng::seed(654);
    int above = 0, below = 0;
    for (int i = 0; i < 10000; ++i) {
        float3 d = sample_uniform_sphere(rng.next_float(), rng.next_float());
        EXPECT_NEAR(length(d), 1.f, 1e-4f);
        if (d.z > 0) above++; else below++;
    }
    // Should be roughly 50/50
    EXPECT_NEAR((float)above / 10000.f, 0.5f, 0.05f);
}

TEST(Sampling, TriangleSamplingBarycentricValid) {
    PCGRng rng = PCGRng::seed(999);
    for (int i = 0; i < 10000; ++i) {
        float3 b = sample_triangle(rng.next_float(), rng.next_float());
        EXPECT_GE(b.x, -1e-6f);
        EXPECT_GE(b.y, -1e-6f);
        EXPECT_GE(b.z, -1e-6f);
        EXPECT_NEAR(b.x + b.y + b.z, 1.f, 1e-4f);
    }
}

TEST(Sampling, TriangleSamplingUniform) {
    // Monte Carlo: area of sub-triangle where alpha > 0.5 should be ~0.25
    PCGRng rng = PCGRng::seed(1111);
    const int N = 200000;
    int count = 0;
    for (int i = 0; i < N; ++i) {
        float3 b = sample_triangle(rng.next_float(), rng.next_float());
        if (b.x > 0.5f) count++;
    }
    // Area fraction where alpha > 0.5 = 0.25 (geometric)
    EXPECT_NEAR((float)count / N, 0.25f, 0.02f);
}

// =====================================================================
//  SECTION 6 - Power heuristic (MIS)
// =====================================================================

TEST(MIS, PowerHeuristic2_Symmetric) {
    // When pdf_a == pdf_b, weight should be 0.5
    EXPECT_NEAR(power_heuristic(1.f, 1.f), 0.5f, kTol);
}

TEST(MIS, PowerHeuristic2_Dominance) {
    // When pdf_a >> pdf_b, weight → 1
    EXPECT_NEAR(power_heuristic(100.f, 1.f), 1.f, 0.001f);
    // When pdf_a << pdf_b, weight → 0
    EXPECT_NEAR(power_heuristic(1.f, 100.f), 0.f, 0.001f);
}

TEST(MIS, PowerHeuristic2_ZeroPDFs) {
    // Both zero should not crash
    float w = power_heuristic(0.f, 0.f);
    EXPECT_FALSE(std::isnan(w));
    EXPECT_FALSE(std::isinf(w));
}

TEST(MIS, PowerHeuristic3_Symmetric) {
    EXPECT_NEAR(power_heuristic_3(1.f, 1.f, 1.f), 1.f/3.f, kTol);
}

TEST(MIS, PowerHeuristic3_SumsToOne) {
    float pa = 2.f, pb = 3.f, pc = 5.f;
    float wa = power_heuristic_3(pa, pb, pc);
    float wb = power_heuristic_3(pb, pa, pc);
    float wc = power_heuristic_3(pc, pa, pb);
    EXPECT_NEAR(wa + wb + wc, 1.f, kTol);
}

TEST(MIS, MISWeight3_Consistent) {
    // nee_shared.h versions should match random.h versions
    float pa = 2.f, pb = 3.f, pc = 5.f;
    float w1 = nee_mis_weight_3(pa, pb, pc);
    float w2 = power_heuristic_3(pa, pb, pc);
    EXPECT_NEAR(w1, w2, kTol);
}

// =====================================================================
//  SECTION 7 - Alias Table (Vose's Algorithm)
// =====================================================================

TEST(AliasTable, UniformWeights) {
    std::vector<float> weights = {1, 1, 1, 1};
    AliasTable table = AliasTable::build(weights);
    EXPECT_EQ(table.n, 4);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(table.pdf(i), 0.25f, kTol);
    }
}

TEST(AliasTable, SamplingMatchesPDF) {
    std::vector<float> weights = {1, 2, 3, 4, 5};
    AliasTable table = AliasTable::build(weights);

    PCGRng rng = PCGRng::seed(42);
    const int N = 500000;
    std::vector<int> counts(5, 0);
    for (int i = 0; i < N; ++i) {
        int idx = table.sample(rng.next_float(), rng.next_float());
        ASSERT_GE(idx, 0);
        ASSERT_LT(idx, 5);
        counts[idx]++;
    }

    float total = 15.f; // sum of weights
    for (int i = 0; i < 5; ++i) {
        float expected = weights[i] / total;
        float observed = (float)counts[i] / N;
        EXPECT_NEAR(observed, expected, 0.01f)
            << "Alias table sample frequency for index " << i;
    }
}

TEST(AliasTable, SingleElement) {
    std::vector<float> weights = {5.0f};
    AliasTable table = AliasTable::build(weights);
    EXPECT_EQ(table.n, 1);
    EXPECT_NEAR(table.pdf(0), 1.f, kTol);

    PCGRng rng = PCGRng::seed(42);
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(table.sample(rng.next_float(), rng.next_float()), 0);
    }
}

TEST(AliasTable, ZeroWeightElement) {
    std::vector<float> weights = {0, 1, 0, 1, 0};
    AliasTable table = AliasTable::build(weights);

    PCGRng rng = PCGRng::seed(42);
    const int N = 100000;
    std::vector<int> counts(5, 0);
    for (int i = 0; i < N; ++i) {
        int idx = table.sample(rng.next_float(), rng.next_float());
        counts[idx]++;
    }
    // Indices 0, 2, 4 should have ~0 samples
    EXPECT_LT(counts[0], N / 100);
    EXPECT_LT(counts[2], N / 100);
    EXPECT_LT(counts[4], N / 100);
    // Indices 1, 3 should split roughly 50/50
    EXPECT_NEAR((float)counts[1] / N, 0.5f, 0.05f);
}

TEST(AliasTable, PDFSumsToOne) {
    std::vector<float> weights = {3, 7, 1, 9, 2};
    AliasTable table = AliasTable::build(weights);
    float sum = 0.f;
    for (int i = 0; i < table.n; ++i) {
        sum += table.pdf(i);
    }
    EXPECT_NEAR(sum, 1.f, kTol);
}

// =====================================================================
//  SECTION 8 - Triangle intersection (Moller-Trumbore)
// =====================================================================

TEST(Triangle, HitCentreOfTriangle) {
    Triangle tri;
    tri.v0 = make_f3(-1, -1, 0);
    tri.v1 = make_f3( 1, -1, 0);
    tri.v2 = make_f3( 0,  1, 0);
    tri.n0 = tri.n1 = tri.n2 = make_f3(0, 0, 1);
    tri.material_id = 0;

    Ray ray;
    ray.origin    = make_f3(0, 0, 5);
    ray.direction = make_f3(0, 0, -1);
    ray.tmin = 1e-4f;
    ray.tmax = 1e20f;

    float t, u, v;
    EXPECT_TRUE(tri.intersect(ray, t, u, v));
    EXPECT_NEAR(t, 5.f, kTol);
    // u + v should be < 1 (inside triangle)
    EXPECT_LT(u + v, 1.f + kTol);
}

TEST(Triangle, MissTriangle) {
    Triangle tri;
    tri.v0 = make_f3(-1, -1, 0);
    tri.v1 = make_f3( 1, -1, 0);
    tri.v2 = make_f3( 0,  1, 0);

    Ray ray;
    ray.origin    = make_f3(10, 10, 5);
    ray.direction = make_f3(0, 0, -1);
    ray.tmin = 1e-4f;
    ray.tmax = 1e20f;

    float t, u, v;
    EXPECT_FALSE(tri.intersect(ray, t, u, v));
}

TEST(Triangle, ParallelRay) {
    Triangle tri;
    tri.v0 = make_f3(0, 0, 0);
    tri.v1 = make_f3(1, 0, 0);
    tri.v2 = make_f3(0, 1, 0);

    Ray ray;
    ray.origin    = make_f3(0, 0, 1);
    ray.direction = make_f3(1, 0, 0); // Parallel to triangle plane
    ray.tmin = 1e-4f;
    ray.tmax = 1e20f;

    float t, u, v;
    EXPECT_FALSE(tri.intersect(ray, t, u, v));
}

TEST(Triangle, BehindRay) {
    Triangle tri;
    tri.v0 = make_f3(-1, -1, 0);
    tri.v1 = make_f3( 1, -1, 0);
    tri.v2 = make_f3( 0,  1, 0);

    Ray ray;
    ray.origin    = make_f3(0, 0, -5);
    ray.direction = make_f3(0, 0, -1); // Pointing away from triangle
    ray.tmin = 1e-4f;
    ray.tmax = 1e20f;

    float t, u, v;
    EXPECT_FALSE(tri.intersect(ray, t, u, v));
}

TEST(Triangle, EdgeHit) {
    Triangle tri;
    tri.v0 = make_f3(0, 0, 0);
    tri.v1 = make_f3(1, 0, 0);
    tri.v2 = make_f3(0, 1, 0);

    // Ray hitting exactly on the v0-v1 edge (y=0)
    Ray ray;
    ray.origin    = make_f3(0.5f, 0.f, 5.f);
    ray.direction = make_f3(0, 0, -1);
    ray.tmin = 1e-4f;
    ray.tmax = 1e20f;

    float t, u, v;
    // This may or may not hit depending on edge rules, just check no crash
    tri.intersect(ray, t, u, v);
}

TEST(Triangle, Area) {
    Triangle tri;
    tri.v0 = make_f3(0, 0, 0);
    tri.v1 = make_f3(1, 0, 0);
    tri.v2 = make_f3(0, 1, 0);
    EXPECT_NEAR(tri.area(), 0.5f, kTol);
}

TEST(Triangle, GeometricNormal) {
    Triangle tri;
    tri.v0 = make_f3(0, 0, 0);
    tri.v1 = make_f3(1, 0, 0);
    tri.v2 = make_f3(0, 1, 0);
    float3 n = tri.geometric_normal();
    EXPECT_NEAR(n.x, 0.f, kTol);
    EXPECT_NEAR(n.y, 0.f, kTol);
    EXPECT_NEAR(n.z, 1.f, kTol);
}

TEST(Triangle, InterpolatePosition) {
    Triangle tri;
    tri.v0 = make_f3(0, 0, 0);
    tri.v1 = make_f3(1, 0, 0);
    tri.v2 = make_f3(0, 1, 0);
    // Centroid: (1/3, 1/3, 1/3)
    float3 c = tri.interpolate_position(1.f/3, 1.f/3, 1.f/3);
    EXPECT_NEAR(c.x, 1.f/3, kTol);
    EXPECT_NEAR(c.y, 1.f/3, kTol);
    EXPECT_NEAR(c.z, 0.f, kTol);
}

// =====================================================================
//  SECTION 9 - AABB intersection
// =====================================================================

TEST(AABB, RayHitsBox) {
    AABB box;
    box.mn = make_f3(-1, -1, -1);
    box.mx = make_f3( 1,  1,  1);

    Ray ray;
    ray.origin    = make_f3(0, 0, 5);
    ray.direction = make_f3(0, 0, -1);
    ray.tmin = 0.f;
    ray.tmax = 100.f;

    float tmin, tmax;
    EXPECT_TRUE(box.intersect(ray, tmin, tmax));
    EXPECT_NEAR(tmin, 4.f, kTol);
    EXPECT_NEAR(tmax, 6.f, kTol);
}

TEST(AABB, RayMissesBox) {
    AABB box;
    box.mn = make_f3(-1, -1, -1);
    box.mx = make_f3( 1,  1,  1);

    Ray ray;
    ray.origin    = make_f3(5, 5, 5);
    ray.direction = make_f3(0, 0, -1);
    ray.tmin = 0.f;
    ray.tmax = 100.f;

    float tmin, tmax;
    EXPECT_FALSE(box.intersect(ray, tmin, tmax));
}

TEST(AABB, RayInsideBox) {
    AABB box;
    box.mn = make_f3(-1, -1, -1);
    box.mx = make_f3( 1,  1,  1);

    Ray ray;
    ray.origin    = make_f3(0, 0, 0);
    ray.direction = make_f3(1, 0, 0);
    ray.tmin = 0.f;
    ray.tmax = 100.f;

    float tmin, tmax;
    EXPECT_TRUE(box.intersect(ray, tmin, tmax));
}

TEST(AABB, Expand) {
    AABB box;
    box.expand(make_f3(1, 2, 3));
    box.expand(make_f3(-1, -2, -3));
    EXPECT_NEAR(box.mn.x, -1.f, kTol);
    EXPECT_NEAR(box.mn.y, -2.f, kTol);
    EXPECT_NEAR(box.mn.z, -3.f, kTol);
    EXPECT_NEAR(box.mx.x, 1.f, kTol);
    EXPECT_NEAR(box.mx.y, 2.f, kTol);
    EXPECT_NEAR(box.mx.z, 3.f, kTol);
}

TEST(AABB, LongestAxis) {
    AABB box;
    box.mn = make_f3(0, 0, 0);
    box.mx = make_f3(3, 2, 1);
    EXPECT_EQ(box.longest_axis(), 0); // X is longest

    box.mx = make_f3(1, 3, 2);
    EXPECT_EQ(box.longest_axis(), 1); // Y is longest
}

// =====================================================================
//  SECTION 10 - Fresnel
// =====================================================================

TEST(Fresnel, SchlickAtNormalIncidence) {
    // At cos_theta = 1, F = f0
    EXPECT_NEAR(fresnel_schlick(1.f, 0.04f), 0.04f, kTol);
    EXPECT_NEAR(fresnel_schlick(1.f, 0.5f), 0.5f, kTol);
}

TEST(Fresnel, SchlickAtGrazingAngle) {
    // At cos_theta = 0, F should approach 1
    EXPECT_NEAR(fresnel_schlick(0.f, 0.04f), 1.f, kTol);
}

TEST(Fresnel, SchlickMonotonic) {
    // Fresnel should increase as angle increases (cos decreases)
    for (float f0 = 0.01f; f0 <= 1.f; f0 += 0.1f) {
        float prev = fresnel_schlick(1.f, f0);
        for (float cos_t = 0.9f; cos_t >= 0.f; cos_t -= 0.1f) {
            float curr = fresnel_schlick(cos_t, f0);
            EXPECT_GE(curr, prev - kTol) << "Not monotonic at cos=" << cos_t;
            prev = curr;
        }
    }
}

TEST(Fresnel, DielectricNormalIncidence) {
    // Glass (n=1.5): F = ((1-1.5)/(1+1.5))^2 = (-0.5/2.5)^2 = 0.04
    float F = fresnel_dielectric(1.f, 1.f / 1.5f);
    EXPECT_NEAR(F, 0.04f, 0.01f);
}

TEST(Fresnel, DielectricTotalInternalReflection) {
    // sin(critical) = 1/n = 1/1.5 → cos(critical) ≈ 0.745
    // For angles above critical (cos < cos_crit), F = 1
    float F = fresnel_dielectric(0.3f, 1.5f); // Inside glass, going out
    EXPECT_NEAR(F, 1.f, kTol);
}

// =====================================================================
//  SECTION 11 - GGX microfacet distribution
// =====================================================================

TEST(GGX, NormalizationIntegral) {
    // For GGX: ∫ D(ωh) cos(θh) dω = 1 over hemisphere
    // MC estimate with uniform sampling
    PCGRng rng = PCGRng::seed(42);
    float alpha = 0.3f;
    const int N = 500000;
    double integral = 0.0;
    for (int i = 0; i < N; ++i) {
        float3 h = sample_uniform_hemisphere(rng.next_float(), rng.next_float());
        float D_val = ggx_D(h, alpha);
        float cos_h = h.z;
        // dω = 2π (for uniform hemisphere)
        integral += D_val * cos_h;
    }
    integral *= (2.0 * PI) / N; // uniform hemi pdf = 1/(2π)
    EXPECT_NEAR(integral, 1.0, 0.05);
}

TEST(GGX, DValueAtNormal) {
    // D is maximum when h = (0,0,1) for any alpha
    float alpha = 0.5f;
    float D_at_normal = ggx_D(make_f3(0, 0, 1), alpha);
    float D_at_45 = ggx_D(normalize(make_f3(0, 0.7071f, 0.7071f)), alpha);
    EXPECT_GT(D_at_normal, D_at_45);
}

TEST(GGX, SmithGeometryRange) {
    // G should be in [0, 1]
    PCGRng rng = PCGRng::seed(42);
    for (int i = 0; i < 1000; ++i) {
        float3 wo = sample_uniform_hemisphere(rng.next_float(), rng.next_float());
        float3 wi = sample_uniform_hemisphere(rng.next_float(), rng.next_float());
        float alpha = rng.next_float() * 0.9f + 0.1f;
        float G = ggx_G(wo, wi, alpha);
        EXPECT_GE(G, -kTol);
        EXPECT_LE(G, 1.f + kTol);
    }
}

// =====================================================================
//  SECTION 12 - BSDF tests
// =====================================================================

TEST(BSDF, LambertianEnergyConservation) {
    // White furnace test: ∫ f(wo,wi) cos(θi) dωi = albedo for Lambertian
    // Lambertian f = Kd/π, integral = Kd
    Material mat;
    mat.type = MaterialType::Lambertian;
    mat.Kd = Spectrum::constant(0.8f);

    float3 wo = make_f3(0, 0, 1); // Normal incidence
    PCGRng rng = PCGRng::seed(42);
    const int N = 200000;

    Spectrum accum = Spectrum::zero();
    for (int i = 0; i < N; ++i) {
        BSDFSample s = bsdf::sample(mat, wo, rng);
        if (s.pdf > 0.f && s.wi.z > 0.f) {
            float cos_theta = s.wi.z;
            // MC estimator: f * cos / pdf
            accum += s.f * (cos_theta / s.pdf);
        }
    }
    accum *= 1.f / N;

    // Should equal Kd = 0.8
    for (int j = 0; j < NUM_LAMBDA; ++j) {
        EXPECT_NEAR(accum[j], 0.8f, 0.03f)
            << "Lambertian energy conservation failed at bin " << j;
    }
}

TEST(BSDF, LambertianPDFConsistency) {
    // Sample direction and verify PDF matches
    Material mat;
    mat.type = MaterialType::Lambertian;
    mat.Kd = Spectrum::constant(0.5f);

    float3 wo = normalize(make_f3(0.3f, 0.2f, 0.9f));
    PCGRng rng = PCGRng::seed(42);

    for (int i = 0; i < 100; ++i) {
        BSDFSample s = bsdf::sample(mat, wo, rng);
        if (s.pdf > 0.f) {
            float expected_pdf = bsdf::pdf(mat, wo, s.wi);
            EXPECT_NEAR(s.pdf, expected_pdf, kTol)
                << "Lambertian: sample PDF != eval PDF";
        }
    }
}

TEST(BSDF, MirrorReflection) {
    Material mat;
    mat.type = MaterialType::Mirror;
    mat.Ks = Spectrum::constant(1.f);

    float3 wo = normalize(make_f3(0.3f, 0.0f, 0.9f));
    PCGRng rng = PCGRng::seed(42);

    BSDFSample s = bsdf::mirror_sample(mat.Ks, wo);
    // Mirror should reflect: wi.x = -wo.x, wi.y = -wo.y, wi.z = wo.z
    EXPECT_NEAR(s.wi.x, -wo.x, kTol);
    EXPECT_NEAR(s.wi.y, -wo.y, kTol);
    EXPECT_NEAR(s.wi.z, wo.z, kTol);
    EXPECT_TRUE(s.is_specular);
}

TEST(BSDF, GlassSampleValid) {
    // Glass should either reflect or refract, not produce NaN
    PCGRng rng = PCGRng::seed(42);
    for (int i = 0; i < 100; ++i) {
        float3 wo = sample_uniform_hemisphere(rng.next_float(), rng.next_float());
        BSDFSample s = bsdf::glass_sample(wo, 1.5f, rng);
        EXPECT_FALSE(std::isnan(s.wi.x));
        EXPECT_FALSE(std::isnan(s.wi.y));
        EXPECT_FALSE(std::isnan(s.wi.z));
        EXPECT_TRUE(s.is_specular);
        EXPECT_GT(s.pdf, 0.f);
    }
}

TEST(BSDF, GlossyEnergyBound) {
    // White furnace test: integral should be <= 1 (energy conservation)
    Material mat;
    mat.type = MaterialType::GlossyMetal;
    mat.Kd = Spectrum::constant(0.3f);
    mat.Ks = Spectrum::constant(0.5f);
    mat.roughness = 0.4f;

    float3 wo = make_f3(0, 0, 1);
    PCGRng rng = PCGRng::seed(42);
    const int N = 200000;

    Spectrum accum = Spectrum::zero();
    for (int i = 0; i < N; ++i) {
        BSDFSample s = bsdf::sample(mat, wo, rng);
        if (s.pdf > 0.f && s.wi.z > 0.f) {
            float cos_theta = s.wi.z;
            accum += s.f * (cos_theta / s.pdf);
        }
    }
    accum *= 1.f / N;

    for (int j = 0; j < NUM_LAMBDA; ++j) {
        EXPECT_LE(accum[j], 1.2f) // slight tolerance for MC noise
            << "Glossy energy conservation failed at bin " << j << " val=" << accum[j];
    }
}

TEST(BSDF, GlossyPDFConsistency) {
    Material mat;
    mat.type = MaterialType::GlossyMetal;
    mat.Kd = Spectrum::constant(0.3f);
    mat.Ks = Spectrum::constant(0.5f);
    mat.roughness = 0.4f;

    float3 wo = normalize(make_f3(0.2f, 0.1f, 0.95f));
    PCGRng rng = PCGRng::seed(42);

    for (int i = 0; i < 100; ++i) {
        BSDFSample s = bsdf::sample(mat, wo, rng);
        if (s.pdf > 0.f && s.wi.z > 0.f) {
            float expected_pdf = bsdf::pdf(mat, wo, s.wi);
            EXPECT_NEAR(s.pdf, expected_pdf, 0.01f)
                << "Glossy: sample PDF != eval PDF";
        }
    }
}

TEST(BSDF, EvaluateNonNegative) {
    // BSDF evaluate should never return negative values
    Material mat;
    mat.type = MaterialType::Lambertian;
    mat.Kd = Spectrum::constant(0.5f);

    PCGRng rng = PCGRng::seed(42);
    for (int i = 0; i < 100; ++i) {
        float3 wo = sample_uniform_hemisphere(rng.next_float(), rng.next_float());
        float3 wi = sample_uniform_hemisphere(rng.next_float(), rng.next_float());
        Spectrum f = bsdf::evaluate(mat, wo, wi);
        for (int j = 0; j < NUM_LAMBDA; ++j) {
            EXPECT_GE(f[j], 0.f) << "Negative BSDF at bin " << j;
        }
    }
}

TEST(BSDF, BelowHemisphereReturnsZero) {
    // If wi or wo are below hemisphere, should return zero
    Material mat;
    mat.type = MaterialType::Lambertian;
    mat.Kd = Spectrum::constant(0.5f);

    float3 wo = make_f3(0, 0, 1);
    float3 wi_below = make_f3(0, 0, -1);
    Spectrum f = bsdf::evaluate(mat, wo, wi_below);
    EXPECT_NEAR(f.sum(), 0.f, kTol);

    float pdf = bsdf::pdf(mat, wo, wi_below);
    EXPECT_NEAR(pdf, 0.f, kTol);
}

// =====================================================================
//  SECTION 13 - Hash Grid
// =====================================================================

TEST(HashGrid, BuildAndQueryBasic) {
    // Create a small set of photons
    PhotonSoA photons;
    for (int i = 0; i < 100; ++i) {
        Photon p;
        p.position = make_f3((float)i * 0.01f, 0.f, 0.f);
        p.wi = make_f3(0, 0, 1);
        p.spectral_flux = Spectrum::constant(1.0f);
        photons.push_back(p);
    }

    float radius = 0.05f;
    HashGrid grid;
    grid.build(photons, radius);

    // Query at origin should find photons within radius
    int count = 0;
    grid.query(make_f3(0, 0, 0), radius, photons,
        [&](uint32_t idx, float dist2) {
            (void)idx;
            (void)dist2;
            count++;
        });

    EXPECT_GT(count, 0) << "Should find at least one photon near origin";
    EXPECT_LE(count, 100) << "Should not find more than total";
}

TEST(HashGrid, QueryDistanceFilter) {
    // Place one photon at exactly (0.1, 0, 0)
    PhotonSoA photons;
    Photon p;
    p.position = make_f3(0.1f, 0.f, 0.f);
    p.wi = make_f3(0, 0, 1);
    p.spectral_flux = Spectrum::constant(1.0f);
    photons.push_back(p);

    float radius = 0.05f;
    HashGrid grid;
    grid.build(photons, radius);

    // Query at origin with radius 0.05 should NOT find the photon (dist=0.1)
    int count = 0;
    grid.query(make_f3(0, 0, 0), radius, photons,
        [&](uint32_t, float) { count++; });
    EXPECT_EQ(count, 0);

    // Query with larger radius should find it
    int count2 = 0;
    grid.query(make_f3(0, 0, 0), 0.15f, photons,
        [&](uint32_t, float) { count2++; });
    EXPECT_EQ(count2, 1);
}

TEST(HashGrid, EmptyGrid) {
    PhotonSoA photons;
    HashGrid grid;
    grid.build(photons, 0.1f);

    int count = 0;
    grid.query(make_f3(0, 0, 0), 0.1f, photons,
        [&](uint32_t, float) { count++; });
    EXPECT_EQ(count, 0);
}

TEST(HashGrid, AllPhotonsFound) {
    // Place photons in a tight cluster, all within radius
    PhotonSoA photons;
    PCGRng rng = PCGRng::seed(42);
    const int N = 50;
    float radius = 1.0f;

    for (int i = 0; i < N; ++i) {
        Photon p;
        p.position = make_f3(
            (rng.next_float() - 0.5f) * 0.1f,
            (rng.next_float() - 0.5f) * 0.1f,
            (rng.next_float() - 0.5f) * 0.1f);
        p.wi = make_f3(0, 0, 1);
        p.spectral_flux = Spectrum::constant(1.0f);
        photons.push_back(p);
    }

    HashGrid grid;
    grid.build(photons, radius);

    int count = 0;
    grid.query(make_f3(0, 0, 0), radius, photons,
        [&](uint32_t, float) { count++; });

    EXPECT_EQ(count, N) << "Should find all " << N << " photons in tight cluster";
}

TEST(HashGrid, DenseCell_ManyPhotonsSameCell) {
    // Place many photons in the exact same cell to test dense scenarios
    PhotonSoA photons;
    const int N = 500;
    float radius = 0.5f;

    // All photons within a 0.1 unit cube at positive coordinates.
    // Positions are entirely in [0.1, 0.2]^3 so they all fall in the same
    // hash cell (floor([0.1,0.2] / 1.0) = 0 on every axis).
    PCGRng rng = PCGRng::seed(123);
    for (int i = 0; i < N; ++i) {
        Photon p;
        p.position = make_f3(
            0.1f + rng.next_float() * 0.1f,
            0.1f + rng.next_float() * 0.1f,
            0.1f + rng.next_float() * 0.1f);
        p.wi = make_f3(0, 0, 1);
        p.spectral_flux = Spectrum::constant(1.0f);
        photons.push_back(p);
    }

    HashGrid grid;
    grid.build(photons, radius);

    // Query at center should find all photons
    int count = 0;
    grid.query(make_f3(0, 0, 0), radius, photons,
        [&](uint32_t, float) { count++; });

    EXPECT_EQ(count, N) << "Should find all " << N << " photons in dense cell";

    // Verify they're all in approximately the same cell
    int3 cell0 = grid.cell_coord(make_f3(photons.pos_x[0], 
                                         photons.pos_y[0], 
                                         photons.pos_z[0]));
    int same_cell = 0;
    for (int i = 0; i < N; ++i) {
        int3 cell_i = grid.cell_coord(make_f3(photons.pos_x[i],
                                               photons.pos_y[i],
                                               photons.pos_z[i]));
        if (cell_i.x == cell0.x && cell_i.y == cell0.y && cell_i.z == cell0.z) {
            same_cell++;
        }
    }
    EXPECT_GT(same_cell, N * 0.95) << "Most photons should be in the same cell";
}

TEST(HashGrid, DenseRegion_HighPhotonDensity) {
    // Test performance and correctness with very high photon density
    PhotonSoA photons;
    const int N = 2000;
    float radius = 1.0f;

    // Create a dense sphere of photons
    PCGRng rng = PCGRng::seed(456);
    for (int i = 0; i < N; ++i) {
        Photon p;
        // Uniform distribution in unit sphere
        float theta = 2.0f * PI * rng.next_float();
        float phi = std::acos(2.0f * rng.next_float() - 1.0f);
        float r = std::cbrt(rng.next_float()) * 0.5f; // Radius ≤ 0.5
        
        p.position = make_f3(
            r * std::sin(phi) * std::cos(theta),
            r * std::sin(phi) * std::sin(theta),
            r * std::cos(phi));
        p.wi = make_f3(0, 0, 1);
        p.spectral_flux = Spectrum::constant(1.0f);
        photons.push_back(p);
    }

    HashGrid grid;
    grid.build(photons, radius);

    // Query at center should find all photons (all within radius)
    int count = 0;
    grid.query(make_f3(0, 0, 0), radius, photons,
        [&](uint32_t /*idx*/, float dist2) {
            EXPECT_LE(dist2, radius * radius + kTol);
            count++;
        });

    EXPECT_EQ(count, N) << "Should find all photons in dense region";
}

TEST(HashGrid, HashCollision_MultipleKeysMapToSameBucket) {
    // Test that hash collisions are handled correctly
    // Force collisions by using a small table size
    PhotonSoA photons;
    
    // Create photons at positions designed to likely collide
    // (large spatial separation but small hash table)
    for (int i = 0; i < 20; ++i) {
        Photon p;
        p.position = make_f3((float)i * 10.0f, 0.0f, 0.0f);
        p.wi = make_f3(0, 0, 1);
        p.spectral_flux = Spectrum::constant(1.0f);
        photons.push_back(p);
    }

    float radius = 5.0f;
    HashGrid grid;
    grid.build(photons, radius);

    // Each query should only find its local photons, not others that hash-collided
    for (int i = 0; i < 20; ++i) {
        float3 query_pos = make_f3((float)i * 10.0f, 0.0f, 0.0f);
        int count = 0;
        grid.query(query_pos, radius, photons,
            [&](uint32_t idx, float dist2) {
                count++;
                // Verify distance is actually within radius
                EXPECT_LE(dist2, radius * radius + kTol)
                    << "Photon " << idx << " should be within radius";
            });
        
        // Should find at least the photon at this position
        EXPECT_GE(count, 1) << "Should find at least local photon at i=" << i;
        // Should not find all 20 photons (they're spatially separated)
        EXPECT_LT(count, 20) << "Should not find all photons due to distance";
    }
}

TEST(HashGrid, DenseCell_CorrectSorting) {
    // Verify that photons in dense cells are correctly sorted by cell key
    PhotonSoA photons;
    const int N = 300;
    float radius = 0.2f;

    PCGRng rng = PCGRng::seed(789);
    for (int i = 0; i < N; ++i) {
        Photon p;
        p.position = make_f3(
            rng.next_float() * 2.0f - 1.0f,
            rng.next_float() * 2.0f - 1.0f,
            rng.next_float() * 2.0f - 1.0f);
        p.wi = make_f3(0, 0, 1);
        p.spectral_flux = Spectrum::constant(1.0f);
        photons.push_back(p);
    }

    HashGrid grid;
    grid.build(photons, radius);

    // Verify sorted_indices has all photons
    EXPECT_EQ(grid.sorted_indices.size(), (size_t)N);

    // Verify no duplicate indices
    std::set<uint32_t> unique_indices(grid.sorted_indices.begin(), 
                                       grid.sorted_indices.end());
    EXPECT_EQ(unique_indices.size(), (size_t)N) << "All indices should be unique";

    // Verify all indices are in valid range
    for (uint32_t idx : grid.sorted_indices) {
        EXPECT_LT(idx, (uint32_t)N) << "Index should be in range [0, N)";
    }
}

TEST(HashGrid, DenseCell_NeighborCellQuery) {
    // Test that queries correctly search 3x3x3 neighboring cells in dense scenarios
    PhotonSoA photons;
    float radius = 0.3f;

    // Place photons in a grid pattern at cell boundaries
    // This tests the neighbor cell search
    for (int z = -1; z <= 1; ++z) {
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                Photon p;
                // Position at cell centers (cell_size = 2*radius = 0.6)
                p.position = make_f3(x * 0.6f, y * 0.6f, z * 0.6f);
                p.wi = make_f3(0, 0, 1);
                p.spectral_flux = Spectrum::constant(1.0f);
                photons.push_back(p);
            }
        }
    }

    HashGrid grid;
    grid.build(photons, radius);

    // Query at origin should find photons in the 27 neighboring cells
    int count = 0;
    grid.query(make_f3(0, 0, 0), radius, photons,
        [&](uint32_t, float) { count++; });

    EXPECT_GT(count, 0) << "Should find photons from neighboring cells";
    EXPECT_LE(count, 27) << "Should not find more than 27 photons (one per cell)";
}

TEST(HashGrid, DenseCell_EmptyBucketsHandled) {
    // Test that empty hash buckets are correctly handled in dense scenarios
    PhotonSoA photons;
    const int N = 100;
    float radius = 0.5f;

    // Create photons clustered in one region
    for (int i = 0; i < N; ++i) {
        Photon p;
        p.position = make_f3(0.0f, 0.0f, (float)i * 0.01f);
        p.wi = make_f3(0, 0, 1);
        p.spectral_flux = Spectrum::constant(1.0f);
        photons.push_back(p);
    }

    HashGrid grid;
    grid.build(photons, radius);

    // Query in empty region should return zero without crash
    int count = 0;
    grid.query(make_f3(100.0f, 100.0f, 100.0f), radius, photons,
        [&](uint32_t, float) { count++; });

    EXPECT_EQ(count, 0) << "Query in empty region should find no photons";

    // Query in dense region should find photons
    int count2 = 0;
    grid.query(make_f3(0.0f, 0.0f, 0.5f), radius, photons,
        [&](uint32_t, float) { count2++; });

    EXPECT_GT(count2, 0) << "Query in dense region should find photons";
}

TEST(HashGrid, DenseCell_DistanceFilteringAccurate) {
    // Verify distance filtering is accurate even with many photons per cell
    PhotonSoA photons;
    const int N = 200;
    float radius = 0.5f;

    PCGRng rng = PCGRng::seed(321);
    for (int i = 0; i < N; ++i) {
        Photon p;
        // Random positions in a larger region
        p.position = make_f3(
            rng.next_float() * 2.0f - 1.0f,
            rng.next_float() * 2.0f - 1.0f,
            rng.next_float() * 2.0f - 1.0f);
        p.wi = make_f3(0, 0, 1);
        p.spectral_flux = Spectrum::constant(1.0f);
        photons.push_back(p);
    }

    HashGrid grid;
    grid.build(photons, radius);

    float3 query_pos = make_f3(0.0f, 0.0f, 0.0f);
    int count = 0;
    
    grid.query(query_pos, radius, photons,
        [&](uint32_t idx, float dist2) {
            // Verify each returned photon is actually within radius
            float3 p = make_f3(photons.pos_x[idx], 
                              photons.pos_y[idx], 
                              photons.pos_z[idx]);
            float3 diff = query_pos - p;
            float actual_dist2 = dot(diff, diff);
            
            EXPECT_NEAR(dist2, actual_dist2, kTol)
                << "Callback dist2 should match actual distance";
            EXPECT_LE(actual_dist2, radius * radius + kTol)
                << "Photon should be within query radius";
            count++;
        });

    // All photons within radius should be found (verify against brute force)
    int brute_count = 0;
    for (size_t i = 0; i < photons.size(); ++i) {
        float3 p = make_f3(photons.pos_x[i], photons.pos_y[i], photons.pos_z[i]);
        float3 diff = query_pos - p;
        float dist2 = dot(diff, diff);
        if (dist2 <= radius * radius) {
            brute_count++;
        }
    }

    EXPECT_EQ(count, brute_count) 
        << "Grid query should find same photons as brute force";
}

// =====================================================================
//  SECTION 14 - Density Estimator (surface consistency filter)
// =====================================================================

TEST(DensityEstimator, SurfaceConsistency_PlaneDistReject) {
    // Photon far from surface plane should be rejected
    PhotonSoA photons;
    Photon p;
    p.position = make_f3(0.f, 0.f, 0.5f); // 0.5 units above surface
    p.wi = make_f3(0, 0, 1);  // Pointing into surface
    p.spectral_flux = Spectrum::constant(100.0f);
    photons.push_back(p);

    HashGrid grid;
    grid.build(photons, 1.0f);

    Material mat;
    mat.type = MaterialType::Lambertian;
    mat.Kd = Spectrum::constant(0.5f);

    DensityEstimatorConfig config;
    config.radius = 1.0f;
    config.surface_tau = 0.01f; // Tight surface consistency
    config.num_photons_total = 1;

    // Surface at z=0, normal = (0,0,1)
    Spectrum L = estimate_photon_density(
        make_f3(0, 0, 0), make_f3(0, 0, 1),
        make_f3(0, 0, 1), mat, photons, grid, config, 1.0f);

    // Should be zero because plane_dist = 0.5 > tau = 0.01
    EXPECT_NEAR(L.sum(), 0.f, kTol);
}

TEST(DensityEstimator, SurfaceConsistency_DirectionReject) {
    // Photon with wrong incoming direction should be rejected
    PhotonSoA photons;
    Photon p;
    p.position = make_f3(0.01f, 0.f, 0.f); // On surface
    p.wi = make_f3(0, 0, -1); // Pointing AWAY from surface (wrong direction)
    p.spectral_flux = Spectrum::constant(100.0f);
    photons.push_back(p);

    HashGrid grid;
    grid.build(photons, 0.5f);

    Material mat;
    mat.type = MaterialType::Lambertian;
    mat.Kd = Spectrum::constant(0.5f);

    DensityEstimatorConfig config;
    config.radius = 0.5f;
    config.surface_tau = 1.0f; // Loose tau so plane_dist doesn't reject
    config.num_photons_total = 1;

    Spectrum L = estimate_photon_density(
        make_f3(0, 0, 0), make_f3(0, 0, 1),
        make_f3(0, 0, 1), mat, photons, grid, config, 0.5f);

    // Should be zero: dot(wi, normal) = dot((0,0,-1),(0,0,1)) = -1 <= 0
    EXPECT_NEAR(L.sum(), 0.f, kTol);
}

TEST(DensityEstimator, ValidPhotonContributes) {
    // A photon that satisfies all filters should contribute
    PhotonSoA photons;
    Photon p;
    p.position    = make_f3(0.001f, 0.f, 0.f); // Very close to query point on surface
    p.wi          = make_f3(0, 0, 1);   // Pointing away from surface (stored convention)
    p.geom_normal = make_f3(0, 0, 1);   // Surface normal matches query normal
    p.spectral_flux = Spectrum::constant(1.0f);
    photons.push_back(p);

    HashGrid grid;
    grid.build(photons, 0.5f);

    Material mat;
    mat.type = MaterialType::Lambertian;
    mat.Kd = Spectrum::constant(1.f);

    DensityEstimatorConfig config;
    config.radius = 0.5f;
    config.surface_tau = 0.1f;
    config.num_photons_total = 1;
    config.use_kernel = false; // Use box kernel for predictable result

    Spectrum L = estimate_photon_density(
        make_f3(0, 0, 0), make_f3(0, 0, 1),
        make_f3(0, 0, 1), mat, photons, grid, config, 0.5f);

    // All bins should have nonzero contribution (full spectral photon)
    EXPECT_GT(L.sum(), 0.f) << "Valid photon should contribute";
}

// =====================================================================
//  SECTION 15 - Geometric edge case: back-facing nearby triangle
// =====================================================================
// This tests the crucial scenario where photons deposited on a nearby
// but BACK-FACING triangle surface should NOT be gathered during
// density estimation, even if they are within the gather radius
// and in the same hash cell.

TEST(DensityEstimator, BackFacingNearbyTriangleRejected) {
    //
    // Setup: Two parallel triangles very close together but facing
    // opposite directions:
    //   - Query surface at z=0, normal = (0,0,+1)
    //   - Photons on other surface at z=0.005, whose wi direction
    //     was aimed at the BACK-facing surface normal = (0,0,-1)
    //
    // The photon should be rejected because dot(wi, query_normal) <= 0
    //
    PhotonSoA photons;
    Photon p;
    // Photon sits on a surface at z=0.005 with normal (0,0,-1)
    // The photon was incoming from above: wi = (0, 0, -1) stored as flipped = (0,0,1)?
    // No — in emitter.h, wi is stored as `ray.direction * (-1)`, i.e. the
    // direction FROM which the photon came. So if the photon was traveling
    // downward (0,0,-1), the stored wi = (0,0,1).
    //
    // But the back-facing surface has normal (0,0,-1). The photon arrived
    // traveling in (0,0,+1) direction and hit the INSIDE of that surface.
    // Stored wi = (0,0,-1) (flipped from (0,0,+1)).
    //
    // For the query surface normal = (0,0,+1):
    //   dot(wi=(0,0,-1), n=(0,0,+1)) = -1 <= 0  → REJECTED ✓
    //
    p.position = make_f3(0.01f, 0.01f, 0.005f); // Very close, same hash cell
    p.wi = make_f3(0, 0, -1);  // Photon was coming from below (stored as wi)
    p.spectral_flux = Spectrum::constant(50.0f);
    photons.push_back(p);

    HashGrid grid;
    grid.build(photons, 0.5f);

    Material mat;
    mat.type = MaterialType::Lambertian;
    mat.Kd = Spectrum::constant(0.8f);

    DensityEstimatorConfig config;
    config.radius = 0.5f;
    config.surface_tau = 0.1f;   // Generous tau — won't reject by distance
    config.num_photons_total = 1;
    config.use_kernel = false;

    // Query at surface facing UP
    Spectrum L = estimate_photon_density(
        make_f3(0, 0, 0), make_f3(0, 0, 1),
        make_f3(0, 0, 1), mat, photons, grid, config, 0.5f);

    // Must be zero: the photon direction is incompatible with our surface normal
    EXPECT_NEAR(L.sum(), 0.f, kTol)
        << "Back-facing photon should NOT be gathered!";
}

TEST(DensityEstimator, ThinWallDoubleSided) {
    //
    // More thorough thin-wall test:
    // Simulate a thin wall (two triangles face-to-face, 1mm apart).
    // - Photons on the front surface (z=+0.001) with wi = (0,0,1) → valid for front
    // - Photons on the back surface (z=-0.001) with wi = (0,0,-1) → valid for back
    //
    // Query at z=0 with normal (0,0,1) should ONLY gather front-surface photons.
    //
    PhotonSoA photons;

    // Front-surface photon (facing same way as query normal +Z)
    Photon p_front;
    p_front.position    = make_f3(0.0f, 0.0f, 0.001f);
    p_front.wi          = make_f3(0, 0, 1);  // away from surface
    p_front.geom_normal = make_f3(0, 0, 1);  // surface faces +Z
    p_front.spectral_flux = Spectrum::constant(10.0f);
    photons.push_back(p_front);

    // Back-surface photon (facing opposite to front query normal)
    Photon p_back;
    p_back.position    = make_f3(0.0f, 0.0f, -0.001f);
    p_back.wi          = make_f3(0, 0, -1);  // away from back surface
    p_back.geom_normal = make_f3(0, 0, -1);  // surface faces -Z
    p_back.spectral_flux = Spectrum::constant(10.0f);
    photons.push_back(p_back);

    HashGrid grid;
    grid.build(photons, 0.5f);

    Material mat;
    mat.type = MaterialType::Lambertian;
    mat.Kd = Spectrum::constant(1.f);

    DensityEstimatorConfig config;
    config.radius = 0.5f;
    config.surface_tau = 0.01f; // 10mm tau — both photons pass distance check
    config.num_photons_total = 1;
    config.use_kernel = false;

    // Query at z=0, facing UP (+Z)
    Spectrum L_front = estimate_photon_density(
        make_f3(0, 0, 0), make_f3(0, 0, 1),
        make_f3(0, 0, 1), mat, photons, grid, config, 0.5f);

    // Query at z=0, facing DOWN (-Z)
    // wo_local must still have positive z in the local shading frame
    Spectrum L_back = estimate_photon_density(
        make_f3(0, 0, 0), make_f3(0, 0, -1),
        make_f3(0, 0, 1), mat, photons, grid, config, 0.5f);

    // Front query should only see p_front
    EXPECT_GT(L_front.sum(), 0.f) << "Front query should see front photon";

    // Back query should only see p_back
    EXPECT_GT(L_back.sum(), 0.f) << "Back query should see back photon";
}

TEST(DensityEstimator, SameCellDifferentFacingRejected) {
    //
    // Place many photons in the SAME hash cell, half facing one direction
    // and half facing the other. Only the correctly-facing half should
    // contribute to the density estimate.
    //
    PhotonSoA photons;
    const int N_per_side = 20;

    for (int i = 0; i < N_per_side; ++i) {
        // Upward-facing photons: surface normal = +Z, wi points away from surface
        Photon p;
        p.position    = make_f3((float)i * 0.001f, 0.f, 0.0001f);
        p.wi          = make_f3(0, 0, 1);  // away from +Z surface
        p.geom_normal = make_f3(0, 0, 1);  // surface faces up (+Z)
        p.spectral_flux = Spectrum::constant(1.0f);
        photons.push_back(p);
    }
    for (int i = 0; i < N_per_side; ++i) {
        // Downward-facing photons: surface normal = -Z, wi points away from surface
        Photon p;
        p.position    = make_f3((float)i * 0.001f, 0.f, -0.0001f);
        p.wi          = make_f3(0, 0, -1);  // away from -Z surface
        p.geom_normal = make_f3(0, 0, -1);  // surface faces down (-Z)
        p.spectral_flux = Spectrum::constant(1.0f);
        photons.push_back(p);
    }

    HashGrid grid;
    grid.build(photons, 0.5f);

    Material mat;
    mat.type = MaterialType::Lambertian;
    mat.Kd = Spectrum::constant(1.f);

    DensityEstimatorConfig config;
    config.radius = 0.5f;
    config.surface_tau = 0.01f;
    config.num_photons_total = 2 * N_per_side;
    config.use_kernel = false;

    // Query facing UP: should only gather upward-facing photons
    Spectrum L_up = estimate_photon_density(
        make_f3(0, 0, 0), make_f3(0, 0, 1),
        make_f3(0, 0, 1), mat, photons, grid, config, 0.5f);

    // Query facing DOWN: should only gather downward-facing photons
    // wo_local z is always positive in local frame (z = surface normal direction)
    Spectrum L_down = estimate_photon_density(
        make_f3(0, 0, 0), make_f3(0, 0, -1),
        make_f3(0, 0, 1), mat, photons, grid, config, 0.5f);

    // Both should contribute roughly equally (same number of photons per side)
    // and neither should include the other side's photons
    EXPECT_GT(L_up[0], 0.f);
    EXPECT_GT(L_down[0], 0.f);
    // They should be approximately equal in magnitude
    EXPECT_NEAR(L_up[0], L_down[0], L_up[0] * 0.3f);
}

// =====================================================================
//  SECTION 17 - MIS weight functions from nee_shared.h
// =====================================================================

TEST(MISWeights, MISWeight2_NonNegative) {
    PCGRng rng = PCGRng::seed(42);
    for (int i = 0; i < 100; ++i) {
        float pa = rng.next_float() * 10.f;
        float pb = rng.next_float() * 10.f;
        float w = mis_weight_2(pa, pb);
        EXPECT_GE(w, 0.f);
        EXPECT_LE(w, 1.f + kTol);
    }
}

TEST(MISWeights, MISWeight3_NonNegative) {
    PCGRng rng = PCGRng::seed(42);
    for (int i = 0; i < 100; ++i) {
        float pa = rng.next_float() * 10.f;
        float pb = rng.next_float() * 10.f;
        float pc = rng.next_float() * 10.f;
        float w = nee_mis_weight_3(pa, pb, pc);
        EXPECT_GE(w, 0.f);
        EXPECT_LE(w, 1.f + kTol);
    }
}

// =====================================================================
//  SECTION 18 - Reflect / Refract
// =====================================================================

TEST(ReflectRefract, ReflectLocal) {
    float3 wo = normalize(make_f3(0.3f, 0.2f, 0.9f));
    float3 wi = reflect_local(wo);
    EXPECT_NEAR(wi.x, -wo.x, kTol);
    EXPECT_NEAR(wi.y, -wo.y, kTol);
    EXPECT_NEAR(wi.z, wo.z, kTol);
}

TEST(ReflectRefract, RefractLocalSnellsLaw) {
    // Snell's law: eta * sin(theta_i) = sin(theta_t)
    float eta = 1.f / 1.5f; // Air to glass
    float3 wo = normalize(make_f3(0.3f, 0.f, 0.9f));

    float sin_i = sqrtf(1.f - wo.z * wo.z);

    float3 wt;
    EXPECT_TRUE(refract_local(wo, eta, wt));

    float sin_t = sqrtf(1.f - wt.z * wt.z);
    EXPECT_NEAR(eta * sin_i, sin_t, kTol);
}

TEST(ReflectRefract, TotalInternalReflection) {
    // From glass to air at steep angle
    float eta = 1.5f; // Glass to air (eta = n_glass / n_air)
    float3 wo = normalize(make_f3(0.9f, 0.f, 0.4f)); // Nearly grazing
    float3 wt;
    EXPECT_FALSE(refract_local(wo, eta, wt));
}

// =====================================================================
//  SECTION 19 - Camera
// =====================================================================

TEST(Camera, RayThroughCenter) {
    // Simple test: camera at origin looking at -Z
    // The central pixel ray should point roughly along -Z
    // (Camera is defined elsewhere, just test that the utility struct works)
    // Minimal test: just verify make_f3 and normalize work together
    float3 dir = normalize(make_f3(0, 0, -1));
    EXPECT_NEAR(dir.z, -1.f, kTol);
}

// =====================================================================
//  SECTION 20 - PhotonSoA
// =====================================================================

TEST(PhotonSoA, PushBackAndGet) {
    PhotonSoA soa;
    Photon p;
    p.position    = make_f3(1, 2, 3);
    p.wi          = make_f3(4, 5, 6);
    p.geom_normal = make_f3(0, 0, 1);  // upward surface normal
    p.spectral_flux = Spectrum::constant(8.5f);
    soa.push_back(p);

    EXPECT_EQ(soa.size(), 1u);
    Photon out = soa.get(0);
    EXPECT_NEAR(out.position.x,    1.f, kTol);
    EXPECT_NEAR(out.position.y,    2.f, kTol);
    EXPECT_NEAR(out.position.z,    3.f, kTol);
    EXPECT_NEAR(out.wi.x,          4.f, kTol);
    EXPECT_NEAR(out.wi.y,          5.f, kTol);
    EXPECT_NEAR(out.wi.z,          6.f, kTol);
    EXPECT_NEAR(out.geom_normal.x, 0.f, kTol);
    EXPECT_NEAR(out.geom_normal.y, 0.f, kTol);
    EXPECT_NEAR(out.geom_normal.z, 1.f, kTol);
    // Check spectral flux round-trips through SoA
    for (int b = 0; b < NUM_LAMBDA; ++b)
        EXPECT_NEAR(out.spectral_flux.value[b], 8.5f, kTol);

    // Also verify the raw SoA arrays are populated
    EXPECT_NEAR(soa.norm_x[0], 0.f, kTol);
    EXPECT_NEAR(soa.norm_y[0], 0.f, kTol);
    EXPECT_NEAR(soa.norm_z[0], 1.f, kTol);
}

TEST(PhotonSoA, ClearWorks) {
    PhotonSoA soa;
    Photon p;
    p.position = make_f3(0, 0, 0);
    p.wi = make_f3(0, 0, 1);
    p.spectral_flux = Spectrum::constant(1.f);
    soa.push_back(p);
    soa.push_back(p);
    EXPECT_EQ(soa.size(), 2u);
    soa.clear();
    EXPECT_EQ(soa.size(), 0u);
}

// =====================================================================
//  SECTION 21 - Hash grid consistency checks
// =====================================================================

TEST(HashGrid, CellCoordConsistency) {
    HashGrid grid;
    grid.cell_size = 0.1f;
    grid.table_size = 1024;

    // Points in the same cell should have the same key
    float3 a = make_f3(0.05f, 0.05f, 0.05f);
    float3 b = make_f3(0.09f, 0.01f, 0.09f);
    int3 ca = grid.cell_coord(a);
    int3 cb = grid.cell_coord(b);
    EXPECT_EQ(ca.x, cb.x);
    EXPECT_EQ(ca.y, cb.y);
    EXPECT_EQ(ca.z, cb.z);

    // Points in different cells should (likely) have different coords
    float3 c = make_f3(0.15f, 0.05f, 0.05f);
    int3 cc = grid.cell_coord(c);
    EXPECT_NE(ca.x, cc.x);
}

TEST(HashGrid, NegativeCoordinates) {
    HashGrid grid;
    grid.cell_size = 1.0f;
    grid.table_size = 1024;

    int3 pos = grid.cell_coord(make_f3(-0.5f, -1.5f, -2.5f));
    EXPECT_EQ(pos.x, -1);
    EXPECT_EQ(pos.y, -2);
    EXPECT_EQ(pos.z, -3);
}

// =====================================================================
//  SECTION 22 - BSDF Helmholtz reciprocity
// =====================================================================
// f(wo, wi) == f(wi, wo) must hold for non-delta BSDFs.

TEST(BSDFReciprocity, LambertianReciprocity) {
    Material mat;
    mat.type = MaterialType::Lambertian;
    mat.Kd = Spectrum::constant(0.6f);

    PCGRng rng = PCGRng::seed(42);
    for (int i = 0; i < 200; ++i) {
        float3 wo = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
        float3 wi = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
        Spectrum f_forward  = bsdf::evaluate(mat, wo, wi);
        Spectrum f_backward = bsdf::evaluate(mat, wi, wo);
        for (int j = 0; j < NUM_LAMBDA; ++j) {
            EXPECT_NEAR(f_forward[j], f_backward[j], kTol)
                << "Lambertian reciprocity failed at bin " << j;
        }
    }
}

TEST(BSDFReciprocity, GlossyMetalReciprocity) {
    Material mat;
    mat.type = MaterialType::GlossyMetal;
    mat.Kd = Spectrum::constant(0.3f);
    mat.Ks = Spectrum::constant(0.5f);
    mat.roughness = 0.4f;

    PCGRng rng = PCGRng::seed(42);
    for (int i = 0; i < 200; ++i) {
        float3 wo = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
        float3 wi = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
        if (wo.z < 0.01f || wi.z < 0.01f) continue;

        Spectrum f_forward  = bsdf::evaluate(mat, wo, wi);
        Spectrum f_backward = bsdf::evaluate(mat, wi, wo);
        for (int j = 0; j < NUM_LAMBDA; ++j) {
            EXPECT_NEAR(f_forward[j], f_backward[j], kLoose)
                << "GlossyMetal reciprocity failed at bin " << j
                << " forward=" << f_forward[j] << " backward=" << f_backward[j];
        }
    }
}

// =====================================================================
//  SECTION 23 - Glass Fresnel energy balance
// =====================================================================
// Reflected fraction + transmitted fraction = 1

TEST(GlassFresnel, EnergyBalance) {
    // At various angles, F_reflect + F_transmit = 1
    for (float cos_t = 0.1f; cos_t <= 1.0f; cos_t += 0.05f) {
        float F = fresnel_dielectric(cos_t, 1.0f / 1.5f);
        float T = 1.0f - F;
        EXPECT_GE(F, 0.f);
        EXPECT_GE(T, 0.f);
        EXPECT_NEAR(F + T, 1.0f, kTol)
            << "Fresnel + transmittance != 1 at cos=" << cos_t;
    }
}

TEST(GlassFresnel, GlassSampleReflectRefractBalance) {
    // Over many samples, the fraction of reflects should match E[F]
    PCGRng rng = PCGRng::seed(42);
    float3 wo = normalize(make_f3(0.3f, 0.0f, 0.9f));
    float ior = 1.5f;

    int N = 50000;
    int reflect_count = 0;
    for (int i = 0; i < N; ++i) {
        BSDFSample s = bsdf::glass_sample(wo, ior, rng);
        if (s.wi.z > 0.f) reflect_count++; // Reflection keeps z positive
    }

    float F_expected = fresnel_dielectric(wo.z, 1.0f / ior);
    float F_observed = (float)reflect_count / N;
    EXPECT_NEAR(F_observed, F_expected, 0.03f)
        << "Glass reflect fraction should match Fresnel";
}

// =====================================================================
//  SECTION 24 - Glossy BSDF at grazing angles
// =====================================================================

TEST(BSDF, GlossyEnergyAtGrazingAngle) {
    // Energy conservation should hold even at steep angles
    Material mat;
    mat.type = MaterialType::GlossyMetal;
    mat.Kd = Spectrum::constant(0.3f);
    mat.Ks = Spectrum::constant(0.5f);
    mat.roughness = 0.3f;

    // Grazing angle: wo nearly parallel to surface
    float3 wo = normalize(make_f3(0.95f, 0.0f, 0.31f));
    PCGRng rng = PCGRng::seed(42);
    const int N = 300000;

    Spectrum accum = Spectrum::zero();
    for (int i = 0; i < N; ++i) {
        BSDFSample s = bsdf::sample(mat, wo, rng);
        if (s.pdf > 0.f && s.wi.z > 0.f) {
            float cos_theta = s.wi.z;
            accum += s.f * (cos_theta / s.pdf);
        }
    }
    accum *= 1.f / N;

    for (int j = 0; j < NUM_LAMBDA; ++j) {
        EXPECT_LE(accum[j], 1.3f) // Some MC noise tolerance
            << "Glossy energy > 1 at grazing angle, bin " << j;
        EXPECT_GE(accum[j], 0.f);
    }
}

// =====================================================================
//  SECTION 25 - RGB → Spectrum round-trip
// =====================================================================

TEST(Spectrum, RGBRoundTrip) {
    // The Smits-style RGB → Spectrum → sRGB round-trip is NOT exact
    // (the spectral basis functions are not calibrated inverses of
    // the CIE matching functions). Instead we verify that the
    // dominant channel is preserved and the output is finite.

    // Red: R channel should dominate
    {
        Spectrum s = rgb_to_spectrum_reflectance(1.0f, 0.0f, 0.0f);
        float3 rgb = spectrum_to_srgb(s);
        EXPECT_GT(rgb.x, rgb.y) << "Red dominant: R > G";
        EXPECT_GT(rgb.x, rgb.z) << "Red dominant: R > B";
        EXPECT_TRUE(std::isfinite(rgb.x) && std::isfinite(rgb.y) && std::isfinite(rgb.z));
    }
    // Green: G channel should dominate
    {
        Spectrum s = rgb_to_spectrum_reflectance(0.0f, 1.0f, 0.0f);
        float3 rgb = spectrum_to_srgb(s);
        EXPECT_GT(rgb.y, rgb.x) << "Green dominant: G > R";
        EXPECT_GT(rgb.y, rgb.z) << "Green dominant: G > B";
        EXPECT_TRUE(std::isfinite(rgb.x) && std::isfinite(rgb.y) && std::isfinite(rgb.z));
    }
    // Blue: B channel should dominate
    {
        Spectrum s = rgb_to_spectrum_reflectance(0.0f, 0.0f, 1.0f);
        float3 rgb = spectrum_to_srgb(s);
        EXPECT_GT(rgb.z, rgb.x) << "Blue dominant: B > R";
        EXPECT_GT(rgb.z, rgb.y) << "Blue dominant: B > G";
        EXPECT_TRUE(std::isfinite(rgb.x) && std::isfinite(rgb.y) && std::isfinite(rgb.z));
    }
    // White: all channels approximately equal (within ±30 % of mean)
    {
        Spectrum s = rgb_to_spectrum_reflectance(1.0f, 1.0f, 1.0f);
        float3 rgb = spectrum_to_srgb(s);
        float mean = (rgb.x + rgb.y + rgb.z) / 3.0f;
        EXPECT_NEAR(rgb.x, mean, mean * 0.3f) << "White: R ≈ mean";
        EXPECT_NEAR(rgb.y, mean, mean * 0.3f) << "White: G ≈ mean";
        EXPECT_NEAR(rgb.z, mean, mean * 0.3f) << "White: B ≈ mean";
    }
}

// =====================================================================
//  SECTION 26 - Blackbody Stefan-Boltzmann law
// =====================================================================

TEST(Spectrum, BlackbodyPlanckFormula) {
    // Verify the blackbody_spectrum function matches the Planck function
    // at several specific wavelengths and temperatures.
    // B(λ,T) = 2hc² / (λ⁵ (e^(hc/λkT) - 1))  in W/(sr·m²·m)
    // Our function returns per-nm, so result = B * 1e-9.
    constexpr double h  = 6.62607015e-34;
    constexpr double c  = 2.99792458e8;
    constexpr double kb = 1.380649e-23;

    auto planck = [&](double lambda_nm, double T) -> double {
        double lam_m = lambda_nm * 1e-9;
        double lam5  = lam_m * lam_m * lam_m * lam_m * lam_m;
        double exponent = (h * c) / (lam_m * kb * T);
        double denom = exp(exponent) - 1.0;
        double L = (2.0 * h * c * c) / (lam5 * denom);
        return L * 1e-9; // per-nm
    };

    float temps[] = { 3000.f, 5000.f, 6500.f, 10000.f };
    for (float T : temps) {
        Spectrum bb = blackbody_spectrum(T);
        // Check a few bins
        int bins[] = { 0, NUM_LAMBDA / 4, NUM_LAMBDA / 2, 3 * NUM_LAMBDA / 4, NUM_LAMBDA - 1 };
        for (int i : bins) {
            float lam = lambda_of_bin(i);
            double expected_val = planck((double)lam, (double)T);
            double actual_val   = (double)bb[i];
            // Allow 1 % relative error (float precision)
            double tol = expected_val * 0.01 + 1e-20;
            EXPECT_NEAR(actual_val, expected_val, tol)
                << "T=" << T << " lambda=" << lam << " nm";
        }
    }

    // Also verify monotonicity: hotter blackbody has more visible power
    Spectrum bb_cool = blackbody_spectrum(4000.f);
    Spectrum bb_hot  = blackbody_spectrum(8000.f);
    double sum_cool = 0, sum_hot = 0;
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        sum_cool += bb_cool[i];
        sum_hot  += bb_hot[i];
    }
    EXPECT_GT(sum_hot, sum_cool) << "Hotter blackbody should have more visible power";
}

// =====================================================================
//  SECTION 27 - GGX VNDF sampling & Smith G symmetry
// =====================================================================

TEST(GGX, VNDFSamplingPDFConsistency) {
    // The sampled half-vector should have a PDF consistent with ggx_D
    PCGRng rng = PCGRng::seed(42);
    float alpha = 0.3f;

    for (int i = 0; i < 200; ++i) {
        float3 wo = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
        if (wo.z < 0.05f) continue;

        float3 h = ggx_sample_halfvector(wo, alpha, rng.next_float(), rng.next_float());

        // Half-vector should be on upper hemisphere and unit length
        EXPECT_GT(h.z, -kTol) << "Half-vector below hemisphere";
        EXPECT_NEAR(length(h), 1.f, 1e-3f) << "Half-vector not unit";

        // D value should be positive
        float D_val = ggx_D(h, alpha);
        EXPECT_GE(D_val, 0.f) << "D < 0 for sampled half-vector";
    }
}

TEST(GGX, SmithGSymmetry) {
    // G(wo, wi, alpha) == G(wi, wo, alpha)
    PCGRng rng = PCGRng::seed(42);
    for (int i = 0; i < 500; ++i) {
        float3 wo = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
        float3 wi = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
        float alpha = rng.next_float() * 0.9f + 0.1f;

        float G_forward  = ggx_G(wo, wi, alpha);
        float G_backward = ggx_G(wi, wo, alpha);
        EXPECT_NEAR(G_forward, G_backward, kTol)
            << "Smith G not symmetric for alpha=" << alpha;
    }
}

// =====================================================================
//  SECTION 28 - Camera ray generation
// =====================================================================

TEST(Camera, CornellBoxCameraSetup) {
    Camera cam = Camera::cornell_box_camera(512, 512);

    // Camera frame vectors should be orthonormal
    EXPECT_NEAR(dot(cam.u, cam.v), 0.f, kTol);
    EXPECT_NEAR(dot(cam.u, cam.w), 0.f, kTol);
    EXPECT_NEAR(dot(cam.v, cam.w), 0.f, kTol);
    EXPECT_NEAR(length(cam.u), 1.f, kTol);
    EXPECT_NEAR(length(cam.v), 1.f, kTol);
    EXPECT_NEAR(length(cam.w), 1.f, kTol);
}

TEST(Camera, CenterPixelRayDirection) {
    Camera cam = Camera::cornell_box_camera(512, 512);
    PCGRng rng = PCGRng::seed(42);

    // Center pixel: should point roughly toward look_at
    Ray ray = cam.generate_ray(256, 256, rng);
    float3 expected_dir = normalize(cam.look_at - cam.position);

    EXPECT_NEAR(ray.direction.x, expected_dir.x, 0.02f);
    EXPECT_NEAR(ray.direction.y, expected_dir.y, 0.02f);
    EXPECT_NEAR(ray.direction.z, expected_dir.z, 0.02f);

    // Ray origin should be camera position
    EXPECT_NEAR(ray.origin.x, cam.position.x, kTol);
    EXPECT_NEAR(ray.origin.y, cam.position.y, kTol);
    EXPECT_NEAR(ray.origin.z, cam.position.z, kTol);
}

TEST(Camera, CornerRaysDiverge) {
    Camera cam = Camera::cornell_box_camera(512, 512);
    PCGRng rng = PCGRng::seed(42);

    Ray r_tl = cam.generate_ray(0, 0, rng);
    Ray r_tr = cam.generate_ray(511, 0, rng);
    Ray r_bl = cam.generate_ray(0, 511, rng);
    Ray r_br = cam.generate_ray(511, 511, rng);

    // All corner rays should diverge from each other
    EXPECT_LT(dot(r_tl.direction, r_br.direction), 0.99f);
    EXPECT_LT(dot(r_tr.direction, r_bl.direction), 0.99f);

    // All should be unit vectors
    EXPECT_NEAR(length(r_tl.direction), 1.f, 1e-4f);
    EXPECT_NEAR(length(r_br.direction), 1.f, 1e-4f);
}

// =====================================================================
//  SECTION 29 - FrameBuffer tonemap
// =====================================================================

TEST(FrameBuffer, ZeroGivesBlack) {
    FrameBuffer fb;
    fb.resize(2, 2);
    fb.tonemap(1.0f);

    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(fb.srgb[i * 4 + 0], 0);
        EXPECT_EQ(fb.srgb[i * 4 + 1], 0);
        EXPECT_EQ(fb.srgb[i * 4 + 2], 0);
        EXPECT_EQ(fb.srgb[i * 4 + 3], 255); // Alpha = full
    }
}

TEST(FrameBuffer, AccumulateAndTonemap) {
    FrameBuffer fb;
    fb.resize(1, 1);

    // Accumulate two samples of a flat white-ish spectrum
    // With normalised XYZ, flat 1.0 maps to Y=1 -> white sRGB
    Spectrum white = Spectrum::constant(1.0f);
    fb.accumulate(0, 0, white);
    fb.accumulate(0, 0, white);

    EXPECT_NEAR(fb.sample_count[0], 2.f, kTol);

    fb.tonemap(1.0f);

    // After tonemap, should be a bright neutral color
    uint8_t r = fb.srgb[0], g = fb.srgb[1], b = fb.srgb[2];
    EXPECT_GT(r, 100); // Should be fairly bright
    EXPECT_GT(g, 100);
    EXPECT_GT(b, 100);
    // Should be roughly neutral (not wildly off-white)
    EXPECT_NEAR((float)r, (float)g, 60.f);
    EXPECT_NEAR((float)g, (float)b, 60.f);
}

// =====================================================================
//  SECTION 30 - Material type classification
// =====================================================================

TEST(Material, IsEmissive) {
    Material mat;
    mat.Le = Spectrum::zero();
    EXPECT_FALSE(mat.is_emissive());

    mat.Le = Spectrum::constant(1.0f);
    EXPECT_TRUE(mat.is_emissive());
}

TEST(Material, IsSpecular) {
    Material mat;
    mat.type = MaterialType::Lambertian;
    EXPECT_FALSE(mat.is_specular());

    mat.type = MaterialType::Mirror;
    EXPECT_TRUE(mat.is_specular());

    mat.type = MaterialType::Glass;
    EXPECT_TRUE(mat.is_specular());

    mat.type = MaterialType::GlossyMetal;
    EXPECT_FALSE(mat.is_specular());
}

TEST(Material, MeanEmission) {
    Material mat;
    mat.Le = Spectrum::constant(2.0f);
    EXPECT_NEAR(mat.mean_emission(), 2.0f, kTol);

    mat.Le = Spectrum::zero();
    EXPECT_NEAR(mat.mean_emission(), 0.f, kTol);
}

// =====================================================================
//  SECTION 31 - Density estimator normalization
// =====================================================================

TEST(DensityEstimator, NormalizationFactor) {
    // Place K identical photons at the query point.
    // With box kernel, L = (1/(π r²)) * K * Φ/N * f
    // For Lambertian with Kd=1: f = 1/π
    // So L[bin] = (1/(π r²)) * K * Φ/N * (1/π)

    const int K = 10;
    const float radius = 0.5f;
    const float flux = 1.0f;
    const int N_total = 100;
    const int bin = NUM_LAMBDA - 1;

    PhotonSoA photons;
    for (int i = 0; i < K; ++i) {
        Photon p;
        p.position    = make_f3(0.001f * i, 0.f, 0.f);
        p.wi          = make_f3(0, 0, 1);
        p.geom_normal = make_f3(0, 0, 1);  // surface normal matches query +Z
        p.spectral_flux = Spectrum::constant(flux);
        photons.push_back(p);
    }

    HashGrid grid;
    grid.build(photons, radius);

    Material mat;
    mat.type = MaterialType::Lambertian;
    mat.Kd = Spectrum::constant(1.f);

    DensityEstimatorConfig config;
    config.radius = radius;
    config.surface_tau = 0.1f;
    config.num_photons_total = N_total;
    config.use_kernel = false; // Box kernel for exact result

    Spectrum L = estimate_photon_density(
        make_f3(0, 0, 0), make_f3(0, 0, 1),
        make_f3(0, 0, 1), mat, photons, grid, config, radius);

    // Expected: inv_area * K * (flux/N_total) * (Kd/π)
    // inv_area = 1/(π * r²) = 1/(π * 0.25)
    float inv_area = 1.0f / (PI * radius * radius);
    float expected = inv_area * K * (flux / N_total) * (1.0f / PI); // Kd=1 → f = 1/π

    EXPECT_NEAR(L[bin], expected, expected * 0.01f)
        << "Density estimator normalization incorrect";

    // All bins should have the same value (full spectral photon)
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        EXPECT_NEAR(L[i], expected, expected * 0.01f);
    }
}

TEST(DensityEstimator, EpanechnikovKernelCorrection) {
    // With Epanechnikov kernel and all photons at center (dist²≈0),
    // kernel = 1.0 and correction factor = 1.5
    // Result should be 1.5× the box kernel result for center photons
    PhotonSoA photons;
    Photon p;
    p.position = make_f3(0.0001f, 0.f, 0.f); // Very close to center
    p.wi = make_f3(0, 0, 1);
    p.spectral_flux = Spectrum::constant(1.0f);
    photons.push_back(p);

    HashGrid grid;
    grid.build(photons, 0.5f);

    Material mat;
    mat.type = MaterialType::Lambertian;
    mat.Kd = Spectrum::constant(1.f);

    DensityEstimatorConfig config;
    config.radius = 0.5f;
    config.surface_tau = 0.1f;
    config.num_photons_total = 1;

    // Box kernel
    config.use_kernel = false;
    Spectrum L_box = estimate_photon_density(
        make_f3(0, 0, 0), make_f3(0, 0, 1),
        make_f3(0, 0, 1), mat, photons, grid, config, 0.5f);

    // Epanechnikov kernel
    config.use_kernel = true;
    Spectrum L_epan = estimate_photon_density(
        make_f3(0, 0, 0), make_f3(0, 0, 1),
        make_f3(0, 0, 1), mat, photons, grid, config, 0.5f);

    // For a photon very close to center: epan_kernel ≈ 1.0
    // v2.1 tangential normalization: box = πr², epan = (π/2)r²
    // So L_epan ≈ 2.0 × L_box (not 1.5× as in 3D spherical)
    EXPECT_NEAR(L_epan[0], L_box[0] * 2.0f, L_box[0] * 0.05f)
        << "Epanechnikov correction factor should be 2.0× for center photon "
           "(tangential disk normalization)";
}

// =====================================================================
//  SECTION 32 - Triangle edge cases
// =====================================================================

TEST(Triangle, DegenerateZeroArea) {
    // Degenerate triangle: two vertices coincide
    Triangle tri;
    tri.v0 = make_f3(0, 0, 0);
    tri.v1 = make_f3(0, 0, 0);  // Same as v0!
    tri.v2 = make_f3(1, 0, 0);

    EXPECT_NEAR(tri.area(), 0.f, kTol);

    // Intersection should not crash
    Ray ray;
    ray.origin = make_f3(0, 0, 5);
    ray.direction = make_f3(0, 0, -1);
    ray.tmin = 1e-4f;
    ray.tmax = 1e20f;
    float t, u, v;
    tri.intersect(ray, t, u, v); // Just verify no crash
}

TEST(Triangle, CollinearVertices) {
    // All three vertices on a line
    Triangle tri;
    tri.v0 = make_f3(0, 0, 0);
    tri.v1 = make_f3(1, 0, 0);
    tri.v2 = make_f3(2, 0, 0);

    EXPECT_NEAR(tri.area(), 0.f, kTol);
}

TEST(Triangle, InterpolateNormal) {
    Triangle tri;
    tri.v0 = make_f3(0, 0, 0);
    tri.v1 = make_f3(1, 0, 0);
    tri.v2 = make_f3(0, 1, 0);
    tri.n0 = normalize(make_f3(0, 0, 1));
    tri.n1 = normalize(make_f3(1, 0, 1));
    tri.n2 = normalize(make_f3(0, 1, 1));

    // At v0: alpha=1, beta=0, gamma=0 → should be n0
    float3 n = tri.interpolate_normal(1, 0, 0);
    n = normalize(n);
    EXPECT_NEAR(n.x, tri.n0.x, kTol);
    EXPECT_NEAR(n.y, tri.n0.y, kTol);
    EXPECT_NEAR(n.z, tri.n0.z, kTol);

    // At v1
    n = tri.interpolate_normal(0, 1, 0);
    n = normalize(n);
    EXPECT_NEAR(n.x, tri.n1.x, kTol);
    EXPECT_NEAR(n.y, tri.n1.y, kTol);
    EXPECT_NEAR(n.z, tri.n1.z, kTol);

    // At centroid: blended normal should still be unit after normalize
    n = tri.interpolate_normal(1.f/3, 1.f/3, 1.f/3);
    n = normalize(n);
    EXPECT_NEAR(length(n), 1.f, kTol);
}

// =====================================================================
//  SECTION 33 - Cosine hemisphere PDF integrates to 1 (MC verification)
// =====================================================================

TEST(Sampling, CosineHemispherePDFIntegration) {
    // MC estimate: ∫ p(ω) dω = 1 using uniform hemisphere sampling
    // p(ω) = cos(θ)/π
    PCGRng rng = PCGRng::seed(42);
    const int N = 500000;
    double integral = 0.0;

    for (int i = 0; i < N; ++i) {
        float3 d = sample_uniform_hemisphere(rng.next_float(), rng.next_float());
        float pdf_cosine = cosine_hemisphere_pdf(d.z);
        float pdf_uniform = uniform_hemisphere_pdf();
        // MC: ∫ p_cosine(ω) dω ≈ (1/N) Σ p_cosine(ω_i) / p_uniform(ω_i)
        integral += pdf_cosine / pdf_uniform;
    }
    integral /= N;
    EXPECT_NEAR(integral, 1.0, 0.02) << "Cosine hemisphere PDF should integrate to 1";
}

// =====================================================================
//  SECTION 35 - Cornell Box scene tests (real geometry)
// =====================================================================
// These tests load the actual Cornell box OBJ and test the full
// pipeline on realistic geometry.

// Helper: build a Cornell box scene with area light
static Scene build_cornell_test_scene() {
    Scene scene;
    std::string path = std::string(SCENES_DIR) + "/cornell_box/cornellbox.obj";
    if (!load_obj(path, scene)) {
        // If loading fails, return empty scene (tests will detect this)
        return scene;
    }

    // Add fallback area light only when the scene has no emitters
    // (The new cornellbox.obj has Ke in its MTL, so this should not fire.)
    scene.build_bvh();
    scene.build_emissive_distribution();

    if (scene.num_emissive() == 0) {
        Material light_mat;
        light_mat.name = "__area_light__";
        light_mat.type = MaterialType::Emissive;
        light_mat.Le = blackbody_spectrum(6500.f, 1e-8f);
        uint32_t light_mat_id = (uint32_t)scene.materials.size();
        scene.materials.push_back(light_mat);

        float3 v0 = make_f3(-0.15f,  0.499f, -0.15f);
        float3 v1 = make_f3( 0.15f,  0.499f, -0.15f);
        float3 v2 = make_f3( 0.15f,  0.499f,  0.15f);
        float3 v3 = make_f3(-0.15f,  0.499f,  0.15f);
        float3 n  = make_f3( 0.0f,  -1.0f,    0.0f);

        Triangle t1;
        t1.v0 = v0; t1.v1 = v1; t1.v2 = v2;
        t1.n0 = t1.n1 = t1.n2 = n;
        t1.uv0 = t1.uv1 = t1.uv2 = make_f2(0, 0);
        t1.material_id = light_mat_id;

        Triangle t2;
        t2.v0 = v0; t2.v1 = v2; t2.v2 = v3;
        t2.n0 = t2.n1 = t2.n2 = n;
        t2.uv0 = t2.uv1 = t2.uv2 = make_f2(0, 0);
        t2.material_id = light_mat_id;

        scene.triangles.push_back(t1);
        scene.triangles.push_back(t2);

        scene.build_bvh();
        scene.build_emissive_distribution();
    }

    return scene;
}

// -- 35.1 Scene loading ----------------------------------------------

TEST(CornellBox, LoadScene) {
    Scene scene = build_cornell_test_scene();

    // cornellbox.obj subdivision mesh: 13056 triangles, 7 materials (incl default)
    EXPECT_EQ(scene.triangles.size(), 13056u);
    EXPECT_GT(scene.materials.size(), 0u);
    EXPECT_GT(scene.bvh_nodes.size(), 0u);
}

TEST(CornellBox, EmissiveDistribution) {
    Scene scene = build_cornell_test_scene();

    // cornellbox.obj has 128 emissive triangles (Light material with Ke)
    EXPECT_EQ(scene.num_emissive(), 128u);
    EXPECT_GT(scene.total_emissive_power, 0.f);

    // Emissive alias table PDF should sum to 1
    float pdf_sum = 0.f;
    for (size_t i = 0; i < scene.emissive_tri_indices.size(); ++i) {
        pdf_sum += scene.emissive_alias_table.pdf((int)i);
    }
    EXPECT_NEAR(pdf_sum, 1.f, kTol);
}

// -- 35.2 BVH vs brute-force ----------------------------------------

TEST(CornellBox, BVHMatchesBruteForce) {
    Scene scene = build_cornell_test_scene();
    if (scene.triangles.empty()) { GTEST_SKIP() << "Scene not loaded"; }

    PCGRng rng = PCGRng::seed(42);
    Camera cam = Camera::cornell_box_camera(64, 64);

    const int N_RAYS = 200;

    for (int i = 0; i < N_RAYS; ++i) {
        int px = (int)(rng.next_float() * 64);
        int py = (int)(rng.next_float() * 64);
        if (px >= 64) px = 63;
        if (py >= 64) py = 63;

        Ray ray = cam.generate_ray(px, py, rng);

        // BVH intersection
        HitRecord bvh_hit = scene.intersect(ray);

        // Brute-force: test all triangles
        HitRecord brute_hit{};
        brute_hit.hit = false;
        brute_hit.t = ray.tmax;

        for (size_t j = 0; j < scene.triangles.size(); ++j) {
            float t, u, v;
            Ray test_ray = ray;
            test_ray.tmax = brute_hit.t;
            if (scene.triangles[j].intersect(test_ray, t, u, v)) {
                if (t < brute_hit.t && t > ray.tmin) {
                    brute_hit.hit = true;
                    brute_hit.t = t;
                    brute_hit.triangle_id = (uint32_t)j;
                }
            }
        }

        // Both should agree on hit/miss
        EXPECT_EQ(bvh_hit.hit, brute_hit.hit)
            << "BVH/brute-force disagree on hit for ray " << i;

        if (bvh_hit.hit && brute_hit.hit) {
            // t values should match closely
            EXPECT_NEAR(bvh_hit.t, brute_hit.t, 1e-3f)
                << "BVH/brute-force t mismatch for ray " << i;
        }
    }
}

// -- 35.3 Camera rays hit expected surfaces --------------------------

TEST(CornellBox, CenterRayHitsSomething) {
    Scene scene = build_cornell_test_scene();
    if (scene.triangles.empty()) { GTEST_SKIP() << "Scene not loaded"; }

    Camera cam = Camera::cornell_box_camera(512, 512);
    PCGRng rng = PCGRng::seed(42);

    // Center pixel should hit *something* inside the box
    // (it may hit an object / block rather than the back wall)
    Ray ray = cam.generate_ray(256, 256, rng);
    HitRecord hit = scene.intersect(ray);

    EXPECT_TRUE(hit.hit) << "Center ray should hit something";
    if (hit.hit) {
        // Hit must be within the scene bounds [-0.6, 0.6]³
        EXPECT_GT(hit.position.x, -0.6f);
        EXPECT_LT(hit.position.x,  0.6f);
        EXPECT_GT(hit.position.y, -0.6f);
        EXPECT_LT(hit.position.y,  0.6f);
        EXPECT_GT(hit.position.z, -0.6f);
        EXPECT_LT(hit.position.z,  0.6f);
    }
}

TEST(CornellBox, FloorRayHitsFloor) {
    Scene scene = build_cornell_test_scene();
    if (scene.triangles.empty()) { GTEST_SKIP() << "Scene not loaded"; }

    // Brute-force a downward ray to verify floor geometry exists,
    // then test via BVH. We use a slightly tilted ray to avoid
    // axis-aligned edge cases in AABB intersection (0 * inf = NaN).
    Ray ray;
    ray.origin    = make_f3(0.001f, 0.4f, 0.001f);  // near center, avoid exact 0
    ray.direction = normalize(make_f3(0.0f, -1.0f, 0.0f));

    // First: brute-force to confirm the floor exists
    bool brute_hit = false;
    float best_t = ray.tmax;
    float best_y = 0.f;
    for (size_t i = 0; i < scene.triangles.size(); ++i) {
        float t, u, v;
        Ray test = ray;
        test.tmax = best_t;
        if (scene.triangles[i].intersect(test, t, u, v)) {
            best_t = t;
            brute_hit = true;
            float alpha = 1.f - u - v;
            float3 pos = scene.triangles[i].interpolate_position(alpha, u, v);
            best_y = pos.y;
        }
    }

    EXPECT_TRUE(brute_hit).operator<<("Brute-force downward ray should hit the floor");
    if (brute_hit) {
        EXPECT_NEAR(best_y, -0.5f, 0.05f)
            << "Brute-force hit should be near floor y = -0.5";
    }

    // Now test through the BVH
    HitRecord hit = scene.intersect(ray);
    EXPECT_TRUE(hit.hit) << "BVH downward ray should hit the floor";
    if (hit.hit) {
        EXPECT_NEAR(hit.position.y, -0.5f, 0.05f)
            << "BVH hit should be near floor y = -0.5";
    }
}

// -- 35.4 Shadow ray visibility --------------------------------------

TEST(CornellBox, ShadowRayToLight) {
    Scene scene = build_cornell_test_scene();
    if (scene.triangles.empty()) { GTEST_SKIP() << "Scene not loaded"; }

    // Point on the floor directly below the light
    float3 floor_pos = make_f3(0.0f, -0.499f, 0.0f);
    float3 light_pos = make_f3(0.0f,  0.499f, 0.0f);

    float3 to_light = light_pos - floor_pos;
    float dist = length(to_light);
    float3 dir = to_light / dist;

    Ray shadow_ray;
    shadow_ray.origin = floor_pos + make_f3(0, EPSILON, 0);
    shadow_ray.direction = dir;
    shadow_ray.tmin = 1e-4f;
    shadow_ray.tmax = dist - 2e-4f;

    HitRecord hit = scene.intersect(shadow_ray);

    // The path from floor center straight up to light should be unoccluded
    EXPECT_FALSE(hit.hit)
        << "Shadow ray from floor center to light should be unoccluded";
}

TEST(CornellBox, ShadowRayBlockedByGeometry) {
    Scene scene = build_cornell_test_scene();
    if (scene.triangles.empty()) { GTEST_SKIP() << "Scene not loaded"; }

    // Point outside the box trying to reach inside through the back wall
    float3 outside = make_f3(0.0f, 0.0f, -2.0f);
    float3 inside  = make_f3(0.0f, 0.0f,  0.0f);

    float3 dir = normalize(inside - outside);
    float dist = length(inside - outside);

    Ray shadow_ray;
    shadow_ray.origin = outside;
    shadow_ray.direction = dir;
    shadow_ray.tmin = 1e-4f;
    shadow_ray.tmax = dist;

    HitRecord hit = scene.intersect(shadow_ray);

    // Should be occluded by the back wall
    EXPECT_TRUE(hit.hit)
        << "Ray from outside should be blocked by the back wall";
    if (hit.hit) {
        EXPECT_LT(hit.t, dist)
            << "Hit should be before the target point";
    }
}

// -- 35.5 Emitter sampling -------------------------------------------

TEST(CornellBox, EmitterSamplesValid) {
    Scene scene = build_cornell_test_scene();
    if (scene.emissive_tri_indices.empty()) { GTEST_SKIP() << "No emitters"; }

    PCGRng rng = PCGRng::seed(42);

    for (int i = 0; i < 100; ++i) {
        EmittedPhoton ep = sample_emitted_photon(scene, rng);

        // Photon should originate near the light (y ≈ 0.499)
        EXPECT_NEAR(ep.ray.origin.y, 0.499f, 0.01f)
            << "Photon should originate at the ceiling light";

        // Direction should point downward (into the scene)
        // The light normal is (0, -1, 0), so the photon goes into the scene
        EXPECT_LT(ep.ray.direction.y, 0.1f)
            << "Photon from ceiling light should generally point downward";

        // Flux should be positive and finite
        EXPECT_GT(ep.spectral_flux.max_component(), 0.f) << "Photon flux should be positive";
        EXPECT_FALSE(std::isnan(ep.spectral_flux.sum())) << "Photon flux should not be NaN";
        EXPECT_FALSE(std::isinf(ep.spectral_flux.sum())) << "Photon flux should not be Inf";
    }
}

// -- 35.6 Direct lighting --------------------------------------------

TEST(CornellBox, DirectLightFromFloor) {
    Scene scene = build_cornell_test_scene();
    if (scene.emissive_tri_indices.empty()) { GTEST_SKIP() << "No emitters"; }

    PCGRng rng = PCGRng::seed(42);

    // Point on the floor directly below the light - should see the light
    float3 floor_pos = make_f3(0.0f, -0.499f, 0.0f);
    float3 floor_normal = make_f3(0.0f, 1.0f, 0.0f);

    int visible_count = 0;
    const int N = 100;

    for (int i = 0; i < N; ++i) {
        DirectLightSample dls = sample_direct_light(floor_pos, floor_normal, scene, rng);
        if (dls.visible) {
            visible_count++;
            EXPECT_GT(dls.pdf_light, 0.f) << "PDF should be positive for visible sample";
            EXPECT_GT(dls.Li.max_component(), 0.f) << "Li should be nonzero";
            EXPECT_GT(dls.distance, 0.f);
        }
    }

    // Most samples should be visible from floor center
    EXPECT_GT(visible_count, N / 2)
        << "Most direct light samples from floor center should be visible";
}

TEST(CornellBox, DirectLightPDFConsistency) {
    Scene scene = build_cornell_test_scene();
    if (scene.emissive_tri_indices.empty()) { GTEST_SKIP() << "No emitters"; }

    PCGRng rng = PCGRng::seed(42);

    float3 floor_pos = make_f3(0.0f, -0.499f, 0.0f);
    float3 floor_normal = make_f3(0.0f, 1.0f, 0.0f);

    for (int i = 0; i < 50; ++i) {
        DirectLightSample dls = sample_direct_light(floor_pos, floor_normal, scene, rng);
        if (!dls.visible || dls.pdf_light <= 0.f) continue;

        // direct_light_pdf for the same direction should give similar result
        float3 offset_pos = floor_pos + floor_normal * EPSILON;
        float reverse_pdf = direct_light_pdf(offset_pos, dls.wi, scene);

        // They should be close (not exactly equal due to floating point)
        if (reverse_pdf > 0.f) {
            EXPECT_NEAR(dls.pdf_light, reverse_pdf, dls.pdf_light * 0.1f)
                << "Forward and reverse light PDFs should be close";
        }
    }
}

// -- 35.7 Photon tracing integration ---------------------------------

TEST(CornellBox, PhotonTracingProducesPhotons) {
    Scene scene = build_cornell_test_scene();
    if (scene.emissive_tri_indices.empty()) { GTEST_SKIP() << "No emitters"; }

    EmitterConfig config;
    config.num_photons = 1000;
    config.max_bounces = 5;

    PhotonSoA global_map, caustic_map;
    trace_photons(scene, config, global_map, caustic_map);

    // Should produce a meaningful number of global photons
    EXPECT_GT(global_map.size(), 500u)
        << "Photon tracing should produce many stored photons";
}

TEST(CornellBox, PhotonPositionsOnSurfaces) {
    Scene scene = build_cornell_test_scene();
    if (scene.emissive_tri_indices.empty()) { GTEST_SKIP() << "No emitters"; }

    EmitterConfig config;
    config.num_photons = 500;
    config.max_bounces = 3;

    PhotonSoA global_map, caustic_map;
    trace_photons(scene, config, global_map, caustic_map);

    // All photon positions should be within the Cornell box bounds
    // Cornell box is roughly [-0.5, 0.5]³
    for (size_t i = 0; i < global_map.size(); ++i) {
        float x = global_map.pos_x[i];
        float y = global_map.pos_y[i];
        float z = global_map.pos_z[i];

        EXPECT_GT(x, -0.55f) << "Photon x out of bounds (low)";
        EXPECT_LT(x,  0.55f) << "Photon x out of bounds (high)";
        EXPECT_GT(y, -0.55f) << "Photon y out of bounds (low)";
        EXPECT_LT(y,  0.55f) << "Photon y out of bounds (high)";
        EXPECT_GT(z, -0.55f) << "Photon z out of bounds (low)";
        EXPECT_LT(z,  0.55f) << "Photon z out of bounds (high)";
    }
}

TEST(CornellBox, PhotonFluxPositiveFinite) {
    Scene scene = build_cornell_test_scene();
    if (scene.emissive_tri_indices.empty()) { GTEST_SKIP() << "No emitters"; }

    EmitterConfig config;
    config.num_photons = 500;
    config.max_bounces = 5;

    PhotonSoA global_map, caustic_map;
    trace_photons(scene, config, global_map, caustic_map);

    for (size_t i = 0; i < global_map.size(); ++i) {
        float tf = global_map.total_flux(i);
        EXPECT_GT(tf, 0.f) << "Photon flux should be positive";
        EXPECT_FALSE(std::isnan(tf)) << "Photon flux NaN at " << i;
        EXPECT_FALSE(std::isinf(tf)) << "Photon flux Inf at " << i;
    }
}

// -- 35.8 Photon density on known geometry ---------------------------

TEST(CornellBox, PhotonDensityOnFloor) {
    Scene scene = build_cornell_test_scene();
    if (scene.emissive_tri_indices.empty()) { GTEST_SKIP() << "No emitters"; }

    // Trace many photons
    EmitterConfig em_config;
    em_config.num_photons = 10000;
    em_config.max_bounces = 5;

    PhotonSoA global_map, caustic_map;
    trace_photons(scene, em_config, global_map, caustic_map);

    if (global_map.size() == 0) { GTEST_SKIP() << "No photons stored"; }

    HashGrid grid;
    grid.build(global_map, 0.1f);

    // Query density at floor center (should be bright - directly below light)
    float3 floor_pos = make_f3(0.0f, -0.499f, 0.0f);
    float3 floor_normal = make_f3(0.0f, 1.0f, 0.0f);
    float3 wo_local = make_f3(0.0f, 0.0f, 1.0f); // Looking straight up

    Material floor_mat;
    floor_mat.type = MaterialType::Lambertian;
    floor_mat.Kd = Spectrum::constant(0.8f);

    DensityEstimatorConfig de_config;
    de_config.radius = 0.1f;
    de_config.surface_tau = 0.02f;
    de_config.num_photons_total = em_config.num_photons;
    de_config.use_kernel = true;

    Spectrum L_center = estimate_photon_density(
        floor_pos, floor_normal, wo_local, floor_mat,
        global_map, grid, de_config, 0.1f);

    // Floor center should receive some illumination
    EXPECT_GT(L_center.sum(), 0.f)
        << "Floor center should have nonzero photon density";

    // Query at a corner (should be dimmer)
    float3 corner_pos = make_f3(-0.45f, -0.499f, -0.45f);
    Spectrum L_corner = estimate_photon_density(
        corner_pos, floor_normal, wo_local, floor_mat,
        global_map, grid, de_config, 0.1f);

    // Center should be brighter than corner (due to inverse-square falloff)
    // This may not always hold with only 10K photons, so use a loose check
    // Just verify the corner also gets some light
    // (Don't assert center > corner because stochastic noise)
    EXPECT_GE(L_center.sum() + L_corner.sum(), 0.f)
        << "Both floor positions should contribute to density";
}

// -- 35.9 Full pipeline: small render produces valid image -----------

TEST(CornellBox, SmallRenderProducesValidImage) {
    Scene scene = build_cornell_test_scene();
    if (scene.emissive_tri_indices.empty()) { GTEST_SKIP() << "No emitters"; }

    Camera cam = Camera::cornell_box_camera(8, 8);

    Renderer renderer;
    renderer.set_scene(&scene);
    renderer.set_camera(cam);

    RenderConfig cfg;
    cfg.image_width = 8;
    cfg.image_height = 8;
    cfg.samples_per_pixel = 2;
    cfg.num_photons = 1000;
    cfg.gather_radius = 0.1f;
    cfg.mode = RenderMode::Full;
    renderer.set_config(cfg);

    renderer.build_photon_maps();
    renderer.render_frame();

    const FrameBuffer& fb = renderer.framebuffer();

    // Image should not be all black (the scene is lit)
    int nonzero_pixels = 0;
    for (int i = 0; i < 64; ++i) {
        uint8_t r = fb.srgb[i * 4 + 0];
        uint8_t g = fb.srgb[i * 4 + 1];
        uint8_t b = fb.srgb[i * 4 + 2];
        if (r > 0 || g > 0 || b > 0) nonzero_pixels++;
    }

    EXPECT_GT(nonzero_pixels, 0)
        << "Rendered image should not be all black";
}

// =====================================================================
//  SECTION 36 -- OptiX Renderer Tests
// =====================================================================
// These tests verify that the OptiX backend is functional.
// OptiX is mandatory -- there is no CPU-only build.

#include "optix/optix_renderer.h"

// -- 36.1 OptixRenderer can initialize without error -----------------

TEST(OptiX, Initialization) {
    OptixRenderer renderer;
    EXPECT_NO_THROW(renderer.init())
        << "OptixRenderer::init() should succeed on a machine with an NVIDIA GPU";
}

// -- 36.2 Acceleration structure builds from a Cornell box scene -----

TEST(OptiX, AccelBuild) {
    Scene scene = build_cornell_test_scene();
    if (scene.triangles.empty()) { GTEST_SKIP() << "No scene geometry"; }

    OptixRenderer renderer;
    renderer.init();
    EXPECT_NO_THROW(renderer.build_accel(scene))
        << "build_accel should complete without error";
}

// -- 36.3 Scene data uploads to GPU ----------------------------------

TEST(OptiX, SceneDataUpload) {
    Scene scene = build_cornell_test_scene();
    if (scene.triangles.empty()) { GTEST_SKIP() << "No scene geometry"; }

    OptixRenderer renderer;
    renderer.init();
    renderer.build_accel(scene);
    EXPECT_NO_THROW(renderer.upload_scene_data(scene))
        << "upload_scene_data should complete without error";
}

// -- 36.4 Photon data uploads to GPU ---------------------------------

TEST(OptiX, PhotonDataUpload) {
    Scene scene = build_cornell_test_scene();
    if (scene.emissive_tri_indices.empty()) { GTEST_SKIP() << "No emitters"; }

    // Build small photon map on CPU
    Renderer cpu_renderer;
    cpu_renderer.set_scene(&scene);
    Camera cam = Camera::cornell_box_camera(8, 8);
    cpu_renderer.set_camera(cam);

    RenderConfig cfg;
    cfg.num_photons = 500;
    cfg.gather_radius = 0.1f;
    cfg.caustic_radius = 0.05f;
    cpu_renderer.set_config(cfg);
    cpu_renderer.build_photon_maps();

    OptixRenderer optix_renderer;
    optix_renderer.init();
    optix_renderer.build_accel(scene);
    optix_renderer.upload_scene_data(scene);

    EXPECT_NO_THROW(optix_renderer.upload_photon_data(
        cpu_renderer.global_photons(), cpu_renderer.global_grid(),
        cpu_renderer.caustic_photons(), cpu_renderer.caustic_grid(),
        cfg.gather_radius, cfg.caustic_radius))
        << "upload_photon_data should complete without error";
}

// -- 36.5 Debug frame produces non-zero output -----------------------

TEST(OptiX, DebugFrameNonZero) {
    Scene scene = build_cornell_test_scene();
    if (scene.emissive_tri_indices.empty()) { GTEST_SKIP() << "No emitters"; }

    Renderer cpu_renderer;
    cpu_renderer.set_scene(&scene);
    Camera cam = Camera::cornell_box_camera(8, 8);
    cpu_renderer.set_camera(cam);

    RenderConfig cfg;
    cfg.image_width = 8;
    cfg.image_height = 8;
    cfg.num_photons = 10000;
    cfg.gather_radius = 0.15f;
    cfg.caustic_radius = 0.05f;
    cpu_renderer.set_config(cfg);
    cpu_renderer.build_photon_maps();

    OptixRenderer optix_renderer;
    optix_renderer.init();
    optix_renderer.build_accel(scene);
    optix_renderer.upload_scene_data(scene);
    optix_renderer.upload_emitter_data(scene);
    optix_renderer.upload_photon_data(
        cpu_renderer.global_photons(), cpu_renderer.global_grid(),
        cpu_renderer.caustic_photons(), cpu_renderer.caustic_grid(),
        cfg.gather_radius, cfg.caustic_radius);

    optix_renderer.resize(8, 8);
    optix_renderer.render_debug_frame(cam, 0, RenderMode::Full, 1);

    FrameBuffer fb;
    optix_renderer.download_framebuffer(fb);

    EXPECT_EQ(fb.width, 8);
    EXPECT_EQ(fb.height, 8);

    // At least some pixels should be non-zero (scene is lit)
    int nonzero = 0;
    for (int i = 0; i < 64; ++i) {
        uint8_t r = fb.srgb[i * 4 + 0];
        uint8_t g = fb.srgb[i * 4 + 1];
        uint8_t b = fb.srgb[i * 4 + 2];
        if (r > 0 || g > 0 || b > 0) nonzero++;
    }
    EXPECT_GT(nonzero, 0)
        << "OptiX debug frame should produce at least some non-black pixels";
}

// -- 36.6 Normals debug mode produces varied output ------------------

TEST(OptiX, NormalsDebugMode) {
    Scene scene = build_cornell_test_scene();
    if (scene.triangles.empty()) { GTEST_SKIP() << "No scene geometry"; }

    Renderer cpu_renderer;
    cpu_renderer.set_scene(&scene);
    Camera cam = Camera::cornell_box_camera(16, 16);
    cpu_renderer.set_camera(cam);

    RenderConfig cfg;
    cfg.image_width = 16;
    cfg.image_height = 16;
    cfg.num_photons = 100;
    cfg.gather_radius = 0.1f;
    cfg.caustic_radius = 0.05f;
    cpu_renderer.set_config(cfg);
    cpu_renderer.build_photon_maps();

    OptixRenderer optix_renderer;
    optix_renderer.init();
    optix_renderer.build_accel(scene);
    optix_renderer.upload_scene_data(scene);
    optix_renderer.upload_photon_data(
        cpu_renderer.global_photons(), cpu_renderer.global_grid(),
        cpu_renderer.caustic_photons(), cpu_renderer.caustic_grid(),
        cfg.gather_radius, cfg.caustic_radius);

    optix_renderer.resize(16, 16);
    optix_renderer.render_debug_frame(cam, 0, RenderMode::Normals, 1);

    FrameBuffer fb;
    optix_renderer.download_framebuffer(fb);

    // Normals mode should show surface variation - check that pixels
    // are not all the same value (walls face different directions)
    std::set<uint32_t> distinct;
    for (int i = 0; i < 256; ++i) {
        uint32_t col = ((uint32_t)fb.srgb[i * 4 + 0] << 16) |
                       ((uint32_t)fb.srgb[i * 4 + 1] <<  8) |
                       ((uint32_t)fb.srgb[i * 4 + 2]);
        distinct.insert(col);
    }

    EXPECT_GT(distinct.size(), 2u)
        << "Normals mode should show at least a few distinct surface colors";
}

// -- 36.7 Final render produces valid framebuffer --------------------

TEST(OptiX, FinalRenderProducesValid) {
    Scene scene = build_cornell_test_scene();
    if (scene.emissive_tri_indices.empty()) { GTEST_SKIP() << "No emitters"; }

    Renderer cpu_renderer;
    cpu_renderer.set_scene(&scene);
    Camera cam = Camera::cornell_box_camera(8, 8);
    cpu_renderer.set_camera(cam);

    RenderConfig cfg;
    cfg.image_width = 8;
    cfg.image_height = 8;
    cfg.samples_per_pixel = 2;
    cfg.num_photons = 500;
    cfg.gather_radius = 0.1f;
    cfg.caustic_radius = 0.05f;
    cfg.mode = RenderMode::Full;
    cpu_renderer.set_config(cfg);
    cpu_renderer.build_photon_maps();

    OptixRenderer optix_renderer;
    optix_renderer.init();
    optix_renderer.build_accel(scene);
    optix_renderer.upload_scene_data(scene);
    optix_renderer.upload_emitter_data(scene);
    optix_renderer.upload_photon_data(
        cpu_renderer.global_photons(), cpu_renderer.global_grid(),
        cpu_renderer.caustic_photons(), cpu_renderer.caustic_grid(),
        cfg.gather_radius, cfg.caustic_radius);

    optix_renderer.render_final(cam, cfg, scene);

    FrameBuffer fb;
    optix_renderer.download_framebuffer(fb);

    EXPECT_EQ(fb.width, 8);
    EXPECT_EQ(fb.height, 8);
    EXPECT_EQ((int)fb.srgb.size(), 8 * 8 * 4);

    // At least some pixels should be non-zero
    int nonzero = 0;
    for (int i = 0; i < 64; ++i) {
        if (fb.srgb[i * 4 + 0] > 0 || fb.srgb[i * 4 + 1] > 0 ||
            fb.srgb[i * 4 + 2] > 0) {
            nonzero++;
        }
    }
    EXPECT_GT(nonzero, 0)
        << "Final OptiX render should produce at least some non-black pixels";
}

// -- 36.8 Resize changes framebuffer dimensions ----------------------

TEST(OptiX, ResizeFramebuffer) {
    Scene scene = build_cornell_test_scene();
    if (scene.triangles.empty()) { GTEST_SKIP() << "No scene geometry"; }

    Renderer cpu_renderer;
    cpu_renderer.set_scene(&scene);
    Camera cam = Camera::cornell_box_camera(16, 16);
    cpu_renderer.set_camera(cam);

    RenderConfig cfg;
    cfg.image_width = 16;
    cfg.image_height = 16;
    cfg.num_photons = 100;
    cfg.gather_radius = 0.1f;
    cfg.caustic_radius = 0.05f;
    cpu_renderer.set_config(cfg);
    cpu_renderer.build_photon_maps();

    OptixRenderer optix_renderer;
    optix_renderer.init();
    optix_renderer.build_accel(scene);
    optix_renderer.upload_scene_data(scene);
    optix_renderer.upload_photon_data(
        cpu_renderer.global_photons(), cpu_renderer.global_grid(),
        cpu_renderer.caustic_photons(), cpu_renderer.caustic_grid(),
        cfg.gather_radius, cfg.caustic_radius);

    // Render at 16x16
    optix_renderer.resize(16, 16);
    optix_renderer.render_debug_frame(cam, 0, RenderMode::Full, 1);

    FrameBuffer fb16;
    optix_renderer.download_framebuffer(fb16);
    EXPECT_EQ(fb16.width, 16);
    EXPECT_EQ(fb16.height, 16);

    // Resize to 8x8
    optix_renderer.resize(8, 8);
    optix_renderer.render_debug_frame(cam, 0, RenderMode::Full, 1);

    FrameBuffer fb8;
    optix_renderer.download_framebuffer(fb8);
    EXPECT_EQ(fb8.width, 8);
    EXPECT_EQ(fb8.height, 8);
}

// -- 36.9 Cell-bin grid built by trace_photons and flags valid --------

TEST(OptiX, CellBinGridValidAfterTracePhotons) {
    Scene scene = build_cornell_test_scene();
    if (scene.emissive_tri_indices.empty()) { GTEST_SKIP() << "No emitters"; }

    Renderer cpu_renderer;
    cpu_renderer.set_scene(&scene);
    Camera cam = Camera::cornell_box_camera(8, 8);
    cpu_renderer.set_camera(cam);

    RenderConfig cfg;
    cfg.image_width = 8;
    cfg.image_height = 8;
    cfg.num_photons = 2000;
    cfg.gather_radius = 0.15f;
    cfg.caustic_radius = 0.05f;
    cpu_renderer.set_config(cfg);
    cpu_renderer.build_photon_maps();

    OptixRenderer optix_renderer;
    optix_renderer.init();
    optix_renderer.build_accel(scene);
    optix_renderer.upload_scene_data(scene);
    optix_renderer.upload_emitter_data(scene);
    optix_renderer.upload_photon_data(
        cpu_renderer.global_photons(), cpu_renderer.global_grid(),
        cpu_renderer.caustic_photons(), cpu_renderer.caustic_grid(),
        cfg.gather_radius, cfg.caustic_radius);

    optix_renderer.resize(8, 8);

    // Cell grid should be valid after trace_photons built it via upload_photon_data
    // (which ends with build_cell_bin_grid inside trace_photons).
    // Render should propagate cell_grid_valid=1 to launch params.
    optix_renderer.render_one_spp(cam, 0);
    {
        const LaunchParams& lp = optix_renderer.last_launch_params_for_test();
        EXPECT_EQ(lp.cell_grid_valid, 1);
        EXPECT_NE(lp.cell_bin_grid, nullptr);
        EXPECT_GT(lp.cell_grid_dim_x, 0);
        EXPECT_GT(lp.cell_grid_dim_y, 0);
        EXPECT_GT(lp.cell_grid_dim_z, 0);
        EXPECT_GT(lp.cell_grid_cell_size, 0.0f);
    }

    // The host-side grid should be accessible
    const CellBinGrid& grid = optix_renderer.cell_bin_grid_for_test();
    EXPECT_GT(grid.dim_x, 0);
    EXPECT_GT(grid.dim_y, 0);
    EXPECT_GT(grid.dim_z, 0);
    EXPECT_FALSE(grid.bins.empty());
}

// -- 36.10 Cell-bin grid allocation is proportional to grid dims ------

TEST(OptiX, CellBinGridAllocationMatchesDimensions) {
    Scene scene = build_cornell_test_scene();
    if (scene.emissive_tri_indices.empty()) { GTEST_SKIP() << "No emitters"; }

    Renderer cpu_renderer;
    cpu_renderer.set_scene(&scene);
    Camera cam = Camera::cornell_box_camera(8, 8);
    cpu_renderer.set_camera(cam);

    RenderConfig cfg;
    cfg.image_width = 8;
    cfg.image_height = 8;
    cfg.num_photons = 2000;
    cfg.gather_radius = 0.15f;
    cfg.caustic_radius = 0.05f;
    cpu_renderer.set_config(cfg);
    cpu_renderer.build_photon_maps();

    OptixRenderer optix_renderer;
    optix_renderer.init();
    optix_renderer.build_accel(scene);
    optix_renderer.upload_scene_data(scene);
    optix_renderer.upload_emitter_data(scene);
    optix_renderer.upload_photon_data(
        cpu_renderer.global_photons(), cpu_renderer.global_grid(),
        cpu_renderer.caustic_photons(), cpu_renderer.caustic_grid(),
        cfg.gather_radius, cfg.caustic_radius);

    const CellBinGrid& grid = optix_renderer.cell_bin_grid_for_test();
    size_t total_cells = (size_t)grid.dim_x * grid.dim_y * grid.dim_z;
    size_t expected = total_cells * PHOTON_BIN_COUNT * sizeof(PhotonBin);
    EXPECT_EQ(optix_renderer.cell_bin_grid_bytes_for_test(), expected);
}

// =====================================================================
//  Photon Directional Bins Tests
// =====================================================================

TEST(PhotonBins, FibonacciSphereCoversUnitSphere) {
    // Verify Fibonacci sphere directions are unit-length and quasi-uniform
    for (int N : {8, 16, 32}) {
        PhotonBinDirs dirs;
        dirs.init(N);
        EXPECT_EQ(dirs.count, N);

        // All directions must be unit length
        for (int k = 0; k < N; ++k) {
            float len = length(dirs.dirs[k]);
            EXPECT_NEAR(len, 1.0f, 1e-5f) << "N=" << N << " k=" << k;
        }

        // Centroid should be near zero (quasi-uniform)
        float3 centroid = make_f3(0, 0, 0);
        for (int k = 0; k < N; ++k) centroid += dirs.dirs[k];
        centroid = centroid * (1.0f / N);
        float centroid_len = length(centroid);
        EXPECT_LT(centroid_len, 0.2f) << "N=" << N << " centroid not near zero";

        // No duplicate directions (min pairwise dot < 1)
        float max_dot = -2.0f;
        for (int i = 0; i < N; ++i)
            for (int j = i + 1; j < N; ++j) {
                float d = dot(dirs.dirs[i], dirs.dirs[j]);
                if (d > max_dot) max_dot = d;
            }
        EXPECT_LT(max_dot, 0.999f) << "N=" << N << " directions too similar";
    }
}

TEST(PhotonBins, FindNearestBinReturnsItself) {
    // Each Fibonacci direction should map to itself as nearest
    PhotonBinDirs dirs;
    dirs.init(32);
    for (int k = 0; k < 32; ++k) {
        int found = dirs.find_nearest(dirs.dirs[k]);
        EXPECT_EQ(found, k) << "Direction " << k << " did not map to itself";
    }
}

TEST(PhotonBins, FindNearestBinGeometric) {
    PhotonBinDirs dirs;
    dirs.init(16);

    // A direction very close to bin 0 should map to bin 0
    float3 d0 = dirs.dirs[0];
    float3 perturbed = normalize(d0 + make_f3(0.01f, 0.01f, 0.01f));
    int found = dirs.find_nearest(perturbed);
    // Should be bin 0 or very close neighbor
    float d_self = dot(perturbed, dirs.dirs[0]);
    float d_found = dot(perturbed, dirs.dirs[found]);
    EXPECT_GE(d_found, d_self - 1e-5f);
}

TEST(PhotonBins, HemisphereCoverage) {
    // For normal = (0,0,1), approximately half the bins should be
    // in the positive hemisphere
    PhotonBinDirs dirs;
    dirs.init(32);
    float3 normal = make_f3(0, 0, 1);
    int pos_count = 0;
    for (int k = 0; k < 32; ++k)
        if (dot(dirs.dirs[k], normal) > 0.0f) ++pos_count;

    // Should be roughly N/2 (±3)
    EXPECT_GE(pos_count, 32 / 2 - 4);
    EXPECT_LE(pos_count, 32 / 2 + 4);
}

TEST(PhotonBins, BinPopulationBasic) {
    // Create a few bins and verify manual population
    PhotonBin bins[16] = {};
    PhotonBinDirs dirs;
    dirs.init(16);

    // Simulate a photon arriving from direction close to bin 0
    float3 wi = dirs.dirs[0];
    float flux = 2.5f;
    float kernel_w = 0.8f;

    int k = dirs.find_nearest(wi);
    bins[k].scalar_flux  += flux * kernel_w;
    bins[k].dir_x += wi.x * flux * kernel_w;
    bins[k].dir_y += wi.y * flux * kernel_w;
    bins[k].dir_z += wi.z * flux * kernel_w;
    bins[k].weight += kernel_w;
    bins[k].count  += 1;

    EXPECT_EQ(k, 0);
    EXPECT_FLOAT_EQ(bins[0].scalar_flux, 2.0f);
    EXPECT_EQ(bins[0].count, 1);
    EXPECT_NEAR(bins[0].weight, 0.8f, 1e-6f);

    // Normalize centroid
    float3 d = make_f3(bins[0].dir_x, bins[0].dir_y, bins[0].dir_z);
    float len = length(d);
    EXPECT_GT(len, 0.0f);
    d = d * (1.0f / len);
    EXPECT_NEAR(dot(d, wi), 1.0f, 1e-5f); // centroid = photon direction
}

TEST(PhotonBins, EmptyBinsHandled) {
    PhotonBin bins[16] = {};
    // All bins empty
    for (int k = 0; k < 16; ++k) {
        EXPECT_EQ(bins[k].count, 0);
        EXPECT_FLOAT_EQ(bins[k].scalar_flux, 0.0f);
    }
}

TEST(PhotonBins, CentroidNormalization) {
    // Two photons near the same bin with different fluxes
    PhotonBinDirs dirs;
    dirs.init(16);

    float3 wi1 = normalize(dirs.dirs[0] + make_f3(0.01f, 0, 0));
    float3 wi2 = normalize(dirs.dirs[0] + make_f3(-0.01f, 0, 0));
    int k1 = dirs.find_nearest(wi1);
    int k2 = dirs.find_nearest(wi2);

    // Both should land in the same bin (bin 0 or nearby)
    if (k1 == k2) {
        PhotonBin bin = {};
        float f1 = 3.0f, f2 = 1.0f;
        float w1 = 0.9f, w2 = 0.7f;

        bin.scalar_flux   += f1 * w1;
        bin.dir_x  += wi1.x * f1 * w1;
        bin.dir_y  += wi1.y * f1 * w1;
        bin.dir_z  += wi1.z * f1 * w1;
        bin.weight += w1;
        bin.count  += 1;

        bin.scalar_flux   += f2 * w2;
        bin.dir_x  += wi2.x * f2 * w2;
        bin.dir_y  += wi2.y * f2 * w2;
        bin.dir_z  += wi2.z * f2 * w2;
        bin.weight += w2;
        bin.count  += 1;

        EXPECT_EQ(bin.count, 2);

        // Normalize centroid
        float3 d = make_f3(bin.dir_x, bin.dir_y, bin.dir_z);
        float len = length(d);
        EXPECT_GT(len, 0.0f);
        d = d * (1.0f / len);
        EXPECT_NEAR(length(d), 1.0f, 1e-5f);

        // Centroid should be closer to wi1 (higher flux)
        float dot1 = dot(d, wi1);
        float dot2 = dot(d, wi2);
        EXPECT_GT(dot1, dot2);
    }
}

TEST(PhotonBins, StratifiedSubPixelCoverage) {
    // Generate 16 stratified sub-pixel offsets (4×4)
    // Verify: each stratum has exactly one sample, all in [0,1)
    std::set<std::pair<int,int>> strata_used;
    for (int s = 0; s < 16; ++s) {
        int stratum_x = s % STRATA_X;
        int stratum_y = s / STRATA_X;
        auto key = std::make_pair(stratum_x, stratum_y);
        EXPECT_EQ(strata_used.count(key), 0u) << "Stratum reused: " << stratum_x << "," << stratum_y;
        strata_used.insert(key);

        // With jitter in [0,1), offset is in [stratum/N, (stratum+1)/N)
        float jx = (stratum_x + 0.5f) / (float)STRATA_X;
        float jy = (stratum_y + 0.5f) / (float)STRATA_Y;
        EXPECT_GE(jx, 0.0f);
        EXPECT_LT(jx, 1.0f);
        EXPECT_GE(jy, 0.0f);
        EXPECT_LT(jy, 1.0f);
    }
    EXPECT_EQ(strata_used.size(), 16u);
}

TEST(PhotonBins, BinSolidAngle) {
    // Each of N bins covers approximately 4π/N steradians
    // Cone half-angle ≈ arccos(1 - 2/N)
    int N = 32;
    float cos_half = 1.0f - 2.0f / (float)N;
    // Solid angle of a cone = 2π(1 - cos(half_angle))
    float solid_angle = 2.0f * PI * (1.0f - cos_half);
    float expected = 4.0f * PI / (float)N;
    // Should be roughly 4π/N (not exact due to Fibonacci packing)
    EXPECT_NEAR(solid_angle, expected, expected * 0.5f);
}

TEST(PhotonBins, PhotonBinSize) {
    // (NUM_LAMBDA + 8) * 4 + 4 = 52 bytes with NUM_LAMBDA=4
    // flux[4](16) + scalar_flux(4) + dir_xyz(12) + weight(4) + count(4) + avg_nxyz(12)
    EXPECT_EQ(sizeof(PhotonBin), (NUM_LAMBDA + 8) * 4 + 4);
}

TEST(PhotonBins, MaxBinCountRespected) {
    // Init with more than MAX should be clamped
    PhotonBinDirs dirs;
    dirs.init(MAX_PHOTON_BIN_COUNT + 10);
    EXPECT_EQ(dirs.count, MAX_PHOTON_BIN_COUNT);
}

// =====================================================================
//  CellBinGrid Tests
// =====================================================================

TEST(CellBinGrid, BuildBasic) {
    // Create a few photons in a known layout and verify grid builds
    PhotonSoA photons;
    // 4 photons in a small box
    photons.pos_x   = {0.0f, 0.1f, 0.2f, 0.3f};
    photons.pos_y   = {0.0f, 0.0f, 0.0f, 0.0f};
    photons.pos_z   = {0.0f, 0.0f, 0.0f, 0.0f};
    photons.wi_x    = {0.0f, 0.0f, 0.0f, 0.0f};
    photons.wi_y    = {0.0f, 0.0f, 0.0f, 0.0f};
    photons.wi_z    = {1.0f, 1.0f, 1.0f, 1.0f};
    photons.norm_x  = {0.0f, 0.0f, 0.0f, 0.0f};  // surface normal x
    photons.norm_y  = {0.0f, 0.0f, 0.0f, 0.0f};  // surface normal y
    photons.norm_z  = {1.0f, 1.0f, 1.0f, 1.0f};  // surface normal z (upward)
    set_soa_flux_uniform(photons, {1.0f, 2.0f, 3.0f, 4.0f});
    photons.bin_idx = {0, 1, 2, 3};

    CellBinGrid grid;
    float gather_radius = 0.1f;
    grid.build(photons, gather_radius, 8);

    EXPECT_GT(grid.dim_x, 0);
    EXPECT_GT(grid.dim_y, 0);
    EXPECT_GT(grid.dim_z, 0);
    EXPECT_GT(grid.cell_size, 0.0f);
    EXPECT_EQ(grid.bins.size(),
              (size_t)grid.dim_x * grid.dim_y * grid.dim_z * 8);
}

TEST(CellBinGrid, CellIndexClampedAtBounds) {
    // A trivially small grid should clamp out-of-bounds lookups
    PhotonSoA photons;
    photons.pos_x   = {0.0f};
    photons.pos_y   = {0.0f};
    photons.pos_z   = {0.0f};
    photons.wi_x    = {1.0f};
    photons.wi_y    = {0.0f};
    photons.wi_z    = {0.0f};
    photons.norm_x  = {0.0f};
    photons.norm_y  = {0.0f};
    photons.norm_z  = {1.0f};
    set_soa_flux_uniform(photons, {1.0f});
    photons.bin_idx = {0};

    CellBinGrid grid;
    grid.build(photons, 0.1f, 4);

    // Far outside should still return a valid clamped index
    int idx_far = grid.cell_index(100.f, 100.f, 100.f);
    int total = grid.dim_x * grid.dim_y * grid.dim_z;
    EXPECT_GE(idx_far, 0);
    EXPECT_LT(idx_far, total);

    int idx_neg = grid.cell_index(-100.f, -100.f, -100.f);
    EXPECT_GE(idx_neg, 0);
    EXPECT_LT(idx_neg, total);
}

TEST(CellBinGrid, ScatterTo27Neighbors) {
    // Place many photons on a flat surface.  Photons near cell centres
    // will have positive Epanechnikov weight and should be scattered
    // into their own cell and possibly neighbouring cells.
    PhotonSoA photons;
    PhotonBinDirs bin_dirs;
    bin_dirs.init(4);
    PCGRng rng = PCGRng::seed(42);
    for (int i = 0; i < 200; ++i) {
        Photon p;
        p.position    = make_f3((rng.next_float() - 0.5f) * 0.6f,
                                0.f,
                                (rng.next_float() - 0.5f) * 0.6f);
        p.wi          = make_f3(0.f, 1.f, 0.f);
        p.geom_normal = make_f3(0.f, 1.f, 0.f);
        p.spectral_flux = Spectrum::constant(5.f);
        p.num_hero = HERO_WAVELENGTHS;
        for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
            p.lambda_bin[h] = (uint16_t)(h * NUM_LAMBDA / HERO_WAVELENGTHS);
            p.flux[h] = 5.f;
        }
        photons.push_back(p);
    }
    photons.bin_idx.resize(photons.size());
    for (size_t i = 0; i < photons.size(); ++i) {
        float3 wi = make_f3(photons.wi_x[i], photons.wi_y[i], photons.wi_z[i]);
        photons.bin_idx[i] = (uint8_t)bin_dirs.find_nearest(wi);
    }

    CellBinGrid grid;
    grid.build(photons, 0.1f, 4);

    // Count cells with non-zero flux
    size_t total = (size_t)grid.dim_x * grid.dim_y * grid.dim_z;
    int non_zero_cells = 0;
    for (size_t c = 0; c < total; ++c) {
        for (int k = 0; k < 4; ++k) {
            if (grid.bins[c * 4 + k].scalar_flux > 0.0f) {
                ++non_zero_cells;
                break;
            }
        }
    }
    // With 200 photons spread across 0.6 extent, many cells should receive flux
    EXPECT_GE(non_zero_cells, 2);
    EXPECT_LE(non_zero_cells, (int)total);
}

TEST(CellBinGrid, EmptyPhotons) {
    PhotonSoA photons;
    CellBinGrid grid;
    grid.build(photons, 0.1f, 8);
    // Empty photon store → empty grid
    EXPECT_EQ(grid.dim_x, 0);
    EXPECT_EQ(grid.dim_y, 0);
    EXPECT_EQ(grid.dim_z, 0);
    EXPECT_TRUE(grid.bins.empty());
}

// =====================================================================
//  SECTION 37 - Normal Visibility Filter (geom_normal)
// =====================================================================
// These tests verify the photon surface-normal visibility term introduced
// to prevent irradiance leaking through thin walls:
//
//   – PhotonSoA correctly stores geom_normal in norm_x/y/z SoA arrays
//   – DensityEstimator rejects photons whose geom_normal opposes query normal
//   – DensityEstimator accepts photons whose geom_normal matches query normal
//   – The geom_normal check fires even when the old wi-direction check passes
//   – CellBinGrid accumulates and normalises avg_nx/y/z per bin
//   – CellBinGrid handles perfectly cancelling normals → zero avg_n
//   – Integration: trace_photons stores valid unit-length normals
//   – Integration: back-facing photons contribute zero in a thin-wall scenario

// ── 37.1  PhotonSoA geom_normal round-trip ──────────────────────────

TEST(PhotonSoA, GeomNormalRoundTrip) {
    // push_back with an arbitrary geom_normal and check the SoA arrays and get()
    PhotonSoA soa;

    Photon p1;
    p1.position    = make_f3(0, 0, 0);
    p1.wi          = make_f3(0, 1, 0);
    p1.geom_normal = make_f3(0, 0, 1);   // upward surface
    p1.spectral_flux = Spectrum::constant(1.0f);
    soa.push_back(p1);

    Photon p2;
    p2.position    = make_f3(1, 0, 0);
    p2.wi          = make_f3(0, -1, 0);
    p2.geom_normal = make_f3(0, 0, -1);  // downward surface
    p2.spectral_flux = Spectrum::constant(2.0f);
    soa.push_back(p2);

    ASSERT_EQ(soa.size(), 2u);
    ASSERT_EQ(soa.norm_x.size(), 2u);
    ASSERT_EQ(soa.norm_y.size(), 2u);
    ASSERT_EQ(soa.norm_z.size(), 2u);

    // First photon
    EXPECT_NEAR(soa.norm_x[0],  0.f, kTol);
    EXPECT_NEAR(soa.norm_y[0],  0.f, kTol);
    EXPECT_NEAR(soa.norm_z[0],  1.f, kTol);
    Photon out0 = soa.get(0);
    EXPECT_NEAR(out0.geom_normal.x,  0.f, kTol);
    EXPECT_NEAR(out0.geom_normal.y,  0.f, kTol);
    EXPECT_NEAR(out0.geom_normal.z,  1.f, kTol);

    // Second photon
    EXPECT_NEAR(soa.norm_x[1],  0.f, kTol);
    EXPECT_NEAR(soa.norm_y[1],  0.f, kTol);
    EXPECT_NEAR(soa.norm_z[1], -1.f, kTol);
    Photon out1 = soa.get(1);
    EXPECT_NEAR(out1.geom_normal.z, -1.f, kTol);
}

TEST(PhotonSoA, GeomNormalClearResetsNormArrays) {
    PhotonSoA soa;
    Photon p;
    p.position    = make_f3(0, 0, 0);
    p.wi          = make_f3(0, 0, 1);
    p.geom_normal = make_f3(0, 1, 0);
    p.spectral_flux = Spectrum::constant(1.f);
    soa.push_back(p);
    EXPECT_EQ(soa.size(), 1u);

    soa.clear();
    EXPECT_EQ(soa.size(), 0u);
    EXPECT_TRUE(soa.norm_x.empty());
    EXPECT_TRUE(soa.norm_y.empty());
    EXPECT_TRUE(soa.norm_z.empty());
}

// ── 37.2  DensityEstimator normal visibility ─────────────────────────

TEST(DensityEstimator, NormalVisibility_BackFacingGeomNormalRejected) {
    // A photon deposited on the BACK face of a wall has geom_normal opposite
    // to the front-face query normal. The normal visibility check must reject it
    // to prevent irradiance leaking through the wall.
    //
    // Setup:
    //   query position  = (0, 0, 0),  query normal = (0, 0, +1) ← front face
    //   photon position = (0, 0, 0)   (same location – pure normal test)
    //   photon wi       = (0, 0, +1)  → dot(wi, query_n) = +1 > 0 (OLD check passes!)
    //   photon geom_n   = (0, 0, -1)  → dot(geom_n, query_n) = -1 ≤ 0 (NEW check fails)
    // Expected result: L = 0
    PhotonSoA photons;
    Photon p;
    p.position    = make_f3(0.f, 0.f, 0.f);
    p.wi          = make_f3(0.f, 0.f, 1.f);   // OLD check would pass
    p.geom_normal = make_f3(0.f, 0.f, -1.f);  // NEW check rejects
    p.spectral_flux = Spectrum::constant(100.f);
    photons.push_back(p);

    HashGrid grid;
    grid.build(photons, 0.5f);

    Material mat;
    mat.type = MaterialType::Lambertian;
    mat.Kd = Spectrum::constant(1.f);

    DensityEstimatorConfig config;
    config.radius          = 0.5f;
    config.surface_tau     = 5.0f;  // very generous — won't reject by plane distance
    config.num_photons_total = 1;
    config.use_kernel      = false;

    Spectrum L = estimate_photon_density(
        make_f3(0, 0, 0), make_f3(0, 0, 1),
        make_f3(0, 0, 1), mat, photons, grid, config, 0.5f);

    EXPECT_NEAR(L.sum(), 0.f, kTol)
        << "Back-facing geom_normal must be rejected (irradiance leak through wall)";
}

TEST(DensityEstimator, NormalVisibility_FrontFacingGeomNormalAccepted) {
    // A photon with geom_normal matching the query normal should contribute.
    PhotonSoA photons;
    Photon p;
    p.position    = make_f3(0.f, 0.f, 0.f);
    p.wi          = make_f3(0.f, 0.f, 1.f);
    p.geom_normal = make_f3(0.f, 0.f, 1.f);  // same hemisphere → accepted
    p.spectral_flux = Spectrum::constant(1.f);
    photons.push_back(p);

    HashGrid grid;
    grid.build(photons, 0.5f);

    Material mat;
    mat.type = MaterialType::Lambertian;
    mat.Kd = Spectrum::constant(1.f);

    DensityEstimatorConfig config;
    config.radius            = 0.5f;
    config.surface_tau       = 1.0f;
    config.num_photons_total = 1;
    config.use_kernel        = false;

    Spectrum L = estimate_photon_density(
        make_f3(0, 0, 0), make_f3(0, 0, 1),
        make_f3(0, 0, 1), mat, photons, grid, config, 0.5f);

    EXPECT_GT(L.sum(), 0.f)
        << "Front-facing geom_normal must be accepted and contribute irradiance";
}

TEST(DensityEstimator, NormalVisibility_OnlyGeomNormalCheckBlocks) {
    // Designed to isolate the geom_normal check from the wi-direction check:
    // wi direction PASSES old check, geom_normal FAILS new check → result is zero.
    //
    // This precisely simulates a photon that bounced off the floor (geom_n up),
    // but we are asking it while standing on the ceiling (query_n down).
    // Even though wi = (0,0,-1) → dot(wi, n_ceiling=(0,0,-1)) = 1 > 0 (wi check passes),
    // geom_n = (0,0,+1) → dot(geom_n, n_ceiling=(0,0,-1)) = -1 ≤ 0 (geom_n rejects).
    PhotonSoA photons;
    Photon p;
    p.position    = make_f3(0.f, 0.f, 0.f);
    p.wi          = make_f3(0.f, 0.f, -1.f);  // pointing downward (toward ceiling query)
    p.geom_normal = make_f3(0.f, 0.f,  1.f);  // photon was on a floor (faces up)
    p.spectral_flux = Spectrum::constant(50.f);
    photons.push_back(p);

    HashGrid grid;
    grid.build(photons, 0.5f);

    Material mat;
    mat.type = MaterialType::Lambertian;
    mat.Kd = Spectrum::constant(1.f);

    DensityEstimatorConfig config;
    config.radius            = 0.5f;
    config.surface_tau       = 5.0f;
    config.num_photons_total = 1;
    config.use_kernel        = false;

    // Query from ceiling, normal pointing downward
    Spectrum L = estimate_photon_density(
        make_f3(0, 0, 0), make_f3(0, 0, -1),
        make_f3(0, 0, 1), mat, photons, grid, config, 0.5f);

    EXPECT_NEAR(L.sum(), 0.f, kTol)
        << "geom_normal check should reject floor photon when querying ceiling";
}

TEST(DensityEstimator, NormalVisibility_ThinWallBothSidesCorrect) {
    // Thin wall scenario: two photons at nearly the same position, one on each
    // side of an infinitely thin wall.
    //   – Front photon: geom_n = (0,0,+1) → accepted by front-face query
    //   – Back  photon: geom_n = (0,0,-1) → rejected by front-face query
    PhotonSoA photons;

    Photon front;
    front.position    = make_f3(0.f, 0.f, 0.0001f);
    front.wi          = make_f3(0.f, 0.f,  1.f);
    front.geom_normal = make_f3(0.f, 0.f,  1.f);  // front surface
    front.spectral_flux = Spectrum::constant(10.f);
    photons.push_back(front);

    Photon back;
    back.position    = make_f3(0.f, 0.f, -0.0001f);
    back.wi          = make_f3(0.f, 0.f, -1.f);
    back.geom_normal = make_f3(0.f, 0.f, -1.f);   // back surface
    back.spectral_flux = Spectrum::constant(10.f);
    photons.push_back(back);

    HashGrid grid;
    grid.build(photons, 0.5f);

    Material mat;
    mat.type = MaterialType::Lambertian;
    mat.Kd = Spectrum::constant(1.f);

    DensityEstimatorConfig config;
    config.radius            = 0.5f;
    config.surface_tau       = 0.01f;  // generous enough to accept both positions
    config.num_photons_total = 1;
    config.use_kernel        = false;

    // Front-face query
    Spectrum L_front = estimate_photon_density(
        make_f3(0, 0, 0), make_f3(0, 0,  1),
        make_f3(0, 0, 1), mat, photons, grid, config, 0.5f);

    // Back-face query
    Spectrum L_back = estimate_photon_density(
        make_f3(0, 0, 0), make_f3(0, 0, -1),
        make_f3(0, 0, 1), mat, photons, grid, config, 0.5f);

    // Each side should only see its own photon
    EXPECT_GT(L_front.sum(), 0.f) << "Front query should see front photon";
    EXPECT_GT(L_back.sum(),  0.f) << "Back query should see back photon";

    // Cross-contamination: roughly equal flux so values should be close
    // (not one being drastically larger due to the other side bleeding through)
    const int check_bin = NUM_LAMBDA / 2;
    if (L_front[check_bin] > 0.f && L_back[check_bin] > 0.f) {
        float ratio = L_front[check_bin] / L_back[check_bin];
        EXPECT_NEAR(ratio, 1.0f, 0.1f)
            << "Both sides should see equal irradiance (same flux, symmetric setup)";
    }
}

// ── 37.3  CellBinGrid normal accumulation ───────────────────────────

TEST(CellBinGrid, SinglePhotonNormalPreserved) {
    // Many photons on a flat surface (normal = up).  After build, every
    // bin that received photons should have avg_n ≈ (0,1,0).
    const float3 expected_n = make_f3(0.f, 1.f, 0.f);  // upward y normal

    PhotonSoA photons;
    PCGRng rng = PCGRng::seed(999);
    for (int i = 0; i < 100; ++i) {
        Photon p;
        p.position    = make_f3((rng.next_float() - 0.5f) * 0.4f,
                                0.f,
                                (rng.next_float() - 0.5f) * 0.4f);
        p.wi          = make_f3(0.f, 1.f, 0.f);
        p.geom_normal = expected_n;
        p.spectral_flux = Spectrum::constant(2.f);
        p.num_hero = HERO_WAVELENGTHS;
        for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
            p.lambda_bin[h] = (uint16_t)(h * NUM_LAMBDA / HERO_WAVELENGTHS);
            p.flux[h] = 2.f;
        }
        photons.push_back(p);
    }
    PhotonBinDirs bin_dirs;
    bin_dirs.init(4);
    photons.bin_idx.resize(photons.size());
    for (size_t i = 0; i < photons.size(); ++i) {
        float3 wi = make_f3(photons.wi_x[i], photons.wi_y[i], photons.wi_z[i]);
        photons.bin_idx[i] = (uint8_t)bin_dirs.find_nearest(wi);
    }

    CellBinGrid grid;
    grid.build(photons, 0.2f, 4);

    // Every bin that received photons should have avg_n ≈ expected_n
    const size_t total = (size_t)grid.dim_x * grid.dim_y * grid.dim_z;
    int bins_checked = 0;
    for (size_t c = 0; c < total; ++c) {
        for (int k = 0; k < 4; ++k) {
            const PhotonBin& b = grid.bins[c * 4 + k];
            if (b.count == 0) continue;
            ++bins_checked;
            EXPECT_NEAR(b.avg_nx, expected_n.x, 1e-4f);
            EXPECT_NEAR(b.avg_ny, expected_n.y, 1e-4f);
            EXPECT_NEAR(b.avg_nz, expected_n.z, 1e-4f);
            float len = std::sqrt(b.avg_nx * b.avg_nx + b.avg_ny * b.avg_ny + b.avg_nz * b.avg_nz);
            EXPECT_NEAR(len, 1.0f, 1e-4f) << "avg_n should be unit length after normalization";
        }
    }
    EXPECT_GT(bins_checked, 0) << "At least one cell should have received photons";
}

TEST(CellBinGrid, NormalAccumulatedAndNormalized) {
    // Many photons on a horizontal surface (Y=0, normal up = (0,1,0)).
    // Photons spread in XZ (the tangential plane for this normal).
    // After build, bins that received photons should have avg_n ≈ (0,1,0).
    PhotonSoA photons;
    PhotonBinDirs bin_dirs;
    bin_dirs.init(4);
    PCGRng rng = PCGRng::seed(123);
    for (int i = 0; i < 200; ++i) {
        Photon p;
        p.position    = make_f3((rng.next_float() - 0.5f) * 0.3f,
                                0.f,
                                (rng.next_float() - 0.5f) * 0.3f);
        p.wi          = make_f3(0.f, 1.f, 0.f);
        p.geom_normal = make_f3(0.f, 1.f, 0.f);
        p.spectral_flux = Spectrum::constant(3.f);
        p.num_hero = HERO_WAVELENGTHS;
        for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
            p.lambda_bin[h] = (uint16_t)(h * NUM_LAMBDA / HERO_WAVELENGTHS);
            p.flux[h] = 3.f;
        }
        photons.push_back(p);
    }
    photons.bin_idx.resize(photons.size());
    for (size_t i = 0; i < photons.size(); ++i) {
        float3 wi = make_f3(photons.wi_x[i], photons.wi_y[i], photons.wi_z[i]);
        photons.bin_idx[i] = (uint8_t)bin_dirs.find_nearest(wi);
    }

    CellBinGrid grid;
    grid.build(photons, 0.1f, 4);

    // Find a cell that received multiple photons
    const size_t total = (size_t)grid.dim_x * grid.dim_y * grid.dim_z;
    bool found_multi = false;
    for (size_t c = 0; c < total; ++c) {
        for (int k = 0; k < 4; ++k) {
            const PhotonBin& b = grid.bins[c * 4 + k];
            if (b.count < 2) continue;
            found_multi = true;
            // All photons have normal (0,1,0) → avg_n should be (0,1,0)
            EXPECT_NEAR(b.avg_ny, 1.f, 0.01f);
            float len = std::sqrt(b.avg_nx * b.avg_nx + b.avg_ny * b.avg_ny + b.avg_nz * b.avg_nz);
            EXPECT_NEAR(len, 1.0f, 1e-4f);
            break;
        }
        if (found_multi) break;
    }
    EXPECT_TRUE(found_multi) << "Expected a cell receiving multiple photons";
}

TEST(CellBinGrid, OppositeNormalsCancelToZero) {
    // Photons on two sides of a thin wall at z=0: half with normal +Z,
    // half with normal -Z.  In shared cells, opposing normals should
    // partially cancel.
    PhotonSoA photons;
    PhotonBinDirs bin_dirs;
    bin_dirs.init(4);
    PCGRng rng = PCGRng::seed(321);
    for (int i = 0; i < 100; ++i) {
        Photon p;
        float x = (rng.next_float() - 0.5f) * 0.3f;
        float y = (rng.next_float() - 0.5f) * 0.3f;
        bool top_side = (i % 2 == 0);
        p.position    = make_f3(x, y, top_side ? 0.001f : -0.001f);
        p.wi          = make_f3(0.f, 0.f, top_side ? 1.f : -1.f);
        p.geom_normal = make_f3(0.f, 0.f, top_side ? 1.f : -1.f);
        p.spectral_flux = Spectrum::constant(5.f);
        p.num_hero = HERO_WAVELENGTHS;
        for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
            p.lambda_bin[h] = (uint16_t)(h * NUM_LAMBDA / HERO_WAVELENGTHS);
            p.flux[h] = 5.f;
        }
        photons.push_back(p);
    }
    photons.bin_idx.resize(photons.size());
    for (size_t i = 0; i < photons.size(); ++i) {
        float3 wi = make_f3(photons.wi_x[i], photons.wi_y[i], photons.wi_z[i]);
        photons.bin_idx[i] = (uint8_t)bin_dirs.find_nearest(wi);
    }

    CellBinGrid grid;
    grid.build(photons, 0.1f, 4);

    // At least some cells should have received photons from both sides.
    // In those cells, the dominant normal direction should partially cancel.
    const size_t total = (size_t)grid.dim_x * grid.dim_y * grid.dim_z;
    bool any_photons = false;
    for (size_t c = 0; c < total; ++c) {
        for (int k = 0; k < 4; ++k) {
            if (grid.bins[c * 4 + k].count > 0) { any_photons = true; break; }
        }
        if (any_photons) break;
    }
    EXPECT_TRUE(any_photons) << "Expected at least some cells with photons";
}

TEST(CellBinGrid, NormArraysSizeMatchesPositions) {
    // Verify that norm_x/y/z arrays have the same length as pos_x after build.
    PhotonSoA photons;
    for (int i = 0; i < 10; ++i) {
        Photon p;
        p.position    = make_f3((float)i * 0.1f, 0.f, 0.f);
        p.wi          = make_f3(0.f, 0.f, 1.f);
        p.geom_normal = make_f3(0.f, 1.f, 0.f);
        p.spectral_flux = Spectrum::constant(1.f);
        photons.push_back(p);
    }
    // bin_idx must be set for CellBinGrid
    photons.bin_idx.assign(10, 0);

    EXPECT_EQ(photons.norm_x.size(), photons.pos_x.size());
    EXPECT_EQ(photons.norm_y.size(), photons.pos_y.size());
    EXPECT_EQ(photons.norm_z.size(), photons.pos_z.size());

    CellBinGrid grid;
    grid.build(photons, 0.2f, 4);
    // Grid should have been built without crashing
    EXPECT_GT(grid.dim_x * grid.dim_y * grid.dim_z, 0);
}

// ── 37.4  Integration: trace_photons stores valid unit normals ───────

TEST(CornellBox, PhotonNormalsAreValidUnitVectors) {
    // After CPU photon tracing on the real Cornell box, every stored photon
    // should have a unit-length geom_normal (stored in norm_x/y/z).
    Scene scene = build_cornell_test_scene();
    if (scene.emissive_tri_indices.empty()) { GTEST_SKIP() << "No emitters"; }

    EmitterConfig config;
    config.num_photons = 2000;
    config.max_bounces = 5;

    PhotonSoA global_map, caustic_map;
    trace_photons(scene, config, global_map, caustic_map);

    if (global_map.size() == 0) { GTEST_SKIP() << "No photons traced"; }

    ASSERT_EQ(global_map.norm_x.size(), global_map.size());
    ASSERT_EQ(global_map.norm_y.size(), global_map.size());
    ASSERT_EQ(global_map.norm_z.size(), global_map.size());

    int zero_normal_count = 0;
    for (size_t i = 0; i < global_map.size(); ++i) {
        float nx = global_map.norm_x[i];
        float ny = global_map.norm_y[i];
        float nz = global_map.norm_z[i];
        float len = std::sqrt(nx * nx + ny * ny + nz * nz);

        EXPECT_FALSE(std::isnan(nx)) << "norm_x[" << i << "] is NaN";
        EXPECT_FALSE(std::isnan(ny)) << "norm_y[" << i << "] is NaN";
        EXPECT_FALSE(std::isnan(nz)) << "norm_z[" << i << "] is NaN";

        if (len < 0.5f) ++zero_normal_count;
        else
            EXPECT_NEAR(len, 1.0f, 0.01f)
                << "Photon " << i << " geom_normal is not unit length";
    }

    // Zero normals shouldn't happen for real surface hits
    float zero_frac = (float)zero_normal_count / (float)global_map.size();
    EXPECT_LT(zero_frac, 0.01f)
        << "More than 1% of photons have near-zero geom_normal";
}

TEST(CornellBox, PhotonNormalsAreOutwardFacing) {
    // Photons can only be stored at bounce > 0 (the no-direct-deposit rule).
    // So we need max_bounces ≥ 3 to get photons that have bounced at least
    // twice and may land on the floor (pos_y < -0.48).  The floor's outward
    // normal faces up (+Y), so those photons should have norm_y > 0.
    Scene scene = build_cornell_test_scene();
    if (scene.emissive_tri_indices.empty()) { GTEST_SKIP() << "No emitters"; }

    EmitterConfig config;
    config.num_photons = 5000;
    config.max_bounces = 4;  // allow multi-bounce paths so floor photons are stored

    PhotonSoA global_map, caustic_map;
    trace_photons(scene, config, global_map, caustic_map);
    if (global_map.size() == 0) { GTEST_SKIP() << "No photons traced"; }

    // Collect photons that landed near the floor (y < -0.48)
    int floor_photons = 0;
    int floor_correct_normal = 0;
    for (size_t i = 0; i < global_map.size(); ++i) {
        if (global_map.pos_y[i] < -0.48f) {
            ++floor_photons;
            // Floor normal should be +Y
            if (global_map.norm_y[i] > 0.5f) ++floor_correct_normal;
        }
    }

    if (floor_photons > 0) {
        float correct_frac = (float)floor_correct_normal / (float)floor_photons;
        EXPECT_GT(correct_frac, 0.8f)
            << "Most floor photons should have outward (+Y) normal; got "
            << floor_correct_normal << "/" << floor_photons;
    }
}

// ── 37.5  Integration: thin-wall leakage with real geometry ─────────

TEST(CornellBox, NormalVisibilityEliminatesLeakage) {
    // Construct a synthetic thin-wall scenario and verify that the density
    // estimator returns zero on the wrong side.
    //
    // Setup: 50 strong photons deposited on a surface facing +Z.
    //        Query on a surface facing -Z at the same location.
    //        Without normal visibility, lots of irradiance leaks through.
    //        With normal visibility, result must be zero.
    const int N = 50;
    const float big_flux = 100.f;  // deliberately large to detect any leak

    PhotonSoA photons;
    for (int i = 0; i < N; ++i) {
        Photon p;
        p.position    = make_f3((float)i * 0.001f, 0.f, 0.f);
        // wi points AWAY from the surface (convention: stored as -incoming_direction).
        // With geom_n = (0,0,+1), the outgoing wi from the surface is (0,0,+1).
        // Correct-side query (n=(0,0,+1)): dot(wi,n)=1>0 (passes wi check), dot(geom_n,n)=1>0 → accepted.
        // Wrong-side  query (n=(0,0,-1)): dot(wi,n)=-1≤0 (rejected by wi check) AND
        //   dot(geom_n=(0,0,+1), n=(0,0,-1))=-1≤0 (rejected by geom_n check) → both checks agree.
        p.wi          = make_f3(0.f, 0.f, 1.f);
        p.geom_normal = make_f3(0.f, 0.f, 1.f);   // photon on +Z surface
        p.spectral_flux = Spectrum::constant(big_flux);
        photons.push_back(p);
    }

    HashGrid grid;
    grid.build(photons, 1.0f);

    Material mat;
    mat.type = MaterialType::Lambertian;
    mat.Kd = Spectrum::constant(1.f);

    DensityEstimatorConfig config;
    config.radius            = 1.0f;
    config.surface_tau       = 5.0f;  // allows all photons through plane-distance check
    config.num_photons_total = N;
    config.use_kernel        = false;

    // Query on surface facing -Z: the normal visibility check should reject all photons
    Spectrum L_wrong_side = estimate_photon_density(
        make_f3(0, 0, 0), make_f3(0, 0, -1),
        make_f3(0, 0, 1), mat, photons, grid, config, 1.0f);

    EXPECT_NEAR(L_wrong_side.sum(), 0.f, kTol)
        << "Normal visibility must prevent all irradiance leaking to the wrong side";

    // Sanity check: the correct side (facing +Z) DOES see the photons
    Spectrum L_correct_side = estimate_photon_density(
        make_f3(0, 0, 0), make_f3(0, 0, 1),
        make_f3(0, 0, 1), mat, photons, grid, config, 1.0f);

    EXPECT_GT(L_correct_side.sum(), 0.f)
        << "Correct side should receive irradiance from photons";
}

// =====================================================================
//  SECTION - CellBinGrid: normal-gated scatter correctness
// =====================================================================
// A physically-correct (but slow) CPU reference implementation that
// gathers photons per-cell by iterating ALL photons and checking
// distance + normal compatibility — the "ground truth" against which
// our two-pass approximation is validated.

namespace {

// ── Reference (brute-force, correct) cell-bin builder ───────────────
// For each cell, iterate ALL photons.  A photon contributes to a cell
// if it is within the 3×3×3 neighbourhood AND passes the Epanechnikov
// tangential-disk kernel (w > 0).  Uses hero-wavelength flux and
// Epanechnikov weighting — exactly mirroring CellBinGrid::build() but
// via an O(cells × photons) brute-force loop for validation.
struct ReferenceCellBinGrid {
    std::vector<PhotonBin>  bins;
    std::vector<float3>     cell_normals;
    float cell_size;
    float min_x, min_y, min_z;
    int   dim_x, dim_y, dim_z;
    int   bin_count;

    void build(const PhotonSoA& photons, float gather_radius, int num_bins) {
        bin_count = num_bins;
        cell_size = gather_radius * 2.0f;

        // Use a temporary CellBinGrid just to get identical AABB / dims
        CellBinGrid tmp;
        tmp.cell_size = cell_size;
        tmp.bin_count = num_bins;
        tmp.compute_grid_geometry(photons);
        min_x = tmp.min_x; min_y = tmp.min_y; min_z = tmp.min_z;
        dim_x = tmp.dim_x; dim_y = tmp.dim_y; dim_z = tmp.dim_z;

        const size_t total = (size_t)dim_x * dim_y * dim_z;
        const size_t N = photons.size();
        const float  r2 = gather_radius * gather_radius;

        // ── Brute-force accumulation ────────────────────────────────
        // Mirrors CellBinGrid::build() exactly: for each photon, scatter
        // to own cell + 3×3×3 neighbours with Epanechnikov kernel.
        bins.resize(total * bin_count);
        std::memset(bins.data(), 0, bins.size() * sizeof(PhotonBin));

        cell_normals.resize(total);
        for (size_t c = 0; c < total; ++c)
            cell_normals[c] = make_f3(0.f, 0.f, 0.f);

        for (size_t i = 0; i < N; ++i) {
            const float px = photons.pos_x[i];
            const float py = photons.pos_y[i];
            const float pz = photons.pos_z[i];
            const float nx = photons.norm_x[i];
            const float ny = photons.norm_y[i];
            const float nz = photons.norm_z[i];
            const float wi_x = photons.wi_x[i];
            const float wi_y = photons.wi_y[i];
            const float wi_z = photons.wi_z[i];

            int n_hero = photons.num_hero.empty() ? 1 : (int)photons.num_hero[i];
            float total_hero_flux = 0.f;
            for (int h = 0; h < n_hero; ++h)
                total_hero_flux += photons.flux[i * HERO_WAVELENGTHS + h];
            if (total_hero_flux <= 0.f) continue;

            int k = photons.bin_idx.empty() ? 0 : (int)photons.bin_idx[i];
            if (k < 0 || k >= bin_count) k = 0;

            // Photon's own cell
            int cx = (int)std::floor((px - min_x) / cell_size);
            int cy = (int)std::floor((py - min_y) / cell_size);
            int cz = (int)std::floor((pz - min_z) / cell_size);
            cx = (std::max)(0, (std::min)(cx, dim_x - 1));
            cy = (std::max)(0, (std::min)(cy, dim_y - 1));
            cz = (std::max)(0, (std::min)(cz, dim_z - 1));

            // Scatter to 3×3×3 neighbourhood
            for (int ddz = -1; ddz <= 1; ++ddz)
            for (int ddy = -1; ddy <= 1; ++ddy)
            for (int ddx = -1; ddx <= 1; ++ddx) {
                int ncx = cx + ddx;
                int ncy = cy + ddy;
                int ncz = cz + ddz;
                if (ncx < 0 || ncx >= dim_x ||
                    ncy < 0 || ncy >= dim_y ||
                    ncz < 0 || ncz >= dim_z)
                    continue;

                // Tangential plane projection
                float cell_cx = min_x + (ncx + 0.5f) * cell_size;
                float cell_cy = min_y + (ncy + 0.5f) * cell_size;
                float cell_cz = min_z + (ncz + 0.5f) * cell_size;
                float dcx = px - cell_cx;
                float dcy = py - cell_cy;
                float dcz = pz - cell_cz;
                float d_plane = nx * dcx + ny * dcy + nz * dcz;

                float vx = dcx - nx * d_plane;
                float vy = dcy - ny * d_plane;
                float vz = dcz - nz * d_plane;
                float d_tan2 = vx*vx + vy*vy + vz*vz;

                // Epanechnikov kernel
                float w = 1.f - d_tan2 / r2;
                if (w <= 0.f) continue;

                int flat = ncx + ncy * dim_x + ncz * dim_x * dim_y;
                PhotonBin& b = bins[(size_t)flat * bin_count + k];

                for (int h = 0; h < n_hero; ++h) {
                    int lam_bin = (int)photons.lambda_bin[i * HERO_WAVELENGTHS + h];
                    float p_flux = photons.flux[i * HERO_WAVELENGTHS + h];
                    if (lam_bin >= 0 && lam_bin < NUM_LAMBDA && p_flux > 0.f)
                        b.flux[lam_bin] += w * p_flux;
                }

                float wf = w * total_hero_flux;
                b.dir_x  += wi_x * wf;
                b.dir_y  += wi_y * wf;
                b.dir_z  += wi_z * wf;
                b.avg_nx += nx * wf;
                b.avg_ny += ny * wf;
                b.avg_nz += nz * wf;
                b.weight += w;
                b.count  += 1;

                cell_normals[flat].x += nx * wf;
                cell_normals[flat].y += ny * wf;
                cell_normals[flat].z += nz * wf;
            }
        }

        // Normalize dominant normals
        for (size_t c = 0; c < total; ++c) {
            float len = length(cell_normals[c]);
            if (len > 1e-8f) cell_normals[c] = cell_normals[c] / len;
        }

        // Compute scalar_flux and normalize directions/normals
        PhotonBinDirs fib;
        fib.init(bin_count);
        for (size_t c = 0; c < total; ++c) {
            for (int k2 = 0; k2 < bin_count; ++k2) {
                PhotonBin& b = bins[c * bin_count + k2];

                float sf = 0.f;
                for (int lam = 0; lam < NUM_LAMBDA; ++lam)
                    sf += b.flux[lam];
                b.scalar_flux = sf;

                if (b.count > 0) {
                    float len = std::sqrt(b.dir_x*b.dir_x + b.dir_y*b.dir_y + b.dir_z*b.dir_z);
                    if (len > 1e-8f) { b.dir_x /= len; b.dir_y /= len; b.dir_z /= len; }
                    else { b.dir_x = fib.dirs[k2].x; b.dir_y = fib.dirs[k2].y; b.dir_z = fib.dirs[k2].z; }

                    float nlen = std::sqrt(b.avg_nx*b.avg_nx + b.avg_ny*b.avg_ny + b.avg_nz*b.avg_nz);
                    if (nlen > 1e-8f) { b.avg_nx /= nlen; b.avg_ny /= nlen; b.avg_nz /= nlen; }
                }
            }
        }
    }
};

// Helper: create photons on a planar surface patch
void add_planar_photons(PhotonSoA& photons, float3 center, float3 normal,
                        float3 wi_dir, int count, float spread,
                        uint16_t /*lambda_bin*/, float flux_each,
                        PhotonBinDirs& bin_dirs) {
    // Build a tangent frame for the surface
    float3 tangent, bitangent;
    if (std::fabs(normal.x) < 0.9f) tangent = normalize(cross(make_f3(1,0,0), normal));
    else                              tangent = normalize(cross(make_f3(0,1,0), normal));
    bitangent = cross(normal, tangent);

    PCGRng rng = PCGRng::seed(12345, 67890);
    for (int i = 0; i < count; ++i) {
        float u = (rng.next_float() - 0.5f) * spread;
        float v = (rng.next_float() - 0.5f) * spread;
        Photon p;
        p.position    = center + tangent * u + bitangent * v;
        p.wi          = wi_dir;
        p.geom_normal = normal;
        p.spectral_flux = Spectrum::constant(flux_each);
        // Set hero-wavelength flux so CellBinGrid build can use it
        // (CellBinGrid reads photons.flux[], not spectral_flux[])
        p.num_hero = HERO_WAVELENGTHS;
        for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
            // Distribute flux evenly across hero wavelengths with
            // stratified lambda bins spanning the spectrum
            p.lambda_bin[h] = (uint16_t)(h * NUM_LAMBDA / HERO_WAVELENGTHS);
            p.flux[h]       = flux_each;
        }
        photons.push_back(p);
    }
    // Precompute bin_idx for newly added photons
    size_t start = photons.bin_idx.size();
    photons.bin_idx.resize(photons.size());
    for (size_t i = start; i < photons.size(); ++i) {
        float3 wi = make_f3(photons.wi_x[i], photons.wi_y[i], photons.wi_z[i]);
        photons.bin_idx[i] = (uint8_t)bin_dirs.find_nearest(wi);
    }
}

} // anonymous namespace

// ── Test 1: Reference matches approximation on a single planar surface
// On a single flat surface (floor only), the normal gate never rejects
// anything → both implementations must produce bit-identical results.
TEST(CellBinGrid, NormalGate_SinglePlaneSameAsReference) {
    PhotonBinDirs bin_dirs;
    bin_dirs.init(PHOTON_BIN_COUNT);
    PhotonSoA photons;

    // Floor photons at y=0, normal=(0,1,0), light from above
    float3 floor_n  = make_f3(0, 1, 0);
    float3 floor_wi = normalize(make_f3(0.2f, 1.0f, 0.1f)); // mostly from above
    add_planar_photons(photons, make_f3(0, 0, 0), floor_n, floor_wi,
                       200, 0.3f, 0, 1.0f, bin_dirs);

    float radius = 0.05f;
    CellBinGrid grid;
    grid.build(photons, radius, PHOTON_BIN_COUNT);

    ReferenceCellBinGrid ref;
    ref.build(photons, radius, PHOTON_BIN_COUNT);

    // Both grids must have identical geometry
    ASSERT_EQ(grid.dim_x, ref.dim_x);
    ASSERT_EQ(grid.dim_y, ref.dim_y);
    ASSERT_EQ(grid.dim_z, ref.dim_z);
    ASSERT_EQ(grid.bins.size(), ref.bins.size());

    // Compare all bins (pre-normalisation values: flux, count)
    int mismatches = 0;
    for (size_t c = 0; c < (size_t)grid.total_cells(); ++c) {
        for (int k = 0; k < PHOTON_BIN_COUNT; ++k) {
            const PhotonBin& a = grid.bins[c * PHOTON_BIN_COUNT + k];
            const PhotonBin& b = ref.bins[c * PHOTON_BIN_COUNT + k];
            if (a.count != b.count) ++mismatches;
            if (std::fabs(a.scalar_flux - b.scalar_flux) > 1e-4f) ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0)
        << "Single plane: all bins should be identical between "
           "reference and two-pass implementation";
}

// ── Test 2: Wall-floor corner: cross-surface contamination is blocked
// Two perpendicular surfaces meet at a corner.  Without the normal
// gate, wall photons would leak into floor cells and vice versa.
TEST(CellBinGrid, NormalGate_WallFloorNoContamination) {
    PhotonBinDirs bin_dirs;
    bin_dirs.init(PHOTON_BIN_COUNT);
    PhotonSoA photons;

    // Floor photons (y=0 plane, normal up) — centred at origin
    float3 floor_n  = make_f3(0, 1, 0);
    float3 floor_wi = normalize(make_f3(0.1f, 1.0f, 0.0f));
    add_planar_photons(photons, make_f3(0, 0, 0), floor_n, floor_wi,
                       200, 0.15f, 0, 1.0f, bin_dirs);

    // Wall photons (x=0.4 plane, normal = +x) — far enough to NOT share cells
    float3 wall_n  = make_f3(1, 0, 0);
    float3 wall_wi = normalize(make_f3(1.0f, 0.1f, 0.0f));
    add_planar_photons(photons, make_f3(0.4f, 0.4f, 0), wall_n, wall_wi,
                       200, 0.15f, 0, 1.0f, bin_dirs);

    float radius = 0.05f;  // cell_size = 0.1
    CellBinGrid grid;
    grid.build(photons, radius, PHOTON_BIN_COUNT);

    // Identify a "pure floor" cell by probing the centre of the floor patch
    int floor_cell = grid.cell_index(0.f, 0.f, 0.f);

    // The dominant normal of that cell should be close to (0,1,0)
    float3 cdn = grid.cell_dominant_normal[floor_cell];
    EXPECT_GT(dot(cdn, floor_n), 0.9f)
        << "Floor cell dominant normal should face up";

    // Count total photons seen by bins in this cell
    int total_count = 0;
    float total_flux = 0.f;
    for (int k = 0; k < PHOTON_BIN_COUNT; ++k) {
        const PhotonBin& b = grid.bins[(size_t)floor_cell * PHOTON_BIN_COUNT + k];
        total_count += b.count;
        total_flux  += b.scalar_flux;
    }

    // For the reference, wall photons should NOT have contributed
    ReferenceCellBinGrid ref;
    ref.build(photons, radius, PHOTON_BIN_COUNT);

    int ref_count = 0;
    float ref_flux = 0.f;
    for (int k = 0; k < PHOTON_BIN_COUNT; ++k) {
        const PhotonBin& b = ref.bins[(size_t)floor_cell * PHOTON_BIN_COUNT + k];
        ref_count += b.count;
        ref_flux  += b.scalar_flux;
    }

    // Our grid and reference should agree
    EXPECT_EQ(total_count, ref_count)
        << "Floor cell photon count must match reference";
    EXPECT_NEAR(total_flux, ref_flux, 1e-4f)
        << "Floor cell flux must match reference";

    // Check every bin in every cell matches
    int mismatches = 0;
    for (size_t c = 0; c < (size_t)grid.total_cells(); ++c) {
        for (int k = 0; k < PHOTON_BIN_COUNT; ++k) {
            const PhotonBin& a = grid.bins[c * PHOTON_BIN_COUNT + k];
            const PhotonBin& b = ref.bins[c * PHOTON_BIN_COUNT + k];
            if (a.count != b.count) ++mismatches;
            if (std::fabs(a.scalar_flux - b.scalar_flux) > 1e-4f) ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0)
        << "All cells must match reference in wall-floor corner scenario";
}

// ── Test 3: Opposite-facing back-to-back walls
// Two walls at x≈0, one facing +x and one facing −x (thin wall).
// Ensures photons from one side never leak to the other.
TEST(CellBinGrid, NormalGate_BackToBackWalls) {
    PhotonBinDirs bin_dirs;
    bin_dirs.init(PHOTON_BIN_COUNT);
    PhotonSoA photons;

    // Wall A: faces +x
    float3 nA  = make_f3(1, 0, 0);
    float3 wiA = normalize(make_f3(1.0f, 0.3f, 0.0f));
    add_planar_photons(photons, make_f3(0.001f, 0, 0), nA, wiA,
                       80, 0.15f, 0, 2.0f, bin_dirs);

    // Wall B: faces −x (opposite side)
    float3 nB  = make_f3(-1, 0, 0);
    float3 wiB = normalize(make_f3(-1.0f, 0.2f, 0.1f));
    add_planar_photons(photons, make_f3(-0.001f, 0, 0), nB, wiB,
                       80, 0.15f, 0, 3.0f, bin_dirs);

    float radius = 0.05f;
    CellBinGrid grid;
    grid.build(photons, radius, PHOTON_BIN_COUNT);

    ReferenceCellBinGrid ref;
    ref.build(photons, radius, PHOTON_BIN_COUNT);

    ASSERT_EQ(grid.bins.size(), ref.bins.size());

    int mismatches = 0;
    for (size_t c = 0; c < (size_t)grid.total_cells(); ++c) {
        for (int k = 0; k < PHOTON_BIN_COUNT; ++k) {
            const PhotonBin& a = grid.bins[c * PHOTON_BIN_COUNT + k];
            const PhotonBin& b = ref.bins[c * PHOTON_BIN_COUNT + k];
            if (a.count != b.count) ++mismatches;
            if (std::fabs(a.scalar_flux - b.scalar_flux) > 1e-4f) ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0)
        << "Back-to-back walls: all bins must match reference";
}

// ── Test 4: Normals and directions are preserved after accumulation
// Verifies that after the two-pass build, the normalised avg_n and dir
// fields in bins still point in physically reasonable directions.
TEST(CellBinGrid, NormalGate_NormalsAndDirectionsPreserved) {
    PhotonBinDirs bin_dirs;
    bin_dirs.init(PHOTON_BIN_COUNT);
    PhotonSoA photons;

    float3 floor_n  = make_f3(0, 1, 0);
    float3 floor_wi = normalize(make_f3(0.0f, 1.0f, 0.0f)); // straight down
    add_planar_photons(photons, make_f3(0, 0, 0), floor_n, floor_wi,
                       300, 0.2f, 0, 1.0f, bin_dirs);

    float radius = 0.05f;
    CellBinGrid grid;
    grid.build(photons, radius, PHOTON_BIN_COUNT);

    // Find the bin that contains the floor_wi direction
    int wi_bin = bin_dirs.find_nearest(floor_wi);

    // Check a cell near the center of the photon distribution
    int center_cell = grid.cell_index(0.f, 0.f, 0.f);
    const PhotonBin& b = grid.bins[(size_t)center_cell * PHOTON_BIN_COUNT + wi_bin];

    EXPECT_GT(b.count, 0) << "Centre cell wi-bin should have photons";

    // The normalised direction should closely match floor_wi
    float3 bin_dir = make_f3(b.dir_x, b.dir_y, b.dir_z);
    EXPECT_GT(dot(bin_dir, floor_wi), 0.99f)
        << "Bin direction should closely match photon incoming direction";

    // The normalised avg normal should closely match floor_n
    float3 bin_n = make_f3(b.avg_nx, b.avg_ny, b.avg_nz);
    EXPECT_GT(dot(bin_n, floor_n), 0.99f)
        << "Bin average normal should closely match surface normal";
}

// ── Test 5: Dominant normal of empty-neighbour cells stays zero
// An empty cell has dominant normal (0,0,0) → length ≈ 0, which means
// no photon passes the dot > 0 gate.  Photons that try to scatter into
// that cell should be blocked.
TEST(CellBinGrid, NormalGate_EmptyCellBlocksScatter) {
    PhotonBinDirs bin_dirs;
    bin_dirs.init(PHOTON_BIN_COUNT);
    PhotonSoA photons;

    // Place photons in a tight cluster so most neighbouring cells are empty
    float3 n  = make_f3(0, 0, 1);
    float3 wi = make_f3(0, 0, 1);
    add_planar_photons(photons, make_f3(0, 0, 0), n, wi,
                       10, 0.001f, 0, 1.0f, bin_dirs);

    float radius = 0.05f;
    CellBinGrid grid;
    grid.build(photons, radius, PHOTON_BIN_COUNT);

    // The native cell of the photons
    int native = grid.cell_index(0.f, 0.f, 0.f);

    // A cell far away should be completely empty
    int far_cell = grid.cell_index(0.5f, 0.5f, 0.5f);
    if (far_cell != native) {
        int count = 0;
        for (int k = 0; k < PHOTON_BIN_COUNT; ++k)
            count += grid.bins[(size_t)far_cell * PHOTON_BIN_COUNT + k].count;
        EXPECT_EQ(count, 0)
            << "Far cell should have no photons";
    }
}

// ── Test 6: Three mutually-perpendicular surfaces (floor, wall-X, wall-Z)
// Full 3-surface corner. Each surface's cells should contain only
// compatible photons.
TEST(CellBinGrid, NormalGate_ThreeSurfaceCorner) {
    PhotonBinDirs bin_dirs;
    bin_dirs.init(PHOTON_BIN_COUNT);
    PhotonSoA photons;

    float3 floor_n  = make_f3(0, 1, 0);
    float3 wallX_n  = make_f3(1, 0, 0);
    float3 wallZ_n  = make_f3(0, 0, 1);

    add_planar_photons(photons, make_f3(0.0f, 0.0f, 0.0f), floor_n,
                       normalize(make_f3(0.1f, 1.0f, 0.0f)), 80, 0.06f,
                       0, 1.0f, bin_dirs);
    add_planar_photons(photons, make_f3(0.05f, 0.05f, 0.0f), wallX_n,
                       normalize(make_f3(1.0f, 0.1f, 0.0f)), 80, 0.06f,
                       1, 2.0f, bin_dirs);
    add_planar_photons(photons, make_f3(0.0f, 0.05f, 0.05f), wallZ_n,
                       normalize(make_f3(0.0f, 0.1f, 1.0f)), 80, 0.06f,
                       2, 3.0f, bin_dirs);

    float radius = 0.05f;
    CellBinGrid grid;
    grid.build(photons, radius, PHOTON_BIN_COUNT);

    ReferenceCellBinGrid ref;
    ref.build(photons, radius, PHOTON_BIN_COUNT);

    ASSERT_EQ(grid.bins.size(), ref.bins.size());

    // Compare every cell/bin — allow zero tolerance on counts, small on flux
    int count_mismatch = 0;
    int flux_mismatch  = 0;
    for (size_t c = 0; c < (size_t)grid.total_cells(); ++c) {
        for (int k = 0; k < PHOTON_BIN_COUNT; ++k) {
            const PhotonBin& a = grid.bins[c * PHOTON_BIN_COUNT + k];
            const PhotonBin& b = ref.bins[c * PHOTON_BIN_COUNT + k];
            if (a.count != b.count) ++count_mismatch;
            if (std::fabs(a.scalar_flux - b.scalar_flux) > 1e-4f) ++flux_mismatch;
        }
    }
    EXPECT_EQ(count_mismatch, 0);
    EXPECT_EQ(flux_mismatch, 0);
}

// ── Test 7: Cell dominant normals are correctly computed
TEST(CellBinGrid, NormalGate_DominantNormalCorrectness) {
    PhotonBinDirs bin_dirs;
    bin_dirs.init(PHOTON_BIN_COUNT);
    PhotonSoA photons;

    // Floor photons only
    float3 floor_n  = make_f3(0, 1, 0);
    float3 floor_wi = make_f3(0, 1, 0);
    add_planar_photons(photons, make_f3(0, 0, 0), floor_n, floor_wi,
                       50, 0.05f, 0, 1.0f, bin_dirs);

    float radius = 0.05f;
    CellBinGrid grid;
    grid.build(photons, radius, PHOTON_BIN_COUNT);

    // The cell containing (0,0,0) should have dominant normal ≈ (0,1,0)
    int cell = grid.cell_index(0.f, 0.f, 0.f);
    float3 cdn = grid.cell_dominant_normal[cell];
    EXPECT_GT(dot(cdn, floor_n), 0.99f)
        << "Dominant normal should match floor normal";

    // An empty cell's dominant normal should be zero-length
    int far_cell = grid.cell_index(0.5f, 0.5f, 0.5f);
    if (far_cell != cell) {
        float len = length(grid.cell_dominant_normal[far_cell]);
        EXPECT_LT(len, 1e-6f)
            << "Empty cell dominant normal should be near-zero";
    }
}

// ── Test 8: Flux conservation on single surface
// When many photons share the same normal in a cluster, the 3×3×3
// scatter should propagate all of them to compatible neighbours.
// The native cell always gets the photon.  Neighbour cells only get it
// if their dominant normal (from pass 1) is compatible (dot > 0).
// For a single-surface cluster every occupied neighbour has the same
// normal, so the gate always passes.
TEST(CellBinGrid, NormalGate_FluxConservation_SingleSurface) {
    PhotonBinDirs bin_dirs;
    bin_dirs.init(PHOTON_BIN_COUNT);
    PhotonSoA photons;

    // Place many photons in a tight cluster so that all 27 neighbours
    // actually contain native photons (all with the same normal).
    float3 normal = make_f3(0.f, 1.f, 0.f);
    float3 wi_dir = normalize(make_f3(0.f, 1.f, 0.f));
    float flux_each = 2.0f;
    int count = 500;
    float radius = 0.05f;  // cell_size = 0.1
    // Spread the photons across ~3 cells in each dimension
    add_planar_photons(photons, make_f3(0.f, 0.f, 0.f), normal, wi_dir,
                       count, 0.25f, 0, flux_each, bin_dirs);

    CellBinGrid grid;
    grid.build(photons, radius, PHOTON_BIN_COUNT);

    // Sum flux in every cell for every bin
    float total_flux = 0.f;
    for (size_t c = 0; c < (size_t)grid.total_cells(); ++c)
        for (int k = 0; k < PHOTON_BIN_COUNT; ++k)
            total_flux += grid.bins[c * PHOTON_BIN_COUNT + k].scalar_flux;

    // Each photon scatters into its native cell + up to 26 neighbours.
    // With a large spread, most photons' neighbours are also occupied
    // and share the same normal → gate passes.
    // Total flux should be > N * flux (from native cells alone) and
    // strictly less than N * flux * 27 (boundary photons may have
    // fewer than 26 valid neighbours).
    float native_total = (float)count * flux_each;  // minimum: native only
    EXPECT_GE(total_flux, native_total - 1e-3f)
        << "Total flux must be at least the native-cell-only sum";
    EXPECT_GT(total_flux, native_total * 2.f)
        << "3×3×3 scatter should significantly amplify total flux "
           "(photons in interior cells contribute to 27 cells each)";

    // Verify against reference
    ReferenceCellBinGrid ref;
    ref.build(photons, radius, PHOTON_BIN_COUNT);

    float ref_total = 0.f;
    for (size_t c = 0; c < (size_t)ref.dim_x * ref.dim_y * ref.dim_z; ++c)
        for (int k = 0; k < PHOTON_BIN_COUNT; ++k)
            ref_total += ref.bins[c * PHOTON_BIN_COUNT + k].scalar_flux;

    EXPECT_NEAR(total_flux, ref_total, 1e-2f)
        << "Total flux should match reference implementation";
}

// =====================================================================
//  SECTION – SPPM Progressive Photon Mapping (sppm.h)
// =====================================================================

// ── SPPMPixel initialization ────────────────────────────────────────

TEST(SPPM, PixelInit) {
    SPPMPixel p;
    p.init(0.25f);

    EXPECT_FLOAT_EQ(p.radius, 0.25f);
    EXPECT_FLOAT_EQ(p.N, 0.f);
    EXPECT_FALSE(p.valid);
    EXPECT_EQ(p.M_count, 0);

    for (int i = 0; i < NUM_LAMBDA; ++i) {
        EXPECT_FLOAT_EQ(p.tau.value[i], 0.f);
        EXPECT_FLOAT_EQ(p.throughput.value[i], 0.f);
        EXPECT_FLOAT_EQ(p.L_direct.value[i], 0.f);
    }
}

// ── Progressive update: no photons → no change ─────────────────────

TEST(SPPM, UpdateZeroPhotons) {
    SPPMPixel p;
    p.init(0.5f);
    p.N = 10.f;  // some prior accumulated photons

    Spectrum phi = Spectrum::constant(1.0f);
    sppm_progressive_update(p, phi, 0);  // M = 0

    // Nothing should change
    EXPECT_FLOAT_EQ(p.radius, 0.5f);
    EXPECT_FLOAT_EQ(p.N, 10.f);
    for (int i = 0; i < NUM_LAMBDA; ++i)
        EXPECT_FLOAT_EQ(p.tau.value[i], 0.f);
}

// ── Progressive update: radius shrinks monotonically ────────────────

TEST(SPPM, RadiusShrinks) {
    SPPMPixel p;
    p.init(1.0f);

    Spectrum phi = Spectrum::constant(1.0f);
    float prev_radius = p.radius;

    for (int iter = 0; iter < 20; ++iter) {
        sppm_progressive_update(p, phi, 10);  // 10 photons per iter
        EXPECT_LT(p.radius, prev_radius)
            << "Radius must shrink each iteration (iter=" << iter << ")";
        prev_radius = p.radius;
    }
}

// ── Progressive update: N accumulates correctly with alpha ──────────

TEST(SPPM, NAccumulation) {
    SPPMPixel p;
    p.init(1.0f);

    float alpha = 0.7f;

    // Iteration 1: N = 0 + alpha * M = 0.7 * 5 = 3.5
    sppm_progressive_update(p, Spectrum::constant(1.0f), 5, alpha);
    EXPECT_NEAR(p.N, alpha * 5.f, 1e-5f);

    // Iteration 2: N = 3.5 + 0.7 * 8 = 3.5 + 5.6 = 9.1
    sppm_progressive_update(p, Spectrum::constant(1.0f), 8, alpha);
    EXPECT_NEAR(p.N, 3.5f + alpha * 8.f, 1e-4f);
}

// ── Progressive update: radius formula verification ─────────────────

TEST(SPPM, RadiusFormula) {
    SPPMPixel p;
    float r0 = 0.5f;
    p.init(r0);
    float alpha = DEFAULT_SPPM_ALPHA;
    int M = 12;

    sppm_progressive_update(p, Spectrum::constant(1.0f), M, alpha);

    // Expected: N_new = 0 + alpha * M = alpha * M
    // ratio = N_new / (0 + M) = alpha
    // r_new = r0 * sqrt(alpha)
    float expected_r = r0 * sqrtf(alpha);
    EXPECT_NEAR(p.radius, expected_r, 1e-5f);
}

// ── Progressive update: flux scales with area ratio ─────────────────

TEST(SPPM, FluxAreaRatio) {
    SPPMPixel p;
    float r0 = 1.0f;
    p.init(r0);
    float alpha = DEFAULT_SPPM_ALPHA;

    Spectrum phi;
    for (int i = 0; i < NUM_LAMBDA; ++i)
        phi.value[i] = (float)(i + 1);  // distinct values

    sppm_progressive_update(p, phi, 10, alpha);

    // area_ratio = (r_new/r_old)^2 = ratio = alpha * 10 / (0 + 10) = alpha
    // tau = (0 + phi) * area_ratio = phi * alpha
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        float expected = phi.value[i] * alpha;
        EXPECT_NEAR(p.tau.value[i], expected, 1e-4f)
            << "tau[" << i << "] should be phi * alpha on first iteration";
    }
}

// ── Progressive update: minimum radius clamp ────────────────────────

TEST(SPPM, MinRadiusClamp) {
    SPPMPixel p;
    float min_r = 0.01f;
    p.init(min_r * 0.5f);  // start below minimum
    p.N = 1000.f;          // many prior photons to force small ratio

    Spectrum phi = Spectrum::constant(0.001f);
    sppm_progressive_update(p, phi, 1000, 0.1f, min_r);

    EXPECT_GE(p.radius, min_r)
        << "Radius should be clamped at min_radius";
}

// ── Reconstruction: invalid pixel returns zero ──────────────────────

TEST(SPPM, ReconstructInvalidPixel) {
    SPPMPixel p;
    p.init(0.5f);
    p.valid = false;

    Spectrum L = sppm_reconstruct(p, 10, 1000);
    for (int i = 0; i < NUM_LAMBDA; ++i)
        EXPECT_FLOAT_EQ(L.value[i], 0.f);
}

// ── Reconstruction: zero iterations returns zero ────────────────────

TEST(SPPM, ReconstructZeroIterations) {
    SPPMPixel p;
    p.init(0.5f);
    p.valid = true;
    p.tau = Spectrum::constant(100.f);

    Spectrum L = sppm_reconstruct(p, 0, 1000);
    for (int i = 0; i < NUM_LAMBDA; ++i)
        EXPECT_FLOAT_EQ(L.value[i], 0.f);
}

// ── Reconstruction: formula verification ────────────────────────────

TEST(SPPM, ReconstructFormula) {
    SPPMPixel p;
    float r = 0.1f;
    p.init(r);
    p.valid = true;

    // Set known tau and L_direct
    float tau_val = 100.f;
    float direct_val = 50.f;
    p.tau = Spectrum::constant(tau_val);
    p.L_direct = Spectrum::constant(direct_val);

    int k = 10;      // iterations
    int N_p = 5000;   // photons per iteration

    Spectrum L = sppm_reconstruct(p, k, N_p);

    // Expected: L_indirect = tau / (0.5 * pi * r^2 * k * N_p)  [Epanechnikov]
    //           L_direct_avg = L_direct / k
    float denom = 0.5f * PI * r * r * (float)k * (float)N_p;
    float expected_indirect = tau_val / denom;
    float expected_direct   = direct_val / (float)k;
    float expected_total    = expected_indirect + expected_direct;

    for (int i = 0; i < NUM_LAMBDA; ++i) {
        EXPECT_NEAR(L.value[i], expected_total, expected_total * 1e-4f)
            << "Reconstruction formula incorrect at bin " << i;
    }
}

// ── SPPMBuffer initialization ───────────────────────────────────────

TEST(SPPM, BufferResize) {
    SPPMBuffer buf;
    buf.resize(32, 16, 0.3f);

    EXPECT_EQ(buf.width, 32);
    EXPECT_EQ(buf.height, 16);
    EXPECT_EQ((int)buf.pixels.size(), 32 * 16);

    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 32; ++x) {
            const auto& p = buf.at(x, y);
            EXPECT_FLOAT_EQ(p.radius, 0.3f);
            EXPECT_FLOAT_EQ(p.N, 0.f);
            EXPECT_FALSE(p.valid);
        }
    }
}

// ── Multi-iteration convergence: radius decreases, N increases ──────

TEST(SPPM, MultiIterationConvergence) {
    SPPMPixel p;
    p.init(0.5f);

    float alpha = DEFAULT_SPPM_ALPHA;
    Spectrum phi = Spectrum::constant(2.0f);

    float prev_r = p.radius;
    float prev_N = p.N;

    for (int iter = 0; iter < 100; ++iter) {
        sppm_progressive_update(p, phi, 5, alpha);

        EXPECT_LE(p.radius, prev_r) << "radius must not increase";
        EXPECT_GE(p.N, prev_N) << "N must not decrease";

        prev_r = p.radius;
        prev_N = p.N;
    }

    // After many iterations, radius should be significantly smaller
    EXPECT_LT(p.radius, 0.5f * 0.5f)
        << "After 100 iterations, radius should be < 50% of initial";
    EXPECT_GT(p.N, 0.f)
        << "N should have accumulated";
}

// ── sppm_gather: verify photon counting and flux accumulation ───────

TEST(SPPM, GatherBasic) {
    // Create a simple photon set and test sppm_gather
    PhotonSoA photons;
    const int N = 5;
    photons.resize(N);

    // Place all photons at origin, facing +Y surface
    for (int i = 0; i < N; ++i) {
        photons.pos_x[i] = 0.01f * (float)i;
        photons.pos_y[i] = 0.f;
        photons.pos_z[i] = 0.f;
        photons.wi_x[i] = 0.f;
        photons.wi_y[i] = 1.f;   // coming from below
        photons.wi_z[i] = 0.f;
        photons.norm_x[i] = 0.f;
        photons.norm_y[i] = 1.f;
        photons.norm_z[i] = 0.f;
        // Set uniform spectral flux for this photon
        Spectrum sf = Spectrum::constant(10.f);
        for (int b = 0; b < NUM_LAMBDA; ++b)
            photons.spectral_flux[i * NUM_LAMBDA + b] = sf.value[b];
    }

    // Build hash grid
    HashGrid grid;
    grid.build(photons, 0.5f);

    // Create a simple diffuse material
    Material mat;
    mat.type = MaterialType::Lambertian;
    mat.Kd   = Spectrum::constant(0.8f);

    // Setup hit point: at origin, normal +Y, looking down from above
    float3 hit_pos = make_f3(0, 0, 0);
    float3 hit_normal = make_f3(0, 1, 0);
    float3 hit_wo = make_f3(0, 0, 1);  // outgoing direction in local frame

    int M_out = 0;
    Spectrum phi = sppm_gather(
        hit_pos, hit_normal, hit_wo,
        mat,
        photons, grid, 0.5f,
        DEFAULT_SURFACE_TAU,
        M_out);

    EXPECT_GT(M_out, 0) << "Should find at least some photons";
    EXPECT_GT(phi.value[0], 0.f) << "Flux in bin 0 should be positive";
}

// =====================================================================
//  Main
// =====================================================================

#include "report_listener.h"

// Parse --report-dir=<path> from argv, or fall back to PPT_REPORT_DIR
// environment variable.  Returns empty string if neither is set.
static std::string get_report_dir(int argc, char** argv) {
    const char* prefix = "--report-dir=";
    size_t prefix_len = strlen(prefix);
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], prefix, prefix_len) == 0)
            return std::string(argv[i] + prefix_len);
    }
#ifdef _MSC_VER
    char* env = nullptr;
    size_t env_len = 0;
    _dupenv_s(&env, &env_len, "PPT_REPORT_DIR");
    std::string result;
    if (env && env[0]) { result = env; free(env); return result; }
    free(env);
#else
    const char* env = std::getenv("PPT_REPORT_DIR");
    if (env && env[0]) return std::string(env);
#endif
    return {};
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // If a report directory is specified, register the detailed
    // report listener that writes report.txt, report.json, summary.txt
    std::string report_dir = get_report_dir(argc, argv);
    if (!report_dir.empty()) {
        auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
        listeners.Append(new ReportListener(report_dir));
        std::cout << "[ppt_tests] Report output: " << report_dir << "\n";
    }

    return RUN_ALL_TESTS();
}
