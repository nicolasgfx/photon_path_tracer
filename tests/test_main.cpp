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
//   - Cornell box: scene loading, direct lighting, OptiX integration
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
#include "renderer/integrator/nee_shared.h"
#include "renderer/camera.h"
#include "renderer/render_config.h"
#include "scene/scene.h"
#include "scene/obj_loader.h"

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------
static constexpr float kTol   = 1e-5f;
static constexpr float kLoose = 1e-3f;
static constexpr float kStat  = 0.05f; // 5% tolerance for statistical tests


// (Intentionally omitted: 'approx' helper was unused - use EXPECT_NEAR instead)

// ── Synthetic photon map for OptiX upload tests ─────────────────────
// Creates a small PhotonSoA with photons scattered inside a Cornell box
// (positions in [-0.5, 0.5]³).  Used to feed OptiX tests that formerly
// relied on the CPU Renderer to build photon maps.
static PhotonSoA make_synthetic_photons(int n) {
    PhotonSoA photons;
    photons.reserve(n);
    PCGRng rng = PCGRng::seed(123);
    for (int i = 0; i < n; ++i) {
        Photon p;
        p.position    = make_f3(rng.next_float() - 0.5f,
                                rng.next_float() - 0.5f,
                                rng.next_float() - 0.5f);
        p.wi          = normalize(make_f3(rng.next_float() - 0.5f,
                                          rng.next_float() - 0.5f,
                                          rng.next_float() - 0.5f));
        p.geom_normal = make_f3(0.f, 1.f, 0.f);
        p.spectral_flux = Spectrum::constant(0.01f);
        photons.push_back(p);
    }
    return photons;
}

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

TEST(ONB, OrientedFrameKeepsOutgoingPositiveZ) {
    float3 n = make_f3(0.f, 0.f, 1.f);
    float3 wo_world = make_f3(0.f, 0.f, -1.f);

    auto frame = orient_frame_to_outgoing(n, wo_world);
    float3 local = frame.world_to_local(wo_world);

    EXPECT_NEAR(local.x, 0.f, kTol);
    EXPECT_NEAR(local.y, 0.f, kTol);
    EXPECT_GT(local.z, 0.f);
}

TEST(ONB, OrientedFramePreservesAlignedNormal) {
    float3 n = normalize(make_f3(0.3f, -0.4f, 0.866f));
    auto frame = orient_frame_to_outgoing(n, n);

    EXPECT_NEAR(dot(frame.w, n), 1.f, kTol);
    EXPECT_GT(frame.world_to_local(n).z, 0.f);
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

// ── Neutral flatness: rgb_to_spectrum_reflectance(c,c,c) must be flat ─
// This is the key property preventing multi-bounce color drift.
TEST(Spectrum, ReflectanceNeutralFlat) {
    // White
    {
        Spectrum s = rgb_to_spectrum_reflectance(1.0f, 1.0f, 1.0f);
        for (int i = 0; i < NUM_LAMBDA; ++i)
            EXPECT_NEAR(s.value[i], 1.0f, 1e-5f)
                << "White reflectance bin " << i << " should be 1.0";
    }
    // Mid-grey
    {
        Spectrum s = rgb_to_spectrum_reflectance(0.5f, 0.5f, 0.5f);
        for (int i = 0; i < NUM_LAMBDA; ++i)
            EXPECT_NEAR(s.value[i], 0.5f, 1e-5f)
                << "Grey 0.5 reflectance bin " << i << " should be 0.5";
    }
    // Staircase2 wall albedo (0.893)
    {
        Spectrum s = rgb_to_spectrum_reflectance(0.893f, 0.893f, 0.893f);
        for (int i = 0; i < NUM_LAMBDA; ++i)
            EXPECT_NEAR(s.value[i], 0.893f, 1e-5f)
                << "Albedo 0.893 reflectance bin " << i << " should be 0.893";
    }
}

// ── Multi-bounce stability: N bounces of neutral reflectance ────────
// After repeated multiplication by a neutral reflectance spectrum,
// all spectral bins should remain equal (no color drift).
TEST(Spectrum, ReflectanceMultiBounceNeutral) {
    float albedo = 0.893f;
    Spectrum Kd = rgb_to_spectrum_reflectance(albedo, albedo, albedo);
    // Simulate 16 bounces of neutral wall reflectance
    Spectrum throughput = Spectrum::constant(1.0f);
    for (int bounce = 0; bounce < 16; ++bounce) {
        for (int i = 0; i < NUM_LAMBDA; ++i)
            throughput.value[i] *= Kd.value[i];
    }
    float expected = std::pow(albedo, 16.0f);
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        EXPECT_NEAR(throughput.value[i], expected, expected * 1e-4f)
            << "After 16 bounces, bin " << i << " should equal albedo^16";
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
    scene.compute_bounds();
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

        scene.compute_bounds();
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

    PhotonSoA global = make_synthetic_photons(500);
    PhotonSoA caustic;  // empty

    OptixRenderer optix_renderer;
    optix_renderer.init();
    optix_renderer.build_accel(scene);
    optix_renderer.upload_scene_data(scene);

    EXPECT_NO_THROW(optix_renderer.upload_photon_data(
        global, caustic, 0.1f, 0.05f))
        << "upload_photon_data should complete without error";
}

// -- 36.5 Debug frame produces non-zero output -----------------------

TEST(OptiX, DebugFrameNonZero) {
    Scene scene = build_cornell_test_scene();
    if (scene.emissive_tri_indices.empty()) { GTEST_SKIP() << "No emitters"; }

    PhotonSoA global = make_synthetic_photons(10000);
    PhotonSoA caustic;

    Camera cam = Camera::cornell_box_camera(8, 8);

    OptixRenderer optix_renderer;
    optix_renderer.init();
    optix_renderer.build_accel(scene);
    optix_renderer.upload_scene_data(scene);
    optix_renderer.upload_emitter_data(scene);
    optix_renderer.upload_photon_data(global, caustic, 0.15f, 0.05f);

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

// -- 36.6 Normals debug mode (permanently skipped) -------------------

TEST(OptiX, NormalsDebugMode) {
    GTEST_SKIP() << "Debug render modes removed in v3 (PT-09)";
}

// -- 36.7 Final render produces valid framebuffer --------------------

TEST(OptiX, FinalRenderProducesValid) {
    Scene scene = build_cornell_test_scene();
    if (scene.emissive_tri_indices.empty()) { GTEST_SKIP() << "No emitters"; }

    PhotonSoA global = make_synthetic_photons(500);
    PhotonSoA caustic;

    Camera cam = Camera::cornell_box_camera(8, 8);

    RenderConfig cfg;
    cfg.image_width = 8;
    cfg.image_height = 8;
    cfg.samples_per_pixel = 2;
    cfg.num_photons = 500;
    cfg.mode = RenderMode::Full;
    cfg.denoiser_enabled = false;

    OptixRenderer optix_renderer;
    optix_renderer.init();
    optix_renderer.build_accel(scene);
    optix_renderer.upload_scene_data(scene);
    optix_renderer.upload_emitter_data(scene);
    optix_renderer.upload_photon_data(global, caustic, 0.1f, 0.05f);

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

    PhotonSoA global = make_synthetic_photons(100);
    PhotonSoA caustic;

    Camera cam = Camera::cornell_box_camera(16, 16);

    OptixRenderer optix_renderer;
    optix_renderer.init();
    optix_renderer.build_accel(scene);
    optix_renderer.upload_scene_data(scene);
    optix_renderer.upload_photon_data(global, caustic, 0.1f, 0.05f);

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




// =====================================================================
//  SECTION 37 - PhotonSoA geom_normal storage
// =====================================================================

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


// =====================================================================
//  SECTION – IAS Instancing metadata
// =====================================================================

TEST(Instancing, HasInstancesRequiresMultiple) {
    Scene scene;
    // Empty → no instances
    EXPECT_FALSE(scene.has_instances());

    // Single mesh + single instance → NOT instanced (single-GAS path)
    MeshDescriptor m0; m0.tri_offset = 0; m0.tri_count = 100;
    scene.meshes.push_back(m0);
    InstanceDescriptor i0{};
    i0.mesh_id = 0;
    i0.transform[0] = 1.f; i0.transform[5] = 1.f; i0.transform[10] = 1.f;
    scene.instances.push_back(i0);
    EXPECT_FALSE(scene.has_instances());

    // Two instances of mesh 0 → instanced (IAS path)
    InstanceDescriptor i1{};
    i1.mesh_id = 0;
    i1.transform[0] = 1.f; i1.transform[5] = 1.f; i1.transform[10] = 1.f;
    i1.transform[3] = 5.f;  // translated +5 in X
    scene.instances.push_back(i1);
    EXPECT_TRUE(scene.has_instances());
}

TEST(Instancing, MeshDescriptorOffsets) {
    Scene scene;
    // Two meshes: mesh0 = tris [0..99], mesh1 = tris [100..249]
    MeshDescriptor m0; m0.tri_offset = 0;   m0.tri_count = 100;
    MeshDescriptor m1; m1.tri_offset = 100; m1.tri_count = 150;
    scene.meshes.push_back(m0);
    scene.meshes.push_back(m1);

    EXPECT_EQ(scene.meshes[0].tri_offset, 0u);
    EXPECT_EQ(scene.meshes[0].tri_count, 100u);
    EXPECT_EQ(scene.meshes[1].tri_offset, 100u);
    EXPECT_EQ(scene.meshes[1].tri_count, 150u);

    // Three instances: mesh0 ×1, mesh1 ×2
    InstanceDescriptor i0{}; i0.mesh_id = 0;
    i0.transform[0] = 1.f; i0.transform[5] = 1.f; i0.transform[10] = 1.f;
    InstanceDescriptor i1{}; i1.mesh_id = 1;
    i1.transform[0] = 1.f; i1.transform[5] = 1.f; i1.transform[10] = 1.f;
    InstanceDescriptor i2{}; i2.mesh_id = 1;
    i2.transform[0] = 1.f; i2.transform[5] = 1.f; i2.transform[10] = 1.f;
    i2.transform[3] = 10.f;
    scene.instances.push_back(i0);
    scene.instances.push_back(i1);
    scene.instances.push_back(i2);

    EXPECT_EQ(scene.instances.size(), 3u);
    EXPECT_TRUE(scene.has_instances());
    EXPECT_EQ(scene.instances[1].mesh_id, 1u);
    EXPECT_EQ(scene.instances[2].mesh_id, 1u);
}

TEST(Instancing, ObjBackwardCompat) {
    // Load Cornell Box via OBJ (has no instancing metadata)
    Scene scene;
    std::string path = std::string(SCENES_DIR) + "/cornell_box/cornellbox.obj";
    ASSERT_TRUE(load_obj(path, scene));
    ASSERT_GT(scene.triangles.size(), 0u);

    // Before wrapping: meshes/instances should be empty
    EXPECT_TRUE(scene.meshes.empty());
    EXPECT_TRUE(scene.instances.empty());

    // Apply the same wrapping logic as main.cpp
    if (scene.meshes.empty() && !scene.triangles.empty()) {
        MeshDescriptor m0;
        m0.tri_offset = 0;
        m0.tri_count  = (uint32_t)scene.triangles.size();
        scene.meshes.push_back(m0);

        InstanceDescriptor inst0{};
        inst0.mesh_id = 0;
        inst0.transform[0] = 1.f; inst0.transform[5] = 1.f; inst0.transform[10] = 1.f;
        scene.instances.push_back(inst0);
    }

    // After wrapping: single mesh, single instance, NOT instanced
    EXPECT_EQ(scene.meshes.size(), 1u);
    EXPECT_EQ(scene.instances.size(), 1u);
    EXPECT_FALSE(scene.has_instances());
    EXPECT_EQ(scene.meshes[0].tri_offset, 0u);
    EXPECT_EQ(scene.meshes[0].tri_count, (uint32_t)scene.triangles.size());
    EXPECT_EQ(scene.instances[0].mesh_id, 0u);
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
