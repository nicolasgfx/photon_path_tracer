// ─────────────────────────────────────────────────────────────────────
// test_medium.cpp – Unit & integration tests for participating medium
// ─────────────────────────────────────────────────────────────────────
// Covers:
//   1. Beer–Lambert transmittance correctness
//   2. Rayleigh phase function normalization
//   3. Rayleigh spectral shape (blue > red)
//   4. Henyey–Greenstein phase normalization
//   5. Medium construction: density / albedo / falloff
//   6. Config defaults compile and have expected values
//   7. Transmittance spectral tests
//   8. Phase colour tests
//   9. Edge cases
// ─────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <cmath>

#include "core/types.h"
#include "core/spectrum.h"
#include "core/config.h"
#include "renderer/render_config.h"
#include "volume/medium.h"

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
    // For azimuthally symmetric p(cosθ), the φ integral reduces to 2π.
    const int N_theta = 500;
    double integral = 0.0;
    double d_theta = PI / N_theta;

    for (int it = 0; it < N_theta; ++it) {
        double theta = (it + 0.5) * d_theta;
        double cos_theta = cos(theta);
        double sin_theta = sin(theta);
        double p = rayleigh_phase((float)cos_theta);
        integral += p * sin_theta * d_theta;
    }
    // Multiply by 2π for the φ integral (no phi dependence → factor of 2π)
    integral *= 2.0 * PI;
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

