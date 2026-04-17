// ─────────────────────────────────────────────────────────────────────
// test_speed_tweaks.cpp – Unit & integration tests for speed tweaks
// ─────────────────────────────────────────────────────────────────────
// Coverage:
//   §1  Chromatic dispersion: Cauchy IOR, Material.Tf, per-bin Fresnel
//   §2  Photon path flags and bounce count
//   §4  Glass BSDF + Tf spectral attenuation
//   §5  Energy conservation with dispersion
//   §7  IORStack nested dielectric tracking
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
#include "scene/material.h"
#include "core/ior_stack.h"
#include "bsdf/bsdf.h"
#include "photon/photon.h"
#include "scene/scene.h"
#include "scene/obj_loader.h"

static constexpr float kTol   = 1e-5f;
static constexpr float kLoose = 1e-3f;

// ── Helpers ─────────────────────────────────────────────────────────

static Photon make_test_photon_flags(float3 pos, float3 wi, float3 norm,
                                     float flux, uint8_t flags) {
    Photon p;
    p.position      = pos;
    p.wi            = wi;
    p.geom_normal   = norm;
    p.spectral_flux = Spectrum::constant(flux);
    p.path_flags    = flags;
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
    EXPECT_EQ(PHOTON_FLAG_VOLUME_SCATTER,  0x04);
    EXPECT_EQ(PHOTON_FLAG_DISPERSION,      0x08);
    EXPECT_EQ(PHOTON_FLAG_CAUSTIC_SPECULAR, 0x10);
}

TEST(PhotonFlags, SoAPushBackPreservesFlags) {
    PhotonSoA soa;
    Photon p = make_test_photon_flags(
        make_f3(1,2,3), make_f3(0,1,0), make_f3(0,0,1),
        1.0f, PHOTON_FLAG_TRAVERSED_GLASS | PHOTON_FLAG_DISPERSION);

    soa.push_back(p);
    Photon got = soa.get(0);

    EXPECT_EQ(got.path_flags, PHOTON_FLAG_TRAVERSED_GLASS | PHOTON_FLAG_DISPERSION);
}

TEST(PhotonFlags, SoAResizeClear) {
    PhotonSoA soa;
    soa.resize(10);
    EXPECT_EQ(soa.path_flags.size(), 10u);

    soa.clear();
    EXPECT_EQ(soa.path_flags.size(), 0u);
}

TEST(PhotonFlags, DefaultsAreZero) {
    Photon p;
    EXPECT_EQ(p.path_flags, 0);
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
// §7  IOR STACK – nested dielectric tracking
// =====================================================================

TEST(IORStack, DefaultIsAir) {
    IORStack stack;
    EXPECT_FLOAT_EQ(stack.top(), 1.0f);
}

TEST(IORStack, PushPop) {
    IORStack stack;

    stack.push(1.5f);
    EXPECT_FLOAT_EQ(stack.top(), 1.5f);

    stack.push(1.33f);
    EXPECT_FLOAT_EQ(stack.top(), 1.33f);

    stack.pop();
    EXPECT_FLOAT_EQ(stack.top(), 1.5f);

    stack.pop();
    EXPECT_FLOAT_EQ(stack.top(), 1.0f);
}

TEST(IORStack, Overflow) {
    IORStack stack;
    // Push more than MAX_DEPTH entries
    for (int i = 0; i < 10; ++i) {
        stack.push(1.0f + 0.1f * i);
    }
    // Should not crash, should cap at MAX_DEPTH
    EXPECT_GT(stack.top(), 1.0f);
}

TEST(IORStack, Underflow) {
    IORStack stack;
    stack.pop();  // Pop from empty
    EXPECT_FLOAT_EQ(stack.top(), 1.0f); // Should remain at air
}
