// ─────────────────────────────────────────────────────────────────────
// tests/unit/test_photon.cpp – Photon system unit tests
//
// Stage 7: Photon struct, storage layout, caustic flags.
// CPU-only struct and type tests.
// ─────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include "photon/photon.h"
#include "core/types.h"
#include "core/config.h"
#include <cmath>

// ── Photon struct ───────────────────────────────────────────────────

TEST(Photon, StructFields) {
    Photon p{};
    p.position = make_f3(1, 2, 3);
    p.wi       = make_f3(0, 0, -1);
    p.geo_normal = make_f3(0, 1, 0);
    p.flux     = make_f3(0.5f, 0.5f, 0.5f);
    p.triangle_id = 2;

    EXPECT_FLOAT_EQ(p.position.x, 1.0f);
    EXPECT_FLOAT_EQ(p.flux.x, 0.5f);
    EXPECT_EQ(p.triangle_id, 2u);
}

TEST(Photon, FluxEnergy) {
    Photon p{};
    p.flux = make_f3(1.0f, 0.5f, 0.25f);
    // luminance = 0.2126*1 + 0.7152*0.5 + 0.0722*0.25
    float energy = 0.2126f * p.flux.x + 0.7152f * p.flux.y + 0.0722f * p.flux.z;
    EXPECT_GT(energy, 0.0f);
    EXPECT_LT(energy, 2.0f);
}

// ── Caustic flags ───────────────────────────────────────────────────

TEST(Photon, CausticFlagDefault) {
    Photon p{};
    p.path_flags = 0;
    EXPECT_EQ(p.path_flags & PHOTON_FLAG_CAUSTIC_SPECULAR, 0u);
    EXPECT_EQ(p.is_caustic, 0u);
}

TEST(Photon, CausticFlagSet) {
    Photon p{};
    p.path_flags = PHOTON_FLAG_CAUSTIC_SPECULAR;
    EXPECT_NE(p.path_flags & PHOTON_FLAG_CAUSTIC_SPECULAR, 0u);
}

TEST(Photon, GlassTraversedFlag) {
    Photon p{};
    p.path_flags = PHOTON_FLAG_TRAVERSED_GLASS;
    EXPECT_NE(p.path_flags & PHOTON_FLAG_TRAVERSED_GLASS, 0u);
    EXPECT_EQ(p.path_flags & PHOTON_FLAG_CAUSTIC_SPECULAR, 0u);
}

TEST(Photon, CombinedFlags) {
    Photon p{};
    p.path_flags = PHOTON_FLAG_CAUSTIC_SPECULAR | PHOTON_FLAG_TRAVERSED_GLASS;
    EXPECT_NE(p.path_flags & PHOTON_FLAG_CAUSTIC_SPECULAR, 0u);
    EXPECT_NE(p.path_flags & PHOTON_FLAG_TRAVERSED_GLASS, 0u);
}

// ── Photon struct metadata ───────────────────────────────────────────

TEST(Photon, IsSmallPOD) {
    // Photon should be a reasonably sized struct for GPU storage
    EXPECT_LT(sizeof(Photon), 128u);
}

TEST(Photon, IsCausticField) {
    Photon p{};
    p.is_caustic = 0;
    EXPECT_FALSE(p.is_caustic);
    p.is_caustic = 1;
    EXPECT_TRUE(p.is_caustic);
}

TEST(Photon, DispersionFlag) {
    Photon p{};
    p.path_flags = PHOTON_FLAG_DISPERSION;
    EXPECT_NE(p.path_flags & PHOTON_FLAG_DISPERSION, 0u);
}
