// ─────────────────────────────────────────────────────────────────────
// test_tri_photon_irradiance.cpp – Edge-case tests for per-triangle
// photon irradiance / count accumulation (heatmap feature)
// ─────────────────────────────────────────────────────────────────────
// Coverage:
//   §1 build_tri_photon_irradiance()
//       - Empty photon set
//       - Zero / negative num_tris
//       - Missing tri_id vector (undersized)
//       - Out-of-range triangle IDs (0xFFFFFFFF sentinel)
//       - Single photon, single triangle
//       - Multiple photons on same triangle (accumulation)
//       - Spread across many triangles (sparse)
//       - Hero flux path vs spectral_flux fallback
//       - Zero flux photons
//       - Negative flux values (shouldn't crash)
//   §2 build_tri_photon_count()
//       - Mirrors the same edge-case matrix for counts
//   §3 PhotonSoA::tri_id field (push_back / get round-trip)
// ─────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include "photon/tri_photon_irradiance.h"
#include "photon/photon.h"
#include "core/config.h"

#include <cstdint>
#include <cmath>
#include <vector>
#include <limits>
#include <numeric>

// ── Helpers ──────────────────────────────────────────────────────────

// Build a minimal PhotonSoA with only the fields needed for irradiance
// accumulation.  Positions / normals / dirs are irrelevant here.
static PhotonSoA make_photon_soa(
    const std::vector<uint32_t>& tri_ids,
    const std::vector<float>&    hero_fluxes,   // HERO_WAVELENGTHS per photon
    const std::vector<uint8_t>&  hero_counts)
{
    const size_t n = tri_ids.size();
    PhotonSoA soa;
    soa.pos_x.resize(n, 0.f); soa.pos_y.resize(n, 0.f); soa.pos_z.resize(n, 0.f);
    soa.wi_x.resize(n, 0.f);  soa.wi_y.resize(n, 0.f);  soa.wi_z.resize(n, 1.f);
    soa.norm_x.resize(n, 0.f); soa.norm_y.resize(n, 1.f); soa.norm_z.resize(n, 0.f);
    soa.spectral_flux.resize(n * NUM_LAMBDA, 0.f);
    soa.lambda_bin.resize(n * HERO_WAVELENGTHS, 0);
    soa.flux = hero_fluxes;
    soa.num_hero = hero_counts;
    soa.source_emissive_idx.resize(n, 0xFFFFu);
    soa.tri_id = tri_ids;
    soa.path_flags.resize(n, 0);
    soa.bounce_count.resize(n, 0);
    return soa;
}

// Overload: single hero flux per photon (fills slot 0, rest zero)
static PhotonSoA make_simple_soa(
    const std::vector<uint32_t>& tri_ids,
    const std::vector<float>&    flux_per_photon)
{
    const size_t n = tri_ids.size();
    std::vector<float> hero_flux(n * HERO_WAVELENGTHS, 0.f);
    std::vector<uint8_t> hero_count(n, 1);
    for (size_t i = 0; i < n; ++i)
        hero_flux[i * HERO_WAVELENGTHS] = flux_per_photon[i];
    return make_photon_soa(tri_ids, hero_flux, hero_count);
}

// =====================================================================
//  §1  build_tri_photon_irradiance
// =====================================================================

// 1.1 Empty photon set → all-zero irradiance
TEST(TriPhotonIrradiance, EmptyPhotons) {
    PhotonSoA empty;
    auto irr = build_tri_photon_irradiance(empty, 10);
    ASSERT_EQ(irr.size(), 10u);
    for (float v : irr)
        EXPECT_FLOAT_EQ(v, 0.f);
}

// 1.2 num_tris = 0 → empty result
TEST(TriPhotonIrradiance, ZeroTriangles) {
    auto soa = make_simple_soa({0}, {1.f});
    auto irr = build_tri_photon_irradiance(soa, 0);
    EXPECT_TRUE(irr.empty());
}

// 1.3 Negative num_tris → empty result (defensive)
TEST(TriPhotonIrradiance, NegativeNumTris) {
    auto soa = make_simple_soa({0}, {1.f});
    auto irr = build_tri_photon_irradiance(soa, -5);
    EXPECT_TRUE(irr.empty());
}

// 1.4 tri_id vector is empty (not populated) → all-zero
TEST(TriPhotonIrradiance, MissingTriIdVector) {
    PhotonSoA soa;
    soa.pos_x.resize(5, 0.f); soa.pos_y.resize(5); soa.pos_z.resize(5);
    soa.wi_x.resize(5); soa.wi_y.resize(5); soa.wi_z.resize(5);
    soa.norm_x.resize(5); soa.norm_y.resize(5); soa.norm_z.resize(5);
    soa.spectral_flux.resize(5 * NUM_LAMBDA, 0.f);
    soa.flux.resize(5 * HERO_WAVELENGTHS, 1.f);
    soa.num_hero.resize(5, 1);
    // tri_id intentionally left empty
    EXPECT_EQ(soa.size(), 5u);

    auto irr = build_tri_photon_irradiance(soa, 10);
    ASSERT_EQ(irr.size(), 10u);
    for (float v : irr)
        EXPECT_FLOAT_EQ(v, 0.f);
}

// 1.5 tri_id vector undersized (has some but fewer than photon count)
TEST(TriPhotonIrradiance, UndersizedTriId) {
    auto soa = make_simple_soa({0, 1, 2, 3, 4}, {1.f, 1.f, 1.f, 1.f, 1.f});
    soa.tri_id.resize(3); // only 3 IDs for 5 photons → should bail
    auto irr = build_tri_photon_irradiance(soa, 10);
    for (float v : irr)
        EXPECT_FLOAT_EQ(v, 0.f);
}

// 1.6 Out-of-range triangle ID (sentinel 0xFFFFFFFF) → skipped
TEST(TriPhotonIrradiance, SentinelTriIdSkipped) {
    auto soa = make_simple_soa({0xFFFFFFFFu, 0xFFFFFFFFu}, {5.f, 3.f});
    auto irr = build_tri_photon_irradiance(soa, 100);
    ASSERT_EQ(irr.size(), 100u);
    for (float v : irr)
        EXPECT_FLOAT_EQ(v, 0.f);
}

// 1.7 Out-of-range ID that is < 0xFFFFFFFF but >= num_tris
TEST(TriPhotonIrradiance, OutOfRangeTriId) {
    auto soa = make_simple_soa({50}, {1.f});
    auto irr = build_tri_photon_irradiance(soa, 10); // tri 50 >= 10
    for (float v : irr)
        EXPECT_FLOAT_EQ(v, 0.f);
}

// 1.8 Single photon on single triangle
TEST(TriPhotonIrradiance, SinglePhotonSingleTri) {
    auto soa = make_simple_soa({0}, {7.5f});
    auto irr = build_tri_photon_irradiance(soa, 1);
    ASSERT_EQ(irr.size(), 1u);
    EXPECT_FLOAT_EQ(irr[0], 7.5f);
}

// 1.9 Multiple photons accumulate on same triangle
TEST(TriPhotonIrradiance, AccumulationSameTriangle) {
    auto soa = make_simple_soa({3, 3, 3, 3}, {1.f, 2.f, 3.f, 4.f});
    auto irr = build_tri_photon_irradiance(soa, 10);
    EXPECT_FLOAT_EQ(irr[3], 10.f); // 1 + 2 + 3 + 4
    // Other triangles untouched
    EXPECT_FLOAT_EQ(irr[0], 0.f);
    EXPECT_FLOAT_EQ(irr[9], 0.f);
}

// 1.10 Photons spread across different triangles
TEST(TriPhotonIrradiance, SpreadAcrossTriangles) {
    auto soa = make_simple_soa({0, 1, 2, 5, 9}, {1.f, 2.f, 3.f, 4.f, 5.f});
    auto irr = build_tri_photon_irradiance(soa, 10);
    EXPECT_FLOAT_EQ(irr[0], 1.f);
    EXPECT_FLOAT_EQ(irr[1], 2.f);
    EXPECT_FLOAT_EQ(irr[2], 3.f);
    EXPECT_FLOAT_EQ(irr[3], 0.f);
    EXPECT_FLOAT_EQ(irr[5], 4.f);
    EXPECT_FLOAT_EQ(irr[9], 5.f);
}

// 1.11 Multiple hero wavelengths are summed
TEST(TriPhotonIrradiance, MultipleHeroWavelengths) {
    // 1 photon with HERO_WAVELENGTHS hero channels all set to 2.0
    std::vector<float> hero(HERO_WAVELENGTHS, 2.f);
    std::vector<uint8_t> nh = {(uint8_t)HERO_WAVELENGTHS};
    auto soa = make_photon_soa({0}, hero, nh);
    auto irr = build_tri_photon_irradiance(soa, 1);
    EXPECT_FLOAT_EQ(irr[0], 2.f * HERO_WAVELENGTHS);
}

// 1.12 num_hero < HERO_WAVELENGTHS → only valid channels summed
TEST(TriPhotonIrradiance, PartialHeroChannels) {
    // 1 photon, 2 out of HERO_WAVELENGTHS channels valid
    std::vector<float> hero(HERO_WAVELENGTHS, 0.f);
    hero[0] = 5.f;
    hero[1] = 3.f; // channel 1
    // channels 2..HERO_WAVELENGTHS-1 = 0 but should be ignored anyway
    if (HERO_WAVELENGTHS > 2) hero[2] = 99.f; // should NOT be counted
    std::vector<uint8_t> nh = {2};
    auto soa = make_photon_soa({0}, hero, nh);
    auto irr = build_tri_photon_irradiance(soa, 1);
    EXPECT_FLOAT_EQ(irr[0], 8.f); // 5 + 3
}

// 1.13 num_hero > HERO_WAVELENGTHS → clamped to HERO_WAVELENGTHS
TEST(TriPhotonIrradiance, ClampedHeroCount) {
    std::vector<float> hero(HERO_WAVELENGTHS, 1.f);
    std::vector<uint8_t> nh = {(uint8_t)(HERO_WAVELENGTHS + 5)}; // too many
    auto soa = make_photon_soa({0}, hero, nh);
    auto irr = build_tri_photon_irradiance(soa, 1);
    EXPECT_FLOAT_EQ(irr[0], (float)HERO_WAVELENGTHS); // clamped
}

// 1.14 Spectral fallback when hero data is absent
TEST(TriPhotonIrradiance, SpectralFallback) {
    PhotonSoA soa;
    soa.pos_x.resize(1); soa.pos_y.resize(1); soa.pos_z.resize(1);
    soa.wi_x.resize(1);  soa.wi_y.resize(1);  soa.wi_z.resize(1);
    soa.norm_x.resize(1); soa.norm_y.resize(1); soa.norm_z.resize(1);

    // Fill spectral flux: 0.5 per bin
    soa.spectral_flux.resize(NUM_LAMBDA, 0.5f);

    // Hero data intentionally too small → fallback path
    soa.flux.clear();
    soa.num_hero.clear();
    soa.lambda_bin.clear();

    soa.tri_id = {0};
    soa.source_emissive_idx.resize(1, 0xFFFFu);
    soa.path_flags.resize(1, 0);
    soa.bounce_count.resize(1, 0);

    auto irr = build_tri_photon_irradiance(soa, 1);
    EXPECT_FLOAT_EQ(irr[0], 0.5f * NUM_LAMBDA);
}

// 1.15 Zero-flux photon → triangle gets 0
TEST(TriPhotonIrradiance, ZeroFluxPhoton) {
    auto soa = make_simple_soa({0}, {0.f});
    auto irr = build_tri_photon_irradiance(soa, 1);
    EXPECT_FLOAT_EQ(irr[0], 0.f);
}

// 1.16 Negative flux → should accumulate without crash (no clamp)
TEST(TriPhotonIrradiance, NegativeFlux) {
    auto soa = make_simple_soa({0, 0}, {-3.f, -2.f});
    auto irr = build_tri_photon_irradiance(soa, 1);
    EXPECT_FLOAT_EQ(irr[0], -5.f);
}

// 1.17 Mixed valid + out-of-range IDs
TEST(TriPhotonIrradiance, MixedValidAndInvalidIds) {
    auto soa = make_simple_soa(
        {0, 0xFFFFFFFFu, 1, 999, 2},
        {1.f, 10.f, 2.f, 20.f, 3.f});
    auto irr = build_tri_photon_irradiance(soa, 5);
    EXPECT_FLOAT_EQ(irr[0], 1.f);
    EXPECT_FLOAT_EQ(irr[1], 2.f);
    EXPECT_FLOAT_EQ(irr[2], 3.f);
    EXPECT_FLOAT_EQ(irr[3], 0.f);
    EXPECT_FLOAT_EQ(irr[4], 0.f);
}

// 1.18 Large triangle count, sparse photons
TEST(TriPhotonIrradiance, LargeSparseScene) {
    const int N_TRIS = 100000;
    auto soa = make_simple_soa({0, 50000, 99999}, {1.f, 2.f, 3.f});
    auto irr = build_tri_photon_irradiance(soa, N_TRIS);
    ASSERT_EQ(irr.size(), (size_t)N_TRIS);
    EXPECT_FLOAT_EQ(irr[0], 1.f);
    EXPECT_FLOAT_EQ(irr[50000], 2.f);
    EXPECT_FLOAT_EQ(irr[99999], 3.f);
    // Spot-check a few empty slots
    EXPECT_FLOAT_EQ(irr[1], 0.f);
    EXPECT_FLOAT_EQ(irr[50001], 0.f);
}

// 1.19 Boundary: triangle ID == num_tris (off by one)
TEST(TriPhotonIrradiance, BoundaryTriIdEqualsNumTris) {
    auto soa = make_simple_soa({5}, {10.f}); // tri 5, but num_tris=5 → out of range
    auto irr = build_tri_photon_irradiance(soa, 5);
    for (float v : irr)
        EXPECT_FLOAT_EQ(v, 0.f);
}

// 1.20 Boundary: triangle ID == num_tris - 1 (last valid)
TEST(TriPhotonIrradiance, BoundaryLastValidTriId) {
    auto soa = make_simple_soa({4}, {10.f}); // tri 4 with num_tris=5
    auto irr = build_tri_photon_irradiance(soa, 5);
    EXPECT_FLOAT_EQ(irr[4], 10.f);
}

// =====================================================================
//  §2  build_tri_photon_count
// =====================================================================

// 2.1 Empty photons
TEST(TriPhotonCount, EmptyPhotons) {
    PhotonSoA empty;
    auto cnt = build_tri_photon_count(empty, 10);
    ASSERT_EQ(cnt.size(), 10u);
    for (uint32_t c : cnt)
        EXPECT_EQ(c, 0u);
}

// 2.2 Zero triangles
TEST(TriPhotonCount, ZeroTriangles) {
    auto soa = make_simple_soa({0}, {1.f});
    auto cnt = build_tri_photon_count(soa, 0);
    EXPECT_TRUE(cnt.empty());
}

// 2.3 Negative num_tris
TEST(TriPhotonCount, NegativeNumTris) {
    auto soa = make_simple_soa({0}, {1.f});
    auto cnt = build_tri_photon_count(soa, -1);
    EXPECT_TRUE(cnt.empty());
}

// 2.4 Missing tri_id
TEST(TriPhotonCount, MissingTriId) {
    PhotonSoA soa;
    soa.pos_x.resize(3); soa.pos_y.resize(3); soa.pos_z.resize(3);
    // tri_id empty
    auto cnt = build_tri_photon_count(soa, 10);
    for (uint32_t c : cnt)
        EXPECT_EQ(c, 0u);
}

// 2.5 Sentinel skipped
TEST(TriPhotonCount, SentinelSkipped) {
    auto soa = make_simple_soa({0xFFFFFFFFu}, {1.f});
    auto cnt = build_tri_photon_count(soa, 10);
    for (uint32_t c : cnt)
        EXPECT_EQ(c, 0u);
}

// 2.6 Out-of-range
TEST(TriPhotonCount, OutOfRangeSkipped) {
    auto soa = make_simple_soa({100}, {1.f});
    auto cnt = build_tri_photon_count(soa, 10);
    for (uint32_t c : cnt)
        EXPECT_EQ(c, 0u);
}

// 2.7 Single photon
TEST(TriPhotonCount, SinglePhoton) {
    auto soa = make_simple_soa({3}, {7.f});
    auto cnt = build_tri_photon_count(soa, 5);
    EXPECT_EQ(cnt[3], 1u);
    EXPECT_EQ(cnt[0], 0u);
}

// 2.8 Multiple photons same triangle
TEST(TriPhotonCount, MultipleSameTriangle) {
    auto soa = make_simple_soa({2, 2, 2, 2, 2}, {1.f, 1.f, 1.f, 1.f, 1.f});
    auto cnt = build_tri_photon_count(soa, 5);
    EXPECT_EQ(cnt[2], 5u);
}

// 2.9 Spread
TEST(TriPhotonCount, SpreadAcrossTriangles) {
    auto soa = make_simple_soa({0, 1, 2, 3, 4}, {1.f, 1.f, 1.f, 1.f, 1.f});
    auto cnt = build_tri_photon_count(soa, 5);
    for (int i = 0; i < 5; ++i)
        EXPECT_EQ(cnt[i], 1u);
}

// 2.10 Mixed valid + invalid
TEST(TriPhotonCount, MixedValidInvalid) {
    auto soa = make_simple_soa(
        {0, 0xFFFFFFFFu, 1, 999, 0},
        {1.f, 1.f, 1.f, 1.f, 1.f});
    auto cnt = build_tri_photon_count(soa, 5);
    EXPECT_EQ(cnt[0], 2u);  // photon 0 and photon 4
    EXPECT_EQ(cnt[1], 1u);
    EXPECT_EQ(cnt[2], 0u);
}

// 2.11 Boundary: ID == num_tris
TEST(TriPhotonCount, BoundaryIdEqualsNumTris) {
    auto soa = make_simple_soa({5}, {1.f});
    auto cnt = build_tri_photon_count(soa, 5);
    for (uint32_t c : cnt)
        EXPECT_EQ(c, 0u);
}

// 2.12 Large sparse
TEST(TriPhotonCount, LargeSparse) {
    const int N = 50000;
    auto soa = make_simple_soa({0, (uint32_t)(N - 1)}, {1.f, 1.f});
    auto cnt = build_tri_photon_count(soa, N);
    EXPECT_EQ(cnt[0], 1u);
    EXPECT_EQ(cnt[N - 1], 1u);
    uint32_t total = 0;
    for (uint32_t c : cnt) total += c;
    EXPECT_EQ(total, 2u);
}

// =====================================================================
//  §3  PhotonSoA::tri_id round-trip (push_back / get)
// =====================================================================

TEST(PhotonSoATriId, PushBackGetRoundTrip) {
    PhotonSoA soa;

    Photon p1;
    p1.triangle_id = 42;
    soa.push_back(p1);

    Photon p2;
    p2.triangle_id = 0xFFFFFFFFu;
    soa.push_back(p2);

    Photon p3;
    p3.triangle_id = 0;
    soa.push_back(p3);

    ASSERT_EQ(soa.size(), 3u);
    EXPECT_EQ(soa.tri_id[0], 42u);
    EXPECT_EQ(soa.tri_id[1], 0xFFFFFFFFu);
    EXPECT_EQ(soa.tri_id[2], 0u);

    // get() round-trip
    EXPECT_EQ(soa.get(0).triangle_id, 42u);
    EXPECT_EQ(soa.get(1).triangle_id, 0xFFFFFFFFu);
    EXPECT_EQ(soa.get(2).triangle_id, 0u);
}

TEST(PhotonSoATriId, ResizeDefault) {
    PhotonSoA soa;
    soa.resize(10);
    ASSERT_EQ(soa.tri_id.size(), 10u);
    for (size_t i = 0; i < 10; ++i)
        EXPECT_EQ(soa.tri_id[i], 0xFFFFFFFFu); // default sentinel
}

TEST(PhotonSoATriId, AppendMergesTriIds) {
    PhotonSoA a, b;
    a.resize(2);
    a.tri_id[0] = 10;
    a.tri_id[1] = 20;

    b.resize(3);
    b.tri_id[0] = 30;
    b.tri_id[1] = 40;
    b.tri_id[2] = 50;

    a.append(b);
    ASSERT_EQ(a.size(), 5u);
    EXPECT_EQ(a.tri_id[0], 10u);
    EXPECT_EQ(a.tri_id[1], 20u);
    EXPECT_EQ(a.tri_id[2], 30u);
    EXPECT_EQ(a.tri_id[3], 40u);
    EXPECT_EQ(a.tri_id[4], 50u);
}

TEST(PhotonSoATriId, ClearResetsTriId) {
    PhotonSoA soa;
    soa.resize(5);
    soa.tri_id[0] = 100;
    soa.clear();
    EXPECT_EQ(soa.size(), 0u);
    EXPECT_TRUE(soa.tri_id.empty());
}
