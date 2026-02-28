// ─────────────────────────────────────────────────────────────────────
// test_light_cache.cpp – Unit tests for per-cell light importance cache
// ─────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include "renderer/light_cache.h"
#include "photon/photon.h"
#include "core/config.h"

#include <cmath>
#include <algorithm>
#include <set>
#include <numeric>

// ── Helper: create a photon at a given position with a source emissive ──
static Photon make_photon(float x, float y, float z,
                          uint16_t source_idx, float flux_val = 1.0f) {
    Photon p;
    p.position   = make_f3(x, y, z);
    p.wi         = make_f3(0, 1, 0);
    p.geom_normal = make_f3(0, 1, 0);
    p.source_emissive_idx = source_idx;
    for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
        p.lambda_bin[h] = 0;
        p.flux[h] = flux_val;
    }
    return p;
}

// ── Helper: build a PhotonSoA from a vector of Photon ───────────────
static PhotonSoA make_soa(const std::vector<Photon>& photons) {
    PhotonSoA soa;
    soa.reserve(photons.size());
    for (const auto& p : photons)
        soa.push_back(p);
    return soa;
}

// ==================================================================
// TEST 1: Empty photon map produces valid but empty cache
// ==================================================================
TEST(LightCache, EmptyPhotonMap) {
    PhotonSoA soa;
    LightCache cache;
    cache.build(soa, 0.1f);

    EXPECT_TRUE(cache.valid());
    EXPECT_EQ(cache.entries.size(), (size_t)LIGHT_CACHE_TABLE_SIZE * NEE_CELL_TOP_K);
    EXPECT_EQ(cache.count.size(), (size_t)LIGHT_CACHE_TABLE_SIZE);

    // All cells should be empty
    for (uint32_t k = 0; k < LIGHT_CACHE_TABLE_SIZE; ++k) {
        EXPECT_EQ(cache.count[k], 0);
        EXPECT_FLOAT_EQ(cache.total_importance[k], 0.f);
    }
}

// ==================================================================
// TEST 2: Single photon → single light in one cell
// ==================================================================
TEST(LightCache, SinglePhotonSingleCell) {
    std::vector<Photon> photons;
    photons.push_back(make_photon(0.5f, 0.5f, 0.5f, 3, 2.0f));
    PhotonSoA soa = make_soa(photons);

    LightCache cache;
    cache.build(soa, 0.1f);

    int out_count = 0;
    float out_total = 0.f;
    const CellLightEntry* entries = cache.query(
        make_f3(0.5f, 0.5f, 0.5f), out_count, out_total);

    ASSERT_NE(entries, nullptr);
    ASSERT_EQ(out_count, 1);
    EXPECT_EQ(entries[0].emissive_idx, 3);
    EXPECT_GT(out_total, 0.f);
    // Total importance should equal the flux: HERO_WAVELENGTHS × 2.0
    EXPECT_FLOAT_EQ(out_total, HERO_WAVELENGTHS * 2.0f);
}

// ==================================================================
// TEST 3: Multiple lights → ranked by importance (top-K selection)
// ==================================================================
TEST(LightCache, TopKSelection) {
    // Place many photons from different lights at the same position.
    // Light 0: 100 photons  (most important)
    // Light 1:  50 photons
    // Light 2:  10 photons
    // Light 3:   1 photon  (least)
    std::vector<Photon> photons;
    float3 pos = make_f3(1.0f, 1.0f, 1.0f);
    for (int i = 0; i < 100; ++i) photons.push_back(make_photon(pos.x, pos.y, pos.z, 0, 1.0f));
    for (int i = 0; i <  50; ++i) photons.push_back(make_photon(pos.x, pos.y, pos.z, 1, 1.0f));
    for (int i = 0; i <  10; ++i) photons.push_back(make_photon(pos.x, pos.y, pos.z, 2, 1.0f));
    for (int i = 0; i <   1; ++i) photons.push_back(make_photon(pos.x, pos.y, pos.z, 3, 1.0f));
    PhotonSoA soa = make_soa(photons);

    LightCache cache;
    cache.build(soa, 0.1f);

    int out_count = 0;
    float out_total = 0.f;
    const CellLightEntry* entries = cache.query(pos, out_count, out_total);

    ASSERT_NE(entries, nullptr);
    ASSERT_EQ(out_count, 4);

    // Entries should be sorted descending by importance
    EXPECT_EQ(entries[0].emissive_idx, 0);
    EXPECT_EQ(entries[1].emissive_idx, 1);
    EXPECT_EQ(entries[2].emissive_idx, 2);
    EXPECT_EQ(entries[3].emissive_idx, 3);

    // Importance ordering
    for (int i = 0; i < out_count - 1; ++i) {
        EXPECT_GE(entries[i].importance, entries[i + 1].importance);
    }
}

// ==================================================================
// TEST 4: More than TOP_K lights → only TOP_K kept
// ==================================================================
TEST(LightCache, MoreThanTopK) {
    std::vector<Photon> photons;
    float3 pos = make_f3(2.0f, 2.0f, 2.0f);

    // Create 20 lights (> NEE_CELL_TOP_K = 16)
    int num_lights = 20;
    for (int l = 0; l < num_lights; ++l) {
        int n_photons = 100 - l * 5;  // light 0 is brightest
        for (int i = 0; i < n_photons; ++i)
            photons.push_back(make_photon(pos.x, pos.y, pos.z, (uint16_t)l, 1.0f));
    }
    PhotonSoA soa = make_soa(photons);

    LightCache cache;
    cache.build(soa, 0.1f);

    int out_count = 0;
    float out_total = 0.f;
    const CellLightEntry* entries = cache.query(pos, out_count, out_total);

    ASSERT_NE(entries, nullptr);
    ASSERT_EQ(out_count, NEE_CELL_TOP_K);  // clamped to top-K

    // The top-K lights should be the 16 brightest (indices 0..15)
    std::set<uint16_t> top_k_set;
    for (int i = 0; i < out_count; ++i) {
        top_k_set.insert(entries[i].emissive_idx);
    }
    for (int l = 0; l < NEE_CELL_TOP_K; ++l) {
        EXPECT_TRUE(top_k_set.count((uint16_t)l) > 0)
            << "Expected light " << l << " in top-K set";
    }
}

// ==================================================================
// TEST 5: Different cells get different light distributions
// ==================================================================
TEST(LightCache, SpatialLocality) {
    std::vector<Photon> photons;

    // Cell A (around 0,0,0): dominated by light 0
    for (int i = 0; i < 50; ++i)
        photons.push_back(make_photon(0.01f, 0.01f, 0.01f, 0, 1.0f));
    for (int i = 0; i < 5; ++i)
        photons.push_back(make_photon(0.01f, 0.01f, 0.01f, 1, 1.0f));

    // Cell B (around 5,5,5): dominated by light 1
    for (int i = 0; i < 50; ++i)
        photons.push_back(make_photon(5.0f, 5.0f, 5.0f, 1, 1.0f));
    for (int i = 0; i < 5; ++i)
        photons.push_back(make_photon(5.0f, 5.0f, 5.0f, 0, 1.0f));

    PhotonSoA soa = make_soa(photons);

    LightCache cache;
    cache.build(soa, 0.5f);  // large cells to avoid hash collision

    // Query cell A
    int countA = 0; float totalA = 0.f;
    const CellLightEntry* entriesA = cache.query(
        make_f3(0.01f, 0.01f, 0.01f), countA, totalA);
    ASSERT_NE(entriesA, nullptr);
    ASSERT_GE(countA, 2);
    EXPECT_EQ(entriesA[0].emissive_idx, 0);  // light 0 dominates cell A

    // Query cell B
    int countB = 0; float totalB = 0.f;
    const CellLightEntry* entriesB = cache.query(
        make_f3(5.0f, 5.0f, 5.0f), countB, totalB);
    ASSERT_NE(entriesB, nullptr);
    ASSERT_GE(countB, 2);
    EXPECT_EQ(entriesB[0].emissive_idx, 1);  // light 1 dominates cell B
}

// ==================================================================
// TEST 6: Photons with invalid source (0xFFFF) are skipped
// ==================================================================
TEST(LightCache, InvalidSourceSkipped) {
    std::vector<Photon> photons;
    // 10 photons without valid source
    for (int i = 0; i < 10; ++i)
        photons.push_back(make_photon(0.5f, 0.5f, 0.5f, 0xFFFFu, 1.0f));

    // 1 photon with valid source
    photons.push_back(make_photon(0.5f, 0.5f, 0.5f, 7, 1.0f));

    PhotonSoA soa = make_soa(photons);

    LightCache cache;
    cache.build(soa, 0.1f);

    int out_count = 0; float out_total = 0.f;
    const CellLightEntry* entries = cache.query(
        make_f3(0.5f, 0.5f, 0.5f), out_count, out_total);

    ASSERT_NE(entries, nullptr);
    ASSERT_EQ(out_count, 1);
    EXPECT_EQ(entries[0].emissive_idx, 7);
}

// ==================================================================
// TEST 7: Empty cell query returns nullptr
// ==================================================================
TEST(LightCache, EmptyCellQuery) {
    std::vector<Photon> photons;
    // All photons at one location
    for (int i = 0; i < 10; ++i)
        photons.push_back(make_photon(0.5f, 0.5f, 0.5f, 0, 1.0f));

    PhotonSoA soa = make_soa(photons);

    LightCache cache;
    cache.build(soa, 0.1f);

    // Query far away (likely a different cell)
    int out_count = 0; float out_total = 0.f;
    const CellLightEntry* entries = cache.query(
        make_f3(100.0f, 100.0f, 100.0f), out_count, out_total);

    // Should return nullptr with count=0 (or entries with count=0)
    EXPECT_EQ(out_count, 0);
    EXPECT_FLOAT_EQ(out_total, 0.f);
}

// ==================================================================
// TEST 8: Total importance equals sum of entry importances
// ==================================================================
TEST(LightCache, TotalImportanceConsistency) {
    std::vector<Photon> photons;
    float3 pos = make_f3(1.0f, 1.0f, 1.0f);
    for (int l = 0; l < 5; ++l) {
        float flux = (float)(l + 1) * 3.0f;
        for (int i = 0; i < 10; ++i)
            photons.push_back(make_photon(pos.x, pos.y, pos.z, (uint16_t)l, flux));
    }
    PhotonSoA soa = make_soa(photons);

    LightCache cache;
    cache.build(soa, 0.1f);

    int out_count = 0; float out_total = 0.f;
    const CellLightEntry* entries = cache.query(pos, out_count, out_total);

    ASSERT_NE(entries, nullptr);
    ASSERT_EQ(out_count, 5);

    // Sum of entry importances should equal total_importance
    float sum = 0.f;
    for (int i = 0; i < out_count; ++i)
        sum += entries[i].importance;
    EXPECT_NEAR(out_total, sum, 1e-3f);
}

// ==================================================================
// TEST 9: Cell size is stored correctly
// ==================================================================
TEST(LightCache, CellSizeStored) {
    PhotonSoA soa;
    LightCache cache;
    cache.build(soa, 0.42f);
    EXPECT_FLOAT_EQ(cache.cell_size, 0.42f);
}

// ==================================================================
// TEST 10: valid() returns false before build
// ==================================================================
TEST(LightCache, ValidBeforeBuild) {
    LightCache cache;
    EXPECT_FALSE(cache.valid());
}

// ==================================================================
// TEST 11: Zero-flux photons are ignored
// ==================================================================
TEST(LightCache, ZeroFluxIgnored) {
    std::vector<Photon> photons;
    photons.push_back(make_photon(0.5f, 0.5f, 0.5f, 5, 0.0f));  // zero flux
    photons.push_back(make_photon(0.5f, 0.5f, 0.5f, 6, 1.0f));  // valid

    PhotonSoA soa = make_soa(photons);

    LightCache cache;
    cache.build(soa, 0.1f);

    int out_count = 0; float out_total = 0.f;
    const CellLightEntry* entries = cache.query(
        make_f3(0.5f, 0.5f, 0.5f), out_count, out_total);

    ASSERT_NE(entries, nullptr);
    ASSERT_EQ(out_count, 1);
    EXPECT_EQ(entries[0].emissive_idx, 6);
}

// ==================================================================
// TEST 12: Cache hash function is deterministic
// ==================================================================
TEST(LightCache, HashDeterministic) {
    int3 cell = make_i3(10, -5, 3);
    uint32_t h1 = LightCache::cache_cell_key(cell);
    uint32_t h2 = LightCache::cache_cell_key(cell);
    EXPECT_EQ(h1, h2);
    EXPECT_LT(h1, LIGHT_CACHE_TABLE_SIZE);
}

// ==================================================================
// TEST 13: Importance accumulates across multiple photons from same light
// ==================================================================
TEST(LightCache, ImportanceAccumulation) {
    std::vector<Photon> photons;
    float3 pos = make_f3(0.5f, 0.5f, 0.5f);

    // 5 photons from light 2, each with flux=3.0
    for (int i = 0; i < 5; ++i)
        photons.push_back(make_photon(pos.x, pos.y, pos.z, 2, 3.0f));

    PhotonSoA soa = make_soa(photons);

    LightCache cache;
    cache.build(soa, 0.1f);

    int out_count = 0; float out_total = 0.f;
    const CellLightEntry* entries = cache.query(pos, out_count, out_total);

    ASSERT_NE(entries, nullptr);
    ASSERT_EQ(out_count, 1);
    EXPECT_EQ(entries[0].emissive_idx, 2);

    // Importance = 5 photons × HERO_WAVELENGTHS × 3.0
    float expected = 5.f * HERO_WAVELENGTHS * 3.0f;
    EXPECT_NEAR(entries[0].importance, expected, 1e-3f);
}

// ==================================================================
// TEST 14: source_emissive_idx in PhotonSoA round-trips through push_back/get
// ==================================================================
TEST(LightCache, PhotonSoASourceRoundTrip) {
    Photon p = make_photon(1.f, 2.f, 3.f, 42, 1.0f);
    PhotonSoA soa;
    soa.push_back(p);

    ASSERT_EQ(soa.size(), 1u);
    ASSERT_EQ(soa.source_emissive_idx.size(), 1u);
    EXPECT_EQ(soa.source_emissive_idx[0], 42);

    Photon out = soa.get(0);
    EXPECT_EQ(out.source_emissive_idx, 42);
}

// ==================================================================
// TEST 15: NEE_CELL_TOP_K and LIGHT_CACHE_TABLE_SIZE are reasonable
// ==================================================================
TEST(LightCache, Constants) {
    EXPECT_GE(NEE_CELL_TOP_K, 4);
    EXPECT_LE(NEE_CELL_TOP_K, 64);
    EXPECT_GE(LIGHT_CACHE_TABLE_SIZE, 1024u);
    EXPECT_GT(NEE_CACHE_FALLBACK_PROB, 0.f);
    EXPECT_LT(NEE_CACHE_FALLBACK_PROB, 1.f);
}
