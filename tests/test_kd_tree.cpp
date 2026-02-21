// ─────────────────────────────────────────────────────────────────────
// test_kd_tree.cpp – Unit tests for KD-tree spatial index
// ─────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include "photon/kd_tree.h"
#include "photon/hash_grid.h"
#include "core/random.h"

#include <set>
#include <cmath>
#include <algorithm>

// ── Helper: create a photon map with N random photons ───────────────
static PhotonSoA make_random_photons(int n, uint32_t seed = 42) {
    PhotonSoA photons;
    photons.reserve(n);
    PCGRng rng = PCGRng::seed(seed, 1);
    for (int i = 0; i < n; ++i) {
        Photon p;
        p.position = make_f3(
            rng.next_float() * 2.0f - 1.0f,
            rng.next_float() * 2.0f - 1.0f,
            rng.next_float() * 2.0f - 1.0f
        );
        p.wi = make_f3(0, 1, 0);
        p.geom_normal = make_f3(0, 1, 0);
        p.lambda_bin = 0;
        p.flux = 1.0f;
        photons.push_back(p);
    }
    return photons;
}

// ── Helper: brute-force range query ─────────────────────────────────
static std::set<uint32_t> brute_force_range(
    float3 pos, float radius, const PhotonSoA& photons)
{
    std::set<uint32_t> result;
    float r2 = radius * radius;
    for (size_t i = 0; i < photons.size(); ++i) {
        float dx = photons.pos_x[i] - pos.x;
        float dy = photons.pos_y[i] - pos.y;
        float dz = photons.pos_z[i] - pos.z;
        if (dx*dx + dy*dy + dz*dz <= r2)
            result.insert((uint32_t)i);
    }
    return result;
}

// ── Helper: brute-force k-NN ────────────────────────────────────────
static std::vector<uint32_t> brute_force_knn(
    float3 pos, int k, const PhotonSoA& photons, float& out_max_dist2)
{
    std::vector<std::pair<float, uint32_t>> dists;
    for (size_t i = 0; i < photons.size(); ++i) {
        float dx = photons.pos_x[i] - pos.x;
        float dy = photons.pos_y[i] - pos.y;
        float dz = photons.pos_z[i] - pos.z;
        dists.push_back({dx*dx + dy*dy + dz*dz, (uint32_t)i});
    }
    std::sort(dists.begin(), dists.end());
    int actual_k = std::min(k, (int)dists.size());
    std::vector<uint32_t> result;
    for (int i = 0; i < actual_k; ++i)
        result.push_back(dists[i].second);
    out_max_dist2 = (actual_k > 0) ? dists[actual_k - 1].first : 0.0f;
    return result;
}

// ═════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════

TEST(KDTree, BuildEmpty) {
    PhotonSoA empty;
    KDTree tree;
    tree.build(empty);
    EXPECT_TRUE(tree.empty());
    EXPECT_EQ(tree.node_count(), 0u);
}

TEST(KDTree, BuildSinglePhoton) {
    PhotonSoA photons;
    Photon p;
    p.position = make_f3(1.0f, 2.0f, 3.0f);
    p.wi = make_f3(0, 1, 0);
    p.geom_normal = make_f3(0, 1, 0);
    p.lambda_bin = 0;
    p.flux = 1.0f;
    photons.push_back(p);

    KDTree tree;
    tree.build(photons);
    EXPECT_FALSE(tree.empty());
    EXPECT_GE(tree.node_count(), 1u);
}

TEST(KDTree, BuildMany) {
    auto photons = make_random_photons(10000);
    KDTree tree;
    tree.build(photons);
    EXPECT_FALSE(tree.empty());
    // A KD-tree with 10000 photons should have multiple nodes
    EXPECT_GT(tree.node_count(), 1u);
}

TEST(KDTree, RangeQueryEmpty) {
    KDTree tree;
    PhotonSoA empty;
    int count = 0;
    tree.query(make_f3(0, 0, 0), 1.0f, empty,
        [&](uint32_t, float) { ++count; });
    EXPECT_EQ(count, 0);
}

TEST(KDTree, RangeQuerySingle) {
    PhotonSoA photons;
    Photon p;
    p.position = make_f3(0.5f, 0.5f, 0.5f);
    p.wi = make_f3(0, 1, 0);
    p.geom_normal = make_f3(0, 1, 0);
    p.lambda_bin = 0;
    p.flux = 1.0f;
    photons.push_back(p);

    KDTree tree;
    tree.build(photons);

    // Query within range
    int count = 0;
    tree.query(make_f3(0.5f, 0.5f, 0.5f), 0.1f, photons,
        [&](uint32_t idx, float dist2) {
            EXPECT_EQ(idx, 0u);
            EXPECT_NEAR(dist2, 0.0f, 1e-6f);
            ++count;
        });
    EXPECT_EQ(count, 1);

    // Query out of range
    count = 0;
    tree.query(make_f3(10.0f, 10.0f, 10.0f), 0.1f, photons,
        [&](uint32_t, float) { ++count; });
    EXPECT_EQ(count, 0);
}

TEST(KDTree, RangeQueryMatchesBruteForce) {
    auto photons = make_random_photons(5000);
    KDTree tree;
    tree.build(photons);

    PCGRng rng = PCGRng::seed(123, 1);

    // Test 20 random query points with varying radii
    for (int q = 0; q < 20; ++q) {
        float3 query_pos = make_f3(
            rng.next_float() * 2.0f - 1.0f,
            rng.next_float() * 2.0f - 1.0f,
            rng.next_float() * 2.0f - 1.0f
        );
        float radius = 0.05f + rng.next_float() * 0.3f;

        auto brute = brute_force_range(query_pos, radius, photons);

        std::set<uint32_t> kd_result;
        tree.query(query_pos, radius, photons,
            [&](uint32_t idx, float) { kd_result.insert(idx); });

        EXPECT_EQ(kd_result, brute)
            << "Range query mismatch at q=" << q
            << " radius=" << radius;
    }
}

TEST(KDTree, RangeQueryVariableRadius) {
    // Verify that different radii at the same point give correct results
    auto photons = make_random_photons(2000);
    KDTree tree;
    tree.build(photons);

    float3 pos = make_f3(0, 0, 0);

    for (float r : {0.01f, 0.05f, 0.1f, 0.2f, 0.5f, 1.0f, 2.0f}) {
        auto brute = brute_force_range(pos, r, photons);
        std::set<uint32_t> kd_result;
        tree.query(pos, r, photons,
            [&](uint32_t idx, float) { kd_result.insert(idx); });
        EXPECT_EQ(kd_result, brute)
            << "Variable radius mismatch at r=" << r;
    }
}

TEST(KDTree, KNNEmpty) {
    KDTree tree;
    PhotonSoA empty;
    std::vector<uint32_t> indices;
    float max_dist2;
    tree.knn(make_f3(0, 0, 0), 10, empty, indices, max_dist2);
    EXPECT_TRUE(indices.empty());
}

TEST(KDTree, KNNSingle) {
    PhotonSoA photons;
    Photon p;
    p.position = make_f3(1.0f, 0.0f, 0.0f);
    p.wi = make_f3(0, 1, 0);
    p.geom_normal = make_f3(0, 1, 0);
    p.lambda_bin = 0;
    p.flux = 1.0f;
    photons.push_back(p);

    KDTree tree;
    tree.build(photons);

    std::vector<uint32_t> indices;
    float max_dist2;
    tree.knn(make_f3(0, 0, 0), 5, photons, indices, max_dist2);
    EXPECT_EQ(indices.size(), 1u);
    EXPECT_EQ(indices[0], 0u);
    EXPECT_NEAR(max_dist2, 1.0f, 1e-6f);
}

TEST(KDTree, KNNMatchesBruteForce) {
    auto photons = make_random_photons(3000);
    KDTree tree;
    tree.build(photons);

    PCGRng rng = PCGRng::seed(456, 1);

    for (int q = 0; q < 15; ++q) {
        float3 query_pos = make_f3(
            rng.next_float() * 2.0f - 1.0f,
            rng.next_float() * 2.0f - 1.0f,
            rng.next_float() * 2.0f - 1.0f
        );
        int k = 10 + (int)(rng.next_float() * 90);  // k in [10, 100]

        float bf_max_dist2;
        auto brute = brute_force_knn(query_pos, k, photons, bf_max_dist2);

        std::vector<uint32_t> kd_indices;
        float kd_max_dist2;
        tree.knn(query_pos, k, photons, kd_indices, kd_max_dist2);

        EXPECT_EQ(kd_indices.size(), brute.size())
            << "k-NN count mismatch at q=" << q;
        EXPECT_NEAR(kd_max_dist2, bf_max_dist2, 1e-5f)
            << "k-NN max distance mismatch at q=" << q;

        // Same set of indices (order may differ)
        std::set<uint32_t> kd_set(kd_indices.begin(), kd_indices.end());
        std::set<uint32_t> bf_set(brute.begin(), brute.end());
        EXPECT_EQ(kd_set, bf_set)
            << "k-NN index set mismatch at q=" << q;
    }
}

TEST(KDTree, KNNAdaptiveRadius) {
    // Verify k-NN gives radius = distance to k-th nearest
    auto photons = make_random_photons(1000);
    KDTree tree;
    tree.build(photons);

    float3 pos = make_f3(0, 0, 0);
    int k = 50;

    std::vector<uint32_t> indices;
    float max_dist2;
    tree.knn(pos, k, photons, indices, max_dist2);

    // The radius from k-NN should contain exactly k photons
    // (possibly more if some are equidistant, but at least k)
    float radius = sqrtf(max_dist2);
    auto brute = brute_force_range(pos, radius, photons);
    EXPECT_GE((int)brute.size(), k);
}

TEST(KDTree, BoundaryPhotons) {
    // Photons exactly on the query boundary
    PhotonSoA photons;
    Photon p;
    p.wi = make_f3(0, 1, 0);
    p.geom_normal = make_f3(0, 1, 0);
    p.lambda_bin = 0;
    p.flux = 1.0f;

    // Place photon exactly at radius distance
    p.position = make_f3(0.1f, 0.0f, 0.0f);
    photons.push_back(p);

    KDTree tree;
    tree.build(photons);

    // Query with radius = exactly the distance
    int count = 0;
    tree.query(make_f3(0, 0, 0), 0.1f, photons,
        [&](uint32_t, float dist2) {
            // dist2 = 0.01, r2 = 0.01 => should be included (<=)
            ++count;
        });
    EXPECT_EQ(count, 1);
}

TEST(KDTree, QueryMatchesHashGrid) {
    // KD-tree and hash grid should return the same photons for same query
    auto photons = make_random_photons(5000);
    float radius = 0.1f;

    KDTree tree;
    tree.build(photons);

    HashGrid grid;
    grid.build(photons, radius);

    PCGRng rng = PCGRng::seed(789, 1);

    for (int q = 0; q < 20; ++q) {
        float3 query_pos = make_f3(
            rng.next_float() * 1.6f - 0.8f,
            rng.next_float() * 1.6f - 0.8f,
            rng.next_float() * 1.6f - 0.8f
        );

        std::set<uint32_t> kd_result;
        tree.query(query_pos, radius, photons,
            [&](uint32_t idx, float) { kd_result.insert(idx); });

        std::set<uint32_t> hg_result;
        grid.query(query_pos, radius, photons,
            [&](uint32_t idx, float) { hg_result.insert(idx); });

        EXPECT_EQ(kd_result, hg_result)
            << "KD-tree vs HashGrid mismatch at q=" << q;
    }
}

TEST(KDTree, LargeDataset) {
    // Stress test with 100k photons
    auto photons = make_random_photons(100000);
    KDTree tree;
    tree.build(photons);

    // Just verify it builds without crashing and queries work
    int count = 0;
    tree.query(make_f3(0, 0, 0), 0.05f, photons,
        [&](uint32_t, float) { ++count; });
    EXPECT_GE(count, 0);  // may be 0 if no photons nearby

    std::vector<uint32_t> indices;
    float max_dist2;
    tree.knn(make_f3(0, 0, 0), 100, photons, indices, max_dist2);
    EXPECT_EQ((int)indices.size(), 100);
}

TEST(KDTree, AllPhotonsSamePosition) {
    // Degenerate case: all photons at the same point
    PhotonSoA photons;
    for (int i = 0; i < 100; ++i) {
        Photon p;
        p.position = make_f3(1.0f, 2.0f, 3.0f);
        p.wi = make_f3(0, 1, 0);
        p.geom_normal = make_f3(0, 1, 0);
        p.lambda_bin = 0;
        p.flux = 1.0f;
        photons.push_back(p);
    }

    KDTree tree;
    tree.build(photons);

    int count = 0;
    tree.query(make_f3(1.0f, 2.0f, 3.0f), 0.01f, photons,
        [&](uint32_t, float) { ++count; });
    EXPECT_EQ(count, 100);
}

TEST(KDTree, KNNMoreThanAvailable) {
    // Request more neighbors than exist
    auto photons = make_random_photons(10);
    KDTree tree;
    tree.build(photons);

    std::vector<uint32_t> indices;
    float max_dist2;
    tree.knn(make_f3(0, 0, 0), 50, photons, indices, max_dist2);
    EXPECT_EQ((int)indices.size(), 10);
}
