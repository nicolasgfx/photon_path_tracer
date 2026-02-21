// ─────────────────────────────────────────────────────────────────────
// test_tangential_gather.cpp – CPU vs GPU gather comparison (§6.3, §6.5)
// ─────────────────────────────────────────────────────────────────────
// Tests cover:
//   1. KD-tree tangential range query correctness
//   2. KD-tree tangential k-NN correctness
//   3. Hash grid tangential range query correctness
//   4. Hash grid shell expansion k-NN correctness
//   5. CPU (KD-tree) vs GPU (hash grid) cross-validation
//   6. Tangential vs 3D distance: tangential ≤ 3D
//   7. Synthetic photon distributions:
//      a) Flat wall (single plane)
//      b) Corner (two perpendicular planes)
//      c) Random 3D cloud
//
// Each critical algorithm has TWO implementations compared:
//   a) KD-tree (CPU ground truth, arbitrary radius)
//   b) Hash grid (GPU primary, O(1) build / shell expansion)
// ─────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include "photon/kd_tree.h"
#include "photon/hash_grid.h"
#include "photon/surface_filter.h"
#include "core/random.h"

#include <set>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>

// ── Helpers ─────────────────────────────────────────────────────────

// Create photons on a flat wall (Y = 0 plane, normal = +Y)
static PhotonSoA make_wall_photons(int n, uint32_t seed = 42) {
    PhotonSoA photons;
    photons.reserve(n);
    PCGRng rng = PCGRng::seed(seed, 1);
    for (int i = 0; i < n; ++i) {
        Photon p;
        p.position = make_f3(
            rng.next_float() * 2.0f - 1.0f,
            rng.next_float() * 0.01f,  // tiny Y variation (near plane)
            rng.next_float() * 2.0f - 1.0f
        );
        p.wi = make_f3(0, 1, 0);             // incoming from +Y
        p.geom_normal = make_f3(0, 1, 0);    // upward normal
        p.lambda_bin[0] = 0;
        p.flux[0] = 1.0f;
        photons.push_back(p);
    }
    return photons;
}

// Create photons in a corner (two perpendicular walls meeting at Y=0, Z=0)
static PhotonSoA make_corner_photons(int n, uint32_t seed = 42) {
    PhotonSoA photons;
    photons.reserve(n);
    PCGRng rng = PCGRng::seed(seed, 1);
    for (int i = 0; i < n; ++i) {
        Photon p;
        if (i % 2 == 0) {
            // Floor: Y ≈ 0, normal = +Y
            p.position = make_f3(
                rng.next_float() * 2.0f - 1.0f,
                rng.next_float() * 0.005f,
                rng.next_float() * 0.5f + 0.1f  // Z > 0 (away from corner)
            );
            p.geom_normal = make_f3(0, 1, 0);
        } else {
            // Wall: Z ≈ 0, normal = +Z
            p.position = make_f3(
                rng.next_float() * 2.0f - 1.0f,
                rng.next_float() * 0.5f + 0.1f,  // Y > 0 (above floor)
                rng.next_float() * 0.005f
            );
            p.geom_normal = make_f3(0, 0, 1);
        }
        p.wi = normalize(make_f3(0.1f, 0.5f, 0.5f));
        p.lambda_bin[0] = 0;
        p.flux[0] = 1.0f;
        photons.push_back(p);
    }
    return photons;
}

// Create uniformly distributed 3D photons
static PhotonSoA make_random_3d_photons(int n, uint32_t seed = 42) {
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
        p.lambda_bin[0] = 0;
        p.flux[0] = 1.0f;
        photons.push_back(p);
    }
    return photons;
}

// Brute-force tangential range query (reference implementation)
static std::set<uint32_t> brute_force_tangential_range(
    float3 pos, float3 normal, float radius, float tau,
    const PhotonSoA& photons)
{
    float r2 = radius * radius;
    std::set<uint32_t> result;
    for (size_t i = 0; i < photons.size(); ++i) {
        float3 ppos = make_f3(photons.pos_x[i], photons.pos_y[i], photons.pos_z[i]);
        TangentialResult tr = compute_tangential(pos, normal, ppos);
        if (fabsf(tr.d_plane) <= tau && tr.d_tan2 <= r2) {
            result.insert((uint32_t)i);
        }
    }
    return result;
}

// Brute-force tangential k-NN (reference implementation)
static void brute_force_tangential_knn(
    float3 pos, float3 normal, int k, float tau,
    const PhotonSoA& photons,
    std::vector<uint32_t>& out_indices, float& out_max_dist2)
{
    std::vector<std::pair<float, uint32_t>> candidates;
    for (size_t i = 0; i < photons.size(); ++i) {
        float3 ppos = make_f3(photons.pos_x[i], photons.pos_y[i], photons.pos_z[i]);
        TangentialResult tr = compute_tangential(pos, normal, ppos);
        if (fabsf(tr.d_plane) <= tau) {
            candidates.push_back({tr.d_tan2, (uint32_t)i});
        }
    }
    std::sort(candidates.begin(), candidates.end());

    out_indices.clear();
    int count = std::min(k, (int)candidates.size());
    for (int i = 0; i < count; ++i) {
        out_indices.push_back(candidates[i].second);
    }
    out_max_dist2 = (count > 0) ? candidates[count - 1].first : 0.0f;
}

// =====================================================================
// 1. KD-tree tangential range query
// =====================================================================

TEST(TangentialGather, KDTreeRangeQueryWall) {
    // Photons on a flat wall — tangential query should find them
    auto photons = make_wall_photons(500);
    KDTree tree;
    tree.build(photons);

    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = make_f3(0, 1, 0);
    float radius  = 0.3f;
    float tau     = 0.02f;

    // KD-tree tangential query
    std::set<uint32_t> kd_results;
    tree.query_tangential(qpos, qnorm, radius, tau, photons,
        [&](uint32_t idx, float) { kd_results.insert(idx); });

    // Brute-force reference
    auto brute_results = brute_force_tangential_range(
        qpos, qnorm, radius, tau, photons);

    EXPECT_EQ(kd_results, brute_results)
        << "KD-tree tangential range query must match brute force";
    EXPECT_GT(kd_results.size(), 0u) << "Should find some photons on the wall";
}

TEST(TangentialGather, KDTreeRangeQueryCorner) {
    // Corner scenario: query on floor should NOT pick up wall photons
    auto photons = make_corner_photons(1000);
    KDTree tree;
    tree.build(photons);

    // Query on the floor near the corner
    float3 qpos  = make_f3(0, 0, 0.05f);
    float3 qnorm = make_f3(0, 1, 0);  // floor normal
    float radius  = 0.5f;
    float tau     = 0.01f;  // tight tau rejects wall photons

    std::set<uint32_t> kd_results;
    tree.query_tangential(qpos, qnorm, radius, tau, photons,
        [&](uint32_t idx, float) { kd_results.insert(idx); });

    auto brute_results = brute_force_tangential_range(
        qpos, qnorm, radius, tau, photons);

    EXPECT_EQ(kd_results, brute_results);

    // Verify: no wall photons (Z ≈ 0) should appear in floor query
    for (uint32_t idx : kd_results) {
        // Floor photons have Y ≈ 0
        EXPECT_LT(fabsf(photons.pos_y[idx]), 0.02f)
            << "Only floor photons should be returned for floor query";
    }
}

// =====================================================================
// 2. KD-tree tangential k-NN
// =====================================================================

TEST(TangentialGather, KDTreeKNNWall) {
    auto photons = make_wall_photons(500);
    KDTree tree;
    tree.build(photons);

    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = make_f3(0, 1, 0);
    int k = 20;
    float tau = 0.02f;

    std::vector<uint32_t> kd_indices;
    float kd_max_dist2;
    tree.knn_tangential(qpos, qnorm, k, tau, photons,
                        kd_indices, kd_max_dist2);

    std::vector<uint32_t> brute_indices;
    float brute_max_dist2;
    brute_force_tangential_knn(qpos, qnorm, k, tau, photons,
                               brute_indices, brute_max_dist2);

    EXPECT_EQ(kd_indices.size(), brute_indices.size());
    EXPECT_NEAR(kd_max_dist2, brute_max_dist2, 1e-4f);

    // Same set of indices
    std::set<uint32_t> kd_set(kd_indices.begin(), kd_indices.end());
    std::set<uint32_t> brute_set(brute_indices.begin(), brute_indices.end());
    EXPECT_EQ(kd_set, brute_set)
        << "KD-tree tangential k-NN must return same photons as brute force";
}

// =====================================================================
// 3. Hash grid tangential range query
// =====================================================================

TEST(TangentialGather, HashGridRangeQueryWall) {
    auto photons = make_wall_photons(500);
    HashGrid grid;
    float radius = 0.3f;
    grid.build(photons, radius);

    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = make_f3(0, 1, 0);
    float tau     = 0.02f;

    std::set<uint32_t> grid_results;
    grid.query_tangential(qpos, qnorm, radius, tau, photons,
        [&](uint32_t idx, float) { grid_results.insert(idx); });

    auto brute_results = brute_force_tangential_range(
        qpos, qnorm, radius, tau, photons);

    EXPECT_EQ(grid_results, brute_results)
        << "Hash grid tangential range query must match brute force";
}

TEST(TangentialGather, HashGridRangeQueryCorner) {
    auto photons = make_corner_photons(1000);
    HashGrid grid;
    float radius = 0.5f;
    grid.build(photons, radius);

    float3 qpos  = make_f3(0, 0, 0.05f);
    float3 qnorm = make_f3(0, 1, 0);
    float tau     = 0.01f;

    std::set<uint32_t> grid_results;
    grid.query_tangential(qpos, qnorm, radius, tau, photons,
        [&](uint32_t idx, float) { grid_results.insert(idx); });

    auto brute_results = brute_force_tangential_range(
        qpos, qnorm, radius, tau, photons);

    EXPECT_EQ(grid_results, brute_results);
}

// =====================================================================
// 4. Hash grid shell expansion k-NN
// =====================================================================

TEST(TangentialGather, ShellExpansionKNNWall) {
    auto photons = make_wall_photons(500);
    HashGrid grid;
    // Use larger cell size so shell expansion covers the search domain
    grid.build(photons, 0.3f);  // cell_size = 0.6

    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = make_f3(0, 1, 0);
    int k = 20;
    float tau = 0.02f;

    std::vector<uint32_t> shell_indices;
    float shell_max_dist2;
    // Use enough layers to cover the domain
    grid.knn_shell_expansion(qpos, qnorm, k, tau, photons,
                             shell_indices, shell_max_dist2, 8);

    std::vector<uint32_t> brute_indices;
    float brute_max_dist2;
    brute_force_tangential_knn(qpos, qnorm, k, tau, photons,
                               brute_indices, brute_max_dist2);

    EXPECT_EQ((int)shell_indices.size(), (int)brute_indices.size());

    // The k-th distance should match closely
    if (!shell_indices.empty() && !brute_indices.empty()) {
        EXPECT_NEAR(shell_max_dist2, brute_max_dist2, 1e-4f);
    }

    // Same set of indices
    std::set<uint32_t> shell_set(shell_indices.begin(), shell_indices.end());
    std::set<uint32_t> brute_set(brute_indices.begin(), brute_indices.end());
    EXPECT_EQ(shell_set, brute_set)
        << "Shell expansion k-NN must return same photons as brute force";
}

TEST(TangentialGather, ShellExpansionKNNRandom3D) {
    auto photons = make_random_3d_photons(500);
    HashGrid grid;
    // Use larger cell size and more layers for random 3D distribution
    grid.build(photons, 0.5f);  // cell_size = 1.0

    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = make_f3(0, 1, 0);
    int k = 10;
    float tau = 1.0f;  // generous tau

    std::vector<uint32_t> shell_indices;
    float shell_max_dist2;
    grid.knn_shell_expansion(qpos, qnorm, k, tau, photons,
                             shell_indices, shell_max_dist2, 8);

    std::vector<uint32_t> brute_indices;
    float brute_max_dist2;
    brute_force_tangential_knn(qpos, qnorm, k, tau, photons,
                               brute_indices, brute_max_dist2);

    EXPECT_EQ((int)shell_indices.size(), (int)brute_indices.size());

    std::set<uint32_t> shell_set(shell_indices.begin(), shell_indices.end());
    std::set<uint32_t> brute_set(brute_indices.begin(), brute_indices.end());
    EXPECT_EQ(shell_set, brute_set);
}

// =====================================================================
// 5. CPU (KD-tree) vs GPU (hash grid) cross-validation
// =====================================================================

TEST(TangentialGather, CPUvsGPU_RangeQueryMatchWall) {
    // CRITICAL: the two implementations MUST return identical results
    auto photons = make_wall_photons(500);

    KDTree tree;
    tree.build(photons);

    HashGrid grid;
    float radius = 0.3f;
    grid.build(photons, radius);

    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = make_f3(0, 1, 0);
    float tau     = 0.02f;

    std::set<uint32_t> kd_results;
    tree.query_tangential(qpos, qnorm, radius, tau, photons,
        [&](uint32_t idx, float) { kd_results.insert(idx); });

    std::set<uint32_t> grid_results;
    grid.query_tangential(qpos, qnorm, radius, tau, photons,
        [&](uint32_t idx, float) { grid_results.insert(idx); });

    EXPECT_EQ(kd_results, grid_results)
        << "KD-tree and hash grid MUST return identical tangential results";
}

TEST(TangentialGather, CPUvsGPU_RangeQueryMatchCorner) {
    auto photons = make_corner_photons(1000);

    KDTree tree;
    tree.build(photons);

    HashGrid grid;
    float radius = 0.5f;
    grid.build(photons, radius);

    float3 qpos  = make_f3(0, 0, 0.05f);
    float3 qnorm = make_f3(0, 1, 0);
    float tau     = 0.01f;

    std::set<uint32_t> kd_results;
    tree.query_tangential(qpos, qnorm, radius, tau, photons,
        [&](uint32_t idx, float) { kd_results.insert(idx); });

    std::set<uint32_t> grid_results;
    grid.query_tangential(qpos, qnorm, radius, tau, photons,
        [&](uint32_t idx, float) { grid_results.insert(idx); });

    EXPECT_EQ(kd_results, grid_results)
        << "CPU vs GPU tangential range query must match at corner";
}

TEST(TangentialGather, CPUvsGPU_RangeQueryMatchRandom) {
    auto photons = make_random_3d_photons(500);

    KDTree tree;
    tree.build(photons);

    HashGrid grid;
    float radius = 0.3f;
    grid.build(photons, radius);

    // Multiple query points
    PCGRng rng = PCGRng::seed(999, 1);
    for (int q = 0; q < 10; ++q) {
        float3 qpos = make_f3(
            rng.next_float() * 1.0f - 0.5f,
            rng.next_float() * 1.0f - 0.5f,
            rng.next_float() * 1.0f - 0.5f
        );
        float3 qnorm = normalize(make_f3(
            rng.next_float() * 2.0f - 1.0f,
            rng.next_float() * 2.0f - 1.0f,
            rng.next_float() * 2.0f - 1.0f
        ));
        float tau = 0.1f;

        std::set<uint32_t> kd_results;
        tree.query_tangential(qpos, qnorm, radius, tau, photons,
            [&](uint32_t idx, float) { kd_results.insert(idx); });

        std::set<uint32_t> grid_results;
        grid.query_tangential(qpos, qnorm, radius, tau, photons,
            [&](uint32_t idx, float) { grid_results.insert(idx); });

        EXPECT_EQ(kd_results, grid_results)
            << "CPU vs GPU mismatch at random query point " << q;
    }
}

TEST(TangentialGather, CPUvsGPU_KNNMatchWall) {
    // KD-tree k-NN vs shell expansion k-NN on wall photons
    auto photons = make_wall_photons(500);

    KDTree tree;
    tree.build(photons);

    HashGrid grid;
    grid.build(photons, 0.1f);

    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = make_f3(0, 1, 0);
    int k = 15;
    float tau = 0.02f;

    std::vector<uint32_t> kd_indices;
    float kd_max_dist2;
    tree.knn_tangential(qpos, qnorm, k, tau, photons,
                        kd_indices, kd_max_dist2);

    std::vector<uint32_t> shell_indices;
    float shell_max_dist2;
    grid.knn_shell_expansion(qpos, qnorm, k, tau, photons,
                             shell_indices, shell_max_dist2);

    EXPECT_EQ(kd_indices.size(), shell_indices.size());
    EXPECT_NEAR(kd_max_dist2, shell_max_dist2, 1e-4f);

    std::set<uint32_t> kd_set(kd_indices.begin(), kd_indices.end());
    std::set<uint32_t> shell_set(shell_indices.begin(), shell_indices.end());
    EXPECT_EQ(kd_set, shell_set)
        << "KD-tree k-NN and shell expansion k-NN must return same photons";
}

// =====================================================================
// 6. Tangential vs 3D: tangential catches coplanar photons missed by 3D
// =====================================================================

TEST(TangentialGather, TangentialFindsCoplanarNotMissed) {
    // Scenario: query with radius r.  Photon is at tangential distance
    // < r but 3D distance > r (because it has large plane distance).
    // With tangential metric, it should NOT be found (plane distance
    // filter rejects it).
    //
    // Conversely: photon with small 3D distance on the wrong side of
    // a thin wall would be found by 3D but rejected by tangential.

    PhotonSoA photons;
    Photon p;

    // Photon A: on same plane, tangential dist = 0.1
    p.position = make_f3(0.1f, 0.001f, 0);
    p.wi = make_f3(0, 1, 0);
    p.geom_normal = make_f3(0, 1, 0);
    p.flux[0] = 1.0f;
    photons.push_back(p);

    // Photon B: close in 3D but on opposite side of wall (Y = -0.05)
    p.position = make_f3(0.01f, -0.05f, 0);
    p.wi = make_f3(0, 1, 0);
    p.geom_normal = make_f3(0, -1, 0);  // opposite normal
    p.flux[0] = 1.0f;
    photons.push_back(p);

    KDTree tree;
    tree.build(photons);

    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = make_f3(0, 1, 0);
    float radius  = 0.2f;
    float tau     = 0.01f;

    // Tangential query: should find A, reject B (plane distance fails)
    std::set<uint32_t> tan_results;
    tree.query_tangential(qpos, qnorm, radius, tau, photons,
        [&](uint32_t idx, float) { tan_results.insert(idx); });

    EXPECT_TRUE(tan_results.count(0) > 0) << "Should find coplanar photon A";
    EXPECT_TRUE(tan_results.count(1) == 0) << "Should reject through-wall photon B";

    // 3D query: would find BOTH (they're both within 3D radius)
    std::set<uint32_t> eucl_results;
    tree.query(qpos, radius, photons,
        [&](uint32_t idx, float) { eucl_results.insert(idx); });

    EXPECT_TRUE(eucl_results.count(0) > 0);
    EXPECT_TRUE(eucl_results.count(1) > 0) << "3D query finds through-wall photon";
}

// =====================================================================
// 7. Legacy 3D queries still work (backward compatibility)
// =====================================================================

TEST(TangentialGather, Legacy3DQueryStillWorks) {
    auto photons = make_random_3d_photons(200);
    KDTree tree;
    tree.build(photons);

    float3 qpos = make_f3(0, 0, 0);
    float radius = 0.5f;

    std::set<uint32_t> results;
    tree.query(qpos, radius, photons,
        [&](uint32_t idx, float) { results.insert(idx); });

    // Verify against brute force 3D
    float r2 = radius * radius;
    std::set<uint32_t> brute;
    for (size_t i = 0; i < photons.size(); ++i) {
        float3 p = make_f3(photons.pos_x[i], photons.pos_y[i], photons.pos_z[i]);
        float3 d = p - qpos;
        if (dot(d, d) <= r2) brute.insert((uint32_t)i);
    }

    EXPECT_EQ(results, brute);
}

TEST(TangentialGather, Legacy3DKNNStillWorks) {
    auto photons = make_random_3d_photons(200);
    KDTree tree;
    tree.build(photons);

    float3 qpos = make_f3(0, 0, 0);
    int k = 10;

    std::vector<uint32_t> indices;
    float max_dist2;
    tree.knn(qpos, k, photons, indices, max_dist2);

    EXPECT_EQ((int)indices.size(), k);

    // All returned photons should be within max_dist2
    for (uint32_t idx : indices) {
        float3 p = make_f3(photons.pos_x[idx], photons.pos_y[idx], photons.pos_z[idx]);
        float3 d = p - qpos;
        EXPECT_LE(dot(d, d), max_dist2 + 1e-5f);
    }
}

// =====================================================================
// 8. Tangential distance values returned by queries are correct
// =====================================================================

TEST(TangentialGather, ReturnedDistancesAreCorrect) {
    auto photons = make_wall_photons(200);

    KDTree tree;
    tree.build(photons);

    HashGrid grid;
    float radius = 0.3f;
    grid.build(photons, radius);

    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = make_f3(0, 1, 0);
    float tau     = 0.02f;

    // KD-tree: verify returned d_tan2 matches independent computation
    tree.query_tangential(qpos, qnorm, radius, tau, photons,
        [&](uint32_t idx, float d_tan2) {
            float3 ppos = make_f3(photons.pos_x[idx],
                                  photons.pos_y[idx],
                                  photons.pos_z[idx]);
            float expected = tangential_distance2(qpos, qnorm, ppos);
            EXPECT_NEAR(d_tan2, expected, 1e-5f)
                << "KD-tree returned wrong tangential distance for photon " << idx;
        });

    // Hash grid: same verification
    grid.query_tangential(qpos, qnorm, radius, tau, photons,
        [&](uint32_t idx, float d_tan2) {
            float3 ppos = make_f3(photons.pos_x[idx],
                                  photons.pos_y[idx],
                                  photons.pos_z[idx]);
            float expected = tangential_distance2(qpos, qnorm, ppos);
            EXPECT_NEAR(d_tan2, expected, 1e-5f)
                << "Hash grid returned wrong tangential distance for photon " << idx;
        });
}

// =====================================================================
// 9. Empty / edge cases
// =====================================================================

TEST(TangentialGather, EmptyPhotonMap) {
    PhotonSoA empty;
    KDTree tree;
    tree.build(empty);

    HashGrid grid;
    grid.build(empty, 0.1f);

    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = make_f3(0, 1, 0);

    int kd_count = 0;
    tree.query_tangential(qpos, qnorm, 1.0f, 1.0f, empty,
        [&](uint32_t, float) { ++kd_count; });
    EXPECT_EQ(kd_count, 0);

    int grid_count = 0;
    grid.query_tangential(qpos, qnorm, 1.0f, 1.0f, empty,
        [&](uint32_t, float) { ++grid_count; });
    EXPECT_EQ(grid_count, 0);

    std::vector<uint32_t> indices;
    float md2;
    tree.knn_tangential(qpos, qnorm, 5, 1.0f, empty, indices, md2);
    EXPECT_EQ(indices.size(), 0u);

    grid.knn_shell_expansion(qpos, qnorm, 5, 1.0f, empty, indices, md2);
    EXPECT_EQ(indices.size(), 0u);
}

TEST(TangentialGather, SinglePhoton) {
    PhotonSoA photons;
    Photon p;
    p.position = make_f3(0.05f, 0.001f, 0);
    p.wi = make_f3(0, 1, 0);
    p.geom_normal = make_f3(0, 1, 0);
    p.flux[0] = 1.0f;
    photons.push_back(p);

    KDTree tree;
    tree.build(photons);

    HashGrid grid;
    grid.build(photons, 0.1f);

    float3 qpos  = make_f3(0, 0, 0);
    float3 qnorm = make_f3(0, 1, 0);

    // Range query
    int kd_count = 0;
    tree.query_tangential(qpos, qnorm, 0.1f, 0.01f, photons,
        [&](uint32_t, float) { ++kd_count; });
    EXPECT_EQ(kd_count, 1);

    int grid_count = 0;
    grid.query_tangential(qpos, qnorm, 0.1f, 0.01f, photons,
        [&](uint32_t, float) { ++grid_count; });
    EXPECT_EQ(grid_count, 1);

    // k-NN
    std::vector<uint32_t> indices;
    float md2;
    tree.knn_tangential(qpos, qnorm, 5, 0.01f, photons, indices, md2);
    EXPECT_EQ(indices.size(), 1u);

    grid.knn_shell_expansion(qpos, qnorm, 5, 0.01f, photons, indices, md2);
    EXPECT_EQ(indices.size(), 1u);
}

// =====================================================================
// 10. Stress test: many query points, CPU vs GPU must always agree
// =====================================================================

TEST(TangentialGather, StressCPUvsGPU_ManyQueries) {
    auto photons = make_wall_photons(1000, 7777);

    KDTree tree;
    tree.build(photons);

    float radius = 0.15f;
    HashGrid grid;
    grid.build(photons, radius);

    float tau = 0.02f;
    PCGRng rng = PCGRng::seed(314159, 1);

    int mismatches = 0;
    int total_queries = 50;

    for (int q = 0; q < total_queries; ++q) {
        float3 qpos = make_f3(
            rng.next_float() * 1.0f - 0.5f,
            rng.next_float() * 0.01f,
            rng.next_float() * 1.0f - 0.5f
        );
        float3 qnorm = make_f3(0, 1, 0);

        std::set<uint32_t> kd_set;
        tree.query_tangential(qpos, qnorm, radius, tau, photons,
            [&](uint32_t idx, float) { kd_set.insert(idx); });

        std::set<uint32_t> grid_set;
        grid.query_tangential(qpos, qnorm, radius, tau, photons,
            [&](uint32_t idx, float) { grid_set.insert(idx); });

        if (kd_set != grid_set) ++mismatches;
    }

    EXPECT_EQ(mismatches, 0)
        << "CPU and GPU tangential queries must ALWAYS agree";
}
