// ---------------------------------------------------------------------
// feature_speed_test.cpp - Unit tests for checklist-critical speed code
// ---------------------------------------------------------------------
#include <gtest/gtest.h>

#include <vector>

#include "core/types.h"
#include "core/cdf.h"
#include "core/nee_sampling.h"
#include "core/photon_density_cache.h"
#include "core/guided_nee.h"

// ---------------------------------------------------------------------
// CDF sampling (binary_search_cdf)
// ---------------------------------------------------------------------

TEST(Cdf, BinarySearchBasicBoundaries) {
    const float cdf[] = {0.25f, 0.5f, 0.75f, 1.0f};

    EXPECT_EQ(binary_search_cdf(cdf, 4, 0.0f), 0);
    EXPECT_EQ(binary_search_cdf(cdf, 4, 0.01f), 0);

    EXPECT_EQ(binary_search_cdf(cdf, 4, 0.24999f), 0);
    // Uses upper_bound semantics for exact boundary; u is continuous so equality is rare.
    EXPECT_EQ(binary_search_cdf(cdf, 4, 0.25f), 1);

    EXPECT_EQ(binary_search_cdf(cdf, 4, 0.49999f), 1);
    EXPECT_EQ(binary_search_cdf(cdf, 4, 0.5f), 2);

    EXPECT_EQ(binary_search_cdf(cdf, 4, 0.99999f), 3);
}

TEST(Cdf, BinarySearchDegenerateOrSmallN) {
    const float cdf1[] = {1.0f};
    EXPECT_EQ(binary_search_cdf(cdf1, 1, 0.0f), 0);
    EXPECT_EQ(binary_search_cdf(cdf1, 1, 0.999f), 0);

    const float* cdf0 = nullptr;
    EXPECT_EQ(binary_search_cdf(cdf0, 0, 0.5f), 0);
}

TEST(Cdf, BinarySearchWithPlateaus) {
    const float cdf[] = {0.2f, 0.2f, 0.2f, 1.0f};

    EXPECT_EQ(binary_search_cdf(cdf, 4, 0.0f), 0);
    EXPECT_EQ(binary_search_cdf(cdf, 4, 0.1999f), 0);
    // At the exact plateau value, upper_bound jumps past the plateau.
    EXPECT_EQ(binary_search_cdf(cdf, 4, 0.2f), 3);
    EXPECT_EQ(binary_search_cdf(cdf, 4, 0.9999f), 3);
}

// ---------------------------------------------------------------------
// NEE sample count policy (bounce-dependent shadow rays)
// ---------------------------------------------------------------------

TEST(NEE, ShadowSampleCountPolicy) {
    EXPECT_EQ(nee_shadow_sample_count(0, 4, 1), 4);
    EXPECT_EQ(nee_shadow_sample_count(1, 4, 1), 1);
    EXPECT_EQ(nee_shadow_sample_count(7, 4, 1), 1);

    // Clamp to at least 1
    EXPECT_EQ(nee_shadow_sample_count(0, 0, 0), 1);
    EXPECT_EQ(nee_shadow_sample_count(1, -10, -2), 1);
    EXPECT_EQ(nee_shadow_sample_count(0, -10, 5), 1);
    EXPECT_EQ(nee_shadow_sample_count(2, 5, -10), 1);
}

// ---------------------------------------------------------------------
// Photon density cache gating
// ---------------------------------------------------------------------

TEST(PhotonDensityCache, ReadWriteGating) {
    float dummy_cache[4] = {0, 0, 0, 0};

    EXPECT_FALSE(should_read_photon_density_cache(false, 0, dummy_cache, 1));
    EXPECT_FALSE(should_read_photon_density_cache(true,  1, dummy_cache, 1));
    EXPECT_FALSE(should_read_photon_density_cache(true,  0, nullptr,     1));
    EXPECT_FALSE(should_read_photon_density_cache(true,  0, dummy_cache, 0));
    EXPECT_TRUE (should_read_photon_density_cache(true,  0, dummy_cache, 1));

    EXPECT_FALSE(should_write_photon_density_cache(false, 0, dummy_cache, 0));
    EXPECT_FALSE(should_write_photon_density_cache(true,  1, dummy_cache, 0));
    EXPECT_FALSE(should_write_photon_density_cache(true,  0, nullptr,     0));
    EXPECT_FALSE(should_write_photon_density_cache(true,  0, dummy_cache, 1));
    EXPECT_TRUE (should_write_photon_density_cache(true,  0, dummy_cache, 0));
}

// ---------------------------------------------------------------------
// Photon directional bins / guided NEE helpers
// ---------------------------------------------------------------------

TEST(PhotonBins, StructSizeMatchesComment) {
    // Important for speed expectations; comment in core/photon_bins.h claims 24 bytes.
    static_assert(sizeof(PhotonBin) == 24, "PhotonBin layout changed; update memory-footprint assumptions");
    EXPECT_EQ(sizeof(PhotonBin), 24u);
}

TEST(PhotonBins, DirsInitAndNearest) {
    PhotonBinDirs dirs;
    dirs.init(32);
    EXPECT_EQ(dirs.count, 32);

    for (int k = 0; k < dirs.count; ++k) {
        float3 d = dirs.dirs[k];
        EXPECT_NEAR(length(d), 1.0f, 1e-4f);
        // Direction should be closest to itself.
        int nearest = dirs.find_nearest(d);
        EXPECT_EQ(nearest, k);
    }
}

TEST(PhotonBins, DirsInitClampsToMax) {
    PhotonBinDirs dirs;
    dirs.init(9999);
    EXPECT_EQ(dirs.count, MAX_PHOTON_BIN_COUNT);
}

TEST(GuidedNEE, FallbackConditions) {
    EXPECT_TRUE(guided_nee_should_fallback(0, 128, 1.0f));
    EXPECT_TRUE(guided_nee_should_fallback(129, 128, 1.0f));
    EXPECT_TRUE(guided_nee_should_fallback(1, 128, 0.0f));
    EXPECT_TRUE(guided_nee_should_fallback(1, 128, -1.0f));
    EXPECT_FALSE(guided_nee_should_fallback(1, 128, 0.001f));
}

TEST(GuidedNEE, BinBoostIsNormalizedAndHemisphereGated) {
    constexpr int N = 4;

    PhotonBinDirs dirs;
    dirs.init(N);

    PhotonBin bins[N] = {};
    bins[0].flux = 10.0f;
    bins[1].flux = 0.0f;
    bins[2].flux = 5.0f;
    bins[3].flux = 5.0f;

    float total = 0.0f;
    for (int i = 0; i < N; ++i) total += bins[i].flux;
    ASSERT_GT(total, 0.0f);

    // Positive hemisphere: normal aligned with wi.
    float3 wi = dirs.dirs[0];
    float3 n  = wi;
    float boost = guided_nee_bin_boost(wi, n, bins, N, dirs, total);
    EXPECT_NEAR(boost, bins[0].flux / total, 1e-6f);

    // Negative hemisphere: no boost.
    float3 n_back = -wi;
    float boost_back = guided_nee_bin_boost(wi, n_back, bins, N, dirs, total);
    EXPECT_NEAR(boost_back, 0.0f, 1e-6f);

    // Total flux zero => no boost.
    float boost_zero = guided_nee_bin_boost(wi, n, bins, N, dirs, 0.0f);
    EXPECT_NEAR(boost_zero, 0.0f, 1e-6f);
}

TEST(GuidedNEE, WeightFormulaMatchesSpec) {
    float p = 0.25f;
    float boost = 1.0f;
    float alpha = 5.0f;
    EXPECT_NEAR(guided_nee_weight(p, boost, alpha), 1.5f, 1e-6f);

    EXPECT_NEAR(guided_nee_weight(0.0f, 1.0f, 5.0f), 0.0f, 1e-6f);
    EXPECT_NEAR(guided_nee_weight(0.25f, 0.0f, 5.0f), 0.25f, 1e-6f);
}
