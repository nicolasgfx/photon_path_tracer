// ---------------------------------------------------------------------
// feature_speed_test.cpp - Unit tests for checklist-critical speed code
// ---------------------------------------------------------------------
#include <gtest/gtest.h>

#include <vector>

#include "core/types.h"
#include "core/random.h"
#include "renderer/nee_shared.h"
#include "photon/photon_bins.h"

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
// Photon directional bins / guided NEE helpers
// ---------------------------------------------------------------------

TEST(PhotonBins, StructSizeMatchesComment) {
    // Per-wavelength flux[NUM_LAMBDA] + scalar_flux + direction + weight + count + avg normal.
    // (NUM_LAMBDA + 8) * 4 + 4 bytes.
    constexpr size_t expected_size = (NUM_LAMBDA + 8) * 4 + 4;
    static_assert(sizeof(PhotonBin) == expected_size, "PhotonBin layout changed; update memory-footprint assumptions");
    EXPECT_EQ(sizeof(PhotonBin), expected_size);
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


