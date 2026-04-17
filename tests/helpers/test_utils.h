#pragma once
// ─────────────────────────────────────────────────────────────────────
// tests/helpers/test_utils.h – Test utilities for GTest tests
//
// Wraps src/testing headers + adds GTest-specific convenience macros.
// ─────────────────────────────────────────────────────────────────────
#include "testing/test_framework.h"
#include <gtest/gtest.h>

namespace test_helpers {

// Re-export utilities
using test_utils::BufferStats;
using test_utils::compute_buffer_stats;
using test_utils::compute_rmse;
using test_utils::mean_luminance;
using test_utils::buffer_is_clean;
using test_utils::buffer_has_nonzero;
using test_utils::normalize_by_spp;

// GTest assertion for float tolerance
#define EXPECT_NEAR_F(a, b, tol) \
    EXPECT_NEAR(static_cast<double>(a), static_cast<double>(b), static_cast<double>(tol))

// Assert buffer has no NaN or Inf
#define ASSERT_BUFFER_CLEAN(data, count) \
    ASSERT_TRUE(test_utils::buffer_is_clean(data, count)) << "Buffer contains NaN/Inf"

// Assert buffer has non-zero values
#define ASSERT_BUFFER_NONZERO(data, count) \
    ASSERT_TRUE(test_utils::buffer_has_nonzero(data, count)) << "Buffer is all zero"

} // namespace test_helpers
