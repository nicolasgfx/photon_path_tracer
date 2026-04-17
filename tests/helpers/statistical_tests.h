#pragma once
// ─────────────────────────────────────────────────────────────────────
// tests/helpers/statistical_tests.h – Statistical test wrappers
//
// Re-exports src/testing statistical + convergence tests for GTest use.
// ─────────────────────────────────────────────────────────────────────
#include "testing/statistical_tests.h"
#include "testing/convergence_test.h"
#include <gtest/gtest.h>

namespace stat_helpers {

// Re-export for convenience
using statistical_tests::ChiSquaredResult;
using statistical_tests::FurnaceResult;
using statistical_tests::ReciprocityResult;
using statistical_tests::chi_squared_test;
using statistical_tests::furnace_test;
using statistical_tests::reciprocity_test;
using statistical_tests::ks_test_uniform;

using convergence_test::ConvergenceBaseline;
using convergence_test::ConvergenceResult;
using convergence_test::LinFitResult;
using convergence_test::log_log_fit;
using convergence_test::compare_to_baseline;
using convergence_test::save_baseline;
using convergence_test::load_baseline;
using convergence_test::create_baseline;

// GTest assertion: chi-squared passes
#define ASSERT_CHI2_PASS(result) \
    ASSERT_TRUE((result).passed) \
        << "chi2=" << (result).chi2 << " p=" << (result).p_value

// GTest assertion: furnace test passes
#define ASSERT_FURNACE_PASS(result) \
    ASSERT_TRUE((result).passed) \
        << "mean_lum=" << (result).mean_luminance \
        << " expected=" << (result).expected

} // namespace stat_helpers
