// ─────────────────────────────────────────────────────────────────────
// tests/diagnostics/test_bottleneck_id.cpp
//
// Meta-skill A: Bottleneck identification from stage metrics.
// Tests that the BottleneckAnalyzer correctly identifies noise sources.
// ─────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include "diagnose/bottleneck_report.h"
#include "diagnose/convergence_analyzer.h"
#include "core/stage_metrics.h"

// Helper: build a FrameMetrics with customizable variance breakdown
static FrameMetrics make_frame(
    float direct_var, float indirect_var, float photon_var,
    float direct_ms = 5.f, float indirect_ms = 10.f, float photon_ms = 3.f)
{
    FrameMetrics fm;
    fm.frame_number = 1;
    fm.spp = 64;

    StageMetrics sm_direct;
    sm_direct.stage_name = "direct-lighting";
    sm_direct.time_ms = direct_ms;
    sm_direct.variance_contribution = direct_var;
    fm.add(sm_direct);

    StageMetrics sm_indirect;
    sm_indirect.stage_name = "path-integrator";
    sm_indirect.time_ms = indirect_ms;
    sm_indirect.variance_contribution = indirect_var;
    fm.add(sm_indirect);

    StageMetrics sm_photon;
    sm_photon.stage_name = "photon-system";
    sm_photon.time_ms = photon_ms;
    sm_photon.variance_contribution = photon_var;
    fm.add(sm_photon);

    return fm;
}

TEST(BottleneckId, DirectLightingDominant) {
    FrameMetrics fm = make_frame(0.8f, 0.15f, 0.05f);

    VarianceStats vs{};
    vs.mean_variance = 0.05f;

    ConvergenceAnalysis ca{};
    ca.rate_alpha = 0.9f;
    ca.converging = true;
    ca.normal_rate = true;

    BottleneckAnalyzer analyzer;
    BottleneckReport report = analyzer.analyze(fm, vs, ca);

    EXPECT_GT(report.direct_variance_fraction, 0.5f);
    EXPECT_EQ(report.bottleneck_stage, "direct-lighting");
}

TEST(BottleneckId, IndirectDominant) {
    FrameMetrics fm = make_frame(0.10f, 0.85f, 0.05f);

    VarianceStats vs{};
    vs.mean_variance = 0.08f;

    ConvergenceAnalysis ca{};
    ca.rate_alpha = 0.6f;
    ca.converging = true;

    BottleneckAnalyzer analyzer;
    BottleneckReport report = analyzer.analyze(fm, vs, ca);

    EXPECT_GT(report.indirect_variance_fraction, 0.5f);
    EXPECT_EQ(report.bottleneck_stage, "path-integrator");
}

TEST(BottleneckId, BalancedNoBottleneck) {
    FrameMetrics fm = make_frame(0.33f, 0.34f, 0.33f);

    VarianceStats vs{};
    vs.mean_variance = 0.02f;

    ConvergenceAnalysis ca{};
    ca.rate_alpha = 1.0f;
    ca.converging = true;
    ca.normal_rate = true;

    BottleneckAnalyzer analyzer;
    BottleneckReport report = analyzer.analyze(fm, vs, ca);

    // No single stage dominates
    EXPECT_LT(report.direct_variance_fraction, 0.5f);
    EXPECT_LT(report.indirect_variance_fraction, 0.5f);
}

TEST(BottleneckId, SlowestStageIdentified) {
    FrameMetrics fm = make_frame(0.3f, 0.5f, 0.2f, 5.f, 20.f, 3.f);

    VarianceStats vs{};
    ConvergenceAnalysis ca{};
    ca.rate_alpha = 0.9f;

    BottleneckAnalyzer analyzer;
    BottleneckReport report = analyzer.analyze(fm, vs, ca);

    EXPECT_EQ(report.slowest_stage, "path-integrator");
    EXPECT_NEAR(report.slowest_stage_ms, 20.f, 1e-5f);
}

TEST(BottleneckId, RecommendationsForDirect) {
    FrameMetrics fm = make_frame(0.8f, 0.15f, 0.05f);
    VarianceStats vs{};
    vs.mean_variance = 0.1f;
    ConvergenceAnalysis ca{};
    ca.rate_alpha = 0.9f;
    ca.converging = true;

    BottleneckAnalyzer analyzer;
    BottleneckReport report = analyzer.analyze(fm, vs, ca);

    // Should suggest improving direct lighting
    EXPECT_FALSE(report.suggestions.empty());
    bool has_direct_suggestion = false;
    for (const auto& s : report.suggestions) {
        if (s.target_stage == "direct-lighting") has_direct_suggestion = true;
    }
    EXPECT_TRUE(has_direct_suggestion);
}
