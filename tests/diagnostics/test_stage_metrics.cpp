// ─────────────────────────────────────────────────────────────────────
// tests/diagnostics/test_stage_metrics.cpp
//
// Meta-skill A: StageMetrics collection and FrameMetrics aggregation.
// ─────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include "core/stage_metrics.h"
#include <string>

TEST(StageMetrics, Construction) {
    StageMetrics sm;
    sm.stage_name = "test-stage";
    sm.time_ms = 5.0f;
    sm.mean_contribution = 0.5f;
    sm.variance_contribution = 0.1f;
    sm.num_samples = 1000;

    EXPECT_STREQ(sm.stage_name, "test-stage");
    EXPECT_FLOAT_EQ(sm.time_ms, 5.0f);
    EXPECT_EQ(sm.num_samples, 1000);
}

TEST(StageMetrics, CustomKeyValue) {
    StageMetrics sm;
    sm.set("photon_count", 50000.0f);
    sm.set("guide_pdf_mean", 0.3f);

    EXPECT_FLOAT_EQ(sm.get("photon_count"), 50000.0f);
    EXPECT_FLOAT_EQ(sm.get("guide_pdf_mean"), 0.3f);
    EXPECT_FLOAT_EQ(sm.get("missing_key", 42.f), 42.f);
}

TEST(StageMetrics, CustomKeyOverwrite) {
    StageMetrics sm;
    sm.set("rate", 1.0f);
    sm.set("rate", 2.0f);

    EXPECT_FLOAT_EQ(sm.get("rate"), 2.0f);
}

TEST(StageMetrics, FrameMetricsAggregation) {
    FrameMetrics fm;
    fm.frame_number = 1;
    fm.spp = 64;

    StageMetrics s1;
    s1.stage_name = "scene";
    s1.time_ms = 2.0f;
    fm.add(s1);

    StageMetrics s2;
    s2.stage_name = "render";
    s2.time_ms = 15.0f;
    fm.add(s2);

    EXPECT_EQ((int)fm.stages.size(), 2);
    EXPECT_NEAR(fm.total_time_ms, 17.0f, 1e-5f);
}

TEST(StageMetrics, FrameMetricsFind) {
    FrameMetrics fm;

    StageMetrics s1;
    s1.stage_name = "lighting";
    s1.time_ms = 8.0f;
    fm.add(s1);

    const StageMetrics* found = fm.find("lighting");
    ASSERT_NE(found, nullptr);
    EXPECT_FLOAT_EQ(found->time_ms, 8.0f);

    EXPECT_EQ(fm.find("nonexistent"), nullptr);
}

TEST(StageMetrics, FrameMetricsEmpty) {
    FrameMetrics fm;
    EXPECT_EQ((int)fm.stages.size(), 0);
    EXPECT_FLOAT_EQ(fm.total_time_ms, 0.0f);
    EXPECT_EQ(fm.find("any"), nullptr);
}
