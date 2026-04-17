#pragma once
// ─────────────────────────────────────────────────────────────────────
// diagnostics/diagnostics.h – Render diagnostics meta-skill (v5)
//
// The "brain" that observes all pipeline stages.  Collects StageMetrics,
// per-pixel variance data, and convergence history.  Produces
// BottleneckReports with actionable recommendations.
// ─────────────────────────────────────────────────────────────────────
#include "core/stage_metrics.h"
#include "diagnose/variance_tracker.h"
#include "diagnose/convergence_analyzer.h"
#include "diagnose/bottleneck_report.h"

class RenderDiagnostics {
public:
    // Feed a completed frame's metrics.
    void submit_frame(const FrameMetrics& metrics);

    // Feed per-pixel variance data from GPU buffers at current SPP.
    void submit_variance(
        const float* d_lum_sum,
        const float* d_lum_sum2,
        const float* d_sample_counts,
        int width, int height,
        int current_spp);

    // Generate full diagnostics report.
    BottleneckReport generate_report() const;

    // Access sub-components.
    VarianceTracker&       variance_tracker()       { return var_tracker_; }
    const VarianceTracker& variance_tracker() const { return var_tracker_; }
    ConvergenceAnalyzer&   convergence_analyzer()       { return conv_analyzer_; }
    const ConvergenceAnalyzer& convergence_analyzer() const { return conv_analyzer_; }

    // Get the most recent variance stats (cached from last submit_variance).
    const VarianceStats& latest_variance_stats() const { return latest_var_stats_; }

    // Get the most recent frame metrics (cached from last submit_frame).
    const FrameMetrics& latest_frame_metrics() const { return latest_frame_; }

    void reset();

private:
    VarianceTracker     var_tracker_;
    ConvergenceAnalyzer conv_analyzer_;
    BottleneckAnalyzer  bottleneck_analyzer_;

    VarianceStats  latest_var_stats_;
    FrameMetrics   latest_frame_;
};
