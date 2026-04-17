// ─────────────────────────────────────────────────────────────────────
// diagnostics/diagnostics.cpp – Render diagnostics implementation
// ─────────────────────────────────────────────────────────────────────
#include "diagnose/diagnostics.h"

void RenderDiagnostics::submit_frame(const FrameMetrics& metrics) {
    latest_frame_ = metrics;
}

void RenderDiagnostics::submit_variance(
    const float* d_lum_sum,
    const float* d_lum_sum2,
    const float* d_sample_counts,
    int width, int height,
    int current_spp)
{
    latest_var_stats_ = var_tracker_.compute_stats(
        d_lum_sum, d_lum_sum2, d_sample_counts, width, height);

    var_tracker_.record_convergence_point(current_spp, latest_var_stats_);
}

BottleneckReport RenderDiagnostics::generate_report() const {
    ConvergenceAnalysis conv = conv_analyzer_.analyze(var_tracker_);
    return bottleneck_analyzer_.analyze(latest_frame_, latest_var_stats_, conv);
}

void RenderDiagnostics::reset() {
    var_tracker_.clear_history();
    latest_var_stats_ = {};
    latest_frame_ = {};
}
