#pragma once
// ─────────────────────────────────────────────────────────────────────
// diagnostics/bottleneck_report.h – Convergence bottleneck identification
//
// Given per-stage StageMetrics and image-level VarianceStats, identifies
// which pipeline stage contributes the most noise and recommends
// parameter changes.
// ─────────────────────────────────────────────────────────────────────
#include "core/stage_metrics.h"
#include "diagnose/variance_tracker.h"
#include "diagnose/convergence_analyzer.h"
#include <string>
#include <vector>

struct Recommendation {
    std::string target_stage;      // e.g. "direct-lighting"
    std::string parameter;         // e.g. "max_bounces"
    std::string action;            // e.g. "increase from 4 to 8"
    std::string rationale;
    float       expected_improvement = 0.f;  // estimated % variance reduction
};

struct BottleneckReport {
    // Overall convergence
    float  total_mse                    = 0.f;
    float  convergence_rate             = 0.f;
    bool   converging_normally          = false;

    // Variance decomposition (fractions summing to ≈1.0)
    float  direct_variance_fraction     = 0.f;  // % from NEE/direct
    float  indirect_variance_fraction   = 0.f;  // % from bounces 2+
    float  photon_variance_fraction     = 0.f;  // % from photon gather
    float  firefly_fraction             = 0.f;  // % from outlier samples

    // Bottleneck identification
    std::string bottleneck_stage;         // dominant noise source
    int    dominant_noise_bounce  = -1;   // which bounce is noisiest (-1 = unknown)

    // Timing breakdown
    float  total_render_ms        = 0.f;
    std::string slowest_stage;
    float  slowest_stage_ms       = 0.f;

    // Recommendations
    std::vector<Recommendation> suggestions;

    // Human-readable summary
    std::string summary;
};

class BottleneckAnalyzer {
public:
    // Analyze metrics and produce a report.
    BottleneckReport analyze(
        const FrameMetrics& frame_metrics,
        const VarianceStats& var_stats,
        const ConvergenceAnalysis& conv_analysis) const
    {
        BottleneckReport report;

        report.total_mse = var_stats.mean_variance;
        report.convergence_rate = conv_analysis.rate_alpha;
        report.converging_normally = conv_analysis.normal_rate;

        // ── Timing analysis ─────────────────────────────────────────
        report.total_render_ms = frame_metrics.total_time_ms;
        for (const auto& stage : frame_metrics.stages) {
            if (stage.time_ms > report.slowest_stage_ms) {
                report.slowest_stage_ms = stage.time_ms;
                report.slowest_stage = stage.stage_name;
            }
        }

        // ── Variance decomposition from stage metrics ───────────────
        float total_variance = 0.f;
        float direct_var = 0.f, indirect_var = 0.f, photon_var = 0.f;

        for (const auto& stage : frame_metrics.stages) {
            total_variance += stage.variance_contribution;

            if (std::string(stage.stage_name) == "direct-lighting" ||
                std::string(stage.stage_name) == "nee")
                direct_var += stage.variance_contribution;
            else if (std::string(stage.stage_name) == "path-integrator" ||
                     std::string(stage.stage_name) == "indirect")
                indirect_var += stage.variance_contribution;
            else if (std::string(stage.stage_name) == "photon-gather" ||
                     std::string(stage.stage_name) == "photon")
                photon_var += stage.variance_contribution;
        }

        if (total_variance > 0.f) {
            report.direct_variance_fraction   = direct_var / total_variance;
            report.indirect_variance_fraction  = indirect_var / total_variance;
            report.photon_variance_fraction    = photon_var / total_variance;
        }

        // Firefly fraction from rejection count
        int total_samples = 0, total_rejected = 0;
        for (const auto& stage : frame_metrics.stages) {
            total_samples += stage.num_samples;
            total_rejected += stage.num_rejected;
        }
        report.firefly_fraction = (total_samples > 0)
            ? (float)total_rejected / (float)total_samples : 0.f;

        // ── Bottleneck identification ───────────────────────────────
        identify_bottleneck_(report);

        // ── Generate recommendations ────────────────────────────────
        generate_recommendations_(report, var_stats, conv_analysis);

        // ── Summary ─────────────────────────────────────────────────
        generate_summary_(report, conv_analysis);

        return report;
    }

private:
    void identify_bottleneck_(BottleneckReport& report) const {
        float max_frac = 0.f;
        if (report.direct_variance_fraction > max_frac) {
            max_frac = report.direct_variance_fraction;
            report.bottleneck_stage = "direct-lighting";
        }
        if (report.indirect_variance_fraction > max_frac) {
            max_frac = report.indirect_variance_fraction;
            report.bottleneck_stage = "path-integrator";
        }
        if (report.photon_variance_fraction > max_frac) {
            max_frac = report.photon_variance_fraction;
            report.bottleneck_stage = "photon-system";
        }
        if (report.firefly_fraction > 0.1f) {
            // High rejection rate → outliers dominate
            report.bottleneck_stage = "path-integrator (fireflies)";
        }
    }

    void generate_recommendations_(
        BottleneckReport& report,
        const VarianceStats& var_stats,
        const ConvergenceAnalysis& conv) const
    {
        if (report.direct_variance_fraction > 0.5f) {
            report.suggestions.push_back({
                "direct-lighting", "max_bounces",
                "check emitter visibility and scene geometry",
                "Direct lighting dominates variance (" +
                    std::to_string((int)(report.direct_variance_fraction * 100)) + "%)",
                report.direct_variance_fraction * 30.f
            });
        }

        if (report.indirect_variance_fraction > 0.5f) {
            report.suggestions.push_back({
                "path-integrator", "samples_per_pixel",
                "increase SPP or reduce clamping to lower indirect variance",
                "Indirect lighting dominates variance (" +
                    std::to_string((int)(report.indirect_variance_fraction * 100)) + "%)",
                report.indirect_variance_fraction * 25.f
            });
        }

        if (report.photon_variance_fraction > 0.3f) {
            report.suggestions.push_back({
                "photon-system", "photon_budget",
                "increase photon budget",
                "Photon gather contributes significant variance (" +
                    std::to_string((int)(report.photon_variance_fraction * 100)) + "%)",
                report.photon_variance_fraction * 20.f
            });
        }

        if (report.firefly_fraction > 0.05f) {
            report.suggestions.push_back({
                "post-processing", "firefly_threshold",
                "lower firefly threshold (more aggressive filtering)",
                "High outlier rate (" +
                    std::to_string((int)(report.firefly_fraction * 100)) + "% rejected)",
                15.f
            });
        }

        if (conv.stalled) {
            report.suggestions.push_back({
                "path-integrator", "max_sample_luminance",
                "check clamping thresholds — possible bias",
                "Convergence stalled (α=" + std::to_string(conv.rate_alpha) + ")",
                0.f
            });
        }

        if (var_stats.noisy_fraction > 0.5f) {
            report.suggestions.push_back({
                "renderer", "samples_per_pixel",
                "increase SPP — many pixels still noisy",
                std::to_string((int)(var_stats.noisy_fraction * 100)) +
                    "% of pixels above noise threshold",
                var_stats.noisy_fraction * 40.f
            });
        }
    }

    void generate_summary_(BottleneckReport& report,
                           const ConvergenceAnalysis& conv) const {
        report.summary = conv.assessment;
        if (!report.bottleneck_stage.empty()) {
            report.summary += " Bottleneck: " + report.bottleneck_stage + ".";
        }
        if (!report.suggestions.empty()) {
            report.summary += " " + std::to_string(report.suggestions.size()) +
                              " recommendation(s) available.";
        }
    }
};
