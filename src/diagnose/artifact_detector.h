#pragma once
// ─────────────────────────────────────────────────────────────────────
// oracle/artifact_detector.h – Classify artifact types and severity
//
// Maps detected image features (noise estimates, metrics) to typed
// ArtifactReports with severity and correction recommendations.
// Encodes the artifact → skill correction table from the plan.
// ─────────────────────────────────────────────────────────────────────
#include "diagnose/image_verdict.h"
#include "diagnose/noise_analyzer.h"
#include "diagnose/image_metrics.h"
#include <string>
#include <vector>
#include <cmath>

namespace artifact_detector {

// Thresholds (tunable)
struct DetectionThresholds {
    float firefly_severity_low   = 0.001f;  // fraction for severity 0.3
    float firefly_severity_high  = 0.01f;   // fraction for severity 1.0
    float splotch_threshold      = 0.15f;   // autocorrelation score for detection
    float noise_high_threshold   = 0.15f;   // global noise level for "HighNoise"
    float noise_very_high        = 0.30f;   // for severity 1.0
};

inline std::vector<ArtifactReport> detect(
    const NoiseEstimate& noise,
    const DetectionThresholds& thresholds = {})
{
    std::vector<ArtifactReport> artifacts;

    // ── Fireflies ───────────────────────────────────────────────────
    if (noise.firefly_fraction > thresholds.firefly_severity_low) {
        float severity = std::min(
            (noise.firefly_fraction - thresholds.firefly_severity_low) /
            (thresholds.firefly_severity_high - thresholds.firefly_severity_low),
            1.f);
        severity = std::max(severity, 0.1f);

        artifacts.push_back({
            ArtifactType::Firefly, severity,
            std::to_string(noise.num_firefly_pixels) + " firefly pixels (" +
                std::to_string((int)(noise.firefly_fraction * 10000) / 100.f) + "%)"
        });
    }

    // ── Splotchy noise ──────────────────────────────────────────────
    if (noise.splotch_score > thresholds.splotch_threshold) {
        float severity = std::min(
            (noise.splotch_score - thresholds.splotch_threshold) /
            (1.f - thresholds.splotch_threshold), 1.f);

        artifacts.push_back({
            ArtifactType::Splotch, severity,
            "Low-frequency structured noise (autocorr=" +
                std::to_string(noise.splotch_score) + ")"
        });
    }

    // ── High noise ──────────────────────────────────────────────────
    if (noise.global_noise_level > thresholds.noise_high_threshold) {
        float severity = std::min(
            (noise.global_noise_level - thresholds.noise_high_threshold) /
            (thresholds.noise_very_high - thresholds.noise_high_threshold),
            1.f);
        severity = std::max(severity, 0.1f);

        artifacts.push_back({
            ArtifactType::HighNoise, severity,
            "Global noise level " + std::to_string(noise.global_noise_level)
        });
    }

    return artifacts;
}

// ── Map artifacts to corrections ────────────────────────────────────
// Encodes the artifact → skill correction knowledge table.

inline std::vector<Correction> suggest_corrections(
    const std::vector<ArtifactReport>& artifacts)
{
    std::vector<Correction> corrections;

    for (const auto& art : artifacts) {
        switch (art.type) {
            case ArtifactType::Firefly:
                corrections.push_back({
                    "path-integrator", "max_sample_luminance",
                    10000.f, 5000.f,
                    "Fireflies detected — lower sample luminance clamp"
                });
                corrections.push_back({
                    "post-processing", "firefly_threshold",
                    4.0f, 3.0f,
                    "Fireflies detected — more aggressive post-filter"
                });
                break;

            case ArtifactType::Splotch:
                corrections.push_back({
                    "renderer", "samples_per_pixel",
                    64.f, 256.f,
                    "Splotchy noise — increase SPP for better convergence"
                });
                break;

            case ArtifactType::HighNoise:
                if (art.severity > 0.5f) {
                    corrections.push_back({
                        "renderer", "samples_per_pixel",
                        64.f, 256.f,
                        "High noise level — increase SPP"
                    });
                }
                corrections.push_back({
                    "renderer", "max_sample_luminance",
                    1e4f, 1e3f,
                    "High noise — tighten sample luminance clamp"
                });
                break;

            case ArtifactType::EnergyLoss:
                corrections.push_back({
                    "path-integrator", "max_bounce_contribution",
                    1e4f, 1e5f,
                    "Energy loss — raise bounce contribution clamp"
                });
                corrections.push_back({
                    "path-integrator", "rr_threshold",
                    0.95f, 0.98f,
                    "Energy loss — raise RR survival threshold"
                });
                break;

            case ArtifactType::Banding:
                corrections.push_back({
                    "camera-system", "jitter_pattern",
                    0.f, 1.f,
                    "Banding — switch to stratified jitter"
                });
                break;

            case ArtifactType::SlowConvergence:
                corrections.push_back({
                    "render-diagnostics", "check_bottleneck",
                    0.f, 0.f,
                    "Slow convergence — run diagnostics to identify bottleneck stage"
                });
                break;
        }
    }

    return corrections;
}

} // namespace artifact_detector
