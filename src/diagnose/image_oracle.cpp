// ─────────────────────────────────────────────────────────────────────
// oracle/image_oracle.cpp – Image oracle implementation
// ─────────────────────────────────────────────────────────────────────
#include "diagnose/image_oracle.h"
#include <cstdio>
#include <string>

ImageVerdict ImageOracle::analyze(
    const float* rgb_image, int width, int height) const
{
    ImageVerdict verdict;

    // 1. Noise estimation
    NoiseEstimate noise = noise_analyzer::analyze(rgb_image, width, height, firefly_K_);
    verdict.noise_level = noise.global_noise_level;

    // 2. Detect artifacts
    verdict.artifacts = artifact_detector::detect(noise, thresholds_);

    // 3. Generate corrections
    verdict.corrections = artifact_detector::suggest_corrections(verdict.artifacts);

    // 4. Summary
    verdict.summary = "Noise level: " + std::to_string(verdict.noise_level);
    if (!verdict.artifacts.empty()) {
        verdict.summary += ". Detected: ";
        for (size_t i = 0; i < verdict.artifacts.size(); ++i) {
            if (i > 0) verdict.summary += ", ";
            verdict.summary += artifact_type_name(verdict.artifacts[i].type);
            verdict.summary += " (sev=" + std::to_string(verdict.artifacts[i].severity) + ")";
        }
    }
    if (!verdict.corrections.empty()) {
        verdict.summary += ". " + std::to_string(verdict.corrections.size()) +
                           " correction(s) suggested.";
    }

    return verdict;
}

ImageVerdict ImageOracle::analyze_with_reference(
    const float* rendered, const float* reference,
    int width, int height) const
{
    // Start with no-reference analysis
    ImageVerdict verdict = analyze(rendered, width, height);

    // Add reference-based metrics
    auto metrics = image_metrics::compute_metrics(rendered, reference, width, height);
    verdict.psnr_vs_reference = metrics.psnr;
    verdict.ssim_vs_reference = metrics.ssim;

    // Update summary
    verdict.summary += " PSNR=" + std::to_string(metrics.psnr) +
                       "dB SSIM=" + std::to_string(metrics.ssim);

    return verdict;
}

ImageVerdict ImageOracle::compare(
    const float* render_a, const float* render_b,
    int width, int height) const
{
    // Analyze B with A as reference
    ImageVerdict verdict = analyze_with_reference(render_b, render_a, width, height);

    // Also run noise analysis on both to determine improvement
    NoiseEstimate noise_a = noise_analyzer::analyze(render_a, width, height, firefly_K_);
    NoiseEstimate noise_b = noise_analyzer::analyze(render_b, width, height, firefly_K_);

    float noise_change = noise_b.global_noise_level - noise_a.global_noise_level;
    if (noise_change < -0.01f) {
        verdict.convergence_note = "B is less noisy than A (improvement)";
    } else if (noise_change > 0.01f) {
        verdict.convergence_note = "B is noisier than A (regression)";
    } else {
        verdict.convergence_note = "Similar noise levels";
    }

    return verdict;
}
