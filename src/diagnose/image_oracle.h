#pragma once
// ─────────────────────────────────────────────────────────────────────
// oracle/image_oracle.h – Main image analysis engine (v5)
//
// Orchestrates noise analysis, artifact detection, and correction
// generation.  Supports both no-reference and reference-based modes.
// ─────────────────────────────────────────────────────────────────────
#include "diagnose/image_verdict.h"
#include "diagnose/image_metrics.h"
#include "diagnose/noise_analyzer.h"
#include "diagnose/artifact_detector.h"

class ImageOracle {
public:
    // No-reference analysis: analyze a single rendered image.
    //   rgb_image: [width*height*3] float RGB (linear HDR, already divided by SPP)
    ImageVerdict analyze(const float* rgb_image, int width, int height) const;

    // Reference-based analysis: compare render against ground truth.
    //   Both images: [width*height*3] float RGB
    ImageVerdict analyze_with_reference(
        const float* rendered, const float* reference,
        int width, int height) const;

    // Compare two renders (A/B differential analysis).
    //   Returns verdict for render_b relative to render_a.
    ImageVerdict compare(
        const float* render_a, const float* render_b,
        int width, int height) const;

    // Configuration
    void set_firefly_K(float K)   { firefly_K_ = K; }
    void set_thresholds(const artifact_detector::DetectionThresholds& t) {
        thresholds_ = t;
    }

private:
    float firefly_K_ = 10.f;
    artifact_detector::DetectionThresholds thresholds_;
};
