#pragma once
// ─────────────────────────────────────────────────────────────────────
// stage_metrics.h – Per-stage convergence telemetry
//
// Every pipeline stage writes a StageMetrics struct during rendering.
// The render-diagnostics meta-skill collects all of them to produce
// convergence reports and bottleneck identification.
// ─────────────────────────────────────────────────────────────────────
#include <string>
#include <vector>
#include <utility>

struct StageMetrics {
    const char* stage_name = "";
    float       time_ms    = 0.f;

    // Variance contribution (set by stages that produce samples)
    float       mean_contribution     = 0.f;
    float       variance_contribution = 0.f;
    int         num_samples           = 0;
    int         num_rejected          = 0;   // zero-contribution or clamped samples

    // Stage-specific key-value pairs
    std::vector<std::pair<std::string, float>> custom;

    void set(const std::string& key, float value) {
        for (auto& kv : custom) {
            if (kv.first == key) { kv.second = value; return; }
        }
        custom.emplace_back(key, value);
    }

    float get(const std::string& key, float default_val = 0.f) const {
        for (const auto& kv : custom) {
            if (kv.first == key) return kv.second;
        }
        return default_val;
    }
};

// Aggregates metrics from all pipeline stages for a single frame/pass.
struct FrameMetrics {
    int                      frame_number = 0;
    int                      spp          = 0;
    float                    total_time_ms = 0.f;
    std::vector<StageMetrics> stages;

    void add(const StageMetrics& m) {
        stages.push_back(m);
        total_time_ms += m.time_ms;
    }

    const StageMetrics* find(const char* name) const {
        for (const auto& s : stages) {
            if (std::string(s.stage_name) == name) return &s;
        }
        return nullptr;
    }
};
