#pragma once
// ─────────────────────────────────────────────────────────────────────
// oracle/image_verdict.h – Image quality verdict & correction structs
// ─────────────────────────────────────────────────────────────────────
#include <string>
#include <vector>

enum class ArtifactType {
    Firefly,          // isolated bright outlier pixels
    Splotch,          // low-frequency blocky noise (under-trained guide)
    HighNoise,        // overall high noise level
    EnergyLoss,       // dark bias (over-clamping)
    Banding,          // quantization artifacts in smooth gradients
    SlowConvergence   // convergence rate below expected 1/N
};

inline const char* artifact_type_name(ArtifactType t) {
    switch (t) {
        case ArtifactType::Firefly:         return "Firefly";
        case ArtifactType::Splotch:         return "Splotch";
        case ArtifactType::HighNoise:       return "HighNoise";
        case ArtifactType::EnergyLoss:      return "EnergyLoss";
        case ArtifactType::Banding:         return "Banding";
        case ArtifactType::SlowConvergence: return "SlowConvergence";
    }
    return "Unknown";
}

struct ArtifactReport {
    ArtifactType type;
    float        severity = 0.f;    // [0,1]
    std::string  description;
};

struct Correction {
    std::string target_skill;        // e.g. "path-integrator"
    std::string parameter;           // e.g. "max_sample_luminance"
    float       current_value  = 0.f;
    float       recommended_value = 0.f;
    std::string rationale;
};

struct ImageVerdict {
    // Overall quality
    float noise_level         = 0.f;  // [0,1] estimated noise
    float psnr_vs_reference   = -1.f; // -1 = no reference
    float ssim_vs_reference   = -1.f; // -1 = no reference

    // Detected artifacts
    std::vector<ArtifactReport> artifacts;

    // Feedback corrections
    std::vector<Correction> corrections;

    // Convergence
    float       convergence_rate      = 0.f;
    bool        converging_normally   = false;
    std::string convergence_note;

    // Summary
    std::string summary;

    bool has_artifact(ArtifactType type) const {
        for (const auto& a : artifacts)
            if (a.type == type) return true;
        return false;
    }

    float artifact_severity(ArtifactType type) const {
        for (const auto& a : artifacts)
            if (a.type == type) return a.severity;
        return 0.f;
    }
};
