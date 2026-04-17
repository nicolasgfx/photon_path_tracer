#pragma once
// ─────────────────────────────────────────────────────────────────────
// analyze_json_output.h – Serialize SceneProfile to JSON (stdout)
//
// Used by ppt_analyze to emit machine-readable analysis results.
// ─────────────────────────────────────────────────────────────────────
#include "core/scene_profile.h"
#include <cstdio>

// String conversions for enums
inline const char* to_string(LightingType t) {
    switch (t) {
        case LightingType::LargeArea:  return "large_area";
        case LightingType::SmallPoint: return "small_point";
        case LightingType::Mixed:      return "mixed";
    }
    return "unknown";
}

inline const char* to_string(EmitterDistribution d) {
    switch (d) {
        case EmitterDistribution::Uniform:        return "uniform";
        case EmitterDistribution::HighVariance:   return "high_variance";
        case EmitterDistribution::SingleDominant: return "single_dominant";
    }
    return "unknown";
}

inline const char* to_string(GeometryComplexity c) {
    switch (c) {
        case GeometryComplexity::Simple:   return "simple";
        case GeometryComplexity::Moderate: return "moderate";
        case GeometryComplexity::Complex:  return "complex";
        case GeometryComplexity::Dense:    return "dense";
    }
    return "unknown";
}

// Print SceneProfile as JSON to stdout.
// Uses printf for zero dependencies (no nlohmann_json needed).
inline void print_scene_profile_json(const SceneProfile& sp) {
    std::printf("{\n");
    std::printf("  \"scene_metrics\": {\n");
    std::printf("    \"num_triangles\": %d,\n",          sp.num_triangles);
    std::printf("    \"num_instances\": %d,\n",          sp.num_instances);
    std::printf("    \"scene_diagonal\": %.4f,\n",       sp.scene_diagonal);
    std::printf("    \"geometry_complexity\": \"%s\",\n", to_string(sp.geometry_complexity));
    std::printf("    \"has_open_geometry\": %s,\n",      sp.has_open_geometry ? "true" : "false");
    std::printf("    \"num_emitters\": %d,\n",           sp.num_emitters);
    std::printf("    \"emitter_size_ratio\": %.6f,\n",   sp.emitter_size_ratio);
    std::printf("    \"emitter_distribution\": \"%s\",\n", to_string(sp.emitter_distribution));
    std::printf("    \"dominant_lighting\": \"%s\",\n",  to_string(sp.dominant_lighting));
    std::printf("    \"mostly_indirect_emitters\": %s,\n", sp.mostly_indirect_emitters ? "true" : "false");
    std::printf("    \"emitter_direct_visibility\": %.4f,\n", sp.emitter_direct_visibility);
    std::printf("    \"has_caustic_paths\": %s,\n",      sp.has_caustic_paths ? "true" : "false");
    std::printf("    \"has_glass\": %s,\n",              sp.has_glass ? "true" : "false");
    std::printf("    \"has_metal\": %s,\n",              sp.has_metal ? "true" : "false");
    std::printf("    \"has_translucent\": %s,\n",        sp.has_translucent ? "true" : "false");
    std::printf("    \"has_clearcoat\": %s,\n",          sp.has_clearcoat ? "true" : "false");
    std::printf("    \"avg_roughness\": %.4f,\n",        sp.avg_roughness);
    std::printf("    \"num_material_types\": %d,\n",     sp.num_material_types);
    std::printf("    \"num_delta_materials\": %d,\n",     sp.num_delta_materials);
    std::printf("    \"num_delta_triangles\": %d,\n",     sp.num_delta_triangles);
    std::printf("    \"delta_area_fraction\": %.6f,\n",   sp.delta_area_fraction);
    std::printf("    \"caustic_geometry_favorable\": %s,\n", sp.caustic_geometry_favorable ? "true" : "false");
    std::printf("    \"total_emissive_flux\": %.4f,\n",   sp.total_emissive_flux);
    std::printf("    \"max_emitter_radiance\": %.4f,\n",   sp.max_emitter_radiance);
    std::printf("    \"emitter_delta_coupling\": %.6f,\n", sp.emitter_delta_coupling);
    std::printf("    \"caustic_difficulty\": %.2f\n",      sp.caustic_difficulty);
    std::printf("  },\n");
    std::printf("  \"recommended_config\": {\n");
    std::printf("    \"max_bounces\": %d,\n",            sp.recommended_max_bounces);
    std::printf("    \"photon_budget\": %d,\n",          sp.recommended_photon_budget);
    std::printf("    \"caustic_photon_budget\": %d,\n",  sp.recommended_caustic_photon_budget);
    std::printf("    \"guide_training_iters\": %d,\n",   sp.recommended_guide_training_iters);
    std::printf("    \"guide_fraction\": %.3f,\n",       sp.recommended_guide_fraction);
    std::printf("    \"gather_radius\": %.6f,\n",        sp.recommended_gather_radius);
    std::printf("    \"caustic_radius\": %.6f,\n",       sp.recommended_caustic_radius);
    std::printf("    \"caustic_enabled\": %s,\n",        sp.recommended_caustic_enabled ? "true" : "false");
    std::printf("    \"caustic_photons_per_frame\": %d,\n", sp.recommended_caustic_photons_per_frame);
    std::printf("    \"caustic_max_splat_luminance\": %.1f,\n", sp.recommended_caustic_max_splat_luminance);
    std::printf("    \"light_tree_enabled\": %s\n", sp.recommended_light_tree_enabled ? "true" : "false");
    std::printf("  }\n");
    std::printf("}\n");
}
