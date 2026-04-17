// ─────────────────────────────────────────────────────────────────────
// app/render_config_json.cpp – JSON load / save for RenderConfig
// ─────────────────────────────────────────────────────────────────────
#include "app/render_config.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <cstdio>

using json = nlohmann::json;

// Helper: read a JSON value into a variable only if the key exists.
template<typename T>
static void read_opt(const json& j, const char* key, T& out) {
    if (j.contains(key)) j.at(key).get_to(out);
}

bool RenderConfig::load_json(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    json j;
    try {
        f >> j;
    } catch (const json::parse_error& e) {
        std::fprintf(stderr, "[Config] JSON parse error in %s: %s\n",
                     path.c_str(), e.what());
        return false;
    }

    // ── Image output ────────────────────────────────────────────────
    read_opt(j, "image_width",       image_width);
    read_opt(j, "image_height",      image_height);

    // ── Core rendering ──────────────────────────────────────────────
    read_opt(j, "samples_per_pixel", samples_per_pixel);
    read_opt(j, "max_bounces",       max_bounces);
    read_opt(j, "min_bounces_rr",    min_bounces_rr);
    read_opt(j, "rr_threshold",      rr_threshold);
    read_opt(j, "max_specular_chain", max_specular_chain);
    read_opt(j, "spp_per_launch",     spp_per_launch);

    // ── Clamping ────────────────────────────────────────────────────
    read_opt(j, "clamping_enabled",        clamping_enabled);
    read_opt(j, "max_bounce_contribution", max_bounce_contribution);
    read_opt(j, "max_path_throughput",     max_path_throughput);
    read_opt(j, "max_nee_contribution",    max_nee_contribution);
    read_opt(j, "max_sample_luminance",    max_sample_luminance);

    // ── Adaptive sampling ───────────────────────────────────────────
    read_opt(j, "adaptive_sampling",  adaptive_sampling);
    read_opt(j, "adaptive_min_spp",   adaptive_min_spp);
    read_opt(j, "adaptive_threshold", adaptive_threshold);
    read_opt(j, "adaptive_radius",    adaptive_radius);

    // ── Tone mapping ────────────────────────────────────────────────
    read_opt(j, "exposure",    exposure);
    read_opt(j, "light_scale", light_scale);

    // ── Denoiser ────────────────────────────────────────────────────
    read_opt(j, "denoiser_enabled",      denoiser_enabled);
    read_opt(j, "denoiser_guide_albedo", denoiser_guide_albedo);
    read_opt(j, "denoiser_guide_normal", denoiser_guide_normal);
    read_opt(j, "denoiser_blend",        denoiser_blend);

    // ── Post-processing ─────────────────────────────────────────────
    read_opt(j, "bloom_enabled",      postfx.bloom_enabled);
    read_opt(j, "bloom_intensity",    postfx.bloom_intensity);
    read_opt(j, "bloom_radius_h",     postfx.bloom_radius_h);
    read_opt(j, "bloom_radius_v",     postfx.bloom_radius_v);
    read_opt(j, "firefly_enabled",    postfx.firefly_enabled);
    read_opt(j, "firefly_radius",     postfx.firefly_radius);
    read_opt(j, "firefly_threshold",  postfx.firefly_threshold);
    read_opt(j, "use_aces",           postfx.use_aces);

    // ── Depth of field ──────────────────────────────────────────────
    read_opt(j, "dof_enabled",        dof_enabled);
    read_opt(j, "dof_focus_distance", dof_focus_distance);
    read_opt(j, "dof_f_number",       dof_f_number);
    // ── Caustic light tracing ──────────────────────────────────────────
    read_opt(j, "caustic_enabled",             caustic_enabled);
    read_opt(j, "caustic_photons_per_frame",   caustic_photons_per_frame);
    read_opt(j, "caustic_max_splat_luminance", caustic_max_splat_luminance);
    // ── Light tree ──────────────────────────────────────────────────
    read_opt(j, "light_tree_enabled",        light_tree_enabled);
    read_opt(j, "light_tree_max_leaf_size",  light_tree_max_leaf_size);
    // ── Render mode (string → enum) ────────────────────────────────
    if (j.contains("render_mode")) {
        std::string m = j["render_mode"].get<std::string>();
        if      (m == "combined" || m == "full") mode = RenderMode::Combined;
        else if (m == "direct")          mode = RenderMode::DirectOnly;
        else if (m == "indirect")        mode = RenderMode::IndirectOnly;
        else if (m == "normals")         mode = RenderMode::Normals;
        else if (m == "material")        mode = RenderMode::MaterialID;
        else if (m == "depth")           mode = RenderMode::Depth;
    }

    std::printf("[Config] Loaded JSON config from %s\n", path.c_str());
    return true;
}

// Helper: render mode → string
static const char* mode_str(RenderMode m) {
    switch (m) {
        case RenderMode::Combined:      return "combined";
        case RenderMode::DirectOnly:    return "direct";
        case RenderMode::IndirectOnly:  return "indirect";
        case RenderMode::Normals:       return "normals";
        case RenderMode::MaterialID:    return "material";
        case RenderMode::Depth:         return "depth";
        default:                        return "combined";
    }
}

bool RenderConfig::save_json(const std::string& path) const {
    json j;

    // ── Image output ────────────────────────────────────────────────
    j["image_width"]       = image_width;
    j["image_height"]      = image_height;

    // ── Core rendering ──────────────────────────────────────────────
    j["samples_per_pixel"] = samples_per_pixel;
    j["max_bounces"]       = max_bounces;
    j["min_bounces_rr"]    = min_bounces_rr;
    j["rr_threshold"]      = rr_threshold;
    j["max_specular_chain"] = max_specular_chain;
    j["spp_per_launch"]     = spp_per_launch;

    // ── Clamping ────────────────────────────────────────────────────
    j["clamping_enabled"]        = clamping_enabled;
    j["max_bounce_contribution"] = max_bounce_contribution;
    j["max_path_throughput"]     = max_path_throughput;
    j["max_nee_contribution"]    = max_nee_contribution;
    j["max_sample_luminance"]    = max_sample_luminance;

    // ── Adaptive sampling ───────────────────────────────────────────
    j["adaptive_sampling"]  = adaptive_sampling;
    j["adaptive_min_spp"]   = adaptive_min_spp;
    j["adaptive_threshold"] = adaptive_threshold;
    j["adaptive_radius"]    = adaptive_radius;

    // ── Tone mapping ────────────────────────────────────────────────
    j["exposure"]    = exposure;
    j["light_scale"] = light_scale;

    // ── Denoiser ────────────────────────────────────────────────────
    j["denoiser_enabled"]      = denoiser_enabled;
    j["denoiser_guide_albedo"] = denoiser_guide_albedo;
    j["denoiser_guide_normal"] = denoiser_guide_normal;
    j["denoiser_blend"]        = denoiser_blend;

    // ── Post-processing ─────────────────────────────────────────────
    j["bloom_enabled"]      = postfx.bloom_enabled;
    j["bloom_intensity"]    = postfx.bloom_intensity;
    j["bloom_radius_h"]     = postfx.bloom_radius_h;
    j["bloom_radius_v"]     = postfx.bloom_radius_v;
    j["firefly_enabled"]    = postfx.firefly_enabled;
    j["firefly_radius"]     = postfx.firefly_radius;
    j["firefly_threshold"]  = postfx.firefly_threshold;
    j["use_aces"]           = postfx.use_aces;

    // ── Depth of field ──────────────────────────────────────────────
    j["dof_enabled"]        = dof_enabled;
    j["dof_focus_distance"] = dof_focus_distance;
    j["dof_f_number"]       = dof_f_number;

    // ── Caustic light tracing ──────────────────────────────────────────
    j["caustic_enabled"]             = caustic_enabled;
    j["caustic_photons_per_frame"]   = caustic_photons_per_frame;
    j["caustic_max_splat_luminance"] = caustic_max_splat_luminance;

    // ── Light tree ──────────────────────────────────────────────────
    j["light_tree_enabled"]        = light_tree_enabled;
    j["light_tree_max_leaf_size"]  = light_tree_max_leaf_size;

    // ── Render mode ─────────────────────────────────────────────────
    j["render_mode"] = mode_str(mode);

    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << j.dump(2) << '\n';
    std::printf("[Config] Saved JSON config to %s\n", path.c_str());
    return true;
}
