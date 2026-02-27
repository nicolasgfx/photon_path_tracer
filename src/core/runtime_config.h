#pragma once
// ─────────────────────────────────────────────────────────────────────
// runtime_config.h — Load render_config.json into RenderConfig + Camera
// ─────────────────────────────────────────────────────────────────────
// Usage:
//   Press R to reload render_config.json and launch a render.
//   Edit the file while the application is running; changes take effect
//   on the next R press.
//
// Supported JSON keys (flat, single-level object):
//   §1  image_width, image_height, spp
//   §2  global_photon_budget, caustic_photon_budget,
//       photon_max_bounces, photon_min_bounces_rr, photon_rr_threshold
//   §3  gather_radius, caustic_radius
//   §4  max_specular_chain, nee_coverage_fraction
//   §5  exposure
//   §6  dof_enabled, dof_focus_distance, dof_f_number,
//       dof_sensor_height, dof_focus_range
//
// Keys with "__" prefix are documentation-only and are silently skipped.
// Unknown keys are silently ignored.
// Returns false (and leaves configs unchanged) if the file cannot be opened.
// ─────────────────────────────────────────────────────────────────────

#include "renderer/renderer.h"    // RenderConfig
#include "renderer/camera.h"      // Camera

#include <fstream>
#include <string>
#include <iostream>
#include <cstdlib>    // atoi, atof
#include <cctype>     // isspace

// ─────────────────────────────────────────────────────────────────────
// Out-params written by load_runtime_config() so the caller can decide
// which follow-up actions are needed.
// ─────────────────────────────────────────────────────────────────────
struct RuntimeConfigFlags {
    bool photon_params_changed = false;  ///< budget / bounces / rr changed → must retrace
    bool gather_radius_changed = false;  ///< gather/caustic radius changed  → must retrace
    bool dof_changed           = false;  ///< DOF field changed              → call cam.update()
    bool resolution_changed    = false;  ///< image_width / height / spp changed
    bool exposure_changed      = false;  ///< exposure changed               → call renderer.set_exposure()
    float exposure             = 1.0f;   ///< new exposure value (valid only when exposure_changed)
};

// ─────────────────────────────────────────────────────────────────────
// load_runtime_config()
//   Reads `path`, parses each "key": value pair and applies it to cfg
//   and cam.  Returns true on success, false if the file cannot be read.
// ─────────────────────────────────────────────────────────────────────
inline bool load_runtime_config(
    const std::string&  path,
    RenderConfig&       cfg,
    Camera&             cam,
    RuntimeConfigFlags& flags)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[Config] Cannot open: " << path << "\n";
        return false;
    }

    // ── Minimal JSON-like parser ──────────────────────────────────────
    // Strategy: iterate line by line, strip // comments, then scan for
    //   "key": value  patterns.  No nesting, no arrays.
    // Values are read as raw token strings (everything after ':' up to
    // ',' or end-of-line, trimmed of whitespace and quotes).

    auto trim_ws = [](const std::string& s) -> std::string {
        size_t l = 0, r = s.size();
        while (l < r && std::isspace((unsigned char)s[l])) ++l;
        while (r > l && std::isspace((unsigned char)s[r - 1])) --r;
        return s.substr(l, r - l);
    };

    auto strip_quotes = [](const std::string& s) -> std::string {
        if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
            return s.substr(1, s.size() - 2);
        return s;
    };

    // Apply one key/value pair.
    // Returns true if the key was recognised (and applied), false if it
    // should be silently ignored.
    auto apply = [&](const std::string& key, const std::string& val) {
        if (key.empty() || val.empty()) return;
        // Documentation-only keys start with __ — skip them.
        if (key.size() >= 2 && key[0] == '_' && key[1] == '_') return;

        auto as_int   = [&]() { return std::atoi(val.c_str()); };
        auto as_float = [&]() { return (float)std::atof(val.c_str()); };
        auto as_bool  = [&]() { return (val == "true" || val == "1"); };

        // §1 – Resolution & sampling
        if      (key == "image_width")  { cfg.image_width  = as_int();   flags.resolution_changed = true; }
        else if (key == "image_height") { cfg.image_height = as_int();   flags.resolution_changed = true; }
        else if (key == "spp")          { cfg.samples_per_pixel = as_int(); flags.resolution_changed = true; }

        // §2 – Photon pass
        else if (key == "global_photon_budget")  {
            cfg.global_photon_budget = as_int();
            cfg.num_photons          = cfg.global_photon_budget;
            flags.photon_params_changed = true;
        }
        else if (key == "caustic_photon_budget") {
            cfg.caustic_photon_budget = as_int();
            flags.photon_params_changed = true;
        }
        else if (key == "photon_max_bounces")    { cfg.max_bounces    = as_int();   flags.photon_params_changed = true; }
        else if (key == "photon_min_bounces_rr") { cfg.min_bounces_rr = as_int();   flags.photon_params_changed = true; }
        else if (key == "photon_rr_threshold")   { cfg.rr_threshold   = as_float(); flags.photon_params_changed = true; }

        // §3 – Gather kernel
        else if (key == "gather_radius")  { cfg.gather_radius  = as_float(); flags.gather_radius_changed = true; }
        else if (key == "caustic_radius") { cfg.caustic_radius = as_float(); flags.gather_radius_changed = true; }

        // §4 – Camera pass & direct lighting
        else if (key == "max_specular_chain")    { cfg.max_specular_chain    = as_int();   }
        else if (key == "nee_coverage_fraction") { cfg.nee_coverage_fraction = as_float(); }

        // §5 – Tone mapping
        else if (key == "exposure") {
            cfg.exposure           = as_float();
            flags.exposure         = cfg.exposure;
            flags.exposure_changed = true;
        }

        // §6 – Depth of field (Camera fields) — only flag change when value differs
        else if (key == "dof_enabled")        { bool  v = as_bool();  if (v != cam.dof_enabled)     { cam.dof_enabled     = v; flags.dof_changed = true; } }
        else if (key == "dof_focus_distance") { float v = as_float(); if (v != cam.dof_focus_dist)  { cam.dof_focus_dist  = v; flags.dof_changed = true; } }
        else if (key == "dof_f_number")       { float v = as_float(); if (v != cam.dof_f_number)    { cam.dof_f_number   = v; flags.dof_changed = true; } }
        else if (key == "dof_sensor_height")  { float v = as_float(); if (v != cam.sensor_height)   { cam.sensor_height  = v; flags.dof_changed = true; } }
        else if (key == "dof_focus_range")    { float v = as_float(); if (v != cam.dof_focus_range) { cam.dof_focus_range = v; flags.dof_changed = true; } }

        // §7 – Denoiser
        else if (key == "denoiser_enabled")       { cfg.denoiser_enabled       = as_bool();  }
        else if (key == "denoiser_guide_albedo")  { cfg.denoiser_guide_albedo  = as_bool();  }
        else if (key == "denoiser_guide_normal")  { cfg.denoiser_guide_normal  = as_bool();  }
        else if (key == "denoiser_blend")         { cfg.denoiser_blend         = as_float(); }
        // All other keys fall through silently.
    };

    // ── Line-by-line parse ────────────────────────────────────────────
    std::string line;
    while (std::getline(file, line)) {
        // Strip trailing whitespace / CR
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' ||
               line.back() == ' ' || line.back() == '\t'))
            line.pop_back();

        // Strip // comments
        auto comment_pos = line.find("//");
        if (comment_pos != std::string::npos)
            line = line.substr(0, comment_pos);

        // Find opening '"' for the key
        auto q1 = line.find('"');
        if (q1 == std::string::npos) continue;
        auto q2 = line.find('"', q1 + 1);
        if (q2 == std::string::npos) continue;
        std::string key = line.substr(q1 + 1, q2 - q1 - 1);

        // Find ':' separator
        auto colon = line.find(':', q2 + 1);
        if (colon == std::string::npos) continue;

        // Extract value token (everything after ':' up to optional ',' or end)
        std::string raw_val = line.substr(colon + 1);
        // Remove trailing comma
        auto comma = raw_val.rfind(',');
        if (comma != std::string::npos) raw_val = raw_val.substr(0, comma);
        // Remove trailing '}' (last field)
        auto brace = raw_val.rfind('}');
        if (brace != std::string::npos) raw_val = raw_val.substr(0, brace);

        std::string val = strip_quotes(trim_ws(raw_val));
        apply(key, val);
    }

    return true;
}
