// ─────────────────────────────────────────────────────────────────────
// photon_map_analysis – Standalone photon map spatial hierarchy analyser
// ─────────────────────────────────────────────────────────────────────
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
// Loads a photon cache + scene, traces CPU reference rays, queries
// every spatial hierarchy at each non-specular hit point with ≥3 hits,
// then writes a pretty-printed JSON comparison report.
//
// Usage:
//   photon_map_analysis <cache.bin> <scene.obj>
//       [--snapshot <snapshot.json>]    # load camera from snapshot JSON
//       [--radius <float>]             # override gather radius
//       [--num-rays <int>]             # number of reference rays (default 100)
//       [--min-hits <int>]             # min non-specular hits (default 3)
//       [--knn-k <int>]               # k for kNN queries   (default 64)
//       [--tau <float>]               # surface filter tau   (default 0.02)
//       [--output <path.json>]         # output file path
// ─────────────────────────────────────────────────────────────────────

#include "photon/photon_io.h"
#include "photon/hash_grid.h"
#include "photon/kd_tree.h"
#include "photon/cell_bin_grid.h"
#include "photon/hash_histogram.h"
#include "photon/cell_cache.h"
#include "photon/photon_analysis.h"
#include "photon/photon_bins.h"
#include "scene/obj_loader.h"
#include "renderer/camera.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iomanip>

// ─────────────────────────────────────────────────────────────────────
// Config defaults
// ─────────────────────────────────────────────────────────────────────
struct ToolConfig {
    std::string cache_path;
    std::string scene_path;         // relative to scenes/ dir
    std::string scenes_dir;         // root scenes directory
    std::string snapshot_path;      // optional: load camera from this
    std::string output_path = "photon_analysis_report.json";

    int   num_rays   = 100;
    int   min_hits   = 3;        // minimum non-specular hits per ray
    int   knn_k      = 64;
    float tau        = 0.02f;
    float radius_override = -1.0f;  // <0 means use from cache header

    // Camera overrides (used if no snapshot JSON provided)
    float cam_pos[3]    = {0.f, 0.f, 0.f};
    float cam_lookat[3] = {0.f, 0.f, -1.f};
    float cam_up[3]     = {0.f, 1.f, 0.f};
    float cam_fov       = 40.0f;
    int   cam_width     = 1024;
    int   cam_height    = 768;
    bool  cam_from_snapshot = false;
};

// ─────────────────────────────────────────────────────────────────────
// HitPoint: a single non-specular surface hit along a reference ray
// ─────────────────────────────────────────────────────────────────────
struct ReferenceHit {
    float3   position;
    float3   normal;
    uint32_t triangle_id;
    uint32_t material_id;
    uint8_t  material_type;  // MaterialType enum
    int      bounce;         // bounce index (0 = primary)
};

struct ReferenceRay {
    int ray_index;
    int pixel_x, pixel_y;
    std::vector<ReferenceHit> hits;
};

// ─────────────────────────────────────────────────────────────────────
// Per-hierarchy gather result at one hit point
// ─────────────────────────────────────────────────────────────────────
struct GatherResult {
    std::string hierarchy_name;
    int    photon_count    = 0;
    float  total_flux      = 0.0f;
    float  min_dist2       = 1e30f;
    float  max_dist2       = 0.0f;
    double build_ms        = 0.0;  // time to build this hierarchy (shared)
    double query_us        = 0.0;  // time to query this single hit point

    // For binned hierarchies (CellBinGrid, HashHistogram)
    int    occupied_bins   = 0;
    float  binned_flux     = 0.0f;
    int    active_bins     = 0;    // bins with flux > 0
    float  concentration   = 0.0f; // f_max / f_total

    // Per-bin directional breakdown (32 bins max)
    struct BinDetail {
        int   idx          = 0;
        float scalar_flux  = 0.0f;
        float avg_nx = 0, avg_ny = 0, avg_nz = 0;
        float dir_x  = 0, dir_y  = 0, dir_z  = 0;  // centroid direction
        int   count        = 0;  // photon count (CellBinGrid only)
    };
    std::vector<BinDetail> bin_details;  // only non-zero bins

    // For counting hierarchies: spatial gather statistics
    float3 centroid          = {0.f, 0.f, 0.f};
    float  position_spread   = 0.0f;  // RMS distance from centroid
    float  normal_coherence  = 0.0f;  // mean dot(photon_normal, hit_normal)
};

// ─────────────────────────────────────────────────────────────────────
// Global Fibonacci bin directions (initialized once)
// ─────────────────────────────────────────────────────────────────────
static PhotonBinDirs g_bin_dirs;

// ─────────────────────────────────────────────────────────────────────
// Split photons into global + caustic by path_flags (for CellInfoCache)
// ─────────────────────────────────────────────────────────────────────
static void split_by_path_flags(
    const PhotonSoA& all,
    PhotonSoA& global_out,
    PhotonSoA& caustic_out)
{
    const size_t N = all.size();
    const bool has_flags = !all.path_flags.empty();

    size_t n_global = 0, n_caustic = 0;
    for (size_t i = 0; i < N; ++i) {
        if (has_flags && (all.path_flags[i] & (PHOTON_FLAG_CAUSTIC_GLASS | PHOTON_FLAG_CAUSTIC_SPECULAR)))
            n_caustic++;
        else
            n_global++;
    }

    global_out.reserve(n_global);
    caustic_out.reserve(n_caustic);
    global_out.clear();
    caustic_out.clear();

    for (size_t i = 0; i < N; ++i) {
        bool is_caustic = has_flags &&
            (all.path_flags[i] & (PHOTON_FLAG_CAUSTIC_GLASS | PHOTON_FLAG_CAUSTIC_SPECULAR));
        PhotonSoA& dst = is_caustic ? caustic_out : global_out;
        size_t di = dst.size();
        dst.resize((uint32_t)(di + 1));
        dst.pos_x[di] = all.pos_x[i];
        dst.pos_y[di] = all.pos_y[i];
        dst.pos_z[di] = all.pos_z[i];
        dst.wi_x[di]  = all.wi_x[i];
        dst.wi_y[di]  = all.wi_y[i];
        dst.wi_z[di]  = all.wi_z[i];
        dst.norm_x[di] = all.norm_x[i];
        dst.norm_y[di] = all.norm_y[i];
        dst.norm_z[di] = all.norm_z[i];
        for (int b = 0; b < NUM_LAMBDA; ++b)
            dst.spectral_flux[di * NUM_LAMBDA + b] = all.spectral_flux[i * NUM_LAMBDA + b];
        for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
            dst.lambda_bin[di * HERO_WAVELENGTHS + h] = all.lambda_bin[i * HERO_WAVELENGTHS + h];
            dst.flux[di * HERO_WAVELENGTHS + h] = all.flux[i * HERO_WAVELENGTHS + h];
        }
        dst.num_hero[di] = all.num_hero[i];
        if (i < all.source_emissive_idx.size())
            dst.source_emissive_idx[di] = all.source_emissive_idx[i];
        if (i < all.bin_idx.size())
            dst.bin_idx[di] = all.bin_idx[i];
        if (i < all.path_flags.size())
            dst.path_flags[di] = all.path_flags[i];
        if (i < all.bounce_count.size())
            dst.bounce_count[di] = all.bounce_count[i];
        if (i < all.tri_id.size())
            dst.tri_id[di] = all.tri_id[i];
    }

    std::printf("[tool] Photon split: %zu global + %zu caustic = %zu total\n",
                global_out.size(), caustic_out.size(), N);
}

// ─────────────────────────────────────────────────────────────────────
// Photon type statistics from path_flags
// ─────────────────────────────────────────────────────────────────────
struct PhotonTypeStats {
    size_t total            = 0;
    size_t global_count     = 0;
    size_t caustic_glass    = 0;
    size_t caustic_specular = 0;
    size_t traversed_glass  = 0;
    size_t volume_scatter   = 0;
    size_t dispersion       = 0;
};

static PhotonTypeStats compute_photon_type_stats(const PhotonSoA& photons) {
    PhotonTypeStats s;
    s.total = photons.size();
    for (size_t i = 0; i < s.total; ++i) {
        uint8_t f = (i < photons.path_flags.size()) ? photons.path_flags[i] : 0;
        if (f & PHOTON_FLAG_CAUSTIC_GLASS)     s.caustic_glass++;
        if (f & PHOTON_FLAG_CAUSTIC_SPECULAR)  s.caustic_specular++;
        if (f & PHOTON_FLAG_TRAVERSED_GLASS)   s.traversed_glass++;
        if (f & PHOTON_FLAG_VOLUME_SCATTER)    s.volume_scatter++;
        if (f & PHOTON_FLAG_DISPERSION)        s.dispersion++;
        if (!(f & (PHOTON_FLAG_CAUSTIC_GLASS | PHOTON_FLAG_CAUSTIC_SPECULAR)))
            s.global_count++;
    }
    return s;
}

// ─────────────────────────────────────────────────────────────────────
// Minimal JSON string escaping
// ─────────────────────────────────────────────────────────────────────
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────
// Parse a simple JSON value (very minimal, no full parser needed)
// ─────────────────────────────────────────────────────────────────────
static bool parse_json_camera(const std::string& json_path, ToolConfig& cfg) {
    std::ifstream f(json_path);
    if (!f.is_open()) {
        std::cerr << "[tool] Cannot open snapshot JSON: " << json_path << "\n";
        return false;
    }
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

    // Simple field extraction via find/sscanf patterns
    auto extract_array3 = [&](const char* key, float out[3]) -> bool {
        auto pos = content.find(std::string("\"") + key + "\"");
        if (pos == std::string::npos) return false;
        auto bracket = content.find('[', pos);
        if (bracket == std::string::npos) return false;
        if (sscanf(content.c_str() + bracket, "[%f, %f, %f]",
                   &out[0], &out[1], &out[2]) == 3)
            return true;
        // Try with %f,%f,%f (no spaces)
        return sscanf(content.c_str() + bracket, "[%f,%f,%f]",
                      &out[0], &out[1], &out[2]) == 3;
    };
    auto extract_float = [&](const char* key, float& out) -> bool {
        auto pos = content.find(std::string("\"") + key + "\"");
        if (pos == std::string::npos) return false;
        auto colon = content.find(':', pos);
        if (colon == std::string::npos) return false;
        return sscanf(content.c_str() + colon + 1, " %f", &out) == 1;
    };
    auto extract_int = [&](const char* key, int& out) -> bool {
        auto pos = content.find(std::string("\"") + key + "\"");
        if (pos == std::string::npos) return false;
        auto colon = content.find(':', pos);
        if (colon == std::string::npos) return false;
        return sscanf(content.c_str() + colon + 1, " %d", &out) == 1;
    };
    auto extract_string = [&](const char* key, std::string& out) -> bool {
        auto pos = content.find(std::string("\"") + key + "\"");
        if (pos == std::string::npos) return false;
        auto colon = content.find(':', pos);
        if (colon == std::string::npos) return false;
        auto q1 = content.find('"', colon + 1);
        if (q1 == std::string::npos) return false;
        auto q2 = content.find('"', q1 + 1);
        if (q2 == std::string::npos) return false;
        out = content.substr(q1 + 1, q2 - q1 - 1);
        return true;
    };

    bool ok = true;
    ok &= extract_array3("position", cfg.cam_pos);
    ok &= extract_array3("look_at",  cfg.cam_lookat);
    extract_array3("up", cfg.cam_up);  // optional, default (0,1,0) is fine
    extract_float("fov_deg", cfg.cam_fov);
    extract_int("width",  cfg.cam_width);
    extract_int("height", cfg.cam_height);

    // Try to extract scene obj_path
    std::string obj;
    if (extract_string("obj_path", obj) && !obj.empty()) {
        cfg.scene_path = obj;
    }

    if (!ok) {
        std::cerr << "[tool] Warning: could not fully parse camera from "
                  << json_path << " (using defaults for missing fields)\n";
    }
    return true;  // partial parse is OK
}

// ─────────────────────────────────────────────────────────────────────
// CPU reference ray tracer
// ─────────────────────────────────────────────────────────────────────
static std::vector<ReferenceRay> trace_reference_rays(
    const Scene& scene, const Camera& camera,
    int num_rays, int min_hits, int max_bounces = 20)
{
    std::vector<ReferenceRay> result;
    result.reserve(num_rays);

    // Deterministic grid sampling over image plane
    int grid_side = (int)std::ceil(std::sqrt((double)num_rays * 4));
    int attempts = 0;
    int ray_idx = 0;
    PCGRng rng = PCGRng::seed(42, 1);

    for (int gy = 0; gy < grid_side && (int)result.size() < num_rays; ++gy) {
        for (int gx = 0; gx < grid_side && (int)result.size() < num_rays; ++gx) {
            ++attempts;
            int px = (int)((float)(gx + 0.5f) / grid_side * camera.width);
            int py = (int)((float)(gy + 0.5f) / grid_side * camera.height);
            px = std::max(0, std::min(px, camera.width - 1));
            py = std::max(0, std::min(py, camera.height - 1));

            Ray ray = generate_camera_ray(
                px, py, rng,
                camera.width, camera.height,
                camera.lower_left, camera.horizontal, camera.vertical,
                camera.position, camera.u, camera.v,
                0.0f, 0.0f, 0.0f);  // no DOF for reference rays

            ReferenceRay rr;
            rr.ray_index = ray_idx++;
            rr.pixel_x = px;
            rr.pixel_y = py;

            for (int bounce = 0; bounce < max_bounces; ++bounce) {
                HitRecord hit = scene.intersect(ray);
                if (!hit.hit) break;

                const Material& mat = scene.materials[hit.material_id];

                // Emissive surfaces: stop tracing
                if (mat.is_emissive()) break;

                // Non-specular hit: record it
                if (!mat.is_specular()) {
                    ReferenceHit rh;
                    rh.position    = hit.position;
                    rh.normal      = hit.normal;
                    rh.triangle_id = hit.triangle_id;
                    rh.material_id = hit.material_id;
                    rh.material_type = (uint8_t)mat.type;
                    rh.bounce      = bounce;
                    rr.hits.push_back(rh);
                }

                // Continue tracing: reflect/refract
                // For specular surfaces, compute perfect reflection/refraction
                // For diffuse surfaces, use cosine-weighted hemisphere sample
                float3 new_dir;
                if (mat.type == MaterialType::Mirror) {
                    new_dir = ray.direction - hit.normal * (2.0f * dot(ray.direction, hit.normal));
                } else if (mat.type == MaterialType::Glass || mat.type == MaterialType::Translucent) {
                    // Simple refraction (Snell's law) with TIR fallback
                    float3 n = hit.normal;
                    float cos_i = -dot(ray.direction, n);
                    float eta = 1.0f / mat.ior;
                    if (cos_i < 0.f) {
                        n = -n;
                        cos_i = -cos_i;
                        eta = mat.ior;
                    }
                    float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
                    if (sin2_t > 1.0f) {
                        // Total internal reflection
                        new_dir = ray.direction - n * (2.0f * dot(ray.direction, n));
                    } else {
                        float cos_t = std::sqrt(1.0f - sin2_t);
                        new_dir = ray.direction * eta + n * (eta * cos_i - cos_t);
                    }
                } else {
                    // Diffuse: random cosine-weighted hemisphere
                    ONB onb = ONB::from_normal(hit.normal);
                    float u1 = rng.next_float();
                    float u2 = rng.next_float();
                    float r_disk = std::sqrt(u1);
                    float theta  = 2.0f * PI * u2;
                    float3 local_dir = make_f3(
                        r_disk * std::cos(theta),
                        r_disk * std::sin(theta),
                        std::sqrt(std::max(0.f, 1.0f - u1)));
                    new_dir = onb.local_to_world(local_dir);
                }

                ray.origin    = hit.position + normalize(new_dir) * 1e-4f;
                ray.direction = normalize(new_dir);
                ray.tmin      = 1e-4f;
                ray.tmax      = 1e20f;
            }

            // Only keep rays with enough non-specular hits
            if ((int)rr.hits.size() >= min_hits) {
                result.push_back(std::move(rr));
            }
        }
    }

    std::cout << "[tool] Traced " << attempts << " camera rays, kept "
              << result.size() << " with >= " << min_hits
              << " non-specular hits\n";
    return result;
}

// ─────────────────────────────────────────────────────────────────────
// Brute-force gather (ground truth)
// ─────────────────────────────────────────────────────────────────────
static GatherResult brute_force_gather(
    const PhotonSoA& photons, float3 pos, float3 normal,
    float radius, float tau)
{
    GatherResult gr;
    gr.hierarchy_name = "brute_force";
    float r2 = radius * radius;
    const size_t N = photons.size();

    // Accumulators for spatial stats
    double cx = 0, cy = 0, cz = 0;  // centroid accumulator
    double norm_dot_sum = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        float3 pp = make_f3(photons.pos_x[i], photons.pos_y[i], photons.pos_z[i]);
        float3 diff = pos - pp;

        // Tangential distance on normal plane
        float d_plane = dot(diff, normal);
        if (std::fabs(d_plane) > tau) continue;

        float3 v_tan = diff - normal * d_plane;
        float d_tan2 = dot(v_tan, v_tan);
        if (d_tan2 > r2) continue;

        float flux_i = photons.total_flux(i);
        gr.photon_count++;
        gr.total_flux += flux_i;
        gr.min_dist2 = std::min(gr.min_dist2, d_tan2);
        gr.max_dist2 = std::max(gr.max_dist2, d_tan2);

        cx += pp.x; cy += pp.y; cz += pp.z;
        float3 pn = make_f3(photons.norm_x[i], photons.norm_y[i], photons.norm_z[i]);
        norm_dot_sum += std::fabs(dot(pn, normal));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    gr.query_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    if (gr.photon_count == 0) {
        gr.min_dist2 = 0.0f;
    } else {
        int n = gr.photon_count;
        gr.centroid = make_f3((float)(cx / n), (float)(cy / n), (float)(cz / n));
        gr.normal_coherence = (float)(norm_dot_sum / n);

        // Second pass for position spread (RMS from centroid)
        double spread_sum = 0;
        for (size_t i = 0; i < N; ++i) {
            float3 pp = make_f3(photons.pos_x[i], photons.pos_y[i], photons.pos_z[i]);
            float3 diff = pos - pp;
            float d_plane = dot(diff, normal);
            if (std::fabs(d_plane) > tau) continue;
            float3 v_tan = diff - normal * d_plane;
            if (dot(v_tan, v_tan) > r2) continue;
            float3 dc = pp - gr.centroid;
            spread_sum += dot(dc, dc);
        }
        gr.position_spread = std::sqrt((float)(spread_sum / n));
    }
    return gr;
}

// ─────────────────────────────────────────────────────────────────────
// HashGrid gather
// ─────────────────────────────────────────────────────────────────────
static GatherResult hash_grid_gather(
    const HashGrid& grid, const PhotonSoA& photons,
    float3 pos, float3 normal, float radius, float tau)
{
    GatherResult gr;
    gr.hierarchy_name = "hash_grid";

    double cx = 0, cy = 0, cz = 0;
    double norm_dot_sum = 0;
    // Collect indices for position_spread second pass
    std::vector<uint32_t> gathered_indices;
    gathered_indices.reserve(256);

    auto t0 = std::chrono::high_resolution_clock::now();
    grid.query_tangential(pos, normal, radius, tau, photons,
        [&](uint32_t idx, float d_tan2) {
            float flux_i = photons.total_flux(idx);
            gr.photon_count++;
            gr.total_flux += flux_i;
            gr.min_dist2 = std::min(gr.min_dist2, d_tan2);
            gr.max_dist2 = std::max(gr.max_dist2, d_tan2);
            cx += photons.pos_x[idx];
            cy += photons.pos_y[idx];
            cz += photons.pos_z[idx];
            float3 pn = make_f3(photons.norm_x[idx], photons.norm_y[idx], photons.norm_z[idx]);
            norm_dot_sum += std::fabs(dot(pn, normal));
            gathered_indices.push_back(idx);
        });
    auto t1 = std::chrono::high_resolution_clock::now();
    gr.query_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    if (gr.photon_count == 0) {
        gr.min_dist2 = 0.0f;
    } else {
        int n = gr.photon_count;
        gr.centroid = make_f3((float)(cx / n), (float)(cy / n), (float)(cz / n));
        gr.normal_coherence = (float)(norm_dot_sum / n);
        double spread_sum = 0;
        for (uint32_t idx : gathered_indices) {
            float3 pp = make_f3(photons.pos_x[idx], photons.pos_y[idx], photons.pos_z[idx]);
            float3 dc = pp - gr.centroid;
            spread_sum += dot(dc, dc);
        }
        gr.position_spread = std::sqrt((float)(spread_sum / n));
    }
    return gr;
}

// ─────────────────────────────────────────────────────────────────────
// KDTree gather
// ─────────────────────────────────────────────────────────────────────
static GatherResult kd_tree_gather(
    const KDTree& tree, const PhotonSoA& photons,
    float3 pos, float3 normal, float radius, float tau)
{
    GatherResult gr;
    gr.hierarchy_name = "kd_tree";

    double cx = 0, cy = 0, cz = 0;
    double norm_dot_sum = 0;
    std::vector<uint32_t> gathered_indices;
    gathered_indices.reserve(256);

    auto t0 = std::chrono::high_resolution_clock::now();
    tree.query_tangential(pos, normal, radius, tau, photons,
        [&](uint32_t idx, float d_tan2) {
            float flux_i = photons.total_flux(idx);
            gr.photon_count++;
            gr.total_flux += flux_i;
            gr.min_dist2 = std::min(gr.min_dist2, d_tan2);
            gr.max_dist2 = std::max(gr.max_dist2, d_tan2);
            cx += photons.pos_x[idx];
            cy += photons.pos_y[idx];
            cz += photons.pos_z[idx];
            float3 pn = make_f3(photons.norm_x[idx], photons.norm_y[idx], photons.norm_z[idx]);
            norm_dot_sum += std::fabs(dot(pn, normal));
            gathered_indices.push_back(idx);
        });
    auto t1 = std::chrono::high_resolution_clock::now();
    gr.query_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    if (gr.photon_count == 0) {
        gr.min_dist2 = 0.0f;
    } else {
        int n = gr.photon_count;
        gr.centroid = make_f3((float)(cx / n), (float)(cy / n), (float)(cz / n));
        gr.normal_coherence = (float)(norm_dot_sum / n);
        double spread_sum = 0;
        for (uint32_t idx : gathered_indices) {
            float3 pp = make_f3(photons.pos_x[idx], photons.pos_y[idx], photons.pos_z[idx]);
            float3 dc = pp - gr.centroid;
            spread_sum += dot(dc, dc);
        }
        gr.position_spread = std::sqrt((float)(spread_sum / n));
    }
    return gr;
}

// ─────────────────────────────────────────────────────────────────────
// CellBinGrid gather (binned hierarchy)
// ─────────────────────────────────────────────────────────────────────
static GatherResult cell_bin_grid_gather(
    const CellBinGrid& grid, float3 pos)
{
    GatherResult gr;
    gr.hierarchy_name = "cell_bin_grid";

    float max_flux = 0.0f;
    auto t0 = std::chrono::high_resolution_clock::now();
    const PhotonBin* bp = grid.lookup(pos);
    if (bp) {
        for (int b = 0; b < grid.bin_count; ++b) {
            float sf = bp[b].scalar_flux;
            if (sf > 0.0f) {
                gr.occupied_bins++;
                gr.binned_flux += sf;
                if (sf > max_flux) max_flux = sf;

                GatherResult::BinDetail bd;
                bd.idx         = b;
                bd.scalar_flux = sf;
                bd.avg_nx      = bp[b].avg_nx;
                bd.avg_ny      = bp[b].avg_ny;
                bd.avg_nz      = bp[b].avg_nz;
                bd.dir_x       = bp[b].dir_x;
                bd.dir_y       = bp[b].dir_y;
                bd.dir_z       = bp[b].dir_z;
                bd.count       = bp[b].count;
                gr.bin_details.push_back(bd);
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    gr.query_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    gr.active_bins = gr.occupied_bins;
    gr.concentration = (gr.binned_flux > 0.f) ? max_flux / gr.binned_flux : 0.f;

    return gr;
}

// ─────────────────────────────────────────────────────────────────────
// HashHistogram gather (binned hierarchy)
// ─────────────────────────────────────────────────────────────────────
static GatherResult hash_histogram_gather(
    const HashHistogram& hist, float3 pos, float /*radius*/)
{
    GatherResult gr;
    gr.hierarchy_name = "hash_histogram";

    float max_flux = 0.0f;
    auto t0 = std::chrono::high_resolution_clock::now();
    if (hist.num_levels > 0) {
        const auto& level = hist.levels[0];  // finest level
        uint32_t bucket = level.cell_hash(pos.x, pos.y, pos.z);
        gr.occupied_bins = level.count_active_bins(bucket);
        gr.binned_flux   = level.bucket_total_flux.empty() ? 0.0f
                         : level.bucket_total_flux[bucket];
        gr.concentration = level.get_concentration(bucket);

        // Per-bin breakdown from gpu_bins
        if (!level.gpu_bins.empty()) {
            for (int k = 0; k < level.bin_count; ++k) {
                size_t slot = (size_t)bucket * level.bin_count + k;
                const GpuGuideBin& gb = level.gpu_bins[slot];
                if (gb.scalar_flux > 0.f) {
                    GatherResult::BinDetail bd;
                    bd.idx         = k;
                    bd.scalar_flux = gb.scalar_flux;
                    bd.avg_nx      = gb.avg_nx;
                    bd.avg_ny      = gb.avg_ny;
                    bd.avg_nz      = gb.avg_nz;
                    // Fibonacci direction for this bin
                    if (k < g_bin_dirs.count) {
                        bd.dir_x = g_bin_dirs.dirs[k].x;
                        bd.dir_y = g_bin_dirs.dirs[k].y;
                        bd.dir_z = g_bin_dirs.dirs[k].z;
                    }
                    gr.bin_details.push_back(bd);
                    if (gb.scalar_flux > max_flux) max_flux = gb.scalar_flux;
                }
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    gr.query_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    gr.active_bins = (int)gr.bin_details.size();

    return gr;
}

// ─────────────────────────────────────────────────────────────────────
// Material type name
// ─────────────────────────────────────────────────────────────────────
static const char* material_type_name(uint8_t t) {
    switch (t) {
        case 0: return "Lambertian";
        case 1: return "Mirror";
        case 2: return "Glass";
        case 3: return "GlossyMetal";
        case 4: return "Emissive";
        case 5: return "GlossyDielectric";
        case 6: return "Translucent";
        case 7: return "Clearcoat";
        case 8: return "Fabric";
        default: return "Unknown";
    }
}

// ─────────────────────────────────────────────────────────────────────
// JSON output writer
// ─────────────────────────────────────────────────────────────────────
static void write_json_report(
    const std::string& path,
    const ToolConfig& cfg,
    const PhotonSoA& photons,
    float gather_radius,
    const PhotonTypeStats& type_stats,
    double build_hash_grid_ms,
    double build_kd_tree_ms,
    double build_cell_bin_grid_ms,
    double build_hash_histogram_ms,
    double build_cell_cache_ms,
    double build_cell_analysis_ms,
    const CellInfoCache& cell_cache,
    const std::vector<CellAnalysis>& cell_analysis,
    const std::vector<ReferenceRay>& rays,
    const std::vector<std::vector<std::vector<GatherResult>>>& all_results)
    // all_results[ray_idx][hit_idx][hierarchy_idx]
{
    std::ofstream jf(path);
    if (!jf.is_open()) {
        std::cerr << "[tool] Cannot write output: " << path << "\n";
        return;
    }
    jf << std::fixed << std::setprecision(6);

    jf << "{\n";

    // ── Config ───────────────────────────────────────────────────────
    jf << "  \"config\": {\n";
    jf << "    \"cache_path\": \"" << json_escape(cfg.cache_path) << "\",\n";
    jf << "    \"scene_path\": \"" << json_escape(cfg.scene_path) << "\",\n";
    jf << "    \"num_rays_requested\": " << cfg.num_rays << ",\n";
    jf << "    \"min_hits\": " << cfg.min_hits << ",\n";
    jf << "    \"gather_radius\": " << gather_radius << ",\n";
    jf << "    \"knn_k\": " << cfg.knn_k << ",\n";
    jf << "    \"tau\": " << cfg.tau << "\n";
    jf << "  },\n";

    // ── Photon map summary with type breakdown ───────────────────────
    jf << "  \"photon_map\": {\n";
    jf << "    \"num_photons\": " << photons.size() << ",\n";
    jf << "    \"global\": " << type_stats.global_count << ",\n";
    jf << "    \"caustic_glass\": " << type_stats.caustic_glass << ",\n";
    jf << "    \"caustic_specular\": " << type_stats.caustic_specular << ",\n";
    jf << "    \"traversed_glass\": " << type_stats.traversed_glass << ",\n";
    jf << "    \"volume_scatter\": " << type_stats.volume_scatter << ",\n";
    jf << "    \"dispersion\": " << type_stats.dispersion << "\n";
    jf << "  },\n";

    // ── Build times ──────────────────────────────────────────────────
    jf << "  \"build_times_ms\": {\n";
    jf << "    \"hash_grid\": " << std::setprecision(3) << build_hash_grid_ms << ",\n";
    jf << "    \"kd_tree\": " << build_kd_tree_ms << ",\n";
    jf << "    \"cell_bin_grid\": " << build_cell_bin_grid_ms << ",\n";
    jf << "    \"hash_histogram\": " << build_hash_histogram_ms << ",\n";
    jf << "    \"cell_info_cache\": " << build_cell_cache_ms << ",\n";
    jf << "    \"cell_analysis\": " << build_cell_analysis_ms << "\n";
    jf << "  },\n";

    jf << std::setprecision(6);

    // ── Aggregate statistics ─────────────────────────────────────────
    struct HierAgg {
        std::string name;
        double total_count_error = 0.0;
        double total_flux_error  = 0.0;
        int    comparisons       = 0;
        double avg_query_us      = 0.0;
        int    query_count       = 0;
    };
    HierAgg agg[5];
    agg[0].name = "brute_force";
    agg[1].name = "hash_grid";
    agg[2].name = "kd_tree";
    agg[3].name = "cell_bin_grid";
    agg[4].name = "hash_histogram";

    // §3 aggregate accumulators
    double sum_guide_fraction = 0, sum_flux_cv = 0, sum_dir_spread = 0;
    double sum_caustic_frac = 0, sum_concentration_cbg = 0, sum_concentration_hh = 0;
    double sum_position_spread = 0, sum_normal_coherence = 0;
    int    n_ca_hits = 0;    // hits with cell analysis data
    int    n_caustic_hot = 0;
    int    n_high_guide   = 0;  // guide_fraction > 0.5

    for (size_t ri = 0; ri < rays.size(); ++ri) {
        for (size_t hi = 0; hi < rays[ri].hits.size(); ++hi) {
            if (ri >= all_results.size() || hi >= all_results[ri].size()) continue;
            const auto& results = all_results[ri][hi];
            if (results.size() < 5) continue;

            const auto& bf = results[0];
            for (int h = 0; h < 5; ++h) {
                agg[h].avg_query_us += results[h].query_us;
                agg[h].query_count++;
            }

            for (int h = 1; h <= 2; ++h) {
                if (bf.photon_count > 0) {
                    agg[h].total_count_error += std::fabs(
                        (double)results[h].photon_count - bf.photon_count)
                        / bf.photon_count;
                    agg[h].total_flux_error += std::fabs(
                        (double)results[h].total_flux - bf.total_flux)
                        / std::max((double)bf.total_flux, 1e-12);
                    agg[h].comparisons++;
                }
            }

            // §3 cell analysis aggregate
            const auto& hit = rays[ri].hits[hi];
            uint32_t ck = cell_cache.cell_key(hit.position);
            if (ck < cell_analysis.size()) {
                const CellAnalysis& ca = cell_analysis[ck];
                const CellCacheInfo& ci = cell_cache.cells[ck];
                if (ca.has_photons) {
                    sum_guide_fraction += ca.guide_fraction;
                    sum_flux_cv        += ca.flux_cv;
                    sum_dir_spread     += ci.directional_spread;
                    sum_caustic_frac   += ca.caustic_fraction;
                    if (ci.is_caustic_hotspot) n_caustic_hot++;
                    if (ca.guide_fraction > 0.5f) n_high_guide++;
                    n_ca_hits++;
                }
            }

            // Spatial stats from brute force
            sum_position_spread += bf.position_spread;
            sum_normal_coherence += bf.normal_coherence;

            // Concentration from binned hierarchies
            sum_concentration_cbg += results[3].concentration;
            sum_concentration_hh  += results[4].concentration;
        }
    }

    jf << "  \"aggregate\": {\n";
    for (int h = 0; h < 5; ++h) {
        jf << "    \"" << agg[h].name << "\": {\n";
        jf << "      \"avg_query_us\": " << std::setprecision(2)
           << (agg[h].query_count > 0 ? agg[h].avg_query_us / agg[h].query_count : 0.0) << ",\n";
        jf << "      \"total_queries\": " << agg[h].query_count;
        if (h >= 1 && h <= 2 && agg[h].comparisons > 0) {
            jf << ",\n";
            jf << "      \"mean_count_error_pct\": " << std::setprecision(4)
               << (agg[h].total_count_error / agg[h].comparisons * 100.0) << ",\n";
            jf << "      \"mean_flux_error_pct\": "
               << (agg[h].total_flux_error / agg[h].comparisons * 100.0);
        }
        jf << "\n    },\n";
    }

    // §3 cell analysis summary
    int nq = agg[0].query_count > 0 ? agg[0].query_count : 1;
    jf << "    \"cell_analysis_summary\": {\n";
    jf << "      \"total_hits_analysed\": " << n_ca_hits << ",\n";
    jf << "      \"mean_guide_fraction\": " << std::setprecision(4)
       << (n_ca_hits > 0 ? sum_guide_fraction / n_ca_hits : 0.0) << ",\n";
    jf << "      \"mean_flux_cv\": "
       << (n_ca_hits > 0 ? sum_flux_cv / n_ca_hits : 0.0) << ",\n";
    jf << "      \"mean_directional_spread\": "
       << (n_ca_hits > 0 ? sum_dir_spread / n_ca_hits : 0.0) << ",\n";
    jf << "      \"mean_caustic_fraction\": "
       << (n_ca_hits > 0 ? sum_caustic_frac / n_ca_hits : 0.0) << ",\n";
    jf << "      \"mean_concentration_cbg\": "
       << (nq > 0 ? sum_concentration_cbg / nq : 0.0) << ",\n";
    jf << "      \"mean_concentration_hh\": "
       << (nq > 0 ? sum_concentration_hh / nq : 0.0) << ",\n";
    jf << "      \"mean_position_spread\": " << std::setprecision(6)
       << (nq > 0 ? sum_position_spread / nq : 0.0) << ",\n";
    jf << "      \"mean_normal_coherence\": " << std::setprecision(4)
       << (nq > 0 ? sum_normal_coherence / nq : 0.0) << ",\n";
    jf << "      \"caustic_hotspot_hits\": " << n_caustic_hot << ",\n";
    jf << "      \"high_guide_fraction_hits\": " << n_high_guide << "\n";
    jf << "    }\n";
    jf << "  },\n";

    jf << std::setprecision(6);

    // ── Per-ray details ──────────────────────────────────────────────
    jf << "  \"rays\": [\n";
    for (size_t ri = 0; ri < rays.size(); ++ri) {
        const auto& rr = rays[ri];
        jf << "    {\n";
        jf << "      \"ray_index\": " << rr.ray_index << ",\n";
        jf << "      \"pixel\": [" << rr.pixel_x << ", " << rr.pixel_y << "],\n";
        jf << "      \"num_hits\": " << rr.hits.size() << ",\n";

        jf << "      \"hits\": [\n";
        for (size_t hi = 0; hi < rr.hits.size(); ++hi) {
            const auto& h = rr.hits[hi];
            jf << "        {\n";
            jf << "          \"bounce\": " << h.bounce << ",\n";
            jf << "          \"position\": [" << h.position.x << ", "
               << h.position.y << ", " << h.position.z << "],\n";
            jf << "          \"normal\": [" << h.normal.x << ", "
               << h.normal.y << ", " << h.normal.z << "],\n";
            jf << "          \"material\": \"" << material_type_name(h.material_type) << "\",\n";
            jf << "          \"triangle_id\": " << h.triangle_id << ",\n";

            // ── §3 Cell Analysis ─────────────────────────────────────
            jf << "          \"cell_analysis\": {\n";
            uint32_t ck = cell_cache.cell_key(h.position);
            if (ck < cell_analysis.size() && cell_analysis[ck].has_photons) {
                const CellAnalysis& ca = cell_analysis[ck];
                const CellCacheInfo& ci = cell_cache.cells[ck];
                jf << "            \"photon_count\": " << ci.photon_count << ",\n";
                jf << "            \"irradiance\": " << ci.irradiance << ",\n";
                jf << "            \"flux_variance\": " << ci.flux_variance << ",\n";
                jf << "            \"flux_density\": " << ca.flux_density << ",\n";
                jf << "            \"flux_cv\": " << std::setprecision(4) << ca.flux_cv << ",\n";
                jf << "            \"directional_spread\": " << ci.directional_spread << ",\n";
                jf << "            \"normal_variance\": " << ci.normal_variance << ",\n";
                jf << "            \"caustic_count\": " << ci.caustic_count << ",\n";
                jf << "            \"caustic_flux\": " << std::setprecision(6) << ci.caustic_flux << ",\n";
                jf << "            \"caustic_cv\": " << std::setprecision(4) << ci.caustic_cv << ",\n";
                jf << "            \"caustic_fraction\": " << ca.caustic_fraction << ",\n";
                jf << "            \"is_caustic_hotspot\": " << (ci.is_caustic_hotspot ? "true" : "false") << ",\n";
                jf << "            \"glass_fraction\": " << ci.glass_fraction << ",\n";
                jf << "            \"adaptive_radius\": " << std::setprecision(6) << ci.adaptive_radius << ",\n";
                jf << "            \"dominant_emitter\": " << ci.dominant_emitter << ",\n";
                jf << "            \"active_bins\": " << ca.active_bins << ",\n";
                jf << "            \"guide_fraction\": " << std::setprecision(4) << ca.guide_fraction << "\n";
                jf << std::setprecision(6);
            } else {
                jf << "            \"has_photons\": false\n";
            }
            jf << "          },\n";

            // ── Gather results per hierarchy ─────────────────────────
            jf << "          \"gather\": {\n";
            if (ri < all_results.size() && hi < all_results[ri].size()) {
                const auto& results = all_results[ri][hi];

                // Photon-counting hierarchies (with spatial stats)
                const char* counting_names[] = {"brute_force", "hash_grid", "kd_tree"};
                for (int idx = 0; idx < 3 && idx < (int)results.size(); ++idx) {
                    const auto& gr = results[idx];
                    jf << "            \"" << counting_names[idx] << "\": {\n";
                    jf << "              \"photon_count\": " << gr.photon_count << ",\n";
                    jf << "              \"total_flux\": " << gr.total_flux << ",\n";
                    jf << "              \"min_dist\": " << std::sqrt(std::max(0.f, gr.min_dist2)) << ",\n";
                    jf << "              \"max_dist\": " << std::sqrt(std::max(0.f, gr.max_dist2)) << ",\n";
                    jf << "              \"centroid\": ["
                       << gr.centroid.x << ", " << gr.centroid.y << ", " << gr.centroid.z << "],\n";
                    jf << "              \"position_spread\": " << gr.position_spread << ",\n";
                    jf << "              \"normal_coherence\": " << std::setprecision(4) << gr.normal_coherence << ",\n";
                    jf << "              \"query_us\": " << std::setprecision(2) << gr.query_us;
                    jf << std::setprecision(6);
                    if (idx > 0 && results[0].photon_count > 0) {
                        float count_err = std::fabs((float)gr.photon_count - results[0].photon_count)
                                        / results[0].photon_count * 100.0f;
                        float flux_err  = results[0].total_flux > 0.0f
                                        ? std::fabs(gr.total_flux - results[0].total_flux)
                                          / results[0].total_flux * 100.0f
                                        : 0.0f;
                        jf << ",\n              \"count_error_pct\": " << std::setprecision(2) << count_err;
                        jf << ",\n              \"flux_error_pct\": " << flux_err;
                        jf << std::setprecision(6);
                    }
                    jf << "\n            },\n";
                }

                // Binned hierarchies (with per-bin breakdown)
                const char* binned_names[] = {"cell_bin_grid", "hash_histogram"};
                for (int idx = 3; idx < 5 && idx < (int)results.size(); ++idx) {
                    const auto& gr = results[idx];
                    jf << "            \"" << binned_names[idx - 3] << "\": {\n";
                    jf << "              \"occupied_bins\": " << gr.occupied_bins << ",\n";
                    jf << "              \"active_bins\": " << gr.active_bins << ",\n";
                    jf << "              \"binned_flux\": " << gr.binned_flux << ",\n";
                    jf << "              \"concentration\": " << std::setprecision(4) << gr.concentration << ",\n";
                    jf << "              \"query_us\": " << std::setprecision(2) << gr.query_us << ",\n";
                    jf << std::setprecision(6);

                    // Per-bin array (non-zero bins only)
                    jf << "              \"bins\": [\n";
                    for (size_t bi = 0; bi < gr.bin_details.size(); ++bi) {
                        const auto& bd = gr.bin_details[bi];
                        jf << "                {\"idx\": " << bd.idx
                           << ", \"flux\": " << bd.scalar_flux
                           << ", \"dir\": [" << std::setprecision(4)
                           << bd.dir_x << ", " << bd.dir_y << ", " << bd.dir_z << "]"
                           << ", \"avg_normal\": ["
                           << bd.avg_nx << ", " << bd.avg_ny << ", " << bd.avg_nz << "]";
                        if (bd.count > 0) jf << ", \"count\": " << bd.count;
                        jf << "}";
                        if (bi + 1 < gr.bin_details.size()) jf << ",";
                        jf << "\n";
                        jf << std::setprecision(6);
                    }
                    jf << "              ]\n";
                    jf << "            }";
                    if (idx < 4) jf << ",";
                    jf << "\n";
                }
            }
            jf << "          }\n";

            jf << "        }";
            if (hi + 1 < rr.hits.size()) jf << ",";
            jf << "\n";
        }
        jf << "      ]\n";

        jf << "    }";
        if (ri + 1 < rays.size()) jf << ",";
        jf << "\n";
    }
    jf << "  ]\n";

    jf << "}\n";
    std::cout << "[tool] Wrote report: " << path << "\n";
}

// ─────────────────────────────────────────────────────────────────────
// Argument parsing
// ─────────────────────────────────────────────────────────────────────
static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <cache.bin> <scene.obj>\n"
              << "  [--snapshot <snapshot.json>]  Load camera from snapshot\n"
              << "  [--radius <float>]           Override gather radius\n"
              << "  [--num-rays <int>]           Number of reference rays (default 100)\n"
              << "  [--min-hits <int>]           Min non-specular hits per ray (default 3)\n"
              << "  [--knn-k <int>]              k for kNN queries (default 64)\n"
              << "  [--tau <float>]              Surface filter tau (default 0.02)\n"
              << "  [--output <path.json>]       Output file path\n"
              << "  [--scenes-dir <path>]        Scenes root directory\n";
}

static bool parse_args(int argc, char* argv[], ToolConfig& cfg) {
    if (argc < 3) {
        print_usage(argv[0]);
        return false;
    }
    cfg.cache_path = argv[1];
    cfg.scene_path = argv[2];

    for (int i = 3; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--snapshot" && i + 1 < argc) {
            cfg.snapshot_path = argv[++i];
            cfg.cam_from_snapshot = true;
        } else if (arg == "--radius" && i + 1 < argc) {
            cfg.radius_override = std::stof(argv[++i]);
        } else if (arg == "--num-rays" && i + 1 < argc) {
            cfg.num_rays = std::stoi(argv[++i]);
        } else if (arg == "--min-hits" && i + 1 < argc) {
            cfg.min_hits = std::stoi(argv[++i]);
        } else if (arg == "--knn-k" && i + 1 < argc) {
            cfg.knn_k = std::stoi(argv[++i]);
        } else if (arg == "--tau" && i + 1 < argc) {
            cfg.tau = std::stof(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            cfg.output_path = argv[++i];
        } else if (arg == "--scenes-dir" && i + 1 < argc) {
            cfg.scenes_dir = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return false;
        }
    }

#ifdef SCENES_DIR
    if (cfg.scenes_dir.empty())
        cfg.scenes_dir = SCENES_DIR;
#endif
    return true;
}

// ─────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    ToolConfig cfg;
    if (!parse_args(argc, argv, cfg)) return 1;

    // ── Parse snapshot JSON if provided ──────────────────────────────
    if (cfg.cam_from_snapshot && !cfg.snapshot_path.empty()) {
        parse_json_camera(cfg.snapshot_path, cfg);
    }

    // ── Load photon cache ────────────────────────────────────────────
    std::cout << "======================================\n";
    std::cout << "  Photon Map Analysis Tool\n";
    std::cout << "======================================\n\n";

    std::cout << "[1/6] Loading photon cache: " << cfg.cache_path << "\n";
    PhotonSoA photons;
    HashGrid  hash_grid;
    float     gather_radius = 0.0f;

    if (!load_photon_cache(cfg.cache_path, photons, hash_grid, gather_radius)) {
        std::cerr << "[tool] FATAL: Failed to load photon cache\n";
        return 1;
    }

    if (cfg.radius_override > 0.0f) {
        std::cout << "[tool] Overriding gather_radius: " << gather_radius
                  << " -> " << cfg.radius_override << "\n";
        gather_radius = cfg.radius_override;
    }

    std::cout << "  Photons: " << photons.size()
              << "  Gather radius: " << gather_radius << "\n";

    // ── Initialize Fibonacci bin directions ──────────────────────────
    g_bin_dirs.init(PHOTON_BIN_COUNT);

    // ── Photon type statistics ───────────────────────────────────────
    PhotonTypeStats type_stats = compute_photon_type_stats(photons);
    std::printf("  Photon types: %zu global, %zu caustic_glass, %zu caustic_specular, "
                "%zu traversed_glass, %zu volume, %zu dispersion\n",
                type_stats.global_count, type_stats.caustic_glass,
                type_stats.caustic_specular, type_stats.traversed_glass,
                type_stats.volume_scatter, type_stats.dispersion);

    // ── Split into global + caustic for CellInfoCache ────────────────
    PhotonSoA global_photons, caustic_photons;
    split_by_path_flags(photons, global_photons, caustic_photons);
    std::cout << "\n";

    // ── Load scene ───────────────────────────────────────────────────
    std::string obj_path = cfg.scenes_dir.empty()
        ? cfg.scene_path
        : cfg.scenes_dir + "/" + cfg.scene_path;

    std::cout << "[2/6] Loading scene: " << obj_path << "\n";
    Scene scene;
    if (!load_obj(obj_path, scene)) {
        std::cerr << "[tool] FATAL: Failed to load scene: " << obj_path << "\n";
        return 1;
    }
    scene.normalize_to_reference();
    scene.build_bvh();
    std::cout << "  Triangles: " << scene.num_triangles()
              << "  Materials: " << scene.num_materials() << "\n\n";

    // ── Set up camera ────────────────────────────────────────────────
    Camera camera;
    camera.position = make_f3(cfg.cam_pos[0], cfg.cam_pos[1], cfg.cam_pos[2]);
    camera.look_at  = make_f3(cfg.cam_lookat[0], cfg.cam_lookat[1], cfg.cam_lookat[2]);
    camera.up       = make_f3(cfg.cam_up[0], cfg.cam_up[1], cfg.cam_up[2]);
    camera.fov_deg  = cfg.cam_fov;
    camera.width    = cfg.cam_width;
    camera.height   = cfg.cam_height;
    camera.update();

    std::cout << "[3/6] Tracing " << cfg.num_rays << " reference rays (min "
              << cfg.min_hits << " non-specular hits)...\n";
    auto rays = trace_reference_rays(scene, camera, cfg.num_rays, cfg.min_hits);

    if (rays.empty()) {
        std::cerr << "[tool] FATAL: No valid reference rays found.\n"
                  << "  Try increasing --num-rays or decreasing --min-hits.\n";
        return 1;
    }

    // Count total hit points
    size_t total_hits = 0;
    for (const auto& r : rays) total_hits += r.hits.size();
    std::cout << "  Valid rays: " << rays.size()
              << "  Total hit points: " << total_hits << "\n\n";

    // ── Build spatial hierarchies ────────────────────────────────────
    std::cout << "[4/6] Building spatial hierarchies...\n";

    // HashGrid was already loaded from cache, but rebuild for timing
    HashGrid fresh_grid;
    auto t0 = std::chrono::high_resolution_clock::now();
    fresh_grid.build(photons, gather_radius);
    auto t1 = std::chrono::high_resolution_clock::now();
    double build_hash_grid_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  HashGrid:       " << std::setprecision(1) << std::fixed
              << build_hash_grid_ms << " ms\n";

    KDTree kd_tree;
    t0 = std::chrono::high_resolution_clock::now();
    kd_tree.build(photons);
    t1 = std::chrono::high_resolution_clock::now();
    double build_kd_tree_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  KDTree:         " << build_kd_tree_ms << " ms\n";

    CellBinGrid cell_bin_grid;
    t0 = std::chrono::high_resolution_clock::now();
    cell_bin_grid.build(photons, gather_radius, PHOTON_BIN_COUNT);
    t1 = std::chrono::high_resolution_clock::now();
    double build_cell_bin_grid_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  CellBinGrid:    " << build_cell_bin_grid_ms << " ms\n";

    HashHistogram hash_hist;
    t0 = std::chrono::high_resolution_clock::now();
    hash_hist.build(photons, gather_radius, PHOTON_BIN_COUNT, 4);  // 4 levels for multi-res analysis
    t1 = std::chrono::high_resolution_clock::now();
    double build_hash_histogram_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  HashHistogram:  " << build_hash_histogram_ms << " ms"
              << "  (" << hash_hist.num_levels << " levels)\n";

    // CellInfoCache (§3 cell metrics)
    float cache_cell_size = gather_radius * GUIDE_LEVEL_SCALES[0];  // 2× gather_radius
    float cell_area = cache_cell_size * cache_cell_size;
    CellInfoCache cell_cache;
    t0 = std::chrono::high_resolution_clock::now();
    cell_cache.build(global_photons, caustic_photons, cache_cell_size, gather_radius);
    t1 = std::chrono::high_resolution_clock::now();
    double build_cell_cache_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  CellInfoCache:  " << build_cell_cache_ms << " ms\n";

    // CellAnalysis (§3.6 conclusions)
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<CellAnalysis> cell_analysis = build_cell_analysis(cell_cache, hash_hist, cell_area);
    t1 = std::chrono::high_resolution_clock::now();
    double build_cell_analysis_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  CellAnalysis:   " << build_cell_analysis_ms << " ms\n\n";

    // ── Query all hierarchies at every hit point ─────────────────────
    std::cout << "[5/6] Querying " << total_hits << " hit points across 5 hierarchies...\n";

    // all_results[ray_idx][hit_idx][hierarchy_idx]
    std::vector<std::vector<std::vector<GatherResult>>> all_results;
    all_results.resize(rays.size());

    for (size_t ri = 0; ri < rays.size(); ++ri) {
        all_results[ri].resize(rays[ri].hits.size());

        for (size_t hi = 0; hi < rays[ri].hits.size(); ++hi) {
            const auto& h = rays[ri].hits[hi];
            auto& results = all_results[ri][hi];
            results.resize(5);

            // 0: Brute force (ground truth)
            results[0] = brute_force_gather(photons, h.position, h.normal,
                                            gather_radius, cfg.tau);

            // 1: HashGrid
            results[1] = hash_grid_gather(fresh_grid, photons, h.position,
                                          h.normal, gather_radius, cfg.tau);

            // 2: KDTree
            results[2] = kd_tree_gather(kd_tree, photons, h.position,
                                        h.normal, gather_radius, cfg.tau);

            // 3: CellBinGrid (binned)
            results[3] = cell_bin_grid_gather(cell_bin_grid, h.position);

            // 4: HashHistogram (binned)
            results[4] = hash_histogram_gather(hash_hist, h.position, gather_radius);
        }
    }

    // Print quick summary
    double bf_total_us = 0, hg_total_us = 0, kd_total_us = 0;
    double cbg_total_us = 0, hh_total_us = 0;
    for (const auto& ray_results : all_results) {
        for (const auto& hit_results : ray_results) {
            bf_total_us  += hit_results[0].query_us;
            hg_total_us  += hit_results[1].query_us;
            kd_total_us  += hit_results[2].query_us;
            cbg_total_us += hit_results[3].query_us;
            hh_total_us  += hit_results[4].query_us;
        }
    }
    double n = (double)total_hits;
    std::cout << "  Avg query time (us/hit):  BF=" << std::setprecision(1)
              << bf_total_us / n << "  HG=" << hg_total_us / n
              << "  KD=" << kd_total_us / n
              << "  CBG=" << cbg_total_us / n
              << "  HH=" << hh_total_us / n << "\n\n";

    // ── Write JSON report ────────────────────────────────────────────
    std::cout << "[6/6] Writing report: " << cfg.output_path << "\n";
    write_json_report(cfg.output_path, cfg, photons, gather_radius,
                      type_stats,
                      build_hash_grid_ms, build_kd_tree_ms,
                      build_cell_bin_grid_ms, build_hash_histogram_ms,
                      build_cell_cache_ms, build_cell_analysis_ms,
                      cell_cache, cell_analysis,
                      rays, all_results);

    std::cout << "\nDone.\n";
    return 0;
}
