// ─────────────────────────────────────────────────────────────────────
// photon_map_analysis – Nearest-photon vs random-in-cell comparison
// ─────────────────────────────────────────────────────────────────────
// Purpose: measure the error of picking a random photon from a hash
// grid cell vs the ideal nearest photon (brute-force O(N) scan).
// Tests three grid resolutions: 100³, 1000³, 10000³.
//
// Usage:
//   photon_map_analysis <cache.bin> <scene.obj>
//       [--snapshot <snapshot.json>]
//       [--num-rays <int>]     (default 100)
//       [--min-hits <int>]     (default 3)
// ─────────────────────────────────────────────────────────────────────
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "photon/photon_io.h"
#include "photon/hash_grid.h"
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
#include <random>
#include <numeric>

// ─────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────
struct ToolConfig {
    std::string cache_path;
    std::string scene_path;
    std::string scenes_dir;
    std::string snapshot_path;
    std::string output_path;    // ignored (kept for renderer compat)

    int   num_rays = 100;
    int   min_hits = 3;
    float radius_override = 0.0f;  // --radius override (0 = use cache value)

    float cam_pos[3]    = {0.f, 0.f, 0.f};
    float cam_lookat[3] = {0.f, 0.f, -1.f};
    float cam_up[3]     = {0.f, 1.f, 0.f};
    float cam_fov       = 40.0f;
    int   cam_width     = 1024;
    int   cam_height    = 768;
    bool  cam_from_snapshot = false;
};

// ─────────────────────────────────────────────────────────────────────
// Hit-point data
// ─────────────────────────────────────────────────────────────────────
struct ReferenceHit {
    float3   position;
    float3   normal;
    uint32_t triangle_id;
    uint32_t material_id;
    uint8_t  material_type;
    int      bounce;
};

struct ReferenceRay {
    int ray_index;
    int pixel_x, pixel_y;
    std::vector<ReferenceHit> hits;
};

// ─────────────────────────────────────────────────────────────────────
// Result of a single nearest / random lookup
// ─────────────────────────────────────────────────────────────────────
struct PhotonResult {
    bool  found    = false;
    int   index    = -1;
    float distance = 1e30f;   // 3D Euclidean distance to hit point
    float3 wi      = {0,0,0}; // incoming direction of the photon
    float flux     = 0.f;     // total scalar flux of the photon
};

// ─────────────────────────────────────────────────────────────────────
// Snapshot JSON camera parser (minimal)
// ─────────────────────────────────────────────────────────────────────
static bool parse_json_camera(const std::string& json_path, ToolConfig& cfg) {
    std::ifstream f(json_path);
    if (!f.is_open()) {
        std::cerr << "[tool] Cannot open snapshot JSON: " << json_path << "\n";
        return false;
    }
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

    auto extract_array3 = [&](const char* key, float out[3]) -> bool {
        auto pos = content.find(std::string("\"") + key + "\"");
        if (pos == std::string::npos) return false;
        auto bracket = content.find('[', pos);
        if (bracket == std::string::npos) return false;
        if (sscanf(content.c_str() + bracket, "[%f, %f, %f]",
                   &out[0], &out[1], &out[2]) == 3)
            return true;
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
    extract_array3("up", cfg.cam_up);
    extract_float("fov_deg", cfg.cam_fov);
    extract_int("width",  cfg.cam_width);
    extract_int("height", cfg.cam_height);

    std::string obj;
    if (extract_string("obj_path", obj) && !obj.empty())
        cfg.scene_path = obj;

    if (!ok)
        std::cerr << "[tool] Warning: partial camera parse from " << json_path << "\n";
    return true;
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
                0.0f, 0.0f, 0.0f);

            ReferenceRay rr;
            rr.ray_index = ray_idx++;
            rr.pixel_x = px;
            rr.pixel_y = py;

            for (int bounce = 0; bounce < max_bounces; ++bounce) {
                HitRecord hit = scene.intersect(ray);
                if (!hit.hit) break;

                const Material& mat = scene.materials[hit.material_id];
                if (mat.is_emissive()) break;

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

                float3 new_dir;
                if (mat.type == MaterialType::Mirror) {
                    new_dir = ray.direction - hit.normal * (2.0f * dot(ray.direction, hit.normal));
                } else if (mat.type == MaterialType::Glass || mat.type == MaterialType::Translucent) {
                    float3 n = hit.normal;
                    float cos_i = -dot(ray.direction, n);
                    float eta = 1.0f / mat.ior;
                    if (cos_i < 0.f) { n = -n; cos_i = -cos_i; eta = mat.ior; }
                    float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
                    if (sin2_t > 1.0f) {
                        new_dir = ray.direction - n * (2.0f * dot(ray.direction, n));
                    } else {
                        float cos_t = std::sqrt(1.0f - sin2_t);
                        new_dir = ray.direction * eta + n * (eta * cos_i - cos_t);
                    }
                } else {
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

            if ((int)rr.hits.size() >= min_hits)
                result.push_back(std::move(rr));
        }
    }

    std::printf("  Traced %d camera rays, kept %zu with >= %d non-specular hits\n",
                attempts, result.size(), min_hits);
    return result;
}

// ─────────────────────────────────────────────────────────────────────
// Brute-force: find the single nearest photon to a point
// ─────────────────────────────────────────────────────────────────────
static PhotonResult find_nearest_photon(
    const PhotonSoA& photons, float3 pos)
{
    PhotonResult best;
    float best_d2 = 1e30f;
    size_t n = photons.size();

    for (size_t i = 0; i < n; ++i) {
        float dx = photons.pos_x[i] - pos.x;
        float dy = photons.pos_y[i] - pos.y;
        float dz = photons.pos_z[i] - pos.z;
        float d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < best_d2) {
            best_d2 = d2;
            best.index    = (int)i;
            best.distance = std::sqrt(d2);
            best.wi       = make_f3(photons.wi_x[i], photons.wi_y[i], photons.wi_z[i]);
            best.flux     = photons.total_flux(i);
            best.found    = true;
        }
    }
    return best;
}

// ─────────────────────────────────────────────────────────────────────
// Hash grid: pick a random photon from the cell containing pos
// Filters by actual cell coordinates to avoid hash-collision contamination.
// ─────────────────────────────────────────────────────────────────────
static PhotonResult random_photon_in_cell(
    const HashGrid& grid, const PhotonSoA& photons,
    float3 pos, std::mt19937& rng)
{
    PhotonResult res;
    uint32_t key = grid.cell_key(pos);
    if (grid.cell_start[key] == 0xFFFFFFFFu) return res; // empty bucket

    uint32_t start = grid.cell_start[key];
    uint32_t end   = grid.cell_end[key];
    if (start >= end) return res;

    // The query position's true cell coordinate
    int3 query_cell = grid.cell_coord(pos);

    // Collect indices of photons that are actually in the same spatial cell
    // (not just the same hash bucket — hash collisions are rampant at fine res)
    thread_local std::vector<uint32_t> candidates;
    candidates.clear();

    for (uint32_t s = start; s < end; ++s) {
        uint32_t idx = grid.sorted_indices[s];
        float3 ppos = make_f3(photons.pos_x[idx], photons.pos_y[idx], photons.pos_z[idx]);
        int3 pcell = grid.cell_coord(ppos);
        if (pcell.x == query_cell.x && pcell.y == query_cell.y && pcell.z == query_cell.z) {
            candidates.push_back(idx);
        }
    }

    if (candidates.empty()) return res; // hash collision only — no true matches

    // Pick a random photon from the true-cell candidates
    std::uniform_int_distribution<uint32_t> dist(0, (uint32_t)candidates.size() - 1);
    uint32_t idx = candidates[dist(rng)];

    float3 ppos = make_f3(photons.pos_x[idx], photons.pos_y[idx], photons.pos_z[idx]);
    float3 diff = pos - ppos;

    res.found    = true;
    res.index    = (int)idx;
    res.distance = std::sqrt(dot(diff, diff));
    res.wi       = make_f3(photons.wi_x[idx], photons.wi_y[idx], photons.wi_z[idx]);
    res.flux     = photons.total_flux(idx);
    return res;
}

// ─────────────────────────────────────────────────────────────────────
// Compute scene AABB from photon positions
// ─────────────────────────────────────────────────────────────────────
struct SceneAABB {
    float3 lo, hi;
    float extent() const {
        float dx = hi.x - lo.x;
        float dy = hi.y - lo.y;
        float dz = hi.z - lo.z;
        return std::max({dx, dy, dz});
    }
};

static SceneAABB compute_photon_aabb(const PhotonSoA& photons) {
    SceneAABB box;
    box.lo = make_f3( 1e30f,  1e30f,  1e30f);
    box.hi = make_f3(-1e30f, -1e30f, -1e30f);
    size_t n = photons.size();
    for (size_t i = 0; i < n; ++i) {
        float x = photons.pos_x[i], y = photons.pos_y[i], z = photons.pos_z[i];
        if (x < box.lo.x) box.lo.x = x;
        if (y < box.lo.y) box.lo.y = y;
        if (z < box.lo.z) box.lo.z = z;
        if (x > box.hi.x) box.hi.x = x;
        if (y > box.hi.y) box.hi.y = y;
        if (z > box.hi.z) box.hi.z = z;
    }
    return box;
}

// ─────────────────────────────────────────────────────────────────────
// Print a percentile row
// ─────────────────────────────────────────────────────────────────────
static void print_percentile_row(
    const char* label, std::vector<double>& vals,
    const char* unit, int label_width = 28)
{
    if (vals.empty()) {
        std::printf("  %-*s  (no data)\n", label_width, label);
        return;
    }
    std::sort(vals.begin(), vals.end());
    auto pctl = [&](double p) -> double {
        double k = (p / 100.0) * (vals.size() - 1);
        size_t lo = (size_t)k;
        size_t hi = std::min(lo + 1, vals.size() - 1);
        double frac = k - lo;
        return vals[lo] + frac * (vals[hi] - vals[lo]);
    };
    std::printf("  %-*s  P50=%-10.4f P90=%-10.4f P99=%-10.4f max=%-10.4f %s\n",
                label_width, label,
                pctl(50), pctl(90), pctl(99), pctl(100), unit);
}

// ─────────────────────────────────────────────────────────────────────
// Cell occupancy statistics for a grid
// ─────────────────────────────────────────────────────────────────────
struct CellStats {
    size_t total_occupied  = 0;
    size_t total_empty     = 0;
    double mean_per_cell   = 0.0;
    double median_per_cell = 0.0;
    double max_per_cell    = 0.0;
};

static CellStats compute_cell_stats(const HashGrid& grid) {
    CellStats cs;
    std::vector<uint32_t> counts;
    counts.reserve(grid.table_size / 4);

    for (uint32_t k = 0; k < grid.table_size; ++k) {
        if (grid.cell_start[k] == 0xFFFFFFFFu) {
            cs.total_empty++;
            continue;
        }
        uint32_t c = grid.cell_end[k] - grid.cell_start[k];
        if (c > 0) {
            counts.push_back(c);
            cs.total_occupied++;
        }
    }

    if (!counts.empty()) {
        double sum = 0;
        for (auto c : counts) sum += c;
        cs.mean_per_cell = sum / counts.size();

        std::sort(counts.begin(), counts.end());
        cs.median_per_cell = counts[counts.size() / 2];
        cs.max_per_cell    = counts.back();
    }
    return cs;
}

// ─────────────────────────────────────────────────────────────────────
// Run comparison for one grid resolution
// ─────────────────────────────────────────────────────────────────────
struct ResolutionResult {
    int    resolution;
    float  cell_size;
    double build_ms;

    // Per-hit error data (only for hits where both found a photon)
    std::vector<double> dist_nearest;
    std::vector<double> dist_random;
    std::vector<double> dist_error_abs;   // |random - nearest|
    std::vector<double> dist_ratio;       // random / nearest
    std::vector<double> angle_error_deg;  // angle between wi_nearest and wi_random
    std::vector<double> flux_error_pct;   // |flux_r - flux_n| / flux_n * 100

    // Coverage
    size_t total_hits             = 0;
    size_t hits_with_photon       = 0;  // true-cell match found
    size_t hits_bucket_nonempty   = 0;  // hash bucket was non-empty (may be collision)
    size_t hits_collision_only    = 0;  // bucket non-empty but all entries were hash collisions

    CellStats cell_stats;
};

static ResolutionResult run_resolution(
    int resolution, float scene_extent,
    const PhotonSoA& photons,
    const std::vector<ReferenceRay>& rays,
    const std::vector<std::vector<PhotonResult>>& bf_nearest)
{
    ResolutionResult rr;
    rr.resolution = resolution;
    rr.cell_size  = scene_extent / (float)resolution;

    // Build hash grid with this cell size
    // HashGrid.build() sets cell_size = radius * 2, so pass radius = cell_size / 2
    HashGrid grid;
    auto t0 = std::chrono::high_resolution_clock::now();
    grid.build(photons, rr.cell_size / 2.0f);
    auto t1 = std::chrono::high_resolution_clock::now();
    rr.build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    rr.cell_stats = compute_cell_stats(grid);

    // Query each hit point
    for (size_t ri = 0; ri < rays.size(); ++ri) {
        for (size_t hi = 0; hi < rays[ri].hits.size(); ++hi) {
            const auto& h = rays[ri].hits[hi];
            const auto& bf = bf_nearest[ri][hi];

            rr.total_hits++;

            // Check if the hash bucket is non-empty (before cell-coord filtering)
            uint32_t key = grid.cell_key(h.position);
            bool bucket_nonempty = (grid.cell_start[key] != 0xFFFFFFFFu &&
                                    grid.cell_end[key] > grid.cell_start[key]);
            if (bucket_nonempty) rr.hits_bucket_nonempty++;

            // Deterministic RNG per hit for reproducibility
            std::mt19937 rng((uint32_t)(ri * 10000 + hi));
            PhotonResult random = random_photon_in_cell(grid, photons, h.position, rng);

            if (random.found) {
                rr.hits_with_photon++;

                if (bf.found && bf.distance > 0.f) {
                    rr.dist_nearest.push_back(bf.distance);
                    rr.dist_random.push_back(random.distance);
                    rr.dist_error_abs.push_back(std::fabs(random.distance - bf.distance));
                    rr.dist_ratio.push_back(random.distance / bf.distance);

                    // Angle between incoming directions
                    float d = dot(bf.wi, random.wi);
                    d = std::max(-1.f, std::min(1.f, d));
                    double angle = std::acos((double)d) * 180.0 / 3.14159265358979;
                    rr.angle_error_deg.push_back(angle);

                    // Flux error
                    if (bf.flux > 0.f) {
                        double fe = std::fabs((double)random.flux - bf.flux) / bf.flux * 100.0;
                        rr.flux_error_pct.push_back(fe);
                    }
                }
            } else if (bucket_nonempty) {
                rr.hits_collision_only++;
            }
        }
    }

    return rr;
}

// ─────────────────────────────────────────────────────────────────────
// Print results for one resolution
// ─────────────────────────────────────────────────────────────────────
static void print_resolution(const ResolutionResult& rr) {
    std::printf("\n");
    std::printf("  ================================================================\n");
    std::printf("  Grid %d^3  (cell_size = %.6f m)   Build: %.1f ms\n",
                rr.resolution, rr.cell_size, rr.build_ms);
    std::printf("  ================================================================\n");

    auto& cs = rr.cell_stats;
    std::printf("  Cells: %zu occupied, %zu empty (%.1f%% empty)\n",
                cs.total_occupied, cs.total_empty,
                (cs.total_occupied + cs.total_empty) > 0
                    ? 100.0 * cs.total_empty / (cs.total_occupied + cs.total_empty)
                    : 0.0);
    std::printf("  Photons/cell: mean=%.1f  median=%.0f  max=%.0f\n",
                cs.mean_per_cell, cs.median_per_cell, cs.max_per_cell);
    std::printf("  Hit coverage: %zu / %zu (%.1f%%) have >= 1 photon in true cell\n",
                rr.hits_with_photon, rr.total_hits,
                rr.total_hits > 0 ? 100.0 * rr.hits_with_photon / rr.total_hits : 0.0);
    if (rr.hits_bucket_nonempty > 0) {
        std::printf("  Hash collisions: %zu bucket-nonempty but %zu had no true-cell match (%.1f%% collision rate)\n",
                    rr.hits_bucket_nonempty, rr.hits_collision_only,
                    100.0 * rr.hits_collision_only / rr.hits_bucket_nonempty);
    }
    std::printf("\n");

    // Make mutable copies for percentile sorting
    auto dist_err  = rr.dist_error_abs;
    auto dist_rat  = rr.dist_ratio;
    auto angle_err = rr.angle_error_deg;
    auto flux_err  = rr.flux_error_pct;
    auto dist_n    = rr.dist_nearest;
    auto dist_r    = rr.dist_random;

    print_percentile_row("Distance nearest (m)",  dist_n,    "m");
    print_percentile_row("Distance random (m)",   dist_r,    "m");
    print_percentile_row("Distance error |r-n|",  dist_err,  "m");
    print_percentile_row("Distance ratio r/n",    dist_rat,  "x");
    print_percentile_row("Direction error",       angle_err, "deg");
    print_percentile_row("Flux error",            flux_err,  "%");
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
              << "  [--output <path>]            (ignored, for renderer compatibility)\n";
}

static bool parse_args(int argc, char* argv[], ToolConfig& cfg) {
    if (argc < 3) { print_usage(argv[0]); return false; }

    cfg.cache_path = argv[1];
    cfg.scene_path = argv[2];

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--snapshot" && i + 1 < argc) {
            cfg.snapshot_path = argv[++i];
            cfg.cam_from_snapshot = true;
        } else if (arg == "--num-rays" && i + 1 < argc) {
            cfg.num_rays = std::atoi(argv[++i]);
        } else if (arg == "--min-hits" && i + 1 < argc) {
            cfg.min_hits = std::atoi(argv[++i]);
        } else if (arg == "--radius" && i + 1 < argc) {
            cfg.radius_override = (float)std::atof(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            cfg.output_path = argv[++i]; // accepted but ignored
        } else {
            std::cerr << "[tool] Unknown argument: " << arg << "\n";
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

    if (cfg.cam_from_snapshot && !cfg.snapshot_path.empty())
        parse_json_camera(cfg.snapshot_path, cfg);

    std::printf("===========================================================\n");
    std::printf("  Photon Map Analysis: Nearest vs Random-in-Cell\n");
    std::printf("===========================================================\n\n");

    // ── [1/4] Load photon cache ──────────────────────────────────────
    std::printf("[1/4] Loading photon cache: %s\n", cfg.cache_path.c_str());
    PhotonSoA photons;
    HashGrid  loaded_grid;
    float     gather_radius = 0.0f;

    if (!load_photon_cache(cfg.cache_path, photons, loaded_grid, gather_radius)) {
        std::cerr << "[tool] FATAL: Failed to load photon cache\n";
        return 1;
    }
    if (cfg.radius_override > 0.0f) {
        std::printf("  Overriding gather_radius: %.6f -> %.6f\n",
                    gather_radius, cfg.radius_override);
        gather_radius = cfg.radius_override;
    }
    std::printf("  Photons: %zu   Gather radius: %.6f\n\n",
                photons.size(), gather_radius);

    // ── Compute scene AABB ───────────────────────────────────────────
    SceneAABB aabb = compute_photon_aabb(photons);
    float extent = aabb.extent();
    std::printf("  Scene AABB: (%.4f, %.4f, %.4f) - (%.4f, %.4f, %.4f)\n",
                aabb.lo.x, aabb.lo.y, aabb.lo.z,
                aabb.hi.x, aabb.hi.y, aabb.hi.z);
    std::printf("  Max extent: %.4f m\n\n", extent);

    // ── [2/4] Load scene ─────────────────────────────────────────────
    std::string obj_path = cfg.scenes_dir.empty()
        ? cfg.scene_path
        : cfg.scenes_dir + "/" + cfg.scene_path;

    std::printf("[2/4] Loading scene: %s\n", obj_path.c_str());
    Scene scene;
    if (!load_obj(obj_path, scene)) {
        std::cerr << "[tool] FATAL: Failed to load scene: " << obj_path << "\n";
        return 1;
    }
    scene.normalize_to_reference();
    scene.build_bvh();

    // Diagnostic: compare scene triangle AABB with photon AABB
    {
        float3 tri_lo = make_f3(1e30f, 1e30f, 1e30f);
        float3 tri_hi = make_f3(-1e30f, -1e30f, -1e30f);
        for (size_t i = 0; i < scene.triangles.size(); ++i) {
            auto update = [&](float3 v) {
                if (v.x < tri_lo.x) tri_lo.x = v.x;
                if (v.y < tri_lo.y) tri_lo.y = v.y;
                if (v.z < tri_lo.z) tri_lo.z = v.z;
                if (v.x > tri_hi.x) tri_hi.x = v.x;
                if (v.y > tri_hi.y) tri_hi.y = v.y;
                if (v.z > tri_hi.z) tri_hi.z = v.z;
            };
            update(scene.triangles[i].v0);
            update(scene.triangles[i].v1);
            update(scene.triangles[i].v2);
        }
        std::printf("  Scene tri AABB (after norm): (%.4f,%.4f,%.4f)-(%.4f,%.4f,%.4f)\n",
                    tri_lo.x, tri_lo.y, tri_lo.z, tri_hi.x, tri_hi.y, tri_hi.z);
        std::printf("  Photon AABB:                 (%.4f,%.4f,%.4f)-(%.4f,%.4f,%.4f)\n\n",
                    aabb.lo.x, aabb.lo.y, aabb.lo.z, aabb.hi.x, aabb.hi.y, aabb.hi.z);
    }
    std::printf("  Triangles: %zu   Materials: %zu\n\n",
                scene.num_triangles(), scene.num_materials());

    // ── Set up camera ────────────────────────────────────────────────
    Camera camera;
    camera.position = make_f3(cfg.cam_pos[0], cfg.cam_pos[1], cfg.cam_pos[2]);
    camera.look_at  = make_f3(cfg.cam_lookat[0], cfg.cam_lookat[1], cfg.cam_lookat[2]);
    camera.up       = make_f3(cfg.cam_up[0], cfg.cam_up[1], cfg.cam_up[2]);
    camera.fov_deg  = cfg.cam_fov;
    camera.width    = cfg.cam_width;
    camera.height   = cfg.cam_height;
    camera.update();

    std::printf("  Camera: pos=(%.4f, %.4f, %.4f)  lookat=(%.4f, %.4f, %.4f)\n",
                camera.position.x, camera.position.y, camera.position.z,
                camera.look_at.x, camera.look_at.y, camera.look_at.z);
    std::printf("          fov=%.1f deg  %dx%d\n\n",
                camera.fov_deg, camera.width, camera.height);

    // ── [3/4] Trace reference rays ───────────────────────────────────
    std::printf("[3/4] Tracing %d reference rays (min %d non-specular hits)...\n",
                cfg.num_rays, cfg.min_hits);
    auto rays = trace_reference_rays(scene, camera, cfg.num_rays, cfg.min_hits);

    if (rays.empty()) {
        std::cerr << "[tool] FATAL: No valid reference rays found.\n";
        return 1;
    }

    size_t total_hits = 0;
    for (const auto& r : rays) total_hits += r.hits.size();
    std::printf("  Valid rays: %zu   Total hit points: %zu\n\n", rays.size(), total_hits);

    // ── Diagnostic: compare hitpoint positions vs photon AABB ────────
    {
        float3 hit_lo = make_f3( 1e30f,  1e30f,  1e30f);
        float3 hit_hi = make_f3(-1e30f, -1e30f, -1e30f);
        for (const auto& r : rays) {
            for (const auto& h : r.hits) {
                if (h.position.x < hit_lo.x) hit_lo.x = h.position.x;
                if (h.position.y < hit_lo.y) hit_lo.y = h.position.y;
                if (h.position.z < hit_lo.z) hit_lo.z = h.position.z;
                if (h.position.x > hit_hi.x) hit_hi.x = h.position.x;
                if (h.position.y > hit_hi.y) hit_hi.y = h.position.y;
                if (h.position.z > hit_hi.z) hit_hi.z = h.position.z;
            }
        }
        std::printf("  Hitpoint AABB: (%.4f, %.4f, %.4f) - (%.4f, %.4f, %.4f)\n",
                    hit_lo.x, hit_lo.y, hit_lo.z, hit_hi.x, hit_hi.y, hit_hi.z);
        std::printf("  Photon  AABB:  (%.4f, %.4f, %.4f) - (%.4f, %.4f, %.4f)\n",
                    aabb.lo.x, aabb.lo.y, aabb.lo.z, aabb.hi.x, aabb.hi.y, aabb.hi.z);

        // Print first 5 hitpoints with their nearest photon distance
        std::printf("\n  Sample hitpoints:\n");
        int shown = 0;
        for (const auto& r : rays) {
            for (const auto& h : r.hits) {
                if (shown >= 5) break;
                std::printf("    hit[%d] pos=(%.5f, %.5f, %.5f)\n",
                            shown, h.position.x, h.position.y, h.position.z);
                ++shown;
            }
            if (shown >= 5) break;
        }
        std::printf("\n");
    }

    // ── Brute-force nearest photon for every hit (ground truth) ──────
    std::printf("  Finding nearest photon (brute force O(N)) for %zu hits...\n", total_hits);
    auto bf_t0 = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<PhotonResult>> bf_nearest(rays.size());
    for (size_t ri = 0; ri < rays.size(); ++ri) {
        bf_nearest[ri].resize(rays[ri].hits.size());
        for (size_t hi = 0; hi < rays[ri].hits.size(); ++hi) {
            bf_nearest[ri][hi] = find_nearest_photon(photons, rays[ri].hits[hi].position);
        }
    }

    auto bf_t1 = std::chrono::high_resolution_clock::now();
    double bf_ms = std::chrono::duration<double, std::milli>(bf_t1 - bf_t0).count();
    std::printf("  Brute-force done in %.1f ms (%.1f ms/hit)\n", bf_ms, bf_ms / total_hits);

    // Diagnostic: nearest-photon distance distribution
    {
        std::vector<float> nn_dists;
        for (const auto& rv : bf_nearest)
            for (const auto& p : rv)
                if (p.found) nn_dists.push_back(p.distance);
        std::sort(nn_dists.begin(), nn_dists.end());
        if (!nn_dists.empty()) {
            auto pctl = [&](double pct) {
                size_t idx = (size_t)((pct / 100.0) * (nn_dists.size() - 1));
                return nn_dists[std::min(idx, nn_dists.size() - 1)];
            };
            std::printf("  Nearest-photon distance: P50=%.6f  P90=%.6f  P99=%.6f  max=%.6f m\n",
                        pctl(50), pctl(90), pctl(99), pctl(100));
        }
        // Show a few specific hitpoints with their nearest photon
        int shown = 0;
        for (size_t ri = 0; ri < rays.size() && shown < 5; ++ri) {
            for (size_t hi = 0; hi < rays[ri].hits.size() && shown < 5; ++hi) {
                const auto& h = rays[ri].hits[hi];
                const auto& p = bf_nearest[ri][hi];
                std::printf("    hit(%zu,%zu) pos=(%.4f,%.4f,%.4f) -> nearest photon dist=%.6f at (%.4f,%.4f,%.4f)\n",
                            ri, hi, h.position.x, h.position.y, h.position.z,
                            p.distance,
                            photons.pos_x[p.index], photons.pos_y[p.index], photons.pos_z[p.index]);
                ++shown;
            }
        }
    }
    // ── Filter hitpoints: only keep those within gather_radius of a photon ──
    // Hitpoints far from any photon are in unlit regions and would never be
    // used in the renderer’s photon density estimation.  Including them in the
    // comparison skews coverage to near-zero and hides the actual error signal.
    float max_dist = gather_radius;  // same radius the GPU uses
    size_t filtered_out = 0;
    std::vector<ReferenceRay> filtered_rays;
    std::vector<std::vector<PhotonResult>> filtered_bf;
    filtered_rays.reserve(rays.size());
    filtered_bf.reserve(rays.size());

    for (size_t ri = 0; ri < rays.size(); ++ri) {
        ReferenceRay fr;
        fr.ray_index = rays[ri].ray_index;
        fr.pixel_x   = rays[ri].pixel_x;
        fr.pixel_y   = rays[ri].pixel_y;
        std::vector<PhotonResult> fp;

        for (size_t hi = 0; hi < rays[ri].hits.size(); ++hi) {
            if (bf_nearest[ri][hi].found && bf_nearest[ri][hi].distance <= max_dist) {
                fr.hits.push_back(rays[ri].hits[hi]);
                fp.push_back(bf_nearest[ri][hi]);
            } else {
                filtered_out++;
            }
        }
        if (!fr.hits.empty()) {
            filtered_rays.push_back(std::move(fr));
            filtered_bf.push_back(std::move(fp));
        }
    }

    size_t filtered_hits = 0;
    for (const auto& r : filtered_rays) filtered_hits += r.hits.size();
    std::printf("\n  Filter: keeping hits with nearest photon <= %.6f m (gather_radius)\n", max_dist);
    std::printf("  Kept %zu / %zu hits (filtered out %zu in unlit regions)\n",
                filtered_hits, total_hits, filtered_out);
    std::printf("  Kept %zu / %zu rays\n\n", filtered_rays.size(), rays.size());

    if (filtered_hits == 0) {
        std::cerr << "[tool] FATAL: no hitpoints within gather_radius of any photon.\n";
        std::cerr << "       Check coordinate space alignment between photons and scene.\n";
        return 1;
    }

    // Replace rays/bf_nearest with filtered versions for the sweep
    rays = std::move(filtered_rays);
    bf_nearest = std::move(filtered_bf);
    total_hits = filtered_hits;

    // ── [4/4] Compare at density-based grid resolutions ────────────
    // Build a coarse pilot grid (cell_size = gather_radius) to measure the
    // actual per-cell density distribution.  This captures caustic hotspots
    // instead of assuming photons are uniformly distributed in the AABB.
    float pilot_cs = std::max(gather_radius, 0.01f);
    int   pilot_res = std::max(1, (int)std::round(extent / pilot_cs));
    HashGrid pilot_grid;
    pilot_grid.build(photons, pilot_cs / 2.0f);

    // Collect occupied-cell photon counts (true spatial cells, not hash buckets)
    // Use a coarse grid where hash collisions are negligible.
    std::vector<uint32_t> occ_counts;
    occ_counts.reserve(pilot_grid.table_size / 4);
    for (uint32_t k = 0; k < pilot_grid.table_size; ++k) {
        if (pilot_grid.cell_start[k] == 0xFFFFFFFFu) continue;
        uint32_t c = pilot_grid.cell_end[k] - pilot_grid.cell_start[k];
        if (c > 0) occ_counts.push_back(c);
    }
    std::sort(occ_counts.begin(), occ_counts.end());

    auto pctl_u32 = [&](double p) -> double {
        if (occ_counts.empty()) return 0.0;
        double k = (p / 100.0) * (occ_counts.size() - 1);
        size_t lo = (size_t)k;
        size_t hi = std::min(lo + 1, occ_counts.size() - 1);
        double frac = k - lo;
        return occ_counts[lo] + frac * (occ_counts[hi] - occ_counts[lo]);
    };

    double pilot_cell_vol = (double)pilot_cs * pilot_cs * pilot_cs;
    double dens_p50 = pctl_u32(50) / pilot_cell_vol;  // photons / m^3
    double dens_p90 = pctl_u32(90) / pilot_cell_vol;
    double dens_p99 = pctl_u32(99) / pilot_cell_vol;
    double dens_max = pctl_u32(100) / pilot_cell_vol;

    float vol_x = aabb.hi.x - aabb.lo.x;
    float vol_y = aabb.hi.y - aabb.lo.y;
    float vol_z = aabb.hi.z - aabb.lo.z;
    double scene_volume = (double)vol_x * vol_y * vol_z;
    double mean_density = photons.size() / scene_volume;

    std::printf("[4/4] Comparing random-in-cell across resolution sweep...\n");
    std::printf("  Pilot grid: cell_size=%.5f m (%d^3)  occupied=%zu cells\n",
                pilot_cs, pilot_res, occ_counts.size());
    std::printf("  Scene volume: %.6f m^3  Mean density: %.0f /m^3\n", scene_volume, mean_density);
    std::printf("  Local density (occupied cells): P50=%.0f  P90=%.0f  P99=%.0f  max=%.0f /m^3\n\n",
                dens_p50, dens_p90, dens_p99, dens_max);

    // Sweep cell sizes targeting different local-density regimes.
    // For each density, cell_size = cbrt(target_per_cell / density)
    // We test: 10 photons/cell at P50, P90, P99, and max density.
    // This ensures caustic hotspots (P99/max) get appropriately fine cells.
    struct DensitySweepEntry {
        const char* label;
        double density;
        double target_per_cell;
    };
    DensitySweepEntry sweep[] = {
        {"P50 density, 10/cell", dens_p50, 10.0},
        {"P90 density, 10/cell", dens_p90, 10.0},
        {"P99 density, 10/cell", dens_p99, 10.0},
        {"Max density, 10/cell", dens_max, 10.0},
        {"P50 density, 1/cell",  dens_p50, 1.0},
        {"P99 density, 1/cell",  dens_p99, 1.0},
    };

    for (const auto& entry : sweep) {
        if (entry.density <= 0.0) continue;
        double cs = std::cbrt(entry.target_per_cell / entry.density);
        int res = std::max(1, std::min(100000, (int)std::round(extent / cs)));

        std::printf("  --- %s -> density=%.0f /m^3  cell_size=%.6f m  resolution=%d ---\n",
                    entry.label, entry.density, cs, res);
        ResolutionResult rr = run_resolution(res, extent, photons, rays, bf_nearest);
        print_resolution(rr);
    }

    std::printf("\n===========================================================\n");
    std::printf("  Done.\n");
    std::printf("===========================================================\n");
    return 0;
}
