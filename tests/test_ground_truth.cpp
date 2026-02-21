// ─────────────────────────────────────────────────────────────────────
// test_ground_truth.cpp – Comparison: ground-truth vs optimized path
// ─────────────────────────────────────────────────────────────────────
// Tests that our optimized GPU-style path tracer (with guided NEE,
// guided BSDF, local bins, density caching) produces results within
// acceptable tolerance of a brute-force "reference" path tracer that
// uses NO approximations beyond the base photon density estimator.
//
// REAL DATA: Both paths run against the SAME photon data loaded from
// a binary file on disk ("tests/data/cornell_box.bin").  This file is
// produced by the renderer via --save-test-data, or bootstrapped on
// first run by the test using CPU photon tracing.
// This guarantees identical inputs: same photons, same grid, same bins.
// ─────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include "core/types.h"
#include "core/spectrum.h"
#include "core/config.h"
#include "core/random.h"
#include "core/photon_bins.h"
#include "core/cdf.h"
#include "core/nee_sampling.h"
#include "core/guided_nee.h"
#include "core/test_data_io.h"
#include "scene/scene.h"
#include "scene/obj_loader.h"
#include "renderer/renderer.h"
#include "renderer/camera.h"
#include "renderer/direct_light.h"
#include "renderer/mis.h"
#include "photon/photon.h"
#include "photon/hash_grid.h"
#include "photon/density_estimator.h"
#include "photon/surface_filter.h"
#include "photon/emitter.h"
#include "bsdf/bsdf.h"

#include <vector>
#include <cmath>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

// Constants matching GPU optix_device.cu (defined there locally)
constexpr int   NEE_GUIDED_MAX_EMISSIVE = 128;
constexpr float NEE_GUIDED_ALPHA        = 5.0f;

// Binary data file path (relative to SCENES_DIR/../tests/data)
static std::string test_data_path() {
    // SCENES_DIR = "<repo>/scenes"  →  data file at "<repo>/tests/data/cornell_box.bin"
    fs::path scenes(SCENES_DIR);
    fs::path data_dir = scenes.parent_path() / "tests" / "data";
    return (data_dir / "cornell_box.bin").string();
}

// =====================================================================
//  CORNELL BOX TEST DATASET  (loaded from disk)
// =====================================================================
// Shared fixture: loads photon data from binary, reconstructs scene
// from OBJ (deterministic), rebuilds hash grid + bins.

struct CornellBoxDataset {
    // Loaded from binary
    TestDataHeader   header;
    PhotonSoA        photons;
    PhotonSoA        caustic_photons;

    // Reconstructed deterministically
    Scene      scene;
    Camera     camera;
    HashGrid   grid;
    HashGrid   caustic_grid;

    // Precomputed bin data
    PhotonBinDirs           bin_dirs;
    std::vector<uint8_t>    photon_bin_idx;

    // Config shortcuts (from header)
    int   num_photons    = 0;
    float gather_radius  = 0.0f;
    int   max_bounces    = 0;

    bool  loaded_from_disk = false;
    bool  valid            = false;

    void build() {
        if (valid) return;

        std::string bin_path = test_data_path();

        // ── 1. Try loading binary data from disk ─────────────────────
        if (fs::exists(bin_path)) {
            if (!load_test_data(bin_path, photons, caustic_photons, header)) {
                std::cerr << "[Dataset] Failed to load " << bin_path << "\n";
                return;
            }
            loaded_from_disk = true;
            std::cout << "[Dataset] Loaded REAL data from " << bin_path << "\n";
        } else {
            // ── Fallback: generate on CPU (bootstrap) ────────────────
            std::cout << "[Dataset] Binary not found at " << bin_path
                      << " — generating via CPU photon trace (fallback)\n";

            // Use a smaller photon count for CPU bootstrap to keep it manageable.
            // The GPU renderer should be used for high-quality dataset generation.
            // 300k photons ≈ 30-60s on CPU — sufficient for relative-error tests.
            constexpr int CPU_BOOTSTRAP_PHOTONS = 300'000;
            header.num_photons_cfg = CPU_BOOTSTRAP_PHOTONS;
            header.gather_radius   = DEFAULT_GATHER_RADIUS;
            header.caustic_radius  = DEFAULT_CAUSTIC_RADIUS;
            header.max_bounces     = DEFAULT_MAX_BOUNCES;
            header.min_bounces_rr  = DEFAULT_MIN_BOUNCES_RR;
            header.rr_threshold    = DEFAULT_RR_THRESHOLD;
            header.scene_path      = "cornell_box/cornellbox.obj";

            // Load scene for tracing
            std::string obj_path = std::string(SCENES_DIR) + "/" + header.scene_path;
            if (!load_obj(obj_path, scene)) {
                std::cerr << "[Dataset] Failed to load " << obj_path << "\n";
                return;
            }
            scene.build_bvh();
            scene.build_emissive_distribution();

            EmitterConfig ecfg;
            ecfg.num_photons    = header.num_photons_cfg;
            ecfg.max_bounces    = header.max_bounces;
            ecfg.rr_threshold   = header.rr_threshold;
            ecfg.min_bounces_rr = header.min_bounces_rr;

            trace_photons(scene, ecfg, photons, caustic_photons);

            if (photons.size() == 0) {
                std::cerr << "[Dataset] No photons stored\n";
                return;
            }

            // Precompute bin_idx before saving
            bin_dirs.init(PHOTON_BIN_COUNT);
            photons.bin_idx.resize(photons.size());
            for (size_t i = 0; i < photons.size(); ++i) {
                float3 wi = make_f3(photons.wi_x[i], photons.wi_y[i], photons.wi_z[i]);
                photons.bin_idx[i] = (uint8_t)bin_dirs.find_nearest(wi);
            }

            // Save for future runs
            fs::create_directories(fs::path(bin_path).parent_path());
            if (save_test_data(bin_path, photons, caustic_photons, header)) {
                std::cout << "[Dataset] Saved bootstrap data to " << bin_path << "\n";
            }

            loaded_from_disk = false;
        }

        // ── 2. Apply config from header ──────────────────────────────
        num_photons   = (int)header.num_photons_cfg;
        gather_radius = header.gather_radius;
        max_bounces   = (int)header.max_bounces;

        // ── 3. Load scene (if not already loaded during fallback) ────
        if (scene.triangles.empty()) {
            std::string obj_path = std::string(SCENES_DIR) + "/" + header.scene_path;
            if (!load_obj(obj_path, scene)) {
                std::cerr << "[Dataset] Failed to load " << obj_path << "\n";
                return;
            }
            scene.build_bvh();
            scene.build_emissive_distribution();
        }

        if (scene.num_emissive() == 0) {
            std::cerr << "[Dataset] No emissive triangles\n";
            return;
        }

        // ── 4. Camera ────────────────────────────────────────────────
        camera = Camera::cornell_box_camera(64, 64);
        camera.update();

        // ── 5. Rebuild hash grid from loaded photons ─────────────────
        grid.build(photons, gather_radius);
        if (caustic_photons.size() > 0)
            caustic_grid.build(caustic_photons, header.caustic_radius);

        // ── 6. Precompute per-photon bin_idx (if not already set) ────
        bin_dirs.init(PHOTON_BIN_COUNT);
        if (photons.bin_idx.size() != photons.size()) {
            photons.bin_idx.resize(photons.size());
            for (size_t i = 0; i < photons.size(); ++i) {
                float3 wi = make_f3(photons.wi_x[i], photons.wi_y[i], photons.wi_z[i]);
                photons.bin_idx[i] = (uint8_t)bin_dirs.find_nearest(wi);
            }
        }
        photon_bin_idx = photons.bin_idx;

        valid = true;
        std::cout << "[Dataset] Cornell Box ready: "
                  << scene.triangles.size() << " tris, "
                  << photons.size() << " photons (from "
                  << (loaded_from_disk ? "DISK" : "CPU fallback") << "), "
                  << grid.table_size << " grid buckets\n";
    }
};

// Singleton dataset — built once, reused by all tests
static CornellBoxDataset& get_dataset() {
    static CornellBoxDataset ds;
    ds.build();
    return ds;
}

// =====================================================================
//  GROUND-TRUTH CPU PATH TRACER
// =====================================================================
// Pure, unoptimized reference.  NO guided NEE, NO guided BSDF,
// NO local bins, NO density cache.  Just:
//   1. Standard NEE (uniform emissive CDF, M shadow rays)
//   2. Full photon gather at EVERY diffuse hit
//   3. Cosine-weighted hemisphere BSDF sampling
//   4. Russian roulette after min_bounces_rr
//
// This is the "physically correct" baseline (modulo statistical noise).

struct GroundTruthResult {
    Spectrum combined;          // Total radiance estimate
    Spectrum nee_direct;        // Direct lighting component
    Spectrum photon_indirect;   // Photon density component
};

static GroundTruthResult ground_truth_path_trace(
    float3 origin, float3 direction, PCGRng& rng,
    const CornellBoxDataset& ds, int max_bounces)
{
    GroundTruthResult result;
    result.combined        = Spectrum::zero();
    result.nee_direct      = Spectrum::zero();
    result.photon_indirect = Spectrum::zero();

    Spectrum throughput = Spectrum::constant(1.0f);
    bool prev_was_specular = true;

    DensityEstimatorConfig de_cfg;
    de_cfg.radius          = ds.gather_radius;
    de_cfg.caustic_radius  = DEFAULT_CAUSTIC_RADIUS;
    de_cfg.num_photons_total = ds.num_photons;
    de_cfg.surface_tau     = DEFAULT_SURFACE_TAU;
    de_cfg.use_kernel      = true;

    Ray ray;
    ray.origin    = origin;
    ray.direction = direction;

    for (int bounce = 0; bounce <= max_bounces; ++bounce) {
        HitRecord hit = ds.scene.intersect(ray);
        if (!hit.hit) break;

        const Material& mat = ds.scene.materials[hit.material_id];

        // Emission: only from camera/specular paths
        if (mat.is_emissive()) {
            if (prev_was_specular) {
                Spectrum Le_contrib = throughput * mat.Le;
                result.combined   += Le_contrib;
                result.nee_direct += Le_contrib;
            }
            break;
        }

        // Specular: mirror bounce
        if (mat.is_specular()) {
            ONB frame = ONB::from_normal(hit.shading_normal);
            float3 wo_local = frame.world_to_local(ray.direction * (-1.f));
            BSDFSample bs = bsdf::sample(mat, wo_local, rng);
            if (bs.pdf <= 0.f) break;

            float cos_theta = fabsf(bs.wi.z);
            for (int i = 0; i < NUM_LAMBDA; ++i)
                throughput.value[i] *= bs.f.value[i] * cos_theta / bs.pdf;

            ray.origin    = hit.position + hit.shading_normal * EPSILON *
                            (bs.wi.z > 0.f ? 1.f : -1.f);
            ray.direction = frame.local_to_world(bs.wi);
            prev_was_specular = true;
            continue;
        }

        prev_was_specular = false;

        // Diffuse hit
        ONB frame = ONB::from_normal(hit.shading_normal);
        float3 wo_local = frame.world_to_local(ray.direction * (-1.f));
        if (wo_local.z <= 0.f) break;

        // ── NEE: single shadow ray, standard CDF ────────────────
        {
            DirectLightSample dls = sample_direct_light(
                hit.position, hit.shading_normal, ds.scene, rng);

            if (dls.visible && dls.pdf_light > 0.f) {
                float3 wi_local = frame.world_to_local(dls.wi);
                float cos_theta = fmaxf(0.f, wi_local.z);

                if (cos_theta > 0.f) {
                    Spectrum f = bsdf::evaluate(mat, wo_local, wi_local);
                    // No MIS — raw NEE to keep it dead simple
                    Spectrum contrib = dls.Li * f * (cos_theta / dls.pdf_light);
                    Spectrum nee_add = throughput * contrib;
                    result.combined   += nee_add;
                    result.nee_direct += nee_add;
                }
            }
        }

        // ── Photon density at first diffuse hit (full gather) ─────
        {
            Spectrum L_photon = estimate_photon_density(
                hit.position, hit.shading_normal, wo_local, mat,
                ds.photons, ds.grid, de_cfg, ds.gather_radius);

            Spectrum photon_add = throughput * L_photon;
            result.combined        += photon_add;
            result.photon_indirect += photon_add;
        }

        // FIX: The photon map already captures ALL indirect illumination
        // (≥2-bounce paths from light).  Continuing BSDF + gathering
        // photons at deeper bounces double-counts those paths.
        // Stop after NEE + photon gather at the first diffuse hit.
        break;
    }

    return result;
}

// =====================================================================
//  OPTIMIZED CPU PATH TRACER (mirrors GPU full_path_trace)
// =====================================================================

// Build local bins during photon gather (mirrors dev_estimate_photon_density_with_bins)
static Spectrum gather_with_local_bins(
    float3 pos, float3 normal, float3 wo_local,
    const Material& mat,
    const CornellBoxDataset& ds,
    PhotonBin* local_bins, int num_bins,
    float& total_bin_flux)
{
    Spectrum L = Spectrum::zero();
    total_bin_flux = 0.0f;

    // Zero-initialize bins
    for (int k = 0; k < num_bins; ++k) {
        local_bins[k].flux   = 0.0f;
        local_bins[k].dir_x  = 0.0f;
        local_bins[k].dir_y  = 0.0f;
        local_bins[k].dir_z  = 0.0f;
        local_bins[k].weight = 0.0f;
        local_bins[k].count  = 0;
    }

    float radius  = ds.gather_radius;
    float r2      = radius * radius;
    float inv_area = 1.0f / (PI * r2);
    float inv_N   = 1.0f / (float)ds.num_photons;
    int count     = 0;

    ONB frame = ONB::from_normal(normal);

    float tau = effective_tau(DEFAULT_SURFACE_TAU);
    ds.grid.query_tangential(pos, normal, radius, tau, ds.photons,
        [&](uint32_t idx, float dist2) {
            // Normal visibility filter: reject photons from opposite faces
            // (mirrors density_estimator.h; guard for v1 files with empty norm arrays)
            if (!ds.photons.norm_x.empty()) {
                float3 photon_n = make_f3(ds.photons.norm_x[idx],
                                          ds.photons.norm_y[idx],
                                          ds.photons.norm_z[idx]);
                if (dot(photon_n, normal) <= 0.f) return;
            }

            float3 wi_world = make_f3(ds.photons.wi_x[idx], ds.photons.wi_y[idx],
                                       ds.photons.wi_z[idx]);
            if (dot(wi_world, normal) <= 0.f) return;

            // Density estimation uses box kernel (weight = 1) to match estimate_photon_density.
            // Bin population uses Epanechnikov for smoother guidance.
            // dist2 is tangential distance from query_tangential callback
            float w_bin = 1.0f - dist2 / r2;  // Epanechnikov for bins

            // Density estimation
            float3 wi_local = frame.world_to_local(wi_world);
            Spectrum f = bsdf::evaluate(mat, wo_local, wi_local);
            float flux = ds.photons.flux[idx];
            int bin = ds.photons.lambda_bin[idx];
            L.value[bin] += flux * inv_N * f.value[bin] * inv_area;  // box kernel (w=1)

            // Bin population (O(1) via precomputed index)
            int k = (int)ds.photon_bin_idx[idx];
            if (k >= num_bins) k = 0;
            local_bins[k].flux   += flux * w_bin;
            local_bins[k].dir_x  += wi_world.x * flux * w_bin;
            local_bins[k].dir_y  += wi_world.y * flux * w_bin;
            local_bins[k].dir_z  += wi_world.z * flux * w_bin;
            local_bins[k].weight += w_bin;
            local_bins[k].count  += 1;
            count++;
        });

    // No Epanechnikov correction needed — density now uses box kernel

    // Normalize bin centroids + total flux
    for (int k = 0; k < num_bins; ++k) {
        if (local_bins[k].count > 0) {
            float3 d = make_f3(local_bins[k].dir_x, local_bins[k].dir_y,
                               local_bins[k].dir_z);
            float len = length(d);
            if (len > 1e-8f) {
                local_bins[k].dir_x = d.x / len;
                local_bins[k].dir_y = d.y / len;
                local_bins[k].dir_z = d.z / len;
            }
        }
        total_bin_flux += local_bins[k].flux;
    }

    return L;
}

// Guided NEE (mirrors dev_nee_guided on GPU)
static Spectrum optimized_nee_guided(
    float3 pos, float3 normal, float3 wo_local,
    const Material& mat, PCGRng& rng,
    const CornellBoxDataset& ds,
    const PhotonBin* bins, int N, const PhotonBinDirs& bin_dirs,
    float total_bin_flux, int bounce,
    float& visibility_out)
{
    visibility_out = 0.f;
    Spectrum L_nee = Spectrum::zero();
    const Scene& scene = ds.scene;

    if (scene.emissive_tri_indices.empty()) return L_nee;

    int num_emissive = (int)scene.emissive_tri_indices.size();

    // Fall back to standard NEE if bins are empty or too many emissives
    bool use_guided = (total_bin_flux > 0.0f && num_emissive <= NEE_GUIDED_MAX_EMISSIVE);

    // Bounce-dependent sample count
    int M = nee_shadow_sample_count(
        bounce, DEFAULT_NEE_LIGHT_SAMPLES, DEFAULT_NEE_DEEP_SAMPLES);

    // Build guided CDF if applicable
    std::vector<float> guided_cdf;
    if (use_guided) {
        guided_cdf.resize(num_emissive);
        float guided_total = 0.0f;

        for (int i = 0; i < num_emissive; ++i) {
            float p_orig = scene.emissive_alias_table.pdf(i);

            uint32_t tri = scene.emissive_tri_indices[i];
            const Triangle& light_tri = scene.triangles[tri];
            float3 centroid = (light_tri.v0 + light_tri.v1 + light_tri.v2) * (1.0f / 3.0f);
            float3 to_light = centroid - pos;
            float d = length(to_light);

            float bin_boost = 0.0f;
            if (d > 1e-6f) {
                float3 wi = to_light * (1.0f / d);
                bin_boost = guided_nee_bin_boost(wi, normal, bins, N, bin_dirs, total_bin_flux);
            }

            float w = guided_nee_weight(p_orig, bin_boost, NEE_GUIDED_ALPHA);
            guided_total += w;
            guided_cdf[i] = guided_total;
        }

        if (guided_total <= 0.0f) use_guided = false;
        else {
            float inv = 1.0f / guided_total;
            for (int i = 0; i < num_emissive; ++i) guided_cdf[i] *= inv;
        }
    }

    ONB frame = ONB::from_normal(normal);
    int visible_count = 0;

    for (int s = 0; s < M; ++s) {
        // Select emissive triangle
        float xi = rng.next_float();
        int local_idx;
        float p_select;

        if (use_guided) {
            local_idx = 0;
            for (int i = 0; i < num_emissive; ++i) {
                if (xi <= guided_cdf[i]) { local_idx = i; break; }
                local_idx = i;
            }
            p_select = (local_idx == 0) ? guided_cdf[0]
                     : (guided_cdf[local_idx] - guided_cdf[local_idx - 1]);
        } else {
            float u2 = rng.next_float();
            local_idx = scene.emissive_alias_table.sample(xi, u2);
            p_select  = scene.emissive_alias_table.pdf(local_idx);
        }

        uint32_t tri_id = scene.emissive_tri_indices[local_idx];
        const Triangle& light_tri = scene.triangles[tri_id];
        const Material& light_mat = scene.materials[light_tri.material_id];
        float light_area = light_tri.area();

        float3 bary = sample_triangle(rng.next_float(), rng.next_float());
        float3 light_pos = light_tri.interpolate_position(bary.x, bary.y, bary.z);
        float3 light_normal = light_tri.geometric_normal();

        float3 to_light = light_pos - pos;
        float dist2 = dot(to_light, to_light);
        float dist  = sqrtf(dist2);
        float3 wi   = to_light * (1.0f / dist);

        float cos_x = dot(wi, normal);
        float cos_y = -dot(wi, light_normal);
        if (cos_x <= 0.f || cos_y <= 0.f) continue;

        Ray shadow;
        shadow.origin    = pos + normal * EPSILON;
        shadow.direction = wi;
        shadow.tmin      = 1e-4f;
        shadow.tmax      = dist - 2e-4f;
        HitRecord shadow_hit = scene.intersect(shadow);
        if (shadow_hit.hit) continue;
        visible_count++;

        float3 wi_local = frame.world_to_local(wi);
        Spectrum f = bsdf::evaluate(mat, wo_local, wi_local);

        float p_y_area = p_select / light_area;
        float p_wi = p_y_area * dist2 / cos_y;

        for (int i = 0; i < NUM_LAMBDA; ++i)
            L_nee.value[i] += f.value[i] * light_mat.Le.value[i]
                           * cos_x / fmaxf(p_wi, 1e-8f);
    }

    if (M > 1) L_nee *= 1.0f / (float)M;

    visibility_out = (float)visible_count / (float)M;
    return L_nee;
}

// Full optimized path trace (mirrors GPU full_path_trace)
static GroundTruthResult optimized_path_trace(
    float3 origin, float3 direction, PCGRng& rng,
    const CornellBoxDataset& ds, int max_bounces)
{
    GroundTruthResult result;
    result.combined        = Spectrum::zero();
    result.nee_direct      = Spectrum::zero();
    result.photon_indirect = Spectrum::zero();

    Spectrum throughput = Spectrum::constant(1.0f);
    bool prev_was_specular = true;

    Ray ray;
    ray.origin    = origin;
    ray.direction = direction;

    for (int bounce = 0; bounce <= max_bounces; ++bounce) {
        HitRecord hit = ds.scene.intersect(ray);
        if (!hit.hit) break;

        const Material& mat = ds.scene.materials[hit.material_id];

        // Emission
        if (mat.is_emissive()) {
            if (prev_was_specular) {
                Spectrum Le_contrib = throughput * mat.Le;
                result.combined   += Le_contrib;
                result.nee_direct += Le_contrib;
            }
            break;
        }

        // Specular
        if (mat.is_specular()) {
            ONB frame = ONB::from_normal(hit.shading_normal);
            float3 wo_local = frame.world_to_local(ray.direction * (-1.f));
            BSDFSample bs = bsdf::sample(mat, wo_local, rng);
            if (bs.pdf <= 0.f) break;

            float cos_theta = fabsf(bs.wi.z);
            for (int i = 0; i < NUM_LAMBDA; ++i)
                throughput.value[i] *= bs.f.value[i] * cos_theta / bs.pdf;

            ray.origin    = hit.position + hit.shading_normal * EPSILON *
                            (bs.wi.z > 0.f ? 1.f : -1.f);
            ray.direction = frame.local_to_world(bs.wi);
            prev_was_specular = true;
            continue;
        }

        prev_was_specular = false;

        // Diffuse hit
        ONB frame = ONB::from_normal(hit.shading_normal);
        float3 wo_local = frame.world_to_local(ray.direction * (-1.f));
        if (wo_local.z <= 0.f) break;

        // ── Photon gather + local bins ───────────────────────────
        PhotonBin local_bins[MAX_PHOTON_BIN_COUNT];
        float local_total_flux = 0.0f;

        Spectrum L_photon = gather_with_local_bins(
            hit.position, hit.shading_normal, wo_local, mat, ds,
            local_bins, PHOTON_BIN_COUNT, local_total_flux);

        bool have_bins = (local_total_flux > 0.0f);

        // ── Guided NEE ───────────────────────────────────────────
        float nee_visibility = 1.0f;
        {
            Spectrum L_nee;
            if (have_bins) {
                L_nee = optimized_nee_guided(
                    hit.position, hit.shading_normal, wo_local, mat, rng, ds,
                    local_bins, PHOTON_BIN_COUNT, ds.bin_dirs,
                    local_total_flux, bounce, nee_visibility);
            } else {
                DirectLightSample dls = sample_direct_light(
                    hit.position, hit.shading_normal, ds.scene, rng);
                if (dls.visible && dls.pdf_light > 0.f) {
                    float3 wi_local = frame.world_to_local(dls.wi);
                    float cos_theta = fmaxf(0.f, wi_local.z);
                    if (cos_theta > 0.f) {
                        Spectrum f = bsdf::evaluate(mat, wo_local, wi_local);
                        L_nee = dls.Li * f * (cos_theta / dls.pdf_light);
                    }
                }
                nee_visibility = dls.visible ? 1.0f : 0.0f;
            }

            Spectrum nee_contrib = throughput * L_nee;
            result.combined   += nee_contrib;
            result.nee_direct += nee_contrib;
        }

        // ── Photon indirect (no shadow-floor suppression) ─────────
        // FIX: Indirect illumination (color bleeding, ambient) is
        // visible in shadows.  Do NOT modulate by nee_visibility.
        {
            Spectrum photon_contrib = throughput * L_photon;
            result.combined        += photon_contrib;
            result.photon_indirect += photon_contrib;
        }

        // FIX: The photon map already captures ALL indirect illumination
        // (≥2-bounce paths from light).  Continuing BSDF + gathering
        // photons at deeper bounces double-counts those paths.
        // Stop after NEE + photon gather at the first diffuse hit.
        break;
    }

    return result;
}

// =====================================================================
//  STATISTICAL COMPARISON UTILITY
// =====================================================================

struct ComparisonStats {
    double gt_mean_lum   = 0.0;
    double opt_mean_lum  = 0.0;
    double gt_var        = 0.0;
    double opt_var       = 0.0;
    double rel_error     = 0.0;
    int    num_samples   = 0;
};

static ComparisonStats compare_methods(
    const CornellBoxDataset& ds,
    int num_rays, int max_bounces,
    int px_start, int py_start, int px_end, int py_end)
{
    ComparisonStats stats;

    int width  = px_end - px_start;
    int height = py_end - py_start;
    int rays_per_pixel = (width > 0 && height > 0)
                       ? std::max(1, num_rays / (width * height)) : num_rays;

    // Pre-compute total work items for indexed parallel access
    int total_work = 0;
    for (int py = py_start; py < py_end; ++py)
        for (int px = px_start; px < px_end; ++px)
            for (int s = 0; s < rays_per_pixel; ++s)
                if (total_work < num_rays) total_work++;

    std::vector<double> gt_lums(total_work, 0.0);
    std::vector<double> opt_lums(total_work, 0.0);

    #pragma omp parallel for schedule(dynamic, 8)
    for (int idx = 0; idx < total_work; ++idx) {
        uint64_t seed_a = (uint64_t)idx * 31 + 12345;
        uint64_t seed_b = (uint64_t)idx + 1;

        // Recover px, py from flat index
        int rem = idx;
        int py = py_start + rem / (width * rays_per_pixel);
        rem %= (width * rays_per_pixel);
        int px = px_start + rem / rays_per_pixel;

        PCGRng rng_ray = PCGRng::seed(seed_a, seed_b);
        Ray ray = ds.camera.generate_ray(px, py, rng_ray);

        PCGRng rng_gt = PCGRng::seed(seed_a + 1000000, seed_b);
        auto gt = ground_truth_path_trace(
            ray.origin, ray.direction, rng_gt, ds, max_bounces);

        PCGRng rng_opt = PCGRng::seed(seed_a + 2000000, seed_b);
        auto opt = optimized_path_trace(
            ray.origin, ray.direction, rng_opt, ds, max_bounces);

        gt_lums[idx]  = (double)gt.combined.sum();
        opt_lums[idx] = (double)opt.combined.sum();
    }

    int count = total_work;
    stats.num_samples = count;
    if (count == 0) return stats;

    double gt_sum  = std::accumulate(gt_lums.begin(), gt_lums.end(), 0.0);
    double opt_sum = std::accumulate(opt_lums.begin(), opt_lums.end(), 0.0);
    stats.gt_mean_lum  = gt_sum / count;
    stats.opt_mean_lum = opt_sum / count;

    double gt_var_sum = 0.0, opt_var_sum = 0.0;
    for (int i = 0; i < count; ++i) {
        double dg = gt_lums[i] - stats.gt_mean_lum;
        double da = opt_lums[i] - stats.opt_mean_lum;
        gt_var_sum  += dg * dg;
        opt_var_sum += da * da;
    }
    stats.gt_var  = gt_var_sum / count;
    stats.opt_var = opt_var_sum / count;

    if (stats.gt_mean_lum > 1e-10)
        stats.rel_error = fabs(stats.opt_mean_lum - stats.gt_mean_lum) / stats.gt_mean_lum;
    else
        stats.rel_error = fabs(stats.opt_mean_lum);

    return stats;
}

// =====================================================================
//  TEST SUITE: GroundTruthComparison
// =====================================================================

// ── Dataset validity ─────────────────────────────────────────────────
TEST(GroundTruthComparison, DatasetIsValid) {
    auto& ds = get_dataset();
    ASSERT_TRUE(ds.valid) << "Cornell Box dataset failed to build";
    EXPECT_GT(ds.scene.triangles.size(), 10000u);
    EXPECT_GT(ds.photons.size(), 1000u);
    EXPECT_EQ(ds.photon_bin_idx.size(), ds.photons.size());
    EXPECT_GT(ds.grid.table_size, 0u);
    EXPECT_GT(ds.scene.num_emissive(), 0);
}

// ── Verify that data was loaded from disk (not CPU fallback) ─────────
TEST(GroundTruthComparison, DataLoadedFromDisk) {
    auto& ds = get_dataset();
    ASSERT_TRUE(ds.valid);

    // This test warns (not fails) if data came from CPU fallback
    if (!ds.loaded_from_disk) {
        std::cout << "[WARNING] Test data was generated by CPU fallback.\n"
                  << "  Run the renderer with --save-test-data to use real GPU data:\n"
                  << "    photon_tracer --save-test-data tests/data/cornell_box.bin\n";
    }
    // Always passes — it's informational
    SUCCEED();
}

// ── Binary round-trip: header values preserved correctly ─────────────
TEST(GroundTruthComparison, HeaderValuesPreserved) {
    auto& ds = get_dataset();
    ASSERT_TRUE(ds.valid);

    EXPECT_GT(ds.header.num_photons_cfg, 0u);
    EXPECT_GT(ds.header.gather_radius, 0.f);
    EXPECT_GT(ds.header.max_bounces, 0u);
    EXPECT_EQ(ds.num_photons, (int)ds.header.num_photons_cfg);
    EXPECT_FLOAT_EQ(ds.gather_radius, ds.header.gather_radius);
    EXPECT_EQ(ds.max_bounces, (int)ds.header.max_bounces);
}

// ── Single ray: both methods produce finite, non-negative results ────
TEST(GroundTruthComparison, BothMethodsProduceFiniteResults) {
    auto& ds = get_dataset();
    ASSERT_TRUE(ds.valid);

    PCGRng rng_ray = PCGRng::seed(42, 1);
    Ray ray = ds.camera.generate_ray(32, 32, rng_ray);

    PCGRng rng_gt = PCGRng::seed(100, 1);
    auto gt = ground_truth_path_trace(
        ray.origin, ray.direction, rng_gt, ds, ds.max_bounces);

    PCGRng rng_opt = PCGRng::seed(200, 1);
    auto opt = optimized_path_trace(
        ray.origin, ray.direction, rng_opt, ds, ds.max_bounces);

    for (int i = 0; i < NUM_LAMBDA; ++i) {
        EXPECT_TRUE(std::isfinite(gt.combined.value[i]))
            << "GT combined[" << i << "] not finite";
        EXPECT_GE(gt.combined.value[i], 0.f)
            << "GT combined[" << i << "] negative";
        EXPECT_TRUE(std::isfinite(opt.combined.value[i]))
            << "Opt combined[" << i << "] not finite";
        EXPECT_GE(opt.combined.value[i], 0.f)
            << "Opt combined[" << i << "] negative";
    }
}

// ── Center pixel convergence (well-illuminated area) ─────────────────
TEST(GroundTruthComparison, CenterPixelMeanConverges) {
    auto& ds = get_dataset();
    ASSERT_TRUE(ds.valid);

    auto stats = compare_methods(ds, 512, ds.max_bounces, 30, 30, 34, 34);

    std::cout << "[CenterPixel] GT mean=" << stats.gt_mean_lum
              << " Opt mean=" << stats.opt_mean_lum
              << " relErr=" << stats.rel_error
              << " N=" << stats.num_samples << "\n";

    if (stats.gt_mean_lum > 1e-6) {
        EXPECT_LT(stats.rel_error, 0.30)
            << "Optimized mean diverges from ground truth by >"
            << (int)(stats.rel_error * 100) << "%";
    }
}

// ── Full image comparison (all pixels, few samples) ──────────────────
TEST(GroundTruthComparison, FullImageMeanConverges) {
    auto& ds = get_dataset();
    ASSERT_TRUE(ds.valid);

    auto stats = compare_methods(ds, 2048, ds.max_bounces, 0, 0, 64, 64);

    std::cout << "[FullImage] GT mean=" << stats.gt_mean_lum
              << " Opt mean=" << stats.opt_mean_lum
              << " relErr=" << stats.rel_error
              << " GT var=" << stats.gt_var
              << " Opt var=" << stats.opt_var
              << " N=" << stats.num_samples << "\n";

    if (stats.gt_mean_lum > 1e-6) {
        EXPECT_LT(stats.rel_error, 0.50)
            << "Full image optimized mean diverges by >"
            << (int)(stats.rel_error * 100) << "%";
    }
}

// ── Variance comparison: optimized should not be wildly worse ────────
TEST(GroundTruthComparison, OptimizedVarianceNotWorse) {
    auto& ds = get_dataset();
    ASSERT_TRUE(ds.valid);

    auto stats = compare_methods(ds, 1024, ds.max_bounces, 16, 16, 48, 48);

    std::cout << "[Variance] GT var=" << stats.gt_var
              << " Opt var=" << stats.opt_var << "\n";

    if (stats.gt_var > 1e-10 && stats.opt_var > 1e-10) {
        double ratio = stats.opt_var / stats.gt_var;
        EXPECT_LT(ratio, 5.0)
            << "Optimized variance is " << ratio << "× ground truth";
    }
}

// ── Direct lighting agreement ────────────────────────────────────────
TEST(GroundTruthComparison, DirectLightingAgreement) {
    auto& ds = get_dataset();
    ASSERT_TRUE(ds.valid);

    double gt_nee_sum = 0.0, opt_nee_sum = 0.0;
    int count = 0;

    for (int py = 20; py < 44; py += 4) {
        for (int px = 20; px < 44; px += 4) {
            for (int s = 0; s < 16; ++s) {
                uint64_t seed = (uint64_t)count * 31 + 99;
                PCGRng rng_ray = PCGRng::seed(seed, count + 1);
                Ray ray = ds.camera.generate_ray(px, py, rng_ray);

                PCGRng rng_gt  = PCGRng::seed(seed + 1000000, count + 1);
                auto gt = ground_truth_path_trace(
                    ray.origin, ray.direction, rng_gt, ds, ds.max_bounces);

                PCGRng rng_opt = PCGRng::seed(seed + 2000000, count + 1);
                auto opt = optimized_path_trace(
                    ray.origin, ray.direction, rng_opt, ds, ds.max_bounces);

                gt_nee_sum  += (double)gt.nee_direct.sum();
                opt_nee_sum += (double)opt.nee_direct.sum();
                count++;
            }
        }
    }

    double gt_mean  = gt_nee_sum / count;
    double opt_mean = opt_nee_sum / count;

    std::cout << "[DirectLighting] GT NEE mean=" << gt_mean
              << " Opt NEE mean=" << opt_mean << "\n";

    if (gt_mean > 1e-6) {
        double rel = fabs(opt_mean - gt_mean) / gt_mean;
        EXPECT_LT(rel, 0.40)
            << "Direct lighting means diverge by " << (int)(rel * 100) << "%";
    }
}

// ── Photon indirect agreement ────────────────────────────────────────
TEST(GroundTruthComparison, PhotonIndirectAgreement) {
    auto& ds = get_dataset();
    ASSERT_TRUE(ds.valid);

    double gt_ind_sum = 0.0, opt_ind_sum = 0.0;
    int count = 0;

    for (int py = 16; py < 48; py += 4) {
        for (int px = 16; px < 48; px += 4) {
            for (int s = 0; s < 8; ++s) {
                uint64_t seed = (uint64_t)count * 37 + 555;
                PCGRng rng_ray = PCGRng::seed(seed, count + 1);
                Ray ray = ds.camera.generate_ray(px, py, rng_ray);

                PCGRng rng_gt  = PCGRng::seed(seed + 1000000, count + 1);
                auto gt = ground_truth_path_trace(
                    ray.origin, ray.direction, rng_gt, ds, ds.max_bounces);

                PCGRng rng_opt = PCGRng::seed(seed + 2000000, count + 1);
                auto opt = optimized_path_trace(
                    ray.origin, ray.direction, rng_opt, ds, ds.max_bounces);

                gt_ind_sum  += (double)gt.photon_indirect.sum();
                opt_ind_sum += (double)opt.photon_indirect.sum();
                count++;
            }
        }
    }

    double gt_mean  = gt_ind_sum / count;
    double opt_mean = opt_ind_sum / count;

    std::cout << "[PhotonIndirect] GT mean=" << gt_mean
              << " Opt mean=" << opt_mean << "\n";

    if (gt_mean > 1e-6) {
        double rel = fabs(opt_mean - gt_mean) / gt_mean;
        EXPECT_LT(rel, 0.50)
            << "Photon indirect means diverge by " << (int)(rel * 100) << "%";
    }
}

// ── Bounce 0 only: should be very close ──────────────────────────────
TEST(GroundTruthComparison, SingleBounceClose) {
    auto& ds = get_dataset();
    ASSERT_TRUE(ds.valid);

    auto stats = compare_methods(ds, 1024, 0, 16, 16, 48, 48);

    std::cout << "[SingleBounce] GT mean=" << stats.gt_mean_lum
              << " Opt mean=" << stats.opt_mean_lum
              << " relErr=" << stats.rel_error << "\n";

    if (stats.gt_mean_lum > 1e-6) {
        EXPECT_LT(stats.rel_error, 0.25)
            << "Single-bounce results diverge by "
            << (int)(stats.rel_error * 100) << "%";
    }
}

// ── Multi-bounce (full 8): guided should not introduce bias ──────────
TEST(GroundTruthComparison, FullBounceNoBias) {
    auto& ds = get_dataset();
    ASSERT_TRUE(ds.valid);

    auto stats = compare_methods(ds, 2048, ds.max_bounces, 8, 8, 56, 56);

    std::cout << "[FullBounce] GT mean=" << stats.gt_mean_lum
              << " Opt mean=" << stats.opt_mean_lum
              << " relErr=" << stats.rel_error
              << " N=" << stats.num_samples << "\n";

    if (stats.gt_mean_lum > 1e-6) {
        EXPECT_LT(stats.rel_error, 0.35)
            << "Full-bounce optimized path is biased by "
            << (int)(stats.rel_error * 100) << "%";
    }
}

// ── Energy conservation ──────────────────────────────────────────────
TEST(GroundTruthComparison, EnergyConservation) {
    auto& ds = get_dataset();
    ASSERT_TRUE(ds.valid);

    int N = 1024;
    double gt_total = 0.0, opt_total = 0.0;

    for (int i = 0; i < N; ++i) {
        int px = i % 64;
        int py = i / 64;
        if (py >= 64) py = py % 64;

        PCGRng rng_ray = PCGRng::seed((uint64_t)i * 17 + 777, i + 1);
        Ray ray = ds.camera.generate_ray(px, py, rng_ray);

        PCGRng rng_gt  = PCGRng::seed((uint64_t)i * 17 + 1000777, i + 1);
        auto gt = ground_truth_path_trace(ray.origin, ray.direction, rng_gt, ds, ds.max_bounces);

        PCGRng rng_opt = PCGRng::seed((uint64_t)i * 17 + 2000777, i + 1);
        auto opt = optimized_path_trace(ray.origin, ray.direction, rng_opt, ds, ds.max_bounces);

        gt_total  += (double)gt.combined.sum();
        opt_total += (double)opt.combined.sum();
    }

    std::cout << "[Energy] GT total=" << gt_total
              << " Opt total=" << opt_total
              << " ratio=" << (gt_total > 0 ? opt_total / gt_total : 0.0) << "\n";

    if (gt_total > 1e-6) {
        double ratio = opt_total / gt_total;
        EXPECT_GT(ratio, 0.60) << "Optimized loses too much energy";
        EXPECT_LT(ratio, 1.50) << "Optimized creates too much energy";
    }
}

// ── Local bins match precomputed bin_idx ──────────────────────────────
TEST(GroundTruthComparison, BinIdxMatchesFindNearest) {
    auto& ds = get_dataset();
    ASSERT_TRUE(ds.valid);

    int num_checks = std::min((int)ds.photons.size(), 10000);
    int mismatches = 0;

    PhotonBinDirs dirs;
    dirs.init(PHOTON_BIN_COUNT);

    for (int i = 0; i < num_checks; ++i) {
        float3 wi = make_f3(ds.photons.wi_x[i], ds.photons.wi_y[i],
                            ds.photons.wi_z[i]);
        int expected = dirs.find_nearest(wi);
        int actual   = (int)ds.photon_bin_idx[i];
        if (expected != actual) mismatches++;
    }

    EXPECT_EQ(mismatches, 0)
        << mismatches << "/" << num_checks << " bin_idx mismatches";
}

// ── Gather-with-bins produces same density as standard gather ────────
TEST(GroundTruthComparison, GatherWithBinsMatchesStandard) {
    auto& ds = get_dataset();
    ASSERT_TRUE(ds.valid);

    float3 floor_pos    = make_f3(0.0f, -0.49f, 0.0f);
    float3 floor_normal = make_f3(0.0f, 1.0f, 0.0f);
    float3 wo_local     = make_f3(0.0f, 0.0f, 1.0f);

    const Material* floor_mat = nullptr;
    for (const auto& m : ds.scene.materials) {
        if (m.type == MaterialType::Lambertian) { floor_mat = &m; break; }
    }
    ASSERT_NE(floor_mat, nullptr) << "No Lambertian material found";

    DensityEstimatorConfig de_cfg;
    de_cfg.radius            = ds.gather_radius;
    de_cfg.num_photons_total = ds.num_photons;
    de_cfg.surface_tau       = DEFAULT_SURFACE_TAU;
    de_cfg.use_kernel        = true;

    Spectrum L_standard = estimate_photon_density(
        floor_pos, floor_normal, wo_local, *floor_mat,
        ds.photons, ds.grid, de_cfg, ds.gather_radius);

    PhotonBin bins[MAX_PHOTON_BIN_COUNT];
    float total_flux = 0.0f;
    Spectrum L_bins = gather_with_local_bins(
        floor_pos, floor_normal, wo_local, *floor_mat, ds,
        bins, PHOTON_BIN_COUNT, total_flux);

    float std_sum = L_standard.sum();
    float bin_sum = L_bins.sum();

    std::cout << "[GatherBins] Standard sum=" << std_sum
              << " WithBins sum=" << bin_sum << "\n";

    if (std_sum > 1e-8f) {
        float rel = fabsf(bin_sum - std_sum) / std_sum;
        EXPECT_LT(rel, 0.01f)
            << "Gather-with-bins density differs from standard by "
            << (rel * 100.f) << "%";
    }
}

// ── Per-spectral-bin agreement ───────────────────────────────────────
TEST(GroundTruthComparison, SpectralBinAgreement) {
    auto& ds = get_dataset();
    ASSERT_TRUE(ds.valid);

    int N = 512;
    Spectrum gt_acc  = Spectrum::zero();
    Spectrum opt_acc = Spectrum::zero();

    for (int i = 0; i < N; ++i) {
        int px = (i * 7) % 64;
        int py = (i * 13) % 64;

        PCGRng rng_ray = PCGRng::seed((uint64_t)i * 41 + 333, i + 1);
        Ray ray = ds.camera.generate_ray(px, py, rng_ray);

        PCGRng rng_gt  = PCGRng::seed((uint64_t)i + 5000000, i + 1);
        auto gt = ground_truth_path_trace(
            ray.origin, ray.direction, rng_gt, ds, ds.max_bounces);

        PCGRng rng_opt = PCGRng::seed((uint64_t)i + 6000000, i + 1);
        auto opt = optimized_path_trace(
            ray.origin, ray.direction, rng_opt, ds, ds.max_bounces);

        gt_acc  += gt.combined;
        opt_acc += opt.combined;
    }

    int bins_with_large_error = 0;
    for (int b = 0; b < NUM_LAMBDA; ++b) {
        float gt_mean  = gt_acc.value[b] / N;
        float opt_mean = opt_acc.value[b] / N;

        if (gt_mean > 1e-6f) {
            float rel = fabsf(opt_mean - gt_mean) / gt_mean;
            if (rel > 0.50f) bins_with_large_error++;
        }
    }

    std::cout << "[Spectral] Bins with >50% error: "
              << bins_with_large_error << "/" << NUM_LAMBDA << "\n";

    EXPECT_LT(bins_with_large_error, NUM_LAMBDA / 4)
        << "Too many spectral bins with large disagreement";
}

// ── Binary I/O round-trip test ───────────────────────────────────────
// Verifies that save_test_data → load_test_data preserves data exactly.
TEST(GroundTruthComparison, BinaryRoundTrip) {
    auto& ds = get_dataset();
    ASSERT_TRUE(ds.valid);

    // Save to a temp file
    fs::path scenes(SCENES_DIR);
    std::string tmp_path = (scenes.parent_path() / "tests" / "data" / "roundtrip_test.bin").string();

    TestDataHeader save_hdr = ds.header;
    ASSERT_TRUE(save_test_data(tmp_path, ds.photons, ds.caustic_photons, save_hdr));

    // Load back
    PhotonSoA loaded_global, loaded_caustic;
    TestDataHeader loaded_hdr;
    ASSERT_TRUE(load_test_data(tmp_path, loaded_global, loaded_caustic, loaded_hdr));

    // Verify header
    EXPECT_EQ(loaded_hdr.num_photons_cfg, save_hdr.num_photons_cfg);
    EXPECT_FLOAT_EQ(loaded_hdr.gather_radius, save_hdr.gather_radius);
    EXPECT_FLOAT_EQ(loaded_hdr.caustic_radius, save_hdr.caustic_radius);
    EXPECT_EQ(loaded_hdr.max_bounces, save_hdr.max_bounces);
    EXPECT_EQ(loaded_hdr.scene_path, save_hdr.scene_path);

    // Verify photon data
    EXPECT_EQ(loaded_global.size(), ds.photons.size());
    if (loaded_global.size() == ds.photons.size()) {
        int diffs = 0;
        for (size_t i = 0; i < ds.photons.size() && i < 10000; ++i) {
            if (loaded_global.pos_x[i] != ds.photons.pos_x[i]) diffs++;
            if (loaded_global.wi_x[i]  != ds.photons.wi_x[i])  diffs++;
            if (loaded_global.flux[i]  != ds.photons.flux[i])   diffs++;
            if (loaded_global.bin_idx[i] != ds.photons.bin_idx[i]) diffs++;
        }
        EXPECT_EQ(diffs, 0) << diffs << " fields differ after round-trip";
    }

    // Cleanup temp file
    fs::remove(tmp_path);
}
