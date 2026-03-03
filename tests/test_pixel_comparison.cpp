// ─────────────────────────────────────────────────────────────────────
// test_pixel_comparison.cpp – Pixel-by-pixel visual comparison at 128×128
// ─────────────────────────────────────────────────────────────────────
// Renders a 128×128 image with both the ground-truth and optimized
// path tracers, then compares per-pixel quantities:
//
//   1. NEE-only first-hit comparison (direct lighting)
//   2. Photon irradiance at first hit (indirect component)
//   3. Combined radiance (NEE + photon)
//   4. Photon-lobe directional accuracy vs actual photon directions
//   5. Shadow region indirect fidelity (was suppressed by vis_weight)
//   6. Per-spectral-bin pixel error heatmap
//   7. RMSE, PSNR, and relative luminance metrics
//
// Both tracers use the SAME photon data (loaded from disk binary).
// ─────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include "core/types.h"
#include "core/spectrum.h"
#include "core/config.h"
#include "core/random.h"
#include "photon/photon_bins.h"
#include "renderer/nee_shared.h"
#include "tests/test_data_io.h"
#include "scene/scene.h"
#include "scene/obj_loader.h"
#include "renderer/camera.h"
#include "renderer/direct_light.h"
#include "photon/photon.h"
#include "photon/hash_grid.h"
#include "photon/density_estimator.h"
#include "photon/surface_filter.h"
#include "photon/emitter.h"
#include "bsdf/bsdf.h"

#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <string>
#include <fstream>
#include <cstdlib>
#include <functional>

// Inlined from deleted core/guided_nee.h
static inline float guided_nee_bin_boost(float3 wi, float3 normal,
                                         const PhotonBin* bins, int N,
                                         const PhotonBinDirs& dirs, float total_flux) {
    (void)N;
    if (total_flux <= 0.f || dot(wi, normal) <= 0.f) return 0.f;
    int idx = dirs.find_nearest(wi);
    return bins[idx].scalar_flux / total_flux;
}
static inline float guided_nee_weight(float p, float boost, float alpha) {
    return p * (1.f + alpha * boost);
}

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────
//  CONSTANTS
// ─────────────────────────────────────────────────────────────────────
constexpr int   PIXEL_CMP_WIDTH             = 128;
constexpr int   PIXEL_CMP_HEIGHT            = 128;
constexpr int   PIXEL_CMP_SPP               = 4;     // samples per pixel
constexpr int   NEE_GUIDED_MAX_EMISSIVE     = 128;
constexpr float NEE_GUIDED_ALPHA            = 5.0f;

// Thresholds
constexpr float MAX_RMSE_LUMINANCE          = 0.15f;  // RMSE on luminance
constexpr float MIN_PSNR_DB                 = 12.0f;  // minimum PSNR
constexpr float MAX_MEAN_REL_ERROR          = 0.35f;  // mean relative error
constexpr float MAX_OUTLIER_FRACTION        = 0.20f;  // fraction of pixels >2× error
constexpr float MAX_SHADOW_INDIRECT_DIFF    = 0.40f;  // shadow region indirect rel err

// ─────────────────────────────────────────────────────────────────────
//  DATASET  (128×128 camera, same photon data)
// ─────────────────────────────────────────────────────────────────────

static std::string test_data_path() {
    fs::path scenes(SCENES_DIR);
    fs::path data_dir = scenes.parent_path() / "tests" / "data";
    return (data_dir / "cornell_box.bin").string();
}

struct PixelCmpDataset {
    TestDataHeader   header;
    PhotonSoA        photons;
    PhotonSoA        caustic_photons;
    Scene            scene;
    Camera           camera;
    HashGrid         grid;
    HashGrid         caustic_grid;
    PhotonBinDirs    bin_dirs;
    std::vector<uint8_t> photon_bin_idx;
    int   num_photons    = 0;
    float gather_radius  = 0.0f;
    int   max_bounces    = 0;
    bool  loaded_from_disk = false;
    bool  valid          = false;

    void build() {
        if (valid) return;
        std::string bin_path = test_data_path();

        if (fs::exists(bin_path)) {
            if (load_test_data(bin_path, photons, caustic_photons, header)) {
                loaded_from_disk = true;
            } else {
                std::cout << "[PixelCmpDataset] Removing stale " << bin_path << "\n";
                fs::remove(bin_path);
            }
        }
        if (!loaded_from_disk) {
            // Reset header so save writes current version/algo_version
            // (load_test_data may have partially overwritten fields).
            header = TestDataHeader{};

            header.num_photons_cfg = DEFAULT_GLOBAL_PHOTON_BUDGET;
            header.gather_radius   = DEFAULT_GATHER_RADIUS;
            header.caustic_radius  = DEFAULT_CAUSTIC_RADIUS;
            header.max_bounces     = DEFAULT_MAX_BOUNCES;
            header.min_bounces_rr  = DEFAULT_MIN_BOUNCES_RR;
            header.rr_threshold    = DEFAULT_RR_THRESHOLD;
            header.scene_path      = "cornell_box/cornellbox.obj";

            std::string obj_path = std::string(SCENES_DIR) + "/" + header.scene_path;
            if (!load_obj(obj_path, scene)) return;
            scene.build_bvh();
            scene.build_emissive_distribution();

            EmitterConfig ecfg;
            ecfg.num_photons    = header.num_photons_cfg;
            ecfg.max_bounces    = header.max_bounces;
            ecfg.rr_threshold   = header.rr_threshold;
            ecfg.min_bounces_rr = header.min_bounces_rr;
            trace_photons(scene, ecfg, photons, caustic_photons);
            if (photons.size() == 0) return;

            bin_dirs.init(PHOTON_BIN_COUNT);
            photons.bin_idx.resize(photons.size());
            for (size_t i = 0; i < photons.size(); ++i) {
                float3 wi = make_f3(photons.wi_x[i], photons.wi_y[i], photons.wi_z[i]);
                photons.bin_idx[i] = (uint8_t)bin_dirs.find_nearest(wi);
            }

            fs::create_directories(fs::path(bin_path).parent_path());
            save_test_data(bin_path, photons, caustic_photons, header);
            loaded_from_disk = false;
        }

        num_photons   = (int)header.num_photons_cfg;
        gather_radius = header.gather_radius;
        max_bounces   = (int)header.max_bounces;

        if (scene.triangles.empty()) {
            std::string obj_path = std::string(SCENES_DIR) + "/" + header.scene_path;
            if (!load_obj(obj_path, scene)) return;
            scene.build_bvh();
            scene.build_emissive_distribution();
        }
        if (scene.num_emissive() == 0) return;

        // 128×128 camera
        camera = Camera::cornell_box_camera(PIXEL_CMP_WIDTH, PIXEL_CMP_HEIGHT);
        camera.update();

        grid.build(photons, gather_radius);
        if (caustic_photons.size() > 0)
            caustic_grid.build(caustic_photons, header.caustic_radius);

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
        std::cout << "[PixelCmpDataset] Ready: " << photons.size() << " photons"
                  << " (" << PIXEL_CMP_WIDTH << "x" << PIXEL_CMP_HEIGHT << " camera)"
                  << " (from " << (loaded_from_disk ? "DISK" : "CPU") << ")\n";
    }
};

static PixelCmpDataset& get_pixel_dataset() {
    static PixelCmpDataset ds;
    ds.build();
    return ds;
}

// ─────────────────────────────────────────────────────────────────────
//  PER-PIXEL RESULT
// ─────────────────────────────────────────────────────────────────────

struct PixelResult {
    Spectrum nee_direct;        // Direct lighting NEE component
    Spectrum photon_indirect;   // Photon gather indirect component
    Spectrum combined;          // nee_direct + photon_indirect
    bool     is_shadow;         // NEE visibility = 0 (no direct light reached)
    bool     is_emissive_hit;   // First hit was emissive (light source)
    float    nee_visibility;    // Shadow ray success fraction

    PixelResult() {
        nee_direct      = Spectrum::zero();
        photon_indirect = Spectrum::zero();
        combined        = Spectrum::zero();
        is_shadow       = false;
        is_emissive_hit = false;
        nee_visibility  = 0.0f;
    }
};

// ─────────────────────────────────────────────────────────────────────
//  GROUND-TRUTH PATH TRACER  (first-diffuse-hit only, no continuation)
// ─────────────────────────────────────────────────────────────────────

static PixelResult ground_truth_trace_pixel(
    float3 origin, float3 direction, PCGRng& rng,
    const PixelCmpDataset& ds, int max_bounces)
{
    PixelResult result;

    Spectrum throughput = Spectrum::constant(1.0f);
    bool prev_was_specular = true;

    DensityEstimatorConfig de_cfg;
    de_cfg.radius            = ds.gather_radius;
    de_cfg.caustic_radius    = DEFAULT_CAUSTIC_RADIUS;
    de_cfg.num_photons_total = ds.num_photons;
    de_cfg.surface_tau       = DEFAULT_SURFACE_TAU;
    de_cfg.use_kernel        = true;

    Ray ray;
    ray.origin    = origin;
    ray.direction = direction;

    for (int bounce = 0; bounce <= max_bounces; ++bounce) {
        HitRecord hit = ds.scene.intersect(ray);
        if (!hit.hit) break;

        const Material& mat = ds.scene.materials[hit.material_id];

        // Emission from camera/specular paths
        if (mat.is_emissive()) {
            if (prev_was_specular) {
                Spectrum Le_contrib = throughput * mat.Le;
                result.combined   += Le_contrib;
                result.nee_direct += Le_contrib;
                result.is_emissive_hit = true;
            }
            break;
        }

        // Specular: mirror bounce, continue
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

        // NEE: single shadow ray
        {
            DirectLightSample dls = sample_direct_light(
                hit.position, hit.shading_normal, ds.scene, rng);

            result.nee_visibility = dls.visible ? 1.0f : 0.0f;
            result.is_shadow = !dls.visible;

            if (dls.visible && dls.pdf_light > 0.f) {
                float3 wi_local = frame.world_to_local(dls.wi);
                float cos_theta = fmaxf(0.f, wi_local.z);
                if (cos_theta > 0.f) {
                    Spectrum f = bsdf::evaluate(mat, wo_local, wi_local);
                    Spectrum contrib = dls.Li * f * (cos_theta / dls.pdf_light);
                    Spectrum nee_add = throughput * contrib;
                    result.combined   += nee_add;
                    result.nee_direct += nee_add;
                }
            }
        }

        // Photon density at first diffuse hit
        {
            Spectrum L_photon = estimate_photon_density(
                hit.position, hit.shading_normal, wo_local, mat,
                ds.photons, ds.grid, de_cfg, ds.gather_radius);

            Spectrum photon_add = throughput * L_photon;
            result.combined        += photon_add;
            result.photon_indirect += photon_add;
        }

        // STOP: no BSDF continuation (photon map has all indirect)
        break;
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────
//  OPTIMIZED PATH TRACER  (first-diffuse-hit only, no continuation)
// ─────────────────────────────────────────────────────────────────────

// Gather + local bins (mirrors GPU dev_estimate_photon_density_with_bins)
static Spectrum gather_with_local_bins(
    float3 pos, float3 normal, float3 wo_local,
    const Material& mat, const PixelCmpDataset& ds,
    PhotonBin* local_bins, int num_bins, float& total_bin_flux)
{
    Spectrum L = Spectrum::zero();
    total_bin_flux = 0.0f;

    for (int k = 0; k < num_bins; ++k) {
        local_bins[k].scalar_flux   = 0.0f;
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

            float3 wi_world = make_f3(ds.photons.wi_x[idx], ds.photons.wi_y[idx],
                                       ds.photons.wi_z[idx]);
            if (dot(wi_world, normal) <= 0.f) return;

            // Normal visibility filter (mirrors density_estimator.h)
            if (!ds.photons.norm_x.empty()) {
                float3 photon_n = make_f3(ds.photons.norm_x[idx], ds.photons.norm_y[idx],
                                          ds.photons.norm_z[idx]);
                if (dot(photon_n, normal) <= 0.f) return;
            }

            float3 wi_local = frame.world_to_local(wi_world);
            Spectrum f = bsdf::evaluate(mat, wo_local, wi_local);
            float flux = ds.photons.flux[idx];
            int bin = ds.photons.lambda_bin[idx];
            // Box kernel (w=1) — matches estimate_photon_density exactly
            L.value[bin] += flux * inv_N * f.value[bin] * inv_area;

            // Epanechnikov weight for bin population only
            float w = 1.0f - dist2 / r2;
            int k = (int)ds.photon_bin_idx[idx];
            if (k >= num_bins) k = 0;
            local_bins[k].scalar_flux   += flux * w;
            local_bins[k].dir_x  += wi_world.x * flux * w;
            local_bins[k].dir_y  += wi_world.y * flux * w;
            local_bins[k].dir_z  += wi_world.z * flux * w;
            local_bins[k].weight += w;
            local_bins[k].count  += 1;
            count++;
        });

    // No Epanechnikov correction needed — box kernel used for density

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
        total_bin_flux += local_bins[k].scalar_flux;
    }

    return L;
}

// Guided NEE (mirrors GPU)
static Spectrum optimized_nee_guided(
    float3 pos, float3 normal, float3 wo_local,
    const Material& mat, PCGRng& rng,
    const PixelCmpDataset& ds,
    const PhotonBin* bins, int N, const PhotonBinDirs& bin_dirs,
    float total_bin_flux, int bounce, float& visibility_out)
{
    visibility_out = 0.f;
    Spectrum L_nee = Spectrum::zero();
    const Scene& scene = ds.scene;
    if (scene.emissive_tri_indices.empty()) return L_nee;

    int num_emissive = (int)scene.emissive_tri_indices.size();
    bool use_guided = (total_bin_flux > 0.0f && num_emissive <= NEE_GUIDED_MAX_EMISSIVE);

    int M = 1;  // v3: single NEE sample per bounce

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

static PixelResult optimized_trace_pixel(
    float3 origin, float3 direction, PCGRng& rng,
    const PixelCmpDataset& ds, int max_bounces)
{
    PixelResult result;

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
                result.is_emissive_hit = true;
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

        // Photon gather + local bins
        PhotonBin local_bins[MAX_PHOTON_BIN_COUNT];
        float local_total_flux = 0.0f;

        Spectrum L_photon = gather_with_local_bins(
            hit.position, hit.shading_normal, wo_local, mat, ds,
            local_bins, PHOTON_BIN_COUNT, local_total_flux);

        bool have_bins = (local_total_flux > 0.0f);

        // Guided NEE
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

            result.nee_visibility = nee_visibility;
            result.is_shadow      = (nee_visibility <= 0.0f);

            Spectrum nee_contrib = throughput * L_nee;
            result.combined   += nee_contrib;
            result.nee_direct += nee_contrib;
        }

        // Photon indirect — no vis_weight suppression
        {
            Spectrum photon_contrib = throughput * L_photon;
            result.combined        += photon_contrib;
            result.photon_indirect += photon_contrib;
        }

        // STOP: no BSDF continuation (photon map has all indirect)
        break;
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────
//  PHOTON-LOBE DIRECTION TRACER  (for directional comparison)
// ─────────────────────────────────────────────────────────────────────

struct PhotonLobeResult {
    float3 mean_photon_dir;       // Flux-weighted mean direction from gathered photons
    float3 dominant_bin_dir;      // Direction of dominant photon bin
    float  total_gathered_flux;
    int    num_gathered;
    bool   valid;

    PhotonLobeResult() {
        mean_photon_dir    = make_f3(0, 0, 0);
        dominant_bin_dir   = make_f3(0, 0, 0);
        total_gathered_flux = 0.0f;
        num_gathered       = 0;
        valid              = false;
    }
};

static PhotonLobeResult trace_photon_lobe(
    float3 origin, float3 direction,
    const PixelCmpDataset& ds, int max_bounces)
{
    PhotonLobeResult result;
    Ray ray;
    ray.origin    = origin;
    ray.direction = direction;

    for (int bounce = 0; bounce <= max_bounces; ++bounce) {
        HitRecord hit = ds.scene.intersect(ray);
        if (!hit.hit) break;

        const Material& mat = ds.scene.materials[hit.material_id];
        if (mat.is_emissive()) break;

        if (mat.is_specular()) {
            ONB frame = ONB::from_normal(hit.shading_normal);
            float3 wo_local = frame.world_to_local(ray.direction * (-1.f));
            // Deterministic reflection (no RNG needed for perfect mirror)
            float3 wi_local = make_f3(-wo_local.x, -wo_local.y, wo_local.z);
            ray.origin    = hit.position + hit.shading_normal * EPSILON;
            ray.direction = frame.local_to_world(wi_local);
            continue;
        }

        // Diffuse hit — gather photons and compute directional info
        float3 normal = hit.shading_normal;
        float3 pos    = hit.position;

        // Raw photon direction gather
        float3 flux_weighted_dir = make_f3(0, 0, 0);
        float total_flux = 0.0f;
        int num = 0;

        ds.grid.query(pos, ds.gather_radius, ds.photons,
            [&](uint32_t idx, float dist2) {
                float3 pp = make_f3(ds.photons.pos_x[idx], ds.photons.pos_y[idx],
                                    ds.photons.pos_z[idx]);
                float3 diff = pp - pos;
                float plane_dist = fabsf(dot(normal, diff));
                if (plane_dist > DEFAULT_SURFACE_TAU) return;

                float3 wi_world = make_f3(ds.photons.wi_x[idx], ds.photons.wi_y[idx],
                                           ds.photons.wi_z[idx]);
                if (dot(wi_world, normal) <= 0.f) return;

                float flux = ds.photons.flux[idx];
                float r2 = ds.gather_radius * ds.gather_radius;
                float w = (1.0f - dist2 / r2) * flux;

                flux_weighted_dir.x += wi_world.x * w;
                flux_weighted_dir.y += wi_world.y * w;
                flux_weighted_dir.z += wi_world.z * w;
                total_flux += w;
                num++;
            });

        if (num > 0 && total_flux > 0.0f) {
            result.mean_photon_dir = flux_weighted_dir * (1.0f / total_flux);
            float len = length(result.mean_photon_dir);
            if (len > 1e-8f)
                result.mean_photon_dir = result.mean_photon_dir * (1.0f / len);

            result.total_gathered_flux = total_flux;
            result.num_gathered = num;

            // Also get the dominant bin direction
            PhotonBin local_bins[MAX_PHOTON_BIN_COUNT];
            float local_total = 0.0f;
            ONB frame = ONB::from_normal(normal);
            float3 wo_local = frame.world_to_local(ray.direction * (-1.f));
            (void)gather_with_local_bins(pos, normal, wo_local,
                mat, ds, local_bins, PHOTON_BIN_COUNT, local_total);

            int best_bin = 0;
            float best_flux = 0.0f;
            for (int k = 0; k < PHOTON_BIN_COUNT; ++k) {
                if (local_bins[k].scalar_flux > best_flux) {
                    best_flux = local_bins[k].scalar_flux;
                    best_bin  = k;
                }
            }
            if (best_flux > 0.0f) {
                result.dominant_bin_dir = make_f3(
                    local_bins[best_bin].dir_x,
                    local_bins[best_bin].dir_y,
                    local_bins[best_bin].dir_z);
                float dlen = length(result.dominant_bin_dir);
                if (dlen > 1e-8f)
                    result.dominant_bin_dir = result.dominant_bin_dir * (1.0f / dlen);
            }

            result.valid = true;
        }

        break;
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────
//  IMAGE RENDERING + METRICS
// ─────────────────────────────────────────────────────────────────────

// Extract luminance (CIE Y) from spectrum
static float spectrum_luminance(const Spectrum& s) {
    float3 xyz = spectrum_to_xyz(s);
    return xyz.y;
}

struct ImageMetrics {
    double rmse_luminance     = 0.0;
    double psnr_db            = 0.0;
    double mean_rel_error     = 0.0;
    double max_rel_error      = 0.0;
    double outlier_fraction   = 0.0;   // pixels with >2× relative error
    int    total_pixels       = 0;
    int    lit_pixels         = 0;     // pixels with nonzero GT luminance

    void print(const char* label) const {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "[" << label << "] RMSE=" << rmse_luminance
                  << " PSNR=" << psnr_db << "dB"
                  << " meanRelErr=" << mean_rel_error
                  << " maxRelErr=" << max_rel_error
                  << " outlierFrac=" << outlier_fraction
                  << " (" << lit_pixels << "/" << total_pixels << " lit)\n";
    }
};

static ImageMetrics compute_image_metrics(
    const std::vector<float>& gt_lum,
    const std::vector<float>& opt_lum,
    int total_pixels)
{
    ImageMetrics m;
    m.total_pixels = total_pixels;

    double sum_sq_err = 0.0;
    double sum_rel    = 0.0;
    double peak       = 0.0;
    int    lit        = 0;
    int    outliers   = 0;

    for (int i = 0; i < total_pixels; ++i) {
        double g = gt_lum[i];
        double o = opt_lum[i];
        double diff = g - o;
        sum_sq_err += diff * diff;

        if (g > peak) peak = g;
        if (o > peak) peak = o;

        if (g > 1e-8) {
            double rel = fabs(diff) / g;
            sum_rel += rel;
            if (rel > m.max_rel_error) m.max_rel_error = rel;
            if (rel > 2.0) outliers++;
            lit++;
        }
    }

    m.lit_pixels = lit;
    m.rmse_luminance = sqrt(sum_sq_err / total_pixels);
    m.psnr_db = (peak > 1e-10 && m.rmse_luminance > 1e-10)
              ? 20.0 * log10(peak / m.rmse_luminance)
              : 100.0;  // Perfect match
    m.mean_rel_error = (lit > 0) ? sum_rel / lit : 0.0;
    m.outlier_fraction = (lit > 0) ? (double)outliers / lit : 0.0;

    return m;
}

// ─────────────────────────────────────────────────────────────────────
//  RENDER FULL IMAGE (both methods)
// ─────────────────────────────────────────────────────────────────────

struct FullImageResult {
    std::vector<PixelResult> gt_pixels;
    std::vector<PixelResult> opt_pixels;
    int width;
    int height;

    int idx(int px, int py) const { return py * width + px; }
};

static FullImageResult render_comparison_image(const PixelCmpDataset& ds)
{
    FullImageResult img;
    img.width  = PIXEL_CMP_WIDTH;
    img.height = PIXEL_CMP_HEIGHT;
    int N = img.width * img.height;
    img.gt_pixels.resize(N);
    img.opt_pixels.resize(N);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int py = 0; py < img.height; ++py) {
        for (int px = 0; px < img.width; ++px) {
            PixelResult gt_acc, opt_acc;

            for (int s = 0; s < PIXEL_CMP_SPP; ++s) {
                uint64_t seed_a = (uint64_t)(py * img.width + px) * 31 + s * 7 + 12345;
                uint64_t seed_b = (uint64_t)s + 1;

                PCGRng rng_ray = PCGRng::seed(seed_a, seed_b);
                Ray ray = ds.camera.generate_ray(px, py, rng_ray);

                PCGRng rng_gt  = PCGRng::seed(seed_a + 1000000, seed_b);
                auto gt = ground_truth_trace_pixel(
                    ray.origin, ray.direction, rng_gt, ds, ds.max_bounces);

                PCGRng rng_opt = PCGRng::seed(seed_a + 2000000, seed_b);
                auto opt = optimized_trace_pixel(
                    ray.origin, ray.direction, rng_opt, ds, ds.max_bounces);

                gt_acc.combined        += gt.combined;
                gt_acc.nee_direct      += gt.nee_direct;
                gt_acc.photon_indirect += gt.photon_indirect;
                if (gt.is_shadow) gt_acc.is_shadow = true;
                if (gt.is_emissive_hit) gt_acc.is_emissive_hit = true;
                gt_acc.nee_visibility  += gt.nee_visibility;

                opt_acc.combined        += opt.combined;
                opt_acc.nee_direct      += opt.nee_direct;
                opt_acc.photon_indirect += opt.photon_indirect;
                if (opt.is_shadow) opt_acc.is_shadow = true;
                if (opt.is_emissive_hit) opt_acc.is_emissive_hit = true;
                opt_acc.nee_visibility  += opt.nee_visibility;
            }

            float inv_spp = 1.0f / (float)PIXEL_CMP_SPP;
            gt_acc.combined        *= inv_spp;
            gt_acc.nee_direct      *= inv_spp;
            gt_acc.photon_indirect *= inv_spp;
            gt_acc.nee_visibility  *= inv_spp;

            opt_acc.combined        *= inv_spp;
            opt_acc.nee_direct      *= inv_spp;
            opt_acc.photon_indirect *= inv_spp;
            opt_acc.nee_visibility  *= inv_spp;

            int i = img.idx(px, py);
            img.gt_pixels[i]  = gt_acc;
            img.opt_pixels[i] = opt_acc;
        }
    }

    return img;
}

// Singleton rendered image — computed once, reused by all tests
static FullImageResult& get_rendered_image() {
    static FullImageResult img;
    static bool rendered = false;
    if (!rendered) {
        auto& ds = get_pixel_dataset();
        if (ds.valid) {
            std::cout << "[PixelComparison] Rendering " << PIXEL_CMP_WIDTH
                      << "x" << PIXEL_CMP_HEIGHT << " @ " << PIXEL_CMP_SPP
                      << " spp (both methods)...\n";
            img = render_comparison_image(ds);
            std::cout << "[PixelComparison] Render complete.\n";
        }
        rendered = true;
    }
    return img;
}

// =====================================================================
//  TEST SUITE: PixelComparison
// =====================================================================

// ── Dataset validity ─────────────────────────────────────────────────
TEST(PixelComparison, DatasetIsValid) {
    auto& ds = get_pixel_dataset();
    ASSERT_TRUE(ds.valid) << "Pixel comparison dataset failed to build";
    EXPECT_GT(ds.scene.triangles.size(), 10000u);
    EXPECT_GT(ds.photons.size(), 1000u);
    EXPECT_GT(ds.scene.num_emissive(), 0);
    EXPECT_EQ(ds.camera.width, PIXEL_CMP_WIDTH);
    EXPECT_EQ(ds.camera.height, PIXEL_CMP_HEIGHT);
}

// ── 1. NEE first-hit comparison (direct lighting only) ───────────────
TEST(PixelComparison, NEEFirstHitComparison) {
    auto& img = get_rendered_image();
    int N = img.width * img.height;
    ASSERT_GT(N, 0);

    std::vector<float> gt_lum(N), opt_lum(N);
    for (int i = 0; i < N; ++i) {
        gt_lum[i]  = spectrum_luminance(img.gt_pixels[i].nee_direct);
        opt_lum[i] = spectrum_luminance(img.opt_pixels[i].nee_direct);
    }

    auto m = compute_image_metrics(gt_lum, opt_lum, N);
    m.print("NEE-DirectOnly");

    EXPECT_LT(m.rmse_luminance, MAX_RMSE_LUMINANCE * 2.0)
        << "NEE direct RMSE too high";
    EXPECT_LT(m.mean_rel_error, MAX_MEAN_REL_ERROR * 1.5)
        << "NEE direct mean relative error too high";
}

// ── 2. Photon irradiance at first hit (indirect only) ────────────────
TEST(PixelComparison, PhotonIrradianceFirstHit) {
    auto& img = get_rendered_image();
    int N = img.width * img.height;
    ASSERT_GT(N, 0);

    std::vector<float> gt_lum(N), opt_lum(N);
    for (int i = 0; i < N; ++i) {
        gt_lum[i]  = spectrum_luminance(img.gt_pixels[i].photon_indirect);
        opt_lum[i] = spectrum_luminance(img.opt_pixels[i].photon_indirect);
    }

    auto m = compute_image_metrics(gt_lum, opt_lum, N);
    m.print("PhotonIndirect");

    EXPECT_LT(m.rmse_luminance, MAX_RMSE_LUMINANCE)
        << "Photon indirect RMSE too high";
    EXPECT_LT(m.mean_rel_error, MAX_MEAN_REL_ERROR)
        << "Photon indirect mean relative error too high";
}

// ── 3. Combined radiance (NEE + photon) ──────────────────────────────
TEST(PixelComparison, CombinedRadianceComparison) {
    auto& img = get_rendered_image();
    int N = img.width * img.height;
    ASSERT_GT(N, 0);

    std::vector<float> gt_lum(N), opt_lum(N);
    for (int i = 0; i < N; ++i) {
        gt_lum[i]  = spectrum_luminance(img.gt_pixels[i].combined);
        opt_lum[i] = spectrum_luminance(img.opt_pixels[i].combined);
    }

    auto m = compute_image_metrics(gt_lum, opt_lum, N);
    m.print("Combined   ");

    EXPECT_LT(m.rmse_luminance, MAX_RMSE_LUMINANCE)
        << "Combined RMSE too high";
    EXPECT_GT(m.psnr_db, MIN_PSNR_DB)
        << "Combined PSNR too low";
    EXPECT_LT(m.mean_rel_error, MAX_MEAN_REL_ERROR)
        << "Combined mean relative error too high";
    EXPECT_LT(m.outlier_fraction, MAX_OUTLIER_FRACTION)
        << "Too many outlier pixels (>2× error)";
}

// ── 4. Photon-lobe directional accuracy vs actual photon dirs ────────
TEST(PixelComparison, PhotonLobeVsActualDirections) {
    auto& ds = get_pixel_dataset();
    ASSERT_TRUE(ds.valid);

    // Sample a grid of pixels and compare bin-lobe direction to actual
    int tested = 0;
    int agreed = 0;
    float total_cos_sim = 0.0f;

    constexpr int STEP = 8;  // test every 8th pixel
    for (int py = 0; py < PIXEL_CMP_HEIGHT; py += STEP) {
        for (int px = 0; px < PIXEL_CMP_WIDTH; px += STEP) {
            uint64_t seed_a = (uint64_t)(py * PIXEL_CMP_WIDTH + px) * 31 + 12345;
            PCGRng rng_ray = PCGRng::seed(seed_a, 1);
            Ray ray = ds.camera.generate_ray(px, py, rng_ray);

            auto lobe = trace_photon_lobe(ray.origin, ray.direction, ds, ds.max_bounces);
            if (!lobe.valid || lobe.num_gathered < 5) continue;

            tested++;

            // Compare mean_photon_dir with dominant_bin_dir
            float cos_sim = dot(lobe.mean_photon_dir, lobe.dominant_bin_dir);
            total_cos_sim += cos_sim;

            // Agreement: cosine similarity > 0.5 (within ~60 degrees)
            if (cos_sim > 0.5f) agreed++;
        }
    }

    if (tested > 0) {
        float avg_cos = total_cos_sim / tested;
        float agree_frac = (float)agreed / tested;

        std::cout << "[PhotonLobe] tested=" << tested
                  << " avgCosSim=" << avg_cos
                  << " agreeFrac=" << agree_frac << "\n";

        EXPECT_GT(agree_frac, 0.50f)
            << "Photon bin lobes poorly match actual photon directions ("
            << (int)(agree_frac * 100) << "% agree)";
        EXPECT_GT(avg_cos, 0.30f)
            << "Average cosine similarity between bin lobe and actual direction too low";
    } else {
        std::cout << "[PhotonLobe] No valid pixels tested (scene may have no diffuse surfaces)\n";
    }
}

// ── 5. Shadow region indirect fidelity ───────────────────────────────
// After the vis_weight fix, shadow regions should have matching
// indirect illumination between GT and optimized.
TEST(PixelComparison, ShadowRegionIndirectFidelity) {
    auto& img = get_rendered_image();
    int N = img.width * img.height;
    ASSERT_GT(N, 0);

    double gt_shadow_sum  = 0.0;
    double opt_shadow_sum = 0.0;
    int shadow_pixels     = 0;

    for (int i = 0; i < N; ++i) {
        // A pixel is "in shadow" if ANY sample had shadow
        bool gt_shadow  = img.gt_pixels[i].is_shadow;
        bool opt_shadow = img.opt_pixels[i].is_shadow;

        if (gt_shadow || opt_shadow) {
            float gt_lum  = spectrum_luminance(img.gt_pixels[i].photon_indirect);
            float opt_lum = spectrum_luminance(img.opt_pixels[i].photon_indirect);
            gt_shadow_sum  += gt_lum;
            opt_shadow_sum += opt_lum;
            shadow_pixels++;
        }
    }

    std::cout << "[ShadowIndirect] shadow_pixels=" << shadow_pixels
              << "/" << N << "\n";

    if (shadow_pixels > 10 && gt_shadow_sum > 1e-6) {
        double gt_mean  = gt_shadow_sum / shadow_pixels;
        double opt_mean = opt_shadow_sum / shadow_pixels;
        double rel_err  = fabs(opt_mean - gt_mean) / gt_mean;

        std::cout << "[ShadowIndirect] GT mean=" << gt_mean
                  << " Opt mean=" << opt_mean
                  << " relErr=" << rel_err << "\n";

        EXPECT_LT(rel_err, MAX_SHADOW_INDIRECT_DIFF)
            << "Shadow region indirect illumination differs by "
            << (int)(rel_err * 100) << "% — vis_weight suppression may still be active";
    } else {
        std::cout << "[ShadowIndirect] Too few shadow pixels to test\n";
    }
}

// ── 6. Per-spectral-bin pixel error distribution ─────────────────────
TEST(PixelComparison, SpectralBinPixelError) {
    auto& img = get_rendered_image();
    int N = img.width * img.height;
    ASSERT_GT(N, 0);

    // For each spectral bin, compute RMSE across all pixels
    int bins_with_high_error = 0;

    std::cout << "[SpectralBin] Per-bin RMSE: ";
    for (int b = 0; b < NUM_LAMBDA; ++b) {
        double sum_sq = 0.0;
        double gt_max = 0.0;
        for (int i = 0; i < N; ++i) {
            double g = img.gt_pixels[i].combined.value[b];
            double o = img.opt_pixels[i].combined.value[b];
            double d = g - o;
            sum_sq += d * d;
            if (g > gt_max) gt_max = g;
        }
        double rmse = sqrt(sum_sq / N);
        double norm_rmse = (gt_max > 1e-8) ? rmse / gt_max : 0.0;
        if (norm_rmse > 0.40) bins_with_high_error++;

        if (b < 8 || b >= NUM_LAMBDA - 4)
            std::cout << std::fixed << std::setprecision(3) << norm_rmse << " ";
        else if (b == 8)
            std::cout << "... ";
    }
    std::cout << "\n";

    std::cout << "[SpectralBin] Bins with >40% normalized RMSE: "
              << bins_with_high_error << "/" << NUM_LAMBDA << "\n";

    EXPECT_LT(bins_with_high_error, NUM_LAMBDA / 4)
        << "Too many spectral bins have high pixel-level error";
}

// ── 7. Global PSNR and energy ratio ──────────────────────────────────
TEST(PixelComparison, GlobalPSNRAndEnergyRatio) {
    auto& img = get_rendered_image();
    int N = img.width * img.height;
    ASSERT_GT(N, 0);

    double gt_total  = 0.0;
    double opt_total = 0.0;
    std::vector<float> gt_lum(N), opt_lum(N);

    for (int i = 0; i < N; ++i) {
        gt_lum[i]  = spectrum_luminance(img.gt_pixels[i].combined);
        opt_lum[i] = spectrum_luminance(img.opt_pixels[i].combined);
        gt_total  += gt_lum[i];
        opt_total += opt_lum[i];
    }

    auto m = compute_image_metrics(gt_lum, opt_lum, N);

    double energy_ratio = (gt_total > 1e-6) ? opt_total / gt_total : 0.0;
    std::cout << "[Global] PSNR=" << m.psnr_db << "dB"
              << " energyRatio=" << energy_ratio
              << " GT_total=" << gt_total
              << " Opt_total=" << opt_total << "\n";

    EXPECT_GT(m.psnr_db, MIN_PSNR_DB)
        << "Global PSNR too low";
    EXPECT_GT(energy_ratio, 0.70)
        << "Optimized loses too much energy";
    EXPECT_LT(energy_ratio, 1.40)
        << "Optimized creates too much energy";
}

// ── 8. Component decomposition (combined ≈ nee + photon) ─────────────
TEST(PixelComparison, ComponentDecomposition) {
    auto& img = get_rendered_image();
    int N = img.width * img.height;
    ASSERT_GT(N, 0);

    int gt_bad  = 0;
    int opt_bad = 0;

    for (int i = 0; i < N; ++i) {
        // GT decomposition check
        float gt_sum     = spectrum_luminance(img.gt_pixels[i].combined);
        float gt_nee     = spectrum_luminance(img.gt_pixels[i].nee_direct);
        float gt_photon  = spectrum_luminance(img.gt_pixels[i].photon_indirect);
        float gt_decomp  = gt_nee + gt_photon;
        if (gt_sum > 1e-6f) {
            float err = fabsf(gt_sum - gt_decomp) / gt_sum;
            if (err > 0.05f) gt_bad++;
        }

        // Opt decomposition check
        float opt_sum     = spectrum_luminance(img.opt_pixels[i].combined);
        float opt_nee     = spectrum_luminance(img.opt_pixels[i].nee_direct);
        float opt_photon  = spectrum_luminance(img.opt_pixels[i].photon_indirect);
        float opt_decomp  = opt_nee + opt_photon;
        if (opt_sum > 1e-6f) {
            float err = fabsf(opt_sum - opt_decomp) / opt_sum;
            if (err > 0.05f) opt_bad++;
        }
    }

    float gt_bad_frac  = (float)gt_bad / N;
    float opt_bad_frac = (float)opt_bad / N;

    std::cout << "[Decomposition] GT bad=" << gt_bad
              << " (" << (gt_bad_frac * 100) << "%)"
              << " Opt bad=" << opt_bad
              << " (" << (opt_bad_frac * 100) << "%)\n";

    EXPECT_LT(gt_bad_frac, 0.05f)
        << "GT combined != nee + photon for too many pixels";
    EXPECT_LT(opt_bad_frac, 0.05f)
        << "Opt combined != nee + photon for too many pixels";
}

// ── 9. Spatial uniformity (no systematic dark/bright regions) ────────
TEST(PixelComparison, SpatialUniformity) {
    auto& img = get_rendered_image();
    ASSERT_GT(img.width * img.height, 0);

    // Divide image into 4×4 blocks and compare block means
    constexpr int BLOCKS = 4;
    int bw = img.width  / BLOCKS;
    int bh = img.height / BLOCKS;

    int blocks_with_large_diff = 0;

    for (int by = 0; by < BLOCKS; ++by) {
        for (int bx = 0; bx < BLOCKS; ++bx) {
            double gt_sum = 0.0, opt_sum = 0.0;
            int count = 0;

            for (int py = by * bh; py < (by + 1) * bh && py < img.height; ++py) {
                for (int px = bx * bw; px < (bx + 1) * bw && px < img.width; ++px) {
                    int i = img.idx(px, py);
                    gt_sum  += spectrum_luminance(img.gt_pixels[i].combined);
                    opt_sum += spectrum_luminance(img.opt_pixels[i].combined);
                    count++;
                }
            }

            if (count > 0 && gt_sum > 1e-6) {
                double gt_mean  = gt_sum / count;
                double opt_mean = opt_sum / count;
                double rel = fabs(opt_mean - gt_mean) / gt_mean;
                if (rel > 0.50) blocks_with_large_diff++;
            }
        }
    }

    std::cout << "[Spatial] Blocks with >50% difference: "
              << blocks_with_large_diff << "/" << (BLOCKS * BLOCKS) << "\n";

    EXPECT_LT(blocks_with_large_diff, BLOCKS * BLOCKS / 2)
        << "Too many spatial blocks have large GT vs Opt difference";
}

// ── 10. No negative pixel values ─────────────────────────────────────
TEST(PixelComparison, NoNegativePixels) {
    auto& img = get_rendered_image();
    int N = img.width * img.height;
    ASSERT_GT(N, 0);

    int gt_neg  = 0;
    int opt_neg = 0;

    for (int i = 0; i < N; ++i) {
        for (int b = 0; b < NUM_LAMBDA; ++b) {
            if (img.gt_pixels[i].combined.value[b] < -1e-6f) { gt_neg++; break; }
        }
        for (int b = 0; b < NUM_LAMBDA; ++b) {
            if (img.opt_pixels[i].combined.value[b] < -1e-6f) { opt_neg++; break; }
        }
    }

    EXPECT_EQ(gt_neg, 0)
        << gt_neg << " GT pixels have negative spectral values";
    EXPECT_EQ(opt_neg, 0)
        << opt_neg << " Opt pixels have negative spectral values";
}

// ── 11. All pixel values finite ──────────────────────────────────────
TEST(PixelComparison, AllPixelsFinite) {
    auto& img = get_rendered_image();
    int N = img.width * img.height;
    ASSERT_GT(N, 0);

    int gt_inf  = 0;
    int opt_inf = 0;

    for (int i = 0; i < N; ++i) {
        for (int b = 0; b < NUM_LAMBDA; ++b) {
            if (!std::isfinite(img.gt_pixels[i].combined.value[b])) { gt_inf++; break; }
        }
        for (int b = 0; b < NUM_LAMBDA; ++b) {
            if (!std::isfinite(img.opt_pixels[i].combined.value[b])) { opt_inf++; break; }
        }
    }

    EXPECT_EQ(gt_inf, 0)
        << gt_inf << " GT pixels have non-finite spectral values";
    EXPECT_EQ(opt_inf, 0)
        << opt_inf << " Opt pixels have non-finite spectral values";
}

// ─────────────────────────────────────────────────────────────────────
//  PPM IMAGE WRITER  (no external dependencies)
// ─────────────────────────────────────────────────────────────────────

static std::string get_output_dir() {
    // If PPT_TEST_OUTPUT_DIR is set, use it; otherwise default to
    // <repo>/tests/output
#ifdef _MSC_VER
    char* env = nullptr;
    size_t env_len = 0;
    _dupenv_s(&env, &env_len, "PPT_TEST_OUTPUT_DIR");
    std::string result;
    if (env && env[0]) { result = env; free(env); return result; }
    free(env);
#else
    const char* env = std::getenv("PPT_TEST_OUTPUT_DIR");
    if (env && env[0]) return std::string(env);
#endif
    fs::path scenes(SCENES_DIR);
    return (scenes.parent_path() / "tests" / "output").string();
}

static bool write_ppm(const std::string& path,
                       const std::vector<uint8_t>& rgb,
                       int w, int h)
{
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << "P6\n" << w << " " << h << "\n255\n";
    f.write(reinterpret_cast<const char*>(rgb.data()), (std::streamsize)(w * h * 3));
    return f.good();
}

// Tone-map a spectrum array to sRGB uint8 image.
// Exposure is auto-computed from the 99th-percentile luminance.
static std::vector<uint8_t> spectrum_to_rgb_image(
    const std::vector<PixelResult>& pixels,
    int w, int h,
    const std::function<Spectrum(const PixelResult&)>& channel)
{
    int N = w * h;
    // Compute luminances for autoexposure
    std::vector<float> lums(N);
    for (int i = 0; i < N; ++i)
        lums[i] = spectrum_luminance(channel(pixels[i]));

    std::vector<float> sorted_lums = lums;
    std::sort(sorted_lums.begin(), sorted_lums.end());
    float p99 = sorted_lums[std::min(N - 1, (int)(N * 0.99f))];
    float exposure = (p99 > 1e-6f) ? 1.0f / p99 : 1.0f;

    std::vector<uint8_t> rgb(N * 3);
    for (int i = 0; i < N; ++i) {
        Spectrum s = channel(pixels[i]) * exposure;
        float3 c = spectrum_to_srgb(s);
        rgb[i * 3 + 0] = (uint8_t)std::min(255, std::max(0, (int)(c.x * 255.f + 0.5f)));
        rgb[i * 3 + 1] = (uint8_t)std::min(255, std::max(0, (int)(c.y * 255.f + 0.5f)));
        rgb[i * 3 + 2] = (uint8_t)std::min(255, std::max(0, (int)(c.z * 255.f + 0.5f)));
    }
    return rgb;
}

// Difference heatmap: black = 0 error, red = large error, white = huge error.
static std::vector<uint8_t> diff_heatmap(
    const std::vector<PixelResult>& gt,
    const std::vector<PixelResult>& opt,
    int w, int h,
    const std::function<Spectrum(const PixelResult&)>& channel,
    float scale = 0.0f)
{
    int N = w * h;
    std::vector<float> errs(N);
    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float gl = spectrum_luminance(channel(gt[i]));
        float ol = spectrum_luminance(channel(opt[i]));
        errs[i] = fabsf(gl - ol);
        if (errs[i] > max_err) max_err = errs[i];
    }
    if (scale <= 0.0f) scale = (max_err > 1e-8f) ? max_err : 1.0f;

    std::vector<uint8_t> rgb(N * 3);
    for (int i = 0; i < N; ++i) {
        float t = std::min(1.0f, errs[i] / scale);
        // Black → Red → Yellow → White
        float r, g, b;
        if (t < 0.33f) {
            r = t / 0.33f; g = 0; b = 0;
        } else if (t < 0.66f) {
            r = 1; g = (t - 0.33f) / 0.33f; b = 0;
        } else {
            r = 1; g = 1; b = (t - 0.66f) / 0.34f;
        }
        rgb[i * 3 + 0] = (uint8_t)(r * 255);
        rgb[i * 3 + 1] = (uint8_t)(g * 255);
        rgb[i * 3 + 2] = (uint8_t)(b * 255);
    }
    return rgb;
}

// Shadow mask: blue = shadow, gray = lit
static std::vector<uint8_t> shadow_mask_image(
    const std::vector<PixelResult>& pixels, int w, int h)
{
    int N = w * h;
    std::vector<uint8_t> rgb(N * 3);
    for (int i = 0; i < N; ++i) {
        if (pixels[i].is_shadow) {
            rgb[i * 3 + 0] = 30;
            rgb[i * 3 + 1] = 60;
            rgb[i * 3 + 2] = 180;
        } else {
            rgb[i * 3 + 0] = 160;
            rgb[i * 3 + 1] = 160;
            rgb[i * 3 + 2] = 160;
        }
    }
    return rgb;
}

// ── 12. Save comparison renderings to output directory ───────────────
TEST(PixelComparison, SaveComparisonImages) {
    auto& img = get_rendered_image();
    int N = img.width * img.height;
    ASSERT_GT(N, 0);

    std::string out_dir = get_output_dir();
    std::cout << "[SaveImages] Output directory: " << out_dir << "\n";

    auto combined   = [](const PixelResult& p) { return p.combined; };
    auto nee        = [](const PixelResult& p) { return p.nee_direct; };
    auto photon_ind = [](const PixelResult& p) { return p.photon_indirect; };

    int saved = 0;
    int w = img.width, h = img.height;

    // Ground truth images
    auto gt_combined_rgb = spectrum_to_rgb_image(img.gt_pixels, w, h, combined);
    if (write_ppm(out_dir + "/gt_combined.ppm", gt_combined_rgb, w, h)) saved++;

    auto gt_nee_rgb = spectrum_to_rgb_image(img.gt_pixels, w, h, nee);
    if (write_ppm(out_dir + "/gt_nee_direct.ppm", gt_nee_rgb, w, h)) saved++;

    auto gt_photon_rgb = spectrum_to_rgb_image(img.gt_pixels, w, h, photon_ind);
    if (write_ppm(out_dir + "/gt_photon_indirect.ppm", gt_photon_rgb, w, h)) saved++;

    // Optimized images
    auto opt_combined_rgb = spectrum_to_rgb_image(img.opt_pixels, w, h, combined);
    if (write_ppm(out_dir + "/opt_combined.ppm", opt_combined_rgb, w, h)) saved++;

    auto opt_nee_rgb = spectrum_to_rgb_image(img.opt_pixels, w, h, nee);
    if (write_ppm(out_dir + "/opt_nee_direct.ppm", opt_nee_rgb, w, h)) saved++;

    auto opt_photon_rgb = spectrum_to_rgb_image(img.opt_pixels, w, h, photon_ind);
    if (write_ppm(out_dir + "/opt_photon_indirect.ppm", opt_photon_rgb, w, h)) saved++;

    // Difference heatmaps
    auto diff_combined_rgb = diff_heatmap(img.gt_pixels, img.opt_pixels, w, h, combined);
    if (write_ppm(out_dir + "/diff_combined.ppm", diff_combined_rgb, w, h)) saved++;

    auto diff_nee_rgb = diff_heatmap(img.gt_pixels, img.opt_pixels, w, h, nee);
    if (write_ppm(out_dir + "/diff_nee_direct.ppm", diff_nee_rgb, w, h)) saved++;

    auto diff_photon_rgb = diff_heatmap(img.gt_pixels, img.opt_pixels, w, h, photon_ind);
    if (write_ppm(out_dir + "/diff_photon_indirect.ppm", diff_photon_rgb, w, h)) saved++;

    // Shadow mask
    auto shadow_rgb = shadow_mask_image(img.gt_pixels, w, h);
    if (write_ppm(out_dir + "/shadow_mask.ppm", shadow_rgb, w, h)) saved++;

    std::cout << "[SaveImages] Wrote " << saved << "/10 images to " << out_dir << "\n";
    EXPECT_EQ(saved, 10) << "Failed to write some comparison images";
}
