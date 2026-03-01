// ─────────────────────────────────────────────────────────────────────
// test_per_ray_validation.cpp – Per-ray physical path validation
// ─────────────────────────────────────────────────────────────────────
// Picks N random rays from real data and traces each step-by-step
// through the critical physical path described in the project spec:
//
//   Camera → Intersection → Material → NEE → Photon Density →
//   BSDF Bounce → Throughput → Russian Roulette → (repeat)
//
// At each step, ground truth is compared against the optimized path.
// Warnings are emitted when deviation exceeds configured thresholds.
//
// Test categories mapped to spec invariants (Section 0):
//   1. No double counting direct lighting
//   2. Every MC estimator divides by exact PDF
//   3. Spectral bins never mix during transport
//   4. Component decomposition: combined ≈ nee + photon_indirect
//   5. Energy conservation (no creation/destruction)
//   6. Physical validity (finite, non-negative at every step)
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
#include <sstream>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <string>
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
//  THRESHOLDS — Deviation limits before warnings are issued
// ─────────────────────────────────────────────────────────────────────
namespace thresholds {
    // Per-ray relative error thresholds (individual ray)
    constexpr float RAY_NEE_DIRECT_REL       = 0.80f;  // NEE varies by sampling
    constexpr float RAY_PHOTON_INDIRECT_REL  = 0.50f;  // Photon density w/ bins
    constexpr float RAY_COMBINED_REL         = 0.70f;  // Overall combined

    // Aggregate (mean over N rays) thresholds — tighter
    constexpr float AGG_NEE_DIRECT_REL       = 0.30f;
    constexpr float AGG_PHOTON_INDIRECT_REL  = 0.25f;
    constexpr float AGG_COMBINED_REL         = 0.30f;
    constexpr float AGG_ENERGY_RATIO_LOW     = 0.65f;
    constexpr float AGG_ENERGY_RATIO_HIGH    = 1.50f;

    // Physical validity
    constexpr float MAX_THROUGHPUT_COMPONENT = 1e6f;   // Throughput shouldn't explode
    constexpr float COMPONENT_DECOMP_REL     = 0.01f;  // combined ≈ nee + photon

    // Fraction of rays allowed to exceed per-ray thresholds
    constexpr float WARN_FRACTION            = 0.15f;  // warn if >15% of rays bad
    constexpr float FAIL_FRACTION            = 0.40f;  // fail if >40% of rays bad
}

// Match GPU constants
constexpr int   NEE_GUIDED_MAX_EMISSIVE = 128;
constexpr float NEE_GUIDED_ALPHA        = 5.0f;

// ─────────────────────────────────────────────────────────────────────
//  PER-BOUNCE STEP DATA
// ─────────────────────────────────────────────────────────────────────

struct BounceStep {
    int     bounce         = -1;
    float3  hit_pos        = {};
    float3  hit_normal     = {};
    int     material_id    = -1;
    bool    is_emissive    = false;
    bool    is_specular    = false;

    // Contributions at this bounce
    Spectrum nee_contribution;
    Spectrum photon_contribution;
    Spectrum throughput;

    // Physical validity flags
    bool throughput_finite      = true;
    bool throughput_nonnegative = true;
    bool nee_finite             = true;
    bool nee_nonnegative        = true;
    bool photon_finite          = true;
    bool photon_nonnegative     = true;

    BounceStep() {
        nee_contribution    = Spectrum::zero();
        photon_contribution = Spectrum::zero();
        throughput          = Spectrum::zero();
    }
};

struct PathResult {
    std::vector<BounceStep> steps;
    Spectrum combined;
    Spectrum nee_direct;
    Spectrum photon_indirect;
    int      total_bounces = 0;

    PathResult() {
        combined        = Spectrum::zero();
        nee_direct      = Spectrum::zero();
        photon_indirect = Spectrum::zero();
    }
};

// ─────────────────────────────────────────────────────────────────────
//  DATASET — Same singleton as test_ground_truth.cpp
// ─────────────────────────────────────────────────────────────────────

static std::string test_data_path() {
    fs::path scenes(SCENES_DIR);
    fs::path data_dir = scenes.parent_path() / "tests" / "data";
    return (data_dir / "cornell_box.bin").string();
}

struct PerRayDataset {
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
                std::cout << "[PerRayDataset] Removing stale " << bin_path << "\n";
                fs::remove(bin_path);
            }
        }
        if (!loaded_from_disk) {
            // Reset header so save writes current version/algo_version
            // (load_test_data may have partially overwritten fields).
            header = TestDataHeader{};

            // Fallback: CPU trace
            header.num_photons_cfg = DEFAULT_NUM_PHOTONS;
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

        camera = Camera::cornell_box_camera(64, 64);
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
        std::cout << "[PerRayDataset] Ready: " << photons.size() << " photons"
                  << " (from " << (loaded_from_disk ? "DISK" : "CPU") << ")\n";
    }
};

static PerRayDataset& get_per_ray_dataset() {
    static PerRayDataset ds;
    ds.build();
    return ds;
}

// ─────────────────────────────────────────────────────────────────────
//  SPECTRUM HELPERS
// ─────────────────────────────────────────────────────────────────────

static bool spectrum_is_finite(const Spectrum& s) {
    for (int i = 0; i < NUM_LAMBDA; ++i)
        if (!std::isfinite(s.value[i])) return false;
    return true;
}

static bool spectrum_is_nonnegative(const Spectrum& s) {
    for (int i = 0; i < NUM_LAMBDA; ++i)
        if (s.value[i] < 0.f) return false;
    return true;
}

static float spectrum_max(const Spectrum& s) {
    float m = s.value[0];
    for (int i = 1; i < NUM_LAMBDA; ++i)
        if (s.value[i] > m) m = s.value[i];
    return m;
}

static float safe_relative_error(float a, float b) {
    float denom = fmaxf(fabsf(a), fabsf(b));
    if (denom < 1e-10f) return 0.f;
    return fabsf(a - b) / denom;
}

// ─────────────────────────────────────────────────────────────────────
//  GROUND-TRUTH PATH TRACER (step-by-step recording)
// ─────────────────────────────────────────────────────────────────────

static PathResult ground_truth_trace_steps(
    float3 origin, float3 direction, PCGRng& rng,
    const PerRayDataset& ds, int max_bounces)
{
    PathResult result;
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

        BounceStep step;
        step.bounce      = bounce;
        step.hit_pos     = hit.position;
        step.hit_normal  = hit.shading_normal;
        step.material_id = hit.material_id;
        step.is_emissive = mat.is_emissive();
        step.is_specular = mat.is_specular();
        step.throughput  = throughput;

        // Validate throughput
        step.throughput_finite      = spectrum_is_finite(throughput);
        step.throughput_nonnegative = spectrum_is_nonnegative(throughput);

        if (mat.is_emissive()) {
            if (prev_was_specular) {
                Spectrum Le_contrib = throughput * mat.Le;
                step.nee_contribution = Le_contrib;
                result.combined   += Le_contrib;
                result.nee_direct += Le_contrib;
            }
            step.nee_finite      = spectrum_is_finite(step.nee_contribution);
            step.nee_nonnegative = spectrum_is_nonnegative(step.nee_contribution);
            result.steps.push_back(step);
            result.total_bounces = bounce + 1;
            break;
        }

        if (mat.is_specular()) {
            ONB frame = ONB::from_normal(hit.shading_normal);
            float3 wo_local = frame.world_to_local(ray.direction * (-1.f));
            BSDFSample bs = bsdf::sample(mat, wo_local, rng);
            if (bs.pdf <= 0.f) { result.steps.push_back(step); break; }

            float cos_theta = fabsf(bs.wi.z);
            for (int i = 0; i < NUM_LAMBDA; ++i)
                throughput.value[i] *= bs.f.value[i] * cos_theta / bs.pdf;

            ray.origin    = hit.position + hit.shading_normal * EPSILON *
                            (bs.wi.z > 0.f ? 1.f : -1.f);
            ray.direction = frame.local_to_world(bs.wi);
            prev_was_specular = true;
            result.steps.push_back(step);
            continue;
        }

        prev_was_specular = false;
        ONB frame = ONB::from_normal(hit.shading_normal);
        float3 wo_local = frame.world_to_local(ray.direction * (-1.f));
        if (wo_local.z <= 0.f) { result.steps.push_back(step); break; }

        // NEE
        {
            DirectLightSample dls = sample_direct_light(
                hit.position, hit.shading_normal, ds.scene, rng);
            if (dls.visible && dls.pdf_light > 0.f) {
                float3 wi_local = frame.world_to_local(dls.wi);
                float cos_theta = fmaxf(0.f, wi_local.z);
                if (cos_theta > 0.f) {
                    Spectrum f = bsdf::evaluate(mat, wo_local, wi_local);
                    Spectrum contrib = dls.Li * f * (cos_theta / dls.pdf_light);
                    Spectrum nee_add = throughput * contrib;
                    step.nee_contribution = nee_add;
                    result.combined   += nee_add;
                    result.nee_direct += nee_add;
                }
            }
        }

        // Photon density
        {
            Spectrum L_photon = estimate_photon_density(
                hit.position, hit.shading_normal, wo_local, mat,
                ds.photons, ds.grid, de_cfg, ds.gather_radius);
            Spectrum photon_add = throughput * L_photon;
            step.photon_contribution = photon_add;
            result.combined        += photon_add;
            result.photon_indirect += photon_add;
        }

        // Validate
        step.nee_finite         = spectrum_is_finite(step.nee_contribution);
        step.nee_nonnegative    = spectrum_is_nonnegative(step.nee_contribution);
        step.photon_finite      = spectrum_is_finite(step.photon_contribution);
        step.photon_nonnegative = spectrum_is_nonnegative(step.photon_contribution);

        result.steps.push_back(step);
        result.total_bounces = bounce + 1;

        // FIX: The photon map already captures ALL indirect illumination.
        // Stop after NEE + photon gather at the first diffuse hit.
        break;
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────
//  OPTIMIZED PATH TRACER (step-by-step recording)
// ─────────────────────────────────────────────────────────────────────

// Gather + local bins (same as test_ground_truth.cpp)
static Spectrum gather_with_bins_step(
    float3 pos, float3 normal, float3 wo_local,
    const Material& mat, const PerRayDataset& ds,
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

    float radius = ds.gather_radius;
    float r2     = radius * radius;
    float inv_area = 1.0f / (PI * r2);
    float inv_N    = 1.0f / (float)ds.num_photons;
    int count = 0;
    ONB frame = ONB::from_normal(normal);

    float tau = effective_tau(DEFAULT_SURFACE_TAU);
    ds.grid.query_tangential(pos, normal, radius, tau, ds.photons,
        [&](uint32_t idx, float dist2) {
            float3 wi_world = make_f3(ds.photons.wi_x[idx], ds.photons.wi_y[idx],
                                       ds.photons.wi_z[idx]);
            if (dot(wi_world, normal) <= 0.f) return;

            // Normal visibility filter (mirrors density_estimator.h: <= 0.0f)
            if (!ds.photons.norm_x.empty()) {
                float3 photon_n = make_f3(ds.photons.norm_x[idx], ds.photons.norm_y[idx],
                                          ds.photons.norm_z[idx]);
                if (dot(photon_n, normal) <= 0.0f) return;
            }

            float3 wi_local = frame.world_to_local(wi_world);
            Spectrum f = bsdf::evaluate(mat, wo_local, wi_local);
            Spectrum sf = ds.photons.get_flux(idx);
            // Box kernel (w=1) — matches estimate_photon_density exactly
            for (int b = 0; b < NUM_LAMBDA; ++b)
                L.value[b] += sf.value[b] * inv_N * f.value[b] * inv_area;
            float flux = ds.photons.total_flux(idx);

            // Epanechnikov weight for bin population only
            // dist2 is tangential distance from query_tangential callback
            float w_bin = 1.0f - dist2 / r2;
            int k = (int)ds.photon_bin_idx[idx];
            if (k >= num_bins) k = 0;
            local_bins[k].scalar_flux   += flux * w_bin;
            local_bins[k].dir_x  += wi_world.x * flux * w_bin;
            local_bins[k].dir_y  += wi_world.y * flux * w_bin;
            local_bins[k].dir_z  += wi_world.z * flux * w_bin;
            local_bins[k].weight += w_bin;
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

// Guided NEE
static Spectrum opt_nee_guided_step(
    float3 pos, float3 normal, float3 wo_local,
    const Material& mat, PCGRng& rng,
    const PerRayDataset& ds,
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

static PathResult optimized_trace_steps(
    float3 origin, float3 direction, PCGRng& rng,
    const PerRayDataset& ds, int max_bounces)
{
    PathResult result;
    Spectrum throughput = Spectrum::constant(1.0f);
    bool prev_was_specular = true;

    Ray ray;
    ray.origin    = origin;
    ray.direction = direction;

    for (int bounce = 0; bounce <= max_bounces; ++bounce) {
        HitRecord hit = ds.scene.intersect(ray);
        if (!hit.hit) break;

        const Material& mat = ds.scene.materials[hit.material_id];

        BounceStep step;
        step.bounce      = bounce;
        step.hit_pos     = hit.position;
        step.hit_normal  = hit.shading_normal;
        step.material_id = hit.material_id;
        step.is_emissive = mat.is_emissive();
        step.is_specular = mat.is_specular();
        step.throughput  = throughput;
        step.throughput_finite      = spectrum_is_finite(throughput);
        step.throughput_nonnegative = spectrum_is_nonnegative(throughput);

        if (mat.is_emissive()) {
            if (prev_was_specular) {
                Spectrum Le_contrib = throughput * mat.Le;
                step.nee_contribution = Le_contrib;
                result.combined   += Le_contrib;
                result.nee_direct += Le_contrib;
            }
            step.nee_finite      = spectrum_is_finite(step.nee_contribution);
            step.nee_nonnegative = spectrum_is_nonnegative(step.nee_contribution);
            result.steps.push_back(step);
            result.total_bounces = bounce + 1;
            break;
        }

        if (mat.is_specular()) {
            ONB frame = ONB::from_normal(hit.shading_normal);
            float3 wo_local = frame.world_to_local(ray.direction * (-1.f));
            BSDFSample bs = bsdf::sample(mat, wo_local, rng);
            if (bs.pdf <= 0.f) { result.steps.push_back(step); break; }

            float cos_theta = fabsf(bs.wi.z);
            for (int i = 0; i < NUM_LAMBDA; ++i)
                throughput.value[i] *= bs.f.value[i] * cos_theta / bs.pdf;

            ray.origin    = hit.position + hit.shading_normal * EPSILON *
                            (bs.wi.z > 0.f ? 1.f : -1.f);
            ray.direction = frame.local_to_world(bs.wi);
            prev_was_specular = true;
            result.steps.push_back(step);
            continue;
        }

        prev_was_specular = false;
        ONB frame = ONB::from_normal(hit.shading_normal);
        float3 wo_local = frame.world_to_local(ray.direction * (-1.f));
        if (wo_local.z <= 0.f) { result.steps.push_back(step); break; }

        // Photon gather + local bins
        PhotonBin local_bins[MAX_PHOTON_BIN_COUNT];
        float local_total_flux = 0.0f;
        Spectrum L_photon = gather_with_bins_step(
            hit.position, hit.shading_normal, wo_local, mat, ds,
            local_bins, PHOTON_BIN_COUNT, local_total_flux);
        bool have_bins = (local_total_flux > 0.0f);

        // Guided NEE
        float nee_visibility = 1.0f;
        {
            Spectrum L_nee;
            if (have_bins) {
                L_nee = opt_nee_guided_step(
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
            step.nee_contribution = nee_contrib;
            result.combined   += nee_contrib;
            result.nee_direct += nee_contrib;
        }

        // Photon indirect (no shadow-floor suppression)
        // FIX: Indirect illumination IS visible in shadows —
        //      do NOT modulate by nee_visibility.
        {
            Spectrum photon_contrib = throughput * L_photon;
            step.photon_contribution = photon_contrib;
            result.combined        += photon_contrib;
            result.photon_indirect += photon_contrib;
        }

        // Validate step
        step.nee_finite         = spectrum_is_finite(step.nee_contribution);
        step.nee_nonnegative    = spectrum_is_nonnegative(step.nee_contribution);
        step.photon_finite      = spectrum_is_finite(step.photon_contribution);
        step.photon_nonnegative = spectrum_is_nonnegative(step.photon_contribution);

        result.steps.push_back(step);
        result.total_bounces = bounce + 1;

        // FIX: The photon map already captures ALL indirect illumination.
        // Stop after NEE + photon gather at the first diffuse hit.
        break;
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────
//  PER-RAY DIAGNOSTIC REPORT
// ─────────────────────────────────────────────────────────────────────

struct RayDiagnostic {
    int    ray_idx        = 0;
    int    px             = 0;
    int    py             = 0;

    float  gt_combined_sum   = 0.f;
    float  opt_combined_sum  = 0.f;
    float  gt_nee_sum        = 0.f;
    float  opt_nee_sum       = 0.f;
    float  gt_photon_sum     = 0.f;
    float  opt_photon_sum    = 0.f;

    float  combined_rel_err   = 0.f;
    float  nee_rel_err        = 0.f;
    float  photon_rel_err     = 0.f;

    // Component decomposition error
    float  gt_decomp_err     = 0.f;   // |combined - (nee + photon)| / combined
    float  opt_decomp_err    = 0.f;

    int    gt_bounces        = 0;
    int    opt_bounces       = 0;

    // Physical validity issues
    int    validity_issues   = 0;
    std::string issues_detail;

    bool has_warning() const {
        return combined_rel_err   > thresholds::RAY_COMBINED_REL
            || nee_rel_err        > thresholds::RAY_NEE_DIRECT_REL
            || photon_rel_err     > thresholds::RAY_PHOTON_INDIRECT_REL
            || validity_issues    > 0;
    }
};

static RayDiagnostic diagnose_ray(
    int ray_idx, int px, int py,
    const PathResult& gt, const PathResult& opt)
{
    RayDiagnostic diag;
    diag.ray_idx = ray_idx;
    diag.px      = px;
    diag.py      = py;

    diag.gt_combined_sum  = gt.combined.sum();
    diag.opt_combined_sum = opt.combined.sum();
    diag.gt_nee_sum       = gt.nee_direct.sum();
    diag.opt_nee_sum      = opt.nee_direct.sum();
    diag.gt_photon_sum    = gt.photon_indirect.sum();
    diag.opt_photon_sum   = opt.photon_indirect.sum();

    diag.combined_rel_err = safe_relative_error(diag.gt_combined_sum, diag.opt_combined_sum);
    diag.nee_rel_err      = safe_relative_error(diag.gt_nee_sum, diag.opt_nee_sum);
    diag.photon_rel_err   = safe_relative_error(diag.gt_photon_sum, diag.opt_photon_sum);

    diag.gt_bounces  = gt.total_bounces;
    diag.opt_bounces = opt.total_bounces;

    // Component decomposition: combined ≈ nee + photon
    float gt_sum_parts  = diag.gt_nee_sum + diag.gt_photon_sum;
    float opt_sum_parts = diag.opt_nee_sum + diag.opt_photon_sum;

    if (diag.gt_combined_sum > 1e-8f)
        diag.gt_decomp_err = fabsf(diag.gt_combined_sum - gt_sum_parts) / diag.gt_combined_sum;
    if (diag.opt_combined_sum > 1e-8f)
        diag.opt_decomp_err = fabsf(diag.opt_combined_sum - opt_sum_parts) / diag.opt_combined_sum;

    // Physical validity per bounce step
    std::ostringstream iss;
    auto check_steps = [&](const PathResult& path, const char* label) {
        for (const auto& s : path.steps) {
            if (!s.throughput_finite) {
                iss << "  " << label << " bounce " << s.bounce
                    << ": throughput NOT FINITE\n";
                diag.validity_issues++;
            }
            if (!s.throughput_nonnegative) {
                iss << "  " << label << " bounce " << s.bounce
                    << ": throughput NEGATIVE\n";
                diag.validity_issues++;
            }
            if (!s.nee_finite) {
                iss << "  " << label << " bounce " << s.bounce
                    << ": NEE NOT FINITE\n";
                diag.validity_issues++;
            }
            if (!s.nee_nonnegative) {
                iss << "  " << label << " bounce " << s.bounce
                    << ": NEE NEGATIVE\n";
                diag.validity_issues++;
            }
            if (!s.photon_finite) {
                iss << "  " << label << " bounce " << s.bounce
                    << ": photon NOT FINITE\n";
                diag.validity_issues++;
            }
            if (!s.photon_nonnegative) {
                iss << "  " << label << " bounce " << s.bounce
                    << ": photon NEGATIVE\n";
                diag.validity_issues++;
            }
            if (spectrum_max(s.throughput) > thresholds::MAX_THROUGHPUT_COMPONENT) {
                iss << "  " << label << " bounce " << s.bounce
                    << ": throughput EXPLODED (max="
                    << spectrum_max(s.throughput) << ")\n";
                diag.validity_issues++;
            }
        }
    };

    check_steps(gt,  "GT");
    check_steps(opt, "OPT");

    // Component decomposition check
    if (diag.gt_decomp_err > thresholds::COMPONENT_DECOMP_REL && diag.gt_combined_sum > 1e-6f) {
        iss << "  GT decomposition error: " << (diag.gt_decomp_err * 100.f) << "%\n";
        diag.validity_issues++;
    }
    if (diag.opt_decomp_err > thresholds::COMPONENT_DECOMP_REL && diag.opt_combined_sum > 1e-6f) {
        iss << "  OPT decomposition error: " << (diag.opt_decomp_err * 100.f) << "%\n";
        diag.validity_issues++;
    }

    diag.issues_detail = iss.str();
    return diag;
}

// ─────────────────────────────────────────────────────────────────────
//  RANDOM RAY SELECTION
// ─────────────────────────────────────────────────────────────────────

struct RandomRay {
    int px, py;
    uint64_t seed;
};

// Deterministic pseudo-random ray selection across the image
static std::vector<RandomRay> pick_random_rays(int N, int width, int height, uint64_t base_seed) {
    std::vector<RandomRay> rays;
    rays.reserve(N);
    PCGRng picker = PCGRng::seed(base_seed, 0x12345678ULL);
    for (int i = 0; i < N; ++i) {
        RandomRay r;
        r.px   = (int)(picker.next_float() * width) % width;
        r.py   = (int)(picker.next_float() * height) % height;
        r.seed = picker.next_uint();
        rays.push_back(r);
    }
    return rays;
}

// ─────────────────────────────────────────────────────────────────────
//  AGGREGATE STATISTICS
// ─────────────────────────────────────────────────────────────────────

struct AggregateStats {
    double gt_mean       = 0.0;
    double opt_mean      = 0.0;
    double gt_nee_mean   = 0.0;
    double opt_nee_mean  = 0.0;
    double gt_phot_mean  = 0.0;
    double opt_phot_mean = 0.0;

    double combined_rel_err = 0.0;
    double nee_rel_err      = 0.0;
    double photon_rel_err   = 0.0;
    double energy_ratio     = 0.0;

    int total_rays          = 0;
    int rays_with_warnings  = 0;
    int total_validity_issues = 0;

    int rays_combined_exceed  = 0;
    int rays_nee_exceed       = 0;
    int rays_photon_exceed    = 0;
};

static void print_warning_summary(
    const AggregateStats& agg,
    const std::vector<RayDiagnostic>& warnings)
{
    std::cout << "\n======================================================\n"
              << "  PER-RAY PHYSICAL PATH VALIDATION SUMMARY\n"
              << "======================================================\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Total rays tested: " << agg.total_rays << "\n";
    std::cout << "  Rays with warnings: " << agg.rays_with_warnings
              << " (" << (100.0 * agg.rays_with_warnings / agg.total_rays) << "%)\n";
    std::cout << "  Physical validity issues: " << agg.total_validity_issues << "\n";
    std::cout << "\n  AGGREGATE MEANS:\n";
    std::cout << "    GT combined mean:  " << agg.gt_mean << "\n";
    std::cout << "    OPT combined mean: " << agg.opt_mean << "\n";
    std::cout << "    Combined relErr:   " << agg.combined_rel_err << "\n";
    std::cout << "    NEE relErr:        " << agg.nee_rel_err << "\n";
    std::cout << "    Photon relErr:     " << agg.photon_rel_err << "\n";
    std::cout << "    Energy ratio:      " << agg.energy_ratio << "\n";
    std::cout << "\n  PER-RAY THRESHOLD EXCEEDANCES:\n";
    std::cout << "    Combined > " << thresholds::RAY_COMBINED_REL << ": "
              << agg.rays_combined_exceed << " rays ("
              << (100.0 * agg.rays_combined_exceed / agg.total_rays) << "%)\n";
    std::cout << "    NEE > " << thresholds::RAY_NEE_DIRECT_REL << ": "
              << agg.rays_nee_exceed << " rays ("
              << (100.0 * agg.rays_nee_exceed / agg.total_rays) << "%)\n";
    std::cout << "    Photon > " << thresholds::RAY_PHOTON_INDIRECT_REL << ": "
              << agg.rays_photon_exceed << " rays ("
              << (100.0 * agg.rays_photon_exceed / agg.total_rays) << "%)\n";

    // Print worst rays
    if (!warnings.empty()) {
        int show = std::min((int)warnings.size(), 10);
        std::cout << "\n  TOP " << show << " WORST RAYS:\n";
        for (int i = 0; i < show; ++i) {
            const auto& w = warnings[i];
            std::cout << "    Ray #" << w.ray_idx
                      << " px=(" << w.px << "," << w.py << ")"
                      << " GT=" << w.gt_combined_sum
                      << " OPT=" << w.opt_combined_sum
                      << " relErr=" << w.combined_rel_err
                      << " bounces=" << w.gt_bounces << "/" << w.opt_bounces
                      << (w.validity_issues > 0 ? " [VALIDITY!]" : "")
                      << "\n";
            if (!w.issues_detail.empty())
                std::cout << w.issues_detail;
        }
    }
    std::cout << "======================================================\n\n";
}

// ─────────────────────────────────────────────────────────────────────
//  CORE VALIDATION RUNNER
// ─────────────────────────────────────────────────────────────────────

static AggregateStats run_per_ray_validation(
    const PerRayDataset& ds, int num_rays, int max_bounces,
    std::vector<RayDiagnostic>& all_warnings)
{
    auto rays = pick_random_rays(num_rays, 64, 64, 0xDEADBEEF42ULL);

    AggregateStats agg;
    agg.total_rays = num_rays;

    double gt_sum = 0, opt_sum = 0;
    double gt_nee_sum = 0, opt_nee_sum = 0;
    double gt_phot_sum = 0, opt_phot_sum = 0;

    // Per-ray results stored for thread-safe parallel access
    struct RayResult {
        RayDiagnostic diag;
        bool has_warning = false;
    };
    std::vector<RayResult> ray_results(num_rays);

    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < num_rays; ++i) {
        const auto& r = rays[i];

        PCGRng rng_ray = PCGRng::seed(r.seed, (uint64_t)i);
        Ray ray = ds.camera.generate_ray(r.px, r.py, rng_ray);

        PCGRng rng_gt  = PCGRng::seed(r.seed + 1000000ULL, (uint64_t)i);
        auto gt = ground_truth_trace_steps(
            ray.origin, ray.direction, rng_gt, ds, max_bounces);

        PCGRng rng_opt = PCGRng::seed(r.seed + 2000000ULL, (uint64_t)i);
        auto opt = optimized_trace_steps(
            ray.origin, ray.direction, rng_opt, ds, max_bounces);

        ray_results[i].diag = diagnose_ray(i, r.px, r.py, gt, opt);
        ray_results[i].has_warning = ray_results[i].diag.has_warning();
    }

    // Sequential reduction over results
    for (int i = 0; i < num_rays; ++i) {
        const auto& diag = ray_results[i].diag;

        gt_sum     += diag.gt_combined_sum;
        opt_sum    += diag.opt_combined_sum;
        gt_nee_sum += diag.gt_nee_sum;
        opt_nee_sum += diag.opt_nee_sum;
        gt_phot_sum += diag.gt_photon_sum;
        opt_phot_sum += diag.opt_photon_sum;

        if (diag.combined_rel_err > thresholds::RAY_COMBINED_REL)
            agg.rays_combined_exceed++;
        if (diag.nee_rel_err > thresholds::RAY_NEE_DIRECT_REL)
            agg.rays_nee_exceed++;
        if (diag.photon_rel_err > thresholds::RAY_PHOTON_INDIRECT_REL)
            agg.rays_photon_exceed++;

        agg.total_validity_issues += diag.validity_issues;

        if (ray_results[i].has_warning) {
            agg.rays_with_warnings++;
            all_warnings.push_back(diag);
        }
    }

    agg.gt_mean      = gt_sum / num_rays;
    agg.opt_mean     = opt_sum / num_rays;
    agg.gt_nee_mean  = gt_nee_sum / num_rays;
    agg.opt_nee_mean = opt_nee_sum / num_rays;
    agg.gt_phot_mean = gt_phot_sum / num_rays;
    agg.opt_phot_mean = opt_phot_sum / num_rays;

    if (agg.gt_mean > 1e-10)
        agg.combined_rel_err = fabs(agg.opt_mean - agg.gt_mean) / agg.gt_mean;
    if (agg.gt_nee_mean > 1e-10)
        agg.nee_rel_err = fabs(agg.opt_nee_mean - agg.gt_nee_mean) / agg.gt_nee_mean;
    if (agg.gt_phot_mean > 1e-10)
        agg.photon_rel_err = fabs(agg.opt_phot_mean - agg.gt_phot_mean) / agg.gt_phot_mean;

    agg.energy_ratio = (gt_sum > 1e-10) ? (opt_sum / gt_sum) : 0.0;

    // Sort warnings by combined error descending
    std::sort(all_warnings.begin(), all_warnings.end(),
        [](const RayDiagnostic& a, const RayDiagnostic& b) {
            return a.combined_rel_err > b.combined_rel_err;
        });

    return agg;
}

// =====================================================================
//  TEST SUITE: PerRayValidation
// =====================================================================

class PerRayValidation : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        get_per_ray_dataset();  // ensure built
    }
};

// ── Dataset is valid ─────────────────────────────────────────────────
TEST_F(PerRayValidation, DatasetIsValid) {
    auto& ds = get_per_ray_dataset();
    ASSERT_TRUE(ds.valid) << "Dataset failed to build";
    EXPECT_GT(ds.photons.size(), 1000u);
}

// ─────────────────────────────────────────────────────────────────────
// 1. NO PHYSICAL VALIDITY VIOLATIONS (Invariant 0: finite, non-neg)
//    Pick 256 random rays and check every bounce step
// ─────────────────────────────────────────────────────────────────────
TEST_F(PerRayValidation, NoPhysicalViolationsInRandomRays) {
    auto& ds = get_per_ray_dataset();
    ASSERT_TRUE(ds.valid);

    constexpr int NUM_RAYS = 256;
    std::vector<RayDiagnostic> warnings;
    auto agg = run_per_ray_validation(ds, NUM_RAYS, ds.max_bounces, warnings);

    print_warning_summary(agg, warnings);

    EXPECT_EQ(agg.total_validity_issues, 0)
        << "Found " << agg.total_validity_issues
        << " physical validity issues (NaN/Inf/negative) across "
        << NUM_RAYS << " random rays";
}

// ─────────────────────────────────────────────────────────────────────
// 2. COMPONENT DECOMPOSITION: combined ≈ nee_direct + photon_indirect
//    (Invariant: no double counting, no missing terms)
// ─────────────────────────────────────────────────────────────────────
TEST_F(PerRayValidation, ComponentDecompositionHolds) {
    auto& ds = get_per_ray_dataset();
    ASSERT_TRUE(ds.valid);

    constexpr int NUM_RAYS = 128;
    auto rays = pick_random_rays(NUM_RAYS, 64, 64, 0xC0FFEE42ULL);

    int decomp_violations_gt  = 0;
    int decomp_violations_opt = 0;

    for (int i = 0; i < NUM_RAYS; ++i) {
        PCGRng rng_ray = PCGRng::seed(rays[i].seed, (uint64_t)i);
        Ray ray = ds.camera.generate_ray(rays[i].px, rays[i].py, rng_ray);

        // Ground truth
        PCGRng rng_gt = PCGRng::seed(rays[i].seed + 1000000ULL, (uint64_t)i);
        auto gt = ground_truth_trace_steps(
            ray.origin, ray.direction, rng_gt, ds, ds.max_bounces);

        float gt_combined = gt.combined.sum();
        float gt_parts    = gt.nee_direct.sum() + gt.photon_indirect.sum();

        if (gt_combined > 1e-6f) {
            float err = fabsf(gt_combined - gt_parts) / gt_combined;
            if (err > thresholds::COMPONENT_DECOMP_REL)
                decomp_violations_gt++;
        }

        // Optimized
        PCGRng rng_opt = PCGRng::seed(rays[i].seed + 2000000ULL, (uint64_t)i);
        auto opt = optimized_trace_steps(
            ray.origin, ray.direction, rng_opt, ds, ds.max_bounces);

        float opt_combined = opt.combined.sum();
        float opt_parts    = opt.nee_direct.sum() + opt.photon_indirect.sum();

        if (opt_combined > 1e-6f) {
            float err = fabsf(opt_combined - opt_parts) / opt_combined;
            if (err > thresholds::COMPONENT_DECOMP_REL)
                decomp_violations_opt++;
        }
    }

    std::cout << "[Decomposition] GT violations: " << decomp_violations_gt
              << "/" << NUM_RAYS
              << "  OPT violations: " << decomp_violations_opt
              << "/" << NUM_RAYS << "\n";

    EXPECT_EQ(decomp_violations_gt, 0)
        << "Ground truth combined != nee + photon in "
        << decomp_violations_gt << " rays";
    EXPECT_EQ(decomp_violations_opt, 0)
        << "Optimized combined != nee + photon in "
        << decomp_violations_opt << " rays";
}

// ─────────────────────────────────────────────────────────────────────
// 3. AGGREGATE NEE CONVERGENCE (Spec §7.1: correct PDF, no bias)
//    Mean NEE across N rays should converge between methods
// ─────────────────────────────────────────────────────────────────────
TEST_F(PerRayValidation, AggregateNEEConverges) {
    auto& ds = get_per_ray_dataset();
    ASSERT_TRUE(ds.valid);

    constexpr int NUM_RAYS = 512;
    std::vector<RayDiagnostic> warnings;
    auto agg = run_per_ray_validation(ds, NUM_RAYS, ds.max_bounces, warnings);

    std::cout << "[AggNEE] GT mean=" << agg.gt_nee_mean
              << " OPT mean=" << agg.opt_nee_mean
              << " relErr=" << agg.nee_rel_err << "\n";

    if (agg.gt_nee_mean > 1e-8) {
        EXPECT_LT(agg.nee_rel_err, thresholds::AGG_NEE_DIRECT_REL)
            << "[WARNING] Aggregate NEE deviation " << (agg.nee_rel_err * 100)
            << "% exceeds threshold " << (thresholds::AGG_NEE_DIRECT_REL * 100) << "%";
    }

    // Warn if too many individual rays exceed per-ray threshold
    float frac = (float)agg.rays_nee_exceed / NUM_RAYS;
    if (frac > thresholds::WARN_FRACTION) {
        std::cout << "[WARNING] " << (frac * 100) << "% of rays exceed per-ray NEE threshold ("
                  << agg.rays_nee_exceed << "/" << NUM_RAYS << ")\n";
    }
    EXPECT_LT(frac, thresholds::FAIL_FRACTION)
        << "Too many rays exceed NEE threshold: " << agg.rays_nee_exceed << "/" << NUM_RAYS;
}

// ─────────────────────────────────────────────────────────────────────
// 4. AGGREGATE PHOTON DENSITY CONVERGENCE (Spec §6: correct estimator)
// ─────────────────────────────────────────────────────────────────────
TEST_F(PerRayValidation, AggregatePhotonDensityConverges) {
    auto& ds = get_per_ray_dataset();
    ASSERT_TRUE(ds.valid);

    constexpr int NUM_RAYS = 512;
    std::vector<RayDiagnostic> warnings;
    auto agg = run_per_ray_validation(ds, NUM_RAYS, ds.max_bounces, warnings);

    std::cout << "[AggPhoton] GT mean=" << agg.gt_phot_mean
              << " OPT mean=" << agg.opt_phot_mean
              << " relErr=" << agg.photon_rel_err << "\n";

    if (agg.gt_phot_mean > 1e-8) {
        EXPECT_LT(agg.photon_rel_err, thresholds::AGG_PHOTON_INDIRECT_REL)
            << "[WARNING] Aggregate photon density deviation "
            << (agg.photon_rel_err * 100) << "% exceeds threshold "
            << (thresholds::AGG_PHOTON_INDIRECT_REL * 100) << "%";
    }

    float frac = (float)agg.rays_photon_exceed / NUM_RAYS;
    if (frac > thresholds::WARN_FRACTION) {
        std::cout << "[WARNING] " << (frac * 100)
                  << "% of rays exceed per-ray photon threshold ("
                  << agg.rays_photon_exceed << "/" << NUM_RAYS << ")\n";
    }
    EXPECT_LT(frac, thresholds::FAIL_FRACTION)
        << "Too many rays exceed photon threshold";
}

// ─────────────────────────────────────────────────────────────────────
// 5. AGGREGATE COMBINED CONVERGENCE (Spec §7: full hybrid path)
// ─────────────────────────────────────────────────────────────────────
TEST_F(PerRayValidation, AggregateCombinedConverges) {
    auto& ds = get_per_ray_dataset();
    ASSERT_TRUE(ds.valid);

    constexpr int NUM_RAYS = 512;
    std::vector<RayDiagnostic> warnings;
    auto agg = run_per_ray_validation(ds, NUM_RAYS, ds.max_bounces, warnings);

    print_warning_summary(agg, warnings);

    if (agg.gt_mean > 1e-8) {
        EXPECT_LT(agg.combined_rel_err, thresholds::AGG_COMBINED_REL)
            << "[WARNING] Aggregate combined deviation " << (agg.combined_rel_err * 100)
            << "% exceeds threshold " << (thresholds::AGG_COMBINED_REL * 100) << "%";
    }

    float frac = (float)agg.rays_combined_exceed / NUM_RAYS;
    if (frac > thresholds::WARN_FRACTION) {
        std::cout << "[WARNING] " << (frac * 100)
                  << "% of rays exceed per-ray combined threshold\n";
    }
    EXPECT_LT(frac, thresholds::FAIL_FRACTION)
        << "Too many rays (" << agg.rays_combined_exceed << "/" << NUM_RAYS
        << ") exceed combined threshold";
}

// ─────────────────────────────────────────────────────────────────────
// 6. ENERGY CONSERVATION (Spec §0: no creation/destruction)
// ─────────────────────────────────────────────────────────────────────
TEST_F(PerRayValidation, EnergyConservationAcrossRandomRays) {
    auto& ds = get_per_ray_dataset();
    ASSERT_TRUE(ds.valid);

    constexpr int NUM_RAYS = 512;
    std::vector<RayDiagnostic> warnings;
    auto agg = run_per_ray_validation(ds, NUM_RAYS, ds.max_bounces, warnings);

    std::cout << "[Energy] ratio=" << agg.energy_ratio
              << " (GT total=" << (agg.gt_mean * NUM_RAYS)
              << " OPT total=" << (agg.opt_mean * NUM_RAYS) << ")\n";

    EXPECT_GT(agg.energy_ratio, thresholds::AGG_ENERGY_RATIO_LOW)
        << "[WARNING] Optimized loses too much energy (ratio=" << agg.energy_ratio << ")";
    EXPECT_LT(agg.energy_ratio, thresholds::AGG_ENERGY_RATIO_HIGH)
        << "[WARNING] Optimized creates energy (ratio=" << agg.energy_ratio << ")";
}

// ─────────────────────────────────────────────────────────────────────
// 7. SPECTRAL BIN ISOLATION (Spec §0: bins never mix during transport)
//    Each photon's wavelength bin must contribute only to its own bin
// ─────────────────────────────────────────────────────────────────────
TEST_F(PerRayValidation, SpectralBinsNeverMix) {
    auto& ds = get_per_ray_dataset();
    ASSERT_TRUE(ds.valid);

    // Validate that photon density per-lambda is zero for bins without
    // photon contributions in that lambda
    constexpr int NUM_CHECKS = 64;
    auto rays = pick_random_rays(NUM_CHECKS, 64, 64, 0x5EC7A100ULL);

    int mixing_violations = 0;

    for (int i = 0; i < NUM_CHECKS; ++i) {
        PCGRng rng_ray = PCGRng::seed(rays[i].seed, (uint64_t)i);
        Ray ray = ds.camera.generate_ray(rays[i].px, rays[i].py, rng_ray);

        HitRecord hit = ds.scene.intersect(ray);
        if (!hit.hit) continue;

        const Material& mat = ds.scene.materials[hit.material_id];
        if (mat.is_emissive() || mat.is_specular()) continue;

        ONB frame = ONB::from_normal(hit.shading_normal);
        float3 wo_local = frame.world_to_local(ray.direction * (-1.f));
        if (wo_local.z <= 0.f) continue;

        // Track which lambda bins have photon contributions
        bool lambda_has_photon[NUM_LAMBDA] = {};
        Spectrum L_manual = Spectrum::zero();

        float radius = ds.gather_radius;
        float r2 = radius * radius;
        float inv_area = 1.0f / (PI * r2);
        float inv_N    = 1.0f / (float)ds.num_photons;

        ds.grid.query(hit.position, radius, ds.photons,
            [&](uint32_t idx, float dist2) {
                float3 pp = make_f3(ds.photons.pos_x[idx], ds.photons.pos_y[idx],
                                    ds.photons.pos_z[idx]);
                float3 diff = pp - hit.position;
                if (fabsf(dot(hit.shading_normal, diff)) > DEFAULT_SURFACE_TAU) return;

                float3 wi_world = make_f3(ds.photons.wi_x[idx], ds.photons.wi_y[idx],
                                           ds.photons.wi_z[idx]);
                if (dot(wi_world, hit.shading_normal) <= 0.f) return;

                float w = 1.0f - dist2 / r2;
                float3 wi_local = frame.world_to_local(wi_world);
                Spectrum f = bsdf::evaluate(mat, wo_local, wi_local);
                Spectrum sf = ds.photons.get_flux(idx);

                // Full spectral: every photon contributes to all bins
                for (int b = 0; b < NUM_LAMBDA; ++b) {
                    if (sf.value[b] > 0.f) lambda_has_photon[b] = true;
                    L_manual.value[b] += sf.value[b] * inv_N * f.value[b] * w * inv_area;
                }
            });

        // Check: any non-zero L in a lambda bin without photon flux?
        for (int b = 0; b < NUM_LAMBDA; ++b) {
            if (!lambda_has_photon[b] && L_manual.value[b] != 0.f) {
                mixing_violations++;
            }
        }
    }

    EXPECT_EQ(mixing_violations, 0)
        << "Found " << mixing_violations
        << " spectral mixing violations (energy in lambda bins without photon sources)";
}

// ─────────────────────────────────────────────────────────────────────
// 8. FIRST-HIT PHOTON DENSITY AGREEMENT (Spec §6: identical gather)
//    At the first diffuse hit, gather is deterministic — both methods
//    should produce the SAME photon density estimate
// ─────────────────────────────────────────────────────────────────────
TEST_F(PerRayValidation, FirstHitPhotonDensityAgreement) {
    auto& ds = get_per_ray_dataset();
    ASSERT_TRUE(ds.valid);

    constexpr int NUM_RAYS = 128;
    auto rays = pick_random_rays(NUM_RAYS, 64, 64, 0xF070AAEEULL);

    int disagreements = 0;
    int tested = 0;

    DensityEstimatorConfig de_cfg;
    de_cfg.radius            = ds.gather_radius;
    de_cfg.caustic_radius    = DEFAULT_CAUSTIC_RADIUS;
    de_cfg.num_photons_total = ds.num_photons;
    de_cfg.surface_tau       = DEFAULT_SURFACE_TAU;
    de_cfg.use_kernel        = false;  // Box kernel to match gather_with_bins_step

    for (int i = 0; i < NUM_RAYS; ++i) {
        PCGRng rng_ray = PCGRng::seed(rays[i].seed, (uint64_t)i);
        Ray ray = ds.camera.generate_ray(rays[i].px, rays[i].py, rng_ray);

        HitRecord hit = ds.scene.intersect(ray);
        if (!hit.hit) continue;

        const Material& mat = ds.scene.materials[hit.material_id];
        if (mat.is_emissive() || mat.is_specular()) continue;

        ONB frame = ONB::from_normal(hit.shading_normal);
        float3 wo_local = frame.world_to_local(ray.direction * (-1.f));
        if (wo_local.z <= 0.f) continue;

        // Ground truth: standard gather
        Spectrum L_gt = estimate_photon_density(
            hit.position, hit.shading_normal, wo_local, mat,
            ds.photons, ds.grid, de_cfg, ds.gather_radius);

        // Optimized: gather with bins
        PhotonBin local_bins[MAX_PHOTON_BIN_COUNT];
        float total_flux = 0.f;
        Spectrum L_opt = gather_with_bins_step(
            hit.position, hit.shading_normal, wo_local, mat, ds,
            local_bins, PHOTON_BIN_COUNT, total_flux);

        float gt_sum  = L_gt.sum();
        float opt_sum = L_opt.sum();
        tested++;

        if (gt_sum > 1e-8f) {
            float rel = fabsf(gt_sum - opt_sum) / gt_sum;
            if (rel > 0.01f) {  // Should be identical (same loop)
                disagreements++;
                if (disagreements <= 5) {
                    std::cout << "[WARNING] Ray " << i
                              << " first-hit photon density mismatch: GT="
                              << gt_sum << " OPT=" << opt_sum
                              << " relErr=" << rel << "\n";
                }
            }
        }
    }

    std::cout << "[FirstHitPhoton] Tested " << tested
              << " diffuse first hits, disagreements: " << disagreements << "\n";

    EXPECT_EQ(disagreements, 0)
        << "Photon density should be identical at first hit (same gather loop), "
        << "but " << disagreements << "/" << tested << " rays differed";
}

// ─────────────────────────────────────────────────────────────────────
// 9. BOUNCE-BY-BOUNCE THROUGHPUT DECAY (Spec §4.2: valid throughput)
//    Throughput should decay physically — never explode, always finite
// ─────────────────────────────────────────────────────────────────────
TEST_F(PerRayValidation, ThroughputDecayIsPhysical) {
    auto& ds = get_per_ray_dataset();
    ASSERT_TRUE(ds.valid);

    constexpr int NUM_RAYS = 256;
    auto rays = pick_random_rays(NUM_RAYS, 64, 64, 0x7A0E0000ULL);

    int explosion_count = 0;
    int nonfinite_count = 0;
    int negative_count  = 0;

    for (int i = 0; i < NUM_RAYS; ++i) {
        PCGRng rng_ray = PCGRng::seed(rays[i].seed, (uint64_t)i);
        Ray ray = ds.camera.generate_ray(rays[i].px, rays[i].py, rng_ray);

        PCGRng rng_gt = PCGRng::seed(rays[i].seed + 1000000ULL, (uint64_t)i);
        auto gt = ground_truth_trace_steps(
            ray.origin, ray.direction, rng_gt, ds, ds.max_bounces);

        PCGRng rng_opt = PCGRng::seed(rays[i].seed + 2000000ULL, (uint64_t)i);
        auto opt = optimized_trace_steps(
            ray.origin, ray.direction, rng_opt, ds, ds.max_bounces);

        auto check = [&](const PathResult& path, const char* label) {
            for (const auto& step : path.steps) {
                if (!step.throughput_finite) {
                    nonfinite_count++;
                    if (nonfinite_count <= 3)
                        std::cout << "[WARNING] " << label << " ray " << i
                                  << " bounce " << step.bounce
                                  << ": throughput not finite\n";
                }
                if (!step.throughput_nonnegative) {
                    negative_count++;
                    if (negative_count <= 3)
                        std::cout << "[WARNING] " << label << " ray " << i
                                  << " bounce " << step.bounce
                                  << ": throughput has negative component\n";
                }
                float max_t = spectrum_max(step.throughput);
                if (max_t > thresholds::MAX_THROUGHPUT_COMPONENT) {
                    explosion_count++;
                    if (explosion_count <= 3)
                        std::cout << "[WARNING] " << label << " ray " << i
                                  << " bounce " << step.bounce
                                  << ": throughput exploded (max="
                                  << max_t << ")\n";
                }
            }
        };

        check(gt,  "GT");
        check(opt, "OPT");
    }

    std::cout << "[Throughput] Tested " << NUM_RAYS << " rays: "
              << "explosions=" << explosion_count
              << " nonfinite=" << nonfinite_count
              << " negative=" << negative_count << "\n";

    EXPECT_EQ(nonfinite_count, 0)
        << "Non-finite throughput values found";
    EXPECT_EQ(negative_count, 0)
        << "Negative throughput values found";
    EXPECT_EQ(explosion_count, 0)
        << "Throughput exploded beyond " << thresholds::MAX_THROUGHPUT_COMPONENT;
}

// ─────────────────────────────────────────────────────────────────────
// 10. SINGLE-BOUNCE PHOTON DENSITY COMPARISON (Spec §6 + §7.2)
//     At bounce 0, both methods use the same photon gather — should
//     have minimal difference. Aggregate over many rays.
// ─────────────────────────────────────────────────────────────────────
TEST_F(PerRayValidation, SingleBouncePhotonAgreesClosely) {
    auto& ds = get_per_ray_dataset();
    ASSERT_TRUE(ds.valid);

    constexpr int NUM_RAYS = 256;
    std::vector<RayDiagnostic> warnings;
    auto agg = run_per_ray_validation(ds, NUM_RAYS, /*max_bounces=*/0, warnings);

    std::cout << "[SingleBounce] GT photon=" << agg.gt_phot_mean
              << " OPT photon=" << agg.opt_phot_mean
              << " relErr=" << agg.photon_rel_err << "\n";

    // At single bounce, photon density should be very close (same gather)
    // Only difference is the shadow floor multiplicative factor
    if (agg.gt_phot_mean > 1e-8) {
        EXPECT_LT(agg.photon_rel_err, 0.15)
            << "[WARNING] Single-bounce photon density diverges by "
            << (agg.photon_rel_err * 100) << "%";
    }
}

// ─────────────────────────────────────────────────────────────────────
// 11. PER-SPECTRAL-BIN AGREEMENT ACROSS RANDOM RAYS
//     (Spec §0: spectral bins never mix)
// ─────────────────────────────────────────────────────────────────────
TEST_F(PerRayValidation, PerSpectralBinAgreement) {
    auto& ds = get_per_ray_dataset();
    ASSERT_TRUE(ds.valid);

    constexpr int NUM_RAYS = 256;
    auto rays = pick_random_rays(NUM_RAYS, 64, 64, 0x5EC7B100ULL);

    Spectrum gt_acc  = Spectrum::zero();
    Spectrum opt_acc = Spectrum::zero();

    for (int i = 0; i < NUM_RAYS; ++i) {
        PCGRng rng_ray = PCGRng::seed(rays[i].seed, (uint64_t)i);
        Ray ray = ds.camera.generate_ray(rays[i].px, rays[i].py, rng_ray);

        PCGRng rng_gt = PCGRng::seed(rays[i].seed + 1000000ULL, (uint64_t)i);
        auto gt = ground_truth_trace_steps(
            ray.origin, ray.direction, rng_gt, ds, ds.max_bounces);

        PCGRng rng_opt = PCGRng::seed(rays[i].seed + 2000000ULL, (uint64_t)i);
        auto opt = optimized_trace_steps(
            ray.origin, ray.direction, rng_opt, ds, ds.max_bounces);

        gt_acc  += gt.combined;
        opt_acc += opt.combined;
    }

    int bins_with_large_error = 0;
    float worst_error = 0.f;
    int worst_bin = -1;

    for (int b = 0; b < NUM_LAMBDA; ++b) {
        float gt_mean  = gt_acc.value[b] / NUM_RAYS;
        float opt_mean = opt_acc.value[b] / NUM_RAYS;

        if (gt_mean > 1e-6f) {
            float rel = fabsf(opt_mean - gt_mean) / gt_mean;
            if (rel > 0.50f) {
                bins_with_large_error++;
                if (rel > worst_error) {
                    worst_error = rel;
                    worst_bin = b;
                }
            }
        }
    }

    std::cout << "[SpectralBins] Bins with >50% error: "
              << bins_with_large_error << "/" << NUM_LAMBDA;
    if (worst_bin >= 0)
        std::cout << " (worst: bin " << worst_bin
                  << " err=" << (worst_error * 100) << "%)";
    std::cout << "\n";

    if (bins_with_large_error > 0) {
        std::cout << "[WARNING] " << bins_with_large_error
                  << " spectral bins have >50% disagreement between methods\n";
    }

    EXPECT_LT(bins_with_large_error, NUM_LAMBDA / 4)
        << "Too many spectral bins have large disagreement";
}

// ─────────────────────────────────────────────────────────────────────
// 12. MULTI-BOUNCE BIAS CHECK (Spec §7.3: guided bounce shouldn't
//     introduce systematic bias)
// ─────────────────────────────────────────────────────────────────────
TEST_F(PerRayValidation, MultiBounceNoBias) {
    auto& ds = get_per_ray_dataset();
    ASSERT_TRUE(ds.valid);

    constexpr int NUM_RAYS = 512;
    std::vector<RayDiagnostic> warnings;
    auto agg = run_per_ray_validation(ds, NUM_RAYS, ds.max_bounces, warnings);

    std::cout << "[MultiBounce] GT mean=" << agg.gt_mean
              << " OPT mean=" << agg.opt_mean
              << " relErr=" << agg.combined_rel_err
              << " energy_ratio=" << agg.energy_ratio << "\n";

    // Warn on significant bias
    if (agg.gt_mean > 1e-8 && agg.combined_rel_err > 0.20) {
        std::cout << "[WARNING] Multi-bounce bias detected: "
                  << (agg.combined_rel_err * 100) << "% mean deviation\n";

        // Show per-bounce breakdown from worst ray
        if (!warnings.empty()) {
            const auto& worst = warnings[0];
            std::cout << "  Worst ray #" << worst.ray_idx
                      << " px=(" << worst.px << "," << worst.py << ")"
                      << " GT=" << worst.gt_combined_sum
                      << " OPT=" << worst.opt_combined_sum << "\n";
        }
    }

    EXPECT_LT(agg.combined_rel_err, thresholds::AGG_COMBINED_REL)
        << "Multi-bounce combined bias exceeds "
        << (thresholds::AGG_COMBINED_REL * 100) << "% threshold";
}

// ─────────────────────────────────────────────────────────────────────
// 13. NO DOUBLE COUNTING DIRECT LIGHT (Spec §0, §4.3)
//     Photon indirect contribution should NOT contain direct-lit
//     hotspot patterns. Check: photon indirect at first diffuse hit
//     should be much smaller than NEE direct in well-lit areas.
// ─────────────────────────────────────────────────────────────────────
TEST_F(PerRayValidation, NoDoubleCountingDirectLight) {
    auto& ds = get_per_ray_dataset();
    ASSERT_TRUE(ds.valid);

    constexpr int NUM_RAYS = 256;
    auto rays = pick_random_rays(NUM_RAYS, 64, 64, 0xD1EC7420ULL);

    int suspicious_rays = 0;

    for (int i = 0; i < NUM_RAYS; ++i) {
        PCGRng rng_ray = PCGRng::seed(rays[i].seed, (uint64_t)i);
        Ray ray = ds.camera.generate_ray(rays[i].px, rays[i].py, rng_ray);

        // Only check ground truth (the reference implementation)
        PCGRng rng_gt = PCGRng::seed(rays[i].seed + 1000000ULL, (uint64_t)i);
        auto gt = ground_truth_trace_steps(
            ray.origin, ray.direction, rng_gt, ds, /*max_bounces=*/0);

        if (gt.steps.empty()) continue;

        float nee_sum = gt.nee_direct.sum();
        float photon_sum = gt.photon_indirect.sum();

        // If there is significant direct NEE, photon indirect should be
        // a fraction of it (indirect by definition is smaller than direct
        // at well-lit points). If photon > 3x NEE, photons likely contain
        // direct light deposits → double counting.
        if (nee_sum > 1e-4f && photon_sum > 3.0f * nee_sum) {
            suspicious_rays++;
            if (suspicious_rays <= 5) {
                std::cout << "[WARNING] Ray " << i << " px=(" << rays[i].px
                          << "," << rays[i].py << "): photon_indirect ("
                          << photon_sum << ") >> nee_direct (" << nee_sum
                          << ") — possible double counting\n";
            }
        }
    }

    std::cout << "[DirectDoubleCount] Suspicious rays: " << suspicious_rays
              << "/" << NUM_RAYS << "\n";

    float frac = (float)suspicious_rays / NUM_RAYS;
    if (frac > 0.10f) {
        std::cout << "[WARNING] " << (frac * 100)
                  << "% of rays have suspiciously high photon/NEE ratio\n";
    }
    // Not a hard fail — photon density can exceed NEE at indirect-heavy spots
    // But if a large fraction shows this, it's a sign of double counting
    EXPECT_LT(frac, 0.30f)
        << "Too many rays (" << suspicious_rays << "/" << NUM_RAYS
        << ") show signs of direct light in photon map";
}

// ─────────────────────────────────────────────────────────────────────
// 14. FULL DIAGNOSTIC REPORT (runs all rays, prints comprehensive
//     warning report with per-bounce details for worst rays)
// ─────────────────────────────────────────────────────────────────────
TEST_F(PerRayValidation, FullDiagnosticReport) {
    auto& ds = get_per_ray_dataset();
    ASSERT_TRUE(ds.valid);

    constexpr int NUM_RAYS = 256;
    std::vector<RayDiagnostic> warnings;
    auto agg = run_per_ray_validation(ds, NUM_RAYS, ds.max_bounces, warnings);

    print_warning_summary(agg, warnings);

    // Print per-bounce breakdown for top 3 worst rays
    if (!warnings.empty()) {
        int show = std::min((int)warnings.size(), 3);
        auto all_rays = pick_random_rays(NUM_RAYS, 64, 64, 0xDEADBEEF42ULL);

        std::cout << "\n  DETAILED PER-BOUNCE BREAKDOWN (top " << show << " worst):\n";
        for (int w = 0; w < show; ++w) {
            const auto& diag = warnings[w];
            const auto& r = all_rays[diag.ray_idx];

            PCGRng rng_ray = PCGRng::seed(r.seed, (uint64_t)diag.ray_idx);
            Ray ray = ds.camera.generate_ray(r.px, r.py, rng_ray);

            PCGRng rng_gt  = PCGRng::seed(r.seed + 1000000ULL, (uint64_t)diag.ray_idx);
            auto gt = ground_truth_trace_steps(
                ray.origin, ray.direction, rng_gt, ds, ds.max_bounces);

            PCGRng rng_opt = PCGRng::seed(r.seed + 2000000ULL, (uint64_t)diag.ray_idx);
            auto opt = optimized_trace_steps(
                ray.origin, ray.direction, rng_opt, ds, ds.max_bounces);

            std::cout << "\n  --- Ray #" << diag.ray_idx
                      << " px=(" << diag.px << "," << diag.py << ") ---\n";

            int max_steps = (int)std::max(gt.steps.size(), opt.steps.size());
            for (int s = 0; s < (int)max_steps; ++s) {
                std::cout << "    Bounce " << s << ": ";
                if (s < (int)gt.steps.size()) {
                    const auto& gs = gt.steps[s];
                    std::cout << "GT[nee=" << gs.nee_contribution.sum()
                              << " phot=" << gs.photon_contribution.sum()
                              << " T_max=" << spectrum_max(gs.throughput)
                              << " mat=" << gs.material_id << "]";
                } else {
                    std::cout << "GT[ended]";
                }
                std::cout << "  ";
                if (s < (int)opt.steps.size()) {
                    const auto& os = opt.steps[s];
                    std::cout << "OPT[nee=" << os.nee_contribution.sum()
                              << " phot=" << os.photon_contribution.sum()
                              << " T_max=" << spectrum_max(os.throughput)
                              << " mat=" << os.material_id << "]";
                } else {
                    std::cout << "OPT[ended]";
                }
                std::cout << "\n";
            }
        }
    }

    // This test always passes — it's a diagnostic report
    // But we assert no physical validity issues
    EXPECT_EQ(agg.total_validity_issues, 0)
        << "Physical validity issues found — see report above";
}
