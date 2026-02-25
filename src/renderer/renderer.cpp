// ─────────────────────────────────────────────────────────────────────
// renderer.cpp – Photon-centric renderer (v2 Architecture)
// ─────────────────────────────────────────────────────────────────────
// Camera rays are first-hit probes with specular chain (§E3).
// All indirect lighting comes from the photon map.
// Direct lighting via NEE at the camera first-hit point.
// ─────────────────────────────────────────────────────────────────────
#include "renderer/renderer.h"
#include "renderer/direct_light.h"
#include "photon/emitter.h"
#include "photon/density_estimator.h"
#include "photon/surface_filter.h"
#include "bsdf/bsdf.h"
#include "core/random.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iomanip>

// ── Build photon maps ───────────────────────────────────────────────

void Renderer::build_photon_maps() {
    if (!scene_) return;

    auto t0 = std::chrono::high_resolution_clock::now();

    EmitterConfig emitter_cfg;
    emitter_cfg.num_photons    = config_.num_photons;
    emitter_cfg.max_bounces    = config_.max_bounces;
    emitter_cfg.rr_threshold   = config_.rr_threshold;
    emitter_cfg.min_bounces_rr = config_.min_bounces_rr;
    emitter_cfg.volume_enabled = false;  // §Q9: volume disabled in v2

    trace_photons(*scene_, emitter_cfg, global_photons_, caustic_photons_, nullptr);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "[Photon] Traced " << config_.num_photons << " photons in "
              << ms << " ms\n";
    std::cout << "[Photon] Global map:  " << global_photons_.size() << " stored\n";
    std::cout << "[Photon] Caustic map: " << caustic_photons_.size() << " stored\n";

    // Build spatial indices
    if (global_photons_.size() > 0) {
        if (config_.use_kdtree) {
            global_kdtree_.build(global_photons_);
            std::cout << "[KDTree] Global KD-tree built ("
                      << global_kdtree_.node_count() << " nodes)\n";
        }
        global_grid_.build(global_photons_, config_.gather_radius);
        std::cout << "[Grid] Global hash grid built ("
                  << global_grid_.table_size << " buckets)\n";
    }

    if (caustic_photons_.size() > 0) {
        if (config_.use_kdtree) {
            caustic_kdtree_.build(caustic_photons_);
            std::cout << "[KDTree] Caustic KD-tree built ("
                      << caustic_kdtree_.node_count() << " nodes)\n";
        }
        caustic_grid_.build(caustic_photons_, config_.caustic_radius);
        std::cout << "[Grid] Caustic hash grid built ("
                  << caustic_grid_.table_size << " buckets)\n";
    }

    // Build CellInfoCache (§10c) — precomputed per-cell statistics
    float cache_cell_size = config_.gather_radius * 2.0f;
    cell_cache_.build(global_photons_, caustic_photons_,
                      cache_cell_size, config_.gather_radius);

    // §10c: Adaptive caustic shooting — trace extra photons toward
    // caustic hotspot cells with high flux variance
    if (caustic_photons_.size() > 0) {
        EmitterConfig caustic_cfg;
        caustic_cfg.num_photons    = config_.num_photons;
        caustic_cfg.max_bounces    = config_.max_bounces;
        caustic_cfg.rr_threshold   = config_.rr_threshold;
        caustic_cfg.min_bounces_rr = config_.min_bounces_rr;
        caustic_cfg.volume_enabled = false;

        size_t before = caustic_photons_.size();
        trace_targeted_caustic_photons(*scene_, caustic_cfg,
                                       cell_cache_, caustic_photons_);

        // Rebuild caustic grid if new photons were added
        if (caustic_photons_.size() > before) {
            if (config_.use_kdtree) {
                caustic_kdtree_.build(caustic_photons_);
            }
            caustic_grid_.build(caustic_photons_, config_.caustic_radius);
            std::cout << "[Grid] Caustic hash grid rebuilt ("
                      << caustic_grid_.table_size << " buckets, "
                      << caustic_photons_.size() << " photons)\n";

            // Rebuild cell cache with the augmented maps
            cell_cache_.build(global_photons_, caustic_photons_,
                              cache_cell_size, config_.gather_radius);
        }
    }
}

// ── k-NN density estimate via KD-tree (§6.3 tangential kernel) ──────
//
// Adaptive-radius gather: find the k nearest photons using tangential
// distance, use distance to the k-th as the effective radius.

static Spectrum estimate_density_knn(
    float3 hit_pos, float3 hit_normal, float3 wo_local,
    const Material& mat,
    const PhotonSoA& photons, const KDTree& tree,
    int k, int num_photons_total)
{
    std::vector<uint32_t> indices;
    float max_dist2 = 0.f;
    float tau = effective_tau(DEFAULT_SURFACE_TAU);
    tree.knn_tangential(hit_pos, hit_normal, k, tau, photons,
                        indices, max_dist2);

    if (indices.empty()) return Spectrum::zero();

    // Normalization uses the k-th tangential distance as effective radius
    float inv_area = 1.0f / (PI * fmaxf(max_dist2, 1e-20f));
    float inv_N    = 1.0f / (float)num_photons_total;
    ONB frame = ONB::from_normal(hit_normal);

    Spectrum L = Spectrum::zero();
    for (uint32_t idx : indices) {
        // Conditions 3 & 4 (already passed tangential + plane in knn)
        if (!photons.norm_x.empty()) {
            float3 pn = make_f3(photons.norm_x[idx],
                                photons.norm_y[idx],
                                photons.norm_z[idx]);
            if (dot(pn, hit_normal) <= 0.0f) continue;
        }

        float3 wi = make_f3(photons.wi_x[idx],
                            photons.wi_y[idx],
                            photons.wi_z[idx]);
        if (dot(wi, hit_normal) <= 0.f) continue;

        float3 wi_loc = frame.world_to_local(wi);
        Spectrum f = bsdf::evaluate_diffuse(mat, wo_local, wi_loc);

        Spectrum photon_flux = photons.get_flux(idx);
        for (int b = 0; b < NUM_LAMBDA; ++b)
            L.value[b] += photon_flux.value[b] * inv_N * f.value[b] * inv_area;
    }
    return L;
}

// ── First-hit + specular chain + NEE + photon gather (v2 §E3) ───────

Renderer::TraceResult Renderer::render_pixel(Ray ray, PCGRng& rng) {
    PixelLighting pl = PixelLighting::zero();
    Spectrum throughput = Spectrum::constant(1.0f);

    DensityEstimatorConfig de_config;
    de_config.radius            = config_.gather_radius;
    de_config.caustic_radius    = config_.caustic_radius;
    de_config.num_photons_total = config_.num_photons;
    de_config.use_kernel        = true; // Epanechnikov (§6.3)

    const int max_spec = config_.max_specular_chain;

    for (int bounce = 0; bounce <= max_spec; ++bounce) {
        HitRecord hit = scene_->intersect(ray);
        if (!hit.hit) break;

        // Apply diffuse texture
        Material mat = scene_->materials[hit.material_id];
        if (mat.diffuse_tex >= 0 &&
            mat.diffuse_tex < (int)scene_->textures.size()) {
            float3 rgb = scene_->textures[mat.diffuse_tex].sample(hit.uv);
            mat.Kd = rgb_to_spectrum_reflectance(rgb.x, rgb.y, rgb.z);
        }

        // Emission: only on first bounce (camera sees a light)
        if (mat.is_emissive() && bounce == 0) {
            pl.emission += throughput * mat.Le;
        }

        // Debug render modes (return immediately)
        if (config_.mode != RenderMode::Combined) {
            switch (config_.mode) {
                case RenderMode::Normals:
                    { PixelLighting d = PixelLighting::zero(); d.emission = render_normals(hit);     return TraceResult::from(d); }
                case RenderMode::MaterialID:
                    { PixelLighting d = PixelLighting::zero(); d.emission = render_material_id(hit); return TraceResult::from(d); }
                case RenderMode::Depth:
                    { PixelLighting d = PixelLighting::zero(); d.emission = render_depth(hit, 5.0f); return TraceResult::from(d); }
                case RenderMode::PhotonMap:
                    { PixelLighting d = PixelLighting::zero(); d.emission = render_photon_density(hit, ray.direction * (-1.f));
                      return TraceResult::from(d); }
                default: break;
            }
        }

        // ── Specular bounce: continue the chain ─────────────────────
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
            continue;  // follow the specular chain
        }

        // ── Diffuse hit: terminate chain, gather radiance ───────────
        ONB frame = ONB::from_normal(hit.shading_normal);
        float3 wo_local = frame.world_to_local(ray.direction * (-1.f));
        if (wo_local.z <= 0.f) break;

        // 1. NEE: direct lighting
        if (config_.mode == RenderMode::Combined ||
            config_.mode == RenderMode::DirectOnly) {

            DirectLightSample dls = sample_direct_light(
                hit.position, hit.shading_normal, *scene_, rng);

            if (dls.visible && dls.pdf_light > 0.f) {
                float3 wi_local = frame.world_to_local(dls.wi);
                float cos_theta = fmaxf(0.f, wi_local.z);
                if (cos_theta > 0.f) {
                    Spectrum f = bsdf::evaluate(mat, wo_local, wi_local);
                    Spectrum contrib = dls.Li * f * (cos_theta / dls.pdf_light);
                    pl.direct_nee += throughput * contrib;
                }
            }
        }

        // 2. Photon gather: indirect lighting (global map)
        if ((config_.mode == RenderMode::Combined ||
             config_.mode == RenderMode::IndirectOnly) &&
            global_photons_.size() > 0) {

            // §10c: Adaptive gather radius from CellInfoCache
            float eff_radius = config_.gather_radius;
            if (!cell_cache_.cells.empty()) {
                eff_radius = cell_cache_.get_adaptive_radius(hit.position);
            }

            Spectrum L_photon;
            if (config_.use_kdtree && !global_kdtree_.empty()) {
                L_photon = estimate_density_knn(
                    hit.position, hit.normal, wo_local, mat,
                    global_photons_, global_kdtree_,
                    DEFAULT_KNN_K, config_.num_photons);
            } else {
                de_config.radius = eff_radius;
                L_photon = estimate_photon_density(
                    hit.position, hit.normal, wo_local, mat,
                    global_photons_, global_grid_, de_config,
                    eff_radius);
            }
            pl.indirect_global += throughput * L_photon;
        }

        // 3. Caustic map gather
        if (caustic_photons_.size() > 0) {
            // §10c: Skip caustic gather if cell has no caustic photons
            bool skip_caustic = false;
            if (!cell_cache_.cells.empty()) {
                const CellCacheInfo& ci = cell_cache_.query(hit.position);
                skip_caustic = (ci.caustic_count == 0);
            }

            if (!skip_caustic) {
                Spectrum L_caustic;
                if (config_.use_kdtree && !caustic_kdtree_.empty()) {
                    L_caustic = estimate_density_knn(
                        hit.position, hit.normal, wo_local, mat,
                        caustic_photons_, caustic_kdtree_,
                        DEFAULT_KNN_K, config_.num_photons);
                } else {
                    L_caustic = estimate_photon_density(
                        hit.position, hit.normal, wo_local, mat,
                        caustic_photons_, caustic_grid_, de_config,
                        config_.caustic_radius);
                }
                pl.indirect_caustic += throughput * L_caustic;
            }
        }

        // ── Glossy continuation: specular reflection for glossy surfaces ──
        // Trace mirror-direction rays weighted by G(wo,wi) × F(cosθ).
        // The NDF D cancels with the sampling PDF, giving the correct energy.
        // Only for near-mirror surfaces (roughness < 0.1) where one ray
        // at the specular peak approximates the lobe well.
        if ((mat.type == MaterialType::GlossyMetal ||
             mat.type == MaterialType::GlossyDielectric) &&
            mat.roughness < 0.1f) {

            constexpr int MAX_GLOSSY_BOUNCES = 4;
            Spectrum glossy_tp = throughput;
            float3 g_pos = hit.position;
            float3 g_dir = ray.direction;
            float3 g_normal = hit.shading_normal;

            for (int gb = 0; gb < MAX_GLOSSY_BOUNCES; ++gb) {
                // Mirror reflection weighted by Cook-Torrance specular
                float3 refl = g_dir - g_normal * (2.f * dot(g_dir, g_normal));
                refl = normalize(refl);
                float cos_v = fabsf(dot(normalize(g_dir * (-1.f)), g_normal));

                float alpha_g = bsdf_roughness_to_alpha(mat.roughness);

                // Smith G for mirror direction (wo = wi angle)
                ONB gf = ONB::from_normal(g_normal);
                float3 wo_g = gf.world_to_local(g_dir * (-1.f));
                float G1 = ggx_G1(wo_g, alpha_g);
                float G_val = G1 * G1;  // G(wo,wi) ≈ G1(wo)·G1(wi)

                // Glossy continuation weight = G · F
                // (the GGX NDF D in the BRDF cancels with the sampling PDF,
                //  so the correct single-sample weight is just G × Fresnel)
                if (mat.type == MaterialType::GlossyMetal) {
                    for (int i = 0; i < NUM_LAMBDA; ++i) {
                        float Fr = fresnel_schlick(cos_v, mat.Ks.value[i]);
                        glossy_tp.value[i] *= G_val * Fr;
                    }
                } else {
                    float f0t = (mat.ior - 1.f) / (mat.ior + 1.f);
                    float F0 = f0t * f0t;
                    float Fr = fresnel_schlick(cos_v, F0);
                    for (int i = 0; i < NUM_LAMBDA; ++i)
                        glossy_tp.value[i] *= G_val * Fr * mat.Ks.value[i];
                }

                if (glossy_tp.max_component() < 0.001f) break;

                // Trace reflection
                Ray refl_ray{g_pos + g_normal * EPSILON, refl};
                HitRecord refl_hit = scene_->intersect(refl_ray);
                if (!refl_hit.hit) break;

                const auto& rmat_ref = scene_->materials[refl_hit.material_id];
                // Apply diffuse texture to reflected material (same pattern as primary hit)
                Material rmat = rmat_ref;
                if (rmat.diffuse_tex >= 0 &&
                    rmat.diffuse_tex < (int)scene_->textures.size()) {
                    float3 rgb = scene_->textures[rmat.diffuse_tex].sample(refl_hit.uv);
                    rmat.Kd = rgb_to_spectrum_reflectance(rgb.x, rgb.y, rgb.z);
                }
                if (rmat.specular_tex >= 0 &&
                    rmat.specular_tex < (int)scene_->textures.size()) {
                    float3 rgb = scene_->textures[rmat.specular_tex].sample(refl_hit.uv);
                    rmat.Ks = rgb_to_spectrum_reflectance(rgb.x, rgb.y, rgb.z);
                }
                if (rmat.is_emissive()) {
                    // CPU glossy continuation uses a deterministic mirror
                    // direction (delta-function BSDF), so pdf_bsdf → ∞
                    // and the 2-way power-heuristic MIS weight collapses to
                    // w_bsdf = 1.0.  NEE at the primary hit uses the *diffuse*
                    // BSDF lobe, which is a different component — not competing
                    // with this specular estimator — so no double counting exists
                    // here.  (DEFAULT_USE_MIS has no numerical effect for the
                    // CPU mirror path; it will matter when stochastic BSDF
                    // sampling replaces the mirror continuation.)
                    pl.glossy_indirect += glossy_tp * rmat.Le;
                    break;
                }

                // NEE at reflected hit
                if (config_.mode == RenderMode::Combined ||
                    config_.mode == RenderMode::DirectOnly) {
                    ONB rf = ONB::from_normal(refl_hit.shading_normal);
                    float3 rwo = rf.world_to_local(refl * (-1.f));
                    if (rwo.z > 0.f) {
                        DirectLightSample rdls = sample_direct_light(
                            refl_hit.position, refl_hit.shading_normal, *scene_, rng);
                        if (rdls.visible && rdls.pdf_light > 0.f) {
                            float3 rwi = rf.world_to_local(rdls.wi);
                            float rcos = fmaxf(0.f, rwi.z);
                            if (rcos > 0.f) {
                                Spectrum rf_val = bsdf::evaluate(rmat, rwo, rwi);
                                pl.glossy_indirect += glossy_tp * rdls.Li * rf_val * (rcos / rdls.pdf_light);
                            }
                        }
                    }
                }

                // Continue if reflected surface is also glossy AND near-mirror
                if ((rmat.type != MaterialType::GlossyMetal &&
                     rmat.type != MaterialType::GlossyDielectric) ||
                    rmat.roughness >= 0.1f) break;

                g_pos = refl_hit.position;
                g_dir = refl;
                g_normal = refl_hit.shading_normal;
            }
        }

        break;  // stop at first diffuse/glossy hit (after glossy continuation)
    }

    return TraceResult::from(pl);
}

// ── Legacy wrapper (backward compatibility) ─────────────────────────

Renderer::TraceResult Renderer::trace_path(Ray ray, PCGRng& rng) {
    return render_pixel(ray, rng);
}

// ── Debug render modes ──────────────────────────────────────────────

Spectrum Renderer::render_normals(const HitRecord& hit) {
    Spectrum s = Spectrum::zero();
    float3 n = hit.shading_normal;
    // Map normal components [-1,1] → [0,1] and store as visible color
    float r = n.x * 0.5f + 0.5f;
    float g = n.y * 0.5f + 0.5f;
    float b = n.z * 0.5f + 0.5f;
    return rgb_to_spectrum_reflectance(r, g, b);
}

Spectrum Renderer::render_material_id(const HitRecord& hit) {
    // Assign distinct hue per material ID
    float hue = fmodf(hit.material_id * 0.618033988f, 1.f);
    float r = fabsf(hue * 6.f - 3.f) - 1.f;
    float g = 2.f - fabsf(hue * 6.f - 2.f);
    float b = 2.f - fabsf(hue * 6.f - 4.f);
    r = fmaxf(0.f, fminf(1.f, r));
    g = fmaxf(0.f, fminf(1.f, g));
    b = fmaxf(0.f, fminf(1.f, b));
    return rgb_to_spectrum_reflectance(r, g, b);
}

Spectrum Renderer::render_depth(const HitRecord& hit, float max_depth) {
    float d = fminf(hit.t / max_depth, 1.f);
    float v = 1.f - d; // Near = bright, far = dark
    return Spectrum::constant(v);
}

Spectrum Renderer::render_photon_density(const HitRecord& hit, float3 wo_world) {
    ONB frame = ONB::from_normal(hit.shading_normal);
    float3 wo_local = frame.world_to_local(wo_world);

    DensityEstimatorConfig de_config;
    de_config.radius = config_.gather_radius;
    de_config.num_photons_total = config_.num_photons;
    de_config.use_kernel        = true; // Epanechnikov (§6.3)

    const Material& mat = scene_->materials[hit.material_id];

    return estimate_photon_density(
        hit.position, hit.normal, wo_local, mat,
        global_photons_, global_grid_, de_config, config_.gather_radius);
}

// ── Render frame ────────────────────────────────────────────────────

void Renderer::render_frame() {
    if (!scene_) return;

    fb_.resize(config_.image_width, config_.image_height);

    auto t0 = std::chrono::high_resolution_clock::now();

    const int height       = config_.image_height;
    const int width        = config_.image_width;
    const int total_pixels = width * height;
    const int spp          = config_.samples_per_pixel;
    const bool adaptive    = config_.adaptive_sampling;
    const int  min_spp     = config_.adaptive_min_spp;
    const int  max_spp     = (config_.adaptive_max_spp > 0)
                                 ? config_.adaptive_max_spp
                                 : spp;
    const float threshold  = config_.adaptive_threshold;
    const int   radius     = config_.adaptive_radius;
    const int   update_int = config_.adaptive_update_interval;

    if (!adaptive) {
        // ── Non-adaptive path (original behaviour) ─────────────────
        #pragma omp parallel for schedule(dynamic, 1)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int pixel_idx = y * width + x;
                for (int s = 0; s < spp; ++s) {
                    PCGRng rng = PCGRng::seed(
                        (uint64_t)pixel_idx * 1000 + s,
                        (uint64_t)pixel_idx + 1);
                    Ray ray = camera_.generate_ray(x, y, rng);
                    auto L = trace_path(ray, rng);
                    fb_.accumulate(x, y, L.combined);
                }
            }
        }
    } else {
        // ── Adaptive path ───────────────────────────────────────────
        // active_mask: 0=done, 1=keep sampling
        std::vector<uint8_t> active(total_pixels, 1u);

        // Helper: recompute noise-based active mask on the CPU
        auto update_mask = [&]() {
            constexpr float eps = 1e-4f;
            #pragma omp parallel for schedule(static)
            for (int py = 0; py < height; ++py) {
                for (int px = 0; px < width; ++px) {
                    int idx = py * width + px;
                    float n = fb_.sample_count[idx];

                    if (n < (float)min_spp) { active[idx] = 1; continue; }
                    if (n >= (float)max_spp) { active[idx] = 0; continue; }

                    // Neighbourhood max relative noise
                    float local_noise = 0.f;
                    for (int dy = -radius; dy <= radius; ++dy) {
                        int ny = py + dy;
                        if (ny < 0 || ny >= height) continue;
                        for (int dx = -radius; dx <= radius; ++dx) {
                            int nx = px + dx;
                            if (nx < 0 || nx >= width) continue;
                            int nidx = ny * width + nx;
                            float nn = fb_.sample_count[nidx];
                            if (nn < 2.f) { local_noise = 1.f; break; }
                            float mu  = fb_.lum_sum[nidx] / nn;
                            float var = std::fmax(fb_.lum_sum2[nidx] / nn
                                                  - mu * mu, 0.f);
                            float se  = std::sqrt(var / nn);
                            float rel = se / (std::fabs(mu) + eps);
                            local_noise = std::fmax(local_noise, rel);
                        }
                    }
                    active[idx] = (local_noise > threshold) ? 1u : 0u;
                }
            }
        };

        for (int pass = 0; pass < max_spp; ++pass) {
            // Phase 1: warmup — always sample
            // Phase 2: adaptive — update mask every update_interval passes
            if (pass >= min_spp && (pass - min_spp) % update_int == 0) {
                update_mask();
                int active_count = 0;
                for (int i = 0; i < total_pixels; ++i)
                    active_count += active[i];
                if (active_count == 0) break;  // fully converged
            }

            #pragma omp parallel for schedule(dynamic, 1)
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int pixel_idx = y * width + x;
                    if (pass >= min_spp && !active[pixel_idx]) continue;

                    PCGRng rng = PCGRng::seed(
                        (uint64_t)pixel_idx * 1000 + pass,
                        (uint64_t)pixel_idx + 1);
                    Ray ray = camera_.generate_ray(x, y, rng);
                    auto L = trace_path(ray, rng);
                    const Spectrum* proxy = ADAPTIVE_NOISE_USE_DIRECT_ONLY
                        ? &L.nee_direct : nullptr;
                    fb_.accumulate(x, y, L.combined, proxy);
                }
            }

            // Progress
            float pct = 100.f * (pass + 1) / max_spp;
            std::cout << "\r[Render] " << (int)pct << "%  pass "
                      << (pass + 1) << "/" << max_spp << "   " << std::flush;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\r[Render] 100%  (" << sec << " s, "
              << config_.samples_per_pixel << " spp"
              << (adaptive ? ", adaptive" : "") << ")\n";

    fb_.tonemap(1.0f);
}
// =====================================================================
//  SPPM render (CPU path)
// =====================================================================
// Full SPPM loop:
//   for each iteration k:
//     1. Camera pass  → find visible points (first diffuse hit)
//     2. Photon pass  → retrace photons, rebuild hash grid
//     3. Gather pass  → query photons per pixel, accumulate flux
//     4. Progressive update → shrink radius, adjust counts
//   Final: reconstruct L = tau / (pi * r^2 * k * N_p) + L_direct/k
// =====================================================================

void Renderer::render_sppm() {
    if (!scene_) return;

    const int width  = config_.image_width;
    const int height = config_.image_height;
    const int K      = config_.sppm_iterations;
    const int N_p    = config_.num_photons;
    const float alpha      = config_.sppm_alpha;
    const float min_radius = config_.sppm_min_radius;

    fb_.resize(width, height);

    // Initialise per-pixel SPPM state
    SPPMBuffer sppm;
    sppm.resize(width, height, config_.sppm_initial_radius);

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < K; ++k) {
        // ── 1. Camera pass: find visible points ─────────────────────
        #pragma omp parallel for schedule(dynamic, 1)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                SPPMPixel& sp = sppm.at(x, y);

                PCGRng rng = PCGRng::seed(
                    (uint64_t)(y * width + x) * 1000 + k,
                    (uint64_t)(y * width + x) + 1);

                Ray ray = camera_.generate_ray(x, y, rng);
                Spectrum throughput = Spectrum::constant(1.f);
                sp.valid = false;

                for (int bounce = 0; bounce <= config_.max_specular_chain; ++bounce) {
                    HitRecord hit = scene_->intersect(ray);
                    if (!hit.hit) break;

                    Material mat = scene_->materials[hit.material_id];

                    // Texture lookup
                    if (mat.diffuse_tex >= 0 &&
                        mat.diffuse_tex < (int)scene_->textures.size()) {
                        float3 rgb = scene_->textures[mat.diffuse_tex].sample(hit.uv);
                        mat.Kd = rgb_to_spectrum_reflectance(rgb.x, rgb.y, rgb.z);
                    }

                    // Emission (camera sees light directly or via specular)
                    if (mat.is_emissive() && bounce == 0) {
                        sp.L_direct += throughput * mat.Le;
                    }

                    // Specular: bounce and continue
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
                        continue;
                    }

                    // ── First diffuse hit: store visible point ──────
                    ONB frame = ONB::from_normal(hit.shading_normal);
                    float3 wo_local = frame.world_to_local(ray.direction * (-1.f));
                    if (wo_local.z <= 0.f) break;

                    sp.position    = hit.position;
                    sp.normal      = hit.normal;  // geometric normal for gather filtering
                    sp.wo_local    = wo_local;
                    sp.material_id = hit.material_id;
                    sp.uv          = hit.uv;
                    sp.throughput  = throughput;
                    sp.valid       = true;

                    // NEE at visible point (direct lighting)
                    DirectLightSample dls = sample_direct_light(
                        hit.position, hit.shading_normal, *scene_, rng);

                    if (dls.visible && dls.pdf_light > 0.f) {
                        float3 wi_local = frame.world_to_local(dls.wi);
                        float cos_theta = fmaxf(0.f, wi_local.z);
                        if (cos_theta > 0.f) {
                            Spectrum f = bsdf::evaluate(mat, wo_local, wi_local);
                            Spectrum contrib = dls.Li * f *
                                               (cos_theta / dls.pdf_light);
                            sp.L_direct += throughput * contrib;
                        }
                    }

                    break;  // stop at first diffuse hit
                }
            }
        }

        // ── 2. Photon pass: retrace photons and rebuild hash grid ───
        EmitterConfig emitter_cfg;
        emitter_cfg.num_photons    = N_p;
        emitter_cfg.max_bounces    = config_.max_bounces;
        emitter_cfg.rr_threshold   = config_.rr_threshold;
        emitter_cfg.min_bounces_rr = config_.min_bounces_rr;
        emitter_cfg.volume_enabled = false;

        global_photons_ = PhotonSoA{};
        caustic_photons_ = PhotonSoA{};
        trace_photons(*scene_, emitter_cfg, global_photons_, caustic_photons_, nullptr);

        if (global_photons_.size() > 0) {
            if (config_.use_kdtree) {
                global_kdtree_.build(global_photons_);
            }
            global_grid_.build(global_photons_, config_.sppm_initial_radius);
        }

        // ── 3 & 4. Gather pass + progressive update ─────────────────
        #pragma omp parallel for schedule(dynamic, 1)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                SPPMPixel& sp = sppm.at(x, y);
                if (!sp.valid) continue;

                Material mat = scene_->materials[sp.material_id];
                if (mat.diffuse_tex >= 0 &&
                    mat.diffuse_tex < (int)scene_->textures.size()) {
                    float3 rgb = scene_->textures[mat.diffuse_tex].sample(sp.uv);
                    mat.Kd = rgb_to_spectrum_reflectance(rgb.x, rgb.y, rgb.z);
                }

                int M = 0;
                Spectrum phi = sppm_gather(
                    sp.position, sp.normal, sp.wo_local, mat,
                    global_photons_, global_grid_,
                    sp.radius, DEFAULT_SURFACE_TAU, M);

                // Bake camera throughput into the flux contribution
                for (int i = 0; i < NUM_LAMBDA; ++i)
                    phi.value[i] *= sp.throughput.value[i];

                sppm_progressive_update(sp, phi, M, alpha, min_radius);
            }
        }

        // Progress
        float pct = 100.f * (k + 1) / K;
        auto t_now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t_now - t_start).count();
        std::cout << "\r[SPPM] " << (int)pct << "%  iteration "
                  << (k + 1) << "/" << K
                  << "  " << std::fixed << std::setprecision(1)
                  << elapsed << "s" << std::flush;
    }

    // ── 5. Reconstruction: L = tau / (pi * r^2 * k * N_p) + L_direct/k ──
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const SPPMPixel& sp = sppm.at(x, y);
            Spectrum L = sppm_reconstruct(sp, K, N_p);
            // Store as a single-sample entry for the framebuffer
            fb_.pixels[y * width + x] = L;
            fb_.sample_count[y * width + x] = 1.f;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "\n[SPPM] Done in " << total_s << " s"
              << "  (" << K << " iterations, " << N_p << " photons/iter)\n";

    fb_.tonemap(1.0f);
}