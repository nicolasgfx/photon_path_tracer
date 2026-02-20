// ─────────────────────────────────────────────────────────────────────
// renderer.cpp – Path tracing core and render pipeline
// ─────────────────────────────────────────────────────────────────────
#include "renderer/renderer.h"
#include "renderer/direct_light.h"
#include "renderer/mis.h"
#include "photon/emitter.h"
#include "photon/density_estimator.h"
#include "bsdf/bsdf.h"
#include "core/random.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdint>

// ── Build photon maps ───────────────────────────────────────────────

void Renderer::build_photon_maps() {
    if (!scene_) return;

    auto t0 = std::chrono::high_resolution_clock::now();

    EmitterConfig emitter_cfg;
    emitter_cfg.num_photons    = config_.num_photons;
    emitter_cfg.max_bounces    = config_.max_bounces;
    emitter_cfg.rr_threshold   = config_.rr_threshold;
    emitter_cfg.min_bounces_rr = config_.min_bounces_rr;

    trace_photons(*scene_, emitter_cfg, global_photons_, caustic_photons_);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "[Photon] Traced " << config_.num_photons << " photons in "
              << ms << " ms\n";
    std::cout << "[Photon] Global map:  " << global_photons_.size() << " stored\n";
    std::cout << "[Photon] Caustic map: " << caustic_photons_.size() << " stored\n";

    // Build hash grids
    if (global_photons_.size() > 0) {
        global_grid_.build(global_photons_, config_.gather_radius);
        std::cout << "[Grid] Global hash grid built ("
                  << global_grid_.table_size << " buckets)\n";
    }

    if (caustic_photons_.size() > 0) {
        caustic_grid_.build(caustic_photons_, config_.caustic_radius);
        std::cout << "[Grid] Caustic hash grid built ("
                  << caustic_grid_.table_size << " buckets)\n";
    }
}

// ── Trace a single camera path ──────────────────────────────────────

Renderer::TraceResult Renderer::trace_path(Ray ray, PCGRng& rng) {
    Spectrum L_total = Spectrum::zero();      // Accumulated radiance
    Spectrum L_nee   = Spectrum::zero();      // Direct-only component (proxy for noise)
    Spectrum throughput = Spectrum::constant(1.0f);  // Path throughput

    DensityEstimatorConfig de_config;
    de_config.radius = config_.gather_radius;
    de_config.caustic_radius = config_.caustic_radius;
    de_config.num_photons_total = config_.num_photons;

    for (int bounce = 0; bounce <= config_.max_bounces; ++bounce) {
        HitRecord hit = scene_->intersect(ray);
        if (!hit.hit) break;

        const Material& mat = scene_->materials[hit.material_id];

        // Handle emission (camera sees a light directly, or via specular bounce)
        if (mat.is_emissive() && bounce == 0) {
            L_total += throughput * mat.Le;
        }

        // Handle debug render modes
        if (config_.mode != RenderMode::Full) {
            switch (config_.mode) {
                case RenderMode::Normals:
                    { auto s = render_normals(hit);     return TraceResult{s, s}; }
                case RenderMode::MaterialID:
                    { auto s = render_material_id(hit); return TraceResult{s, s}; }
                case RenderMode::Depth:
                    { auto s = render_depth(hit, 5.0f); return TraceResult{s, s}; }
                case RenderMode::PhotonMap:
                    { auto s = render_photon_density(hit, ray.direction * (-1.f));
                      return TraceResult{s, s}; }
                default:
                    break;
            }
        }

        // For specular materials, just bounce (no direct light / photon map)
        if (mat.is_specular()) {
            ONB frame = ONB::from_normal(hit.shading_normal);
            float3 wo_local = frame.world_to_local(ray.direction * (-1.f));

            BSDFSample bs = bsdf::sample(mat, wo_local, rng);
            if (bs.pdf <= 0.f) break;

            float cos_theta = fabsf(bs.wi.z);

            // Update throughput per wavelength
            for (int i = 0; i < NUM_LAMBDA; ++i) {
                throughput.value[i] *= bs.f.value[i] * cos_theta / bs.pdf;
            }

            // Set up next ray
            ray.origin    = hit.position + hit.shading_normal * EPSILON *
                            (bs.wi.z > 0.f ? 1.f : -1.f);
            ray.direction = frame.local_to_world(bs.wi);
            continue;
        }

        // ── Diffuse hit: combine direct light + photon map + BSDF ──

        ONB frame = ONB::from_normal(hit.shading_normal);
        float3 wo_local = frame.world_to_local(ray.direction * (-1.f));
        if (wo_local.z <= 0.f) break;

        // ── 1. Direct light sampling (NEE) ──────────────────────────
        if (config_.mode == RenderMode::Full ||
            config_.mode == RenderMode::DirectOnly) {

            DirectLightSample dls = sample_direct_light(
                hit.position, hit.shading_normal, *scene_, rng);

            if (dls.visible && dls.pdf_light > 0.f) {
                float3 wi_local = frame.world_to_local(dls.wi);
                float cos_theta = fmaxf(0.f, wi_local.z);

                if (cos_theta > 0.f) {
                    Spectrum f = bsdf::evaluate(mat, wo_local, wi_local);
                    float pdf_bsdf = bsdf::pdf(mat, wo_local, wi_local);

                    float pdf_photon = 0.f;
                    if (config_.use_photon_guided && global_photons_.size() > 0) {
                        pdf_photon = photon_guided_pdf(
                            dls.wi, hit.position, hit.shading_normal,
                            global_photons_, global_grid_, config_.gather_radius);
                    }

                    float w = config_.use_mis
                        ? mis_weight_3(dls.pdf_light, pdf_bsdf, pdf_photon)
                        : 1.0f;

                    Spectrum contrib = dls.Li * f * (cos_theta * w / dls.pdf_light);
                    L_total += throughput * contrib;
                    L_nee   += throughput * contrib;  // NEE shadow-ray: low-variance proxy
                }
            }
        }

        // ── 2. BSDF sampling ────────────────────────────────────────
        BSDFSample bs = bsdf::sample(mat, wo_local, rng);
        if (bs.pdf > 0.f && !bs.is_specular) {
            float3 wi_world = frame.local_to_world(bs.wi);
            float cos_theta = fabsf(bs.wi.z);

            // Check if this direction hits a light
            Ray bsdf_ray;
            bsdf_ray.origin    = hit.position + hit.shading_normal * EPSILON;
            bsdf_ray.direction = wi_world;

            HitRecord bsdf_hit = scene_->intersect(bsdf_ray);
            if (bsdf_hit.hit) {
                const Material& bsdf_mat = scene_->materials[bsdf_hit.material_id];
                if (bsdf_mat.is_emissive()) {
                    float pdf_light = direct_light_pdf(
                        hit.position + hit.shading_normal * EPSILON,
                        wi_world, *scene_);

                    float pdf_photon = 0.f;
                    if (config_.use_photon_guided && global_photons_.size() > 0) {
                        pdf_photon = photon_guided_pdf(
                            wi_world, hit.position, hit.shading_normal,
                            global_photons_, global_grid_, config_.gather_radius);
                    }

                    float w = config_.use_mis
                        ? mis_weight_3(bs.pdf, pdf_light, pdf_photon)
                        : 1.0f;

                    Spectrum contrib = bsdf_mat.Le * bs.f * (cos_theta * w / bs.pdf);
                    L_total += throughput * contrib;
                    L_nee   += throughput * contrib;  // BSDF-sampled emitter: also direct
                }
            }
        }

        // ── 3. Photon density estimate (indirect) ───────────────────
        if ((config_.mode == RenderMode::Full ||
             config_.mode == RenderMode::IndirectOnly) &&
            global_photons_.size() > 0) {

            Spectrum L_photon = estimate_photon_density(
                hit.position, hit.shading_normal, wo_local, mat,
                global_photons_, global_grid_, de_config, config_.gather_radius);

            L_total += throughput * L_photon;
        }

        // ── 4. Caustic map contribution ─────────────────────────────
        if (caustic_photons_.size() > 0) {
            Spectrum L_caustic = estimate_photon_density(
                hit.position, hit.shading_normal, wo_local, mat,
                caustic_photons_, caustic_grid_, de_config, config_.caustic_radius);

            L_total += throughput * L_caustic;
        }

        // ── Continue path (throughput update) ───────────────────────
        if (bs.pdf <= 0.f) break;

        float cos_theta = fabsf(bs.wi.z);
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            throughput.value[i] *= bs.f.value[i] * cos_theta / bs.pdf;
        }

        // Russian roulette
        if (bounce >= config_.min_bounces_rr) {
            float p_rr = fminf(config_.rr_threshold, throughput.max_component());
            if (rng.next_float() >= p_rr) break;
            throughput *= 1.0f / p_rr;
        }

        // Next ray
        float3 wi_world = frame.local_to_world(bs.wi);
        ray.origin    = hit.position + hit.shading_normal * EPSILON;
        ray.direction = wi_world;
    }

    return TraceResult{L_total, L_nee};
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

    const Material& mat = scene_->materials[hit.material_id];

    return estimate_photon_density(
        hit.position, hit.shading_normal, wo_local, mat,
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
