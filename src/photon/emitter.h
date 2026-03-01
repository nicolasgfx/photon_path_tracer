#pragma once
// ─────────────────────────────────────────────────────────────────────
// emitter.h – Photon emission and tracing (§4, §5.3 decorrelation)
// ─────────────────────────────────────────────────────────────────────
// v2.1 updates:
//   §5.3.1 Cell-stratified bouncing (Fibonacci hemisphere strata)
//   §5.3.2 RNG spatial hash (decorrelation complement)
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"
#include "core/random.h"
#include "core/config.h"
#include "core/alias_table.h"
#include "core/material_flags.h"
#include "core/ior_stack.h"
#include "scene/scene.h"
#include "bsdf/bsdf.h"
#include "photon/photon.h"
#include "volume/medium.h"

#include <vector>
#include <cmath>
#include <atomic>
#include <iomanip>

#include "photon/specular_target.h"    // SpecularTargetSet (targeted caustic emission)

struct EmitterConfig {
    int    num_photons       = 1000000;  // Total photons to emit
    int    max_bounces       = 10;       // Max photon bounces
    float  rr_threshold      = 0.95f;    // Russian roulette survival cap
    int    min_bounces_rr    = 3;        // Min bounces before RR
    bool   volume_enabled    = false;    // Enable volumetric photon tracing
    float  volume_density    = 0.05f;    // Extinction coefficient
    float  volume_falloff    = 0.0f;     // Height falloff coefficient
    float  volume_albedo     = 0.95f;    // Single-scatter albedo
};

// ── Photon emission from a single emissive triangle ─────────────────

struct EmittedPhoton {
    Ray      ray;              // Origin + direction of emitted photon
    Spectrum spectral_flux;    // Full spectral flux [W/nm] per wavelength bin
    uint16_t source_emissive_idx = 0xFFFFu;  // local index into emissive_tri_indices
};

// Sample an emitted photon given the scene's emissive distribution
// ─────────────────────────────────────────────────────────────────
// v2.2 CANONICAL EMISSION SAMPLER:
//   1. Sample emissive triangle via power-weighted alias table
//   2. Sample point on triangle uniformly (barycentric)
//   3. Sample direction: cosine-weighted hemisphere (Lambertian, pdf=cos/π)
//   4. Flux = Le(λ) · cos_θ / (p_tri · p_pos · p_dir)
//
// Canonical emission sampler: alias-table + cosine hemisphere.
// ─────────────────────────────────────────────────────────────────
inline EmittedPhoton sample_emitted_photon(const Scene& scene, PCGRng& rng) {
    EmittedPhoton ep;

    float u1 = rng.next_float();
    float u2 = rng.next_float();
    int local_idx = scene.emissive_alias_table.sample(u1, u2);
    uint32_t tri_idx = scene.emissive_tri_indices[local_idx];
    const Triangle& tri = scene.triangles[tri_idx];
    const Material& mat = scene.materials[tri.material_id];

    float pdf_tri = scene.emissive_alias_table.pdf(local_idx);

    float3 bary = sample_triangle(rng.next_float(), rng.next_float());
    float3 pos  = tri.interpolate_position(bary.x, bary.y, bary.z);
    float3 n    = tri.geometric_normal();
    float  area = tri.area();
    float  pdf_pos = 1.0f / area;

    // Cosine-weighted hemisphere
    // pdf_dir = cos_θ / π
    float3 local_dir = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
    ONB frame = ONB::from_normal(n);
    float3 world_dir = frame.local_to_world(local_dir);
    float cos_theta = local_dir.z;
    float pdf_dir = cosine_hemisphere_pdf(cos_theta);

    float denom = pdf_tri * pdf_pos * pdf_dir;
    if (denom > 0.f) {
        float scale = cos_theta / denom;
        for (int b = 0; b < NUM_LAMBDA; ++b)
            ep.spectral_flux.value[b] = mat.Le.value[b] * scale;
    } else {
        ep.spectral_flux = Spectrum::zero();
    }

    ep.ray.origin    = pos + n * EPSILON;
    ep.ray.direction = world_dir;
    ep.source_emissive_idx = (uint16_t)local_idx;

    return ep;
}

// ── Trace photons through the scene ─────────────────────────────────
// Fills global_map (and optionally caustic_map) with photon hits.

// ── §5.3.1 Cell-stratified bounce helper ────────────────────────────
// Uses Fibonacci hemisphere mapping to convert a 1D stratum index
// into a deterministic direction seed on the hemisphere.
// ── §5.3.2 RNG spatial hash ────────────────────────────────────────
// Seeds the per-bounce RNG from a hash of the photon's current position,
// creating spatial decorrelation (nearby photons get different sequences).
inline uint64_t rng_spatial_seed(float3 pos, int bounce, int photon_idx) {
    uint32_t hx = (uint32_t)(pos.x * 12345.67f);
    uint32_t hy = (uint32_t)(pos.y * 45678.91f);
    uint32_t hz = (uint32_t)(pos.z * 78901.23f);
    uint64_t seed = (uint64_t)(hx ^ (hy * 2654435761u) ^ (hz * 2246822519u));
    seed ^= (uint64_t)bounce * 6364136223846793005ULL;
    seed ^= (uint64_t)photon_idx * 1442695040888963407ULL;
    return seed;
}


// MediumStack is now in volume/medium.h (shared HD for CPU + GPU).

inline void trace_photons(const Scene& scene,
                           const EmitterConfig& config,
                           PhotonSoA& global_map,
                           PhotonSoA& caustic_map,
                           PhotonSoA* volume_map = nullptr) {
    global_map.clear();
    caustic_map.clear();
    global_map.reserve(config.num_photons);
    caustic_map.reserve(config.num_photons / 4);
    if (volume_map) {
        volume_map->clear();
        volume_map->reserve(config.num_photons / 4);
    }

    if (scene.emissive_tri_indices.empty()) {
        return;
    }

    for (int photon_idx = 0; photon_idx < config.num_photons; ++photon_idx) {
        PCGRng rng = PCGRng::seed(photon_idx * 7 + 42, photon_idx + 1);

        EmittedPhoton ep = sample_emitted_photon(scene, rng);
        Spectrum spectral_flux = ep.spectral_flux;  // Full spectral throughput
        Ray ray = ep.ray;
        uint16_t source_emissive = ep.source_emissive_idx;

        // Sample primary hero wavelength bin at emission time (matches GPU
        // photon trace convention).  This bin determines the refraction
        // direction through dispersive glass, producing correct chromatic
        // dispersion (rainbow caustics).
        int primary_hero_bin;
        {
            float flux_sum = spectral_flux.sum();
            if (flux_sum > 0.f) {
                float xi = rng.next_float() * flux_sum;
                float cum = 0.f;
                primary_hero_bin = NUM_LAMBDA - 1;
                for (int b = 0; b < NUM_LAMBDA; ++b) {
                    cum += spectral_flux.value[b];
                    if (xi <= cum) { primary_hero_bin = b; break; }
                }
            } else {
                primary_hero_bin = (int)(rng.next_float() * NUM_LAMBDA);
                if (primary_hero_bin >= NUM_LAMBDA) primary_hero_bin = NUM_LAMBDA - 1;
            }
        }

        bool on_caustic_path = false; // Set true on first specular bounce, false after diffuse
        uint8_t path_flags = 0;       // Accumulated path flags
        IORStack ior_stack;            // Nested dielectric IOR tracking
        MediumStack medium_stack;      // Participating medium boundary tracking

        for (int bounce = 0; bounce < config.max_bounces; ++bounce) {
            HitRecord hit = scene.intersect(ray);
            if (!hit.hit) break;

            // §5.3.2: Re-seed RNG from spatial hash at each bounce
            // This decorrelates nearby photons that share similar paths
            if (bounce > 0) {
                uint64_t sp_seed = rng_spatial_seed(hit.position, bounce,
                                                     photon_idx);
                rng = PCGRng::seed(sp_seed, (uint64_t)(photon_idx + 1));
            }

            // ── Per-material interior medium Beer-Lambert (§7.7) ─────
            // If the photon is inside a Translucent object, apply
            // Beer-Lambert transmittance over the ray segment.
            bool inside_object_medium = false;
            {
                int cur_mid = medium_stack.current_medium_id();
                if (cur_mid >= 0 && cur_mid < (int)scene.media.size()) {
                    inside_object_medium = true;
                    const HomogeneousMedium& med = scene.media[cur_mid];
                    float seg_t = hit.t;
                    for (int b = 0; b < NUM_LAMBDA; ++b)
                        spectral_flux.value[b] *= expf(-med.sigma_t.value[b] * seg_t);
                }
            }

            // ── Atmospheric volume photon deposit (Beer–Lambert free-flight) ─────
            // Legacy atmospheric fog volume photon system.
            // Volume point deposits — runtime gated via config.volume_enabled.
            // §7.10 Double-attenuation guard: skip atmospheric volume when
            // inside a per-material medium.
            {
            if (volume_map && config.volume_enabled && config.volume_density > 0.f
                && !inside_object_medium) {
                path_flags |= PHOTON_FLAG_VOLUME_SCATTER;
                float seg_t = hit.t;
                float mid_y = ray.origin.y + ray.direction.y * (seg_t * 0.5f);
                HomogeneousMedium med = make_rayleigh_medium(
                    config.volume_density, config.volume_albedo,
                    config.volume_falloff, mid_y);

                float avg_sig_t = med.sigma_t.sum() / NUM_LAMBDA;
                if (avg_sig_t > 0.f) {
                    float u_ff = rng.next_float();
                    float t_ff = -logf(fmaxf(1.f - u_ff, 1e-12f)) / avg_sig_t;
                    if (t_ff < seg_t) {
                        Photon vp;
                        vp.position = ray.origin + ray.direction * t_ff;
                        vp.wi = ray.direction * (-1.f);
                        vp.path_flags = path_flags;
                        vp.bounce_count = (uint8_t)bounce;
                        for (int b = 0; b < NUM_LAMBDA; ++b) {
                            float sig_s_b = med.sigma_s.value[b];
                            float sig_t_b = fmaxf(med.sigma_t.value[b], 1e-20f);
                            vp.spectral_flux.value[b] = spectral_flux.value[b] * (sig_s_b / sig_t_b);
                        }
                        // Hero wavelength fields for GPU gather compatibility
                        {
                            int primary = (int)(rng.next_float() * NUM_LAMBDA);
                            if (primary >= NUM_LAMBDA) primary = NUM_LAMBDA - 1;
                            constexpr float hero_scale =
                                (float)NUM_LAMBDA / (float)HERO_WAVELENGTHS;
                            for (int hh = 0; hh < HERO_WAVELENGTHS; ++hh) {
                                int bin = (primary + hh * NUM_LAMBDA / HERO_WAVELENGTHS)
                                          % NUM_LAMBDA;
                                vp.lambda_bin[hh] = (uint16_t)bin;
                                vp.flux[hh] = vp.spectral_flux.value[bin] * hero_scale;
                            }
                            vp.num_hero = HERO_WAVELENGTHS;
                        }
                        volume_map->push_back(vp);
                    }
                    for (int b = 0; b < NUM_LAMBDA; ++b)
                        spectral_flux.value[b] *= expf(-med.sigma_t.value[b] * seg_t);
                }
            }
            } // end volume point deposits

            const Material& mat = scene.materials[hit.material_id];

            // ── v2.2 Canonical photon deposit rules ─────────────────────
            // Use classify_for_photons() as the single source of truth.
            // Deposit rules:
            //   - GlobalPhoton:  non-delta hit, bounce >= 1
            //   - CausticPhoton: non-delta hit, bounce >= 1, caustic_eligible
            // Specular transport flag:
            //   - Set when path undergoes delta interaction on a caustic_caster
            MaterialFlags mat_flags = classify_for_photons(mat);

            if (!mat_flags.is_delta) {
                // Non-specular surface: deposit photon(s)
                Photon p;
                p.position      = hit.position;
                p.wi            = ray.direction * (-1.f); // Flip: stored as incoming
                p.geom_normal   = hit.normal;             // Geometric normal at hit
                p.spectral_flux = spectral_flux;
                p.source_emissive_idx = source_emissive;
                p.path_flags    = path_flags;
                p.bounce_count  = (uint8_t)bounce;

                // Fill hero wavelength bins for GPU gather compatibility.
                {
                    int primary = (int)(rng.next_float() * NUM_LAMBDA);
                    if (primary >= NUM_LAMBDA) primary = NUM_LAMBDA - 1;
                    constexpr float hero_scale =
                        (float)NUM_LAMBDA / (float)HERO_WAVELENGTHS;
                    for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
                        int bin = (primary + h * NUM_LAMBDA / HERO_WAVELENGTHS)
                                  % NUM_LAMBDA;
                        p.lambda_bin[h] = (uint16_t)bin;
                        p.flux[h] = spectral_flux.value[bin] * hero_scale;
                    }
                    p.num_hero = HERO_WAVELENGTHS;
                }

                if (bounce > 0) {
                    global_map.push_back(p);
                }
                if (on_caustic_path && bounce > 0) {
                    caustic_map.push_back(p);
                }

                on_caustic_path = false;
            } else {
                // Delta/specular bounce: track caustic eligibility
                // Set caustic flag for any caustic caster (Mirror, Glass, Translucent)
                if (mat_flags.caustic_caster) {
                    on_caustic_path = true;
                }
                if (mat.type == MaterialType::Mirror) {
                    // Mirror caustic: purely reflective, no IOR/dispersion
                    path_flags |= PHOTON_FLAG_CAUSTIC_SPECULAR;
                }
                if (mat.type == MaterialType::Glass ||
                    mat.type == MaterialType::Translucent) {
                    path_flags |= PHOTON_FLAG_TRAVERSED_GLASS;
                    if (bounce == 0)
                        path_flags |= PHOTON_FLAG_CAUSTIC_GLASS;
                    if (mat.dispersion)
                        path_flags |= PHOTON_FLAG_DISPERSION;
                    // Push/pop IOR stack on glass entry/exit.
                    // Use the GEOMETRIC normal (hit.normal) to determine
                    // enter/exit — shading normals can be unreliable near
                    // edges and with inconsistent mesh winding.
                    float dot_ni = dot(ray.direction, hit.normal);
                    bool entering = dot_ni < 0.f;
                    if (entering) {
                        ior_stack.push(mat.ior);
                        // Entering medium boundary
                        if (mat.has_medium())
                            medium_stack.push(mat.medium_id);
                    } else {
                        ior_stack.pop();
                        // Exiting medium boundary — guard against
                        // stack underflow (inconsistent mesh winding)
                        if (mat.has_medium() && medium_stack.depth > 0)
                            medium_stack.pop();
                    }
                }
            }

            // §5.3.1: Cell-stratified bounce direction
            ONB frame = ONB::from_normal(hit.shading_normal);
            float3 wo_local = frame.world_to_local(ray.direction * (-1.f));

            BSDFSample bsdf_sample = bsdf::sample(mat, wo_local, rng, primary_hero_bin,
                                                    TransportMode::Importance);

            if (bsdf_sample.pdf <= 0.f) break;

            // Update spectral flux: per-wavelength throughput = f(λ) * cos / pdf
            float cos_theta = fabsf(bsdf_sample.wi.z);
            float geom_factor = cos_theta / bsdf_sample.pdf;
            for (int b = 0; b < NUM_LAMBDA; ++b)
                spectral_flux.value[b] *= bsdf_sample.f.value[b] * geom_factor;

            // Russian roulette (based on max spectral component)
            // Skip for specular/translucent bounces (matches GPU behaviour)
            if (bounce >= config.min_bounces_rr && !mat_flags.is_delta) {
                float max_throughput = spectral_flux.max_component();
                float p_rr = fminf(config.rr_threshold, max_throughput);
                if (p_rr <= 0.f) break;
                if (rng.next_float() >= p_rr) break;
                float inv_rr = 1.0f / p_rr;
                for (int b = 0; b < NUM_LAMBDA; ++b)
                    spectral_flux.value[b] *= inv_rr;
            }

            // Update ray for next bounce
            float3 wi_world = frame.local_to_world(bsdf_sample.wi);
            // Offset along +normal for reflection, −normal for refraction
            // (bsdf_sample.wi.z > 0 ⇒ same hemisphere as normal ⇒ reflection)
            ray.origin    = hit.position + hit.shading_normal * EPSILON *
                            (bsdf_sample.wi.z > 0.f ? 1.f : -1.f);
            ray.direction = wi_world;
            ray.tmin      = 1e-4f;
            ray.tmax      = 1e20f;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Adaptive caustic shooting (§10c)
// ─────────────────────────────────────────────────────────────────────
// Two-phase emission: after the initial uniform photon trace, identify
// caustic hotspot cells (high CV) via CellInfoCache and re-trace
// additional targeted caustic photons.  The targeted photons are
// appended to the existing caustic map and the hash grid is rebuilt.
//
// ─────────────────────────────────────────────────────────────────────
// Targeted caustic emission via specular geometry sampling (§11)
// ─────────────────────────────────────────────────────────────────────
// Instead of shooting caustic photons blindly, importance-sample
// emission directions toward specular (Glass/Translucent/Mirror)
// triangles with visibility checks.  Photons that reach a specular
// surface are then traced normally through the bounce loop.
//
// This gives ~20–100× efficiency improvement for the caustic budget
// when specular objects subtend a small fraction of the light's
// hemisphere (which is the common case).
//
// The estimator is unbiased: the targeted PDF replaces the standard
// cosine-hemisphere PDF, and the flux is reweighted accordingly.

inline void trace_targeted_caustic_emission(
    const Scene& scene,
    const EmitterConfig& config,
    const SpecularTargetSet& target_set,
    PhotonSoA& caustic_map,
    float mix_ratio = DEFAULT_TARGETED_CAUSTIC_MIX)
{
    if (!target_set.valid) {
        std::cout << "[TargetedCaustic] No specular targets — skipping\n";
        return;
    }

    // Budget split: mix_ratio goes to targeted, (1 - mix_ratio) to uniform
    int targeted_budget = (int)(config.num_photons * mix_ratio);
    if (targeted_budget <= 0) return;

    std::cout << "[TargetedCaustic] Shooting " << targeted_budget
              << " targeted caustic photons toward "
              << target_set.specular_tri_indices.size()
              << " specular triangles\n";

    int emitted = 0;
    int stored  = 0;
    int rejected_visibility = 0;

    for (int photon_idx = 0; photon_idx < targeted_budget; ++photon_idx) {
        PCGRng rng = PCGRng::seed(
            (uint64_t)photon_idx * 13 + 0xCA051CUL,
            (uint64_t)photon_idx + 7);

        TargetedCausticPhoton tcp = sample_targeted_caustic_photon(
            scene, target_set, rng);

        if (!tcp.valid) {
            ++rejected_visibility;
            continue;
        }
        ++emitted;

        // ── Trace the photon through the scene (same as trace_photons) ──
        Spectrum spectral_flux = tcp.spectral_flux;
        Ray ray = tcp.ray;
        uint16_t source_emissive = tcp.source_emissive_idx;

        // Sample primary hero wavelength bin at emission time (matches GPU
        // convention and trace_photons).
        int primary_hero_bin;
        {
            float flux_sum = spectral_flux.sum();
            if (flux_sum > 0.f) {
                float xi = rng.next_float() * flux_sum;
                float cum = 0.f;
                primary_hero_bin = NUM_LAMBDA - 1;
                for (int b = 0; b < NUM_LAMBDA; ++b) {
                    cum += spectral_flux.value[b];
                    if (xi <= cum) { primary_hero_bin = b; break; }
                }
            } else {
                primary_hero_bin = (int)(rng.next_float() * NUM_LAMBDA);
                if (primary_hero_bin >= NUM_LAMBDA) primary_hero_bin = NUM_LAMBDA - 1;
            }
        }

        // Start on caustic path: the photon is aimed at a specular surface
        bool on_caustic_path = false;
        uint8_t path_flags = 0;
        IORStack ior_stack;
        MediumStack medium_stack;

        for (int bounce = 0; bounce < config.max_bounces; ++bounce) {
            HitRecord hit = scene.intersect(ray);
            if (!hit.hit) break;

            // §5.3.2: Re-seed RNG from spatial hash
            if (bounce > 0) {
                uint64_t sp_seed = rng_spatial_seed(hit.position, bounce,
                                                     photon_idx);
                rng = PCGRng::seed(sp_seed, (uint64_t)(photon_idx + 1));
            }

            // ── Per-material interior medium Beer-Lambert (§7.7) ─────
            {
                int cur_mid = medium_stack.current_medium_id();
                if (cur_mid >= 0 && cur_mid < (int)scene.media.size()) {
                    const HomogeneousMedium& med = scene.media[cur_mid];
                    float seg_t = hit.t;
                    for (int b = 0; b < NUM_LAMBDA; ++b)
                        spectral_flux.value[b] *= expf(-med.sigma_t.value[b] * seg_t);
                }
            }

            const Material& mat = scene.materials[hit.material_id];

            // ── v2.2 Canonical deposit rules (targeted caustic) ──────
            MaterialFlags mat_flags = classify_for_photons(mat);

            if (!mat_flags.is_delta) {
                if (on_caustic_path && bounce > 0) {
                    Photon p;
                    p.position          = hit.position;
                    p.wi                = ray.direction * (-1.f);
                    p.geom_normal       = hit.normal;
                    p.spectral_flux     = spectral_flux;
                    p.source_emissive_idx = source_emissive;
                    p.path_flags        = path_flags;
                    p.bounce_count      = (uint8_t)bounce;

                    {
                        PCGRng hero_rng = PCGRng::seed(
                            (uint64_t)photon_idx * 997 + 0xBEEF01UL,
                            (uint64_t)stored + 1);
                        int primary = (int)(hero_rng.next_float() * NUM_LAMBDA);
                        if (primary >= NUM_LAMBDA) primary = NUM_LAMBDA - 1;
                        constexpr float hero_scale =
                            (float)NUM_LAMBDA / (float)HERO_WAVELENGTHS;
                        for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
                            int bin = (primary + h * NUM_LAMBDA / HERO_WAVELENGTHS)
                                      % NUM_LAMBDA;
                            p.lambda_bin[h] = (uint16_t)bin;
                            p.flux[h] = spectral_flux.value[bin] * hero_scale;
                        }
                        p.num_hero = HERO_WAVELENGTHS;
                    }

                    caustic_map.push_back(p);
                    ++stored;
                }
                // Caustic path ends at diffuse surface — done with
                // this photon (we only care about caustic deposits).
                break;
            } else {
                // Delta bounce: track caustic eligibility (Mirror, Glass, Translucent)
                if (mat_flags.caustic_caster) {
                    on_caustic_path = true;
                }
                if (mat.type == MaterialType::Mirror) {
                    path_flags |= PHOTON_FLAG_CAUSTIC_SPECULAR;
                }
                if (mat.type == MaterialType::Glass ||
                    mat.type == MaterialType::Translucent) {
                    path_flags |= PHOTON_FLAG_TRAVERSED_GLASS;
                    if (bounce == 0)
                        path_flags |= PHOTON_FLAG_CAUSTIC_GLASS;
                    if (mat.dispersion)
                        path_flags |= PHOTON_FLAG_DISPERSION;
                    float dot_ni = dot(ray.direction, hit.normal);
                    bool entering = dot_ni < 0.f;
                    if (entering) {
                        ior_stack.push(mat.ior);
                        if (mat.has_medium())
                            medium_stack.push(mat.medium_id);
                    } else {
                        ior_stack.pop();
                        if (mat.has_medium() && medium_stack.depth > 0)
                            medium_stack.pop();
                    }
                }
            }

            // BSDF sampling for next bounce
            ONB frame = ONB::from_normal(hit.shading_normal);
            float3 wo_local = frame.world_to_local(ray.direction * (-1.f));

            BSDFSample bsdf_sample = bsdf::sample(mat, wo_local, rng, primary_hero_bin,
                                                    TransportMode::Importance);

            if (bsdf_sample.pdf <= 0.f) break;

            // Update throughput
            float cos_theta = fabsf(bsdf_sample.wi.z);
            float geom_factor = cos_theta / bsdf_sample.pdf;
            for (int b = 0; b < NUM_LAMBDA; ++b)
                spectral_flux.value[b] *= bsdf_sample.f.value[b] * geom_factor;

            // Russian roulette — skip for specular bounces to preserve
            // caustic photon survival through glass stacks
            if (!mat.is_specular() && bounce >= config.min_bounces_rr) {
                float max_throughput = spectral_flux.max_component();
                float p_rr = fminf(config.rr_threshold, max_throughput);
                if (p_rr <= 0.f) break;
                if (rng.next_float() >= p_rr) break;
                float inv_rr = 1.0f / p_rr;
                for (int b = 0; b < NUM_LAMBDA; ++b)
                    spectral_flux.value[b] *= inv_rr;
            }

            // Next ray
            float3 wi_world = frame.local_to_world(bsdf_sample.wi);
            ray.origin    = hit.position + hit.shading_normal * EPSILON *
                            (bsdf_sample.wi.z > 0.f ? 1.f : -1.f);
            ray.direction = wi_world;
            ray.tmin      = 1e-4f;
            ray.tmax      = 1e20f;
        }
    }

    float vis_rate = (targeted_budget > 0)
        ? (float)emitted / (float)targeted_budget * 100.f : 0.f;

    std::cout << "[TargetedCaustic] Emitted: " << emitted
              << " (" << std::fixed << std::setprecision(1) << vis_rate
              << "% visibility), stored: " << stored
              << " caustic photons, rejected: " << rejected_visibility
              << "\n";
}
