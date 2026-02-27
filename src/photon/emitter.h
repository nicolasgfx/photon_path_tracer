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
#include "scene/scene.h"
#include "bsdf/bsdf.h"
#include "photon/photon.h"
#include "core/medium.h"
#include "core/phase_function.h"
#include "volume/photon_beam.h"

#include <vector>
#include <cmath>
#include <atomic>
#include <iomanip>

#include "photon/adaptive_emission.h"   // AdaptiveEmissionContext (used by trace_photons)
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

// ── Beam tracing parameters (subset of RenderConfig) ────────────────
struct BeamTracingConfig {
    bool  enabled          = false;
    int   max_segments     = 5000000;
    float min_segment_len  = 1e-4f;
    float hg_g             = 0.0f;
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
// The EmitterPointSet primary path is disabled by default
// (DEFAULT_USE_EMITTER_POINT_SET = false).  The cosine cone has been
// replaced with full cosine hemisphere for physically correct emission.
// ─────────────────────────────────────────────────────────────────
inline EmittedPhoton sample_emitted_photon(const Scene& scene, PCGRng& rng) {
    EmittedPhoton ep;

    const auto& epts = scene.emitter_points.points;

    if constexpr (DEFAULT_USE_EMITTER_POINT_SET) {
     if (!epts.empty()) {
        // ── Legacy EmitterPointSet path (gated, off by default) ──────
        int pt_idx = (int)(rng.next_float() * (float)epts.size());
        if (pt_idx >= (int)epts.size()) pt_idx = (int)epts.size() - 1;
        const EmitterPoint& ept = epts[pt_idx];

        float3 pos = ept.position;
        float3 n   = ept.normal;

        uint32_t tri_idx = ept.global_tri_idx;
        const Triangle& tri = scene.triangles[tri_idx];
        const Material& mat = scene.materials[tri.material_id];
        float area = tri.area();

        float pdf_point = 1.0f / (float)epts.size();

        // Cosine-weighted hemisphere emission (replaces cone)
        float3 local_dir = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
        ONB frame = ONB::from_normal(n);
        float3 world_dir = frame.local_to_world(local_dir);
        float cos_theta = local_dir.z;
        float pdf_dir = cosine_hemisphere_pdf(cos_theta);

        float denom = pdf_point * pdf_dir;
        if (denom > 0.f) {
            float scale = cos_theta * area / denom;
            for (int b = 0; b < NUM_LAMBDA; ++b)
                ep.spectral_flux.value[b] = mat.Le.value[b] * scale;
        } else {
            ep.spectral_flux = Spectrum::zero();
        }

        ep.ray.origin    = pos + n * EPSILON;
        ep.ray.direction = world_dir;
        ep.source_emissive_idx = ept.emissive_local_idx;
        return ep;
     } // end if (!epts.empty()) — fall through to canonical path
    } // end if constexpr

    {
        // ── Canonical path: alias-table + cosine hemisphere ──────────
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

        // v2.2: Cosine-weighted hemisphere (replaces cosine cone)
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
    }

    return ep;
}

// ── View-adaptive emitter sampling ─────────────────────────────────
// Uses the same deduped emitter point pool as sample_emitted_photon()
// but picks a point via the AEC's view-weighted alias table instead
// of uniformly.  Everything else (position, normal, cone direction,
// flux formula) is identical — only the point-selection PDF changes.

inline EmittedPhoton sample_emitted_photon_adaptive(
    const Scene& scene, const AdaptiveEmissionContext& aec, PCGRng& rng)
{
    EmittedPhoton ep;
    const auto& epts = scene.emitter_points.points;
    if (epts.empty()) return sample_emitted_photon(scene, rng);

    // Pick an emitter point from the view-weighted alias table
    float pdf_point;
    int pt_idx = aec.sample_point(rng.next_float(), rng.next_float(), pdf_point);
    if (pt_idx < 0 || pt_idx >= (int)epts.size()) {
        // Fallback: uniform
        return sample_emitted_photon(scene, rng);
    }

    const EmitterPoint& ept = epts[pt_idx];
    float3 pos = ept.position;
    float3 n   = ept.normal;

    uint32_t tri_idx = ept.global_tri_idx;
    const Triangle& tri = scene.triangles[tri_idx];
    const Material& mat = scene.materials[tri.material_id];
    float area = tri.area();

    // Direction sampling — cosine-weighted hemisphere (v2.2 canonical)
    float3 local_dir = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
    ONB frame = ONB::from_normal(n);
    float3 world_dir = frame.local_to_world(local_dir);
    float cos_theta = local_dir.z;
    float pdf_dir = cosine_hemisphere_pdf(cos_theta);

    // Flux: Le * cos_theta * area / (pdf_point * pdf_dir)
    float denom = pdf_point * pdf_dir;
    if (denom > 0.f) {
        float scale = cos_theta * area / denom;
        for (int b = 0; b < NUM_LAMBDA; ++b)
            ep.spectral_flux.value[b] = mat.Le.value[b] * scale;
    } else {
        ep.spectral_flux = Spectrum::zero();
    }

    ep.ray.origin    = pos + n * EPSILON;
    ep.ray.direction = world_dir;
    ep.source_emissive_idx = ept.emissive_local_idx;

    return ep;
}

// ── Trace photons through the scene ─────────────────────────────────
// Fills global_map (and optionally caustic_map) with photon hits.

// ── §5.3.1 Cell-stratified bounce helper ────────────────────────────
// Uses Fibonacci hemisphere mapping to convert a 1D stratum index
// into a deterministic direction seed on the hemisphere.
// N_strata = DEFAULT_PHOTON_BOUNCE_STRATA (config.h)

struct CellStratifier {
    // Use a spatial hash to assign each photon-hit position to a cell,
    // and use an atomic counter per cell to produce stratum indices.
    static constexpr uint32_t TABLE_SIZE = 4096;
    std::vector<std::atomic<uint32_t>> counters;
    int num_strata;

    CellStratifier(int strata = DEFAULT_PHOTON_BOUNCE_STRATA)
        : counters(TABLE_SIZE), num_strata(strata)
    {
        for (auto& c : counters) c.store(0, std::memory_order_relaxed);
    }

    // Spatial hash of a 3D position → cell index
    static uint32_t spatial_hash(float3 pos, float cell_size) {
        int cx = (int)floorf(pos.x / cell_size);
        int cy = (int)floorf(pos.y / cell_size);
        int cz = (int)floorf(pos.z / cell_size);
        uint32_t h = (uint32_t)(cx * 73856093u)
                   ^ (uint32_t)(cy * 19349663u)
                   ^ (uint32_t)(cz * 83492791u);
        return h % TABLE_SIZE;
    }

    // Get the next stratum index for a cell (atomic increment)
    uint32_t next_stratum(uint32_t cell_idx) {
        uint32_t s = counters[cell_idx].fetch_add(1, std::memory_order_relaxed);
        return s % (uint32_t)num_strata;
    }

    // Fibonacci hemisphere: map stratum s ∈ [0, N) to (theta, phi)
    // Returns a direction in local coordinates (z = up)
    static float3 fibonacci_hemisphere(int s, int N) {
        constexpr float GOLDEN_RATIO = 1.6180339887f;
        float i = (float)s + 0.5f;
        float cos_theta = 1.0f - i / (float)N;  // uniform in cos(theta) on [0,1]
        float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
        float phi = 2.0f * PI * i / GOLDEN_RATIO;
        return make_f3(
            sin_theta * cosf(phi),
            sin_theta * sinf(phi),
            cos_theta
        );
    }
};

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

// ── IOR stack for nested dielectrics ────────────────────────────────
struct IORStack {
    float stack[4] = {1.0f, 0.f, 0.f, 0.f};
    int   depth    = 0;
    float current_ior() const { return depth > 0 ? stack[depth-1] : 1.0f; }
    void  push(float ior) { if (depth < 4) stack[depth++] = ior; }
    void  pop()           { if (depth > 0) --depth; }
};

// ── Medium stack for participating media boundary tracking ──────────
struct MediumStack {
    static constexpr int MAX_DEPTH = 4;
    int stack[MAX_DEPTH] = {-1, -1, -1, -1};  // medium_id per nesting level
    int depth    = 0;
    int current_medium_id() const { return depth > 0 ? stack[depth-1] : -1; }
    void push(int medium_id) {
        if (depth < MAX_DEPTH)
            stack[depth++] = medium_id;
        // else: overflow — clamped at MAX_DEPTH (logged in debug builds)
#ifndef NDEBUG
        else
            printf("[MediumStack] WARNING: overflow at depth %d\n", depth);
#endif
    }
    void pop() {
        if (depth > 0)
            --depth;
        // else: underflow — likely inconsistent mesh winding
#ifndef NDEBUG
        else
            printf("[MediumStack] WARNING: underflow (pop on empty stack)\n");
#endif
    }
};

inline void trace_photons(const Scene& scene,
                           const EmitterConfig& config,
                           PhotonSoA& global_map,
                           PhotonSoA& caustic_map,
                           PhotonSoA* volume_map = nullptr,
                           PhotonBeamMap* beam_map = nullptr,
                           const BeamTracingConfig* beam_cfg = nullptr,
                           const AdaptiveEmissionContext* aec = nullptr) {
    global_map.clear();
    caustic_map.clear();
    global_map.reserve(config.num_photons);
    caustic_map.reserve(config.num_photons / 4);
    if (volume_map) {
        volume_map->clear();
        volume_map->reserve(config.num_photons / 4);
    }

    // Beam map parameters
    const bool beam_enabled = beam_map && beam_cfg && beam_cfg->enabled;
    const int  beam_max_segs = beam_cfg ? beam_cfg->max_segments : 5000000;
    const float beam_min_len = beam_cfg ? beam_cfg->min_segment_len : 1e-4f;
    const float beam_hg_g    = beam_cfg ? beam_cfg->hg_g : 0.0f;

    if (scene.emissive_tri_indices.empty()) {
        return;
    }

    // §5.3.1: Cell stratifier for bounce decorrelation
    CellStratifier stratifier(DEFAULT_PHOTON_BOUNCE_STRATA);
    const float strat_cell_size = DEFAULT_GATHER_RADIUS * 2.0f;

    for (int photon_idx = 0; photon_idx < config.num_photons; ++photon_idx) {
        PCGRng rng = PCGRng::seed(photon_idx * 7 + 42, photon_idx + 1);

        EmittedPhoton ep = (aec && aec->valid)
            ? sample_emitted_photon_adaptive(scene, *aec, rng)
            : sample_emitted_photon(scene, rng);
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

            // ── Volume photon deposit (Beer–Lambert free-flight) ─────
            // Legacy atmospheric fog volume photon system.
            // Volume point deposits — runtime gated via config.volume_enabled.
            // IMPORTANT (H3 fix): skip this when the photon is inside an
            // object-attached medium that will be handled by the beam
            // system below, to prevent double-attenuation.
            {
            int cur_med_id_legacy = medium_stack.current_medium_id();
            bool beam_handles_medium = beam_enabled && cur_med_id_legacy >= 0 &&
                cur_med_id_legacy < (int)scene.media.size();

            if (volume_map && config.volume_enabled && config.volume_density > 0.f
                && !beam_handles_medium) {
                path_flags |= PHOTON_FLAG_VOLUME_SEGMENT;
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

            // ── Photon-beam medium free-flight + segment recording ─────
            // When the photon is inside a participating medium (tracked
            // by medium_stack), record a beam segment and optionally
            // scatter within the medium before hitting the surface.
            //
            // Sampling strategy (hero-wavelength scheme):
            //   - Pick a "hero" wavelength bin uniformly at random.
            //   - Sample free-flight distance t from the hero bin's
            //     exponential: pdf(t) = σ_t[hero] · exp(-σ_t[hero]·t).
            //   - For the scatter event the MC weight per wavelength is:
            //       w(λ) = σ_s(λ) · exp(-σ_t(λ)·t)           [numerator]
            //            / ( (1/N) · Σ_j σ_t[j]·exp(-σ_t[j]·t) )  [avg pdf]
            //     This is the "spectral MIS" one-sample estimator
            //     (Eq 4 in Hero-wavelength spectral sampling, 2014).
            //   - For the no-scatter (surface hit) event the weight is:
            //       w(λ) = exp(-σ_t(λ)·seg_t)                [Tr to surface]
            //            / ( (1/N) · Σ_j exp(-σ_t[j]·seg_t) )  [avg survival]
            //     which correctly accounts for the probability of the
            //     free-flight distance exceeding seg_t.
            int cur_med_id = medium_stack.current_medium_id();
            if (beam_enabled && cur_med_id >= 0 &&
                cur_med_id < (int)scene.media.size()) {

                const HomogeneousMedium& med = scene.media[cur_med_id];
                float seg_t = hit.t;  // distance to next surface

                // Pick hero wavelength uniformly
                int hero = (int)(rng.next_float() * NUM_LAMBDA);
                if (hero >= NUM_LAMBDA) hero = NUM_LAMBDA - 1;
                float sig_t_hero = fmaxf(med.sigma_t.value[hero], 1e-20f);

                if (sig_t_hero > 0.f) {
                    // Sample free-flight distance from hero exponential
                    float u_ff = rng.next_float();
                    float t_ff = -logf(fmaxf(u_ff, 1e-12f)) / sig_t_hero;

                    if (t_ff < seg_t) {
                        // ── Medium scattering event (before surface) ──
                        float3 p_scatter = ray.origin + ray.direction * t_ff;

                        // Record beam segment: ray.origin → p_scatter
                        // Store beta = spectral_flux ATTENUATED to the
                        // scatter point (Convention 1: beta at segment
                        // start, without attenuation — see C2 note below).
                        // We store the flux at segment start; the estimator
                        // will apply extinction from start to closest-approach.
                        if (t_ff >= beam_min_len &&
                            (int)beam_map->size() < beam_max_segs) {
                            PhotonBeamSegment bseg;
                            bseg.p0 = ray.origin;
                            bseg.p1 = p_scatter;
                            bseg.beta = spectral_flux;  // power at segment start
                            bseg.medium_id = (uint32_t)cur_med_id;
                            bseg.t0 = 0.f;
                            bseg.t1 = t_ff;
                            beam_map->push_back(bseg);
                        }

                        // Spectral MIS weight for scatter event:
                        //   numerator(λ) = σ_s(λ) · exp(-σ_t(λ)·t_ff)
                        //   pdf_avg      = (1/N) Σ_j σ_t[j]·exp(-σ_t[j]·t_ff)
                        //   weight(λ)    = numerator(λ) / pdf_avg
                        float pdf_sum = 0.f;
                        for (int b = 0; b < NUM_LAMBDA; ++b) {
                            float st = fmaxf(med.sigma_t.value[b], 1e-20f);
                            pdf_sum += st * expf(-st * t_ff);
                        }
                        float inv_pdf_avg = (float)NUM_LAMBDA / fmaxf(pdf_sum, 1e-30f);

                        for (int b = 0; b < NUM_LAMBDA; ++b) {
                            float sig_s_b = med.sigma_s.value[b];
                            float sig_t_b = fmaxf(med.sigma_t.value[b], 1e-20f);
                            float Tr_b = expf(-sig_t_b * t_ff);
                            // weight = σ_s(λ) · Tr(λ) / pdf_avg
                            spectral_flux.value[b] *= sig_s_b * Tr_b * inv_pdf_avg;
                        }

                        // Sample new direction from HG phase function
                        // Use per-medium g; fall back to global beam_hg_g
                        float med_g = med.g;
                        if (fabsf(med_g) < 1e-6f) med_g = beam_hg_g;
                        float3 hg_local = sample_henyey_greenstein(
                            med_g, rng.next_float(), rng.next_float());
                        // Transform from local (z=forward) to world:
                        // forward direction = current ray.direction
                        ONB phase_frame = ONB::from_normal(ray.direction);
                        float3 new_dir = phase_frame.local_to_world(hg_local);

                        // Phase PDF cancels when sampling == eval (importance
                        // sampling the phase function), so no division needed.

                        // Russian roulette in medium
                        if (bounce >= config.min_bounces_rr) {
                            float max_tp = spectral_flux.max_component();
                            float p_rr = fminf(config.rr_threshold, max_tp);
                            if (p_rr <= 0.f) break;
                            if (rng.next_float() >= p_rr) break;
                            float inv_rr = 1.0f / p_rr;
                            for (int b = 0; b < NUM_LAMBDA; ++b)
                                spectral_flux.value[b] *= inv_rr;
                        }

                        // Continue from scatter point with new direction
                        ray.origin    = p_scatter;
                        ray.direction = new_dir;
                        ray.tmin      = 1e-4f;
                        ray.tmax      = 1e20f;
                        continue; // next bounce iteration
                    } else {
                        // ── No scatter: photon reaches the surface ────
                        // Record beam segment for the full travel
                        if (seg_t >= beam_min_len &&
                            (int)beam_map->size() < beam_max_segs) {
                            PhotonBeamSegment bseg;
                            bseg.p0 = ray.origin;
                            bseg.p1 = hit.position;
                            bseg.beta = spectral_flux;  // power at segment start
                            bseg.medium_id = (uint32_t)cur_med_id;
                            bseg.t0 = 0.f;
                            bseg.t1 = seg_t;
                            beam_map->push_back(bseg);
                        }

                        // Spectral MIS weight for surface-hit event:
                        //   P(t > seg_t | hero) = exp(-σ_t[hero]·seg_t)
                        //   numerator(λ) = exp(-σ_t(λ)·seg_t)   [Tr to surface]
                        //   P_avg(t>seg_t) = (1/N) Σ_j exp(-σ_t[j]·seg_t)
                        //   weight(λ) = Tr(λ) / P_avg
                        float surv_sum = 0.f;
                        for (int b = 0; b < NUM_LAMBDA; ++b)
                            surv_sum += expf(-fmaxf(med.sigma_t.value[b], 1e-20f) * seg_t);
                        float inv_surv_avg = (float)NUM_LAMBDA / fmaxf(surv_sum, 1e-30f);

                        for (int b = 0; b < NUM_LAMBDA; ++b) {
                            float Tr_b = expf(-fmaxf(med.sigma_t.value[b], 1e-20f) * seg_t);
                            spectral_flux.value[b] *= Tr_b * inv_surv_avg;
                        }
                    }
                }
            }

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
            // Use the Fibonacci stratum as a deterministic seed that
            // modulates the BSDF sample, ensuring spatial variety.
            ONB frame = ONB::from_normal(hit.shading_normal);
            float3 wo_local = frame.world_to_local(ray.direction * (-1.f));

            BSDFSample bsdf_sample;
            if (DEFAULT_PHOTON_BOUNCE_STRATA > 0 && !mat.is_specular()
                && (mat.type == MaterialType::Lambertian ||
                    mat.type == MaterialType::Emissive)) {
                // §5.3.1: Stratified bounce — Fibonacci lattice in [0,1)²
                // maps stratum → cosine-hemisphere direction with correct
                // importance sampling (pdf = cos/π).
                uint32_t cell = CellStratifier::spatial_hash(
                    hit.position, strat_cell_size);
                uint32_t s = stratifier.next_stratum(cell);
                int N = DEFAULT_PHOTON_BOUNCE_STRATA;

                // Fibonacci lattice → stratified (u1, u2)
                constexpr float INV_GOLDEN = 0.6180339887f;
                float base_u1 = ((float)s + 0.5f) / (float)N;
                float base_u2 = fmodf((float)s * INV_GOLDEN, 1.0f);
                float u1 = fmodf(base_u1 + rng.next_float() / (float)N, 1.0f);
                float u2 = fmodf(base_u2 + rng.next_float() / (float)N, 1.0f);

                bsdf_sample.wi  = sample_cosine_hemisphere(u1, u2);
                bsdf_sample.pdf = cosine_hemisphere_pdf(bsdf_sample.wi.z);
                bsdf_sample.f   = mat.Kd * INV_PI;
                bsdf_sample.is_specular = false;

                if (bsdf_sample.pdf <= 0.f) {
                    bsdf_sample = bsdf::sample(mat, wo_local, rng, primary_hero_bin);
                }
            } else {
                bsdf_sample = bsdf::sample(mat, wo_local, rng, primary_hero_bin);
            }

            if (bsdf_sample.pdf <= 0.f) break;

            // Update spectral flux: per-wavelength throughput = f(λ) * cos / pdf
            float cos_theta = fabsf(bsdf_sample.wi.z);
            float geom_factor = cos_theta / bsdf_sample.pdf;
            for (int b = 0; b < NUM_LAMBDA; ++b)
                spectral_flux.value[b] *= bsdf_sample.f.value[b] * geom_factor;

            // Russian roulette (based on max spectral component)
            if (bounce >= config.min_bounces_rr) {
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
// Uses standard trace_photons internally — the "targeting" is achieved
// by simply tracing extra photons uniformly (more of them) and only
// keeping the caustic deposits.  A more sophisticated scheme would
// bias emission toward glass objects, but this approach is simple and
// correct: more samples → lower variance.

#include "core/cell_cache.h"

inline void trace_targeted_caustic_photons(
    const Scene& scene,
    const EmitterConfig& config,
    const CellInfoCache& cell_cache,
    PhotonSoA& caustic_map)
{
    // Determine how many extra photons to emit
    auto hotspot_keys = cell_cache.get_caustic_hotspot_keys();
    if (hotspot_keys.empty()) {
        std::cout << "[AdaptiveCaustic] No hotspot cells — skipping\n";
        return;
    }

    int targeted_budget = (int)(config.num_photons * CAUSTIC_TARGETED_FRACTION);
    if (targeted_budget <= 0) return;

    std::cout << "[AdaptiveCaustic] " << hotspot_keys.size()
              << " hotspot cells, tracing " << targeted_budget
              << " targeted photons\n";

    for (int iter = 0; iter < CAUSTIC_MAX_TARGETED_ITERS; ++iter) {
        // Trace extra photons — only keep caustic deposits
        PhotonSoA extra_global;
        PhotonSoA extra_caustic;
        extra_caustic.reserve(targeted_budget / 4);

        EmitterConfig extra_cfg = config;
        extra_cfg.num_photons    = targeted_budget;
        extra_cfg.volume_enabled = false;

        trace_photons(scene, extra_cfg, extra_global, extra_caustic, nullptr);

        if (extra_caustic.size() == 0) break;

        // Append targeted caustic photons to the main map
        size_t n = extra_caustic.size();
        for (size_t i = 0; i < n; ++i) {
            Photon p = extra_caustic.get(i);
            caustic_map.push_back(p);
        }

        std::cout << "[AdaptiveCaustic] Iter " << (iter + 1)
                  << ": +" << n << " caustic photons (total "
                  << caustic_map.size() << ")\n";

        // Reduce budget for subsequent iterations
        targeted_budget /= 2;
        if (targeted_budget < 1000) break;
    }
}

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

    // §5.3.1: Cell stratifier for bounce decorrelation
    CellStratifier stratifier(DEFAULT_PHOTON_BOUNCE_STRATA);
    const float strat_cell_size = DEFAULT_GATHER_RADIUS * 2.0f;

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

            BSDFSample bsdf_sample;
            if (DEFAULT_PHOTON_BOUNCE_STRATA > 0 && !mat.is_specular()
                && (mat.type == MaterialType::Lambertian ||
                    mat.type == MaterialType::Emissive)) {
                // §5.3.1: Stratified bounce (see trace_photons)
                uint32_t cell = CellStratifier::spatial_hash(
                    hit.position, strat_cell_size);
                uint32_t s = stratifier.next_stratum(cell);
                int N = DEFAULT_PHOTON_BOUNCE_STRATA;

                constexpr float INV_GOLDEN = 0.6180339887f;
                float base_u1 = ((float)s + 0.5f) / (float)N;
                float base_u2 = fmodf((float)s * INV_GOLDEN, 1.0f);
                float u1 = fmodf(base_u1 + rng.next_float() / (float)N, 1.0f);
                float u2 = fmodf(base_u2 + rng.next_float() / (float)N, 1.0f);

                bsdf_sample.wi  = sample_cosine_hemisphere(u1, u2);
                bsdf_sample.pdf = cosine_hemisphere_pdf(bsdf_sample.wi.z);
                bsdf_sample.f   = mat.Kd * INV_PI;
                bsdf_sample.is_specular = false;

                if (bsdf_sample.pdf <= 0.f) {
                    bsdf_sample = bsdf::sample(mat, wo_local, rng, primary_hero_bin);
                }
            } else {
                bsdf_sample = bsdf::sample(mat, wo_local, rng, primary_hero_bin);
            }

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
