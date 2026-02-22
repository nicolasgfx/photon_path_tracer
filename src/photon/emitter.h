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
#include "scene/scene.h"
#include "bsdf/bsdf.h"
#include "photon/photon.h"
#include "core/medium.h"

#include <vector>
#include <cmath>
#include <atomic>

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
inline EmittedPhoton sample_emitted_photon(const Scene& scene, PCGRng& rng) {
    EmittedPhoton ep;

    // 1. Sample emissive triangle via alias table
    float u1 = rng.next_float();
    float u2 = rng.next_float();
    int local_idx = scene.emissive_alias_table.sample(u1, u2);
    uint32_t tri_idx = scene.emissive_tri_indices[local_idx];
    const Triangle& tri = scene.triangles[tri_idx];
    const Material& mat = scene.materials[tri.material_id];

    float pdf_tri = scene.emissive_alias_table.pdf(local_idx);

    // 2. Sample position on triangle (uniform barycentric)
    float3 bary = sample_triangle(rng.next_float(), rng.next_float());
    float3 pos  = tri.interpolate_position(bary.x, bary.y, bary.z);
    float3 n    = tri.geometric_normal();
    float  area = tri.area();
    float  pdf_pos = 1.0f / area;

    // 3. Full spectral flux: compute flux for ALL wavelength bins.
    //    No wavelength sampling needed — each photon carries all bins.

    // 4. Sample cosine-weighted direction within emission cone
    const float cone_half_rad = DEFAULT_LIGHT_CONE_HALF_ANGLE_DEG * (PI / 180.0f);
    const float cos_cone_max  = cosf(cone_half_rad);
    float3 local_dir = sample_cosine_cone(rng.next_float(), rng.next_float(), cos_cone_max);
    ONB frame = ONB::from_normal(n);
    float3 world_dir = frame.local_to_world(local_dir);
    float cos_theta = local_dir.z;
    float pdf_dir = cosine_cone_pdf(cos_theta, cos_cone_max);

    // 5. Compute spectral photon flux for each wavelength bin:
    //    Phi(lambda) = Le(lambda) * cos_theta / (p_tri * p_pos * p_dir)
    //    (No p_lambda division — we carry all wavelengths simultaneously.)
    float denom = pdf_tri * pdf_pos * pdf_dir;
    if (denom > 0.f) {
        float scale = cos_theta / denom;
        for (int b = 0; b < NUM_LAMBDA; ++b)
            ep.spectral_flux.value[b] = mat.Le.value[b] * scale;
    } else {
        ep.spectral_flux = Spectrum::zero();
    }

    ep.ray.origin    = pos + n * EPSILON; // Offset to avoid self-intersection
    ep.ray.direction = world_dir;
    ep.source_emissive_idx = (uint16_t)local_idx;

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

    // §5.3.1: Cell stratifier for bounce decorrelation
    CellStratifier stratifier(DEFAULT_PHOTON_BOUNCE_STRATA);
    const float strat_cell_size = DEFAULT_GATHER_RADIUS * 2.0f;

    for (int photon_idx = 0; photon_idx < config.num_photons; ++photon_idx) {
        PCGRng rng = PCGRng::seed(photon_idx * 7 + 42, photon_idx + 1);

        EmittedPhoton ep = sample_emitted_photon(scene, rng);
        Spectrum spectral_flux = ep.spectral_flux;  // Full spectral throughput
        Ray ray = ep.ray;
        uint16_t source_emissive = ep.source_emissive_idx;

        bool on_caustic_path = false; // Set true on first specular bounce, false after diffuse
        uint8_t path_flags = 0;       // Accumulated path flags
        IORStack ior_stack;            // Nested dielectric IOR tracking

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
            if (volume_map && config.volume_enabled && config.volume_density > 0.f) {
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
                        volume_map->push_back(vp);
                    }
                    for (int b = 0; b < NUM_LAMBDA; ++b)
                        spectral_flux.value[b] *= expf(-med.sigma_t.value[b] * seg_t);
                }
            }

            const Material& mat = scene.materials[hit.material_id];

            // Store photon at diffuse surfaces
            if (!mat.is_specular()) {
                Photon p;
                p.position      = hit.position;
                p.wi            = ray.direction * (-1.f); // Flip: stored as incoming
                p.geom_normal   = hit.normal;             // Geometric normal at hit
                p.spectral_flux = spectral_flux;
                p.source_emissive_idx = source_emissive;
                p.path_flags    = path_flags;
                p.bounce_count  = (uint8_t)bounce;

                if (on_caustic_path && bounce > 0) {
                    caustic_map.push_back(p);
                }
                if (bounce > 0) {
                    global_map.push_back(p);
                }

                on_caustic_path = false;
            } else {
                // Specular bounce: track glass traversal and caustic flags
                on_caustic_path = true;
                if (mat.type == MaterialType::Glass) {
                    path_flags |= PHOTON_FLAG_TRAVERSED_GLASS;
                    if (bounce == 0)
                        path_flags |= PHOTON_FLAG_CAUSTIC_GLASS;
                    if (mat.dispersion)
                        path_flags |= PHOTON_FLAG_DISPERSION;
                    // Push/pop IOR stack on glass entry/exit
                    float dot_ni = dot(ray.direction, hit.normal);
                    if (dot_ni < 0.f)
                        ior_stack.push(mat.ior);
                    else
                        ior_stack.pop();
                }
            }

            // §5.3.1: Cell-stratified bounce direction
            // Use the Fibonacci stratum as a deterministic seed that
            // modulates the BSDF sample, ensuring spatial variety.
            ONB frame = ONB::from_normal(hit.shading_normal);
            float3 wo_local = frame.world_to_local(ray.direction * (-1.f));

            BSDFSample bsdf_sample;
            if (DEFAULT_PHOTON_BOUNCE_STRATA > 0 && !mat.is_specular()) {
                // Get stratum for this cell + bounce
                uint32_t cell = CellStratifier::spatial_hash(
                    hit.position, strat_cell_size);
                uint32_t s = stratifier.next_stratum(cell);
                int N = DEFAULT_PHOTON_BOUNCE_STRATA;

                // Fibonacci hemisphere direction as jittered seed
                float3 fib_dir = CellStratifier::fibonacci_hemisphere(
                    (int)s, N);

                // Jitter within the stratum using the RNG
                float jitter_u = rng.next_float() / (float)N;
                float jitter_v = rng.next_float() / (float)N;
                fib_dir.x += jitter_u * 0.1f;
                fib_dir.y += jitter_v * 0.1f;
                float len = sqrtf(fib_dir.x*fib_dir.x + fib_dir.y*fib_dir.y
                                + fib_dir.z*fib_dir.z);
                if (len > 0.f) {
                    fib_dir.x /= len;
                    fib_dir.y /= len;
                    fib_dir.z /= len;
                }

                // Use the stratified direction for diffuse surfaces
                bsdf_sample.wi  = fib_dir;
                bsdf_sample.pdf = cosine_hemisphere_pdf(
                    fmaxf(0.0f, fib_dir.z));
                bsdf_sample.f   = bsdf::evaluate(mat, wo_local, fib_dir);

                if (bsdf_sample.pdf <= 0.f) {
                    // Fall back to standard BSDF sampling
                    bsdf_sample = bsdf::sample(mat, wo_local, rng);
                }
            } else {
                bsdf_sample = bsdf::sample(mat, wo_local, rng);
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
        size_t before = caustic_map.size();
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
