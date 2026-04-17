#pragma once
// ─────────────────────────────────────────────────────────────────────
// scene_analyzer.h – Analyze a loaded Scene to produce a SceneProfile
//
// Called once after scene loading.  Classifies lighting type, material
// mix, geometric complexity, delta objects, emitter distribution, and
// computes convergence hints consumed by all downstream stages.
// ─────────────────────────────────────────────────────────────────────
#include "core/scene_profile.h"
#include "scene/scene.h"
#include <cmath>
#include <algorithm>

inline SceneProfile analyze_scene(const Scene& scene) {
    SceneProfile sp;

    // ── Geometry ────────────────────────────────────────────────────
    sp.num_triangles = (int)scene.triangles.size();
    sp.num_instances = (int)scene.instances.size();
    sp.scene_diagonal = scene.scene_bounding_radius() * 2.0f;

    if (sp.num_triangles < 10000)
        sp.geometry_complexity = GeometryComplexity::Simple;
    else if (sp.num_triangles < 500000)
        sp.geometry_complexity = GeometryComplexity::Moderate;
    else if (sp.num_triangles < 5000000)
        sp.geometry_complexity = GeometryComplexity::Complex;
    else
        sp.geometry_complexity = GeometryComplexity::Dense;

    // ── Open geometry detection ─────────────────────────────────────
    // Heuristic: check if any non-emissive triangles are single-sided
    // (scene has triangles that don't form closed manifolds).
    // We sample a subset for performance.
    {
        int sample_count = (std::min)((int)scene.triangles.size(), 1000);
        int step = (std::max)(1, (int)scene.triangles.size() / sample_count);
        int degenerate = 0;
        for (int i = 0; i < (int)scene.triangles.size(); i += step) {
            const auto& t = scene.triangles[i];
            float area = t.area();
            if (area < 1e-10f) degenerate++;
        }
        sp.has_open_geometry = (degenerate > sample_count / 20);
    }

    // ── Emitters ────────────────────────────────────────────────────
    sp.num_emitters = (int)scene.emissive_tri_indices.size();

    // Emitter centroids + flux, used by visibility probe and coupling.
    struct EmInfo { float3 centroid; float3 normal; float flux; };
    std::vector<EmInfo> emitters;

    if (sp.num_emitters > 0) {
        float total_emitter_area = 0.f;
        float min_area = 1e30f, max_area = 0.f;
        float total_flux = 0.f;
        float max_radiance = 0.f;
        for (uint32_t idx : scene.emissive_tri_indices) {
            const auto& t = scene.triangles[idx];
            float a = t.area();
            total_emitter_area += a;
            min_area = fminf(min_area, a);
            max_area = fmaxf(max_area, a);
            const auto& mat = scene.materials[t.material_id];
            float lum = mat.Le.sum() / 3.f;  // mean luminance
            total_flux += lum * a;
            max_radiance = fmaxf(max_radiance, lum);
        }
        sp.total_emissive_flux  = total_flux;
        sp.max_emitter_radiance = max_radiance;

        AABB bb = AABB::empty();
        for (const auto& t : scene.triangles) {
            bb.expand(t.v0); bb.expand(t.v1); bb.expand(t.v2);
        }
        float3 ext = bb.extent();
        float bbox_area = 2.f * (ext.x * ext.y + ext.y * ext.z + ext.x * ext.z);
        sp.emitter_size_ratio = (bbox_area > 0.f) ? total_emitter_area / bbox_area : 0.f;

        // Emitter distribution classification
        if (sp.num_emitters == 1) {
            sp.emitter_distribution = EmitterDistribution::SingleDominant;
        } else if (max_area > 0.f && min_area > 0.f) {
            float ratio = max_area / min_area;
            sp.emitter_distribution = (ratio > 10.f)
                ? EmitterDistribution::HighVariance
                : EmitterDistribution::Uniform;
        }

        // ── Group emitter triangles by material → per-light centroid ─
        // Shared by visibility probe and emitter-to-delta coupling.
        struct EmGroup { int mat_id; float3 pos_sum; float3 nrm_sum; float flux; int count; };
        std::vector<EmGroup> em_groups;
        for (uint32_t idx : scene.emissive_tri_indices) {
            const auto& t = scene.triangles[idx];
            int mid = t.material_id;
            float3 c = (t.v0 + t.v1 + t.v2) * (1.f / 3.f);
            float3 n = t.geometric_normal();
            float lum = scene.materials[mid].Le.sum() / 3.f;
            float a = t.area();
            EmGroup* found = nullptr;
            for (auto& g : em_groups)
                if (g.mat_id == mid) { found = &g; break; }
            if (found) {
                found->pos_sum = found->pos_sum + c;
                found->nrm_sum = found->nrm_sum + n;
                found->flux += lum * a;
                found->count++;
            } else {
                em_groups.push_back({mid, c, n, lum * a, 1});
            }
        }
        for (auto& g : em_groups) {
            float inv = 1.f / (float)g.count;
            float3 avg_n = g.nrm_sum * inv;
            float len = length(avg_n);
            if (len > 1e-8f) avg_n = avg_n * (1.f / len);
            emitters.push_back({g.pos_sum * inv, avg_n, g.flux});
        }

        // ── Emitter direct visibility (occlusion probe) ────────────
        // Cast shadow rays from sampled non-emissive surfaces toward
        // emitter centroids.  Low visibility → lights are mostly
        // indirect (e.g. lamps behind covers, as in veach-bidir).
        {
            // Collect non-emissive triangle indices for sampling
            std::vector<uint32_t> surface_tris;
            surface_tris.reserve(scene.triangles.size());
            for (uint32_t i = 0; i < (uint32_t)scene.triangles.size(); ++i) {
                const auto& mat = scene.materials[scene.triangles[i].material_id];
                if (!mat.is_emissive())
                    surface_tris.push_back(i);
            }

            // Simple LCG for deterministic analysis
            uint32_t rng_state = 0x12345678u;
            auto lcg = [&]() -> uint32_t {
                rng_state = rng_state * 1664525u + 1013904223u;
                return rng_state;
            };
            auto rng_float = [&]() -> float {
                return (float)(lcg() & 0x00FFFFFFu) / (float)0x01000000u;
            };

            // For large scenes, subsample the occluder set to cap
            // brute-force cost (~50K tris max).
            constexpr uint32_t MAX_OCCLUDER_TRIS = 50000u;
            std::vector<uint32_t> occluder_indices;
            if (scene.triangles.size() <= MAX_OCCLUDER_TRIS) {
                occluder_indices.resize(scene.triangles.size());
                for (uint32_t i = 0; i < (uint32_t)scene.triangles.size(); ++i)
                    occluder_indices[i] = i;
            } else {
                // Reservoir-sample MAX_OCCLUDER_TRIS indices
                occluder_indices.resize(MAX_OCCLUDER_TRIS);
                for (uint32_t i = 0; i < MAX_OCCLUDER_TRIS; ++i)
                    occluder_indices[i] = i;
                for (uint32_t i = MAX_OCCLUDER_TRIS; i < (uint32_t)scene.triangles.size(); ++i) {
                    uint32_t j = lcg() % (i + 1);
                    if (j < MAX_OCCLUDER_TRIS)
                        occluder_indices[j] = i;
                }
            }

            const int NUM_PROBES = 512;
            int visible = 0;
            int total_probes = 0;

            if (!surface_tris.empty() && !emitters.empty()) {
                for (int probe = 0; probe < NUM_PROBES; ++probe) {
                    // Pick a random non-emissive triangle
                    uint32_t surf_idx = surface_tris[lcg() % surface_tris.size()];
                    const auto& st = scene.triangles[surf_idx];

                    // Random point on triangle (uniform barycentric)
                    float u = rng_float();
                    float v = rng_float();
                    if (u + v > 1.f) { u = 1.f - u; v = 1.f - v; }
                    float3 origin = st.v0 * (1.f - u - v) + st.v1 * u + st.v2 * v;
                    float3 surf_n = st.geometric_normal();

                    // Test against each emitter centroid
                    for (const auto& em : emitters) {
                        float3 to_light = em.centroid - origin;
                        float dist = length(to_light);
                        if (dist < 1e-6f) continue;
                        float3 dir = to_light * (1.f / dist);

                        // Skip if surface faces away from emitter
                        if (dot(dir, surf_n) < 0.01f) continue;
                        // Skip if emitter faces away from surface
                        if (dot(dir * (-1.f), em.normal) < 0.01f) continue;

                        ++total_probes;

                        // Trace shadow ray (CPU brute-force against subsampled tris)
                        bool occluded = false;
                        Ray shadow_ray;
                        shadow_ray.origin = origin + surf_n * 1e-4f;
                        shadow_ray.direction = dir;
                        shadow_ray.tmin = 0.f;
                        shadow_ray.tmax = dist - 1e-4f;

                        for (uint32_t oi : occluder_indices) {
                            const auto& tri = scene.triangles[oi];
                            float t_hit, u_hit, v_hit;
                            if (tri.intersect(shadow_ray, t_hit, u_hit, v_hit)) {
                                occluded = true;
                                break;
                            }
                        }
                        if (!occluded) ++visible;
                    }
                }
            }

            if (total_probes > 0) {
                sp.emitter_direct_visibility = (float)visible / (float)total_probes;
                sp.mostly_indirect_emitters = (sp.emitter_direct_visibility < 0.50f);
            } else {
                sp.emitter_direct_visibility = 0.f;
                sp.mostly_indirect_emitters = false;
            }
        }
    }

    // ── Materials ───────────────────────────────────────────────────
    uint8_t type_seen[16] = {};
    float roughness_sum = 0.f;
    int roughness_count = 0;
    int delta_count = 0;

    for (const auto& mat : scene.materials) {
        if ((int)mat.type < 16) type_seen[(int)mat.type] = 1;

        if (mat.type == MaterialType::Glass || mat.type == MaterialType::Translucent)
            sp.has_glass = true;
        if (mat.type == MaterialType::GlossyMetal || mat.type == MaterialType::Mirror)
            sp.has_metal = true;
        if (mat.type == MaterialType::Translucent)
            sp.has_translucent = true;
        if (mat.type == MaterialType::Clearcoat)
            sp.has_clearcoat = true;
        if (mat.type == MaterialType::Mirror || mat.type == MaterialType::Glass)
            delta_count++;

        if (mat.type != MaterialType::Emissive) {
            roughness_sum += mat.roughness;
            roughness_count++;
        }
    }

    int distinct_types = 0;
    for (int i = 0; i < 16; ++i) distinct_types += type_seen[i];
    sp.num_material_types = distinct_types;
    sp.num_delta_materials = delta_count;
    sp.avg_roughness = (roughness_count > 0) ? roughness_sum / roughness_count : 0.5f;

    // ── Caustic detection ───────────────────────────────────────────
    sp.has_caustic_paths = sp.has_glass && sp.num_emitters > 0;

    // ── Delta surface geometry (for caustic budget sizing) ──────────
    {
        int delta_tris = 0;
        float delta_area = 0.f;
        float total_area = 0.f;
        for (const auto& t : scene.triangles) {
            float a = t.area();
            total_area += a;
            auto mt = scene.materials[t.material_id].type;
            if (mt == MaterialType::Mirror || mt == MaterialType::Glass ||
                mt == MaterialType::Translucent) {
                delta_tris++;
                delta_area += a;
            }
        }
        sp.num_delta_triangles = delta_tris;
        sp.delta_area_fraction = (total_area > 0.f) ? delta_area / total_area : 0.f;

        // Heuristic: delta objects are "favorable" for caustics when
        // they have non-trivial surface area and lights exist.
        sp.caustic_geometry_favorable =
            sp.has_caustic_paths && sp.delta_area_fraction > 0.001f;
    }

    // ── Emitter-to-delta coupling ───────────────────────────────────
    // Approximate solid angle of delta surfaces as seen from emitters.
    // Drives caustic budget: low coupling → need more photons or
    // direction biasing (which the kernel handles with 50/50 split).
    if (sp.has_caustic_paths && !emitters.empty() && sp.num_delta_triangles > 0) {
        // Subsample delta tris for coupling estimate (cap CPU cost)
        constexpr int MAX_DELTA_SAMPLE = 1000;
        std::vector<uint32_t> delta_sample;
        delta_sample.reserve((std::min)(sp.num_delta_triangles, MAX_DELTA_SAMPLE));

        uint32_t rng_c = 0xCAFEBABEu;
        auto lcg_c = [&]() -> uint32_t {
            rng_c = rng_c * 1664525u + 1013904223u;
            return rng_c;
        };

        for (uint32_t i = 0; i < (uint32_t)scene.triangles.size(); ++i) {
            auto mt = scene.materials[scene.triangles[i].material_id].type;
            if (mt == MaterialType::Mirror || mt == MaterialType::Glass ||
                mt == MaterialType::Translucent) {
                if ((int)delta_sample.size() < MAX_DELTA_SAMPLE) {
                    delta_sample.push_back(i);
                } else {
                    // Reservoir sampling
                    uint32_t j = lcg_c() % ((uint32_t)delta_sample.size() + 1u);
                    if (j < MAX_DELTA_SAMPLE)
                        delta_sample[j] = i;
                }
            }
        }

        // For each emitter group, estimate solid angle of delta surfaces
        float weighted_coupling = 0.f;
        float total_em_flux = sp.total_emissive_flux;
        if (total_em_flux <= 0.f) total_em_flux = 1.f;

        for (const auto& em : emitters) {
            float solid_angle = 0.f;
            for (uint32_t di : delta_sample) {
                const auto& dt = scene.triangles[di];
                float3 centroid = (dt.v0 + dt.v1 + dt.v2) * (1.f / 3.f);
                float3 to_delta = centroid - em.centroid;
                float dist2 = dot(to_delta, to_delta);
                if (dist2 < 1e-8f) continue;
                float dist = sqrtf(dist2);
                float3 dir = to_delta * (1.f / dist);

                // Cosine at delta surface (both sides count)
                float3 dn = dt.geometric_normal();
                float cos_delta = fabsf(dot(dir, dn));
                if (cos_delta < 1e-6f) continue;

                // Projected solid angle contribution
                solid_angle += dt.area() * cos_delta / dist2;
            }

            // Scale by subsample ratio when reservoir-sampled
            if (sp.num_delta_triangles > MAX_DELTA_SAMPLE)
                solid_angle *= (float)sp.num_delta_triangles / (float)MAX_DELTA_SAMPLE;

            // Normalize to hemisphere (2π sr)
            float coupling = solid_angle / (2.f * PI);
            weighted_coupling += coupling * (em.flux / total_em_flux);
        }

        sp.emitter_delta_coupling = fminf(weighted_coupling, 1.f);
        // Caustic difficulty: budget multiplier relative to a well-coupled reference
        constexpr float REF_COUPLING = 0.10f;
        float c = fmaxf(sp.emitter_delta_coupling, 0.001f);
        sp.caustic_difficulty = fminf(REF_COUPLING / c, 100.f);
        sp.caustic_difficulty = fmaxf(sp.caustic_difficulty, 1.f);
    }

    // ── Lighting classification ─────────────────────────────────────
    if (sp.emitter_size_ratio > 0.01f)
        sp.dominant_lighting = LightingType::LargeArea;
    else if (sp.emitter_size_ratio <= 0.01f && sp.num_emitters > 0)
        sp.dominant_lighting = LightingType::SmallPoint;
    else
        sp.dominant_lighting = LightingType::Mixed;

    // ── Convergence hints ───────────────────────────────────────────

    // Bounces: more for glass/caustics, fewer for simple diffuse
    sp.recommended_max_bounces = sp.has_glass ? 12 : 8;
    if (sp.has_translucent) sp.recommended_max_bounces = 16;

    // Photon budget: scale with scene complexity and lighting
    if (sp.has_caustic_paths)
        sp.recommended_photon_budget = 4000000;
    else if (sp.dominant_lighting == LightingType::SmallPoint)
        sp.recommended_photon_budget = 2000000;
    else
        sp.recommended_photon_budget = 1000000;

    // Caustic photon budget: separate allocation for delta materials
    if (sp.has_caustic_paths) {
        sp.recommended_caustic_photon_budget =
            sp.num_delta_materials > 3 ? 2000000 : 1000000;
    }

    // Guide training iterations
    sp.recommended_guide_training_iters = 10;

    // Guide fraction: higher for complex indirect
    sp.recommended_guide_fraction =
        (sp.dominant_lighting == LightingType::LargeArea) ? 0.3f : 0.5f;

    // ── Mostly-indirect emitters → rely more on photons and guide ───
    // When lights are occluded from most surfaces (e.g. lamps behind
    // covers), NEE is largely wasted; path guiding and photon mapping
    // become the primary convergence drivers.
    if (sp.mostly_indirect_emitters) {
        sp.recommended_photon_budget =
            (std::max)(sp.recommended_photon_budget, 4000000);
        sp.recommended_guide_fraction =
            fmaxf(sp.recommended_guide_fraction, 0.6f);
        sp.recommended_max_bounces =
            (std::max)(sp.recommended_max_bounces, 12);
        sp.recommended_guide_training_iters =
            (std::max)(sp.recommended_guide_training_iters, 15);
    }

    // Gather radius: scale with scene size
    float base_radius = sp.scene_diagonal * 0.005f;
    sp.recommended_gather_radius = base_radius;
    sp.recommended_caustic_radius = base_radius * 0.3f;

    // ── Caustic light tracing budget (coupling-driven) ────────────
    if (sp.has_caustic_paths && sp.caustic_geometry_favorable) {
        sp.recommended_caustic_enabled = true;

        // Scale budget by coupling difficulty.  Direction biasing in the
        // kernel handles the worst cases, but poorly-coupled scenes
        // still benefit from a larger budget.
        float multiplier = fminf(sp.caustic_difficulty, 8.f);
        int base_budget = (int)((float)DEFAULT_CAUSTIC_PHOTONS_PER_FRAME * multiplier);
        if (sp.mostly_indirect_emitters)
            base_budget *= 2;
        // Clamp to [256K, 2M]
        base_budget = (std::max)(base_budget, DEFAULT_CAUSTIC_PHOTONS_PER_FRAME);
        base_budget = (std::min)(base_budget, 2097152);
        sp.recommended_caustic_photons_per_frame = base_budget;

        // Splat luminance: relax for dim emitters, tighten for bright
        float flux = fmaxf(sp.total_emissive_flux, 1.f);
        sp.recommended_caustic_max_splat_luminance =
            fminf(fmaxf(flux * 2.f, 50.f), 500.f);
    }

    // ── Light tree: disable for trivial (≤1 emitter) scenes ──────────
    sp.recommended_light_tree_enabled = (sp.num_emitters > 1);

    return sp;
}
