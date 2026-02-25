#pragma once
// ─────────────────────────────────────────────────────────────────────
// emitter_points.h – Precomputed representative points on emissive
//                    surfaces for directional-coverage shadow ray NEE
// ─────────────────────────────────────────────────────────────────────
// Pipeline:
//   1. Per-triangle stratified point generation (area-proportional)
//   2. Cross-triangle density deduplication (spatial, no averaging)
//   3. The resulting global point set feeds the per-cell hemisphere
//      coverage algorithm in LightCache (light_cache.h).
//
// Key design decisions:
//   - Points are deterministic, not random — enables reproducible
//     directional coverage tests across frames.
//   - Deduplication never averages: merged points may not lie on any
//     emitter surface.  Instead, redundant points are simply removed.
//   - Each point records its source emissive_local_idx and global
//     triangle index for GPU vertex lookup.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "scene/triangle.h"
#include "scene/material.h"

#include <vector>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <chrono>
#include <unordered_set>
#include <unordered_map>

// ── Configuration ───────────────────────────────────────────────────

// Target area per representative point (fraction of scene extent²).
// Smaller = more points per triangle, denser coverage.
//   0.0001 = ~10k points on unit²  |  0.001 = ~1k points on unit²
constexpr float EMITTER_POINT_DENSITY = 0.0002f;

// Minimum distance between two accepted points (fraction of scene extent).
// Points closer than this are considered redundant.
//   0.005 = 0.5% of unit scene  |  0.01 = 1%
constexpr float EMITTER_DEDUP_MIN_DIST = 0.005f;

// Every triangle gets at least this many representative points.
constexpr int EMITTER_MIN_POINTS_PER_TRI = 1;

// Maximum total points (safety cap to prevent memory explosion).
constexpr int EMITTER_MAX_TOTAL_POINTS = 100000;

// ── Data structures ─────────────────────────────────────────────────

// A single representative point on an emissive surface.
// Stored on CPU, uploaded to GPU for shadow ray targeting.
struct EmitterPoint {
    float3   position;           // world-space position on emitter surface
    float3   normal;             // geometric normal of the emitter triangle
    uint16_t emissive_local_idx; // index into emissive_tri_indices[]
    uint32_t global_tri_idx;     // global triangle index (for vertex lookup)
};

// Complete set of deduplicated emitter representative points.
struct EmitterPointSet {
    std::vector<EmitterPoint> points;

    // ── Step 1: Per-triangle stratified point generation ────────────
    // For each emissive triangle, generate n_points = max(1, area/target_area)
    // points using stratified grid sampling within barycentric coords.
    void generate_per_triangle(
        const std::vector<Triangle>&  triangles,
        const std::vector<Material>&  materials,
        const std::vector<uint32_t>&  emissive_tri_indices,
        float target_area_per_point)
    {
        points.clear();
        (void)materials;  // reserved for future emission-weighted point density

        if (emissive_tri_indices.empty()) return;

        for (uint16_t local_idx = 0; local_idx < (uint16_t)emissive_tri_indices.size(); ++local_idx) {
            uint32_t tri_idx = emissive_tri_indices[local_idx];
            const Triangle& tri = triangles[tri_idx];
            float area = tri.area();

            // Number of points proportional to area
            int n = (std::max)(EMITTER_MIN_POINTS_PER_TRI,
                              (int)std::ceil(area / (std::max)(target_area_per_point, 1e-12f)));

            // Cap per-triangle to prevent single huge triangles from dominating
            n = (std::min)(n, 256);

            float3 gn = tri.geometric_normal();

            if (n == 1) {
                // Single point: centroid
                EmitterPoint ep;
                ep.position = (tri.v0 + tri.v1 + tri.v2) * (1.0f / 3.0f);
                ep.normal = gn;
                ep.emissive_local_idx = local_idx;
                ep.global_tri_idx = tri_idx;
                points.push_back(ep);
            } else {
                // Stratified grid in barycentric space
                // Use a grid of strata_n × strata_n in the unit triangle
                int strata_n = (int)std::ceil(std::sqrt((float)n));
                int generated = 0;

                for (int si = 0; si < strata_n && generated < n; ++si) {
                    for (int sj = 0; sj <= si && generated < n; ++sj) {
                        // Map (si, sj) to barycentric center of stratum
                        // Triangle subdivision: split into strata_n² sub-triangles
                        float u = ((float)si + 0.5f) / (float)strata_n;
                        float v = ((float)sj + 0.5f) / (float)strata_n;

                        // Ensure within triangle: u + v <= 1
                        if (u + v > 1.0f) {
                            u = 1.0f - u;
                            v = 1.0f - v;
                        }

                        float w = 1.0f - u - v;

                        EmitterPoint ep;
                        ep.position = tri.v0 * w + tri.v1 * u + tri.v2 * v;
                        ep.normal = gn;
                        ep.emissive_local_idx = local_idx;
                        ep.global_tri_idx = tri_idx;
                        points.push_back(ep);
                        ++generated;
                    }
                }

                // If we haven't reached n yet, fill with centroid-offset points
                while (generated < n) {
                    // Distribute remaining points using low-discrepancy offsets
                    float t = (float)(generated + 1) / (float)(n + 1);
                    float su = std::sqrt(t);
                    float alpha = 1.0f - su;
                    float beta = 0.5f * su;  // mid-value for v
                    float gamma = 1.0f - alpha - beta;

                    EmitterPoint ep;
                    ep.position = tri.v0 * alpha + tri.v1 * beta + tri.v2 * gamma;
                    ep.normal = gn;
                    ep.emissive_local_idx = local_idx;
                    ep.global_tri_idx = tri_idx;
                    points.push_back(ep);
                    ++generated;
                }
            }
        }

        std::printf("[EmitterPoints] Step 1: Generated %zu raw points from %zu emissive triangles\n",
                    points.size(), emissive_tri_indices.size());
    }

    // ── Step 2: Cross-triangle density deduplication ────────────────
    // Remove points that are too close to already-accepted points.
    // Process order: largest triangles first, so small isolated lights
    // are not consumed by their neighbors.
    // NEVER averages positions — only removes redundant points.
    void deduplicate(
        const std::vector<Triangle>&  triangles,
        const std::vector<uint32_t>&  emissive_tri_indices,
        float min_distance)
    {
        if (points.empty()) return;

        auto t_start = std::chrono::high_resolution_clock::now();
        const float min_dist2 = min_distance * min_distance;

        // Sort points by source triangle area (descending)
        // Points from larger triangles get priority
        std::sort(points.begin(), points.end(),
            [&](const EmitterPoint& a, const EmitterPoint& b) {
                float area_a = triangles[emissive_tri_indices[a.emissive_local_idx]].area();
                float area_b = triangles[emissive_tri_indices[b.emissive_local_idx]].area();
                return area_a > area_b;
            });

        // Simple spatial grid for efficient neighbor queries
        const float cell_size = min_distance * 2.0f;
        const float inv_cell = 1.0f / cell_size;

        // Use a hash set of occupied cells for quick proximity checks
        auto cell_hash = [](int x, int y, int z) -> uint64_t {
            return (uint64_t)((uint32_t)(x * 73856093u) ^
                              (uint32_t)(y * 19349663u) ^
                              (uint32_t)(z * 83492791u));
        };

        // Store accepted points with their cell coordinates
        struct AcceptedEntry {
            float3 pos;
            int cx, cy, cz;
        };

        // Multi-map: cell_hash → list of accepted points in that cell
        std::unordered_multimap<uint64_t, size_t> cell_map;
        std::vector<AcceptedEntry> accepted_entries;
        std::vector<EmitterPoint> accepted;
        accepted.reserve(points.size());
        accepted_entries.reserve(points.size());

        for (const auto& pt : points) {
            int cx = (int)std::floor(pt.position.x * inv_cell);
            int cy = (int)std::floor(pt.position.y * inv_cell);
            int cz = (int)std::floor(pt.position.z * inv_cell);

            // Check 3×3×3 neighbourhood for any accepted point within min_dist
            bool too_close = false;
            for (int dx = -1; dx <= 1 && !too_close; ++dx) {
                for (int dy = -1; dy <= 1 && !too_close; ++dy) {
                    for (int dz = -1; dz <= 1 && !too_close; ++dz) {
                        uint64_t h = cell_hash(cx + dx, cy + dy, cz + dz);
                        auto range = cell_map.equal_range(h);
                        for (auto it = range.first; it != range.second; ++it) {
                            const auto& ae = accepted_entries[it->second];
                            float3 diff = pt.position - ae.pos;
                            if (dot(diff, diff) < min_dist2) {
                                too_close = true;
                                break;
                            }
                        }
                    }
                }
            }

            if (!too_close) {
                size_t idx = accepted_entries.size();
                uint64_t h = cell_hash(cx, cy, cz);
                cell_map.emplace(h, idx);
                accepted_entries.push_back({pt.position, cx, cy, cz});
                accepted.push_back(pt);
            }
        }

        size_t before = points.size();
        points = std::move(accepted);

        // Safety cap
        if ((int)points.size() > EMITTER_MAX_TOTAL_POINTS) {
            points.resize(EMITTER_MAX_TOTAL_POINTS);
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        std::printf("[EmitterPoints] Step 2: Deduplicated %zu → %zu points  "
                    "(min_dist=%.4f)  %.1f ms\n",
                    before, points.size(), min_distance, ms);
    }

    // ── Full pipeline: generate + deduplicate ───────────────────────
    void build(
        const std::vector<Triangle>&  triangles,
        const std::vector<Material>&  materials,
        const std::vector<uint32_t>&  emissive_tri_indices,
        float scene_extent = 1.0f)
    {
        float target_area = EMITTER_POINT_DENSITY * scene_extent * scene_extent;
        float min_dist    = EMITTER_DEDUP_MIN_DIST * scene_extent;

        generate_per_triangle(triangles, materials, emissive_tri_indices, target_area);
        deduplicate(triangles, emissive_tri_indices, min_dist);

        std::printf("[EmitterPoints] Final: %zu representative emitter points\n",
                    points.size());
    }

    size_t size() const { return points.size(); }
    bool   empty() const { return points.empty(); }
};
