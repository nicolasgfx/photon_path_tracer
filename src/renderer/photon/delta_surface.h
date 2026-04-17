#pragma once
// ─────────────────────────────────────────────────────────────────────
// photon/delta_surface.h – Area-weighted CDF over delta surface tris
//
// Built once at scene load time.  Uploaded to GPU for the caustic
// light tracer to importance-sample target points on mirror/glass/
// translucent surfaces.
// ─────────────────────────────────────────────────────────────────────
#include "scene/scene.h"
#include <vector>
#include <cstdint>

struct DeltaSurfaceDistribution {
    std::vector<uint32_t> tri_indices; // global triangle indices
    std::vector<float>    cdf;         // cumulative area-weighted PDF [0,1]
    float                 total_area = 0.f;

    bool empty() const { return tri_indices.empty(); }
    int  count() const { return (int)tri_indices.size(); }
};

// Scan the scene for delta-material triangles and build an area-weighted CDF.
inline DeltaSurfaceDistribution build_delta_surface_distribution(const Scene& scene) {
    DeltaSurfaceDistribution d;

    for (uint32_t i = 0; i < (uint32_t)scene.triangles.size(); ++i) {
        const auto& t = scene.triangles[i];
        auto mt = scene.materials[t.material_id].type;
        if (mt == MaterialType::Mirror || mt == MaterialType::Glass ||
            mt == MaterialType::Translucent) {
            float a = t.area();
            if (a > 0.f) {
                d.tri_indices.push_back(i);
                d.total_area += a;
                d.cdf.push_back(d.total_area);
            }
        }
    }

    // Normalize CDF to [0, 1]
    if (d.total_area > 0.f) {
        float inv = 1.f / d.total_area;
        for (float& v : d.cdf) v *= inv;
        d.cdf.back() = 1.f; // ensure exact endpoint
    }

    return d;
}
