#pragma once
// ─────────────────────────────────────────────────────────────────────
// photon_beam.h – Stub for photon beam tracing (future use)
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"
#include <vector>
#include <cstdint>

/// A photon beam segment stored for volumetric radiance estimation.
struct PhotonBeamSegment {
    float3   p0;           // segment start (world)
    float3   p1;           // segment end   (world)
    Spectrum beta;          // spectral power at segment start
    uint32_t medium_id;     // index into Scene::media[]
    float    t0;            // parametric start
    float    t1;            // parametric end
};

/// Placeholder photon beam map (collection of beam segments).
struct PhotonBeamMap {
    std::vector<PhotonBeamSegment> segments;
    void clear() { segments.clear(); }
    void reserve(size_t n) { segments.reserve(n); }
    size_t size() const { return segments.size(); }
    void push_back(const PhotonBeamSegment& s) { segments.push_back(s); }
};
