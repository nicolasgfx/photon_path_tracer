#pragma once
// ─────────────────────────────────────────────────────────────────────
// accel_types.h – OptiX acceleration types, SBT records, constants
//
// Phase 3: Acceleration stage.  Defines pipeline configuration,
// SBT record layouts, and the AccelStructure handle wrapper.
// ─────────────────────────────────────────────────────────────────────
#include <cstdint>

#ifdef PPT_USE_OPTIX
#include <optix.h>
#endif

// ── Pipeline configuration constants ─────────────────────────────────
// Must match the payload layout in optix_programs.cu / optix_utils.cuh.
namespace accel {

constexpr int NUM_PAYLOAD_VALUES   = 18;  // float3 pos + float3 sn + t + mat + tri + hit + float3 gn + float2 uv + float3 tangent
constexpr int NUM_ATTRIBUTE_VALUES = 2;   // barycentrics (built-in triangles)
constexpr int MAX_TRACE_DEPTH      = 2;   // radiance + shadow
constexpr int STACK_SIZE           = 16384;

// Ray types (SBT offset)
constexpr int RAY_TYPE_RADIANCE = 0;
constexpr int RAY_TYPE_SHADOW   = 1;
constexpr int NUM_RAY_TYPES     = 2;

// Scene epsilon for self-intersection avoidance
constexpr float SCENE_EPSILON    = 1e-4f;
constexpr float DEFAULT_RAY_TMAX = 1e20f;

} // namespace accel

// ── SBT record types ─────────────────────────────────────────────────
// The v5 pipeline stores all geometry/material data in LaunchParams
// (__constant__).  SBT records carry only the OptiX header; per-record
// data fields are minimal.

struct RayGenSbtRecord {
    char pad[1]; // avoid zero-size struct
};

struct MissSbtRecord {
    float background_r, background_g, background_b;
};

struct HitGroupSbtRecord {
    char pad[1]; // geometry data lives in LaunchParams, not SBT
};

// ── SBT record with alignment ───────────────────────────────────────
#ifdef PPT_USE_OPTIX

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4324) // structure was padded due to alignment
#endif

template <typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T    data;
};

using RayGenRecord   = SbtRecord<RayGenSbtRecord>;
using MissRecord     = SbtRecord<MissSbtRecord>;
using HitGroupRecord = SbtRecord<HitGroupSbtRecord>;

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif // PPT_USE_OPTIX

// ── AccelStructure ──────────────────────────────────────────────────
// Lightweight handle wrapper returned by the builder.
struct AccelStructure {
#ifdef PPT_USE_OPTIX
    OptixTraversableHandle handle = 0;
#else
    uint64_t               handle = 0;
#endif
    bool     instanced    = false;  // true when IAS path was used
    size_t   compacted_bytes = 0;   // total GPU memory for accel
    int      num_triangles   = 0;
    int      num_instances   = 0;
};
