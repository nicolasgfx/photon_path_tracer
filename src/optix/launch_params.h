#pragma once
// ─────────────────────────────────────────────────────────────────────
// launch_params.h – Shared data between host and OptiX device code
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"
#include "core/config.h"

#ifdef PPT_USE_OPTIX
#include <optix.h>
#endif

// ── Render mode (matches RenderMode in renderer.h) ──────────────────
// Duplicated as int for device compatibility
constexpr int RENDER_MODE_FULL          = 0;
constexpr int RENDER_MODE_DIRECT_ONLY   = 1;
constexpr int RENDER_MODE_INDIRECT_ONLY = 2;
constexpr int RENDER_MODE_PHOTON_MAP    = 3;
constexpr int RENDER_MODE_NORMALS       = 4;
constexpr int RENDER_MODE_MATERIAL_ID   = 5;
constexpr int RENDER_MODE_DEPTH         = 6;

// ── Launch parameters (accessible from all OptiX programs) ──────────

struct LaunchParams {
    // Output buffer
    float*    spectrum_buffer;   // [width * height * NUM_LAMBDA]
    float*    sample_counts;     // [width * height]
    uint8_t*  srgb_buffer;       // [width * height * 4]

    // Component output buffers (for debug PNGs)
    float*    nee_direct_buffer;       // [width * height * NUM_LAMBDA]
    float*    photon_indirect_buffer;  // [width * height * NUM_LAMBDA]

    // Image dimensions
    int width;
    int height;

    // Camera
    float3 cam_pos;
    float3 cam_u;
    float3 cam_v;
    float3 cam_w;
    float3 cam_lower_left;
    float3 cam_horizontal;
    float3 cam_vertical;

    // Rendering parameters
    int    samples_per_pixel;
    int    max_bounces;
    int    photon_max_bounces;    // max bounces for photon tracing (separate from render)
    int    frame_number;
    int    render_mode;          // one of RENDER_MODE_* constants
    int    is_final_render;      // 0 = debug first-hit, 1 = full path tracing
    int    debug_shadow_rays;    // 1 = debug_first_hit casts shadow rays (NEE PNG)
    int    nee_light_samples;    // M: shadow-ray samples at bounce 0
    int    nee_deep_samples;     // shadow-ray samples at bounce >= 1

    // Scene geometry (device pointers)
    float3*   vertices;          // [num_tris * 3]
    float3*   normals;           // [num_tris * 3]
    uint32_t* material_ids;      // [num_tris]

    // Material spectral data (device pointers, flattened)
    int       num_materials;
    float*    Kd;                // [num_materials * NUM_LAMBDA]
    float*    Ks;                // [num_materials * NUM_LAMBDA]
    float*    Le;                // [num_materials * NUM_LAMBDA]
    float*    roughness;         // [num_materials]
    float*    ior;               // [num_materials]
    uint8_t*  mat_type;          // [num_materials]

    // Photon map (device pointers)
    float*    photon_pos_x;
    float*    photon_pos_y;
    float*    photon_pos_z;
    float*    photon_wi_x;
    float*    photon_wi_y;
    float*    photon_wi_z;
    uint16_t* photon_lambda;
    float*    photon_flux;
    int       num_photons;

    // Hash grid (device pointers)
    uint32_t* grid_sorted_indices;
    uint32_t* grid_cell_start;
    uint32_t* grid_cell_end;
    float     grid_cell_size;
    uint32_t  grid_table_size;

    // Photon gather radius
    float gather_radius;

    // Emitter data (for GPU photon tracing)
    uint32_t* emissive_tri_indices;  // [num_emissive]
    float*    emissive_cdf;          // [num_emissive] cumulative distribution
    int       num_emissive;
    float     total_emissive_power;

    // Photon output buffers (for __raygen__photon_trace)
    float*    out_photon_pos_x;
    float*    out_photon_pos_y;
    float*    out_photon_pos_z;
    float*    out_photon_wi_x;
    float*    out_photon_wi_y;
    float*    out_photon_wi_z;
    uint16_t* out_photon_lambda;
    float*    out_photon_flux;
    unsigned int* out_photon_count;  // atomic counter (device)
    int       max_stored_photons;

    // ── GPU kernel profiling (per-pixel clock64 accumulators) ────────
    // Each buffer is [width * height] long long values.
    // Set to nullptr to disable profiling.
    long long* prof_total;           // total kernel time per pixel
    long long* prof_ray_trace;       // time in optixTrace (primary + bounces)
    long long* prof_nee;             // time in NEE direct lighting
    long long* prof_photon_gather;   // time in photon density estimation
    long long* prof_bsdf;            // time in BSDF eval + continuation

#ifdef PPT_USE_OPTIX
    OptixTraversableHandle traversable;
#endif
};

// ── SBT record (Shader Binding Table) ───────────────────────────────

struct RayGenSbtRecord {
    char pad[1]; // avoid zero-size struct
};

struct MissSbtRecord {
    float3 background_color;
};

struct HitGroupSbtRecord {
    // Per-triangle geometry (indexed by primitive ID)
    float3*   vertices;       // [num_tris * 3]
    float3*   normals;        // [num_tris * 3]
    float2*   texcoords;      // [num_tris * 3]
    uint32_t* material_ids;   // [num_tris]

    // Material arrays (indexed by material_id, flattened spectral)
    float*    Kd;             // [num_materials * NUM_LAMBDA]
    float*    Ks;             // [num_materials * NUM_LAMBDA]
    float*    Le;             // [num_materials * NUM_LAMBDA]
    float*    roughness;      // [num_materials]
    float*    ior;            // [num_materials]
    uint8_t*  mat_type;       // [num_materials]
};
