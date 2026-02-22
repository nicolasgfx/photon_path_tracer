#pragma once
// ─────────────────────────────────────────────────────────────────────
// launch_params.h – Shared data between host and OptiX device code
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"
#include "core/config.h"
#include "core/photon_bins.h"
#include "core/light_cache.h"

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

// ── GPU texture descriptor (for flat atlas lookup) ──────────────────
struct GpuTexDesc {
    int offset;    // start index in the flat RGBA atlas (in floats)
    int width;
    int height;
};

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
    float  cam_lens_radius;  // >0 enables thin-lens DOF
    float  cam_focus_dist;   // focus distance (scene units), needed for focus-range jitter
    float  cam_focus_range;  // absolute slab depth (scene units) around focus plane

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
    float2*   texcoords;         // [num_tris * 3]
    uint32_t* material_ids;      // [num_tris]

    // Material spectral data (device pointers, flattened)
    int       num_materials;
    float*    Kd;                // [num_materials * NUM_LAMBDA]
    float*    Ks;                // [num_materials * NUM_LAMBDA]
    float*    Le;                // [num_materials * NUM_LAMBDA]
    float*    roughness;         // [num_materials]
    float*    ior;               // [num_materials]
    uint8_t*  mat_type;          // [num_materials]
    int*      diffuse_tex;        // [num_materials]  texture ID or -1
    int*      emission_tex;       // [num_materials]  texture ID or -1 (map_Ke)

    // Texture atlas (flat RGBA float buffer, all textures concatenated)
    float*      tex_atlas;        // [total_texels * 4]
    GpuTexDesc* tex_descs;        // [num_textures]
    int         num_textures;

    // Photon map (device pointers)
    float*    photon_pos_x;
    float*    photon_pos_y;
    float*    photon_pos_z;
    float*    photon_wi_x;
    float*    photon_wi_y;
    float*    photon_wi_z;
    float*    photon_norm_x;   // geometric surface normal at photon hit
    float*    photon_norm_y;
    float*    photon_norm_z;
    uint16_t* photon_lambda;   // [num_photons * HERO_WAVELENGTHS] wavelength bins
    float*    photon_flux;     // [num_photons * HERO_WAVELENGTHS] per-hero flux
    uint8_t*  photon_num_hero; // [num_photons] valid hero count per photon
    uint8_t*  photon_bin_idx;      // [num_photons] precomputed Fibonacci bin index
    int       num_photons;
    int       num_photons_emitted; // N_emitted (for density normalisation, §5.3)
    int       photon_map_seed;     // RNG seed offset for multi-map re-tracing

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
    float*    out_photon_norm_x;   // geometric normal at photon hit (output)
    float*    out_photon_norm_y;
    float*    out_photon_norm_z;
    uint16_t* out_photon_lambda;   // [max_stored * HERO_WAVELENGTHS]
    float*    out_photon_flux;     // [max_stored * HERO_WAVELENGTHS]
    uint8_t*  out_photon_num_hero; // [max_stored] valid hero count per photon
    uint16_t* out_photon_source_emissive; // [max_stored] source emissive local index
    unsigned int* out_photon_count;  // atomic counter (device)
    int       max_stored_photons;

    // Volume photon output buffers (for __raygen__photon_trace)
    float*    out_vol_photon_pos_x;
    float*    out_vol_photon_pos_y;
    float*    out_vol_photon_pos_z;
    float*    out_vol_photon_wi_x;
    float*    out_vol_photon_wi_y;
    float*    out_vol_photon_wi_z;
    uint16_t* out_vol_photon_lambda;
    float*    out_vol_photon_flux;
    unsigned int* out_vol_photon_count;  // atomic counter (device)
    int       max_stored_vol_photons;

    // ── GPU kernel profiling (per-pixel clock64 accumulators) ────────
    // Each buffer is [width * height] long long values.
    // Set to nullptr to disable profiling.
    long long* prof_total;           // total kernel time per pixel
    long long* prof_ray_trace;       // time in optixTrace (primary + bounces)
    long long* prof_nee;             // time in NEE direct lighting
    long long* prof_photon_gather;   // time in photon density estimation
    long long* prof_bsdf;            // time in BSDF eval + continuation

    // ── Per-pixel lobe balance (Bresenham accumulator) ────────────────
    // Persistent across frames.  Positive = specular deficit,
    // negative = diffuse deficit.  Guarantees optimal lobe coverage
    // so each pixel converges without redundant same-lobe paths.
    // nullptr disables the feature (falls back to random coin flip).
    float*   lobe_balance;  // [width * height]

    // ── Adaptive sampling ────────────────────────────────────────────
    // All three pointers are nullptr when adaptive sampling is disabled.
    float*   lum_sum;       // [width * height]  Σ Y_i   (linear luminance)
    float*   lum_sum2;      // [width * height]  Σ Y_i²
    uint8_t* active_mask;   // [width * height]  1 = trace this pixel, 0 = skip

    // ── Participating medium (volumetric scattering) ────────────────
    int   volume_enabled;       // 0 = off, 1 = on
    float volume_density;       // base extinction scale
    float volume_falloff;       // exponential height decay (0 = homogeneous)
    float volume_albedo;        // σ_s / σ_t
    int   volume_samples;       // medium samples per ray segment
    float volume_max_t;         // max march distance for miss rays

    // ── Dense 3D cell-bin grid (replaces per-pixel bin cache) ────────
    // Precomputed on CPU, uploaded once.  Each grid cell contains
    // PHOTON_BIN_COUNT directional bins with accumulated flux from
    // the 3×3×3 photon neighbourhood.  O(1) lookup at render time.
    PhotonBin* cell_bin_grid;        // [grid_total_cells * photon_bin_count]
    int        photon_bin_count;     // runtime copy of PHOTON_BIN_COUNT
    int        cell_grid_valid;      // 1 = grid uploaded, 0 = not available
    int        use_dense_grid_gather; // 1 = use cell-bin path, 0 = hash-grid walk
    float      cell_grid_min_x;     // AABB min corner
    float      cell_grid_min_y;
    float      cell_grid_min_z;
    float      cell_grid_cell_size; // cell size (2 × gather_radius)
    int        cell_grid_dim_x;     // grid dimensions
    int        cell_grid_dim_y;
    int        cell_grid_dim_z;

    // ── Dense 3D cell-bin grid for VOLUME photons ────────────────────
    // Same structure as the surface grid but stores photons deposited
    // inside the medium (free-flight scattering events).  Used for
    // multi-scatter volume radiance via phase-function gathering.
    PhotonBin* vol_cell_bin_grid;        // [vol_grid_total_cells * photon_bin_count]
    int        vol_cell_grid_valid;      // 1 = grid uploaded, 0 = not available
    float      vol_cell_grid_min_x;
    float      vol_cell_grid_min_y;
    float      vol_cell_grid_min_z;
    float      vol_cell_grid_cell_size;
    int        vol_cell_grid_dim_x;
    int        vol_cell_grid_dim_y;
    int        vol_cell_grid_dim_z;

    // ── Light importance cache (per-cell top-K lights for NEE) ────────
    // Built from photon deposition statistics.  Each hash cell stores
    // the top NEE_CELL_TOP_K most important light sources ranked by
    // accumulated photon flux.  NEE samples from this cache instead
    // of the global power CDF, reducing shadow-ray variance.
    CellLightEntry* light_cache_entries;         // [LIGHT_CACHE_TABLE_SIZE * NEE_CELL_TOP_K]
    int*            light_cache_count;            // [LIGHT_CACHE_TABLE_SIZE]
    float*          light_cache_total_importance; // [LIGHT_CACHE_TABLE_SIZE]
    float           light_cache_cell_size;        // cell size (same as hash grid)
    int             light_cache_valid;            // 1 = cache uploaded, 0 = not available
    int             use_light_cache;              // 1 = use cached NEE, 0 = standard NEE

    // ── SPPM (Stochastic Progressive Photon Mapping) ────────────────
    // Per-pixel visible-point buffers (written by camera pass, read by
    // gather kernel).  All buffers are [width * height].
    int    sppm_mode;                ///< 0 = off, 1 = camera pass, 2 = gather pass
    int    sppm_iteration;           ///< current iteration index k
    int    sppm_photons_per_iter;    ///< N_p photons emitted this iteration
    float  sppm_alpha;               ///< radius shrinkage factor α
    float  sppm_min_radius;          ///< floor for radius shrinkage

    // Visible-point storage (written by SPPM camera pass)
    float*    sppm_vp_pos_x;         ///< [W*H] hit position x
    float*    sppm_vp_pos_y;         ///< [W*H] hit position y
    float*    sppm_vp_pos_z;         ///< [W*H] hit position z
    float*    sppm_vp_norm_x;        ///< [W*H] geometric normal x (for gather filtering)
    float*    sppm_vp_norm_y;        ///< [W*H] geometric normal y
    float*    sppm_vp_norm_z;        ///< [W*H] geometric normal z
    float*    sppm_vp_wo_x;          ///< [W*H] outgoing direction (local) x
    float*    sppm_vp_wo_y;          ///< [W*H] outgoing direction (local) y
    float*    sppm_vp_wo_z;          ///< [W*H] outgoing direction (local) z
    uint32_t* sppm_vp_mat_id;        ///< [W*H] material index
    float*    sppm_vp_uv_u;          ///< [W*H] texture coord u
    float*    sppm_vp_uv_v;          ///< [W*H] texture coord v
    float*    sppm_vp_throughput;     ///< [W*H*NUM_LAMBDA] camera-path throughput
    uint8_t*  sppm_vp_valid;         ///< [W*H] 1 = valid visible point

    // Progressive per-pixel state (persists across iterations)
    float*    sppm_radius;           ///< [W*H] current gather radius r_i
    float*    sppm_N;                ///< [W*H] accumulated photon count N_i
    float*    sppm_tau;              ///< [W*H*NUM_LAMBDA] accumulated flux τ_i

    // Direct lighting accumulator (summed over iterations)
    float*    sppm_L_direct;         ///< [W*H*NUM_LAMBDA] summed NEE radiance

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
