#pragma once
// ─────────────────────────────────────────────────────────────────────
// launch_params.h – Shared data between host and OptiX device code (v3)
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"
#include "core/config.h"
#include "photon/photon_bins.h"
#include "volume/medium.h"

#ifdef PPT_USE_OPTIX
#include <optix.h>
#endif

// RenderMode enum class is now defined in core/types.h
// (shared between CPU and GPU — no more duplicated int constants).

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

    // AOV buffers for OptiX denoiser guide layers (§5 config.h)
    // Written at the first non-specular hit during full_path_trace.
    float*    albedo_buffer;           // [width * height * 4] RGBA linear diffuse albedo
    float*    normal_buffer;           // [width * height * 4] world-space shading normal

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
    int    max_bounces;           // [legacy] max specular chain bounces
    int    max_bounces_camera;    // [v3] max camera path depth (§4)
    int    min_bounces_rr;        // [v3] guaranteed bounces before RR
    float  rr_threshold;          // [v3] max RR survival probability
    float  guide_fraction;        // [v3] photon-guided fraction (0..1)
    int    guide_fallback_bounce; // [v3] switch to photon gather after this bounce
    int    photon_final_gather;   // [v3] 1 = use photon map at terminal bounce
    int    preview_mode;          // 1 = fast preview (skip kNN guide/caustic/gather, 3-bounce cap)
    int    photon_max_bounces;    // max bounces for photon tracing (separate from render)

    int    frame_number;
    RenderMode render_mode;      // shared enum from core/types.h
    float  exposure;             // linear exposure multiplier (applied before tone mapping)
    int    skip_tonemap;         // 1 = skip inline tonemap in __raygen__render (use post-process kernel)

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
    float*    opacity;            // [num_materials]  0..1 (from MTL 'd')
    float*    clearcoat_weight;   // [num_materials]  clearcoat layer weight
    float*    clearcoat_roughness;// [num_materials]  clearcoat roughness
    float*    sheen;              // [num_materials]  sheen weight
    float*    sheen_tint;         // [num_materials]  sheen tint factor
    uint8_t*  mat_dispersion;     // [num_materials]  1 = Cauchy dispersion enabled
    float*    cauchy_A;           // [num_materials]  Cauchy A coefficient
    float*    cauchy_B;           // [num_materials]  Cauchy B coefficient
    uint8_t*  mat_thin;           // [num_materials]  1 = thin dielectric (no refraction offset)

    // Per-material interior medium (§7.7 Translucent surfaces)
    int*              mat_medium_id;  // [num_materials]  medium index or -1
    HomogeneousMedium* media;         // [num_media]  flat array of interior media
    int               num_media;      // number of entries in the media table

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
    int       num_photons;
    int       num_photons_emitted; // N_emitted (for density normalisation, §5.3)
    int       photon_map_seed;     // RNG seed offset for multi-map re-tracing

    // ── Dense 3D grid (surface photon lookup) ────────────────────────
    uint32_t* dense_sorted_indices;     // [num_photons] sorted by cell
    uint32_t* dense_cell_start;         // [total_cells]
    uint32_t* dense_cell_end;           // [total_cells]
    int       dense_valid;              // 1 = built, 0 = not available
    float     dense_min_x, dense_min_y, dense_min_z;  // AABB min
    float     dense_cell_size;          // cell edge length
    int       dense_dim_x, dense_dim_y, dense_dim_z;  // grid resolution
    float     guide_cone_cos_half_angle;  // cos(half-angle) for photon wi cone jitter (1.0 = off)

    // Per-triangle photon irradiance heatmap (precomputed on CPU, for preview)
    float*    tri_photon_irradiance;  // [num_triangles] accumulated scalar irradiance
    int       num_triangles;          // total scene triangles
    int       show_photon_heatmap;    // 0 = off, 1 = show heatmap overlay

    // Photon gather radius
    float gather_radius;

    // ── Dual-budget caustic system (Jensen 1996 two-budget) ─────────
    int       num_caustic_emitted;    // N_caustic for normalisation
    float     caustic_gather_radius;  // gather radius for caustic budget
    int       caustic_only_store;     // 1 = only store caustic photons

    // Emitter data (for GPU photon tracing)
    uint32_t* emissive_tri_indices;  // [num_emissive]
    float*    emissive_cdf;          // [num_emissive] cumulative distribution
    int*      emissive_local_idx;    // [num_triangles] tri_id → local emissive index (-1 = not emissive)
    int       num_emissive;
    float     total_emissive_power;

    // ── Targeted caustic emission (Jensen §9.2) ─────────────────────
    // Specular triangle set for importance-sampled caustic photon emission.
    // The alias table enables O(1) area-weighted specular triangle sampling.
    uint32_t* targeted_spec_tri_indices;  // [num_targeted_spec_tris] global triangle indices
    float*    targeted_spec_alias_prob;   // [num_targeted_spec_tris] alias table probabilities
    uint32_t* targeted_spec_alias_idx;    // [num_targeted_spec_tris] alias table redirect indices
    float*    targeted_spec_pdf;          // [num_targeted_spec_tris] normalized PDF values
    float*    targeted_spec_areas;        // [num_targeted_spec_tris] per-triangle areas
    int       num_targeted_spec_tris;     // number of specular triangles
    int       targeted_mode;              // 1 = this launch is targeted emission

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
    uint8_t*  out_photon_is_caustic;     // [max_stored] 1 = caustic path, 0 = global
    uint8_t*  out_photon_path_flags;     // [max_stored] PHOTON_FLAG_* bit field (F2 debug overlay)
    uint32_t* out_photon_tri_id;         // [max_stored] scene triangle index at deposit
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

    // ── Adaptive sampling ────────────────────────────────────────────
    // All three pointers are nullptr when adaptive sampling is disabled.
    float*   lum_sum;       // [width * height]  Σ Y_i   (linear luminance)
    float*   lum_sum2;      // [width * height]  Σ Y_i²
    uint8_t* active_mask;   // [width * height]  1 = trace this pixel, 0 = skip

    // ── Per-bounce AOV buffers (DB-04, §10.3) ────────────────────────
    // When enabled, capture the radiance contribution at each of the
    // first MAX_AOV_BOUNCES bounce depths for diagnostic visualisation.
    float*    bounce_aov[MAX_AOV_BOUNCES]; // [W*H*NUM_LAMBDA] per-bounce radiance, nullptr = disabled
    int       bounce_aov_enabled;          // 0 = off, 1 = on

    // ── Participating medium (volumetric scattering) ────────────────
    int   volume_enabled;       // 0 = off, 1 = on
    float volume_density;       // base extinction scale
    float volume_falloff;       // exponential height decay (0 = homogeneous)
    float volume_albedo;        // σ_s / σ_t
    int   volume_samples;       // medium samples per ray segment
    float volume_max_t;         // max march distance for miss rays

    // ── Dense 3D cell-bin grid (retained for volume guide only) ────────
    // Surface guided sampling now uses per-hitpoint kNN (optix_guided.cuh).
    int        photon_bin_count;     // runtime copy of PHOTON_BIN_COUNT

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

    // ── Volume photon kNN gather data (MT-07) ────────────────────────
    // Uploaded from volume_photons_ SoA + volume_grid_ HashGrid for
    // GPU-side kNN density estimation at terminal scatter events.
    float*    vol_photon_pos_x;
    float*    vol_photon_pos_y;
    float*    vol_photon_pos_z;
    float*    vol_photon_wi_x;
    float*    vol_photon_wi_y;
    float*    vol_photon_wi_z;
    uint16_t* vol_photon_lambda;       // [num_vol_photons] wavelength bin
    float*    vol_photon_flux;         // [num_vol_photons] per-photon flux
    int       num_vol_photons;
    int       num_vol_photons_emitted; // N_emitted for density normalisation
    uint32_t* vol_grid_sorted_indices;
    uint32_t* vol_grid_cell_start;
    uint32_t* vol_grid_cell_end;
    float     vol_grid_cell_size;
    uint32_t  vol_grid_table_size;
    float     vol_gather_radius;       // search radius for volume kNN

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
