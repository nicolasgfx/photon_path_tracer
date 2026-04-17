#pragma once
// ─────────────────────────────────────────────────────────────────────
// launch_params.h – Shared data between host and OptiX device code (v5)
//
// RGB pipeline: all colour data uses 3-channel float (Color3 / float3).
// Spectral types removed.
//
// This struct grows as pipeline stages are added.  Each stage's fields
// are grouped under a section comment.  Stages read from LaunchParams
// via the device-side `extern "C" { __constant__ LaunchParams params; }`.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/config.h"
#include "lighting/light_tree_node.h"

#ifdef PPT_USE_OPTIX
#include <optix.h>
#endif

// (GpuTexDesc removed — replaced by CUDA texture objects)

// ── Launch parameters ───────────────────────────────────────────────

struct LaunchParams {

    // ── §0  Output buffers ───────────────────────────────────────────
    // SoA layout: one float array per channel for coalesced GPU access.
    float*    color_r;          // [width * height] R accumulator
    float*    color_g;          // [width * height] G accumulator
    float*    color_b;          // [width * height] B accumulator
    float*    sample_counts;    // [width * height]
    uint8_t*  srgb_buffer;      // [width * height * 4] display RGBA8

    // AOV guide layers for denoiser (written at first non-specular hit)
    float*    albedo_buffer;    // [width * height * 4] (nullable)
    float*    normal_buffer;    // [width * height * 4] (nullable)

    int width;
    int height;

    // ── §1  Camera ───────────────────────────────────────────────────
    float3 cam_pos;
    float3 cam_u;
    float3 cam_v;
    float3 cam_w;
    float  cam_lens_radius;     // >0 enables thin-lens DOF
    float  cam_focus_dist;

    // ── §2  Rendering parameters ─────────────────────────────────────
    int       samples_per_pixel;
    int       max_bounces;
    int       min_bounces_rr;       // guaranteed bounces before RR
    float     rr_threshold;         // max RR survival probability
    int       frame_number;
    RenderMode render_mode;
    float     exposure;

    // ── §2a  Clamping (runtime-tunable via JSON config) ──────────────
    int       clamping_enabled;         // master gate (0 = bypass all clamps)
    float     max_bounce_contribution;  // per-bounce f*cos/pdf clamp
    float     max_path_throughput;      // cumulative throughput clamp
    float     max_nee_contribution;     // per-bounce NEE clamp
    float     max_sample_luminance;     // final per-sample luminance clamp

    // ── §3  Scene geometry (device pointers) ─────────────────────────
    float3*   vertices;         // [num_triangles * 3]
    float3*   normals;          // [num_triangles * 3]
    float3*   tangents;         // [num_triangles * 3] (nullable)
    float2*   texcoords;        // [num_triangles * 3]
    uint32_t* material_ids;     // [num_triangles]
    int       num_triangles;

    // ── §4  Material data (device pointers, RGB) ─────────────────────
    int       num_materials;
    float*    Kd;               // [num_materials * 3] RGB diffuse
    float*    Ks;               // [num_materials * 3] RGB specular
    float*    Le;               // [num_materials * 3] RGB emission
    float*    Tf;               // [num_materials * 3] RGB transmittance
    float*    roughness;        // [num_materials]
    float*    ior;              // [num_materials]
    uint8_t*  mat_type;         // [num_materials]
    int*      diffuse_tex;      // [num_materials] texture ID or -1
    int*      specular_tex;     // [num_materials] texture ID or -1
    int*      emission_tex;     // [num_materials] texture ID or -1
    int*      bump_tex;         // [num_materials] texture ID or -1 (height map)
    int*      normal_tex;       // [num_materials] texture ID or -1 (tangent-space normal)
    int*      alpha_tex;        // [num_materials] texture ID or -1
    int*      displacement_tex; // [num_materials] texture ID or -1 (future)
    float*    displacement_scale; // [num_materials] (future)
    float*    opacity;          // [num_materials] 0..1
    uint8_t*  mat_thin;         // [num_materials] 1 = thin dielectric
    float*    cauchy_A;          // [num_materials] Cauchy dispersion A coefficient
    float*    cauchy_B;          // [num_materials] Cauchy dispersion B coefficient (nm²)

    // Extended PBR material fields
    float*    clearcoat_weight;    // [num_materials]
    float*    clearcoat_roughness; // [num_materials]
    float*    sheen;               // [num_materials]
    float*    sheen_tint;          // [num_materials]

    // ── §5  Texture atlas (CUDA texture objects) ───────────────────────
    cudaTextureObject_t* textures;    // [num_textures] hardware-sampled
    int                  num_textures;

    // ── §6  Medium data ──────────────────────────────────────────────
    // (Added by material stage — Phase 4)
    int*       mat_medium_id;   // [num_materials] medium index or -1 (nullable)
    // HomogeneousMedium* media; // deferred until Phase 4
    int        num_media;

    // ── §7  Emitter data ─────────────────────────────────────────────
    uint32_t* emissive_tri_indices;  // [num_emissive]
    float*    emissive_cdf;          // [num_emissive]
    int*      emissive_local_idx;    // [num_triangles] → local emissive index (-1 = not)
    int       num_emissive;
    float     total_emissive_power;

    // ── §7a  Light tree (importance-driven NEE) ──────────────────────
    // Flat BVH over emissive triangles for spatial importance sampling.
    // When light_tree_nodes != nullptr, NEE uses tree traversal instead
    // of the global power-weighted CDF.
    LightTreeNode* light_tree_nodes;      // [num_light_tree_nodes] (nullable)
    uint32_t*      light_tree_tri_order;  // [num_emissive] reordered tri indices
    int            num_light_tree_nodes;
    int            light_tree_root;       // root node index
    int            light_tree_enabled;    // 0 = use CDF, 1 = use tree

    // ── §9  Caustic light tracing ──────────────────────────────────────
    // Separate SoA buffers for caustic splatting (additive, not averaged).
    // Composited with color buffers at PostFx normalization time.
    float*    caustic_r;            // [width * height] (nullable)
    float*    caustic_g;            // [width * height] (nullable)
    float*    caustic_b;            // [width * height] (nullable)
    int       caustic_num_photons;  // photons to emit this frame
    int       caustic_frame_number; // for RNG decorrelation
    float     caustic_max_splat_luminance; // per-splat clamp

    // Delta surface distribution (area-weighted CDF over mirror/glass/translucent tris)
    uint32_t* delta_tri_indices;    // [num_delta_tris] (nullable)
    float*    delta_cdf;            // [num_delta_tris] (nullable)
    int       num_delta_tris;
    float     delta_total_area;     // total area of all delta surfaces (for PDF)

    // ── §11 Adaptive sampling ────────────────────────────────────────
    float*   lum_sum;           // [width * height] (nullable)
    float*   lum_sum2;          // [width * height] (nullable)
    uint8_t* active_mask;       // [width * height] (nullable)

    // ── §13 Pre-pass diagnostics ─────────────────────────────────────
    // Global atomic counters for view-dependent analysis pre-pass.
    // All nullable — zero overhead when prepass_active == 0.
    unsigned int* prepass_nee_attempts;  // [1] atomic (nullable)
    unsigned int* prepass_nee_hits;      // [1] atomic (nullable)
    unsigned int* prepass_zero_paths;    // [1] atomic (nullable)
    unsigned int* prepass_bounce_sum;    // [1] atomic (nullable)
    unsigned int* prepass_total_paths;   // [1] atomic (nullable)
    int           prepass_active;        // 0 = normal, 1 = collect counters

    // ── §12 OptiX traversable ────────────────────────────────────────
#ifdef PPT_USE_OPTIX
    OptixTraversableHandle traversable;
#endif
    int has_instances;          // 1 = IAS active, apply object→world xform
};
