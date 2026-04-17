#pragma once
// ─────────────────────────────────────────────────────────────────────
// scene_profile.h – Scene classification for convergence-driven tuning
//
// Produced by analyze_scene() after loading.
// Consumed by ALL downstream stages + ppt_analyze JSON output.
// ─────────────────────────────────────────────────────────────────────

// ── Lighting type classification ────────────────────────────────────
enum class LightingType {
    LargeArea,    // large emissive surfaces (Cornell box, diffuse panels)
    SmallPoint,   // small point/spot lights, many-lights scenes
    Mixed         // combination of the above
};

// ── Emitter size distribution ───────────────────────────────────────
enum class EmitterDistribution {
    Uniform,      // emitters roughly same size
    HighVariance, // mix of large and tiny emitters → need photon balancing
    SingleDominant// one emitter dominates total power
};

// ── Geometric complexity tier ───────────────────────────────────────
enum class GeometryComplexity {
    Simple,    // < 10k triangles
    Moderate,  // 10k–100k
    Complex,   // 100k–1M
    Dense      // > 1M triangles
};

// ── Scene profile ───────────────────────────────────────────────────
// Classifies a scene's characteristics at load time.
// Every pipeline stage reads it to select convergence-optimal defaults.
struct SceneProfile {
    // ── Lighting classification ─────────────────────────────────────
    LightingType        dominant_lighting     = LightingType::Mixed;
    EmitterDistribution emitter_distribution  = EmitterDistribution::Uniform;
    bool                has_caustic_paths     = false;
    int                 num_emitters          = 0;
    float               emitter_size_ratio    = 0.f;
    bool                mostly_indirect_emitters = false; // lights mostly occluded from surfaces
    float               emitter_direct_visibility = 1.f;  // fraction of shadow rays that reach emitters

    // ── Geometric complexity ────────────────────────────────────────
    GeometryComplexity  geometry_complexity   = GeometryComplexity::Simple;
    int                 num_triangles         = 0;
    int                 num_instances         = 0;
    float               scene_diagonal        = 1.f;
    bool                has_open_geometry     = false;

    // ── Material complexity ─────────────────────────────────────────
    bool                has_glass             = false;
    bool                has_metal             = false;
    bool                has_translucent       = false;
    bool                has_clearcoat         = false;
    float               avg_roughness         = 0.5f;
    int                 num_material_types    = 0;
    int                 num_delta_materials   = 0; // mirror + glass count
    int                 num_delta_triangles   = 0; // total delta surface triangles
    float               delta_area_fraction   = 0.f; // delta surface area / total scene area
    bool                caustic_geometry_favorable = false; // delta objects near emitters

    // ── Emitter flux & caustic coupling ─────────────────────────────
    float               total_emissive_flux   = 0.f; // sum of luminance(Le) × area
    float               max_emitter_radiance  = 0.f; // peak emitter Le luminance
    float               emitter_delta_coupling = 0.f; // solid angle fraction of delta surfaces [0,1]
    float               caustic_difficulty    = 1.f;  // budget multiplier (higher = harder caustics)

    // ── Derived convergence hints ───────────────────────────────────
    int   recommended_max_bounces             = 8;
    int   recommended_photon_budget           = 1000000;
    int   recommended_caustic_photon_budget   = 0;
    int   recommended_guide_training_iters    = 10;
    float recommended_guide_fraction          = 0.5f;
    float recommended_gather_radius           = -1.f; // -1 = use default
    float recommended_caustic_radius          = -1.f;
    bool  recommended_caustic_enabled         = false;
    int   recommended_caustic_photons_per_frame = 0;
    float recommended_caustic_max_splat_luminance = 0.f; // 0 = use default
    bool  recommended_light_tree_enabled       = true;
};
