#pragma once
// ─────────────────────────────────────────────────────────────────────
// config.h – Central configuration for the photon-centric renderer v2.1
// ─────────────────────────────────────────────────────────────────────
// Architecture (doc/architecture/revised_guideline_v2.md):
//   Camera rays: first-hit only → NEE + photon gather (§2, §7).
//   Photon rays: full path tracing from lights (§5).
//   Gather kernel: tangential disk on surface, not 3D sphere (§6.3).
//   Spatial index: KD-tree (CPU), hash grid (GPU) (§6.1–6.2).
// ─────────────────────────────────────────────────────────────────────
// Only scene-level and algorithm-level tunables belong here.
// Low-level implementation constants (OptiX pipeline sizes, hash primes,
// KD-tree leaf size, etc.) belong in the owning source files.
// ─────────────────────────────────────────────────────────────────────

// =====================================================================
// §0  SCENE SELECTION
// =====================================================================
// Uncomment exactly ONE to set the compile-time active scene.
// Runtime switching via keys 1–9 uses SCENE_PROFILES[] below.

//#define SCENE_CORNELL_BOX
#define SCENE_CONFERENCE
//#define SCENE_LIVING_ROOM
//#define SCENE_SIBENIK

// =====================================================================
// §1  OUTPUT RESOLUTION & SAMPLING
// =====================================================================

// Image dimensions in pixels.
//   Preview: 512×512  |  Default: 1024×768  |  Final: 1920×1080+
constexpr int DEFAULT_IMAGE_WIDTH  = 1024;
constexpr int DEFAULT_IMAGE_HEIGHT = 768;

// Samples per pixel (anti-aliasing + noise averaging).
//   Preview: 1–4  |  Default: 16–64  |  Final: 64–256
constexpr int DEFAULT_SPP = 32;

// Stratified sub-pixel jitter grid (§7.1).
// Constraint: STRATA_X × STRATA_Y == DEFAULT_SPP (one sample per stratum).
//   4×8 = 32 matches DEFAULT_SPP = 32.
constexpr int STRATA_X = 4;
constexpr int STRATA_Y = 8;

// =====================================================================
// §2  PHOTON PASS — BUDGETS & TRANSPORT (§5)
// =====================================================================

// ── Multi-hero wavelength transport (PBRT v4 §14.3) ────────────────
// Each photon carries HERO_WAVELENGTHS spectral bins with stratified
// offsets.  The density estimator normalises by 1/HERO_WAVELENGTHS
// so that each physical photon counts once, not once per hero bin.
//   1 = mono-hero (legacy)  |  4 = recommended (PBRT v4)
constexpr int HERO_WAVELENGTHS = 4;

// ── Photon budgets (§Q4) ────────────────────────────────────────────
// Total photons emitted from lights per pass.  Higher = less noise,
// longer precomputation.  The photon map carries ALL multi-bounce
// indirect transport in v2.
//   Preview: 50k–200k  |  Default: 500k–1M  |  Final: 1M–5M
constexpr int DEFAULT_GLOBAL_PHOTON_BUDGET  = 5000000;  // diffuse indirect photons
constexpr int DEFAULT_CAUSTIC_PHOTON_BUDGET = 1000000;   // specular→diffuse caustic photons

// ── Photon path depth (§5.2) ────────────────────────────────────────
// Maximum bounce depth for photon rays (the real path tracers in v2).
// Camera rays do NOT use this — see DEFAULT_MAX_SPECULAR_CHAIN.
//   Preview: 3–4  |  Default: 4  |  High-quality: 6–8
constexpr int DEFAULT_PHOTON_MAX_BOUNCES = 4;

// ── Russian roulette (§5.2.2) ───────────────────────────────────────
// After MIN_BOUNCES_RR bounces, each photon terminates with probability
// (1 − min(max_spectrum(throughput), RR_THRESHOLD)).  Throughput is
// divided by the survival probability to keep the estimator unbiased.
//   min_bounces: 2–3  |  threshold: 0.80–0.90
constexpr int   DEFAULT_PHOTON_MIN_BOUNCES_RR = 2;
constexpr float DEFAULT_PHOTON_RR_THRESHOLD   = 0.85f;

// ── Photon emission mixture (§5.1, variance reduction) ──────────────
// Fraction of photons emitted with area-uniform (rather than power-
// weighted) triangle selection.  Reduces variance when emissive
// surfaces have very different brightness.
//   0.0 = pure power-weighted  |  0.10 = recommended  |  1.0 = pure area
constexpr float DEFAULT_PHOTON_EMITTER_UNIFORM_MIX = 0.10f;

// ── Photon bounce decorrelation (§5.3.1) ────────────────────────────
// Hemisphere strata for cell-stratified bouncing.  Successive photons
// in the same grid cell bounce into different strata, avoiding
// redundant coverage of the same directions.
//   0 = disable (pure random BSDF)  |  32–64 = recommended
constexpr int DEFAULT_PHOTON_BOUNCE_STRATA = 64;

// ── Photon emission cone angle (§5.1.2) ─────────────────────────────
// Half-angle of the cosine-weighted emission cone (degrees).
// Controls ONLY the initial photon emission from light sources;
// subsequent bounces are unrestricted (material-dependent BSDF).
// 90° = full hemisphere (Lambertian).  60° = 120° FOV (directional,
// e.g. sunlight through windows).  Shared by CPU emitter.h
// and GPU __raygen__photon_trace.
constexpr float DEFAULT_LIGHT_CONE_HALF_ANGLE_DEG = 90.0f;

// Debug: stop photon after first intersection (validate emission only).
constexpr bool DEBUG_PHOTON_SINGLE_BOUNCE = false;

// =====================================================================
// §3  SPATIAL INDEX & GATHER KERNEL (§6)
// =====================================================================

// ── Gather radius (§3.4, §6.6) ──────────────────────────────────────
// Radius of the tangential disk kernel (§6.3) for photon density
// estimation.  Smaller = sharper but noisier; larger = smoother but
// more bias.  Both maps use the tangential (surface) distance metric.
// Scene is normalised to [-0.5, 0.5]³ (SCENE_REF_EXTENT = 1.0):
//   10% of scene = 0.10  |  5% = 0.05  |  Final quality: 0.02–0.05
constexpr float DEFAULT_GATHER_RADIUS  = 0.05f;   // global (diffuse) map — 10% of scene extent
constexpr float DEFAULT_CAUSTIC_RADIUS = 0.05f;   // caustic map — 5% of scene extent

// ── Dense cell-bin grid gather (§3.5, §6.7) ─────────────────────────
// When true, the GPU gather uses the precomputed dense 3D cell-bin grid
// (CellBinGrid) instead of the per-photon hash-grid walk.  The cell-bin
// grid accumulates photon flux into PHOTON_BIN_COUNT directional bins per
// spatial cell using the full tangential-disk kernel (§7.1) baked at
// CPU build time.  GPU query is O(PHOTON_BIN_COUNT) instead of O(N_cell).
//
// The tangential plane projection (§7.1 guideline) is preserved:
//   Build: Epanechnikov weight max(0, 1 - d_tan²/r²) relative to cell
//          centre + surface tau filter (§6.4) applied per photon.
//   Query: per-bin normal gate (avg_n · filter_normal > 0) + hemisphere
//          gate (bin_dir · filter_normal > 0).
//
// Expected speedup: ~200× for the gather kernel (see §G12 analysis).
// Mild approximation: indirect spectral distribution is flattened
// (hero-wavelength flux summed to scalar; BSDF provides surface colour).
// Toggle at runtime with G key.
//   true = dense grid (default)  |  false = per-photon hash-grid walk
constexpr bool DEFAULT_USE_DENSE_GRID = true;

// ── k-NN adaptive radius (§C2, §6.5) ────────────────────────────────
// In k-NN gather mode, the radius adapts to local photon density:
//   radius = tangential distance to the k-th nearest photon.
// CPU: KD-tree k-NN.  GPU: grid shell-expansion k-NN (§6.5).
//   Typical: 50–200  |  Default: 100
constexpr int DEFAULT_KNN_K = 100;

// ── GPU grid maximum gather radius (§6.5) ───────────────────────────
// Upper bound for GPU hash-grid shell expansion.  In sparse regions
// k-NN accepts fewer than k photons rather than exceeding this limit.
// CPU KD-tree is unbounded.  Must be ≥ DEFAULT_GATHER_RADIUS.
//   Typical: 0.1–1.0  |  Default: 0.5
constexpr float DEFAULT_GPU_MAX_GATHER_RADIUS = 0.5f;

// ── Surface consistency filter (§6.3, §6.4) ─────────────────────────
// Plane-distance threshold τ: photons farther than τ from the query
// tangent plane are rejected.  Prevents cross-surface leakage (walls,
// floors, nearby geometry).
// Effective τ at runtime: max(surface_tau, PLANE_TAU_EPSILON_FACTOR × ray_eps).
//   Typical: 0.01–0.05  |  Default: 0.02
constexpr float DEFAULT_SURFACE_TAU = 0.02f;

// Robust τ floor multiplier (§6.3).  Ensures τ is never smaller than
// the ray-offset epsilon, preventing rejection of photons deposited on
// the exact same surface due to floating-point offset.
//   Typical: 5–20  |  Default: 10
constexpr float PLANE_TAU_EPSILON_FACTOR = 10.0f;

// =====================================================================
// §4  CAMERA PASS & DIRECT LIGHTING (§7)
// =====================================================================

// ── Camera specular chain limit (§E3, §7.1) ─────────────────────────
// Camera ray follows specular (mirror/glass) bounces until hitting a
// diffuse surface, then evaluates NEE + photon gather.  Only specular
// bounces are allowed — no diffuse continuation.
//   Typical: 4–16  |  Default: 8
constexpr int DEFAULT_MAX_SPECULAR_CHAIN = 8;

// ── Glossy BSDF continuation bounces (§7.1.1) ──────────────────────
// After the first non-specular hit, glossy surfaces can trace
// additional BSDF-sampled reflection bounces to capture scene
// reflections (not just specular highlights of light sources).
//   0 = no glossy continuation  |  2–3 = recommended  |  5+ = expensive
constexpr int DEFAULT_MAX_GLOSSY_BOUNCES = 2;

// ── NEE shadow ray count (§7.2) ─────────────────────────────────────
// Shadow rays cast to light sources at the camera first-hit.
// More = softer shadows, less noise.  First-hit only in v2.
//   Preview: 4–8  |  Default: 16–64  |  Final: 64+
constexpr int DEFAULT_NEE_LIGHT_SAMPLES = 64;

// ── NEE coverage-aware sampling (§7.2.1) ────────────────────────────
// Mixture weight between power-weighted and area-weighted emitter
// selection for shadow rays.  Prevents large dim emitters from being
// undersampled while still concentrating rays on bright sources.
//   0.0 = pure power-weighted  |  0.3 = recommended  |  1.0 = pure area
constexpr float DEFAULT_NEE_COVERAGE_FRACTION = 0.3f;

// ── Adaptive sampling noise metric ──────────────────────────────────
// When true, adaptive sampling mask uses NEE direct-only variance
// (stable signal for the first-hit architecture).  When false, uses
// full combined path radiance (legacy behavior).
constexpr bool ADAPTIVE_NOISE_USE_DIRECT_ONLY = false;

// ── MIS: NEE vs BSDF continuation (§7.3 — 2-way power heuristic) ───
// Applies balance between NEE shadow rays and stochastic BSDF-sampled
// rays that coincidentally hit a light (glossy continuation bounces).
// Without MIS, both paths add their full contribution → firefly spikes
// on metallic/glossy surfaces.  With MIS, the power heuristic weights
// each estimator by its squared PDF, removing double counting.
//
// GPU: applied in dev_nee_direct() and dev_guided_nee() for every
//      glossy BSDF continuation bounce (DEFAULT_MAX_GLOSSY_BOUNCES > 0).
// CPU: pure mirror reflection has delta-BSDF → MIS weight = 1.0 always,
//      so this flag has no numerical effect on the CPU path currently.
//      It will matter when stochastic BSDF sampling is added to CPU.
//   false = no MIS (double-counts emitters on glossy surfaces)
//   true  = power heuristic MIS (eliminates glossy fireflies)
constexpr bool DEFAULT_USE_MIS = true;

// =====================================================================
// §5  TONE MAPPING (§Q8)
// =====================================================================

// ACES Filmic tone mapping replaces Reinhard (§Q8).
// Better highlight rolloff and color preservation.
//   true = ACES Filmic  |  false = Reinhard (legacy)
constexpr bool USE_ACES_TONEMAPPING = true;

// Exposure multiplier applied to linear XYZ before tone mapping.
// Used by both GPU dev_spectrum_to_srgb() and CPU renderer tonemap().
//   0.5 = darker  |  1.0 = neutral  |  2.0 = brighter
constexpr float DEFAULT_EXPOSURE = 1.0f;

// ── Multi-map photon re-tracing (§1.2) ────────────────────────────
// Re-trace the photon map with a new RNG seed every N samples to
// decorrelate photon noise from camera noise.  0 = single map.
// With DEFAULT_SPP=32: group=4 → 8 unique maps; group=8 → 4 maps.
//   0 = disabled  |  4 = recommended for 32 spp  |  8 = quality renders
constexpr int MULTI_MAP_SPP_GROUP = 4;

// =====================================================================
// §6  DEPTH OF FIELD (thin-lens camera)
// =====================================================================
// Physically-based DOF via stochastic lens sampling.
// Exposure stays constant regardless of aperture (artist-friendly).

constexpr bool  DEFAULT_DOF_ENABLED        = false;
constexpr float DEFAULT_DOF_FOCUS_DISTANCE = 0.1f;    // distance to focus plane (scene units)
constexpr float DEFAULT_DOF_F_NUMBER       = 8.0f;    // f-stop: lower = shallower DOF.  Range: 1.4–22
constexpr float DEFAULT_DOF_SENSOR_HEIGHT  = 0.024f;  // sensor height in meters (0.024 = 24 mm full-frame)
constexpr float DEFAULT_DOF_FOCUS_RANGE    = 0.2f;    // in-focus slab depth (scene units). 0 = razor-thin

// =====================================================================
// §7  PHOTON MAP PERSISTENCE (§20)
// =====================================================================
// The photon map is a static, view-independent light-field snapshot.
// Saving it to disk enables instant startup on unchanged scenes.
// Scene hash auto-invalidation ensures stale caches are never used.
// P key forces recomputation at runtime (§20.8).

constexpr bool DEFAULT_SAVE_PHOTON_CACHE = true;   // auto-save after tracing
constexpr bool DEFAULT_LOAD_PHOTON_CACHE = true;   // auto-load if scene hash matches

// =====================================================================
// §8  PHOTON DIRECTIONAL BINS (§21 — Phase 7 optimization)
// =====================================================================
// Fixed-size directional bin cache per pixel (Fibonacci sphere layout).
// Used for gather caching and photon-guided bouncing in future passes.
// Not critical for core v2 rendering.
//   Typical: 16–64  |  Default: 32

constexpr int PHOTON_BIN_COUNT     = 32;   // directional bins per pixel
constexpr int MAX_PHOTON_BIN_COUNT = 32;   // compile-time upper bound (≥ PHOTON_BIN_COUNT)

// =====================================================================
// §9  LIGHTING & SCENE NORMALIZATION
// =====================================================================

// Runtime-adjustable light intensity multiplier (+/− keys).
// Applied to all emissive materials: L_e_scaled = light_scale × L_e_material (§5.1.1).
constexpr float DEFAULT_LIGHT_SCALE = 1.0f;
constexpr float LIGHT_SCALE_STEP    = 1.25f;   // multiplicative step per +/− key press
constexpr float LIGHT_SCALE_MIN     = 0.01f;
constexpr float LIGHT_SCALE_MAX     = 100.0f;

// Scene normalization: all scenes are scaled to fit within this extent
// centered at the origin, matching the Cornell Box reference frame ([-0.5, 0.5]³).
constexpr float SCENE_REF_EXTENT = 1.0f;

// =====================================================================
// §10  VOLUME RENDERING (§Q9 — TEMPORARILY DISABLED)
// =====================================================================
// Disabled during v2 rewrite.  Re-enable after surface transport is
// validated and tested.  All constants retained for future use.

constexpr bool  DEFAULT_VOLUME_ENABLED  = false;   // master switch (§Q9: disabled in v2)
constexpr float DEFAULT_VOLUME_DENSITY  = 0.15f;   // base extinction σ_t. Range: 0.01–1.0
constexpr float DEFAULT_VOLUME_FALLOFF  = 0.0f;    // height falloff (0 = homogeneous, >0 = exp decay)
constexpr float DEFAULT_VOLUME_ALBEDO   = 0.95f;   // single-scattering albedo σ_s/σ_t. Range: [0, 1]
constexpr int   DEFAULT_VOLUME_SAMPLES  = 2;       // medium samples per ray segment. Range: 1–8
constexpr float DEFAULT_VOLUME_MAX_T    = 2.0f;    // max march distance for miss rays (scene units)

// =====================================================================
// §11  BACKWARD COMPATIBILITY (v1 → v2 migration aliases)
// =====================================================================
// These constants are deprecated in the v2 photon-centric architecture
// but retained because existing code still references them.  They have
// NO effect on v2 rendering.  Remove once all v1 code paths are deleted.

// v1 multi-bounce camera features — all disabled in v2 (§22).
// ── Dead v1 flags (kept for backward compat — no effect in v2.1) ───
// These constants are unused by the rendering pipeline.  They remain
// only to avoid breaking external callers that reference them.
// TODO: Remove when all references have been cleaned from tests.
// DEFAULT_USE_MIS is now defined in §4 (CAMERA PASS) — not a dead flag.
constexpr bool  DEFAULT_USE_PHOTON_GUIDED = false;   // v1 guided camera bounce (removed)
constexpr float DEFAULT_GUIDED_BSDF_MIX   = 0.80f;  // v1 guided BSDF mix       (removed)
constexpr int   DEFAULT_NEE_DEEP_SAMPLES  = 4;       // v1 deep NEE samples       (removed)

// Photon-prefixed names → legacy aliases used across codebase.
constexpr int   DEFAULT_MAX_BOUNCES    = DEFAULT_PHOTON_MAX_BOUNCES;
constexpr int   DEFAULT_MIN_BOUNCES_RR = DEFAULT_PHOTON_MIN_BOUNCES_RR;
constexpr float DEFAULT_RR_THRESHOLD   = DEFAULT_PHOTON_RR_THRESHOLD;
constexpr int   DEFAULT_NUM_PHOTONS    = DEFAULT_GLOBAL_PHOTON_BUDGET;

// =====================================================================
// §12  SCENE PROFILES (runtime switching via keys 1–9)
// =====================================================================
// Each profile defines: OBJ path (relative to SCENES_DIR), camera
// position/look-at/FOV, and move speed suited to the scene scale.

// After normalisation every scene lives in approximately [-0.5, 0.5]³,
// so all camera/speed profiles now share the same coordinate system.

#ifdef SCENE_CORNELL_BOX
  constexpr const char* SCENE_OBJ_PATH        = "cornell_box/cornellbox.obj";
  constexpr const char* SCENE_DISPLAY_NAME     = "Cornell Box";
  constexpr bool  SCENE_IS_REFERENCE           = true;   // already in ref frame
  constexpr float SCENE_CAM_POS[]              = { 0.0f,  0.0f,  0.0f };
  constexpr float SCENE_CAM_LOOKAT[]           = { 0.0f,  0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV                = 40.0f;
  constexpr float SCENE_CAM_SPEED              = 0.5f;

#elif defined(SCENE_CONFERENCE)
  constexpr const char* SCENE_OBJ_PATH        = "conference/conference.obj";
  constexpr const char* SCENE_DISPLAY_NAME     = "Conference Room";
  constexpr bool  SCENE_IS_REFERENCE           = false;
  constexpr float SCENE_CAM_POS[]              = { 0.0f,  0.0f,  0.0f };
  constexpr float SCENE_CAM_LOOKAT[]           = { 0.0f,  0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV                = 50.0f;
  constexpr float SCENE_CAM_SPEED              = 0.5f;

#elif defined(SCENE_LIVING_ROOM)
  constexpr const char* SCENE_OBJ_PATH        = "living_room/living_room.obj";
  constexpr const char* SCENE_DISPLAY_NAME     = "Living Room";
  constexpr bool  SCENE_IS_REFERENCE           = false;
  constexpr float SCENE_CAM_POS[]              = { 0.0f,  0.0f,  0.0f };
  constexpr float SCENE_CAM_LOOKAT[]           = { 0.0f,  0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV                = 50.0f;
  constexpr float SCENE_CAM_SPEED              = 0.5f;

#elif defined(SCENE_SIBENIK)
  constexpr const char* SCENE_OBJ_PATH        = "sibenik/sibnek.obj";
  constexpr const char* SCENE_DISPLAY_NAME     = "Sibenik Cathedral";
  constexpr bool  SCENE_IS_REFERENCE           = false;
  constexpr float SCENE_CAM_POS[]              = { 0.0f,  0.0f,  0.0f };
  constexpr float SCENE_CAM_LOOKAT[]           = { 0.0f,  0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV                = 50.0f;
  constexpr float SCENE_CAM_SPEED              = 0.5f;

#else
  #error "No scene selected! Uncomment one SCENE_* define in config.h"
#endif

// ── Scene lighting mode ──────────────────────────────────────────────
// Describes how light enters the scene (from scenes_description.md).
enum class SceneLightMode {
    FromMTL,            // Emissive surfaces defined in .mtl file
    DirectionalToFloor, // Directional light aimed at center floor (e.g. Sibenik)
    HemisphereEnv,      // Upper hemisphere sky dome (e.g. Sponza open atrium)
    SphericalEnv,       // Full sphere environment light (e.g. Hairball)
};

// ── Scene complexity level ──────────────────────────────────────────
// Used to auto-select photon budget, gather radius, spp, etc.
enum class SceneComplexity {
    Low,     // e.g. Cornell Box — fast to render
    Medium,  // e.g. Conference, Living Room — moderate cost
    High,    // e.g. Sibenik, Sponza, Hairball — expensive
};

// ── Complexity-based rendering presets ──────────────────────────────
struct ComplexityPreset {
    int   global_photon_budget;
    int   caustic_photon_budget;
    float gather_radius;
    float caustic_radius;
    int   spp;
    int   photon_max_bounces;
    int   nee_light_samples;
};

// Indexed by SceneComplexity enum value
constexpr ComplexityPreset COMPLEXITY_PRESETS[3] = {
    // Low:    small scenes, fast convergence
    { 500000,  100000, 0.10f, 0.05f, 64, 4, 64 },
    // Medium: moderate geometry, balanced quality
    { 1000000, 250000, 0.10f, 0.05f, 64, 4, 64 },
    // High:   complex geometry, high photon budget for coverage
    { 2000000, 500000, 0.10f, 0.05f, 32, 6, 32 },
};

// ── Runtime scene profile (for hotkey scene switching) ──────────────
struct SceneProfile {
    const char*     obj_path;       // relative to SCENES_DIR
    const char*     display_name;
    bool            is_reference;
    float           cam_pos[3];
    float           cam_lookat[3];
    float           cam_fov;
    float           cam_speed;
    SceneLightMode  light_mode;     // how light enters the scene
    SceneComplexity complexity;     // auto-selects render presets

    // Convenience: get the complexity-based preset
    constexpr const ComplexityPreset& preset() const {
        return COMPLEXITY_PRESETS[static_cast<int>(complexity)];
    }
};

constexpr int NUM_SCENE_PROFILES = 8;

// Keys 1–8 map to indices 0–7 (from scenes_description.md)
constexpr SceneProfile SCENE_PROFILES[NUM_SCENE_PROFILES] = {
    // 1: Cornell Box — low complexity, emitters in MTL
    { "cornell_box/cornellbox.obj",              "Cornell Box",       true,
      { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -1.0f }, 40.0f, 0.5f,
      SceneLightMode::FromMTL, SceneComplexity::Low },

    // 2: Conference Room — medium complexity, emitters in MTL
    { "conference/conference.obj",               "Conference Room",   false,
      { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -1.0f }, 50.0f, 0.5f,
      SceneLightMode::FromMTL, SceneComplexity::Medium },

    // 3: Living Room — medium complexity, emitters in MTL
    { "living_room/living_room.obj",             "Living Room",       false,
      { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -1.0f }, 50.0f, 0.5f,
      SceneLightMode::FromMTL, SceneComplexity::Medium },

    // 4: Fireplace Room — medium complexity, emitters in MTL
    { "fireplace_room/fireplace_room.obj",       "Fireplace Room",    false,
      { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -1.0f }, 50.0f, 0.5f,
      SceneLightMode::FromMTL, SceneComplexity::Medium },

    // 5: Salle de Bain — medium complexity, emitters in MTL
    { "salle_de_bain/salle_de_bain.obj",         "Salle de Bain",     false,
      { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -1.0f }, 50.0f, 0.5f,
      SceneLightMode::FromMTL, SceneComplexity::Medium },

    // 6: Sibenik Cathedral — high complexity, directional light to floor
    { "sibenik/sibnek.obj",                      "Sibenik Cathedral",  false,
      { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -1.0f }, 50.0f, 0.5f,
      SceneLightMode::DirectionalToFloor, SceneComplexity::High },

    // 7: Sponza — high complexity, hemisphere sky dome from above
    { "sponza/sponza.obj",                       "Sponza",            false,
      { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -1.0f }, 60.0f, 0.5f,
      SceneLightMode::HemisphereEnv, SceneComplexity::High },

    // 8: Hairball — high complexity, spherical environment light
    { "hairball/hairball.obj",                   "Hairball",           false,
      { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -1.0f }, 50.0f, 0.5f,
      SceneLightMode::SphericalEnv, SceneComplexity::High },
};

