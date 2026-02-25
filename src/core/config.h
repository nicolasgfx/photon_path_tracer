#pragma once
#include <cstdint>
// ─────────────────────────────────────────────────────────────────────
// config.h – Central configuration for the photon-centric renderer v2.2
// ─────────────────────────────────────────────────────────────────────
//
//   Architecture: photon-centric (doc/architecture/architecture.md)
//     Camera rays  → first-hit only → NEE + photon gather
//     Photon rays  → full path tracing from lights (the real tracers)
//     Gather       → tangential disk on surface, not 3D sphere
//     Spatial index → KD-tree (CPU), hash grid (GPU)
//
//   Only scene-level and algorithm-level tunables belong here.
//   Low-level constants (OptiX pipeline, hash primes, KD-tree leaf
//   size, etc.) belong in the owning source files.
//
//   Constants marked [R] are overridable at runtime via render_config.json.
//   Constants marked [K] are togglable via keyboard at runtime.
//
//   Ranges: Fast / Balanced / Quality
//   Current defaults target the "Balanced" sweet spot.
//
// ─────────────────────────────────────────────────────────────────────


// =====================================================================
//  §0  SCENE SELECTION
// =====================================================================
// Uncomment exactly ONE.  Runtime switching via keys 1–8 uses
// SCENE_PROFILES[] at the bottom of this file.

#define SCENE_CORNELL_BOX
//#define SCENE_CONFERENCE
//#define SCENE_LIVING_ROOM
//#define SCENE_SIBENIK

// ── Photon tracing backend ──────────────────────────────────────────
// Set to 1 to use the CPU photon tracer (emitter.h) instead of the GPU
// OptiX kernels.  Useful for A/B comparison; GPU rendering (camera rays,
// gather, etc.) still runs on the GPU — only photon emission is on CPU.
#define USE_CPU_PHOTON_TRACE 0

// ── v2.2 Consistency Reset Flags ────────────────────────────────────
// GPU photon tracing: disabled by default (CPU is ground truth until
// a fully equivalent OptiX photon tracer exists).
constexpr bool DEFAULT_USE_GPU_PHOTON_TRACING = false;

// Bresenham per-pixel lobe balance (GPU BSDF heuristic):
// disabled by default for CPU↔GPU consistency.  Enable only after
// CPU has the same mechanism and equivalence tests pass.
constexpr bool DEFAULT_ENABLE_BRESENHAM_BSDF = false;

// EmitterPointSet primary emission path: disabled by default.
// v2.2 uses alias-table + cosine hemisphere only, for consistency.
constexpr bool DEFAULT_USE_EMITTER_POINT_SET = false;


// =====================================================================
//  §1  IMAGE OUTPUT
// =====================================================================

constexpr int DEFAULT_IMAGE_WIDTH  = 512;           // [R]
constexpr int DEFAULT_IMAGE_HEIGHT = 512;           // [R]


// =====================================================================
//  §2  CORE RENDERING
// =====================================================================
// The parameters that most directly control output quality and speed.
// Adjust these first when tuning a render.

// ── Samples per pixel ───────────────────────────────────────────────
// Anti-aliasing + noise averaging.  This is the single biggest
// quality/speed knob.
//   Fast: 4–8  |  Balanced: 16  |  Quality: 32–64  |  Final: 128–256
constexpr int DEFAULT_SPP = 4;                       // [R]

// Sub-pixel stratified jitter grid.
// Constraint: STRATA_X × STRATA_Y == DEFAULT_SPP.
constexpr int STRATA_X = 2;                           // 4 × 4 = 16 = DEFAULT_SPP
constexpr int STRATA_Y = 2;

// ── Photon budgets ──────────────────────────────────────────────────
// Total photons emitted per pass.  The photon map carries ALL indirect
// transport in the v2 architecture.
//   Fast: 100k  |  Balanced: 500k–1M  |  Quality: 2M–5M
constexpr int DEFAULT_GLOBAL_PHOTON_BUDGET  = 1000000;   // [R]  diffuse indirect
constexpr int DEFAULT_CAUSTIC_PHOTON_BUDGET = 750000;   // [R]  specular→diffuse caustics

// ── Gather radii ────────────────────────────────────────────────────
// Tangential disk kernel radius for photon density estimation.
// Smaller = sharper but noisier;  larger = smoother but more bias.
// Values are fractions of SCENE_REF_EXTENT (scene in [-0.5, 0.5]³).
//   Fast: 0.08–0.10  |  Balanced: 0.05  |  Quality: 0.02–0.03
constexpr float DEFAULT_GATHER_RADIUS  = 0.05f;      // 0.05[R]  global (diffuse) map
constexpr float DEFAULT_CAUSTIC_RADIUS = 0.025f;     // 0.025[R]  caustic map (tighter for sharp caustics)

// ── NEE shadow rays ─────────────────────────────────────────────────
// Shadow rays per shading point (bounce 0).  The bin/cache system
// improves *which* emitters are chosen, but M still controls how
// many shadow rays are cast.  See DEFAULT_NEE_DEEP_SAMPLES for
// bounces ≥ 1.
//   Fast: 4–8  |  Balanced: 16  |  Quality: 32–64
constexpr int DEFAULT_NEE_LIGHT_SAMPLES = 4;          // [R]


// =====================================================================
//  §3  PHOTON TRANSPORT
// =====================================================================
// Controls how photon rays bounce through the scene.

// Maximum bounce depth for photon rays.
// Camera rays do NOT use this — see DEFAULT_MAX_SPECULAR_CHAIN.
// Must be high enough for multiple glass layers (2 bounces each for
// enter+exit) plus subsequent diffuse bounces.  4 glass layers = 8
// transmission bounces before reaching a diffuse surface.
//   Fast: 4–6  |  Balanced: 10  |  Quality: 12–16
constexpr int DEFAULT_PHOTON_MAX_BOUNCES = 12;        // [R]

// ── Russian roulette ────────────────────────────────────────────────
// After MIN_BOUNCES_RR guaranteed bounces, each continuation is
// probabilistic: survive = min(max_spectrum(throughput), RR_THRESHOLD).
// Throughput is divided by survival probability (unbiased).
// Set high to guarantee photon survival through glass stacks — RR
// should not kill photons mid-transmission through nested dielectrics.
//   MIN_BOUNCES_RR — Fast: 3–4  |  Balanced: 8  |  Quality: 10
//   RR_THRESHOLD   — 0.80 (aggressive) .. 0.95 (conservative)
constexpr int   DEFAULT_PHOTON_MIN_BOUNCES_RR = 10;    // [R]
constexpr float DEFAULT_PHOTON_RR_THRESHOLD   = 0.90f;// [R]

// ── Spectral transport (PBRT v4 §14.3) ─────────────────────────────
// Hero wavelengths per photon.  4 = PBRT v4 recommended.
// Changing this requires full rebuild — it's wired into the photon struct.
constexpr int HERO_WAVELENGTHS = 4;

// ── Emission cone angle ─────────────────────────────────────────────
// Half-angle (degrees) of cosine-weighted emission from light sources.
// 90° = full hemisphere (Lambertian).  Smaller = directional emitters.
constexpr float DEFAULT_LIGHT_CONE_HALF_ANGLE_DEG = 90.0f;

// ── Emission variance reduction ─────────────────────────────────────
// Fraction of photons emitted with area-uniform (vs power-weighted)
// triangle selection.  Helps when emitters have wildly different power.
//   0.0 = pure power  |  0.10 = balanced  |  1.0 = pure area
constexpr float DEFAULT_PHOTON_EMITTER_UNIFORM_MIX = 0.10f;

// Hemisphere strata for cell-stratified bounce decorrelation.
//   0 = disable (pure random BSDF)  |  32–64 = recommended
constexpr int DEFAULT_PHOTON_BOUNCE_STRATA = 64;

// Multi-map photon re-tracing: re-trace the photon map with a new
// RNG seed every N camera samples to decorrelate photon/camera noise.
//   0 = single map  |  4 = balanced  |  8 = quality
constexpr int MULTI_MAP_SPP_GROUP = 4;


// =====================================================================
//  §4  CAMERA RAYS & DIRECT LIGHTING
// =====================================================================

// ── Specular chain ──────────────────────────────────────────────────
// Camera ray follows mirror/glass bounces until hitting a diffuse
// surface, then evaluates NEE + photon gather.
//   Fast: 4  |  Balanced: 8  |  Quality: 12–16
constexpr int DEFAULT_MAX_SPECULAR_CHAIN = 12;         // [R]

// ── Glossy continuation bounces ─────────────────────────────────────
// After the first non-specular hit, glossy surfaces can trace extra
// BSDF-sampled reflection bounces.
//   0 = off  |  2 = balanced  |  3–4 = quality (expensive)
constexpr int DEFAULT_MAX_GLOSSY_BOUNCES = 2;

// ── NEE emitter selection mix ───────────────────────────────────────
// Power-weighted vs area-weighted emitter selection for shadow rays.
//   0.0 = pure power  |  0.3 = balanced  |  1.0 = pure area
constexpr float DEFAULT_NEE_COVERAGE_FRACTION = 0.3f; // [R]

// ── MIS (power heuristic) ───────────────────────────────────────────
// Balances NEE shadow rays vs BSDF-sampled rays that hit emitters.
// Should always be true — disabling causes double-counting on glossy.
constexpr bool DEFAULT_USE_MIS = true;


// =====================================================================
//  §5  TONE MAPPING & DISPLAY
// =====================================================================

constexpr bool  USE_ACES_TONEMAPPING = true;          // true = ACES Filmic, false = Reinhard
constexpr float DEFAULT_EXPOSURE     = 1.0f;          // [R]  0.5 dark .. 1.0 neutral .. 2.0 bright

// Runtime light intensity multiplier (+/− keys).
constexpr float DEFAULT_LIGHT_SCALE = 1.0f;           // [K]
constexpr float LIGHT_SCALE_STEP    = 1.25f;          //      multiplicative step per key press
constexpr float LIGHT_SCALE_MIN     = 0.01f;
constexpr float LIGHT_SCALE_MAX     = 100.0f;

// Progress snapshot PNGs at power-of-2 SPP intervals (near-zero overhead).
constexpr bool PROGRESS_SNAPSHOT_ENABLED = true;


// =====================================================================
//  §6  SPATIAL INDEX & SURFACE FILTER
// =====================================================================
// Parameters controlling photon gather geometry and the underlying
// spatial acceleration.  Rarely need changing.

// ── Surface consistency filter (τ) ──────────────────────────────────
// Plane-distance threshold: photons farther than τ from the query
// tangent plane are rejected (prevents cross-surface leakage).
//   Tight: 0.01  |  Balanced: 0.02  |  Loose: 0.05
constexpr float DEFAULT_SURFACE_TAU        = 0.02f;
constexpr float PLANE_TAU_EPSILON_FACTOR   = 10.0f;   // robust τ floor = factor × ray_eps

// ── k-NN gather ─────────────────────────────────────────────────────
// Adaptive radius = distance to the k-th nearest photon.
// CPU: KD-tree.  GPU: grid shell expansion (capped by MAX_GATHER_RADIUS).
constexpr int   DEFAULT_KNN_K                = 100;
constexpr float DEFAULT_GPU_MAX_GATHER_RADIUS = 0.5f;  // upper bound for GPU shell expansion

// ── Cell cache (per-cell photon statistics) ─────────────────────────
// Adaptive gather radius, empty-region skip, caustic hotspot detection.
constexpr uint32_t CELL_CACHE_TABLE_SIZE     = 65536u;  // 64K cells

// ── Directional photon bins (Fibonacci sphere) ──────────────────────
constexpr int PHOTON_BIN_COUNT     = 32;    // runtime bin count (quasi-uniform S²)
constexpr int MAX_PHOTON_BIN_COUNT = 64;    // compile-time upper bound for fixed arrays

constexpr float ADAPTIVE_RADIUS_MIN_FACTOR  = 0.25f;   // never shrink below 25% of base
constexpr float ADAPTIVE_RADIUS_MAX_FACTOR  = 2.0f;    // never grow above 200% of base
constexpr float ADAPTIVE_RADIUS_TARGET_K    = 100.f;   // desired photons in gather disk

// ── Adaptive caustic emission ───────────────────────────────────────
constexpr float CAUSTIC_TARGETED_FRACTION   = 0.30f;   // fraction of caustic budget targeted
constexpr int   CAUSTIC_MIN_FOR_ANALYSIS    = 10;      // min photons per cell for CV analysis
constexpr float CAUSTIC_CV_THRESHOLD        = 0.50f;   // CV above this = "hot" cell
constexpr int   CAUSTIC_MAX_TARGETED_ITERS  = 3;       // max adaptive refinement passes

// ── Targeted caustic emission (§11: specular geometry sampling) ─────
// Fraction of caustic budget directed at specular surfaces via
// importance-sampled emission (the remainder uses uniform emission).
//   0.0 = all uniform (disabled)  |  0.5 = balanced  |  0.8 = aggressive
constexpr float DEFAULT_TARGETED_CAUSTIC_MIX = 1.0f;  // 0.7 [R]


// =====================================================================
//  §7  DEPTH OF FIELD (thin-lens camera)
// =====================================================================
// Physically-based DOF via stochastic lens sampling.
// Exposure stays constant regardless of aperture.

constexpr bool  DEFAULT_DOF_ENABLED        = false;    // [R]
constexpr float DEFAULT_DOF_FOCUS_DISTANCE = 0.1f;    // [R]  to focus plane (scene units)
constexpr float DEFAULT_DOF_F_NUMBER       = 8.0f;    // [R]  f-stop (1.4 shallow .. 22 deep)
constexpr float DEFAULT_DOF_SENSOR_HEIGHT  = 0.024f;  // [R]  sensor height in m (0.024 = 24mm)
constexpr float DEFAULT_DOF_FOCUS_RANGE    = 0.05f;   // [R]  fraction of focus dist that stays sharp


// =====================================================================
//  §8  VOLUME RENDERING (disabled — future use)
// =====================================================================
// Plumbing is wired but gated behind the master switch.
// Re-enable after surface transport is fully validated.

constexpr bool  DEFAULT_VOLUME_ENABLED  = false;       // master switch
constexpr float DEFAULT_VOLUME_DENSITY  = 0.15f;       // σ_t extinction (0.01–1.0)
constexpr float DEFAULT_VOLUME_FALLOFF  = 0.0f;        // height falloff (0 = homogeneous)
constexpr float DEFAULT_VOLUME_ALBEDO   = 0.95f;       // σ_s / σ_t (0–1)
constexpr int   DEFAULT_VOLUME_SAMPLES  = 2;           // medium samples per segment (1–8)
constexpr float DEFAULT_VOLUME_MAX_T    = 2.0f;        // max march distance (scene units)


// =====================================================================
//  §9  SCENE NORMALIZATION
// =====================================================================

// All scenes are scaled to fit within [-0.5, 0.5]³.  Gather radii,
// surface tau, and camera speeds are relative to this.
constexpr float SCENE_REF_EXTENT = 1.0f;


// =====================================================================
//  §10  DEBUG
// =====================================================================

constexpr bool DEBUG_PHOTON_SINGLE_BOUNCE = false;     // stop photon after 1st hit
constexpr bool DEBUG_PHOTON_INDIRECT_PNG  = false;     // emit photon-indirect preview PNGs at launch
constexpr bool DEBUG_CAUSTIC_PNG          = false;     // emit caustic-only debug PNGs (needs above)
constexpr bool DEBUG_COVERAGE_PNG         = false;     // emit coverage debug PNGs
constexpr bool ADAPTIVE_NOISE_USE_DIRECT_ONLY = false; // adaptive noise uses direct-only proxy

// ── Dense grid toggle ───────────────────────────────────────────────
constexpr bool DEFAULT_USE_DENSE_GRID = false;          // use cell-bin dense grid path


// =====================================================================
//  §11  LEGACY ALIASES
// =====================================================================
// Still referenced across the GPU pipeline and tests.
// TODO: migrate callers to the canonical names, then remove.

constexpr int   DEFAULT_NEE_DEEP_SAMPLES = 4;         // v1 deep NEE (still wired in optix_renderer)
constexpr int   DEFAULT_MAX_BOUNCES      = DEFAULT_PHOTON_MAX_BOUNCES;
constexpr int   DEFAULT_MIN_BOUNCES_RR   = DEFAULT_PHOTON_MIN_BOUNCES_RR;
constexpr float DEFAULT_RR_THRESHOLD     = DEFAULT_PHOTON_RR_THRESHOLD;
constexpr int   DEFAULT_NUM_PHOTONS      = DEFAULT_GLOBAL_PHOTON_BUDGET;


// =====================================================================
//  §12  SCENE PROFILES
// =====================================================================

#ifdef SCENE_CORNELL_BOX
  constexpr const char* SCENE_OBJ_PATH    = "cornell_box/cornellbox.obj";
  constexpr const char* SCENE_DISPLAY_NAME = "Cornell Box";
  constexpr bool  SCENE_IS_REFERENCE       = true;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 40.0f;
  constexpr float SCENE_CAM_SPEED          = 0.1f;

#elif defined(SCENE_CONFERENCE)
  constexpr const char* SCENE_OBJ_PATH    = "conference/conference.obj";
  constexpr const char* SCENE_DISPLAY_NAME = "Conference Room";
  constexpr bool  SCENE_IS_REFERENCE       = false;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 50.0f;
  constexpr float SCENE_CAM_SPEED          = 0.1f;

#elif defined(SCENE_LIVING_ROOM)
  constexpr const char* SCENE_OBJ_PATH    = "living_room/living_room.obj";
  constexpr const char* SCENE_DISPLAY_NAME = "Living Room";
  constexpr bool  SCENE_IS_REFERENCE       = false;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 50.0f;
  constexpr float SCENE_CAM_SPEED          = 0.1f;

#elif defined(SCENE_SIBENIK)
  constexpr const char* SCENE_OBJ_PATH    = "sibenik/sibnek.obj";
  constexpr const char* SCENE_DISPLAY_NAME = "Sibenik Cathedral";
  constexpr bool  SCENE_IS_REFERENCE       = false;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 50.0f;
  constexpr float SCENE_CAM_SPEED          = 0.1f;

#else
  #error "No scene selected! Uncomment one SCENE_* define in config.h §0"
#endif

// ── Scene lighting mode ─────────────────────────────────────────────
enum class SceneLightMode {
    FromMTL,             // emissive surfaces from .mtl
    DirectionalToFloor,  // directional light to centre floor (Sibenik)
    HemisphereEnv,       // upper hemisphere sky dome (Sponza)
    SphericalEnv,        // full sphere environment (Mori Knob)
};

struct SceneProfile {
    const char*    obj_path;
    const char*    display_name;
    bool           is_reference;
    float          cam_pos[3];
    float          cam_lookat[3];
    float          cam_fov;
    float          cam_speed;
    SceneLightMode light_mode;
};

constexpr int NUM_SCENE_PROFILES = 8;

constexpr SceneProfile SCENE_PROFILES[NUM_SCENE_PROFILES] = {
    { "cornell_box/cornellbox.obj",      "Cornell Box",       true,
      {0,0,0}, {0,0,-1}, 40.f, 0.1f, SceneLightMode::FromMTL },
    { "conference/conference.obj",       "Conference Room",   false,
      {0,0,0}, {0,0,-1}, 50.f, 0.1f, SceneLightMode::FromMTL },
    { "living_room/living_room.obj",     "Living Room",       false,
      {0,0,0}, {0,0,-1}, 50.f, 0.1f, SceneLightMode::FromMTL },
    { "Interior/interior.obj",           "Interior",          false,
      {0,0,0}, {0,0,-1}, 50.f, 0.1f, SceneLightMode::FromMTL },
    { "salle_de_bain/salle_de_bain.obj", "Salle de Bain",     false,
      {0,0,0}, {0,0,-1}, 50.f, 0.1f, SceneLightMode::FromMTL },
    { "sibenik/sibnek.obj",              "Sibenik Cathedral",  false,
      {0,0,0}, {0,0,-1}, 50.f, 0.1f, SceneLightMode::DirectionalToFloor },
    { "sponza/sponza.obj",              "Sponza",             false,
      {0,0,0}, {0,0,-1}, 60.f, 0.1f, SceneLightMode::HemisphereEnv },
    { "mori_knob/testObj.obj",           "Mori Knob",          false,
      {0,0,0}, {0,0,-1}, 50.f, 0.1f, SceneLightMode::FromMTL },
};

