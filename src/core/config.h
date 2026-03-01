#pragma once
#include <cstdint>
// ─────────────────────────────────────────────────────────────────────
// config.h – Central configuration for the photon-guided path tracer v3
// ─────────────────────────────────────────────────────────────────────
//
//   Architecture: photon-guided path tracing (doc/architecture/architecture.md)
//     Pass 1: photon tracing → photon map + cell-bin grid
//     Pass 2: camera path tracing → iterative bounce loop with photon guide
//     Gather → adaptive kNN on tangent plane (final gather at path termination)
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
//  MASTER GATE  (must precede all sections that reference it)
// =====================================================================
// Gate runtime statistics collection and debug file output.  When false,
// the compiler eliminates all stats code paths (zero overhead).
// See §10 for debug-specific flags that are subordinate to this gate.
constexpr bool ENABLE_STATS = true;


// =====================================================================
//  §0  SCENE SELECTION
// =====================================================================
// Uncomment exactly ONE.  Runtime switching via keys 1–8 uses
// SCENE_PROFILES[] at the bottom of this file.

//#define SCENE_CORNELL_BOX
//#define SCENE_CORNELL_SPHERE
//#define SCENE_CORNELL_MIRROR
//#define SCENE_CORNELL_WATER
#define SCENE_LIVING_ROOM
//#define SCENE_CONFERENCE
//#define SCENE_SALLE_DE_BAIN
//#define SCENE_MORI_KNOB

// ── v3 flags ────────────────────────────────────────────────────────
// (Bresenham lobe balance removed in v3 — no per-pixel lobe balance)


// =====================================================================
//  §1  IMAGE OUTPUT
// =====================================================================

constexpr int DEFAULT_IMAGE_WIDTH  = 1920;           // [R]
constexpr int DEFAULT_IMAGE_HEIGHT = 1080;           // [R]


// =====================================================================
//  §2  CORE RENDERING
// =====================================================================
// The parameters that most directly control output quality and speed.
// Adjust these first when tuning a render.

// ── Samples per pixel ───────────────────────────────────────────────
// Anti-aliasing + noise averaging.  This is the single biggest
// quality/speed knob.
//   Fast: 4–8  |  Balanced: 16  |  Quality: 32–64  |  Final: 128–256
constexpr int DEFAULT_SPP = 256;                       // [R]

// Sub-pixel stratified jitter grid.
// Constraint: STRATA_X × STRATA_Y == DEFAULT_SPP.
constexpr int STRATA_X = 16;                           // 4 × 4 = 16 = DEFAULT_SPP
constexpr int STRATA_Y = 16;

// ── Photon budgets ──────────────────────────────────────────────────
// Total photons emitted per pass.  The photon map carries ALL indirect
// transport in the v2 architecture.
//   Fast: 100k  |  Balanced: 500k–1M  |  Quality: 2M–5M
constexpr int DEFAULT_GLOBAL_PHOTON_BUDGET  = 2000000;   // [R]  diffuse indirect
constexpr int DEFAULT_CAUSTIC_PHOTON_BUDGET = 2000000;   // [R]  specular→diffuse caustics

// ── Gather radii (max kNN search radius) ────────────────────────────
// These set the MAXIMUM search radius for k-NN photon gathering.
// The actual gather radius per hitpoint is adaptive: the tangential
// distance to the K-th nearest photon (see DEFAULT_KNN_K).
// These caps prevent pathologically large searches in sparse regions.
// Values are fractions of SCENE_REF_EXTENT (scene in [-0.5, 0.5]³).
//   Fast: 0.08–0.10  |  Balanced: 0.05  |  Quality: 0.02–0.03
constexpr float DEFAULT_GATHER_RADIUS  = 0.05f;      // 0.05[R]  global (diffuse) map
constexpr float DEFAULT_CAUSTIC_RADIUS = 0.025f;     // 0.025[R]  caustic map (tighter for sharp caustics)


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
constexpr int DEFAULT_PHOTON_MAX_BOUNCES = 10;        // [R]

// ── Russian roulette ────────────────────────────────────────────────
// After MIN_BOUNCES_RR guaranteed bounces, each continuation is
// probabilistic: survive = min(max_spectrum(throughput), RR_THRESHOLD).
// Throughput is divided by survival probability (unbiased).
// Set high to guarantee photon survival through glass stacks — RR
// should not kill photons mid-transmission through nested dielectrics.
//   MIN_BOUNCES_RR — Fast: 3–4  |  Balanced: 8  |  Quality: 10
//   RR_THRESHOLD   — 0.80 (aggressive) .. 0.95 (conservative)
constexpr int   DEFAULT_PHOTON_MIN_BOUNCES_RR = 8;    // [R]
constexpr float DEFAULT_PHOTON_RR_THRESHOLD   = 0.90f;// [R]

// ── Spectral transport (PBRT v4 §14.3) ─────────────────────────────
// Hero wavelengths per photon.  4 = PBRT v4 recommended.
// Changing this requires full rebuild — it's wired into the photon struct.
constexpr int HERO_WAVELENGTHS = 4;

// ── Emission cone angle ─────────────────────────────────────────────
// Half-angle (degrees) of cosine-weighted emission from light sources.
// 90° = full hemisphere (Lambertian).  Smaller = directional emitters.
constexpr float DEFAULT_LIGHT_CONE_HALF_ANGLE_DEG = 90.0f;

// Multi-map photon re-tracing: re-trace the photon map with a new
// RNG seed every N camera samples to decorrelate photon/camera noise.
//   0 = single map  |  4 = balanced  |  8 = quality


// =====================================================================
//  §4  CAMERA PATH TRACING (v3 — Photon-Guided)
// =====================================================================

// ── Path depth ──────────────────────────────────────────────────────
// v3: single iterative bounce loop replaces specular chain + glossy.
constexpr int DEFAULT_MAX_BOUNCES_CAMERA = 12;         // [R]  max camera path depth
constexpr int DEFAULT_MIN_BOUNCES_RR     = 3;          // [R]  guaranteed bounces before RR
constexpr float DEFAULT_RR_THRESHOLD     = 0.95f;      // [R]  max survival probability

// ── Photon-guided sampling ──────────────────────────────────────────
constexpr float DEFAULT_GUIDE_FRACTION   = 0.5f;       // [R]  probability of guided vs BSDF sample
constexpr bool  DEFAULT_USE_GUIDE        = true;        // [K]  enable/disable guided sampling

// ── Photon density fallback ─────────────────────────────────────────
constexpr int  DEFAULT_GUIDE_FALLBACK_BOUNCE = 3;       // [R]  switch to photon gather after this bounce
constexpr bool DEFAULT_PHOTON_FINAL_GATHER   = true;    // [K]  use photon map as final gather at terminal bounces

// ── Per-bounce AOV debug buffers (DB-04, §10.3) ─────────────────────
constexpr int   MAX_AOV_BOUNCES = 4;                    // first N bounces captured

// ── Legacy aliases (still referenced by SPPM, CPU renderer, NEE) ────
constexpr int   DEFAULT_MAX_SPECULAR_CHAIN     = DEFAULT_MAX_BOUNCES_CAMERA;  // alias for SPPM camera pass
constexpr bool  DEFAULT_USE_MIS                = true;


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

// ── OptiX AI Denoiser ────────────────────────────────────────────────
// When enabled, the final render applies the OptiX HDR denoiser after
// accumulation and before tone mapping.  Requires albedo + normal AOVs
// written during the camera pass.  Guide layers (albedo, normal) improve
// edge preservation and detail retention.
//   true  = denoise final render (adds ~20–50 ms overhead)
//   false = raw Monte Carlo output (legacy behaviour)
constexpr bool DEFAULT_DENOISER_ENABLED        = true;   // [R]
constexpr bool DEFAULT_DENOISER_GUIDE_ALBEDO   = true;   //     use albedo guide layer
constexpr bool DEFAULT_DENOISER_GUIDE_NORMAL   = true;   //     use normal guide layer
constexpr float DEFAULT_DENOISER_BLEND         = 0.0f;   //     0 = fully denoised, 1 = original

// Progress snapshot PNGs at power-of-2 SPP intervals (near-zero overhead).
// Subordinate to ENABLE_STATS — snapshots are a debugging/analysis tool.
constexpr bool PROGRESS_SNAPSHOT_ENABLED = ENABLE_STATS && true;
constexpr int  PROGRESS_SNAPSHOT_INTERVAL = 0;        // 0 = power-of-2 only

// Write per-component debug PNGs (NEE direct, photon indirect, caustic).
// Subordinate to ENABLE_STATS.
constexpr bool DEBUG_COMPONENT_PNGS = ENABLE_STATS && false;


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
//  §10  DEBUG & STATISTICS
// =====================================================================

// ENABLE_STATS is defined at the top of this file (MASTER GATE).
// All debug flags below are subordinate to it.

constexpr bool DEBUG_PHOTON_SINGLE_BOUNCE = false;     // stop photon after 1st hit
constexpr bool DEBUG_PHOTON_INDIRECT_PNG  = ENABLE_STATS && false;  // emit photon-indirect preview PNGs at launch
constexpr bool DEBUG_CAUSTIC_PNG          = ENABLE_STATS && false;  // emit caustic-only debug PNGs (needs above)
constexpr bool DEBUG_COVERAGE_PNG         = ENABLE_STATS && false;  // emit coverage debug PNGs
constexpr bool ADAPTIVE_NOISE_USE_DIRECT_ONLY = false; // adaptive noise uses direct-only proxy

// ── Dense grid toggle ───────────────────────────────────────────────
constexpr bool DEFAULT_USE_DENSE_GRID = false;          // use cell-bin dense grid path


// =====================================================================
//  §11  LEGACY ALIASES
// =====================================================================
// Still referenced across the GPU pipeline and tests.
// TODO(v3): migrate callers to canonical names, then remove.

constexpr int   DEFAULT_MAX_BOUNCES      = DEFAULT_PHOTON_MAX_BOUNCES;
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

#elif defined(SCENE_CORNELL_SPHERE)
  constexpr const char* SCENE_OBJ_PATH    = "cornell_sphere/CornellBox-Sphere.obj";
  constexpr const char* SCENE_DISPLAY_NAME = "Cornell Sphere";
  constexpr bool  SCENE_IS_REFERENCE       = false;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 40.0f;
  constexpr float SCENE_CAM_SPEED          = 0.1f;

#elif defined(SCENE_CORNELL_MIRROR)
  constexpr const char* SCENE_OBJ_PATH    = "cornell_mirror/CornellBox-Mirror.obj";
  constexpr const char* SCENE_DISPLAY_NAME = "Cornell Mirror";
  constexpr bool  SCENE_IS_REFERENCE       = false;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 40.0f;
  constexpr float SCENE_CAM_SPEED          = 0.1f;

#elif defined(SCENE_CORNELL_WATER)
  constexpr const char* SCENE_OBJ_PATH    = "cornell_water/CornellBox-Water.obj";
  constexpr const char* SCENE_DISPLAY_NAME = "Cornell Water";
  constexpr bool  SCENE_IS_REFERENCE       = false;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 40.0f;
  constexpr float SCENE_CAM_SPEED          = 0.1f;

#elif defined(SCENE_LIVING_ROOM)
  constexpr const char* SCENE_OBJ_PATH    = "living_room/living_room.obj";
  constexpr const char* SCENE_DISPLAY_NAME = "Living Room";
  constexpr bool  SCENE_IS_REFERENCE       = false;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 50.0f;
  constexpr float SCENE_CAM_SPEED          = 0.1f;

#elif defined(SCENE_CONFERENCE)
  constexpr const char* SCENE_OBJ_PATH    = "conference/conference.obj";
  constexpr const char* SCENE_DISPLAY_NAME = "Conference Room";
  constexpr bool  SCENE_IS_REFERENCE       = false;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 50.0f;
  constexpr float SCENE_CAM_SPEED          = 0.1f;

#elif defined(SCENE_SALLE_DE_BAIN)
  constexpr const char* SCENE_OBJ_PATH    = "salle_de_bain/salle_de_bain.obj";
  constexpr const char* SCENE_DISPLAY_NAME = "Salle de Bain";
  constexpr bool  SCENE_IS_REFERENCE       = false;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 50.0f;
  constexpr float SCENE_CAM_SPEED          = 0.1f;

#elif defined(SCENE_MORI_KNOB)
  constexpr const char* SCENE_OBJ_PATH    = "mori_knob/testObj.obj";
  constexpr const char* SCENE_DISPLAY_NAME = "Mori Knob";
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
    { "cornell_box/cornellbox.obj",              "Cornell Box",       true,
      {0,0,0}, {0,0,-1}, 40.f, 0.1f, SceneLightMode::FromMTL },
    { "cornell_sphere/CornellBox-Sphere.obj",    "Cornell Sphere",    false,
      {0,0,0}, {0,0,-1}, 40.f, 0.1f, SceneLightMode::FromMTL },
    { "fireplace_room/fireplace_room.obj",        "Fireplace Room",    false,
      {0,0,0}, {0,0,-1}, 50.f, 0.1f, SceneLightMode::FromMTL },
    { "cornell_water/CornellBox-Water.obj",      "Cornell Water",     false,
      {0,0,0}, {0,0,-1}, 40.f, 0.1f, SceneLightMode::FromMTL },
    { "living_room/living_room.obj",             "Living Room",       false,
      {0,0,0}, {0,0,-1}, 50.f, 0.1f, SceneLightMode::FromMTL },
    { "conference/conference.obj",               "Conference Room",   false,
      {0,0,0}, {0,0,-1}, 50.f, 0.1f, SceneLightMode::FromMTL },
    { "salle_de_bain/salle_de_bain.obj",         "Salle de Bain",     false,
      {0,0,0}, {0,0,-1}, 50.f, 0.1f, SceneLightMode::FromMTL },
    { "mori_knob/testObj.obj",                   "Mori Knob",         false,
      {0,0,0}, {0,0,-1}, 50.f, 0.1f, SceneLightMode::FromMTL },
};

