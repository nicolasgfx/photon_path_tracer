#pragma once
#include <cstdint>
// ─────────────────────────────────────────────────────────────────────
// config.h – Central configuration for the photon-centric renderer v2.3
// ─────────────────────────────────────────────────────────────────────
//
//   Architecture: photon-centric (doc/architecture/architecture.md)
//     Camera rays  → first-hit only → NEE + photon gather
//     Photon rays  → full path tracing from lights (the real tracers)
//     Gather       → adaptive kNN on tangent plane, not fixed-radius 3D sphere
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
// See §9 for debug-specific flags that are subordinate to this gate.
constexpr bool ENABLE_STATS = true;


// =====================================================================
//  §0  SCENE SELECTION
// =====================================================================
// Uncomment exactly ONE.  Runtime switching via keys 1–9,0 uses
// SCENE_PROFILES[] at the bottom of this file.

//#define SCENE_CORNELL_BOX
//#define SCENE_LIVING_ROOM
//#define SCENE_SALLE_DE_BAIN
//#define SCENE_FIREPLACE_ROOM
//#define SCENE_CONFERENCE
//#define SCENE_LIVING_ROOM_2
#define SCENE_STAIRCASE
//#define SCENE_STAIRCASE_2
//#define SCENE_BEDROOM
//#define SCENE_BATHROOM


// =====================================================================
//  §1  IMAGE OUTPUT
// =====================================================================

constexpr int DEFAULT_IMAGE_WIDTH  = 1920;           // [R]
constexpr int DEFAULT_IMAGE_HEIGHT = 1024;           // [R]


// =====================================================================
//  §2  CORE RENDERING
// =====================================================================
// The parameters that most directly control output quality and speed.
// Adjust these first when tuning a render.

// ── Sub-pixel stratified jitter grid ────────────────────────────────
constexpr int STRATA_X = 32;
constexpr int STRATA_Y = 32;                           // 16 × 16 = 256 strata

// ── Samples per pixel ───────────────────────────────────────────────
// Anti-aliasing + noise averaging.  This is the single biggest
// quality/speed knob.
//   Fast: 4–8  |  Balanced: 16  |  Quality: 32–64  |  Final: 128–256
constexpr int DEFAULT_SPP = STRATA_X * STRATA_Y;       // [R]

// ── Photon budgets ──────────────────────────────────────────────────
// Total photons emitted per pass.  The photon map carries ALL indirect
// transport in the v2 architecture.
//   Fast: 100k  |  Balanced: 500k–1M  |  Quality: 2M–5M
constexpr int DEFAULT_GLOBAL_PHOTON_BUDGET  = 1000000;   // [R]  diffuse indirect
constexpr int DEFAULT_CAUSTIC_PHOTON_BUDGET = 1000000;   // [R]  specular→diffuse caustics

// ── Gather radii (max kNN search radius) ────────────────────────────
// These set the MAXIMUM search radius for k-NN photon gathering.
// The actual gather radius per hitpoint is adaptive: the tangential
// distance to the K-th nearest photon (see DEFAULT_KNN_K).
// These caps prevent pathologically large searches in sparse regions.
// Values are fractions of scene extent (scene normalised to [-0.5, 0.5]³).
//   Fast: 0.08–0.10  |  Balanced: 0.05  |  Quality: 0.02–0.03
constexpr float DEFAULT_GATHER_RADIUS  = 0.05f;      // 0.05[R]  global (diffuse) map
constexpr float DEFAULT_CAUSTIC_RADIUS = 0.025f;     // 0.025[R]  caustic map (tighter for sharp caustics)

// Enable/disable the photon density estimation pass at first camera hit.
// When enabled, a separate raygen program gathers photon radiance at the
// first diffuse hit and writes it to photon_gather_buffer.  The result is
// added to the path-traced output during tonemap / spectrum-to-HDR.
constexpr bool DEFAULT_PHOTON_GATHER_ENABLED = true;


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
constexpr int DEFAULT_MAX_BOUNCES = 8;                // [R]

// ── Russian roulette ────────────────────────────────────────────────
// After MIN_BOUNCES_RR guaranteed bounces, each continuation is
// probabilistic: survive = min(max_spectrum(throughput), RR_THRESHOLD).
// Throughput is divided by survival probability (unbiased).
//   MIN_BOUNCES_RR — Fast: 3  |  Balanced: 5  |  Quality: 8
//   RR_THRESHOLD   — 0.80 (aggressive) .. 0.95 (conservative)
constexpr int   DEFAULT_MIN_BOUNCES_RR = 5;            // [R]
constexpr float DEFAULT_RR_THRESHOLD   = 0.95f;        // [R]

// ── Spectral transport (PBRT v4 §14.3) ─────────────────────────────
// Hero wavelengths per photon.  4 = PBRT v4 recommended.
// Changing this requires full rebuild — it's wired into the photon struct.
constexpr int HERO_WAVELENGTHS = 4;

// ── Emission cone angle ─────────────────────────────────────────────
// Half-angle (degrees) of cosine-weighted emission from light sources.
// 90° = full hemisphere (Lambertian).  Smaller = directional emitters.
constexpr float DEFAULT_LIGHT_CONE_HALF_ANGLE_DEG = 90.0f;

// ── Preview mode (interactive navigation) ───────────────────────────
// During interactive camera motion the renderer uses plain unguided
// path tracing with a reduced bounce cap for speed.  After
// IDLE_TIMEOUT_SEC of no input the viewer switches to full-quality
// photon-guided accumulation.
constexpr int   PREVIEW_MAX_BOUNCES = 2;             //  bounce cap in preview mode
constexpr float IDLE_TIMEOUT_SEC    = 1.0f;


// =====================================================================
//  §4  CAMERA RAYS & DIRECT LIGHTING
// =====================================================================

// ── Specular chain ──────────────────────────────────────────────────
// Camera ray follows mirror/glass bounces until hitting a diffuse
// surface, then evaluates NEE + photon gather.
//   Fast: 4  |  Balanced: 8  |  Quality: 12–16
constexpr int DEFAULT_MAX_SPECULAR_CHAIN = 8;          // [R]

// ── Camera ray max bounces (full path trace) ───────────────────────
// Total bounce limit for camera rays in the full path-trace loop
// (specular chain + glossy continuation + final gather).
// With RR starting at bounce 3, most energy is captured in 6–8 bounces.
//   Fast: 6  |  Balanced: 8  |  Quality: 12–16
constexpr int DEFAULT_MAX_BOUNCES_CAMERA = 8;             // [R]

// ── Final gather ────────────────────────────────────────────────────
// When true, the last diffuse bounce performs a photon density estimate
// instead of terminating.  Improves colour bleeding at the cost of one
// extra gather per camera ray.
constexpr bool DEFAULT_PHOTON_FINAL_GATHER = true;


// =====================================================================
//  §4a  PHOTON-GUIDED PATH TRACING (one-sample MIS)
// =====================================================================
// Dense-grid guided sampling: at each non-delta camera bounce, a coin
// flip (p = guide_fraction) chooses between picking a photon direction
// from the dense grid ("guide") and a normal BSDF importance sample.
// One-sample MIS (balance heuristic) weights the result using the
// marginal PDF over all eligible photons.
//
// All constants below control this subsystem.  The dense grid itself
// is built from the merged photon SoA (global + caustic + targeted)
// after every photon trace pass.

// Master switch — disables the entire guide path when false.
constexpr bool  DEFAULT_USE_GUIDE = true;        // [K]

constexpr int MAX_GUIDE_PDF_PHOTONS = 32; // was 64

// Probability of choosing the guided strategy vs pure BSDF.
//   0.0 = BSDF only  |  0.5 = balanced  |  1.0 = guide only
constexpr float DEFAULT_GUIDE_FRACTION   = 0.5f;        // [R]

// ── Dense grid ──────────────────────────────────────────────────────
// Uniform 3D grid over the photon AABB.  Each cell stores start/end
// indices into a sorted photon array for O(1) cell lookup.
constexpr float DENSE_GRID_CELL_SIZE     = 0.01f;  // 0.01f = 1cm cell side-length (metres)

// ── Cone jitter ─────────────────────────────────────────────────────
// Half-angle (radians) applied to photon wi when sampling a guided
// direction.  Widens the stochastic axis around the photon's incoming
// direction, improving convergence.
//   0.0  = no jitter (use photon wi exactly)
//   0.15 = ~8.6°  balanced default
constexpr float DEFAULT_PHOTON_GUIDE_CONE_HALF_ANGLE = 0.15f; // 0.15f // [R] radians

// ── Firefly clamp ───────────────────────────────────────────────────
// Safety clamp on per-bounce f·cos/pdf contribution.  Prevents residual
// firefly outliers from numerical edge cases.  Slightly biased (energy
// loss in extreme situations) but makes convergence much more robust.
//   10–20 = aggressive  |  50 = balanced  |  200+ = nearly unbiased
constexpr float MAX_BOUNCE_CONTRIBUTION  = 50.f;

// Per-path throughput clamp.  After each bounce, the cumulative
// throughput magnitude is capped to this value.  Prevents compounding
// of per-bounce clamps (50² = 2500 after two clamped bounces) from
// creating extreme NEE contributions downstream.  Slightly biased but
// eliminates persistent fireflies visible even at high frame counts.
//   50 = aggressive  |  100 = balanced  |  500+ = nearly unbiased
constexpr float MAX_PATH_THROUGHPUT      = 100.f;


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

// ── OptiX AI Denoiser ───────────────────────────────────────────────
constexpr bool  DEFAULT_DENOISER_ENABLED       = false;   // [K]  enable OptiX denoiser
constexpr bool  DEFAULT_DENOISER_GUIDE_ALBEDO  = true;    //      feed albedo AOV to denoiser
constexpr bool  DEFAULT_DENOISER_GUIDE_NORMAL  = true;    //      feed normal AOV to denoiser
constexpr float DEFAULT_DENOISER_BLEND         = 0.0f;    // [R]  0 = full denoise, 1 = no denoise


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

// ── Cell cache (per-cell photon statistics) ─────────────────────────
// Adaptive gather radius, empty-region skip, caustic hotspot detection.
constexpr uint32_t CELL_CACHE_TABLE_SIZE     = 65536u;  // 64K cells

// ── Directional photon bins (Fibonacci sphere) ──────────────────────
constexpr int PHOTON_BIN_COUNT     = 32;    // runtime bin count (quasi-uniform S²)
constexpr int MAX_PHOTON_BIN_COUNT = 64;    // compile-time upper bound for fixed arrays

// ── Adaptive caustic emission ───────────────────────────────────────
constexpr int   CAUSTIC_MIN_FOR_ANALYSIS    = 10;      // min photons per cell for CV analysis
constexpr float CAUSTIC_CV_THRESHOLD        = 0.50f;   // CV above this = "hot" cell

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
//  §8  VOLUME RENDERING (V key toggle, off by default)
// =====================================================================

constexpr bool  DEFAULT_VOLUME_ENABLED  = false;       // master switch
constexpr float DEFAULT_VOLUME_DENSITY  = 0.15f;       // σ_t extinction (0.01–1.0)
constexpr float DEFAULT_VOLUME_FALLOFF  = 0.0f;        // height falloff (0 = homogeneous)
constexpr float DEFAULT_VOLUME_ALBEDO   = 0.95f;       // σ_s / σ_t (0–1)
constexpr int   DEFAULT_VOLUME_SAMPLES  = 2;           // medium samples per segment (1–8)
constexpr float DEFAULT_VOLUME_MAX_T    = 2.0f;        // max march distance (scene units)


// =====================================================================
//  §9  DEBUG
// =====================================================================

constexpr bool DEBUG_PHOTON_SINGLE_BOUNCE = false;     // stop photon after 1st hit

// ── Per-bounce AOV debug buffers (DB-04, §10.3) ─────────────────────
constexpr int   MAX_AOV_BOUNCES = 4;                    // first N bounces captured

constexpr bool ADAPTIVE_NOISE_USE_DIRECT_ONLY = false; // adaptive noise uses direct-only proxy


// =====================================================================
//  §10  SCENE PROFILES
// =====================================================================

#ifdef SCENE_CORNELL_BOX
  constexpr const char* SCENE_OBJ_PATH    = "cornell_box/cornellbox.obj";
  constexpr const char* SCENE_DISPLAY_NAME = "Cornell Box";
  constexpr bool  SCENE_IS_REFERENCE       = true;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 70.0f;
  constexpr float SCENE_CAM_SPEED          = 0.1f;

#elif defined(SCENE_LIVING_ROOM)
  constexpr const char* SCENE_OBJ_PATH    = "living_room/scene-v4.obj";
  constexpr const char* SCENE_DISPLAY_NAME = "Living Room";
  constexpr bool  SCENE_IS_REFERENCE       = false;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 70.0f;
  constexpr float SCENE_CAM_SPEED          = 0.1f;

#elif defined(SCENE_SALLE_DE_BAIN)
  constexpr const char* SCENE_OBJ_PATH    = "salle_de_bain/salle_de_bain.obj";
  constexpr const char* SCENE_DISPLAY_NAME = "Salle de Bain";
  constexpr bool  SCENE_IS_REFERENCE       = false;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 70.0f;
  constexpr float SCENE_CAM_SPEED          = 0.1f;

#elif defined(SCENE_FIREPLACE_ROOM)
  constexpr const char* SCENE_OBJ_PATH    = "fireplace_room/fireplace_room.obj";
  constexpr const char* SCENE_DISPLAY_NAME = "Fire Place";
  constexpr bool  SCENE_IS_REFERENCE       = false;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 70.0f;
  constexpr float SCENE_CAM_SPEED          = 0.1f;

#elif defined(SCENE_CONFERENCE)
  constexpr const char* SCENE_OBJ_PATH    = "conference/conference.obj";
  constexpr const char* SCENE_DISPLAY_NAME = "Conference Room";
  constexpr bool  SCENE_IS_REFERENCE       = false;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 70.0f;
  constexpr float SCENE_CAM_SPEED          = 0.1f;

#elif defined(SCENE_LIVING_ROOM_2)
  constexpr const char* SCENE_OBJ_PATH    = "living_room_2/scene-v4.obj";
  constexpr const char* SCENE_DISPLAY_NAME = "Living Room 2";
  constexpr bool  SCENE_IS_REFERENCE       = false;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 70.0f;
  constexpr float SCENE_CAM_SPEED          = 0.1f;

#elif defined(SCENE_STAIRCASE)
  constexpr const char* SCENE_OBJ_PATH    = "staircase/scene-v4.obj";
  constexpr const char* SCENE_DISPLAY_NAME = "Staircase";
  constexpr bool  SCENE_IS_REFERENCE       = false;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 100.0f;
  constexpr float SCENE_CAM_SPEED          = 0.1f;

#elif defined(SCENE_STAIRCASE_2)
  constexpr const char* SCENE_OBJ_PATH    = "staircase2/scene-v4.obj";
  constexpr const char* SCENE_DISPLAY_NAME = "Staircase 2";
  constexpr bool  SCENE_IS_REFERENCE       = false;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 70.0f;
  constexpr float SCENE_CAM_SPEED          = 0.1f;

#elif defined(SCENE_BEDROOM)
  constexpr const char* SCENE_OBJ_PATH    = "bedroom/scene-v4.obj";
  constexpr const char* SCENE_DISPLAY_NAME = "Bedroom";
  constexpr bool  SCENE_IS_REFERENCE       = false;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 70.0f;
  constexpr float SCENE_CAM_SPEED          = 0.1f;

#elif defined(SCENE_BATHROOM)
  constexpr const char* SCENE_OBJ_PATH    = "bathroom/scene-v4.obj";
  constexpr const char* SCENE_DISPLAY_NAME = "Bathroom";
  constexpr bool  SCENE_IS_REFERENCE       = false;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 70.0f;
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

constexpr int NUM_SCENE_PROFILES = 10;

constexpr SceneProfile SCENE_PROFILES[NUM_SCENE_PROFILES] = {
    { "cornell_box/cornellbox.obj",              "Cornell Box",       true,
      {0,0,0}, {0,0,-1}, 70.f, 0.1f, SceneLightMode::FromMTL },
    { "living_room/scene-v4.obj",                "Living Room",       false,
      {0,0,0}, {0,0,-1}, 70.f, 0.1f, SceneLightMode::FromMTL },
    { "salle_de_bain/salle_de_bain.obj",         "Salle de Bain",     false,
      {0,0,0}, {0,0,-1}, 70.f, 0.1f, SceneLightMode::FromMTL },
    { "fireplace_room/fireplace_room.obj",       "Fire Place",        false,
      {0,0,0}, {0,0,-1}, 70.f, 0.1f, SceneLightMode::FromMTL },
    { "conference/conference.obj",               "Conference Room",   false,
      {0,0,0}, {0,0,-1}, 70.f, 0.1f, SceneLightMode::FromMTL },
    { "living_room_2/scene-v4.obj",              "Living Room 2",     false,
      {0,0,0}, {0,0,-1}, 70.f, 0.1f, SceneLightMode::FromMTL },
    { "staircase/scene-v4.obj",                  "Staircase",         false,
      {0,0,0}, {0,0,-1}, 70.f, 0.1f, SceneLightMode::FromMTL },
    { "staircase2/scene-v4.obj",                 "Staircase 2",       false,
      {0,0,0}, {0,0,-1}, 70.f, 0.1f, SceneLightMode::FromMTL },
    { "bedroom/scene-v4.obj",                    "Bedroom",           false,
      {0,0,0}, {0,0,-1}, 70.f, 0.1f, SceneLightMode::FromMTL },
    { "bathroom/scene-v4.obj",                   "Bathroom",          false,
      {0,0,0}, {0,0,-1}, 70.f, 0.1f, SceneLightMode::FromMTL },
};

