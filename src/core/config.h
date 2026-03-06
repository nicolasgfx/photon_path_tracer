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
constexpr bool ENABLE_STATS = false;

// Gate guided-sampling photon-count diagnostics (min / max / avg eligible
// photons per neighbourhood).  When true, four device-side atomic counters
// are updated on every guided bounce — substantial overhead on GPU.
constexpr bool ENABLE_GUIDE_STATS = false;


// =====================================================================
//  §0  SCENE SELECTION
// =====================================================================
// Uncomment exactly ONE.  Runtime switching via keys 1–9,0,Shift+1 uses
// SCENE_PROFILES[] at the bottom of this file.

#define SCENE_CORNELL_BOX
//#define SCENE_FIREPLACE_ROOM
//#define SCENE_STAIRCASE
//#define SCENE_STAIRCASE_2
//#define SCENE_BATHROOM
//#define SCENE_LIVING_ROOM_2
//#define SCENE_BEDROOM
//#define SCENE_CROWN
//#define SCENE_CLASSROOM
//#define SCENE_ZERO_DAY
//#define SCENE_KROKEN


// =====================================================================
//  §1  IMAGE OUTPUT
// =====================================================================

constexpr int DEFAULT_IMAGE_WIDTH  = 2560;           // [R]
constexpr int DEFAULT_IMAGE_HEIGHT = 1440;           // [R]


// =====================================================================
//  §2  CORE RENDERING
// =====================================================================
// The parameters that most directly control output quality and speed.
// Adjust these first when tuning a render.

// ── Sub-pixel stratified jitter grid ────────────────────────────────
constexpr int STRATA_X = 16;
constexpr int STRATA_Y = 16;                           // 16 × 16 = 256 strata

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
constexpr int   DEFAULT_MIN_BOUNCES_RR = 8;            // [R]
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
constexpr float IDLE_TIMEOUT_SEC    = 4.0f;


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
//  §4a  FIRST-HIT GUIDED BRUTE-FORCE PATH TRACING
// =====================================================================
// Direction-map guided sampling: at the FIRST non-delta camera hit,
// a coin flip (p = guide_fraction) chooses between picking a direction
// from the precomputed directional SPP framebuffer and a normal BSDF
// importance sample.  One-sample MIS (balance heuristic) weights the
// result.  All subsequent bounces use pure BSDF + NEE (brute force).
//
// Direction map build (one per photon pass, per subpixel):
//   1. Cast primary ray → first non-delta hitpoint.
//   2. Max-heap kNN: find the K=64 closest photons (3D Euclidean).
//   3. Shadow-ray filter: for each of the K photons trace a shadow ray
//      from the hitpoint.  Accept/reject by material type.
//   4. Epanechnikov-weighted Fibonacci sphere histogram (128 bins) from
//      accepted photons.  Distance kernel: w × max(0, 1 − d²/r_k²).
//   5. Sample one direction per SPP for MIS.

// Master switch — disables the entire first-hit guide when false.
constexpr bool  DEFAULT_USE_GUIDE = !true;           // [K]

// Max photons gathered by kNN before shadow-ray filtering.
constexpr int MAX_GUIDE_PDF_PHOTONS = 64;

// Max shadow rays per pixel in direction map build.  Set lower than
// MAX_GUIDE_PDF_PHOTONS to cap GPU workload on complex BVH scenes.
// Closest photons are evaluated first (most likely to be unoccluded).
//   32 halves shadow-ray cost vs. 64 with negligible quality loss.
constexpr int MAX_DM_SHADOW_RAYS = 32;

// Minimum photons found by kNN before proceeding to shadow-ray
// filtering & histogram.  Below this threshold the guide data is
// too noisy to be useful — skip shadow rays (saves GPU work in
// photon-sparse regions).  0 = disabled (legacy behaviour).
constexpr int MIN_GUIDE_PHOTONS = 8;

// Maximum shell-expansion layers in the direction-map kNN search.
// Layer 0 = center cell, layer 1 = 3×3×3, layer 2 = 5×5×5, etc.
// Early termination (heap root < next-shell boundary) usually exits
// well before this limit.  Caps worst-case cell visits.
constexpr int DM_KNN_MAX_LAYERS = 8;

// Probability of choosing the guided strategy vs pure BSDF (1st hit only).
//   0.0 = BSDF only  |  0.5 = balanced  |  1.0 = guide only
constexpr float DEFAULT_GUIDE_FRACTION   = 0.5f;        // [R]

// ── Directional histogram ───────────────────────────────────────────
// Number of Fibonacci sphere directional bins for the per-subpixel
// histogram.  128 bins ≈ 0.10 sr per bin ≈ 18° angular radius.
constexpr int   DIR_MAP_SPHERE_BINS      = 128;

// Resolution multiplier: subpixel grid is (W * FACTOR) × (H * FACTOR).
constexpr int   DIR_MAP_SUBPIXEL_FACTOR  = 1;

// ── Shadow-ray photon weights ───────────────────────────────────────
// After kNN, a shadow ray from the hitpoint to each photon classifies
// the path.  These flat weights are multiplied by the Epanechnikov
// kernel value before accumulation into the histogram.
constexpr float DIR_MAP_DEFAULT_WEIGHT     = 1.0f;  // miss (no obstruction)
constexpr float DIR_MAP_DELTA_WEIGHT       = 4.0f;  // hit delta material (glass/mirror)
constexpr float DIR_MAP_TRANSLUCENT_WEIGHT = 4.0f;  // hit translucent material

// ── Cone jitter ─────────────────────────────────────────────────────
// Half-angle (radians) applied when sampling a guided direction from
// the direction map.  Widens the stochastic axis around the bin
// centroid, improving convergence.
constexpr float DEFAULT_PHOTON_GUIDE_CONE_HALF_ANGLE = 0.15f; // [R] radians

// ── Direction map hash grid ─────────────────────────────────────────
// Teschner spatial hash for kNN photon lookup.  Cell edge length (m).
// Chosen so the search sphere covers ~5×5×5 – 7×7×7 cells for good
// kNN performance.  Built with radius = cell_size / 2.
constexpr float DIR_MAP_HASH_CELL_SIZE = 0.02f;  // 2 cm cell side-length

// ── Guide search radius (kNN cap) ──────────────────────────────────
// Max 3D Euclidean distance for the kNN photon gather.  The adaptive
// radius (distance to the K-th nearest photon) is usually smaller;
// this is the upper bound for sparse regions.  Matches the main
// photon gather radius so photon density is consistent.
constexpr float DEFAULT_GUIDE_RADIUS = DEFAULT_GATHER_RADIUS;  // 0.05 m

// ── Periodic photon + direction-map rebuild during final render ─────
// Every N SPP, re-trace the photon map (with a fresh seed) and rebuild
// the direction map.  Decorrelates guide directions across the render.
constexpr int DEFAULT_GUIDE_REMAP_INTERVAL = 500;  // [R] SPP between rebuilds

// ── NaN / infinity safety net ────────────────────────────────────────
// With correct one-sample MIS (balance heuristic) and physical BSDFs,
// f·cos / combined_pdf is analytically bounded:
//   Lambert:  f·cos / pdf  ≤  2 · Kd  (< 2 for albedo < 1)
//   GGR spec: importance sampling tracks the NDF, ratio ≈ G·F/4cos  (< ~8)
//   Fabric:   same as Lambert + small sheen term  (< ~3)
// So the clamps below should NEVER trigger under normal rendering.
// They exist purely as guards against numerical edge cases (division by
// near-epsilon, NaN propagation from degenerate geometry, etc.).
// Set to very large values — if they fire, investigate the root cause.
constexpr float MAX_BOUNCE_CONTRIBUTION  = 1e4f;
constexpr float MAX_PATH_THROUGHPUT      = 1e4f;

// ── Spectral outlier clamp (photon-referenced) ──────────────────────
// Uses per-pixel photon irradiance estimates as a low-variance reference
// to soft-clamp high-variance PT samples per wavelength channel.
// A PT sample value above threshold × photon_ref is clamped to that limit.
constexpr bool  DEFAULT_SPECTRAL_CLAMP_ENABLED   = true;    // [K] X key toggle
constexpr float DEFAULT_SPECTRAL_CLAMP_THRESHOLD = 20.0f;   // max ratio PT/ref before clamping


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

// ── Bloom / glow (2D post-FX, §11) ──────────────────────────────────
constexpr bool  DEFAULT_BLOOM_ENABLED   = false;       // [K]  B key toggle
constexpr float DEFAULT_BLOOM_INTENSITY = 0.5f;        //      additive strength (0 = off, 1 = full)
constexpr float DEFAULT_BLOOM_RADIUS_H  = 15.0f;       //      horizontal blur radius in pixels (full res)
constexpr float DEFAULT_BLOOM_RADIUS_V  = 15.0f;       //      vertical blur radius in pixels (full res)


// =====================================================================
//  §6  SPATIAL INDEX & SURFACE FILTER
// =====================================================================
// Parameters controlling photon gather geometry and the underlying
// spatial acceleration.  Rarely need changing.

// ── Surface consistency filter (τ) ──────────────────────────────────
// Plane-distance threshold: photons farther than τ from the query
// tangent plane are rejected (prevents cross-surface leakage).
//   Tight: 0.01  |  Balanced: 0.03  |  Loose: 0.05
// 0.03 is slightly relaxed vs the old 0.02 to reduce black (no-guidance)
// patches on curved or slightly offset surfaces in the direction map.
constexpr float DEFAULT_SURFACE_TAU        = 0.05f;
constexpr float PLANE_TAU_EPSILON_FACTOR   = 10.0f;   // robust τ floor = factor × ray_eps

// ── k-NN gather ─────────────────────────────────────────────────────
// Adaptive radius = distance to the k-th nearest photon.
// CPU: KD-tree.  GPU: grid shell expansion (capped by MAX_GATHER_RADIUS).
constexpr int   DEFAULT_KNN_K                = 100;

// ── 3×3×3 neighbourhood filter for CPU/volume kNN ───────────────────
// When true, CPU shell expansion and GPU volume photon kNN are
// limited to the 3×3×3 cell neighbourhood.  Does NOT affect the
// direction map (which always uses full max-heap kNN).
constexpr bool  KNN_3X3X3_FILTER             = false;

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

// ── Sparse-gather photon reliability thresholds (B4: §6 edge-case) ──
// When fewer than K photons are found within gather_radius, the
// density estimate falls back to a wider radius.  These constants
// control how that fallback behaves.
constexpr int   SPARSE_GATHER_MIN_PHOTONS     = 5;     // below this, suppress photon contribution
constexpr float SPARSE_GATHER_RELIABILITY_K   = 0.25f; // fraction of DEFAULT_KNN_K used as soft reliability threshold


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

#elif defined(SCENE_ZERO_DAY)
  constexpr const char* SCENE_OBJ_PATH    = "zero_day/frame25.obj";
  constexpr const char* SCENE_DISPLAY_NAME = "Zero Day";
  constexpr bool  SCENE_IS_REFERENCE       = false;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 70.0f;
  constexpr float SCENE_CAM_SPEED          = 0.1f;

#elif defined(SCENE_CROWN)
  constexpr const char* SCENE_OBJ_PATH    = "../tools/pbrtv4_scenes/pbrt-v4-scenes/crown/crown.pbrt";
  constexpr const char* SCENE_DISPLAY_NAME = "Crown";
  constexpr bool  SCENE_IS_REFERENCE       = false;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 47.0f;
  constexpr float SCENE_CAM_SPEED          = 0.01f;

#elif defined(SCENE_SAN_MIGUEL)
  constexpr const char* SCENE_OBJ_PATH    = "sanmiguel/sanmiguel-courtyard.obj";
  constexpr const char* SCENE_DISPLAY_NAME = "San Miguel";
  constexpr bool  SCENE_IS_REFERENCE       = false;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 84.0f;
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

#elif defined(SCENE_FIREPLACE_ROOM)
  constexpr const char* SCENE_OBJ_PATH    = "fireplace_room/fireplace_room.obj";
  constexpr const char* SCENE_DISPLAY_NAME = "Fire Place";
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

#elif defined(SCENE_CLASSROOM)
  constexpr const char* SCENE_OBJ_PATH    = "../tools/pbrtv4_scenes/pbrt-v4-scenes/classroom/scene-v4.pbrt";
  constexpr const char* SCENE_DISPLAY_NAME = "Classroom";
  constexpr bool  SCENE_IS_REFERENCE       = false;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 36.0f;
  constexpr float SCENE_CAM_SPEED          = 0.01f;

#elif defined(SCENE_KROKEN)
  constexpr const char* SCENE_OBJ_PATH    = "../tools/pbrtv4_scenes/pbrt-v4-scenes/kroken/camera-1.pbrt";
  constexpr const char* SCENE_DISPLAY_NAME = "Kroken";
  constexpr bool  SCENE_IS_REFERENCE       = false;
  constexpr float SCENE_CAM_POS[]          = { 0.0f, 0.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]       = { 0.0f, 0.0f, -1.0f };
  constexpr float SCENE_CAM_FOV            = 17.0f;
  constexpr float SCENE_CAM_SPEED          = 0.01f;

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
    bool           rotate_x_180;    // rotate geometry 180° around X axis (false = no change)
};

constexpr int NUM_SCENE_PROFILES = 11;

constexpr SceneProfile SCENE_PROFILES[NUM_SCENE_PROFILES] = {
    // Key 1 – Cornell Box (reference scene)
    { "cornell_box/cornellbox.obj",              "Cornell Box",       true,
      {0,0,0}, {0,0,-1}, 90.f, 0.1f, SceneLightMode::FromMTL, false },
    // Key 2 – Fire Place
    { "fireplace_room/fireplace_room.obj",       "Fire Place",        false,
      {0,0,0}, {0,0,-1}, 90.f, 0.1f, SceneLightMode::FromMTL, false },
    // Key 3 – Staircase
    { "staircase/scene-v4.obj",                  "Staircase",         false,
      {0,0,0}, {0,0,-1}, 90.f, 0.1f, SceneLightMode::FromMTL, false },
    // Key 4 – Staircase 2
    { "staircase2/scene-v4.obj",                 "Staircase 2",       false,
      {0,0,0}, {0,0,-1}, 70.f, 0.1f, SceneLightMode::FromMTL, false },
    // Key 5 – Bathroom
    { "bathroom/scene-v4.obj",                   "Bathroom",          false,
      {0,0,0}, {0,0,-1}, 90.f, 0.1f, SceneLightMode::FromMTL, false },
    // Key 6 – Living Room 2
    { "living_room_2/scene-v4.obj",              "Living Room 2",     false,
      {0,0,0}, {0,0,-1}, 90.f, 0.1f, SceneLightMode::FromMTL, false },
    // Key 7 – Bedroom
    { "bedroom/scene-v4.obj",                    "Bedroom",           false,
      {0,0,0}, {0,0,-1}, 90.f, 0.1f, SceneLightMode::FromMTL, false },
    // Key 8 – Crown
    { "../tools/pbrtv4_scenes/pbrt-v4-scenes/crown/crown.pbrt", "Crown", false,
      {0,0,0}, {0,0,-1}, 47.f, 0.01f, SceneLightMode::FromMTL, false },
    // Key 9 – Classroom
    { "../tools/pbrtv4_scenes/pbrt-v4-scenes/classroom/scene-v4.pbrt", "Classroom", false,
      {0,0,0}, {0,0,-1}, 36.f, 0.01f, SceneLightMode::FromMTL, false },
    // Key 0 – Zero Day
    { "zero_day/frame25.obj",   "Zero Day",          false,
      {0,0,0}, {0,0,-1}, 70.f, 0.01f, SceneLightMode::FromMTL, false },
    // Shift+1 – Kroken
    { "../tools/pbrtv4_scenes/pbrt-v4-scenes/kroken/camera-1.pbrt", "Kroken", false,
      {0,0,0}, {0,0,-1}, 90.f, 0.1f, SceneLightMode::FromMTL, false },
};

