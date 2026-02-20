#pragma once
// ─────────────────────────────────────────────────────────────────────
// config.h – Central configuration constants for the renderer
// ─────────────────────────────────────────────────────────────────────
// Keep this file focused on the knobs you actually tweak while iterating.
// Low-level implementation constants (OptiX pipeline sizes, hash primes,
// stratification layout, etc.) belong in the owning source files.
// ─────────────────────────────────────────────────────────────────────

// NOTE: If you find yourself adding a constant here that is only used in
// one .cpp/.cu file, prefer making it a file-local constexpr instead.

// ── Active scene (switch here) ─────────────────────────────────────
// Uncomment exactly ONE of these:
//#define SCENE_CORNELL_BOX
#define SCENE_CONFERENCE
//#define SCENE_LIVING_ROOM
//#define SCENE_BREAKFAST_ROOM
//#define SCENE_SIBENIK

// =====================================================================
// 1) CHANGE THESE FIRST (crucial iteration knobs)
// =====================================================================

// ── Output resolution (also used for final PNG) ─────────────────────
// Recommendation:
//   - Fast iteration: 512×512 or 800×800
//   - Final:          1080p+ (expect much slower)
constexpr int DEFAULT_IMAGE_WIDTH  = 1024;
constexpr int DEFAULT_IMAGE_HEIGHT = 1024;

// ── Sampling / path depth ───────────────────────────────────────────
// Recommendation:
//   - Preview:  1–4 spp
//   - Default:  16 spp (good balance)
//   - Final:    64–256 spp depending on noise tolerance
constexpr int   DEFAULT_SPP            = 16;   // samples per pixel
constexpr int   DEFAULT_MAX_BOUNCES    = 8;    // path depth
constexpr int   DEFAULT_MIN_BOUNCES_RR = 3;    // start Russian roulette after this
constexpr float DEFAULT_RR_THRESHOLD   = 0.95f;

// ── Photon mapping (performance ↔ smoothness) ───────────────────────
// Recommendation:
//   - Preview:  50k–200k photons, radius 0.07–0.12
//   - Default:  500k photons,     radius 0.05
//   - Final:    1M–5M photons,    radius 0.02–0.05
constexpr int   DEFAULT_NUM_PHOTONS    = 5000000;
constexpr float DEFAULT_GATHER_RADIUS  = 0.1f;
constexpr float DEFAULT_CAUSTIC_RADIUS = 0.02f;

// Debug aid: true = photons stop after 1st hit (useful to validate emission)
constexpr bool  DEBUG_PHOTON_SINGLE_BOUNCE = false;

// ── Direct lighting (NEE) ───────────────────────────────────────────
// `DEFAULT_NEE_LIGHT_SAMPLES` affects the first bounce the most.
// Recommendation:
//   - Preview:  1
//   - Default:  4
//   - Final:    8–16 for softer shadows (scene dependent)
constexpr int   DEFAULT_NEE_LIGHT_SAMPLES = 8;
constexpr int   DEFAULT_NEE_DEEP_SAMPLES  = 1;  // bounces >= 1 (throughput attenuated)

// ── Integrator toggles ──────────────────────────────────────────────
// Recommendation:
//   - Leave MIS enabled for most scenes.
//   - Photon-guided sampling helps convergence once photon bins are populated.
constexpr bool  DEFAULT_USE_MIS           = true;
constexpr bool  DEFAULT_USE_PHOTON_GUIDED = true;

// ── Photon directional bins (guided NEE / caching) ──────────────────
// Higher counts can improve guidance but increase memory/time.
// Recommendation: 32 is a good default.
constexpr int   PHOTON_BIN_COUNT      = 32;   // directional bins per pixel (Fibonacci sphere)
constexpr float PHOTON_BIN_HORIZON_EPS = 0.05f;
constexpr int   PHOTON_BIN_NEE_TOP_K  = 4;    // top-K bins for guided NEE direction bias
// Compile-time upper bound for fixed-size arrays (must be >= PHOTON_BIN_COUNT).
constexpr int   MAX_PHOTON_BIN_COUNT  = 32;

// =====================================================================
// 2) ADVANCED (rarely changed; shared by tests/device)
// =====================================================================

// Stratified sub-pixel sampling grid.
// Only used when `DEFAULT_SPP == STRATA_X * STRATA_Y`.
constexpr int   STRATA_X = 4;
constexpr int   STRATA_Y = 4;

// Photon gather surface-consistency threshold (plane distance along normal).
// Higher = more photons accepted across nearby surfaces (smoother but more bias).
constexpr float DEFAULT_SURFACE_TAU = 0.02f;

// ── Scene profiles ──────────────────────────────────────────────────
// Each profile defines: OBJ path (relative to SCENES_DIR), camera
// position/look-at/FOV, and move speed suited to the scene scale.

// After normalisation every scene lives in approximately [-0.5, 0.5]³,
// so all camera/speed profiles now share the same coordinate system.

#ifdef SCENE_CORNELL_BOX
  constexpr const char* SCENE_OBJ_PATH        = "cornell_box/cornellbox.obj";
  constexpr const char* SCENE_DISPLAY_NAME     = "Cornell Box";
  constexpr bool  SCENE_IS_REFERENCE           = true;   // already in ref frame
  constexpr float SCENE_CAM_POS[]              = { 0.0f,  0.0f,  2.5f };
  constexpr float SCENE_CAM_LOOKAT[]           = { 0.0f,  0.0f,  0.0f };
  constexpr float SCENE_CAM_FOV                = 40.0f;
  constexpr float SCENE_CAM_SPEED              = 0.5f;

#elif defined(SCENE_CONFERENCE)
  constexpr const char* SCENE_OBJ_PATH        = "conference/conference.obj";
  constexpr const char* SCENE_DISPLAY_NAME     = "Conference Room";
  constexpr bool  SCENE_IS_REFERENCE           = false;
  constexpr float SCENE_CAM_POS[]              = { 0.0f,  0.0f,  2.5f };
  constexpr float SCENE_CAM_LOOKAT[]           = { 0.0f,  0.0f,  0.0f };
  constexpr float SCENE_CAM_FOV                = 50.0f;
  constexpr float SCENE_CAM_SPEED              = 0.5f;

#elif defined(SCENE_LIVING_ROOM)
  constexpr const char* SCENE_OBJ_PATH        = "living_room/living_room.obj";
  constexpr const char* SCENE_DISPLAY_NAME     = "Living Room";
  constexpr bool  SCENE_IS_REFERENCE           = false;
  constexpr float SCENE_CAM_POS[]              = { 0.0f,  0.0f,  2.5f };
  constexpr float SCENE_CAM_LOOKAT[]           = { 0.0f,  0.0f,  0.0f };
  constexpr float SCENE_CAM_FOV                = 50.0f;
  constexpr float SCENE_CAM_SPEED              = 0.5f;

#elif defined(SCENE_BREAKFAST_ROOM)
  constexpr const char* SCENE_OBJ_PATH        = "breakfast_room/breakfast_room.obj";
  constexpr const char* SCENE_DISPLAY_NAME     = "Breakfast Room";
  constexpr bool  SCENE_IS_REFERENCE           = false;
  constexpr float SCENE_CAM_POS[]              = { 0.0f,  0.0f,  2.5f };
  constexpr float SCENE_CAM_LOOKAT[]           = { 0.0f,  0.0f,  0.0f };
  constexpr float SCENE_CAM_FOV                = 50.0f;
  constexpr float SCENE_CAM_SPEED              = 0.5f;

#elif defined(SCENE_SIBENIK)
  constexpr const char* SCENE_OBJ_PATH        = "sibenik/sibenik.obj";
  constexpr const char* SCENE_DISPLAY_NAME     = "Sibenik Cathedral";
  constexpr bool  SCENE_IS_REFERENCE           = false;
  constexpr float SCENE_CAM_POS[]              = { 0.0f,  0.0f,  2.5f };
  constexpr float SCENE_CAM_LOOKAT[]           = { 0.0f,  0.0f,  0.0f };
  constexpr float SCENE_CAM_FOV                = 50.0f;
  constexpr float SCENE_CAM_SPEED              = 0.5f;

#else
  #error "No scene selected! Uncomment one SCENE_* define in config.h"
#endif

// ── Scene normalisation (Cornell Box = reference frame) ────────────
// The standard Cornell Box spans [-0.5, 0.5] on every axis, giving a
// unit cube centred at the origin.  All other scenes are scaled and
// translated to fit inside this reference volume so that camera
// defaults, gather radius, light placement, etc. transfer directly.
constexpr float SCENE_REF_EXTENT           = 1.0f;   // target longest-axis extent
constexpr float SCENE_REF_CENTER[]         = { 0.0f, 0.0f, 0.0f };

