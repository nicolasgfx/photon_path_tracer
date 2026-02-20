#pragma once
// ─────────────────────────────────────────────────────────────────────
// config.h – Central configuration constants for the renderer
// ─────────────────────────────────────────────────────────────────────
// All tunable parameters live here so they can be adjusted in one
// place.  Compile-time constants use constexpr; run-time defaults are
// plain constants that feed into RenderConfig / EmitterConfig.
// ─────────────────────────────────────────────────────────────────────

#include <cstdint>

// ── Active scene (change this to switch) ────────────────────────────
// Uncomment exactly ONE of these:
 //#define SCENE_CORNELL_BOX
#define SCENE_CONFERENCE
//#define SCENE_LIVING_ROOM
 //#define SCENE_BREAKFAST_ROOM
// #define SCENE_SIBENIK

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

// ── Window & display ────────────────────────────────────────────────
constexpr int   DEFAULT_WINDOW_WIDTH       = 1024;
constexpr int   DEFAULT_WINDOW_HEIGHT      = 768;

// ── Image output ────────────────────────────────────────────────────
constexpr int   DEFAULT_IMAGE_WIDTH        = 1024;
constexpr int   DEFAULT_IMAGE_HEIGHT       = 768;

// ── Sampling ────────────────────────────────────────────────────────
constexpr int   DEFAULT_SPP                = 16;     // samples per pixel
constexpr int   DEFAULT_MAX_BOUNCES        = 8;
constexpr int   DEFAULT_MIN_BOUNCES_RR     = 3;      // start Russian roulette after this
constexpr float DEFAULT_RR_THRESHOLD       = 0.95f;

// ── Photon mapping ──────────────────────────────────────────────────
constexpr int DEFAULT_NUM_PHOTONS        = 1000000;
constexpr float DEFAULT_GATHER_RADIUS      = 0.05f;
constexpr float DEFAULT_CAUSTIC_RADIUS     = 0.02f;
constexpr bool  DEBUG_PHOTON_SINGLE_BOUNCE = false;  // true = photons stop after 1st hit (debugging)
// DEBUG_CAMERA_SINGLE_BOUNCE removed: use RenderMode::FirstHitOnly instead

// ── NEE (Next Event Estimation) ─────────────────────────────────────
constexpr int   DEFAULT_NEE_LIGHT_SAMPLES  = 4;   // M: shadow-ray samples at bounce 0 (1=fast, 4-16=soft)
constexpr int   DEFAULT_NEE_DEEP_SAMPLES   = 1;    // shadow-ray samples at bounce >= 1 (throughput-attenuated)

// ── MIS ─────────────────────────────────────────────────────────────
constexpr bool  DEFAULT_USE_MIS            = true;
constexpr bool  DEFAULT_USE_PHOTON_GUIDED  = true;

// ── Hash grid ───────────────────────────────────────────────────────
constexpr float HASHGRID_CELL_FACTOR       = 2.0f;  // cell_size = factor * radius
constexpr int   HASHGRID_NEIGHBOUR_RANGE   = 1;     // ±1 → 3×3×3 query
constexpr uint32_t HASHGRID_PRIME_1        = 73856093u;
constexpr uint32_t HASHGRID_PRIME_2        = 19349663u;
constexpr uint32_t HASHGRID_PRIME_3        = 83492791u;

// ── Density estimator ───────────────────────────────────────────────
constexpr float DEFAULT_SURFACE_TAU        = 0.02f;  // plane-distance filter
constexpr bool  DEFAULT_USE_KERNEL         = true;
constexpr float PHOTON_SHADOW_FLOOR        = 0.1f;   // min visibility weight for photon gather
                                                      // 0 = photons fully suppressed in shadow
                                                      // 1 = no suppression (old behaviour)

// ── Camera ──────────────────────────────────────────────────────────
constexpr float DEFAULT_CORNELL_FOV        = 40.0f;  // degrees (vertical)

// ── Scene normalisation (Cornell Box = reference frame) ────────────
// The standard Cornell Box spans [-0.5, 0.5] on every axis, giving a
// unit cube centred at the origin.  All other scenes are scaled and
// translated to fit inside this reference volume so that camera
// defaults, gather radius, light placement, etc. transfer directly.
constexpr float SCENE_REF_EXTENT           = 1.0f;   // target longest-axis extent
constexpr float SCENE_REF_CENTER[]         = { 0.0f, 0.0f, 0.0f };

// ── Scene / geometry ────────────────────────────────────────────────
constexpr float DEFAULT_RAY_TMIN           = 1e-4f;
constexpr float DEFAULT_RAY_TMAX           = 1e20f;

// ── Camera movement ─────────────────────────────────────────────────
constexpr float DEFAULT_CAM_MOVE_SPEED     = 1.0f;    // units/sec
constexpr float DEFAULT_CAM_MOUSE_SENS     = 0.0005f; // radians/pixel

// ── OptiX ───────────────────────────────────────────────────────────
constexpr int   OPTIX_NUM_PAYLOAD_VALUES   = 14;    // payloads per trace call
constexpr int   OPTIX_NUM_ATTRIBUTE_VALUES = 2;     // barycentrics
constexpr int   OPTIX_MAX_TRACE_DEPTH      = 2;     // radiance + shadow
constexpr int   OPTIX_STACK_SIZE           = 2048;
constexpr float OPTIX_SCENE_EPSILON        = 1e-4f;

// ── CUDA ────────────────────────────────────────────────────────────
constexpr int   CUDA_BLOCK_SIZE_1D         = 256;
constexpr int   CUDA_BLOCK_SIZE_2D         = 16;    // 16×16 = 256 threads

// ── Output ──────────────────────────────────────────────────────────
constexpr const char* DEFAULT_OUTPUT_DIR   = "output";
constexpr const char* DEFAULT_OUTPUT_EXT   = ".png";

