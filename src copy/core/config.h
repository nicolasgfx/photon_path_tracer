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
 #define SCENE_CORNELL_BOX
//#define SCENE_CONFERENCE
// #define SCENE_LIVING_ROOM
// #define SCENE_BREAKFAST_ROOM
// #define SCENE_SIBENIK

// ── Scene profiles ──────────────────────────────────────────────────
// Each profile defines: OBJ path (relative to SCENES_DIR), camera
// position/look-at/FOV, and move speed suited to the scene scale.

#ifdef SCENE_CORNELL_BOX
  constexpr const char* SCENE_OBJ_PATH        = "cornell_box/cornellbox.obj";
  constexpr const char* SCENE_DISPLAY_NAME     = "Cornell Box";
  constexpr float SCENE_CAM_POS[]              = { 0.0f,  0.0f,  2.5f };
  constexpr float SCENE_CAM_LOOKAT[]           = { 0.0f,  0.0f,  0.0f };
  constexpr float SCENE_CAM_FOV                = 40.0f;
  constexpr float SCENE_CAM_SPEED              = 0.5f;
  // Compatibility: FULL — all MTL features supported, has emitter.

#elif defined(SCENE_CONFERENCE)
  constexpr const char* SCENE_OBJ_PATH        = "conference/conference.obj";
  constexpr const char* SCENE_DISPLAY_NAME     = "Conference Room";
  constexpr float SCENE_CAM_POS[]              = { 10.0f, 4.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]           = { 0.0f,  3.0f, 0.0f };
  constexpr float SCENE_CAM_FOV                = 60.0f;
  constexpr float SCENE_CAM_SPEED              = 2.0f;
  // Compatibility: GOOD — no textures needed, has Ke emitter (ceiling
  //   lights). All illum 2 → Lambertian or GlossyMetal. Ka ignored (OK).

#elif defined(SCENE_LIVING_ROOM)
  constexpr const char* SCENE_OBJ_PATH        = "living_room/living_room.obj";
  constexpr const char* SCENE_DISPLAY_NAME     = "Living Room";
  constexpr float SCENE_CAM_POS[]              = { 1.5f,  1.5f,  3.0f };
  constexpr float SCENE_CAM_LOOKAT[]           = { 0.0f,  1.0f,  0.0f };
  constexpr float SCENE_CAM_FOV                = 55.0f;
  constexpr float SCENE_CAM_SPEED              = 1.0f;
  // Compatibility: PARTIAL — has Ke emitters (4 lights), but heavily
  //   texture-dependent (map_Kd). Without texture loading, many surfaces
  //   render as flat Kd color. Backslash texture paths may need fixing.

#elif defined(SCENE_BREAKFAST_ROOM)
  constexpr const char* SCENE_OBJ_PATH        = "breakfast_room/breakfast_room.obj";
  constexpr const char* SCENE_DISPLAY_NAME     = "Breakfast Room";
  constexpr float SCENE_CAM_POS[]              = { 0.0f,  1.5f,  3.0f };
  constexpr float SCENE_CAM_LOOKAT[]           = { 0.0f,  1.0f,  0.0f };
  constexpr float SCENE_CAM_FOV                = 60.0f;
  constexpr float SCENE_CAM_SPEED              = 1.0f;
  // Compatibility: LOW — NO emitters (will be dark, auto-light added).
  //   All materials illum 4 → misclassified as Glass. Tf unsupported.
  //   Textures (map_Kd) not loaded. Needs work.

#elif defined(SCENE_SIBENIK)
  constexpr const char* SCENE_OBJ_PATH        = "sibenik/sibenik.obj";
  constexpr const char* SCENE_DISPLAY_NAME     = "Sibenik Cathedral";
  constexpr float SCENE_CAM_POS[]              = { -8.0f, -2.0f, 0.0f };
  constexpr float SCENE_CAM_LOOKAT[]           = { 0.0f,  5.0f,  0.0f };
  constexpr float SCENE_CAM_FOV                = 60.0f;
  constexpr float SCENE_CAM_SPEED              = 3.0f;
  // Compatibility: LOW — NO emitters (will be dark, auto-light added).
  //   Uses Tf, Tr, map_bump, bump — all unsupported. Stained glass
  //   windows lose color. Textures not loaded. Needs work.

#else
  #error "No scene selected! Uncomment one SCENE_* define in config.h"
#endif

// ── Window & display ────────────────────────────────────────────────
constexpr int   DEFAULT_WINDOW_WIDTH       = 1920;
constexpr int   DEFAULT_WINDOW_HEIGHT      = 1024;

// ── Image output ────────────────────────────────────────────────────
constexpr int   DEFAULT_IMAGE_WIDTH        = 1920;
constexpr int   DEFAULT_IMAGE_HEIGHT       = 1024;

// ── Sampling ────────────────────────────────────────────────────────
constexpr int   DEFAULT_SPP                = 16;     // samples per pixel
constexpr int   DEFAULT_MAX_BOUNCES        = 8;
constexpr int   DEFAULT_MIN_BOUNCES_RR     = 3;      // start Russian roulette after this
constexpr float DEFAULT_RR_THRESHOLD       = 0.95f;

// ── Photon mapping ──────────────────────────────────────────────────
constexpr int   DEFAULT_NUM_PHOTONS        = 500000;
constexpr float DEFAULT_GATHER_RADIUS      = 0.05f;
constexpr float DEFAULT_CAUSTIC_RADIUS     = 0.02f;

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

// ── Camera ──────────────────────────────────────────────────────────
constexpr float DEFAULT_CORNELL_FOV        = 40.0f;  // degrees (vertical)

// ── Scene / geometry ────────────────────────────────────────────────
constexpr float DEFAULT_RAY_TMIN           = 1e-4f;
constexpr float DEFAULT_RAY_TMAX           = 1e20f;

// ── Debug overlay ───────────────────────────────────────────────────
constexpr float DEBUG_PHOTON_BRIGHTNESS    = 2.0f;

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

