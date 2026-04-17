#pragma once
#include <cstdint>
// ─────────────────────────────────────────────────────────────────────
// config.h – Central configuration for the decomposed renderer (v5)
//
// TWO-TIER CONFIGURATION:
//
//   Tier 1 (Runtime):  All tunable rendering parameters live in
//       RenderConfig (render_config.h).  They can be overridden via
//       a JSON config file (--config path.json) or CLI args at
//       startup — no recompilation needed.  The DEFAULT_* constants
//       below are compile-time fallbacks used when a JSON field is
//       absent.
//
//   Tier 2 (Compile-time):  Structural constants that affect data
//       layout, OptiX pipeline config, or array sizes stay here as
//       constexpr.  Changing these requires a rebuild.
//
// [R] = runtime-overridable (RenderConfig field)
// [C] = compile-time only (constexpr, rebuild required)
// ─────────────────────────────────────────────────────────────────────


// =====================================================================
//  §0  COMPILE-TIME STRUCTURAL CONSTANTS  [C]
// =====================================================================

// ── OptiX pipeline layout ───────────────────────────────────────────
constexpr int   OPTIX_MAX_TRACE_DEPTH        = 2;      // [C] radiance + shadow
constexpr int   OPTIX_STACK_SIZE             = 16384;  // [C] OptiX stack bytes

// ── Array-size constants ────────────────────────────────────────────
constexpr int   MAX_AOV_BOUNCES              = 4;      // [C] debug AOV bounce limit


// =====================================================================
//  §1  RUNTIME DEFAULTS  (used as RenderConfig initializers)  [R]
//
//  These are the fallback values when no JSON config or CLI arg is
//  provided.  Skills / agents change rendering behavior by writing
//  a JSON config file — NOT by editing these constants.
// =====================================================================

// ── Image output ────────────────────────────────────────────────────
constexpr int   DEFAULT_IMAGE_WIDTH          = 1440;
constexpr int   DEFAULT_IMAGE_HEIGHT         = 1440;

// ── Core rendering ──────────────────────────────────────────────────
constexpr int   DEFAULT_SPP                  = 256;
constexpr int   DEFAULT_MAX_BOUNCES_CAMERA   = 8;
constexpr int   DEFAULT_MIN_BOUNCES_RR       = 3;
constexpr float DEFAULT_RR_THRESHOLD         = 0.95f;
constexpr int   DEFAULT_MAX_SPECULAR_CHAIN   = 8;

// ── Preview mode ────────────────────────────────────────────────────
constexpr int   PREVIEW_MAX_BOUNCES          = 2;
constexpr float IDLE_TIMEOUT_SEC             = 1.0f;

// ── Clamping (safety nets — now piped to GPU via LaunchParams) ──────
constexpr bool  DEFAULT_CLAMPING_ENABLED        = true;
constexpr float DEFAULT_MAX_BOUNCE_CONTRIBUTION = 1e4f;
constexpr float DEFAULT_MAX_PATH_THROUGHPUT     = 1e4f;
constexpr float DEFAULT_MAX_NEE_CONTRIBUTION    = 1e4f;
constexpr float DEFAULT_MAX_SAMPLE_LUMINANCE    = 10000.f;



// ── Post-processing ─────────────────────────────────────────────────
constexpr bool  USE_ACES_TONEMAPPING             = false;
constexpr float DEFAULT_EXPOSURE                 = 1.0f;
constexpr float DEFAULT_LIGHT_SCALE              = 1.0f;
constexpr float LIGHT_SCALE_STEP                 = 1.25f;
constexpr float LIGHT_SCALE_MIN                  = 0.01f;
constexpr float LIGHT_SCALE_MAX                  = 100.0f;
constexpr bool  DEFAULT_DENOISER_ENABLED         = false;
constexpr bool  DEFAULT_DENOISER_GUIDE_ALBEDO    = true;
constexpr bool  DEFAULT_DENOISER_GUIDE_NORMAL    = true;
constexpr float DEFAULT_DENOISER_BLEND           = 0.0f;
constexpr bool  DEFAULT_FIREFLY_FILTER_ENABLED   = true;
constexpr int   FIREFLY_FILTER_RADIUS            = 1;
constexpr float FIREFLY_FILTER_THRESHOLD         = 4.0f;
constexpr bool  DEFAULT_BLOOM_ENABLED            = false;
constexpr float DEFAULT_BLOOM_INTENSITY          = 0.5f;
constexpr float DEFAULT_BLOOM_RADIUS_H           = 15.0f;
constexpr float DEFAULT_BLOOM_RADIUS_V           = 15.0f;

// ── Depth of field ──────────────────────────────────────────────────
constexpr bool  DEFAULT_DOF_ENABLED              = false;
constexpr float DEFAULT_DOF_FOCUS_DISTANCE       = 0.1f;
constexpr float DEFAULT_DOF_F_NUMBER             = 8.0f;
constexpr float DEFAULT_DOF_SENSOR_HEIGHT        = 0.024f;
constexpr float DEFAULT_DOF_FOCUS_RANGE          = 0.05f;

// ── Caustic light tracing ────────────────────────────────────────────
constexpr bool  DEFAULT_CAUSTIC_ENABLED            = true;
constexpr int   DEFAULT_CAUSTIC_PHOTONS_PER_FRAME  = 262144;  // 256K
constexpr float DEFAULT_CAUSTIC_MAX_SPLAT_LUMINANCE = 100.f;

// ── Light tree (importance-driven emitter sampling) ──────────────────
constexpr bool  DEFAULT_LIGHT_TREE_ENABLED         = true;
constexpr int   DEFAULT_LIGHT_TREE_MAX_LEAF_SIZE   = 4;

// ── Adaptive sampling ───────────────────────────────────────────────
constexpr bool  DEFAULT_ADAPTIVE_SAMPLING        = false;
constexpr int   ADAPTIVE_MIN_SPP                 = 4;
constexpr int   ADAPTIVE_UPDATE_INTERVAL         = 1;
constexpr float ADAPTIVE_THRESHOLD               = 0.02f;
constexpr int   ADAPTIVE_RADIUS                  = 1;

// ── Scene epsilon ───────────────────────────────────────────────────
constexpr float OPTIX_SCENE_EPSILON              = 1e-4f;


// =====================================================================
//  §13  SCENE PROFILES (runtime scene selection)
// =====================================================================

enum class SceneLightMode {
    FromMTL,
    DirectionalToFloor,
    HemisphereEnv,
    SphericalEnv,
};

// Scene profile for runtime selection (keys 1–9, 0).
// NOTE: In the new architecture, SceneProfile (scene_profile.h) is the
// AI-analysed classification struct used for convergence tuning.  This
// struct is just the UI scene selector — will be renamed to ScenePreset
// during migration to avoid confusion.
struct ScenePreset {
    const char*    obj_path;
    const char*    display_name;
    bool           is_reference;
    float          cam_pos[3];
    float          cam_lookat[3];
    float          cam_fov;
    float          cam_speed;
    SceneLightMode light_mode;
    bool           rotate_x_180;
};

constexpr int NUM_SCENE_PRESETS = 4;

// Indices 0-9 → keys 1-9,0.  Index 10+ → Shift+1, Shift+2, ...
inline constexpr char scene_hotkey_char(int idx) {
    if (idx < 9)  return char('1' + idx);
    if (idx == 9) return '0';
    return char('!' + (idx - 10));  // Shift+1='!', Shift+2='@', ...
}

constexpr ScenePreset SCENE_PRESETS[NUM_SCENE_PRESETS] = {
    { "cornell_glass_boxes/scene-v4.pbrt",       "Cornell Glass Boxes", false,
      {0,0,0}, {0,0,-1}, 50.f, 0.1f, SceneLightMode::FromMTL, false },
    { "veach-bidir/scene-v4.pbrt",               "Veach Bidir",       false,
      {0,0,0}, {0,0,-1}, 50.f, 0.1f, SceneLightMode::FromMTL, false },
    { "staircase/scene-v4.pbrt",                 "Staircase",         false,
      {0,0,0}, {0,0,-1}, 90.f, 0.1f, SceneLightMode::FromMTL, false },
    { "staircase2/scene-v4.pbrt",                "Staircase 2",       false,
      {0,0,0}, {0,0,-1}, 70.f, 0.1f, SceneLightMode::FromMTL, false },
};

inline const ScenePreset& get_scene_preset(int idx) {
    return SCENE_PRESETS[idx];
}

