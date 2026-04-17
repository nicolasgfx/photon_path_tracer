#pragma once
// ─────────────────────────────────────────────────────────────────────
// app/render_config.h – Runtime render configuration (v5, RGB)
//
// All tunable rendering parameters live here.  Populated from:
//   1. config.h DEFAULT_* constants  (compile-time fallbacks)
//   2. JSON config file              (--config path.json)
//   3. CLI arguments                 (highest priority)
//
// Consumed by all pipeline stages.  GPU clamping values are forwarded
// to LaunchParams by render_session.cpp.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/color.h"
#include "core/config.h"
#include "postfx/postfx_params.h"

#include <cstdint>
#include <string>
#include <vector>

// ── Render configuration ────────────────────────────────────────────

struct RenderConfig {
    // ── Image output ────────────────────────────────────────────────
    int    image_width       = DEFAULT_IMAGE_WIDTH;
    int    image_height      = DEFAULT_IMAGE_HEIGHT;

    // ── Core rendering ──────────────────────────────────────────────
    int    samples_per_pixel = DEFAULT_SPP;
    int    max_bounces       = DEFAULT_MAX_BOUNCES_CAMERA;
    int    min_bounces_rr    = DEFAULT_MIN_BOUNCES_RR;
    float  rr_threshold      = DEFAULT_RR_THRESHOLD;
    int    max_specular_chain = DEFAULT_MAX_SPECULAR_CHAIN;
    int    spp_per_launch    = 1;           // SPP per GPU kernel launch (higher = less sync overhead)

    // ── Clamping (forwarded to LaunchParams for GPU) ────────────────
    bool   clamping_enabled        = DEFAULT_CLAMPING_ENABLED;
    float  max_bounce_contribution = DEFAULT_MAX_BOUNCE_CONTRIBUTION;
    float  max_path_throughput     = DEFAULT_MAX_PATH_THROUGHPUT;
    float  max_nee_contribution    = DEFAULT_MAX_NEE_CONTRIBUTION;
    float  max_sample_luminance    = DEFAULT_MAX_SAMPLE_LUMINANCE;

    // ── Adaptive sampling ───────────────────────────────────────────
    bool   adaptive_sampling        = DEFAULT_ADAPTIVE_SAMPLING;
    int    adaptive_min_spp         = ADAPTIVE_MIN_SPP;
    float  adaptive_threshold       = ADAPTIVE_THRESHOLD;
    int    adaptive_radius          = ADAPTIVE_RADIUS;

    // ── View-dependent pre-pass ─────────────────────────────────────
    int    prepass_spp              = 8;          // SPP for pre-pass (0 = disabled)
    int    prepass_scale_divisor    = 4;          // resolution = full / divisor

    // ── Tone mapping ────────────────────────────────────────────────
    float  exposure           = DEFAULT_EXPOSURE;
    float  light_scale        = DEFAULT_LIGHT_SCALE;

    // ── OptiX AI Denoiser ───────────────────────────────────────────
    bool   denoiser_enabled       = DEFAULT_DENOISER_ENABLED;
    bool   denoiser_guide_albedo  = DEFAULT_DENOISER_GUIDE_ALBEDO;
    bool   denoiser_guide_normal  = DEFAULT_DENOISER_GUIDE_NORMAL;
    float  denoiser_blend         = DEFAULT_DENOISER_BLEND;

    // ── Post-processing effects ─────────────────────────────────────
    PostFxParams postfx;

    // ── Depth of field ──────────────────────────────────────────────
    bool   dof_enabled         = DEFAULT_DOF_ENABLED;
    float  dof_focus_distance  = DEFAULT_DOF_FOCUS_DISTANCE;
    float  dof_f_number        = DEFAULT_DOF_F_NUMBER;

    // ── Caustic light tracing ────────────────────────────────────────
    bool   caustic_enabled              = DEFAULT_CAUSTIC_ENABLED;
    int    caustic_photons_per_frame    = 0;       // 0 = use SceneProfile recommendation
    float  caustic_max_splat_luminance  = DEFAULT_CAUSTIC_MAX_SPLAT_LUMINANCE;

    // ── Light tree (importance-driven emitter sampling) ────────────
    bool   light_tree_enabled        = DEFAULT_LIGHT_TREE_ENABLED;
    int    light_tree_max_leaf_size  = DEFAULT_LIGHT_TREE_MAX_LEAF_SIZE;

    // ── Debug ───────────────────────────────────────────────────────
    RenderMode mode            = RenderMode::Combined;
    bool  bounce_aov_enabled   = false;

    // ── JSON I/O ────────────────────────────────────────────────────
    // Load from a JSON file.  Only fields present in the file are
    // overwritten; absent fields keep their current values.
    // Returns true on success, false if file cannot be read.
    bool load_json(const std::string& path);

    // Save the full config to a JSON file (for agent round-tripping).
    bool save_json(const std::string& path) const;
};

// ── CPU-side framebuffer (RGB, v5) ──────────────────────────────────
// For display / PNG output. GPU rendering writes directly to
// DeviceBuffer<float>, then downloads here for display.

struct FrameBuffer {
    int width  = 0;
    int height = 0;
    std::vector<float>    rgb;          // Linear RGB accumulator (w×h×3)
    std::vector<float>    sample_count; // Per-pixel sample count
    std::vector<uint8_t>  srgb;         // Final sRGB output (RGBA, w×h×4)

    void resize(int w, int h) {
        width  = w;
        height = h;
        rgb.resize(w * h * 3, 0.f);
        sample_count.resize(w * h, 0.f);
        srgb.resize(w * h * 4, 0);
    }

    void clear() {
        std::fill(rgb.begin(), rgb.end(), 0.f);
        std::fill(sample_count.begin(), sample_count.end(), 0.f);
    }

    int num_pixels() const { return width * height; }

    // Get averaged pixel color at (x,y)
    Color3 get_pixel(int x, int y) const {
        int idx = (y * width + x) * 3;
        float n = sample_count[y * width + x];
        if (n < 1.f) return Color3::zero();
        float inv = 1.f / n;
        return Color3::from_rgb(rgb[idx] * inv, rgb[idx+1] * inv, rgb[idx+2] * inv);
    }

    // Tonemap RGB accumulator → sRGB output buffer
    void tonemap(float exposure = 1.0f) {
        for (int i = 0; i < width * height; ++i) {
            float n = sample_count[i];
            float r = 0.f, g = 0.f, b = 0.f;
            if (n > 0.f) {
                float inv = exposure / n;
                r = rgb[i * 3 + 0] * inv;
                g = rgb[i * 3 + 1] * inv;
                b = rgb[i * 3 + 2] * inv;
            }

            // ACES tonemap
            if (USE_ACES_TONEMAPPING) {
                auto aces = [](float x) {
                    float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
                    return (x * (a * x + b)) / (x * (c * x + d) + e);
                };
                r = aces(r);
                g = aces(g);
                b = aces(b);
            }

            // Linear → sRGB gamma
            auto to_srgb = [](float v) -> uint8_t {
                v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
                float s = v <= 0.0031308f
                    ? v * 12.92f
                    : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
                return (uint8_t)(s * 255.f + 0.5f);
            };

            srgb[i * 4 + 0] = to_srgb(r);
            srgb[i * 4 + 1] = to_srgb(g);
            srgb[i * 4 + 2] = to_srgb(b);
            srgb[i * 4 + 3] = 255;
        }
    }
};
