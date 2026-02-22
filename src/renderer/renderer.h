#pragma once
// ─────────────────────────────────────────────────────────────────────
// renderer.h – Top-level render pipeline
// ─────────────────────────────────────────────────────────────────────
// Orchestrates the full photon + path tracing pipeline:
//   1. Build scene → BVH + emissive distribution
//   2. Photon pass → global map + caustic map
//   3. Build hash grids
//   4. Camera pass → spectral image   (CPU fallback or OptiX)
//   5. Tonemap → sRGB framebuffer
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"
#include "core/config.h"
#include "renderer/camera.h"
#include "scene/scene.h"
#include "photon/photon.h"
#include "photon/hash_grid.h"
#include "photon/kd_tree.h"
#include "photon/density_estimator.h"
#include "core/sppm.h"

#include <vector>
#include <string>

// ── Render configuration ────────────────────────────────────────────

enum class RenderMode {
    Combined,            // Direct (NEE) + Indirect (photon map)  [default]
    Full = Combined,     // Legacy alias for Combined
    DirectOnly,          // NEE direct lighting only (no photon gather)
    IndirectOnly,        // Photon map indirect only (no NEE)
    PhotonMap,           // Visualize photon density (heatmap)
    Normals,             // Debug: surface normals
    MaterialID,          // Debug: material colors
    Depth                // Debug: depth buffer
};

struct RenderConfig {
    int    image_width       = DEFAULT_IMAGE_WIDTH;
    int    image_height      = DEFAULT_IMAGE_HEIGHT;
    int    samples_per_pixel = DEFAULT_SPP;
    int    max_bounces       = DEFAULT_MAX_BOUNCES;
    int    min_bounces_rr    = DEFAULT_MIN_BOUNCES_RR;
    float  rr_threshold      = DEFAULT_RR_THRESHOLD;

    // Photon mapping
    int    num_photons       = DEFAULT_NUM_PHOTONS;
    int    global_photon_budget  = DEFAULT_GLOBAL_PHOTON_BUDGET;   // §Q4
    int    caustic_photon_budget = DEFAULT_CAUSTIC_PHOTON_BUDGET;  // §Q4
    float  gather_radius     = DEFAULT_GATHER_RADIUS;
    float  caustic_radius    = DEFAULT_CAUSTIC_RADIUS;

    // MIS and guided-BSDF (v1 features, removed in v2.1)
    // These fields are kept for API compatibility only — they have no
    // effect; the renderer always uses the v2.1 first-hit + photon pipeline.
    // TODO: Remove in a future breaking-API cleanup.

    // Camera specular chain (§E3)
    int    max_specular_chain = DEFAULT_MAX_SPECULAR_CHAIN;

    // KD-tree adaptive radius (k-NN, §C2)
    int    knn_k              = DEFAULT_KNN_K;
    bool   use_kdtree         = true;   // primary spatial index
    bool   use_knn_adaptive   = false;  // k-NN adaptive gather radius (§C2)

    // NEE coverage fraction (§7.2.1)
    float  nee_coverage_fraction = DEFAULT_NEE_COVERAGE_FRACTION;

    // Participating medium (volumetric scattering)
    bool   volume_enabled     = DEFAULT_VOLUME_ENABLED;
    float  volume_density     = DEFAULT_VOLUME_DENSITY;
    float  volume_falloff     = DEFAULT_VOLUME_FALLOFF;
    float  volume_albedo      = DEFAULT_VOLUME_ALBEDO;
    int    volume_samples     = DEFAULT_VOLUME_SAMPLES;
    float  volume_max_t       = DEFAULT_VOLUME_MAX_T;

    // SPPM (Stochastic Progressive Photon Mapping)
    bool  sppm_enabled           = false;
    int   sppm_iterations        = DEFAULT_SPPM_ITERATIONS;
    float sppm_alpha             = DEFAULT_SPPM_ALPHA;
    float sppm_initial_radius    = DEFAULT_SPPM_INITIAL_RADIUS;
    float sppm_min_radius        = DEFAULT_SPPM_MIN_RADIUS;

    // Adaptive sampling
    bool  adaptive_sampling      = false;
    int   adaptive_min_spp       = 4;      ///< warmup passes (uniform)
    int   adaptive_max_spp       = 0;      ///< 0 = use samples_per_pixel
    int   adaptive_update_interval = 1;   ///< recompute mask every N passes
    float adaptive_threshold     = 0.02f; ///< relative-noise threshold to keep sampling
    int   adaptive_radius        = 1;     ///< neighbourhood half-width (pixels)

    // Debug
    RenderMode mode          = RenderMode::Combined;
};

// ── Spectral framebuffer ────────────────────────────────────────────

struct FrameBuffer {
    int width, height;
    std::vector<Spectrum> pixels;       // Spectral radiance accumulator
    std::vector<float>    sample_count; // Per-pixel sample count
    std::vector<float>    lum_sum;      // Σ Y_i  (linear luminance, for adaptive)
    std::vector<float>    lum_sum2;     // Σ Y_i² (for variance estimate)
    std::vector<uint8_t>  srgb;         // Final sRGB output (RGBA)

    void resize(int w, int h) {
        width  = w;
        height = h;
        pixels.resize(w * h, Spectrum::zero());
        sample_count.resize(w * h, 0.f);
        lum_sum.resize(w * h, 0.f);
        lum_sum2.resize(w * h, 0.f);
        srgb.resize(w * h * 4, 0);
    }

    void clear() {
        for (auto& p : pixels)       p = Spectrum::zero();
        for (auto& s : sample_count) s = 0.f;
        for (auto& v : lum_sum)      v = 0.f;
        for (auto& v : lum_sum2)     v = 0.f;
    }

    // proxy_L: if non-null, use this spectrum (e.g. NEE direct only) for the
    // luminance noise estimate instead of the full combined L.
    // Controlled by ADAPTIVE_NOISE_USE_DIRECT_ONLY in config.h.
    void accumulate(int x, int y, const Spectrum& L,
                    const Spectrum* proxy_L = nullptr) {
        int idx = y * width + x;
        pixels[idx] += L;
        sample_count[idx] += 1.f;
        // Luminance moments for adaptive sampling noise estimate
        const Spectrum& lum_src = (proxy_L != nullptr) ? *proxy_L : L;
        float3 xyz = spectrum_to_xyz(lum_src);
        float  Y   = xyz.y;               // CIE Y = linear luminance
        lum_sum[idx]  += Y;
        lum_sum2[idx] += Y * Y;
    }

    // Tonemap and convert to sRGB (ACES Filmic, §Q8)
    void tonemap(float exposure = 1.0f) {
        for (int i = 0; i < width * height; ++i) {
            Spectrum avg = (sample_count[i] > 0.f)
                ? pixels[i] / sample_count[i]
                : Spectrum::zero();

            // Apply exposure
            avg *= exposure;

            // Spectrum → sRGB with ACES tone mapping
            float3 rgb = USE_ACES_TONEMAPPING
                ? spectrum_to_srgb_aces(avg)
                : spectrum_to_srgb(avg);

            // Clamp
            rgb.x = fminf(fmaxf(rgb.x, 0.f), 1.f);
            rgb.y = fminf(fmaxf(rgb.y, 0.f), 1.f);
            rgb.z = fminf(fmaxf(rgb.z, 0.f), 1.f);

            srgb[i * 4 + 0] = (uint8_t)(rgb.x * 255.f);
            srgb[i * 4 + 1] = (uint8_t)(rgb.y * 255.f);
            srgb[i * 4 + 2] = (uint8_t)(rgb.z * 255.f);
            srgb[i * 4 + 3] = 255;
        }
    }
};

// ── Renderer ────────────────────────────────────────────────────────

class Renderer {
public:
    // Setup
    void set_scene(Scene* scene) { scene_ = scene; }
    void set_camera(const Camera& cam) { camera_ = cam; }
    void set_config(const RenderConfig& cfg) { config_ = cfg; }

    // Pipeline stages
    void build_photon_maps();
    void render_frame();

    /// SPPM rendering: iterative photon tracing + per-pixel progressive
    /// radius shrinking.  Uses the existing photon maps and hash grids.
    void render_sppm();

    // Access results
    FrameBuffer&       framebuffer()       { return fb_; }
    const FrameBuffer& framebuffer() const { return fb_; }
    const RenderConfig& config() const { return config_; }

    // Photon map stats
    size_t global_photon_count()  const { return global_photons_.size(); }
    size_t caustic_photon_count() const { return caustic_photons_.size(); }

    // Debug access
    const PhotonSoA& global_photons()  const { return global_photons_; }
    const PhotonSoA& caustic_photons() const { return caustic_photons_; }
    const HashGrid&  global_grid()     const { return global_grid_; }
    const HashGrid&  caustic_grid()    const { return caustic_grid_; }

    // Result of a single camera ray (first-hit + specular chain)
    struct TraceResult {
        Spectrum combined;    ///< full radiance (NEE + photon + emission)
        Spectrum nee_direct;  ///< direct-lighting-only component
    };

    // v2: First-hit camera ray with specular chain (§E3).
    // Follows specular bounces, then NEE + photon gather at first diffuse hit.
    TraceResult render_pixel(Ray ray, PCGRng& rng);

    // Legacy: multi-bounce trace_path (kept for backward compat, wraps render_pixel)
    TraceResult trace_path(Ray ray, PCGRng& rng);

private:

    // Debug render modes
    Spectrum render_normals(const HitRecord& hit);
    Spectrum render_material_id(const HitRecord& hit);
    Spectrum render_depth(const HitRecord& hit, float max_depth);
    Spectrum render_photon_density(const HitRecord& hit, float3 wo_world);

    Scene*       scene_  = nullptr;
    Camera       camera_;
    RenderConfig config_;
    FrameBuffer  fb_;

    // Photon maps
    PhotonSoA    global_photons_;
    PhotonSoA    caustic_photons_;
    HashGrid     global_grid_;
    HashGrid     caustic_grid_;
    KDTree       global_kdtree_;     // v2: primary spatial index
    KDTree       caustic_kdtree_;    // v2: caustic KD-tree
};
