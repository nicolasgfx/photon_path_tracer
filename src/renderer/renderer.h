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
#include "photon/density_estimator.h"

#include <vector>
#include <string>

// ── Render configuration ────────────────────────────────────────────

enum class RenderMode {
    Full,                // Complete photon + path tracing
    DirectOnly,          // Direct illumination only
    IndirectOnly,        // Photon map indirect only
    PhotonMap,           // Visualize photon density
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
    float  gather_radius     = DEFAULT_GATHER_RADIUS;
    float  caustic_radius    = DEFAULT_CAUSTIC_RADIUS;

    // MIS
    bool   use_mis           = DEFAULT_USE_MIS;
    bool   use_photon_guided = DEFAULT_USE_PHOTON_GUIDED;

    // Debug
    RenderMode mode          = RenderMode::Full;
};

// ── Spectral framebuffer ────────────────────────────────────────────

struct FrameBuffer {
    int width, height;
    std::vector<Spectrum> pixels;       // Spectral radiance accumulator
    std::vector<float>    sample_count; // Per-pixel sample count
    std::vector<uint8_t>  srgb;         // Final sRGB output (RGBA)

    void resize(int w, int h) {
        width  = w;
        height = h;
        pixels.resize(w * h, Spectrum::zero());
        sample_count.resize(w * h, 0.f);
        srgb.resize(w * h * 4, 0);
    }

    void clear() {
        for (auto& p : pixels)       p = Spectrum::zero();
        for (auto& s : sample_count) s = 0.f;
    }

    void accumulate(int x, int y, const Spectrum& L) {
        int idx = y * width + x;
        pixels[idx] += L;
        sample_count[idx] += 1.f;
    }

    // Tonemap and convert to sRGB
    void tonemap(float exposure = 1.0f) {
        for (int i = 0; i < width * height; ++i) {
            Spectrum avg = (sample_count[i] > 0.f)
                ? pixels[i] / sample_count[i]
                : Spectrum::zero();

            // Apply exposure
            avg *= exposure;

            // Spectrum → sRGB
            float3 rgb = spectrum_to_srgb(avg);

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

private:
    // Trace a single camera path and return spectral radiance
    Spectrum trace_path(Ray ray, PCGRng& rng);

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
};
