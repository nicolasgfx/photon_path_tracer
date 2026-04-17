#pragma once
// ─────────────────────────────────────────────────────────────────────
// app/render_session.h – GPU render session orchestrator (v5, RGB)
//
// Owns the full GPU pipeline: AccelBuilder, LightingUploader,
// DeviceBuffers for output, and postfx.
// Provides init(), render_frame(), and download() for both
// interactive (Viewer) and headless (batch) use.
// ─────────────────────────────────────────────────────────────────────
#include "accel/accel_builder.h"
#include "accel/launch_params.h"
#include "accel/lighting_upload.h"
#include "core/camera.h"
#include "core/device_buffer.h"
#include "core/types.h"
#include "core/config.h"
#include "scene/scene.h"
#include "postfx/postfx_pipeline.h"
#include "app/render_config.h"
#include "postfx/optix_denoiser.h"
#include "analyze/prepass_metrics.h"
#include "photon/delta_surface.h"

#include <string>
#include <vector>

class RenderSession {
public:
    RenderSession() = default;
    ~RenderSession() = default;

    // Non-copyable
    RenderSession(const RenderSession&) = delete;
    RenderSession& operator=(const RenderSession&) = delete;

    // ── Initialization ───────────────────────────────────────────────
    bool init(const Scene& scene, const RenderConfig& config,
              const std::string& ptx_path);

    bool is_ready() const { return ready_; }

    // ── Per-frame rendering (1 SPP progressive accumulation) ────────
    void render_frame(const Camera& camera, int frame_number,
                      const RenderConfig& config);

    // ── Caustic light tracing (call after render_frame) ─────────────
    void launch_caustics(const Camera& camera, int frame_number,
                         const RenderConfig& config);

    // ── Download result ──────────────────────────────────────────────
    void download_color(std::vector<float>& rgb_out) const;
    void download_sample_counts(std::vector<float>& counts_out) const;

    // ── Post-processing (GPU-side) ─────────────────────────────────
    void apply_postfx(const PostFxParams& params);
    void apply_snapshot_postfx(const PostFxParams& params);
    void download_srgb(std::vector<uint8_t>& srgb_out) const;
    void download_hdr(std::vector<float>& hdr_out) const;

    // ── View-dependent pre-pass ─────────────────────────────────────
    PrePassMetrics run_prepass(const Camera& camera, const RenderConfig& config);

    // ── Accessors ────────────────────────────────────────────────────
    int width()  const { return width_; }
    int height() const { return height_; }
    const AccelBuilder& accel() const { return builder_; }
    int  accumulated_spp() const { return accumulated_spp_; }
    int  caustic_frames()  const { return caustic_frames_; }

    void reset_accumulation();

private:
    void fill_launch_params(const Camera& camera, int frame_number,
                            const RenderConfig& config);
    void fill_caustic_params(const RenderConfig& config);

    bool ready_ = false;
    int  width_  = 0;
    int  height_ = 0;
    int  accumulated_spp_ = 0;

    // Pipeline components
    AccelBuilder      builder_;
    LightingUploader  lighting_;

    // GPU output buffers (SoA color channels for coalesced GPU access)
    DeviceBuffer<float>     d_color_r_;       // [w * h]
    DeviceBuffer<float>     d_color_g_;       // [w * h]
    DeviceBuffer<float>     d_color_b_;       // [w * h]
    DeviceBuffer<float>     d_sample_counts_; // [w * h]
    DeviceBuffer<float>     d_albedo_;        // [w * h * 4]
    DeviceBuffer<float>     d_normal_;        // [w * h * 4]
    DeviceBuffer<uint8_t>   d_srgb_;          // [w * h * 4] PostFx output

    // Post-processing pipeline
    PostFxPipeline          postfx_;

    // OptiX AI Denoiser (optional)
    DenoiserSession         denoiser_;
    bool                    denoiser_enabled_ = false;

    // Caustic light tracing
    DeltaSurfaceDistribution  delta_dist_;
    DeviceBuffer<uint32_t>    d_delta_tri_indices_;
    DeviceBuffer<float>       d_delta_cdf_;
    DeviceBuffer<float>       d_caustic_r_;
    DeviceBuffer<float>       d_caustic_g_;
    DeviceBuffer<float>       d_caustic_b_;
    int                       caustic_frames_ = 0;
    bool                      caustic_ready_  = false;

    // Shared launch params (rebuilt each frame)
    LaunchParams lp_ = {};
};
