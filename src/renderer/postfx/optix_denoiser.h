#pragma once
// ─────────────────────────────────────────────────────────────────────
// postfx/optix_denoiser.h – OptiX AI denoiser wrapper (v5, RGB)
//
// Wraps the OptiX HDR denoiser with optional albedo + normal guides.
// Operates entirely on device memory (float4 HDR in/out).
//
// Lifecycle:
//   DenoiserSession ds;
//   ds.init(optix_ctx, width, height, use_guides);
//   ds.denoise(d_hdr_in_out, d_albedo, d_normal, width, height, blend);
//   ds.cleanup();
// ─────────────────────────────────────────────────────────────────────
#include <optix.h>
#include <cstddef>

class DenoiserSession {
public:
    DenoiserSession() = default;
    ~DenoiserSession() { cleanup(); }

    // Non-copyable
    DenoiserSession(const DenoiserSession&) = delete;
    DenoiserSession& operator=(const DenoiserSession&) = delete;

    // ── Setup ────────────────────────────────────────────────────────
    // Call once after the OptiX context is created.
    // use_guides: enable albedo + normal guide layers (better quality).
    bool init(OptixDeviceContext ctx, int width, int height,
              bool use_guides = true);

    void cleanup();

    bool is_ready() const { return denoiser_ != nullptr; }

    // ── Per-frame denoise ────────────────────────────────────────────
    // d_hdr:    [width*height*4] float (RGBA): input AND output
    // d_albedo: [width*height*4] float (nullable if !use_guides)
    // d_normal: [width*height*4] float (nullable if !use_guides)
    // blend:    0.0 = full denoise, 1.0 = passthrough
    void denoise(float* d_hdr,
                 const float* d_albedo,
                 const float* d_normal,
                 int width, int height,
                 float blend = 0.f);

private:
    OptixDenoiser denoiser_   = nullptr;
    void*   d_state_          = nullptr;
    void*   d_scratch_        = nullptr;
    void*   d_intensity_      = nullptr;
    void*   d_output_         = nullptr;  // denoised result (float4)

    size_t  state_size_       = 0;
    size_t  scratch_size_     = 0;

    int     alloc_w_          = 0;
    int     alloc_h_          = 0;
    bool    use_guides_       = false;
};
