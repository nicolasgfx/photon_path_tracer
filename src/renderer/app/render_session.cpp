// ─────────────────────────────────────────────────────────────────────
// app/render_session.cpp – GPU render session implementation (v5, RGB)
//
// Orchestrates: AccelBuilder → LightingUploader → progressive render
//               → postfx / denoiser → download.
// ─────────────────────────────────────────────────────────────────────
#include "app/render_session.h"
#include "core/config.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

// ── Initialization ──────────────────────────────────────────────────

bool RenderSession::init(const Scene& scene, const RenderConfig& config,
                         const std::string& ptx_path) {
    width_  = config.image_width;
    height_ = config.image_height;
    int num_pixels = width_ * height_;

    std::printf("[RenderSession] Initialising %dx%d  ptx=%s\n",
                width_, height_, ptx_path.c_str());
    auto t0 = std::chrono::high_resolution_clock::now();

    // ── 1. Build OptiX pipeline + acceleration structure ────────────
    builder_.init();
    builder_.build(scene, ptx_path);
    builder_.upload_geometry(scene);
    builder_.upload_materials(scene);
    builder_.upload_textures(scene);

    // ── 2. Upload lighting data (emissives + light tree) ────────────
    lighting_.upload_emissives(scene);
    if (config.light_tree_enabled)
        lighting_.upload_light_tree(scene, config.light_tree_max_leaf_size);

    // ── 3. Allocate output buffers ──────────────────────────────────
    d_color_r_.alloc(num_pixels);
    d_color_g_.alloc(num_pixels);
    d_color_b_.alloc(num_pixels);
    d_sample_counts_.alloc(num_pixels);
    d_albedo_.alloc(num_pixels * 4);
    d_normal_.alloc(num_pixels * 4);

    d_color_r_.zero();
    d_color_g_.zero();
    d_color_b_.zero();
    d_sample_counts_.zero();
    d_albedo_.zero();
    d_normal_.zero();

    // ── 3b. PostFx sRGB buffer + pipeline init ──────────────────────
    d_srgb_.alloc(num_pixels * 4);
    d_srgb_.zero();
    postfx_.init(width_, height_);

    // ── 4. OptiX AI denoiser (optional) ─────────────────────────────
    denoiser_enabled_ = false;
    if (config.denoiser_enabled) {
        bool use_guides = config.denoiser_guide_albedo && config.denoiser_guide_normal;
        denoiser_enabled_ = denoiser_.init(builder_.optix_context(), width_, height_, use_guides);
        if (!denoiser_enabled_) {
            std::fprintf(stderr, "[RenderSession] Warning: Denoiser init failed, continuing without denoiser\n");
        }
    }

    // ── 5. Caustic light tracing (delta surface CDF + splat buffers) ─
    delta_dist_ = build_delta_surface_distribution(scene);
    if (!delta_dist_.empty() && config.caustic_enabled) {
        d_delta_tri_indices_.alloc(delta_dist_.count());
        d_delta_cdf_.alloc(delta_dist_.count());
        d_delta_tri_indices_.upload(delta_dist_.tri_indices.data(), delta_dist_.count());
        d_delta_cdf_.upload(delta_dist_.cdf.data(), delta_dist_.count());

        d_caustic_r_.alloc(num_pixels);
        d_caustic_g_.alloc(num_pixels);
        d_caustic_b_.alloc(num_pixels);
        d_caustic_r_.zero();
        d_caustic_g_.zero();
        d_caustic_b_.zero();
        caustic_frames_ = 0;
        caustic_ready_ = true;

        std::printf("[RenderSession] Caustic buffers ready  delta_tris=%d  total_area=%.3f\n",
                    delta_dist_.count(), delta_dist_.total_area);
    } else {
        caustic_ready_ = false;
    }

    accumulated_spp_ = 0;
    ready_ = true;

    auto t1 = std::chrono::high_resolution_clock::now();
    std::printf("[RenderSession] Ready in %.1f ms  GPU: %s  VRAM: %.0f MB\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                builder_.gpu_name().c_str(),
                builder_.gpu_vram_total() / (1024.0 * 1024.0));
    return true;
}

// ── Per-frame rendering ─────────────────────────────────────────────

void RenderSession::render_frame(const Camera& camera, int frame_number,
                                 const RenderConfig& config) {
    if (!ready_) return;

    fill_launch_params(camera, frame_number, config);

    // Batch multiple SPP per kernel launch to amortize sync overhead.
    int spp = (std::max)(config.spp_per_launch, 1);
    lp_.samples_per_pixel = spp;

    builder_.launch_progressive(width_, height_, lp_);

    accumulated_spp_ += spp;
}

// ── Caustic light tracing ───────────────────────────────────────────

void RenderSession::launch_caustics(const Camera& camera, int frame_number,
                                    const RenderConfig& config) {
    if (!ready_ || !caustic_ready_ || !config.caustic_enabled) return;

    fill_launch_params(camera, frame_number, config);
    fill_caustic_params(config);

    int budget = config.caustic_photons_per_frame;
    if (budget <= 0) budget = DEFAULT_CAUSTIC_PHOTONS_PER_FRAME;

    if (caustic_frames_ == 0)
        std::printf("[Caustic] budget=%d  delta_tris=%d  delta_area=%.3f  splat_clamp=%.1f\n",
                    budget, (int)delta_dist_.count(), delta_dist_.total_area,
                    config.caustic_max_splat_luminance);

    builder_.launch_caustic(budget, lp_);
    ++caustic_frames_;
}

// ── Download ────────────────────────────────────────────────────────

void RenderSession::download_color(std::vector<float>& rgb_out) const {
    int n = width_ * height_;
    rgb_out.resize(n * 3);
    // Interleave SoA channels into AoS for host-side consumers
    std::vector<float> r(n), g(n), b(n);
    d_color_r_.download(r.data(), n);
    d_color_g_.download(g.data(), n);
    d_color_b_.download(b.data(), n);
    for (int i = 0; i < n; ++i) {
        rgb_out[i * 3 + 0] = r[i];
        rgb_out[i * 3 + 1] = g[i];
        rgb_out[i * 3 + 2] = b[i];
    }
}

void RenderSession::download_sample_counts(std::vector<float>& counts_out) const {
    d_sample_counts_.download(counts_out);
}

// ── Post-processing ─────────────────────────────────────────────────

void RenderSession::apply_postfx(const PostFxParams& params) {
    if (!ready_) return;

    // Inject caustic composition data into a mutable copy
    PostFxParams p = params;
    if (caustic_ready_ && caustic_frames_ > 0) {
        p.caustic_r      = d_caustic_r_.data();
        p.caustic_g      = d_caustic_g_.data();
        p.caustic_b      = d_caustic_b_.data();
        p.caustic_frames = caustic_frames_;
    }

    if (denoiser_enabled_ && denoiser_.is_ready()) {
        float* d_hdr = postfx_.apply_phase1(
            d_color_r_.data(), d_color_g_.data(), d_color_b_.data(),
            d_sample_counts_.data(),
            width_, height_, p);

        denoiser_.denoise(d_hdr,
                          d_albedo_.data(),
                          d_normal_.data(),
                          width_, height_,
                          p.denoiser_blend);

        postfx_.apply_phase2(d_srgb_.data(), width_, height_, p);
    } else {
        postfx_.apply(d_color_r_.data(),
                      d_color_g_.data(),
                      d_color_b_.data(),
                      d_sample_counts_.data(),
                      d_srgb_.data(),
                      nullptr,
                      width_, height_,
                      p);
    }
}

void RenderSession::apply_snapshot_postfx(const PostFxParams& params) {
    if (!ready_) return;

    // Inject caustic composition data (same as apply_postfx)
    PostFxParams p = params;
    if (caustic_ready_ && caustic_frames_ > 0) {
        p.caustic_r      = d_caustic_r_.data();
        p.caustic_g      = d_caustic_g_.data();
        p.caustic_b      = d_caustic_b_.data();
        p.caustic_frames = caustic_frames_;
    }

    // Lazy-init denoiser if not already active
    if (!denoiser_enabled_ || !denoiser_.is_ready()) {
        denoiser_enabled_ = denoiser_.init(
            builder_.optix_context(), width_, height_, /*use_guides=*/true);
        if (denoiser_enabled_)
            std::printf("[Snapshot] Denoiser initialised for snapshot\n");
    }

    // Run full postfx chain with denoiser
    if (denoiser_enabled_ && denoiser_.is_ready()) {
        float* d_hdr = postfx_.apply_phase1(
            d_color_r_.data(), d_color_g_.data(), d_color_b_.data(),
            d_sample_counts_.data(),
            width_, height_, p);

        denoiser_.denoise(d_hdr,
                          d_albedo_.data(),
                          d_normal_.data(),
                          width_, height_,
                          0.f);  // blend=0: fully denoised

        postfx_.apply_phase2(d_srgb_.data(), width_, height_, p);
    } else {
        std::fprintf(stderr, "[Snapshot] Warning: denoiser unavailable, saving without denoising\n");
        apply_postfx(params);
    }
}

void RenderSession::download_srgb(std::vector<uint8_t>& srgb_out) const {
    d_srgb_.download(srgb_out);
}

void RenderSession::download_hdr(std::vector<float>& hdr_out) const {
    float* d_hdr = postfx_.hdr_buffer();
    if (!d_hdr) return;
    size_t n = (size_t)width_ * height_ * 4;
    hdr_out.resize(n);
    CUDA_CHECK(cudaMemcpy(hdr_out.data(), d_hdr,
                          n * sizeof(float), cudaMemcpyDeviceToHost));
}

// ── View-dependent pre-pass ─────────────────────────────────────────

PrePassMetrics RenderSession::run_prepass(const Camera& camera,
                                          const RenderConfig& config) {
    PrePassMetrics result;
    if (!ready_) return result;

    const int prepass_spp = config.prepass_spp;
    if (prepass_spp <= 0) return result;

    const int pw = (std::max)(width_  / config.prepass_scale_divisor, 1);
    const int ph = (std::max)(height_ / config.prepass_scale_divisor, 1);
    const int num_pixels = pw * ph;

    result.prepass_spp    = prepass_spp;
    result.prepass_width  = pw;
    result.prepass_height = ph;

    std::printf("[PrePass] Running %d SPP at %dx%d (%.1f%% of full res)...\n",
                prepass_spp, pw, ph,
                100.f * (float)num_pixels / (float)(width_ * height_));

    auto t0 = std::chrono::high_resolution_clock::now();

    // ── 1. Allocate temporary quarter-res buffers ───────────────────
    DeviceBuffer<float>        pp_color_r;   pp_color_r.alloc(num_pixels);
    DeviceBuffer<float>        pp_color_g;   pp_color_g.alloc(num_pixels);
    DeviceBuffer<float>        pp_color_b;   pp_color_b.alloc(num_pixels);
    DeviceBuffer<float>        pp_counts;    pp_counts.alloc(num_pixels);
    DeviceBuffer<float>        pp_lum_sum;   pp_lum_sum.alloc(num_pixels);
    DeviceBuffer<float>        pp_lum_sum2;  pp_lum_sum2.alloc(num_pixels);
    DeviceBuffer<unsigned int> pp_nee_attempts; pp_nee_attempts.alloc(1);
    DeviceBuffer<unsigned int> pp_nee_hits;     pp_nee_hits.alloc(1);
    DeviceBuffer<unsigned int> pp_zero_paths;   pp_zero_paths.alloc(1);
    DeviceBuffer<unsigned int> pp_bounce_sum;   pp_bounce_sum.alloc(1);
    DeviceBuffer<unsigned int> pp_total_paths;  pp_total_paths.alloc(1);

    pp_color_r.zero();
    pp_color_g.zero();
    pp_color_b.zero();
    pp_counts.zero();
    pp_lum_sum.zero();
    pp_lum_sum2.zero();
    pp_nee_attempts.zero();
    pp_nee_hits.zero();
    pp_zero_paths.zero();
    pp_bounce_sum.zero();
    pp_total_paths.zero();

    // ── 2. Build launch params with pre-pass overrides ──────────────
    for (int spp = 0; spp < prepass_spp; ++spp) {
        fill_launch_params(camera, spp, config);

        // Override output buffers to use temporary quarter-res buffers
        lp_.color_r       = pp_color_r.data();
        lp_.color_g       = pp_color_g.data();
        lp_.color_b       = pp_color_b.data();
        lp_.sample_counts = pp_counts.data();
        lp_.albedo_buffer = nullptr;
        lp_.normal_buffer = nullptr;
        lp_.width         = pw;
        lp_.height        = ph;

        // Variance tracking
        lp_.lum_sum       = pp_lum_sum.data();
        lp_.lum_sum2      = pp_lum_sum2.data();

        // Pre-pass atomic counters
        lp_.prepass_nee_attempts = pp_nee_attempts.data();
        lp_.prepass_nee_hits     = pp_nee_hits.data();
        lp_.prepass_zero_paths   = pp_zero_paths.data();
        lp_.prepass_bounce_sum   = pp_bounce_sum.data();
        lp_.prepass_total_paths  = pp_total_paths.data();
        lp_.prepass_active       = 1;

        lp_.samples_per_pixel = 1;

        builder_.launch_progressive(pw, ph, lp_);
    }

    // ── 3. Download results ─────────────────────────────────────────
    std::vector<float> h_lum_sum(num_pixels), h_lum_sum2(num_pixels);
    pp_lum_sum.download(h_lum_sum.data(), num_pixels);
    pp_lum_sum2.download(h_lum_sum2.data(), num_pixels);

    unsigned int counters[5] = {};
    pp_nee_attempts.download(&counters[0], 1);
    pp_nee_hits.download(&counters[1], 1);
    pp_zero_paths.download(&counters[2], 1);
    pp_bounce_sum.download(&counters[3], 1);
    pp_total_paths.download(&counters[4], 1);

    auto t1 = std::chrono::high_resolution_clock::now();
    float elapsed = (float)std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ── 4. Compute metrics ──────────────────────────────────────────
    result = compute_prepass_metrics(
        h_lum_sum.data(), h_lum_sum2.data(), num_pixels, prepass_spp,
        counters[0], counters[1], counters[2], counters[3], counters[4]);
    result.prepass_spp    = prepass_spp;
    result.prepass_width  = pw;
    result.prepass_height = ph;
    result.time_ms        = elapsed;

    std::printf("[PrePass] Done in %.1f ms  NEE hit=%.2f%%  zero=%.2f%%  "
                "avg_bounce=%.1f  var_mean=%.4f  var_p99=%.4f\n",
                elapsed,
                result.nee_hit_rate * 100.f,
                result.zero_path_fraction * 100.f,
                result.avg_bounce_depth,
                result.mean_pixel_variance,
                result.variance_p99);

    // ── 5. Save diagnostics ─────────────────────────────────────────
    save_variance_heatmap("output/prepass_variance.png",
                          h_lum_sum.data(), h_lum_sum2.data(), pw, ph, prepass_spp);
    save_prepass_json("output/prepass_metrics.json", result);
    std::printf("[PrePass] Saved output/prepass_variance.png + prepass_metrics.json\n");

    // Temporary buffers freed by RAII destructors
    return result;
}

// ── Reset accumulation ──────────────────────────────────────────────

void RenderSession::reset_accumulation() {
    if (!ready_) return;
    d_color_r_.zero();
    d_color_g_.zero();
    d_color_b_.zero();
    d_sample_counts_.zero();
    d_albedo_.zero();
    d_normal_.zero();
    accumulated_spp_ = 0;

    if (caustic_ready_) {
        d_caustic_r_.zero();
        d_caustic_g_.zero();
        d_caustic_b_.zero();
        caustic_frames_ = 0;
    }
}

// ── Fill LaunchParams ───────────────────────────────────────────────

void RenderSession::fill_launch_params(const Camera& camera,
                                       int frame_number,
                                       const RenderConfig& config) {
    memset(&lp_, 0, sizeof(LaunchParams));

    // Output buffers (SoA color channels)
    lp_.color_r       = d_color_r_.data();
    lp_.color_g       = d_color_g_.data();
    lp_.color_b       = d_color_b_.data();
    lp_.sample_counts = d_sample_counts_.data();
    lp_.srgb_buffer   = nullptr;
    lp_.albedo_buffer = d_albedo_.data();
    lp_.normal_buffer = d_normal_.data();
    lp_.width         = width_;
    lp_.height        = height_;

    // Camera convention expected by kernel:
    //   direction = normalize(cam_w + cam_u*(2u-1) + cam_v*(2v-1))
    //   v_ndc is top->bottom, so cam_v must point down (negated vertical).
    float focus_factor = (camera.lens_radius > 0.f) ? camera.dof_focus_dist : 1.f;
    lp_.cam_pos         = camera.position;
    lp_.cam_u           = camera.horizontal * 0.5f;
    lp_.cam_v           = camera.vertical * -0.5f;
    lp_.cam_w           = camera.w * (-focus_factor);
    lp_.cam_lens_radius = camera.lens_radius;
    lp_.cam_focus_dist  = camera.dof_focus_dist;

    // Rendering parameters
    lp_.samples_per_pixel = 1;
    lp_.max_bounces       = config.max_bounces;
    lp_.min_bounces_rr    = config.min_bounces_rr;
    lp_.rr_threshold      = config.rr_threshold;
    lp_.frame_number      = frame_number;
    lp_.render_mode       = config.mode;
    lp_.exposure          = config.exposure;

    // Clamping (runtime-tunable via JSON config)
    lp_.clamping_enabled        = config.clamping_enabled ? 1 : 0;
    lp_.max_bounce_contribution = config.max_bounce_contribution;
    lp_.max_path_throughput     = config.max_path_throughput;
    lp_.max_nee_contribution    = config.max_nee_contribution;
    lp_.max_sample_luminance    = config.max_sample_luminance;

    // Geometry + materials
    builder_.fill_geometry_params(lp_);
    builder_.fill_material_params(lp_);

    // Lighting
    lighting_.fill_params(lp_);

#ifdef PPT_USE_OPTIX
    lp_.traversable = builder_.accel().handle;
#endif
    lp_.has_instances = builder_.accel().instanced ? 1 : 0;
}

void RenderSession::fill_caustic_params(const RenderConfig& config) {
    // Caustic splat buffers
    lp_.caustic_r = d_caustic_r_.data();
    lp_.caustic_g = d_caustic_g_.data();
    lp_.caustic_b = d_caustic_b_.data();

    int budget = config.caustic_photons_per_frame;
    if (budget <= 0) budget = DEFAULT_CAUSTIC_PHOTONS_PER_FRAME;
    lp_.caustic_num_photons = budget;
    lp_.caustic_frame_number = caustic_frames_;
    lp_.caustic_max_splat_luminance = config.caustic_max_splat_luminance;

    // Delta surface distribution
    lp_.delta_tri_indices = d_delta_tri_indices_.data();
    lp_.delta_cdf         = d_delta_cdf_.data();
    lp_.num_delta_tris    = delta_dist_.count();
    lp_.delta_total_area  = delta_dist_.total_area;
}
