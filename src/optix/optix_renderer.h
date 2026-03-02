#pragma once
// ---------------------------------------------------------------------
// optix_renderer.h -- OptiX host-side pipeline management
// ---------------------------------------------------------------------
// Owns the OptiX context, module, pipeline, SBT, GAS/IAS, and device
// buffers.  Used by both the debug interactive viewer and the final
// "press-R" high-quality render.
// OptiX is MANDATORY -- there is no CPU fallback.
// ---------------------------------------------------------------------
#include "core/types.h"
#include "core/spectrum.h"
#include "core/config.h"
#include "scene/scene.h"
#include "renderer/camera.h"
#include "renderer/renderer.h"   // RenderConfig, RenderMode, FrameBuffer
#include "photon/photon.h"
#include "photon/hash_grid.h"
#include "photon/cell_bin_grid.h"
#include "photon/hash_histogram.h"
#include "debug/stats_collector.h"
#include "optix/launch_params.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include <functional>

// -- CUDA error helpers -----------------------------------------------

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t rc = call;                                                \
        if (rc != cudaSuccess) {                                              \
            std::cerr << "[CUDA] " << cudaGetErrorString(rc) << " at "        \
                      << __FILE__ << ":" << __LINE__ << "\n";                 \
            throw std::runtime_error("CUDA error");                           \
        }                                                                     \
    } while (0)

#define OPTIX_CHECK(call)                                                     \
    do {                                                                      \
        OptixResult rc = call;                                                \
        if (rc != OPTIX_SUCCESS) {                                            \
            std::cerr << "[OptiX] Error " << (int)rc << " at "               \
                      << __FILE__ << ":" << __LINE__ << "\n";                 \
            throw std::runtime_error("OptiX error");                          \
        }                                                                     \
    } while (0)

// -- DeviceBuffer -- RAII wrapper for CUdeviceptr ---------------------

struct DeviceBuffer {
    void*  d_ptr  = nullptr;
    size_t bytes  = 0;

    void alloc(size_t n) {
        free();
        bytes = n;
        CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
    }

    /// Like alloc(), but only (re)allocates if the current buffer is
    /// too small.  Avoids cudaFree+cudaMalloc round-trips when the
    /// size hasn't changed (the common case for LaunchParams).
    void ensure_alloc(size_t n) {
        if (bytes >= n) return;   // already large enough
        alloc(n);
    }

    void alloc_zero(size_t n) {
        alloc(n);
        CUDA_CHECK(cudaMemset(d_ptr, 0, bytes));
    }

    template <typename T>
    void upload(const std::vector<T>& data) {
        alloc(data.size() * sizeof(T));
        CUDA_CHECK(cudaMemcpy(d_ptr, data.data(), bytes, cudaMemcpyHostToDevice));
    }

    template <typename T>
    void upload(const T* data, size_t count) {
        alloc(count * sizeof(T));
        CUDA_CHECK(cudaMemcpy(d_ptr, data, bytes, cudaMemcpyHostToDevice));
    }

    template <typename T>
    void download(std::vector<T>& out) const {
        out.resize(bytes / sizeof(T));
        CUDA_CHECK(cudaMemcpy(out.data(), d_ptr, bytes, cudaMemcpyDeviceToHost));
    }

    void free() {
        if (d_ptr) { cudaFree(d_ptr); d_ptr = nullptr; bytes = 0; }
    }

    ~DeviceBuffer() { free(); }

    // Non-copyable
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : d_ptr(o.d_ptr), bytes(o.bytes) {
        o.d_ptr = nullptr; o.bytes = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        free(); d_ptr = o.d_ptr; bytes = o.bytes;
        o.d_ptr = nullptr; o.bytes = 0;
        return *this;
    }

    template <typename T> T* as() { return reinterpret_cast<T*>(d_ptr); }
    template <typename T> const T* as() const { return reinterpret_cast<const T*>(d_ptr); }
};

// ── Photon map pool slot ────────────────────────────────────────────

// -- OptiX debug render mode (matches RenderMode enum) ----------------

enum class OptixDebugMode : int {
    Full          = 0,
    DirectOnly    = 1,
    IndirectOnly  = 2,
    PhotonMap     = 3,
    Normals       = 4,
    MaterialID    = 5,
    Depth         = 6
};

// ---------------------------------------------------------------------
// OptixRenderer -- the main OptiX host-side class
// ---------------------------------------------------------------------

class OptixRenderer {
public:
    OptixRenderer() = default;
    ~OptixRenderer() { cleanup(); }

    // -- Initialisation -----------------------------------------------
    void init();
    void build_accel(const Scene& scene);
    void upload_scene_data(const Scene& scene);

    /// Set clearcoat/fabric per-material pointers in launch params.
    /// Called after fill_common_params() at every launch site.
    void fill_clearcoat_fabric_params(LaunchParams& lp) const;

    void upload_photon_data(const PhotonSoA& global_photons,
                            const HashGrid& global_grid,
                            const PhotonSoA& caustic_photons,
                            const HashGrid& caustic_grid,
                            float gather_radius,
                            float caustic_radius,
                            int num_photons_emitted = 0);

    /// Build per-cell photon analysis from CellInfoCache + CellBinGrid
    /// and upload the result arrays to GPU (PA-07/PA-08).
    void upload_cell_analysis(const CellInfoCache& cell_cache,
                              const CellBinGrid&   bin_grid,
                              float                cell_area);

    /// Upload emitter data to device (for GPU photon tracing)
    void upload_emitter_data(const Scene& scene);



    /// Trace photons entirely on the GPU. Downloads results, builds
    /// the hash grid on CPU, uploads the grid + photon data back.
    /// @param grid_radius_override  If > 0, use this radius instead of
    ///     config.gather_radius for the hash grid build and GPU cell size.
    ///     Used by SPPM to size cells for the SPPM initial radius.
    void trace_photons(const Scene& scene, const RenderConfig& config,
                       float grid_radius_override = 0.f,
                       int photon_map_seed = 0);

    /// VA-01/02/03: Build view-adaptive emitter CDF from stored photon
    /// source indices and re-upload to GPU.  Returns true if the CDF
    /// was updated (i.e. enough photons had valid source indices).
    bool build_view_adaptive_cdf(const Scene& scene, float beta);

    // -- Rendering ----------------------------------------------------

    /// Launch a single frame of the debug viewer (1 spp, progressive)
    /// @param shadow_rays  (unused in v3 — kept for API compat)
    void render_debug_frame(const Camera& camera, int frame_number,
                            RenderMode mode, int spp = 1,
                            bool shadow_rays = false);

    /// Launch the full final render (multi-spp, blocking)
    void render_final(const Camera& camera, const RenderConfig& config,
                      const Scene& scene);

    /// Launch the SPPM render (iterative photon mapping, blocking).
    /// Each iteration: camera pass → photon trace → gather → update.
    /// @param iter_callback  Optional callback invoked after every completed
    ///                       iteration.  Arguments: (0-based iteration index,
    ///                       current accumulated FrameBuffer).  Use this to
    ///                       save per-iteration PNGs for progress tracking.
    void render_sppm(const Camera& camera, const RenderConfig& config,
                     const Scene& scene,
                     std::function<void(int, const FrameBuffer&)> iter_callback = {});

    /// Launch a single sample of full path tracing (progressive)
    void render_one_spp(const Camera& camera, int frame_number,
                        int max_bounces = DEFAULT_MAX_BOUNCES);

    /// Render a caustic-only debug pass using only caustic-flagged photons.
    /// Call after render_final() completes.  Outputs caustic-only photon
    /// indirect into the provided spectral / sample-count buffers.
    void render_caustic_debug_pass(const Camera& camera,
                                   const RenderConfig& config,
                                   std::vector<float>& caustic_spec,
                                   std::vector<float>& samp_counts);

    /// Non-destructive caustic snapshot: saves/restores progressive
    /// accumulation state around render_caustic_debug_pass().
    /// Safe to call during progressive rendering (e.g. progress snapshots).
    void render_caustic_snapshot(const Camera& camera,
                                const RenderConfig& config,
                                std::vector<float>& caustic_spec,
                                std::vector<float>& caustic_samp);

    /// Read back the sRGB buffer from the device (denoised when enabled)
    void download_framebuffer(FrameBuffer& fb) const;

    /// Read back the raw (un-denoised) sRGB buffer; valid only after render_final with denoiser
    void download_raw_framebuffer(FrameBuffer& fb) const;

    /// Download component spectral buffers to CPU (for debug PNGs)
    /// Returns raw spectral data [width*height*NUM_LAMBDA] floats.
    void download_component_buffers(
        std::vector<float>& nee_direct,
        std::vector<float>& photon_indirect,
        std::vector<float>& sample_counts) const;

    /// Trace a single ray and return the hit record (for debug hover)
    HitRecord trace_single_ray(float3 origin, float3 direction) const;

    // -- Accessors ----------------------------------------------------
    int width()  const { return width_; }
    int height() const { return height_; }
    void resize(int w, int h);
    void clear_buffers();  ///< Zero accumulation buffers (camera moved)
    bool is_initialised() const { return initialised_; }

    /// Runtime toggle for participating medium (V key in interactive viewer).
    /// Overrides the compile-time DEFAULT_VOLUME_ENABLED for all render paths.
    void set_volume_enabled(bool v) { runtime_volume_enabled_ = v; }
    bool is_volume_enabled()  const { return runtime_volume_enabled_; }

    /// Runtime exposure multiplier (render_config.json §5, applied before tone mapping).
    void set_exposure(float e)   { exposure_ = e; }
    float get_exposure()   const { return exposure_; }

    /// Toggle per-triangle photon irradiance heatmap in preview mode.
    void set_photon_heatmap(bool v) { show_photon_heatmap_ = v; }
    bool is_photon_heatmap() const  { return show_photon_heatmap_; }

    /// Toggle OptiX AI denoiser for final render.
    void set_denoiser_enabled(bool v) { denoiser_enabled_ = v; }
    bool is_denoiser_enabled() const  { return denoiser_enabled_; }

    /// Toggle dense cell-bin grid path vs hash-grid walk.
    void set_use_dense_grid(bool v) { use_dense_grid_ = v; }
    bool is_use_dense_grid() const  { return use_dense_grid_; }

    /// Runtime guide fraction (T key: toggle guided/unguided).
    void set_guide_fraction(float f) { guide_fraction_ = f; }
    float get_guide_fraction() const  { return guide_fraction_; }

    /// Toggle histogram-only conclusion mode (C key).
    void set_histogram_only(bool v) { histogram_only_ = v; }
    bool is_histogram_only() const  { return histogram_only_; }

    /// GPU device info (populated during init()).
    const std::string& gpu_name()    const { return gpu_name_; }
    size_t  gpu_vram_total()         const { return gpu_vram_total_; }
    int     gpu_sm_count()           const { return gpu_sm_count_; }
    int     gpu_cc_major()           const { return gpu_cc_major_; }
    int     gpu_cc_minor()           const { return gpu_cc_minor_; }

    /// Fill CellBinGrid / guidance params into LaunchParams.
    /// Called after fill_common_params at every launch site.
    void fill_cell_grid_params(LaunchParams& lp) const;

    /// Render coverage debug PNG (stub — no-op when DEBUG_COMPONENT_PNGS is false).
    template<typename CamT, typename FbT>
    void render_coverage_debug_png(const CamT&, const FbT&) {}

    /// Download GPU kernel profiling data and print a summary.
    /// Call after the final render completes.
    void print_kernel_profiling() const;

    /// Snapshot of renderer statistics (for JSON export).
    struct RenderStats {
        // Image
        int    image_width          = 0;
        int    image_height         = 0;
        int    accumulated_spp      = 0;

        // Photon map
        int    photons_emitted      = 0;
        int    photons_stored       = 0;
        int    caustic_emitted      = 0;
        float  gather_radius        = 0.f;
        float  caustic_radius       = 0.f;
        int    caustic_stored       = 0;   // tag-2 (targeted) count
        int    noncaustic_stored    = 0;   // tag-0
        int    global_caustic_stored = 0;  // tag-1

        // Cell analysis / guidance
        int    cell_analysis_cells  = 0;
        float  avg_guide_fraction   = 0.f;
        float  avg_guide_fraction_populated = 0.f;  // avg over cells with photons only
        int    guide_populated_cells = 0;            // cells with guide_fraction > 0
        float  avg_caustic_fraction = 0.f;
        GuideFractionDist guide_dist;
        ConclusionCounters conclusions;

        // Hash histogram (multi-resolution guide)
        HashHistStats hash_hist;

        // Config
        int    max_bounces_camera   = 0;
        int    max_bounces_photon   = 0;
        int    min_bounces_rr       = 0;
        float  rr_threshold         = 0.f;
        float  guide_fraction       = 0.f;
        float  exposure             = 0.f;
        bool   denoiser_enabled     = false;
        int    knn_k                = 0;
        float  surface_tau          = 0.f;

        // Scene
        std::string scene_name;
        int    num_triangles        = 0;
        int    num_emissive_tris    = 0;
    };

    /// Collect current renderer statistics into a struct.
    RenderStats gather_stats(const char* scene_name) const;

    /// Access the stored photon data and hash grid (available after
    /// trace_photons() completes)
    const PhotonSoA& photons() const { return stored_photons_; }
    const HashGrid&  grid()    const { return stored_grid_; }

    /// Cell-bin grid test accessors (returns empty grid when dense grid is disabled)
    const CellBinGrid& cell_bin_grid_for_test() const { return cell_bin_grid_; }
    size_t cell_bin_grid_bytes_for_test() const {
        return (size_t)cell_bin_grid_.total_cells() * cell_bin_grid_.bin_count * sizeof(PhotonBin);
    }

    /// Test hook: returns the last LaunchParams struct used for an OptiX launch
    /// (copied on the host just before uploading to the device).
    const LaunchParams& last_launch_params_for_test() const { return last_launch_params_host_; }

    /// Test hooks for volume photon grid.
    const PhotonSoA& volume_photons() const { return volume_photons_; }

    void cleanup();

private:
    void create_context();
    void create_module();
    void create_programs();
    void create_pipeline();
    void build_sbt(const Scene& scene);

    // Denoiser
    void setup_denoiser(int w, int h, bool guide_albedo, bool guide_normal);
    void run_denoiser(float blend_factor);
    void cleanup_denoiser();

    // OptiX handles
    OptixDeviceContext       context_      = nullptr;
    OptixModule              module_       = nullptr;
    OptixPipeline            pipeline_     = nullptr;

    // Denoiser handles
    OptixDenoiser            denoiser_     = nullptr;
    DeviceBuffer             d_denoiser_state_;
    DeviceBuffer             d_denoiser_scratch_;
    int                      denoiser_width_  = 0;
    int                      denoiser_height_ = 0;
    bool                     denoiser_guide_albedo_ = false;
    bool                     denoiser_guide_normal_ = false;

    // Program groups
    OptixProgramGroup        raygen_pg_          = nullptr;
    OptixProgramGroup        raygen_photon_pg_   = nullptr;  // photon trace
    OptixProgramGroup        raygen_targeted_pg_ = nullptr;  // targeted caustic photon trace
    OptixProgramGroup        miss_pg_            = nullptr;
    OptixProgramGroup        miss_shadow_pg_     = nullptr;
    OptixProgramGroup        hitgroup_pg_        = nullptr;
    OptixProgramGroup        hitgroup_shadow_pg_ = nullptr;

    // SBT
    OptixShaderBindingTable  sbt_          = {};

    // Acceleration structure
    OptixTraversableHandle   gas_handle_   = 0;
    DeviceBuffer             gas_buffer_;

    // Device buffers
    DeviceBuffer d_spectrum_buffer_;   // float [W*H*NUM_LAMBDA]
    DeviceBuffer d_sample_counts_;     // float [W*H]
    DeviceBuffer d_srgb_buffer_;       // uint8 [W*H*4]  (final — denoised when enabled)
    DeviceBuffer d_srgb_raw_buffer_;   // uint8 [W*H*4]  (raw tonemap, before denoiser)

    // AOV buffers for denoiser guide layers
    DeviceBuffer d_albedo_buffer_;     // float [W*H*4] linear diffuse albedo
    DeviceBuffer d_normal_buffer_;     // float [W*H*4] world-space shading normal

    // HDR intermediate buffer (denoiser input/output)
    DeviceBuffer d_hdr_buffer_;        // float [W*H*4] linear HDR RGB
    DeviceBuffer d_hdr_denoised_;      // float [W*H*4] denoised output

    // Component output buffers (for debug PNGs)
    DeviceBuffer d_nee_direct_buffer_;       // float [W*H*NUM_LAMBDA]
    DeviceBuffer d_photon_indirect_buffer_;  // float [W*H*NUM_LAMBDA]

    // Adaptive sampling buffers
    DeviceBuffer d_lum_sum_;          // float [W*H]
    DeviceBuffer d_lum_sum2_;         // float [W*H]
    DeviceBuffer d_active_mask_;      // uint8_t [W*H]
    DeviceBuffer d_pixel_max_spp_;    // uint16_t [W*H] — AS-02 per-pixel budget

    // Per-bounce AOV buffers (DB-04, §10.3)
    DeviceBuffer d_bounce_aov_[MAX_AOV_BOUNCES]; // float [W*H*NUM_LAMBDA] each

    // GPU kernel profiling buffers (long long [W*H] each)
    DeviceBuffer d_prof_total_;
    DeviceBuffer d_prof_ray_trace_;
    DeviceBuffer d_prof_nee_;
    DeviceBuffer d_prof_photon_gather_;
    DeviceBuffer d_prof_bsdf_;

    // Test hook: last launch params copied on host (for unit/integration tests)
    LaunchParams last_launch_params_host_ = {};

    // Scene geometry (device)
    DeviceBuffer d_vertices_;
    DeviceBuffer d_normals_;
    DeviceBuffer d_texcoords_;
    DeviceBuffer d_material_ids_;

    // Material data (device)
    DeviceBuffer d_Kd_;
    DeviceBuffer d_Ks_;
    DeviceBuffer d_Le_;
    DeviceBuffer d_roughness_;
    DeviceBuffer d_ior_;
    DeviceBuffer d_cauchy_A_;           // float  [num_materials]
    DeviceBuffer d_cauchy_B_;           // float  [num_materials]
    DeviceBuffer d_mat_dispersion_;     // uint8  [num_materials]
    DeviceBuffer d_mat_type_;
    DeviceBuffer d_diffuse_tex_;    // int [num_materials]
    DeviceBuffer d_emission_tex_;   // int [num_materials]
    DeviceBuffer d_opacity_;        // float [num_materials]

    // Per-material interior medium (§7.7 Translucent)
    DeviceBuffer d_mat_medium_id_;  // int [num_materials]  medium index or -1
    DeviceBuffer d_media_;          // HomogeneousMedium [num_media]

    // Clearcoat / Fabric per-material data
    DeviceBuffer d_clearcoat_weight_;    // float [num_materials]
    DeviceBuffer d_clearcoat_roughness_; // float [num_materials]
    DeviceBuffer d_sheen_;               // float [num_materials]
    DeviceBuffer d_sheen_tint_;          // float [num_materials]

    // Texture atlas (device)
    DeviceBuffer d_tex_atlas_;      // float [total_texels * 4]
    DeviceBuffer d_tex_descs_;      // GpuTexDesc [num_textures]

    // Photon data (device -- for hash grid lookups in render)
    DeviceBuffer d_photon_pos_x_, d_photon_pos_y_, d_photon_pos_z_;
    DeviceBuffer d_photon_wi_x_,  d_photon_wi_y_,  d_photon_wi_z_;
    DeviceBuffer d_photon_norm_x_, d_photon_norm_y_, d_photon_norm_z_;  // surface normals
    DeviceBuffer d_photon_lambda_, d_photon_flux_;
    DeviceBuffer d_photon_num_hero_;  // uint8_t [num_photons] hero count per photon
    DeviceBuffer d_photon_is_caustic_pass_;  // uint8_t [num_photons] 0=global, 1=caustic-targeted
    DeviceBuffer d_grid_sorted_indices_, d_grid_cell_start_, d_grid_cell_end_;

    // GPU hash grid build scratch buffers (CUB radix sort)
    DeviceBuffer d_grid_keys_in_, d_grid_keys_out_;
    DeviceBuffer d_grid_indices_in_;
    DeviceBuffer d_grid_cub_temp_;

    // Emitter data (device -- for GPU photon tracing)
    DeviceBuffer d_emissive_indices_;
    DeviceBuffer d_emissive_cdf_;
    DeviceBuffer d_emissive_local_idx_;  // int [num_tris] inverse lookup

    // Targeted caustic specular triangle data (device)
    DeviceBuffer d_targeted_spec_tri_indices_;   // uint32_t [N]
    DeviceBuffer d_targeted_spec_alias_prob_;    // float    [N]
    DeviceBuffer d_targeted_spec_alias_idx_;     // uint32_t [N]
    DeviceBuffer d_targeted_spec_pdf_;           // float    [N]
    DeviceBuffer d_targeted_spec_areas_;         // float    [N]
    int          num_targeted_spec_tris_ = 0;

    // Photon output buffers (device -- written by __raygen__photon_trace)
    DeviceBuffer d_out_photon_pos_x_, d_out_photon_pos_y_, d_out_photon_pos_z_;
    DeviceBuffer d_out_photon_wi_x_,  d_out_photon_wi_y_,  d_out_photon_wi_z_;
    DeviceBuffer d_out_photon_norm_x_, d_out_photon_norm_y_, d_out_photon_norm_z_;  // surface normals
    DeviceBuffer d_out_photon_lambda_, d_out_photon_flux_;
    DeviceBuffer d_out_photon_num_hero_;  // uint8_t [max_stored] hero count per photon
    DeviceBuffer d_out_photon_source_emissive_;  // uint16_t [max_stored] source emissive idx
    DeviceBuffer d_out_photon_is_caustic_;       // uint8_t  [max_stored] caustic flag
    DeviceBuffer d_out_photon_path_flags_;       // uint8_t  [max_stored] PHOTON_FLAG_* bits
    DeviceBuffer d_out_photon_tri_id_;           // uint32_t [max_stored] deposit triangle
    DeviceBuffer d_out_photon_count_;

    // Per-triangle photon irradiance heatmap (for preview visualization)
    DeviceBuffer d_tri_photon_irradiance_;   // float [num_tris]
    int          num_tris_ = 0;              // host triangle count

    // Volume photon output buffers (device -- written by __raygen__photon_trace)
    DeviceBuffer d_out_vol_photon_pos_x_, d_out_vol_photon_pos_y_, d_out_vol_photon_pos_z_;
    DeviceBuffer d_out_vol_photon_wi_x_,  d_out_vol_photon_wi_y_,  d_out_vol_photon_wi_z_;
    DeviceBuffer d_out_vol_photon_lambda_, d_out_vol_photon_flux_;
    DeviceBuffer d_out_vol_photon_count_;

    // SBT record buffers
    DeviceBuffer d_raygen_record_;
    DeviceBuffer d_raygen_photon_record_;
    DeviceBuffer d_raygen_targeted_record_;
    DeviceBuffer d_miss_records_;
    DeviceBuffer d_hitgroup_records_;

    // Launch params (on device)
    DeviceBuffer d_launch_params_;

    // Stored photon data & hash grid (CPU side, after trace_photons())
    PhotonSoA stored_photons_;
    HashGrid  stored_grid_;
    std::vector<uint8_t> caustic_flags_;  // per-photon caustic flag (downloaded from GPU)

    // Host-side scene triangles for debug ray picking/hover inspection.
    std::vector<Triangle> host_triangles_;

    // Host-side cell-bin grid (kept after build for save/test access)
    CellBinGrid cell_bin_grid_;  // empty unless dense grid is built

    // Device-side cell-bin grid (for volume guide / legacy dense gather)
    DeviceBuffer d_cell_bin_grid_;  // PhotonBin [total_cells * bin_count]

    // ── Multi-resolution hash histogram (replaces CellBinGrid for guide) ──
    HashHistogram hash_histogram_;
    DeviceBuffer  d_guide_histogram_[MAX_GUIDE_LEVELS];  // GpuGuideBin per level

    // Per-cell photon analysis (PA-08: GPU upload buffers)
    DeviceBuffer d_cell_guide_fraction_;
    DeviceBuffer d_cell_caustic_fraction_;
    DeviceBuffer d_cell_flux_density_;
    int          cell_analysis_count_ = 0;  // number of cells uploaded
    ConclusionCounters cell_conclusions_;     // from last build_cell_analysis()

    // Volume photon storage + spatial indices (VP-02/03)
    PhotonSoA   volume_photons_;
    HashGrid    volume_grid_;          // VP-02: 3D hash grid for volume photon kNN
    CellBinGrid vol_cell_bin_grid_;    // VP-03: directional histogram grid for volume guide

    // Device-side volume cell-bin grid (VP-07)
    DeviceBuffer d_vol_cell_bin_grid_; // PhotonBin [vol_total_cells * bin_count]

    // Device-side volume photon kNN data (MT-07)
    DeviceBuffer d_vol_photon_pos_x_, d_vol_photon_pos_y_, d_vol_photon_pos_z_;
    DeviceBuffer d_vol_photon_wi_x_,  d_vol_photon_wi_y_,  d_vol_photon_wi_z_;
    DeviceBuffer d_vol_photon_lambda_, d_vol_photon_flux_;
    DeviceBuffer d_vol_grid_sorted_indices_, d_vol_grid_cell_start_, d_vol_grid_cell_end_;

    // SPPM per-pixel buffers (GPU side)
    DeviceBuffer d_sppm_vp_pos_x_,  d_sppm_vp_pos_y_,  d_sppm_vp_pos_z_;
    DeviceBuffer d_sppm_vp_norm_x_, d_sppm_vp_norm_y_, d_sppm_vp_norm_z_;
    DeviceBuffer d_sppm_vp_wo_x_,   d_sppm_vp_wo_y_,   d_sppm_vp_wo_z_;
    DeviceBuffer d_sppm_vp_mat_id_;
    DeviceBuffer d_sppm_vp_uv_u_, d_sppm_vp_uv_v_;
    DeviceBuffer d_sppm_vp_throughput_;   // float [W*H*NUM_LAMBDA]
    DeviceBuffer d_sppm_vp_valid_;        // uint8_t [W*H]
    DeviceBuffer d_sppm_radius_;          // float [W*H]
    DeviceBuffer d_sppm_N_;               // float [W*H]
    DeviceBuffer d_sppm_tau_;             // float [W*H*NUM_LAMBDA]
    DeviceBuffer d_sppm_L_direct_;        // float [W*H*NUM_LAMBDA]

    int  width_       = DEFAULT_IMAGE_WIDTH;
    int  height_      = DEFAULT_IMAGE_HEIGHT;
    bool initialised_ = false;
    int  num_emissive_ = 0;
    int  num_photons_emitted_ = 0;  // N_emitted for density normalisation
    int  num_caustic_emitted_ = 0;  // N_caustic for dual-budget caustic normalisation
    std::vector<uint8_t> caustic_pass_flags_;  // per-photon: 0=global, 1=caustic-targeted
    bool runtime_volume_enabled_ = DEFAULT_VOLUME_ENABLED;  // toggled via V key
    bool show_photon_heatmap_    = false;                    // toggled via F-key
    bool use_dense_grid_         = false;                    // dense cell-bin grid path
    bool denoiser_enabled_       = DEFAULT_DENOISER_ENABLED; // OptiX AI denoiser
    float gather_radius_ = DEFAULT_GATHER_RADIUS;
    float caustic_radius_ = DEFAULT_CAUSTIC_RADIUS;   // tighter radius for caustic gather
    float exposure_      = DEFAULT_EXPOSURE;          // runtime exposure (set_exposure / R key)

    // Runtime guide fraction (T key toggle: 0 = unguided, DEFAULT = guided)
    float guide_fraction_ = DEFAULT_GUIDE_FRACTION;
    bool  histogram_only_ = false;  // C key: use only histogram conclusions

    // GPU device info (populated in init())
    std::string gpu_name_;
    size_t      gpu_vram_total_ = 0;
    int         gpu_sm_count_   = 0;
    int         gpu_cc_major_   = 0;
    int         gpu_cc_minor_   = 0;
};

// ---------------------------------------------------------------------
// Inline / template implementations
// ---------------------------------------------------------------------

inline void OptixRenderer::resize(int w, int h) {
    // Guard: skip if dimensions match AND buffers are already allocated.
    // Without the d_ptr check the very first call (where w/h already
    // equal the default members) would return without allocating.
    if (w == width_ && h == height_ && d_spectrum_buffer_.d_ptr) return;
    width_  = w;
    height_ = h;
    // Cell-bin grid is independent of resolution; don't invalidate here.

    size_t pixels = (size_t)w * h;
    d_spectrum_buffer_.alloc(pixels * NUM_LAMBDA * sizeof(float));
    d_sample_counts_.alloc(pixels * sizeof(float));
    d_srgb_buffer_.alloc(pixels * 4 * sizeof(uint8_t));
    d_srgb_raw_buffer_.alloc(pixels * 4 * sizeof(uint8_t));

    // AOV + HDR buffers for denoiser
    d_albedo_buffer_.alloc(pixels * 4 * sizeof(float));
    d_normal_buffer_.alloc(pixels * 4 * sizeof(float));
    d_hdr_buffer_.alloc(pixels * 4 * sizeof(float));
    d_hdr_denoised_.alloc(pixels * 4 * sizeof(float));

    // Adaptive sampling buffers
    d_lum_sum_.alloc(pixels * sizeof(float));
    d_lum_sum2_.alloc(pixels * sizeof(float));
    d_active_mask_.alloc(pixels * sizeof(uint8_t));
    d_pixel_max_spp_.alloc(pixels * sizeof(uint16_t));

    // Component buffers
    d_nee_direct_buffer_.alloc(pixels * NUM_LAMBDA * sizeof(float));
    d_photon_indirect_buffer_.alloc(pixels * NUM_LAMBDA * sizeof(float));

    // Per-bounce AOV buffers (DB-04)
    for (int b = 0; b < MAX_AOV_BOUNCES; ++b)
        d_bounce_aov_[b].alloc(pixels * NUM_LAMBDA * sizeof(float));

    // Profiling buffers
    d_prof_total_.alloc(pixels * sizeof(long long));
    d_prof_ray_trace_.alloc(pixels * sizeof(long long));
    d_prof_nee_.alloc(pixels * sizeof(long long));
    d_prof_photon_gather_.alloc(pixels * sizeof(long long));
    d_prof_bsdf_.alloc(pixels * sizeof(long long));

    // Zero the buffers
    CUDA_CHECK(cudaMemset(d_spectrum_buffer_.d_ptr, 0, d_spectrum_buffer_.bytes));
    CUDA_CHECK(cudaMemset(d_sample_counts_.d_ptr,   0, d_sample_counts_.bytes));
    CUDA_CHECK(cudaMemset(d_srgb_buffer_.d_ptr,     0, d_srgb_buffer_.bytes));
    CUDA_CHECK(cudaMemset(d_srgb_raw_buffer_.d_ptr, 0, d_srgb_raw_buffer_.bytes));
    CUDA_CHECK(cudaMemset(d_albedo_buffer_.d_ptr,   0, d_albedo_buffer_.bytes));
    CUDA_CHECK(cudaMemset(d_normal_buffer_.d_ptr,   0, d_normal_buffer_.bytes));
    CUDA_CHECK(cudaMemset(d_hdr_buffer_.d_ptr,      0, d_hdr_buffer_.bytes));
    CUDA_CHECK(cudaMemset(d_hdr_denoised_.d_ptr,    0, d_hdr_denoised_.bytes));
    CUDA_CHECK(cudaMemset(d_lum_sum_.d_ptr,          0, d_lum_sum_.bytes));
    CUDA_CHECK(cudaMemset(d_lum_sum2_.d_ptr,         0, d_lum_sum2_.bytes));
    CUDA_CHECK(cudaMemset(d_active_mask_.d_ptr,      0, d_active_mask_.bytes));
    CUDA_CHECK(cudaMemset(d_pixel_max_spp_.d_ptr,   0, d_pixel_max_spp_.bytes));
    CUDA_CHECK(cudaMemset(d_nee_direct_buffer_.d_ptr, 0, d_nee_direct_buffer_.bytes));
    CUDA_CHECK(cudaMemset(d_photon_indirect_buffer_.d_ptr, 0, d_photon_indirect_buffer_.bytes));
    for (int b = 0; b < MAX_AOV_BOUNCES; ++b)
        CUDA_CHECK(cudaMemset(d_bounce_aov_[b].d_ptr, 0, d_bounce_aov_[b].bytes));

    // Profiling buffers
    CUDA_CHECK(cudaMemset(d_prof_total_.d_ptr, 0, d_prof_total_.bytes));
    CUDA_CHECK(cudaMemset(d_prof_ray_trace_.d_ptr, 0, d_prof_ray_trace_.bytes));
    CUDA_CHECK(cudaMemset(d_prof_nee_.d_ptr, 0, d_prof_nee_.bytes));
    CUDA_CHECK(cudaMemset(d_prof_photon_gather_.d_ptr, 0, d_prof_photon_gather_.bytes));
    CUDA_CHECK(cudaMemset(d_prof_bsdf_.d_ptr, 0, d_prof_bsdf_.bytes));
}

/// Zero the accumulation buffers without reallocating (for camera movement)
inline void OptixRenderer::clear_buffers() {
    if (d_spectrum_buffer_.d_ptr)
        CUDA_CHECK(cudaMemset(d_spectrum_buffer_.d_ptr, 0, d_spectrum_buffer_.bytes));
    if (d_sample_counts_.d_ptr)
        CUDA_CHECK(cudaMemset(d_sample_counts_.d_ptr, 0, d_sample_counts_.bytes));
    if (d_srgb_buffer_.d_ptr)
        CUDA_CHECK(cudaMemset(d_srgb_buffer_.d_ptr, 0, d_srgb_buffer_.bytes));
    if (d_srgb_raw_buffer_.d_ptr)
        CUDA_CHECK(cudaMemset(d_srgb_raw_buffer_.d_ptr, 0, d_srgb_raw_buffer_.bytes));
    if (d_albedo_buffer_.d_ptr)
        CUDA_CHECK(cudaMemset(d_albedo_buffer_.d_ptr, 0, d_albedo_buffer_.bytes));
    if (d_normal_buffer_.d_ptr)
        CUDA_CHECK(cudaMemset(d_normal_buffer_.d_ptr, 0, d_normal_buffer_.bytes));
    if (d_lum_sum_.d_ptr)
        CUDA_CHECK(cudaMemset(d_lum_sum_.d_ptr, 0, d_lum_sum_.bytes));
    if (d_lum_sum2_.d_ptr)
        CUDA_CHECK(cudaMemset(d_lum_sum2_.d_ptr, 0, d_lum_sum2_.bytes));
    if (d_active_mask_.d_ptr)
        CUDA_CHECK(cudaMemset(d_active_mask_.d_ptr, 0, d_active_mask_.bytes));
    if (d_nee_direct_buffer_.d_ptr)
        CUDA_CHECK(cudaMemset(d_nee_direct_buffer_.d_ptr, 0, d_nee_direct_buffer_.bytes));
    if (d_photon_indirect_buffer_.d_ptr)
        CUDA_CHECK(cudaMemset(d_photon_indirect_buffer_.d_ptr, 0, d_photon_indirect_buffer_.bytes));
    for (int b = 0; b < MAX_AOV_BOUNCES; ++b)
        if (d_bounce_aov_[b].d_ptr)
            CUDA_CHECK(cudaMemset(d_bounce_aov_[b].d_ptr, 0, d_bounce_aov_[b].bytes));
    if (d_prof_total_.d_ptr)
        CUDA_CHECK(cudaMemset(d_prof_total_.d_ptr, 0, d_prof_total_.bytes));
    if (d_prof_ray_trace_.d_ptr)
        CUDA_CHECK(cudaMemset(d_prof_ray_trace_.d_ptr, 0, d_prof_ray_trace_.bytes));
    if (d_prof_nee_.d_ptr)
        CUDA_CHECK(cudaMemset(d_prof_nee_.d_ptr, 0, d_prof_nee_.bytes));
    if (d_prof_photon_gather_.d_ptr)
        CUDA_CHECK(cudaMemset(d_prof_photon_gather_.d_ptr, 0, d_prof_photon_gather_.bytes));
    if (d_prof_bsdf_.d_ptr)
        CUDA_CHECK(cudaMemset(d_prof_bsdf_.d_ptr, 0, d_prof_bsdf_.bytes));
}

inline void OptixRenderer::download_framebuffer(FrameBuffer& fb) const {
    fb.resize(width_, height_);
    CUDA_CHECK(cudaMemcpy(fb.srgb.data(), d_srgb_buffer_.d_ptr,
                          fb.srgb.size(), cudaMemcpyDeviceToHost));
}

inline void OptixRenderer::download_raw_framebuffer(FrameBuffer& fb) const {
    fb.resize(width_, height_);
    CUDA_CHECK(cudaMemcpy(fb.srgb.data(), d_srgb_raw_buffer_.d_ptr,
                          fb.srgb.size(), cudaMemcpyDeviceToHost));
}

inline void OptixRenderer::download_component_buffers(
    std::vector<float>& nee_direct,
    std::vector<float>& photon_indirect,
    std::vector<float>& sample_counts) const
{
    size_t pixels = (size_t)width_ * height_;
    size_t spec_size = pixels * NUM_LAMBDA;

    nee_direct.resize(spec_size);
    photon_indirect.resize(spec_size);
    sample_counts.resize(pixels);

    if (d_nee_direct_buffer_.d_ptr)
        CUDA_CHECK(cudaMemcpy(nee_direct.data(), d_nee_direct_buffer_.d_ptr,
                              spec_size * sizeof(float), cudaMemcpyDeviceToHost));
    if (d_photon_indirect_buffer_.d_ptr)
        CUDA_CHECK(cudaMemcpy(photon_indirect.data(), d_photon_indirect_buffer_.d_ptr,
                              spec_size * sizeof(float), cudaMemcpyDeviceToHost));
    if (d_sample_counts_.d_ptr)
        CUDA_CHECK(cudaMemcpy(sample_counts.data(), d_sample_counts_.d_ptr,
                              pixels * sizeof(float), cudaMemcpyDeviceToHost));
}

inline void OptixRenderer::cleanup() {
    if (denoiser_)             { optixDenoiserDestroy(denoiser_);               denoiser_ = nullptr; }
    if (pipeline_)             { optixPipelineDestroy(pipeline_);              pipeline_ = nullptr; }
    if (raygen_pg_)            { optixProgramGroupDestroy(raygen_pg_);         raygen_pg_ = nullptr; }
    if (raygen_photon_pg_)     { optixProgramGroupDestroy(raygen_photon_pg_);  raygen_photon_pg_ = nullptr; }
    if (raygen_targeted_pg_)   { optixProgramGroupDestroy(raygen_targeted_pg_); raygen_targeted_pg_ = nullptr; }
    if (miss_pg_)              { optixProgramGroupDestroy(miss_pg_);           miss_pg_ = nullptr; }
    if (miss_shadow_pg_)       { optixProgramGroupDestroy(miss_shadow_pg_);    miss_shadow_pg_ = nullptr; }
    if (hitgroup_pg_)          { optixProgramGroupDestroy(hitgroup_pg_);       hitgroup_pg_ = nullptr; }
    if (hitgroup_shadow_pg_)   { optixProgramGroupDestroy(hitgroup_shadow_pg_);hitgroup_shadow_pg_ = nullptr; }
    if (module_)               { optixModuleDestroy(module_);                  module_ = nullptr; }
    if (context_)              { optixDeviceContextDestroy(context_);          context_ = nullptr; }
    initialised_ = false;
}

