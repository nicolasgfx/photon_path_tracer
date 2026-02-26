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
#include "core/cell_bin_grid.h"
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

    /// Upload emitter data to device (for GPU photon tracing)
    void upload_emitter_data(const Scene& scene);

    /// CPU photon tracing fallback: uses emitter.h trace_photons() +
    /// targeted caustic emission on the CPU, then merges, builds hash
    /// grid, and uploads to GPU for the OptiX gather.  Enabled when
    /// USE_CPU_PHOTON_TRACE is 1 in config.h.
    void cpu_trace_photons(const Scene& scene, const RenderConfig& config);

    /// Trace photons entirely on the GPU. Downloads results, builds
    /// the hash grid on CPU, uploads the grid + photon data back.
    /// @param grid_radius_override  If > 0, use this radius instead of
    ///     config.gather_radius for the hash grid build and GPU cell size.
    ///     Used by SPPM to size cells for the SPPM initial radius.
    void trace_photons(const Scene& scene, const RenderConfig& config,
                       float grid_radius_override = 0.f,
                       int photon_map_seed = 0);

    // -- Rendering ----------------------------------------------------

    /// Launch a single frame of the debug viewer (1 spp, progressive)
    /// @param shadow_rays  If true, debug_first_hit casts shadow rays (for NEE PNG)
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

    /// Read back the sRGB buffer from the device
    void download_framebuffer(FrameBuffer& fb) const;

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

    /// Toggle dense cell-bin grid path vs hash-grid walk.
    void set_use_dense_grid(bool v) { use_dense_grid_ = v; }
    bool is_use_dense_grid() const  { return use_dense_grid_; }

    /// Render coverage debug PNG (stub — no-op when DEBUG_COMPONENT_PNGS is false).
    template<typename CamT, typename FbT>
    void render_coverage_debug_png(const CamT&, const FbT&) {}

    /// Download GPU kernel profiling data and print a summary.
    /// Call after the final render completes.
    void print_kernel_profiling() const;

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

    // OptiX handles
    OptixDeviceContext       context_      = nullptr;
    OptixModule              module_       = nullptr;
    OptixPipeline            pipeline_     = nullptr;

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
    DeviceBuffer d_srgb_buffer_;       // uint8 [W*H*4]

    // Component output buffers (for debug PNGs)
    DeviceBuffer d_nee_direct_buffer_;       // float [W*H*NUM_LAMBDA]
    DeviceBuffer d_photon_indirect_buffer_;  // float [W*H*NUM_LAMBDA]

    // Per-pixel lobe balance (Bresenham accumulator)
    DeviceBuffer d_lobe_balance_;     // float [W*H]

    // Adaptive sampling buffers
    DeviceBuffer d_lum_sum_;          // float [W*H]
    DeviceBuffer d_lum_sum2_;         // float [W*H]
    DeviceBuffer d_active_mask_;      // uint8_t [W*H]

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

    // Volume photon storage
    PhotonSoA volume_photons_;

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
    float gather_radius_ = DEFAULT_GATHER_RADIUS;
    float caustic_radius_ = DEFAULT_CAUSTIC_RADIUS;   // tighter radius for caustic gather
    float exposure_      = DEFAULT_EXPOSURE;          // runtime exposure (set_exposure / R key)
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

    // Per-pixel lobe balance
    d_lobe_balance_.alloc(pixels * sizeof(float));

    // Adaptive sampling buffers
    d_lum_sum_.alloc(pixels * sizeof(float));
    d_lum_sum2_.alloc(pixels * sizeof(float));
    d_active_mask_.alloc(pixels * sizeof(uint8_t));

    // Component buffers
    d_nee_direct_buffer_.alloc(pixels * NUM_LAMBDA * sizeof(float));
    d_photon_indirect_buffer_.alloc(pixels * NUM_LAMBDA * sizeof(float));

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
    CUDA_CHECK(cudaMemset(d_lobe_balance_.d_ptr,     0, d_lobe_balance_.bytes));
    CUDA_CHECK(cudaMemset(d_lum_sum_.d_ptr,          0, d_lum_sum_.bytes));
    CUDA_CHECK(cudaMemset(d_lum_sum2_.d_ptr,         0, d_lum_sum2_.bytes));
    CUDA_CHECK(cudaMemset(d_active_mask_.d_ptr,      0, d_active_mask_.bytes));
    CUDA_CHECK(cudaMemset(d_nee_direct_buffer_.d_ptr, 0, d_nee_direct_buffer_.bytes));
    CUDA_CHECK(cudaMemset(d_photon_indirect_buffer_.d_ptr, 0, d_photon_indirect_buffer_.bytes));

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

