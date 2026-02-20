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
    void upload_photon_data(const PhotonSoA& global_photons,
                            const HashGrid& global_grid,
                            const PhotonSoA& caustic_photons,
                            const HashGrid& caustic_grid,
                            float gather_radius,
                            float caustic_radius);

    /// Upload emitter data to device (for GPU photon tracing)
    void upload_emitter_data(const Scene& scene);

    /// Trace photons entirely on the GPU. Downloads results, builds
    /// the hash grid on CPU, uploads the grid + photon data back.
    void trace_photons(const Scene& scene, const RenderConfig& config);

    // -- Rendering ----------------------------------------------------

    /// Launch a single frame of the debug viewer (1 spp, progressive)
    /// @param shadow_rays  If true, debug_first_hit casts shadow rays (for NEE PNG)
    void render_debug_frame(const Camera& camera, int frame_number,
                            RenderMode mode, int spp = 1,
                            bool shadow_rays = false);

    /// Launch the full final render (multi-spp, blocking)
    void render_final(const Camera& camera, const RenderConfig& config);

    /// Launch a single sample of full path tracing (progressive)
    void render_one_spp(const Camera& camera, int frame_number,
                        int max_bounces = DEFAULT_MAX_BOUNCES);

    /// Build the dense 3D cell-bin grid from stored photons and upload
    /// it to the device.  Called once after trace_photons().
    void build_cell_bin_grid();

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

    /// Download GPU kernel profiling data and print a summary.
    /// Call after the final render completes.
    void print_kernel_profiling() const;

    /// Access the stored photon data and hash grid (available after
    /// trace_photons() completes)
    const PhotonSoA& photons() const { return stored_photons_; }
    const HashGrid&  grid()    const { return stored_grid_; }

    /// Test hook: returns the last LaunchParams struct used for an OptiX launch
    /// (copied on the host just before uploading to the device).
    const LaunchParams& last_launch_params_for_test() const { return last_launch_params_host_; }

    /// Test hook: allocated byte size for cell-bin grid.
    size_t cell_bin_grid_bytes_for_test() const { return d_cell_bin_grid_.bytes; }
    /// Test hook: access the host-side cell-bin grid.
    const CellBinGrid& cell_bin_grid_for_test() const { return cell_bin_grid_; }

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

    // Dense 3D cell-bin grid
    DeviceBuffer d_cell_bin_grid_;          // PhotonBin [total_cells*PHOTON_BIN_COUNT]

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
    DeviceBuffer d_mat_type_;

    // Photon data (device -- for hash grid lookups in render)
    DeviceBuffer d_photon_pos_x_, d_photon_pos_y_, d_photon_pos_z_;
    DeviceBuffer d_photon_wi_x_,  d_photon_wi_y_,  d_photon_wi_z_;
    DeviceBuffer d_photon_lambda_, d_photon_flux_;
    DeviceBuffer d_photon_bin_idx_;  // uint8_t [num_photons] precomputed bin index
    DeviceBuffer d_grid_sorted_indices_, d_grid_cell_start_, d_grid_cell_end_;

    // Emitter data (device -- for GPU photon tracing)
    DeviceBuffer d_emissive_indices_;
    DeviceBuffer d_emissive_cdf_;

    // Photon output buffers (device -- written by __raygen__photon_trace)
    DeviceBuffer d_out_photon_pos_x_, d_out_photon_pos_y_, d_out_photon_pos_z_;
    DeviceBuffer d_out_photon_wi_x_,  d_out_photon_wi_y_,  d_out_photon_wi_z_;
    DeviceBuffer d_out_photon_lambda_, d_out_photon_flux_;
    DeviceBuffer d_out_photon_count_;

    // SBT record buffers
    DeviceBuffer d_raygen_record_;
    DeviceBuffer d_raygen_photon_record_;
    DeviceBuffer d_miss_records_;
    DeviceBuffer d_hitgroup_records_;

    // Launch params (on device)
    DeviceBuffer d_launch_params_;

    // Stored photon data & hash grid (CPU side, after trace_photons())
    PhotonSoA stored_photons_;
    HashGrid  stored_grid_;

    // Host-side scene triangles for debug ray picking/hover inspection.
    std::vector<Triangle> host_triangles_;

    // Host-side cell-bin grid (kept after build for save/test access)
    CellBinGrid cell_bin_grid_;
    bool cell_grid_uploaded_ = false;

    int  width_       = DEFAULT_IMAGE_WIDTH;
    int  height_      = DEFAULT_IMAGE_HEIGHT;
    bool initialised_ = false;
    int  num_emissive_ = 0;
    float gather_radius_ = DEFAULT_GATHER_RADIUS;
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
    if (miss_pg_)              { optixProgramGroupDestroy(miss_pg_);           miss_pg_ = nullptr; }
    if (miss_shadow_pg_)       { optixProgramGroupDestroy(miss_shadow_pg_);    miss_shadow_pg_ = nullptr; }
    if (hitgroup_pg_)          { optixProgramGroupDestroy(hitgroup_pg_);       hitgroup_pg_ = nullptr; }
    if (hitgroup_shadow_pg_)   { optixProgramGroupDestroy(hitgroup_shadow_pg_);hitgroup_shadow_pg_ = nullptr; }
    if (module_)               { optixModuleDestroy(module_);                  module_ = nullptr; }
    if (context_)              { optixDeviceContextDestroy(context_);          context_ = nullptr; }
    initialised_ = false;
}

