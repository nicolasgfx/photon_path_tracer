#pragma once
// ─────────────────────────────────────────────────────────────────────
// accel_builder.h – OptiX pipeline and acceleration structure builder
//
// Phase 3: Acceleration stage.  Focused component that manages:
//   - OptiX context, module, program groups, pipeline
//   - GAS (single geometry) and IAS (instanced) acceleration builds
//   - SBT (shader binding table) construction
//   - Scene geometry upload to device
//
// Usage:
//   AccelBuilder builder;
//   builder.init();
//   builder.build(scene, ptx_path);
//   builder.upload_geometry(scene);
//   auto accel = builder.accel();  // traversable handle + stats
//   builder.launch_test(width, height, camera);  // diagnostic
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/device_buffer.h"
#include "accel/accel_types.h"
#include "accel/launch_params.h"
#include "scene/scene.h"

#include <optix.h>
#include <optix_stubs.h>
#include <string>
#include <vector>

class AccelBuilder {
public:
    AccelBuilder() = default;
    ~AccelBuilder() { cleanup(); }

    // Non-copyable
    AccelBuilder(const AccelBuilder&) = delete;
    AccelBuilder& operator=(const AccelBuilder&) = delete;

    // ── Initialization ───────────────────────────────────────────────
    // Initialize CUDA + OptiX. Must be called before anything else.
    void init();
    bool is_initialised() const { return initialised_; }

    // ── Full build (context + accel + module + pipeline + SBT) ───────
    // Call this with the scene and path to the compiled PTX file.
    void build(const Scene& scene, const std::string& ptx_path);

    // ── Scene data upload ────────────────────────────────────────────
    // Uploads geometry (vertices, normals, texcoords, material_ids)
    // and material arrays to device memory.  Must be called after
    // build() so that SBT records can reference the device pointers.
    void upload_geometry(const Scene& scene);
    void upload_materials(const Scene& scene);
    void upload_textures(const Scene& scene);

    // ── Accessors ────────────────────────────────────────────────────
    const AccelStructure& accel() const { return accel_; }
    OptixPipeline         pipeline() const { return pipeline_; }
    const OptixShaderBindingTable& sbt() const { return sbt_; }

    // Fill geometry and traversable fields in a LaunchParams struct.
    // Non-const: returns mutable device pointers from owned buffers.
    void fill_geometry_params(LaunchParams& lp);
    void fill_material_params(LaunchParams& lp);

    // ── Diagnostic launch ────────────────────────────────────────────
    // Launches __raygen__test_normals and downloads the result.
    // color_out is resized to [width * height * 3].
    void launch_test_normals(int width, int height,
                             float3 cam_pos, float3 cam_u,
                             float3 cam_v, float3 cam_w,
                             std::vector<float>& color_out);

    // Launches __raygen__test_nee for direct lighting diagnostics.
    // Requires emissive data to be filled in LaunchParams.
    void launch_test_nee(int width, int height,
                         float3 cam_pos, float3 cam_u,
                         float3 cam_v, float3 cam_w,
                         const LaunchParams& extra_params,
                         std::vector<float>& color_out);

    // Launches __raygen__render for full path tracing.
    // extra_params must have emissive data filled in.
    // Rendering parameters (bounces, RR, SPP) are set from extra_params.
    void launch_render(int width, int height,
                       float3 cam_pos, float3 cam_u,
                       float3 cam_v, float3 cam_w,
                       const LaunchParams& extra_params,
                       std::vector<float>& color_out);

    // Low-level progressive launch: uploads lp as-is and calls optixLaunch.
    // No buffer allocation, no zeroing, no download.  For use by RenderSession
    // which manages its own persistent device buffers across frames.
    void launch_progressive(int width, int height, const LaunchParams& lp);

    // Launch __raygen__caustic for caustic light tracing.
    // Grid is (num_photons, 1, 1) — one thread per photon.
    void launch_caustic(int num_photons, const LaunchParams& lp);

    // ── GPU info ─────────────────────────────────────────────────────
    const std::string& gpu_name() const { return gpu_name_; }
    size_t gpu_vram_total()       const { return gpu_vram_total_; }
    int    gpu_sm_count()         const { return gpu_sm_count_; }

        // ── OptiX context (needed by subsystems that reuse the context) ──
        OptixDeviceContext optix_context() const { return context_; }

    // ── Cleanup ──────────────────────────────────────────────────────
    void cleanup();

private:
    void create_context();
    void create_module(const std::string& ptx);
    void create_programs();
    void create_pipeline();
    void build_gas(const Scene& scene);
    void build_ias(const Scene& scene);
    void build_sbt();

    // State
    bool initialised_ = false;

    // OptiX handles
    OptixDeviceContext  context_  = nullptr;
    OptixModule         module_   = nullptr;
    OptixPipeline       pipeline_ = nullptr;

    // Program groups (2 ray types: radiance + shadow)
    OptixProgramGroup   raygen_test_pg_   = nullptr; // test_normals
    OptixProgramGroup   raygen_nee_pg_    = nullptr; // test_nee
    OptixProgramGroup   raygen_render_pg_ = nullptr; // __raygen__render
    OptixProgramGroup   raygen_caustic_pg_ = nullptr; // __raygen__caustic
    OptixProgramGroup   miss_pg_          = nullptr; // radiance
    OptixProgramGroup   miss_shadow_pg_   = nullptr; // shadow
    OptixProgramGroup   hitgroup_pg_      = nullptr; // radiance
    OptixProgramGroup   hitgroup_shadow_pg_ = nullptr; // shadow

    // SBT
    OptixShaderBindingTable sbt_ = {};
    OptixShaderBindingTable sbt_nee_ = {};     // SBT with raygen_nee
    OptixShaderBindingTable sbt_render_ = {};  // SBT with raygen_render
    OptixShaderBindingTable sbt_caustic_ = {}; // SBT with raygen_caustic
    DeviceBuffer<RayGenRecord>   d_raygen_record_;
    DeviceBuffer<RayGenRecord>   d_raygen_nee_record_;
    DeviceBuffer<RayGenRecord>   d_raygen_render_record_;
    DeviceBuffer<RayGenRecord>   d_raygen_caustic_record_;
    DeviceBuffer<MissRecord>     d_miss_records_;
    DeviceBuffer<HitGroupRecord> d_hitgroup_records_;

    // Acceleration structure
    AccelStructure accel_;
    DeviceBuffer<uint8_t> gas_buffer_;        // compacted GAS
    DeviceBuffer<uint8_t> d_vertices_raw_;    // vertex buffer for GAS build

    // IAS path
    std::vector<OptixTraversableHandle> ias_gas_handles_;
    std::vector<DeviceBuffer<uint8_t>>  ias_gas_buffers_;
    DeviceBuffer<uint8_t>               ias_buffer_;
    DeviceBuffer<OptixInstance>          d_ias_instances_;

    // Scene geometry (device)
    DeviceBuffer<float3>   d_vertices_;
    DeviceBuffer<float3>   d_normals_;
    DeviceBuffer<float3>   d_tangents_;
    DeviceBuffer<float2>   d_texcoords_;
    DeviceBuffer<uint32_t> d_material_ids_;

    // Material arrays (device)
    DeviceBuffer<float>    d_Kd_;
    DeviceBuffer<float>    d_Ks_;
    DeviceBuffer<float>    d_Le_;
    DeviceBuffer<float>    d_Tf_;
    DeviceBuffer<float>    d_roughness_;
    DeviceBuffer<float>    d_ior_;
    DeviceBuffer<uint8_t>  d_mat_type_;
    DeviceBuffer<float>    d_opacity_;
    DeviceBuffer<uint8_t>  d_mat_thin_;
    DeviceBuffer<float>    d_cauchy_A_;
    DeviceBuffer<float>    d_cauchy_B_;
    DeviceBuffer<float>    d_clearcoat_weight_;
    DeviceBuffer<float>    d_clearcoat_roughness_;
    DeviceBuffer<float>    d_sheen_;
    DeviceBuffer<float>    d_sheen_tint_;

    // Per-material texture ID arrays (device)
    DeviceBuffer<int>      d_diffuse_tex_;
    DeviceBuffer<int>      d_specular_tex_;
    DeviceBuffer<int>      d_emission_tex_;
    DeviceBuffer<int>      d_bump_tex_;
    DeviceBuffer<int>      d_normal_tex_;
    DeviceBuffer<int>      d_alpha_tex_;
    DeviceBuffer<int>      d_displacement_tex_;
    DeviceBuffer<float>    d_displacement_scale_;

    // CUDA texture objects (GPU hardware-sampled textures)
    std::vector<cudaArray_t>         tex_arrays_;
    std::vector<cudaTextureObject_t> tex_objects_;
    DeviceBuffer<cudaTextureObject_t> d_tex_objects_;
    void destroy_textures();

    // Launch params (device)
    DeviceBuffer<LaunchParams> d_launch_params_;

    // Diagnostic output (SoA layout for standalone test launches)
    DeviceBuffer<float>    d_test_color_r_;
    DeviceBuffer<float>    d_test_color_g_;
    DeviceBuffer<float>    d_test_color_b_;
    DeviceBuffer<float>    d_sample_counts_;

    // GPU info
    std::string gpu_name_;
    size_t      gpu_vram_total_ = 0;
    int         gpu_sm_count_   = 0;

    // Pipeline mode
    bool instanced_pipeline_ = false;
};
