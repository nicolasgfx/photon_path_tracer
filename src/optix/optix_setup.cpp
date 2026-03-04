// ---------------------------------------------------------------------
// optix_setup.cpp -- OptiX pipeline creation, accel build, SBT
// ---------------------------------------------------------------------
// Extracted from optix_renderer.cpp (§1.8):
//   init(), build_accel(), create_context(), create_module(),
//   create_programs(), create_pipeline(), build_sbt()
// ---------------------------------------------------------------------
#include "optix/optix_renderer.h"

#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <cstdio>
#include <iostream>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

// ---------------------------------------------------------------------
// Module-local constants (pipeline configuration)
// ---------------------------------------------------------------------
namespace {
    // Must match the payload layout documented in optix_device.cu.
    constexpr int OPTIX_NUM_PAYLOAD_VALUES   = 15;
    constexpr int OPTIX_NUM_ATTRIBUTE_VALUES = 2;     // barycentrics
    constexpr int OPTIX_MAX_TRACE_DEPTH      = 2;     // radiance + shadow

    // Conservative stack size: increased for per-ray local arrays.
    constexpr int OPTIX_STACK_SIZE           = 16384;
}

// -- Helper: read PTX from file ---------------------------------------
static std::string read_ptx_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open PTX file: " + filename);
    }
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

// -- OptiX log callback -----------------------------------------------
static void optix_log_callback(unsigned int level, const char* tag,
                                const char* message, void* /*cbdata*/) {
    std::cerr << "[OptiX][" << level << "][" << tag << "] " << message << "\n";
}

// -- SBT record helpers -----------------------------------------------
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4324) // structure was padded due to alignment
#endif
template <typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T    data;
};

using RayGenRecord   = SbtRecord<RayGenSbtRecord>;
using MissRecord     = SbtRecord<MissSbtRecord>;
using HitGroupRecord = SbtRecord<HitGroupSbtRecord>;
#ifdef _MSC_VER
#pragma warning(pop)
#endif

// =====================================================================
// init()
// =====================================================================
void OptixRenderer::init() {
    if (initialised_) return;

    // Initialise CUDA
    CUDA_CHECK(cudaFree(nullptr)); // forces context creation

    // Query GPU device properties
    {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        gpu_name_       = prop.name;
        gpu_vram_total_ = prop.totalGlobalMem;
        gpu_sm_count_   = prop.multiProcessorCount;
        gpu_cc_major_   = prop.major;
        gpu_cc_minor_   = prop.minor;
        std::printf("[GPU] %s  |  %.0f MB VRAM  |  %d SMs  |  CC %d.%d\n",
                    gpu_name_.c_str(),
                    (double)gpu_vram_total_ / (1024.0 * 1024.0),
                    gpu_sm_count_, gpu_cc_major_, gpu_cc_minor_);
    }

    // Initialise OptiX
    OPTIX_CHECK(optixInit());

    create_context();
    // Module, programs, pipeline are created after build_accel

    resize(width_, height_);
    initialised_ = true;
    std::cout << "[OptiX] Renderer initialised\n";
}

// =====================================================================
// build_accel() -- Build GAS from triangle soup
// =====================================================================
void OptixRenderer::build_accel(const Scene& scene) {
    if (!context_) throw std::runtime_error("OptiX context not created");

    // Keep a host-side copy for debug hover ray queries.
    host_triangles_ = scene.triangles;
    num_tris_ = (int)scene.triangles.size();

    // Flatten triangle vertices into a contiguous float3 array
    size_t num_tris = scene.triangles.size();
    std::vector<float3> vertices(num_tris * 3);
    for (size_t i = 0; i < num_tris; ++i) {
        vertices[i * 3 + 0] = scene.triangles[i].v0;
        vertices[i * 3 + 1] = scene.triangles[i].v1;
        vertices[i * 3 + 2] = scene.triangles[i].v2;
    }

    // Upload vertices
    d_vertices_.upload(vertices);

    // Build input
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    auto& tri_input = build_input.triangleArray;
    CUdeviceptr vertex_ptr = reinterpret_cast<CUdeviceptr>(d_vertices_.d_ptr);
    tri_input.vertexBuffers       = &vertex_ptr;
    tri_input.numVertices         = (unsigned int)(num_tris * 3);
    tri_input.vertexFormat         = OPTIX_VERTEX_FORMAT_FLOAT3;
    tri_input.vertexStrideInBytes  = sizeof(float3);

    // All triangles share one SBT record (single material group)
    unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;
    tri_input.flags                = &flags;
    tri_input.numSbtRecords        = 1;

    // Compute memory requirements
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                               OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes buf_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context_, &accel_options, &build_input, 1, &buf_sizes));

    // Allocate temp + output
    DeviceBuffer temp_buffer;
    temp_buffer.alloc(buf_sizes.tempSizeInBytes);

    DeviceBuffer output_buffer;
    output_buffer.alloc(buf_sizes.outputSizeInBytes);

    // Build (with compaction property)
    DeviceBuffer compacted_size_buf;
    compacted_size_buf.alloc(sizeof(size_t));

    OptixAccelEmitDesc emit_desc;
    emit_desc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_desc.result = reinterpret_cast<CUdeviceptr>(compacted_size_buf.d_ptr);

    auto t_gas0 = std::chrono::high_resolution_clock::now();
    OPTIX_CHECK(optixAccelBuild(
        context_, nullptr,
        &accel_options,
        &build_input, 1,
        reinterpret_cast<CUdeviceptr>(temp_buffer.d_ptr), temp_buffer.bytes,
        reinterpret_cast<CUdeviceptr>(output_buffer.d_ptr), output_buffer.bytes,
        &gas_handle_,
        &emit_desc, 1));

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t_gas1 = std::chrono::high_resolution_clock::now();

    size_t compacted_size;
    CUDA_CHECK(cudaMemcpy(&compacted_size, compacted_size_buf.d_ptr,
                           sizeof(size_t), cudaMemcpyDeviceToHost));

    gas_buffer_.alloc(compacted_size);
    OPTIX_CHECK(optixAccelCompact(
        context_, nullptr,
        gas_handle_,
        reinterpret_cast<CUdeviceptr>(gas_buffer_.d_ptr), compacted_size,
        &gas_handle_));

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t_gas2 = std::chrono::high_resolution_clock::now();

    double ms_build   = std::chrono::duration<double, std::milli>(t_gas1 - t_gas0).count();
    double ms_compact = std::chrono::duration<double, std::milli>(t_gas2 - t_gas1).count();
    std::printf("[OptiX] GAS built: %zu tris  uncompressed=%.1f KB  compacted=%.1f KB  "
                "build=%.1f ms  compact=%.1f ms\n",
                num_tris,
                (double)output_buffer.bytes / 1024.0,
                (double)compacted_size / 1024.0,
                ms_build, ms_compact);

    // Now create module, programs, pipeline, SBT
    // Module/programs/pipeline are shader-dependent, not scene-dependent.
    // Only create them on the first call so build_accel is safe to re-invoke
    // when hot-swapping scenes at runtime (keys 1-5).
    if (!module_) {
        create_module();
        create_programs();
        create_pipeline();
    }
    build_sbt(scene);
}

// =====================================================================
// Private: OptiX pipeline creation
// =====================================================================

void OptixRenderer::create_context() {
    CUcontext cu_ctx = nullptr;
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = optix_log_callback;
    options.logCallbackLevel   = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context_));
    std::cout << "[OptiX] Context created\n";
}

void OptixRenderer::create_module() {
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifndef NDEBUG
    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur                  = false;
    pipeline_options.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.numPayloadValues                 = OPTIX_NUM_PAYLOAD_VALUES;
    pipeline_options.numAttributeValues               = OPTIX_NUM_ATTRIBUTE_VALUES;
    pipeline_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName  = "params";

    std::string ptx = read_ptx_file(PTX_FILE_PATH);

    char log[2048];
    size_t log_size = sizeof(log);

    OPTIX_CHECK(optixModuleCreate(
        context_,
        &module_compile_options,
        &pipeline_options,
        ptx.c_str(), ptx.size(),
        log, &log_size,
        &module_));

    if (log_size > 1) std::cout << "[OptiX Module] " << log << "\n";
    std::cout << "[OptiX] Module created from PTX (" << ptx.size() << " bytes)\n";
}

void OptixRenderer::create_programs() {
    OptixProgramGroupOptions pg_options = {};
    char log[2048];
    size_t log_size;

    // Raygen (render)
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = module_;
        desc.raygen.entryFunctionName = "__raygen__render";
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_, &desc, 1, &pg_options, log, &log_size, &raygen_pg_));
    }

    // Raygen (photon trace)
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = module_;
        desc.raygen.entryFunctionName = "__raygen__photon_trace";
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_, &desc, 1, &pg_options, log, &log_size, &raygen_photon_pg_));
    }

    // Raygen (targeted caustic photon trace)
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = module_;
        desc.raygen.entryFunctionName = "__raygen__targeted_photon_trace";
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_, &desc, 1, &pg_options, log, &log_size, &raygen_targeted_pg_));
    }

    // Raygen (photon density gather at first camera hit)
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = module_;
        desc.raygen.entryFunctionName = "__raygen__photon_gather";
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_, &desc, 1, &pg_options, log, &log_size, &raygen_gather_pg_));
    }

    // Miss (radiance)
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module = module_;
        desc.miss.entryFunctionName = "__miss__radiance";
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_, &desc, 1, &pg_options, log, &log_size, &miss_pg_));
    }

    // Miss (shadow)
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module = module_;
        desc.miss.entryFunctionName = "__miss__shadow";
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_, &desc, 1, &pg_options, log, &log_size, &miss_shadow_pg_));
    }

    // Hit group (radiance)
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH = module_;
        desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        desc.hitgroup.moduleAH = module_;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_, &desc, 1, &pg_options, log, &log_size, &hitgroup_pg_));
    }

    // Hit group (shadow)
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH = module_;
        desc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
        desc.hitgroup.moduleAH = module_;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__shadow";
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_, &desc, 1, &pg_options, log, &log_size, &hitgroup_shadow_pg_));
    }

    std::cout << "[OptiX] Program groups created (render + photon trace + targeted + gather)\n";
}

void OptixRenderer::create_pipeline() {
    OptixProgramGroup program_groups[] = {
        raygen_pg_, raygen_photon_pg_, raygen_targeted_pg_, raygen_gather_pg_,
        miss_pg_, miss_shadow_pg_,
        hitgroup_pg_, hitgroup_shadow_pg_
    };

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = OPTIX_MAX_TRACE_DEPTH;

    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur                  = false;
    pipeline_options.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.numPayloadValues                 = OPTIX_NUM_PAYLOAD_VALUES;
    pipeline_options.numAttributeValues               = OPTIX_NUM_ATTRIBUTE_VALUES;
    pipeline_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName  = "params";

    char log[2048];
    size_t log_size = sizeof(log);

    OPTIX_CHECK(optixPipelineCreate(
        context_,
        &pipeline_options,
        &link_options,
        program_groups, 7,
        log, &log_size,
        &pipeline_));

    // Set stack sizes
    // Last param is maxTraversableGraphDepth: 1 for single GAS (no IAS)
    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline_,
        OPTIX_STACK_SIZE,
        OPTIX_STACK_SIZE,
        OPTIX_STACK_SIZE,
        1));

    std::cout << "[OptiX] Pipeline created\n";
}

void OptixRenderer::build_sbt(const Scene& scene) {
    // Raygen record (render)
    {
        RayGenRecord rec = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_pg_, &rec));
        d_raygen_record_.upload(&rec, 1);
        sbt_.raygenRecord = reinterpret_cast<CUdeviceptr>(d_raygen_record_.d_ptr);
    }

    // Raygen record (photon trace) -- stored separately, swapped in for photon launch
    {
        RayGenRecord rec = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_photon_pg_, &rec));
        d_raygen_photon_record_.upload(&rec, 1);
    }

    // Raygen record (targeted caustic photon trace)
    {
        RayGenRecord rec = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_targeted_pg_, &rec));
        d_raygen_targeted_record_.upload(&rec, 1);
    }

    // Raygen record (photon density gather at first camera hit)
    {
        RayGenRecord rec = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_gather_pg_, &rec));
        d_raygen_gather_record_.upload(&rec, 1);
    }

    // Miss records (radiance + shadow)
    {
        std::vector<MissRecord> miss_records(2);

        OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg_, &miss_records[0]));
        miss_records[0].data.background_color = make_f3(0, 0, 0);

        OPTIX_CHECK(optixSbtRecordPackHeader(miss_shadow_pg_, &miss_records[1]));
        miss_records[1].data.background_color = make_f3(0, 0, 0);

        d_miss_records_.upload(miss_records);
        sbt_.missRecordBase          = reinterpret_cast<CUdeviceptr>(d_miss_records_.d_ptr);
        sbt_.missRecordStrideInBytes = sizeof(MissRecord);
        sbt_.missRecordCount         = 2;
    }

    // Hit group records (radiance + shadow)
    {
        std::vector<HitGroupRecord> hg_records(2);

        auto fill_hg = [&](HitGroupRecord& rec) {
            rec.data.vertices     = d_vertices_.as<float3>();
            rec.data.normals      = d_normals_.as<float3>();
            rec.data.texcoords    = d_texcoords_.as<float2>();
            rec.data.material_ids = d_material_ids_.as<uint32_t>();
            rec.data.Kd           = d_Kd_.as<float>();
            rec.data.Ks           = d_Ks_.as<float>();
            rec.data.Le           = d_Le_.as<float>();
            rec.data.roughness    = d_roughness_.as<float>();
            rec.data.ior          = d_ior_.as<float>();
            rec.data.mat_type     = d_mat_type_.as<uint8_t>();
        };

        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pg_, &hg_records[0]));
        fill_hg(hg_records[0]);

        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_shadow_pg_, &hg_records[1]));
        fill_hg(hg_records[1]);

        d_hitgroup_records_.upload(hg_records);
        sbt_.hitgroupRecordBase          = reinterpret_cast<CUdeviceptr>(d_hitgroup_records_.d_ptr);
        sbt_.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
        sbt_.hitgroupRecordCount         = 2;
    }

    std::cout << "[OptiX] SBT built (render + photon trace)\n";
    (void)scene;
}
