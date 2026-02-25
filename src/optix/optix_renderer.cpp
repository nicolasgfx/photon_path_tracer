// ---------------------------------------------------------------------
// optix_renderer.cpp -- OptiX host-side pipeline implementation
// ---------------------------------------------------------------------
#include "optix/optix_renderer.h"
#include "core/config.h"
#include "optix/adaptive_sampling.h"
#include "photon/specular_target.h"   // SpecularTargetSet (for GPU upload)
#include "photon/tri_photon_irradiance.h"  // per-triangle irradiance heatmap
#include "photon/emitter.h"                // CPU photon tracing (USE_CPU_PHOTON_TRACE)

#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <functional>
#include <unordered_set>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

// ---------------------------------------------------------------------
// Module-local implementation constants
// (kept out of core/config.h to avoid clutter)
// ---------------------------------------------------------------------
namespace {
    // Must match the payload layout documented in optix_device.cu.
    constexpr int OPTIX_NUM_PAYLOAD_VALUES   = 15;
    constexpr int OPTIX_NUM_ATTRIBUTE_VALUES = 2;     // barycentrics
    constexpr int OPTIX_MAX_TRACE_DEPTH      = 2;     // radiance + shadow

    // Conservative stack size: increased for per-ray local arrays.
    constexpr int OPTIX_STACK_SIZE           = 16384;

    // Ray epsilon to avoid self-intersections.
    constexpr float OPTIX_SCENE_EPSILON      = 1e-4f;

    // Large tmax avoids clipping long rays in normalized scenes.
    constexpr float DEFAULT_RAY_TMAX         = 1e20f;

    // HashGrid::build() uses cell_size = radius * 2.0f.
    constexpr float HASHGRID_CELL_FACTOR     = 2.0f;
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
// upload_scene_data() -- Materials, normals, texcoords to device
// =====================================================================
void OptixRenderer::upload_scene_data(const Scene& scene) {
    size_t num_tris = scene.triangles.size();

    // Normals (per-vertex, 3 per triangle)
    std::vector<float3> normals(num_tris * 3);
    std::vector<float2> texcoords(num_tris * 3);
    std::vector<uint32_t> mat_ids(num_tris);

    for (size_t i = 0; i < num_tris; ++i) {
        normals[i * 3 + 0] = scene.triangles[i].n0;
        normals[i * 3 + 1] = scene.triangles[i].n1;
        normals[i * 3 + 2] = scene.triangles[i].n2;
        texcoords[i * 3 + 0] = scene.triangles[i].uv0;
        texcoords[i * 3 + 1] = scene.triangles[i].uv1;
        texcoords[i * 3 + 2] = scene.triangles[i].uv2;
        mat_ids[i] = scene.triangles[i].material_id;
    }

    d_normals_.upload(normals);
    d_texcoords_.upload(texcoords);
    d_material_ids_.upload(mat_ids);

    // Materials
    size_t num_mats = scene.materials.size();
    std::vector<float> Kd(num_mats * NUM_LAMBDA);
    std::vector<float> Ks(num_mats * NUM_LAMBDA);
    std::vector<float> Le(num_mats * NUM_LAMBDA);
    std::vector<float> roughness(num_mats);
    std::vector<float> ior(num_mats);
    std::vector<uint8_t> mat_type(num_mats);

    for (size_t m = 0; m < num_mats; ++m) {
        const Material& mat = scene.materials[m];
        for (int l = 0; l < NUM_LAMBDA; ++l) {
            // For glass materials, the GPU uses the Kd slot as the
            // transmittance filter (Tf).  Copy Tf into Kd so that
            // dev_get_Kd() on the device returns the correct filter.
            if (mat.type == MaterialType::Glass ||
                mat.type == MaterialType::Translucent)
                Kd[m * NUM_LAMBDA + l] = mat.Tf.value[l];
            else
                Kd[m * NUM_LAMBDA + l] = mat.Kd.value[l];
            Ks[m * NUM_LAMBDA + l] = mat.Ks.value[l];
            Le[m * NUM_LAMBDA + l] = mat.Le.value[l];
        }
        roughness[m] = mat.roughness;
        ior[m]       = mat.ior;
        mat_type[m]  = (uint8_t)mat.type;
    }

    d_Kd_.upload(Kd);
    d_Ks_.upload(Ks);
    d_Le_.upload(Le);
    d_roughness_.upload(roughness);
    d_ior_.upload(ior);
    d_mat_type_.upload(mat_type);

    // Per-material chromatic dispersion (Cauchy equation)
    std::vector<float>   cauchy_A(num_mats);
    std::vector<float>   cauchy_B(num_mats);
    std::vector<uint8_t> mat_dispersion(num_mats);
    for (size_t m = 0; m < num_mats; ++m) {
        const Material& mat = scene.materials[m];
        cauchy_A[m]       = mat.cauchy_A;
        cauchy_B[m]       = mat.cauchy_B;
        mat_dispersion[m] = mat.dispersion ? (uint8_t)1 : (uint8_t)0;
    }
    d_cauchy_A_.upload(cauchy_A);
    d_cauchy_B_.upload(cauchy_B);
    d_mat_dispersion_.upload(mat_dispersion);

    // Per-material diffuse texture ID (-1 = none)
    std::vector<int> diffuse_tex(num_mats);
    for (size_t m = 0; m < num_mats; ++m)
        diffuse_tex[m] = scene.materials[m].diffuse_tex;
    d_diffuse_tex_.upload(diffuse_tex);

    // Per-material emission texture ID (-1 = none)
    std::vector<int> emission_tex(num_mats);
    for (size_t m = 0; m < num_mats; ++m)
        emission_tex[m] = scene.materials[m].emission_tex;
    d_emission_tex_.upload(emission_tex);

    // Per-material opacity (from MTL 'd' keyword, default 1.0)
    std::vector<float> opacity(num_mats);
    for (size_t m = 0; m < num_mats; ++m)
        opacity[m] = scene.materials[m].opacity;
    d_opacity_.upload(opacity);

    // Per-material clearcoat / fabric data
    std::vector<float> clearcoat_weight(num_mats);
    std::vector<float> clearcoat_roughness(num_mats);
    std::vector<float> sheen(num_mats);
    std::vector<float> sheen_tint(num_mats);
    for (size_t m = 0; m < num_mats; ++m) {
        clearcoat_weight[m]    = scene.materials[m].pb_clearcoat;
        clearcoat_roughness[m] = (scene.materials[m].pb_clearcoat_roughness >= 0.f)
                                   ? scene.materials[m].pb_clearcoat_roughness
                                   : 0.03f;  // default coat roughness
        sheen[m]               = scene.materials[m].pb_sheen;
        sheen_tint[m]          = scene.materials[m].pb_sheen_tint;
    }
    d_clearcoat_weight_.upload(clearcoat_weight);
    d_clearcoat_roughness_.upload(clearcoat_roughness);
    d_sheen_.upload(sheen);
    d_sheen_tint_.upload(sheen_tint);

    // Texture atlas: concatenate all textures into one flat RGBA float buffer
    size_t num_textures = scene.textures.size();
    if (num_textures > 0) {
        std::vector<GpuTexDesc> descs(num_textures);
        size_t total_floats = 0;
        for (size_t t = 0; t < num_textures; ++t) {
            descs[t].offset = (int)total_floats;
            descs[t].width  = scene.textures[t].width;
            descs[t].height = scene.textures[t].height;
            total_floats += scene.textures[t].data.size();
        }
        std::vector<float> atlas(total_floats);
        for (size_t t = 0; t < num_textures; ++t) {
            std::memcpy(&atlas[descs[t].offset],
                        scene.textures[t].data.data(),
                        scene.textures[t].data.size() * sizeof(float));
        }
        d_tex_atlas_.upload(atlas);
        d_tex_descs_.upload(descs);
        std::cout << "[OptiX] Uploaded " << num_textures << " textures ("
                  << (total_floats * sizeof(float) / (1024*1024)) << " MB atlas)\n";
    } else {
        d_tex_atlas_.free();
        d_tex_descs_.free();
    }

    // Compute total GPU scene data size
    size_t scene_bytes =
        d_normals_.bytes + d_texcoords_.bytes + d_material_ids_.bytes +
        d_Kd_.bytes + d_Ks_.bytes + d_Le_.bytes +
        d_roughness_.bytes + d_ior_.bytes + d_mat_type_.bytes +
        d_cauchy_A_.bytes + d_cauchy_B_.bytes + d_mat_dispersion_.bytes +
        d_diffuse_tex_.bytes + d_emission_tex_.bytes +
        d_clearcoat_weight_.bytes + d_clearcoat_roughness_.bytes +
        d_sheen_.bytes + d_sheen_tint_.bytes +
        d_tex_atlas_.bytes + d_tex_descs_.bytes;
    std::printf("[OptiX] Scene data: %zu mats  %zu tris  %zu textures  total=%.2f MB\n",
                num_mats, num_tris, num_textures,
                (double)scene_bytes / (1024.0 * 1024.0));
}

// =====================================================================
// fill_clearcoat_fabric_params() -- set coat/sheen pointers in LaunchParams
// =====================================================================
void OptixRenderer::fill_clearcoat_fabric_params(LaunchParams& lp) const {
    lp.clearcoat_weight    = d_clearcoat_weight_.d_ptr    ? const_cast<float*>(d_clearcoat_weight_.as<float>())    : nullptr;
    lp.clearcoat_roughness = d_clearcoat_roughness_.d_ptr ? const_cast<float*>(d_clearcoat_roughness_.as<float>()) : nullptr;
    lp.sheen               = d_sheen_.d_ptr               ? const_cast<float*>(d_sheen_.as<float>())               : nullptr;
    lp.sheen_tint          = d_sheen_tint_.d_ptr          ? const_cast<float*>(d_sheen_tint_.as<float>())          : nullptr;
}

// =====================================================================
// upload_photon_data() -- Upload CPU-side photon map to device
// =====================================================================
void OptixRenderer::upload_photon_data(
    const PhotonSoA& global_photons,
    const HashGrid& global_grid,
    const PhotonSoA& /*caustic_photons*/,
    const HashGrid& /*caustic_grid*/,
    float gather_radius,
    float /*caustic_radius*/,
    int num_photons_emitted)
{
    if (global_photons.size() == 0) return;

    // Record N_emitted; fall back to N_stored if caller passes 0
    num_photons_emitted_ = (num_photons_emitted > 0)
                               ? num_photons_emitted
                               : (int)global_photons.size();

    d_photon_pos_x_.upload(global_photons.pos_x);
    d_photon_pos_y_.upload(global_photons.pos_y);
    d_photon_pos_z_.upload(global_photons.pos_z);
    d_photon_wi_x_.upload(global_photons.wi_x);
    d_photon_wi_y_.upload(global_photons.wi_y);
    d_photon_wi_z_.upload(global_photons.wi_z);
    d_photon_norm_x_.upload(global_photons.norm_x);
    d_photon_norm_y_.upload(global_photons.norm_y);
    d_photon_norm_z_.upload(global_photons.norm_z);
    d_photon_lambda_.upload(global_photons.lambda_bin);
    d_photon_flux_.upload(global_photons.flux);
    if (!global_photons.num_hero.empty())
        d_photon_num_hero_.upload(global_photons.num_hero);
    else
        d_photon_num_hero_.free();

    if (global_grid.sorted_indices.size() > 0) {
        d_grid_sorted_indices_.upload(global_grid.sorted_indices);
        d_grid_cell_start_.upload(global_grid.cell_start);
        d_grid_cell_end_.upload(global_grid.cell_end);
    }

    std::cout << "[OptiX] Uploaded " << global_photons.size()
              << " photons to device\n";
    gather_radius_ = gather_radius;

    // Copy into stored_photons_
    stored_photons_ = global_photons;
}

// =====================================================================
// upload_emitter_data() -- Upload emitter CDF for GPU photon tracing
// =====================================================================
void OptixRenderer::upload_emitter_data(const Scene& scene) {
    size_t n = scene.emissive_tri_indices.size();
    num_emissive_ = (int)n;
    if (n == 0) {
        std::cout << "[OptiX] No emissive triangles found\n";
        return;
    }

    // Upload indices
    d_emissive_indices_.upload(scene.emissive_tri_indices);

    // Build CDF from weights
    std::vector<float> weights(n);
    for (size_t i = 0; i < n; ++i) {
        uint32_t tri_idx = scene.emissive_tri_indices[i];
        const auto& tri = scene.triangles[tri_idx];
        const auto& mat = scene.materials[tri.material_id];
        weights[i] = tri.area() * mat.mean_emission();
    }
    float total = 0.f;
    for (auto w : weights) total += w;

    std::vector<float> cdf(n);
    float cum = 0.f;
    for (size_t i = 0; i < n; ++i) {
        cum += weights[i] / total;
        cdf[i] = cum;
    }
    cdf[n-1] = 1.0f; // ensure exact 1.0

    d_emissive_cdf_.upload(cdf);

    std::printf("[OptiX] Emitter data: %zu tris  total_power=%.4f  max_w=%.4f  min_w=%.4f\n",
                n, total,
                *std::max_element(weights.begin(), weights.end()),
                *std::min_element(weights.begin(), weights.end()));
}

// =====================================================================
// cpu_trace_photons() -- CPU photon tracing fallback (A/B comparison)
//   Uses emitter.h trace_photons() + targeted caustic emission on CPU,
//   then merges global + caustic maps, builds hash grid, and uploads
//   everything to the GPU for OptiX gather.
// =====================================================================
void OptixRenderer::cpu_trace_photons(const Scene& scene, const RenderConfig& config) {
    if (scene.emissive_tri_indices.empty()) {
        std::cout << "[CPU-Photon] Skipping (no emissive triangles)\n";
        return;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    gather_radius_ = config.gather_radius;
    caustic_radius_ = config.caustic_radius;

    // ── 1. CPU photon tracing (global + caustic) ─────────────────────
    EmitterConfig ecfg;
    ecfg.num_photons    = config.num_photons;
    ecfg.max_bounces    = config.max_bounces;
    ecfg.rr_threshold   = config.rr_threshold;
    ecfg.min_bounces_rr = config.min_bounces_rr;
    ecfg.volume_enabled = false;

    PhotonSoA global_map, caustic_map;
    ::trace_photons(scene, ecfg, global_map, caustic_map);

    num_photons_emitted_ = config.num_photons;

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms1 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("[CPU-Photon] Traced %d photons in %.1f ms  global=%zu  caustic=%zu\n",
                config.num_photons, ms1, global_map.size(), caustic_map.size());

    // ── 2. Targeted caustic emission (separate map) ────────────────
    //   Shoot photons directionally toward specular/translucent geometry.
    //   Results go into a FRESH map (not caustic_map which duplicates
    //   global_map entries) to avoid double-counting at gather.
    PhotonSoA targeted_caustic_map;
    int targeted_budget = 0;
    if (config.targeted_caustic_emission_enabled) {
        SpecularTargetSet target_set = SpecularTargetSet::build(scene);
        if (target_set.valid) {
            EmitterConfig tcfg = ecfg;
            tcfg.num_photons = config.caustic_photon_budget;
            targeted_budget = (int)(config.caustic_photon_budget * config.targeted_caustic_mix);
            trace_targeted_caustic_emission(
                scene, tcfg, target_set, targeted_caustic_map,
                config.targeted_caustic_mix);
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    // ── 3. Merge global + targeted caustic into stored_photons_ ──────
    //  Tag 0 = global (non-caustic diffuse indirect)
    //  Tag 1 = global-pass caustic path (L→S+→D) – skipped at gather
    //  Tag 2 = dedicated caustic-targeted pass
    //
    //  CPU trace_photons deposits caustic-path photons in BOTH global_map
    //  and caustic_map.  To mirror the GPU tag system we mark those
    //  global_map entries as tag 1 (skip at gather) so that caustic
    //  contribution only comes from the dedicated tag-2 targeted map.
    size_t n_global           = global_map.size();
    size_t n_targeted_caustic = targeted_caustic_map.size();
    size_t total              = n_global + n_targeted_caustic;

    stored_photons_.clear();
    stored_photons_.reserve(total);
    stored_photons_.append(global_map);
    stored_photons_.append(targeted_caustic_map);

    // Build caustic pass flags ────────────────────────────────────────
    caustic_pass_flags_.assign(total, 0);

    // Mark global-pass caustic photons as tag 1 (skip at gather).
    // caustic_map from trace_photons contains exact duplicates of
    // caustic-path entries in global_map (bit-exact positions).
    if (caustic_map.size() > 0 && n_targeted_caustic > 0) {
        std::unordered_set<uint64_t> caustic_pos_hashes;
        caustic_pos_hashes.reserve(caustic_map.size());
        for (size_t ci = 0; ci < caustic_map.size(); ++ci) {
            uint32_t hx, hy, hz;
            std::memcpy(&hx, &caustic_map.pos_x[ci], 4);
            std::memcpy(&hy, &caustic_map.pos_y[ci], 4);
            std::memcpy(&hz, &caustic_map.pos_z[ci], 4);
            uint64_t key = (uint64_t)hx ^ ((uint64_t)hy << 21) ^ ((uint64_t)hz << 42);
            caustic_pos_hashes.insert(key);
        }
        size_t n_tag1 = 0;
        for (size_t gi = 0; gi < n_global; ++gi) {
            uint32_t hx, hy, hz;
            std::memcpy(&hx, &global_map.pos_x[gi], 4);
            std::memcpy(&hy, &global_map.pos_y[gi], 4);
            std::memcpy(&hz, &global_map.pos_z[gi], 4);
            uint64_t key = (uint64_t)hx ^ ((uint64_t)hy << 21) ^ ((uint64_t)hz << 42);
            if (caustic_pos_hashes.count(key)) {
                caustic_pass_flags_[gi] = 1;  // skip at gather
                ++n_tag1;
            }
        }
        std::printf("[CPU-Photon] Tagged %zu global photons as tag-1 (caustic-path, skip)\n", n_tag1);
    }

    // Tag targeted caustics as 2
    if (n_targeted_caustic > 0) {
        std::fill(caustic_pass_flags_.begin() + n_global,
                  caustic_pass_flags_.end(), (uint8_t)2);
    }

    // N_caustic for dual-budget gather (matches GPU: only targeted budget)
    num_caustic_emitted_ = (n_targeted_caustic > 0) ? targeted_budget : 0;

    std::printf("[CPU-Photon] Merged: %zu global + %zu targeted-caustic = %zu total  "
                "N_emit=%d  N_caustic_emit=%d\n",
                n_global, n_targeted_caustic, total,
                num_photons_emitted_, num_caustic_emitted_);

    // ── 4. Build hash grid on CPU ────────────────────────────────────
    stored_grid_.build(stored_photons_, gather_radius_);

    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_grid = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::printf("[CPU-Photon] Hash grid built in %.1f ms  buckets=%u\n",
                ms_grid, stored_grid_.table_size);

    // ── 5. Upload to GPU ─────────────────────────────────────────────
    d_photon_pos_x_.upload(stored_photons_.pos_x);
    d_photon_pos_y_.upload(stored_photons_.pos_y);
    d_photon_pos_z_.upload(stored_photons_.pos_z);
    d_photon_wi_x_.upload(stored_photons_.wi_x);
    d_photon_wi_y_.upload(stored_photons_.wi_y);
    d_photon_wi_z_.upload(stored_photons_.wi_z);
    d_photon_norm_x_.upload(stored_photons_.norm_x);
    d_photon_norm_y_.upload(stored_photons_.norm_y);
    d_photon_norm_z_.upload(stored_photons_.norm_z);
    d_photon_lambda_.upload(stored_photons_.lambda_bin);
    d_photon_flux_.upload(stored_photons_.flux);
    d_photon_num_hero_.upload(stored_photons_.num_hero);
    d_grid_sorted_indices_.upload(stored_grid_.sorted_indices);
    d_grid_cell_start_.upload(stored_grid_.cell_start);
    d_grid_cell_end_.upload(stored_grid_.cell_end);
    d_photon_is_caustic_pass_.upload(caustic_pass_flags_);

    // Heatmap
    {
        auto irr = build_tri_photon_irradiance(stored_photons_, num_tris_);
        if (!irr.empty())
            d_tri_photon_irradiance_.upload(irr);
        else
            d_tri_photon_irradiance_.free();
    }

    auto t4 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t4 - t0).count();
    std::printf("[CPU-Photon] Total: %.1f ms  (uploaded to GPU)\n", total_ms);
}

// =====================================================================
// trace_photons() -- GPU photon tracing
//   1. Launch __raygen__photon_trace on the GPU
//   2. Download results to CPU
//   3. Build hash grid on CPU
//   4. Upload photons + grid back to device
// =====================================================================
void OptixRenderer::trace_photons(const Scene& scene, const RenderConfig& config,
                                  float grid_radius_override, int photon_map_seed) {
    if (num_emissive_ <= 0) {
        std::cout << "[OptiX] Skipping photon trace (no emissive triangles)\n";
        return;
    }

    auto t_phase_start = std::chrono::high_resolution_clock::now();
    auto t_lap = t_phase_start;

    // Use override radius for the hash grid if provided (SPPM mode).
    // The member gather_radius_ drives both grid build AND fill_common_params.
    gather_radius_ = (grid_radius_override > 0.f)
                         ? grid_radius_override
                         : config.gather_radius;
    caustic_radius_ = config.caustic_radius;

    int num_photons = config.num_photons;
    int max_stored  = num_photons * DEFAULT_MAX_BOUNCES; // upper bound on stored photons
    num_photons_emitted_ = num_photons;  // record N_emitted for density normalisation

    // Estimate GPU buffer footprint (bytes per photon: 9 float pos/wi/norm + Hero*(uint16+float) + 1 uint8)
    size_t bytes_per_photon = 9 * sizeof(float)
                             + HERO_WAVELENGTHS * (sizeof(uint16_t) + sizeof(float))
                             + sizeof(uint8_t);
    double buf_mb = (double)(max_stored * bytes_per_photon) / (1024.0 * 1024.0);
    std::printf("[OptiX] Tracing %d photons  max_stored=%d  buf=%.1f MB  radius=%.5f  bounces=%d\n",
                num_photons, max_stored, buf_mb, gather_radius_, config.max_bounces);

    // Allocate output buffers
    d_out_photon_pos_x_.alloc(max_stored * sizeof(float));
    d_out_photon_pos_y_.alloc(max_stored * sizeof(float));
    d_out_photon_pos_z_.alloc(max_stored * sizeof(float));
    d_out_photon_wi_x_.alloc(max_stored * sizeof(float));
    d_out_photon_wi_y_.alloc(max_stored * sizeof(float));
    d_out_photon_wi_z_.alloc(max_stored * sizeof(float));
    d_out_photon_norm_x_.alloc(max_stored * sizeof(float));
    d_out_photon_norm_y_.alloc(max_stored * sizeof(float));
    d_out_photon_norm_z_.alloc(max_stored * sizeof(float));
    d_out_photon_lambda_.alloc(max_stored * HERO_WAVELENGTHS * sizeof(uint16_t));
    d_out_photon_flux_.alloc(max_stored * HERO_WAVELENGTHS * sizeof(float));
    d_out_photon_num_hero_.alloc(max_stored * sizeof(uint8_t));
    d_out_photon_source_emissive_.alloc(max_stored * sizeof(uint16_t));
    d_out_photon_is_caustic_.alloc(max_stored * sizeof(uint8_t));
    d_out_photon_tri_id_.alloc(max_stored * sizeof(uint32_t));
    d_out_photon_count_.alloc_zero(sizeof(unsigned int)); // zero the counter

    // Allocate volume photon output buffers
    int max_vol_stored = max_stored / 2; // conservative upper bound
    d_out_vol_photon_pos_x_.alloc(max_vol_stored * sizeof(float));
    d_out_vol_photon_pos_y_.alloc(max_vol_stored * sizeof(float));
    d_out_vol_photon_pos_z_.alloc(max_vol_stored * sizeof(float));
    d_out_vol_photon_wi_x_.alloc(max_vol_stored * sizeof(float));
    d_out_vol_photon_wi_y_.alloc(max_vol_stored * sizeof(float));
    d_out_vol_photon_wi_z_.alloc(max_vol_stored * sizeof(float));
    d_out_vol_photon_lambda_.alloc(max_vol_stored * sizeof(uint16_t));
    d_out_vol_photon_flux_.alloc(max_vol_stored * sizeof(float));
    d_out_vol_photon_count_.alloc_zero(sizeof(unsigned int));

    // Build launch params for photon trace
    LaunchParams lp = {};
    lp.traversable      = gas_handle_;
    lp.vertices          = d_vertices_.as<float3>();
    lp.normals           = d_normals_.as<float3>();
    lp.texcoords         = d_texcoords_.as<float2>();
    lp.material_ids      = d_material_ids_.as<uint32_t>();

    lp.num_materials     = (int)(d_mat_type_.bytes / sizeof(uint8_t));
    lp.Kd                = d_Kd_.as<float>();
    lp.Ks                = d_Ks_.as<float>();
    lp.Le                = d_Le_.as<float>();
    lp.roughness         = d_roughness_.as<float>();
    lp.ior               = d_ior_.as<float>();
    lp.cauchy_A          = d_cauchy_A_.d_ptr       ? d_cauchy_A_.as<float>()       : nullptr;
    lp.cauchy_B          = d_cauchy_B_.d_ptr       ? d_cauchy_B_.as<float>()       : nullptr;
    lp.mat_dispersion    = d_mat_dispersion_.d_ptr ? d_mat_dispersion_.as<uint8_t>() : nullptr;
    lp.mat_type          = d_mat_type_.as<uint8_t>();
    lp.diffuse_tex       = d_diffuse_tex_.d_ptr ? d_diffuse_tex_.as<int>() : nullptr;
    lp.emission_tex      = d_emission_tex_.d_ptr ? d_emission_tex_.as<int>() : nullptr;
    lp.opacity           = d_opacity_.d_ptr ? d_opacity_.as<float>() : nullptr;
    lp.clearcoat_weight    = d_clearcoat_weight_.d_ptr    ? d_clearcoat_weight_.as<float>()    : nullptr;
    lp.clearcoat_roughness = d_clearcoat_roughness_.d_ptr ? d_clearcoat_roughness_.as<float>() : nullptr;
    lp.sheen               = d_sheen_.d_ptr               ? d_sheen_.as<float>()               : nullptr;
    lp.sheen_tint          = d_sheen_tint_.d_ptr          ? d_sheen_tint_.as<float>()          : nullptr;
    lp.tex_atlas         = d_tex_atlas_.d_ptr   ? d_tex_atlas_.as<float>() : nullptr;
    lp.tex_descs         = d_tex_descs_.d_ptr   ? d_tex_descs_.as<GpuTexDesc>() : nullptr;
    lp.num_textures      = (int)(d_tex_descs_.bytes / sizeof(GpuTexDesc));

    lp.num_photons       = num_photons;
    lp.max_bounces       = config.max_bounces;
    lp.photon_max_bounces = DEBUG_PHOTON_SINGLE_BOUNCE ? 1 : config.max_bounces;

    // Emitter data
    lp.emissive_tri_indices = d_emissive_indices_.as<uint32_t>();
    lp.emissive_cdf         = d_emissive_cdf_.as<float>();
    lp.num_emissive         = num_emissive_;
    lp.total_emissive_power = scene.total_emissive_power;

    // Photon output buffers
    lp.out_photon_pos_x  = d_out_photon_pos_x_.as<float>();
    lp.out_photon_pos_y  = d_out_photon_pos_y_.as<float>();
    lp.out_photon_pos_z  = d_out_photon_pos_z_.as<float>();
    lp.out_photon_wi_x   = d_out_photon_wi_x_.as<float>();
    lp.out_photon_wi_y   = d_out_photon_wi_y_.as<float>();
    lp.out_photon_wi_z   = d_out_photon_wi_z_.as<float>();
    lp.out_photon_norm_x = d_out_photon_norm_x_.as<float>();
    lp.out_photon_norm_y = d_out_photon_norm_y_.as<float>();
    lp.out_photon_norm_z = d_out_photon_norm_z_.as<float>();
    lp.out_photon_lambda  = d_out_photon_lambda_.as<uint16_t>();
    lp.out_photon_flux    = d_out_photon_flux_.as<float>();
    lp.out_photon_num_hero = d_out_photon_num_hero_.as<uint8_t>();
    lp.out_photon_source_emissive = d_out_photon_source_emissive_.as<uint16_t>();
    lp.out_photon_is_caustic  = d_out_photon_is_caustic_.d_ptr
                                   ? d_out_photon_is_caustic_.as<uint8_t>() : nullptr;
    lp.out_photon_tri_id  = d_out_photon_tri_id_.d_ptr
                                   ? d_out_photon_tri_id_.as<uint32_t>() : nullptr;
    lp.out_photon_count   = d_out_photon_count_.as<unsigned int>();
    lp.max_stored_photons = max_stored;

    // Volume photon output buffers
    lp.out_vol_photon_pos_x  = d_out_vol_photon_pos_x_.as<float>();
    lp.out_vol_photon_pos_y  = d_out_vol_photon_pos_y_.as<float>();
    lp.out_vol_photon_pos_z  = d_out_vol_photon_pos_z_.as<float>();
    lp.out_vol_photon_wi_x   = d_out_vol_photon_wi_x_.as<float>();
    lp.out_vol_photon_wi_y   = d_out_vol_photon_wi_y_.as<float>();
    lp.out_vol_photon_wi_z   = d_out_vol_photon_wi_z_.as<float>();
    lp.out_vol_photon_lambda = d_out_vol_photon_lambda_.as<uint16_t>();
    lp.out_vol_photon_flux   = d_out_vol_photon_flux_.as<float>();
    lp.out_vol_photon_count  = d_out_vol_photon_count_.as<unsigned int>();
    lp.max_stored_vol_photons = max_vol_stored;

    // Volume params for photon medium interaction
    lp.volume_enabled  = config.volume_enabled ? 1 : 0;
    lp.volume_density  = config.volume_density;
    lp.volume_falloff  = config.volume_falloff;
    lp.volume_albedo   = config.volume_albedo;
    lp.volume_samples  = config.volume_samples;
    lp.volume_max_t    = config.volume_max_t;

    // Multi-map seed for RNG decorrelation
    lp.photon_map_seed = photon_map_seed;

    // Upload params
    d_launch_params_.alloc(sizeof(LaunchParams));
    CUDA_CHECK(cudaMemcpy(d_launch_params_.d_ptr, &lp,
                           sizeof(LaunchParams), cudaMemcpyHostToDevice));

    // Swap SBT to photon raygen
    CUdeviceptr saved_raygen = sbt_.raygenRecord;
    sbt_.raygenRecord = reinterpret_cast<CUdeviceptr>(d_raygen_photon_record_.d_ptr);

    // Launch 1D: one thread per photon
    OPTIX_CHECK(optixLaunch(
        pipeline_,
        nullptr,
        reinterpret_cast<CUdeviceptr>(d_launch_params_.d_ptr),
        sizeof(LaunchParams),
        &sbt_,
        num_photons, 1, 1));

    CUDA_CHECK(cudaDeviceSynchronize());

    // Restore SBT
    sbt_.raygenRecord = saved_raygen;

    {
        auto t_now = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_now - t_lap).count();
        std::printf("[Timing] Photon GPU trace:  %8.1f ms\n", ms);
        t_lap = t_now;
    }

    // Download photon count
    unsigned int stored_count = 0;
    CUDA_CHECK(cudaMemcpy(&stored_count, d_out_photon_count_.d_ptr,
                           sizeof(unsigned int), cudaMemcpyDeviceToHost));

    if ((int)stored_count > max_stored)
        stored_count = (unsigned int)max_stored;

    {
        float stored_pct = 100.f * (float)stored_count / (float)(std::max)(1, num_photons);
        float photons_per_emitted = (float)stored_count / (float)(std::max)(1, num_photons);
        std::printf("[OptiX] Stored %u photons  (%.1f%% of %d emitted = %.2f stored/ray)\n",
                    stored_count, stored_pct, num_photons, photons_per_emitted);
    }

    if (stored_count == 0) return;

    // Download photon data
    stored_photons_.resize(stored_count);
    CUDA_CHECK(cudaMemcpy(stored_photons_.pos_x.data(),  d_out_photon_pos_x_.d_ptr,  stored_count*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(stored_photons_.pos_y.data(),  d_out_photon_pos_y_.d_ptr,  stored_count*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(stored_photons_.pos_z.data(),  d_out_photon_pos_z_.d_ptr,  stored_count*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(stored_photons_.wi_x.data(),   d_out_photon_wi_x_.d_ptr,   stored_count*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(stored_photons_.wi_y.data(),   d_out_photon_wi_y_.d_ptr,   stored_count*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(stored_photons_.wi_z.data(),   d_out_photon_wi_z_.d_ptr,   stored_count*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(stored_photons_.norm_x.data(), d_out_photon_norm_x_.d_ptr, stored_count*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(stored_photons_.norm_y.data(), d_out_photon_norm_y_.d_ptr, stored_count*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(stored_photons_.norm_z.data(), d_out_photon_norm_z_.d_ptr, stored_count*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(stored_photons_.lambda_bin.data(), d_out_photon_lambda_.d_ptr, stored_count*HERO_WAVELENGTHS*sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(stored_photons_.flux.data(),   d_out_photon_flux_.d_ptr,   stored_count*HERO_WAVELENGTHS*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(stored_photons_.num_hero.data(), d_out_photon_num_hero_.d_ptr, stored_count*sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(stored_photons_.source_emissive_idx.data(), d_out_photon_source_emissive_.d_ptr, stored_count*sizeof(uint16_t), cudaMemcpyDeviceToHost));
    if (d_out_photon_tri_id_.d_ptr)
        CUDA_CHECK(cudaMemcpy(stored_photons_.tri_id.data(), d_out_photon_tri_id_.d_ptr, stored_count*sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Download per-photon caustic flags (only when caustic debug is enabled)
    // Always download caustic flags (needed for progress snapshot caustic PNG)
    if (d_out_photon_is_caustic_.d_ptr) {
        caustic_flags_.resize(stored_count);
        CUDA_CHECK(cudaMemcpy(caustic_flags_.data(), d_out_photon_is_caustic_.d_ptr, stored_count*sizeof(uint8_t), cudaMemcpyDeviceToHost));
        size_t n_caustic = 0;
        for (size_t i = 0; i < stored_count; ++i)
            if (caustic_flags_[i]) ++n_caustic;
        std::printf("[OptiX] Caustic photons: %zu / %u (%.1f%%)\n",
                    n_caustic, stored_count,
                    100.f * (float)n_caustic / (float)(std::max)(1u, stored_count));
    } else {
        caustic_flags_.clear();
    }

    {
        auto t_now = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_now - t_lap).count();
        std::printf("[Timing] Photon download:   %8.1f ms\n", ms);
        t_lap = t_now;
    }

    // ── Dual-budget caustic pass (Jensen 1996 two-budget) ────────────
    // When targeted caustic emission is enabled, we skip this blind
    // caustic-only pass entirely — the targeted pass replaces it for
    // consistency with the CPU pipeline (2 maps: global + targeted).
    // When targeted is disabled, fall back to the blind caustic-only
    // pass that re-emits photons with caustic_only_store=1.
    const unsigned int global_count = stored_count;
    const int caustic_budget = config.caustic_photon_budget;
    num_caustic_emitted_ = 0;

    if (caustic_budget > 0 && !config.targeted_caustic_emission_enabled) {
        num_caustic_emitted_ = caustic_budget;
        std::printf("[OptiX] Caustic pass: emitting %d targeted photons\n", caustic_budget);

        // Zero the GPU counter for the second pass
        CUDA_CHECK(cudaMemset(d_out_photon_count_.d_ptr, 0, sizeof(unsigned int)));

        // Configure launch params for caustic-only mode
        lp.num_photons       = caustic_budget;
        lp.caustic_only_store = 1;
        lp.photon_map_seed   = photon_map_seed + 7919;  // decorrelate from global pass

        CUDA_CHECK(cudaMemcpy(d_launch_params_.d_ptr, &lp,
                               sizeof(LaunchParams), cudaMemcpyHostToDevice));

        sbt_.raygenRecord = reinterpret_cast<CUdeviceptr>(d_raygen_photon_record_.d_ptr);

        OPTIX_CHECK(optixLaunch(
            pipeline_, nullptr,
            reinterpret_cast<CUdeviceptr>(d_launch_params_.d_ptr),
            sizeof(LaunchParams), &sbt_,
            caustic_budget, 1, 1));

        CUDA_CHECK(cudaDeviceSynchronize());
        sbt_.raygenRecord = saved_raygen;

        // Reset caustic_only_store so it doesn't leak to render passes
        lp.caustic_only_store = 0;

        // Download caustic-pass photon count
        unsigned int caustic_stored = 0;
        CUDA_CHECK(cudaMemcpy(&caustic_stored, d_out_photon_count_.d_ptr,
                               sizeof(unsigned int), cudaMemcpyDeviceToHost));
        if ((int)caustic_stored > max_stored)
            caustic_stored = (unsigned int)max_stored;

        std::printf("[OptiX] Caustic pass stored %u / %d photons (%.1f%%)\n",
                    caustic_stored, caustic_budget,
                    100.f * (float)caustic_stored / (float)(std::max)(1, caustic_budget));

        if (caustic_stored > 0) {
            // Grow stored_photons_ to hold both global + caustic photons
            size_t total = (size_t)global_count + caustic_stored;
            stored_photons_.resize((uint32_t)total);

            // Download caustic photons into the tail of stored_photons_
            size_t off = global_count;
            CUDA_CHECK(cudaMemcpy(stored_photons_.pos_x.data()  + off, d_out_photon_pos_x_.d_ptr,  caustic_stored*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(stored_photons_.pos_y.data()  + off, d_out_photon_pos_y_.d_ptr,  caustic_stored*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(stored_photons_.pos_z.data()  + off, d_out_photon_pos_z_.d_ptr,  caustic_stored*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(stored_photons_.wi_x.data()   + off, d_out_photon_wi_x_.d_ptr,   caustic_stored*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(stored_photons_.wi_y.data()   + off, d_out_photon_wi_y_.d_ptr,   caustic_stored*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(stored_photons_.wi_z.data()   + off, d_out_photon_wi_z_.d_ptr,   caustic_stored*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(stored_photons_.norm_x.data() + off, d_out_photon_norm_x_.d_ptr, caustic_stored*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(stored_photons_.norm_y.data() + off, d_out_photon_norm_y_.d_ptr, caustic_stored*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(stored_photons_.norm_z.data() + off, d_out_photon_norm_z_.d_ptr, caustic_stored*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(stored_photons_.lambda_bin.data() + off*HERO_WAVELENGTHS, d_out_photon_lambda_.d_ptr, caustic_stored*HERO_WAVELENGTHS*sizeof(uint16_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(stored_photons_.flux.data()   + off*HERO_WAVELENGTHS, d_out_photon_flux_.d_ptr,   caustic_stored*HERO_WAVELENGTHS*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(stored_photons_.num_hero.data() + off, d_out_photon_num_hero_.d_ptr, caustic_stored*sizeof(uint8_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(stored_photons_.source_emissive_idx.data() + off, d_out_photon_source_emissive_.d_ptr, caustic_stored*sizeof(uint16_t), cudaMemcpyDeviceToHost));
            if (d_out_photon_tri_id_.d_ptr)
                CUDA_CHECK(cudaMemcpy(stored_photons_.tri_id.data() + off, d_out_photon_tri_id_.d_ptr, caustic_stored*sizeof(uint32_t), cudaMemcpyDeviceToHost));

            // Update stored_count to include caustic photons
            stored_count = (unsigned int)total;
        }

        {
            auto t_now = std::chrono::high_resolution_clock::now();
            double ms_c = std::chrono::duration<double, std::milli>(t_now - t_lap).count();
            std::printf("[Timing] Caustic pass:      %8.1f ms\n", ms_c);
            t_lap = t_now;
        }
    } else {
        num_caustic_emitted_ = 0;
    }

    // ── GPU targeted caustic emission (Jensen §9.2) ────────────────────
    // Shoot photons directly at specular geometry on the GPU using OptiX
    // ray tracing.  Results are appended to stored_photons_.
    if (config.targeted_caustic_emission_enabled) {
        SpecularTargetSet target_set = SpecularTargetSet::build(scene);
        std::printf("[Targeted] targeted_caustic_emission_enabled=true  SpecularTargetSet.valid=%d  spec_tris=%zu\n",
                    (int)target_set.valid, target_set.specular_tri_indices.size());
        // Dump first few specular triangle indices + materials
        for (size_t si = 0; si < (std::min)(target_set.specular_tri_indices.size(), (size_t)5); ++si) {
            uint32_t ti = target_set.specular_tri_indices[si];
            uint32_t mi = scene.triangles[ti].material_id;
            const char* mname = "";
            if (mi < scene.materials.size()) {
                auto mt = scene.materials[mi].type;
                if (mt == MaterialType::Glass) mname = "Glass";
                else if (mt == MaterialType::Mirror) mname = "Mirror";
                else if (mt == MaterialType::Translucent) mname = "Translucent";
                else mname = "OTHER";
            }
            std::printf("[Targeted]   spec_tri[%zu] = tri#%u  mat#%u (%s)  area=%.6f\n",
                        si, ti, mi, mname, target_set.tri_areas[si]);
        }
        if (target_set.valid) {
            int targeted_budget = (int)(config.caustic_photon_budget * config.targeted_caustic_mix);
            if (targeted_budget > 0) {
                auto t_tgt0 = std::chrono::high_resolution_clock::now();

                // Upload specular target data to GPU
                num_targeted_spec_tris_ = (int)target_set.specular_tri_indices.size();
                d_targeted_spec_tri_indices_.upload(target_set.specular_tri_indices);

                // Extract alias table arrays from SpecularTargetSet
                {
                    int n = num_targeted_spec_tris_;
                    std::vector<float>    alias_prob(n);
                    std::vector<uint32_t> alias_idx(n);
                    std::vector<float>    pdf_vals(n);
                    for (int i = 0; i < n; ++i) {
                        alias_prob[i] = target_set.area_alias_table.entries[i].prob;
                        alias_idx[i]  = target_set.area_alias_table.entries[i].alias;
                        pdf_vals[i]   = target_set.area_alias_table.pdf_values[i];
                    }
                    d_targeted_spec_alias_prob_.upload(alias_prob);
                    d_targeted_spec_alias_idx_.upload(alias_idx);
                    d_targeted_spec_pdf_.upload(pdf_vals);
                    d_targeted_spec_areas_.upload(target_set.tri_areas);
                }

                // Zero the GPU counter for this pass
                CUDA_CHECK(cudaMemset(d_out_photon_count_.d_ptr, 0, sizeof(unsigned int)));

                // Configure launch params for targeted mode
                lp.num_photons       = targeted_budget;
                lp.caustic_only_store = 1;
                lp.photon_map_seed   = photon_map_seed + 15731;  // decorrelate

                lp.targeted_spec_tri_indices = d_targeted_spec_tri_indices_.as<uint32_t>();
                lp.targeted_spec_alias_prob  = d_targeted_spec_alias_prob_.as<float>();
                lp.targeted_spec_alias_idx   = d_targeted_spec_alias_idx_.as<uint32_t>();
                lp.targeted_spec_pdf         = d_targeted_spec_pdf_.as<float>();
                lp.targeted_spec_areas       = d_targeted_spec_areas_.as<float>();
                lp.num_targeted_spec_tris    = num_targeted_spec_tris_;
                lp.targeted_mode             = 1;

                CUDA_CHECK(cudaMemcpy(d_launch_params_.d_ptr, &lp,
                                       sizeof(LaunchParams), cudaMemcpyHostToDevice));

                // Swap to targeted raygen SBT
                sbt_.raygenRecord = reinterpret_cast<CUdeviceptr>(d_raygen_targeted_record_.d_ptr);

                OPTIX_CHECK(optixLaunch(
                    pipeline_, nullptr,
                    reinterpret_cast<CUdeviceptr>(d_launch_params_.d_ptr),
                    sizeof(LaunchParams), &sbt_,
                    targeted_budget, 1, 1));

                CUDA_CHECK(cudaDeviceSynchronize());
                sbt_.raygenRecord = saved_raygen;

                // Reset targeted mode
                lp.targeted_mode = 0;

                // Download targeted photon count
                unsigned int targeted_stored = 0;
                CUDA_CHECK(cudaMemcpy(&targeted_stored, d_out_photon_count_.d_ptr,
                                       sizeof(unsigned int), cudaMemcpyDeviceToHost));
                if ((int)targeted_stored > max_stored)
                    targeted_stored = (unsigned int)max_stored;

                auto t_tgt1 = std::chrono::high_resolution_clock::now();
                double tgt_ms = std::chrono::duration<double, std::milli>(t_tgt1 - t_tgt0).count();

                std::printf("[OptiX] Targeted caustic pass: %u / %d stored (%.1f%%) in %.1f ms\n",
                            targeted_stored, targeted_budget,
                            100.f * (float)targeted_stored / (float)(std::max)(1, targeted_budget),
                            tgt_ms);

                if (targeted_stored > 0) {
                    // Grow stored_photons_ to hold targeted caustic photons
                    size_t total = (size_t)stored_count + targeted_stored;
                    stored_photons_.resize((uint32_t)total);

                    // Download targeted photons into the tail of stored_photons_
                    size_t off = stored_count;
                    CUDA_CHECK(cudaMemcpy(stored_photons_.pos_x.data()  + off, d_out_photon_pos_x_.d_ptr,  targeted_stored*sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(stored_photons_.pos_y.data()  + off, d_out_photon_pos_y_.d_ptr,  targeted_stored*sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(stored_photons_.pos_z.data()  + off, d_out_photon_pos_z_.d_ptr,  targeted_stored*sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(stored_photons_.wi_x.data()   + off, d_out_photon_wi_x_.d_ptr,   targeted_stored*sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(stored_photons_.wi_y.data()   + off, d_out_photon_wi_y_.d_ptr,   targeted_stored*sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(stored_photons_.wi_z.data()   + off, d_out_photon_wi_z_.d_ptr,   targeted_stored*sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(stored_photons_.norm_x.data() + off, d_out_photon_norm_x_.d_ptr, targeted_stored*sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(stored_photons_.norm_y.data() + off, d_out_photon_norm_y_.d_ptr, targeted_stored*sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(stored_photons_.norm_z.data() + off, d_out_photon_norm_z_.d_ptr, targeted_stored*sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(stored_photons_.lambda_bin.data() + off*HERO_WAVELENGTHS, d_out_photon_lambda_.d_ptr, targeted_stored*HERO_WAVELENGTHS*sizeof(uint16_t), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(stored_photons_.flux.data()   + off*HERO_WAVELENGTHS, d_out_photon_flux_.d_ptr,   targeted_stored*HERO_WAVELENGTHS*sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(stored_photons_.num_hero.data() + off, d_out_photon_num_hero_.d_ptr, targeted_stored*sizeof(uint8_t), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(stored_photons_.source_emissive_idx.data() + off, d_out_photon_source_emissive_.d_ptr, targeted_stored*sizeof(uint16_t), cudaMemcpyDeviceToHost));
                    if (d_out_photon_tri_id_.d_ptr)
                        CUDA_CHECK(cudaMemcpy(stored_photons_.tri_id.data() + off, d_out_photon_tri_id_.d_ptr, targeted_stored*sizeof(uint32_t), cudaMemcpyDeviceToHost));

                    stored_count = (unsigned int)total;
                }

                // Account for targeted photons in caustic normalisation
                num_caustic_emitted_ += targeted_budget;

                {
                    auto t_now = std::chrono::high_resolution_clock::now();
                    double ms_t = std::chrono::duration<double, std::milli>(t_now - t_lap).count();
                    std::printf("[Timing] Targeted pass:     %8.1f ms\n", ms_t);
                    t_lap = t_now;
                }
            }
        }
    }

    // Build per-photon caustic tags (3-valued):
    //   0 = global-pass, non-caustic  → accumulate to L_indirect  (1/N_global)
    //   1 = global-pass, caustic path → SKIP at gather (superseded by caustic map)
    //   2 = caustic-targeted pass     → accumulate to L_caustic   (1/N_caustic)
    // This avoids double-counting: non-caustic indirect from global map,
    // caustic contribution from dedicated caustic map, summed at the end.
    caustic_pass_flags_.resize(stored_count, 0);
    for (size_t i = 0; i < global_count; ++i) {
        bool is_caustic_path = (i < caustic_flags_.size() && caustic_flags_[i]);
        caustic_pass_flags_[i] = is_caustic_path ? (uint8_t)1 : (uint8_t)0;
    }
    if (stored_count > global_count)
        std::fill(caustic_pass_flags_.begin() + global_count, caustic_pass_flags_.end(), (uint8_t)2);

    // ── Diagnostic: tag distribution ─────────────────────────────────
    {
        size_t n_tag0 = 0, n_tag1 = 0, n_tag2 = 0;
        for (size_t i = 0; i < stored_count; ++i) {
            uint8_t t = caustic_pass_flags_[i];
            if (t == 0) ++n_tag0;
            else if (t == 1) ++n_tag1;
            else if (t == 2) ++n_tag2;
        }
        std::printf("[OptiX] Tag distribution: tag0=%zu (noncaustic)  tag1=%zu (skip)  "
                    "tag2=%zu (caustic-pass)  total=%u\n",
                    n_tag0, n_tag1, n_tag2, stored_count);
        std::printf("[OptiX] Dual-budget: N_global_emitted=%d  N_caustic_emitted=%d  "
                    "r_global=%.5f  r_caustic=%.5f\n",
                    num_photons_emitted_, num_caustic_emitted_,
                    gather_radius_, caustic_radius_);
        // Sample a few tag-2 photons (if any) to check flux and position
        if (n_tag2 > 0) {
            int shown = 0;
            for (size_t i = 0; i < stored_count && shown < 3; ++i) {
                if (caustic_pass_flags_[i] != 2) continue;
                float px = stored_photons_.pos_x[i];
                float py = stored_photons_.pos_y[i];
                float pz = stored_photons_.pos_z[i];
                float f0 = stored_photons_.flux[i * HERO_WAVELENGTHS + 0];
                int lb0 = (int)stored_photons_.lambda_bin[i * HERO_WAVELENGTHS + 0];
                std::printf("[OptiX]   tag2[%zu]: pos=(%.3f,%.3f,%.3f)  flux[0]=%.4e  bin[0]=%d\n",
                            i, (double)px, (double)py, (double)pz, (double)f0, lb0);
                ++shown;
            }
        }
    }

    // Build hash grid on CPU (merged global + caustic photons)
    stored_grid_.build(stored_photons_, gather_radius_);

    {
        // Compute hash grid quality metrics
        int occupied_cells = 0;
        uint32_t chain_max = 0;
        uint64_t chain_sum = 0;
        for (uint32_t ci = 0; ci < stored_grid_.table_size; ++ci) {
            uint32_t len = stored_grid_.cell_end[ci] - stored_grid_.cell_start[ci];
            if (len > 0) {
                ++occupied_cells;
                chain_sum += len;
                if (len > chain_max) chain_max = len;
            }
        }
        float load_factor  = (float)occupied_cells / (float)stored_grid_.table_size;
        float mean_chain   = occupied_cells > 0 ? (float)chain_sum / (float)occupied_cells : 0.f;
        size_t grid_mem_kb = (stored_grid_.sorted_indices.size() * sizeof(uint32_t)
                             + stored_grid_.cell_start.size()    * sizeof(uint32_t)
                             + stored_grid_.cell_end.size()      * sizeof(uint32_t)) / 1024;
        auto t_now = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_now - t_lap).count();
        std::printf("[Timing] Hash grid build:   %8.1f ms  "
                    "buckets=%u  occupied=%d  load=%.2f  mean_chain=%.1f  max_chain=%u  mem=%zu KB\n",
                    ms, stored_grid_.table_size, occupied_cells, load_factor,
                    mean_chain, chain_max, grid_mem_kb);
        t_lap = t_now;
    }

    // Upload photons + grid back to device
    d_photon_pos_x_.upload(stored_photons_.pos_x);
    d_photon_pos_y_.upload(stored_photons_.pos_y);
    d_photon_pos_z_.upload(stored_photons_.pos_z);
    d_photon_wi_x_.upload(stored_photons_.wi_x);
    d_photon_wi_y_.upload(stored_photons_.wi_y);
    d_photon_wi_z_.upload(stored_photons_.wi_z);
    d_photon_norm_x_.upload(stored_photons_.norm_x);
    d_photon_norm_y_.upload(stored_photons_.norm_y);
    d_photon_norm_z_.upload(stored_photons_.norm_z);
    d_photon_lambda_.upload(stored_photons_.lambda_bin);
    d_photon_flux_.upload(stored_photons_.flux);
    d_photon_num_hero_.upload(stored_photons_.num_hero);
    d_grid_sorted_indices_.upload(stored_grid_.sorted_indices);
    d_grid_cell_start_.upload(stored_grid_.cell_start);
    d_grid_cell_end_.upload(stored_grid_.cell_end);
    // Upload per-photon caustic-pass tags (for dual-budget gather)
    d_photon_is_caustic_pass_.upload(caustic_pass_flags_);

    // Build and upload per-triangle photon irradiance heatmap (for preview)
    {
        auto irr = build_tri_photon_irradiance(stored_photons_, num_tris_);
        if (!irr.empty())
            d_tri_photon_irradiance_.upload(irr);
        else
            d_tri_photon_irradiance_.free();
    }

    std::cout << "[OptiX] Photon data uploaded to device\n";

    {
        auto t_now = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_now - t_lap).count();
        std::printf("[Timing] Photon upload:     %8.1f ms\n", ms);
        t_lap = t_now;
    }

    // (Light cache / oct-bin infrastructure removed — golden-ratio NEE used instead)

    // ── Download and process volume photons ──────────────────────────
    if (config.volume_enabled && config.volume_density > 0.f) {
        unsigned int vol_count = 0;
        CUDA_CHECK(cudaMemcpy(&vol_count, d_out_vol_photon_count_.d_ptr,
                               sizeof(unsigned int), cudaMemcpyDeviceToHost));
        if ((int)vol_count > max_vol_stored)
            vol_count = (unsigned int)max_vol_stored;

        std::cout << "[OptiX] GPU photon trace stored " << vol_count << " volume photons\n";

        if (vol_count > 0) {
            volume_photons_.resize(vol_count);
            CUDA_CHECK(cudaMemcpy(volume_photons_.pos_x.data(),  d_out_vol_photon_pos_x_.d_ptr,  vol_count*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(volume_photons_.pos_y.data(),  d_out_vol_photon_pos_y_.d_ptr,  vol_count*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(volume_photons_.pos_z.data(),  d_out_vol_photon_pos_z_.d_ptr,  vol_count*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(volume_photons_.wi_x.data(),   d_out_vol_photon_wi_x_.d_ptr,   vol_count*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(volume_photons_.wi_y.data(),   d_out_vol_photon_wi_y_.d_ptr,   vol_count*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(volume_photons_.wi_z.data(),   d_out_vol_photon_wi_z_.d_ptr,   vol_count*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(volume_photons_.lambda_bin.data(), d_out_vol_photon_lambda_.d_ptr, vol_count*sizeof(uint16_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(volume_photons_.flux.data(),   d_out_vol_photon_flux_.d_ptr,   vol_count*sizeof(float), cudaMemcpyDeviceToHost));
        }
    }

    {
        auto t_now = std::chrono::high_resolution_clock::now();
        double total = std::chrono::duration<double, std::milli>(t_now - t_phase_start).count();
        std::printf("[Timing] Photon total:      %8.1f ms\n", total);
    }

    // ── Adaptive gather radius (k-NN, §C2) ──────────────────────────
    // When use_knn_adaptive is set, compute a scene-representative
    // gather radius by running knn_shell_expansion() on a random sample
    // of stored photon positions and taking the median k-th distance.
    // This replaces the fixed gather_radius_ for subsequent renders.
    if (config.use_knn_adaptive && stored_photons_.size() > 0) {
        const int   k           = config.knn_k;
        const float tau         = DEFAULT_SURFACE_TAU;
        const float max_radius  = DEFAULT_GPU_MAX_GATHER_RADIUS;
        const int   num_samples = (std::min)((int)stored_photons_.size(), 256);
        const size_t stride     = (std::max)((size_t)1,
                                           stored_photons_.size() / num_samples);

        std::vector<float> knn_radii;
        knn_radii.reserve(num_samples);

        std::vector<uint32_t> tmp_indices;
        float tmp_max_dist2 = 0.f;

        for (int s = 0; s < num_samples; ++s) {
            size_t idx = (size_t)s * stride;
            float3 pos    = make_f3(stored_photons_.pos_x[idx],
                                    stored_photons_.pos_y[idx],
                                    stored_photons_.pos_z[idx]);
            float3 normal = make_f3(stored_photons_.norm_x[idx],
                                    stored_photons_.norm_y[idx],
                                    stored_photons_.norm_z[idx]);

            stored_grid_.knn_shell_expansion(
                pos, normal, k, tau,
                stored_photons_, tmp_indices, tmp_max_dist2);

            if (!tmp_indices.empty()) {
                float r = sqrtf(fminf(tmp_max_dist2, max_radius * max_radius));
                knn_radii.push_back(r);
            }
        }

        if (!knn_radii.empty()) {
            std::sort(knn_radii.begin(), knn_radii.end());
            float median_r = knn_radii[knn_radii.size() / 2];
            median_r = fminf(fmaxf(median_r, 1e-4f), max_radius);
            std::printf("[k-NN] Adaptive gather radius: %.5f  (k=%d, %zu samples)\n",
                        median_r, k, knn_radii.size());
            gather_radius_ = median_r;
        }
    }
}

// =====================================================================
// fill_common_params() -- helper to fill LaunchParams
// =====================================================================
static void fill_common_params(
    LaunchParams& p,
    const DeviceBuffer& spectrum, const DeviceBuffer& samples,
    const DeviceBuffer& srgb, int w, int h,
    const Camera& camera,
    const DeviceBuffer& vertices, const DeviceBuffer& normals,
    const DeviceBuffer& texcoords_buf,
    const DeviceBuffer& material_ids,
    const DeviceBuffer& Kd, const DeviceBuffer& Ks,
    const DeviceBuffer& Le, const DeviceBuffer& roughness,
    const DeviceBuffer& ior, const DeviceBuffer& mat_type,
    const DeviceBuffer& diffuse_tex_buf,
    const DeviceBuffer& tex_atlas_buf, const DeviceBuffer& tex_descs_buf,
    int num_textures,
    const DeviceBuffer& photon_pos_x, const DeviceBuffer& photon_pos_y,
    const DeviceBuffer& photon_pos_z,
    const DeviceBuffer& photon_wi_x, const DeviceBuffer& photon_wi_y,
    const DeviceBuffer& photon_wi_z,
    const DeviceBuffer& photon_norm_x, const DeviceBuffer& photon_norm_y,
    const DeviceBuffer& photon_norm_z,
    const DeviceBuffer& photon_lambda, const DeviceBuffer& photon_flux,
    const DeviceBuffer& grid_sorted, const DeviceBuffer& grid_start,
    const DeviceBuffer& grid_end,
    const DeviceBuffer& emissive_idx, const DeviceBuffer& emissive_cdf,
    int num_emissive, float total_emissive_power,
    OptixTraversableHandle gas_handle,
    float gather_radius,
    int   num_photons_emitted,
    const DeviceBuffer& nee_direct_buf,
    const DeviceBuffer& photon_indirect_buf,
    const DeviceBuffer& prof_total,
    const DeviceBuffer& prof_ray_trace,
    const DeviceBuffer& prof_nee,
    const DeviceBuffer& prof_photon_gather,
    const DeviceBuffer& prof_bsdf,
    // ── Volume participating-medium params ──
    bool volume_enabled, float volume_density, float volume_falloff,
    float volume_albedo, int volume_samples, float volume_max_t)
{
    p.spectrum_buffer = const_cast<float*>(spectrum.as<float>());
    p.sample_counts   = const_cast<float*>(samples.as<float>());
    p.srgb_buffer     = const_cast<uint8_t*>(srgb.as<uint8_t>());
    p.nee_direct_buffer      = nee_direct_buf.d_ptr
        ? const_cast<float*>(nee_direct_buf.as<float>()) : nullptr;
    p.photon_indirect_buffer = photon_indirect_buf.d_ptr
        ? const_cast<float*>(photon_indirect_buf.as<float>()) : nullptr;

    // Profiling buffers (may be nullptr)
    p.prof_total         = prof_total.d_ptr
        ? reinterpret_cast<long long*>(prof_total.d_ptr) : nullptr;
    p.prof_ray_trace     = prof_ray_trace.d_ptr
        ? reinterpret_cast<long long*>(prof_ray_trace.d_ptr) : nullptr;
    p.prof_nee           = prof_nee.d_ptr
        ? reinterpret_cast<long long*>(prof_nee.d_ptr) : nullptr;
    p.prof_photon_gather = prof_photon_gather.d_ptr
        ? reinterpret_cast<long long*>(prof_photon_gather.d_ptr) : nullptr;
    p.prof_bsdf          = prof_bsdf.d_ptr
        ? reinterpret_cast<long long*>(prof_bsdf.d_ptr) : nullptr;
    p.width           = w;
    p.height          = h;

    p.cam_pos         = camera.position;
    p.cam_u           = camera.u;
    p.cam_v           = camera.v;
    p.cam_w           = camera.w;
    p.cam_lower_left  = camera.lower_left;
    p.cam_horizontal  = camera.horizontal;
    p.cam_vertical    = camera.vertical;
    p.cam_lens_radius = camera.lens_radius;
    p.cam_focus_dist  = camera.dof_focus_dist;
    // dof_focus_range is stored as a fraction; convert to absolute slab depth
    p.cam_focus_range = camera.dof_focus_range * camera.dof_focus_dist;

    p.vertices     = const_cast<float3*>(vertices.as<float3>());
    p.normals      = const_cast<float3*>(normals.as<float3>());
    p.texcoords    = texcoords_buf.d_ptr
        ? const_cast<float2*>(texcoords_buf.as<float2>()) : nullptr;
    p.material_ids = const_cast<uint32_t*>(material_ids.as<uint32_t>());

    p.num_materials = (int)(mat_type.bytes / sizeof(uint8_t));
    p.Kd            = const_cast<float*>(Kd.as<float>());
    p.Ks            = const_cast<float*>(Ks.as<float>());
    p.Le            = const_cast<float*>(Le.as<float>());
    p.roughness     = const_cast<float*>(roughness.as<float>());
    p.ior           = const_cast<float*>(ior.as<float>());
    p.mat_type      = const_cast<uint8_t*>(mat_type.as<uint8_t>());
    p.diffuse_tex   = diffuse_tex_buf.d_ptr
        ? const_cast<int*>(diffuse_tex_buf.as<int>()) : nullptr;
    p.tex_atlas     = tex_atlas_buf.d_ptr
        ? const_cast<float*>(tex_atlas_buf.as<float>()) : nullptr;
    p.tex_descs     = tex_descs_buf.d_ptr
        ? reinterpret_cast<GpuTexDesc*>(tex_descs_buf.d_ptr) : nullptr;
    p.num_textures  = num_textures;

    // num_photons = flux bytes / (HERO_WAVELENGTHS * sizeof(float))
    p.num_photons       = (int)(photon_flux.bytes / (HERO_WAVELENGTHS * sizeof(float)));
    p.num_photons_emitted = num_photons_emitted;
    p.photon_map_seed   = 0;  // default; overridden by render_final for multi-map
    p.photon_pos_x      = const_cast<float*>(photon_pos_x.as<float>());
    p.photon_pos_y      = const_cast<float*>(photon_pos_y.as<float>());
    p.photon_pos_z      = const_cast<float*>(photon_pos_z.as<float>());
    p.photon_wi_x       = const_cast<float*>(photon_wi_x.as<float>());
    p.photon_wi_y       = const_cast<float*>(photon_wi_y.as<float>());
    p.photon_wi_z       = const_cast<float*>(photon_wi_z.as<float>());
    p.photon_norm_x     = photon_norm_x.d_ptr ? const_cast<float*>(photon_norm_x.as<float>()) : nullptr;
    p.photon_norm_y     = photon_norm_y.d_ptr ? const_cast<float*>(photon_norm_y.as<float>()) : nullptr;
    p.photon_norm_z     = photon_norm_z.d_ptr ? const_cast<float*>(photon_norm_z.as<float>()) : nullptr;
    p.photon_lambda     = const_cast<uint16_t*>(photon_lambda.as<uint16_t>());
    p.photon_flux       = const_cast<float*>(photon_flux.as<float>());
    p.photon_num_hero   = nullptr;  // set by callers that have the buffer

    p.grid_sorted_indices = const_cast<uint32_t*>(grid_sorted.as<uint32_t>());
    p.grid_cell_start     = const_cast<uint32_t*>(grid_start.as<uint32_t>());
    p.grid_cell_end       = const_cast<uint32_t*>(grid_end.as<uint32_t>());
    p.gather_radius       = gather_radius;

    if (grid_sorted.d_ptr) {
        p.grid_cell_size  = gather_radius * HASHGRID_CELL_FACTOR;
        p.grid_table_size = (uint32_t)(grid_start.bytes / sizeof(uint32_t));
    }

    // Emitter data (for NEE in render and photon trace)
    p.emissive_tri_indices = const_cast<uint32_t*>(emissive_idx.as<uint32_t>());
    p.emissive_cdf         = const_cast<float*>(emissive_cdf.as<float>());
    p.num_emissive         = num_emissive;
    p.total_emissive_power = total_emissive_power;

    p.traversable = gas_handle;

    // ── Participating medium ─────────────────────────────────────────
    p.volume_enabled = volume_enabled ? 1 : 0;
    p.volume_density = volume_density;
    p.volume_falloff = volume_falloff;
    p.volume_albedo  = volume_albedo;
    p.volume_samples = volume_samples;
    p.volume_max_t   = volume_max_t;
}

// =====================================================================
// render_debug_frame() -- first-hit only (is_final_render = 0)
// =====================================================================
void OptixRenderer::render_debug_frame(
    const Camera& camera, int frame_number,
    RenderMode mode, int spp, bool shadow_rays)
{
    LaunchParams lp = {};
    fill_common_params(lp,
        d_spectrum_buffer_, d_sample_counts_, d_srgb_buffer_,
        width_, height_, camera,
        d_vertices_, d_normals_, d_texcoords_,
        d_material_ids_,
        d_Kd_, d_Ks_, d_Le_, d_roughness_, d_ior_, d_mat_type_,
        d_diffuse_tex_, d_tex_atlas_, d_tex_descs_,
        (int)(d_tex_descs_.bytes / sizeof(GpuTexDesc)),
        d_photon_pos_x_, d_photon_pos_y_, d_photon_pos_z_,
        d_photon_wi_x_, d_photon_wi_y_, d_photon_wi_z_,
        d_photon_norm_x_, d_photon_norm_y_, d_photon_norm_z_,
        d_photon_lambda_, d_photon_flux_,
        d_grid_sorted_indices_, d_grid_cell_start_, d_grid_cell_end_,
        d_emissive_indices_, d_emissive_cdf_,
        num_emissive_, 0.f,
        gas_handle_,
        gather_radius_,
        num_photons_emitted_,
        DeviceBuffer(), DeviceBuffer(),
        DeviceBuffer(), DeviceBuffer(), DeviceBuffer(),
        DeviceBuffer(), DeviceBuffer(),
        DEFAULT_VOLUME_ENABLED, DEFAULT_VOLUME_DENSITY, DEFAULT_VOLUME_FALLOFF,
        DEFAULT_VOLUME_ALBEDO, DEFAULT_VOLUME_SAMPLES, DEFAULT_VOLUME_MAX_T);
    fill_clearcoat_fabric_params(lp);

    // Dual-budget caustic params
    lp.photon_is_caustic_pass = d_photon_is_caustic_pass_.d_ptr
        ? d_photon_is_caustic_pass_.as<uint8_t>() : nullptr;
    lp.num_caustic_emitted    = num_caustic_emitted_;
    lp.caustic_gather_radius  = caustic_radius_;

    // Runtime toggle (V key) overrides the compile-time default
    lp.volume_enabled = (int)runtime_volume_enabled_;
    lp.photon_num_hero = d_photon_num_hero_.d_ptr ? d_photon_num_hero_.as<uint8_t>() : nullptr;

    lp.samples_per_pixel = spp;
    lp.max_bounces       = DEFAULT_MAX_BOUNCES;
    lp.frame_number      = frame_number;
    lp.render_mode       = (int)mode;
    lp.is_final_render   = 0;  // DEBUG: first-hit + direct lighting only
    lp.debug_shadow_rays = shadow_rays ? 1 : 0;
    lp.nee_light_samples = shadow_rays ? DEFAULT_NEE_LIGHT_SAMPLES : 1;
    lp.nee_deep_samples   = shadow_rays ? DEFAULT_NEE_DEEP_SAMPLES  : 1;
    lp.exposure           = exposure_;

    // Per-triangle photon irradiance heatmap
    lp.tri_photon_irradiance = d_tri_photon_irradiance_.d_ptr
        ? d_tri_photon_irradiance_.as<float>() : nullptr;
    lp.num_triangles         = num_tris_;
    lp.show_photon_heatmap   = show_photon_heatmap_ ? 1 : 0;

    last_launch_params_host_ = lp;

    // Upload launch params
    d_launch_params_.alloc(sizeof(LaunchParams));
    CUDA_CHECK(cudaMemcpy(d_launch_params_.d_ptr, &lp,
                           sizeof(LaunchParams), cudaMemcpyHostToDevice));

    // Launch
    OPTIX_CHECK(optixLaunch(
        pipeline_,
        nullptr,
        reinterpret_cast<CUdeviceptr>(d_launch_params_.d_ptr),
        sizeof(LaunchParams),
        &sbt_,
        width_, height_, 1));

    CUDA_CHECK(cudaDeviceSynchronize());
}

// =====================================================================
// render_one_spp() -- launch a single sample of full path tracing
// =====================================================================
void OptixRenderer::render_one_spp(
    const Camera& camera, int frame_number, int max_bounces)
{
    LaunchParams lp = {};
    fill_common_params(lp,
        d_spectrum_buffer_, d_sample_counts_, d_srgb_buffer_,
        width_, height_, camera,
        d_vertices_, d_normals_, d_texcoords_,
        d_material_ids_,
        d_Kd_, d_Ks_, d_Le_, d_roughness_, d_ior_, d_mat_type_,
        d_diffuse_tex_, d_tex_atlas_, d_tex_descs_,
        (int)(d_tex_descs_.bytes / sizeof(GpuTexDesc)),
        d_photon_pos_x_, d_photon_pos_y_, d_photon_pos_z_,
        d_photon_wi_x_, d_photon_wi_y_, d_photon_wi_z_,
        d_photon_norm_x_, d_photon_norm_y_, d_photon_norm_z_,
        d_photon_lambda_, d_photon_flux_,
        d_grid_sorted_indices_, d_grid_cell_start_, d_grid_cell_end_,
        d_emissive_indices_, d_emissive_cdf_,
        num_emissive_, 0.f,
        gas_handle_,
        gather_radius_,
        num_photons_emitted_,
        d_nee_direct_buffer_, d_photon_indirect_buffer_,
        d_prof_total_, d_prof_ray_trace_, d_prof_nee_,
        d_prof_photon_gather_, d_prof_bsdf_,
        DEFAULT_VOLUME_ENABLED, DEFAULT_VOLUME_DENSITY, DEFAULT_VOLUME_FALLOFF,
        DEFAULT_VOLUME_ALBEDO, DEFAULT_VOLUME_SAMPLES, DEFAULT_VOLUME_MAX_T);
    fill_clearcoat_fabric_params(lp);

    // Dual-budget caustic params
    lp.photon_is_caustic_pass = d_photon_is_caustic_pass_.d_ptr
        ? d_photon_is_caustic_pass_.as<uint8_t>() : nullptr;
    lp.num_caustic_emitted    = num_caustic_emitted_;
    lp.caustic_gather_radius  = caustic_radius_;

    // Runtime toggle (V key) overrides the compile-time default
    lp.volume_enabled = (int)runtime_volume_enabled_;
    lp.photon_num_hero = d_photon_num_hero_.d_ptr ? d_photon_num_hero_.as<uint8_t>() : nullptr;

    lp.samples_per_pixel  = 1;
    lp.max_bounces        = max_bounces;
    lp.frame_number       = frame_number;
    lp.render_mode        = RENDER_MODE_FULL;
    lp.is_final_render    = 1;
    lp.nee_light_samples  = DEFAULT_NEE_LIGHT_SAMPLES;
    lp.nee_deep_samples   = DEFAULT_NEE_DEEP_SAMPLES;
    lp.exposure           = exposure_;

    last_launch_params_host_ = lp;

    d_launch_params_.alloc(sizeof(LaunchParams));
    CUDA_CHECK(cudaMemcpy(d_launch_params_.d_ptr, &lp,
                           sizeof(LaunchParams), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(
        pipeline_,
        nullptr,
        reinterpret_cast<CUdeviceptr>(d_launch_params_.d_ptr),
        sizeof(LaunchParams),
        &sbt_,
        width_, height_, 1));

    CUDA_CHECK(cudaDeviceSynchronize());
}

// =====================================================================
// render_caustic_debug_pass() -- re-render using only caustic photons
//   1. Filter stored photons to caustic-only subset
//   2. Upload caustic photons, rebuild hash grid + cell-bin grid
//   3. Clear accumulation buffers
//   4. Launch a short camera pass (CAUSTIC_DEBUG_SPP samples)
//   5. Download photon_indirect_buffer as the caustic-only component
// =====================================================================
void OptixRenderer::render_caustic_debug_pass(
    const Camera& camera,
    const RenderConfig& config,
    std::vector<float>& caustic_spec,
    std::vector<float>& samp_counts)
{
    constexpr int CAUSTIC_DEBUG_SPP = 8;

    if (caustic_pass_flags_.empty() || stored_photons_.size() == 0) {
        std::cout << "[CausticDebug] No photon data — skipping.\n";
        return;
    }

    std::cout << "[CausticDebug] Building tag-2 caustic-only photon map...\n";
    auto t_start = std::chrono::high_resolution_clock::now();

    // 1. Filter stored photons to targeted-caustic only (tag 2)
    //    caustic_pass_flags_: 0=global-noncaustic, 1=global-caustic(skip), 2=caustic-targeted
    //    Only include tag 2 — the dedicated targeted caustic pass.
    PhotonSoA caustic_soa;
    for (size_t i = 0; i < stored_photons_.size(); ++i) {
        if (i >= caustic_pass_flags_.size() || caustic_pass_flags_[i] != 2) continue;
        caustic_soa.pos_x.push_back(stored_photons_.pos_x[i]);
        caustic_soa.pos_y.push_back(stored_photons_.pos_y[i]);
        caustic_soa.pos_z.push_back(stored_photons_.pos_z[i]);
        caustic_soa.wi_x.push_back(stored_photons_.wi_x[i]);
        caustic_soa.wi_y.push_back(stored_photons_.wi_y[i]);
        caustic_soa.wi_z.push_back(stored_photons_.wi_z[i]);
        caustic_soa.norm_x.push_back(stored_photons_.norm_x[i]);
        caustic_soa.norm_y.push_back(stored_photons_.norm_y[i]);
        caustic_soa.norm_z.push_back(stored_photons_.norm_z[i]);
        // Copy hero wavelength data (interleaved: HERO_WAVELENGTHS per photon)
        for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
            caustic_soa.lambda_bin.push_back(stored_photons_.lambda_bin[i * HERO_WAVELENGTHS + h]);
            caustic_soa.flux.push_back(stored_photons_.flux[i * HERO_WAVELENGTHS + h]);
        }
        caustic_soa.num_hero.push_back(stored_photons_.num_hero[i]);
        if (!stored_photons_.source_emissive_idx.empty())
            caustic_soa.source_emissive_idx.push_back(stored_photons_.source_emissive_idx[i]);
    }

    size_t n_caustic = caustic_soa.size();
    if (n_caustic == 0) {
        std::cout << "[CausticDebug] No caustic photons found — skipping.\n";
        return;
    }
    std::printf("[CausticDebug] Filtered %zu tag-2 caustic photons from %zu total\n",
                n_caustic, stored_photons_.size());

    // Diagnostic: flux statistics of caustic photons
    {
        float min_flux = 1e30f, max_flux = 0.f, sum_flux = 0.f;
        for (size_t i = 0; i < n_caustic; ++i) {
            for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
                float f = caustic_soa.flux[i * HERO_WAVELENGTHS + h];
                if (f > 0.f) {
                    min_flux = (std::min)(min_flux, f);
                    max_flux = (std::max)(max_flux, f);
                    sum_flux += f;
                }
            }
        }
        float avg_flux = sum_flux / (std::max)((float)(n_caustic * HERO_WAVELENGTHS), 1.f);
        std::printf("[CausticDebug] Flux stats: min=%.4e  max=%.4e  avg=%.4e  total=%.4e\n",
                    (double)min_flux, (double)max_flux, (double)avg_flux, (double)sum_flux);
    }

    // 2. Build hash grid from caustic photons
    HashGrid caustic_grid;
    caustic_grid.build(caustic_soa, gather_radius_);

    // 3. Upload caustic photons to device (overwrites current photon data)
    d_photon_pos_x_.upload(caustic_soa.pos_x);
    d_photon_pos_y_.upload(caustic_soa.pos_y);
    d_photon_pos_z_.upload(caustic_soa.pos_z);
    d_photon_wi_x_.upload(caustic_soa.wi_x);
    d_photon_wi_y_.upload(caustic_soa.wi_y);
    d_photon_wi_z_.upload(caustic_soa.wi_z);
    d_photon_norm_x_.upload(caustic_soa.norm_x);
    d_photon_norm_y_.upload(caustic_soa.norm_y);
    d_photon_norm_z_.upload(caustic_soa.norm_z);
    d_photon_lambda_.upload(caustic_soa.lambda_bin);
    d_photon_flux_.upload(caustic_soa.flux);
    d_photon_num_hero_.upload(caustic_soa.num_hero);
    d_grid_sorted_indices_.upload(caustic_grid.sorted_indices);
    d_grid_cell_start_.upload(caustic_grid.cell_start);
    d_grid_cell_end_.upload(caustic_grid.cell_end);

    auto t_setup = std::chrono::high_resolution_clock::now();
    double ms_setup = std::chrono::duration<double, std::milli>(t_setup - t_start).count();
    std::printf("[CausticDebug] Caustic map uploaded (%.1f ms)\n", ms_setup);

    // 5. Clear accumulation buffers for the caustic-only pass
    CUDA_CHECK(cudaMemset(d_spectrum_buffer_.d_ptr, 0, d_spectrum_buffer_.bytes));
    CUDA_CHECK(cudaMemset(d_sample_counts_.d_ptr,   0, d_sample_counts_.bytes));
    CUDA_CHECK(cudaMemset(d_photon_indirect_buffer_.d_ptr, 0, d_photon_indirect_buffer_.bytes));
    CUDA_CHECK(cudaMemset(d_nee_direct_buffer_.d_ptr, 0, d_nee_direct_buffer_.bytes));
    CUDA_CHECK(cudaMemset(d_lobe_balance_.d_ptr, 0, d_lobe_balance_.bytes));

    // 6. Launch camera pass at CAUSTIC_DEBUG_SPP
    std::printf("[CausticDebug] Rendering %d spp with caustic-only photon map...\n",
                CAUSTIC_DEBUG_SPP);

    for (int s = 0; s < CAUSTIC_DEBUG_SPP; ++s) {
        LaunchParams lp = {};
        fill_common_params(lp,
            d_spectrum_buffer_, d_sample_counts_, d_srgb_buffer_,
            width_, height_, camera,
            d_vertices_, d_normals_, d_texcoords_,
            d_material_ids_,
            d_Kd_, d_Ks_, d_Le_, d_roughness_, d_ior_, d_mat_type_,
            d_diffuse_tex_, d_tex_atlas_, d_tex_descs_,
            (int)(d_tex_descs_.bytes / sizeof(GpuTexDesc)),
            d_photon_pos_x_, d_photon_pos_y_, d_photon_pos_z_,
            d_photon_wi_x_, d_photon_wi_y_, d_photon_wi_z_,
            d_photon_norm_x_, d_photon_norm_y_, d_photon_norm_z_,
            d_photon_lambda_, d_photon_flux_,
            d_grid_sorted_indices_, d_grid_cell_start_, d_grid_cell_end_,
            d_emissive_indices_, d_emissive_cdf_,
            num_emissive_, 0.f,
            gas_handle_,
            gather_radius_,
            num_caustic_emitted_,  // tag-2 only: normalise by caustic budget
            d_nee_direct_buffer_, d_photon_indirect_buffer_,
            d_prof_total_, d_prof_ray_trace_, d_prof_nee_,
            d_prof_photon_gather_, d_prof_bsdf_,
            config.volume_enabled, config.volume_density, config.volume_falloff,
            config.volume_albedo, config.volume_samples, config.volume_max_t);
        fill_clearcoat_fabric_params(lp);

        // Disable dual-budget for this debug pass — all photons are
        // tag-2 caustic, so single-budget gather with N_caustic is correct.
        lp.photon_is_caustic_pass = nullptr;
        lp.num_caustic_emitted    = 0;
        lp.caustic_gather_radius  = caustic_radius_;

        lp.volume_enabled     = (int)runtime_volume_enabled_;
        lp.photon_num_hero    = d_photon_num_hero_.d_ptr ? d_photon_num_hero_.as<uint8_t>() : nullptr;
        lp.samples_per_pixel  = 1;
        lp.max_bounces        = config.max_bounces;
        lp.frame_number       = s;
        lp.render_mode        = RENDER_MODE_FULL;
        lp.is_final_render    = 1;
        lp.nee_light_samples  = DEFAULT_NEE_LIGHT_SAMPLES;
        lp.nee_deep_samples   = DEFAULT_NEE_DEEP_SAMPLES;
        lp.exposure           = exposure_;

        d_launch_params_.alloc(sizeof(LaunchParams));
        CUDA_CHECK(cudaMemcpy(d_launch_params_.d_ptr, &lp,
                               sizeof(LaunchParams), cudaMemcpyHostToDevice));

        OPTIX_CHECK(optixLaunch(
            pipeline_,
            nullptr,
            reinterpret_cast<CUdeviceptr>(d_launch_params_.d_ptr),
            sizeof(LaunchParams),
            &sbt_,
            width_, height_, 1));

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 7. Download photon_indirect_buffer (caustic-only component)
    size_t pixels    = (size_t)width_ * height_;
    size_t spec_size = pixels * NUM_LAMBDA;
    caustic_spec.resize(spec_size);
    samp_counts.resize(pixels);

    CUDA_CHECK(cudaMemcpy(caustic_spec.data(), d_photon_indirect_buffer_.d_ptr,
                          spec_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(samp_counts.data(), d_sample_counts_.d_ptr,
                          pixels * sizeof(float), cudaMemcpyDeviceToHost));

    // 8. Restore full photon map to GPU so subsequent renders use all photons
    d_photon_pos_x_.upload(stored_photons_.pos_x);
    d_photon_pos_y_.upload(stored_photons_.pos_y);
    d_photon_pos_z_.upload(stored_photons_.pos_z);
    d_photon_wi_x_.upload(stored_photons_.wi_x);
    d_photon_wi_y_.upload(stored_photons_.wi_y);
    d_photon_wi_z_.upload(stored_photons_.wi_z);
    d_photon_norm_x_.upload(stored_photons_.norm_x);
    d_photon_norm_y_.upload(stored_photons_.norm_y);
    d_photon_norm_z_.upload(stored_photons_.norm_z);
    d_photon_lambda_.upload(stored_photons_.lambda_bin);
    d_photon_flux_.upload(stored_photons_.flux);
    if (!stored_photons_.num_hero.empty())
        d_photon_num_hero_.upload(stored_photons_.num_hero);
    d_grid_sorted_indices_.upload(stored_grid_.sorted_indices);
    d_grid_cell_start_.upload(stored_grid_.cell_start);
    d_grid_cell_end_.upload(stored_grid_.cell_end);

    auto t_end = std::chrono::high_resolution_clock::now();
    double ms_total = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::printf("[CausticDebug] Done (%.1f ms total, %d spp, full photon map restored)\n",
                ms_total, CAUSTIC_DEBUG_SPP);
}

// =====================================================================
// render_caustic_snapshot() -- non-destructive caustic debug pass
//   Saves progressive accumulation buffers, runs caustic debug pass,
//   then restores the saved state so progressive rendering can continue.
// =====================================================================
void OptixRenderer::render_caustic_snapshot(
    const Camera& camera,
    const RenderConfig& config,
    std::vector<float>& caustic_spec,
    std::vector<float>& caustic_samp)
{
    size_t pixels    = (size_t)width_ * height_;
    size_t spec_bytes = pixels * NUM_LAMBDA * sizeof(float);
    size_t samp_bytes = pixels * sizeof(float);

    // 1. Save progressive accumulation buffers to CPU
    std::vector<float> saved_spectrum(pixels * NUM_LAMBDA);
    std::vector<float> saved_nee(pixels * NUM_LAMBDA);
    std::vector<float> saved_photon(pixels * NUM_LAMBDA);
    std::vector<float> saved_samp(pixels);
    std::vector<float> saved_lobe(pixels);

    if (d_spectrum_buffer_.d_ptr)
        CUDA_CHECK(cudaMemcpy(saved_spectrum.data(), d_spectrum_buffer_.d_ptr,
                              spec_bytes, cudaMemcpyDeviceToHost));
    if (d_nee_direct_buffer_.d_ptr)
        CUDA_CHECK(cudaMemcpy(saved_nee.data(), d_nee_direct_buffer_.d_ptr,
                              spec_bytes, cudaMemcpyDeviceToHost));
    if (d_photon_indirect_buffer_.d_ptr)
        CUDA_CHECK(cudaMemcpy(saved_photon.data(), d_photon_indirect_buffer_.d_ptr,
                              spec_bytes, cudaMemcpyDeviceToHost));
    if (d_sample_counts_.d_ptr)
        CUDA_CHECK(cudaMemcpy(saved_samp.data(), d_sample_counts_.d_ptr,
                              samp_bytes, cudaMemcpyDeviceToHost));
    if (d_lobe_balance_.d_ptr)
        CUDA_CHECK(cudaMemcpy(saved_lobe.data(), d_lobe_balance_.d_ptr,
                              samp_bytes, cudaMemcpyDeviceToHost));

    // 2. Run destructive caustic debug pass (it restores photon map internally)
    render_caustic_debug_pass(camera, config, caustic_spec, caustic_samp);

    // 3. Restore progressive accumulation buffers
    if (d_spectrum_buffer_.d_ptr)
        CUDA_CHECK(cudaMemcpy(d_spectrum_buffer_.d_ptr, saved_spectrum.data(),
                              spec_bytes, cudaMemcpyHostToDevice));
    if (d_nee_direct_buffer_.d_ptr)
        CUDA_CHECK(cudaMemcpy(d_nee_direct_buffer_.d_ptr, saved_nee.data(),
                              spec_bytes, cudaMemcpyHostToDevice));
    if (d_photon_indirect_buffer_.d_ptr)
        CUDA_CHECK(cudaMemcpy(d_photon_indirect_buffer_.d_ptr, saved_photon.data(),
                              spec_bytes, cudaMemcpyHostToDevice));
    if (d_sample_counts_.d_ptr)
        CUDA_CHECK(cudaMemcpy(d_sample_counts_.d_ptr, saved_samp.data(),
                              samp_bytes, cudaMemcpyHostToDevice));
    if (d_lobe_balance_.d_ptr)
        CUDA_CHECK(cudaMemcpy(d_lobe_balance_.d_ptr, saved_lobe.data(),
                              samp_bytes, cudaMemcpyHostToDevice));
}

// =====================================================================
// render_final() -- full path tracing (is_final_render = 1)
// =====================================================================
void OptixRenderer::render_final(
    const Camera& camera, const RenderConfig& config, const Scene& scene)
{
    // Reset accumulation
    resize(config.image_width, config.image_height);

    const int total_spp    = config.samples_per_pixel;
    const int max_spp      = (config.adaptive_max_spp > 0)
                                 ? config.adaptive_max_spp
                                 : total_spp;
    const int min_spp      = config.adaptive_min_spp;
    const bool adaptive    = config.adaptive_sampling && (max_spp > min_spp);

    const long long total_pixels = (long long)config.image_width * config.image_height;
    std::cout << "[Render] Starting: " << config.image_width << "x"
              << config.image_height << " @ " << total_spp
              << " spp" << (adaptive ? " (adaptive)" : "")
              << " (" << total_pixels << " pixels)\n";

    auto t_start = std::chrono::high_resolution_clock::now();

    // Helper lambda: launch one pass with the given frame number and optional mask
    auto launch_pass = [&](int frame_number, bool use_mask) {
        LaunchParams lp = {};
        fill_common_params(lp,
            d_spectrum_buffer_, d_sample_counts_, d_srgb_buffer_,
            width_, height_, camera,
            d_vertices_, d_normals_, d_texcoords_,
            d_material_ids_,
            d_Kd_, d_Ks_, d_Le_, d_roughness_, d_ior_, d_mat_type_,
            d_diffuse_tex_, d_tex_atlas_, d_tex_descs_,
            (int)(d_tex_descs_.bytes / sizeof(GpuTexDesc)),
            d_photon_pos_x_, d_photon_pos_y_, d_photon_pos_z_,
            d_photon_wi_x_, d_photon_wi_y_, d_photon_wi_z_,
            d_photon_norm_x_, d_photon_norm_y_, d_photon_norm_z_,
            d_photon_lambda_, d_photon_flux_,
            d_grid_sorted_indices_, d_grid_cell_start_, d_grid_cell_end_,
            d_emissive_indices_, d_emissive_cdf_,
            num_emissive_, 0.f,
            gas_handle_,
            gather_radius_,
            num_photons_emitted_,
            d_nee_direct_buffer_, d_photon_indirect_buffer_,
            d_prof_total_, d_prof_ray_trace_, d_prof_nee_,
            d_prof_photon_gather_, d_prof_bsdf_,
            config.volume_enabled, config.volume_density, config.volume_falloff,
            config.volume_albedo, config.volume_samples, config.volume_max_t);
        fill_clearcoat_fabric_params(lp);

        // Dual-budget caustic params
        lp.photon_is_caustic_pass = d_photon_is_caustic_pass_.d_ptr
            ? d_photon_is_caustic_pass_.as<uint8_t>() : nullptr;
        lp.num_caustic_emitted    = num_caustic_emitted_;
        lp.caustic_gather_radius  = caustic_radius_;

        // Runtime toggle (V key) overrides the RenderConfig value
        lp.volume_enabled = (int)runtime_volume_enabled_;
        lp.photon_num_hero = d_photon_num_hero_.d_ptr ? d_photon_num_hero_.as<uint8_t>() : nullptr;

        lp.samples_per_pixel  = 1;
        lp.max_bounces        = config.max_bounces;
        lp.frame_number       = frame_number;
        lp.render_mode        = RENDER_MODE_FULL;
        lp.is_final_render    = 1;  // FINAL: full path tracing
        lp.nee_light_samples  = DEFAULT_NEE_LIGHT_SAMPLES;
        lp.nee_deep_samples   = DEFAULT_NEE_DEEP_SAMPLES;
        lp.exposure           = exposure_;

        // Adaptive buffers
        lp.lum_sum    = adaptive
            ? reinterpret_cast<float*>(d_lum_sum_.d_ptr)   : nullptr;
        lp.lum_sum2   = adaptive
            ? reinterpret_cast<float*>(d_lum_sum2_.d_ptr)  : nullptr;
        lp.active_mask = (adaptive && use_mask)
            ? reinterpret_cast<uint8_t*>(d_active_mask_.d_ptr) : nullptr;

        // Per-pixel lobe balance (Bresenham accumulator, persists across frames)
        lp.lobe_balance = reinterpret_cast<float*>(d_lobe_balance_.d_ptr);

        d_launch_params_.alloc(sizeof(LaunchParams));
        CUDA_CHECK(cudaMemcpy(d_launch_params_.d_ptr, &lp,
                               sizeof(LaunchParams), cudaMemcpyHostToDevice));
        last_launch_params_host_ = lp;

        OPTIX_CHECK(optixLaunch(
            pipeline_,
            nullptr,
            reinterpret_cast<CUdeviceptr>(d_launch_params_.d_ptr),
            sizeof(LaunchParams),
            &sbt_,
            width_, height_, 1));

        CUDA_CHECK(cudaDeviceSynchronize());
    };

    // ── Progress helper ──────────────────────────────────────────────
    auto print_progress = [&](int done, int total, int active_pixels) {
        auto t_now = std::chrono::high_resolution_clock::now();
        double elapsed_s = std::chrono::duration<double>(t_now - t_start).count();
        float pct = 100.f * done / total;
        double eta_s = (done < total)
            ? elapsed_s * (total - done) / done : 0.0;
        constexpr int BAR_W = 20;
        int filled = (int)(pct / 100.f * BAR_W);
        char bar[BAR_W + 1];
        for (int i = 0; i < BAR_W; ++i)
            bar[i] = (i < filled) ? '=' : ' ';
        if (filled < BAR_W) bar[filled] = '>';
        bar[BAR_W] = '\0';
        char line[256];
        if (active_pixels >= 0)
            snprintf(line, sizeof(line),
                     "\r[Render] [%s] %3d%%  %d/%d spp  active=%d  %.1fs  ETA %.1fs   ",
                     bar, (int)pct, done, total, active_pixels, elapsed_s, eta_s);
        else
            snprintf(line, sizeof(line),
                     "\r[Render] [%s] %3d%%  %d/%d spp  %.1fs  ETA %.1fs   ",
                     bar, (int)pct, done, total, elapsed_s, eta_s);
        std::cout << line << std::flush;
    };

    if (!adaptive) {
        // ── Non-adaptive path ────────────────────────────────────────
        // §1.2 Multi-map: re-trace photons every MULTI_MAP_SPP_GROUP
        // samples with a different RNG seed to decorrelate the photon map.
        for (int s = 0; s < total_spp; ++s) {
            if (MULTI_MAP_SPP_GROUP > 0 && (s % MULTI_MAP_SPP_GROUP) == 0) {
                int map_seed = s / MULTI_MAP_SPP_GROUP;
                std::printf("\n[Render] Re-tracing photon map (seed=%d) ...\n", map_seed);
                trace_photons(scene, config, /*grid_radius_override=*/0.f,
                              /*photon_map_seed=*/map_seed);
            }
            launch_pass(s, /*use_mask=*/false);
            print_progress(s + 1, total_spp, /*active=*/-1);
        }
    } else {
        // ── Adaptive path ─────────────────────────────────────────────
        // Phase 1: warmup — render min_spp passes uniformly
        for (int s = 0; s < min_spp; ++s) {
            if (MULTI_MAP_SPP_GROUP > 0 && (s % MULTI_MAP_SPP_GROUP) == 0) {
                int map_seed = s / MULTI_MAP_SPP_GROUP;
                std::printf("\n[Render] Re-tracing photon map (seed=%d) ...\n", map_seed);
                trace_photons(scene, config, /*grid_radius_override=*/0.f,
                              /*photon_map_seed=*/map_seed);
            }
            launch_pass(s, /*use_mask=*/false);
            print_progress(s + 1, max_spp, /*active=*/(int)total_pixels);
        }

        // Phase 2: adaptive — update mask every update_interval passes
        int active_pixels = (int)total_pixels;
        int frame = min_spp;
        const int update_interval = config.adaptive_update_interval;

        for (int s = min_spp; s < max_spp; ++s) {
            // §1.2 Multi-map: re-trace every MULTI_MAP_SPP_GROUP samples
            if (MULTI_MAP_SPP_GROUP > 0 && (s % MULTI_MAP_SPP_GROUP) == 0) {
                int map_seed = s / MULTI_MAP_SPP_GROUP;
                std::printf("\n[Render] Re-tracing photon map (seed=%d) ...\n", map_seed);
                trace_photons(scene, config, /*grid_radius_override=*/0.f,
                              /*photon_map_seed=*/map_seed);
            }
            // Recompute mask at start of adaptive phase and every N passes
            if ((s - min_spp) % update_interval == 0) {
                AdaptiveParams ap;
                ap.sample_counts  = reinterpret_cast<float*>(d_sample_counts_.d_ptr);
                ap.lum_sum        = reinterpret_cast<float*>(d_lum_sum_.d_ptr);
                ap.lum_sum2       = reinterpret_cast<float*>(d_lum_sum2_.d_ptr);
                ap.active_mask    = reinterpret_cast<uint8_t*>(d_active_mask_.d_ptr);
                ap.width          = width_;
                ap.height         = height_;
                ap.min_spp        = min_spp;
                ap.max_spp        = max_spp;
                ap.threshold      = config.adaptive_threshold;
                ap.radius         = config.adaptive_radius;
                active_pixels = adaptive_update_mask(ap);

                if (active_pixels == 0) break;  // fully converged
            }

            launch_pass(frame++, /*use_mask=*/true);
            print_progress(s + 1, max_spp, active_pixels);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(t_end - t_start).count();
    double mrays = (double)total_pixels * total_spp * config.max_bounces / total_s / 1e6;
    std::cout << "\n[Render] Done in " << std::fixed
              << std::setprecision(1) << total_s << "s"
              << "  (~" << (int)mrays << " Mray-bounces/s)\n";
}

// =====================================================================
// render_sppm() -- SPPM iterative rendering
//   For each iteration k:
//     1. Camera pass   → store visible points per pixel
//     2. Photon pass   → trace new photons, rebuild hash grid
//     3. Gather pass   → query hash grid per pixel, progressive update
// =====================================================================
void OptixRenderer::render_sppm(
    const Camera& camera, const RenderConfig& config, const Scene& scene,
    std::function<void(int, const FrameBuffer&)> iter_callback)
{
    resize(config.image_width, config.image_height);

    const int K    = config.sppm_iterations;
    const int N_p  = config.num_photons;
    const size_t pixels = (size_t)width_ * height_;

    std::cout << "[SPPM] Starting: " << width_ << "x" << height_
              << " @ " << K << " iterations, " << N_p << " photons/iter\n";

    // ── Allocate SPPM per-pixel buffers ──────────────────────────────
    d_sppm_vp_pos_x_.alloc_zero(pixels * sizeof(float));
    d_sppm_vp_pos_y_.alloc_zero(pixels * sizeof(float));
    d_sppm_vp_pos_z_.alloc_zero(pixels * sizeof(float));
    d_sppm_vp_norm_x_.alloc_zero(pixels * sizeof(float));
    d_sppm_vp_norm_y_.alloc_zero(pixels * sizeof(float));
    d_sppm_vp_norm_z_.alloc_zero(pixels * sizeof(float));
    d_sppm_vp_wo_x_.alloc_zero(pixels * sizeof(float));
    d_sppm_vp_wo_y_.alloc_zero(pixels * sizeof(float));
    d_sppm_vp_wo_z_.alloc_zero(pixels * sizeof(float));
    d_sppm_vp_mat_id_.alloc_zero(pixels * sizeof(uint32_t));
    d_sppm_vp_uv_u_.alloc_zero(pixels * sizeof(float));
    d_sppm_vp_uv_v_.alloc_zero(pixels * sizeof(float));
    d_sppm_vp_throughput_.alloc_zero(pixels * NUM_LAMBDA * sizeof(float));
    d_sppm_vp_valid_.alloc_zero(pixels * sizeof(uint8_t));
    d_sppm_N_.alloc_zero(pixels * sizeof(float));
    d_sppm_tau_.alloc_zero(pixels * NUM_LAMBDA * sizeof(float));
    d_sppm_L_direct_.alloc_zero(pixels * NUM_LAMBDA * sizeof(float));

    // Initialise per-pixel radius to sppm_initial_radius
    {
        std::vector<float> init_radius(pixels, config.sppm_initial_radius);
        d_sppm_radius_.upload(init_radius);
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < K; ++k) {
        // ── 1. Camera pass ──────────────────────────────────────────
        {
            LaunchParams lp = {};
            fill_common_params(lp,
                d_spectrum_buffer_, d_sample_counts_, d_srgb_buffer_,
                width_, height_, camera,
                d_vertices_, d_normals_, d_texcoords_,
                d_material_ids_,
                d_Kd_, d_Ks_, d_Le_, d_roughness_, d_ior_, d_mat_type_,
                d_diffuse_tex_, d_tex_atlas_, d_tex_descs_,
                (int)(d_tex_descs_.bytes / sizeof(GpuTexDesc)),
                d_photon_pos_x_, d_photon_pos_y_, d_photon_pos_z_,
                d_photon_wi_x_, d_photon_wi_y_, d_photon_wi_z_,
                d_photon_norm_x_, d_photon_norm_y_, d_photon_norm_z_,
                d_photon_lambda_, d_photon_flux_,
                d_grid_sorted_indices_, d_grid_cell_start_, d_grid_cell_end_,
                d_emissive_indices_, d_emissive_cdf_,
                num_emissive_, 0.f,
                gas_handle_,
                gather_radius_,
                num_photons_emitted_,
                d_nee_direct_buffer_, d_photon_indirect_buffer_,
                d_prof_total_, d_prof_ray_trace_, d_prof_nee_,
                d_prof_photon_gather_, d_prof_bsdf_,
                false, 0.f, 0.f, 0.f, 0, 0.f);  // volume disabled for SPPM
            fill_clearcoat_fabric_params(lp);

            // Dual-budget caustic params
            lp.photon_is_caustic_pass = d_photon_is_caustic_pass_.d_ptr
                ? d_photon_is_caustic_pass_.as<uint8_t>() : nullptr;
            lp.num_caustic_emitted    = num_caustic_emitted_;
            lp.caustic_gather_radius  = caustic_radius_;

            lp.sppm_mode            = 1;  // camera pass
            lp.photon_num_hero = d_photon_num_hero_.d_ptr ? d_photon_num_hero_.as<uint8_t>() : nullptr;
            lp.sppm_iteration       = k;
            lp.sppm_photons_per_iter = N_p;
            lp.sppm_alpha           = config.sppm_alpha;
            lp.sppm_min_radius      = config.sppm_min_radius;
            lp.max_bounces          = config.max_bounces;
            lp.nee_light_samples    = DEFAULT_NEE_LIGHT_SAMPLES;
            lp.nee_deep_samples     = DEFAULT_NEE_DEEP_SAMPLES;
            lp.is_final_render      = 1;
            lp.samples_per_pixel    = 1;
            lp.frame_number         = k;
            lp.exposure             = exposure_;

            // SPPM visible-point buffers
            lp.sppm_vp_pos_x     = d_sppm_vp_pos_x_.as<float>();
            lp.sppm_vp_pos_y     = d_sppm_vp_pos_y_.as<float>();
            lp.sppm_vp_pos_z     = d_sppm_vp_pos_z_.as<float>();
            lp.sppm_vp_norm_x    = d_sppm_vp_norm_x_.as<float>();
            lp.sppm_vp_norm_y    = d_sppm_vp_norm_y_.as<float>();
            lp.sppm_vp_norm_z    = d_sppm_vp_norm_z_.as<float>();
            lp.sppm_vp_wo_x      = d_sppm_vp_wo_x_.as<float>();
            lp.sppm_vp_wo_y      = d_sppm_vp_wo_y_.as<float>();
            lp.sppm_vp_wo_z      = d_sppm_vp_wo_z_.as<float>();
            lp.sppm_vp_mat_id    = d_sppm_vp_mat_id_.as<uint32_t>();
            lp.sppm_vp_uv_u      = d_sppm_vp_uv_u_.as<float>();
            lp.sppm_vp_uv_v      = d_sppm_vp_uv_v_.as<float>();
            lp.sppm_vp_throughput = d_sppm_vp_throughput_.as<float>();
            lp.sppm_vp_valid      = d_sppm_vp_valid_.as<uint8_t>();
            lp.sppm_radius        = d_sppm_radius_.as<float>();
            lp.sppm_N             = d_sppm_N_.as<float>();
            lp.sppm_tau           = d_sppm_tau_.as<float>();
            lp.sppm_L_direct      = d_sppm_L_direct_.as<float>();

            d_launch_params_.alloc(sizeof(LaunchParams));
            CUDA_CHECK(cudaMemcpy(d_launch_params_.d_ptr, &lp,
                                   sizeof(LaunchParams), cudaMemcpyHostToDevice));
            last_launch_params_host_ = lp;

            OPTIX_CHECK(optixLaunch(pipeline_, nullptr,
                reinterpret_cast<CUdeviceptr>(d_launch_params_.d_ptr),
                sizeof(LaunchParams), &sbt_,
                width_, height_, 1));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // ── 2. Photon pass (reuse existing infrastructure) ──────────
        // Override hash grid cell size for SPPM: cells must be >= 2 × max
        // per-pixel radius so the 3×3×3 neighbour query always covers the
        // full gather disc.  On iteration 0 the max radius is the initial
        // SPPM radius; on subsequent iterations it can only shrink.
        float max_radius = (k == 0) ? config.sppm_initial_radius
                                    : config.sppm_initial_radius; // radii only shrink
        trace_photons(scene, config, /*grid_radius_override=*/ max_radius);

        // ── 3. Gather pass ──────────────────────────────────────────
        {
            LaunchParams lp = {};
            fill_common_params(lp,
                d_spectrum_buffer_, d_sample_counts_, d_srgb_buffer_,
                width_, height_, camera,
                d_vertices_, d_normals_, d_texcoords_,
                d_material_ids_,
                d_Kd_, d_Ks_, d_Le_, d_roughness_, d_ior_, d_mat_type_,
                d_diffuse_tex_, d_tex_atlas_, d_tex_descs_,
                (int)(d_tex_descs_.bytes / sizeof(GpuTexDesc)),
                d_photon_pos_x_, d_photon_pos_y_, d_photon_pos_z_,
                d_photon_wi_x_, d_photon_wi_y_, d_photon_wi_z_,
                d_photon_norm_x_, d_photon_norm_y_, d_photon_norm_z_,
                d_photon_lambda_, d_photon_flux_,
                d_grid_sorted_indices_, d_grid_cell_start_, d_grid_cell_end_,
                d_emissive_indices_, d_emissive_cdf_,
                num_emissive_, 0.f,
                gas_handle_,
                gather_radius_,
                num_photons_emitted_,
                d_nee_direct_buffer_, d_photon_indirect_buffer_,
                d_prof_total_, d_prof_ray_trace_, d_prof_nee_,
                d_prof_photon_gather_, d_prof_bsdf_,
                false, 0.f, 0.f, 0.f, 0, 0.f);
            fill_clearcoat_fabric_params(lp);

            // Dual-budget caustic params
            lp.photon_is_caustic_pass = d_photon_is_caustic_pass_.d_ptr
                ? d_photon_is_caustic_pass_.as<uint8_t>() : nullptr;
            lp.num_caustic_emitted    = num_caustic_emitted_;
            lp.caustic_gather_radius  = caustic_radius_;

            lp.sppm_mode            = 2;  // gather pass
            lp.exposure             = exposure_;
            lp.photon_num_hero = d_photon_num_hero_.d_ptr ? d_photon_num_hero_.as<uint8_t>() : nullptr;
            lp.sppm_iteration       = k;
            lp.sppm_photons_per_iter = N_p;
            lp.sppm_alpha           = config.sppm_alpha;
            lp.sppm_min_radius      = config.sppm_min_radius;
            lp.is_final_render      = 1;
            lp.samples_per_pixel    = 1;

            lp.sppm_vp_pos_x     = d_sppm_vp_pos_x_.as<float>();
            lp.sppm_vp_pos_y     = d_sppm_vp_pos_y_.as<float>();
            lp.sppm_vp_pos_z     = d_sppm_vp_pos_z_.as<float>();
            lp.sppm_vp_norm_x    = d_sppm_vp_norm_x_.as<float>();
            lp.sppm_vp_norm_y    = d_sppm_vp_norm_y_.as<float>();
            lp.sppm_vp_norm_z    = d_sppm_vp_norm_z_.as<float>();
            lp.sppm_vp_wo_x      = d_sppm_vp_wo_x_.as<float>();
            lp.sppm_vp_wo_y      = d_sppm_vp_wo_y_.as<float>();
            lp.sppm_vp_wo_z      = d_sppm_vp_wo_z_.as<float>();
            lp.sppm_vp_mat_id    = d_sppm_vp_mat_id_.as<uint32_t>();
            lp.sppm_vp_uv_u      = d_sppm_vp_uv_u_.as<float>();
            lp.sppm_vp_uv_v      = d_sppm_vp_uv_v_.as<float>();
            lp.sppm_vp_throughput = d_sppm_vp_throughput_.as<float>();
            lp.sppm_vp_valid      = d_sppm_vp_valid_.as<uint8_t>();
            lp.sppm_radius        = d_sppm_radius_.as<float>();
            lp.sppm_N             = d_sppm_N_.as<float>();
            lp.sppm_tau           = d_sppm_tau_.as<float>();
            lp.sppm_L_direct      = d_sppm_L_direct_.as<float>();

            d_launch_params_.alloc(sizeof(LaunchParams));
            CUDA_CHECK(cudaMemcpy(d_launch_params_.d_ptr, &lp,
                                   sizeof(LaunchParams), cudaMemcpyHostToDevice));
            last_launch_params_host_ = lp;

            OPTIX_CHECK(optixLaunch(pipeline_, nullptr,
                reinterpret_cast<CUdeviceptr>(d_launch_params_.d_ptr),
                sizeof(LaunchParams), &sbt_,
                width_, height_, 1));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // ── Per-iteration callback (e.g. save progress PNG) ────────
        if (iter_callback) {
            FrameBuffer iter_fb;
            download_framebuffer(iter_fb);
            iter_callback(k, iter_fb);
        }

        // ── Progress ────────────────────────────────────────────────
        {
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            float pct = 100.f * (k + 1) / K;
            double eta = (k + 1 < K)
                ? elapsed * (K - k - 1) / (k + 1) : 0.0;
            std::printf("\r[SPPM] %3d%%  iter %d/%d  %.1fs  ETA %.1fs   ",
                        (int)pct, k + 1, K, elapsed, eta);
            std::fflush(stdout);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(t_end - t_start).count();
    std::printf("\n[SPPM] Done in %.1f s  (%d iterations, %d photons/iter)\n",
                total_s, K, N_p);
}

// =====================================================================
// print_kernel_profiling() -- download GPU timing data and print summary
// =====================================================================
void OptixRenderer::print_kernel_profiling() const {
    if (!d_prof_total_.d_ptr) {
        std::cout << "[Profiling] No kernel profiling data available.\n";
        return;
    }

    size_t pixels = (size_t)width_ * height_;
    std::vector<long long> total(pixels), ray(pixels), nee(pixels);
    std::vector<long long> pg(pixels), bsdf(pixels);

    CUDA_CHECK(cudaMemcpy(total.data(), d_prof_total_.d_ptr,
                          pixels * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ray.data(), d_prof_ray_trace_.d_ptr,
                          pixels * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(nee.data(), d_prof_nee_.d_ptr,
                          pixels * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pg.data(), d_prof_photon_gather_.d_ptr,
                          pixels * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bsdf.data(), d_prof_bsdf_.d_ptr,
                          pixels * sizeof(long long), cudaMemcpyDeviceToHost));

    // Compute statistics
    auto stats = [&](const std::vector<long long>& v, const char* label) {
        long long sum = 0, mn = LLONG_MAX, mx = 0;
        int active = 0;
        for (size_t i = 0; i < v.size(); ++i) {
            if (v[i] > 0) {
                sum += v[i];
                if (v[i] < mn) mn = v[i];
                if (v[i] > mx) mx = v[i];
                active++;
            }
        }
        double avg = active > 0 ? (double)sum / active : 0.0;
        std::printf("  %-20s  sum %12.0f  avg %10.0f  min %10lld  max %10lld  (%d px)\n",
                    label, (double)sum, avg, (active > 0 ? mn : 0LL), mx, active);
        return sum;
    };

    std::cout << "\n========================================\n";
    std::cout << "  GPU Kernel Profiling (clock64 ticks)\n";
    std::cout << "  Resolution: " << width_ << "x" << height_
              << " (" << pixels << " pixels)\n";
    std::cout << "========================================\n";

    long long sum_total = stats(total, "Total kernel");
    long long sum_ray   = stats(ray,   "Ray trace");
    long long sum_nee   = stats(nee,   "NEE (shadow)");
    long long sum_pg    = stats(pg,    "Photon gather");
    long long sum_bsdf  = stats(bsdf,  "BSDF continuation");

    long long sum_accounted = sum_ray + sum_nee + sum_pg + sum_bsdf;

    std::cout << "  ----------------------------------------\n";
    if (sum_total > 0) {
        std::printf("  %-20s  %5.1f%%\n", "Ray trace",
                    100.0 * sum_ray / sum_total);
        std::printf("  %-20s  %5.1f%%\n", "NEE (shadow)",
                    100.0 * sum_nee / sum_total);
        std::printf("  %-20s  %5.1f%%\n", "Photon gather",
                    100.0 * sum_pg / sum_total);
        std::printf("  %-20s  %5.1f%%\n", "BSDF continuation",
                    100.0 * sum_bsdf / sum_total);
        std::printf("  %-20s  %5.1f%%\n", "Overhead/other",
                    100.0 * (sum_total - sum_accounted) / sum_total);
    }
    std::cout << "========================================\n\n";
}

// =====================================================================
// trace_single_ray() -- for debug hover inspection
// =====================================================================
HitRecord OptixRenderer::trace_single_ray(float3 origin, float3 direction) const {
    HitRecord hit = {};
    hit.hit = false;
    hit.t   = DEFAULT_RAY_TMAX;

    if (host_triangles_.empty()) {
        return hit;
    }

    Ray ray;
    ray.origin = origin;
    ray.direction = normalize(direction);
    ray.tmin = OPTIX_SCENE_EPSILON;
    ray.tmax = DEFAULT_RAY_TMAX;

    for (uint32_t tri_id = 0; tri_id < (uint32_t)host_triangles_.size(); ++tri_id) {
        const Triangle& tri = host_triangles_[tri_id];
        float t, u, v;

        Ray test_ray = ray;
        test_ray.tmax = hit.t;
        if (!tri.intersect(test_ray, t, u, v)) continue;

        if (t < hit.t) {
            float alpha = 1.f - u - v;
            hit.hit = true;
            hit.t = t;
            hit.triangle_id = tri_id;
            hit.material_id = tri.material_id;
            hit.position = tri.interpolate_position(alpha, u, v);
            hit.normal = tri.geometric_normal();
            hit.shading_normal = tri.interpolate_normal(alpha, u, v);
            hit.uv = tri.interpolate_uv(alpha, u, v);
        }
    }

    return hit;
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

    std::cout << "[OptiX] Program groups created (render + photon trace + targeted)\n";
}

void OptixRenderer::create_pipeline() {
    OptixProgramGroup program_groups[] = {
        raygen_pg_, raygen_photon_pg_, raygen_targeted_pg_,
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
        program_groups, 6,
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