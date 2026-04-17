// ─────────────────────────────────────────────────────────────────────
// accel_builder.cpp – OptiX pipeline and acceleration structure builder
//
// Phase 3 implementation.  Ported from optix_setup.cpp (v4) with:
//   - Spectrum→Color3 material upload
//   - DeviceBuffer<T> typed RAII instead of raw DeviceBuffer
//   - Decomposed from monolithic OptixRenderer class
//   - CUDA texture object creation and per-material texture ID upload
// ─────────────────────────────────────────────────────────────────────
#include "accel/accel_builder.h"
#include "core/error.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <fstream>
#include <sstream>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <cmath>

// ── Helper: read PTX from file ──────────────────────────────────────
static std::string read_ptx_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open PTX file: " + filename);
    }
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

// ── OptiX log callback ──────────────────────────────────────────────
static void optix_log_callback(unsigned int level, const char* tag,
                                const char* message, void* /*cbdata*/) {
    std::cerr << "[OptiX][" << level << "][" << tag << "] " << message << "\n";
}

static void apply_default_clamp_params(LaunchParams& lp) {
    if (lp.max_bounce_contribution <= 0.f)
        lp.max_bounce_contribution = DEFAULT_MAX_BOUNCE_CONTRIBUTION;
    if (lp.max_path_throughput <= 0.f)
        lp.max_path_throughput = DEFAULT_MAX_PATH_THROUGHPUT;
    if (lp.max_nee_contribution <= 0.f)
        lp.max_nee_contribution = DEFAULT_MAX_NEE_CONTRIBUTION;
    if (lp.max_sample_luminance <= 0.f)
        lp.max_sample_luminance = DEFAULT_MAX_SAMPLE_LUMINANCE;
}

// =====================================================================
// init()
// =====================================================================
void AccelBuilder::init() {
    if (initialised_) return;

    // Force CUDA context creation
    CUDA_CHECK(cudaFree(nullptr));

    // Query GPU properties
    {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        gpu_name_       = prop.name;
        gpu_vram_total_ = prop.totalGlobalMem;
        gpu_sm_count_   = prop.multiProcessorCount;
        std::printf("[GPU] %s  |  %.0f MB VRAM  |  %d SMs\n",
                    gpu_name_.c_str(),
                    (double)gpu_vram_total_ / (1024.0 * 1024.0),
                    gpu_sm_count_);
    }

    // Initialize OptiX
    OPTIX_CHECK(optixInit());
    create_context();

    initialised_ = true;
    std::printf("[AccelBuilder] Initialized\n");
}

// =====================================================================
// build() — Full pipeline: accel + module + programs + pipeline + SBT
// =====================================================================
void AccelBuilder::build(const Scene& scene, const std::string& ptx_path) {
    if (!initialised_) {
        throw std::runtime_error("AccelBuilder::build() called before init()");
    }

    bool need_instanced = scene.has_instances();
    instanced_pipeline_ = need_instanced;

    if (need_instanced) {
        build_ias(scene);
    } else {
        build_gas(scene);
    }
    accel_.instanced = need_instanced;
    accel_.num_triangles = (int)scene.triangles.size();
    accel_.num_instances = need_instanced ? (int)scene.instances.size() : 0;

    std::string ptx = read_ptx_file(ptx_path);
    create_module(ptx);
    create_programs();
    create_pipeline();
    build_sbt();
}

// =====================================================================
// upload_geometry()
// =====================================================================
void AccelBuilder::upload_geometry(const Scene& scene) {
    size_t num_tris = scene.triangles.size();

    // Flatten triangle vertices
    std::vector<float3> vertices(num_tris * 3);
    std::vector<float3> normals(num_tris * 3);
    std::vector<float2> texcoords(num_tris * 3);
    std::vector<uint32_t> mat_ids(num_tris);

    for (size_t i = 0; i < num_tris; ++i) {
        const auto& t = scene.triangles[i];
        vertices[i * 3 + 0] = t.v0;
        vertices[i * 3 + 1] = t.v1;
        vertices[i * 3 + 2] = t.v2;
        normals[i * 3 + 0]  = t.n0;
        normals[i * 3 + 1]  = t.n1;
        normals[i * 3 + 2]  = t.n2;
        texcoords[i * 3 + 0] = t.uv0;
        texcoords[i * 3 + 1] = t.uv1;
        texcoords[i * 3 + 2] = t.uv2;
        mat_ids[i] = t.material_id;
    }

    d_vertices_.upload(vertices);
    d_normals_.upload(normals);
    d_texcoords_.upload(texcoords);
    d_material_ids_.upload(mat_ids);

    // Compute per-triangle tangents from UV gradients (Lengyel's method)
    std::vector<float3> tangents(num_tris * 3);
    for (size_t i = 0; i < num_tris; ++i) {
        const auto& t = scene.triangles[i];
        float3 e1 = t.v1 - t.v0;
        float3 e2 = t.v2 - t.v0;
        float duv1_x = t.uv1.x - t.uv0.x;
        float duv1_y = t.uv1.y - t.uv0.y;
        float duv2_x = t.uv2.x - t.uv0.x;
        float duv2_y = t.uv2.y - t.uv0.y;

        float det = duv1_x * duv2_y - duv2_x * duv1_y;
        float3 T;
        if (std::fabs(det) > 1e-8f) {
            float inv_det = 1.f / det;
            T = normalize((e1 * duv2_y - e2 * duv1_y) * inv_det);
        } else {
            // Degenerate UVs: generate arbitrary tangent orthogonal to n0
            float3 n = t.n0;
            float3 up = (std::fabs(n.y) < 0.999f) ? make_f3(0,1,0) : make_f3(1,0,0);
            T = normalize(cross(n, up));
        }
        // Store same tangent for all 3 vertices (per-triangle)
        tangents[i * 3 + 0] = T;
        tangents[i * 3 + 1] = T;
        tangents[i * 3 + 2] = T;
    }
    d_tangents_.upload(tangents);

    std::printf("[AccelBuilder] Geometry uploaded: %zu tris  (tangents computed)\n", num_tris);
}

// =====================================================================
// upload_materials()
// =====================================================================
void AccelBuilder::upload_materials(const Scene& scene) {
    size_t num_mats = scene.materials.size();

    // Flatten material arrays (RGB, 3 floats per material)
    std::vector<float> Kd(num_mats * 3);
    std::vector<float> Ks(num_mats * 3);
    std::vector<float> Le(num_mats * 3);
    std::vector<float> Tf(num_mats * 3);
    std::vector<float> roughness(num_mats);
    std::vector<float> ior(num_mats);
    std::vector<uint8_t> mat_type(num_mats);
    std::vector<float> opacity(num_mats);
    std::vector<uint8_t> mat_thin(num_mats);

    for (size_t i = 0; i < num_mats; ++i) {
        const auto& m = scene.materials[i];
        Kd[i * 3 + 0] = m.Kd.r;  Kd[i * 3 + 1] = m.Kd.g;  Kd[i * 3 + 2] = m.Kd.b;
        Ks[i * 3 + 0] = m.Ks.r;  Ks[i * 3 + 1] = m.Ks.g;  Ks[i * 3 + 2] = m.Ks.b;
        Le[i * 3 + 0] = m.Le.r;  Le[i * 3 + 1] = m.Le.g;  Le[i * 3 + 2] = m.Le.b;
        Tf[i * 3 + 0] = m.Tf.r;  Tf[i * 3 + 1] = m.Tf.g;  Tf[i * 3 + 2] = m.Tf.b;
        roughness[i]  = m.roughness;
        ior[i]        = m.ior;
        mat_type[i]   = static_cast<uint8_t>(m.type);
        opacity[i]    = m.opacity;
        mat_thin[i]   = m.pb_thin ? 1 : 0;
    }

    d_Kd_.upload(Kd);
    d_Ks_.upload(Ks);
    d_Le_.upload(Le);
    d_Tf_.upload(Tf);
    d_roughness_.upload(roughness);
    d_ior_.upload(ior);
    d_mat_type_.upload(mat_type);
    d_opacity_.upload(opacity);
    d_mat_thin_.upload(mat_thin);

    // Cauchy dispersion coefficients
    // For Glass/Translucent materials without explicit Cauchy params,
    // auto-compute from IOR assuming typical crown glass dispersion:
    //   B = 4200 nm², A = IOR - B/589² (so IOR matches at sodium D-line)
    constexpr float DEFAULT_CAUCHY_B = 4200.f;
    constexpr float LAMBDA_D_SQ = 589.f * 589.f;   // sodium D-line
    std::vector<float> cauchy_A(num_mats);
    std::vector<float> cauchy_B(num_mats);
    for (size_t i = 0; i < num_mats; ++i) {
        const auto& m = scene.materials[i];
        bool is_glass = (m.type == MaterialType::Glass || m.type == MaterialType::Translucent);
        if (m.dispersion) {
            cauchy_A[i] = m.cauchy_A;
            cauchy_B[i] = m.cauchy_B;
        } else if (is_glass) {
            cauchy_B[i] = DEFAULT_CAUCHY_B;
            cauchy_A[i] = m.ior - DEFAULT_CAUCHY_B / LAMBDA_D_SQ;
        } else {
            cauchy_A[i] = 0.f;
            cauchy_B[i] = 0.f;
        }
    }
    d_cauchy_A_.upload(cauchy_A);
    d_cauchy_B_.upload(cauchy_B);

    // Per-material texture ID arrays
    std::vector<int> diffuse_tex_ids(num_mats);
    std::vector<int> specular_tex_ids(num_mats);
    std::vector<int> emission_tex_ids(num_mats);
    std::vector<int> bump_tex_ids(num_mats);
    std::vector<int> normal_tex_ids(num_mats);
    std::vector<int> alpha_tex_ids(num_mats);
    std::vector<int> displacement_tex_ids(num_mats);
    std::vector<float> displacement_scales(num_mats);

    for (size_t i = 0; i < num_mats; ++i) {
        const auto& m = scene.materials[i];
        diffuse_tex_ids[i]     = m.diffuse_tex;
        specular_tex_ids[i]    = m.specular_tex;
        emission_tex_ids[i]    = m.emission_tex;
        bump_tex_ids[i]        = m.bump_tex;
        normal_tex_ids[i]      = m.normal_tex;
        alpha_tex_ids[i]       = m.alpha_tex;
        displacement_tex_ids[i] = m.displacement_tex;
        displacement_scales[i]  = m.displacement_scale;
    }

    d_diffuse_tex_.upload(diffuse_tex_ids);
    d_specular_tex_.upload(specular_tex_ids);
    d_emission_tex_.upload(emission_tex_ids);
    d_bump_tex_.upload(bump_tex_ids);
    d_normal_tex_.upload(normal_tex_ids);
    d_alpha_tex_.upload(alpha_tex_ids);
    d_displacement_tex_.upload(displacement_tex_ids);
    d_displacement_scale_.upload(displacement_scales);

    // Extended PBR: clearcoat + sheen
    std::vector<float> clearcoat_weight(num_mats);
    std::vector<float> clearcoat_roughness(num_mats);
    std::vector<float> sheen(num_mats);
    std::vector<float> sheen_tint(num_mats);
    for (size_t i = 0; i < num_mats; ++i) {
        const auto& m = scene.materials[i];
        clearcoat_weight[i]    = m.pb_clearcoat;
        clearcoat_roughness[i] = (m.pb_clearcoat_roughness >= 0.f)
                                     ? m.pb_clearcoat_roughness : m.roughness;
        sheen[i]     = m.pb_sheen;
        sheen_tint[i] = m.pb_sheen_tint;
    }
    d_clearcoat_weight_.upload(clearcoat_weight);
    d_clearcoat_roughness_.upload(clearcoat_roughness);
    d_sheen_.upload(sheen);
    d_sheen_tint_.upload(sheen_tint);

    std::printf("[AccelBuilder] Materials uploaded: %zu\n", num_mats);
}

// =====================================================================
// upload_textures() — Create CUDA texture objects from Scene::textures
// =====================================================================
void AccelBuilder::upload_textures(const Scene& scene) {
    destroy_textures();

    size_t num_tex = scene.textures.size();
    if (num_tex == 0) {
        std::printf("[AccelBuilder] No textures to upload\n");
        return;
    }

    tex_arrays_.resize(num_tex, nullptr);
    tex_objects_.resize(num_tex, 0);

    for (size_t i = 0; i < num_tex; ++i) {
        const auto& tex = scene.textures[i];
        if (tex.width == 0 || tex.height == 0) continue;

        // Allocate CUDA array (float4 per texel)
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
        CUDA_CHECK(cudaMallocArray(&tex_arrays_[i], &desc, tex.width, tex.height));

        // Pack RGBA data as float4 (texture.data is already RGBA float)
        size_t num_texels = (size_t)tex.width * tex.height;
        std::vector<float4> texels(num_texels);
        for (size_t j = 0; j < num_texels; ++j) {
            size_t base = j * tex.channels;
            texels[j] = make_float4(
                tex.data[base + 0],
                tex.data[base + 1],
                tex.data[base + 2],
                tex.channels >= 4 ? tex.data[base + 3] : 1.f);
        }

        CUDA_CHECK(cudaMemcpy2DToArray(
            tex_arrays_[i], 0, 0,
            texels.data(), tex.width * sizeof(float4),
            tex.width * sizeof(float4), tex.height,
            cudaMemcpyHostToDevice));

        // Create texture object
        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = tex_arrays_[i];

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0]   = cudaAddressModeWrap;
        tex_desc.addressMode[1]   = cudaAddressModeWrap;
        tex_desc.filterMode       = cudaFilterModeLinear;
        tex_desc.readMode         = cudaReadModeElementType;
        tex_desc.normalizedCoords = 1;

        CUDA_CHECK(cudaCreateTextureObject(&tex_objects_[i], &res_desc, &tex_desc, nullptr));
    }

    d_tex_objects_.upload(tex_objects_);
    std::printf("[AccelBuilder] Textures uploaded: %zu  (CUDA texture objects)\n", num_tex);
}

// =====================================================================
// destroy_textures() — Clean up CUDA texture objects and arrays
// =====================================================================
void AccelBuilder::destroy_textures() {
    for (auto& obj : tex_objects_) {
        if (obj) cudaDestroyTextureObject(obj);
    }
    tex_objects_.clear();

    for (auto& arr : tex_arrays_) {
        if (arr) cudaFreeArray(arr);
    }
    tex_arrays_.clear();
}

// =====================================================================
// fill_geometry_params()
// =====================================================================
void AccelBuilder::fill_geometry_params(LaunchParams& lp) {
    lp.vertices     = d_vertices_.data();
    lp.normals      = d_normals_.data();
    lp.tangents     = d_tangents_.empty() ? nullptr : d_tangents_.data();
    lp.texcoords    = d_texcoords_.data();
    lp.material_ids = d_material_ids_.data();
    lp.num_triangles = accel_.num_triangles;
    lp.traversable   = accel_.handle;
    lp.has_instances = accel_.instanced ? 1 : 0;
}

// =====================================================================
// fill_material_params()
// =====================================================================
void AccelBuilder::fill_material_params(LaunchParams& lp) {
    lp.Kd           = d_Kd_.data();
    lp.Ks           = d_Ks_.data();
    lp.Le           = d_Le_.data();
    lp.Tf           = d_Tf_.data();
    lp.roughness    = d_roughness_.data();
    lp.ior          = d_ior_.data();
    lp.mat_type     = d_mat_type_.data();
    lp.opacity      = d_opacity_.data();
    lp.mat_thin     = d_mat_thin_.data();
    lp.cauchy_A      = d_cauchy_A_.empty()  ? nullptr : d_cauchy_A_.data();
    lp.cauchy_B      = d_cauchy_B_.empty()  ? nullptr : d_cauchy_B_.data();
    lp.num_materials = (int)d_mat_type_.size();

    // Per-material texture IDs
    lp.diffuse_tex      = d_diffuse_tex_.empty()  ? nullptr : d_diffuse_tex_.data();
    lp.specular_tex     = d_specular_tex_.empty()  ? nullptr : d_specular_tex_.data();
    lp.emission_tex     = d_emission_tex_.empty()  ? nullptr : d_emission_tex_.data();
    lp.bump_tex         = d_bump_tex_.empty()      ? nullptr : d_bump_tex_.data();
    lp.normal_tex       = d_normal_tex_.empty()    ? nullptr : d_normal_tex_.data();
    lp.alpha_tex        = d_alpha_tex_.empty()     ? nullptr : d_alpha_tex_.data();
    lp.displacement_tex   = d_displacement_tex_.empty()   ? nullptr : d_displacement_tex_.data();
    lp.displacement_scale = d_displacement_scale_.empty() ? nullptr : d_displacement_scale_.data();

    // CUDA texture objects
    lp.textures      = d_tex_objects_.empty() ? nullptr : d_tex_objects_.data();
    lp.num_textures  = (int)tex_objects_.size();

    // Extended PBR: clearcoat + sheen
    lp.clearcoat_weight    = d_clearcoat_weight_.empty()    ? nullptr : d_clearcoat_weight_.data();
    lp.clearcoat_roughness = d_clearcoat_roughness_.empty() ? nullptr : d_clearcoat_roughness_.data();
    lp.sheen               = d_sheen_.empty()               ? nullptr : d_sheen_.data();
    lp.sheen_tint          = d_sheen_tint_.empty()          ? nullptr : d_sheen_tint_.data();
}

// =====================================================================
// launch_test_normals()
// =====================================================================
void AccelBuilder::launch_test_normals(
    int width, int height,
    float3 cam_pos, float3 cam_u, float3 cam_v, float3 cam_w,
    std::vector<float>& color_out)
{
    // Allocate output buffers (SoA)
    size_t num_pixels = (size_t)width * height;
    d_test_color_r_.ensure_alloc(num_pixels);
    d_test_color_g_.ensure_alloc(num_pixels);
    d_test_color_b_.ensure_alloc(num_pixels);
    d_test_color_r_.zero();
    d_test_color_g_.zero();
    d_test_color_b_.zero();
    d_sample_counts_.ensure_alloc(num_pixels);
    d_sample_counts_.zero();

    // Fill launch params
    LaunchParams lp = {};
    fill_geometry_params(lp);
    fill_material_params(lp);

    lp.color_r        = d_test_color_r_.data();
    lp.color_g        = d_test_color_g_.data();
    lp.color_b        = d_test_color_b_.data();
    lp.sample_counts  = d_sample_counts_.data();
    lp.srgb_buffer    = nullptr;
    lp.albedo_buffer  = nullptr;
    lp.normal_buffer  = nullptr;
    lp.width          = width;
    lp.height         = height;
    lp.cam_pos        = cam_pos;
    lp.cam_u          = cam_u;
    lp.cam_v          = cam_v;
    lp.cam_w          = cam_w;
    lp.samples_per_pixel = 1;
    lp.frame_number   = 0;

    // Null out optional fields
    fill_material_params(lp);
    lp.mat_medium_id  = nullptr;
    lp.num_media      = 0;
    lp.emissive_tri_indices = nullptr;
    lp.emissive_cdf   = nullptr;
    lp.emissive_local_idx = nullptr;
    lp.num_emissive   = 0;
    lp.lum_sum        = nullptr;
    lp.lum_sum2       = nullptr;
    lp.active_mask    = nullptr;

    // Upload and launch
    d_launch_params_.upload(&lp, 1);

    OPTIX_CHECK(optixLaunch(
        pipeline_,
        nullptr,  // CUDA stream
        reinterpret_cast<CUdeviceptr>(d_launch_params_.data()),
        sizeof(LaunchParams),
        &sbt_,
        width, height, 1));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download result (SoA → interleaved AoS for callers)
    color_out.resize(num_pixels * 3);
    std::vector<float> r(num_pixels), g(num_pixels), b(num_pixels);
    d_test_color_r_.download(r.data(), num_pixels);
    d_test_color_g_.download(g.data(), num_pixels);
    d_test_color_b_.download(b.data(), num_pixels);
    for (size_t i = 0; i < num_pixels; ++i) {
        color_out[i * 3 + 0] = r[i];
        color_out[i * 3 + 1] = g[i];
        color_out[i * 3 + 2] = b[i];
    }
}

// =====================================================================
// launch_test_nee()
// =====================================================================
void AccelBuilder::launch_test_nee(
    int width, int height,
    float3 cam_pos, float3 cam_u, float3 cam_v, float3 cam_w,
    const LaunchParams& extra_params,
    std::vector<float>& color_out)
{
    size_t num_pixels = (size_t)width * height;
    d_test_color_r_.ensure_alloc(num_pixels);
    d_test_color_g_.ensure_alloc(num_pixels);
    d_test_color_b_.ensure_alloc(num_pixels);
    d_test_color_r_.zero();
    d_test_color_g_.zero();
    d_test_color_b_.zero();
    d_sample_counts_.ensure_alloc(num_pixels);
    d_sample_counts_.zero();

    // Start from extra_params (already has emissive data)
    LaunchParams lp = extra_params;

    // Fill geometry and material from our owned buffers
    fill_geometry_params(lp);
    fill_material_params(lp);

    lp.color_r        = d_test_color_r_.data();
    lp.color_g        = d_test_color_g_.data();
    lp.color_b        = d_test_color_b_.data();
    lp.sample_counts  = d_sample_counts_.data();
    lp.srgb_buffer    = nullptr;
    lp.albedo_buffer  = nullptr;
    lp.normal_buffer  = nullptr;
    lp.width          = width;
    lp.height         = height;
    lp.cam_pos        = cam_pos;
    lp.cam_u          = cam_u;
    lp.cam_v          = cam_v;
    lp.cam_w          = cam_w;
    lp.samples_per_pixel = 1;
    lp.frame_number   = 0;

    // Null out fields not set by caller
    fill_material_params(lp);
    lp.mat_medium_id  = nullptr;
    lp.num_media      = 0;
    lp.lum_sum        = nullptr;
    lp.lum_sum2       = nullptr;
    lp.active_mask    = nullptr;

    d_launch_params_.upload(&lp, 1);

    OPTIX_CHECK(optixLaunch(
        pipeline_,
        nullptr,
        reinterpret_cast<CUdeviceptr>(d_launch_params_.data()),
        sizeof(LaunchParams),
        &sbt_nee_,
        width, height, 1));
    CUDA_CHECK(cudaDeviceSynchronize());

    color_out.resize(num_pixels * 3);
    std::vector<float> r(num_pixels), g(num_pixels), b(num_pixels);
    d_test_color_r_.download(r.data(), num_pixels);
    d_test_color_g_.download(g.data(), num_pixels);
    d_test_color_b_.download(b.data(), num_pixels);
    for (size_t i = 0; i < num_pixels; ++i) {
        color_out[i * 3 + 0] = r[i];
        color_out[i * 3 + 1] = g[i];
        color_out[i * 3 + 2] = b[i];
    }
}

// =====================================================================
// launch_render()
// =====================================================================
void AccelBuilder::launch_render(
    int width, int height,
    float3 cam_pos, float3 cam_u, float3 cam_v, float3 cam_w,
    const LaunchParams& extra_params,
    std::vector<float>& color_out)
{
    size_t num_pixels = (size_t)width * height;
    d_test_color_r_.ensure_alloc(num_pixels);
    d_test_color_g_.ensure_alloc(num_pixels);
    d_test_color_b_.ensure_alloc(num_pixels);
    d_test_color_r_.zero();
    d_test_color_g_.zero();
    d_test_color_b_.zero();
    d_sample_counts_.ensure_alloc(num_pixels);
    d_sample_counts_.zero();

    LaunchParams lp = extra_params;

    fill_geometry_params(lp);
    fill_material_params(lp);

    lp.color_r        = d_test_color_r_.data();
    lp.color_g        = d_test_color_g_.data();
    lp.color_b        = d_test_color_b_.data();
    lp.sample_counts  = d_sample_counts_.data();
    lp.srgb_buffer    = nullptr;
    lp.albedo_buffer  = nullptr;
    lp.normal_buffer  = nullptr;
    lp.width          = width;
    lp.height         = height;
    lp.cam_pos        = cam_pos;
    lp.cam_u          = cam_u;
    lp.cam_v          = cam_v;
    lp.cam_w          = cam_w;
    lp.cam_lens_radius = 0.f;
    lp.cam_focus_dist  = 1.f;

    // Use render params from extra_params (already set by caller)
    // But null out unused fields
    fill_material_params(lp);
    lp.mat_medium_id  = nullptr;
    lp.num_media      = 0;
    lp.lum_sum        = nullptr;
    lp.lum_sum2       = nullptr;
    lp.active_mask    = nullptr;
    apply_default_clamp_params(lp);

    d_launch_params_.upload(&lp, 1);

    OPTIX_CHECK(optixLaunch(
        pipeline_,
        nullptr,
        reinterpret_cast<CUdeviceptr>(d_launch_params_.data()),
        sizeof(LaunchParams),
        &sbt_render_,
        width, height, 1));
    CUDA_CHECK(cudaDeviceSynchronize());

    color_out.resize(num_pixels * 3);
    std::vector<float> r(num_pixels), g(num_pixels), b(num_pixels);
    d_test_color_r_.download(r.data(), num_pixels);
    d_test_color_g_.download(g.data(), num_pixels);
    d_test_color_b_.download(b.data(), num_pixels);
    for (size_t i = 0; i < num_pixels; ++i) {
        color_out[i * 3 + 0] = r[i];
        color_out[i * 3 + 1] = g[i];
        color_out[i * 3 + 2] = b[i];
    }
}

// =====================================================================
// launch_progressive()
// Uploads lp unchanged and launches __raygen__render.
// No buffer zeroing/allocation — caller owns the device buffers.
// Used by RenderSession for multi-frame progressive accumulation.
// =====================================================================
void AccelBuilder::launch_progressive(int width, int height,
                                       const LaunchParams& lp)
{
    LaunchParams launch_params = lp;
    apply_default_clamp_params(launch_params);
    d_launch_params_.upload(&launch_params, 1);

    OPTIX_CHECK(optixLaunch(
        pipeline_,
        nullptr,
        reinterpret_cast<CUdeviceptr>(d_launch_params_.data()),
        sizeof(LaunchParams),
        &sbt_render_,
        width, height, 1));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// =====================================================================
// launch_caustic()
// Launches __raygen__caustic: 1 thread per photon, 1D grid.
// =====================================================================
void AccelBuilder::launch_caustic(int num_photons, const LaunchParams& lp)
{
    LaunchParams launch_params = lp;
    apply_default_clamp_params(launch_params);
    d_launch_params_.upload(&launch_params, 1);

    OPTIX_CHECK(optixLaunch(
        pipeline_,
        nullptr,
        reinterpret_cast<CUdeviceptr>(d_launch_params_.data()),
        sizeof(LaunchParams),
        &sbt_caustic_,
        num_photons, 1, 1));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// =====================================================================
// Private: create_context()
// =====================================================================
void AccelBuilder::create_context() {
    CUcontext cu_ctx = nullptr;
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = optix_log_callback;
    options.logCallbackLevel   = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context_));
    std::printf("[AccelBuilder] OptiX context created\n");
}

// =====================================================================
// Private: create_module()
// =====================================================================
void AccelBuilder::create_module(const std::string& ptx) {
    OptixModuleCompileOptions module_opts = {};
    module_opts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifndef NDEBUG
    module_opts.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
    module_opts.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

    OptixPipelineCompileOptions pipeline_opts = {};
    pipeline_opts.usesMotionBlur                 = false;
    pipeline_opts.traversableGraphFlags          = instanced_pipeline_
        ? OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING
        : OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_opts.numPayloadValues               = accel::NUM_PAYLOAD_VALUES;
    pipeline_opts.numAttributeValues             = accel::NUM_ATTRIBUTE_VALUES;
    pipeline_opts.exceptionFlags                 = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_opts.pipelineLaunchParamsVariableName = "params";

    char log[2048];
    size_t log_size = sizeof(log);

    OPTIX_CHECK_LOG(optixModuleCreate(
        context_,
        &module_opts,
        &pipeline_opts,
        ptx.c_str(), ptx.size(),
        log, &log_size,
        &module_), log, log_size);

    std::printf("[AccelBuilder] Module created (%zu bytes PTX)\n", ptx.size());
}

// =====================================================================
// Private: create_programs()
// =====================================================================
void AccelBuilder::create_programs() {
    OptixProgramGroupOptions pg_options = {};
    char log[2048];
    size_t log_size;

    // Raygen (test_normals)
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = module_;
        desc.raygen.entryFunctionName = "__raygen__test_normals";
        log_size = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context_, &desc, 1, &pg_options, log, &log_size, &raygen_test_pg_),
            log, log_size);
    }

    // Raygen (test_nee)
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = module_;
        desc.raygen.entryFunctionName = "__raygen__test_nee";
        log_size = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context_, &desc, 1, &pg_options, log, &log_size, &raygen_nee_pg_),
            log, log_size);
    }

    // Raygen (render)
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = module_;
        desc.raygen.entryFunctionName = "__raygen__render";
        log_size = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context_, &desc, 1, &pg_options, log, &log_size, &raygen_render_pg_),
            log, log_size);
    }

    // Raygen (caustic light tracing)
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = module_;
        desc.raygen.entryFunctionName = "__raygen__caustic";
        log_size = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context_, &desc, 1, &pg_options, log, &log_size, &raygen_caustic_pg_),
            log, log_size);
    }

    // Miss (radiance)
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module = module_;
        desc.miss.entryFunctionName = "__miss__radiance";
        log_size = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context_, &desc, 1, &pg_options, log, &log_size, &miss_pg_),
            log, log_size);
    }

    // Miss (shadow)
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module = module_;
        desc.miss.entryFunctionName = "__miss__shadow";
        log_size = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context_, &desc, 1, &pg_options, log, &log_size, &miss_shadow_pg_),
            log, log_size);
    }

    // Hit group (radiance): closesthit + anyhit
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH            = module_;
        desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        desc.hitgroup.moduleAH            = module_;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
        log_size = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context_, &desc, 1, &pg_options, log, &log_size, &hitgroup_pg_),
            log, log_size);
    }

    // Hit group (shadow): anyhit only (closesthit disabled by ray flag)
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleAH            = module_;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__shadow";
        // CH disabled — shadow rays use OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT
        // but we still set a closesthit stub for completeness
        desc.hitgroup.moduleCH            = module_;
        desc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
        log_size = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context_, &desc, 1, &pg_options, log, &log_size, &hitgroup_shadow_pg_),
            log, log_size);
    }

    std::printf("[AccelBuilder] Program groups created\n");
}

// =====================================================================
// Private: create_pipeline()
// =====================================================================
void AccelBuilder::create_pipeline() {
    OptixProgramGroup program_groups[] = {
        raygen_test_pg_,
        raygen_nee_pg_,
        raygen_render_pg_,
        raygen_caustic_pg_,
        miss_pg_, miss_shadow_pg_,
        hitgroup_pg_, hitgroup_shadow_pg_
    };

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = accel::MAX_TRACE_DEPTH;

    OptixPipelineCompileOptions pipeline_opts = {};
    pipeline_opts.usesMotionBlur                 = false;
    pipeline_opts.traversableGraphFlags          = instanced_pipeline_
        ? OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING
        : OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_opts.numPayloadValues               = accel::NUM_PAYLOAD_VALUES;
    pipeline_opts.numAttributeValues             = accel::NUM_ATTRIBUTE_VALUES;
    pipeline_opts.exceptionFlags                 = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_opts.pipelineLaunchParamsVariableName = "params";

    char log[2048];
    size_t log_size = sizeof(log);

    OPTIX_CHECK_LOG(optixPipelineCreate(
        context_,
        &pipeline_opts,
        &link_options,
        program_groups, sizeof(program_groups) / sizeof(program_groups[0]),
        log, &log_size,
        &pipeline_), log, log_size);

    // Stack sizes: 1 for single GAS, 2 for IAS
    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline_,
        accel::STACK_SIZE,
        accel::STACK_SIZE,
        accel::STACK_SIZE,
        instanced_pipeline_ ? 2 : 1));

    std::printf("[AccelBuilder] Pipeline created\n");
}

// =====================================================================
// Private: build_gas() — Single GAS (no instancing)
// =====================================================================
void AccelBuilder::build_gas(const Scene& scene) {
    size_t num_tris = scene.triangles.size();

    // Flatten vertices for GAS build
    std::vector<float3> vertices(num_tris * 3);
    for (size_t i = 0; i < num_tris; ++i) {
        vertices[i * 3 + 0] = scene.triangles[i].v0;
        vertices[i * 3 + 1] = scene.triangles[i].v1;
        vertices[i * 3 + 2] = scene.triangles[i].v2;
    }

    d_vertices_raw_.upload(reinterpret_cast<const uint8_t*>(vertices.data()),
                           vertices.size() * sizeof(float3));

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    auto& tri = build_input.triangleArray;

    CUdeviceptr vertex_ptr = reinterpret_cast<CUdeviceptr>(d_vertices_raw_.data());
    tri.vertexBuffers       = &vertex_ptr;
    tri.numVertices         = (unsigned int)(num_tris * 3);
    tri.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    tri.vertexStrideInBytes = sizeof(float3);

    unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;
    tri.flags          = &flags;
    tri.numSbtRecords  = 1;

    OptixAccelBuildOptions accel_opts = {};
    accel_opts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                            OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_opts.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes buf_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context_, &accel_opts, &build_input, 1, &buf_sizes));

    DeviceBuffer<uint8_t> temp_buf;
    temp_buf.alloc(buf_sizes.tempSizeInBytes);
    DeviceBuffer<uint8_t> output_buf;
    output_buf.alloc(buf_sizes.outputSizeInBytes);

    // Request compacted size
    DeviceBuffer<size_t> compacted_size_buf;
    compacted_size_buf.alloc(1);
    OptixAccelEmitDesc emit_desc;
    emit_desc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_desc.result = reinterpret_cast<CUdeviceptr>(compacted_size_buf.data());

    OPTIX_CHECK(optixAccelBuild(
        context_, nullptr,
        &accel_opts,
        &build_input, 1,
        reinterpret_cast<CUdeviceptr>(temp_buf.data()), temp_buf.bytes(),
        reinterpret_cast<CUdeviceptr>(output_buf.data()), output_buf.bytes(),
        &accel_.handle,
        &emit_desc, 1));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compact
    size_t compacted_size;
    CUDA_CHECK(cudaMemcpy(&compacted_size, compacted_size_buf.data(),
                           sizeof(size_t), cudaMemcpyDeviceToHost));
    gas_buffer_.alloc(compacted_size);
    OPTIX_CHECK(optixAccelCompact(
        context_, nullptr, accel_.handle,
        reinterpret_cast<CUdeviceptr>(gas_buffer_.data()), compacted_size,
        &accel_.handle));
    CUDA_CHECK(cudaDeviceSynchronize());

    accel_.compacted_bytes = compacted_size;
    std::printf("[AccelBuilder] GAS built: %zu tris  compacted=%.1f KB\n",
                num_tris, (double)compacted_size / 1024.0);
}

// =====================================================================
// Private: build_ias() — Per-mesh GAS + top-level IAS
// =====================================================================
void AccelBuilder::build_ias(const Scene& scene) {
    size_t num_tris = scene.triangles.size();

    // Upload global vertex buffer
    std::vector<float3> vertices(num_tris * 3);
    for (size_t i = 0; i < num_tris; ++i) {
        vertices[i * 3 + 0] = scene.triangles[i].v0;
        vertices[i * 3 + 1] = scene.triangles[i].v1;
        vertices[i * 3 + 2] = scene.triangles[i].v2;
    }
    d_vertices_raw_.upload(reinterpret_cast<const uint8_t*>(vertices.data()),
                           vertices.size() * sizeof(float3));
    CUdeviceptr base_vertex_ptr = reinterpret_cast<CUdeviceptr>(d_vertices_raw_.data());

    // Build one GAS per mesh
    size_t num_meshes = scene.meshes.size();
    ias_gas_handles_.resize(num_meshes);
    ias_gas_buffers_.resize(num_meshes);

    OptixAccelBuildOptions accel_opts = {};
    accel_opts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                            OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_opts.operation  = OPTIX_BUILD_OPERATION_BUILD;

    size_t total_gas_bytes = 0;

    for (size_t mi = 0; mi < num_meshes; ++mi) {
        const auto& mesh = scene.meshes[mi];

        OptixBuildInput build_input = {};
        build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        auto& tri = build_input.triangleArray;

        CUdeviceptr mesh_vertex_ptr = base_vertex_ptr +
            (size_t)mesh.tri_offset * 3 * sizeof(float3);
        tri.vertexBuffers       = &mesh_vertex_ptr;
        tri.numVertices         = mesh.tri_count * 3;
        tri.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
        tri.vertexStrideInBytes = sizeof(float3);
        tri.primitiveIndexOffset = mesh.tri_offset;

        unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;
        tri.flags          = &flags;
        tri.numSbtRecords  = 1;

        OptixAccelBufferSizes buf_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            context_, &accel_opts, &build_input, 1, &buf_sizes));

        DeviceBuffer<uint8_t> temp_buf, out_buf;
        temp_buf.alloc(buf_sizes.tempSizeInBytes);
        out_buf.alloc(buf_sizes.outputSizeInBytes);

        DeviceBuffer<size_t> compact_size_buf;
        compact_size_buf.alloc(1);
        OptixAccelEmitDesc emit_desc;
        emit_desc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_desc.result = reinterpret_cast<CUdeviceptr>(compact_size_buf.data());

        OPTIX_CHECK(optixAccelBuild(
            context_, nullptr,
            &accel_opts, &build_input, 1,
            reinterpret_cast<CUdeviceptr>(temp_buf.data()), temp_buf.bytes(),
            reinterpret_cast<CUdeviceptr>(out_buf.data()), out_buf.bytes(),
            &ias_gas_handles_[mi],
            &emit_desc, 1));
        CUDA_CHECK(cudaDeviceSynchronize());

        size_t compacted_size;
        CUDA_CHECK(cudaMemcpy(&compacted_size, compact_size_buf.data(),
                               sizeof(size_t), cudaMemcpyDeviceToHost));

        ias_gas_buffers_[mi].alloc(compacted_size);
        OPTIX_CHECK(optixAccelCompact(
            context_, nullptr, ias_gas_handles_[mi],
            reinterpret_cast<CUdeviceptr>(ias_gas_buffers_[mi].data()), compacted_size,
            &ias_gas_handles_[mi]));
        CUDA_CHECK(cudaDeviceSynchronize());
        total_gas_bytes += compacted_size;
    }

    std::printf("[AccelBuilder] IAS: %zu per-mesh GAS  total=%.1f MB\n",
                num_meshes, (double)total_gas_bytes / (1024.0 * 1024.0));

    // Build OptixInstance array
    size_t num_instances = scene.instances.size();
    std::vector<OptixInstance> optix_instances(num_instances);
    std::memset(optix_instances.data(), 0, num_instances * sizeof(OptixInstance));

    for (size_t i = 0; i < num_instances; ++i) {
        const auto& desc = scene.instances[i];
        auto& oi = optix_instances[i];
        std::memcpy(oi.transform, desc.transform, sizeof(float) * 12);
        oi.instanceId        = (unsigned int)i;
        oi.sbtOffset         = 0;
        oi.visibilityMask    = 255;
        oi.flags             = OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
        oi.traversableHandle = ias_gas_handles_[desc.mesh_id];
    }

    d_ias_instances_.upload(optix_instances);

    // Build top-level IAS
    OptixBuildInput ias_input = {};
    ias_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    ias_input.instanceArray.instances    = reinterpret_cast<CUdeviceptr>(d_ias_instances_.data());
    ias_input.instanceArray.numInstances = (unsigned int)num_instances;

    OptixAccelBuildOptions ias_opts = {};
    ias_opts.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    ias_opts.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buf_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context_, &ias_opts, &ias_input, 1, &ias_buf_sizes));

    DeviceBuffer<uint8_t> ias_temp;
    ias_temp.alloc(ias_buf_sizes.tempSizeInBytes);
    ias_buffer_.alloc(ias_buf_sizes.outputSizeInBytes);

    OptixTraversableHandle ias_handle;
    OPTIX_CHECK(optixAccelBuild(
        context_, nullptr,
        &ias_opts, &ias_input, 1,
        reinterpret_cast<CUdeviceptr>(ias_temp.data()), ias_temp.bytes(),
        reinterpret_cast<CUdeviceptr>(ias_buffer_.data()), ias_buffer_.bytes(),
        &ias_handle,
        nullptr, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    accel_.handle = ias_handle;
    accel_.compacted_bytes = total_gas_bytes + ias_buf_sizes.outputSizeInBytes;

    std::printf("[AccelBuilder] IAS built: %zu instances  IAS=%.1f KB\n",
                num_instances, (double)ias_buf_sizes.outputSizeInBytes / 1024.0);
}

// =====================================================================
// Private: build_sbt()
// =====================================================================
void AccelBuilder::build_sbt() {
    // Raygen record
    {
        RayGenRecord rec = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_test_pg_, &rec));
        d_raygen_record_.upload(&rec, 1);
        sbt_.raygenRecord = reinterpret_cast<CUdeviceptr>(d_raygen_record_.data());
    }

    // Miss records (radiance + shadow)
    {
        std::vector<MissRecord> miss_records(accel::NUM_RAY_TYPES);

        OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg_, &miss_records[0]));
        miss_records[0].data = {0.f, 0.f, 0.f}; // black background

        OPTIX_CHECK(optixSbtRecordPackHeader(miss_shadow_pg_, &miss_records[1]));
        miss_records[1].data = {0.f, 0.f, 0.f};

        d_miss_records_.upload(miss_records);
        sbt_.missRecordBase          = reinterpret_cast<CUdeviceptr>(d_miss_records_.data());
        sbt_.missRecordStrideInBytes = sizeof(MissRecord);
        sbt_.missRecordCount         = accel::NUM_RAY_TYPES;
    }

    // Hit group records (radiance + shadow)
    {
        std::vector<HitGroupRecord> hg_records(accel::NUM_RAY_TYPES);

        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pg_, &hg_records[0]));
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_shadow_pg_, &hg_records[1]));

        d_hitgroup_records_.upload(hg_records);
        sbt_.hitgroupRecordBase          = reinterpret_cast<CUdeviceptr>(d_hitgroup_records_.data());
        sbt_.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
        sbt_.hitgroupRecordCount         = accel::NUM_RAY_TYPES;
    }

    std::printf("[AccelBuilder] SBT built (%d ray types)\n", accel::NUM_RAY_TYPES);

    // NEE raygen SBT (same miss + hitgroup, different raygen)
    {
        RayGenRecord rec = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_nee_pg_, &rec));
        d_raygen_nee_record_.upload(&rec, 1);

        sbt_nee_ = sbt_;  // copy miss + hitgroup from normal SBT
        sbt_nee_.raygenRecord = reinterpret_cast<CUdeviceptr>(d_raygen_nee_record_.data());
    }

    // Render raygen SBT
    {
        RayGenRecord rec = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_render_pg_, &rec));
        d_raygen_render_record_.upload(&rec, 1);

        sbt_render_ = sbt_;  // copy miss + hitgroup
        sbt_render_.raygenRecord = reinterpret_cast<CUdeviceptr>(d_raygen_render_record_.data());
    }

    // Caustic raygen SBT
    {
        RayGenRecord rec = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_caustic_pg_, &rec));
        d_raygen_caustic_record_.upload(&rec, 1);

        sbt_caustic_ = sbt_;  // same miss + hitgroup as render
        sbt_caustic_.raygenRecord = reinterpret_cast<CUdeviceptr>(d_raygen_caustic_record_.data());
    }
}

// =====================================================================
// cleanup()
// =====================================================================
void AccelBuilder::cleanup() {
    destroy_textures();

    auto destroy_pg = [](OptixProgramGroup& pg) {
        if (pg) { optixProgramGroupDestroy(pg); pg = nullptr; }
    };
    destroy_pg(raygen_test_pg_);
    destroy_pg(raygen_nee_pg_);
    destroy_pg(raygen_render_pg_);
    destroy_pg(raygen_caustic_pg_);
    destroy_pg(miss_pg_);
    destroy_pg(miss_shadow_pg_);
    destroy_pg(hitgroup_pg_);
    destroy_pg(hitgroup_shadow_pg_);

    if (pipeline_) { optixPipelineDestroy(pipeline_); pipeline_ = nullptr; }
    if (module_)   { optixModuleDestroy(module_); module_ = nullptr; }
    if (context_)  { optixDeviceContextDestroy(context_); context_ = nullptr; }

    initialised_ = false;
}
