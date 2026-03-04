// ---------------------------------------------------------------------
// optix_renderer.cpp -- OptiX host-side rendering & photon tracing
// ---------------------------------------------------------------------
#include "optix/optix_renderer.h"
#include "core/config.h"
#include "optix/adaptive_sampling.h"
#include "photon/specular_target.h"   // SpecularTargetSet (for GPU upload)
#include "photon/tri_photon_irradiance.h"  // per-triangle irradiance heatmap
#include "photon/emitter.h"                // CPU photon tracing
#include "photon/hash_grid.h"              // gpu_build_hash_grid
#include "photon/cell_cache.h"             // CellInfoCache (for PA-08 cell analysis)
#include "photon/photon_bins.h"            // PhotonBinDirs (Fibonacci sphere bin directions)
#include "renderer/tonemap.h"              // launch_tonemap_kernel etc.

#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <climits>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <functional>
#include <unordered_set>

#include <optix.h>
#include <optix_stubs.h>

// ---------------------------------------------------------------------
// Module-local implementation constants
// ---------------------------------------------------------------------
namespace {
    // Ray epsilon to avoid self-intersections.
    constexpr float OPTIX_SCENE_EPSILON      = 1e-4f;

    // Large tmax avoids clipping long rays in normalized scenes.
    constexpr float DEFAULT_RAY_TMAX         = 1e20f;

    // HashGrid::build() uses cell_size = radius * 2.0f.
    constexpr float HASHGRID_CELL_FACTOR     = 2.0f;
}

// fill_common_params() -- helper to fill LaunchParams
void fill_common_params(
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
    // â”€â”€ Volume participating-medium params â”€â”€
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

    // num_photons from position buffer
    p.num_photons       = (int)(photon_pos_x.bytes / sizeof(float));
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

    p.gather_radius       = gather_radius;

    // Emitter data (for NEE in render and photon trace)
    p.emissive_tri_indices = const_cast<uint32_t*>(emissive_idx.as<uint32_t>());
    p.emissive_cdf         = const_cast<float*>(emissive_cdf.as<float>());
    p.emissive_local_idx   = nullptr;  // set by callers that have the inverse-index buffer
    p.num_emissive         = num_emissive;
    p.total_emissive_power = total_emissive_power;

    p.traversable = gas_handle;

    // â”€â”€ Participating medium â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    p.volume_enabled = volume_enabled ? 1 : 0;
    p.volume_density = volume_density;
    p.volume_falloff = volume_falloff;
    p.volume_albedo  = volume_albedo;
    p.volume_samples = volume_samples;
    p.volume_max_t   = volume_max_t;

    // Default-off bounce AOV (callers that need it override after fill_common_params)
    p.bounce_aov_enabled = 0;
    for (int b = 0; b < MAX_AOV_BOUNCES; ++b)
        p.bounce_aov[b] = nullptr;

    // Terminal-bounce photon gather (H-1 fix: was never set → always 0)
    p.photon_final_gather = DEFAULT_PHOTON_FINAL_GATHER ? 1 : 0;

    // Environment map defaults (overridden by fill_envmap_params)
    p.has_envmap = 0;
    p.envmap_pixels = nullptr;
    p.envmap_marginal_cdf = nullptr;
    p.envmap_conditional_cdf = nullptr;
    p.envmap_width = 0;
    p.envmap_height = 0;
    p.envmap_scale = 1.0f;
    p.envmap_total_power = 0.f;
    p.envmap_selection_prob = 0.f;
    p.envmap_scene_center = make_f3(0, 0, 0);
    p.envmap_scene_radius = 1.0f;
    p.envmap_rot_row0 = make_f3(1, 0, 0);
    p.envmap_rot_row1 = make_f3(0, 1, 0);
    p.envmap_rot_row2 = make_f3(0, 0, 1);
    p.envmap_inv_rot_row0 = make_f3(1, 0, 0);
    p.envmap_inv_rot_row1 = make_f3(0, 1, 0);
    p.envmap_inv_rot_row2 = make_f3(0, 0, 1);
}

// fill_cell_grid_params() -- Wire volume CellBinGrid + kNN data into LaunchParams
void OptixRenderer::fill_cell_grid_params(LaunchParams& lp) const {
    // Surface guided sampling now uses per-hitpoint kNN (no histogram upload).
    lp.photon_bin_count = PHOTON_BIN_COUNT;

    // â”€â”€ VP-07: Volume cell-bin grid params â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if (d_vol_cell_bin_grid_.d_ptr && vol_cell_bin_grid_.total_cells() > 0) {
        lp.vol_cell_bin_grid       = reinterpret_cast<PhotonBin*>(d_vol_cell_bin_grid_.d_ptr);
        lp.vol_cell_grid_valid     = 1;
        lp.vol_cell_grid_min_x     = vol_cell_bin_grid_.min_x;
        lp.vol_cell_grid_min_y     = vol_cell_bin_grid_.min_y;
        lp.vol_cell_grid_min_z     = vol_cell_bin_grid_.min_z;
        lp.vol_cell_grid_cell_size = vol_cell_bin_grid_.cell_size;
        lp.vol_cell_grid_dim_x     = vol_cell_bin_grid_.dim_x;
        lp.vol_cell_grid_dim_y     = vol_cell_bin_grid_.dim_y;
        lp.vol_cell_grid_dim_z     = vol_cell_bin_grid_.dim_z;
    } else {
        lp.vol_cell_bin_grid       = nullptr;
        lp.vol_cell_grid_valid     = 0;
    }

    // â”€â”€ MT-07: Volume photon kNN data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if (d_vol_grid_sorted_indices_.d_ptr && volume_photons_.size() > 0) {
        lp.vol_photon_pos_x         = reinterpret_cast<float*>(d_vol_photon_pos_x_.d_ptr);
        lp.vol_photon_pos_y         = reinterpret_cast<float*>(d_vol_photon_pos_y_.d_ptr);
        lp.vol_photon_pos_z         = reinterpret_cast<float*>(d_vol_photon_pos_z_.d_ptr);
        lp.vol_photon_wi_x          = reinterpret_cast<float*>(d_vol_photon_wi_x_.d_ptr);
        lp.vol_photon_wi_y          = reinterpret_cast<float*>(d_vol_photon_wi_y_.d_ptr);
        lp.vol_photon_wi_z          = reinterpret_cast<float*>(d_vol_photon_wi_z_.d_ptr);
        lp.vol_photon_lambda        = reinterpret_cast<uint16_t*>(d_vol_photon_lambda_.d_ptr);
        lp.vol_photon_flux          = reinterpret_cast<float*>(d_vol_photon_flux_.d_ptr);
        lp.num_vol_photons          = (int)volume_photons_.size();
        lp.num_vol_photons_emitted  = num_photons_emitted_;
        lp.vol_grid_sorted_indices  = reinterpret_cast<uint32_t*>(d_vol_grid_sorted_indices_.d_ptr);
        lp.vol_grid_cell_start      = reinterpret_cast<uint32_t*>(d_vol_grid_cell_start_.d_ptr);
        lp.vol_grid_cell_end        = reinterpret_cast<uint32_t*>(d_vol_grid_cell_end_.d_ptr);
        lp.vol_grid_cell_size       = volume_grid_.cell_size;
        lp.vol_grid_table_size      = volume_grid_.table_size;
        lp.vol_gather_radius        = gather_radius_;
    } else {
        lp.vol_photon_pos_x        = nullptr;
        lp.vol_photon_pos_y        = nullptr;
        lp.vol_photon_pos_z        = nullptr;
        lp.vol_photon_wi_x         = nullptr;
        lp.vol_photon_wi_y         = nullptr;
        lp.vol_photon_wi_z         = nullptr;
        lp.vol_photon_lambda       = nullptr;
        lp.vol_photon_flux         = nullptr;
        lp.num_vol_photons         = 0;
        lp.num_vol_photons_emitted = 0;
        lp.vol_grid_sorted_indices = nullptr;
        lp.vol_grid_cell_start     = nullptr;
        lp.vol_grid_cell_end       = nullptr;
        lp.vol_grid_cell_size      = 0.f;
        lp.vol_grid_table_size     = 0;
        lp.vol_gather_radius       = 0.f;
    }
}

// fill_dense_grid_params() -- Wire dense grid device pointers into LaunchParams
void OptixRenderer::fill_dense_grid_params(LaunchParams& lp) {
    if (d_dense_sorted_indices_.d_ptr && stored_dense_grid_.total_cells() > 0) {
        lp.dense_sorted_indices = reinterpret_cast<uint32_t*>(d_dense_sorted_indices_.d_ptr);
        lp.dense_cell_start     = reinterpret_cast<uint32_t*>(d_dense_cell_start_.d_ptr);
        lp.dense_cell_end       = reinterpret_cast<uint32_t*>(d_dense_cell_end_.d_ptr);
        lp.dense_valid          = 1;
        lp.dense_min_x          = stored_dense_grid_.min_x;
        lp.dense_min_y          = stored_dense_grid_.min_y;
        lp.dense_min_z          = stored_dense_grid_.min_z;
        lp.dense_cell_size      = stored_dense_grid_.cell_size;
        lp.dense_dim_x          = stored_dense_grid_.dim_x;
        lp.dense_dim_y          = stored_dense_grid_.dim_y;
        lp.dense_dim_z          = stored_dense_grid_.dim_z;
    } else {
        lp.dense_sorted_indices = nullptr;
        lp.dense_cell_start     = nullptr;
        lp.dense_cell_end       = nullptr;
        lp.dense_valid          = 0;
    }
}

// fill_dm_hash_grid_params() -- Wire direction-map hash grid into LaunchParams
void OptixRenderer::fill_dm_hash_grid_params(LaunchParams& lp) {
    if (d_dm_hash_sorted_indices_.d_ptr && dm_hash_grid_.table_size > 0) {
        lp.dm_hash_sorted_indices = reinterpret_cast<uint32_t*>(d_dm_hash_sorted_indices_.d_ptr);
        lp.dm_hash_cell_start     = reinterpret_cast<uint32_t*>(d_dm_hash_cell_start_.d_ptr);
        lp.dm_hash_cell_end       = reinterpret_cast<uint32_t*>(d_dm_hash_cell_end_.d_ptr);
        lp.dm_hash_cell_size      = dm_hash_grid_.cell_size;
        lp.dm_hash_table_size     = dm_hash_grid_.table_size;
        lp.dm_hash_valid          = 1;
    } else {
        lp.dm_hash_sorted_indices = nullptr;
        lp.dm_hash_cell_start     = nullptr;
        lp.dm_hash_cell_end       = nullptr;
        lp.dm_hash_cell_size      = 0.f;
        lp.dm_hash_table_size     = 0;
        lp.dm_hash_valid          = 0;
    }
}

// =====================================================================
// Direction map: fill params, build (OptiX launch), download
// =====================================================================

void OptixRenderer::fill_direction_map_params(LaunchParams& lp) const {
    const int factor = DIR_MAP_SUBPIXEL_FACTOR;
    const int dm_w = width_  * factor;
    const int dm_h = height_ * factor;

    if (d_dir_map_buffer_.d_ptr && d_dir_map_buffer_.bytes >= (size_t)dm_w * dm_h * sizeof(DirMapEntry)) {
        lp.dir_map_buffer   = reinterpret_cast<DirMapEntry*>(d_dir_map_buffer_.d_ptr);
        lp.dir_map_width    = dm_w;
        lp.dir_map_height   = dm_h;
        lp.dir_map_valid    = 1;
    } else {
        lp.dir_map_buffer   = nullptr;
        lp.dir_map_width    = 0;
        lp.dir_map_height   = 0;
        lp.dir_map_valid    = 0;
    }
    lp.dir_map_spp_seed = 0;  // default; overridden per-SPP in render loops
}

void OptixRenderer::build_direction_map(const Camera& camera, int spp_seed) {
    const int factor = DIR_MAP_SUBPIXEL_FACTOR;
    const int dm_w = width_  * factor;
    const int dm_h = height_ * factor;
    const size_t dm_bytes = (size_t)dm_w * dm_h * sizeof(DirMapEntry);

    // Allocate device buffer if needed
    d_dir_map_buffer_.ensure_alloc(dm_bytes);

    // Build a minimal LaunchParams for the direction map pass
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
    fill_dense_grid_params(lp);
    fill_dm_hash_grid_params(lp);

    // Direction map specific fields
    lp.dir_map_buffer   = reinterpret_cast<DirMapEntry*>(d_dir_map_buffer_.d_ptr);
    lp.dir_map_width    = dm_w;
    lp.dir_map_height   = dm_h;
    lp.dir_map_valid    = 1;
    lp.dir_map_spp_seed = spp_seed;
    lp.guide_radius     = DEFAULT_GUIDE_RADIUS;

    // ── Debug: dump LaunchParams state before launch ──────────────
    std::printf("[DirMap] build_direction_map: dm_w=%d dm_h=%d  spp_seed=%d\n", dm_w, dm_h, spp_seed);
    std::printf("[DirMap]   dm_hash_valid=%d  table_size=%u  cell_size=%.6f\n",
               lp.dm_hash_valid, lp.dm_hash_table_size, lp.dm_hash_cell_size);
    std::printf("[DirMap]   dm_hash_sorted_indices=%p  cell_start=%p  cell_end=%p\n",
               (void*)lp.dm_hash_sorted_indices, (void*)lp.dm_hash_cell_start, (void*)lp.dm_hash_cell_end);
    std::printf("[DirMap]   num_photons=%d  guide_radius=%.6f  traversable=%llu\n",
               lp.num_photons, lp.guide_radius, (unsigned long long)lp.traversable);
    std::printf("[DirMap]   dir_map_buffer=%p  dir_map_valid=%d\n",
               (void*)lp.dir_map_buffer, lp.dir_map_valid);
    std::printf("[DirMap]   cam_pos=(%.3f, %.3f, %.3f)\n",
               lp.cam_pos.x, lp.cam_pos.y, lp.cam_pos.z);
    std::printf("[DirMap]   photon_pos_x=%p  photon_wi_x=%p\n",
               (void*)lp.photon_pos_x, (void*)lp.photon_wi_x);

    // Upload params and swap SBT to direction map raygen
    d_launch_params_.ensure_alloc(sizeof(LaunchParams));
    CUDA_CHECK(cudaMemcpy(d_launch_params_.d_ptr, &lp,
                           sizeof(LaunchParams), cudaMemcpyHostToDevice));

    OptixShaderBindingTable dirmap_sbt = sbt_;
    dirmap_sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(d_raygen_dirmap_record_.d_ptr);

    OPTIX_CHECK(optixLaunch(
        pipeline_,
        nullptr,
        reinterpret_cast<CUdeviceptr>(d_launch_params_.d_ptr),
        sizeof(LaunchParams),
        &dirmap_sbt,
        dm_w, dm_h, 1));

    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("[DirMap] OptiX launch completed\n");
}

void OptixRenderer::download_direction_map() {
    direction_map_.resize(width_, height_);
    if (d_dir_map_buffer_.d_ptr) {
        CUDA_CHECK(cudaMemcpy(direction_map_.entries.data(),
                              d_dir_map_buffer_.d_ptr,
                              direction_map_.entries.size() * sizeof(DirMapEntry),
                              cudaMemcpyDeviceToHost));
    }

    // ── Debug: analyse downloaded direction map ──────────────────
    int n_total = (int)direction_map_.entries.size();
    int n_hit = 0, n_eligible = 0, n_has_dir = 0;
    float max_pdf = 0.f;
    uint16_t max_elig = 0;
    for (int i = 0; i < n_total; ++i) {
        const auto& e = direction_map_.entries[i];
        if (e.hit_x != 0.f || e.hit_y != 0.f || e.hit_z != 0.f) ++n_hit;
        if (e.num_eligible > 0) ++n_eligible;
        if (e.dir_x != 0.f || e.dir_y != 0.f || e.dir_z != 0.f) ++n_has_dir;
        if (e.pdf > max_pdf) max_pdf = e.pdf;
        if (e.num_eligible > max_elig) max_elig = e.num_eligible;
    }
    std::printf("[DirMap] download: %d entries  base=%dx%d  factor=%d\n",
               n_total, direction_map_.base_width, direction_map_.base_height,
               direction_map_.factor);
    int denom = n_total > 0 ? n_total : 1;
    std::printf("[DirMap]   n_hit=%d (%.1f%%)  n_eligible=%d (%.1f%%)  n_has_dir=%d (%.1f%%)\n",
               n_hit, 100.f * n_hit / denom,
               n_eligible, 100.f * n_eligible / denom,
               n_has_dir, 100.f * n_has_dir / denom);
    std::printf("[DirMap]   max_pdf=%.6f  max_eligible=%d\n", max_pdf, (int)max_elig);
    // Print first 5 non-zero entries
    int printed = 0;
    for (int i = 0; i < n_total && printed < 5; ++i) {
        const auto& e = direction_map_.entries[i];
        if (e.num_eligible > 0) {
            std::printf("[DirMap]   entry[%d]: dir=(%.4f,%.4f,%.4f) pdf=%.6f elig=%d pos=(%.3f,%.3f,%.3f)\n",
                       i, e.dir_x, e.dir_y, e.dir_z, e.pdf, (int)e.num_eligible,
                       e.hit_x, e.hit_y, e.hit_z);
            ++printed;
        }
    }
    if (printed == 0)
        std::printf("[DirMap]   WARNING: all entries have num_eligible=0 — direction map is empty!\n");
}

// render_debug_frame() -- first-hit preview (v3: always traces full path)
void OptixRenderer::render_debug_frame(
    const Camera& camera, int frame_number,
    RenderMode mode, int spp, bool /*shadow_rays*/)
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
    fill_cell_grid_params(lp);
    fill_dense_grid_params(lp);
    fill_direction_map_params(lp);

    lp.num_caustic_emitted    = num_caustic_emitted_;
    lp.caustic_gather_radius  = caustic_radius_;

    // Runtime toggle (V key) overrides the compile-time default
    lp.volume_enabled = (int)runtime_volume_enabled_;

    lp.samples_per_pixel = spp;
    lp.max_bounces_camera = DEFAULT_MAX_BOUNCES_CAMERA;
    lp.min_bounces_rr    = DEFAULT_MIN_BOUNCES_RR;
    lp.rr_threshold      = DEFAULT_RR_THRESHOLD;
    lp.guide_fraction    = guide_fraction_;
    lp.guide_radius      = DEFAULT_GUIDE_RADIUS;
    lp.dir_map_spp_seed  = frame_number;  // vary direction map per frame
    lp.preview_mode      = preview_mode_ ? 1 : 0;
    lp.frame_number      = frame_number;
    lp.render_mode       = mode;
    lp.exposure           = exposure_;

    // Per-triangle photon irradiance heatmap
    lp.tri_photon_irradiance = d_tri_photon_irradiance_.d_ptr
        ? d_tri_photon_irradiance_.as<float>() : nullptr;
    lp.num_triangles         = num_tris_;
    lp.show_photon_heatmap   = show_photon_heatmap_ ? 1 : 0;

    last_launch_params_host_ = lp;

    // Upload launch params
    d_launch_params_.ensure_alloc(sizeof(LaunchParams));
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

    // ── Post-FX in interactive preview ──────────────────────────────
    // When bloom is enabled, re-derive HDR from the spectral accumulator,
    // apply bloom, then tone-map to overwrite d_srgb_buffer_.
    // The inline-tonemapped sRGB from the raygen shader is discarded.
    if (postfx_params_.bloom_enabled) {
        launch_spectrum_to_hdr_kernel(
            (const float*)d_spectrum_buffer_.d_ptr,
            (const float*)d_sample_counts_.d_ptr,
            (float*)d_hdr_buffer_.d_ptr,
            width_, height_, exposure_);
        CUDA_CHECK(cudaDeviceSynchronize());

        postfx_pipeline_.apply((float*)d_hdr_buffer_.d_ptr,
                               width_, height_, postfx_params_);

        launch_tonemap_hdr_kernel(
            (const float*)d_hdr_buffer_.d_ptr,
            (uint8_t*)d_srgb_buffer_.d_ptr,
            width_, height_);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// render_one_spp() -- launch a single sample of full path tracing
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
    fill_cell_grid_params(lp);
    fill_dense_grid_params(lp);
    fill_direction_map_params(lp);

    lp.num_caustic_emitted    = num_caustic_emitted_;
    lp.caustic_gather_radius  = caustic_radius_;

    // Runtime toggle (V key) overrides the compile-time default
    lp.volume_enabled = (int)runtime_volume_enabled_;

    lp.samples_per_pixel  = 1;
    lp.max_bounces_camera = DEFAULT_MAX_BOUNCES_CAMERA;
    lp.min_bounces_rr    = DEFAULT_MIN_BOUNCES_RR;
    lp.rr_threshold      = DEFAULT_RR_THRESHOLD;
    lp.guide_fraction    = guide_fraction_;
    lp.guide_radius      = DEFAULT_GUIDE_RADIUS;
    lp.dir_map_spp_seed  = frame_number;  // vary direction map per frame
    lp.preview_mode      = 0;  // render_one_spp is always full quality
    lp.frame_number       = frame_number;
    lp.render_mode        = RenderMode::Full;
    lp.exposure           = exposure_;

    last_launch_params_host_ = lp;

    d_launch_params_.ensure_alloc(sizeof(LaunchParams));
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
// build_view_adaptive_cdf() -- VA-01/02/03
// Reads stored_photons_.source_emissive_idx, aggregates per-emitter
// usefulness, builds a mixture CDF, and re-uploads it to the GPU.
// =====================================================================
bool OptixRenderer::build_view_adaptive_cdf(const Scene& scene, float beta)
{
    const size_t n_emitters = scene.emissive_tri_indices.size();
    if (n_emitters == 0 || stored_photons_.size() == 0) return false;
    if (stored_photons_.source_emissive_idx.empty()) return false;

    // â”€â”€ VA-01/02: Per-emitter usefulness â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // Accumulate total flux originating from each emitter.
    std::vector<double> usefulness(n_emitters, 0.0);
    const size_t n_photons = stored_photons_.size();
    for (size_t i = 0; i < n_photons; ++i) {
        uint16_t eidx = stored_photons_.source_emissive_idx[i];
        if (eidx >= n_emitters) continue;
        // Sum hero-wavelength flux
        double flux_sum = 0.0;
        int n_hero = stored_photons_.num_hero.empty() ? 1
                     : (int)stored_photons_.num_hero[i];
        for (int h = 0; h < n_hero; ++h)
            flux_sum += stored_photons_.flux[i * HERO_WAVELENGTHS + h];
        usefulness[eidx] += flux_sum;
    }

    // Normalize usefulness â†’ p_view
    double total_use = 0.0;
    for (auto u : usefulness) total_use += u;
    if (total_use <= 0.0) return false;  // all photons had invalid emitters

    std::vector<float> p_view(n_emitters);
    for (size_t i = 0; i < n_emitters; ++i)
        p_view[i] = (float)(usefulness[i] / total_use);

    // â”€â”€ VA-03: Mixture CDF â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // p_power from scene's area Ã— Le weights
    std::vector<float> p_power(n_emitters);
    {
        float total_pow = 0.f;
        for (size_t i = 0; i < n_emitters; ++i) {
            uint32_t tri_idx = scene.emissive_tri_indices[i];
            const auto& tri = scene.triangles[tri_idx];
            const auto& mat = scene.materials[tri.material_id];
            p_power[i] = tri.area() * mat.mean_emission();
            total_pow += p_power[i];
        }
        if (total_pow > 0.f)
            for (auto& w : p_power) w /= total_pow;
    }

    // Mixture: p(k) = (1-Î²)Â·p_power(k) + Î²Â·p_view(k)
    std::vector<float> cdf(n_emitters);
    float cum = 0.f;
    for (size_t i = 0; i < n_emitters; ++i) {
        float mix = (1.f - beta) * p_power[i] + beta * p_view[i];
        cum += mix;
        cdf[i] = cum;
    }
    // Normalize to ensure last entry is exactly 1.0
    if (cum > 0.f)
        for (auto& c : cdf) c /= cum;
    cdf[n_emitters - 1] = 1.0f;

    // Upload the new CDF
    d_emissive_cdf_.upload(cdf);

    std::printf("[VA] View-adaptive CDF built: Î²=%.2f  top_use=%.4e  emitters=%zu\n",
                beta, *std::max_element(usefulness.begin(), usefulness.end()),
                n_emitters);
    return true;
}

// render_final() -- full path tracing (v3)
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
        fill_cell_grid_params(lp);
        fill_dense_grid_params(lp);
        fill_direction_map_params(lp);

        lp.num_caustic_emitted    = num_caustic_emitted_;
        lp.caustic_gather_radius  = caustic_radius_;

        // Runtime toggle (V key) overrides the RenderConfig value
        lp.volume_enabled = (int)runtime_volume_enabled_;

        lp.samples_per_pixel  = 1;
        lp.max_bounces_camera = DEFAULT_MAX_BOUNCES_CAMERA;
        lp.min_bounces_rr    = DEFAULT_MIN_BOUNCES_RR;
        lp.rr_threshold      = DEFAULT_RR_THRESHOLD;
        lp.guide_fraction    = guide_fraction_;
        lp.guide_radius       = DEFAULT_GUIDE_RADIUS;
        lp.dir_map_spp_seed   = frame_number;  // vary direction map per SPP
        lp.frame_number       = frame_number;
        lp.render_mode        = RenderMode::Full;
        lp.exposure           = exposure_;
        lp.skip_tonemap       = 1;  // defer tonemap to post-process kernel

        // Denoiser AOV buffers (written on first sample of first frame)
        lp.albedo_buffer = d_albedo_buffer_.d_ptr
            ? reinterpret_cast<float*>(d_albedo_buffer_.d_ptr) : nullptr;
        lp.normal_buffer = d_normal_buffer_.d_ptr
            ? reinterpret_cast<float*>(d_normal_buffer_.d_ptr) : nullptr;

        // Adaptive buffers
        lp.lum_sum    = adaptive
            ? reinterpret_cast<float*>(d_lum_sum_.d_ptr)   : nullptr;
        lp.lum_sum2   = adaptive
            ? reinterpret_cast<float*>(d_lum_sum2_.d_ptr)  : nullptr;
        lp.active_mask = (adaptive && use_mask)
            ? reinterpret_cast<uint8_t*>(d_active_mask_.d_ptr) : nullptr;

        // Per-bounce AOV buffers (DB-04)
        lp.bounce_aov_enabled = config.bounce_aov_enabled ? 1 : 0;
        for (int b = 0; b < MAX_AOV_BOUNCES; ++b)
            lp.bounce_aov[b] = (config.bounce_aov_enabled && d_bounce_aov_[b].d_ptr)
                ? reinterpret_cast<float*>(d_bounce_aov_[b].d_ptr) : nullptr;

        d_launch_params_.ensure_alloc(sizeof(LaunchParams));
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

    // â”€â”€ Progress helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    // â”€â”€ Build single photon map â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    std::printf("\n[Render] Building photon map ...\n");
    trace_photons(scene, config, /*grid_radius_override=*/0.f,
                  /*photon_map_seed=*/0);

    // â”€â”€ VA-04/05: View-adaptive photon re-trace â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // After the initial photon map is built, compute per-emitter
    // usefulness from stored_photons_.source_emissive_idx, construct
    // a mixture CDF, and optionally re-trace with the updated CDF.
    if (config.view_adaptive_enabled && config.view_adaptive_retrace > 0) {
        const int n_retrace = config.view_adaptive_retrace;
        for (int r = 0; r < n_retrace; ++r) {
            float beta = config.view_adaptive_beta;
            // Progressive: ramp Î² from 0 to target over iterations
            if (n_retrace > 1)
                beta *= (float)(r + 1) / (float)n_retrace;

            if (build_view_adaptive_cdf(scene, beta)) {
                std::printf("[VA] Re-tracing photons (iteration %d/%d, Î²=%.2f) ...\n",
                            r + 1, n_retrace, beta);
                trace_photons(scene, config, /*grid_radius_override=*/0.f,
                              /*photon_map_seed=*/r + 1);
            } else {
                std::printf("[VA] Skipping re-trace: no valid emitter usefulness data\n");
                break;
            }
        }
    }

    // ── Build direction map (first-hit guidance) ─────────────────────
    // Uses the current photon map + camera to precompute one guided
    // direction per subpixel.  Currently built once (spp_seed=0);
    // the same direction is reused across all SPP.
    // TODO(H-2): rebuild per SPP inside the loop (with incrementing
    //   spp_seed) to decorrelate guide samples and improve convergence.
    //   Requires profiling: one extra OptiX launch per SPP at
    //   (W*4)×(H*4) = ~12M threads.  Alternative: store multiple
    //   candidate directions in DirMapEntry and index by frame_number.
    std::printf("[Render] Building direction map (%dx%d subpixels) ...\n",
                width_ * DIR_MAP_SUBPIXEL_FACTOR,
                height_ * DIR_MAP_SUBPIXEL_FACTOR);
    build_direction_map(camera, /*spp_seed=*/0);

    if (!adaptive) {
        // â”€â”€ Non-adaptive path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        for (int s = 0; s < total_spp; ++s) {
            // Periodic photon + direction map rebuild
            if (DEFAULT_GUIDE_REMAP_INTERVAL > 0 && s > 0
                && (s % DEFAULT_GUIDE_REMAP_INTERVAL) == 0) {
                std::printf("\n[Render] Rebuilding photon map + direction map (spp %d) ...\n", s);
                trace_photons(scene, config, 0.f, /*photon_map_seed=*/s);
                build_direction_map(camera, /*spp_seed=*/s);
            }
            launch_pass(s, /*use_mask=*/false);
            print_progress(s + 1, total_spp, /*active=*/-1);
        }
    } else {
        // â”€â”€ Adaptive path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        // Phase 1: warmup â€” render min_spp passes uniformly
        for (int s = 0; s < min_spp; ++s) {
            launch_pass(s, /*use_mask=*/false);
            print_progress(s + 1, max_spp, /*active=*/(int)total_pixels);
        }

        // â”€â”€ AS-01/02: Cost map from pilot pass + photon analysis â”€â”€â”€â”€â”€
        // After the pilot, compute per-pixel SPP budgets.
        {
            CostMapParams cp;
            cp.lum_sum             = reinterpret_cast<float*>(d_lum_sum_.d_ptr);
            cp.lum_sum2            = reinterpret_cast<float*>(d_lum_sum2_.d_ptr);
            cp.sample_counts       = reinterpret_cast<float*>(d_sample_counts_.d_ptr);
            cp.cell_guide_fraction = nullptr;
            cp.cell_caustic_fraction = nullptr;
            cp.cell_flux_density   = nullptr;
            cp.spectrum_buffer     = reinterpret_cast<float*>(d_spectrum_buffer_.d_ptr);
            cp.width               = width_;
            cp.height              = height_;
            cp.base_spp            = total_spp;
            cp.min_spp_clamp       = min_spp;
            cp.max_spp_clamp       = max_spp;
            cp.pixel_max_spp       = reinterpret_cast<uint16_t*>(d_pixel_max_spp_.d_ptr);
            compute_pixel_cost_map(cp);
            std::printf("\n[Render] AS-02: Per-pixel cost map computed (pilot=%d SPP)\n", min_spp);
        }

        // Phase 2: adaptive â€” update mask every update_interval passes
        int active_pixels = (int)total_pixels;
        int frame = min_spp;
        const int update_interval = config.adaptive_update_interval;
        long long total_samples_done = (long long)min_spp * total_pixels;

        for (int s = min_spp; s < max_spp; ++s) {
            // Periodic photon + direction map rebuild
            if (DEFAULT_GUIDE_REMAP_INTERVAL > 0 && s > 0
                && (s % DEFAULT_GUIDE_REMAP_INTERVAL) == 0) {
                std::printf("\n[Render] Rebuilding photon map + direction map (spp %d) ...\n", s);
                trace_photons(scene, config, 0.f, /*photon_map_seed=*/s);
                build_direction_map(camera, /*spp_seed=*/s);
            }
            // Recompute mask at start of adaptive phase and every N passes
            if ((s - min_spp) % update_interval == 0) {
                AdaptiveParams ap;
                ap.sample_counts  = reinterpret_cast<float*>(d_sample_counts_.d_ptr);
                ap.lum_sum        = reinterpret_cast<float*>(d_lum_sum_.d_ptr);
                ap.lum_sum2       = reinterpret_cast<float*>(d_lum_sum2_.d_ptr);
                ap.active_mask    = reinterpret_cast<uint8_t*>(d_active_mask_.d_ptr);
                ap.pixel_max_spp  = reinterpret_cast<uint16_t*>(d_pixel_max_spp_.d_ptr);
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
            total_samples_done += active_pixels;  // AS-05: track actual work
            print_progress(s + 1, max_spp, active_pixels);
        }
    }

    // â”€â”€ Post-process: optionally denoise, then tonemap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // The per-SPP inline tonemap was skipped (skip_tonemap=1).
    if (config.denoiser_enabled) {
        // 1. Convert spectral accumulator â†’ linear HDR float4
        launch_spectrum_to_hdr_kernel(
            (const float*)d_spectrum_buffer_.d_ptr,
            (const float*)d_sample_counts_.d_ptr,
            (float*)d_hdr_buffer_.d_ptr,
            width_, height_, exposure_);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 2. Tone-map the raw (un-denoised) HDR â†’ sRGB for side-by-side comparison
        launch_tonemap_hdr_kernel(
            (const float*)d_hdr_buffer_.d_ptr,
            (uint8_t*)d_srgb_raw_buffer_.d_ptr,
            width_, height_);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 3. Set up and run the OptiX AI denoiser
        setup_denoiser(width_, height_,
                       config.denoiser_guide_albedo,
                       config.denoiser_guide_normal);
        run_denoiser(config.denoiser_blend);

        // 3b. Apply post-FX (bloom) on denoised HDR before tone mapping
        postfx_pipeline_.apply((float*)d_hdr_denoised_.d_ptr,
                               width_, height_, postfx_params_);

        // 4. Tone map the denoised HDR → sRGB
        launch_tonemap_hdr_kernel(
            (const float*)d_hdr_denoised_.d_ptr,
            (uint8_t*)d_srgb_buffer_.d_ptr,
            width_, height_);
        CUDA_CHECK(cudaDeviceSynchronize());

        cleanup_denoiser();
        std::cout << "\n[Denoiser] OptiX AI denoiser applied"
                  << (config.denoiser_guide_albedo ? " +albedo" : "")
                  << (config.denoiser_guide_normal ? " +normal" : "")
                  << " (blend=" << config.denoiser_blend << ")";
    } else {
        // Non-denoiser path: spectrum → HDR → post-FX → sRGB
        // (Previously went straight spectrum → sRGB, but post-FX needs
        //  the HDR intermediate for luminance-aware bloom.)
        launch_spectrum_to_hdr_kernel(
            (const float*)d_spectrum_buffer_.d_ptr,
            (const float*)d_sample_counts_.d_ptr,
            (float*)d_hdr_buffer_.d_ptr,
            width_, height_, exposure_);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Apply post-FX (bloom) on HDR
        postfx_pipeline_.apply((float*)d_hdr_buffer_.d_ptr,
                               width_, height_, postfx_params_);

        // Tone map HDR → sRGB
        launch_tonemap_hdr_kernel(
            (const float*)d_hdr_buffer_.d_ptr,
            (uint8_t*)d_srgb_buffer_.d_ptr,
            width_, height_);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(t_end - t_start).count();
    double mrays = (double)total_pixels * total_spp * config.max_bounces / total_s / 1e6;
    std::cout << "\n[Render] Done in " << std::fixed
              << std::setprecision(1) << total_s << "s"
              << "  (~" << (int)mrays << " Mray-bounces/s)\n";
}


// print_kernel_profiling() -- download GPU timing data and print summary
// Gated by ENABLE_STATS — heavy D2H copy, not needed in production.
void OptixRenderer::print_kernel_profiling() const {
    if constexpr (!ENABLE_STATS) return;
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

// gather_stats() -- collect renderer statistics for JSON export
OptixRenderer::RenderStats OptixRenderer::gather_stats(const char* scene_name) const {
    RenderStats s;
    s.image_width         = width_;
    s.image_height        = height_;

    // Accumulated SPP: read from sample_counts buffer (pixel 0)
    if (d_sample_counts_.d_ptr) {
        float n = 0.f;
        cudaMemcpy(&n, d_sample_counts_.d_ptr, sizeof(float), cudaMemcpyDeviceToHost);
        s.accumulated_spp = (int)n;
    }

    // Photon map
    s.photons_emitted     = num_photons_emitted_;
    s.photons_stored      = (int)stored_photons_.size();
    s.caustic_emitted     = num_caustic_emitted_;
    s.gather_radius       = gather_radius_;
    s.caustic_radius      = caustic_radius_;

    // Tag distribution
    s.noncaustic_stored    = 0;
    s.global_caustic_stored = 0;
    s.caustic_stored       = 0;
    for (size_t i = 0; i < caustic_pass_flags_.size(); ++i) {
        uint8_t t = caustic_pass_flags_[i];
        if (t == 0) ++s.noncaustic_stored;
        else if (t == 1) ++s.global_caustic_stored;
        else if (t == 2) ++s.caustic_stored;
    }

    // Cell analysis removed (dense grid replaces hash grid)
    s.cell_analysis_cells   = 0;

    // Config
    s.max_bounces_camera  = DEFAULT_MAX_BOUNCES_CAMERA;
    s.max_bounces_photon  = DEFAULT_MAX_BOUNCES;
    s.min_bounces_rr      = DEFAULT_MIN_BOUNCES_RR;
    s.rr_threshold        = DEFAULT_RR_THRESHOLD;
    s.guide_fraction      = guide_fraction_;
    s.exposure            = exposure_;
    s.denoiser_enabled    = denoiser_enabled_;
    s.knn_k               = DEFAULT_KNN_K;
    s.surface_tau         = DEFAULT_SURFACE_TAU;

    // Hash histogram stats removed — surface guide now uses per-hitpoint kNN.
    {
        auto& hh = s.hash_hist;
        hh.num_levels        = 0;
        hh.photons_processed = (int)stored_photons_.size();
        hh.gather_radius     = gather_radius_;
        hh.total_gpu_bytes   = 0;
    }

    // Scene
    s.scene_name          = scene_name ? scene_name : "";
    s.num_triangles       = num_tris_;
    s.num_emissive_tris   = num_emissive_;

    return s;
}

// trace_single_ray() -- for debug hover inspection
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
// Incremental SPPM helpers (one iteration per frame for interactive idle-refine)
// =====================================================================


// ── download_hdr_buffer() ──────────────────────────────────────────
void OptixRenderer::download_hdr_buffer(
    std::vector<float>& out, bool convert_from_spectrum)
{
    const size_t pixels = (size_t)width_ * height_;

    // Ensure the HDR device buffer exists
    d_hdr_buffer_.alloc(pixels * 4 * sizeof(float));

    if (convert_from_spectrum) {
        launch_spectrum_to_hdr_kernel(
            d_spectrum_buffer_.as<float>(),
            d_sample_counts_.as<float>(),
            reinterpret_cast<float*>(d_hdr_buffer_.d_ptr),
            width_, height_, exposure_);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    out.resize(pixels * 4);
    CUDA_CHECK(cudaMemcpy(out.data(), d_hdr_buffer_.d_ptr,
                          pixels * 4 * sizeof(float),
                          cudaMemcpyDeviceToHost));
}
