// ---------------------------------------------------------------------
// optix_caustic_debug.cpp -- Caustic debug rendering (extracted from optix_renderer.cpp)
// ---------------------------------------------------------------------
#include "optix/optix_renderer.h"
#include "core/config.h"
#include "renderer/tonemap.h"

#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <iostream>

#include <optix.h>
#include <optix_stubs.h>

namespace {
    constexpr float HASHGRID_CELL_FACTOR = 2.0f;
}

// Forward declaration — defined in optix_renderer.cpp
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
    bool volume_enabled, float volume_density, float volume_falloff,
    float volume_albedo, int volume_samples, float volume_max_t,
    const DeviceBuffer& cell_guide, const DeviceBuffer& cell_caustic,
    const DeviceBuffer& cell_density);

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
            config.volume_albedo, config.volume_samples, config.volume_max_t,
            d_cell_guide_fraction_, d_cell_caustic_fraction_, d_cell_flux_density_);
        fill_clearcoat_fabric_params(lp);
        fill_cell_grid_params(lp);

        // Disable dual-budget for this debug pass — all photons are
        // tag-2 caustic, so single-budget gather with N_caustic is correct.
        lp.photon_is_caustic_pass = nullptr;
        lp.num_caustic_emitted    = 0;
        lp.caustic_gather_radius  = caustic_radius_;

        lp.volume_enabled     = (int)runtime_volume_enabled_;
        lp.photon_num_hero    = d_photon_num_hero_.d_ptr ? d_photon_num_hero_.as<uint8_t>() : nullptr;
        lp.samples_per_pixel  = 1;
        lp.max_bounces        = config.max_bounces;
        lp.max_bounces_camera = DEFAULT_MAX_BOUNCES_CAMERA;
        lp.min_bounces_rr    = DEFAULT_MIN_BOUNCES_RR;
        lp.rr_threshold      = DEFAULT_RR_THRESHOLD;
        lp.guide_fraction    = guide_fraction_;
        lp.frame_number       = s;
        lp.render_mode        = RenderMode::Full;
        lp.exposure           = exposure_;

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
    d_photon_is_caustic_pass_.upload(caustic_pass_flags_);

    // Re-build GPU hash grid for the restored full photon set
    {
        const int    n         = (int)stored_photons_.size();
        const float  cell_size = gather_radius_ * 2.0f;
        const uint32_t tbl_sz  = (uint32_t)(std::max)(n, 1024);

        d_grid_sorted_indices_.alloc(n * sizeof(uint32_t));
        d_grid_cell_start_.alloc(tbl_sz * sizeof(uint32_t));
        d_grid_cell_end_.alloc(tbl_sz * sizeof(uint32_t));
        d_grid_keys_in_.alloc(n * sizeof(uint32_t));
        d_grid_keys_out_.alloc(n * sizeof(uint32_t));
        d_grid_indices_in_.alloc(n * sizeof(uint32_t));

        size_t cub_tmp = 0;
        gpu_build_hash_grid(
            d_photon_pos_x_.as<float>(), d_photon_pos_y_.as<float>(), d_photon_pos_z_.as<float>(),
            n, cell_size, tbl_sz,
            d_grid_keys_in_.as<uint32_t>(), d_grid_keys_out_.as<uint32_t>(),
            d_grid_indices_in_.as<uint32_t>(), d_grid_sorted_indices_.as<uint32_t>(),
            d_grid_cell_start_.as<uint32_t>(), d_grid_cell_end_.as<uint32_t>(),
            nullptr, cub_tmp);
        d_grid_cub_temp_.alloc(cub_tmp);
        gpu_build_hash_grid(
            d_photon_pos_x_.as<float>(), d_photon_pos_y_.as<float>(), d_photon_pos_z_.as<float>(),
            n, cell_size, tbl_sz,
            d_grid_keys_in_.as<uint32_t>(), d_grid_keys_out_.as<uint32_t>(),
            d_grid_indices_in_.as<uint32_t>(), d_grid_sorted_indices_.as<uint32_t>(),
            d_grid_cell_start_.as<uint32_t>(), d_grid_cell_end_.as<uint32_t>(),
            d_grid_cub_temp_.d_ptr, cub_tmp);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

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
}