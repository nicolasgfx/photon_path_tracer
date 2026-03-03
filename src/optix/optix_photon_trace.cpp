// ---------------------------------------------------------------------
// optix_photon_trace.cpp -- GPU photon tracing (extracted from optix_renderer.cpp)
// ---------------------------------------------------------------------
#include "optix/optix_renderer.h"
#include "core/config.h"
#include "photon/specular_target.h"
#include "photon/tri_photon_irradiance.h"
#include "photon/emitter.h"
#include "photon/dense_grid.h"
#include "photon/hash_grid.h"
#include "photon/photon_bins.h"

#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <unordered_set>

#include <optix.h>
#include <optix_stubs.h>

namespace {
    constexpr float HASHGRID_CELL_FACTOR = 2.0f;
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

    // Use override radius for the hash grid if provided.
    // The member gather_radius_ drives both grid build AND fill_common_params.
    gather_radius_ = (grid_radius_override > 0.f)
                         ? grid_radius_override
                         : config.gather_radius;
    caustic_radius_ = config.caustic_radius;

    int num_photons = config.num_photons;
    int max_stored  = num_photons * DEFAULT_MAX_BOUNCES; // upper bound on stored photons
    num_photons_emitted_ = num_photons;  // record N_emitted for density normalisation

    // Estimate GPU buffer footprint (bytes per photon: 9 float pos/wi/norm)
    size_t bytes_per_photon = 9 * sizeof(float);
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
    d_out_photon_path_flags_.alloc(max_stored * sizeof(uint8_t));
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

    // ── Pre-allocate photon accumulation buffers for GPU-side grid build ──
    // These will hold the merged photon SoA from all passes (global +
    // caustic/targeted).  D→D copies populate them directly, eliminating
    // the costly H→D re-upload.
    const size_t max_total = (size_t)max_stored * 2;  // global + caustic/targeted
    d_photon_pos_x_.alloc(max_total * sizeof(float));
    d_photon_pos_y_.alloc(max_total * sizeof(float));
    d_photon_pos_z_.alloc(max_total * sizeof(float));
    d_photon_wi_x_.alloc(max_total * sizeof(float));
    d_photon_wi_y_.alloc(max_total * sizeof(float));
    d_photon_wi_z_.alloc(max_total * sizeof(float));
    d_photon_norm_x_.alloc(max_total * sizeof(float));
    d_photon_norm_y_.alloc(max_total * sizeof(float));
    d_photon_norm_z_.alloc(max_total * sizeof(float));

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

    // Per-material interior medium (§7.7 Translucent)
    lp.mat_medium_id = d_mat_medium_id_.d_ptr ? d_mat_medium_id_.as<int>() : nullptr;
    lp.media         = d_media_.d_ptr ? d_media_.as<HomogeneousMedium>() : nullptr;
    lp.num_media     = d_media_.d_ptr ? (int)(d_media_.bytes / sizeof(HomogeneousMedium)) : 0;

    lp.num_photons       = num_photons;
    lp.max_bounces       = config.max_bounces;
    lp.photon_max_bounces = DEBUG_PHOTON_SINGLE_BOUNCE ? 1 : config.max_bounces;

    // Emitter data
    lp.emissive_tri_indices = d_emissive_indices_.as<uint32_t>();
    lp.emissive_cdf         = d_emissive_cdf_.as<float>();
    lp.emissive_local_idx   = d_emissive_local_idx_.d_ptr
        ? d_emissive_local_idx_.as<int>() : nullptr;
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
    lp.out_photon_path_flags = d_out_photon_path_flags_.d_ptr
                                   ? d_out_photon_path_flags_.as<uint8_t>() : nullptr;
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
    d_launch_params_.ensure_alloc(sizeof(LaunchParams));
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

    // ── GPU-side: D→D copy global photons into accumulation buffers ──
    // The photon SoA is already on the GPU in d_out_photon_* buffers.
    // Copy directly to d_photon_* (offset 0) to avoid D→H + H→D round-trip.
    CUDA_CHECK(cudaMemcpy(d_photon_pos_x_.d_ptr,  d_out_photon_pos_x_.d_ptr,  stored_count*sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_photon_pos_y_.d_ptr,  d_out_photon_pos_y_.d_ptr,  stored_count*sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_photon_pos_z_.d_ptr,  d_out_photon_pos_z_.d_ptr,  stored_count*sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_photon_wi_x_.d_ptr,   d_out_photon_wi_x_.d_ptr,   stored_count*sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_photon_wi_y_.d_ptr,   d_out_photon_wi_y_.d_ptr,   stored_count*sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_photon_wi_z_.d_ptr,   d_out_photon_wi_z_.d_ptr,   stored_count*sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_photon_norm_x_.d_ptr, d_out_photon_norm_x_.d_ptr, stored_count*sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_photon_norm_y_.d_ptr, d_out_photon_norm_y_.d_ptr, stored_count*sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_photon_norm_z_.d_ptr, d_out_photon_norm_z_.d_ptr, stored_count*sizeof(float), cudaMemcpyDeviceToDevice));

    // Download photon data to CPU (still needed for diagnostics, irradiance
    // heatmap, and k-NN adaptive radius).
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

    // Download per-photon path flags (PHOTON_FLAG_* bits for F2 debug overlay)
    if (d_out_photon_path_flags_.d_ptr) {
        stored_photons_.path_flags.resize(stored_count);
        CUDA_CHECK(cudaMemcpy(stored_photons_.path_flags.data(), d_out_photon_path_flags_.d_ptr, stored_count*sizeof(uint8_t), cudaMemcpyDeviceToHost));
    }

    // Download per-photon caustic flags (needed for progress snapshot caustic PNG)
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

            // D→D copy caustic photons into accumulation buffers at offset
            {
                size_t boff  = global_count;
                auto   d2d   = cudaMemcpyDeviceToDevice;
                CUDA_CHECK(cudaMemcpy((float*)d_photon_pos_x_.d_ptr  + boff, d_out_photon_pos_x_.d_ptr,  caustic_stored*sizeof(float), d2d));
                CUDA_CHECK(cudaMemcpy((float*)d_photon_pos_y_.d_ptr  + boff, d_out_photon_pos_y_.d_ptr,  caustic_stored*sizeof(float), d2d));
                CUDA_CHECK(cudaMemcpy((float*)d_photon_pos_z_.d_ptr  + boff, d_out_photon_pos_z_.d_ptr,  caustic_stored*sizeof(float), d2d));
                CUDA_CHECK(cudaMemcpy((float*)d_photon_wi_x_.d_ptr   + boff, d_out_photon_wi_x_.d_ptr,   caustic_stored*sizeof(float), d2d));
                CUDA_CHECK(cudaMemcpy((float*)d_photon_wi_y_.d_ptr   + boff, d_out_photon_wi_y_.d_ptr,   caustic_stored*sizeof(float), d2d));
                CUDA_CHECK(cudaMemcpy((float*)d_photon_wi_z_.d_ptr   + boff, d_out_photon_wi_z_.d_ptr,   caustic_stored*sizeof(float), d2d));
                CUDA_CHECK(cudaMemcpy((float*)d_photon_norm_x_.d_ptr + boff, d_out_photon_norm_x_.d_ptr, caustic_stored*sizeof(float), d2d));
                CUDA_CHECK(cudaMemcpy((float*)d_photon_norm_y_.d_ptr + boff, d_out_photon_norm_y_.d_ptr, caustic_stored*sizeof(float), d2d));
                CUDA_CHECK(cudaMemcpy((float*)d_photon_norm_z_.d_ptr + boff, d_out_photon_norm_z_.d_ptr, caustic_stored*sizeof(float), d2d));
            }

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

                    // D→D copy targeted photons into accumulation buffers at offset
                    {
                        size_t boff = stored_count;
                        auto   d2d  = cudaMemcpyDeviceToDevice;
                        CUDA_CHECK(cudaMemcpy((float*)d_photon_pos_x_.d_ptr  + boff, d_out_photon_pos_x_.d_ptr,  targeted_stored*sizeof(float), d2d));
                        CUDA_CHECK(cudaMemcpy((float*)d_photon_pos_y_.d_ptr  + boff, d_out_photon_pos_y_.d_ptr,  targeted_stored*sizeof(float), d2d));
                        CUDA_CHECK(cudaMemcpy((float*)d_photon_pos_z_.d_ptr  + boff, d_out_photon_pos_z_.d_ptr,  targeted_stored*sizeof(float), d2d));
                        CUDA_CHECK(cudaMemcpy((float*)d_photon_wi_x_.d_ptr   + boff, d_out_photon_wi_x_.d_ptr,   targeted_stored*sizeof(float), d2d));
                        CUDA_CHECK(cudaMemcpy((float*)d_photon_wi_y_.d_ptr   + boff, d_out_photon_wi_y_.d_ptr,   targeted_stored*sizeof(float), d2d));
                        CUDA_CHECK(cudaMemcpy((float*)d_photon_wi_z_.d_ptr   + boff, d_out_photon_wi_z_.d_ptr,   targeted_stored*sizeof(float), d2d));
                        CUDA_CHECK(cudaMemcpy((float*)d_photon_norm_x_.d_ptr + boff, d_out_photon_norm_x_.d_ptr, targeted_stored*sizeof(float), d2d));
                        CUDA_CHECK(cudaMemcpy((float*)d_photon_norm_y_.d_ptr + boff, d_out_photon_norm_y_.d_ptr, targeted_stored*sizeof(float), d2d));
                        CUDA_CHECK(cudaMemcpy((float*)d_photon_norm_z_.d_ptr + boff, d_out_photon_norm_z_.d_ptr, targeted_stored*sizeof(float), d2d));
                    }

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

    // ── Reconstruct spectral_flux from hero wavelength data ─────────
    // The GPU emits photons with HERO_WAVELENGTHS per-hero (lambda_bin,
    // flux) pairs.  The CPU-side spectral_flux array (NUM_LAMBDA floats
    // per photon) is not populated by the GPU download, so we rebuild it
    // here so that total_flux() / get_flux() return correct values for
    // debug overlays (F1/F2 photon visualisation, heatmap, etc.).
    {
        const size_t n = stored_photons_.size();
        // spectral_flux was zero-initialised by resize(); splat hero data
        for (size_t i = 0; i < n; ++i) {
            int nh = stored_photons_.num_hero[i];
            for (int h = 0; h < nh; ++h) {
                uint16_t bin = stored_photons_.lambda_bin[i * HERO_WAVELENGTHS + h];
                float    f   = stored_photons_.flux[i * HERO_WAVELENGTHS + h];
                if (bin < NUM_LAMBDA)
                    stored_photons_.spectral_flux[i * NUM_LAMBDA + bin] += f;
            }
        }
    }

    // ── GPU-side caustic tag computation ──────────────────────────────
    // Build 3-valued tags directly on GPU, avoiding CPU loop + upload.
    // Uses d_out_photon_is_caustic_ from the global pass (already on GPU
    // in the first global_count entries of d_photon_* accumulation buffers).
    // NOTE: The d_out_photon_is_caustic_ buffer was written by the global
    // pass and may have been overwritten by caustic/targeted passes, but
    // we already saved a copy to d_photon_* offset 0 via D→D copy.
    // We still build CPU tags for diagnostics.
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
        std::printf("[OptiX] Tag distribution: tag0=%zu (noncaustic)  tag1=%zu (global-caustic)  "
                    "tag2=%zu (targeted-caustic)  total=%u\n",
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

    // ── Build dense grid on CPU from downloaded photon SoA ───────────
    {
        DenseGridData dg = build_dense_grid(stored_photons_, DENSE_GRID_CELL_SIZE);
        d_dense_sorted_indices_.upload(dg.sorted_indices);
        d_dense_cell_start_.upload(dg.cell_start);
        d_dense_cell_end_.upload(dg.cell_end);

        size_t grid_mem_kb = (dg.sorted_indices.size() * sizeof(uint32_t)
                             + dg.cell_start.size() * sizeof(uint32_t) * 2) / 1024;
        auto t_now = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_now - t_lap).count();
        std::printf("[Timing] Dense grid build:  %8.1f ms  "
                    "%dx%dx%d = %d cells  mem=%zu KB\n",
                    ms, dg.dim_x, dg.dim_y, dg.dim_z, dg.total_cells(),
                    grid_mem_kb);
        t_lap = t_now;
        stored_dense_grid_ = std::move(dg);
    }

    // Build and upload per-triangle photon irradiance heatmap (for preview)
    {
        auto irr = build_tri_photon_irradiance(stored_photons_, num_tris_);
        if (!irr.empty())
            d_tri_photon_irradiance_.upload(irr);
        else
            d_tri_photon_irradiance_.free();
    }

    std::cout << "[OptiX] GPU grid + photon data ready on device\n";

    {
        auto t_now = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_now - t_lap).count();
        std::printf("[Timing] Irradiance build:  %8.1f ms\n", ms);
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

            // ── VP-02: Build 3D hash grid for volume photon kNN ──────
            volume_grid_.build(volume_photons_, gather_radius_);
            std::printf("[OptiX] Volume hash grid:  %u photons  table=%u\n",
                        vol_count, volume_grid_.table_size);

            // ── VP-03: Build volume CellBinGrid (directional histograms) ──
            // Precompute bin_idx for volume photons
            {
                PhotonBinDirs vbin_dirs;
                vbin_dirs.init(PHOTON_BIN_COUNT);
                volume_photons_.bin_idx.resize(vol_count);
                for (size_t vi = 0; vi < vol_count; ++vi) {
                    float3 wi = make_f3(volume_photons_.wi_x[vi],
                                        volume_photons_.wi_y[vi],
                                        volume_photons_.wi_z[vi]);
                    volume_photons_.bin_idx[vi] = (uint8_t)vbin_dirs.find_nearest(wi);
                }
            }
            vol_cell_bin_grid_.build_volume(volume_photons_, gather_radius_, PHOTON_BIN_COUNT);

            // ── VP-07: Upload volume cell-bin grid to GPU ────────────
            if (vol_cell_bin_grid_.total_cells() > 0 && !vol_cell_bin_grid_.bins.empty()) {
                d_vol_cell_bin_grid_.upload(vol_cell_bin_grid_.bins);
                std::printf("[OptiX] Volume cell-bin grid uploaded:  %d cells  %.2f MB\n",
                            vol_cell_bin_grid_.total_cells(),
                            (double)(vol_cell_bin_grid_.bins.size() * sizeof(PhotonBin)) / (1024.0 * 1024.0));
            } else {
                d_vol_cell_bin_grid_.free();
            }

            // ── MT-07: Upload volume photon kNN data to GPU ─────────
            // Re-use the output buffers that still hold the photon data
            // on GPU, but also upload hash grid index arrays.
            d_vol_photon_pos_x_.upload(volume_photons_.pos_x);
            d_vol_photon_pos_y_.upload(volume_photons_.pos_y);
            d_vol_photon_pos_z_.upload(volume_photons_.pos_z);
            d_vol_photon_wi_x_.upload(volume_photons_.wi_x);
            d_vol_photon_wi_y_.upload(volume_photons_.wi_y);
            d_vol_photon_wi_z_.upload(volume_photons_.wi_z);
            d_vol_photon_lambda_.upload(volume_photons_.lambda_bin);
            d_vol_photon_flux_.upload(volume_photons_.flux);
            d_vol_grid_sorted_indices_.upload(volume_grid_.sorted_indices);
            d_vol_grid_cell_start_.upload(volume_grid_.cell_start);
            d_vol_grid_cell_end_.upload(volume_grid_.cell_end);
            std::printf("[OptiX] Volume photon kNN data uploaded:  %u photons  table=%u\n",
                        vol_count, volume_grid_.table_size);

            {
                auto t_now = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t_now - t_lap).count();
                std::printf("[Timing] Volume grid build: %8.1f ms\n", ms);
                t_lap = t_now;
            }
        }
    }

    {
        auto t_now = std::chrono::high_resolution_clock::now();
        double total = std::chrono::duration<double, std::milli>(t_now - t_phase_start).count();
        std::printf("[Timing] Photon total:      %8.1f ms\n", total);
    }

    // Dense grid replaces hash grid + kNN + cell analysis.
    // No CPU hash grid build or bin_idx precompute needed.
}