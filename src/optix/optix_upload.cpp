// ---------------------------------------------------------------------
// optix_upload.cpp -- Scene / photon / emitter data upload to device
// ---------------------------------------------------------------------
// Extracted from optix_renderer.cpp (§1.8):
//   upload_scene_data(), fill_clearcoat_fabric_params(),
//   upload_photon_data(), upload_emitter_data()
// ---------------------------------------------------------------------
#include "optix/optix_renderer.h"
#include "photon/dense_grid.h"

#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <cstdio>
#include <numeric>
#include <algorithm>
#include <iostream>

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

    // Per-material interior medium id (-1 = no medium)
    std::vector<int> medium_ids(num_mats);
    for (size_t m = 0; m < num_mats; ++m)
        medium_ids[m] = scene.materials[m].medium_id;
    d_mat_medium_id_.upload(medium_ids);

    // Interior media table (HomogeneousMedium array, indexed by medium_id)
    if (!scene.media.empty()) {
        d_media_.upload(scene.media);
        std::printf("[OptiX] Uploaded %zu interior media\n", scene.media.size());
    } else {
        d_media_.free();
    }

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
        d_mat_medium_id_.bytes + d_media_.bytes +
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
    // Emissive inverse-index (O(1) light PDF lookup)
    lp.emissive_local_idx  = d_emissive_local_idx_.d_ptr  ? const_cast<int*>(d_emissive_local_idx_.as<int>())      : nullptr;

    // Per-material interior medium (§7.7 Translucent)
    lp.mat_medium_id = d_mat_medium_id_.d_ptr ? const_cast<int*>(d_mat_medium_id_.as<int>()) : nullptr;
    lp.media         = d_media_.d_ptr ? const_cast<HomogeneousMedium*>(d_media_.as<HomogeneousMedium>()) : nullptr;
    lp.num_media     = d_media_.d_ptr ? (int)(d_media_.bytes / sizeof(HomogeneousMedium)) : 0;
}

// =====================================================================
// upload_photon_data() -- Upload CPU-side photon map to device
// =====================================================================
void OptixRenderer::upload_photon_data(
    const PhotonSoA& global_photons,
    const PhotonSoA& /*caustic_photons*/,
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

    std::cout << "[OptiX] Uploaded " << global_photons.size()
              << " photons to device\n";
    gather_radius_ = gather_radius;

    // Copy into stored_photons_
    stored_photons_ = global_photons;

    // ── Build and upload dense grid ───────────────────────────────────
    {
        DenseGridData dg = build_dense_grid(stored_photons_, DENSE_GRID_CELL_SIZE);
        d_dense_sorted_indices_.upload(dg.sorted_indices);
        d_dense_cell_start_.upload(dg.cell_start);
        d_dense_cell_end_.upload(dg.cell_end);
        std::printf("[OptiX] Dense grid: %dx%dx%d = %d cells  (%zu photons)\n",
                    dg.dim_x, dg.dim_y, dg.dim_z, dg.total_cells(),
                    dg.sorted_indices.size());
        stored_dense_grid_ = std::move(dg);
    }
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

    // Build inverse-index: global tri_id → local emissive index (-1 = not emissive)
    std::vector<int> local_idx(scene.triangles.size(), -1);
    for (size_t i = 0; i < n; ++i)
        local_idx[scene.emissive_tri_indices[i]] = (int)i;
    d_emissive_local_idx_.upload(local_idx);

    std::printf("[OptiX] Emitter data: %zu tris  total_power=%.4f  max_w=%.4f  min_w=%.4f\n",
                n, total,
                *std::max_element(weights.begin(), weights.end()),
                *std::min_element(weights.begin(), weights.end()));
}


