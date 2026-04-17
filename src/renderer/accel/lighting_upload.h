#pragma once
// ─────────────────────────────────────────────────────────────────────
// lighting_upload.h – Upload emissive data to GPU (v5)
//
// Fills LaunchParams §7 (emitter data).
// Pure CUDA upload — no OptiX dependency.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/device_buffer.h"
#include "accel/launch_params.h"
#include "lighting/light_tree.h"
#include "scene/scene.h"

class LightingUploader {
public:
    // Upload emissive triangle distribution to device.
    // Must be called after Scene::build_emissive_distribution().
    void upload_emissives(const Scene& scene) {
        int num_emissive = (int)scene.emissive_tri_indices.size();
        if (num_emissive == 0) {
            std::printf("[LightingUploader] No emissive triangles to upload\n");
            return;
        }

        // Emissive triangle indices
        d_emissive_tri_indices_.upload(scene.emissive_tri_indices);

        // Build CDF from alias table pdf_values (cumulative)
        std::vector<float> cdf(num_emissive);
        float running = 0.f;
        for (int i = 0; i < num_emissive; ++i) {
            running += scene.emissive_alias_table.pdf_values[i];
            cdf[i] = running;
        }
        // Ensure last entry is exactly 1.0
        if (num_emissive > 0) cdf[num_emissive - 1] = 1.0f;
        d_emissive_cdf_.upload(cdf);

        // Build inverse lookup: triangle_id → local emissive index (-1 = not)
        int num_tris = (int)scene.triangles.size();
        std::vector<int> local_idx(num_tris, -1);
        for (int i = 0; i < num_emissive; ++i) {
            uint32_t tri = scene.emissive_tri_indices[i];
            if (tri < (uint32_t)num_tris)
                local_idx[tri] = i;
        }
        d_emissive_local_idx_.upload(local_idx);

        total_emissive_power_ = scene.total_emissive_power;
        num_emissive_ = num_emissive;

        std::printf("[LightingUploader] Emissives uploaded: %d tris, power=%.4f\n",
                    num_emissive, total_emissive_power_);
    }

    // Fill LaunchParams §7 + §7a
    void fill_params(LaunchParams& lp) const {
        // §7 Emitter data
        lp.emissive_tri_indices = d_emissive_tri_indices_.empty() ? nullptr : const_cast<uint32_t*>(d_emissive_tri_indices_.data());
        lp.emissive_cdf         = d_emissive_cdf_.empty() ? nullptr : const_cast<float*>(d_emissive_cdf_.data());
        lp.emissive_local_idx   = d_emissive_local_idx_.empty() ? nullptr : const_cast<int*>(d_emissive_local_idx_.data());
        lp.num_emissive         = num_emissive_;
        lp.total_emissive_power = total_emissive_power_;

        // §7a Light tree
        lp.light_tree_nodes     = d_light_tree_nodes_.empty() ? nullptr : const_cast<LightTreeNode*>(d_light_tree_nodes_.data());
        lp.light_tree_tri_order = d_light_tree_tri_order_.empty() ? nullptr : const_cast<uint32_t*>(d_light_tree_tri_order_.data());
        lp.num_light_tree_nodes = num_light_tree_nodes_;
        lp.light_tree_root      = light_tree_root_;
        lp.light_tree_enabled   = light_tree_enabled_ ? 1 : 0;
    }

    // Build and upload light tree.  Call after upload_emissives().
    void upload_light_tree(const Scene& scene, int max_leaf_size = 4) {
        LightTree tree;
        tree.build(scene, max_leaf_size);
        if (tree.nodes.empty()) {
            light_tree_enabled_ = false;
            return;
        }
        d_light_tree_nodes_.upload(tree.nodes);
        d_light_tree_tri_order_.upload(tree.tri_order);
        num_light_tree_nodes_ = (int)tree.nodes.size();
        light_tree_root_ = tree.root;
        light_tree_enabled_ = true;
    }

private:
    // Emissive
    DeviceBuffer<uint32_t> d_emissive_tri_indices_;
    DeviceBuffer<float>    d_emissive_cdf_;
    DeviceBuffer<int>      d_emissive_local_idx_;
    int   num_emissive_ = 0;
    float total_emissive_power_ = 0.f;

    // Light tree
    DeviceBuffer<LightTreeNode> d_light_tree_nodes_;
    DeviceBuffer<uint32_t>      d_light_tree_tri_order_;
    int  num_light_tree_nodes_ = 0;
    int  light_tree_root_ = 0;
    bool light_tree_enabled_ = false;
};
