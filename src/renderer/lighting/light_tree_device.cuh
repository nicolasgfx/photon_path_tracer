#ifndef PPT_LIGHTING_LIGHT_TREE_DEVICE_CUH
#define PPT_LIGHTING_LIGHT_TREE_DEVICE_CUH
// light_tree_device.cuh - GPU stochastic light tree traversal
//
// Top-down traversal using importance-based child selection.
// At each interior node the traversal probabilistically picks the
// child with higher estimated contribution to the shading point,
// accumulating the path probability for correct MIS.
//
// Reference: Estevez & Kulla, HPG 2018

#include "accel/launch_params.h"
#include "core/random.h"

// Forward: extern params from optix_utils.cuh
extern "C" { extern __constant__ LaunchParams params; }

// ---- Importance bound for a node seen from shading point ----

__forceinline__ __device__
float dev_light_tree_importance(const LightTreeNode& node, float3 pos) {
    float3 clamped;
    clamped.x = fmaxf(node.bbox_lo.x, fminf(pos.x, node.bbox_hi.x));
    clamped.y = fmaxf(node.bbox_lo.y, fminf(pos.y, node.bbox_hi.y));
    clamped.z = fmaxf(node.bbox_lo.z, fminf(pos.z, node.bbox_hi.z));
    float3 diff = pos - clamped;
    float dist2 = fmaxf(dot(diff, diff), 1e-4f);

    float geo_factor = 1.f;

    // orientation cone: soft falloff when cone faces mostly away
    if (node.cos_theta_o > -0.99f) {
        float3 center = (node.bbox_lo + node.bbox_hi) * 0.5f;
        float3 to_pos = pos - center;
        float d = length(to_pos);
        if (d > 1e-6f) {
            float cos_dir = dot(node.axis, to_pos * (-1.f / d));
            if (cos_dir < node.cos_theta_o) {
                float overshoot = node.cos_theta_o - cos_dir;
                geo_factor = fmaxf(1.f - overshoot * 2.f, 0.01f);
            }
        }
    }

    return node.flux * geo_factor / dist2;
}

// ---- Compute child selection probability for an interior node ----

__forceinline__ __device__
float dev_light_tree_p_left(const LightTreeNode* nodes,
                            int left, int right, float3 pos) {
    float imp_l = dev_light_tree_importance(nodes[left], pos);
    float imp_r = dev_light_tree_importance(nodes[right], pos);
    float total = imp_l + imp_r;

    if (total <= 0.f) {
        imp_l = nodes[left].flux;
        imp_r = nodes[right].flux;
        total = imp_l + imp_r;
        if (total <= 0.f) return 0.5f;
    }

    return fmaxf(fminf(imp_l / total, 0.999f), 0.001f);
}

// ---- AABB centroid containment test ----

__forceinline__ __device__
bool dev_aabb_contains(const LightTreeNode& node, float3 centroid) {
    return centroid.x >= node.bbox_lo.x && centroid.x <= node.bbox_hi.x &&
           centroid.y >= node.bbox_lo.y && centroid.y <= node.bbox_hi.y &&
           centroid.z >= node.bbox_lo.z && centroid.z <= node.bbox_hi.z;
}

// ---- Stochastic traversal: select one emissive triangle ----

__forceinline__ __device__
int dev_light_tree_sample(float3 pos, PCGRng& rng, float& pdf_out) {
    const LightTreeNode* nodes = params.light_tree_nodes;
    int node_idx = params.light_tree_root;
    float pdf = 1.f;

    for (int depth = 0; depth < 64; ++depth) {
        const LightTreeNode& node = nodes[node_idx];

        if (node.tri_count > 0) {
            int pick = 0;
            if (node.tri_count > 1) {
                pick = (int)(rng.next_float() * node.tri_count);
                if (pick >= node.tri_count) pick = node.tri_count - 1;
            }
            pdf *= 1.f / (float)node.tri_count;
            pdf_out = pdf;
            return (int)params.light_tree_tri_order[node.child_left + pick];
        }

        float p_left = dev_light_tree_p_left(nodes,
            node.child_left, node.child_right, pos);

        if (rng.next_float() < p_left) {
            node_idx = node.child_left;
            pdf *= p_left;
        } else {
            node_idx = node.child_right;
            pdf *= (1.f - p_left);
        }
    }

    pdf_out = 1.f;
    return (int)params.light_tree_tri_order[0];
}

// ---- PDF within a subtree (iterative, used for overlap fallback) ----

__forceinline__ __device__
float dev_light_tree_pdf_in(const LightTreeNode* nodes, int node_idx,
                            float3 pos, float3 centroid, uint32_t tri_id) {
    float pdf = 1.f;
    for (int d = 0; d < 64; ++d) {
        const LightTreeNode& node = nodes[node_idx];
        if (node.tri_count > 0) {
            for (int i = 0; i < node.tri_count; ++i) {
                if (params.light_tree_tri_order[node.child_left + i] == tri_id)
                    return pdf / (float)node.tri_count;
            }
            return 0.f;
        }
        float p_l = dev_light_tree_p_left(nodes,
            node.child_left, node.child_right, pos);

        if (dev_aabb_contains(nodes[node.child_left], centroid)) {
            pdf *= p_l;
            node_idx = node.child_left;
        } else {
            pdf *= (1.f - p_l);
            node_idx = node.child_right;
        }
    }
    return 0.f;
}

// ---- PDF evaluation for a known emissive triangle ----
// Used for MIS when a BSDF-sampled ray hits an emissive triangle.

__forceinline__ __device__
float dev_light_tree_pdf(float3 pos, uint32_t global_tri_id) {
    const LightTreeNode* nodes = params.light_tree_nodes;
    int node_idx = params.light_tree_root;
    float pdf = 1.f;

    float3 v0 = params.vertices[global_tri_id * 3 + 0];
    float3 v1 = params.vertices[global_tri_id * 3 + 1];
    float3 v2 = params.vertices[global_tri_id * 3 + 2];
    float3 centroid = (v0 + v1 + v2) * (1.f / 3.f);

    for (int depth = 0; depth < 64; ++depth) {
        const LightTreeNode& node = nodes[node_idx];

        if (node.tri_count > 0) {
            for (int i = 0; i < node.tri_count; ++i) {
                if (params.light_tree_tri_order[node.child_left + i] == global_tri_id)
                    return pdf / (float)node.tri_count;
            }
            return 0.f;
        }

        float p_left = dev_light_tree_p_left(nodes,
            node.child_left, node.child_right, pos);

        bool in_left  = dev_aabb_contains(nodes[node.child_left], centroid);
        bool in_right = dev_aabb_contains(nodes[node.child_right], centroid);

        if (in_left && !in_right) {
            pdf *= p_left;
            node_idx = node.child_left;
        } else if (in_right && !in_left) {
            pdf *= (1.f - p_left);
            node_idx = node.child_right;
        } else if (in_left) {
            // in both - check left subtree first
            float pdf_l = dev_light_tree_pdf_in(
                nodes, node.child_left, pos, centroid, global_tri_id);
            if (pdf_l > 0.f) return pdf * p_left * pdf_l;
            pdf *= (1.f - p_left);
            node_idx = node.child_right;
        } else {
            return 0.f;
        }
    }
    return 0.f;
}

#endif // PPT_LIGHTING_LIGHT_TREE_DEVICE_CUH
