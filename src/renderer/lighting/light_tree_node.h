#pragma once
// light_tree_node.h — GPU-compatible light BVH node (POD)
//
// Shared between CPU builder (light_tree.h) and GPU traversal
// (light_tree_device.cuh).  Included by launch_params.h.

#include "core/types.h"

struct LightTreeNode {
    float3 bbox_lo;       // AABB min
    float3 bbox_hi;       // AABB max
    float  flux;          // aggregate emissive power (Σ area × Le)
    float  cos_theta_o;   // cone: cos(half-angle) of normals, -1 = omni
    float3 axis;          // cone: flux-weighted average normal (unit)
    int    child_left;    // interior: left child index; leaf: first tri_order index
    int    child_right;   // interior: right child index; leaf: -1
    int    tri_count;     // 0 = interior, >0 = leaf
};
