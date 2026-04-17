#pragma once
// sd_tree.h — CPU spatial-directional tree (Müller 2017)
//
// Binary KD-tree with adaptive split-axis selection and a directional quadtree (DQuad)
// at each leaf.  Trained from camera-path samples using geometric
// doubling.  Flattened to GPU-ready arrays for device-side sampling.

#include "guide/sd_tree_gpu.h"
#include "guide/dquad_gpu.h"
#include "core/types.h"
#include "core/config.h"

#include <vector>
#include <cstdint>

class SDTree {
public:
    // --- GPU upload payload
    struct FlatData {
        std::vector<SDNodeGPU>    sd_nodes;
        std::vector<DQuadNodeGPU> dquad_nodes;
        float3 bbox_min;
        float3 bbox_max;
    };

    SDTree() = default;

    // initialize root spanning [lo, hi]
    void init(float3 lo, float3 hi);

    // deposit a training sample (position, direction, weight)
    void deposit(float3 pos, float3 dir, float weight);

    // spatial refinement: median split on leaves exceeding threshold
    // threshold = SD_TREE_SPLIT_THRESHOLD * sqrt(2^training_iter)
    void refine_spatial(int training_iter);

    // directional refinement: subdivide high-variance DQuad bins
    void refine_directional();

    // zero all flux counters, keep tree structure
    void reset_flux();

    // serialize to GPU-ready flat arrays
    FlatData flatten() const;

    // diagnostics
    int num_nodes()  const { return (int)nodes_.size(); }
    int num_leaves() const;
    int total_dquad_nodes() const;

    bool is_valid() const { return !nodes_.empty(); }

private:
    // --- cylindrical equal-area mapping
    static void dir_to_uv(float3 d, float& u, float& v);

    // --- internal node
    struct Node {
        bool  is_leaf  = true;
        int   axis     = 0;      // 0=x, 1=y, 2=z
        float split_pos = 0.f;
        int   left     = -1;
        int   right    = -1;
        int   depth    = 0;
        float3 cell_min = {};    // spatial cell bounds (for midpoint fallback)
        float3 cell_max = {};

        // leaf data: DQuad flux array (complete tree, DQUAD_NODES_PER_LEAF entries)
        std::vector<float> dquad_flux;
        int num_deposits = 0;

        // full sample positions for adaptive split-axis selection
        std::vector<float3> sample_positions;
    };

    // find leaf containing position
    int find_leaf(float3 pos) const;

    // map position to leaf, clamped to bbox
    int find_leaf_clamped(float3 pos) const;

    // deposit into a DQuad: traverse by UV, accumulate flux
    void dquad_deposit(std::vector<float>& dquad, float u, float v, float weight);

    // split a leaf node into two children (median or midpoint)
    bool split_leaf(int idx);

    // flatten one DQuad leaf array → GPU DQuadNodeGPU array
    void flatten_dquad(const std::vector<float>& flux,
                       std::vector<DQuadNodeGPU>& out) const;

    // recursive flatten for spatial tree
    void flatten_node(int idx, std::vector<SDNodeGPU>& sd_out,
                      std::vector<DQuadNodeGPU>& dq_out,
                      std::vector<int>& leaf_dquad_offsets) const;

    std::vector<Node> nodes_;
    float3 bbox_min_ = {};
    float3 bbox_max_ = {};
};
