// sd_tree.cpp — CPU spatial-directional tree implementation (Müller 2017)
//
// Binary KD-tree with adaptive split-axis selection, median splitting, and a
// fixed-depth directional quadtree (DQuad) at each leaf.
// Trained from camera-path samples.  Serialized to GPU via flatten().

#include "guide/sd_tree.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

// -----------------------------------------------
// cylindrical equal-area sphere → UV mapping
// -----------------------------------------------

void SDTree::dir_to_uv(float3 d, float& u, float& v) {
    // u = phi / (2*pi),  v = (1 - cos_theta) / 2
    float cos_theta = fmaxf(-1.f, fminf(1.f, d.z));
    float phi = atan2f(d.y, d.x);
    if (phi < 0.f) phi += 2.f * 3.14159265358979f;
    u = phi / (2.f * 3.14159265358979f);
    v = (1.f - cos_theta) * 0.5f;
    u = fmaxf(0.f, fminf(u, 0.99999f));
    v = fmaxf(0.f, fminf(v, 0.99999f));
}

// -----------------------------------------------
// init / deposit / find_leaf
// -----------------------------------------------

void SDTree::init(float3 lo, float3 hi) {
    nodes_.clear();
    bbox_min_ = lo;
    bbox_max_ = hi;

    // root node: leaf spanning entire scene
    Node root;
    root.is_leaf  = true;
    root.axis     = 0;
    root.depth    = 0;
    root.cell_min = lo;
    root.cell_max = hi;
    root.dquad_flux.assign(DQUAD_NODES_PER_LEAF, 0.f);
    nodes_.push_back(std::move(root));
}

int SDTree::find_leaf(float3 pos) const {
    int idx = 0;
    while (!nodes_[idx].is_leaf) {
        const Node& n = nodes_[idx];
        float val = (n.axis == 0) ? pos.x : (n.axis == 1) ? pos.y : pos.z;
        idx = (val < n.split_pos) ? n.left : n.right;
    }
    return idx;
}

int SDTree::find_leaf_clamped(float3 pos) const {
    // clamp to bbox with small inset to avoid boundary issues
    constexpr float EPS = 1e-6f;
    pos.x = fmaxf(bbox_min_.x + EPS, fminf(pos.x, bbox_max_.x - EPS));
    pos.y = fmaxf(bbox_min_.y + EPS, fminf(pos.y, bbox_max_.y - EPS));
    pos.z = fmaxf(bbox_min_.z + EPS, fminf(pos.z, bbox_max_.z - EPS));
    return find_leaf(pos);
}

void SDTree::deposit(float3 pos, float3 dir, float weight) {
    if (nodes_.empty()) return;
    int leaf = find_leaf_clamped(pos);
    Node& n = nodes_[leaf];

    float u, v;
    dir_to_uv(dir, u, v);
    dquad_deposit(n.dquad_flux, u, v, weight);
    n.num_deposits++;

    // keep the full sample position so planar leaves can split on their tangential axes
    n.sample_positions.push_back(pos);
}

// -----------------------------------------------
// DQuad deposit: traverse complete quad tree by UV
// -----------------------------------------------

void SDTree::dquad_deposit(std::vector<float>& dquad, float u, float v, float weight) {
    // traverse the complete quad tree: node i has children 4i+1 .. 4i+4
    // quadrants: (u < mid, v < mid)=0, (u >= mid, v < mid)=1,
    //            (u < mid, v >= mid)=2, (u >= mid, v >= mid)=3
    int idx = 0;
    float u0 = 0.f, u1 = 1.f, v0 = 0.f, v1 = 1.f;

    for (int d = 0; d < SD_TREE_DQUAD_DEPTH; ++d) {
        dquad[idx] += weight;  // accumulate flux at each level

        float u_mid = (u0 + u1) * 0.5f;
        float v_mid = (v0 + v1) * 0.5f;

        int child;
        if (u < u_mid) {
            if (v < v_mid) { child = 0; u1 = u_mid; v1 = v_mid; }
            else           { child = 2; u1 = u_mid; v0 = v_mid; }
        } else {
            if (v < v_mid) { child = 1; u0 = u_mid; v1 = v_mid; }
            else           { child = 3; u0 = u_mid; v0 = v_mid; }
        }
        idx = 4 * idx + 1 + child;
    }
    dquad[idx] += weight;  // leaf level
}

// -----------------------------------------------
// split a single leaf into two children
// -----------------------------------------------

bool SDTree::split_leaf(int idx) {
    constexpr float MIN_AXIS_SPAN = 1e-6f;

    // reserve to prevent push_back from invalidating indices
    nodes_.reserve(nodes_.size() + 2);

    Node& node = nodes_[idx];
    if (!node.is_leaf) return false;

    const auto axis_value = [](float3 p, int axis) {
        return (axis == 0) ? p.x : (axis == 1) ? p.y : p.z;
    };

    const auto cell_bound = [&](int axis, bool upper) {
        const float3 bound = upper ? node.cell_max : node.cell_min;
        return axis_value(bound, axis);
    };

    int axis = -1;
    float best_spread = 0.f;
    if (!node.sample_positions.empty()) {
        for (int offset = 0; offset < 3; ++offset) {
            int candidate_axis = (node.axis + offset) % 3;
            float min_v = axis_value(node.sample_positions[0], candidate_axis);
            float max_v = min_v;
            for (size_t i = 1; i < node.sample_positions.size(); ++i) {
                float v = axis_value(node.sample_positions[i], candidate_axis);
                min_v = fminf(min_v, v);
                max_v = fmaxf(max_v, v);
            }

            float spread = max_v - min_v;
            if (spread > best_spread + MIN_AXIS_SPAN) {
                best_spread = spread;
                axis = candidate_axis;
            }
        }
    }

    // if the deposited samples are degenerate, fall back to the widest cell extent
    if (axis < 0) {
        float best_extent = 0.f;
        for (int offset = 0; offset < 3; ++offset) {
            int candidate_axis = (node.axis + offset) % 3;
            float extent = cell_bound(candidate_axis, true) - cell_bound(candidate_axis, false);
            if (extent > best_extent + MIN_AXIS_SPAN) {
                best_extent = extent;
                axis = candidate_axis;
            }
        }
        if (axis < 0) return false;
    }

    const int depth = node.depth;
    const float lo = cell_bound(axis, false);
    const float hi = cell_bound(axis, true);
    if (hi - lo <= MIN_AXIS_SPAN) return false;

    // compute median split position on the selected axis, with midpoint fallback near boundaries
    float split = 0.5f * (lo + hi);
    if (!node.sample_positions.empty() && best_spread > MIN_AXIS_SPAN) {
        std::vector<float> projected;
        projected.reserve(node.sample_positions.size());
        for (const float3 sample_pos : node.sample_positions)
            projected.push_back(axis_value(sample_pos, axis));

        auto mid = projected.begin() + projected.size() / 2;
        std::nth_element(projected.begin(), mid, projected.end());
        float median = *mid;
        if (median > lo + MIN_AXIS_SPAN && median < hi - MIN_AXIS_SPAN)
            split = median;
    }
    if (split <= lo + MIN_AXIS_SPAN || split >= hi - MIN_AXIS_SPAN) return false;

    std::vector<float3> left_positions;
    std::vector<float3> right_positions;
    left_positions.reserve(node.sample_positions.size());
    right_positions.reserve(node.sample_positions.size());
    for (const float3 sample_pos : node.sample_positions) {
        if (axis_value(sample_pos, axis) < split) left_positions.push_back(sample_pos);
        else                                      right_positions.push_back(sample_pos);
    }

    const int left_count = (int)left_positions.size();
    const int right_count = (int)right_positions.size();
    if (left_count == 0 || right_count == 0) return false;

    const int total_count = left_count + right_count;
    const float left_ratio =
        (total_count > 0) ? ((float)left_count / (float)total_count) : 0.5f;
    const float right_ratio =
        (total_count > 0) ? ((float)right_count / (float)total_count) : 0.5f;

    // build children with inherited cell bounds, narrowed on the split axis
    float3 left_max = node.cell_max;
    float3 right_min = node.cell_min;
    if (axis == 0)      { left_max.x = split; right_min.x = split; }
    else if (axis == 1) { left_max.y = split; right_min.y = split; }
    else                { left_max.z = split; right_min.z = split; }

    Node left_child;
    left_child.is_leaf  = true;
    left_child.depth    = depth + 1;
    left_child.axis     = (axis + 1) % 3;
    left_child.cell_min = node.cell_min;
    left_child.cell_max = left_max;
    left_child.dquad_flux.assign(DQUAD_NODES_PER_LEAF, 0.f);
    left_child.num_deposits = left_count;
    left_child.sample_positions = std::move(left_positions);

    Node right_child;
    right_child.is_leaf  = true;
    right_child.depth    = depth + 1;
    right_child.axis     = (axis + 1) % 3;
    right_child.cell_min = right_min;
    right_child.cell_max = node.cell_max;
    right_child.dquad_flux.assign(DQUAD_NODES_PER_LEAF, 0.f);
    right_child.num_deposits = right_count;
    right_child.sample_positions = std::move(right_positions);

    // split parent's DQuad flux by observed child occupancy
    for (int i = 0; i < DQUAD_NODES_PER_LEAF; ++i) {
        left_child.dquad_flux[i]  = node.dquad_flux[i] * left_ratio;
        right_child.dquad_flux[i] = node.dquad_flux[i] * right_ratio;
    }

    int left_idx  = (int)nodes_.size();
    int right_idx = left_idx + 1;
    nodes_.push_back(std::move(left_child));
    nodes_.push_back(std::move(right_child));

    // convert leaf to internal — access by index (safe after reserve)
    node.is_leaf   = false;
    node.axis      = axis;
    node.split_pos = split;
    node.left      = left_idx;
    node.right     = right_idx;
    node.dquad_flux.clear();
    node.dquad_flux.shrink_to_fit();
    node.num_deposits = 0;
    node.sample_positions.clear();
    node.sample_positions.shrink_to_fit();
    return true;
}

// -----------------------------------------------
// spatial refinement: median split
// -----------------------------------------------

void SDTree::refine_spatial(int training_iter) {
    // threshold adapted per Müller 2017 §5.3
    float threshold = (float)SD_TREE_SPLIT_THRESHOLD *
                      sqrtf(powf(2.f, (float)training_iter));

    int splits = 0;

    // pass 1: split leaves exceeding deposit threshold
    int n_before = (int)nodes_.size();
    for (int i = 0; i < n_before; ++i) {
        if (nodes_[i].is_leaf &&
            nodes_[i].num_deposits > (int)threshold &&
            nodes_[i].depth < SD_TREE_MAX_DEPTH) {
            if (split_leaf(i))
                ++splits;
        }
    }

    // pass 2 (iter 0 only): cascade forced splits until all leaves reach min depth
    if (training_iter == 0) {
        bool any = true;
        while (any) {
            any = false;
            int n = (int)nodes_.size();
            for (int i = 0; i < n; ++i) {
                if (nodes_[i].is_leaf &&
                    nodes_[i].num_deposits > 0 &&
                    nodes_[i].depth < SD_TREE_MIN_DEPTH) {
                    if (split_leaf(i)) {
                        ++splits;
                        any = true;
                    }
                }
            }
        }
    }

    if (splits > 0) {
        std::printf("[SDTree] Refined: %d splits, %d total nodes, %d leaves\n",
                    splits, num_nodes(), num_leaves());
    }
}

// -----------------------------------------------
// directional refinement (no-op for fixed-depth DQuad)
// -----------------------------------------------

void SDTree::refine_directional() {
    // fixed-depth DQuad: no adaptive directional splits needed.
    // all leaves have depth SD_TREE_DQUAD_DEPTH from init.
}

// -----------------------------------------------
// reset flux
// -----------------------------------------------

void SDTree::reset_flux() {
    for (auto& n : nodes_) {
        if (n.is_leaf) {
            std::fill(n.dquad_flux.begin(), n.dquad_flux.end(), 0.f);
            n.num_deposits = 0;
            n.sample_positions.clear();
        }
    }
}

// -----------------------------------------------
// flatten to GPU arrays (DFS ordering)
// -----------------------------------------------

SDTree::FlatData SDTree::flatten() const {
    FlatData out;
    out.bbox_min = bbox_min_;
    out.bbox_max = bbox_max_;

    if (nodes_.empty()) return out;

    // DFS flatten: left child at idx+1, right child stored explicitly
    struct StackEntry {
        int cpu_idx;    // index in nodes_
        int gpu_idx;    // index in output sd_nodes
    };

    // first pass: count total nodes for pre-allocation
    int total_sd = num_nodes();
    int total_leaves = num_leaves();
    out.sd_nodes.reserve(total_sd);
    out.dquad_nodes.reserve(total_leaves * DQUAD_NODES_PER_LEAF);

    // DFS traversal
    std::vector<StackEntry> stack;
    stack.push_back({0, 0});
    out.sd_nodes.resize(total_sd);

    // we'll fill sd_nodes in DFS order and track gpu indices
    std::vector<int> cpu_to_gpu(nodes_.size(), -1);

    // iterative DFS to assign gpu positions
    int next_gpu = 0;
    {
        struct Item { int cpu; };
        std::vector<Item> dfs_stack;
        dfs_stack.push_back({0});
        while (!dfs_stack.empty()) {
            int cpu = dfs_stack.back().cpu;
            dfs_stack.pop_back();
            int gpu = next_gpu++;
            cpu_to_gpu[cpu] = gpu;
            if (!nodes_[cpu].is_leaf) {
                // push right first so left is processed first (DFS: left at gpu+1)
                dfs_stack.push_back({nodes_[cpu].right});
                dfs_stack.push_back({nodes_[cpu].left});
            }
        }
    }

    // second pass: build GPU nodes
    for (int cpu_idx = 0; cpu_idx < (int)nodes_.size(); ++cpu_idx) {
        int gpu_idx = cpu_to_gpu[cpu_idx];
        if (gpu_idx < 0) continue;
        const Node& n = nodes_[cpu_idx];

        if (n.is_leaf) {
            int dquad_offset = (int)out.dquad_nodes.size();
            flatten_dquad(n.dquad_flux, out.dquad_nodes);
            float total_flux = n.dquad_flux.empty() ? 0.f : n.dquad_flux[0];
            out.sd_nodes[gpu_idx] = sd_make_leaf(dquad_offset, total_flux);
        } else {
            int right_gpu = cpu_to_gpu[n.right];
            out.sd_nodes[gpu_idx] = sd_make_internal(n.axis, right_gpu, n.split_pos);
        }
    }

    return out;
}

void SDTree::flatten_dquad(const std::vector<float>& flux,
                           std::vector<DQuadNodeGPU>& out) const {
    int base = (int)out.size();
    out.resize(base + DQUAD_NODES_PER_LEAF);

    for (int i = 0; i < DQUAD_NODES_PER_LEAF; ++i) {
        float f = (i < (int)flux.size()) ? flux[i] : 0.f;
        int first_child_abs = base + 4 * i + 1;

        // leaf if children would exceed the complete tree array
        if (4 * i + 1 >= DQUAD_NODES_PER_LEAF) {
            out[base + i] = dquad_make_leaf(f);
        } else {
            out[base + i] = dquad_make_internal(f, first_child_abs);
        }
    }
}

// -----------------------------------------------
// diagnostics
// -----------------------------------------------

int SDTree::num_leaves() const {
    int count = 0;
    for (const auto& n : nodes_)
        if (n.is_leaf) ++count;
    return count;
}

int SDTree::total_dquad_nodes() const {
    return num_leaves() * DQUAD_NODES_PER_LEAF;
}
