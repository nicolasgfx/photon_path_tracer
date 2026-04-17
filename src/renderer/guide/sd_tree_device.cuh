#ifndef PPT_GUIDE_SD_TREE_DEVICE_CUH
#define PPT_GUIDE_SD_TREE_DEVICE_CUH
// sd_tree_device.cuh — GPU device functions for SD-tree sampling/PDF
//
// Müller 2017 §4: bilateral MIS with spatial-directional tree.
// Cylindrical equal-area mapping: u = φ/(2π), v = (1 - cosθ)/2.
//
// Requires: LaunchParams params (extern "C" __constant__) with §10 fields.

#include "guide/sd_tree_gpu.h"
#include "guide/dquad_gpu.h"

constexpr float SD_PI      = 3.14159265358979f;
constexpr float SD_TWO_PI  = 6.28318530717959f;
constexpr float SD_FOUR_PI = 12.5663706143592f;
constexpr float SD_INV_4PI = 0.07957747154594f;    // 1 / (4π)

// -----------------------------------------------
// spatial KD-tree lookup → DQuad offset
// -----------------------------------------------

__forceinline__ __device__
int dev_sd_tree_lookup(float3 pos,
                       const SDNodeGPU* sd_nodes, int num_nodes,
                       float3 bbox_min, float3 bbox_max) {
    if (num_nodes <= 0) return -1;

    // clamp to bounds
    constexpr float EPS = 1e-6f;
    pos.x = fmaxf(bbox_min.x + EPS, fminf(pos.x, bbox_max.x - EPS));
    pos.y = fmaxf(bbox_min.y + EPS, fminf(pos.y, bbox_max.y - EPS));
    pos.z = fmaxf(bbox_min.z + EPS, fminf(pos.z, bbox_max.z - EPS));

    int idx = 0;
    while (!sd_is_leaf(sd_nodes[idx])) {
        int axis = sd_axis(sd_nodes[idx]);
        float sp = sd_nodes[idx].split_pos;
        float val = (axis == 0) ? pos.x : (axis == 1) ? pos.y : pos.z;
        int next = (val < sp) ? (idx + 1) : sd_right_child(sd_nodes[idx]);
        if (next <= idx || next >= num_nodes) return -1; // safety: prevent OOB / loops
        idx = next;
    }
    return sd_dquad_offset(sd_nodes[idx]);
}

// lookup variant that also returns the spatial cell extent (Müller 2017 §5.4)
__forceinline__ __device__
int dev_sd_tree_lookup_cell(float3 pos,
                            const SDNodeGPU* sd_nodes, int num_nodes,
                            float3 bbox_min, float3 bbox_max,
                            float3& cell_extent) {
    if (num_nodes <= 0) { cell_extent = make_f3(0,0,0); return -1; }

    constexpr float EPS = 1e-6f;
    pos.x = fmaxf(bbox_min.x + EPS, fminf(pos.x, bbox_max.x - EPS));
    pos.y = fmaxf(bbox_min.y + EPS, fminf(pos.y, bbox_max.y - EPS));
    pos.z = fmaxf(bbox_min.z + EPS, fminf(pos.z, bbox_max.z - EPS));

    float3 lo = bbox_min;
    float3 hi = bbox_max;

    int idx = 0;
    while (!sd_is_leaf(sd_nodes[idx])) {
        int axis = sd_axis(sd_nodes[idx]);
        float sp = sd_nodes[idx].split_pos;
        float val = (axis == 0) ? pos.x : (axis == 1) ? pos.y : pos.z;
        if (val < sp) {
            if (axis == 0) hi.x = sp; else if (axis == 1) hi.y = sp; else hi.z = sp;
            int next = idx + 1;
            if (next >= num_nodes) { cell_extent = make_f3(0,0,0); return -1; }
            idx = next;
        } else {
            if (axis == 0) lo.x = sp; else if (axis == 1) lo.y = sp; else lo.z = sp;
            int next = sd_right_child(sd_nodes[idx]);
            if (next <= idx || next >= num_nodes) { cell_extent = make_f3(0,0,0); return -1; }
            idx = next;
        }
    }
    cell_extent = make_f3(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z);
    return sd_dquad_offset(sd_nodes[idx]);
}

// -----------------------------------------------
// direction ↔ UV (cylindrical equal-area)
// -----------------------------------------------

__forceinline__ __device__
void sd_dir_to_uv(float3 d, float& u, float& v) {
    float cos_theta = fmaxf(-1.f, fminf(1.f, d.z));
    float phi = atan2f(d.y, d.x);
    if (phi < 0.f) phi += SD_TWO_PI;
    u = phi / SD_TWO_PI;
    v = (1.f - cos_theta) * 0.5f;
    u = fmaxf(0.f, fminf(u, 0.99999f));
    v = fmaxf(0.f, fminf(v, 0.99999f));
}

__forceinline__ __device__
float3 sd_uv_to_dir(float u, float v) {
    float cos_theta = 1.f - 2.f * v;
    float sin_theta = sqrtf(fmaxf(0.f, 1.f - cos_theta * cos_theta));
    float phi = u * SD_TWO_PI;
    return make_f3(sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta);
}

// -----------------------------------------------
// DQuad hierarchical CDF sampling
// -----------------------------------------------

__forceinline__ __device__
float3 dev_sd_tree_sample_dquad(const DQuadNodeGPU* dq, int root,
                                float& pdf_omega,
                                float r1, float r2) {
    int idx = root;
    float u0 = 0.f, u1 = 1.f, v0 = 0.f, v1 = 1.f;

    for (int d = 0; d < SD_TREE_DQUAD_DEPTH; ++d) {
        int fc = 4 * (idx - root) + 1 + root;  // first child in absolute index

        float f0 = fmaxf(dq[fc + 0].flux, 0.f);
        float f1 = fmaxf(dq[fc + 1].flux, 0.f);
        float f2 = fmaxf(dq[fc + 2].flux, 0.f);
        float f3 = fmaxf(dq[fc + 3].flux, 0.f);
        float total = f0 + f1 + f2 + f3;

        if (total <= 0.f) {
            // uniform fallback within current cell
            break;
        }

        float inv_total = 1.f / total;
        float c0 = f0 * inv_total;
        float c1 = c0 + f1 * inv_total;
        float c2 = c1 + f2 * inv_total;

        float u_mid = (u0 + u1) * 0.5f;
        float v_mid = (v0 + v1) * 0.5f;

        int child;
        if (r1 < c0) {
            child = 0;
            r1 = r1 / fmaxf(c0, 1e-10f);  // remap for next level
            u1 = u_mid; v1 = v_mid;
        } else if (r1 < c1) {
            child = 1;
            r1 = (r1 - c0) / fmaxf(c1 - c0, 1e-10f);
            u0 = u_mid; v1 = v_mid;
        } else if (r1 < c2) {
            child = 2;
            r1 = (r1 - c1) / fmaxf(c2 - c1, 1e-10f);
            u1 = u_mid; v0 = v_mid;
        } else {
            child = 3;
            r1 = (r1 - c2) / fmaxf(1.f - c2, 1e-10f);
            u0 = u_mid; v0 = v_mid;
        }
        idx = fc + child;
    }

    // uniform sample within the selected leaf cell
    float u = u0 + r1 * (u1 - u0);
    float v = v0 + r2 * (v1 - v0);

    // compute PDF per steradian
    // pdf_uv = (leaf_flux / root_flux) / leaf_area
    // jacobian: dω = 4π · du · dv  (cylindrical equal-area)
    // pdf_omega = pdf_uv / (4π)
    float leaf_area = (u1 - u0) * (v1 - v0);
    float root_flux = fmaxf(dq[root].flux, 1e-20f);
    float leaf_flux = fmaxf(dq[idx].flux, 0.f);

    // handle near-zero root flux → uniform sphere
    if (root_flux < 1e-10f) {
        pdf_omega = SD_INV_4PI;
        return sd_uv_to_dir(r1, r2);
    }

    pdf_omega = (leaf_flux / root_flux) / fmaxf(leaf_area, 1e-10f) / SD_FOUR_PI;
    if (pdf_omega < 1e-10f) pdf_omega = SD_INV_4PI;

    return sd_uv_to_dir(u, v);
}

// -----------------------------------------------
// DQuad PDF evaluation for a given direction
// -----------------------------------------------

__forceinline__ __device__
float dev_sd_tree_pdf_dquad(const DQuadNodeGPU* dq, int root,
                            float3 dir) {
    float u, v;
    sd_dir_to_uv(dir, u, v);

    int idx = root;
    float u0 = 0.f, u1 = 1.f, v0 = 0.f, v1 = 1.f;

    for (int d = 0; d < SD_TREE_DQUAD_DEPTH; ++d) {
        int fc = 4 * (idx - root) + 1 + root;
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
        idx = fc + child;
    }

    float leaf_area = (u1 - u0) * (v1 - v0);
    float root_flux = fmaxf(dq[root].flux, 1e-20f);
    float leaf_flux = fmaxf(dq[idx].flux, 0.f);

    if (root_flux < 1e-10f) return SD_INV_4PI;

    float pdf = (leaf_flux / root_flux) / fmaxf(leaf_area, 1e-10f) / SD_FOUR_PI;
    return fmaxf(pdf, 1e-10f);
}

// -----------------------------------------------
// convenience wrappers reading from LaunchParams
// -----------------------------------------------

__forceinline__ __device__
float3 dev_sd_tree_sample(float3 pos, float& pdf_omega,
                          float r1, float r2) {
    if (!params.sd_tree_valid || !params.sd_tree_nodes || !params.sd_tree_dquad) {
        pdf_omega = SD_INV_4PI;
        // uniform sphere
        float cos_theta = 1.f - 2.f * r2;
        float sin_theta = sqrtf(fmaxf(0.f, 1.f - cos_theta * cos_theta));
        float phi = r1 * SD_TWO_PI;
        return make_f3(sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta);
    }

    int dq_root = dev_sd_tree_lookup(pos,
        params.sd_tree_nodes, params.sd_tree_num_nodes,
        params.sd_tree_bbox_min, params.sd_tree_bbox_max);

    if (dq_root < 0) {
        pdf_omega = SD_INV_4PI;
        float cos_theta = 1.f - 2.f * r2;
        float sin_theta = sqrtf(fmaxf(0.f, 1.f - cos_theta * cos_theta));
        float phi = r1 * SD_TWO_PI;
        return make_f3(sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta);
    }

    return dev_sd_tree_sample_dquad(params.sd_tree_dquad, dq_root,
                                    pdf_omega, r1, r2);
}

__forceinline__ __device__
float dev_sd_tree_pdf(float3 pos, float3 dir) {
    if (!params.sd_tree_valid || !params.sd_tree_nodes || !params.sd_tree_dquad)
        return SD_INV_4PI;

    int dq_root = dev_sd_tree_lookup(pos,
        params.sd_tree_nodes, params.sd_tree_num_nodes,
        params.sd_tree_bbox_min, params.sd_tree_bbox_max);

    if (dq_root < 0) return SD_INV_4PI;

    return dev_sd_tree_pdf_dquad(params.sd_tree_dquad, dq_root, dir);
}

// convenience: sample with RNG object
__forceinline__ __device__
float3 dev_sd_tree_sample(float3 pos, PCGRng& rng, float& pdf_omega) {
    return dev_sd_tree_sample(pos, pdf_omega,
                              rng.next_float(), rng.next_float());
}

// -----------------------------------------------
// nearest-boundary blend — smooth cell transitions
// -----------------------------------------------
// Extends Müller 2017 §5.4: instead of stochastic jitter (which adds
// variance), deterministically blend the current cell's DQuad with its
// nearest neighbor.  Weight is linear: 1.0 at cell centre, 0.5 at face.
// One-sample mixture sampling keeps MIS consistency:
//   pdf = w·pdf_A + (1-w)·pdf_B

struct SDBlendInfo {
    int   dq_primary;     // DQuad root of cell containing pos
    int   dq_neighbor;    // DQuad root of nearest neighbor (-1 = none)
    float w;              // weight for primary cell [0.5, 1.0]
};

__forceinline__ __device__
SDBlendInfo dev_sd_tree_get_blend(float3 pos) {
    SDBlendInfo bi;
    bi.dq_primary  = -1;
    bi.dq_neighbor = -1;
    bi.w           = 1.f;

    if (!params.sd_tree_valid || !params.sd_tree_nodes || !params.sd_tree_dquad)
        return bi;

    const SDNodeGPU* nodes = params.sd_tree_nodes;
    const int        num   = params.sd_tree_num_nodes;
    const float3     bmin  = params.sd_tree_bbox_min;
    const float3     bmax  = params.sd_tree_bbox_max;

    constexpr float EPS = 1e-6f;
    pos.x = fmaxf(bmin.x + EPS, fminf(pos.x, bmax.x - EPS));
    pos.y = fmaxf(bmin.y + EPS, fminf(pos.y, bmax.y - EPS));
    pos.z = fmaxf(bmin.z + EPS, fminf(pos.z, bmax.z - EPS));

    // traverse to leaf, tracking cell bounds
    float3 lo = bmin;
    float3 hi = bmax;
    int idx = 0;
    while (!sd_is_leaf(nodes[idx])) {
        int axis  = sd_axis(nodes[idx]);
        float sp  = nodes[idx].split_pos;
        float val = (axis == 0) ? pos.x : (axis == 1) ? pos.y : pos.z;
        if (val < sp) {
            if (axis == 0) hi.x = sp; else if (axis == 1) hi.y = sp; else hi.z = sp;
            int next = idx + 1;
            if (next >= num) return bi;
            idx = next;
        } else {
            if (axis == 0) lo.x = sp; else if (axis == 1) lo.y = sp; else lo.z = sp;
            int next = sd_right_child(nodes[idx]);
            if (next <= idx || next >= num) return bi;
            idx = next;
        }
    }
    bi.dq_primary = sd_dquad_offset(nodes[idx]);

    // distance to each of the 6 cell faces
    float dists[6] = {
        pos.x - lo.x, hi.x - pos.x,
        pos.y - lo.y, hi.y - pos.y,
        pos.z - lo.z, hi.z - pos.z
    };
    int best = 0;
    for (int i = 1; i < 6; ++i)
        if (dists[i] < dists[best]) best = i;

    int  best_axis  = best / 2;
    bool lower_face = (best % 2 == 0);
    float half_ext  = ((best_axis == 0) ? (hi.x - lo.x)
                     : (best_axis == 1) ? (hi.y - lo.y)
                                        : (hi.z - lo.z)) * 0.5f;
    if (half_ext < EPS) return bi;

    // linear weight: 1.0 at centre → 0.5 at face
    bi.w = 0.5f + 0.5f * dists[best] / half_ext;
    bi.w = fmaxf(0.5f, fminf(1.f, bi.w));

    // reflect pos across nearest face → lands in neighbor cell
    float boundary = lower_face
        ? ((best_axis == 0) ? lo.x : (best_axis == 1) ? lo.y : lo.z)
        : ((best_axis == 0) ? hi.x : (best_axis == 1) ? hi.y : hi.z);
    float orig = (best_axis == 0) ? pos.x : (best_axis == 1) ? pos.y : pos.z;
    float refl = 2.f * boundary - orig;

    float3 reflected = pos;
    if (best_axis == 0)      reflected.x = refl;
    else if (best_axis == 1) reflected.y = refl;
    else                     reflected.z = refl;

    // bail if reflected pos is outside the scene bbox
    if (reflected.x < bmin.x + EPS || reflected.x > bmax.x - EPS ||
        reflected.y < bmin.y + EPS || reflected.y > bmax.y - EPS ||
        reflected.z < bmin.z + EPS || reflected.z > bmax.z - EPS) {
        bi.w = 1.f;
        return bi;
    }

    bi.dq_neighbor = dev_sd_tree_lookup(reflected, nodes, num, bmin, bmax);
    if (bi.dq_neighbor < 0) bi.w = 1.f;
    return bi;
}

// sample from nearest-boundary mixture of two cells
__forceinline__ __device__
float3 dev_sd_tree_sample_blend(float3 pos, PCGRng& rng, float& pdf_omega) {
    SDBlendInfo bi = dev_sd_tree_get_blend(pos);

    if (bi.dq_primary < 0) {
        pdf_omega = SD_INV_4PI;
        float r1 = rng.next_float(), r2 = rng.next_float();
        float ct = 1.f - 2.f * r2;
        float st = sqrtf(fmaxf(0.f, 1.f - ct * ct));
        float phi = r1 * SD_TWO_PI;
        return make_f3(st * cosf(phi), st * sinf(phi), ct);
    }

    const DQuadNodeGPU* dq = params.sd_tree_dquad;

    // stochastic cell selection preserves one-sample MIS correctness
    bool use_primary = (bi.dq_neighbor < 0 || rng.next_float() < bi.w);
    int  root = use_primary ? bi.dq_primary : bi.dq_neighbor;

    float r1 = rng.next_float(), r2 = rng.next_float();
    float sample_pdf = 0.f;
    float3 dir = dev_sd_tree_sample_dquad(dq, root, sample_pdf, r1, r2);

    // combined mixture PDF
    if (bi.dq_neighbor >= 0) {
        float pdf_a = dev_sd_tree_pdf_dquad(dq, bi.dq_primary,  dir);
        float pdf_b = dev_sd_tree_pdf_dquad(dq, bi.dq_neighbor, dir);
        pdf_omega = bi.w * pdf_a + (1.f - bi.w) * pdf_b;
    } else {
        pdf_omega = sample_pdf;
    }
    if (pdf_omega < 1e-10f) pdf_omega = SD_INV_4PI;
    return dir;
}

// PDF evaluation with nearest-boundary blend
__forceinline__ __device__
float dev_sd_tree_pdf_blend(float3 pos, float3 dir) {
    SDBlendInfo bi = dev_sd_tree_get_blend(pos);
    if (bi.dq_primary < 0) return SD_INV_4PI;

    const DQuadNodeGPU* dq = params.sd_tree_dquad;
    float pdf_a = dev_sd_tree_pdf_dquad(dq, bi.dq_primary, dir);

    if (bi.dq_neighbor >= 0) {
        float pdf_b = dev_sd_tree_pdf_dquad(dq, bi.dq_neighbor, dir);
        return bi.w * pdf_a + (1.f - bi.w) * pdf_b;
    }
    return pdf_a;
}

#endif // PPT_GUIDE_SD_TREE_DEVICE_CUH
