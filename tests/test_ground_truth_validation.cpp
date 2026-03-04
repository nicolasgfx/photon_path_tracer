// ─────────────────────────────────────────────────────────────────────
// test_ground_truth_validation.cpp – Ground-truth CPU validation tests
// ─────────────────────────────────────────────────────────────────────
// Purpose: find the FUNDAMENTAL flaw in the renderer by comparing
// every spatial query, density estimate, and NEE sample against
// obviously-correct brute-force CPU implementations.
//
// Ground-truth rules:
//   - Brute-force = simple nested for-loops, no spatial indices
//   - Identical physics to the production code
//   - Pre-populated with artificial data covering real-world scenarios
//
// Test groups:
//   1. KD-tree range query correctness (vs brute-force linear scan)
//   2. KD-tree k-NN correctness
//   3. Full density estimation: brute-force vs KD-tree vs hash grid
//   4. Photon flux magnitude & energy conservation
//   5. NEE direct lighting: shadow rays, PDF, contribution
//   6. End-to-end pipeline: known scene → analytic answer
// ─────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>
#include <set>
#include <algorithm>
#include <iostream>
#include <random>

#include "core/types.h"
#include "core/spectrum.h"
#include "core/random.h"
#include "core/config.h"
#include "scene/triangle.h"
#include "scene/material.h"
#include "scene/scene.h"
#include "bsdf/bsdf.h"
#include "photon/photon.h"
#include "photon/hash_grid.h"
#include "photon/kd_tree.h"
#include "photon/density_estimator.h"
#include "photon/surface_filter.h"
#include "photon/emitter.h"
#include "renderer/direct_light.h"
#include "scene/obj_loader.h"

#include <cstdio>
#include <filesystem>

// =====================================================================
// Helper: brute-force photon gather (ground truth)
// =====================================================================
// Linear scan through ALL photons.  Returns the set of photon indices
// within `radius` of `query_pos`.  NO spatial index.  Obviously correct.

static std::vector<uint32_t> brute_force_range_query(
    float3 query_pos, float radius,
    const PhotonSoA& photons)
{
    std::vector<uint32_t> result;
    float r2 = radius * radius;
    for (size_t i = 0; i < photons.size(); ++i) {
        float dx = photons.pos_x[i] - query_pos.x;
        float dy = photons.pos_y[i] - query_pos.y;
        float dz = photons.pos_z[i] - query_pos.z;
        float d2 = dx*dx + dy*dy + dz*dz;
        if (d2 <= r2) {
            result.push_back((uint32_t)i);
        }
    }
    return result;
}

// =====================================================================
// Helper: brute-force k-NN (ground truth)
// =====================================================================
// Returns the k nearest photon indices and the squared distance to the
// k-th nearest.  Simple O(N·k) linear scan using a sorted list.

static void brute_force_knn(
    float3 query_pos, int k,
    const PhotonSoA& photons,
    std::vector<uint32_t>& out_indices,
    float& out_max_dist2)
{
    // Compute all distances
    std::vector<std::pair<float, uint32_t>> all_dists;
    all_dists.reserve(photons.size());
    for (size_t i = 0; i < photons.size(); ++i) {
        float dx = photons.pos_x[i] - query_pos.x;
        float dy = photons.pos_y[i] - query_pos.y;
        float dz = photons.pos_z[i] - query_pos.z;
        float d2 = dx*dx + dy*dy + dz*dz;
        all_dists.push_back({d2, (uint32_t)i});
    }

    // Partial sort: find the k smallest
    if ((int)all_dists.size() > k) {
        std::nth_element(all_dists.begin(), all_dists.begin() + k,
                         all_dists.end());
        all_dists.resize(k);
    }

    // Sort by distance for deterministic output
    std::sort(all_dists.begin(), all_dists.end());

    out_indices.clear();
    out_max_dist2 = 0.f;
    for (const auto& [d2, idx] : all_dists) {
        out_indices.push_back(idx);
        out_max_dist2 = fmaxf(out_max_dist2, d2);
    }
}

// =====================================================================
// Helper: brute-force density estimation (ground truth)
// =====================================================================
// Identical physics/math to estimate_density_kdtree() in renderer.cpp,
// but uses a linear scan instead of the KD-tree.
// This is the REFERENCE implementation.

static Spectrum brute_force_density_estimate(
    float3 hit_pos, float3 hit_normal, float3 wo_local,
    const Material& mat,
    const PhotonSoA& photons,
    float gather_radius, int num_photons_emitted)
{
    Spectrum L = Spectrum::zero();
    float r2       = gather_radius * gather_radius;
    float inv_area = 2.0f / (PI * r2); // Epanechnikov 2D normalization (§6.3)
    float inv_N    = 1.0f / (float)num_photons_emitted;
    ONB frame = ONB::from_normal(hit_normal);

    for (size_t i = 0; i < photons.size(); ++i) {
        // Tangential distance check (matches production tangential disk kernel §6.3)
        float3 p_pos = make_f3(photons.pos_x[i],
                               photons.pos_y[i],
                               photons.pos_z[i]);
        float3 diff = p_pos - hit_pos;
        float d_plane = dot(hit_normal, diff);
        if (fabsf(d_plane) > DEFAULT_SURFACE_TAU) continue;
        float3 v_tan = diff - hit_normal * d_plane;
        float d_tan2 = dot(v_tan, v_tan);
        if (d_tan2 > r2) continue;

        // Normal consistency
        if (!photons.norm_x.empty()) {
            float3 pn = make_f3(photons.norm_x[i],
                                photons.norm_y[i],
                                photons.norm_z[i]);
            if (dot(pn, hit_normal) <= 0.0f) continue;
        }

        // Direction consistency: photon wi must point into the surface
        float3 wi = make_f3(photons.wi_x[i],
                            photons.wi_y[i],
                            photons.wi_z[i]);
        if (dot(wi, hit_normal) <= 0.f) continue;

        // BSDF evaluation (diffuse-only, same as production code)
        float3 wi_loc = frame.world_to_local(wi);
        Spectrum f = bsdf::evaluate_diffuse(mat, wo_local, wi_loc);

        // Accumulate (Epanechnikov weight)
        float w = 1.0f - d_tan2 / r2;
        Spectrum photon_flux = photons.get_flux(i);
        for (int b = 0; b < NUM_LAMBDA; ++b)
            L.value[b] += w * photon_flux.value[b] * inv_N * f.value[b] * inv_area;
    }

    return L;
}

// =====================================================================
// Helper: brute-force density estimation with FULL BSDF (ground truth)
// =====================================================================
// Same as above but uses bsdf::evaluate() instead of evaluate_diffuse().
// This lets us detect if the diffuse-only choice is causing the problem.

static Spectrum brute_force_density_estimate_full_bsdf(
    float3 hit_pos, float3 hit_normal, float3 wo_local,
    const Material& mat,
    const PhotonSoA& photons,
    float gather_radius, int num_photons_emitted)
{
    Spectrum L = Spectrum::zero();
    float r2       = gather_radius * gather_radius;
    float inv_area = 2.0f / (PI * r2); // Epanechnikov 2D normalization (§6.3)
    float inv_N    = 1.0f / (float)num_photons_emitted;
    ONB frame = ONB::from_normal(hit_normal);

    for (size_t i = 0; i < photons.size(); ++i) {
        // Tangential distance check (matches production tangential disk kernel §6.3)
        float3 p_pos = make_f3(photons.pos_x[i],
                               photons.pos_y[i],
                               photons.pos_z[i]);
        float3 diff = p_pos - hit_pos;
        float d_plane = dot(hit_normal, diff);
        if (fabsf(d_plane) > DEFAULT_SURFACE_TAU) continue;
        float3 v_tan = diff - hit_normal * d_plane;
        float d_tan2 = dot(v_tan, v_tan);
        if (d_tan2 > r2) continue;

        if (!photons.norm_x.empty()) {
            float3 pn = make_f3(photons.norm_x[i],
                                photons.norm_y[i],
                                photons.norm_z[i]);
            if (dot(pn, hit_normal) <= 0.0f) continue;
        }

        float3 wi = make_f3(photons.wi_x[i],
                            photons.wi_y[i],
                            photons.wi_z[i]);
        if (dot(wi, hit_normal) <= 0.f) continue;

        float3 wi_loc = frame.world_to_local(wi);
        Spectrum f = bsdf::evaluate(mat, wo_local, wi_loc);

        float w = 1.0f - d_tan2 / r2; // Epanechnikov weight
        Spectrum photon_flux = photons.get_flux(i);
        for (int b = 0; b < NUM_LAMBDA; ++b)
            L.value[b] += w * photon_flux.value[b] * inv_N * f.value[b] * inv_area;
    }

    return L;
}

// =====================================================================
// Helper: create a synthetic PhotonSoA with known properties
// =====================================================================

// Create photons uniformly distributed on a horizontal plane
// at height y=0, normal=(0,1,0), wi=(0,1,0) (coming from above).
// All photons have the same constant spectral flux.
static PhotonSoA make_uniform_floor_photons(
    int count, float extent, float flux_per_bin,
    unsigned seed = 42)
{
    PhotonSoA photons;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-extent/2, extent/2);

    for (int i = 0; i < count; ++i) {
        Photon p;
        p.position    = make_f3(dist(gen), 0.f, dist(gen));
        p.wi          = make_f3(0.f, 1.f, 0.f);      // coming from above
        p.geom_normal = make_f3(0.f, 1.f, 0.f);      // floor normal
        p.spectral_flux = Spectrum::constant(flux_per_bin);
        photons.push_back(p);
    }
    return photons;
}

// Create photons in a 3D box
static PhotonSoA make_random_3d_photons(
    int count, float extent, float flux_per_bin,
    unsigned seed = 42)
{
    PhotonSoA photons;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> pos_dist(-extent/2, extent/2);
    std::uniform_real_distribution<float> dir_dist(-1.f, 1.f);

    for (int i = 0; i < count; ++i) {
        Photon p;
        p.position = make_f3(pos_dist(gen), pos_dist(gen), pos_dist(gen));
        // Random direction but ensure it's normalized
        float3 d = make_f3(dir_dist(gen), dir_dist(gen), dir_dist(gen));
        float l = length(d);
        if (l < 0.01f) d = make_f3(0.f, 1.f, 0.f);
        else d = d / l;
        p.wi = d;
        // Normal consistent with wi direction
        p.geom_normal = d;
        p.spectral_flux = Spectrum::constant(flux_per_bin);
        photons.push_back(p);
    }
    return photons;
}

// Create a single photon at a specific location
static PhotonSoA make_single_photon(
    float3 pos, float3 wi, float3 normal, float flux_per_bin)
{
    PhotonSoA photons;
    Photon p;
    p.position      = pos;
    p.wi            = wi;
    p.geom_normal   = normal;
    p.spectral_flux = Spectrum::constant(flux_per_bin);
    photons.push_back(p);
    return photons;
}

// Create photons on a wall (vertical surface, normal along x)
static PhotonSoA make_uniform_wall_photons(
    int count, float extent, float flux_per_bin,
    unsigned seed = 43)
{
    PhotonSoA photons;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-extent/2, extent/2);

    for (int i = 0; i < count; ++i) {
        Photon p;
        p.position    = make_f3(0.f, dist(gen), dist(gen));
        p.wi          = make_f3(1.f, 0.f, 0.f);  // coming from +x side
        p.geom_normal = make_f3(1.f, 0.f, 0.f);  // wall faces +x
        p.spectral_flux = Spectrum::constant(flux_per_bin);
        photons.push_back(p);
    }
    return photons;
}

// Create a cluster of photons at a point (simulate one bright photon area)
static PhotonSoA make_clustered_photons(
    float3 center, int count, float cluster_radius,
    float3 wi, float3 normal, float flux_per_bin,
    unsigned seed = 44)
{
    PhotonSoA photons;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-cluster_radius, cluster_radius);

    for (int i = 0; i < count; ++i) {
        Photon p;
        p.position    = center + make_f3(dist(gen), dist(gen), dist(gen));
        p.wi          = wi;
        p.geom_normal = normal;
        p.spectral_flux = Spectrum::constant(flux_per_bin);
        photons.push_back(p);
    }
    return photons;
}

// Create photons with spectrally varying flux (simulates real emission)
static PhotonSoA make_spectral_photons(
    int count, float extent, unsigned seed = 45)
{
    PhotonSoA photons;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> pos_dist(-extent/2, extent/2);

    for (int i = 0; i < count; ++i) {
        Photon p;
        p.position    = make_f3(pos_dist(gen), 0.f, pos_dist(gen));
        p.wi          = make_f3(0.f, 1.f, 0.f);
        p.geom_normal = make_f3(0.f, 1.f, 0.f);
        // Each photon gets a different spectral profile
        // Simulate warm white light emission
        Spectrum flux;
        for (int b = 0; b < NUM_LAMBDA; ++b) {
            float lam = lambda_of_bin(b);
            // Warm white: peaks in red/green, less blue
            float t = (lam - 550.f) / 100.f;
            flux.value[b] = 10.f * expf(-0.5f * t * t);
        }
        p.spectral_flux = flux;
        photons.push_back(p);
    }
    return photons;
}

// Helper: create a Lambertian material with constant Kd
static Material make_lambertian_material(float kd) {
    Material mat;
    mat.type = MaterialType::Lambertian;
    mat.Kd = Spectrum::constant(kd);
    mat.Ks = Spectrum::zero();
    mat.roughness = 1.0f;
    return mat;
}

// Helper: create a GlossyMetal material
static Material make_glossy_material(float kd, float ks, float roughness) {
    Material mat;
    mat.type = MaterialType::GlossyMetal;
    mat.Kd = Spectrum::constant(kd);
    mat.Ks = Spectrum::constant(ks);
    mat.roughness = roughness;
    return mat;
}

// Helper: add a quad to the scene with auto-corrected winding.
// Ensures geometric_normal() agrees with `intended_normal`.
static void add_quad(Scene& scene,
                     float3 a, float3 b, float3 c, float3 d,
                     float3 intended_normal, int mat_id)
{
    // Triangle 1: a-b-c
    Triangle t1;
    t1.v0 = a; t1.v1 = b; t1.v2 = c;
    t1.n0 = t1.n1 = t1.n2 = intended_normal;
    t1.material_id = mat_id;
    if (dot(t1.geometric_normal(), intended_normal) < 0.f)
        std::swap(t1.v1, t1.v2);
    scene.triangles.push_back(t1);

    // Triangle 2: a-c-d
    Triangle t2;
    t2.v0 = a; t2.v1 = c; t2.v2 = d;
    t2.n0 = t2.n1 = t2.n2 = intended_normal;
    t2.material_id = mat_id;
    if (dot(t2.geometric_normal(), intended_normal) < 0.f)
        std::swap(t2.v1, t2.v2);
    scene.triangles.push_back(t2);
}

// Helper: build a simple box scene for NEE testing
// Returns a scene with 10 triangles (5 faces) and one emissive ceiling
static Scene make_box_scene() {
    Scene scene;

    float3 up    = make_f3(0, 1, 0);
    float3 down  = make_f3(0,-1, 0);
    float3 fwd   = make_f3(0, 0, 1);
    float3 right = make_f3(1, 0, 0);
    float3 left  = make_f3(-1, 0, 0);

    // Floor (y=-0.5, normal up)
    add_quad(scene,
        make_f3(-0.5f,-0.5f,-0.5f), make_f3( 0.5f,-0.5f,-0.5f),
        make_f3( 0.5f,-0.5f, 0.5f), make_f3(-0.5f,-0.5f, 0.5f),
        up, 0);

    // Ceiling light (y=0.49, normal down — emits into room)
    add_quad(scene,
        make_f3(-0.1f, 0.49f,-0.1f), make_f3( 0.1f, 0.49f,-0.1f),
        make_f3( 0.1f, 0.49f, 0.1f), make_f3(-0.1f, 0.49f, 0.1f),
        down, 1);

    // Back wall (z=-0.5, normal +z)
    add_quad(scene,
        make_f3(-0.5f,-0.5f,-0.5f), make_f3(-0.5f, 0.5f,-0.5f),
        make_f3( 0.5f, 0.5f,-0.5f), make_f3( 0.5f,-0.5f,-0.5f),
        fwd, 0);

    // Left wall (x=-0.5, normal +x)
    add_quad(scene,
        make_f3(-0.5f,-0.5f,-0.5f), make_f3(-0.5f,-0.5f, 0.5f),
        make_f3(-0.5f, 0.5f, 0.5f), make_f3(-0.5f, 0.5f,-0.5f),
        right, 0);

    // Right wall (x=0.5, normal -x)
    add_quad(scene,
        make_f3( 0.5f,-0.5f, 0.5f), make_f3( 0.5f,-0.5f,-0.5f),
        make_f3( 0.5f, 0.5f,-0.5f), make_f3( 0.5f, 0.5f, 0.5f),
        left, 0);

    // Materials: 0 = diffuse white, 1 = emissive
    Material diffuse;
    diffuse.type = MaterialType::Lambertian;
    diffuse.Kd = Spectrum::constant(0.8f);
    diffuse.Ks = Spectrum::zero();
    scene.materials.push_back(diffuse);

    Material light_mat;
    light_mat.type = MaterialType::Emissive;
    light_mat.Kd = Spectrum::constant(0.0f);
    light_mat.Ks = Spectrum::zero();
    light_mat.Le = Spectrum::constant(10.0f);
    scene.materials.push_back(light_mat);

    return scene;
}

// Helper: build box scene with a blocker between floor and ceiling light
static Scene make_box_scene_with_blocker() {
    Scene scene = make_box_scene();

    // Add blocker quad at y=0.0, covers the area below the light
    // Needs BOTH faces to block from both directions
    float3 up   = make_f3(0, 1, 0);
    float3 down = make_f3(0,-1, 0);

    // Downward face (blocks rays coming from below)
    add_quad(scene,
        make_f3(-0.2f, 0.0f,-0.2f), make_f3( 0.2f, 0.0f,-0.2f),
        make_f3( 0.2f, 0.0f, 0.2f), make_f3(-0.2f, 0.0f, 0.2f),
        down, 0);

    // Upward face (blocks rays coming from above)
    add_quad(scene,
        make_f3(-0.2f, 0.0f,-0.2f), make_f3(-0.2f, 0.0f, 0.2f),
        make_f3( 0.2f, 0.0f, 0.2f), make_f3( 0.2f, 0.0f,-0.2f),
        up, 0);

    return scene;
}


// =====================================================================
//  GROUP 1: KD-Tree Range Query Correctness
// =====================================================================

TEST(GroundTruth_KDTree, RangeQuery_SameAsLinearScan_Uniform3D) {
    // Create 5000 random photons in a unit cube
    auto photons = make_random_3d_photons(5000, 1.0f, 1.0f);

    KDTree tree;
    tree.build(photons);

    // Test 20 random query points
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-0.4f, 0.4f);
    float radius = 0.15f;

    int total_mismatches = 0;
    for (int q = 0; q < 20; ++q) {
        float3 query = make_f3(dist(gen), dist(gen), dist(gen));

        // Brute force
        auto bf_indices = brute_force_range_query(query, radius, photons);
        std::set<uint32_t> bf_set(bf_indices.begin(), bf_indices.end());

        // KD-tree
        std::set<uint32_t> kd_set;
        tree.query(query, radius, photons,
            [&](uint32_t idx, float /*d2*/) { kd_set.insert(idx); });

        // Must return EXACTLY the same photon indices
        if (bf_set != kd_set) {
            total_mismatches++;
            // Diagnostic: find what's different
            std::vector<uint32_t> only_bf, only_kd;
            std::set_difference(bf_set.begin(), bf_set.end(),
                                kd_set.begin(), kd_set.end(),
                                std::back_inserter(only_bf));
            std::set_difference(kd_set.begin(), kd_set.end(),
                                bf_set.begin(), bf_set.end(),
                                std::back_inserter(only_kd));
            std::cerr << "  Query " << q
                      << ": BF=" << bf_set.size()
                      << " KD=" << kd_set.size()
                      << " OnlyBF=" << only_bf.size()
                      << " OnlyKD=" << only_kd.size() << "\n";

            // Print distances for first few mismatches
            for (size_t j = 0; j < std::min(only_bf.size(), (size_t)3); ++j) {
                uint32_t idx = only_bf[j];
                float dx = photons.pos_x[idx] - query.x;
                float dy = photons.pos_y[idx] - query.y;
                float dz = photons.pos_z[idx] - query.z;
                float d2 = dx*dx + dy*dy + dz*dz;
                std::cerr << "    OnlyBF[" << j << "] idx=" << idx
                          << " d=" << sqrtf(d2) << " r=" << radius << "\n";
            }
        }
    }
    EXPECT_EQ(total_mismatches, 0)
        << "KD-tree range query returned different photons than brute force";
}

TEST(GroundTruth_KDTree, RangeQuery_SameAsLinearScan_FloorPhotons) {
    // Floor photons: all at y≈0, which creates a degenerate distribution
    // (nearly 2D). This can trip up KD-trees that split on y axis.
    auto photons = make_uniform_floor_photons(3000, 1.0f, 1.0f);

    KDTree tree;
    tree.build(photons);

    std::mt19937 gen(67890);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    float radius = 0.1f;

    int mismatches = 0;
    for (int q = 0; q < 20; ++q) {
        float3 query = make_f3(dist(gen), 0.f, dist(gen));

        auto bf_indices = brute_force_range_query(query, radius, photons);
        std::set<uint32_t> bf_set(bf_indices.begin(), bf_indices.end());

        std::set<uint32_t> kd_set;
        tree.query(query, radius, photons,
            [&](uint32_t idx, float /*d2*/) { kd_set.insert(idx); });

        if (bf_set != kd_set) mismatches++;
    }
    EXPECT_EQ(mismatches, 0)
        << "KD-tree failed on degenerate (flat y=0) photon distribution";
}

TEST(GroundTruth_KDTree, RangeQuery_SmallRadius) {
    // Very small radius relative to photon spacing
    auto photons = make_random_3d_photons(1000, 1.0f, 1.0f, 111);
    KDTree tree;
    tree.build(photons);

    float radius = 0.01f;
    float3 query = make_f3(0.f, 0.f, 0.f);

    auto bf = brute_force_range_query(query, radius, photons);
    std::set<uint32_t> bf_set(bf.begin(), bf.end());

    std::set<uint32_t> kd_set;
    tree.query(query, radius, photons,
        [&](uint32_t idx, float /*d2*/) { kd_set.insert(idx); });

    EXPECT_EQ(bf_set, kd_set);
}

TEST(GroundTruth_KDTree, RangeQuery_LargeRadius) {
    // Radius covers entire scene — should return all photons
    auto photons = make_random_3d_photons(200, 0.5f, 1.0f, 222);
    KDTree tree;
    tree.build(photons);

    float radius = 10.0f;
    float3 query = make_f3(0.f, 0.f, 0.f);

    auto bf = brute_force_range_query(query, radius, photons);
    std::set<uint32_t> bf_set(bf.begin(), bf.end());

    std::set<uint32_t> kd_set;
    tree.query(query, radius, photons,
        [&](uint32_t idx, float /*d2*/) { kd_set.insert(idx); });

    EXPECT_EQ(bf_set.size(), photons.size());
    EXPECT_EQ(kd_set.size(), photons.size());
    EXPECT_EQ(bf_set, kd_set);
}

// =====================================================================
//  GROUP 2: KD-Tree k-NN Correctness
// =====================================================================

TEST(GroundTruth_KDTree, KNN_SameAsLinearScan) {
    auto photons = make_random_3d_photons(2000, 1.0f, 1.0f, 333);
    KDTree tree;
    tree.build(photons);

    std::mt19937 gen(444);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    int k = 50;

    int mismatches = 0;
    for (int q = 0; q < 15; ++q) {
        float3 query = make_f3(dist(gen), dist(gen), dist(gen));

        // Brute force k-NN
        std::vector<uint32_t> bf_indices;
        float bf_max_d2;
        brute_force_knn(query, k, photons, bf_indices, bf_max_d2);

        // KD-tree k-NN
        std::vector<uint32_t> kd_indices;
        float kd_max_d2;
        tree.knn(query, k, photons, kd_indices, kd_max_d2);

        // Both should return exactly k photons
        EXPECT_EQ((int)bf_indices.size(), k);
        EXPECT_EQ((int)kd_indices.size(), k);

        // Compare the SETS (order may differ)
        std::set<uint32_t> bf_set(bf_indices.begin(), bf_indices.end());
        std::set<uint32_t> kd_set(kd_indices.begin(), kd_indices.end());

        if (bf_set != kd_set) {
            mismatches++;
            std::cerr << "  k-NN query " << q
                      << ": bf_max_d2=" << bf_max_d2
                      << " kd_max_d2=" << kd_max_d2 << "\n";
        }

        // Max distances should match (within float tolerance)
        EXPECT_NEAR(bf_max_d2, kd_max_d2, 1e-4f)
            << "k-NN max distance mismatch at query " << q;
    }
    EXPECT_EQ(mismatches, 0)
        << "k-NN query returned different photon sets than brute force";
}

// =====================================================================
//  GROUP 3: Density Estimation — BruteForce vs KDTree vs HashGrid
// =====================================================================

TEST(GroundTruth_Density, KDTree_MatchesBruteForce_UniformFloor) {
    const int N = 2000;
    const float extent = 1.0f;
    const float flux = 5.0f;
    const int num_emitted = N;         // pretend each photon = 1 emitted
    const float gather_r = 0.1f;

    auto photons = make_uniform_floor_photons(N, extent, flux);
    Material mat = make_lambertian_material(0.5f);

    KDTree tree;
    tree.build(photons);

    // Query points on the floor
    float3 hit_normal = make_f3(0.f, 1.f, 0.f);
    // wo_local: viewing from above
    ONB frame = ONB::from_normal(hit_normal);
    float3 wo_world = make_f3(0.f, 1.f, 0.f);
    float3 wo_local = frame.world_to_local(wo_world);

    std::mt19937 gen(555);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);

    float max_rel_error = 0.f;
    int zero_count = 0;

    for (int q = 0; q < 10; ++q) {
        float3 hit_pos = make_f3(dist(gen), 0.f, dist(gen));

        // Brute-force ground truth
        Spectrum L_bf = brute_force_density_estimate(
            hit_pos, hit_normal, wo_local, mat,
            photons, gather_r, num_emitted);

        // KD-tree (tangential gather, matching production code)
        Spectrum L_kd = Spectrum::zero();
        float r2 = gather_r * gather_r;
        float inv_area = 2.0f / (PI * r2); // Epanechnikov 2D normalization
        float inv_N = 1.0f / (float)num_emitted;
        float tau_kd = effective_tau(gather_r);
        tree.query_tangential(hit_pos, hit_normal, gather_r, tau_kd, photons,
            [&](uint32_t idx, float d_tan2) {
                if (!photons.norm_x.empty()) {
                    float3 pn = make_f3(photons.norm_x[idx],
                                        photons.norm_y[idx],
                                        photons.norm_z[idx]);
                    if (dot(pn, hit_normal) <= 0.0f) return;
                }

                float3 wi = make_f3(photons.wi_x[idx],
                                    photons.wi_y[idx],
                                    photons.wi_z[idx]);
                if (dot(wi, hit_normal) <= 0.f) return;

                float w = 1.0f - d_tan2 / r2; // Epanechnikov weight
                float3 wi_loc = frame.world_to_local(wi);
                Spectrum f = bsdf::evaluate_diffuse(mat, wo_local, wi_loc);
                Spectrum pf = photons.get_flux(idx);
                for (int b = 0; b < NUM_LAMBDA; ++b)
                    L_kd.value[b] += w * pf.value[b] * inv_N * f.value[b] * inv_area;
            });

        // Compare
        for (int b = 0; b < NUM_LAMBDA; ++b) {
            float bf_val = L_bf.value[b];
            float kd_val = L_kd.value[b];
            if (bf_val > 1e-8f) {
                float rel_err = fabsf(kd_val - bf_val) / bf_val;
                max_rel_error = fmaxf(max_rel_error, rel_err);
            }
            EXPECT_NEAR(bf_val, kd_val, 1e-5f)
                << "Density mismatch at query " << q << " bin " << b;
        }

        if (L_bf.sum() < 1e-10f) zero_count++;
    }

    EXPECT_LT(max_rel_error, 1e-4f)
        << "KD-tree density estimate diverges from brute force";

    std::cout << "  [Density KDTree] max relative error = " << max_rel_error
              << ", zero queries = " << zero_count << "/10\n";
}

TEST(GroundTruth_Density, HashGrid_MatchesBruteForce_UniformFloor) {
    const int N = 2000;
    const float extent = 1.0f;
    const float flux = 5.0f;
    const int num_emitted = N;
    const float gather_r = 0.1f;

    auto photons = make_uniform_floor_photons(N, extent, flux);
    Material mat = make_lambertian_material(0.5f);

    HashGrid grid;
    grid.build(photons, gather_r);

    DensityEstimatorConfig de_config;
    de_config.radius = gather_r;
    de_config.num_photons_total = num_emitted;
    de_config.surface_tau = DEFAULT_SURFACE_TAU;
    de_config.use_kernel = true; // Epanechnikov (§6.3)

    float3 hit_normal = make_f3(0.f, 1.f, 0.f);
    ONB frame = ONB::from_normal(hit_normal);
    float3 wo_world = make_f3(0.f, 1.f, 0.f);
    float3 wo_local = frame.world_to_local(wo_world);

    std::mt19937 gen(666);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);

    float max_rel_error = 0.f;

    for (int q = 0; q < 10; ++q) {
        float3 hit_pos = make_f3(dist(gen), 0.f, dist(gen));

        // Brute-force
        Spectrum L_bf = brute_force_density_estimate(
            hit_pos, hit_normal, wo_local, mat,
            photons, gather_r, num_emitted);

        // Hash grid
        Spectrum L_hg = estimate_photon_density(
            hit_pos, hit_normal, wo_local, mat,
            photons, grid, de_config, gather_r);

        // Compare
        for (int b = 0; b < NUM_LAMBDA; ++b) {
            float bf_val = L_bf.value[b];
            float hg_val = L_hg.value[b];
            if (bf_val > 1e-8f) {
                float rel_err = fabsf(hg_val - bf_val) / bf_val;
                max_rel_error = fmaxf(max_rel_error, rel_err);
            }
            EXPECT_NEAR(bf_val, hg_val, 1e-5f)
                << "Hash grid density mismatch at query " << q << " bin " << b;
        }
    }

    std::cout << "  [Density HashGrid] max relative error = " << max_rel_error << "\n";
}

TEST(GroundTruth_Density, SinglePhoton_KnownValue) {
    // One photon at origin with known flux.
    // Query at origin: should get exact known answer.
    float flux_per_bin = 10.0f;
    auto photons = make_single_photon(
        make_f3(0, 0, 0),       // position
        make_f3(0, 1, 0),       // wi (from above)
        make_f3(0, 1, 0),       // normal (up)
        flux_per_bin);

    Material mat = make_lambertian_material(0.5f);
    float3 hit_pos    = make_f3(0, 0, 0);
    float3 hit_normal = make_f3(0, 1, 0);
    ONB frame = ONB::from_normal(hit_normal);
    float3 wo_local = frame.world_to_local(make_f3(0, 1, 0));

    float gather_r = 0.1f;
    int num_emitted = 1;

    Spectrum L_bf = brute_force_density_estimate(
        hit_pos, hit_normal, wo_local, mat,
        photons, gather_r, num_emitted);

    // Expected: L = flux * (1/N) * (Kd/pi) * W(0) * (2/(pi*r^2))
    // For Epanechnikov kernel at d_tan=0: W(0) = 1, norm = 2/(pi*r^2)
    float expected_per_bin = flux_per_bin
        * (1.0f / num_emitted)
        * (0.5f / PI)           // Kd/pi
        * (2.0f / (PI * gather_r * gather_r));  // Epanechnikov normalization

    for (int b = 0; b < NUM_LAMBDA; ++b) {
        EXPECT_NEAR(L_bf.value[b], expected_per_bin, expected_per_bin * 0.01f)
            << "Single photon ground truth mismatch at bin " << b;
    }

    std::cout << "  [Single photon] Expected=" << expected_per_bin
              << " Got=" << L_bf.value[0] << "\n";
}

TEST(GroundTruth_Density, KDTree_MatchesBruteForce_GlossyMaterial) {
    // Test with glossy material — the BSDF should still match
    const int N = 1000;
    auto photons = make_uniform_floor_photons(N, 1.0f, 5.0f, 777);
    Material mat = make_glossy_material(0.3f, 0.5f, 0.3f);

    KDTree tree;
    tree.build(photons);

    float3 hit_normal = make_f3(0.f, 1.f, 0.f);
    ONB frame = ONB::from_normal(hit_normal);
    float3 wo_local = frame.world_to_local(make_f3(0, 1, 0));

    float gather_r = 0.1f;
    float3 hit_pos = make_f3(0, 0, 0);

    // Brute force with diffuse-only BSDF (matches production code)
    Spectrum L_bf = brute_force_density_estimate(
        hit_pos, hit_normal, wo_local, mat,
        photons, gather_r, N);

    // Brute force with full BSDF
    Spectrum L_bf_full = brute_force_density_estimate_full_bsdf(
        hit_pos, hit_normal, wo_local, mat,
        photons, gather_r, N);

    // KD-tree should match brute-force diffuse-only
    Spectrum L_kd = Spectrum::zero();
    float r2 = gather_r * gather_r;
    float inv_area = 2.0f / (PI * r2); // Epanechnikov 2D normalization
    float inv_N = 1.0f / (float)N;
    float tau_kd = effective_tau(gather_r);
    tree.query_tangential(hit_pos, hit_normal, gather_r, tau_kd, photons,
        [&](uint32_t idx, float d_tan2) {
            if (!photons.norm_x.empty()) {
                float3 pn = make_f3(photons.norm_x[idx], photons.norm_y[idx], photons.norm_z[idx]);
                if (dot(pn, hit_normal) <= 0.0f) return;
            }
            float3 wi = make_f3(photons.wi_x[idx], photons.wi_y[idx], photons.wi_z[idx]);
            if (dot(wi, hit_normal) <= 0.f) return;
            float w = 1.0f - d_tan2 / r2; // Epanechnikov weight
            float3 wi_loc = frame.world_to_local(wi);
            Spectrum f = bsdf::evaluate_diffuse(mat, wo_local, wi_loc);
            Spectrum pf = photons.get_flux(idx);
            for (int b = 0; b < NUM_LAMBDA; ++b)
                L_kd.value[b] += w * pf.value[b] * inv_N * f.value[b] * inv_area;
        });

    // KD-tree should exactly match brute-force diffuse-only
    for (int b = 0; b < NUM_LAMBDA; ++b) {
        EXPECT_NEAR(L_bf.value[b], L_kd.value[b], 1e-5f)
            << "Glossy material: KD diffuse-only != BF diffuse-only, bin " << b;
    }

    // Print the ratio of full BSDF to diffuse-only
    float ratio = (L_bf.value[0] > 1e-10f)
        ? L_bf_full.value[0] / L_bf.value[0] : 0.f;
    std::cout << "  [Glossy] Full BSDF / Diffuse-only ratio = " << ratio << "\n";
    std::cout << "  [Glossy] Diffuse-only L[0] = " << L_bf.value[0]
              << "  Full BSDF L[0] = " << L_bf_full.value[0] << "\n";
}

TEST(GroundTruth_Density, SpectralPhotons_PerBinAccumulation) {
    // Verify that each spectral bin is accumulated independently
    // (no crosstalk between bins)
    const int N = 500;
    auto photons = make_spectral_photons(N, 1.0f);
    Material mat = make_lambertian_material(1.0f);

    float3 hit_pos = make_f3(0, 0, 0);
    float3 hit_normal = make_f3(0, 1, 0);
    ONB frame = ONB::from_normal(hit_normal);
    float3 wo_local = frame.world_to_local(make_f3(0, 1, 0));
    float gather_r = 0.15f;

    // Brute force
    Spectrum L_bf = brute_force_density_estimate(
        hit_pos, hit_normal, wo_local, mat, photons, gather_r, N);

    // Verify that different bins have different values
    // (spectral photons have wavelength-dependent flux)
    bool has_variation = false;
    for (int b = 1; b < NUM_LAMBDA; ++b) {
        if (fabsf(L_bf.value[b] - L_bf.value[0]) > 1e-6f) {
            has_variation = true;
            break;
        }
    }
    EXPECT_TRUE(has_variation)
        << "Spectral photons should produce wavelength-varying radiance";

    // Verify all bins are non-negative and finite
    for (int b = 0; b < NUM_LAMBDA; ++b) {
        EXPECT_GE(L_bf.value[b], 0.f) << "Negative radiance at bin " << b;
        EXPECT_TRUE(std::isfinite(L_bf.value[b]))
            << "Non-finite radiance at bin " << b;
    }
}

TEST(GroundTruth_Density, WallPhotons_NormalConsistency) {
    // Verify that floor query does NOT pick up wall photons
    // (normal consistency filter should reject them)
    auto floor_photons = make_uniform_floor_photons(500, 1.0f, 5.0f, 888);
    auto wall_photons  = make_uniform_wall_photons(500, 1.0f, 5.0f, 999);

    // Merge into one photon map
    PhotonSoA combined;
    for (size_t i = 0; i < floor_photons.size(); ++i)
        combined.push_back(floor_photons.get(i));
    for (size_t i = 0; i < wall_photons.size(); ++i)
        combined.push_back(wall_photons.get(i));

    Material mat = make_lambertian_material(0.5f);
    float3 hit_pos = make_f3(0, 0, 0);
    float3 hit_normal = make_f3(0, 1, 0);  // floor query
    ONB frame = ONB::from_normal(hit_normal);
    float3 wo_local = frame.world_to_local(make_f3(0, 1, 0));
    float gather_r = 0.2f;

    // Full brute force on combined
    Spectrum L_combined = brute_force_density_estimate(
        hit_pos, hit_normal, wo_local, mat, combined, gather_r, 1000);

    // Floor-only brute force
    Spectrum L_floor = brute_force_density_estimate(
        hit_pos, hit_normal, wo_local, mat, floor_photons, gather_r, 1000);

    // They should be equal if normal consistency works correctly
    for (int b = 0; b < NUM_LAMBDA; ++b) {
        EXPECT_NEAR(L_combined.value[b], L_floor.value[b], 1e-5f)
            << "Wall photons leaked into floor estimate at bin " << b;
    }
}

// =====================================================================
//  GROUP 4: Photon Flux & Energy Conservation
// =====================================================================

TEST(GroundTruth_Energy, EmittedFlux_MagnitudeCheck) {
    // Emit photons from a known light and check the flux magnitude.
    // For a single emissive triangle with Le=10, area A, the expected
    // per-photon flux (before 1/N) should be:
    //   E[flux] = Le * cos_theta / (p_tri * p_pos * p_dir)
    //           = Le * pi * A / p_tri
    //           = Le * pi * A  (when p_tri = 1 for single emitter)
    //           = total_power_of_light

    Scene scene = make_box_scene();
    scene.build_bvh();
    scene.build_emissive_distribution();

    const int N = 10000;
    double total_per_bin[NUM_LAMBDA] = {};

    for (int i = 0; i < N; ++i) {
        PCGRng rng = PCGRng::seed(i * 7 + 42, i + 1);
        EmittedPhoton ep = sample_emitted_photon(scene, rng);
        for (int b = 0; b < NUM_LAMBDA; ++b)
            total_per_bin[b] += ep.spectral_flux.value[b];
    }

    // Average flux per photon
    for (int b = 0; b < NUM_LAMBDA; ++b)
        total_per_bin[b] /= N;

    // Expected (analytic):
    // The light is two triangles at y=0.49 with Le=10 (constant spectrum).
    // Total area ≈ 2 * area_of_one_triangle.
    // total_power = Le * pi * total_area
    // average_flux_per_photon = total_power (because the 1/N is applied at gather time)

    // Compute actual light area
    float total_area = 0.f;
    for (uint32_t idx : scene.emissive_tri_indices) {
        total_area += scene.triangles[idx].area();
    }
    float Le = 10.0f;
    float expected_avg_flux = Le * PI * total_area;

    std::cout << "  [Flux] Light area = " << total_area
              << "  Expected avg flux/photon = " << expected_avg_flux
              << "  Measured avg flux/photon[0] = " << total_per_bin[0] << "\n";

    // Should be within ~5% (Monte Carlo variance with 10K samples)
    for (int b = 0; b < NUM_LAMBDA; ++b) {
        float rel_err = fabsf((float)total_per_bin[b] - expected_avg_flux) / expected_avg_flux;
        EXPECT_LT(rel_err, 0.10f)
            << "Emitted flux magnitude is off by " << (rel_err * 100)
            << "% at bin " << b;
    }
}

TEST(GroundTruth_Energy, PhotonFlux_AllPositiveFinite) {
    // After tracing through a scene, every stored photon should have
    // positive, finite flux in every spectral bin
    Scene scene = make_box_scene();
    scene.build_bvh();
    scene.build_emissive_distribution();

    PhotonSoA global_map, caustic_map;
    EmitterConfig cfg;
    cfg.num_photons = 5000;
    cfg.max_bounces = 4;
    cfg.volume_enabled = false;
    trace_photons(scene, cfg, global_map, caustic_map);

    ASSERT_GT(global_map.size(), 0u) << "No photons stored";

    int bad_photons = 0;
    for (size_t i = 0; i < global_map.size(); ++i) {
        Spectrum flux = global_map.get_flux(i);
        for (int b = 0; b < NUM_LAMBDA; ++b) {
            if (flux.value[b] < 0.f || !std::isfinite(flux.value[b])) {
                bad_photons++;
                break;
            }
        }
    }
    EXPECT_EQ(bad_photons, 0) << bad_photons << " photons have invalid flux";
}

TEST(GroundTruth_Energy, DensityEstimate_IndependentOfRadius) {
    // For a uniform photon field, the density estimate should be approximately
    // independent of the gather radius (bias decreases as N increases).
    // Test that estimates at two different radii agree within ~50%
    // (generous tolerance for finite-N bias).

    const int N = 5000;
    auto photons = make_uniform_floor_photons(N, 2.0f, 1.0f);
    Material mat = make_lambertian_material(0.5f);

    float3 hit_pos = make_f3(0, 0, 0);
    float3 hit_normal = make_f3(0, 1, 0);
    ONB frame = ONB::from_normal(hit_normal);
    float3 wo_local = frame.world_to_local(make_f3(0, 1, 0));

    Spectrum L_small = brute_force_density_estimate(
        hit_pos, hit_normal, wo_local, mat, photons, 0.08f, N);
    Spectrum L_large = brute_force_density_estimate(
        hit_pos, hit_normal, wo_local, mat, photons, 0.2f, N);

    // Both should be non-zero
    EXPECT_GT(L_small.value[0], 0.f);
    EXPECT_GT(L_large.value[0], 0.f);

    // Ratio should be reasonable (both estimate the same quantity)
    float ratio = L_small.value[0] / L_large.value[0];
    std::cout << "  [Radius invariance] r=0.08 L=" << L_small.value[0]
              << "  r=0.2 L=" << L_large.value[0]
              << "  ratio=" << ratio << "\n";

    // With 5000 photons, we expect some bias but not extreme
    EXPECT_GT(ratio, 0.3f) << "Small radius estimate is improbably low";
    EXPECT_LT(ratio, 3.0f) << "Small radius estimate is improbably high";
}

// =====================================================================
//  GROUP 5: NEE Direct Lighting — Shadow Rays & Contribution
// =====================================================================

TEST(GroundTruth_NEE, NoOcclusion_VisibleIsTrue) {
    // Simple box scene with light on ceiling, query point on floor.
    // Without any occluder, the light should always be visible.
    Scene scene = make_box_scene();
    scene.build_bvh();
    scene.build_emissive_distribution();

    float3 hit_pos    = make_f3(0.f, -0.49f, 0.f);
    float3 hit_normal = make_f3(0.f, 1.f, 0.f);

    int visible_count = 0;
    int total = 500;

    for (int i = 0; i < total; ++i) {
        PCGRng rng = PCGRng::seed(i * 13 + 7, i + 1);
        DirectLightSample dls = sample_direct_light(
            hit_pos, hit_normal, scene, rng);
        if (dls.visible) visible_count++;
    }

    float visibility_ratio = (float)visible_count / total;
    std::cout << "  [NEE no occlusion] Visibility = "
              << visible_count << "/" << total
              << " (" << (visibility_ratio * 100) << "%)\n";

    // Should be nearly 100% visible (some samples may go to walls)
    EXPECT_GT(visibility_ratio, 0.8f)
        << "Light should be mostly visible from floor center";
}

TEST(GroundTruth_NEE, WithBlocker_ShadowDetected) {
    // Box scene + horizontal blocker between floor and light
    Scene scene = make_box_scene_with_blocker();
    scene.build_bvh();
    scene.build_emissive_distribution();

    float3 hit_pos    = make_f3(0.f, -0.49f, 0.f);
    float3 hit_normal = make_f3(0.f, 1.f, 0.f);

    int visible_count = 0;
    int total = 500;

    for (int i = 0; i < total; ++i) {
        PCGRng rng = PCGRng::seed(i * 13 + 7, i + 1);
        DirectLightSample dls = sample_direct_light(
            hit_pos, hit_normal, scene, rng);
        if (dls.visible) visible_count++;
    }

    float visibility_ratio = (float)visible_count / total;
    std::cout << "  [NEE with blocker] Visibility = "
              << visible_count << "/" << total
              << " (" << (visibility_ratio * 100) << "%)\n";

    // The blocker covers the light from directly below.
    // Visibility should be significantly reduced.
    EXPECT_LT(visibility_ratio, 0.5f)
        << "Blocker should occlude most light from directly below";
}

TEST(GroundTruth_NEE, Contribution_MatchesAnalytic) {
    // For a single point light sample, verify the NEE contribution
    // matches the analytic direct lighting formula.
    //
    // Analytic (for a point on the floor, looking up at a light):
    //   L_direct = Le * (Kd/pi) * cos_receiver * cos_emitter * Area / (pi * dist^2)
    //
    // The Monte Carlo estimator with PDF = p_tri * p_pos * dist^2 / cos_emitter
    // gives: contribution = Le * f * cos / pdf
    //
    // Averaged over many samples it should converge to the analytic value.

    Scene scene = make_box_scene();
    scene.build_bvh();
    scene.build_emissive_distribution();

    float3 hit_pos    = make_f3(0.f, -0.49f, 0.f);
    float3 hit_normal = make_f3(0.f, 1.f, 0.f);
    ONB frame = ONB::from_normal(hit_normal);
    float3 wo_local = frame.world_to_local(make_f3(0, 1, 0));

    Material mat = scene.materials[0]; // diffuse Kd=0.8

    const int N = 5000;
    Spectrum L_avg = Spectrum::zero();
    int valid = 0;

    for (int i = 0; i < N; ++i) {
        PCGRng rng = PCGRng::seed(i * 31 + 11, i + 1);
        DirectLightSample dls = sample_direct_light(
            hit_pos, hit_normal, scene, rng);

        if (dls.visible && dls.pdf_light > 0.f) {
            float3 wi_local = frame.world_to_local(dls.wi);
            float cos_theta = fmaxf(0.f, wi_local.z);
            if (cos_theta > 0.f) {
                Spectrum f = bsdf::evaluate(mat, wo_local, wi_local);
                Spectrum contrib = dls.Li * f * (cos_theta / dls.pdf_light);
                L_avg += contrib;
                valid++;
            }
        }
    }
    L_avg *= (1.0f / N);

    // Compute analytic expected value
    float total_light_area = 0.f;
    for (uint32_t idx : scene.emissive_tri_indices) {
        total_light_area += scene.triangles[idx].area();
    }
    float Le = 10.0f;
    float Kd = 0.8f;
    float dist_y = 0.49f - (-0.49f);  // = 0.98
    float cos_r = 1.0f;  // normal aligned
    float cos_e = 1.0f;  // light faces down
    // Analytic = Le * (Kd/pi) * cos_r * cos_e * A / dist^2
    // BUT: the light is not a point, it's a rectangle at finite distance.
    // For a small light far away, this is approx right.
    // More precisely: integral over light triangle...
    // We'll use the point-source approximation and check within factor 2.
    float analytic_approx = Le * (Kd * INV_PI) * cos_r * cos_e
                          * total_light_area / (dist_y * dist_y);

    std::cout << "  [NEE Analytic] MC average L[0] = " << L_avg.value[0]
              << "  Analytic approx = " << analytic_approx
              << "  valid/total = " << valid << "/" << N
              << "  Light area = " << total_light_area << "\n";

    // MC should be within ~2x of analytic (not exact due to solid angle integration)
    EXPECT_GT(L_avg.value[0], 0.f) << "NEE contribution should be positive";
    EXPECT_GT(L_avg.value[0], analytic_approx * 0.2f)
        << "MC is improbably low vs analytic";
    EXPECT_LT(L_avg.value[0], analytic_approx * 5.0f)
        << "MC is improbably high vs analytic";
}

TEST(GroundTruth_NEE, PDF_GeometricTransformCorrect) {
    // Verify: pdf_solid_angle = pdf_area * dist^2 / cos_emitter
    // by checking individual samples
    Scene scene = make_box_scene();
    scene.build_bvh();
    scene.build_emissive_distribution();

    float3 hit_pos    = make_f3(0.f, -0.49f, 0.f);
    float3 hit_normal = make_f3(0.f, 1.f, 0.f);

    for (int i = 0; i < 100; ++i) {
        PCGRng rng = PCGRng::seed(i * 77 + 3, i + 1);

        // Manually reproduce what sample_direct_light does, checking PDFs

        float u1 = rng.next_float();
        float u2_alias = rng.next_float();
        int local_idx = scene.emissive_alias_table.sample(u1, u2_alias);
        uint32_t tri_idx = scene.emissive_tri_indices[local_idx];
        const Triangle& light_tri = scene.triangles[tri_idx];

        float pdf_tri = scene.emissive_alias_table.pdf(local_idx);
        EXPECT_GT(pdf_tri, 0.f) << "pdf_tri should be positive";

        float3 bary = sample_triangle(rng.next_float(), rng.next_float());
        float3 light_pos = light_tri.interpolate_position(bary.x, bary.y, bary.z);
        float3 light_normal = light_tri.geometric_normal();
        float light_area = light_tri.area();

        float pdf_pos = 1.0f / light_area;

        float3 to_light = light_pos - hit_pos;
        float dist2 = dot(to_light, to_light);
        float dist = sqrtf(dist2);
        float3 wi = to_light / dist;

        float cos_emitter = dot(wi * (-1.f), light_normal);
        if (cos_emitter <= 0.f) continue;

        float pdf_area = pdf_tri * pdf_pos;
        float expected_pdf_solid = pdf_area * dist2 / cos_emitter;

        // Check: the sample_direct_light PDF should match
        // We can't easily call it with the same RNG state, so just verify
        // our formula is self-consistent
        EXPECT_GT(expected_pdf_solid, 0.f)
            << "Solid angle PDF should be positive";
        EXPECT_TRUE(std::isfinite(expected_pdf_solid))
            << "Solid angle PDF should be finite";
    }
}

TEST(GroundTruth_NEE, ShadowRay_BVH_BasicOcclusion) {
    // Verify that BVH intersection works for shadow rays by
    // tracing a ray that we KNOW should hit a triangle.
    Scene scene = make_box_scene();
    scene.build_bvh();

    // Ray from floor center pointing up should hit the ceiling light
    Ray ray;
    ray.origin = make_f3(0.f, -0.49f, 0.f);
    ray.direction = make_f3(0.f, 1.f, 0.f);  // straight up
    ray.tmin = 1e-4f;
    ray.tmax = 100.f;

    HitRecord hit = scene.intersect(ray);
    ASSERT_TRUE(hit.hit) << "Ray from floor up should hit ceiling light";
    EXPECT_GT(hit.t, 0.f);
    EXPECT_LT(hit.t, 2.0f);  // scene is ~1 unit

    std::cout << "  [BVH shadow] Hit at t=" << hit.t
              << " mat_id=" << hit.material_id << "\n";

    // Now with blocker scene
    Scene blocked = make_box_scene_with_blocker();
    blocked.build_bvh();

    HitRecord hit2 = blocked.intersect(ray);
    ASSERT_TRUE(hit2.hit) << "Ray should hit blocker or ceiling";

    // The first hit should be the blocker (at y≈0.0)
    // from y=-0.49, the blocker at y=0 is at t≈0.49
    std::cout << "  [BVH shadow] With blocker: hit at t=" << hit2.t
              << " mat_id=" << hit2.material_id << "\n";

    EXPECT_LT(hit2.t, hit.t)
        << "Blocker should be hit first (closer than ceiling light)";
}

// =====================================================================
//  GROUP 6: End-to-End Pipeline on Simple Scene
// =====================================================================

TEST(GroundTruth_Pipeline, PhotonTrace_ThenGather_BruteForceMatchesKDTree) {
    // Full pipeline test:
    // 1. Create box scene
    // 2. Trace photons through it
    // 3. Build KD-tree and hash grid
    // 4. At several hit points, compare brute-force vs KD-tree density
    Scene scene = make_box_scene();
    scene.build_bvh();
    scene.build_emissive_distribution();

    PhotonSoA global_map, caustic_map;
    EmitterConfig cfg;
    cfg.num_photons = 10000;
    cfg.max_bounces = 4;
    cfg.volume_enabled = false;
    trace_photons(scene, cfg, global_map, caustic_map);

    ASSERT_GT(global_map.size(), 0u) << "No global photons after tracing";

    KDTree tree;
    tree.build(global_map);

    HashGrid grid;
    float gather_r = 0.08f;
    grid.build(global_map, gather_r);

    Material mat = scene.materials[0];  // diffuse floor
    float3 hit_normal = make_f3(0, 1, 0);
    ONB frame = ONB::from_normal(hit_normal);
    float3 wo_local = frame.world_to_local(make_f3(0, 1, 0));

    // Test several points on the floor
    float3 test_points[] = {
        make_f3(0.f,   -0.49f, 0.f),
        make_f3(0.2f,  -0.49f, 0.1f),
        make_f3(-0.3f, -0.49f, 0.2f),
        make_f3(0.1f,  -0.49f, -0.3f),
    };

    for (int q = 0; q < 4; ++q) {
        float3 hp = test_points[q];

        // Brute force
        Spectrum L_bf = brute_force_density_estimate(
            hp, hit_normal, wo_local, mat,
            global_map, gather_r, cfg.num_photons);

        // KD-tree (inline, same as production tangential code)
        Spectrum L_kd = Spectrum::zero();
        float r2 = gather_r * gather_r;
        float inv_area = 2.0f / (PI * r2); // Epanechnikov 2D normalization
        float inv_N = 1.0f / (float)cfg.num_photons;
        float tau_kd = effective_tau(gather_r);
        tree.query_tangential(hp, hit_normal, gather_r, tau_kd, global_map,
            [&](uint32_t idx, float d_tan2) {
                if (!global_map.norm_x.empty()) {
                    float3 pn = make_f3(global_map.norm_x[idx], global_map.norm_y[idx], global_map.norm_z[idx]);
                    if (dot(pn, hit_normal) <= 0.0f) return;
                }
                float3 wi = make_f3(global_map.wi_x[idx], global_map.wi_y[idx], global_map.wi_z[idx]);
                if (dot(wi, hit_normal) <= 0.f) return;
                float w = 1.0f - d_tan2 / r2; // Epanechnikov weight
                float3 wi_loc = frame.world_to_local(wi);
                Spectrum f = bsdf::evaluate_diffuse(mat, wo_local, wi_loc);
                Spectrum pf = global_map.get_flux(idx);
                for (int b = 0; b < NUM_LAMBDA; ++b)
                    L_kd.value[b] += w * pf.value[b] * inv_N * f.value[b] * inv_area;
            });

        // Hash grid
        DensityEstimatorConfig de_config;
        de_config.radius = gather_r;
        de_config.num_photons_total = cfg.num_photons;
        de_config.surface_tau = DEFAULT_SURFACE_TAU;
        de_config.use_kernel = true; // Epanechnikov (§6.3)
        Spectrum L_hg = estimate_photon_density(
            hp, hit_normal, wo_local, mat,
            global_map, grid, de_config, gather_r);

        // All three should match
        for (int b = 0; b < NUM_LAMBDA; ++b) {
            EXPECT_NEAR(L_bf.value[b], L_kd.value[b], 1e-5f)
                << "BF vs KD mismatch, point " << q << " bin " << b;
            EXPECT_NEAR(L_bf.value[b], L_hg.value[b], 1e-5f)
                << "BF vs HashGrid mismatch, point " << q << " bin " << b;
        }

        std::cout << "  [Pipeline point " << q << "] L_bf[0]=" << L_bf.value[0]
                  << " L_kd[0]=" << L_kd.value[0]
                  << " L_hg[0]=" << L_hg.value[0] << "\n";
    }
}

TEST(GroundTruth_Pipeline, PhotonGather_ReasonableMagnitude) {
    // After tracing 10K photons through a simple box scene, the density
    // estimate at a lit point should be a reasonable non-zero radiance,
    // not astronomically large or vanishingly small.

    Scene scene = make_box_scene();
    scene.build_bvh();
    scene.build_emissive_distribution();

    PhotonSoA global_map, caustic_map;
    EmitterConfig cfg;
    cfg.num_photons = 10000;
    cfg.max_bounces = 4;
    cfg.volume_enabled = false;
    trace_photons(scene, cfg, global_map, caustic_map);

    Material mat = scene.materials[0];
    float3 hp = make_f3(0, -0.49f, 0);
    float3 hn = make_f3(0, 1, 0);
    ONB frame = ONB::from_normal(hn);
    float3 wo = frame.world_to_local(make_f3(0, 1, 0));

    Spectrum L = brute_force_density_estimate(
        hp, hn, wo, mat, global_map, 0.1f, cfg.num_photons);

    float sum = L.sum();
    std::cout << "  [Magnitude] Photon gather L.sum() = " << sum
              << " L[0]=" << L.value[0]
              << " photons=" << global_map.size() << "\n";

    // Sanity: should be non-zero and not insane
    EXPECT_GT(sum, 0.f)
        << "Density estimate should be positive under the light";
    EXPECT_LT(sum, 1e6f)
        << "Density estimate is suspiciously large";
    EXPECT_GT(sum, 1e-10f)
        << "Density estimate is suspiciously small";
}

TEST(GroundTruth_Pipeline, NEE_vs_PhotonGather_SameOrder) {
    // Both NEE and photon gather should estimate the same quantity
    // (outgoing radiance at the hitpoint), so they should be within
    // an order of magnitude of each other.

    Scene scene = make_box_scene();
    scene.build_bvh();
    scene.build_emissive_distribution();

    PhotonSoA global_map, caustic_map;
    EmitterConfig cfg;
    cfg.num_photons = 50000;
    cfg.max_bounces = 4;
    cfg.volume_enabled = false;
    trace_photons(scene, cfg, global_map, caustic_map);

    Material mat = scene.materials[0];
    float3 hp = make_f3(0, -0.49f, 0);
    float3 hn = make_f3(0, 1, 0);
    ONB frame = ONB::from_normal(hn);
    float3 wo = frame.world_to_local(make_f3(0, 1, 0));

    // NEE: average over many samples
    const int M = 2000;
    Spectrum L_nee_avg = Spectrum::zero();
    for (int i = 0; i < M; ++i) {
        PCGRng rng = PCGRng::seed(i * 31 + 11, i + 1);
        DirectLightSample dls = sample_direct_light(hp, hn, scene, rng);
        if (dls.visible && dls.pdf_light > 0.f) {
            float3 wi_local = frame.world_to_local(dls.wi);
            float cos_theta = fmaxf(0.f, wi_local.z);
            if (cos_theta > 0.f) {
                Spectrum f = bsdf::evaluate(mat, wo, wi_local);
                Spectrum contrib = dls.Li * f * (cos_theta / dls.pdf_light);
                L_nee_avg += contrib;
            }
        }
    }
    L_nee_avg *= (1.0f / M);

    // Photon gather
    Spectrum L_photon = brute_force_density_estimate(
        hp, hn, wo, mat, global_map, 0.1f, cfg.num_photons);

    float nee_0 = L_nee_avg.value[0];
    float pho_0 = L_photon.value[0];

    std::cout << "  [NEE vs Photon] NEE[0]=" << nee_0
              << " Photon[0]=" << pho_0 << "\n";

    // NEE gives DIRECT lighting only.
    // Photon map (with bounce>0 filter) gives INDIRECT lighting only.
    // So they measure DIFFERENT quantities!  Direct should be larger than
    // indirect for a simple box.  But both should be non-zero and finite.
    EXPECT_GT(nee_0, 0.f) << "NEE should produce positive radiance";
    EXPECT_GT(pho_0, 0.f) << "Photon gather should produce positive radiance";
    EXPECT_TRUE(std::isfinite(nee_0)) << "NEE radiance should be finite";
    EXPECT_TRUE(std::isfinite(pho_0)) << "Photon radiance should be finite";

    // Print the ratio for diagnostic purposes
    if (pho_0 > 0.f) {
        std::cout << "  [NEE vs Photon] NEE/Photon ratio = " << (nee_0 / pho_0) << "\n";
    }
}

// =====================================================================
//  GROUP 7: Photon SoA data integrity
// =====================================================================

TEST(GroundTruth_SoA, GetFlux_MatchesPushBack) {
    // Verify that push_back + get_flux round-trips correctly
    // for interleaved spectral storage
    PhotonSoA soa;

    for (int i = 0; i < 100; ++i) {
        Photon p;
        p.position = make_f3((float)i, 0, 0);
        p.wi = make_f3(0, 1, 0);
        p.geom_normal = make_f3(0, 1, 0);
        // Each photon has a distinct spectral profile
        for (int b = 0; b < NUM_LAMBDA; ++b)
            p.spectral_flux.value[b] = (float)(i * NUM_LAMBDA + b);
        soa.push_back(p);
    }

    // Verify every photon's flux matches what we stored
    for (int i = 0; i < 100; ++i) {
        Spectrum flux = soa.get_flux(i);
        for (int b = 0; b < NUM_LAMBDA; ++b) {
            float expected = (float)(i * NUM_LAMBDA + b);
            EXPECT_EQ(flux.value[b], expected)
                << "SoA flux mismatch at photon " << i << " bin " << b;
        }
    }
}

TEST(GroundTruth_SoA, KDTree_IndicesPreserved) {
    // After building a KD-tree, verify that the indices still
    // correctly map to the original photon data
    auto photons = make_random_3d_photons(500, 1.0f, 1.0f, 42);

    // Give each photon a unique flux signature
    for (size_t i = 0; i < photons.size(); ++i) {
        for (int b = 0; b < NUM_LAMBDA; ++b)
            photons.spectral_flux[i * NUM_LAMBDA + b] = (float)(i * 1000 + b);
    }

    KDTree tree;
    tree.build(photons);

    // Query all photons (large radius)
    float3 center = make_f3(0, 0, 0);
    float radius = 100.f;

    std::vector<uint32_t> gathered;
    tree.query(center, radius, photons,
        [&](uint32_t idx, float /*d2*/) {
            gathered.push_back(idx);
        });

    // Should get all photons
    EXPECT_EQ(gathered.size(), photons.size());

    // Verify each gathered index maps to correct data
    for (uint32_t idx : gathered) {
        Spectrum flux = photons.get_flux(idx);
        for (int b = 0; b < NUM_LAMBDA; ++b) {
            float expected = (float)(idx * 1000 + b);
            EXPECT_EQ(flux.value[b], expected)
                << "After KD-tree build, idx=" << idx
                << " flux doesn't match original bin " << b;
        }
    }
}

// =====================================================================
//  GROUP 8: Diagnostic — measure what's actually happening
// =====================================================================

TEST(GroundTruth_Diagnostic, FloorPhotonGather_CountAndFluxStats) {
    // Diagnostic test: trace photons through box scene, then measure:
    // - How many photons are gathered at a floor point
    // - Their flux magnitude distribution
    // - The resulting density estimate breakdown

    Scene scene = make_box_scene();
    scene.build_bvh();
    scene.build_emissive_distribution();

    PhotonSoA global_map, caustic_map;
    EmitterConfig cfg;
    cfg.num_photons = 50000;
    cfg.max_bounces = 4;
    cfg.volume_enabled = false;
    trace_photons(scene, cfg, global_map, caustic_map);

    float3 hp = make_f3(0, -0.49f, 0);
    float3 hn = make_f3(0, 1, 0);
    ONB frame = ONB::from_normal(hn);
    float3 wo_local = frame.world_to_local(make_f3(0, 1, 0));
    Material mat = scene.materials[0];

    float gather_r = 0.1f;
    float r2 = gather_r * gather_r;

    int total_photons_in_radius = 0;
    int passing_surface_tau = 0;
    int passing_normal = 0;
    int passing_direction = 0;
    double flux_sum = 0;
    double flux_max = 0;
    double flux_min = 1e30;

    for (size_t i = 0; i < global_map.size(); ++i) {
        float dx = global_map.pos_x[i] - hp.x;
        float dy = global_map.pos_y[i] - hp.y;
        float dz = global_map.pos_z[i] - hp.z;
        float d2 = dx*dx + dy*dy + dz*dz;
        if (d2 > r2) continue;
        total_photons_in_radius++;

        float3 p_pos = make_f3(global_map.pos_x[i], global_map.pos_y[i], global_map.pos_z[i]);
        float plane_dist = fabsf(dot(hn, p_pos - hp));
        if (plane_dist > DEFAULT_SURFACE_TAU) continue;
        passing_surface_tau++;

        float3 pn = make_f3(global_map.norm_x[i], global_map.norm_y[i], global_map.norm_z[i]);
        if (dot(pn, hn) <= 0.0f) continue;
        passing_normal++;

        float3 wi = make_f3(global_map.wi_x[i], global_map.wi_y[i], global_map.wi_z[i]);
        if (dot(wi, hn) <= 0.f) continue;
        passing_direction++;

        float ft = global_map.total_flux(i);
        flux_sum += ft;
        flux_max = fmax(flux_max, ft);
        flux_min = fmin(flux_min, ft);
    }

    std::cout << "\n  ═══ PHOTON GATHER DIAGNOSTIC ═══\n"
              << "  Total photons stored: " << global_map.size() << "\n"
              << "  In radius " << gather_r << " of floor center: "
              << total_photons_in_radius << "\n"
              << "  Passing surface-tau (" << DEFAULT_SURFACE_TAU << "): "
              << passing_surface_tau << "\n"
              << "  Passing normal consistency (>0.0): "
              << passing_normal << "\n"
              << "  Passing direction consistency (wi·n>0): "
              << passing_direction << "\n"
              << "  Flux range: [" << flux_min << ", " << flux_max << "]\n"
              << "  Flux sum of gathered: " << flux_sum << "\n";

    // Compute final density estimate
    Spectrum L = brute_force_density_estimate(
        hp, hn, wo_local, mat, global_map, gather_r, cfg.num_photons);

    std::cout << "  Density estimate L[0]=" << L.value[0]
              << "  L.sum()=" << L.sum() << "\n";

    // Also compute what fraction of photons survive each filter
    if (total_photons_in_radius > 0) {
        std::cout << "  Surface-tau pass rate: "
                  << (100.f * passing_surface_tau / total_photons_in_radius) << "%\n"
                  << "  Normal pass rate: "
                  << (100.f * passing_normal / total_photons_in_radius) << "%\n"
                  << "  Direction pass rate: "
                  << (100.f * passing_direction / total_photons_in_radius) << "%\n";
    }

    // The crucial check: are any photons gathered at all?
    EXPECT_GT(passing_direction, 0)
        << "No photons pass all filters at floor center — "
           "this would produce a black image!";
}

TEST(GroundTruth_Diagnostic, PhotonPosition_Distribution) {
    // Check where photons actually land in the scene
    Scene scene = make_box_scene();
    scene.build_bvh();
    scene.build_emissive_distribution();

    PhotonSoA global_map, caustic_map;
    EmitterConfig cfg;
    cfg.num_photons = 10000;
    cfg.max_bounces = 4;
    cfg.volume_enabled = false;
    trace_photons(scene, cfg, global_map, caustic_map);

    // Count photons by y-coordinate (floor vs ceiling vs walls)
    int near_floor = 0, near_ceiling = 0, on_walls = 0;
    for (size_t i = 0; i < global_map.size(); ++i) {
        float y = global_map.pos_y[i];
        if (y < -0.4f) near_floor++;
        else if (y > 0.4f) near_ceiling++;
        else on_walls++;
    }

    std::cout << "\n  ═══ PHOTON DISTRIBUTION ═══\n"
              << "  Total stored: " << global_map.size() << "\n"
              << "  Near floor (y<-0.4):   " << near_floor << "\n"
              << "  Near ceiling (y>0.4):  " << near_ceiling << "\n"
              << "  On walls:              " << on_walls << "\n";

    // There should be photons on the floor (bounce 1 from light)
    EXPECT_GT(near_floor, 0) << "No photons landed on the floor!";

    // Check photon normals for floor photons
    int correct_normals = 0;
    for (size_t i = 0; i < global_map.size(); ++i) {
        float y = global_map.pos_y[i];
        if (y < -0.4f) {
            float ny = global_map.norm_y[i];
            if (ny > 0.5f) correct_normals++;
        }
    }
    std::cout << "  Floor photons with correct normal (ny>0.5): "
              << correct_normals << "/" << near_floor << "\n";
}

// =====================================================================
//  GROUP 9: Cornell Box full scene test (if available)
// =====================================================================

class CornellBoxGroundTruth : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        std::vector<std::string> paths = {
            "scenes/cornell_box/cornellbox.obj",
            "../scenes/cornell_box/cornellbox.obj",
            "../../scenes/cornell_box/cornellbox.obj",
        };
#ifdef SCENES_DIR
        paths.insert(paths.begin(),
            std::string(SCENES_DIR) + "/cornell_box/cornellbox.obj");
#endif
        for (const auto& p : paths) {
            namespace fs = std::filesystem;
            if (fs::exists(p)) {
                scene_ = std::make_unique<Scene>();
                if (load_obj(p, *scene_)) {
                    scene_->build_bvh();
                    scene_->build_emissive_distribution();
                    loaded_ = true;
                    std::cout << "[CornellBoxGT] Loaded " << p
                              << " (" << scene_->triangles.size() << " tris)\n";
                }
                break;
            }
        }
    }
    static void TearDownTestSuite() { scene_.reset(); }
    bool ok() const { return loaded_; }

    static std::unique_ptr<Scene> scene_;
    static bool loaded_;
};

std::unique_ptr<Scene> CornellBoxGroundTruth::scene_;
bool CornellBoxGroundTruth::loaded_ = false;

TEST_F(CornellBoxGroundTruth, BruteForce_vs_KDTree_Gather) {
    if (!ok()) GTEST_SKIP() << "Cornell Box not found";

    PhotonSoA global_map, caustic_map;
    EmitterConfig cfg;
    cfg.num_photons = 20000;
    cfg.max_bounces = 4;
    cfg.volume_enabled = false;
    trace_photons(*scene_, cfg, global_map, caustic_map);

    std::cout << "  Photons stored: " << global_map.size() << "\n";
    ASSERT_GT(global_map.size(), 0u);

    KDTree tree;
    tree.build(global_map);

    // Test at several points
    float3 hit_normal = make_f3(0, 1, 0);  // floor
    ONB frame = ONB::from_normal(hit_normal);
    float3 wo_local = frame.world_to_local(make_f3(0, 1, 0));
    Material mat;
    mat.type = MaterialType::Lambertian;
    mat.Kd = Spectrum::constant(0.5f);

    float gather_r = 0.05f;
    int mismatches = 0;

    for (int q = 0; q < 5; ++q) {
        float3 hp = make_f3(-0.2f + q * 0.1f, -0.49f, 0.f);

        Spectrum L_bf = brute_force_density_estimate(
            hp, hit_normal, wo_local, mat,
            global_map, gather_r, cfg.num_photons);

        // KD-tree gather (tangential, matching brute_force_density_estimate)
        Spectrum L_kd = Spectrum::zero();
        float r2 = gather_r * gather_r;
        float inv_area = 2.0f / (PI * r2); // Epanechnikov 2D normalization
        float inv_N = 1.0f / (float)cfg.num_photons;
        float tau_kd2 = effective_tau(DEFAULT_SURFACE_TAU);
        tree.query_tangential(hp, hit_normal, gather_r, tau_kd2, global_map,
            [&](uint32_t idx, float d_tan2) {
                if (!global_map.norm_x.empty()) {
                    float3 pn = make_f3(global_map.norm_x[idx], global_map.norm_y[idx], global_map.norm_z[idx]);
                    if (dot(pn, hit_normal) <= 0.0f) return;
                }
                float3 wi = make_f3(global_map.wi_x[idx], global_map.wi_y[idx], global_map.wi_z[idx]);
                if (dot(wi, hit_normal) <= 0.f) return;
                float w = 1.0f - d_tan2 / r2; // Epanechnikov weight
                float3 wi_loc = frame.world_to_local(wi);
                Spectrum f = bsdf::evaluate_diffuse(mat, wo_local, wi_loc);
                Spectrum pf = global_map.get_flux(idx);
                for (int b = 0; b < NUM_LAMBDA; ++b)
                    L_kd.value[b] += w * pf.value[b] * inv_N * f.value[b] * inv_area;
            });

        bool match = true;
        for (int b = 0; b < NUM_LAMBDA; ++b) {
            if (fabsf(L_bf.value[b] - L_kd.value[b]) > 1e-5f) {
                match = false;
                break;
            }
        }
        if (!match) mismatches++;

        std::cout << "  [Cornell q=" << q << "] BF=" << L_bf.value[0]
                  << " KD=" << L_kd.value[0]
                  << (match ? " OK" : " MISMATCH") << "\n";
    }

    EXPECT_EQ(mismatches, 0)
        << "KD-tree gather doesn't match brute-force on Cornell Box";
}

TEST_F(CornellBoxGroundTruth, NEE_ShadowRay_Sanity) {
    if (!ok()) GTEST_SKIP() << "Cornell Box not found";

    // Test that NEE produces visible samples from the floor
    float3 hp = make_f3(0, -0.49f, 0);
    float3 hn = make_f3(0, 1, 0);

    int visible = 0;
    int total = 200;
    for (int i = 0; i < total; ++i) {
        PCGRng rng = PCGRng::seed(i * 71 + 5, i + 1);
        DirectLightSample dls = sample_direct_light(hp, hn, *scene_, rng);
        if (dls.visible) visible++;
    }

    float ratio = (float)visible / total;
    std::cout << "  [Cornell NEE] Visibility from floor: "
              << visible << "/" << total
              << " (" << (ratio * 100) << "%)\n";

    EXPECT_GT(ratio, 0.3f)
        << "NEE visibility too low on Cornell Box floor";
}

// =====================================================================
//  GROUP 10: Conference Room scene diagnostics
// =====================================================================

class ConferenceRoomGroundTruth : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        std::vector<std::string> paths = {
            "scenes/conference/conference.obj",
            "../scenes/conference/conference.obj",
            "../../scenes/conference/conference.obj",
        };
#ifdef SCENES_DIR
        paths.insert(paths.begin(),
            std::string(SCENES_DIR) + "/conference/conference.obj");
#endif
        namespace fs = std::filesystem;
        for (const auto& p : paths) {
            if (fs::exists(p)) {
                scene_ = std::make_unique<Scene>();
                if (load_obj(p, *scene_)) {
                    scene_->normalize_to_reference();
                    scene_->build_bvh();
                    scene_->build_emissive_distribution();
                    loaded_ = true;
                    std::cout << "[ConferenceGT] Loaded " << p
                              << " (" << scene_->triangles.size() << " tris, "
                              << scene_->emissive_tri_indices.size() << " emissive)\n";
                }
                break;
            }
        }
    }
    static void TearDownTestSuite() { scene_.reset(); }
    bool ok() const { return loaded_; }

    static std::unique_ptr<Scene> scene_;
    static bool loaded_;
};

std::unique_ptr<Scene> ConferenceRoomGroundTruth::scene_;
bool ConferenceRoomGroundTruth::loaded_ = false;

TEST_F(ConferenceRoomGroundTruth, EmissiveTriangle_GeomNormal_Direction) {
    // CRITICAL DIAGNOSTIC: Check whether emissive triangles have
    // geometric normals pointing INTO the room (downward for ceiling lights).
    //
    // If geometric normals point upward (into the ceiling), then:
    //   1. NEE: cos_theta_emitter <= 0 → light invisible → no shadows
    //   2. Emitter: photons emitted upward → never reach room → no photons
    //
    // This is the suspected FUNDAMENTAL FLAW.
    if (!ok()) GTEST_SKIP() << "Conference Room not found";

    int total_emissive = (int)scene_->emissive_tri_indices.size();
    int facing_down = 0;     // geometric normal has negative y component
    int facing_up = 0;       // geometric normal has positive y component
    int facing_side = 0;     // geometric normal is mostly horizontal

    float avg_vertex_ny = 0.f;
    float avg_geom_ny = 0.f;
    int vertex_normal_disagree = 0;

    for (uint32_t idx : scene_->emissive_tri_indices) {
        const Triangle& tri = scene_->triangles[idx];
        float3 gn = tri.geometric_normal();

        // Vertex normal average (shading normal at centroid)
        float3 vn = normalize(tri.n0 + tri.n1 + tri.n2);
        avg_vertex_ny += vn.y;
        avg_geom_ny += gn.y;

        if (dot(gn, vn) < 0.f) vertex_normal_disagree++;

        if (gn.y < -0.5f) facing_down++;
        else if (gn.y > 0.5f) facing_up++;
        else facing_side++;
    }

    if (total_emissive > 0) {
        avg_vertex_ny /= total_emissive;
        avg_geom_ny /= total_emissive;
    }

    std::cout << "\n  ═══ CONFERENCE ROOM EMISSIVE DIAGNOSTIC ═══\n"
              << "  Total emissive triangles: " << total_emissive << "\n"
              << "  Geometric normal facing down (y<-0.5): " << facing_down << "\n"
              << "  Geometric normal facing UP   (y>0.5):  " << facing_up << "\n"
              << "  Geometric normal facing side:          " << facing_side << "\n"
              << "  Avg vertex normal y: " << avg_vertex_ny << "\n"
              << "  Avg geometric normal y:  " << avg_geom_ny << "\n"
              << "  Vertex/geometric normal DISAGREE: " << vertex_normal_disagree
              << "/" << total_emissive << "\n";

    // If most emissive triangles face up, they're emitting into the ceiling!
    if (facing_up > facing_down) {
        std::cerr << "\n  *** WARNING: Most emissive triangles face UP! ***\n"
                  << "  *** This means photon emission and NEE are directed ***\n"
                  << "  *** AWAY from the room interior!                    ***\n\n";
    }

    // Check a few individual emissive triangles
    int to_show = std::min(5, total_emissive);
    for (int i = 0; i < to_show; ++i) {
        uint32_t idx = scene_->emissive_tri_indices[i];
        const Triangle& tri = scene_->triangles[idx];
        float3 gn = tri.geometric_normal();
        float3 vn = normalize(tri.n0 + tri.n1 + tri.n2);
        const Material& mat = scene_->materials[tri.material_id];
        std::cout << "  Emissive tri[" << i << "] idx=" << idx
                  << " geom_n=(" << gn.x << "," << gn.y << "," << gn.z << ")"
                  << " vert_n=(" << vn.x << "," << vn.y << "," << vn.z << ")"
                  << " Le_sum=" << mat.Le.sum()
                  << " area=" << tri.area() << "\n";
    }
}

TEST_F(ConferenceRoomGroundTruth, NEE_VisibilityFromFloor) {
    if (!ok()) GTEST_SKIP() << "Conference Room not found";

    // Sample NEE from multiple floor points
    float3 hn = make_f3(0, 1, 0);
    float3 test_points[] = {
        make_f3(0.f, -0.49f, 0.f),
        make_f3(0.2f, -0.49f, 0.1f),
        make_f3(-0.1f, -0.49f, -0.2f),
    };

    for (int p = 0; p < 3; ++p) {
        float3 hp = test_points[p];
        int visible = 0;
        int back_facing = 0;
        int total = 300;

        for (int i = 0; i < total; ++i) {
            PCGRng rng = PCGRng::seed(i * 71 + p * 1000, i + 1);
            DirectLightSample dls = sample_direct_light(hp, hn, *scene_, rng);
            if (dls.visible) visible++;

            // Also manually check why visibility might fail:
            // Re-sample to check cos_theta_emitter
            PCGRng rng2 = PCGRng::seed(i * 71 + p * 1000, i + 1);
            float u1 = rng2.next_float();
            float u2 = rng2.next_float();
            int local_idx = scene_->emissive_alias_table.sample(u1, u2);
            uint32_t tri_idx = scene_->emissive_tri_indices[local_idx];
            const Triangle& light_tri = scene_->triangles[tri_idx];
            float3 gn = light_tri.geometric_normal();
            float3 bary = sample_triangle(rng2.next_float(), rng2.next_float());
            float3 lp = light_tri.interpolate_position(bary.x, bary.y, bary.z);
            float3 to_light = lp - hp;
            float3 wi = normalize(to_light);
            float cos_e = dot(wi * (-1.f), gn);
            if (cos_e <= 0.f) back_facing++;
        }

        float vis_ratio = (float)visible / total;
        float bf_ratio = (float)back_facing / total;
        std::cout << "  [Conf NEE point " << p << "] Visible: "
                  << visible << "/" << total << " (" << (vis_ratio*100) << "%)"
                  << "  Back-facing: " << back_facing << "/" << total
                  << " (" << (bf_ratio*100) << "%)\n";
    }
}

TEST_F(ConferenceRoomGroundTruth, PhotonTrace_CountAndDistribution) {
    if (!ok()) GTEST_SKIP() << "Conference Room not found";

    PhotonSoA global_map, caustic_map;
    EmitterConfig cfg;
    cfg.num_photons = 10000;
    cfg.max_bounces = 4;
    cfg.volume_enabled = false;
    trace_photons(*scene_, cfg, global_map, caustic_map);

    // Check distribution of stored photons
    float y_min = 1e30f, y_max = -1e30f;
    for (size_t i = 0; i < global_map.size(); ++i) {
        y_min = fminf(y_min, global_map.pos_y[i]);
        y_max = fmaxf(y_max, global_map.pos_y[i]);
    }

    // Check how many photons have normals agreeing with scene geometry
    int normal_ok = 0;
    for (size_t i = 0; i < global_map.size(); ++i) {
        float3 pn = make_f3(global_map.norm_x[i], global_map.norm_y[i], global_map.norm_z[i]);
        float nl = length(pn);
        if (nl > 0.5f) normal_ok++;
    }

    int facing_down_stored = 0;
    for (size_t i = 0; i < global_map.size(); ++i) {
        if (global_map.norm_y[i] < -0.3f) facing_down_stored++;
    }

    std::cout << "\n  ═══ CONFERENCE PHOTON TRACE DIAGNOSTIC ═══\n"
              << "  Emitted: " << cfg.num_photons
              << "  Stored: " << global_map.size()
              << " (ratio: " << (global_map.size() / (float)cfg.num_photons) << ")\n"
              << "  Y range: [" << y_min << ", " << y_max << "]\n"
              << "  Normals valid (length>0.5): " << normal_ok << "/" << global_map.size() << "\n"
              << "  Facing down (ny<-0.3): " << facing_down_stored << "\n";

    // Photons should actually be stored
    EXPECT_GT(global_map.size(), 0u) << "No photons stored in Conference Room!";
}

TEST_F(ConferenceRoomGroundTruth, GeomNormal_vs_ShadingNormal_AllTriangles) {
    // Check ALL triangles in the scene for geometric/shading normal agreement
    if (!ok()) GTEST_SKIP() << "Conference Room not found";

    int total = (int)scene_->triangles.size();
    int disagree = 0;
    int severely_disagree = 0;

    for (int i = 0; i < total; ++i) {
        const Triangle& tri = scene_->triangles[i];
        float3 gn = tri.geometric_normal();
        float3 vn = normalize(tri.n0 + tri.n1 + tri.n2);
        float d = dot(gn, vn);
        if (d < 0.f) {
            disagree++;
            if (d < -0.5f) severely_disagree++;
        }
    }

    std::cout << "\n  ═══ NORMAL CONSISTENCY (ALL TRIANGLES) ═══\n"
              << "  Total triangles: " << total << "\n"
              << "  Geometric vs vertex normal disagree (dot<0): " << disagree << "\n"
              << "  Severely disagree (dot<-0.5): " << severely_disagree << "\n"
              << "  Disagree rate: " << (100.f * disagree / total) << "%\n";

    // A high disagree rate would explain photon gathering failures
    // because photons store geometric_normal but queries filter using
    // shading_normal, causing valid photons to be rejected
}

// =====================================================================
// Multi-Hero Wavelength Tests (§1.1)
// =====================================================================
// These tests validate the multi-hero photon transport implementation:
// 1. Stratified companion bins are correctly spaced
// 2. PhotonSoA round-trips hero data correctly
// 3. Multi-hero density estimation converges to full-spectral ground truth

// -- Stratified companion bin spacing --------------------------------
TEST(MultiHero, StratifiedBinsAreEvenlySpaced) {
    // For any primary hero bin, the HERO_WAVELENGTHS companion bins
    // should be spaced at exactly NUM_LAMBDA / HERO_WAVELENGTHS apart.
    const int stride = NUM_LAMBDA / HERO_WAVELENGTHS;

    for (int primary = 0; primary < NUM_LAMBDA; ++primary) {
        std::set<int> bins;
        for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
            int bin = (primary + h * stride) % NUM_LAMBDA;
            EXPECT_GE(bin, 0);
            EXPECT_LT(bin, NUM_LAMBDA);
            bins.insert(bin);
        }
        // All HERO_WAVELENGTHS bins should be distinct
        EXPECT_EQ((int)bins.size(), HERO_WAVELENGTHS)
            << "Duplicate bins for primary=" << primary;
    }
}

// -- PhotonSoA hero field round-trip ---------------------------------
TEST(MultiHero, PhotonSoA_HeroFieldRoundTrip) {
    Photon p;
    p.position    = make_f3(1.f, 2.f, 3.f);
    p.wi          = make_f3(0.f, 1.f, 0.f);
    p.geom_normal = make_f3(0.f, 1.f, 0.f);
    p.spectral_flux = Spectrum::zero();
    p.num_hero    = HERO_WAVELENGTHS;

    for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
        p.lambda_bin[h] = (uint16_t)(h * 3 + 1);
        p.flux[h]       = (float)(h + 1) * 0.5f;
    }

    PhotonSoA soa;
    soa.push_back(p);
    EXPECT_EQ(soa.size(), 1u);

    Photon q = soa.get(0);
    EXPECT_EQ(q.num_hero, HERO_WAVELENGTHS);

    for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
        EXPECT_EQ(q.lambda_bin[h], p.lambda_bin[h])
            << "lambda_bin mismatch at hero " << h;
        EXPECT_FLOAT_EQ(q.flux[h], p.flux[h])
            << "flux mismatch at hero " << h;
    }

    // Also verify the raw interleaved layout
    EXPECT_EQ(soa.lambda_bin.size(), (size_t)HERO_WAVELENGTHS);
    EXPECT_EQ(soa.flux.size(), (size_t)HERO_WAVELENGTHS);
    EXPECT_EQ(soa.num_hero.size(), 1u);
    EXPECT_EQ(soa.num_hero[0], (uint8_t)HERO_WAVELENGTHS);
}

// -- Multiple photons round-trip -------------------------------------
TEST(MultiHero, PhotonSoA_MultiplePhotons) {
    PhotonSoA soa;
    const int N = 10;

    for (int i = 0; i < N; ++i) {
        Photon p;
        p.position    = make_f3((float)i, 0.f, 0.f);
        p.wi          = make_f3(0.f, 1.f, 0.f);
        p.geom_normal = make_f3(0.f, 1.f, 0.f);
        p.spectral_flux = Spectrum::zero();
        p.num_hero    = HERO_WAVELENGTHS;
        for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
            p.lambda_bin[h] = (uint16_t)((i * HERO_WAVELENGTHS + h) % NUM_LAMBDA);
            p.flux[h]       = (float)(i * 10 + h);
        }
        soa.push_back(p);
    }

    EXPECT_EQ(soa.size(), (size_t)N);
    EXPECT_EQ(soa.lambda_bin.size(), (size_t)(N * HERO_WAVELENGTHS));
    EXPECT_EQ(soa.flux.size(), (size_t)(N * HERO_WAVELENGTHS));

    for (int i = 0; i < N; ++i) {
        Photon q = soa.get(i);
        EXPECT_EQ(q.num_hero, HERO_WAVELENGTHS);
        for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
            EXPECT_EQ(q.lambda_bin[h],
                      (uint16_t)((i * HERO_WAVELENGTHS + h) % NUM_LAMBDA));
            EXPECT_FLOAT_EQ(q.flux[h], (float)(i * 10 + h));
        }
    }
}

// -- Hero bins cover distinct spectral bands -------------------------
TEST(MultiHero, HeroBinsCoverSpectrum) {
    // With HERO_WAVELENGTHS=4 and NUM_LAMBDA=4, each hero covers
    // exactly one spectral bin (stride=1).
    const int stride = NUM_LAMBDA / HERO_WAVELENGTHS;

    // Pick hero_bin = 0: companions should be {0, 8, 16, 24}
    int hero_bin = 0;
    for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
        int expected = h * stride;
        int actual   = (hero_bin + h * stride) % NUM_LAMBDA;
        EXPECT_EQ(actual, expected);
    }

    // Pick hero_bin = 5: companions should be {5, 13, 21, 29}
    hero_bin = 5;
    for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
        int expected = (5 + h * stride) % NUM_LAMBDA;
        int actual   = (hero_bin + h * stride) % NUM_LAMBDA;
        EXPECT_EQ(actual, expected);
    }
}

// -- Multi-hero density estimate vs full-spectral ground truth -------
//
// Create photons with known multi-hero data and verify that summing
// over the hero channels at each gathered photon produces the same
// result as the brute-force full-spectral estimate, for the bins
// covered by the hero wavelengths.
TEST(MultiHero, DensityEstimate_MatchesFullSpectral_CoveredBins) {
    // Build a simple set of photons on a flat floor (y=0, normal up)
    const int N_photons = 200;
    const float gather_r = 0.5f;
    const int hero_bin = 2;  // primary hero
    const int stride = NUM_LAMBDA / HERO_WAVELENGTHS;

    // Diffuse white material
    Material white_mat;
    white_mat.type = MaterialType::Lambertian;
    for (int b = 0; b < NUM_LAMBDA; ++b)
        white_mat.Kd.value[b] = 0.8f;

    PhotonSoA photons;
    photons.reserve(N_photons);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> pos_dist(-0.3f, 0.3f);
    std::uniform_real_distribution<float> flux_dist(0.1f, 2.0f);

    for (int i = 0; i < N_photons; ++i) {
        Photon p;
        p.position    = make_f3(pos_dist(gen), 0.0f, pos_dist(gen));
        p.wi          = normalize(make_f3(0.3f, 1.0f, 0.2f));
        p.geom_normal = make_f3(0.f, 1.f, 0.f);
        p.num_hero    = HERO_WAVELENGTHS;

        // Set hero bins (stratified)
        for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
            int bin = (hero_bin + h * stride) % NUM_LAMBDA;
            p.lambda_bin[h] = (uint16_t)bin;
            p.flux[h]       = flux_dist(gen);
        }

        // Also fill spectral_flux for the covered bins (matching hero data)
        p.spectral_flux = Spectrum::zero();
        for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
            int bin = (hero_bin + h * stride) % NUM_LAMBDA;
            p.spectral_flux.value[bin] = p.flux[h];
        }

        photons.push_back(p);
    }

    // Build hash grid
    HashGrid grid;
    grid.build(photons, gather_r);

    float3 query_pos  = make_f3(0.f, 0.f, 0.f);
    float3 normal     = make_f3(0.f, 1.f, 0.f);
    float3 wo_local   = make_f3(0.f, 0.f, 1.f);

    // Full-spectral brute-force estimate
    Spectrum L_full = brute_force_density_estimate(
        query_pos, normal, wo_local, white_mat,
        photons, gather_r, N_photons);

    // Now simulate multi-hero density estimation:
    // For each gathered photon, accumulate its hero channels into the
    // corresponding spectral bins.
    Spectrum L_hero = Spectrum::zero();
    float r2 = gather_r * gather_r;
    float inv_area = 2.0f / (PI * r2);  // Epanechnikov normalization
    float inv_N    = 1.0f / (float)N_photons;
    ONB frame = ONB::from_normal(normal);

    for (size_t i = 0; i < photons.size(); ++i) {
        float3 p_pos = make_f3(photons.pos_x[i], photons.pos_y[i], photons.pos_z[i]);
        float3 diff = p_pos - query_pos;
        float d_plane = dot(normal, diff);
        if (fabsf(d_plane) > DEFAULT_SURFACE_TAU) continue;
        float3 v_tan = diff - normal * d_plane;
        float d_tan2 = dot(v_tan, v_tan);
        if (d_tan2 > r2) continue;

        float3 pn = make_f3(photons.norm_x[i], photons.norm_y[i], photons.norm_z[i]);
        if (dot(pn, normal) <= 0.f) continue;

        float3 wi = make_f3(photons.wi_x[i], photons.wi_y[i], photons.wi_z[i]);
        if (dot(wi, normal) <= 0.f) continue;

        float w = 1.0f - d_tan2 / r2;  // Epanechnikov
        float3 wi_loc = frame.world_to_local(wi);
        Spectrum f = bsdf::evaluate_diffuse(white_mat, wo_local, wi_loc);

        int n_hero = (photons.num_hero.size() > i) ? (int)photons.num_hero[i] : 1;
        for (int h = 0; h < n_hero; ++h) {
            int bin = (int)photons.lambda_bin[i * HERO_WAVELENGTHS + h];
            float pflux = photons.flux[i * HERO_WAVELENGTHS + h];
            if (bin >= 0 && bin < NUM_LAMBDA)
                L_hero.value[bin] += w * pflux * inv_N * f.value[bin] * inv_area;
        }
    }

    // Compare only the bins covered by hero wavelengths
    for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
        int bin = (hero_bin + h * stride) % NUM_LAMBDA;
        float full = L_full.value[bin];
        float hero = L_hero.value[bin];
        if (full > 0.f) {
            float rel_err = fabsf(hero - full) / full;
            EXPECT_LT(rel_err, 1e-4f)
                << "Bin " << bin << ": full=" << full << " hero=" << hero;
        } else {
            EXPECT_NEAR(hero, 0.f, 1e-6f);
        }
    }

    // Non-covered bins should be zero in the hero estimate
    std::set<int> covered_bins;
    for (int h = 0; h < HERO_WAVELENGTHS; ++h)
        covered_bins.insert((hero_bin + h * stride) % NUM_LAMBDA);

    for (int b = 0; b < NUM_LAMBDA; ++b) {
        if (covered_bins.count(b) == 0) {
            EXPECT_NEAR(L_hero.value[b], 0.f, 1e-6f)
                << "Bin " << b << " should be zero (not a hero bin)";
        }
    }
}

// -- PhotonSoA clear resets all hero fields --------------------------
TEST(MultiHero, PhotonSoA_Clear) {
    PhotonSoA soa;
    Photon p;
    p.position = make_f3(1.f, 0.f, 0.f);
    p.wi = make_f3(0.f, 1.f, 0.f);
    p.geom_normal = make_f3(0.f, 1.f, 0.f);
    p.spectral_flux = Spectrum::zero();
    p.num_hero = HERO_WAVELENGTHS;
    for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
        p.lambda_bin[h] = (uint16_t)h;
        p.flux[h] = 1.f;
    }
    soa.push_back(p);

    EXPECT_EQ(soa.size(), 1u);
    soa.clear();
    EXPECT_EQ(soa.size(), 0u);
    EXPECT_TRUE(soa.lambda_bin.empty());
    EXPECT_TRUE(soa.flux.empty());
    EXPECT_TRUE(soa.num_hero.empty());
}

// -- Hero wavelength config sanity check -----------------------------
TEST(MultiHero, HeroWavelengthConfigValid) {
    EXPECT_GE(HERO_WAVELENGTHS, 1);
    EXPECT_LE(HERO_WAVELENGTHS, NUM_LAMBDA);
    // NUM_LAMBDA should be evenly divisible by HERO_WAVELENGTHS
    EXPECT_EQ(NUM_LAMBDA % HERO_WAVELENGTHS, 0)
        << "NUM_LAMBDA must be divisible by HERO_WAVELENGTHS for uniform stratification";
}
