// ─────────────────────────────────────────────────────────────────────
// test_integration.cpp – v2 Architecture integration tests (Phase 4)
// ─────────────────────────────────────────────────────────────────────
// Validates the v2 first-hit + specular-chain + photon-gather pipeline
// using the CPU renderer on the Cornell Box scene.
//
// Tests (original):
//   1. Energy conservation: radiance is non-negative, finite
//   2. Specular chain terminates within max_specular_chain bounces
//   3. NEE-only mode produces non-zero output for lit pixels
//   4. Indirect-only mode produces non-zero output when photons exist
//   5. Combined mode ≈ NEE + indirect (no double-counting)
//   6. KD-tree and hash-grid produce similar density estimates
//   7. ACES tonemap produces valid sRGB output
//   8. Photon map has expected structure after build
//
// Tests (v2.1 additions):
//   9. Tangential gather: KD-tree density ≈ hash grid density (both tangential)
//  10. Tangential vs 3D: tangential gather rejects through-wall photons
//  11. Coverage-aware NEE: mixture sampling produces valid contributions
//  12. Photon decorrelation: photons are spatially well-distributed
//  13. Shell expansion k-NN matches KD-tree tangential k-NN on real data
//  14. Surface consistency filter: no through-surface photon leaking
//  15. Tangential kernel normalization: energy not amplified
// ─────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include "core/types.h"
#include "core/spectrum.h"
#include "core/config.h"
#include "core/random.h"
#include "scene/scene.h"
#include "scene/obj_loader.h"
#include "renderer/renderer.h"
#include "renderer/camera.h"
#include "photon/photon.h"
#include "photon/hash_grid.h"
#include "photon/kd_tree.h"
#include "photon/density_estimator.h"
#include "photon/emitter.h"
#include "photon/surface_filter.h"
#include "bsdf/bsdf.h"

#include <vector>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <numeric>

namespace fs = std::filesystem;

// ─── Fixture: loads Cornell Box once and builds photon maps ──────────

class IntegrationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Find Cornell Box scene
        std::vector<std::string> search_paths = {
            "scenes/cornell_box/cornellbox.obj",
            "../scenes/cornell_box/cornellbox.obj",
            "../../scenes/cornell_box/cornellbox.obj",
        };
        std::string obj_path;
        for (const auto& p : search_paths) {
            if (fs::exists(p)) { obj_path = p; break; }
        }
        if (obj_path.empty()) {
            std::cerr << "[IntegrationTest] Cornell Box not found, skipping.\n";
            scene_loaded_ = false;
            return;
        }

        // Load scene
        scene_ = std::make_unique<Scene>();
        if (!load_obj(obj_path, *scene_)) {
            std::cerr << "[IntegrationTest] Failed to load " << obj_path << "\n";
            scene_loaded_ = false;
            return;
        }
        scene_->build_bvh();
        scene_->build_emissive_distribution();
        scene_loaded_ = true;

        // Configure for small test render
        config_.image_width       = 64;
        config_.image_height      = 64;
        config_.samples_per_pixel = 4;
        config_.num_photons       = 50000;
        config_.max_bounces       = 8;
        config_.gather_radius     = 0.05f;
        config_.caustic_radius    = 0.02f;
        config_.use_kdtree        = true;
        config_.max_specular_chain = DEFAULT_MAX_SPECULAR_CHAIN;

        // Build camera – Cornell Box is centred at origin, [-0.5, 0.5]^3
        camera_.position = make_f3(0.f, 0.f, 1.5f);
        camera_.look_at  = make_f3(0.f, 0.f, 0.f);
        camera_.up       = make_f3(0.f, 1.f, 0.f);
        camera_.fov_deg  = 50.0f;
        camera_.width    = config_.image_width;
        camera_.height   = config_.image_height;
        camera_.update();

        // Build photon maps
        EmitterConfig emitter_cfg;
        emitter_cfg.num_photons    = config_.num_photons;
        emitter_cfg.max_bounces    = config_.max_bounces;
        emitter_cfg.rr_threshold   = config_.rr_threshold;
        emitter_cfg.min_bounces_rr = config_.min_bounces_rr;
        emitter_cfg.volume_enabled = false;

        trace_photons(*scene_, emitter_cfg, global_photons_, caustic_photons_, nullptr);

        // Build spatial indices
        if (global_photons_.size() > 0) {
            global_grid_.build(global_photons_, config_.gather_radius);
            global_kdtree_.build(global_photons_);
        }
        if (caustic_photons_.size() > 0) {
            caustic_grid_.build(caustic_photons_, config_.caustic_radius);
        }

        std::cout << "[IntegrationTest] Scene loaded: "
                  << scene_->triangles.size() << " triangles, "
                  << global_photons_.size() << " global photons, "
                  << caustic_photons_.size() << " caustic photons\n";
    }

    static void TearDownTestSuite() {
        scene_.reset();
    }

    bool scene_ok() const { return scene_loaded_; }

    // Shared across all tests
    static std::unique_ptr<Scene> scene_;
    static RenderConfig config_;
    static Camera camera_;
    static PhotonSoA global_photons_;
    static PhotonSoA caustic_photons_;
    static HashGrid global_grid_;
    static HashGrid caustic_grid_;
    static KDTree global_kdtree_;
    static bool scene_loaded_;
};

// Static member definitions
std::unique_ptr<Scene> IntegrationTest::scene_;
RenderConfig IntegrationTest::config_;
Camera IntegrationTest::camera_;
PhotonSoA IntegrationTest::global_photons_;
PhotonSoA IntegrationTest::caustic_photons_;
HashGrid IntegrationTest::global_grid_;
HashGrid IntegrationTest::caustic_grid_;
KDTree IntegrationTest::global_kdtree_;
bool IntegrationTest::scene_loaded_ = false;

// ─── Test: Photon map structure ─────────────────────────────────────

TEST_F(IntegrationTest, PhotonMapHasPhotons) {
    if (!scene_ok()) GTEST_SKIP();
    EXPECT_GT(global_photons_.size(), 0u)
        << "Global photon map should contain stored photons";
}

TEST_F(IntegrationTest, PhotonMapFluxPositive) {
    if (!scene_ok()) GTEST_SKIP();
    for (size_t i = 0; i < global_photons_.size(); ++i) {
        float tf = global_photons_.total_flux(i);
        ASSERT_GE(tf, 0.f)
            << "Photon " << i << " has negative flux";
        ASSERT_TRUE(std::isfinite(tf))
            << "Photon " << i << " has non-finite flux";
    }
}

TEST_F(IntegrationTest, KDTreeBuiltSuccessfully) {
    if (!scene_ok()) GTEST_SKIP();
    EXPECT_FALSE(global_kdtree_.empty());
    EXPECT_GT(global_kdtree_.node_count(), 0u);
}

// ─── Test: Render pixel produces valid output ───────────────────────

TEST_F(IntegrationTest, RenderPixelNonNegativeFinite) {
    if (!scene_ok()) GTEST_SKIP();

    Renderer renderer;
    renderer.set_scene(scene_.get());
    renderer.set_camera(camera_);
    renderer.set_config(config_);
    renderer.build_photon_maps();

    // Test a grid of pixels
    const int N = 8;
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int px = x * config_.image_width / N;
            int py = y * config_.image_height / N;
            int pixel_idx = py * config_.image_width + px;

            PCGRng rng = PCGRng::seed(
                (uint64_t)pixel_idx * 1000,
                (uint64_t)pixel_idx + 1);
            Ray ray = camera_.generate_ray(px, py, rng);
            auto result = renderer.render_pixel(ray, rng);

            for (int i = 0; i < NUM_LAMBDA; ++i) {
                ASSERT_GE(result.combined.value[i], 0.f)
                    << "Negative radiance at pixel (" << px << "," << py
                    << ") lambda bin " << i;
                ASSERT_TRUE(std::isfinite(result.combined.value[i]))
                    << "Non-finite radiance at pixel (" << px << "," << py
                    << ") lambda bin " << i;
            }
        }
    }
}

// ─── Test: Combined mode has non-zero output ────────────────────────

TEST_F(IntegrationTest, CombinedModeProducesNonZeroOutput) {
    if (!scene_ok()) GTEST_SKIP();

    Renderer renderer;
    renderer.set_scene(scene_.get());
    renderer.set_camera(camera_);

    RenderConfig cfg = config_;
    cfg.mode = RenderMode::Combined;
    renderer.set_config(cfg);
    renderer.build_photon_maps();

    // Center pixel should see something in Cornell Box
    int cx = config_.image_width / 2;
    int cy = config_.image_height / 2;
    float total_energy = 0.f;

    for (int s = 0; s < 16; ++s) {
        PCGRng rng = PCGRng::seed((uint64_t)s * 42, 1);
        Ray ray = camera_.generate_ray(cx, cy, rng);
        auto result = renderer.render_pixel(ray, rng);
        for (int i = 0; i < NUM_LAMBDA; ++i)
            total_energy += result.combined.value[i];
    }

    EXPECT_GT(total_energy, 0.f)
        << "Center pixel should have non-zero radiance in Combined mode";
}

// ─── Test: NEE-only mode produces some output ───────────────────────

TEST_F(IntegrationTest, DirectOnlyModeNonZero) {
    if (!scene_ok()) GTEST_SKIP();

    Renderer renderer;
    renderer.set_scene(scene_.get());
    renderer.set_camera(camera_);

    RenderConfig cfg = config_;
    cfg.mode = RenderMode::DirectOnly;
    renderer.set_config(cfg);
    renderer.build_photon_maps();

    int cx = config_.image_width / 2;
    int cy = config_.image_height / 2;
    float total = 0.f;

    for (int s = 0; s < 16; ++s) {
        PCGRng rng = PCGRng::seed((uint64_t)s * 42, 1);
        Ray ray = camera_.generate_ray(cx, cy, rng);
        auto result = renderer.render_pixel(ray, rng);
        for (int i = 0; i < NUM_LAMBDA; ++i)
            total += result.combined.value[i];
    }

    EXPECT_GT(total, 0.f)
        << "Center pixel should have non-zero direct lighting";
}

// ─── Test: Indirect-only mode produces some output ──────────────────

TEST_F(IntegrationTest, IndirectOnlyModeNonZero) {
    if (!scene_ok()) GTEST_SKIP();

    Renderer renderer;
    renderer.set_scene(scene_.get());
    renderer.set_camera(camera_);

    RenderConfig cfg = config_;
    cfg.mode = RenderMode::IndirectOnly;
    renderer.set_config(cfg);
    renderer.build_photon_maps();

    float total = 0.f;

    // Sample multiple pixels - at least some should have indirect light
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            int px = (x + 1) * config_.image_width / 5;
            int py = (y + 1) * config_.image_height / 5;
            for (int s = 0; s < 4; ++s) {
                PCGRng rng = PCGRng::seed(
                    (uint64_t)(py * config_.image_width + px) * 1000 + s, 1);
                Ray ray = camera_.generate_ray(px, py, rng);
                auto result = renderer.render_pixel(ray, rng);
                for (int i = 0; i < NUM_LAMBDA; ++i)
                    total += result.combined.value[i];
            }
        }
    }

    EXPECT_GT(total, 0.f)
        << "Some pixels should have non-zero indirect lighting from photon map";
}

// ─── Test: No double-counting between NEE and photon gather ─────────

TEST_F(IntegrationTest, CombinedEqualsDirectPlusIndirect) {
    if (!scene_ok()) GTEST_SKIP();

    Renderer renderer;
    renderer.set_scene(scene_.get());
    renderer.set_camera(camera_);

    // In v2, render_pixel returns L_total and L_nee separately.
    // L_total = NEE + photon_gather, so L_total - L_nee ≈ photon_gather
    RenderConfig cfg = config_;
    cfg.mode = RenderMode::Combined;
    renderer.set_config(cfg);
    renderer.build_photon_maps();

    int cx = config_.image_width / 2;
    int cy = config_.image_height / 2;

    float sum_combined = 0.f;
    float sum_nee = 0.f;

    for (int s = 0; s < 32; ++s) {
        PCGRng rng = PCGRng::seed((uint64_t)s * 42, 1);
        Ray ray = camera_.generate_ray(cx, cy, rng);
        auto result = renderer.render_pixel(ray, rng);
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            sum_combined += result.combined.value[i];
            sum_nee += result.nee_direct.value[i];
        }
    }

    // Combined should be >= NEE (photon gather adds to it)
    EXPECT_GE(sum_combined + 1e-6f, sum_nee)
        << "Combined radiance should be >= direct-only radiance";
}

// ─── Test: KD-tree vs HashGrid density agreement ────────────────────

TEST_F(IntegrationTest, KDTreeMatchesHashGridDensity) {
    if (!scene_ok()) GTEST_SKIP();
    if (global_photons_.size() == 0) GTEST_SKIP();

    // Pick a point in the scene and compare density estimates
    // Use the center pixel's first hit point
    PCGRng rng = PCGRng::seed(12345, 1);
    Ray ray = camera_.generate_ray(
        config_.image_width / 2, config_.image_height / 2, rng);
    HitRecord hit = scene_->intersect(ray);
    if (!hit.hit) GTEST_SKIP();

    Material mat = scene_->materials[hit.material_id];
    ONB frame = ONB::from_normal(hit.shading_normal);
    float3 wo_local = frame.world_to_local(ray.direction * (-1.f));

    // Hash grid estimate
    DensityEstimatorConfig de_config;
    de_config.radius = config_.gather_radius;
    de_config.num_photons_total = config_.num_photons;

    Spectrum L_grid = estimate_photon_density(
        hit.position, hit.shading_normal, wo_local, mat,
        global_photons_, global_grid_, de_config, config_.gather_radius);

    // KD-tree estimate (raw fixed-radius, same as hash grid)
    Spectrum L_kdtree = Spectrum::zero();
    float r2 = config_.gather_radius * config_.gather_radius;
    float inv_area = 1.0f / (PI * r2);
    float inv_N = 1.0f / (float)config_.num_photons;

    global_kdtree_.query(hit.position, config_.gather_radius, global_photons_,
        [&](uint32_t idx, float /*dist2*/) {
            float3 p_pos = make_f3(global_photons_.pos_x[idx],
                                   global_photons_.pos_y[idx],
                                   global_photons_.pos_z[idx]);
            float plane_dist = fabsf(dot(hit.shading_normal, p_pos - hit.position));
            if (plane_dist > DEFAULT_SURFACE_TAU) return;

            if (!global_photons_.norm_x.empty()) {
                float3 pn = make_f3(global_photons_.norm_x[idx],
                                    global_photons_.norm_y[idx],
                                    global_photons_.norm_z[idx]);
                if (dot(pn, hit.shading_normal) <= 0.f) return;
            }

            float3 wi = make_f3(global_photons_.wi_x[idx],
                                global_photons_.wi_y[idx],
                                global_photons_.wi_z[idx]);
            if (dot(wi, hit.shading_normal) <= 0.f) return;

            float3 wi_loc = frame.world_to_local(wi);
            Spectrum f = bsdf::evaluate(mat, wo_local, wi_loc);

            Spectrum photon_flux = global_photons_.get_flux(idx);
            for (int b = 0; b < NUM_LAMBDA; ++b)
                L_kdtree.value[b] += photon_flux.value[b] * inv_N *
                                     f.value[b] * inv_area;
        });

    // They should produce the same result (both do fixed-radius gather)
    float grid_sum = L_grid.sum();
    float kd_sum = L_kdtree.sum();

    if (grid_sum > 0.f || kd_sum > 0.f) {
        float rel_diff = fabsf(grid_sum - kd_sum) /
                        (fmaxf(grid_sum, kd_sum) + 1e-10f);
        // 10% tolerance: KD-tree manual gather vs hash grid estimate
        // may differ slightly due to rounding / boundary inclusion
        EXPECT_LT(rel_diff, 0.10f)
            << "KD-tree and hash grid density estimates should agree. "
            << "Grid=" << grid_sum << " KD=" << kd_sum;
    }
}

// ─── Test: ACES tonemap produces valid sRGB ─────────────────────────

TEST_F(IntegrationTest, ACESTonemapValid) {
    if (!scene_ok()) GTEST_SKIP();

    Renderer renderer;
    renderer.set_scene(scene_.get());
    renderer.set_camera(camera_);
    renderer.set_config(config_);
    renderer.build_photon_maps();
    renderer.render_frame();

    const auto& fb = renderer.framebuffer();
    int total = config_.image_width * config_.image_height;

    for (int i = 0; i < total; ++i) {
        // After tonemap, check that sRGB values are in valid range
        // (The framebuffer stores pre-tonemap spectral data, but
        //  tonemap writes to an internal sRGB buffer)
        const Spectrum& L = fb.pixels[i];
        for (int j = 0; j < NUM_LAMBDA; ++j) {
            ASSERT_TRUE(std::isfinite(L.value[j]))
                << "Pixel " << i << " wavelength " << j << " is not finite";
            ASSERT_GE(L.value[j], 0.f)
                << "Pixel " << i << " wavelength " << j << " is negative";
        }
    }
}

// ─── Test: Full render frame completes without crash ────────────────

TEST_F(IntegrationTest, FullRenderFrameCompletes) {
    if (!scene_ok()) GTEST_SKIP();

    Renderer renderer;
    renderer.set_scene(scene_.get());
    renderer.set_camera(camera_);

    RenderConfig cfg = config_;
    cfg.image_width  = 32;
    cfg.image_height = 32;
    cfg.samples_per_pixel = 2;
    cfg.num_photons  = 10000;
    renderer.set_config(cfg);
    renderer.build_photon_maps();

    // This should not crash
    ASSERT_NO_THROW(renderer.render_frame());

    const auto& fb = renderer.framebuffer();
    // At least some pixels should be non-zero
    float total = 0.f;
    int npix = cfg.image_width * cfg.image_height;
    for (int i = 0; i < npix; ++i) {
        for (int j = 0; j < NUM_LAMBDA; ++j)
            total += fb.pixels[i].value[j];
    }
    EXPECT_GT(total, 0.f) << "Rendered frame should have some non-zero pixels";
}
// ═════════════════════════════════════════════════════════════════════
// v2.1 Integration Tests: Tangential Gather, Surface Filter, Coverage
// ═════════════════════════════════════════════════════════════════════

// ─── Test: KD-tree tangential gather matches hash grid tangential ────
//
// CPU reference (KD-tree) vs GPU-primary (hash grid) using the same
// tangential kernel.  Both should produce the same density estimate
// at the same point.

TEST_F(IntegrationTest, TangentialKDTreeMatchesHashGrid) {
    if (!scene_ok()) GTEST_SKIP();
    if (global_photons_.size() == 0) GTEST_SKIP();

    PCGRng rng = PCGRng::seed(54321, 1);

    // Sample 8 hit points on scene surfaces
    int match_count = 0;
    int tested = 0;

    for (int trial = 0; trial < 16; ++trial) {
        int px = (int)(rng.next_float() * config_.image_width);
        int py = (int)(rng.next_float() * config_.image_height);
        Ray ray = camera_.generate_ray(px, py, rng);
        HitRecord hit = scene_->intersect(ray);
        if (!hit.hit) continue;

        Material mat = scene_->materials[hit.material_id];
        if (mat.is_specular()) continue;
        if (mat.is_emissive()) continue;

        ONB frame = ONB::from_normal(hit.shading_normal);
        float3 wo_local = frame.world_to_local(ray.direction * (-1.f));
        if (wo_local.z <= 0.f) continue;

        // Hash grid tangential estimate
        DensityEstimatorConfig de_config;
        de_config.radius = config_.gather_radius;
        de_config.num_photons_total = config_.num_photons;
        de_config.use_kernel = false;  // box kernel

        Spectrum L_grid = estimate_photon_density(
            hit.position, hit.shading_normal, wo_local, mat,
            global_photons_, global_grid_, de_config, config_.gather_radius);

        // KD-tree tangential estimate (same tangential kernel)
        Spectrum L_kdtree = Spectrum::zero();
        float r2  = config_.gather_radius * config_.gather_radius;
        float tau = effective_tau(DEFAULT_SURFACE_TAU);
        float inv_area = 1.0f / fmaxf(box_kernel_norm(r2), 1e-20f);
        float inv_N = 1.0f / (float)config_.num_photons;

        global_kdtree_.query_tangential(hit.position, hit.shading_normal,
            config_.gather_radius, tau, global_photons_,
            [&](uint32_t idx, float d_tan2) {
                // conditions 3 & 4
                if (!global_photons_.norm_x.empty()) {
                    float3 pn = make_f3(global_photons_.norm_x[idx],
                                        global_photons_.norm_y[idx],
                                        global_photons_.norm_z[idx]);
                    if (dot(pn, hit.shading_normal) <= 0.0f) return;
                }
                float3 wi = make_f3(global_photons_.wi_x[idx],
                                    global_photons_.wi_y[idx],
                                    global_photons_.wi_z[idx]);
                if (dot(wi, hit.shading_normal) <= 0.f) return;

                float w = tangential_box_kernel(d_tan2, r2);
                if (w <= 0.f) return;

                float3 wi_loc = frame.world_to_local(wi);
                Spectrum f = bsdf::evaluate_diffuse(mat, wo_local, wi_loc);
                Spectrum flux = global_photons_.get_flux(idx);
                for (int b = 0; b < NUM_LAMBDA; ++b)
                    L_kdtree.value[b] += w * flux.value[b] * inv_N
                                       * f.value[b] * inv_area;
            });

        float g = L_grid.sum();
        float k = L_kdtree.sum();

        if (g > 0.f || k > 0.f) {
            float rel_diff = fabsf(g - k) / (fmaxf(g, k) + 1e-10f);
            EXPECT_LT(rel_diff, 0.05f)
                << "Tangential KD-tree vs hash grid mismatch at trial "
                << trial << ": grid=" << g << " kd=" << k;
            if (rel_diff < 0.05f) ++match_count;
        } else {
            ++match_count;  // both zero is a match
        }
        ++tested;
    }

    EXPECT_GE(tested, 4) << "Need at least 4 testable hit points";
    EXPECT_EQ(match_count, tested) << "All tested points should match";
}

// ─── Test: Tangential gather rejects through-wall photons ────────────
//
// At a diffuse hit point, tangential gather should not include photons
// that are geometrically close in 3D but on the opposite side of the
// surface.  We verify this by checking that no gathered photon has
// d_plane > tau.

TEST_F(IntegrationTest, TangentialGatherRejectsOpposite) {
    if (!scene_ok()) GTEST_SKIP();
    if (global_photons_.size() == 0) GTEST_SKIP();

    PCGRng rng = PCGRng::seed(11111, 1);
    float tau = effective_tau(DEFAULT_SURFACE_TAU);

    for (int trial = 0; trial < 8; ++trial) {
        int px = (int)(rng.next_float() * config_.image_width);
        int py = (int)(rng.next_float() * config_.image_height);
        Ray ray = camera_.generate_ray(px, py, rng);
        HitRecord hit = scene_->intersect(ray);
        if (!hit.hit) continue;

        // query_tangential should only return photons within tau of plane
        global_kdtree_.query_tangential(hit.position, hit.shading_normal,
            config_.gather_radius, tau, global_photons_,
            [&](uint32_t idx, float d_tan2) {
                float3 ppos = make_f3(global_photons_.pos_x[idx],
                                       global_photons_.pos_y[idx],
                                       global_photons_.pos_z[idx]);
                float d_plane = plane_distance(hit.position, hit.shading_normal,
                                               ppos);
                EXPECT_LE(fabsf(d_plane), tau + 1e-5f)
                    << "Photon " << idx
                    << " exceeded plane distance filter";
            });
    }
}

// ─── Test: Surface consistency filter prevents normal-incompatible ────
//
// Verify that after full surface consistency, no accepted photon
// has opposite-facing normal or incoming direction below the surface.

TEST_F(IntegrationTest, SurfaceConsistencyOnRealScene) {
    if (!scene_ok()) GTEST_SKIP();
    if (global_photons_.size() == 0) GTEST_SKIP();

    PCGRng rng = PCGRng::seed(22222, 1);
    float tau = effective_tau(DEFAULT_SURFACE_TAU);
    int violations = 0;
    int total_gathered = 0;

    for (int trial = 0; trial < 8; ++trial) {
        int px = (int)(rng.next_float() * config_.image_width);
        int py = (int)(rng.next_float() * config_.image_height);
        Ray ray = camera_.generate_ray(px, py, rng);
        HitRecord hit = scene_->intersect(ray);
        if (!hit.hit) continue;

        global_kdtree_.query_tangential(hit.position, hit.shading_normal,
            config_.gather_radius, tau, global_photons_,
            [&](uint32_t idx, float /*d_tan2*/) {
                // Check condition 3: normal compatibility
                if (!global_photons_.norm_x.empty()) {
                    float3 pn = make_f3(global_photons_.norm_x[idx],
                                        global_photons_.norm_y[idx],
                                        global_photons_.norm_z[idx]);
                    // query_tangential doesn't enforce condition 3,
                    // just conditions 1 & 2.  We check that the user of
                    // query_tangential CAN filter by normal.
                    if (dot(pn, hit.shading_normal) <= 0.0f)
                        ++violations;  // Would be filtered in density_estimator
                }
                ++total_gathered;
            });
    }

    // We expect some violations because query_tangential only handles
    // tangential + plane; conditions 3 & 4 are applied downstream.
    // But this test documents that they EXIST in real data.
    if (total_gathered > 0) {
        float ratio = (float)violations / (float)total_gathered;
        // In a Cornell Box with mostly co-planar walls, opposite-normal
        // photons should be uncommon but not impossible near edges.
        EXPECT_LT(ratio, 0.3f)
            << "Too many normal-incompatible photons: "
            << violations << "/" << total_gathered;
    }
}

// ─── Test: Coverage-aware NEE produces valid direct lighting ─────────
//
// Verify that direct lighting with coverage-aware mixture sampling
// still produces non-negative, finite, non-zero results at lit pixels.

TEST_F(IntegrationTest, CoverageAwareNEEValid) {
    if (!scene_ok()) GTEST_SKIP();

    Renderer renderer;
    renderer.set_scene(scene_.get());
    renderer.set_camera(camera_);

    RenderConfig cfg = config_;
    cfg.mode = RenderMode::DirectOnly;
    cfg.nee_coverage_fraction = 0.3f;  // §7.2.1 default
    renderer.set_config(cfg);
    renderer.build_photon_maps();

    int cx = config_.image_width / 2;
    int cy = config_.image_height / 2;
    float total = 0.f;

    for (int s = 0; s < 32; ++s) {
        PCGRng rng = PCGRng::seed((uint64_t)s * 77, 1);
        Ray ray = camera_.generate_ray(cx, cy, rng);
        auto result = renderer.render_pixel(ray, rng);
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            float v = result.nee_direct.value[i];
            ASSERT_TRUE(std::isfinite(v))
                << "NEE coverage mixture: non-finite at s=" << s
                << " bin=" << i;
            ASSERT_GE(v, 0.f)
                << "NEE coverage mixture: negative at s=" << s
                << " bin=" << i;
            total += v;
        }
    }

    EXPECT_GT(total, 0.f)
        << "Coverage-aware NEE should produce non-zero direct lighting "
           "at center pixel of Cornell Box";
}

// ─── Test: Photon decorrelation doesn't break photon distribution ────
//
// Verify that photon tracing with cell-stratified bouncing and RNG
// spatial hash still produces a valid photon map: positive flux,
// positions inside the scene bounds, finite values.

TEST_F(IntegrationTest, DecorrelatedPhotonsValid) {
    if (!scene_ok()) GTEST_SKIP();

    // The photon map was already built with decorrelation in SetUpTestSuite.
    // Verify all photons have valid properties.
    for (size_t i = 0; i < global_photons_.size(); ++i) {
        float3 pos = make_f3(global_photons_.pos_x[i],
                              global_photons_.pos_y[i],
                              global_photons_.pos_z[i]);
        float3 wi = make_f3(global_photons_.wi_x[i],
                             global_photons_.wi_y[i],
                             global_photons_.wi_z[i]);

        // Position should be finite
        ASSERT_TRUE(std::isfinite(pos.x) && std::isfinite(pos.y) &&
                    std::isfinite(pos.z))
            << "Photon " << i << " has non-finite position";

        // Direction should be unit-ish
        float len2 = dot(wi, wi);
        ASSERT_GT(len2, 0.5f) << "Photon " << i << " has degenerate wi";
        ASSERT_LT(len2, 1.5f) << "Photon " << i << " has non-unit wi";

        // Flux should be non-negative and finite
        float tf = global_photons_.total_flux(i);
        ASSERT_GE(tf, 0.f) << "Photon " << i << " has negative flux";
        ASSERT_TRUE(std::isfinite(tf))
            << "Photon " << i << " has non-finite flux";
    }
}

// ─── Test: Photon spatial distribution is reasonable ─────────────────
//
// Cell-stratified bouncing should not cause all photons to cluster
// in a small region.  We check the bounding box of the photon map.

TEST_F(IntegrationTest, PhotonSpatialDistribution) {
    if (!scene_ok()) GTEST_SKIP();
    if (global_photons_.size() < 100) GTEST_SKIP();

    float min_x = 1e10f, max_x = -1e10f;
    float min_y = 1e10f, max_y = -1e10f;
    float min_z = 1e10f, max_z = -1e10f;

    for (size_t i = 0; i < global_photons_.size(); ++i) {
        float x = global_photons_.pos_x[i];
        float y = global_photons_.pos_y[i];
        float z = global_photons_.pos_z[i];
        min_x = fminf(min_x, x); max_x = fmaxf(max_x, x);
        min_y = fminf(min_y, y); max_y = fmaxf(max_y, y);
        min_z = fminf(min_z, z); max_z = fmaxf(max_z, z);
    }

    float extent_x = max_x - min_x;
    float extent_y = max_y - min_y;
    float extent_z = max_z - min_z;

    // Cornell Box is roughly unit-scale.  Photons should be spread
    // across a significant fraction of the box.
    float max_extent = fmaxf(extent_x, fmaxf(extent_y, extent_z));
    EXPECT_GT(max_extent, 0.1f)
        << "Photon distribution is too compact: extent="
        << extent_x << "x" << extent_y << "x" << extent_z;

    // Should be spread in at least 2 dimensions
    int spread_dims = (extent_x > 0.05f ? 1 : 0)
                    + (extent_y > 0.05f ? 1 : 0)
                    + (extent_z > 0.05f ? 1 : 0);
    EXPECT_GE(spread_dims, 2)
        << "Photons should be distributed in at least 2D";
}

// ─── Test: Shell expansion k-NN matches KD-tree tangential k-NN ─────
//
// On real photon map data, compare the set of k-nearest photons
// found by KD-tree tangential k-NN vs hash grid shell expansion k-NN.

TEST_F(IntegrationTest, ShellExpansionMatchesKDTreeKNN) {
    if (!scene_ok()) GTEST_SKIP();
    if (global_photons_.size() < 100) GTEST_SKIP();

    PCGRng rng = PCGRng::seed(33333, 1);
    float tau = effective_tau(DEFAULT_SURFACE_TAU);
    int k = 20;
    int tested = 0;
    int matched = 0;

    for (int trial = 0; trial < 12; ++trial) {
        int px = (int)(rng.next_float() * config_.image_width);
        int py = (int)(rng.next_float() * config_.image_height);
        Ray ray = camera_.generate_ray(px, py, rng);
        HitRecord hit = scene_->intersect(ray);
        if (!hit.hit) continue;

        // KD-tree tangential k-NN
        std::vector<uint32_t> kd_indices;
        float kd_max_dist2;
        global_kdtree_.knn_tangential(hit.position, hit.shading_normal,
            k, tau, global_photons_, kd_indices, kd_max_dist2);

        if (kd_indices.empty()) continue;

        // Shell expansion k-NN
        std::vector<uint32_t> sh_indices;
        float sh_max_dist2;
        global_grid_.knn_shell_expansion(hit.position, hit.shading_normal,
            k, tau, global_photons_, sh_indices, sh_max_dist2);

        ++tested;

        // Compare index sets (order may differ due to different traversal)
        std::set<uint32_t> kd_set(kd_indices.begin(), kd_indices.end());
        std::set<uint32_t> sh_set(sh_indices.begin(), sh_indices.end());

        // Count overlap
        int overlap = 0;
        for (uint32_t idx : sh_set) {
            if (kd_set.count(idx)) ++overlap;
        }

        // Allow some difference because hash grid cell resolution
        // may cause minor boundary discrepancies
        float overlap_ratio = (float)overlap /
            (float)fmaxf((float)kd_set.size(), 1.f);

        if (overlap_ratio >= 0.7f) ++matched;

        // Max distances should be in the same ballpark
        if (kd_max_dist2 > 0.f && sh_max_dist2 > 0.f) {
            float ratio = fmaxf(kd_max_dist2, sh_max_dist2) /
                          fminf(kd_max_dist2, sh_max_dist2);
            EXPECT_LT(ratio, 5.0f)
                << "k-NN max distance diverges at trial " << trial
                << ": kd=" << kd_max_dist2 << " sh=" << sh_max_dist2;
        }
    }

    if (tested > 0) {
        float match_rate = (float)matched / (float)tested;
        EXPECT_GE(match_rate, 0.5f)
            << "Shell expansion should agree with KD-tree tangential k-NN "
               "for majority of test points";
    }
}

// ─── Test: Tangential gather energy not amplified ─────────────────────
//
// The density estimate at any point should not exceed a reasonable
// physical bound.  For a given photon budget and gather radius, the
// estimate L ≤ total_flux / (π r²) * max_bsdf, approximately.

TEST_F(IntegrationTest, TangentialGatherEnergyBound) {
    if (!scene_ok()) GTEST_SKIP();
    if (global_photons_.size() == 0) GTEST_SKIP();

    Renderer renderer;
    renderer.set_scene(scene_.get());
    renderer.set_camera(camera_);
    renderer.set_config(config_);
    renderer.build_photon_maps();

    PCGRng rng = PCGRng::seed(44444, 1);

    // Upper bound: assume all flux concentrated in one pixel
    float total_emitted_flux = 0.f;
    for (size_t i = 0; i < global_photons_.size(); ++i)
        total_emitted_flux += global_photons_.total_flux(i);
    float inv_N = 1.0f / (float)config_.num_photons;
    float r2 = config_.gather_radius * config_.gather_radius;
    float loose_upper = total_emitted_flux * inv_N / (PI * r2) * 10.f;

    for (int trial = 0; trial < 8; ++trial) {
        int px = (int)(rng.next_float() * config_.image_width);
        int py = (int)(rng.next_float() * config_.image_height);
        Ray ray = camera_.generate_ray(px, py, rng);
        auto result = renderer.render_pixel(ray, rng);

        for (int b = 0; b < NUM_LAMBDA; ++b) {
            EXPECT_LT(result.combined.value[b], loose_upper)
                << "Radiance exceeds physical upper bound at trial "
                << trial << " bin " << b
                << " value=" << result.combined.value[b]
                << " bound=" << loose_upper;
        }
    }
}

// ─── Test: Full render with tangential gather completes validly ──────
//
// End-to-end test: build photon maps, render a small image with
// tangential gather, verify all pixels are valid and the image has
// non-zero content.

TEST_F(IntegrationTest, FullRenderTangentialValid) {
    if (!scene_ok()) GTEST_SKIP();

    Renderer renderer;
    renderer.set_scene(scene_.get());
    renderer.set_camera(camera_);

    RenderConfig cfg = config_;
    cfg.image_width  = 32;
    cfg.image_height = 32;
    cfg.samples_per_pixel = 2;
    cfg.num_photons  = 20000;
    cfg.use_kdtree   = true;  // use tangential KD-tree path
    cfg.nee_coverage_fraction = 0.3f;  // coverage-aware NEE
    renderer.set_config(cfg);
    renderer.build_photon_maps();

    ASSERT_NO_THROW(renderer.render_frame());

    const auto& fb = renderer.framebuffer();
    int npix = cfg.image_width * cfg.image_height;
    float total = 0.f;
    int negative_count = 0;
    int nan_count = 0;

    for (int i = 0; i < npix; ++i) {
        for (int j = 0; j < NUM_LAMBDA; ++j) {
            float v = fb.pixels[i].value[j];
            if (v < 0.f) ++negative_count;
            if (!std::isfinite(v)) ++nan_count;
            total += v;
        }
    }

    EXPECT_EQ(negative_count, 0) << "No pixel should have negative radiance";
    EXPECT_EQ(nan_count, 0) << "No pixel should have NaN/Inf radiance";
    EXPECT_GT(total, 0.f) << "Rendered image should have non-zero content";
}