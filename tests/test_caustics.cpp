// test_caustics.cpp — Tests for targeted caustic photon emission
#include <gtest/gtest.h>
#include "scene/scene.h"
#include "scene/obj_loader.h"
#include "photon/specular_target.h"
#include "photon/emitter.h"
#include "core/config.h"
#include "core/material_flags.h"
#include "bsdf/bsdf.h"
#include <cstdio>
#include <string>

static std::string scene_path(const char* rel) {
    return std::string(SCENES_DIR) + "/" + rel;
}

// ── Test that SpecularTargetSet finds glass/translucent triangles ──────
TEST(Caustics, SpecularTargetSetFindsGlass) {
    Scene scene;
    ASSERT_TRUE(load_obj(scene_path("cornell_water/CornellBox-Water.obj"), scene));
    SpecularTargetSet ts = SpecularTargetSet::build(scene);
    std::printf("[Test] SpecularTargetSet: valid=%d  n_spec_tris=%zu\n",
                (int)ts.valid, ts.specular_tri_indices.size());
    // Dump material types of all specular triangles
    for (size_t i = 0; i < ts.specular_tri_indices.size() && i < 10; ++i) {
        uint32_t ti = ts.specular_tri_indices[i];
        uint32_t mi = scene.triangles[ti].material_id;
        const Material& mat = scene.materials[mi];
        std::printf("[Test]   tri#%u mat#%u type=%d ior=%.2f area=%.6f\n",
                    ti, mi, (int)mat.type, mat.ior, ts.tri_areas[i]);
    }
    EXPECT_TRUE(ts.valid) << "No specular/glass/translucent triangles found in cornell_water";
    EXPECT_GT(ts.specular_tri_indices.size(), 0u);
}

// ── Test that targeted caustic photon sampling produces valid photons ──
TEST(Caustics, SampleTargetedCausticPhotons) {
    Scene scene;
    ASSERT_TRUE(load_obj(scene_path("cornell_water/CornellBox-Water.obj"), scene));
    scene.build_bvh();
    scene.build_emissive_distribution();
    SpecularTargetSet ts = SpecularTargetSet::build(scene);
    ASSERT_TRUE(ts.valid);

    int total = 1000;
    int valid = 0;
    int rejected_backface = 0;
    int rejected_emitter_cos = 0;
    int rejected_visibility = 0;

    for (int i = 0; i < total; ++i) {
        PCGRng rng = PCGRng::seed((uint64_t)i * 13 + 42, (uint64_t)i + 7);
        TargetedCausticPhoton tcp = sample_targeted_caustic_photon(scene, ts, rng);
        if (tcp.valid) {
            ++valid;
            if (valid <= 3) {
                std::printf("[Test] Valid photon %d: origin=(%.3f,%.3f,%.3f) dir=(%.3f,%.3f,%.3f) flux_sum=%.4e\n",
                            valid, tcp.ray.origin.x, tcp.ray.origin.y, tcp.ray.origin.z,
                            tcp.ray.direction.x, tcp.ray.direction.y, tcp.ray.direction.z,
                            tcp.spectral_flux.sum());
            }
        }
    }

    float acceptance_rate = (float)valid / (float)total;
    std::printf("[Test] Acceptance: %d / %d = %.1f%%\n", valid, total, acceptance_rate * 100.f);
    EXPECT_GT(valid, 0) << "No valid targeted caustic photons at all!";
    EXPECT_GT(acceptance_rate, 0.01f) << "Acceptance rate below 1%";
}

// ── Test that trace_targeted_caustic_emission produces stored caustic photons ──
TEST(Caustics, TraceTargetedCausticEmission) {
    Scene scene;
    ASSERT_TRUE(load_obj(scene_path("cornell_water/CornellBox-Water.obj"), scene));
    scene.build_bvh();
    scene.build_emissive_distribution();
    SpecularTargetSet ts = SpecularTargetSet::build(scene);
    ASSERT_TRUE(ts.valid);

    EmitterConfig ecfg;
    ecfg.num_photons    = 10000;
    ecfg.max_bounces    = 16;
    ecfg.rr_threshold   = DEFAULT_RR_THRESHOLD;
    ecfg.min_bounces_rr = DEFAULT_MIN_BOUNCES_RR;
    ecfg.volume_enabled = false;

    PhotonSoA caustic_map;
    trace_targeted_caustic_emission(scene, ecfg, ts, caustic_map, 1.0f);
    std::printf("[Test] Targeted emission: budget=%d  stored=%zu\n",
                ecfg.num_photons, caustic_map.size());

    if (caustic_map.size() > 0) {
        // Sample a few photons
        for (size_t i = 0; i < (std::min)(caustic_map.size(), (size_t)3); ++i) {
            float px = caustic_map.pos_x[i];
            float py = caustic_map.pos_y[i];
            float pz = caustic_map.pos_z[i];
            float f0 = caustic_map.flux[i * HERO_WAVELENGTHS];
            std::printf("[Test]   photon[%zu]: pos=(%.3f,%.3f,%.3f) flux[0]=%.4e\n",
                        i, px, py, pz, (double)f0);
        }
    }

    EXPECT_GT(caustic_map.size(), 0u) << "No caustic photons stored after targeted emission!";
}

// ── Detailed step-by-step bounce trace for a single targeted photon ──
TEST(Caustics, SinglePhotonBounceTrace) {
    Scene scene;
    ASSERT_TRUE(load_obj(scene_path("cornell_water/CornellBox-Water.obj"), scene));
    scene.build_bvh();
    scene.build_emissive_distribution();
    SpecularTargetSet ts = SpecularTargetSet::build(scene);
    ASSERT_TRUE(ts.valid);

    // Try multiple seeds until we find a valid targeted photon
    TargetedCausticPhoton tcp;
    int seed_used = -1;
    for (int s = 0; s < 100; ++s) {
        PCGRng rng = PCGRng::seed((uint64_t)s * 13 + 42, (uint64_t)s + 7);
        tcp = sample_targeted_caustic_photon(scene, ts, rng);
        if (tcp.valid) { seed_used = s; break; }
    }
    ASSERT_TRUE(tcp.valid) << "Could not find a valid targeted photon in 100 tries";

    std::printf("[Trace] Seed=%d  origin=(%.4f,%.4f,%.4f)  dir=(%.4f,%.4f,%.4f)\n",
                seed_used, tcp.ray.origin.x, tcp.ray.origin.y, tcp.ray.origin.z,
                tcp.ray.direction.x, tcp.ray.direction.y, tcp.ray.direction.z);
    std::printf("[Trace] flux_sum=%.6e  source_emissive=%u\n",
                (double)tcp.spectral_flux.sum(), (unsigned)tcp.source_emissive_idx);

    // Now trace the photon bounce-by-bounce
    Ray ray = tcp.ray;
    Spectrum spectral_flux = tcp.spectral_flux;
    bool on_caustic_path = false;
    int max_bounces = 16;

    for (int bounce = 0; bounce < max_bounces; ++bounce) {
        HitRecord hit = scene.intersect(ray);
        if (!hit.hit) {
            std::printf("[Trace] bounce %d: MISS (no intersection)\n", bounce);
            break;
        }

        const Material& mat = scene.materials[hit.material_id];
        MaterialFlags mat_flags = classify_for_photons(mat);

        std::printf("[Trace] bounce %d: hit tri#%u mat#%u type=%d(%s)  is_delta=%d  caustic_caster=%d\n",
                    bounce, hit.triangle_id, hit.material_id, (int)mat.type,
                    mat.type == MaterialType::Lambertian ? "Lambertian" :
                    mat.type == MaterialType::Mirror ? "Mirror" :
                    mat.type == MaterialType::Glass ? "Glass" :
                    mat.type == MaterialType::Translucent ? "Translucent" :
                    mat.type == MaterialType::Emissive ? "Emissive" : "Other",
                    (int)mat_flags.is_delta, (int)mat_flags.caustic_caster);
        std::printf("[Trace]   hit_pos=(%.4f,%.4f,%.4f) hit_normal=(%.4f,%.4f,%.4f)\n",
                    hit.position.x, hit.position.y, hit.position.z,
                    hit.normal.x, hit.normal.y, hit.normal.z);
        std::printf("[Trace]   ray_dir=(%.4f,%.4f,%.4f)  dot(ray,normal)=%.6f\n",
                    ray.direction.x, ray.direction.y, ray.direction.z,
                    dot(ray.direction, hit.normal));

        if (!mat_flags.is_delta) {
            std::printf("[Trace]   -> DIFFUSE surface. on_caustic_path=%d  bounce=%d  -> %s\n",
                        (int)on_caustic_path, bounce,
                        (on_caustic_path && bounce > 0) ? "STORE" : "SKIP (not caustic)");
            break;
        } else {
            if (mat_flags.caustic_caster) on_caustic_path = true;

            // BSDF sample
            ONB frame = ONB::from_normal(hit.shading_normal);
            float3 wo_local = frame.world_to_local(ray.direction * (-1.f));

            std::printf("[Trace]   wo_local=(%.4f,%.4f,%.4f)  entering=%d\n",
                        wo_local.x, wo_local.y, wo_local.z, (int)(wo_local.z > 0.f));

            PCGRng bounce_rng = PCGRng::seed((uint64_t)(seed_used * 1000 + bounce), 42);
            BSDFSample bs = bsdf::sample(mat, wo_local, bounce_rng);

            std::printf("[Trace]   bsdf: wi=(%.4f,%.4f,%.4f)  pdf=%.6f  is_specular=%d  f_max=%.6e\n",
                        bs.wi.x, bs.wi.y, bs.wi.z, bs.pdf, (int)bs.is_specular,
                        (double)bs.f.max_component());

            if (bs.pdf <= 0.f) {
                std::printf("[Trace]   -> BSDF pdf=0, photon DIES\n");
                break;
            }

            // Update flux
            float cos_theta = fabsf(bs.wi.z);
            float geom_factor = cos_theta / bs.pdf;
            float flux_before = spectral_flux.sum();
            for (int b = 0; b < NUM_LAMBDA; ++b)
                spectral_flux.value[b] *= bs.f.value[b] * geom_factor;
            float flux_after = spectral_flux.sum();
            std::printf("[Trace]   flux: %.4e -> %.4e  (geom_factor=%.4f)\n",
                        (double)flux_before, (double)flux_after, geom_factor);

            // Next ray
            float3 wi_world = frame.local_to_world(bs.wi);
            ray.origin    = hit.position + hit.shading_normal * 1e-4f *
                            (bs.wi.z > 0.f ? 1.f : -1.f);
            ray.direction = wi_world;
            ray.tmin      = 1e-4f;
            ray.tmax      = 1e20f;

            std::printf("[Trace]   next_ray: origin=(%.4f,%.4f,%.4f) dir=(%.4f,%.4f,%.4f)\n",
                        ray.origin.x, ray.origin.y, ray.origin.z,
                        ray.direction.x, ray.direction.y, ray.direction.z);
        }
    }
}
