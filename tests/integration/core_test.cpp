// ─────────────────────────────────────────────────────────────────────
// core_test.cpp – Compile-time verification that all core headers work
//
// This file includes every core header and exercises key types to
// ensure the v5 RGB foundation compiles and links correctly.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/color.h"
#include "core/random.h"
#include "core/config.h"
#include "core/hash.h"
#include "core/ior_stack.h"
#include "core/alias_table.h"
#include "core/material_flags.h"
#include "core/scene_profile.h"
#include "core/stage_metrics.h"

#include <cassert>
#include <cstdio>

int main() {
    // ── types.h ─────────────────────────────────────────────────────
    float3 a = make_f3(1.f, 2.f, 3.f);
    float3 b = make_f3(4.f, 5.f, 6.f);
    float  d = dot(a, b);
    assert(d == 32.f);  // 1*4+2*5+3*6

    float3 c = cross(a, b);
    assert(dot(c, a) < 1e-6f);

    float3 n = normalize(make_f3(0, 0, 5.f));
    assert(fabsf(length(n) - 1.f) < 1e-6f);

    ONB onb = ONB::from_normal(make_f3(0, 1, 0));
    float3 local = onb.world_to_local(make_f3(0, 1, 0));
    assert(fabsf(local.z - 1.f) < 1e-5f);

    Ray ray;
    ray.origin    = make_f3(0, 0, 0);
    ray.direction = make_f3(0, 0, -1);

    HitRecord hit{};
    hit.hit = false;

    AABB box = AABB::empty();
    box.expand(make_f3(-1, -1, -1));
    box.expand(make_f3(1, 1, 1));
    assert(fabsf(box.diagonal() - sqrtf(12.f)) < 1e-4f);

    // ── color.h ─────────────────────────────────────────────────────
    Color3 white = Color3::one();
    Color3 half  = white * 0.5f;
    assert(fabsf(half.luminance() - 0.5f) < 1e-5f);
    assert(half.is_finite());

    Color3 red = Color3::from_rgb(1.f, 0.f, 0.f);
    assert(fabsf(red.luminance() - 0.2126f) < 1e-4f);

    Color3 added = red + half;
    assert(fabsf(added.r - 1.5f) < 1e-5f);

    Color3 clamped = Color3::from_rgb(-1.f, 2.f, 0.5f).clamped_non_negative();
    assert(clamped.r == 0.f && clamped.g == 2.f);

    Color3 toned = tonemap_aces_srgb(Color3::constant(1.f));
    assert(toned.r > 0.f && toned.r <= 1.f);

    // ── random.h ────────────────────────────────────────────────────
    PCGRng rng = PCGRng::seed(42, 1);
    float u = rng.next_float();
    assert(u >= 0.f && u < 1.f);

    float3 dir = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
    assert(dir.z >= 0.f);

    float pdf = cosine_hemisphere_pdf(dir.z);
    assert(pdf >= 0.f);

    float mis = power_heuristic(1.f, 2.f);
    assert(mis > 0.f && mis < 1.f);

    // ── config.h ────────────────────────────────────────────────────
    static_assert(DEFAULT_SPP > 0, "SPP must be positive");
    static_assert(DEFAULT_KNN_K > 0, "kNN K must be positive");
    static_assert(DEFAULT_GATHER_RADIUS > 0.f, "gather radius must be positive");

    // ── hash.h ──────────────────────────────────────────────────────
    uint32_t h1 = teschner_hash(make_i3(1, 2, 3), 1024);
    uint32_t h2 = teschner_hash(make_i3(4, 5, 6), 1024);
    assert(h1 != h2);  // statistically improbable to collide

    uint32_t ph = hash_pixel(0, 0);
    (void)ph;

    // ── ior_stack.h ─────────────────────────────────────────────────
    IORStack stack;
    assert(stack.top() == 1.0f);
    stack.push(1.5f);
    assert(fabsf(stack.top() - 1.5f) < 1e-6f);
    stack.pop();
    assert(stack.top() == 1.0f);

    // ── alias_table.h ───────────────────────────────────────────────
    AliasTable at = AliasTable::build({1.f, 2.f, 3.f});
    assert(at.n == 3);
    assert(fabsf(at.total_weight - 6.f) < 1e-5f);
    int sampled = at.sample(0.5f, 0.5f);
    assert(sampled >= 0 && sampled < 3);

    // ── material_flags.h ────────────────────────────────────────────
    MaterialFlags glass_f = classify_for_photons_by_type(2);  // Glass
    assert(glass_f.is_delta && glass_f.caustic_caster && !glass_f.is_emissive);

    MaterialFlags diff_f = classify_for_photons_by_type(0);  // Lambertian
    assert(!diff_f.is_delta && !diff_f.caustic_caster);

    // ── scene_profile.h ─────────────────────────────────────────────
    SceneProfile profile;
    profile.dominant_lighting = LightingType::LargeArea;
    profile.num_triangles     = 50000;
    assert(profile.recommended_max_bounces == 8);

    // ── stage_metrics.h ─────────────────────────────────────────────
    StageMetrics metrics;
    metrics.stage_name = "test-stage";
    metrics.time_ms    = 42.f;
    metrics.set("photons_deposited", 1e6f);
    assert(fabsf(metrics.get("photons_deposited") - 1e6f) < 1.f);

    FrameMetrics frame;
    frame.add(metrics);
    assert(frame.stages.size() == 1);
    assert(frame.find("test-stage") != nullptr);

    printf("core_test PASSED: all %d core headers compile and link correctly.\n", 10);
    return 0;
}
