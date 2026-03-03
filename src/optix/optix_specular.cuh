#pragma once

// optix_specular.cuh – Cauchy dispersion, IOR stack,
//                       SpecularBounceResult, dev_specular_bounce
//
// fresnel_dielectric() now lives in bsdf/bsdf_shared.h (shared HD).

// =====================================================================
// Cauchy dispersion helpers (§10.1)
// n(λ) = A + B / λ²   (λ in nm).
// Falls back to constant IOR when dispersion is disabled.
// =====================================================================
__forceinline__ __device__
float dev_ior_at_lambda(uint32_t mat_id, float lambda_nm) {
    if (params.mat_dispersion && params.mat_dispersion[mat_id])
        return params.cauchy_A[mat_id] + params.cauchy_B[mat_id] / (lambda_nm * lambda_nm);
    return dev_get_ior(mat_id);
}

__forceinline__ __device__
bool dev_has_dispersion(uint32_t mat_id) {
    return params.mat_dispersion && params.mat_dispersion[mat_id];
}

// =====================================================================
// dev_specular_bounce — Shared helper for glass (Fresnel dielectric) and
// mirror reflection.  Returns the new ray direction, offset position, and
// a spectral throughput filter.
//
// The filter already accounts for the stochastic Fresnel choice: the
// hero wavelength decides reflect vs refract (probability F_hero for
// reflection, 1-F_hero for refraction), and the filter re-weights every
// wavelength bin by the ratio of that bin's Fresnel to the hero's.
//
// This ensures identical energy accounting to the CPU BSDF
// (bsdf.h glass_sample), where:
//   reflect: throughput *= (F_b / F_hero)                 — no Tf
//   refract: throughput *= Tf[b] * (1 - F_b) / (1 - F_hero) — with Tf
//   (bins with TIR get zero weight)
//
// Chromatic dispersion (Cauchy equation, §10.1):
// When dispersion is enabled, the hero wavelength (hero_bins[0] if
// provided, else bin 0) determines the refraction direction.  All other
// wavelength bins share that direction but receive per-wavelength Fresnel
// weights in `filter`.
//
// Without dispersion: all bins share the same IOR/Fresnel, the ratio
// is 1 everywhere: reflect → filter = 1.0, refract → filter = Tf.
// This recovers the original non-dispersive behaviour exactly.
// =====================================================================

// ── IOR stack for nested dielectric tracking ─────────────────────────
// (Moved to core/ior_stack.h — shared HD implementation)
// =====================================================================

struct SpecularBounceResult {
    float3   new_dir;
    float3   new_pos;
    Spectrum filter;   // throughput multiplier for the chosen path
};

__forceinline__ __device__
SpecularBounceResult dev_specular_bounce(
    float3 dir, float3 pos, float3 normal, float3 geo_normal,
    uint32_t mat_id, float2 uv, PCGRng& rng,
    const int* hero_bins = nullptr, int num_hero = 0,
    IORStack* ior_stack = nullptr,
    TransportMode mode = TransportMode::Radiance,
    MediumStack* medium_stack = nullptr)
{
    SpecularBounceResult r;
    r.filter = Spectrum::constant(1.0f);

    if (dev_is_glass(mat_id) || dev_is_translucent(mat_id)) {
        // ── Thin dielectric shortcut ─────────────────────────────────
        // Thin glass: no refraction direction change, no IOR stack.
        // Stochastic reflect/transmit; transmitted rays pass straight
        // through with Fresnel attenuation and transmission filter.
        if (dev_is_thin(mat_id)) {
            bool entering = dot(dir, geo_normal) < 0.f;
            float3 outward_n   = entering ? normal : normal * (-1.f);
            float3 outward_geo = entering ? geo_normal : geo_normal * (-1.f);
            float cos_i = fabsf(dot(dir, outward_n));
            float mat_ior = dev_get_ior(mat_id);
            float eta = entering ? (1.0f / mat_ior) : mat_ior;
            float F = fresnel_dielectric(cos_i, eta);

            if (rng.next_float() < F) {
                // Reflect
                r.new_dir = dir - outward_n * (2.f * dot(dir, outward_n));
                r.new_pos = pos + outward_geo * OPTIX_SCENE_EPSILON;
            } else {
                // Transmit straight through (no bending)
                r.new_dir = dir;
                r.new_pos = pos - outward_geo * OPTIX_SCENE_EPSILON;
                Spectrum Tf = dev_get_Tf(mat_id, uv);
                r.filter = Tf;
            }
            return r;
        }

        // Use geometric normal for inside/outside test (immune to
        // shading-normal interpolation artefacts on curved meshes).
        bool entering = dot(dir, geo_normal) < 0.f;

        // Flip shading normal to match the entering decision from geo_normal.
        float3 outward_n = entering ? normal : normal * (-1.f);
        // Geometric outward normal for epsilon offset (push past geometry).
        float3 outward_geo = entering ? geo_normal : geo_normal * (-1.f);

        float cos_i = fabsf(dot(dir, outward_n));

        // Outside medium IOR from the IOR stack (air = 1.0 when empty).
        float outside_ior = ior_stack ? ior_stack->top() : 1.0f;

        // Hero wavelength determines direction when dispersion is enabled.
        // When called from the camera path (no hero_bins), use the D-line
        // bin (~589 nm) so the refraction direction uses the nominal IOR
        // instead of the extreme UV bin 0 (380 nm, IOR much too high).
        constexpr int DLINE_BIN = (int)((589.0f - LAMBDA_MIN) / LAMBDA_STEP);
        int hero_bin = (hero_bins && num_hero > 0) ? hero_bins[0] : DLINE_BIN;
        float hero_ior = dev_ior_at_lambda(mat_id, lambda_of_bin(hero_bin));
        float eta_hero = entering ? (outside_ior / hero_ior) : (hero_ior / outside_ior);

        // Fresnel reflectance at hero wavelength
        float F_hero = fresnel_dielectric(cos_i, eta_hero);

        Spectrum Tf = dev_get_Tf(mat_id, uv);
        bool do_reflect = (rng.next_float() < F_hero);

        if (do_reflect) {
            // Reflection: no IOR stack change.
            r.new_dir = dir - outward_n * (2.f * dot(dir, outward_n));
            r.new_pos = pos + outward_geo * OPTIX_SCENE_EPSILON;

            if (dev_has_dispersion(mat_id)) {
                // Reflection filter: F_b / F_hero  (Tf NOT applied)
                float inv_F_hero = 1.f / fmaxf(F_hero, 1e-8f);
                for (int b = 0; b < NUM_LAMBDA; ++b) {
                    float n_b   = dev_ior_at_lambda(mat_id, lambda_of_bin(b));
                    float eta_b = entering ? (outside_ior / n_b) : (n_b / outside_ior);
                    float F_b   = fresnel_dielectric(cos_i, eta_b);
                    r.filter.value[b] = F_b * inv_F_hero;
                }
            }
            // else: non-dispersive → F_b == F_hero for all bins,
            //       ratio = 1.0, filter stays constant(1.0).  Correct.
        } else {
            // Refract — direction from hero wavelength IOR.
            // Update IOR stack: push on enter, pop on exit.
            if (ior_stack) {
                if (entering)
                    ior_stack->push(hero_ior);
                else
                    ior_stack->pop();
            }
            // Update medium stack for Translucent surfaces (§7.10).
            if (medium_stack && dev_is_translucent(mat_id)) {
                int mid = dev_get_medium_id(mat_id);
                if (mid >= 0) {
                    if (entering)
                        medium_stack->push(mid);
                    else if (medium_stack->depth > 0)
                        medium_stack->pop();
                }
            }

            float sin2_hero = eta_hero * eta_hero * (1.f - cos_i * cos_i);
            float3 refracted = dir * eta_hero +
                outward_n * (eta_hero * cos_i - sqrtf(fmaxf(0.f, 1.f - sin2_hero)));
            r.new_dir = normalize(refracted);
            r.new_pos = pos - outward_geo * OPTIX_SCENE_EPSILON;

            if (dev_has_dispersion(mat_id)) {
                // Refraction filter: Tf[b] * (1 - F_b) / (1 - F_hero)
                // Bins with TIR (F_b == 1) get zero weight.
                float inv_one_minus_F = 1.f / fmaxf(1.f - F_hero, 1e-8f);
                for (int b = 0; b < NUM_LAMBDA; ++b) {
                    float n_b   = dev_ior_at_lambda(mat_id, lambda_of_bin(b));
                    float eta_b = entering ? (outside_ior / n_b) : (n_b / outside_ior);
                    float F_b   = fresnel_dielectric(cos_i, eta_b);
                    if (F_b >= 1.f) {
                        r.filter.value[b] = 0.f;  // TIR at this wavelength
                    } else {
                        r.filter.value[b] = Tf.value[b] * (1.f - F_b) * inv_one_minus_F;
                        // §2.2 η² adjoint correction for importance transport
                        if (mode == TransportMode::Importance)
                            r.filter.value[b] *= eta_b * eta_b;
                    }
                }
            } else {
                // Non-dispersive: (1-F)/(1-F) = 1 for all bins → filter = Tf
                r.filter = Tf;
                // §2.2 η² adjoint correction for importance transport
                if (mode == TransportMode::Importance) {
                    float eta_sq = eta_hero * eta_hero;
                    for (int b = 0; b < NUM_LAMBDA; ++b)
                        r.filter.value[b] *= eta_sq;
                }
            }
        }
    } else {
        // Mirror: pure reflection (use shading normal for direction,
        // geometric normal for epsilon offset).
        r.new_dir = dir - normal * (2.f * dot(dir, normal));
        r.new_pos = pos + geo_normal * OPTIX_SCENE_EPSILON;
    }
    return r;
}
