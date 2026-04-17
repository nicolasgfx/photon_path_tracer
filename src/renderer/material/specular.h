#pragma once
// ─────────────────────────────────────────────────────────────────────
// specular.h – Specular bounce helper for glass / mirror (v5 RGB)
//
// Performs a full specular bounce with:
//   - Fresnel reflection / refraction decision
//   - IOR stack update (nested dielectrics)
//   - Medium stack update (participating media)
//   - Epsilon offset along geometric normal
//   - Adjoint η² correction for importance transport
//   - Thin dielectric shortcut
//
// Ported from v4 optix_specular.cuh: Spectrum→Color3, no per-channel
// dispersion (RGB uses scalar IOR + Color3 Tf filter).
//
// This is a CPU/GPU-shared helper.  GPU renderers pass material
// properties directly via the SpecularBounceParams struct, avoiding
// dependency on device-side global `params`.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/color.h"
#include "core/random.h"
#include "core/ior_stack.h"
#include "scene/medium.h"
#include "material/bsdf_shared.h"

constexpr float SPECULAR_EPSILON = 1e-4f;

// ── Parameters for a specular bounce ────────────────────────────────
struct SpecularBounceParams {
    float   ior;         // material IOR
    Color3  Tf;          // transmittance filter
    bool    is_glass;    // true = glass/translucent (refract), false = mirror
    bool    is_thin;     // thin dielectric shortcut
    int     medium_id;   // interior medium (-1 = none)
};

// ── Result of a specular bounce ─────────────────────────────────────
struct SpecularBounceResult {
    float3 new_dir;
    float3 new_pos;
    Color3 filter;  // throughput multiplier
};

// ── Specular bounce (CPU/GPU shared) ─────────────────────────────────

inline HD SpecularBounceResult specular_bounce(
    float3 dir, float3 pos, float3 normal, float3 geo_normal,
    const SpecularBounceParams& bp, PCGRng& rng,
    IORStack* ior_stack = nullptr,
    TransportMode mode = TransportMode::Radiance,
    MediumStack* medium_stack = nullptr)
{
    SpecularBounceResult r;
    r.filter = Color3::one();

    if (bp.is_glass) {
        // ── Thin dielectric shortcut ─────────────────────────────
        if (bp.is_thin) {
            bool entering = dot(dir, geo_normal) < 0.f;
            float3 outward_n   = entering ? normal : normal * (-1.f);
            float3 outward_geo = entering ? geo_normal : geo_normal * (-1.f);
            float cos_i = fabsf(dot(dir, outward_n));
            float eta = entering ? (1.0f / bp.ior) : bp.ior;
            float F = fresnel_dielectric(cos_i, eta);

            if (rng.next_float() < F) {
                r.new_dir = dir - outward_n * (2.f * dot(dir, outward_n));
                r.new_pos = pos + outward_geo * SPECULAR_EPSILON;
            } else {
                r.new_dir = dir;
                r.new_pos = pos - outward_geo * SPECULAR_EPSILON;
                r.filter = bp.Tf;
            }
            return r;
        }

        // Use geometric normal for inside/outside test
        bool entering = dot(dir, geo_normal) < 0.f;
        float3 outward_n   = entering ? normal : normal * (-1.f);
        float3 outward_geo = entering ? geo_normal : geo_normal * (-1.f);
        float cos_i = fabsf(dot(dir, outward_n));

        // Outside medium IOR from stack
        float outside_ior;
        if (entering) {
            outside_ior = ior_stack ? ior_stack->top() : 1.0f;
        } else {
            outside_ior = (ior_stack && ior_stack->depth >= 2)
                        ? ior_stack->iors[ior_stack->depth - 2]
                        : 1.0f;
        }

        float eta = entering ? (outside_ior / bp.ior) : (bp.ior / outside_ior);
        float F = fresnel_dielectric(cos_i, eta);

        bool do_reflect = (rng.next_float() < F);

        if (do_reflect) {
            r.new_dir = dir - outward_n * (2.f * dot(dir, outward_n));
            r.new_pos = pos + outward_geo * SPECULAR_EPSILON;
            // No Tf on reflection, filter stays 1.0
        } else {
            // Update IOR stack
            if (ior_stack) {
                if (entering)
                    ior_stack->push(bp.ior);
                else
                    ior_stack->pop();
            }

            // Update medium stack
            if (medium_stack) {
                if (bp.medium_id >= 0) {
                    if (entering)
                        medium_stack->push(bp.medium_id);
                    else if (medium_stack->depth > 0)
                        medium_stack->pop();
                } else {
                    // Glass with no interior medium (e.g. air pocket)
                    if (entering)
                        medium_stack->push(-1);
                    else if (medium_stack->depth > 0)
                        medium_stack->pop();
                }
            }

            // Compute refracted direction
            float sin2 = eta * eta * (1.f - cos_i * cos_i);
            float3 refracted = dir * eta +
                outward_n * (eta * cos_i - sqrtf(fmaxf(0.f, 1.f - sin2)));
            r.new_dir = normalize(refracted);
            r.new_pos = pos - outward_geo * SPECULAR_EPSILON;
            r.filter = bp.Tf;

            // Adjoint η² correction for importance transport
            if (mode == TransportMode::Importance) {
                r.filter = r.filter * (eta * eta);
            }
        }
    } else {
        // Mirror: pure specular reflection
        r.new_dir = dir - normal * (2.f * dot(dir, normal));
        r.new_pos = pos + geo_normal * SPECULAR_EPSILON;
    }
    return r;
}
