#pragma once
// ─────────────────────────────────────────────────────────────────────
// mis.h – Multiple Importance Sampling (Section 8)
// ─────────────────────────────────────────────────────────────────────
// 3-way MIS between:
//   1. Light sampling (direct illumination)
//   2. BSDF sampling
//   3. Photon-guided sampling
//
// Uses power heuristic with exponent 2.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"
#include "core/random.h"

// ── MIS weight computation ──────────────────────────────────────────

// 2-way power heuristic: w_a = pa^2 / (pa^2 + pb^2)
inline HD float mis_weight_2(float pdf_a, float pdf_b) {
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    return a2 / fmaxf(a2 + b2, 1e-30f);
}

// 3-way power heuristic: w_a = pa^2 / (pa^2 + pb^2 + pc^2)
inline HD float mis_weight_3(float pdf_a, float pdf_b, float pdf_c) {
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    float c2 = pdf_c * pdf_c;
    return a2 / fmaxf(a2 + b2 + c2, 1e-30f);
}

// ── MIS-combined estimator ──────────────────────────────────────────
// For each sample from strategy k:
//   contribution += w_k * f(x) / p_k(x)
//
// With power heuristic:
//   w_k(x) = p_k(x)^2 / sum_j p_j(x)^2

struct MISContribution {
    Spectrum radiance;
    float    weight;
};

// Combine light sample with MIS
inline MISContribution mis_light_sample(
    Spectrum Li,                // Le from light
    Spectrum f_bsdf,            // BSDF value for the light direction
    float    cos_theta,         // cos of angle at receiver
    float    pdf_light,         // PDF from light sampling
    float    pdf_bsdf,          // PDF of BSDF for same direction
    float    pdf_photon = 0.f)  // PDF of photon-guided for same direction
{
    MISContribution c;
    float w = mis_weight_3(pdf_light, pdf_bsdf, pdf_photon);
    c.weight = w;

    // Contribution = w * Li * f * cos / pdf_light
    if (pdf_light > 0.f) {
        c.radiance = Li * f_bsdf * (cos_theta * w / pdf_light);
    } else {
        c.radiance = Spectrum::zero();
    }
    return c;
}

// Combine BSDF sample with MIS
inline MISContribution mis_bsdf_sample(
    Spectrum Le,                // Emission at hit point (if light was hit)
    Spectrum f_bsdf,            // BSDF value
    float    cos_theta,
    float    pdf_bsdf,          // PDF from BSDF sampling
    float    pdf_light,         // PDF of light sampling for same direction
    float    pdf_photon = 0.f)
{
    MISContribution c;
    float w = mis_weight_3(pdf_bsdf, pdf_light, pdf_photon);
    c.weight = w;

    if (pdf_bsdf > 0.f) {
        c.radiance = Le * f_bsdf * (cos_theta * w / pdf_bsdf);
    } else {
        c.radiance = Spectrum::zero();
    }
    return c;
}

// Combine photon-guided sample with MIS
inline MISContribution mis_photon_sample(
    Spectrum Li,
    Spectrum f_bsdf,
    float    cos_theta,
    float    pdf_photon,        // PDF from photon-guided sampling
    float    pdf_light,
    float    pdf_bsdf)
{
    MISContribution c;
    float w = mis_weight_3(pdf_photon, pdf_light, pdf_bsdf);
    c.weight = w;

    if (pdf_photon > 0.f) {
        c.radiance = Li * f_bsdf * (cos_theta * w / pdf_photon);
    } else {
        c.radiance = Spectrum::zero();
    }
    return c;
}
