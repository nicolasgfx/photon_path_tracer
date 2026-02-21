#pragma once
// ─────────────────────────────────────────────────────────────────────
// random.h – PCG-based random number generator (host + device)
// ─────────────────────────────────────────────────────────────────────
#include "types.h"

struct PCGRng {
    uint64_t state;
    uint64_t inc;

    HD static PCGRng seed(uint64_t initstate, uint64_t initseq = 1u) {
        PCGRng rng;
        rng.state = 0u;
        rng.inc   = (initseq << 1u) | 1u;
        rng.next_uint();
        rng.state += initstate;
        rng.next_uint();
        return rng;
    }

    HD uint32_t next_uint() {
        uint64_t oldstate = state;
        state = oldstate * 6364136223846793005ULL + inc;
        uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = (uint32_t)(oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
    }

    // Uniform [0, 1)
    HD float next_float() {
        return (float)next_uint() / 4294967296.0f;
    }

    // Uniform [0, 1)² 
    HD float2 next_float2() {
        float u = next_float();
        float v = next_float();
        return make_f2(u, v);
    }

    // Spatial decorrelation: mix an arbitrary key into the RNG state
    // to break coherent patterns between neighbouring generators.
    HD void advance(uint32_t delta) {
        state += (uint64_t)delta * 6364136223846793005ULL;
        next_uint();  // consume one step to fully mix
    }
};

// ── Sampling utilities ──────────────────────────────────────────────

// Cosine-weighted hemisphere sampling (Malley's method)
inline HD float3 sample_cosine_hemisphere(float u1, float u2) {
    float r   = sqrtf(u1);
    float phi = TWO_PI * u2;
    float x   = r * cosf(phi);
    float y   = r * sinf(phi);
    float z   = sqrtf(fmaxf(0.f, 1.f - u1));
    return make_f3(x, y, z); // z-up local space
}

inline HD float cosine_hemisphere_pdf(float cos_theta) {
    return fmaxf(0.f, cos_theta) * INV_PI;
}

// Uniform hemisphere sampling
inline HD float3 sample_uniform_hemisphere(float u1, float u2) {
    float z   = u1;
    float r   = sqrtf(fmaxf(0.f, 1.f - z * z));
    float phi = TWO_PI * u2;
    return make_f3(r * cosf(phi), r * sinf(phi), z);
}

inline HD float uniform_hemisphere_pdf() {
    return INV_2PI;
}

// Uniform sphere sampling
inline HD float3 sample_uniform_sphere(float u1, float u2) {
    float z   = 1.f - 2.f * u1;
    float r   = sqrtf(fmaxf(0.f, 1.f - z * z));
    float phi = TWO_PI * u2;
    return make_f3(r * cosf(phi), r * sinf(phi), z);
}

// Uniform triangle sampling (standard barycentric)
inline HD float3 sample_triangle(float u, float v) {
    float su = sqrtf(u);
    float alpha = 1.f - su;
    float beta  = v * su;
    float gamma = 1.f - alpha - beta;
    return make_f3(alpha, beta, gamma); // barycentric coords
}

// Concentric disk sampling (Shirley & Chiu)
// Maps uniform [0,1)² → unit disk with low distortion.
inline HD float2 sample_concentric_disk(float u1, float u2) {
    float a = 2.f * u1 - 1.f;
    float b = 2.f * u2 - 1.f;
    if (a == 0.f && b == 0.f) return make_f2(0.f, 0.f);
    float r, phi;
    if (a * a > b * b) {
        r   = a;
        phi = (PI / 4.f) * (b / a);
    } else {
        r   = b;
        phi = (PI / 2.f) - (PI / 4.f) * (a / b);
    }
    return make_f2(r * cosf(phi), r * sinf(phi));
}

// Power heuristic for MIS (beta = 2)
inline HD float power_heuristic(float pdf_a, float pdf_b) {
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    return a2 / fmaxf(a2 + b2, 1e-30f);
}

// 3-way power heuristic
inline HD float power_heuristic_3(float pdf_a, float pdf_b, float pdf_c) {
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    float c2 = pdf_c * pdf_c;
    return a2 / fmaxf(a2 + b2 + c2, 1e-30f);
}
