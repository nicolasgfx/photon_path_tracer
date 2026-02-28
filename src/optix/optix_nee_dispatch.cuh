#pragma once

// optix_nee_dispatch.cuh – NEE shadow variants: coverage-aware selection,
//                           golden-ratio stratified NEE, dispatch

// =====================================================================
// NEE DIRECT LIGHTING (with shadow ray)
// M light samples per hitpoint, averaged.  Coverage-aware CDF selection.
// =====================================================================

// Helper: coverage-aware emitter selection + PDF
__forceinline__ __device__
int dev_nee_select_global(PCGRng& rng, float& p_tri_out) {
    const float c = DEFAULT_NEE_COVERAGE_FRACTION;
    int local_idx;
    if (c > 0.f && rng.next_float() < c) {
        local_idx = (int)(rng.next_float() * (float)params.num_emissive);
        if (local_idx >= params.num_emissive) local_idx = params.num_emissive - 1;
    } else {
        float xi = rng.next_float();
        local_idx = binary_search_cdf(
            params.emissive_cdf, params.num_emissive, xi);
        if (local_idx >= params.num_emissive) local_idx = params.num_emissive - 1;
    }
    float p_power = (local_idx == 0)
        ? params.emissive_cdf[0]
        : params.emissive_cdf[local_idx] - params.emissive_cdf[local_idx - 1];
    float p_uniform = 1.0f / (float)params.num_emissive;
    p_tri_out = (1.0f - c) * p_power + c * p_uniform;
    return local_idx;
}

// Helper: global coverage-aware PDF for a given emissive index
__forceinline__ __device__
float dev_nee_global_pdf(int local_idx) {
    const float c = DEFAULT_NEE_COVERAGE_FRACTION;
    float p_power = (local_idx == 0)
        ? params.emissive_cdf[0]
        : params.emissive_cdf[local_idx] - params.emissive_cdf[local_idx - 1];
    float p_uniform = 1.0f / (float)params.num_emissive;
    return (1.0f - c) * p_power + c * p_uniform;
}

__forceinline__ __device__
NeeResult dev_nee_direct(float3 pos, float3 normal, float3 wo_local,
                         uint32_t mat_id, PCGRng& rng, int bounce,
                         float2 uv)
{
    NeeResult result;
    result.L = Spectrum::zero();
    result.visibility = 0.f;
    if (params.num_emissive <= 0) return result;

    const int M = nee_shadow_sample_count(
        bounce, params.nee_light_samples, params.nee_deep_samples);
    int visible_count = 0;
    ONB frame = ONB::from_normal(normal);

    for (int s = 0; s < M; ++s) {
        float p_tri;
        int local_idx = dev_nee_select_global(rng, p_tri);

        NeeSampleResult sr = dev_nee_evaluate_sample(
            local_idx, p_tri, pos, normal, wo_local, mat_id, frame, uv, rng);
        if (sr.visible) visible_count++;
        result.L += sr.L;
    }

    if (M > 1) {
        float inv_M = 1.f / (float)M;
        for (int i = 0; i < NUM_LAMBDA; ++i)
            result.L.value[i] *= inv_M;
    }
    result.visibility = (float)visible_count / (float)M;
    return result;
}

// =====================================================================
// Golden-Ratio Stratified CDF NEE (§7.2.2)
//
// Low-discrepancy shadow ray placement using the golden ratio
// φ = (√5−1)/2 ≈ 0.618.  A random base offset (from RNG) provides
// spatial decorrelation, while successive samples s = 0..M−1 are
// spaced by φ across the power-weighted emissive CDF, guaranteeing
// each shadow ray targets a different region of the distribution.
// Cost: O(M log N_e) per shading point — no precomputation needed.
// =====================================================================

static constexpr float GOLDEN_RATIO_CONJ = 0.6180339887498949f; // (√5−1)/2

__forceinline__ __device__
NeeResult dev_nee_golden_stratified(float3 pos, float3 normal, float3 wo_local,
                                    uint32_t mat_id, PCGRng& rng, int bounce,
                                    float2 uv = make_float2(0.f, 0.f))
{
    NeeResult result;
    result.L = Spectrum::zero();
    result.visibility = 0.f;
    if (params.num_emissive <= 0) return result;

    const int M = nee_shadow_sample_count(
        bounce, params.nee_light_samples, params.nee_deep_samples);
    int visible_count = 0;
    ONB frame = ONB::from_normal(normal);

    // Random base offset — Cranley-Patterson rotation for decorrelation
    float base = rng.next_float();

    for (int s = 0; s < M; ++s) {
        // Golden-ratio stratified CDF sample
        float xi = base + s * GOLDEN_RATIO_CONJ;
        xi = xi - floorf(xi);  // fract — wrap to [0,1)

        int local_idx = binary_search_cdf(
            params.emissive_cdf, params.num_emissive, xi);
        if (local_idx >= params.num_emissive) local_idx = params.num_emissive - 1;

        // PDF for this emitter under the power-weighted CDF
        float p_tri = (local_idx == 0)
            ? params.emissive_cdf[0]
            : params.emissive_cdf[local_idx] - params.emissive_cdf[local_idx - 1];

        NeeSampleResult sr = dev_nee_evaluate_sample(
            local_idx, p_tri, pos, normal, wo_local, mat_id, frame, uv, rng);
        if (sr.visible) visible_count++;
        result.L += sr.L;
    }

    if (M > 1) {
        float inv_M = 1.f / (float)M;
        for (int i = 0; i < NUM_LAMBDA; ++i)
            result.L.value[i] *= inv_M;
    }
    result.visibility = (float)visible_count / (float)M;
    return result;
}

// =====================================================================
// dev_nee_dispatch -- route to golden-ratio stratified NEE
// =====================================================================
__forceinline__ __device__
NeeResult dev_nee_dispatch(float3 pos, float3 normal, float3 wo_local,
                           uint32_t mat_id, PCGRng& rng, int bounce,
                           float2 uv)
{
    return dev_nee_golden_stratified(pos, normal, wo_local, mat_id, rng, bounce, uv);
}
