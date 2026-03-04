#pragma once
// ─────────────────────────────────────────────────────────────────────
// optix_envmap.cuh – GPU device-side environment map utilities
// ─────────────────────────────────────────────────────────────────────
// Functions:
//   dev_envmap_dir_to_uv    – World direction → lat-long UV (with rotation)
//   dev_envmap_uv_to_dir    – UV → world direction (with inverse rotation)
//   dev_envmap_eval_rgb     – Evaluate envmap RGB radiance for a direction
//   dev_envmap_eval         – Evaluate envmap spectral radiance
//   dev_envmap_sample       – Importance-sample direction from envmap CDF
//   dev_envmap_pdf          – Solid-angle PDF for a given direction
//   dev_nee_envmap_sample   – NEE: sample envmap + shadow + BSDF MIS
// ─────────────────────────────────────────────────────────────────────

// ── Rotation matrix apply (device) ──────────────────────────────────
__forceinline__ __device__
float3 dev_envmap_rotate(float3 v) {
    return make_f3(
        dot(params.envmap_rot_row0, v),
        dot(params.envmap_rot_row1, v),
        dot(params.envmap_rot_row2, v));
}

__forceinline__ __device__
float3 dev_envmap_inv_rotate(float3 v) {
    return make_f3(
        dot(params.envmap_inv_rot_row0, v),
        dot(params.envmap_inv_rot_row1, v),
        dot(params.envmap_inv_rot_row2, v));
}

// ── Direction ↔ UV conversion ───────────────────────────────────────
__forceinline__ __device__
float2 dev_envmap_dir_to_uv(float3 world_dir) {
    float3 local_dir = dev_envmap_rotate(world_dir);
    float theta = acosf(fminf(fmaxf(local_dir.y, -1.0f), 1.0f));
    float phi   = atan2f(local_dir.z, local_dir.x);
    if (phi < 0.f) phi += 2.0f * PI;
    return make_f2(phi * (0.5f * INV_PI), theta * INV_PI);
}

__forceinline__ __device__
float3 dev_envmap_uv_to_dir(float2 uv) {
    float theta = uv.y * PI;
    float phi   = uv.x * 2.0f * PI;
    float sin_theta = sinf(theta);
    float3 local_dir = make_f3(
        sin_theta * cosf(phi),
        cosf(theta),
        sin_theta * sinf(phi));
    return dev_envmap_inv_rotate(local_dir);
}

// ── Bilinear interpolation of envmap RGB ────────────────────────────
__forceinline__ __device__
float3 dev_envmap_eval_rgb(float3 world_dir) {
    float2 uv = dev_envmap_dir_to_uv(world_dir);
    int w = params.envmap_width;
    int h = params.envmap_height;

    float fx = uv.x * w - 0.5f;
    float fy = uv.y * h - 0.5f;
    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    float dx = fx - x0;
    float dy = fy - y0;

    // Wrap x, clamp y
    int x0w = ((x0 % w) + w) % w;
    int x1w = ((x0 + 1) % w + w) % w;
    int y0w = max(0, min(y0, h - 1));
    int y1w = max(0, min(y0 + 1, h - 1));

    auto px = [&](int x, int y) -> float3 {
        int idx = (y * w + x) * 3;
        return make_f3(params.envmap_pixels[idx],
                       params.envmap_pixels[idx + 1],
                       params.envmap_pixels[idx + 2]);
    };

    float3 c00 = px(x0w, y0w);
    float3 c10 = px(x1w, y0w);
    float3 c01 = px(x0w, y1w);
    float3 c11 = px(x1w, y1w);

    float3 c0 = c00 * (1.f - dx) + c10 * dx;
    float3 c1 = c01 * (1.f - dx) + c11 * dx;
    float3 rgb = c0 * (1.f - dy) + c1 * dy;

    return rgb * params.envmap_scale;
}

// ── Device-side RGB → Spectrum (emission) ───────────────────────────
// Same matrix as cpu-side rgb_to_spectrum_emission() in spectrum.h
__forceinline__ __device__
Spectrum dev_rgb_to_spectrum_emission(float r, float g, float b) {
    Spectrum s;
    s.value[0] = fmaxf(0.f, 0.01381753f * r + 0.07280016f * g + 0.78052238f * b);
    s.value[1] = fmaxf(0.f, 0.06839557f * r + 0.82078527f * g + 0.09598912f * b);
    s.value[2] = fmaxf(0.f, 0.69628317f * r + 0.39308480f * g - 0.03413205f * b);
    s.value[3] = fmaxf(0.f, 0.00013491f * r + 0.00030314f * g + 0.00001524f * b);
    return s;
}

// ── Evaluate envmap spectral radiance ───────────────────────────────
__forceinline__ __device__
Spectrum dev_envmap_eval(float3 world_dir) {
    float3 rgb = dev_envmap_eval_rgb(world_dir);
    return dev_rgb_to_spectrum_emission(rgb.x, rgb.y, rgb.z);
}

// ── Binary search in device CDF array ───────────────────────────────
__forceinline__ __device__
int dev_envmap_bsearch(const float* cdf, int n, float u) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cdf[mid] < u) lo = mid + 1;
        else              hi = mid;
    }
    return lo;
}

// ── Importance-sample a direction from the envmap ───────────────────
// Returns: world-space direction, solid-angle PDF, spectral Le
struct DevEnvSample {
    float3   direction;
    float    pdf;
    Spectrum Le;
};

__forceinline__ __device__
DevEnvSample dev_envmap_sample(PCGRng& rng) {
    DevEnvSample s;
    s.pdf = 0.f;
    s.Le = Spectrum::zero();
    s.direction = make_f3(0, 1, 0);

    int w = params.envmap_width;
    int h = params.envmap_height;
    if (w == 0 || h == 0) return s;

    float u1 = rng.next_float();
    float u2 = rng.next_float();

    // Sample row (marginal CDF)
    int y = dev_envmap_bsearch(params.envmap_marginal_cdf, h, u1);
    y = min(y, h - 1);

    float cdf_y_lo = (y > 0) ? params.envmap_marginal_cdf[y - 1] : 0.f;
    float cdf_y_hi = params.envmap_marginal_cdf[y];

    // Sample column (conditional CDF within row)
    const float* row_cdf = params.envmap_conditional_cdf + y * w;
    int x = dev_envmap_bsearch(row_cdf, w, u2);
    x = min(x, w - 1);

    // Direction
    float u_coord = (x + 0.5f) / w;
    float v_coord = (y + 0.5f) / h;
    s.direction = dev_envmap_uv_to_dir(make_f2(u_coord, v_coord));

    // PDF
    float pdf_y = cdf_y_hi - cdf_y_lo;
    float cdf_x_lo = (x > 0) ? row_cdf[x - 1] : 0.f;
    float pdf_x = row_cdf[x] - cdf_x_lo;

    float theta = v_coord * PI;
    float sin_theta = sinf(theta);
    if (sin_theta < 1e-10f) sin_theta = 1e-10f;

    float pdf_uv = pdf_y * h * pdf_x * w;
    s.pdf = pdf_uv / (2.0f * PI * PI * sin_theta);

    // Radiance at this pixel
    int idx = (y * w + x) * 3;
    float3 rgb = make_f3(params.envmap_pixels[idx],
                          params.envmap_pixels[idx + 1],
                          params.envmap_pixels[idx + 2]) * params.envmap_scale;
    s.Le = dev_rgb_to_spectrum_emission(rgb.x, rgb.y, rgb.z);

    return s;
}

// ── PDF for a given world direction ─────────────────────────────────
__forceinline__ __device__
float dev_envmap_pdf(float3 world_dir) {
    int w = params.envmap_width;
    int h = params.envmap_height;
    if (w == 0 || h == 0) return 0.f;

    float2 uv = dev_envmap_dir_to_uv(world_dir);
    int x = min((int)(uv.x * w), w - 1);
    int y = min((int)(uv.y * h), h - 1);

    float cdf_y_lo = (y > 0) ? params.envmap_marginal_cdf[y - 1] : 0.f;
    float pdf_y = params.envmap_marginal_cdf[y] - cdf_y_lo;

    const float* row_cdf = params.envmap_conditional_cdf + y * w;
    float cdf_x_lo = (x > 0) ? row_cdf[x - 1] : 0.f;
    float pdf_x = row_cdf[x] - cdf_x_lo;

    float theta = uv.y * PI;
    float sin_theta = sinf(theta);
    if (sin_theta < 1e-10f) return 0.f;

    float pdf_uv = pdf_y * h * pdf_x * w;
    return pdf_uv / (2.0f * PI * PI * sin_theta);
}

// NOTE: dev_nee_envmap_sample() is in optix_nee.cuh (after NeeSampleResult def)
