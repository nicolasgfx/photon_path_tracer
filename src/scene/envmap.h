#pragma once
// ─────────────────────────────────────────────────────────────────────
// envmap.h – HDR environment map loading, importance sampling, and
//            analytical lookup for infinite-light IBL
// ─────────────────────────────────────────────────────────────────────
// Supports:
//   - Loading .exr files via tinyexr
//   - Marginal-conditional CDF for importance sampling (luminance × sinθ)
//   - Rotation via 3×3 matrix (from Euler angles)
//   - CPU eval / sample / pdf for CPU path tracing
//   - Flat data arrays ready for GPU upload
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"
#include "core/config.h"

#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <numeric>

// ── 3×3 rotation matrix (row-major) ─────────────────────────────────
struct RotationMatrix3 {
    float3 row0, row1, row2;

    HD float3 apply(float3 v) const {
        return make_f3(
            dot(row0, v),
            dot(row1, v),
            dot(row2, v));
    }
};

inline RotationMatrix3 rotation_identity() {
    return { make_f3(1,0,0), make_f3(0,1,0), make_f3(0,0,1) };
}

inline RotationMatrix3 rotation_transpose(const RotationMatrix3& m) {
    return {
        make_f3(m.row0.x, m.row1.x, m.row2.x),
        make_f3(m.row0.y, m.row1.y, m.row2.y),
        make_f3(m.row0.z, m.row1.z, m.row2.z)
    };
}

// Build rotation from Euler angles (degrees) in XYZ convention
inline RotationMatrix3 rotation_from_euler_deg(float rx, float ry, float rz) {
    constexpr float DEG2RAD = PI / 180.0f;
    float cx = cosf(rx * DEG2RAD), sx = sinf(rx * DEG2RAD);
    float cy = cosf(ry * DEG2RAD), sy = sinf(ry * DEG2RAD);
    float cz = cosf(rz * DEG2RAD), sz = sinf(rz * DEG2RAD);

    // R = Rz * Ry * Rx
    RotationMatrix3 m;
    m.row0 = make_f3( cy*cz,  sx*sy*cz - cx*sz,  cx*sy*cz + sx*sz);
    m.row1 = make_f3( cy*sz,  sx*sy*sz + cx*cz,  cx*sy*sz - sx*cz);
    m.row2 = make_f3(-sy,     sx*cy,              cx*cy);
    return m;
}

// ── EnvironmentMap ──────────────────────────────────────────────────

struct EnvironmentMap {
    // Pixel data (RGB float, row-major, top-to-bottom)
    std::vector<float> pixels;   // [width * height * 3]
    int width  = 0;
    int height = 0;

    // Rotation
    RotationMatrix3 rotation;       // world → envmap local
    RotationMatrix3 inv_rotation;   // envmap local → world
    float scale = 1.0f;

    // Marginal-conditional CDF for importance sampling
    // conditional_cdf[y * width + x] = CDF of column x in row y
    // marginal_cdf[y] = CDF of selecting row y
    std::vector<float> conditional_cdf;  // [height * width]
    std::vector<float> marginal_cdf;     // [height]

    // Precomputed total power (for envmap selection probability)
    float total_power = 0.f;

    // Scene bounding sphere (for photon emission from infinity)
    float3 scene_center = make_f3(0, 0, 0);
    float  scene_radius = 1.0f;

    // ── Direction ↔ UV conversion ───────────────────────────────────
    // Uses latitude-longitude (equirectangular) mapping.
    // θ ∈ [0, π] from +Y down, φ ∈ [0, 2π] from +X through +Z
    // u = φ / (2π), v = θ / π

    static inline HD float2 direction_to_uv(float3 dir) {
        // θ = acos(clamp(dir.y, -1, 1))
        float theta = acosf(fminf(fmaxf(dir.y, -1.0f), 1.0f));
        float phi   = atan2f(dir.z, dir.x);
        if (phi < 0.f) phi += 2.0f * PI;
        float u = phi * (0.5f * INV_PI);
        float v = theta * INV_PI;
        return make_f2(u, v);
    }

    static inline HD float3 uv_to_direction(float2 uv) {
        float theta = uv.y * PI;
        float phi   = uv.x * 2.0f * PI;
        float sin_theta = sinf(theta);
        return make_f3(
            sin_theta * cosf(phi),
            cosf(theta),
            sin_theta * sinf(phi));
    }

    // ── Evaluate radiance for a world-space direction ───────────────
    float3 eval_rgb(float3 world_dir) const {
        float3 local_dir = rotation.apply(world_dir);
        float2 uv = direction_to_uv(local_dir);

        // Bilinear interpolation
        float fx = uv.x * width  - 0.5f;
        float fy = uv.y * height - 0.5f;
        int x0 = (int)floorf(fx);
        int y0 = (int)floorf(fy);
        float dx = fx - x0;
        float dy = fy - y0;

        // Wrap coordinates
        auto wrap_x = [&](int x) { return ((x % width) + width) % width; };
        auto wrap_y = [&](int y) { return std::max(0, std::min(y, height - 1)); };

        int x0w = wrap_x(x0), x1w = wrap_x(x0 + 1);
        int y0w = wrap_y(y0), y1w = wrap_y(y0 + 1);

        auto px = [&](int x, int y) -> float3 {
            int idx = (y * width + x) * 3;
            return make_f3(pixels[idx], pixels[idx+1], pixels[idx+2]);
        };

        float3 c00 = px(x0w, y0w);
        float3 c10 = px(x1w, y0w);
        float3 c01 = px(x0w, y1w);
        float3 c11 = px(x1w, y1w);

        float3 c0 = c00 * (1.f - dx) + c10 * dx;
        float3 c1 = c01 * (1.f - dx) + c11 * dx;
        float3 rgb = c0 * (1.f - dy) + c1 * dy;

        return rgb * scale;
    }

    Spectrum eval(float3 world_dir) const {
        float3 rgb = eval_rgb(world_dir);
        return rgb_to_spectrum_emission(rgb.x, rgb.y, rgb.z);
    }

    // ── Build marginal-conditional CDF ──────────────────────────────
    // Weight = luminance(pixel) × sin(θ) for solid-angle correction
    void build_distribution() {
        if (width == 0 || height == 0) return;

        conditional_cdf.resize(width * height);
        marginal_cdf.resize(height);

        std::vector<float> row_weights(height, 0.f);

        for (int y = 0; y < height; ++y) {
            float theta = PI * (y + 0.5f) / height;
            float sin_theta = sinf(theta);

            // Build conditional CDF for this row
            float row_sum = 0.f;
            for (int x = 0; x < width; ++x) {
                int idx = (y * width + x) * 3;
                float r = pixels[idx], g = pixels[idx+1], b = pixels[idx+2];
                float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
                float w = fmaxf(lum, 0.f) * sin_theta;
                row_sum += w;
                conditional_cdf[y * width + x] = row_sum;
            }

            // Normalize conditional CDF
            if (row_sum > 0.f) {
                float inv = 1.f / row_sum;
                for (int x = 0; x < width; ++x)
                    conditional_cdf[y * width + x] *= inv;
            } else {
                // Uniform fallback
                for (int x = 0; x < width; ++x)
                    conditional_cdf[y * width + x] = (x + 1.f) / width;
            }
            conditional_cdf[y * width + (width - 1)] = 1.0f;

            row_weights[y] = row_sum;
        }

        // Build marginal CDF
        float total = 0.f;
        for (int y = 0; y < height; ++y) {
            total += row_weights[y];
            marginal_cdf[y] = total;
        }
        if (total > 0.f) {
            float inv = 1.f / total;
            for (int y = 0; y < height; ++y)
                marginal_cdf[y] *= inv;
        } else {
            for (int y = 0; y < height; ++y)
                marginal_cdf[y] = (y + 1.f) / height;
        }
        marginal_cdf[height - 1] = 1.0f;

        // Total power ≈ ∑ lum * sin(θ) * (π/H) * (2π/W) * scale
        // This gives approximate radiant power integrated over the sphere.
        float pixel_solid_angle = (2.0f * PI / width) * (PI / height);
        total_power = total * pixel_solid_angle * scale;

        std::printf("[EnvMap] Distribution built: %dx%d  total_power=%.4f  scale=%.2f\n",
                    width, height, total_power, scale);
    }

    // ── Binary search helper ────────────────────────────────────────
    static int binary_search_cdf(const float* cdf, int n, float u) {
        int lo = 0, hi = n - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (cdf[mid] < u) lo = mid + 1;
            else              hi = mid;
        }
        return lo;
    }

    // ── Importance-sample a direction from the envmap ────────────────
    // Returns (world_dir, pdf_solid_angle, RGB_radiance)
    struct EnvSample {
        float3 direction;
        float  pdf;
        float3 rgb;
    };

    EnvSample sample(float u1, float u2) const {
        EnvSample s;
        s.pdf = 0.f;
        s.rgb = make_f3(0, 0, 0);
        s.direction = make_f3(0, 1, 0);

        if (width == 0 || height == 0) return s;

        // Sample row (marginal)
        int y = binary_search_cdf(marginal_cdf.data(), height, u1);
        y = std::min(y, height - 1);

        // Remap u1 within the row's CDF interval
        float cdf_y_lo = (y > 0) ? marginal_cdf[y - 1] : 0.f;
        float cdf_y_hi = marginal_cdf[y];
        float u1_remapped = (u1 - cdf_y_lo) / fmaxf(cdf_y_hi - cdf_y_lo, 1e-10f);
        u1_remapped = fminf(fmaxf(u1_remapped, 0.f), 1.f - 1e-7f);

        // Sample column (conditional) — use u2
        const float* row_cdf = conditional_cdf.data() + y * width;
        int x = binary_search_cdf(row_cdf, width, u2);
        x = std::min(x, width - 1);

        // Texel centre UV
        float u = (x + 0.5f) / width;
        float v = (y + 0.5f) / height;

        // Direction in envmap local space
        float3 local_dir = uv_to_direction(make_f2(u, v));

        // Rotate to world space
        s.direction = inv_rotation.apply(local_dir);

        // Compute PDF in solid angle measure
        // pdf_pixel = marginal_pdf(y) * conditional_pdf(x|y)
        float pdf_y = cdf_y_hi - cdf_y_lo;
        float cdf_x_lo = (x > 0) ? row_cdf[x - 1] : 0.f;
        float pdf_x = row_cdf[x] - cdf_x_lo;

        float theta = v * PI;
        float sin_theta = sinf(theta);
        if (sin_theta < 1e-10f) sin_theta = 1e-10f;

        // pdf_uv = pdf_y * height * pdf_x * width
        // pdf_solid_angle = pdf_uv / (2π² sin θ)
        float pdf_uv = pdf_y * height * pdf_x * width;
        s.pdf = pdf_uv / (2.0f * PI * PI * sin_theta);

        // Evaluate radiance at this pixel
        int idx = (y * width + x) * 3;
        s.rgb = make_f3(pixels[idx], pixels[idx+1], pixels[idx+2]) * scale;

        return s;
    }

    // ── PDF for a given world direction ──────────────────────────────
    float pdf(float3 world_dir) const {
        if (width == 0 || height == 0) return 0.f;

        float3 local_dir = rotation.apply(world_dir);
        float2 uv = direction_to_uv(local_dir);

        float u = uv.x, v = uv.y;
        int x = std::min((int)(u * width),  width  - 1);
        int y = std::min((int)(v * height), height - 1);

        // Marginal PDF
        float cdf_y_lo = (y > 0) ? marginal_cdf[y - 1] : 0.f;
        float pdf_y = marginal_cdf[y] - cdf_y_lo;

        // Conditional PDF
        const float* row_cdf = conditional_cdf.data() + y * width;
        float cdf_x_lo = (x > 0) ? row_cdf[x - 1] : 0.f;
        float pdf_x = row_cdf[x] - cdf_x_lo;

        float theta = v * PI;
        float sin_theta = sinf(theta);
        if (sin_theta < 1e-10f) return 0.f;

        float pdf_uv = pdf_y * height * pdf_x * width;
        return pdf_uv / (2.0f * PI * PI * sin_theta);
    }

    // ── Sample a photon origin on the bounding disk ─────────────────
    // Photon comes from infinity: creates a ray on a disk perpendicular
    // to the sampled direction, positioned at the scene bounding sphere.
    struct PhotonEmitSample {
        float3 origin;
        float3 direction;   // towards scene
        float  pdf;         // combined direction + area PDF
        float3 rgb;         // envmap radiance
    };

    PhotonEmitSample sample_photon(float u1, float u2, float u3, float u4) const {
        PhotonEmitSample pe;
        pe.pdf = 0.f;
        pe.rgb = make_f3(0, 0, 0);

        // Sample direction from envmap
        EnvSample es = sample(u1, u2);
        pe.direction = es.direction * (-1.f);  // Photon travels towards scene
        pe.rgb = es.rgb;

        if (es.pdf <= 0.f) return pe;

        // Create orthonormal frame around the incoming direction
        ONB frame = ONB::from_normal(es.direction);

        // Uniform disk sampling: p(r,θ) = 1/(π R²)
        float r = scene_radius * sqrtf(u3);
        float theta = 2.0f * PI * u4;
        float disk_x = r * cosf(theta);
        float disk_y = r * sinf(theta);

        // Origin on the disk, far away along the direction
        pe.origin = scene_center + es.direction * scene_radius
                  + frame.u * disk_x + frame.v * disk_y;

        // Combined PDF: pdf_dir(ω) / (π R²)
        float disk_area = PI * scene_radius * scene_radius;
        pe.pdf = es.pdf / disk_area;

        return pe;
    }
};

// ── Load environment map from .exr file ─────────────────────────────
// Uses tinyexr (already linked in the project).
// Note: This function must be called from a compilation unit that has
// TINYEXR_IMPLEMENTATION defined (e.g., viewer.cpp or a dedicated .cpp).
// Here we just declare the interface.

bool load_environment_map(const std::string& exr_path,
                          float scale,
                          float3 rotation_deg,
                          EnvironmentMap& envmap);

// ── Create a constant-colour environment map (no file needed) ───────
// Useful for PBRT scenes with `LightSource "infinite"` that specify a
// constant blackbody instead of an EXR texture.
inline bool create_constant_envmap(float r, float g, float b,
                                   float scale_factor,
                                   float3 rotation_deg,
                                   EnvironmentMap& envmap) {
    constexpr int W = 8, H = 4;  // tiny – constant colour, no detail needed
    envmap.width  = W;
    envmap.height = H;
    envmap.scale  = scale_factor;

    envmap.pixels.resize(W * H * 3);
    for (int i = 0; i < W * H; ++i) {
        envmap.pixels[i * 3 + 0] = fmaxf(r, 0.f);
        envmap.pixels[i * 3 + 1] = fmaxf(g, 0.f);
        envmap.pixels[i * 3 + 2] = fmaxf(b, 0.f);
    }

    envmap.rotation     = rotation_from_euler_deg(rotation_deg.x,
                                                   rotation_deg.y,
                                                   rotation_deg.z);
    envmap.inv_rotation = rotation_transpose(envmap.rotation);

    envmap.build_distribution();

    std::printf("[EnvMap] Created constant envmap: rgb=(%.3f,%.3f,%.3f) "
                "scale=%.2f  total_power=%.4f\n",
                r, g, b, scale_factor, envmap.total_power);
    return true;
}
