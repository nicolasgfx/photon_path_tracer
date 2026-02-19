#pragma once
// ─────────────────────────────────────────────────────────────────────
// triangle.h – Triangle mesh data structures
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"

struct Triangle {
    float3   v0, v1, v2;           // Vertex positions
    float3   n0, n1, n2;           // Vertex normals (for shading)
    float2   uv0, uv1, uv2;       // Texture coords
    uint32_t material_id;

    HD float3 geometric_normal() const {
        return normalize(cross(v1 - v0, v2 - v0));
    }

    HD float area() const {
        return 0.5f * length(cross(v1 - v0, v2 - v0));
    }

    // Interpolate position from barycentric coordinates
    HD float3 interpolate_position(float alpha, float beta, float gamma) const {
        return v0 * alpha + v1 * beta + v2 * gamma;
    }

    // Interpolate shading normal from barycentric coordinates
    HD float3 interpolate_normal(float alpha, float beta, float gamma) const {
        return normalize(n0 * alpha + n1 * beta + n2 * gamma);
    }

    // Interpolate UVs from barycentric coordinates
    HD float2 interpolate_uv(float alpha, float beta, float gamma) const {
        return make_f2(
            uv0.x * alpha + uv1.x * beta + uv2.x * gamma,
            uv0.y * alpha + uv1.y * beta + uv2.y * gamma
        );
    }

    // Möller–Trumbore ray-triangle intersection
    HD bool intersect(const Ray& ray, float& t_out, float& u_out, float& v_out) const {
        float3 e1 = v1 - v0;
        float3 e2 = v2 - v0;
        float3 h  = cross(ray.direction, e2);
        float  a  = dot(e1, h);

        if (fabsf(a) < EPSILON) return false;

        float  f = 1.f / a;
        float3 s = ray.origin - v0;
        float  u = f * dot(s, h);
        if (u < 0.f || u > 1.f) return false;

        float3 q = cross(s, e1);
        float  v = f * dot(ray.direction, q);
        if (v < 0.f || u + v > 1.f) return false;

        float t = f * dot(e2, q);
        if (t < ray.tmin || t > ray.tmax) return false;

        t_out = t;
        u_out = u;
        v_out = v;
        return true;
    }
};

// ── AABB ────────────────────────────────────────────────────────────
struct AABB {
    float3 mn = { 1e30f,  1e30f,  1e30f};
    float3 mx = {-1e30f, -1e30f, -1e30f};

    HD void expand(float3 p) {
        mn = fminf3(mn, p);
        mx = fmaxf3(mx, p);
    }

    HD void expand(const AABB& b) {
        mn = fminf3(mn, b.mn);
        mx = fmaxf3(mx, b.mx);
    }

    HD float3 center() const {
        return (mn + mx) * 0.5f;
    }

    HD float3 extent() const {
        return mx - mn;
    }

    HD int longest_axis() const {
        float3 e = extent();
        if (e.x > e.y && e.x > e.z) return 0;
        if (e.y > e.z) return 1;
        return 2;
    }

    HD bool intersect(const Ray& ray, float& tmin_out, float& tmax_out) const {
        float3 invd = make_f3(1.f/ray.direction.x, 1.f/ray.direction.y, 1.f/ray.direction.z);
        float3 t0 = (mn - ray.origin) * invd;
        float3 t1 = (mx - ray.origin) * invd;

        float3 tsmall = fminf3(t0, t1);
        float3 tbig   = fmaxf3(t0, t1);

        float tmin = fmaxf(fmaxf(tsmall.x, tsmall.y), fmaxf(tsmall.z, ray.tmin));
        float tmax = fminf(fminf(tbig.x, tbig.y), fminf(tbig.z, ray.tmax));

        tmin_out = tmin;
        tmax_out = tmax;
        return tmin <= tmax;
    }
};
