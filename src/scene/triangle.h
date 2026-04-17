#pragma once
// ─────────────────────────────────────────────────────────────────────
// triangle.h – Triangle mesh data structures (v5)
//
// Exact port from v4: no spectral fields, purely geometric.
// AABB is defined in core/types.h; this file is geometry-only.
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

    HD float3 interpolate_position(float alpha, float beta, float gamma) const {
        return v0 * alpha + v1 * beta + v2 * gamma;
    }

    HD float3 interpolate_normal(float alpha, float beta, float gamma) const {
        return normalize(n0 * alpha + n1 * beta + n2 * gamma);
    }

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
