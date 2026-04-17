#pragma once
// ─────────────────────────────────────────────────────────────────────
// types.h – Fundamental math types with host/device annotations
//
// RGB pipeline (v5). Spectral types removed; all transport uses Color3.
// ─────────────────────────────────────────────────────────────────────
#include <cmath>
#include <cstdint>
#include <algorithm>

#ifdef __CUDACC__
#define HD __host__ __device__
#define DEV __device__
#else
#define HD
#define DEV
// Include CUDA vector types for host CXX compilation so that float3,
// float2, int3, etc. are the exact same types used by the CUDA runtime.
#include <vector_types.h>
#endif

// ── Constructors ────────────────────────────────────────────────────
inline HD float3 make_f3(float x, float y, float z) { return {x, y, z}; }
inline HD float2 make_f2(float x, float y) { return {x, y}; }
inline HD int3   make_i3(int x, int y, int z) { return {x, y, z}; }

// ── Arithmetic ──────────────────────────────────────────────────────
inline HD float3 operator+(float3 a, float3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline HD float3 operator-(float3 a, float3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline HD float3 operator*(float3 a, float3 b) { return {a.x*b.x, a.y*b.y, a.z*b.z}; }
inline HD float3 operator*(float3 a, float s)  { return {a.x*s, a.y*s, a.z*s}; }
inline HD float3 operator*(float s, float3 a)  { return {a.x*s, a.y*s, a.z*s}; }
inline HD float3 operator/(float3 a, float s)  { return {a.x/s, a.y/s, a.z/s}; }
inline HD float3 operator-(float3 a)           { return {-a.x, -a.y, -a.z}; }

inline HD float3& operator+=(float3& a, float3 b) { a.x+=b.x; a.y+=b.y; a.z+=b.z; return a; }
inline HD float3& operator-=(float3& a, float3 b) { a.x-=b.x; a.y-=b.y; a.z-=b.z; return a; }
inline HD float3& operator*=(float3& a, float s)  { a.x*=s;   a.y*=s;   a.z*=s;   return a; }
inline HD float3& operator/=(float3& a, float s)  { a.x/=s;   a.y/=s;   a.z/=s;   return a; }

// ── Vector operations ───────────────────────────────────────────────
inline HD float  dot(float3 a, float3 b)   { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline HD float3 cross(float3 a, float3 b) {
    return {a.y*b.z - a.z*b.y,
            a.z*b.x - a.x*b.z,
            a.x*b.y - a.y*b.x};
}
inline HD float  length(float3 v)     { return sqrtf(dot(v, v)); }
inline HD float  length_sq(float3 v)  { return dot(v, v); }
inline HD float3 normalize(float3 v) {
    float d = dot(v, v);
    if (d <= 0.f) return make_f3(0, 0, 0);
#ifdef __CUDA_ARCH__
    return v * rsqrtf(d);
#else
    return v * (1.f / sqrtf(d));
#endif
}

inline HD float3 fminf3(float3 a, float3 b) {
    return {fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)};
}
inline HD float3 fmaxf3(float3 a, float3 b) {
    return {fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)};
}

// ── Scalar reductions on float3 ─────────────────────────────────────
inline HD float luminance(float3 v) {
    return 0.2126f * v.x + 0.7152f * v.y + 0.0722f * v.z;
}
inline HD float max_component(float3 v) {
    return fmaxf(v.x, fmaxf(v.y, v.z));
}
inline HD bool is_finite_f3(float3 v) {
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

// Reflect direction d about normal n (d points toward surface).
inline HD float3 reflect(float3 d, float3 n) {
    return d - 2.f * dot(d, n) * n;
}

// ── Coordinate frame from normal ────────────────────────────────────
struct ONB {
    float3 u, v, w; // w = normal

    HD static ONB from_normal(float3 n) {
        ONB onb;
        onb.w = n;
        float3 a = (fabsf(n.x) > 0.9f) ? make_f3(0, 1, 0) : make_f3(1, 0, 0);
        onb.v = normalize(cross(n, a));
        onb.u = cross(onb.w, onb.v);
        return onb;
    }

    HD float3 local_to_world(float3 d) const {
        return u * d.x + v * d.y + w * d.z;
    }

    HD float3 world_to_local(float3 d) const {
        return make_f3(dot(d, u), dot(d, v), dot(d, w));
    }
};

// Build ONB from a normal and an explicit tangent (Gram-Schmidt).
// Falls back to from_normal() when the tangent is degenerate.
inline HD ONB onb_from_normal_and_tangent(float3 n, float3 t) {
    ONB onb;
    onb.w = n;
    // Gram-Schmidt: remove component of t along n
    float3 t_proj = t - dot(t, n) * n;
    float len2 = dot(t_proj, t_proj);
    if (len2 < 1e-8f) return ONB::from_normal(n);
    onb.u = t_proj * (1.f / sqrtf(len2));
    onb.v = cross(onb.w, onb.u);
    return onb;
}

// ── ShadingFrame — all per-hit normal preparation in one place ──────
//
// Consolidates frame orientation, geo_normal reorientation, and
// entering detection so callers don't repeat ad-hoc flipping logic.
// The shading_normal input can come from vertex interpolation or
// a normal map — downstream code is source-agnostic.
struct ShadingFrame {
    ONB    frame;     // shading ONB, w = oriented shading normal
    float3 geo_n;     // geometric normal, oriented to camera side
    bool   entering;  // ray enters surface (dot(wo, raw_geo_n) > 0)
    bool   valid;     // wo_local.z > 0 after frame construction

    // Build wo in the local frame.  Callers should check `valid` first.
    HD float3 wo_local(float3 wo_world) const {
        return frame.world_to_local(wo_world);
    }
};

// Factory: builds the shading frame with all orientation applied.
//   shading_normal — interpolated (or normal-mapped) surface normal
//   geo_normal     — raw cross-product face normal from hit
//   wo_world       — outgoing direction (= -ray_direction)
inline HD ShadingFrame build_shading_frame(
        float3 shading_normal, float3 geo_normal, float3 wo_world) {
    ShadingFrame sf;

    // 1. Build ONB from shading normal, flip so wo stays in +z
    sf.frame = ONB::from_normal(shading_normal);
    if (dot(sf.frame.w, wo_world) < 0.f) {
        sf.frame.u = -sf.frame.u;
        sf.frame.v = -sf.frame.v;
        sf.frame.w = -sf.frame.w;
    }

    // 2. entering test against raw geometric normal (before reorientation)
    sf.entering = dot(wo_world, geo_normal) > 0.f;

    // 3. orient geometric normal to match the shading hemisphere
    sf.geo_n = geo_normal;
    if (dot(sf.geo_n, sf.frame.w) < 0.f)
        sf.geo_n = -sf.geo_n;

    // 4. validate
    float wo_z = dot(wo_world, sf.frame.w);
    sf.valid = wo_z > 0.f;

    return sf;
}

// Legacy wrapper — keep until all call-sites are migrated.
inline HD ONB orient_frame_to_outgoing(float3 shading_normal, float3 wo_world) {
    ONB onb = ONB::from_normal(shading_normal);
    if (dot(onb.w, wo_world) < 0.f) {
        onb.u = -onb.u;
        onb.v = -onb.v;
        onb.w = -onb.w;
    }
    return onb;
}

// ── Ray ─────────────────────────────────────────────────────────────
struct Ray {
    float3 origin;
    float3 direction;
    float  tmin = 1e-4f;
    float  tmax = 1e20f;
};

// ── Hit record ──────────────────────────────────────────────────────
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4324) // structure was padded due to alignment
#endif
struct HitRecord {
    float3   position;
    float3   normal;          // geometric normal (face normal)
    float3   shading_normal;  // interpolated normal
    float2   uv;
    float    t;
    uint32_t triangle_id;
    uint32_t material_id;
    bool     hit;
};
#ifdef _MSC_VER
#pragma warning(pop)
#endif

// ── Material type (shared between CPU and GPU) ─────────────────────
// Plain (unscoped) enum so device code can compare against uint8_t
// without casts.  Qualified syntax MaterialType::Mirror still works.
enum MaterialType : uint8_t {
    Lambertian           = 0,
    Mirror               = 1,
    Glass                = 2,
    GlossyMetal          = 3,   // Cook-Torrance metallic Fresnel (F0 = Ks)
    Emissive             = 4,
    GlossyDielectric     = 5,   // Cook-Torrance + Lambertian (dielectric Fresnel)
    Translucent          = 6,   // Surface BSDF + interior participating medium
    Clearcoat            = 7,   // Layered: dielectric coat over base BRDF
    Fabric               = 8,   // Diffuse + sheen lobe (cloth)
    DiffuseTransmission  = 9    // Two-sided Lambert: Kd/π reflect, Tf/π transmit
};

// ── Transport mode (adjoint correction) ─────────────────────────────
// Radiance  = camera / eye paths — standard BSDF.
// Importance = light / photon paths — eta^2 correction at refractive
//              interfaces.
enum class TransportMode : int {
    Radiance   = 0,
    Importance = 1
};

// ── Render mode (shared between CPU and GPU) ────────────────────────
enum class RenderMode : int {
    Combined      = 0,
    Full          = Combined,  // Legacy alias
    DirectOnly    = 1,
    IndirectOnly  = 2,
    Normals       = 4,
    MaterialID    = 5,
    Depth         = 6,
    CausticOnly   = 8,
    Coverage      = 9
};

// ── AABB ────────────────────────────────────────────────────────────
struct AABB {
    float3 lo;
    float3 hi;

    HD static AABB empty() {
        return {make_f3(1e30f, 1e30f, 1e30f), make_f3(-1e30f, -1e30f, -1e30f)};
    }

    HD void expand(float3 p) {
        lo = fminf3(lo, p);
        hi = fmaxf3(hi, p);
    }

    HD void expand(const AABB& other) {
        lo = fminf3(lo, other.lo);
        hi = fmaxf3(hi, other.hi);
    }

    HD float3 center() const { return (lo + hi) * 0.5f; }
    HD float3 extent() const { return hi - lo; }

    HD float diagonal() const {
        float3 e = extent();
        return sqrtf(e.x * e.x + e.y * e.y + e.z * e.z);
    }
};

// ── Constants ───────────────────────────────────────────────────────
constexpr float PI       = 3.14159265358979323846f;
constexpr float TWO_PI   = 6.28318530717958647692f;
constexpr float INV_PI   = 0.31830988618379067153f;
constexpr float INV_2PI  = 0.15915494309189533577f;
constexpr float EPSILON  = 1e-6f;
