#pragma once
// ─────────────────────────────────────────────────────────────────────
// types.h – Fundamental math types with host/device annotations
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
inline HD float3& operator*=(float3& a, float s)  { a.x*=s;   a.y*=s;   a.z*=s;   return a; }

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
    if (d <= 0.f) return make_f3(0,0,0);
#ifdef __CUDA_ARCH__
    return v * rsqrtf(d);
#else
    return v * (1.f / sqrtf(d));
#endif
}

inline HD float3 fminf3(float3 a, float3 b) {
    return {fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z)};
}
inline HD float3 fmaxf3(float3 a, float3 b) {
    return {fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z)};
}

// ── Coordinate frame from normal ────────────────────────────────────
struct ONB {
    float3 u, v, w; // w = normal

    HD static ONB from_normal(float3 n) {
        ONB onb;
        onb.w = n;
        float3 a = (fabsf(n.x) > 0.9f) ? make_f3(0,1,0) : make_f3(1,0,0);
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
    float3   normal;       // geometric normal
    float3   shading_normal;
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
    Lambertian        = 0,
    Mirror            = 1,
    Glass             = 2,
    GlossyMetal       = 3,   // Cook-Torrance metallic Fresnel (F0 = Ks)
    Emissive          = 4,
    GlossyDielectric  = 5,   // Cook-Torrance + Lambertian (dielectric Fresnel)
    Translucent       = 6,   // Surface BSDF + interior participating medium
    Clearcoat         = 7,   // Layered: dielectric coat over base BRDF
    Fabric            = 8    // Diffuse + sheen lobe (cloth)
};

// ── Transport mode (adjoint correction, §2.2) ───────────────────────
// Radiance  = camera / eye paths — standard BSDF.
// Importance = light / photon paths — η² correction at refractive interfaces.
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
    PhotonMap     = 3,
    Normals       = 4,
    MaterialID    = 5,
    Depth         = 6,
    GuideMap      = 7,    // Visualise cell-bin histogram as false-colour directional map
    CausticOnly   = 8,    // Caustic photon density estimate only
    Coverage      = 9     // (legacy)
};

// ── Constants ───────────────────────────────────────────────────────
constexpr float PI       = 3.14159265358979323846f;
constexpr float TWO_PI   = 6.28318530717958647692f;
constexpr float INV_PI   = 0.31830988618379067153f;
constexpr float INV_2PI  = 0.15915494309189533577f;
constexpr float EPSILON  = 1e-6f;
