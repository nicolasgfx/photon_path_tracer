#pragma once
// ─────────────────────────────────────────────────────────────────────
// color.h – RGB color type for all light transport
//
// Replaces the 4-bin Spectrum struct.  All rendering, photon storage,
// and BSDF evaluation now use Color3 (linear RGB float).  sRGB gamma
// and tone mapping are applied only at the final display stage.
// ─────────────────────────────────────────────────────────────────────
#include "types.h"

// ── Color3 — linear RGB ─────────────────────────────────────────────
struct Color3 {
    float r, g, b;

    // ── Factory methods ─────────────────────────────────────────────
    HD static Color3 zero()         { return {0.f, 0.f, 0.f}; }
    HD static Color3 one()          { return {1.f, 1.f, 1.f}; }
    HD static Color3 constant(float v) { return {v, v, v}; }
    HD static Color3 from_rgb(float r, float g, float b) { return {r, g, b}; }

    // ── Accessors ───────────────────────────────────────────────────
    HD float  operator[](int i) const { return (&r)[i]; }
    HD float& operator[](int i)       { return (&r)[i]; }

    // ── CIE Y luminance (ITU-R BT.709) ─────────────────────────────
    HD float luminance() const {
        return 0.2126f * r + 0.7152f * g + 0.0722f * b;
    }

    HD float max_component() const {
        return fmaxf(r, fmaxf(g, b));
    }

    HD float min_component() const {
        return fminf(r, fminf(g, b));
    }

    HD float sum() const { return r + g + b; }

    HD bool is_zero() const {
        return r == 0.f && g == 0.f && b == 0.f;
    }

    HD bool is_finite() const {
        return isfinite(r) && isfinite(g) && isfinite(b);
    }

    HD bool is_black() const {
        return r <= 0.f && g <= 0.f && b <= 0.f;
    }

    // ── Clamping ────────────────────────────────────────────────────
    HD Color3 clamped(float lo = 0.f, float hi = 1e30f) const {
        return {fmaxf(lo, fminf(r, hi)),
                fmaxf(lo, fminf(g, hi)),
                fmaxf(lo, fminf(b, hi))};
    }

    HD Color3 clamped_non_negative() const {
        return {fmaxf(0.f, r), fmaxf(0.f, g), fmaxf(0.f, b)};
    }

    // ── Conversion to CUDA float3 ──────────────────────────────────
    HD float3 to_float3() const { return make_f3(r, g, b); }
    HD static Color3 from_float3(float3 v) { return {v.x, v.y, v.z}; }
};

// ── Arithmetic operators ────────────────────────────────────────────
inline HD Color3 operator+(Color3 a, Color3 b) { return {a.r+b.r, a.g+b.g, a.b+b.b}; }
inline HD Color3 operator-(Color3 a, Color3 b) { return {a.r-b.r, a.g-b.g, a.b-b.b}; }
inline HD Color3 operator*(Color3 a, Color3 b) { return {a.r*b.r, a.g*b.g, a.b*b.b}; }
inline HD Color3 operator*(Color3 a, float s)  { return {a.r*s, a.g*s, a.b*s}; }
inline HD Color3 operator*(float s, Color3 a)  { return {a.r*s, a.g*s, a.b*s}; }
inline HD Color3 operator/(Color3 a, float s)  { return {a.r/s, a.g/s, a.b/s}; }
inline HD Color3 operator/(Color3 a, Color3 b) { return {a.r/b.r, a.g/b.g, a.b/b.b}; }

inline HD Color3& operator+=(Color3& a, Color3 b) { a.r+=b.r; a.g+=b.g; a.b+=b.b; return a; }
inline HD Color3& operator-=(Color3& a, Color3 b) { a.r-=b.r; a.g-=b.g; a.b-=b.b; return a; }
inline HD Color3& operator*=(Color3& a, Color3 b) { a.r*=b.r; a.g*=b.g; a.b*=b.b; return a; }
inline HD Color3& operator*=(Color3& a, float s)  { a.r*=s;   a.g*=s;   a.b*=s;   return a; }
inline HD Color3& operator/=(Color3& a, float s)  { a.r/=s;   a.g/=s;   a.b/=s;   return a; }

// ── Color4 — RGBA (for framebuffer output) ──────────────────────────
struct Color4 {
    float r, g, b, a;

    HD static Color4 zero() { return {0.f, 0.f, 0.f, 0.f}; }
    HD static Color4 from_rgb(Color3 c, float alpha = 1.f) {
        return {c.r, c.g, c.b, alpha};
    }

    HD Color3 rgb() const { return {r, g, b}; }

    HD float  operator[](int i) const { return (&r)[i]; }
    HD float& operator[](int i)       { return (&r)[i]; }
};

// ── Tone mapping & gamma ────────────────────────────────────────────

// sRGB gamma encoding (IEC 61966-2-1)
inline HD float srgb_gamma(float c) {
    if (c <= 0.0031308f) return 12.92f * c;
    return 1.055f * powf(c, 1.f / 2.4f) - 0.055f;
}

// sRGB gamma decoding (linear from sRGB)
inline HD float srgb_to_linear(float c) {
    if (c <= 0.04045f) return c / 12.92f;
    return powf((c + 0.055f) / 1.055f, 2.4f);
}

// ACES Filmic tone mapping (Narkowicz 2015 fit, per-channel)
inline HD float aces_filmic(float x) {
    return (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f);
}

// Full pipeline: linear RGB → tone map → gamma → [0,1]
inline HD Color3 tonemap_aces_srgb(Color3 c, float exposure = 1.f) {
    Color3 exposed = c * exposure;
    return {
        fminf(1.f, fmaxf(0.f, srgb_gamma(aces_filmic(fmaxf(0.f, exposed.r))))),
        fminf(1.f, fmaxf(0.f, srgb_gamma(aces_filmic(fmaxf(0.f, exposed.g))))),
        fminf(1.f, fmaxf(0.f, srgb_gamma(aces_filmic(fmaxf(0.f, exposed.b)))))
    };
}
