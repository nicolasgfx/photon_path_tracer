#pragma once
// ─────────────────────────────────────────────────────────────────────
// ior_stack.h – Nested-dielectric IOR tracking (host + device)
//
// Tracks which dielectric medium the ray is currently inside.  Empty
// stack → air (IOR 1.0).  Push on enter, pop on exit, no change on
// reflect.  Works with Cauchy dispersion by storing the material's
// nominal IOR (per-wavelength IOR is recomputed from the material).
//
// Single HD implementation shared by CPU emitter (emitter.h) and
// GPU specular bounces (optix_device.cu).
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"   // HD macro

struct IORStack {
    static constexpr int MAX_DEPTH = 4;
    float iors[MAX_DEPTH] = {0.f, 0.f, 0.f, 0.f};
    int   depth           = 0;

    HD float top() const { return depth > 0 ? iors[depth - 1] : 1.0f; }
    HD void  push(float ior) { if (depth < MAX_DEPTH) iors[depth++] = ior; }
    HD void  pop()           { if (depth > 0) --depth; }
};
