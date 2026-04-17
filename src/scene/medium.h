#pragma once
// ─────────────────────────────────────────────────────────────────────
// medium.h – Homogeneous participating medium (v5 RGB)
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/color.h"

struct HomogeneousMedium {
    Color3 sigma_s = Color3::zero();   // scattering coefficient
    Color3 sigma_a = Color3::zero();   // absorption coefficient
    Color3 sigma_t = Color3::zero();   // extinction = sigma_s + sigma_a
    float  g       = 0.0f;            // Henyey-Greenstein asymmetry
};

// ── Medium stack (nested-object interior tracking) ──────────────────
struct MediumStack {
    static constexpr int MAX_DEPTH = 4;
    int  stack[MAX_DEPTH] = {-1, -1, -1, -1};
    int  depth            = 0;

    HD int  current_medium_id() const { return depth > 0 ? stack[depth - 1] : -1; }
    HD void push(int medium_id) { if (depth < MAX_DEPTH) stack[depth++] = medium_id; }
    HD void pop()               { if (depth > 0) --depth; }
};
