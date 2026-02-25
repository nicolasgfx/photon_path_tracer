#pragma once
// ─────────────────────────────────────────────────────────────────────
// adaptive_emission.h – View-adaptive photon emission context (stub)
// ─────────────────────────────────────────────────────────────────────
// Provides the AdaptiveEmissionContext used by emitter.h's
// sample_emitted_photon_adaptive() to bias photon emission towards
// camera-visible regions.  Currently a minimal stub.
// ─────────────────────────────────────────────────────────────────────
#include "core/alias_table.h"

/// Context for view-adaptive photon emission.
/// Built per-frame from camera visibility information.
struct AdaptiveEmissionContext {
    AliasTable alias_table;  // view-weighted emitter point selection
    bool       valid = false;

    /// Sample an emitter point index using the alias table.
    /// Returns point index and output PDF.
    int sample_point(float u1, float u2, float& out_pdf) const {
        if (!valid || alias_table.n == 0) {
            out_pdf = 0.f;
            return -1;
        }
        int idx = alias_table.sample(u1, u2);
        out_pdf = (alias_table.n > 0) ? alias_table.pdf(idx) : 0.f;
        return idx;
    }
};
