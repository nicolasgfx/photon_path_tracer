#pragma once
// ─────────────────────────────────────────────────────────────────────
// russian_roulette.h – Unbiased path termination (v5 RGB)
//
// After min_bounces_rr guaranteed bounces, probabilistically terminate
// paths with low throughput.  Surviving paths are boosted by 1/p.
//
// Host/device shared (HD qualifier).
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"

// ── RR result ───────────────────────────────────────────────────────
struct RRResult {
    bool  terminate;     // true → kill this path
    float inv_survival;  // multiply throughput by this (1/p or 1.0 if not applied)
};

// ── Evaluate Russian Roulette ───────────────────────────────────────
// max_tp:     max component of current throughput
// rr_threshold: maximum survival probability (e.g. 0.95)
// xi:         uniform random in [0,1)

inline HD RRResult russian_roulette(float max_tp, float rr_threshold, float xi) {
    RRResult r;
    float p_survive = fminf(rr_threshold, max_tp);

    if (p_survive < 1e-4f) {
        r.terminate = true;
        r.inv_survival = 1.f;
        return r;
    }

    if (xi >= p_survive) {
        r.terminate = true;
        r.inv_survival = 1.f;
        return r;
    }

    r.terminate = false;
    r.inv_survival = 1.f / p_survive;
    return r;
}
