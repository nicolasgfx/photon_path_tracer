#pragma once
// ─────────────────────────────────────────────────────────────────────
// nee_sampling.h – NEE sample-count policy (host + device)
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"

// Matches the device-side behavior:
//   - bounce 0 uses nee_light_samples
//   - bounce >=1 uses nee_deep_samples
//   - clamps to at least 1
inline HD int nee_shadow_sample_count(int bounce, int nee_light_samples, int nee_deep_samples) {
    const int cfg = (bounce == 0) ? nee_light_samples : nee_deep_samples;
    return (cfg > 0) ? cfg : 1;
}
