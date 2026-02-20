#pragma once
// ─────────────────────────────────────────────────────────────────────
// cdf.h – Small utilities for monotone CDF sampling (host + device)
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"

// Binary search on a monotone non-decreasing CDF array where:
//   - cdf has length n
//   - cdf[n-1] should be 1.0 (or very close)
//   - u is in [0,1)
// Returns an index in [0, n-1].
inline HD int binary_search_cdf(const float* cdf, int n, float u) {
    if (n <= 1) return 0;
    if (u <= 0.f) return 0;

    int lo = 0;
    int hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cdf[mid] <= u) lo = mid + 1;
        else hi = mid;
    }

    if (lo < 0) lo = 0;
    if (lo >= n) lo = n - 1;
    return lo;
}
