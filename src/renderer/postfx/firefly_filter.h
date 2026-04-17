#pragma once
// ─────────────────────────────────────────────────────────────────────
// postfx/firefly_filter.h – Median-based outlier suppression (v5)
// ─────────────────────────────────────────────────────────────────────

// Launch firefly filter on an HDR float4 buffer in-place.
// d_temp must be at least width*height*4 floats (scratch space).
// Radius is clamped to [1,2] (3×3 or 5×5 median window).
// Threshold is the MAD multiplier K (typically 3–5).
void launch_firefly_filter(float* d_hdr, float* d_temp,
                           int width, int height,
                           int radius, float threshold);
