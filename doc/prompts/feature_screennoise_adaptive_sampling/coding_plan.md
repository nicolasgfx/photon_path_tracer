
# Feature: Screen-noise adaptive sampling

## Goal
Allocate more samples (rays) to pixels whose neighborhood is still noisy, while letting low-noise regions stop early. This should improve perceived convergence for a fixed render-time / sample budget.

This feature is **screen-space adaptive sampling**:
- **Input:** current per-pixel Monte Carlo statistics.
- **Output:** a per-pixel decision (active mask / target spp) controlling where the next sampling pass traces rays.

## Non-goals (v1)
- No new denoiser.
- No new UI panels; configuration via existing `RenderConfig` only.
- No tile-based scheduler rework (unless required for performance).

## Where this fits in the codebase
Two render paths exist:
- **CPU fallback**: `Renderer::render_frame()` in `src/renderer/renderer.cpp` loops `y/x/spp`.
- **OptiX GPU**: `OptixRenderer::render_final()` in `src/optix/optix_renderer.cpp` launches **1 spp per pass** for progress feedback.

Adaptive sampling is easiest to implement and most valuable in the OptiX path first (it already renders in 1-spp batches).

## Key design decisions

### 1) Noise metric (per pixel)
We need a metric that correlates with Monte Carlo noise, not just image detail.

Recommended: **running variance of linear luminance** computed from per-sample contributions.

Per pixel maintain:
- `n` = sample count
- `sumY` = Σ Y
- `sumY2` = Σ Y²

Compute:
$$\mu = \frac{\text{sumY}}{n}$$
$$\sigma^2 = \max\left(\frac{\text{sumY2}}{n} - \mu^2, 0\right)$$
Standard error (noise of the mean):
$$\text{se} = \sqrt{\frac{\sigma^2}{n}}$$
Relative noise (scale invariance):
$$r = \frac{\text{se}}{|\mu| + \epsilon}$$

Notes:
- Use **linear** luminance (pre-tonemap), not sRGB.
- Clamp/guard for `n < 2`.
- `epsilon` ~ 1e-4 to avoid exploding in dark pixels.

### 2) “Noise around a pixel” (neighborhood aggregation)
Compute a local neighborhood statistic on the per-pixel relative noise map `r`:

- window: `(2R+1)×(2R+1)`; start with `R=1` (3×3).
- aggregation: `max()` or `mean()`.

Recommended for v1: `localNoise = max(r in window)`.
Rationale: aggressively focuses samples on speckle/caustic noise islands and avoids averaging away localized noise.

### 3) Sampling policy
We need a policy that turns `localNoise` into “trace more rays here”.

Two viable policies:

**Policy A — Active mask + iterative 1 spp passes (recommended v1)**
- Always render in 1-spp passes.
- After each update interval, compute `localNoise`.
- Set `active[p] = (localNoise[p] > threshold) && (n[p] < max_spp)`.
- Next pass traces rays only for active pixels.
- Stop when no active pixels remain or the global budget is hit.

**Policy B — Per-pixel target spp (v2)**
- Compute `target_spp[p] = clamp(min_spp + f(localNoise), min_spp, max_spp)`.
- Keep sampling until `n[p] >= target_spp[p]`.

Policy A is simpler and integrates well with the existing OptiX “1 spp loop”.

## Data/buffer changes

### CPU path (`FrameBuffer`)
Extend `FrameBuffer` with luminance moments for noise estimation:
- `std::vector<float> lum_sum;`
- `std::vector<float> lum_sum2;`
- optionally `std::vector<float> noise;` (debug)

Update `FrameBuffer::clear()` and `resize()` accordingly.

### OptiX path (device buffers + LaunchParams)
Add optional device buffers:
- `float* lum_sum;`  // [W*H]
- `float* lum_sum2;` // [W*H]
- `uint8_t* active_mask;` // [W*H] 0/1 (or `uint32_t` for alignment)

Add these pointers to `LaunchParams` in `src/optix/launch_params.h`.
The OptiX device code should treat `nullptr` as “feature disabled” to avoid breaking other modes.

## Implementation steps (OptiX / GPU first)

### Step 1 — Add config toggles
In `RenderConfig` (`src/renderer/renderer.h`), add:
- `bool adaptive_sampling = false;`
- `int adaptive_min_spp = 4;`        // warmup samples everywhere
- `int adaptive_max_spp = samples_per_pixel;` // cap
- `int adaptive_update_interval = 1;` // how often to recompute mask (in spp)
- `float adaptive_threshold = 0.02f;` // relative noise threshold
- `int adaptive_radius = 1;`         // neighborhood radius

Keep defaults conservative and keep `adaptive_sampling=false` by default.

### Step 2 — Allocate and clear new device buffers
In `OptixRenderer`:
- Add `DeviceBuffer d_lum_sum_; d_lum_sum2_; d_active_mask_;`.
- Allocate in `resize()` alongside existing `d_sample_counts_`.
- Clear in `clear_buffers()` and at start of `render_final()`.

### Step 3 — Accumulate luminance moments per sample in the kernel
In `src/optix/optix_device.cu` where per-pixel contributions are accumulated:
- After computing the per-sample radiance contribution `L_accum` (linear), compute luminance `Y`.
	- Prefer using a linear XYZ conversion and take `Y`, or a linear RGB luminance if RGB is already available in linear space.
- `atomicAdd(&params.lum_sum[p], Y);`
- `atomicAdd(&params.lum_sum2[p], Y * Y);`

Notes:
- If accumulation is single-thread-per-pixel in the raygen program, atomics may not be needed; verify the execution model.
- Keep this behind `if (params.lum_sum && params.lum_sum2)`.

### Step 4 — Compute `active_mask` from luminance moments
Add a CUDA kernel (not OptiX) invoked from the host after a pass:

Inputs: `sample_counts`, `lum_sum`, `lum_sum2`, parameters (`threshold`, `radius`, `max_spp`, image dims)
Output: `active_mask`.

Implementation outline:
1. Compute per-pixel relative noise `r` (can be in registers; no need to store a full `r` buffer for v1).
2. For each pixel, compute `localNoise = max(r in neighborhood)`.
3. `active[p] = (n >= adaptive_min_spp) && (localNoise > threshold) && (n < adaptive_max_spp)`.
4. Also compute a device-side reduction `active_count` for early exit (optional but useful).

Because neighborhood reads are required, consider:
- Simple global memory reads first (v1).
- Later optimize with shared memory tiles.

### Step 5 — Make raygen skip inactive pixels
In the OptiX raygen program:
- Compute `pixel_idx`.
- If `params.active_mask != nullptr` and `params.active_mask[pixel_idx] == 0`, **return early** without tracing.

Important: do not update `sample_counts` for skipped pixels.

### Step 6 — Restructure `render_final()` loop
Current behavior: loop `s=0..total_spp-1`, always launch full frame.

New behavior when `config.adaptive_sampling`:
1. Clear accumulation buffers.
2. Warmup phase: render `adaptive_min_spp` passes with `active_mask=nullptr` (or mask=all-ones).
3. Adaptive phase: for `pass = adaptive_min_spp .. adaptive_max_spp-1`:
	 - recompute mask every `adaptive_update_interval` passes.
	 - if mask is empty (no active pixels), break.
	 - launch 1-spp pass with `active_mask` set.
4. Keep progress output meaningful (report effective spp stats: min/avg/max sample count).

Keep non-adaptive path unchanged.

### Step 7 — Debug outputs (high value for tuning)
Add optional debug dumps (guarded by a flag or only in debug builds):
- `active_mask` visualization (black/white PNG).
- `sample_counts` heatmap.
- `localNoise` heatmap (optional; may require storing a float buffer).

This makes it obvious whether the metric is selecting true noise vs. edges.

## CPU fallback implementation (secondary)
Add the same luminance moment accumulators into the CPU path:
- In the inner sampling loop in `Renderer::render_frame()`, after tracing `Spectrum L`, compute `Y` and update `lum_sum/lum_sum2`.
- Implement Policy A but on CPU:
	- warmup `min_spp`.
	- compute `active_mask` on CPU.
	- for each pass, only sample pixels with `active_mask[p]`.

CPU note: because the CPU loop is currently `for y/x { for s<spp }`, adaptive needs restructuring to `for pass { for y/x if active }`.

## Correctness and bias considerations
Adaptive sampling decisions depend on noisy estimates; “optional stopping” can introduce subtle bias in rare cases.

Mitigations:
- Enforce a non-trivial `adaptive_min_spp` (e.g. 4–16) so variance estimates are stable.
- Compute noise on **linear** luminance before tonemap.
- Use conservative thresholds.
- If bias becomes visible in tests, consider a two-stream approach (use a small set of held-out samples for the stopping decision).

## Testing / validation

### Unit tests (where feasible)
- Add a small deterministic test for noise estimator math (variance/relative error) using synthetic moment values.
- Add a test that `active_mask` becomes empty for a uniform image with zero variance.

### Image/regression tests
Use existing test harness patterns in `tests/`:
- Render a tiny scene (e.g. 64×64) with fixed RNG seeds.
- Compare against a baseline image or compare statistics:
	- adaptive should reach equal-or-better error for same total traced samples.

### Performance checks
- Track overhead of mask computation and divergence.
- Measure: total traced rays, time per pass, and final error.

## Benefits
- **Faster perceived convergence**: spends rays on noisy regions (shadows, caustics, glossy paths).
- **More uniform noise** across the image at a fixed sample budget.
- **Better time-to-first-clean-image** in final renders, especially for scenes with large smooth areas.

## Risks / tradeoffs
- **False positives on edges/textures** if the metric uses spatial variance of the image instead of sample variance; can oversample detail.
	- Mitigation: use per-pixel sample variance (recommended above) and neighborhood `max` aggregation.
- **Overhead and GPU divergence**: skipping pixels makes warp work uneven; mask computation costs extra bandwidth.
	- Mitigation: update mask every N passes, not every pass; later move to tile-based compaction if needed.
- **Dark-pixel instability**: relative error metric can blow up when mean luminance is near zero.
	- Mitigation: `epsilon`, clamp relative error, optionally use absolute error below a luminance floor.
- **Potential bias from optional stopping** (subtle, but real in some Monte Carlo setups).
	- Mitigation: conservative thresholds + minimum spp; consider two-stream stopping if needed.

## Rollout plan
1. Land the OptiX adaptive path behind `RenderConfig::adaptive_sampling`.
2. Add debug heatmaps for mask + sample counts.
3. Tune defaults on a few representative scenes.
4. Port the same logic to the CPU fallback path.

