# Speed Test Checklist: Precomputed Photon Bins + Optimized NEE

This document is meant to answer two questions:

1) Does the **actual** code path in the current build really use the precomputed-bin changes and the optimized NEE / guided sampling?
2) What is the **current configuration** (shadow rays, gather neighborhood, bounce-dependent reductions) so we can tweak performance intentionally?

Scope note: this checklist is written for the OptiX path (GPU final render), not the CPU fallback.

---

## 1) Checklist — verify the real code path

### A. Precomputed photon directional bins are actually used

- [ ] Bin cache is allocated at render resolution.
	- Where: [src/optix/optix_renderer.h](src/optix/optix_renderer.h) (`OptixRenderer::resize` allocates `d_photon_bin_cache_` + `d_photon_density_cache_`).
	- Sanity: allocation size scales with `W*H*PHOTON_BIN_COUNT`.

- [ ] The bin-population pass is executed before the performance measurement.
	- Where: [src/main.cpp](src/main.cpp) calls `optix_renderer.populate_photon_bins(...)` right after the “NEE debug PNG” block for the progressive render.
	- What to look for in console:
		- `[Bins] Populating photon directional bin cache (... bins per pixel)...`
		- `[Bins] Population done: ... ms (...)`

- [ ] Population pass triggers the special kernel path (`populate_bins_mode = 1`).
	- Where: [src/optix/optix_renderer.cpp](src/optix/optix_renderer.cpp) sets `lp.populate_bins_mode = 1` in `OptixRenderer::populate_photon_bins`.
	- Device-side switch: [src/optix/optix_device.cu](src/optix/optix_device.cu) `__raygen__render` early-exits into `dev_populate_bins_for_pixel(...)` when `params.populate_bins_mode` is set.

- [ ] Final render launches set `photon_bins_valid = 1` after population.
	- Where: [src/optix/optix_renderer.cpp](src/optix/optix_renderer.cpp) sets `lp.photon_bins_valid = bins_populated_ ? 1 : 0` in `render_one_spp` (and similar in `render_final`).
	- Gate used on device: [src/optix/optix_device.cu](src/optix/optix_device.cu) `use_bins = (params.photon_bins_valid == 1 && params.photon_bin_cache != nullptr)`.

- [ ] Bin population is “first diffuse hit” oriented (center ray, follow specular, stop at first diffuse).
	- Where: [src/optix/optix_device.cu](src/optix/optix_device.cu) `dev_populate_bins_for_pixel(...)` traces the center ray and follows specular bounces up to a fixed count, then gathers photons at that first diffuse hit.

### B. Cached photon density is actually saving work (frame > 0)

- [ ] Density cache is written on frame 0 and read on subsequent frames (bounce 0 only).
	- Where: [src/optix/optix_device.cu](src/optix/optix_device.cu) inside `full_path_trace(...)`:
		- Read path: `use_bins && bounce == 0 && photon_density_cache != nullptr && frame_number > 0`.
		- Write path: `use_bins && bounce == 0 && photon_density_cache != nullptr && frame_number == 0`.
	- Expectation: on spp>0, “Photon gather” time in GPU profiling should drop significantly if the cache hit-rate is high.
	- How to verify: [src/optix/optix_renderer.cpp](src/optix/optix_renderer.cpp) `OptixRenderer::print_kernel_profiling()` prints `Photon gather` totals.

- [ ] The density cache is not accidentally disabled by missing pointers.
	- Where: [src/optix/launch_params.h](src/optix/launch_params.h) requires `photon_density_cache` to be non-null.
	- Host wiring: [src/optix/optix_renderer.cpp](src/optix/optix_renderer.cpp) `fill_common_params(...)` passes device pointers when the buffers exist.

### C. Guided NEE (bin-aware) is actually active

- [ ] On final render, direct lighting uses guided NEE when bins are valid.
	- Where: [src/optix/optix_device.cu](src/optix/optix_device.cu) inside `full_path_trace(...)`:
		- `use_bins` → `dev_nee_guided(...)`
		- else → `dev_nee_direct(...)`

- [ ] Guided NEE fallback conditions are understood (these can kill any expected change).
	- Where: [src/optix/optix_device.cu](src/optix/optix_device.cu) `dev_nee_guided(...)` falls back to `dev_nee_direct(...)` when:
		- bins not valid,
		- emissive triangle count exceeds an internal max (stack budget),
		- total bin flux is 0.
	- Action item for speed tests: print or log the scene’s `num_emissive` (host already prints “Uploaded N emissive triangles”). If N is large, guided NEE may be bypassed.

- [ ] Shadow rays are real OptiX visibility rays (terminate on first hit).
	- Where: [src/optix/optix_device.cu](src/optix/optix_device.cu) `trace_shadow(...)` uses `OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT` and disables closest-hit.

### D. Bounce-dependent shadow-ray reduction is really happening

- [ ] NEE sample count changes with bounce index.
	- Where: [src/optix/optix_device.cu](src/optix/optix_device.cu) in both `dev_nee_direct(...)` and `dev_nee_guided(...)`:
		- bounce 0 → `params.nee_light_samples`
		- bounce ≥1 → `params.nee_deep_samples`

- [ ] Host actually sets both parameters for final render launches.
	- Where: [src/optix/optix_renderer.cpp](src/optix/optix_renderer.cpp) sets `lp.nee_light_samples` and `lp.nee_deep_samples` in `render_one_spp(...)`.

### E. Guided BSDF continuation from bins is actually active (bounce 0)

- [ ] The first diffuse bounce direction is sampled from bin flux (not cosine).
	- Where: [src/optix/optix_device.cu](src/optix/optix_device.cu) `full_path_trace(...)`:
		- if `use_bins && bounce == 0` → `dev_sample_guided_bounce(...)`
		- else → cosine hemisphere.

---

## 2) Current configuration snapshot (speed-relevant knobs)

This section describes what the current build is effectively doing by default.

### A. Shadow rays (NEE)

- Bounce-0 shadow rays per diffuse hit:
	- Default: `DEFAULT_NEE_LIGHT_SAMPLES = 4`
	- Source: [src/core/config.h](src/core/config.h)
	- Used by: [src/optix/optix_renderer.cpp](src/optix/optix_renderer.cpp) when setting `LaunchParams.nee_light_samples`.

- Deeper-bounce shadow rays per diffuse hit (bounce ≥ 1):
	- Default: `DEFAULT_NEE_DEEP_SAMPLES = 1`
	- Source: [src/core/config.h](src/core/config.h)
	- Device logic: [src/optix/optix_device.cu](src/optix/optix_device.cu) `M_cfg = (bounce == 0) ? nee_light_samples : nee_deep_samples`.

- Guided NEE internal constants (extra work per shading point):
	- `NEE_GUIDED_MAX_EMISSIVE = 128` (fallback threshold)
	- `NEE_GUIDED_ALPHA = 5.0` (flux boost strength)
	- Source: [src/optix/optix_device.cu](src/optix/optix_device.cu)

### B. Photon gather neighborhood size

- Gather radius (`r`):
	- Default: `DEFAULT_GATHER_RADIUS = 0.05`
	- Source: [src/core/config.h](src/core/config.h)
	- Runtime: comes from `RenderConfig.gather_radius` → uploaded into `LaunchParams.gather_radius`.
	- Host path: [src/optix/optix_renderer.cpp](src/optix/optix_renderer.cpp) stores it in `gather_radius_` via `upload_photon_data(...)`.

- Hash-grid cell size:
	- Current: `cell_size = gather_radius * HASHGRID_CELL_FACTOR`
	- Default factor: `HASHGRID_CELL_FACTOR = 2.0` → `cell_size = 2r` (diameter)
	- Source: [src/core/config.h](src/core/config.h)
	- Host wiring: [src/optix/optix_renderer.cpp](src/optix/optix_renderer.cpp) sets `LaunchParams.grid_cell_size`.

- Effective cell neighborhood scanned per gather:
	- Device side computes cell bounds from `(pos ± r) / cell_size`.
	- With `cell_size = 2r`, each axis usually spans 1–2 cells ⇒ typically up to 8 hash buckets checked per gather.
	- Source: [src/optix/optix_device.cu](src/optix/optix_device.cu) `dev_estimate_photon_density(...)`.

- Surface-plane filter (reduces gathered photons):
	- `DEFAULT_SURFACE_TAU = 0.02`
	- Source: [src/core/config.h](src/core/config.h)
	- Used by: [src/optix/optix_device.cu](src/optix/optix_device.cu) in both density estimation and bin population.

### C. Shadowing of photon indirect via NEE visibility

- Indirect lighting is multiplied by a visibility-derived weight:
	- `vis_weight = max(nee_visibility, PHOTON_SHADOW_FLOOR)`
	- Default floor: `PHOTON_SHADOW_FLOOR = 0.1`
	- Source: [src/core/config.h](src/core/config.h)
	- Device usage: [src/optix/optix_device.cu](src/optix/optix_device.cu) in `full_path_trace(...)`.

### D. Photon directional bins (precompute settings)

- Bin count per pixel:
	- `PHOTON_BIN_COUNT = 32`
	- Source: [src/core/config.h](src/core/config.h)

- Memory footprint (important for speed expectations):
	- `PhotonBin` is 24 bytes, so bin cache is `24 * PHOTON_BIN_COUNT * W * H` bytes.
	- Example comment: at 1024×768 and 32 bins, bin cache alone is ~604 MB.
	- Source: [src/core/photon_bins.h](src/core/photon_bins.h)

- When bins/density cache are used:
	- Only when `photon_bins_valid == 1` and the cache pointers are non-null.
	- Source: [src/optix/optix_device.cu](src/optix/optix_device.cu)

### E. Bounce / Russian roulette

- Max bounces:
	- Host sets `LaunchParams.max_bounces` from `RenderConfig.max_bounces` for final rendering.
	- Host path: [src/optix/optix_renderer.cpp](src/optix/optix_renderer.cpp)

- Russian roulette (OptiX final render path):
	- Uses compile-time defaults `DEFAULT_MIN_BOUNCES_RR` and `DEFAULT_RR_THRESHOLD`.
	- Device path: [src/optix/optix_device.cu](src/optix/optix_device.cu) uses these constants directly in `full_path_trace(...)`.
	- Note: this means `RenderConfig.min_bounces_rr` / `RenderConfig.rr_threshold` do not currently affect the OptiX kernel.

---

## 3) Quick “why no speedup?” triage (things this checklist can reveal)

- [ ] Guided NEE is silently falling back (e.g., too many emissive triangles), so you’re paying the bin-population cost without getting the guided NEE behavior.
- [ ] Density caching is not taking effect (e.g., cache pointer null, `photon_bins_valid` off, or `frame_number` not progressing as expected), so photon gather remains expensive every spp.
- [ ] Guided NEE adds per-hit overhead (per-hit temporary CDF construction), possibly offsetting savings from cached photon density.

