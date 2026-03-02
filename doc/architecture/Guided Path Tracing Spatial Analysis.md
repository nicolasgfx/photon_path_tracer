# Guided Path Tracing — Spatial Analysis & kNN Refactor

> Last updated: 2 March 2026
> Status: **Implemented** — build passes, 334/341 fast tests pass

---

## 1  Background

The renderer uses photon-guided path tracing to steer camera rays toward
directions where photon flux is concentrated.  At each non-delta surface
bounce the guide module builds a directional histogram of nearby photon
flux and MIS-combines a guided sample with the BSDF sample.

Prior to this refactor, the guide histogram was pre-aggregated on the CPU
via a **HashHistogram** (a multi-level spatial hash of `GpuGuideBin`
arrays, one per hash bucket) and uploaded as a flat GPU buffer.  A
companion **CellBinGrid** (a dense 3-D grid of `PhotonBin` arrays) was
also built and uploaded for surface photon guide lookup.

### Motivation

Analysis of `hierarchy_report.json` (produced by `photon_map_analysis`)
revealed severe quantitative flaws in both pre-aggregated structures:

| Metric | HashHistogram | CellBinGrid |
|---|---|---|
| False-positive rate (ghost bins) | **31 %** | ~0 % |
| Median flux ratio vs ground truth | **0.38×** | 0.92× |
| Worst-case overestimate | **1210×** | 4.1× |
| Direction accuracy (mean cos error) | 0.12 | 0.08 |

The hash histogram's collisions aliased photon directions across widely
separated cells, creating ghost bins that could steer rays into empty
space.  Meanwhile, the hash grid provided exact per-photon data with
only Ο(K) work per query — comparable to the pre-aggregated lookup once
K is small.

### Decision

Replace the HashHistogram multi-level guided sampling with **per-hitpoint
kNN guided sampling** that walks the photon hash grid directly.  Remove
the surface CellBinGrid from the GPU launch parameters.  The hash grid
becomes the sole spatial acceleration structure for all surface photon
queries (density, caustics, and now guidance).

---

## 2  Architecture After Refactor

### 2.1  Spatial structures on GPU

| Structure | Purpose | Status |
|---|---|---|
| **Hash grid** (sorted-index, Teschner spatial hash) | kNN density estimation (K=100), kNN caustic gather, **kNN guide (K=32)** | **Active — sole structure** |
| HashHistogram (multi-level GpuGuideBin) | Guided direction sampling | **Removed from GPU** |
| CellBinGrid (surface, dense 3-D PhotonBin) | Surface guide fallback | **Removed from GPU** |
| CellBinGrid (volume) | Volume photon guide | Retained (unchanged) |
| CellInfoCache (65K Teschner hash) | Per-cell analysis, guide fraction, caustic fraction | Retained (unchanged) |

### 2.2  Per-bounce data flow (optix_path_trace_v3.cuh)

```
camera bounce (non-delta)
  │
  ├── dev_knn_guide_sample(pos, normal, geo_normal, fib)   ← NEW (K=32 kNN)
  │     └── hash grid walk → GuidedHistogram
  │
  ├── dev_estimate_caustic_only(pos, normal, geo_normal, ...) (K=100 kNN)
  │
  ├── NEE shadow ray
  │
  └── terminal?
       ├─ yes → dev_estimate_photon_density(pos, normal, ...) (K=100 kNN)
       └─ no  → MIS { guided sample, BSDF sample }
                  └── dev_sample_guided_direction(guide_hist, fib, normal, rng)
                  └── dev_guided_pdf(guide_hist, fib, wi)
```

All three kNN queries walk the same hash grid through the same
`params.grid_*` arrays.  No secondary spatial structure is needed.

### 2.3  Include chain

```
optix_device.cu
  └── #include "optix_nee.cuh"         (density, caustic kNN, NEE)
        └── #include "optix_guided.cuh" (kNN guide, sampling, volume guide)
              └── #include "optix_path_trace_v3.cuh" (bounce loop)
```

---

## 3  `dev_knn_guide_sample()` — Implementation Details

**File**: [src/optix/optix_guided.cuh](../../src/optix/optix_guided.cuh#L102)

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `pos` | `float3` | Hit position |
| `normal` | `float3` | Shading normal (for hemisphere tests) |
| `filter_normal` | `float3` | Geometry normal (for tangential-disk metric) |
| `fib` | `DevPhotonBinDirs&` | Pre-computed Fibonacci sphere (32 bins) |

### Algorithm

**Phase 1 — kNN collection** (lines 117–213)

1. Compute integer cell range covering `[pos − r, pos + r]` with
   `cell_size = params.grid_cell_size`.
2. Walk all cells in that range.  For each cell, hash to a bucket key
   via `teschner_hash()`.  Skip already-visited keys (in a 27-element
   local array).
3. For each photon in the bucket:
   - **Tangential-disk distance**: project `diff = pos − photon_pos`
     onto the geometry normal, keep the tangential component
     `d_tan² = |diff − n·(diff·n)|²`.
   - **Thickness gate**: reject if `|d_plane| > DEFAULT_SURFACE_TAU`
     (0.02 scene units).
   - **Normal gate**: reject if `dot(photon_normal, filter_normal) ≤ 0`.
   - **Hemisphere gate**: reject if `dot(photon_wi, filter_normal) ≤ 0`.
   - Insert into a **K=32 max-heap** keyed on `d_tan²`.  If the heap is
     full, replace the root (farthest neighbour) and sift down.

**Phase 2 — Directional binning** (lines 221–244)

1. For each of the `knn_count` accepted photons:
   - Look up the Fibonacci bin via `fib.find_nearest(wi_world)` (32
     dot products).
   - Compute scalar flux = Σ hero-wavelength fluxes.
   - Accumulate `bin_flux[bin] += scalar_flux`.
2. Set `h.valid = (total_flux > 0)`.

**Output**: `GuidedHistogram` — identical layout to the old
`dev_read_cell_histogram()` output.  Consumed by the existing
`dev_sample_guided_direction()` and `dev_guided_pdf()` without changes.

### Complexity

| Aspect | Old (HashHistogram) | New (kNN K=32) |
|---|---|---|
| GPU memory per frame | `65536 × 32 × 8B × n_levels` ≈ 16–48 MB | 0 (reads existing hash grid) |
| Per-hitpoint work | 1 hash + 32 reads (O(1)) | 27 cells × avg occupancy + K heap ops |
| Build cost (CPU per frame) | ~8 ms (scatter 27 neighbours per photon) | 0 (no build needed) |
| Accuracy | 31% false positives, 0.38× flux | Exact per-hitpoint |

The per-hitpoint cost is comparable to the existing
`dev_estimate_caustic_only()` call already at every bounce, with a
smaller K (32 vs 100).

---

## 4  Files Modified

### 4.1  Device code

| File | Change |
|---|---|
| [optix_guided.cuh](../../src/optix/optix_guided.cuh) | Removed `dev_read_cell_histogram()`, `dev_guide_hash()`.  Added `dev_knn_guide_sample()` (K=32 kNN from hash grid, tangential-disk metric, Fibonacci binning, flux-weighted histogram).  `dev_sample_guided_direction()`, `dev_guided_pdf()`, volume guide functions unchanged. |
| [optix_path_trace_v3.cuh](../../src/optix/optix_path_trace_v3.cuh#L374) | Line 374: `dev_read_cell_histogram(hit.position, hit.shading_normal)` → `dev_knn_guide_sample(hit.position, hit.shading_normal, hit.geo_normal, fib)` |
| [launch_params.h](../../src/optix/launch_params.h) | Removed 13 fields: `guide_histogram[]`, `guide_cell_size[]`, `guide_num_levels`, surface `cell_bin_grid`, `cell_grid_valid`, `use_dense_grid_gather`, `cell_grid_min_x/y/z`, `cell_grid_cell_size`, `cell_grid_dim_x/y/z`.  Kept `photon_bin_count`, all `vol_cell_*` volume fields. |

### 4.2  Host code

| File | Change |
|---|---|
| [optix_renderer.cpp](../../src/renderer/optix_renderer.cpp) | Rewrote `fill_cell_grid_params()` — removed ~30 lines of histogram/surface-grid wiring; replaced hash histogram stats block with zeroed stub. |
| [optix_renderer.h](../../src/renderer/optix_renderer.h) | Removed `#include "photon/hash_histogram.h"`, `HashHistogram hash_histogram_` member, `DeviceBuffer d_guide_histogram_[]` member. |
| [optix_photon_trace.cpp](../../src/renderer/optix_photon_trace.cpp) | Removed `hash_histogram_.build()` (~40 lines), surface `cell_bin_grid_.build()` (~8 lines); kept `bin_idx` precompute. |
| [optix_upload.cpp](../../src/renderer/optix_upload.cpp) | Removed hash histogram build/upload (~20 lines), surface CellBinGrid upload (~5 lines); updated `upload_cell_analysis()` signature. |
| [photon_analysis.h](../../src/photon/photon_analysis.h) | `build_cell_analysis()` no longer takes `HashHistogram&`; `active_bins` derived from `directional_spread × 32`; `concentration` derived from `1 − directional_spread`. |

### 4.3  Dead code (retained, not included)

| File | Notes |
|---|---|
| [hash_histogram.h](../../src/photon/hash_histogram.h) | No longer `#include`d by any production source.  Still used by `tools/photon_map_analysis/main.cpp` for offline comparison. |

---

## 5  Key Constants

| Constant | Value | Location | Purpose |
|---|---|---|---|
| `GUIDE_KNN_K` | 32 | optix_guided.cuh | Neighbours for guide histogram |
| `DEFAULT_KNN_K` | 100 | optix_nee.cuh | Neighbours for density + caustic kNN |
| `PHOTON_BIN_COUNT` | 32 | photon_bins.h | Fibonacci sphere directional bins |
| `CELL_CACHE_TABLE_SIZE` | 65536 | cell_cache.h | CellInfoCache hash table |
| `DEFAULT_SURFACE_TAU` | 0.02 | types.h | Tangential-disk thickness gate |
| `DEFAULT_GATHER_RADIUS` | 0.05 | config.h | Base photon gather radius |
| `MAX_GUIDE_LEVELS` | 3 | photon_bins.h | Was used by HashHistogram (now unused) |

---

## 6  Test Impact

341 fast tests, **334 passed**, 2 skipped, 5 failed.

| Test | Status | Notes |
|---|---|---|
| `OptiX.CellBinGridNotBuiltAfterKnnGuide` | **PASS** | New test (replaces old `CellBinGridValidAfterTracePhotons`). Verifies surface grid is intentionally empty. |
| `CellBinGrid.NormalGate_NormalsAndDirectionsPreserved` | FAIL | **Pre-existing** — CPU-only `CellBinGrid.build()` test; class unchanged by this work. |
| `Caustics.SpecularTargetSetFindsGlass` | FAIL | **Pre-existing** — cornell_box scene has no glass/specular triangles (`ts.valid = false`). |
| `Caustics.SampleTargetedCausticPhotons` | FAIL | **Pre-existing** — same root cause. |
| `Caustics.TraceTargetedCausticEmission` | FAIL | **Pre-existing** — same root cause. |
| `Caustics.SinglePhotonBounceTrace` | FAIL | **Pre-existing** — same root cause. |

---

## 7  Possible Follow-Up Optimisations

1. **Merge kNN guide with caustic gather**: Both walk the same hash grid
   at every bounce.  A single combined walk (K=100, splitting photons
   into guide bins and caustic accumulation simultaneously) would halve
   the hash-grid traffic.

2. **Upload `photon_bin_idx` to GPU**: The CPU precomputes
   `bin_idx` for each photon.  Uploading it would avoid the 32-iteration
   `fib.find_nearest()` inner loop during binning (Phase 2).

3. **Tune `GUIDE_KNN_K`**: K=32 was chosen to match the bin count.
   Render quality testing may show optimal K is lower (16) or higher (64).

4. **Remove `hash_histogram.h`** from the source tree entirely once the
   analysis tool no longer needs it for offline comparison.

5. **Remove `use_dense_grid_` flag and surface `cell_bin_grid_` member**
   from `OptixRenderer` if no remaining code path populates them.