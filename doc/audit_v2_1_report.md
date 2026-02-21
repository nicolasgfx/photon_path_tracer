# Guideline v2.1 Compliance Audit Report

**Date:** 2025  
**Scope:** CPU ground truth algorithms in unit tests vs. guideline v2.1  
**Status:** READ-ONLY AUDIT — no code changes made  

---

## Summary

| Category | Compliant | Non-Compliant | Warnings |
|----------|-----------|---------------|----------|
| A. Ground truth algorithm correctness | 10 | 5 | 3 |
| B. Test coverage per §15 | 6 | 3 | 0 |
| C. Test correctness (wrong/outdated) | 5 | 4 | 2 |
| D. Missing tests per guideline | — | 12 | — |

---

## A) Ground Truth Algorithm Correctness per Guideline v2.1

### A1. First-hit camera architecture (§2, §7)

| File | Line(s) | Status | Severity |
|------|---------|--------|----------|
| `src/renderer/renderer.cpp` | L246–L300 | **COMPLIANT** | — |
| `tests/test_ground_truth.cpp` | L208–L338 | **COMPLIANT** | — |
| `tests/test_per_ray_validation.cpp` | L280–L390 | **COMPLIANT** | — |
| `tests/test_pixel_comparison.cpp` | L210–L310 | **COMPLIANT** | — |

**Detail:** All CPU path tracers correctly implement first-hit + specular chain. Camera rays
follow specular bounces (`continue`), then at the first diffuse hit perform NEE + photon
gather and `break`. No multi-bounce camera continuation. Matches §7 and §E3.

---

### A2. Tangential disk kernel (§6.3) — production code

| File | Line(s) | Status | Severity |
|------|---------|--------|----------|
| `src/photon/density_estimator.h` | L59–L106 | **COMPLIANT** | — |
| `src/photon/surface_filter.h` | L35–L80 | **COMPLIANT** | — |
| `src/photon/kd_tree.h` | L98–L128 | **COMPLIANT** | — |
| `src/renderer/renderer.cpp` | L79–L113 | **COMPLIANT** | — |

**Detail:** Production `estimate_photon_density()` calls `grid.query_tangential()` with
tangential distance. `estimate_density_kdtree()` calls `tree.query_tangential()`. Both
use `tangential_box_kernel()` / `tangential_epanechnikov_kernel()` on `d_tan2`. The
3D query method is labelled "legacy" and is not used in the render path.

---

### A3. Tangential disk kernel (§6.3) — test ground truth functions

| File | Line(s) | Status | Severity |
|------|---------|--------|----------|
| `tests/test_ground_truth_validation.cpp` | L56–L76 | **NON-COMPLIANT** | **CRITICAL** |
| `tests/test_ground_truth_validation.cpp` | L77–L98 | **NON-COMPLIANT** | **CRITICAL** |
| `tests/test_ground_truth_validation.cpp` | L100–L150 | **NON-COMPLIANT** | **HIGH** |

**Detail:**  
- `brute_force_range_query()` (L56): Uses **3D Euclidean distance** (`dx*dx+dy*dy+dz*dz`),
  not tangential — the "ground truth" baseline itself violates §6.3.
- `brute_force_knn()` (L77): Also uses **3D Euclidean distance**.
- `brute_force_density_estimate()` (L100): Uses 3D Euclidean for the range check
  (`dist2 = dx*dx + dy*dy + dz*dz`), then applies surface filters. The primary range
  check does not use tangential distance.

**Impact:** Tests that compare brute-force (3D) against KD-tree (tangential) are
comparing **different distance metrics**. The tests will pass coincidentally for
flat surfaces but produce wrong results at corners and edges.

---

### A4. Surface consistency filter (§6.4)

| File | Line(s) | Status | Severity |
|------|---------|--------|----------|
| `src/photon/surface_filter.h` | L93–L127 | **COMPLIANT** | — |
| `src/photon/density_estimator.h` | L78–L96 | **COMPLIANT** | — |
| `src/renderer/renderer.cpp` | L96–L105 | **COMPLIANT** | — |

**Detail:** Production code correctly implements all 4 conditions:
1. Tangential distance < r² ✓
2. Plane distance < τ ✓
3. Normal compatibility: `dot(n_photon, n_query) > 0` ✓
4. Direction consistency: `dot(wi, n_query) > 0` ✓

---

### A5. Normal compatibility threshold inconsistency

| File | Line(s) | Status | Severity |
|------|---------|--------|----------|
| `tests/test_ground_truth_validation.cpp` | L139 | **NON-COMPLIANT** | **MEDIUM** |
| `tests/test_per_ray_validation.cpp` | L438 | **NON-COMPLIANT** | **MEDIUM** |
| `src/photon/density_estimator.h` | L84 | **COMPLIANT** (uses `> 0.0f`) | — |
| `src/photon/surface_filter.h` | L115 | **COMPLIANT** (uses `> 0.0f`) | — |

**Detail:** Guideline §6.4 condition 3 states `dot(n_photon, n_query) > 0`. The
production code correctly uses `> 0.0f`. However:
- `brute_force_density_estimate()` in test_ground_truth_validation.cpp uses `> 0.25f`
  (a stricter threshold that rejects more photons than the spec requires).
- `gather_with_bins_step()` in test_per_ray_validation.cpp uses `<= 0.25f` return
  (same `0.25` threshold, inconsistent with production code's `<= 0.0f`).

---

### A6. Geometric vs. shading normal usage (§15.1.2)

| File | Line(s) | Status | Severity |
|------|---------|--------|----------|
| `src/renderer/renderer.cpp` | L240, L244, L256, L271 | **NON-COMPLIANT** | **HIGH** |
| `tests/test_ground_truth.cpp` | L260–L300 | **NON-COMPLIANT** | **HIGH** |
| `tests/test_per_ray_validation.cpp` | L305, L330, L348 | **NON-COMPLIANT** | **HIGH** |
| `tests/test_pixel_comparison.cpp` | L260–L310 | **NON-COMPLIANT** | **HIGH** |
| `src/photon/emitter.h` | L240 | **COMPLIANT** (stores `hit.normal` = geometric) | — |
| `src/photon/surface_filter.h` | L98 comment | **COMPLIANT** (says "geometric normals") | — |

**Detail:** §15.1.2 mandates:
- Photons store **geometric normal** → emitter.h correctly stores `hit.normal` (geometric) ✓
- Query point uses **geometric normal** for tangential metric/plane filter ✗
- Shading normals only for BSDF evaluation ✗

The CPU renderer (`renderer.cpp`) and ALL test ground-truth path tracers use
`hit.shading_normal` for density estimation calls:
```cpp
estimate_photon_density(hit.position, hit.shading_normal, ...)
estimate_density_kdtree(hit.position, hit.shading_normal, ...)
```
This should be `hit.normal` (geometric) per §15.1.2. The `hit.shading_normal` is
the interpolated vertex normal, which can diverge from the face normal near edges,
breaking the plane distance filter.

---

### A7. Photon deposition at lightPathDepth ≥ 2 (§5.2.3)

| File | Line(s) | Status | Severity |
|------|---------|--------|----------|
| `src/photon/emitter.h` | L237–L245 | **WARNING** | **LOW** |

**Detail:** The emitter deposits photons with `if (bounce > 0)` for the global map
and `if (on_caustic_path && bounce > 0)` for caustics. Since `bounce` is 0-indexed
and the photon was emitted from the light (depth 1), `bounce > 0` corresponds to
`lightPathDepth >= 2`. However, the guideline notation uses "lightPathDepth" starting
from 1 at the light source. The implementation is consistent with the intent: direct
illumination photons are never stored. **Functionally compliant.**

---

### A8. SPPM tangential gather (§8)

| File | Line(s) | Status | Severity |
|------|---------|--------|----------|
| `src/photon/density_estimator.h` | L226–L278 | **COMPLIANT** | — |
| `src/renderer/renderer.cpp` | L505–L530 | **COMPLIANT** | — |
| `src/core/sppm.h` | L100–L140 | **COMPLIANT** | — |

**Detail:** `sppm_gather()` correctly uses `grid.query_tangential()` with tangential
distance. The progressive update uses tangential radius. The reconstruction denominator
uses `(π/2) r²` for Epanechnikov kernel. Matches §8 and §6.3.

---

### A9. NEE area-sampling Jacobian (§7.2)

| File | Line(s) | Status | Severity |
|------|---------|--------|----------|
| `tests/test_per_ray_validation.cpp` | L540–L580 | **COMPLIANT** | — |
| `tests/test_pixel_comparison.cpp` | L450–L495 | **COMPLIANT** | — |

**Detail:** NEE implementations correctly compute:
- `p_y_area = p_select / light_area`
- `p_wi = p_y_area * dist2 / cos_y`
- Contribution: `f * Le * cos_x / p_wi`

This matches the area-sampling form in §7.2.

---

### A10. ACES filmic tone mapping (§9)

| File | Line(s) | Status | Severity |
|------|---------|--------|----------|
| `src/renderer/renderer.h` | L153–L154 | **COMPLIANT** | — |

**Detail:** `FrameBuffer::tonemap()` uses `USE_ACES_TONEMAPPING` flag and calls
`spectrum_to_srgb_aces()`. Matches §9.

---

### A11. Cell-stratified bouncing (§5.3)

| File | Line(s) | Status | Severity |
|------|---------|--------|----------|
| `src/photon/emitter.h` | L117–L157 | **COMPLIANT** | — |
| `src/photon/emitter.h` | L248–L280 | **COMPLIANT** | — |

**Detail:** `CellStratifier` implements Fibonacci hemisphere strata with spatial hashing,
atomic counters per cell, and jitter. Applied to diffuse bounces via
`DEFAULT_PHOTON_BOUNCE_STRATA`. Matches §5.3.1 and §5.3.2.

---

### A12. Russian roulette (§5.2.2)

| File | Line(s) | Status | Severity |
|------|---------|--------|----------|
| `src/photon/emitter.h` | L301–L311 | **COMPLIANT** | — |

**Detail:** RR fires after `min_bounces_rr` with survival probability capped at
`rr_threshold`. Uses spectral max component. Matches §5.2.2.

---

### A13. Kernel normalization (§15.1.3)

| File | Line(s) | Status | Severity |
|------|---------|--------|----------|
| `src/photon/surface_filter.h` | L162–L170 | **COMPLIANT** | — |

**Detail:** Box = `π r²`, Epanechnikov = `(π/2) r²`. Matches §15.1.3 table exactly.

---

### A14. Test ground truth uses `bsdf::evaluate()` vs `bsdf::evaluate_diffuse()`

| File | Line(s) | Status | Severity |
|------|---------|--------|----------|
| `tests/test_per_ray_validation.cpp` | L370 | Uses `evaluate()` (full BSDF) | **WARNING** |
| `tests/test_pixel_comparison.cpp` | L297 | Uses `evaluate()` (full BSDF) | **WARNING** |
| `src/photon/density_estimator.h` | L94 | Uses `evaluate_diffuse()` | — |
| `src/renderer/renderer.cpp` | L107 | Uses `evaluate_diffuse()` | — |

**Detail:** Production code uses `bsdf::evaluate_diffuse()` for photon gather (correct:
density estimation at a diffuse hit only evaluates the diffuse lobe). However, several
test ground truth functions use `bsdf::evaluate()` (full BSDF including specular/glossy
lobes). For purely diffuse materials these are identical, but for glossy materials this
could cause test vs. production divergence.

---

### A15. Photon `wi` sign convention (§15.1.1)

| File | Line(s) | Status | Severity |
|------|---------|--------|----------|
| `src/photon/emitter.h` | L239 | **COMPLIANT** | — |

**Detail:** Photon `wi` is stored as `ray.direction * (-1.f)` — the incoming direction
toward the surface, consistent with §15.1.1. The filter's condition 4 checks
`dot(wi, n) > 0` which correctly validates this convention.

---

## B) Test Coverage per §15 Required Categories

### B1. KD-Tree

| Required | Status | File | Notes |
|----------|--------|------|-------|
| Build | ✓ PRESENT | `test_kd_tree.cpp` | BuildEmpty, BuildSinglePhoton, BuildMany |
| Range query | ✓ PRESENT | `test_kd_tree.cpp` | RangeQueryMatchesBruteForce, etc. |
| k-NN | ✓ PRESENT | `test_kd_tree.cpp` | KNNMatchesBruteForce, KNNAdaptiveRadius |
| Empty | ✓ PRESENT | `test_kd_tree.cpp` | BuildEmpty, RangeQueryEmpty, KNNEmpty |
| Single photon | ✓ PRESENT | `test_kd_tree.cpp` | BuildSinglePhoton, RangeQuerySingle |
| Boundary | ✓ PRESENT | `test_kd_tree.cpp` | BoundaryPhotons |

**Status: COMPLIANT** ✓  
**Note:** KD-tree tests in test_kd_tree.cpp use 3D Euclidean distance only, but
tangential KD-tree tests exist in test_tangential_gather.cpp.

---

### B2. KDTree vs HashGrid

| Required | Status | File | Notes |
|----------|--------|------|-------|
| Same query results | ✓ PRESENT | `test_kd_tree.cpp` L317 | QueryMatchesHashGrid |
| Tangential results | ✓ PRESENT | `test_tangential_gather.cpp` | CPUvsGPU range/kNN cross-validation (5 tests) |
| | ✓ PRESENT | `test_integration.cpp` L470 | TangentialKDTreeMatchesHashGrid |

**Status: COMPLIANT** ✓

---

### B3. Tangential kernel

| Required | Status | File | Notes |
|----------|--------|------|-------|
| Tangential distance analytic | ✓ PRESENT | `test_tangential_gather.cpp` | ReturnedDistancesAreCorrect |
| Planar geometry tests | ✓ PRESENT | `test_tangential_gather.cpp` | KDTreeRangeQueryWall, Corner, etc. |
| Legacy 3D comparison | ✓ PRESENT | `test_tangential_gather.cpp` | TangentialFindsCoplanarNotMissed, Legacy3DQueryStillWorks |

**Status: COMPLIANT** ✓

---

### B4. Surface filter

| Required | Status | File | Notes |
|----------|--------|------|-------|
| Cross-wall rejection | ✓ PRESENT | `test_surface_filter.cpp` | OppositeNormalRejected |
| Same-surface acceptance | ✓ PRESENT | `test_surface_filter.cpp` | SameSurfaceAccepted |
| All 4 conditions | ✓ PRESENT | `test_surface_filter.cpp` | TangentialRadiusReject, PlaneDistanceReject, OppositeNormal, WrongDirection |

**Status: COMPLIANT** ✓

---

### B5. Shell expansion k-NN

| Required | Status | File | Notes |
|----------|--------|------|-------|
| GPU k-NN matches CPU k-NN | ✓ PRESENT | `test_tangential_gather.cpp` | ShellExpansionKNNWall, ShellExpansionKNNRandom3D |
| | ✓ PRESENT | `test_integration.cpp` L662 | ShellExpansionMatchesKDTreeKNN |

**Status: COMPLIANT** ✓

---

### B6. CPU Renderer (first-hit)

| Required | Status | File | Notes |
|----------|--------|------|-------|
| NEE correctness | ✓ PRESENT | `test_per_ray_validation.cpp` | AggregateNEEConverges |
| Photon gather correctness | ✓ PRESENT | `test_per_ray_validation.cpp` | AggregatePhotonDensityConverges, FirstHitPhotonDensityAgreement |
| Combined | ✓ PRESENT | `test_per_ray_validation.cpp` | AggregateCombinedConverges |

**Status: COMPLIANT** ✓

---

### B7. Integration (CPU↔GPU) — All tests from §12.3

| Required Test | Status | Notes |
|---------------|--------|-------|
| `CPU_GPU.DirectLightingMatch` | **MISSING** | Not in test_integration.cpp |
| `CPU_GPU.PhotonIndirectMatch` | **MISSING** | Not in test_integration.cpp |
| `CPU_GPU.CombinedMatch` | **MISSING** | Not in test_integration.cpp |
| `CPU_GPU.SPPMConvergence` | **MISSING** | Not in test_integration.cpp |
| `CPU_GPU.CausticMapMatch` | **MISSING** | Not in test_integration.cpp |
| `CPU_GPU.AdaptiveRadiusMatch` | **MISSING** | Not in test_integration.cpp |
| `CPU_GPU.EnergyConservation` | **MISSING** | Not in test_integration.cpp |
| `CPU_GPU.NoNegativeValues` | **MISSING** | Not in test_integration.cpp |
| `CPU_GPU.SpectralBinIsolation` | **MISSING** | Not in test_integration.cpp |
| `CPU_GPU.DifferenceImage` | **MISSING** | Not in test_integration.cpp |

**Status: NON-COMPLIANT — 0/10 tests present** ⛔  
**Severity: HIGH**

**Note:** test_integration.cpp contains v2.1-specific tests (tangential gather, surface
consistency, decorrelated photons, shell expansion) but does NOT contain the 10 CPU↔GPU
comparison tests listed in §12.3. The existing tests compare CPU algorithms (KD-tree vs
hash grid), not CPU vs GPU renders.

---

### B8. Adaptive radius

| Required | Status | File | Notes |
|----------|--------|------|-------|
| k-NN tangential radius matches expected density | ✓ PRESENT | `test_kd_tree.cpp` | KNNAdaptiveRadius |
| | ✓ PRESENT | `test_tangential_gather.cpp` | ShellExpansionKNNWall/Random3D |

**Status: COMPLIANT** ✓

---

### B9. SPPM + KD-tree

| Required | Status | File | Notes |
|----------|--------|------|-------|
| Progressive convergence identical to hash grid | **MISSING** | — | No SPPM-specific tests exist anywhere |

**Status: NON-COMPLIANT** ⛔  
**Severity: MEDIUM**

---

## C) Test Correctness — Wrong or Outdated Behavior

### C1. Brute-force ground truth uses 3D Euclidean (should use tangential)

| File | Line(s) | Status | Severity |
|------|---------|--------|----------|
| `tests/test_ground_truth_validation.cpp` | L56–L98 | **INCORRECT** | **CRITICAL** |

**Detail:** `brute_force_range_query()` and `brute_force_knn()` use 3D Euclidean distance
as the "ground truth" baseline, but v2.1 mandates tangential distance for all gather.
Tests comparing KD-tree (tangential) against brute-force (3D) are comparing different
metrics, making the "ground truth" itself wrong per v2.1.

---

### C2. Normal compatibility threshold 0.25 in tests vs. 0.0 in production

| File | Line(s) | Status | Severity |
|------|---------|--------|----------|
| `tests/test_ground_truth_validation.cpp` | L139 | **INCORRECT** | **MEDIUM** |
| `tests/test_per_ray_validation.cpp` | L438 | **INCORRECT** | **MEDIUM** |

**Detail:** These test functions use `dot(photon_n, normal) <= 0.25f` as the normal
compatibility check, but §6.4 and the production code use `<= 0.0f`. The test
"optimized" path tracers are stricter than the spec, rejecting photons the production
code would accept.

---

### C3. Obsolete includes — cell_bin_grid.h, MIS headers

| File | Line(s) | Status | Severity |
|------|---------|--------|----------|
| `tests/test_main.cpp` | L14 | **OUTDATED** | **LOW** |
| `tests/test_main.cpp` | L559–L596 | **OUTDATED** | **LOW** |
| `tests/test_medium.cpp` | L36 | **OUTDATED** | **LOW** |

**Detail:** `test_main.cpp` includes `core/cell_bin_grid.h` and `renderer/mis.h`.
`test_medium.cpp` includes `renderer/mis.h`. §16.3 states cell-bin grid was a "removed
approach" and MIS weights are disabled in v2. The tests for `MIS`, `CellBinGrid`, and
`MISWeights` test suites in test_main.cpp exercise removed/deprecated features.
Per §Q7 ("remove obsolete tests"), these should be flagged for removal.

---

### C4. test_medium.cpp includes volume render tests that are self-skipped

| File | Line(s) | Status | Severity |
|------|---------|--------|----------|
| `tests/test_medium.cpp` | L594 | **OUTDATED** | **LOW** |
| `tests/test_medium.cpp` | L685 | **OUTDATED** | **LOW** |

**Detail:** `HigherDensity_ReducesSurfaceRadiance` and `VolumePhotonsContributeInScatter`
contain `GTEST_SKIP()` with comment "v2: CPU renderer does not implement volume transport".
These tests are permanently disabled and serve no validation purpose.

---

### C5. test_pixel_comparison.cpp photon lobe direction uses 3D query

| File | Line(s) | Status | Severity |
|------|---------|--------|----------|
| `tests/test_pixel_comparison.cpp` | L666 | **INCORRECT** | **MEDIUM** |

**Detail:** `trace_photon_lobe()` uses `ds.grid.query()` (3D Euclidean range query)
instead of `ds.grid.query_tangential()` for photon direction gathering. The directional
comparison test (PhotonLobeVsActualDirections) thus gathers with a non-v2.1-compliant
metric.

---

### C6. test_per_ray_validation.cpp SpectralBinsNeverMix uses 3D query

| File | Line(s) | Status | Severity |
|------|---------|--------|----------|
| `tests/test_per_ray_validation.cpp` | L1290 | **INCORRECT** | **MEDIUM** |

**Detail:** The `SpectralBinsNeverMix` test manually gathers photons using
`ds.grid.query()` (3D Euclidean) rather than `ds.grid.query_tangential()`. While this
doesn't invalidate the spectral isolation check per se (it's checking a different property),
the gather set is different from production, potentially hiding mismatches.

---

### C7. test_pixel_comparison.cpp `gather_with_local_bins` uses single-wavelength accumulation

| File | Line(s) | Status | Severity |
|------|---------|--------|----------|
| `tests/test_pixel_comparison.cpp` | L381–L385 | **WARNING** | **LOW** |

**Detail:** In `gather_with_local_bins()`, the density contribution line reads:
```cpp
L.value[bin] += flux * inv_N * f.value[bin] * inv_area;
```
This accumulates into a single wavelength bin (`photons.lambda_bin[idx]`), treating
photons as single-wavelength carriers. But the production code
(`estimate_photon_density()`) accumulates into ALL bins via `photons.get_flux(idx)`,
which returns the full spectral flux. This discrepancy means the "optimized" test path
tracer computes a different radiance than production.

---

## D) Missing Tests per Guideline

### D1. CPU↔GPU Integration Tests (§12.3) — ALL 10 MISSING

| Test Name | Guideline Reference |
|-----------|-------------------|
| `CPU_GPU.DirectLightingMatch` | §12.3 — PSNR > 40 dB |
| `CPU_GPU.PhotonIndirectMatch` | §12.3 — PSNR > 30 dB |
| `CPU_GPU.CombinedMatch` | §12.3 — PSNR > 30 dB |
| `CPU_GPU.SPPMConvergence` | §12.3 — PSNR > 25 dB |
| `CPU_GPU.CausticMapMatch` | §12.3 — PSNR > 25 dB |
| `CPU_GPU.AdaptiveRadiusMatch` | §12.3 — PSNR > 25 dB |
| `CPU_GPU.EnergyConservation` | §12.3 — ratio ∈ [0.95, 1.05] |
| `CPU_GPU.NoNegativeValues` | §12.3 — max(min_pixel) ≥ 0 |
| `CPU_GPU.SpectralBinIsolation` | §12.3 — exact match |
| `CPU_GPU.DifferenceImage` | §12.3 — always pass (logs) |

**Severity: HIGH**

---

### D2. SPPM + KD-tree Convergence Test (§15)

No test verifies that SPPM progressive radius shrinking produces identical results
whether the spatial index is KD-tree or hash grid. The `render_sppm()` function in
renderer.cpp always builds both, but no test compares them.

**Severity: MEDIUM**

---

## E) Positive Findings (Compliant Areas)

| Area | Files | Notes |
|------|-------|-------|
| Tangential gather implementation | `density_estimator.h`, `kd_tree.h`, `surface_filter.h` | Production code fully v2.1-compliant |
| Tangential gather tests | `test_tangential_gather.cpp` | Comprehensive: 16 tests covering CPU/GPU cross-validation, wall/corner geometry, shell expansion |
| Surface filter tests | `test_surface_filter.cpp` | All 4 conditions tested, kernel functions verified |
| First-hit architecture | `renderer.cpp`, all test path tracers | Consistent `break` at first diffuse hit |
| Specular chain | `renderer.cpp` L226–L242 | Correctly follows specular bounces with throughput accumulation |
| SPPM implementation | `sppm.h`, `renderer.cpp` L460–L660 | Uses tangential gather, Epanechnikov kernel, correct reconstruction formula |
| Cell-stratified bouncing | `emitter.h` | Fibonacci hemisphere strata with spatial hash |
| Photon wi convention | `emitter.h` L239 | Correctly stored as incoming direction |
| Photon geometric normal storage | `emitter.h` L240 | `hit.normal` (geometric) stored per photon |
| Kernel normalization | `surface_filter.h` | Box = πr², Epa = (π/2)r² — matches §15.1.3 |
| Per-ray validation | `test_per_ray_validation.cpp` | 14 tests covering physical validity, energy conservation, decomposition |
| Pixel comparison | `test_pixel_comparison.cpp` | 12 tests with RMSE/PSNR metrics and image output |

---

## F) Priority Remediation Order

1. **CRITICAL:** Update `brute_force_range_query()` and `brute_force_knn()` in
   test_ground_truth_validation.cpp to use tangential distance (A3)
2. **HIGH:** Fix all test path tracers and renderer.cpp to pass `hit.normal` (geometric)
   instead of `hit.shading_normal` to density estimation functions (A6)
3. **HIGH:** Implement the 10 CPU↔GPU integration tests from §12.3 (D1)
4. **MEDIUM:** Fix normal compatibility threshold from `0.25f` to `0.0f` in test
   brute-force and optimized gatherers (A5, C2)
5. **MEDIUM:** Add SPPM convergence test comparing KD-tree vs hash grid (D2)
6. **MEDIUM:** Fix `trace_photon_lobe()` and `SpectralBinsNeverMix` to use
   `query_tangential()` instead of `query()` (C5, C6)
7. **LOW:** Remove or update obsolete MIS/CellBinGrid tests in test_main.cpp (C3)
8. **LOW:** Remove or unskip permanently-skipped volume tests in test_medium.cpp (C4)
9. **LOW:** Reconcile `bsdf::evaluate()` vs `bsdf::evaluate_diffuse()` in test ground
   truth functions (A14)
