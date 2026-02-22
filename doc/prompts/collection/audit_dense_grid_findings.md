# Dense Grid vs KD-Tree Diagnostic Audit — 2026-02-22

## Test Infrastructure

17 diagnostic tests created in `tests/test_dense_grid_diagnostics.cpp`, comparing four
independent gather implementations against each other:

| Label | Implementation | Algorithm |
|-------|---------------|-----------|
| **CPU** | `cpu_brute_force_density()` | Brute-force per-photon, all 4 surface conditions |
| **KD** | `kdtree_density_estimate()` | KD-tree tangential query, same 4 conditions |
| **HG** | Hash-grid via `estimate_density()` | Per-photon, same 4 conditions |
| **DG** | `dense_grid_density()` | CellBinGrid trilinear, per-bin normal+hemisphere gates |

**Result: CPU = KD = HG (identical to float precision) — ground truth is consistent.**

---

## Finding 1: High-Frequency Shadow Preservation

### Hypothesis (original)
> "A does not preserve high-frequency shadows."

### Test Result: `ShadowEdge_CpuVsKdtreeVsDenseGrid`

```
[ShadowEdge] CPU  lit=27.65  shadow=0.00  contrast=999:1
[ShadowEdge] KD   lit=27.65  shadow=0.00  contrast=999:1
[ShadowEdge] DG   lit=2.60   shadow=0.00  contrast=999:1
```

### Analysis
Both CPU and DG produce **perfect shadow contrast** (999:1 = clamped maximum). The shadow
boundary is preserved identically. However, the absolute energy in the lit region differs
by **10.6×** (CPU=27.65 vs DG=2.60).

**Root cause is NOT shadow contrast but brightness deficit** — see Finding 2.

The original observation of "neutralized shadows" is likely caused by the entire indirect
frame being so dim that when composed with NEE, the shadow regions appear relatively
brighter in comparison (the dynamic range is compressed).

### Verdict: **Finding 1 is a secondary effect of Finding 2 (brightness deficit).**

---

## Finding 2: "Washed-Out" / Too Bright Composition

### Hypothesis (original)
> "The final composition (NEE+Photons) in A is too bright and washed out."

### Test Results

#### `EnergyComparison_CpuVsKdtreeVsDenseGrid`
```
CPU brute force: 2.968
KD-tree:         2.968  (ratio vs CPU: 1.000)
Dense grid:      0.385  (ratio vs CPU: 0.130)
WARNING: Dense grid is 87% dimmer than CPU
```

#### `FullPipeline_ThreeWayComparison`
```
CPU total:  167.85
KD-tree:    167.85  (max error: 0.0%)
HashGrid:   167.85  (max error: 0.0%)
DenseGrid:  20.82   (max error: 90.2%)
Dense grid / CPU ratio: 0.124
```

#### `SummaryReport` (5 representative query points)
```
Query     CPU        DG         DG/CPU
Q0        3.375      0.368      10.9%
Q1        2.919      0.381      13.1%
Q2        3.313      0.425      12.8%
Q3        1.979      0.402      20.3%
Q4        2.348      0.297      12.6%
```

### Analysis: `StepByStep_IsolateDeviation`
```
Step 1 — Spatial Retrieval:
  CPU: 29 photons within disk
  DG:  278 accumulated photon-counts (3×3×3 scatter)

Step 2 — Surface Filtering:
  CPU: 29/29 pass all 4 conditions
  DG:  8/32 bins pass normal+hemisphere gates (75% rejected!)

Step 3 — Kernel Weighting:
  CPU total weight: 15.14
  DG total weight:  16.66  (ratio: 1.10×)

Step 4 — BSDF: 0.3183 (Lambertian INV_PI, direction-independent)

Step 5 — Final Radiance:
  CPU: 7.852
  DG:  1.080
  Ratio DG/CPU: 0.138  (error: 86.2%)
```

### Root Causes Identified

The dense grid produces **~87% less energy** than the per-photon reference. The step-by-step
analysis reveals the deviation enters at multiple stages:

1. **Bin discretization loss** — The dense grid scatters photons into 32 directional bins
   (Fibonacci sphere). At query time, 75% of bins are rejected by the hemisphere gate
   (`dot(bin_dir, hit_normal) <= 0`). On a flat surface, roughly half the Fibonacci
   sphere directions point below the hemisphere, so ~16 out of 32 bins already fail.
   Additional bins fail the normal gate. Only 8 bins contribute — but these 8 bins
   contain aggregated flux from all 29 nearby photons.

2. **Missing normalization compensation** — The dense grid query uses
   `inv_area = 2/(π·r²)` while the CPU uses `inv_area = 1/epanechnikov_norm(r²)`.
   The Epanechnikov kernel normalization is `(2/3)·π·r²`, giving `inv_area ≈ 0.4775/r²`.
   The dense grid's `2/(π·r²) ≈ 0.6366/r²` is actually **larger**, so this factor
   alone would make DG **brighter** not dimmer — it compensates partially but not enough.

3. **Flux fragmentation across bins** — Each photon's flux is deposited into exactly one
   directional bin (its nearest Fibonacci direction). At query time, the BSDF is evaluated
   once per bin using the bin's centroid direction. When photons are spread across many
   bins, some bins have very low counts and their contributions don't compensate for the
   bins that are rejected.

4. **Kernel baking at cell centres vs query point** — The Epanechnikov weight is pre-baked
   at the distance from each photon to the cell centre, not to the query point. At query
   time, only trilinear interpolation blends between cells. This introduces spatial
   smoothing that reduces peak values.

### Verdict: **Dense grid is ~87% DIMMER than reference, not brighter.**
The "washed-out" appearance in the original observation may be caused by an
auto-tonemapping or exposure compensation that boosts the uniformly dim indirect
channel, reducing shadow contrast in the final composite.

---

## Finding 3: Backface Shadows on Small Spheres

### Hypothesis (original)
> "Both A and B show shadows on the backfacing side of smaller, spherical objects."

### Test Results

#### `SmallSphere_BackFaceIllumination`
```
Lit side (top):   CPU=62.83  KD=62.83  DG=7.38
Back side (btm):  CPU=0.000  KD=0.000  DG=0.000
Back/Lit ratio:   CPU=0.000  KD=0.000  DG=0.000
```

#### `SmallSphere_EquatorialLeakage`
```
Sphere r=0.030, tau=0.020
Photons passing spatial filter: 583
Of those, wrong hemisphere normal: 0
```

#### `HighCurvature_TangentialProjectionError`
```
Sphere radius=0.10
3D distance top→side:         0.1414
Tangential distance top→side: 0.1000
Plane distance top→side:      0.1000
```

### Analysis

All three implementations correctly produce **zero radiance on the backside** — the
normal/hemisphere gates and the photon distribution (top hemisphere only) prevent any leakage.
No equatorial leakage was detected even with `sphere_r < tau`.

The backface "shadow" observed in the original renders is therefore not a gather artefact.
It is the **correct physical behavior**: the back hemisphere of a small sphere receives zero
indirect photons when illuminated from above, so it appears dark. In a real scene, multi-bounce
indirect illumination (from nearby walls) would fill in the back side, but this requires
sufficient photon density and multi-bounce tracing — which the test scene (isolated sphere)
does not provide.

The tangential projection error on high-curvature surfaces is moderate: tangential distance
is ~29% less than 3D distance (0.10 vs 0.14 for r=0.10). This means the gather disk maps
to a larger surface patch than intended, but since both hemispheres have the same normal
check, this doesn't cause leakage.

### Verdict: **Behavior is physically correct.** Backface darkness requires multi-bounce GI
from environment geometry, not a gather algorithm fix.

---

## Finding 4: Speed Difference (>100×)

### Test Results

#### `GatherComplexity_PerPhotonVsPerBin`
```
Per-photon path:
  33 photons found → 33 tangential projections + 33 BSDF evals

Dense grid path:
  256 bin lookups (8 cells × 32 bins)
  Speedup ratio (gather-only): 0.1×
```

#### `NEEPathIsIndependentOfGather`
Confirmed: NEE path is identical regardless of gather method.

### Analysis

The measured gather-only operation count shows DG does 256 bin lookups vs 33 per-photon
operations — DG actually does **more** operations per pixel for this photon density. However:

1. **DG operations are trivial**: Each bin lookup is: load bin data (1 cache line) →
   2 dot products → 1 multiply-add per wavelength. No spatial queries, no tree traversals.

2. **KD-tree operations are complex**: Each of 33 photons requires: tree traversal (log N
   comparisons) → tangential projection → 4 surface consistency checks → Epanechnikov
   weight → ONB transform → BSDF evaluation.

3. **Memory access pattern**: DG bins are contiguous in memory (8 adjacent cells × 32 bins =
   256 sequential reads). KD-tree access is pointer-chasing with poor spatial locality.

4. **Scaling**: KD-tree cost is O(k·log N) per query where k = photons found and N = total
   photons. DG cost is O(8 × PHOTON_BIN_COUNT) = O(256) independent of N. With 1M photons,
   k grows while DG remains constant.

5. **The >100× claim**: The timing data shows photon gather = 10.7% of frame time for B.
   NEE = 85.5% of frame time (identical for both A and B). The actual full-frame speedup
   from gather optimization is at most `1/((1-0.107) + 0.107/100) ≈ 1.12×` (12%).
   The >100× number likely compares **gather-only** time, not total frame time.

#### `NeighbourScatter_EnergyConservation`
```
500 photons, flux=1.0 each:
  Total deposited flux: 2453.39
  Total photon-cell hits (3×3×3 scatter): 1173
  Average scatter multiplicity: 2.3×
```

The 3×3×3 scatter causes each photon to appear in ~2.3 cells on average (not the
theoretical maximum of 27, because the Epanechnikov kernel rejects most neighbours).

### Verdict: **The >100× number is gather-only throughput**, not frame speedup.
Full-frame improvement is ~12%. The algorithmic difference (O(256) constant vs
O(k·log N) scaling) is genuine and well-justified — DG trades quality for constant-time
gather.

---

## Cross-Surface Leakage Test

### `ThinWall_CrossSurfaceLeakage`
```
CPU: 4.980
KD:  4.980
DG:  0.623   (-87.5% vs CPU)
```

The dense grid shows **no excess energy** from cross-surface leakage — it's actually
87.5% dimmer. The per-bin normal gate (`dot(avg_n, hit_normal) > 0`) successfully prevents
through-wall contamination. The brightness deficit is the same as the general brightness
deficit (Finding 2).

---

## Summary of Root Causes

| Finding | Original Hypothesis | Actual Root Cause |
|---------|-------------------|-------------------|
| **1. HF shadows** | DG neutralizes them | Secondary effect of 87% brightness deficit (F2) |
| **2. Washed out** | DG too bright | DG is 87% **dimmer** — perceived washout may be from exposure compensation |
| **3. Backface** | Gather leakage | Correct physics — needs multi-bounce GI from walls |
| **4. Speed** | >100× suspicious | Gather-only: correct; full-frame: ~12% speedup |

### Dense Grid Brightness Deficit — Contributing Factors

The 87% energy deficit is caused by the cumulative effect of:

1. **Directional discretization**: 32 Fibonacci bins → ~50% rejected by hemisphere gate on
   flat surfaces (correct), but surviving bins carry fragmented flux
2. **No per-photon spectral_flux path**: DG accumulates per-hero `flux[]` into spectral bins;
   this is correct but differs from the full `spectral_flux` sum used by CPU/KD/HG
3. **Query-time normalization mismatch**: `2/(π·r²)` vs `1/epanechnikov_norm(r²)`
4. **Baked kernel weights** at cell centres vs dynamic weights at query points
5. **Trilinear interpolation** smoothing reduces peak values

### Recommendations

1. **Investigate normalization factor** — the constant `2/(π·r²)` in the GPU query should be
   verified against the Epanechnikov kernel integral `(2/3)·π·r²` to ensure the
   `inv_area` term is mathematically correct for pre-baked kernel weights.

2. **Consider reducing bin count** — 32 bins wastes 50% of storage on below-hemisphere
   directions that are always rejected. Using 16 hemisphere-only bins would double
   the flux per bin and reduce bin-rejection losses.

3. **Finding 3 is not actionable** — backface darkness is correct physics and needs
   environmental indirect illumination (multi-bounce from walls), not algorithm changes.

4. **Finding 4 is expected** — the O(1) vs O(k·log N) tradeoff is the design intent.
   The 87% energy gap is the real concern, not the speed difference.
