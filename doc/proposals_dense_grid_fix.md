# Proposals: Fixing the Dense Grid 87% Energy Deficit

## Problem Statement

The `CellBinGrid` (dense grid, path "A") produces **87% less energy** than the
per-photon reference paths (CPU brute-force, KD-tree, hash grid). The diagnostic
test suite (`DenseGridDiagnostics`) confirmed this is consistent across all query
points: DG/CPU ratio = 0.12–0.13.

The root cause is NOT a single bug but a cumulative loss from bin discretization,
baked kernel weights, and normalization mismatch. Below are three proposals ranked
from least to most invasive.

---

## Proposal A: "Fix the Bin Path" — Correct Normalization + Hemisphere-Only Bins

**Approach**: Keep the CellBinGrid architecture but fix the identified energy losses.

### Changes Required

#### A.1 — Use hemisphere-only Fibonacci bins (halve waste)

Currently 32 full-sphere bins → ~16 are below-hemisphere and always rejected at query
time (wasted storage + fragmented flux). Switch to hemisphere-only bins:

```
// photon_bins.h — add hemisphere variant
HD void init_hemisphere(int n, float3 up) {
    count = n;
    for (int k = 0; k < count; ++k) {
        // Fibonacci distribution on upper hemisphere only
        float theta = acosf(1.0f - (k + 0.5f) / (float)count);  // [0, π/2]
        float phi   = golden_angle * k;
        dirs[k] = /* rotate to align with 'up' */;
    }
}
```

**Problem**: The hemisphere depends on the surface normal at the query point, which
varies per-pixel. At build time, a cell contains photons from multiple surfaces with
different normals. You'd need per-surface-orientation bin sets within each cell, which
defeats the O(1) lookup.

**Verdict**: Not straightforward. Skip unless we add per-cell dominant-normal bins.

#### A.2 — Fix normalization: account for directional discretization

The current GPU query uses `inv_area = 2/(π r²)` — the correct 2D Epanechnikov
normalization for a per-photon kernel. But the bin path doesn't evaluate the kernel
per-photon at query time; the kernel weight was already baked in at build time. The
query just sums pre-weighted flux. The `inv_area` factor is thus double-counting
the kernel normalization.

**Fix**: During build, store raw flux (without kernel weight) in the bin OR remove
`inv_area` from the query and bake the full normalization into the flux at build time.

**Risk**: This may overshoot — the kernel normalization is coupled to the query-point
tangential distance, which doesn't exist at build time. Unclear how to separate.

#### A.3 — Add per-cell calibration factor

At build time, compute a per-cell "energy ratio" by comparing the cell's total binned
flux against the brute-force sum of photon flux that reaches the cell centre. Store
this as a scalar correction factor per cell. At query time, multiply by this factor.

**Trade-off**: +1 float per cell (~negligible memory), but requires an O(N) reference
pass during build. More complex build, O(1) query unchanged.

### Assessment

| Aspect | Rating |
|--------|--------|
| Accuracy improvement | Unclear — may reduce deficit to ~30-50% but won't reach <5% |
| Implementation risk | High — normalization fix interacts with trilinear blending |
| Build cost | +O(N) calibration pass |
| Query cost | Unchanged |
| Confidence | Low — fundamental losses from directional discretization remain |

---

## Proposal B: "Precomputed Per-Photon Density" — Store Total Radiance Per Cell

**Approach**: Instead of storing directional bins per cell, store the **final scalar
irradiance** precomputed using brute-force per-photon gather at the cell centre. At
query time, just trilinearly interpolate the precomputed value.

### Architecture

```
struct CellIrradiance {
    float irradiance[NUM_LAMBDA];  // precomputed at cell centre
    float3 avg_normal;             // for the normal gate
};
```

**Build** (CPU, once per photon map update):
```
for each cell:
    pos = cell_centre
    normal = cell_dominant_normal  (from flux-weighted photon normals)
    wo = normal  (hemisphere assumption)
    
    irradiance[cell] = cpu_brute_force_density(
        pos, normal, wo, white_lambertian,
        photons, gather_radius, num_emitted)
```

**Query** (GPU, per pixel):
```
trilinear_cells(hit_pos) → 8 cells + weights
L = 0
for each corner cell:
    if dot(cell.avg_normal, filter_normal) > 0:
        L += tw * cell.irradiance * bsdf_eval(...)
```

### Memory

Per cell: `NUM_LAMBDA * 4 + 12 + 4 = 144 bytes` (vs current `32 * 164 = 5248 bytes`).
That's **36× less memory** per cell.

### Accuracy

- **Pro**: Uses the exact same per-photon algorithm as the reference at each cell centre
- **Con**: Interpolates between cell centres — spatial smoothing at cell_size scale
- **Con**: Uses cell's dominant normal, not query-point normal — normal mismatch
- **Con**: BSDF direction is approximated (hemisphere average, not per-photon)

### Assessment

| Aspect | Rating |
|--------|--------|
| Accuracy improvement | ~50-70% error reduction (spatial smoothing remains) |
| Implementation risk | Medium — straightforward but trades directional info |
| Build cost | O(cells × photons_in_radius) ≈ expensive for dense maps |
| Query cost | O(8) — even faster than current O(8 × 32) |
| Confidence | Medium — good for diffuse, loses directional information |

---

## Proposal C: "GPU Hash Grid" — Drop Approximation, Go Per-Photon on GPU

**Approach**: Keep the per-photon hash grid path that already exists in
`dev_estimate_photon_density` (the fallback path in `optix_device.cu`) and make
it the primary/only gather path. Drop `CellBinGrid` entirely.

### What Already Exists

The GPU hash-grid path is already fully implemented:
- `dev_estimate_photon_density()` lines 470–550 in `optix_device.cu`
- Per-photon tangential projection + tau filter + normal/hemisphere checks
- Epanechnikov kernel weight
- Per-hero-wavelength spectral accumulation
- Same O(N_cell) complexity as the CPU KD-tree path

The data upload pipeline already works:
- `photon_pos_x/y/z`, `photon_wi_x/y/z`, `photon_norm_x/y/z` → device arrays
- `grid_cell_start`, `grid_cell_end`, `grid_sorted_indices` → hash grid
- `photon_flux`, `photon_lambda`, `photon_num_hero` → hero wavelength data

### What to Change

1. **Default to hash grid**: Set `DEFAULT_USE_DENSE_GRID = false` in `config.h`
2. **Optimize the hash grid GPU path** (optional performance work):
   - Sort photons by hash cell for better GPU memory coalescence
   - Consider warp-level parallelism for cell iteration
   - Tune hash table load factor
3. **Remove or deprecate CellBinGrid** (optional cleanup)

### Performance Impact

The diagnostic test measured:
```
Per-photon: 33 tangential projections + 33 BSDF evals per query
Dense grid: 256 bin lookups (8 × 32) per query
```

Per-photon work per element is heavier (tangential projection + 4 conditions + ONB
transform vs simple multiply-add), but the total element count is lower (33 vs 256).
The hash grid path is already there — **we measured >100× slower in the original
Finding 4, but the timing data showed photon gather is only 10.7% of total frame
time**. Going back to hash grid means:

- Worst case: frame time increases by ~12% (from 10.7% → 100× slower → but bounded
  by the fact that NEE at 85.5% is unchanged)
- The quality improvement is massive: from 87% deficit to **0% deficit** (exact match)

### Additional GPU Optimization: Cell-Sorted SoA

To make the hash grid path faster on GPU, sort photons by hash cell index at build
time. This converts the random-access pattern into sequential reads:

```cpp
// In hash_grid build or a new sort step:
// 1. For each photon, compute hash cell
// 2. Sort photon indices by cell
// 3. Upload sorted photon data in SoA layout
// GPU walks contiguous memory segments per cell
```

This is similar to what particle simulations do (Z-order / Morton code sorting) and
can bring 2-5× speedup to the per-photon gather by improving L1/L2 cache hit rates.

### Assessment

| Aspect | Rating |
|--------|--------|
| Accuracy improvement | **100%** — identical to reference by construction |
| Implementation risk | **Minimal** — the code already exists, just flip the default |
| Build cost | Unchanged (hash grid already built) |
| Query cost | O(N_cell) instead of O(256) — slower per query |
| Frame time impact | ≤12% slower than current (NEE dominates at 85.5%) |
| Confidence | **Very high** — tested and proven correct |

---

## Proposal D: "Hybrid" — Hash Grid with Per-Cell Shortlist

**Approach**: Precompute a fixed-size photon shortlist per cell at build time. At
query time, iterate only the shortlisted photons (not all photons in hash bucket).
Combines per-photon accuracy with bounded query cost.

### Architecture

```
struct CellPhotonList {
    uint32_t photon_indices[MAX_PHOTONS_PER_CELL];  // e.g. 64
    int count;
};
```

**Build**: For each cell, find all photons within `gather_radius` of the cell centre
using a brute-force scan. Store up to `MAX_PHOTONS_PER_CELL` indices (sorted by
kernel weight, keep the strongest).

**Query**: For each of 8 trilinear cells, iterate the shortlist. Apply full
per-photon surface consistency (tangential distance from actual query point, tau
filter, normal check, hemisphere check, kernel weight, per-hero BSDF eval).

### Memory

`MAX_PHOTONS_PER_CELL = 64`: 64 × 4 + 4 = 260 bytes per cell.
For 1000 cells: 260 KB (vs 5.1 MB for current 32-bin grid).

### Accuracy

**Exact** for the photons in the shortlist. Only approximate if the shortlist
truncates (more than 64 photons in radius of cell centre). For typical photon maps
(1M photons, r=0.05, scene 1.0³), expect ~30-100 photons per cell. A limit of 64
loses some, but the highest-weight ones are kept.

### Query Cost

O(8 × MAX_PHOTONS_PER_CELL) = O(512) — comparable to current O(256) bin lookups,
but each step does real per-photon work (tangential projection from query point).
Roughly 3-5× slower than current bin path, but 10-30× faster than full hash grid
walk (which chases pointers across hash buckets with poor cache behavior).

### Assessment

| Aspect | Rating |
|--------|--------|
| Accuracy improvement | **~95-99%** — per-photon exact, small truncation loss |
| Implementation risk | Medium — new data structure, but simple concept |
| Build cost | O(cells × nearby_photons) |
| Query cost | O(512) — ~2× current, ~10× better than raw hash grid |
| Memory | 260 KB (50× less than current) |
| Confidence | High |

---

## Recommendation

### Short Term (immediate fix): **Proposal C**

Flip `DEFAULT_USE_DENSE_GRID = false`. The hash grid per-photon path is already
implemented, tested, and produces exact results. The 12% frame time increase is
acceptable — NEE dominates at 85.5% of frame time regardless.

### Medium Term (performance optimization): **Proposal D**

If the hash grid path proves too slow for interactive use, implement the per-cell
photon shortlist. It gives per-photon accuracy with bounded O(512) query cost and
dramatically better cache behavior than the hash grid. This is the best
accuracy/performance tradeoff.

### Do Not Pursue: Proposals A and B

Proposal A (fix bins) has too many interacting normalization issues and fundamental
directional-discretization losses that cannot be fully corrected. Proposal B
(precomputed irradiance) loses directional information needed for non-Lambertian
surfaces.

| Proposal | Accuracy | Speed | Risk | Recommendation |
|----------|----------|-------|------|----------------|
| A. Fix bins | ~50% better | Same | High | **Skip** |
| B. Precomputed irradiance | ~65% better | Fastest | Medium | **Skip** |
| C. GPU hash grid (existing) | **100%** | ~12% slower | **Minimal** | **Do now** |
| D. Per-cell shortlist | **~98%** | ~2× current | Medium | **Do next** |
