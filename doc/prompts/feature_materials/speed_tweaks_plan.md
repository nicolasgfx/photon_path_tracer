# Speed Tweaks & Pre-computation Plan

> Design document for three optimisation topics:
> 1. Additional photon data for transparency/refraction
> 2. Hash-grid precomputed cell data bridging photon→render pass
> 3. Adaptive photon shooting for caustic convergence

---

## Current Architecture Snapshot

| Component | Key Facts |
|-----------|-----------|
| **PhotonSoA** | `pos`, `wi`, `norm`, `spectral_flux[N×32]`, `lambda_bin`, `flux`, `num_hero`, `bin_idx`, `source_emissive_idx` |
| **HashGrid** | `cell_size = 2r`, sorted-index + cell_start/end, `query_tangential`, `knn_shell_expansion` |
| **LightCache** | Per-cell top-16 emissive sources by flux (64 K cells, ~6.5 MB). Already proves the hash-grid precomputation pattern |
| **DensityEstimator** | Tangential disk kernel (Epanechnikov), surface-consistency filter (4 conditions), O(N_cell) per query |
| **Emitter** | Cell-stratified bouncing (Fibonacci 64 strata), RNG spatial hash, Russian roulette |
| **Camera pass** | specular chain → first diffuse → NEE + global gather + caustic gather + glossy continuation |
| **Glass BSDF** | `glass_sample` is colour-neutral (`Spectrum::constant`), no spectral Tf, no Beer-Lambert |

---

## Topic 1 — Photon Data for Transparency & Refraction

### 1.1  Problem Statement

Glass/transparent surfaces currently block photon deposition entirely:
`trace_photons` deposits only at `!mat.is_specular()` surfaces. The photon
*passes through* glass via `bsdf::sample` (mirror or refract), but:

- The throughput (`spectral_flux`) is multiplied by `Spectrum::constant(…)` — **no spectral absorption**.
- No record of whether the photon traversed glass, how many interfaces it crossed, or what IOR stack it saw.
- `glass_sample` uses a single scalar IOR; future wavelength-dependent IOR (dispersion) will need per-photon wavelength tracking.

At the **camera pass**, specular chain bounces follow the same colour-neutral glass. Photon gather at the first diffuse hit cannot tell whether nearby photons arrived through glass or not.

### 1.2  Proposed New Photon Fields

| Field | Type per photon | Size (10 M photons) | Purpose |
|-------|----------------|---------------------|---------|
| `path_flags` | `uint8_t` | 10 MB | Bit flags: bit 0 = "traversed glass", bit 1 = "caustic origin is glass", bit 2 = "has volumetric segment" |
| `Tf` (transmittance filter) | `Spectrum` (32 floats) or **packed** `half4` | 640 MB or 80 MB | Cumulative spectral attenuation from all glass/transparent segments the photon passed through |
| `ior_at_deposit` | `float` | 40 MB | IOR of the medium the photon was travelling in when deposited (1.0 = air). Needed for correct `n²` factor in Veach's non-symmetric correction for BSDF evaluation |
| `path_length` | `float` | 40 MB | Total geometric path length from emission to deposit — enables Beer-Lambert volumetric absorption at gather time |
| `bounce_count` | `uint8_t` | 10 MB | Total bounces (existing `bounce` variable, just not stored). Useful for filtering (e.g. ignore single-bounce photons in caustic map variance analysis) |

#### Memory Budget Trade-off

Full `Spectrum Tf` per photon is prohibitively large (640 MB at 10 M photons, 32 floats). Options:

| Encoding | Bytes/photon | 10 M total | Accuracy |
|----------|-------------|-----------|----------|
| None (current) | 0 | 0 | No glass colour |
| 3×float (RGB Tf) | 12 | 120 MB | Good for smooth Tf; spectral aliasing for sharp filters |
| **Tf baked into `spectral_flux`** | 0 | 0 | **Perfect — but changes meaning** |
| `uint8_t path_flags` only | 1 | 10 MB | Binary: "was glass on path?" |

**Recommended approach — bake Tf into spectral\_flux:**

Instead of storing a separate Tf array, multiply the photon's `spectral_flux`
by the material's spectral transmittance `Tf(λ)` at each glass bounce. The
deposited photon then automatically carries the correct colour-filtered flux.
This is the standard photon mapping approach: the throughput already absorbs
all BSDF evaluations along the path.

**Required code changes:**

1. **`material.h`**: Add `Spectrum Tf = Spectrum::constant(1.0f);` field (spectral transmittance filter).
2. **`bsdf.h → glass_sample`**: Return `s.f = mat.Tf * factor` instead of `Spectrum::constant(factor)`. Both reflection and refraction branches should apply Tf.
3. **`emitter.h → trace_photons`**: Already multiplies throughput by `bsdf_sample.f` — no change needed if `glass_sample` correctly returns Tf-weighted BSDF.
4. **`renderer.cpp → render_pixel`**: The specular chain already applies `bs.f * cos_theta / bs.pdf` to throughput — will automatically incorporate Tf.

**Low-cost extras worth adding:**

```cpp
// In PhotonSoA — 2 bytes per photon, 20 MB at 10M
std::vector<uint8_t>  path_flags;     // bit field (see below)
std::vector<uint8_t>  bounce_count;   // total bounces at deposit
```

Path flag bits:
```
bit 0: traversed_glass      — photon passed through ≥1 glass interface
bit 1: caustic_glass         — caustic path starts with glass refraction
bit 2: has_volume_segment   — photon had a volume interaction
bit 3: dispersion_active     — different hero wavelengths took different paths (future)
bits 4-7: reserved
```

### 1.3  Dispersion Path (Future)

When wavelength-dependent IOR is enabled:
- Hero wavelengths may split at a glass interface (different refraction angles).
- The per-hero `lambda_bin` + `flux` arrays already support per-wavelength flux.
- A "dispersion" flag (`path_flags` bit 3) would indicate that the photon's `wi`
  direction is approximate — the true direction varies per wavelength.
- For caustic map accuracy, each hero wavelength should be deposited separately
  when dispersion is significant (`|Δθ| > threshold`).

### 1.4  IOR Stack Tracking (Photon Emitter)

For nested dielectrics (glass inside liquid), the emitter needs an IOR stack:

```cpp
// Lightweight stack for trace_photons inner loop
struct IORStack {
    float stack[4] = {1.0f, 0, 0, 0};
    int   depth    = 0;
    float current_ior() const { return depth > 0 ? stack[depth-1] : 1.0f; }
    void  push(float ior) { if (depth < 4) stack[depth++] = ior; }
    void  pop()           { if (depth > 0) --depth; }
};
```

During `trace_photons`, maintain the stack as the photon enters/exits glass
surfaces. Pass `ior_stack.current_ior()` to `glass_sample` instead of `mat.ior`.

### 1.5  Implementation Order

| Step | Files | Effort | Impact |
|------|-------|--------|--------|
| 1. Add `Spectrum Tf` to Material | `material.h`, OBJ loader | Small | Enables coloured glass |
| 2. Wire Tf into `glass_sample` | `bsdf.h` | Small | Throughput automatically absorbs Tf |
| 3. Add `path_flags` + `bounce_count` to PhotonSoA | `photon.h` | Small | Metadata for Topics 2 & 3 |
| 4. Set flags in `trace_photons` | `emitter.h` | Small | Populate new fields |
| 5. IOR stack in trace_photons | `emitter.h` | Medium | Correct nested dielectrics |
| 6. Dispersion (per-hero split) | `emitter.h`, `photon.h` | Large | Spectral caustics (future) |

---

## Topic 2 — Hash-Grid Precomputed Cell Data

### 2.1  Concept

The LightCache already demonstrates the pattern: after photon tracing, aggregate
per-cell statistics from the photon map, use them at render time for O(1) lookup.

Extend this to a **CellInfoCache** — a second hash table (same spatial hash as
LightCache, 64 K cells) storing precomputed per-cell aggregate statistics derived
from the photon map and caustic map. The camera pass queries this cache to:
- Adapt gather radius
- Skip empty regions
- Guide NEE sample allocation
- Detect caustic regions for special treatment

### 2.2  Per-Cell Data Fields

```cpp
struct CellInfo {
    // ── From global photon map ──────────────────────────────────
    float    irradiance;          // Average scalar flux in cell (total_flux / count)
    float    flux_variance;       // Variance of per-photon flux in cell
    uint32_t photon_count;        // Number of global photons in this cell
    float    density;             // photon_count / cell_volume — for radius adaptation

    // ── Directional statistics ──────────────────────────────────
    float3   avg_wi;              // Mean incoming direction (flux-weighted)
    float    directional_spread;  // 1 - |avg_wi_normalized| — high = isotropic

    // ── Caustic detection ───────────────────────────────────────
    uint32_t caustic_count;       // Number of caustic photons in this cell
    float    caustic_flux;        // Total caustic flux in this cell
    uint8_t  is_caustic_hotspot;  // Binary: caustic_count > threshold

    // ── Glass interaction flags ─────────────────────────────────
    float    glass_fraction;      // Fraction of photons with traversed_glass flag

    // ── Surface geometry ────────────────────────────────────────
    float3   avg_normal;          // Average surface normal (for planarity check)
    float    normal_variance;     // Spread of normals — high = curved/edge region
};
```

**Memory**: ~80 bytes × 64 K cells = **5 MB** (tiny, independent of photon count).

### 2.3  Build Algorithm

Insert into `Renderer::build_photon_maps()`, after hash grid construction:

```
for each photon i in global_map:
    cell_key = hash(pos_i)
    cell.photon_count++
    cell.irradiance += total_flux(i)
    cell.avg_wi     += wi_i * total_flux(i)
    cell.avg_normal += norm_i
    // (running variance via Welford's algorithm)

for each photon j in caustic_map:
    cell_key = hash(pos_j)
    cell.caustic_count++
    cell.caustic_flux += total_flux(j)

// Finalise
for each cell:
    cell.irradiance      /= max(cell.photon_count, 1)
    cell.avg_wi           = normalize(cell.avg_wi)
    cell.avg_normal       = normalize(cell.avg_normal)
    cell.density          = cell.photon_count / cell_volume
    cell.is_caustic_hotspot = (cell.caustic_count > threshold)
    cell.directional_spread = 1.0 - length(cell.avg_wi)   // before normalise
    cell.normal_variance    = ...
```

Complexity: O(N_photons) — same as LightCache build, runs in parallel.

### 2.4  Render-Time Usage

#### 2.4.1  Adaptive Gather Radius

```cpp
float adapt_radius(float3 pos, float base_radius, const CellInfoCache& cache) {
    auto& cell = cache.query(pos);
    if (cell.photon_count == 0) return base_radius;

    // Shrink radius in high-density regions for sharper detail
    // Grow radius in low-density regions to reduce noise
    float target_photons = 100.f;  // desired photons in gather disk
    float estimated_in_disk = cell.density * PI * base_radius * base_radius;
    float ratio = target_photons / max(estimated_in_disk, 1.f);
    return clamp(base_radius * sqrt(ratio), min_radius, max_radius);
}
```

This replaces the current fixed `config_.gather_radius` with a per-query
adaptive radius similar to k-NN but without the expansion loop overhead.
The k-NN can still be used as a fallback when the CellInfo density is too
coarse (e.g. small cells with 0-1 photons).

#### 2.4.2  NEE Sample Count Allocation

```cpp
int nee_samples_for_point(float3 pos, const CellInfoCache& cache) {
    auto& cell = cache.query(pos);

    // More NEE samples in bright/high-variance regions
    // Fewer samples in dark/converged regions
    if (cell.irradiance < low_threshold) return 4;   // dark region: few NEE rays
    if (cell.flux_variance > high_var)   return 32;  // noisy: more samples
    return DEFAULT_NEE_CACHED_LIGHT_SAMPLES;          // normal
}
```

This steers computational effort toward regions that need it most. Photon map
variance is a good proxy for render-time indirect lighting variance.

#### 2.4.3  Skip Empty Regions

```cpp
// In render_pixel, before photon gather:
auto& cell = cell_cache.query(hit.position);
if (cell.photon_count == 0 && cell.caustic_count == 0) {
    // No photons anywhere nearby — skip gather entirely
    // (saves the 3×3×3 hash lookups + per-photon surface filter)
    goto skip_gather;
}
```

For scenes with large unlit regions (e.g. exterior scenes with interior
lighting), this skips the gather loop entirely for empty cells.

#### 2.4.4  Caustic-Aware Gather

When `cell.is_caustic_hotspot`, the renderer can:
- Use a **smaller gather radius** for caustic photons (sharper caustics)
- Allocate **more samples** to the caustic map gather
- **Enable k-NN** in the caustic region (adaptive radius tracks the caustic edge)

```cpp
if (cell.is_caustic_hotspot) {
    float caustic_r = config_.caustic_radius * 0.5f;  // tighter radius
    L_caustic = estimate_photon_density(
        hit.position, hit.normal, wo_local, mat,
        caustic_photons_, caustic_grid_, de_config, caustic_r);
}
```

#### 2.4.5  Photon-Guided Importance Sampling Boost

The existing `sample_photon_guided` scans all photons in the hash cell at O(N_cell).
With precomputed `CellInfo.avg_wi` and `directional_spread`, the guided sample
proposal can be constructed in O(1):

```cpp
// O(1) guided proposal instead of O(N_cell) discrete scan
PhotonGuidedSample fast_guided(float3 hit_pos, const CellInfoCache& cache, PCGRng& rng) {
    auto& cell = cache.query(hit_pos);
    if (cell.photon_count == 0) return {.valid = false};

    // Use precomputed average direction ± spread as a vMF proposal
    float kappa = 1.0f / max(cell.directional_spread, 0.01f);
    float3 wi = sample_vMF(cell.avg_wi, kappa, rng);
    float pdf = vMF_pdf(wi, cell.avg_wi, kappa);
    return {wi, pdf, true};
}
```

This is a **quality improvement** (better importance sampling) *and* a **speed
improvement** (O(1) instead of O(N_cell)).

### 2.5  Relationship to LightCache

The CellInfoCache and LightCache are complementary:
- **LightCache**: "which lights contribute here?" → steers NEE shadow rays toward relevant emitters
- **CellInfoCache**: "what is the photon field like here?" → steers gather radius, sample counts, strategy selection

Both are built after photon tracing, both are 64 K cell hash tables, both are O(1) query. They could share the `cell_coord` / `cache_cell_key` functions (or even be members of a single `CellCache` struct).

### 2.6  Implementation Order

| Step | Description | Effort |
|------|-------------|--------|
| 1. Define `CellInfo` struct + `CellInfoCache` class | Small |
| 2. Build from global + caustic photon maps | Small (follows LightCache pattern) |
| 3. Adaptive gather radius in `render_pixel` | Medium (careful clamping/fallback) |
| 4. Empty-region skip | Small |
| 5. Caustic hotspot detection + tighter radius | Small |
| 6. NEE sample count modulation | Small |
| 7. O(1) guided sampling via vMF | Medium |

---

## Topic 3 — Adaptive Photon Shooting for Caustics

### 3.1  Concept

During the photon map pass, some cells receive many caustic photons with high
variance (e.g. the focused caustic from a glass sphere). Other cells receive
few or zero. Standard uniform photon emission wastes budget on already-converged
regions and under-samples critical caustic features.

**Goal**: Detect high-frequency / high-variance caustic cells *during* photon
tracing, then adaptively emit more photons targeted at those cells until
the per-cell variance drops below a convergence threshold.

### 3.2  Two-Phase Photon Emission

```
Phase 1: Uniform emission (70% of caustic budget)
    - Standard trace_photons for caustic map
    - After phase 1: build CellInfoCache (§2) from caustic photons
    - Identify "hot" cells: cells with caustic_count > 0 AND flux_variance > threshold

Phase 2: Targeted emission (remaining 30% of caustic budget)
    - For each hot cell:
        - Estimate which emissive triangle(s) contribute to this cell
          (from source_emissive_idx of photons already in the cell)
        - Emit additional photons ONLY from those emitters
        - Direct them toward the cell centre (cone-constrained emission)
        - Re-deposit into the same caustic map
    - Repeat variance check; stop when variance < target or budget exhausted
```

### 3.3  Hot Cell Detection

After phase 1, for each cell with caustic photons:

```cpp
struct CausticCellStats {
    float mean_flux;
    float variance;
    float cv;            // coefficient of variation = σ / μ
    uint32_t count;
    uint16_t top_source; // emissive_idx that contributes most flux
};

bool is_hot_cell(const CausticCellStats& s) {
    return s.count >= MIN_PHOTONS_FOR_ANALYSIS   // need enough samples
        && s.cv > CAUSTIC_CV_THRESHOLD;          // high relative variance
}
```

The coefficient of variation (CV = σ/μ) is scale-invariant: a bright caustic
with low variance won't trigger, but a bright caustic with high variance
(under-sampled) will.

**Typical thresholds:**
- `MIN_PHOTONS_FOR_ANALYSIS = 10`
- `CAUSTIC_CV_THRESHOLD = 0.5` (50% relative standard deviation)

### 3.4  Targeted Emission Strategy

For each hot cell, we know:
1. **Which emitter** produced the caustic (`source_emissive_idx` majority vote)
2. **Where** the caustic lands (cell centre position)
3. **How much variance** remains

The targeted emitter emits photons from the known source triangle with a
**direction cone aimed at the cell centre**:

```cpp
// Targeted caustic emission for a single hot cell
void emit_targeted_caustics(
    const Scene& scene,
    int source_emissive_idx,
    float3 target_center,
    float cone_angle,         // half-angle in radians
    int num_photons,
    PhotonSoA& caustic_map)
{
    uint32_t tri_idx = scene.emissive_tri_indices[source_emissive_idx];
    const Triangle& tri = scene.triangles[tri_idx];

    for (int i = 0; i < num_photons; ++i) {
        // Sample position on source triangle
        float3 pos = sample_point_on_triangle(tri, rng);
        // Direction: aim at target with cone jitter
        float3 aim = normalize(target_center - pos);
        float3 dir = sample_within_cone(aim, cone_angle, rng);
        // Trace and deposit (reuse existing bounce logic)
        trace_single_photon(scene, pos, dir, source_emissive_idx, caustic_map);
    }
}
```

### 3.5  Convergence Criterion

After each batch of targeted photons, recompute the cell's CV:

```
do:
    emit N_batch targeted photons
    update running mean/variance for cell
until cv < CAUSTIC_CV_TARGET || total_emitted > max_budget_per_cell

CAUSTIC_CV_TARGET = 0.2  (20% relative std dev)
```

**Welford's online algorithm** updates mean and variance incrementally without
re-scanning all photons:

```cpp
void update_welford(float new_flux, float& mean, float& M2, int& n) {
    n++;
    float delta = new_flux - mean;
    mean += delta / n;
    float delta2 = new_flux - mean;
    M2 += delta * delta2;
    // variance = M2 / (n - 1)
}
```

### 3.6  Budget Allocation

The targeted photon budget should be allocated proportionally to cell variance:

```
total_targeted_budget = CAUSTIC_PHOTON_BUDGET * 0.30    (e.g. 1.5 M)
per_cell_budget(c) = total_targeted_budget * (cv(c)² / Σ cv(c)²)
```

Cells with higher variance get more photons. The `cv²` weighting ensures
variance reduction is approximately uniform across all hot cells after
targeted shooting.

### 3.7  Interaction with Cell Stratifier

The existing `CellStratifier` provides decorrelation for photon bounces at
each surface. Targeted emission should:
- Use a **separate** CellStratifier instance (reset counters) for the targeted pass
- Ensure targeted photons don't alias with phase-1 photons in the Fibonacci strata

### 3.8  Implementation Order

| Step | Description | Effort |
|------|-------------|--------|
| 1. Add Welford running stats to CellInfoCache build | Small |
| 2. Hot cell detection (CV threshold) | Small |
| 3. Two-phase `trace_photons` wrapper | Medium |
| 4. Targeted emission from known source → cell centre | Medium |
| 5. Iterative convergence loop with budget cap | Medium |
| 6. Merge targeted photons into existing caustic map + rebuild grid | Small |
| 7. Tunable thresholds in config.h | Small |

### 3.9  Expected Impact

| Metric | Before | After (estimated) |
|--------|--------|-------------------|
| Caustic noise (CV in hotspot cells) | 0.5–2.0 | 0.1–0.3 |
| Photons wasted in empty caustic regions | ~70% | ~20% |
| Caustic map build time | 1× | 1.3–1.5× (two phases) |
| Caustic visual quality at equal budget | Baseline | 2–4× (variance-to-noise) |

The build time increase is modest because Phase 2 only traces photons for hot
cells (typically 1–5% of all cells). The quality improvement is substantial
because those cells receive 10–50× more photons than under uniform emission.

---

## Cross-Topic Dependencies

```
Topic 1 (Photon Data)
    └─ path_flags.traversed_glass ──→ Topic 2 (CellInfo.glass_fraction)
    └─ bounce_count                ──→ Topic 3 (filter single-bounce for CV analysis)

Topic 2 (CellInfoCache)
    └─ caustic detection           ──→ Topic 3 (hot cell identification)
    └─ variance stats              ──→ Topic 3 (convergence criterion)

Topic 3 (Adaptive Shooting)
    └─ source_emissive_idx         ──→ Already exists in PhotonSoA
    └─ CellInfoCache               ──→ Must be built after Phase 1
```

**Recommended implementation sequence:**

1. **Topic 1, steps 1–3** — Add Tf to Material, wire into glass_sample, add path_flags/bounce_count to PhotonSoA
2. **Topic 2, steps 1–4** — CellInfoCache struct, build, adaptive radius, empty-region skip
3. **Topic 3, steps 1–6** — Two-phase caustic emission with convergence
4. **Topic 2, steps 5–7** — Caustic hotspot in CellInfoCache, NEE modulation, O(1) guided sampling
5. **Topic 1, steps 5–6** — IOR stack, dispersion (if needed)

---

## Config Constants to Add

```cpp
// §10  CELL INFO CACHE (Topics 2 & 3)
constexpr uint32_t CELL_CACHE_TABLE_SIZE = 65536u;  // Same as LightCache

// §10.1  Adaptive gather radius
constexpr float ADAPTIVE_RADIUS_MIN_FACTOR = 0.25f;  // Never shrink below 25% of base
constexpr float ADAPTIVE_RADIUS_MAX_FACTOR = 2.0f;   // Never grow above 200% of base
constexpr float ADAPTIVE_RADIUS_TARGET_K   = 100.f;  // Desired photons in gather disk

// §10.2  Adaptive caustic emission
constexpr float CAUSTIC_TARGETED_FRACTION  = 0.30f;  // 30% of caustic budget is targeted
constexpr int   CAUSTIC_MIN_FOR_ANALYSIS   = 10;     // Min photons per cell for CV
constexpr float CAUSTIC_CV_THRESHOLD       = 0.50f;  // CV threshold to classify as "hot"
constexpr float CAUSTIC_CV_TARGET          = 0.20f;  // Target CV after adaptive shooting
constexpr float CAUSTIC_CONE_HALF_ANGLE    = 15.f;   // Degrees, targeting cone half-angle
constexpr int   CAUSTIC_MAX_TARGETED_ITERS = 3;      // Max adaptive refinement passes
```

---

## File Impact Summary

| File | Topic 1 | Topic 2 | Topic 3 |
|------|---------|---------|---------|
| `material.h` | Add `Tf` | — | — |
| `bsdf.h` | Wire Tf into `glass_sample` | — | — |
| `photon.h` | Add `path_flags`, `bounce_count` | — | — |
| `emitter.h` | Set flags, IOR stack | — | Two-phase wrapper, targeted emission |
| `hash_grid.h` | — | — | — |
| `light_cache.h` | — | (reference pattern) | — |
| **`cell_cache.h`** (new) | — | CellInfoCache struct + build + query | Hot cell detection, convergence |
| `density_estimator.h` | — | Adaptive radius | — |
| `renderer.cpp` | — | Query CellInfoCache, adapt radius, skip empty | Call two-phase trace |
| `config.h` | — | Adaptive radius constants | Caustic adaptive constants |
| `direct_light.h` | — | NEE sample modulation | — |
