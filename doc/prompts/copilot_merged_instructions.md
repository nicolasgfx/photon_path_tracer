# Spectral Photon + Path Tracing Renderer — Single Copilot Instruction Set

Audience: **GitHub Copilot** implementing/maintaining this renderer.

Priority order:
1. **Physical correctness** (unbiased estimators, correct PDFs, no double counting)
2. **Deterministic + debuggable** (component outputs, clear invariants)
3. **Simplicity over performance** (fixed radii, fixed bins, explicit code)

OptiX is mandatory for all ray tracing in the final product. Do not add a CPU rendering fallback.

---

## 0) Non‑Negotiable Invariants (Read First)

- **Never double count direct lighting**.
  - Direct illumination at camera hitpoints is estimated by **NEE only** (shadow rays).
  - Photon maps must **not** contain “direct-from-light to first diffuse” deposits.
- **Every Monte Carlo estimator must divide by the exact PDF of how it sampled**.
- **Spectral bins never mix during transport**. Convert spectral → RGB only at output.
- **Photons store radiant flux per wavelength bin** (a power packet), not radiance.

---

## 1) Physical Units & Definitions

Rendering equation (spectral):

$$
L_o(x, \omega_o, \lambda) = \int_{\Omega} L_i(x, \omega_i, \lambda)\, f_s(x, \omega_i, \omega_o, \lambda)\, \cos\theta_i\, d\omega_i
$$

- Radiance $L$ : $[W\,/(sr\cdot m^2\cdot nm)]$
- Flux $\Phi$ : $[W\,/nm]$
- Irradiance $E$ : $[W\,/(m^2\cdot nm)]$

A stored photon represents **radiant flux** in one wavelength bin.

---

## 2) Core Data Structures (Conceptual)

- `Spectrum`: fixed-size array of `NUM_LAMBDA` bins.
- `Photon` (SoA on GPU):
  - `position`
  - `wi` (incoming direction at the deposit point, pointing **into** the surface)
  - `lambda_bin`
  - `flux` (radiant flux in that bin)
- `HashGrid`: hashed uniform grid for neighbor queries within fixed radius.

---

## 3) Single Shared Light Distribution (Photon Emission + NEE)

Build ONE distribution over all emissive triangles and reuse it for:
- photon emission (light → scene)
- NEE (surface hitpoint → light)

Triangle weight:

$$
w_t = A_t \cdot \bar{L}_{e,t}
$$

- $A_t$ = triangle area
- $\bar{L}_{e,t}$ = a deterministic “average emission proxy” (e.g., mean of `Le` spectrum; for textured emission, sample a small fixed set of UVs)

Implementation may use an **alias table** or a **CDF**. What matters is:
- sampling returns `tri_id`
- you can query `p_tri` = normalized probability of selecting that triangle.

---

## 4) Photon Pass (Build Photon Maps)

### 4.1 Emit 1 Photon (Per-photon steps)

A) Sample emissive triangle: `(tri_id, p_tri)` from the shared distribution.

B) Sample uniform point on triangle (area-uniform):

$$
\alpha = 1-\sqrt{u},\quad \beta = v\sqrt{u},\quad \gamma = 1-\alpha-\beta
$$

$$
x = \alpha v_0 + \beta v_1 + \gamma v_2
$$

PDF on area of chosen triangle:

$$
p_{pos} = 1/A_{tri}
$$

C) Sample emission direction.
- For a Lambertian emitter: cosine-weighted hemisphere oriented around triangle normal.

$$
p_{dir}(\omega) = \cos\theta/\pi
$$

D) Sample wavelength bin proportional to emission spectrum at the sampled point:

$$
p_{\lambda}(i\mid x) = \frac{L_e(x,\lambda_i)}{\sum_j L_e(x,\lambda_j)}
$$

E) Compute initial photon flux (power packet) for the chosen bin:

$$
\Phi = \frac{L_e(x,\omega,\lambda)\,\cos\theta}{p_{tri}\, p_{pos}\, p_{dir}\, p_{\lambda}}
$$

Correctness note: Sampling triangle weights by $A\cdot\bar{L_e}$ is fine; unbiasedness is preserved because $\Phi$ divides by the *actual* sampling PDF.

### 4.2 Photon Transport (Bounce loop)

Maintain:
- `flux` (scalar for the photon’s single `lambda_bin`)
- `hasSpecularChain` (true if any specular event occurred since emission)
- `lightPathDepth` (number of surface interactions so far after emission; define unambiguously)

Update rule at each bounce (conceptual throughput):

$$
flux \leftarrow flux \cdot \frac{f_s\cos\theta}{p(\omega)}
$$

- For Lambertian sampled with cosine hemisphere, $f_s=\rho/\pi$ and $p(\omega)=\cos\theta/\pi$, so the factor reduces to $\rho$.

### 4.3 Photon Deposition Rule (Critical: avoids double counting)

We must ensure photon maps contain **indirect** (and caustic) energy, not direct.

Define **lightPathDepth** as:
- 1 at the *first* surface hit after emission,
- 2 at the second surface hit, etc.

Deposit photons only when:
- hit material event is **diffuse-like** (non-delta)
- AND `lightPathDepth >= 2`

This is equivalent to “skip the first diffuse hit from the light source”.

### 4.4 Global vs Caustic Maps

- **Global photon map**: deposited diffuse hits with `hasSpecularChain == false`.
- **Caustic photon map**: deposited diffuse hits with `hasSpecularChain == true`.

Caustic photons typically use a smaller gather radius.

---

## 5) Spatial Hash Grid (Fixed-Radius Neighbor Search)

- Choose `cellSize ≈ 2 * radius`.
- Key = hash(floor(pos / cellSize)).
- Sort photons by key; build `cellStart[key] / cellEnd[key]`.
- Query scans neighbor cells covering the radius (commonly 3×3×3 when cellSize=2r).

Correctness note: Because different cells can collide into the same hash bucket, avoid double-processing a bucket during a query.

---

## 6) Photon Density Estimator (Indirect + Caustics)

At a diffuse camera hitpoint $x$ with outgoing direction $\omega_o$:

$$
L_{photon}(x,\omega_o,\lambda) = \frac{1}{\pi r^2}\sum_{i\in N(x)} \Phi_i(\lambda)\, f_s(x,\omega_i,\omega_o,\lambda)
$$

Optional radial kernel (e.g., Epanechnikov):
- Multiply each photon by kernel weight $W(\|x-x_i\|)$.
- If you use Epanechnikov, apply the correct normalization constant.

### 6.1 Surface Consistency Filter (Recommended)

Reject photon $i$ unless:

- hemisphere consistency: photon arrives from above surface
  - `dot(wi_photon, n_x) > 0` if `wi_photon` points into the surface
- plane-distance check:

$$
|n_x \cdot (x_i - x)| < \tau
$$

Purpose: avoid cross-surface contamination through thin walls/gaps.

---

## 7) Camera Pass (Hybrid Path Tracing)

Per camera path:
- `throughput T(λ) = 1`
- `radiance L(λ) = 0`

### 7.1 Direct Lighting: NEE (Soft Shadows)

At a diffuse hitpoint:
1) sample `(tri_id, p_tri)` from shared light distribution
2) sample uniform point `y` on that triangle (PDF `1/A`)
3) compute `wi = normalize(y-x)`, distance
4) cast shadow ray (x→y); if occluded: 0
5) evaluate `Le(y, -wi, λ)` and BSDF `f(x, wi, wo, λ)`

Convert PDF from area to solid angle:

$$
 p_{y,area} = p_{tri}\cdot\frac{1}{A_{tri}},\quad
 p_{\omega} = p_{y,area}\cdot\frac{\|x-y\|^2}{\cos\theta_y}
$$

Contribution per sample:

$$
L_{direct} += \frac{f\,Le\,\cos\theta_x}{p_{\omega}}
$$

Average over $M$ NEE samples.

### 7.2 Indirect Lighting: Photon Density

Add photon-map estimate separately:

$$
L += T \cdot L_{photon}
$$

### 7.3 Path Continuation (Optional; if enabled)

Strategies (conceptual):
- BSDF sampling
- photon-guided sampling
- (optional) light sampling as continuation (if you do full MIS path tracing)

If combining multiple proposals, use power heuristic:

$$
 w_k = \frac{p_k^2}{\sum_j p_j^2}
$$

Correctness note: If photon-guided sampling is a **discrete** proposal over a set of directions, you must be explicit about the measure and how its PDF is compared in MIS against continuous PDFs. If this is unclear, disable photon-guided MIS until the measure is defined unambiguously.

---

## 8) Spectral → RGB Output

After accumulation:
1. integrate spectrum against CIE XYZ curves
2. convert XYZ → linear sRGB
3. tone map
4. gamma correct

Perform the exact same conversion for component buffers so they are visually comparable.

---

## 9) Required Debug / Component Outputs

Every final render iteration/frame must be able to output at least:
- `out_nee_direct.png` (NEE-only direct component)
- `out_photon_indirect.png` (photon density only)
- `out_combined.png` (sum of the above)

Recommended additional outputs (high value for debugging):
- `out_photon_caustic.png`
- `out_indirect_total.png` (= global + caustic)
- `out_photon_count.png` (gathered photon count)

### 9.1 File Naming Convention

When writing multiple frames/iterations, prefix with a frame counter:
- `frame_0001_out_nee_direct.png`
- `frame_0001_out_photon_indirect.png`
- `frame_0001_out_photon_caustic.png`
- `frame_0001_out_indirect_total.png`
- `frame_0001_out_combined.png`

---

## 9b) Debug Viewer UX (Project Requirement)

### Key Bindings

- F1: toggle photon points
- F2: toggle global map
- F3: toggle caustic map
- F4: toggle hash grid cells
- F5: toggle photon directions
- F6: toggle PDF display
- F7: toggle radius sphere
- F8: toggle MIS weights
- F9: toggle spectral coloring mode
- TAB: cycle render modes

### Hover Cell Overlay

When the mouse hovers a hash-grid cell, display at least:
- cell coordinate
- photon count
- sum flux
- average flux
- dominant wavelength bin and wavelength (nm)
- gather radius
- map type (global/caustic)

---

## 9c) Render Modes (Debug + Validation)

Expose a single `RenderMode` enum and ensure each mode is a strict subset of terms:

- **First-hit debug**: normals/material ID/depth (no photon gather, no continuation)
- **DirectOnly**: direct lighting terms only (NEE + specular-chain visible emission if supported)
- **IndirectOnly**: photon density terms only
- **Full/Combined**: direct + indirect
- **PhotonMap**: visualize the photon density estimate itself (not a lit result)

Correctness requirement: if a mode disables NEE, it must also not include any direct-light photon deposits (those must never be stored).

---

## 10) Minimal Acceptance Tests (Must Pass)

1. Direct-only (photons off): soft shadows converge, brightness stable.
2. Indirect-only (NEE off): no “direct-lit” hotspot patterns.
3. Combined: `combined ≈ nee_direct + photon_indirect (+ caustic if present)`.
4. Cornell box: indirect color bleeding visible.
5. Glass caustics: concentrated patterns show up in caustic component.

---

## 11) Complete Execution Order (Single Frame)

1. Load scene (OBJ/MTL).
2. Identify emissive triangles.
3. Build shared light distribution (alias table or CDF) over emissive triangles.
4. Photon pass:
  - emit photons with tracked PDFs and spectral sampling
  - trace/scatter photons
  - deposit photons using the “no-direct-deposits” rule
  - build hash grid(s) (global + caustic)
5. Camera pass:
  - for each pixel: trace camera ray
  - at hits: add NEE direct (if enabled)
  - gather photon density (global + caustic) (if enabled)
  - continue path with BSDF sampling (and MIS if enabled)
6. Accumulate spectral radiance buffers.
7. Convert spectral → RGB and write PNGs (final + components).

---

## 12) Common Bugs to Avoid

- Forgetting area→solid-angle Jacobian (`dist^2 / cos_y`) in NEE.
- Using the wrong cosine in that Jacobian (must be emitter-side `cos_y`).
- Depositing photons at the first diffuse hit from the light (double counts with NEE).
- Using triangle-uniform sampling instead of power/area-weighted emission.
- Not offsetting shadow rays / new rays with an epsilon.
- Mixing wavelength bins during transport.

---

## 13) Unit Test Coverage Expectations

At minimum, keep tests covering:
- alias table / CDF sampling correctness (PDF sums to 1, sampling matches PDF)
- triangle sampling uniformity
- NEE PDF conversion correctness (area → solid angle)
- hash grid build/query correctness (distance filter + collision handling)
- density estimator surface consistency filters
- spectral conversions

---

## 14) “Reality Check” Notes for This Repo

This document is a **target** spec. When implementing, always reconcile with:
- the OptiX device programs (final behavior)
- the host-side OptiX glue (what gets uploaded / which buffers exist)

If any part of the spec conflicts with the running OptiX path, prefer:
1) fixing the code (if it’s a bug), or
2) updating this spec (if the spec was too ambitious/unclear).

---

## 15) Photon Directional Bins — Detailed Coding Plan

This section is a merged-in, detailed implementation plan for a **directional bin cache** to reduce redundant photon gathers and to enable photon-guided sampling. It is a *proposal/plan*, and must preserve the invariants above (especially: no double counting direct lighting; correct PDFs).

### 15.1 Review of the Proposal

#### 15.1.1 Core Idea

Replace the per-sample photon gather loop (currently iterating 100-500+
photons per hitpoint, per SPP sample) with a **fixed-size directional bin
cache**.  At each pixel's first diffuse hit, gather photons once and
distribute their flux into `N` pre-defined directional bins covering the
full sphere.  Subsequent SPP samples read from this cache instead of
re-querying the hash grid.

Each bin stores:
- Accumulated spectral flux `Spectrum` from photons whose `wi` falls into
  that bin's solid angle
- Total weight (Epanechnikov-weighted photon count) for normalization
- A representative direction (flux-weighted centroid of photon `wi` vectors)

#### 15.1.2 Bin Geometry

Bins tile the **full sphere** (not just a hemisphere) to support caustics
and glass transport.  At diffuse hitpoints, only bins in the positive
hemisphere of the surface normal are used for BSDF continuation.

**Bin layout:** Fibonacci sphere (quasi-uniform distribution of N points on
S²).  This gives near-equal solid-angle bins with no polar concentration
artifacts, and is trivially computable from a bin index with no lookup
table.

```
// Fibonacci sphere: bin direction for index k ∈ [0, N)
golden_ratio = (1 + √5) / 2
θ_k = arccos(1 − 2(k + 0.5) / N)
φ_k = 2π · k / golden_ratio
dir_k = (sin θ_k cos φ_k,  sin θ_k sin φ_k,  cos θ_k)
```

Finding which bin a photon direction maps to = nearest-neighbor search
among N Fibonacci directions.  For N ≤ 64, a brute-force `dot(wi, dir_k)`
scan is cheaper than any spatial index (16-64 dot products vs. a single
hash grid lookup of 500+ photons).

#### 15.1.3 Edge Case: Hemisphere Boundary

A bin whose center lies slightly below the tangent plane may still capture
photons arriving from just above.  The plan:

1. During population, accept ALL photons into their nearest bin regardless
   of hemisphere (full-sphere binning).
2. During sampling (BSDF bounce), skip bins whose center has
   `dot(dir_k, normal) < -ε` (deep below horizon).
3. For bins near the horizon (`-ε < dot < +ε`): accept the bin for
   sampling, but **clamp the jittered bounce direction** to the positive
   hemisphere:
   ```
   if dot(wi_jittered, normal) <= 0:
       wi_jittered = reflect across tangent plane
   ```
   This preserves energy from near-grazing photons without shooting
   rays into the surface.

---

### 15.2 Comparison with Existing Approaches

#### 15.2.1 Literature Context

| Technique | Source | How bins relate |
|-----------|--------|-----------------|
| **Photon Map Density Estimation** | Jensen 2001 | Our current approach. Bins replace the per-query gather loop with pre-aggregated directional data. |
| **Irradiance Caching** | Ward et al. 1988 | Similar motivation (cache lighting at sparse points), but bins operate per-pixel at the photon gather level, not at the irradiance level. |
| **Spherical Harmonics for Photon Maps** | Schregle 2003 | Encodes photon distribution into SH coefficients. Bins are a discrete alternative that preserves directional peaks better (SH smooths them). |
| **Photon-guided path tracing** | Vorba et al. 2014 | Learns a parametric mixture model (VMF) for importance sampling. Our bins approximate this with a fixed non-parametric histogram — simpler, no fitting, deterministic. |
| **Radiance caching / lightcuts** | Walter et al. 2005 | Hierarchical light representation. Bins are a flat per-pixel version suited to GPU parallelism. |

#### 15.2.2 Current Code vs. Proposed

```
CURRENT PIPELINE (per pixel, per SPP sample):
═══════════════════════════════════════════════

  for s in 0..SPP:                          // 16 iterations
    ┌─ trace primary ray ──────────────┐
    │  hit diffuse surface             │
    │                                  │
    │  ┌─ NEE (4 shadow rays) ───────┐ │
    │  │  random light sample × 4    │ │    ← NEE: 4 shadow rays
    │  └─────────────────────────────┘ │
    │                                  │
    │  ┌─ Photon Gather ─────────────┐ │
    │  │  iterate ALL photons in     │ │
    │  │  3×3×3 cells (~100-500+)    │ │    ← REDUNDANT: same photons × 16
    │  │  for each: distance test,   │ │
    │  │  plane test, kernel weight, │ │
    │  │  BSDF eval, accumulate      │ │
    │  └─────────────────────────────┘ │
    │                                  │
    │  ┌─ BSDF Bounce ──────────────┐  │
    │  │  cosine hemisphere sample   │  │    ← BLIND: ignores photon info
    │  │  (random direction)         │  │
    │  └─────────────────────────────┘  │
    └──────────────────────────────────┘

  Total photon lookups: SPP × photons_in_radius ≈ 16 × 300 = 4,800
  Bounce quality: random (no light awareness)
```

```
PROPOSED PIPELINE (per pixel):
═══════════════════════════════

  ┌─ PHASE 1: Bin Population (once per pixel) ──────────┐
  │  trace reference ray (pixel center)                  │
  │  hit diffuse surface                                 │
  │  iterate ALL photons in 3×3×3 cells (~100-500+)     │
  │  for each photon:                                    │
  │    find nearest bin k = argmax dot(wi, dir_k)        │
  │    bin[k].flux[λ] += photon_flux × kernel_weight     │
  │    bin[k].weight  += kernel_weight                   │
  │    bin[k].dir     += wi × kernel_weight  (centroid)  │
  │  normalize bin directions                            │
  │  write bins[0..N-1] to per-pixel cache               │
  └──────────────────────────────────────────────────────┘

  for s in 0..SPP:                          // 16 iterations
    ┌─ trace primary ray (jittered) ───────────────┐
    │  hit diffuse surface                         │
    │                                              │
    │  ┌─ NEE (guided by bins) ──────────────────┐ │
    │  │  sort/select top-K bins by flux         │ │
    │  │  cast shadow rays toward bin centers    │ │  ← GUIDED: best K dirs
    │  │  (jittered within bin solid angle)      │ │
    │  └─────────────────────────────────────────┘ │
    │                                              │
    │  ┌─ Photon Contribution (from cache) ──────┐ │
    │  │  read N bins from cache                 │ │
    │  │  sum flux × BSDF for hemisphere bins    │ │  ← N lookups, not 500
    │  └─────────────────────────────────────────┘ │
    │                                              │
    │  ┌─ BSDF Bounce (guided by bins) ─────────┐  │
    │  │  select bin proportional to flux        │  │
    │  │  jitter within bin solid angle          │  │  ← GUIDED: follow light
    │  │  clamp to positive hemisphere           │  │
    │  └─────────────────────────────────────────┘  │
    └──────────────────────────────────────────────┘

  Total photon lookups: 1 × photons_in_radius ≈ 300  (÷16 reduction)
  Bounce quality: informed by photon directional distribution
```

---

### 15.3 Impact Analysis

#### 15.3.1 Speed

| Component | Current | Proposed | Speedup |
|-----------|---------|----------|---------|
| **Photon gather** (per pixel) | SPP × iterate_all_photons = 16 × ~300 = 4,800 photon evaluations | 1 × ~300 (populate) + SPP × N (read) = 300 + 16×32 = 812 | **~6×** |
| **BSDF bounce** (per sample) | 1 RNG + 1 trig (cosine hemisphere) | 1 bin select + 1 jitter + 1 clamp | ~same |
| **NEE shadow rays** | 4 random light samples × SPP | K directed shadow rays × SPP (K ≤ 4) | same count, **better variance** |
| **Memory** | 0 extra | width × height × N × sizeof(PhotonBin) | ~32 bins × 148B = 4.6KB/pixel → **~3.6 GB** at 1024×768 for N=32 with full Spectrum. See §15.3.3 for reduction. |

#### 15.3.2 Quality

| Aspect | Current | Proposed | Improvement |
|--------|---------|----------|-------------|
| **Bounce directions** | Cosine hemisphere (blind) | Flux-proportional (informed) | **Major**: samples go where light is |
| **NEE shadow rays** | Random light triangles | Directed at strongest photon flux | **Moderate**: less wasted shadow rays |
| **Photon estimate** | Exact (all photons weighted) | Discretized into N bins | **Minor loss**: smoothed by bin solid angle |
| **Temporal coherence** | None (new RNG each frame) | Bin cache stable across SPP | Reduced noise flicker |

#### 15.3.3 Memory Budget & Bin Data Compression

Full `Spectrum` per bin (32 floats × 4 bytes = 128 bytes) is too expensive.
Since each photon only contributes to ONE wavelength bin, we can store
per-bin data compactly:

**Compact PhotonBin layout (GPU):**
```c
struct PhotonBin {    // 24 bytes per bin
    float flux;       // total weighted flux (summed across all λ)
    float dir_x;      // flux-weighted centroid direction x
    float dir_y;      // centroid y
    float dir_z;      // centroid z
    float weight;     // Epanechnikov weight sum (for normalization)
    uint16_t count;   // number of photons accumulated
    uint16_t pad;
};
```

For spectral contribution, we still need wavelength info.  Options:
- **(A) Per-bin Spectrum** (128 + 20 = 148 bytes × 32 bins = 4.6 KB/pixel).
  At 1024×768 = 3.6 GB — too much for most GPUs.
- **(B) Dominant-λ per bin** (store dominant wavelength bin index + scalar
  flux = 24 bytes × 32 = 768 B/pixel = 600 MB).  Loses spectral
  resolution within each directional bin.
- **(C) Packed spectral channels** (4-8 representative wavelengths per bin =
  48-80 bytes × 32 = 1.5-2.5 KB/pixel = 1.2-2.0 GB).
- **→ Recommended (D): Two-pass approach**
  - Pass 1 (bin population): store scalar flux + direction only (24 bytes
    per bin × 32 = 768 B/pixel = **600 MB**).  Fits in VRAM.
  - Pass 2 (render SPP loop): use bin directions for guided sampling, but
    evaluate BSDF × photon contribution on-the-fly during the first sample
    only (re-read spectrum from photon map once using the bin's direction as
    a filter).  Eliminates storing Spectrum per bin.

**Recommended initial implementation: Option D with N=16 bins = 384 B/pixel
≈ 300 MB at 1024×768.**

---

### 15.4 Detailed Design

#### 15.4.1 New Data Structures

##### `src/core/photon_bins.h` (NEW FILE)
```cpp
#pragma once
#include "core/types.h"
#include "core/config.h"
#include <cmath>

// Number of directional bins — compile-time constant from config.h
// using PHOTON_BIN_COUNT

// ── Fibonacci sphere bin directions ─────────────────────────────────
// Quasi-uniform distribution of N points on S².
// Stored as constexpr for GPU use.

struct PhotonBinDirs {
    float3 dirs[MAX_PHOTON_BIN_COUNT];  // MAX_PHOTON_BIN_COUNT = 64 (upper bound)
    int    count;

    HD void init(int n) {
        count = n;
        const float golden_angle = PI * (3.0f - sqrtf(5.0f));
        for (int k = 0; k < n; ++k) {
            float theta = acosf(1.0f - 2.0f * (k + 0.5f) / (float)n);
            float phi   = golden_angle * k;
            dirs[k] = make_f3(
                sinf(theta) * cosf(phi),
                sinf(theta) * sinf(phi),
                cosf(theta));
        }
    }

    // Find nearest bin for direction wi (brute-force, fine for N≤64)
    HD int find_nearest(float3 wi) const {
        int   best = 0;
        float best_dot = -2.0f;
        for (int k = 0; k < count; ++k) {
            float d = dot(wi, dirs[k]);
            if (d > best_dot) { best_dot = d; best = k; }
        }
        return best;
    }
};

// ── Per-bin data (GPU cache) ────────────────────────────────────────
struct PhotonBin {
    float flux;       // total Epanechnikov-weighted flux
    float dir_x;      // flux-weighted centroid direction x
    float dir_y;      // flux-weighted centroid direction y
    float dir_z;      // flux-weighted centroid direction z
    float weight;     // total Epanechnikov weight (for normalization)
    int   count;      // number of photons in this bin
};
// sizeof(PhotonBin) = 24 bytes
// Per-pixel: 24 × PHOTON_BIN_COUNT
// At N=16, 1024×768: 24 × 16 × 786,432 = 302 MB
// At N=32, 1024×768: 24 × 32 × 786,432 = 604 MB
```

#### 15.4.2 Config Constants

Add to `config.h`:
```cpp
// ── Photon directional bins ─────────────────────────────────────────
constexpr int   PHOTON_BIN_COUNT          = 16;    // directional bins per pixel
constexpr int   MAX_PHOTON_BIN_COUNT      = 64;    // compile-time upper bound
constexpr float PHOTON_BIN_HORIZON_EPS    = 0.05f; // bins below -eps are skipped
constexpr int   PHOTON_BIN_NEE_TOP_K      = 4;     // top-K bins for guided NEE
```

#### 15.4.3 LaunchParams Additions

Add to `launch_params.h`:
```cpp
    // Photon directional bin cache (device pointers)
    PhotonBin* photon_bin_cache;  // [width * height * PHOTON_BIN_COUNT]
    int        photon_bin_count;  // runtime copy of PHOTON_BIN_COUNT
    int        photon_bins_valid; // 1 = cache populated, 0 = need population pass
```

#### 15.4.4 Two-Pass Render Architecture

**Current** `__raygen__render` does everything in one kernel launch.  The
new design splits into:

```
Pass 1: __raygen__populate_bins
  - ONE launch per frame (not per SPP)
  - Each pixel: trace center ray, find first diffuse hit
  - Query hash grid, populate N bins
  - Write to photon_bin_cache[pixel * N + k]

Pass 2: __raygen__render (modified)
  - SPP loop as before
  - Instead of dev_estimate_photon_density(), read bins from cache
  - Use bins for guided BSDF bounce
  - Use bins for guided NEE direction selection
```

This requires a new raygen program + SBT entry in the OptiX pipeline.

#### 15.4.5 Bin Population Kernel (Pass 1)

```
__raygen__populate_bins():
  pixel = optixGetLaunchIndex()
  trace center ray → first diffuse hit (pos, normal, mat_id)
  
  // Initialize N bins to zero
  for k in 0..N:  bins[k] = {0}
  
  // Build ONB at hitpoint
  frame = ONB::from_normal(normal)
  
  // Iterate photons in gather radius (same hash grid query)
  for each cell in 3×3×3 neighborhood:
    for each photon j in cell:
      diff = pos - photon_pos[j]
      d² = dot(diff, diff)
      if d² > r²: continue
      if |dot(diff, normal)| > TAU: continue
      
      w = 1 - d²/r²                          // Epanechnikov
      wi_world = photon_wi[j]
      k = find_nearest_bin(wi_world)          // O(N) dot products
      
      bins[k].flux   += photon_flux[j] * w
      bins[k].dir_x  += wi_world.x * photon_flux[j] * w
      bins[k].dir_y  += wi_world.y * photon_flux[j] * w
      bins[k].dir_z  += wi_world.z * photon_flux[j] * w
      bins[k].weight += w
      bins[k].count  += 1
  
  // Normalize centroid directions
  for k in 0..N:
    if bins[k].count > 0:
      len = length(bins[k].dir)
      if len > 0: bins[k].dir /= len
      else: bins[k].dir = fibonacci_dir[k]    // fallback to bin center
  
  // Write to cache
  for k in 0..N:
    photon_bin_cache[pixel * N + k] = bins[k]
```

#### 15.4.6 Modified Render Kernel (Pass 2)

##### 15.4.6.1 Photon Contribution from Bins

Replace `dev_estimate_photon_density()` call with:

```
dev_estimate_from_bins(pixel_idx, normal, wo_local, mat_id):
  L = Spectrum::zero()
  frame = ONB::from_normal(normal)
  
  for k in 0..N:
    bin = photon_bin_cache[pixel_idx * N + k]
    if bin.count == 0: continue
    
    // Hemisphere check (diffuse)
    wi_world = make_f3(bin.dir_x, bin.dir_y, bin.dir_z)
    if dot(wi_world, normal) <= 0: continue
    
    wi_local = frame.world_to_local(wi_world)
    f = dev_bsdf_evaluate(mat_id, wo_local, wi_local)
    
    // Flux normalization same as density estimator
    norm = 1.5 / (PI * r² * num_photons)
    for i in 0..NUM_LAMBDA:
      L.value[i] += f.value[i] * bin.flux * norm
  
  return L
```

**Note:** This loses per-wavelength-bin specificity (the current code
only adds flux to the photon's specific λ bin).  To preserve this, each
PhotonBin needs per-λ accumulation.  For N=16 and NUM_LAMBDA=32 this is
16 × 32 × 4 = 2 KB/pixel = 1.5 GB at 1024×768.  Too much.

**Solution:** Keep the scalar flux in bins for direction guidance, but for
the actual spectral density estimate, perform the gather only on the first
SPP sample and cache the resulting Spectrum (one per pixel, 128 bytes).
Direction-guided bounces read from bins; spectral contribution reads from
the single cached Spectrum.

##### 15.4.6.2 Guided BSDF Bounce

Replace `sample_cosine_hemisphere_dev(rng)` with:

```
dev_sample_guided_bounce(bins, N, normal, rng):
  // Build CDF from bin fluxes (hemisphere only)
  float cdf[N]
  total = 0
  for k in 0..N:
    wi = make_f3(bins[k].dir_x, bins[k].dir_y, bins[k].dir_z)
    cos_n = dot(wi, normal)
    if cos_n <= -PHOTON_BIN_HORIZON_EPS or bins[k].count == 0:
      cdf[k] = total       // skip bin
      continue
    cdf[k] = total + bins[k].flux * max(cos_n, 0)
    total = cdf[k]
  
  if total <= 0: return sample_cosine_hemisphere(rng)   // fallback
  
  // Select bin
  xi = rng.next_float() * total
  selected_k = binary_search(cdf, N, xi)
  
  // Jitter within bin solid angle
  // Bin solid angle ≈ 4π/N steradians → half-angle ≈ arccos(1 - 2/N)
  // Generate direction within cone around bin centroid
  bin_dir = make_f3(bins[selected_k].dir...)
  half_angle = acosf(1.0f - 2.0f / N)
  jittered_dir = sample_cone(bin_dir, half_angle, rng)
  
  // Clamp to positive hemisphere
  if dot(jittered_dir, normal) <= 0:
    jittered_dir = reflect(jittered_dir, normal)
  
  return jittered_dir
```

The corresponding PDF for MIS:
```
dev_guided_pdf(wi, bins, N, normal):
  k = find_nearest_bin(wi)
  if bins[k].count == 0: return 0
  cos_n = dot(bins[k].dir, normal)
  if cos_n <= -eps: return 0
  
  total_flux = sum of active bins flux * max(cos_n, 0)
  bin_flux = bins[k].flux * max(cos_n, 0)
  
  // Probability of selecting this bin × density within cone
  p_bin = bin_flux / total_flux
  cone_half = acosf(1 - 2/N)
  p_cone = 1.0 / (2π(1 - cos(cone_half)))    // uniform cone PDF
  return p_bin * p_cone
```

##### 15.4.6.3 Guided NEE

Replace random light CDF sampling with bin-guided selection:

```
dev_nee_guided(bins, N, normal, ...):
  // Sort bins by flux (find top-K)
  // For K=4 shadow rays, select top-4 bins
  
  top_k = find_top_k_bins(bins, N, normal, K=4)
  
  for each bin_k in top_k:
    // Cast shadow ray toward bin centroid direction (jittered)
    wi = jitter_in_cone(bins[bin_k].dir, half_angle, rng)
    <standard NEE evaluation along wi>
```

**Important caveat:** NEE must still sample actual light triangles (for
correct PDF computation).  Bins can BIAS the selection but the geometric
light PDF must be computed for the actual triangle hit.  The cleanest
approach: use bins to filter which emissive triangles to target.

**Simpler alternative for NEE:** Keep the existing CDF-based light sampling
for the first shadow ray (bounce 0, already has 4 samples).  Use bins only
for bounce ≥ 1 direction guidance (where we currently have 1 shadow ray).

---

### 15.5 Stratified SPP (B3)

Replace the current random sub-pixel jitter:
```cpp
float u = ((float)px + rng.next_float()) / (float)params.width;
float v = ((float)py + rng.next_float()) / (float)params.height;
```

With stratified sampling:
```cpp
int stratum_x = s % STRATA_X;              // e.g., 4 for 4×4
int stratum_y = s / STRATA_X;
float u = ((float)px + (stratum_x + rng.next_float()) / STRATA_X) / (float)params.width;
float v = ((float)py + (stratum_y + rng.next_float()) / STRATA_Y) / (float)params.height;
```

For SPP=16: STRATA_X=4, STRATA_Y=4 → 4×4 stratified grid with jitter
within each stratum.

---

### 15.6 Stochastic Photon Cap (A2) — Optional

During bin population (Pass 1), if a cell contains more than
`MAX_PHOTONS_PER_CELL` photons, randomly subsample:

```
for j in start..end:
  if (end - start) > MAX_PHOTONS_PER_CELL:
    if rng.next_float() > MAX_PHOTONS_PER_CELL / (float)(end-start):
      continue
    scale = (float)(end-start) / MAX_PHOTONS_PER_CELL
  else:
    scale = 1.0

  // accumulate into bin with flux *= scale
```

This caps the population loop at ~64-128 photons per cell regardless of
density.

---

### 15.7 Implementation Plan — Files to Change

#### 15.7.1 New Files

| File | Purpose |
|------|---------|
| `src/core/photon_bins.h` | `PhotonBinDirs`, `PhotonBin` structs, Fibonacci sphere generation, `find_nearest_bin()` |

#### 15.7.2 Modified Files

| File | Changes |
|------|---------|
| `src/core/config.h` | Add `PHOTON_BIN_COUNT`, `MAX_PHOTON_BIN_COUNT`, `PHOTON_BIN_HORIZON_EPS`, `PHOTON_BIN_NEE_TOP_K`, `STRATA_X/Y`, `MAX_PHOTONS_PER_CELL` |
| `src/optix/launch_params.h` | Add `PhotonBin* photon_bin_cache`, `int photon_bin_count`, `int photon_bins_valid` |
| `src/optix/optix_device.cu` | (1) New `__raygen__populate_bins` kernel. (2) New `dev_estimate_from_bins()` replacing `dev_estimate_photon_density()` in render path. (3) New `dev_sample_guided_bounce()` replacing `sample_cosine_hemisphere_dev()` for BSDF continuation. (4) Stratified sub-pixel offsets in `__raygen__render`. (5) Stochastic photon cap in populate_bins. (6) Guided NEE using bin flux ranking. |
| `src/optix/optix_renderer.h` | Add `DeviceBuffer d_photon_bin_cache_`, method `populate_bins()`, Pipeline/SBT entries for new raygen program |
| `src/optix/optix_renderer.cpp` | (1) Allocate `d_photon_bin_cache_` in `resize()`. (2) Build new raygen program for `__raygen__populate_bins`. (3) New `populate_bins()` method that launches the bin population kernel. (4) Call `populate_bins()` before `render_one_spp()` loop. (5) Set `photon_bin_cache` pointer in `fill_common_params()`. |
| `src/main.cpp` | Call `populate_bins()` after photon tracing, before render loop. |
| `tests/test_main.cpp` | New test cases (see §15.8). |

#### 15.7.3 Unchanged Files

| File | Reason |
|------|--------|
| `src/photon/hash_grid.h` | Hash grid still used for bin population |
| `src/photon/photon.h` | PhotonSoA unchanged |
| `src/photon/density_estimator.h` | CPU-side estimator unchanged (not used in GPU path) |
| `src/bsdf/bsdf.h` | BSDF evaluation unchanged |
| `src/scene/*` | Scene loading unchanged |
| `src/renderer/camera.h` | Camera unchanged |

---

### 15.8 Unit Tests

#### 15.8.1 New Test Cases

```
TEST(PhotonBins, FibonacciSphereCoversUnitSphere)
  // Generate N=16,32,64 Fibonacci directions
  // Verify: all directions have unit length
  // Verify: min pairwise angle > 0 (no duplicates)
  // Verify: centroid ≈ (0,0,0) (quasi-uniform)
  // Verify: solid angle per bin ≈ 4π/N ± tolerance

TEST(PhotonBins, FindNearestBinCorrect)
  // For each Fibonacci direction, verify find_nearest returns itself
  // For a direction between two bins, verify nearest is geometrically correct

TEST(PhotonBins, HemisphereCoverage)
  // For normal = (0,0,1), count bins with dot(dir,normal) > 0
  // Verify: approximately N/2 bins in positive hemisphere

TEST(PhotonBins, BinPopulationBasic)
  // Create 3 photons with known directions and flux
  // Populate bins
  // Verify: photons land in correct bins
  // Verify: flux accumulated correctly
  // Verify: centroid direction matches photon wi

TEST(PhotonBins, BinPopulationEpanechnikov)
  // Place photon at distance d from gather point
  // Verify: kernel weight = 1 - d²/r² applied to flux

TEST(PhotonBins, CentroidNormalization)
  // Two photons with different fluxes pointing near same bin
  // Verify: centroid is flux-weighted average, normalized

TEST(PhotonBins, EmptyBinsHandled)
  // No photons in any cell
  // Verify: all bins have count=0, flux=0
  // Verify: guided bounce falls back to cosine hemisphere

TEST(PhotonBins, GuidedBouncePDF)
  // Populate bins with known flux distribution
  // Sample many guided bounces
  // Verify: empirical distribution matches flux-proportional PDF
  // Verify: all samples in positive hemisphere

TEST(PhotonBins, HorizonEdgeCase)
  // Place photon at near-grazing angle (dot(wi, normal) ≈ 0)
  // Verify: bin is populated
  // Verify: jittered bounce clamped to positive hemisphere

TEST(PhotonBins, StochasticCapPreservesFlux)
  // Populate with 1000 photons, MAX_PHOTONS_PER_CELL = 64
  // Verify: total flux across bins ≈ total flux of all photons (within %)

TEST(PhotonBins, StratifiedSubPixelCoverage)
  // Generate 16 stratified sub-pixel offsets (4×4)
  // Verify: each stratum has exactly one sample
  // Verify: all offsets in [0, 1)²

TEST(PhotonBins, GuidedNEESelectsTopBins)
  // Populate bins with one dominant flux bin
  // Verify: top-K selection returns it
  // Verify: shadow ray direction falls within bin solid angle

TEST(PhotonBins, SpectralConsistency)
  // Compare: full density estimate vs bin-cached estimate
  // For a simple setup (1 photon, 1 bin covering its direction)
  // Verify: results match within discretization tolerance
```

#### 15.8.2 Integration Tests (require OptiX)

```
TEST(OptiX, BinPopulationKernelRuns)
  // Load Cornell box, trace photons, launch populate_bins
  // Download bin cache, verify non-zero bins exist

TEST(OptiX, BinGuidedRenderConverges)
  // Render 16 SPP with bins vs 16 SPP without bins
  // Verify: both produce positive finite output
  // Compare: variance of bin-guided should be ≤ cosine-hemisphere
```

---

### 15.9 Phased Implementation Schedule

#### Phase 1: Foundation (data structures + bin population)
1. Create `src/core/photon_bins.h` with `PhotonBinDirs`, `PhotonBin`
2. Add config constants to `config.h`
3. Add cache pointer to `launch_params.h`
4. Write `__raygen__populate_bins` kernel in `optix_device.cu`
5. Allocate buffer + launch in `optix_renderer.cpp`
6. Wire up in `main.cpp`
7. Unit tests for Fibonacci sphere, bin lookup, population

#### Phase 2: Cached photon contribution (A1 — per-pixel cache)
1. Replace `dev_estimate_photon_density()` with `dev_estimate_from_bins()`
2. Cache Spectrum result for reuse across SPP
3. Verify: output matches baseline (within binning tolerance)

#### Phase 3: Guided BSDF bounce (B1)
1. Implement `dev_sample_guided_bounce()` using bin flux CDF
2. Implement cone jittering within selected bin
3. Implement hemisphere clamping for edge-case bins
4. Add guided PDF for MIS
5. MIS weight between guided and cosine hemisphere
6. Unit tests for PDF, hemisphere coverage

#### Phase 4: Stratified SPP (B3)
1. Add STRATA_X/Y to config
2. Modify `__raygen__render` sub-pixel offset calculation
3. Unit test for stratum coverage

#### Phase 5: Stochastic photon cap (A2) — Optional
1. Add MAX_PHOTONS_PER_CELL to config
2. Implement subsample + scale in populate_bins
3. Unit test for flux preservation

#### Phase 6: Guided NEE (B2) — Required
1. Rank bins by flux in populate_bins or render kernel
2. Use top-K bins to bias shadow ray directions
3. Maintain correct PDF computation for MIS

---

### 15.10 Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| VRAM overflow (bin cache too large) | High | Start with N=16 (300 MB). Offer N=8 fallback. |
| Binning artifacts (N too small → visible lobes) | Medium | Jitter within bin solid angle smooths artifacts. MIS with cosine hemisphere fills gaps. |
| Spectral accuracy loss | Medium | Keep first-sample full gather for spectral density; use bins only for direction guidance. |
| Two-pass latency | Low | populate_bins is a single kernel launch, same pixel count as render. Expected ~same cost as 1 SPP. |
| Pipeline complexity (new raygen program) | Low | OptiX supports multiple raygen programs. Same SBT/GAS/pipeline. |
| Kernel register pressure (N bin array on stack) | Medium | For N=16: 16 × 24B = 384B register/local mem. Within budget. N=32 may spill to local memory. |

---

### 15.11 Compile-Time Kernel Variants (Optional)

The proposal mentions multiple shader kernels for different bin counts.
This is achievable with C++ templates or `#if` blocks:

```cpp
// In optix_device.cu
#if PHOTON_BIN_COUNT == 16
  // unrolled 16-bin loop
  #pragma unroll
  for (int k = 0; k < 16; ++k) { ... }
#elif PHOTON_BIN_COUNT == 32
  #pragma unroll
  for (int k = 0; k < 32; ++k) { ... }
#elif PHOTON_BIN_COUNT == 64
  #pragma unroll
  for (int k = 0; k < 64; ++k) { ... }
#endif
```

Since `PHOTON_BIN_COUNT` is a `constexpr int` in `config.h`, NVCC will
unroll the loop automatically without `#if` guards — a simple:
```cpp
#pragma unroll
for (int k = 0; k < PHOTON_BIN_COUNT; ++k)
```
is sufficient and selects at compile time.  No need for multiple kernel
entry points unless different bin counts are needed simultaneously at
runtime.

---

### 15.12 Summary

This proposal combines all five acceleration strategies (A1 per-pixel
cache, A2 stochastic cap, B1 guided bounces, B2 guided NEE, B3 stratified
SPP) into a unified directional bin architecture.  The key insight is that
**bins serve as the single-source-of-truth for both cached photon data AND
directional guidance**, eliminating the current redundancy (16× photon
gather) and blindness (cosine hemisphere bounce) in one architectural
change.

Expected net impact:
- **Speed: 4-8× improvement** on the photon gather phase (dominant for
  high photon counts), with bounded worst-case via stochastic cap.
- **Quality: significant variance reduction** at equal SPP, especially in
  scenes with complex indirect lighting where random cosine bounces waste
  most samples in low-contribution directions.
