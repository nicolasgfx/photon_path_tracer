# Spectral Photon-Centric Renderer — Revised Guideline v2

Audience: **GitHub Copilot** implementing/maintaining this renderer.

**Revision rationale:** The previous approach (hybrid path tracer + photon
density estimation at every camera bounce) is replaced by a *photon-centric*
architecture where photon rays carry the full transport burden and camera rays
stop at the first hit.  This simplifies the camera pass dramatically and
concentrates algorithmic sophistication in the physically-motivated photon
tracing, where coverage and convergence matter most.

Priority order (unchanged):
1. **Physical correctness** (unbiased estimators, correct PDFs, no double counting)
2. **Deterministic + debuggable** (component outputs, clear invariants)
3. **Simplicity over performance** (explicit code, correct first)

OptiX is mandatory for the GPU path.  A CPU reference renderer must exist
for validation; it computes identical results via the same algorithms, just
without GPU acceleration.

---

## 0) Non-Negotiable Invariants (Read First)

- **Never double-count direct lighting.**
  Direct illumination at camera hit-points is estimated by **NEE only**.
  Photon maps must NOT contain first-bounce deposits (lightPathDepth < 2).
- **Every Monte Carlo estimator must divide by its exact sampling PDF.**
- **Spectral bins never mix during transport.** Convert spectral → RGB only
  at output.
- **Photons store radiant flux per wavelength bin** (a power packet), not radiance.
- **CPU and GPU must produce statistically identical results** when given the
  same RNG seed and scene.  Divergence is a bug.
- **KD-tree is the canonical spatial query structure.**  Hash grid may be
  retained as a secondary GPU-optimized option but KD-tree is the reference.

---

## 1) Physical Units & Definitions

Rendering equation (spectral):

$$
L_o(x, \omega_o, \lambda) = L_e(x, \omega_o, \lambda) + \int_{\Omega} L_i(x, \omega_i, \lambda)\, f_s(x, \omega_i, \omega_o, \lambda)\, \cos\theta_i\, d\omega_i
$$

- Radiance $L$ : $[W\,/(sr\cdot m^2\cdot nm)]$
- Flux $\Phi$ : $[W\,/nm]$
- Irradiance $E$ : $[W\,/(m^2\cdot nm)]$

A stored photon represents **radiant flux** in one wavelength bin.

---

## 2) Architecture Overview — Photon-Centric Model

### 2.1 Design Philosophy

Camera rays are **cheap probes**: they find the first visible surface point
and evaluate direct lighting via NEE.  All global illumination (indirect
diffuse, caustics, color bleeding, subsurface-like effects) is carried
entirely by the photon map, which is queried at the camera hit-point.

Photon rays are the **real path tracers**: they start from light sources and
bounce through the scene with full spectral transport, sophisticated
importance sampling, and Russian roulette.  The photon distribution encodes
the complete indirect light field.

### 2.2 Render Pipeline (Single Frame)

```
1. Load scene → build BVH / OptiX GAS
2. Build emitter distribution (alias table / CDF over emissive triangles)
3. ═══ PHOTON PASS ═══  (may be precomputed — see §20)
   a. IF cached photon binary exists AND scene unchanged:
        Load photon map + spatial index from binary file
        SKIP steps b–c
   b. Emit N photons from lights (shared emitter distribution)
   c. Trace each photon through scene with full bounce logic:
      - Bounce 0: guided hemisphere/sphere coverage
      - Bounce 1+: BSDF importance sampling + Russian roulette
      - Deposit at each qualifying diffuse hit (lightPathDepth ≥ 2)
   d. Build spatial index (KD-tree on CPU, hash grid on GPU)
   e. Save photon map + spatial index to binary file (optional)
4. ═══ CAMERA PASS ═══  (first-hit only, independent of photon pass)
   a. For each pixel: trace ONE camera ray → find first hit
   b. At first hit:
      - NEE: shadow rays to light sample points (reuse emitter samples)
      - Photon gather: query KD-tree / hash grid within adaptive radius
      - Combine: L = L_direct(NEE) + L_indirect(photon density)
   c. No further bounces of camera rays.
5. Spectral → RGB conversion, tone mapping, output
```

**Key design principle:** The photon pass and camera pass are **fully
independent**.  The photon map is a static light-field snapshot that can
be precomputed to any desired quality (millions of photons, many bounces,
expensive spatial index) and saved to disk.  The camera pass loads this
map and renders interactively.  This decoupling means we can trade
arbitrarily long precomputation for rendering speed.

### 2.3 Two Implementations

| | CPU Reference | GPU (OptiX) |
|---|---|---|
| Purpose | Ground truth, validation | Interactive, production |
| Spatial index | KD-tree (arbitrary radius) | KD-tree or hash grid (fixed cell) |
| Photon transport | Identical algorithm | Identical algorithm |
| Gather | KD-tree range query | GPU-optimized query |
| Allowed tweaks | None — exact physics | Approximate kernels, capped photon counts |
| RNG | PCG, deterministic seed | PCG, same seed → same result |

---

## 3) Core Data Structures

### 3.1 Existing (kept)

- `Spectrum`: fixed-size array of `NUM_LAMBDA=32` bins (380–780 nm)
- `PhotonSoA`: SoA storage (pos, wi, normal, lambda_bin, flux)
- `Scene`, `Triangle`, `Material`, `Camera`, `BSDFSample`

### 3.2 New: KD-Tree

```cpp
struct KDTree {
    struct Node {
        int     split_axis;    // 0=x, 1=y, 2=z; -1 = leaf
        float   split_pos;     // split plane position
        int     left, right;   // child indices (internal) or photon range (leaf)
    };
    std::vector<Node>     nodes;
    std::vector<uint32_t> indices;  // photon indices in leaf order

    void build(const PhotonSoA& photons);

    // Variable-radius range query: returns all photons within `radius`
    // of `pos`.  No cell-size constraint.
    template<typename Callback>
    void query(float3 pos, float radius, const PhotonSoA& photons,
               Callback callback) const;

    // k-nearest-neighbor query (for adaptive radius)
    void knn(float3 pos, int k, const PhotonSoA& photons,
             std::vector<uint32_t>& out_indices, float& out_max_dist2) const;
};
```

The KD-tree enables **per-hitpoint adaptive radius** — each camera hit can
search with whatever radius is locally appropriate (density-adaptive,
SPPM-shrinking, or k-NN bounded).

### 3.3 Existing: Hash Grid (GPU secondary)

The hash grid is kept for the GPU path as an O(1)-build, O(cells)-query
alternative.  Its cell size is set to `2 × max_radius`, so it works for any
query radius ≤ `max_radius`.  The GPU gather kernel already handles
per-pixel variable radii within this bound (see SPPM implementation).

### 3.4 Gather Radius Strategy

| Mode | Radius per hitpoint |
|---|---|
| Fixed | `config.gather_radius` (same for all pixels) |
| k-NN adaptive | Find k nearest photons → radius = distance to k-th |
| SPPM progressive | Per-pixel shrinking radius (Hachisuka & Jensen 2009) |

All three strategies are supported by the KD-tree.  The hash grid supports
fixed and SPPM (with `max_radius` bound).

---

## 4) Emitter Distribution (Shared by Photon Emission + NEE)

Build ONE distribution over all emissive triangles.  Reuse for:
- Photon emission (light → scene)
- NEE (surface hitpoint → light)

Triangle weight:

$$
w_t = A_t \cdot \bar{L}_{e,t}
$$

Implementation: alias table or CDF.  Returns `(tri_id, p_tri)`.

### 4.1 Sample Point Reuse (NEE ↔ Photon Emission)

Each frame, pre-generate a set of **light sample points** by sampling the
emitter distribution:
1. Sample triangle `(tri_id, p_tri)` from distribution
2. Sample uniform point on triangle → `(position, normal)`
3. Store `(position, normal, tri_id, p_tri, Le)` in a sample pool

These sample points serve double duty:
- **Photon emission**: each photon starts from one of these points
- **NEE shadow rays**: camera pass sends shadow rays to these same points

This avoids redundant light sampling and ensures the set of light sample
points is shared infrastructure.

---

## 5) Photon Pass (The Real Path Tracer)

### 5.1 Photon Emission

Per photon (steps A–E):

**A) Sample emissive triangle:**
`(tri_id, p_tri)` from the shared emitter distribution.

**B) Sample uniform point on triangle (area-uniform):**

$$
\alpha = 1-\sqrt{u},\quad \beta = v\sqrt{u},\quad \gamma = 1-\alpha-\beta
$$
$$
x = \alpha v_0 + \beta v_1 + \gamma v_2
$$

PDF on area of chosen triangle: $p_{pos} = 1/A_{tri}$

**C) Sample emission direction:**
- For a Lambertian emitter: cosine-weighted hemisphere oriented around
  triangle normal.
$$
p_{dir}(\omega) = \cos\theta / \pi
$$

**D) Sample wavelength bin** proportional to emission spectrum at the
sampled point:
$$
p_{\lambda}(i \mid x) = \frac{L_e(x,\lambda_i)}{\sum_j L_e(x,\lambda_j)}
$$

**E) Compute initial photon flux** (power packet) for the chosen bin:
$$
\Phi = \frac{L_e(x,\omega,\lambda)\,\cos\theta}{p_{tri}\, p_{pos}\, p_{dir}\, p_{\lambda}}
$$

**Correctness note:** Sampling triangle weights by $A \cdot \bar{L_e}$
is fine; unbiasedness is preserved because $\Phi$ divides by the *actual*
sampling PDF.

### 5.2 Photon Bounce Logic — Sophisticated Path Tracing

Photon rays are the primary path tracers.  Their bounce strategy must
achieve **full statistical coverage** of the light transport.

#### 5.2.1 Bounce 0 (First Interaction After Emission)

**Goal:** Full coverage of the hemisphere (diffuse/glossy) or full sphere
(glass/translucent) at the first surface hit after emission.

Strategy:
- For **diffuse/glossy** surfaces: cosine-weighted hemisphere sampling with
  optional guided importance sampling from prior photon data
- For **glass/translucent** surfaces: Fresnel-weighted reflection/refraction
  with full spherical coverage
- Respect wavelength: dispersion for glass materials (wavelength-dependent IOR)

The key constraint is that bounce-0 receives the highest photon budget.
If N photons hit a surface at bounce 0, each hemisphere direction should be
well-represented statistically.  This is naturally achieved by having many
photons; no special stratification beyond BSDF sampling is needed.

#### 5.2.2 Bounce 1+ (Deeper Bounces)

Standard BSDF importance sampling:
- Lambertian: cosine hemisphere, throughput *= albedo
- Mirror: perfect reflection
- Glass: Fresnel-weighted reflect/refract
- Glossy: GGX/VNDF sampling

Russian roulette after `MIN_BOUNCES_RR` (default 3):
- Continue probability = `min(max_spectrum(throughput), RR_THRESHOLD)`
- Throughput /= continue_probability

#### 5.2.3 Photon Deposition Rule (No Double Counting)

Deposit photon when:
- Hit material is **diffuse-like** (non-delta)
- AND `lightPathDepth >= 2` (skip first hit from light)

This prevents double counting with NEE at camera hitpoints.

#### 5.2.4 Separate Maps

- **Global photon map**: diffuse deposits with `hasSpecularChain == false`
- **Caustic photon map**: diffuse deposits with `hasSpecularChain == true`
  (uses smaller gather radius for sharper features)

### 5.3 Photon Path Decorrelation (Anti-Redundancy)

**Problem:** Two photons $P_1$ and $P_2$ emitted from the same (or nearby)
emitter point $E$ in similar directions will hit nearly the same surface
patch $S$.  If both then bounce via standard BSDF sampling with independent
RNG draws, there is a non-negligible chance they pick similar bounce
directions — especially on Lambertian surfaces where the cosine-weighted
PDF has a broad lobe.  Result: two photons doing identical work (same
emitter → same surface → same bounce direction → same second surface),
while other directions from $S$ receive zero coverage.

```
  E (emitter)
  │╲
  │  ╲  nearly parallel emission
  ▼    ▼
  P1   P2    ←── both hit surface patch S
  │    │
  ▼    ▼     ←── by chance, similar BSDF bounce direction
  same region of scene
  
  ✗ WASTED: doubled coverage of one direction
  ✗ MISSED: other directions from S get zero photons
```

This is not a bias problem (the estimator is still correct on average),
but it is a **variance problem** — the photon budget is spent inefficiently.

#### 5.3.1 Cell-Stratified Bouncing

**Core idea:** Divide each hash grid cell's bounce hemisphere into strata.
Assign each photon landing in the cell to a stratum based on an arrival
counter, so successive photons in the same cell bounce in different
directions by construction.

**Mechanism:**

1. Maintain a per-cell atomic counter `cell_photon_count[cell_key]`
   (initialized to 0 at the start of each photon pass).

2. When photon $P_k$ hits a diffuse surface in cell $C$:
   ```
   my_index = atomicAdd(&cell_photon_count[hash(C)], 1)
   ```

3. Use `my_index` to select a stratum in the hemisphere:
   ```
   // Fibonacci hemisphere stratification
   num_strata = min(expected_photons_per_cell, MAX_STRATA)  // e.g. 64
   stratum    = my_index % num_strata
   
   // Stratified direction: cosine hemisphere, but restricted to stratum
   golden_angle = π × (3 - √5)
   θ = acos(sqrt((stratum + ξ₁) / num_strata))   // stratified cos²
   φ = golden_angle × stratum + ξ₂ × (2π / num_strata)  // jittered azimuth
   wi_local = (sin θ cos φ, sin θ sin φ, cos θ)
   ```

4. The PDF is still cosine-hemisphere ($\cos\theta / \pi$) because the
   stratification is just a variance-reduction technique over the same
   distribution — each stratum covers equal probability mass under the
   cosine distribution.

**Result:** If 20 photons hit cell $C$, they bounce in 20 distinct strata
of the hemisphere instead of 20 random draws that might cluster.

**Config:**
```cpp
// config.h
// Maximum number of hemisphere strata for photon bounce decorrelation.
// Higher = finer stratification, but diminishing returns beyond ~64.
// Set to 0 to disable (pure random BSDF sampling).
constexpr int DEFAULT_PHOTON_BOUNCE_STRATA = 64;
```

#### 5.3.2 RNG Decorrelation by Spatial Hash

A lighter-weight alternative (or complement): seed each photon's bounce
RNG from a hash of both its photon index AND its hit cell coordinate.
This ensures that photons in the same cell use maximally different random
sequences.

```
// At each bounce, inject the cell key into the RNG state
uint32_t cell_key = hash_cell(floor(hit_pos / cell_size));
rng.advance(cell_key * 0x9E3779B9u);  // golden-ratio hash scramble
```

This does not *guarantee* stratified coverage like §5.3.1, but makes
accidental correlation exponentially unlikely.  It is trivial to implement
(one line per bounce) and has zero memory cost.

#### 5.3.3 Photon Merging (Optional, Advanced)

When two photons in the same cell have very similar directions (within a
cone half-angle $\alpha$), merge them:

1. Keep one photon, add the other's flux to it
2. The surviving photon's direction becomes the flux-weighted average
3. The killed photon's budget is "freed" — it can be re-emitted from the
   light source in a new direction (photon splitting/recycling)

This is more complex and not recommended for the initial implementation.
It requires tracking live photons per cell and doing pairwise direction
comparisons.  Mention here for completeness.

#### 5.3.4 Summary: Which Decorrelation to Use

| Method | Complexity | Memory | Coverage guarantee | Recommended |
|--------|-----------|--------|-------------------|-------------|
| **Cell-stratified bouncing** (§5.3.1) | Medium | 1 atomic int / cell | **Yes** — deterministic | **CPU + GPU** |
| **RNG spatial hash** (§5.3.2) | Trivial | Zero | Statistical only | **GPU fallback** |
| **Photon merging** (§5.3.3) | High | Per-cell photon lists | Indirect (frees budget) | Future work |

**Recommendation:** Implement both §5.3.1 (cell-stratified) and §5.3.2
(RNG hash).  Use §5.3.1 for the CPU reference (deterministic coverage).
Use §5.3.1 on GPU where atomic counters are cheap; fall back to §5.3.2
if atomics cause contention.

### 5.4 Photon Transport Throughput Update

$$
\text{flux} \leftarrow \text{flux} \cdot \frac{f_s \cos\theta}{p(\omega)}
$$

For Lambertian with cosine sampling: factor = albedo ρ.

---

## 6) Spatial Index — KD-Tree (Primary) + Hash Grid (GPU Secondary)

### 6.1 KD-Tree

**Build**: Median-split construction O(N log N).  Split axis cycles x→y→z
or is chosen by largest-extent heuristic.  Leaf nodes contain ≤ `MAX_LEAF`
photons (8–16).

**Range query**: Recursive descent, pruning subtrees whose bounding box is
farther than `radius` from the query point.  No cell-size constraint →
**any radius works identically**.

**k-NN query**: Standard priority-queue k-NN on the KD-tree.  Returns the
k closest photons and the distance to the k-th (= adaptive radius).

Both CPU and GPU implementations.  GPU KD-tree uses a flat array layout
(left child at `2i+1`, right at `2i+2` or explicit index).

### 6.2 Hash Grid (GPU, secondary)

Retained for GPU gather when a fixed max radius is known (interactive
preview, SPPM with bounded initial radius).  Build and query as currently
implemented.

### 6.3 Density Estimation

At a diffuse camera hitpoint $x$ with outgoing direction $\omega_o$:

$$
L_{\text{photon}}(x,\omega_o,\lambda) = \frac{1}{\pi r^2}\sum_{i\in N(x)} W(d_i, r)\, \Phi_i(\lambda)\, f_s(x,\omega_i,\omega_o,\lambda)
$$

Where $W(d, r)$ is the kernel:
- **Box kernel**: $W = 1$ (flat, used by GPU fast path)
- **Epanechnikov kernel**: $W = 1 - d^2/r^2$, normalisation denominator
  becomes $\frac{\pi}{2} r^2$ instead of $\pi r^2$

### 6.4 Surface Consistency Filter

Reject photon $i$ unless ALL pass:
1. **Radial distance**: $\|x_i - x\|^2 < r^2$
2. **Plane distance**: $|n_x \cdot (x_i - x)| < \tau$
3. **Normal visibility**: $\text{dot}(n_{\text{photon}}, n_{\text{query}}) > 0$
4. **Direction consistency**: $\text{dot}(\omega_i, n_{\text{query}}) > 0$

---

## 7) Camera Pass (First-Hit Only)

Camera rays are simple probes.  No bouncing.

### 7.1 Per Pixel

1. Generate camera ray (with DOF if enabled, stratified sub-pixel jitter)
2. Trace ray → find first intersection
3. If miss: background color / environment
4. If hit emissive: add emission directly
5. If hit non-emissive surface:

### 7.2 NEE (Direct Lighting)

At the first-hit diffuse/glossy surface:
1. Sample M light sample points (from the shared emitter pool or fresh samples)
2. For each: cast shadow ray, evaluate BSDF, compute NEE contribution
3. Average over M samples

$$
L_{\text{direct}} = \frac{1}{M}\sum_{j=1}^{M} \frac{f_s(x,\omega_j,\omega_o,\lambda)\, L_e(y_j,-\omega_j,\lambda)\, \cos\theta_x}{p_\omega(\omega_j)} \cdot V(x, y_j)
$$

### 7.2.1 Shadow Ray Coverage Guarantee

**Problem:** The sample pool contains $N_{\text{pool}}$ light sample points
distributed across all emissive triangles (e.g., 1000).  Each pixel picks
$M$ of them for shadow rays (e.g., 100).  With pure importance sampling
(power-weighted), unlucky draws can cluster all $M$ picks onto a small
fraction of the total emissive area.

Example: A scene has a bright desk lamp (5% of total emitting area, 60%
of total power) and a large dim ceiling panel (95% area, 40% power).
Pure power-weighted sampling picks ~60 of 100 shadow rays toward the desk
lamp.  The ceiling panel gets ~40 samples spread over 20× more area —
badly undersampled.  Result: noisy soft shadows on surfaces lit mainly by
the ceiling.

**Solution: Coverage-aware stratified sampling** — a mixture of
importance-weighted selection and area-stratified selection, controlled by
a single parameter $c \in [0, 1]$ (`DEFAULT_NEE_COVERAGE_FRACTION`):

$$
p_{\text{select}}(i) = (1 - c) \cdot p_{\text{power}}(i) \;+\; c \cdot p_{\text{area}}(i)
$$

Where:
- $p_{\text{power}}(i)$: probability proportional to $A_t \cdot \bar{L}_{e,t}$
  (the existing emitter CDF — puts more rays where light power is high)
- $p_{\text{area}}(i)$: probability proportional to $A_t$ alone
  (pure area weighting — puts rays in proportion to emitting surface area,
  ensuring geometric coverage)
- $c$: coverage fraction. At $c = 0$, pure power-weighted (current
  behavior). At $c = 1$, pure area-weighted (maximum coverage, ignores
  brightness). At $c = 0.3$, 30% of the sampling budget goes to
  area-covering the light surfaces, 70% goes to importance sampling.

**Implementation:**

1. Build two CDFs over emissive triangles: `power_cdf` (existing) and
   `area_cdf` (new, weight = triangle area only)
2. When selecting $M$ shadow-ray targets from the pool:
   - With probability $(1 - c)$: sample from `power_cdf`
   - With probability $c$: sample from `area_cdf`
3. The PDF for the selected sample is the mixture:
   $p(i) = (1-c) \cdot p_{\text{power}}(i) + c \cdot p_{\text{area}}(i)$
4. The NEE estimator divides by this mixture PDF → **unbiased** for all $c$

**Config constant:**

```cpp
// config.h
// NEE shadow-ray coverage fraction.  0 = pure importance (power-weighted),
// 1 = pure area coverage.  0.3 is a good default: ensures every large
// emitter gets shadow rays proportional to its visible area.
constexpr float DEFAULT_NEE_COVERAGE_FRACTION = 0.3f;
```

**Why $c = 0.3$ is the recommended default:**

| $c$ | Behavior | When to use |
|-----|----------|-------------|
| 0.0 | Pure power-weighted | Scenes with 1–2 light sources of similar area |
| 0.3 | 30% coverage + 70% importance | **General-purpose default** — small lights get enough power-weighted samples, large dim lights get area coverage |
| 0.5 | Equal split | Scenes with extreme area/power imbalance |
| 1.0 | Pure area-weighted | Debug / validation only |

**Invariant:** The mixture PDF is always positive for any emissive triangle,
so no triangle can be missed entirely (except at $c = 0$ where a triangle
with zero power has $p = 0$).  At any $c > 0$, every emissive triangle has
a non-zero chance of being sampled.

### 7.3 Photon Density Estimation (Indirect Lighting)

Query the photon map (KD-tree or hash grid) at the hit position:
$$
L_{\text{indirect}} = L_{\text{photon}}(x, \omega_o, \lambda)
$$

### 7.4 Final Pixel Radiance

$$
L_{\text{pixel}} = L_{\text{emission}} + L_{\text{direct}} + L_{\text{indirect}}
$$

No camera ray continuation.  All multi-bounce transport is encoded in the
photon map.

---

## 8) SPPM (Progressive Photon Mapping)

SPPM remains supported.  The per-pixel progressive radius update is
unchanged (Hachisuka & Jensen 2009).  The KD-tree naturally handles
per-pixel variable radii (each visible point queries with its own `radius`).

### 8.1 Per-Iteration Steps

1. **Camera pass**: trace eye ray to first diffuse hit → visible point
2. **Photon pass**: emit N_p photons, trace with full bounce logic, build
   spatial index
3. **Gather pass**: for each visible point, query photons within `r_i`
4. **Progressive update**: shrink radius, accumulate flux
5. **Reconstruct**: $L = L_{\text{direct}} + \tau / (k_w \cdot r^2 \cdot K \cdot N_p)$
   where $k_w$ is the kernel normalisation constant ($\pi$ for box,
   $\pi/2$ for Epanechnikov)

---

## 9) Spectral → RGB Output

1. Integrate spectrum against CIE XYZ curves
2. Convert XYZ → linear sRGB
3. Tone map (**ACES Filmic** — decided, replaces Reinhard)
4. Gamma correct (sRGB transfer function)

Same pipeline for all component buffers (NEE, indirect, combined).
Both CPU and GPU must use the identical ACES pipeline for fair PSNR comparison.

---

## 10) Debug / Component Outputs

Required outputs per frame:
- `out_nee_direct.png` — NEE-only direct component
- `out_photon_indirect.png` — photon density only
- `out_combined.png` — sum of above
- `out_photon_caustic.png` — caustic map contribution

### 10.1 Render Modes

Expose a single `RenderMode` enum.  Each mode is a strict subset of terms:

- **First-hit debug** — normals / material ID / depth (no photon gather, no NEE)
- **DirectOnly** — NEE direct lighting only (no photon gather)
- **IndirectOnly** — photon density terms only (no NEE)
- **Combined** — direct + indirect
- **PhotonMap** — raw photon density visualization (heat map of gather count / flux)

**Correctness requirement:** if a mode disables NEE, it must also not
include any direct-light photon deposits (those must never be stored).
Conversely, IndirectOnly must show only photon-map energy — not emission.

### 10.2 File Naming Convention

When writing multiple frames/iterations, prefix with a frame counter:
- `frame_0001_out_nee_direct.png`
- `frame_0001_out_photon_indirect.png`
- `frame_0001_out_photon_caustic.png`
- `frame_0001_out_indirect_total.png`
- `frame_0001_out_combined.png`

### 10.3 Debug Viewer UX

#### Key Bindings

| Key | Action |
|-----|--------|
| F1 | Toggle photon point visualization |
| F2 | Toggle global photon map |
| F3 | Toggle caustic photon map |
| F4 | Toggle hash grid / KD-tree cell visualization |
| F5 | Toggle photon direction arrows |
| F6 | Toggle PDF display |
| F7 | Toggle gather radius sphere |
| F8 | Toggle MIS weights |
| F9 | Toggle spectral coloring mode |
| TAB | Cycle render modes |
| **P** | **Recompute photons** (re-run photon pass, rebuild spatial index, save binary) |
| 1–4 | Switch scene |
| +/− | Adjust light intensity |

#### Hover Cell Overlay

When the mouse hovers over a spatial cell, display:
- Cell coordinate (grid position or KD-tree node ID)
- Photon count in cell
- Sum flux
- Average flux
- Dominant wavelength bin and wavelength (nm)
- Gather radius
- Map type (global / caustic)

---

## 11) CPU vs GPU: Dual Implementation Design

### 11.1 Shared Code (Header-Only or Templated)

| Component | Location | Shared? |
|---|---|---|
| `Spectrum`, `PhotonSoA`, `SPPMPixel` | `src/core/` | Yes |
| `KDTree` build + query | `src/photon/kd_tree.h` | Yes (CPU); GPU gets device copy |
| `HashGrid` build | `src/photon/hash_grid.h` | Yes (CPU build); GPU query in CUDA |
| BSDF evaluate/sample/pdf | `src/bsdf/bsdf.h` | Yes (HD annotated) |
| Emitter sampling | `src/photon/emitter.h` | Yes |
| Density estimator | `src/photon/density_estimator.h` | Yes (CPU reference) |
| SPPM update/reconstruct | `src/core/sppm.h` | Yes |

### 11.2 CPU Reference Renderer (`src/renderer/`)

- `Renderer::build_photon_maps()` — trace photons, build KD-tree
- `Renderer::render_frame()` — simplified: first-hit → NEE + photon gather
- `Renderer::render_sppm()` — iterate: camera pass, photon pass, gather, update

### 11.3 GPU Renderer (`src/optix/`)

- `OptixRenderer::trace_photons()` — GPU photon emission + bounce
- `OptixRenderer::render_one_spp()` — first-hit, NEE, photon gather
- `OptixRenderer::render_sppm()` — 3-pass loop

### 11.4 Allowed GPU Tweaks

The GPU implementation may:
- Use hash grid instead of KD-tree for gather (with max_radius bound)
- Cap photons-per-cell to limit kernel divergence
- Use box kernel instead of Epanechnikov for speed
- Use lower photon counts for interactive preview

The GPU implementation must NOT:
- Change the deposition rule (lightPathDepth ≥ 2)
- Change the BSDF formulas or PDFs
- Mix wavelength bins
- Include direct lighting in the photon map

---

## 12) Integration Tests (CPU ↔ GPU Comparison)

### 12.1 Test Framework Design

The integration tests render the **binary Cornell Box** (simplest possible
scene: 5 gray/colored walls, 1 emissive quad) with both CPU and GPU
renderers using the same parameters, and compare results.

### 12.2 Test Harness

```cpp
struct IntegrationTestConfig {
    int    width         = 64;     // small for fast tests
    int    height        = 64;
    int    num_photons   = 100000;
    float  gather_radius = 0.1f;
    int    nee_samples   = 16;
    int    spp           = 16;
    int    max_bounces   = 4;
    uint32_t rng_seed    = 42;     // deterministic
};

struct IntegrationTestResult {
    FrameBuffer cpu_fb;
    FrameBuffer gpu_fb;
    FrameBuffer diff_fb;         // |cpu - gpu| per pixel
    float       psnr;            // peak signal-to-noise ratio
    float       max_pixel_error; // max absolute difference (any channel)
    float       mean_pixel_error;
    bool        passed;          // psnr > threshold
};
```

### 12.3 Test Cases

| Test | What it validates | Threshold |
|---|---|---|
| `CPU_GPU.DirectLightingMatch` | NEE-only renders agree | PSNR > 40 dB |
| `CPU_GPU.PhotonIndirectMatch` | Photon density agreement | PSNR > 30 dB |
| `CPU_GPU.CombinedMatch` | Full render agreement | PSNR > 30 dB |
| `CPU_GPU.SPPMConvergence` | SPPM after 16 iterations agrees | PSNR > 25 dB |
| `CPU_GPU.CausticMapMatch` | Caustic photon contribution agrees | PSNR > 25 dB |
| `CPU_GPU.AdaptiveRadiusMatch` | k-NN adaptive radius agrees | PSNR > 25 dB |
| `CPU_GPU.EnergyConservation` | Total energy within 5% | ratio ∈ [0.95, 1.05] |
| `CPU_GPU.NoNegativeValues` | No negative radiance in either | max(min_pixel) ≥ 0 |
| `CPU_GPU.SpectralBinIsolation` | No cross-bin contamination | exact match |
| `CPU_GPU.DifferenceImage` | Save diff image for visual inspection | always pass (logs) |

### 12.4 Output Artifacts

Each test run saves to `tests/output/integration/`:
- `cpu_combined.png`, `gpu_combined.png`
- `cpu_nee.png`, `gpu_nee.png`
- `cpu_indirect.png`, `gpu_indirect.png`
- `diff_combined.png` (amplified difference)
- `integration_report.txt` (PSNR, max error, mean error, per-component)

### 12.5 Difference Image Generation

```cpp
// Per-pixel difference with amplification for visibility
for (pixel) {
    float diff = abs(cpu_srgb[pixel] - gpu_srgb[pixel]);
    diff_srgb[pixel] = clamp(diff * amplification_factor, 0, 255);
}
```

---

## 13) Acceptance Tests

1. **Direct-only** (photons off): soft shadows converge, brightness stable
2. **Indirect-only** (NEE off): no direct-lit hotspot patterns
3. **Combined**: `combined ≈ nee_direct + photon_indirect + caustic`
4. **Cornell box**: indirect color bleeding visible
5. **Glass caustics**: concentrated patterns in caustic component
6. **CPU = GPU**: integration tests pass (PSNR thresholds)
7. **Energy conservation**: total output energy within 5% of analytic for
   simple scenes (diffuse box with known solution)

---

## 14) Common Bugs to Avoid

- Forgetting area→solid-angle Jacobian (`dist² / cos_y`) in NEE
- Using wrong cosine in Jacobian (must be emitter-side `cos_y`)
- Depositing photons at first diffuse hit from light (double counts with NEE)
- Using triangle-uniform instead of power/area-weighted emission
- Not offsetting shadow/new rays with epsilon
- Mixing wavelength bins during transport
- KD-tree: incorrect median split leading to unbalanced trees
- KD-tree: not updating max-distance during k-NN backtrack
- Hash grid: using cell_size < query radius (misses neighbors)
- CPU vs GPU: different floating-point order → use tolerance in comparisons

---

## 15) Unit Test Coverage

Existing tests (~268) remain.  New tests required:

| Category | Tests |
|---|---|
| KDTree | Build, range query, k-NN, empty, single photon, boundary |
| KDTree vs HashGrid | Same query results for same photons and radius |
| CPU Renderer (first-hit) | NEE correctness, photon gather correctness |
| Integration (CPU↔GPU) | All tests from §12.3 |
| Adaptive radius | k-NN radius matches expected density |
| SPPM + KD-tree | Progressive convergence identical to hash grid |

---

# PART II — IMPLEMENTATION PLAN

## 16) Change Impact Analysis

### 16.1 New Files

| File | Purpose | Effort |
|---|---|---|
| `src/photon/kd_tree.h` | KD-tree build + range query + k-NN | **Large** (300–500 lines) |
| `src/photon/kd_tree_device.h` | GPU-friendly flat KD-tree layout | **Medium** (200–300 lines) |
| `src/photon/photon_io.h` | Binary save/load API (§20) | **Medium** (200 lines) |
| `src/photon/photon_io.cpp` | Binary I/O implementation | **Medium** (300 lines) |
| `tests/test_kd_tree.cpp` | KD-tree unit tests | **Medium** (300 lines) |
| `tests/test_integration.cpp` | CPU↔GPU integration tests | **Large** (500–800 lines) |

### 16.2 Modified Files

| File | Changes | Effort |
|---|---|---|
| **`src/renderer/renderer.h`** | Remove multi-bounce `trace_path`, add `render_first_hit()`. Add `KDTree` member. | Small |
| **`src/renderer/renderer.cpp`** | Rewrite `trace_path()` → single-hit `render_pixel()`. Replace hash grid queries with KD-tree. Remove BSDF continuation. ~50% rewrite of 608 lines. | **Large** |
| **`src/optix/optix_device.cu`** | Simplify `__raygen__render`: remove bounce loop, keep only first-hit + NEE + photon gather. Optionally add KD-tree device query. ~30% rewrite of 2194 lines. | **Large** |
| **`src/optix/optix_renderer.h`** | Add KD-tree device buffers. Remove multi-bounce render paths. | Medium |
| **`src/optix/optix_renderer.cpp`** | Add KD-tree upload. Simplify `render_one_spp()`. ~20% change of 1790 lines. | **Large** |
| **`src/optix/launch_params.h`** | Add KD-tree device pointers. Remove unused bounce params. | Small |
| **`src/photon/density_estimator.h`** | Add KD-tree query mode alongside hash grid. | Medium |
| **`src/photon/hash_grid.h`** | No structural changes. Kept as GPU secondary. | None |
| **`src/core/config.h`** | Add KD-tree config constants. Separate `global_photon_budget` / `caustic_photon_budget`. ACES tone map flag. Rename/adjust defaults. | Small |
| **`src/core/sppm.h`** | Minor: ensure compatibility with KD-tree queries. | Small |
| **`src/main.cpp`** | Remove multi-bounce camera controls. Add `--spatial kdtree|hashgrid`, `--adaptive-radius`, `--photon-file`, `--force-recompute`, `--no-save-photons`, `--global-photons`, `--caustic-photons`. Photon cache load/save logic. P-key handler. | Medium |
| **`tests/test_main.cpp`** | Update tests that depend on multi-bounce camera paths. Add KD-tree tests. | Medium |
| **`doc/architecture/architecture.md`** | Rewrite to match new architecture. | Medium |

### 16.3 Removed / Deprecated Functionality

| Feature | Reason |
|---|---|
| Camera ray bouncing (bounce > 0) | Replaced by photon-only indirect lighting |
| MIS 3-way weighting (NEE + BSDF + photon) | Camera has no BSDF continuation |
| Photon-guided camera BSDF sampling | No camera BSDF bounces |
| `RenderMode::Full` with multi-bounce | Replaced by first-hit + NEE + photon |
| `dev_sample_guided_bounce()` in camera pass | Photon bounces use standard BSDF |
| Volume rendering (Rayleigh, Beer-Lambert, volume photons) | **Temporarily disabled** — re-enable after surface transport is validated |
| Dense cell-bin grid (`CellBinGrid`) | **Deleted entirely** (~800 lines across 4 files). Not needed — KD-tree handles spatial queries. Camera specular chains (E3) don't use cell-bin guidance. |

### 16.4 Effort Estimate

| Category | Estimated effort |
|---|---|
| KD-tree implementation + tests | 3–4 days |
| CPU renderer rewrite (first-hit only) | 1–2 days |
| GPU renderer simplification | 2–3 days |
| Integration test framework | 2–3 days |
| Photon map persistence (binary I/O) | 1–2 days |
| Wiring + config + CLI + UI | 1 day |
| Testing + debugging | 2–3 days |
| **Total** | **~13–18 days** |

---

## 17) Phased Implementation Plan

### Phase 1: KD-Tree (Foundation)
1. Implement `KDTree` in `src/photon/kd_tree.h` (header-only)
2. Unit tests: build, range query, k-NN, edge cases
3. Validate: query results match hash grid queries for same photon set + radius

### Phase 2: CPU Reference Renderer (First-Hit Only)  
1. Rewrite `Renderer::trace_path()` → `Renderer::render_pixel()`
   - Single hit, NEE, photon gather via KD-tree
2. Wire into `render_frame()` (remove bounce loop)
3. Add adaptive radius mode (k-NN)
4. Unit tests: direct-only, indirect-only, combined for Cornell box

### Phase 3: GPU Renderer (Simplified)
1. Simplify `__raygen__render` in `optix_device.cu`:
   - Remove bounce loop
   - Keep: ray trace → first hit → NEE → photon gather → output
2. Upload KD-tree to device OR keep hash grid as gather backend
3. Update `render_one_spp()` and `fill_common_params()`

### Phase 4: Integration Tests
1. Build `test_integration.cpp` framework
2. Implement all CPU↔GPU comparison tests from §12.3
3. Difference image generation and PSNR computation
4. Test with binary Cornell box

### Phase 5: Photon Map Persistence (Binary Save/Load)
1. Define binary format (see §21)
2. Implement `PhotonMap::save()` and `PhotonMap::load()` in `src/photon/photon_io.h`
3. Scene hash computation (geometry + materials + lights)
4. CLI flags: `--photon-file`, `--force-recompute`
5. Wire into `main.cpp` and `OptixRenderer`
6. UI button (P key) for recomputation
7. Unit tests: round-trip save/load correctness

### Phase 6: Polish
1. SPPM with KD-tree (verify convergence matches hash grid)
2. Adaptive radius CLI flags
3. Documentation update
4. Performance profiling

---

## 18) Proposals (Choose One Per Category)

### PROPOSAL A: Spatial Index for GPU

| Option | Description | Pros | Cons |
|---|---|---|---|
| **A1: Hash grid only** (current) | Keep existing hash grid for GPU | Zero GPU work. Proven fast. | Fixed max radius; doesn't scale to arbitrary per-pixel radii. |
| **A2: GPU KD-tree** | Upload flat KD-tree array to GPU, traverse in device code | True arbitrary radius. Matches CPU reference. | More complex device code. Potential warp divergence. ~500 lines new CUDA. |
| **A3: Hybrid** | KD-tree for CPU reference; hash grid for GPU with `max_radius` ceiling | Simple GPU code. CPU validates physics. | GPU gather has a radius cap — not truly arbitrary. |

**Recommendation: A2** — since the photon pass is precomputed offline, the
KD-tree build cost is irrelevant.  The GPU needs a KD-tree for truly
adaptive per-pixel gather radius (k-NN, SPPM).  The ~500 lines of CUDA
traversal code pay for themselves in correctness and flexibility.  The
hash grid can be retained as a fast-preview fallback.

### PROPOSAL B: Photon Bounce Strategy at Bounce 0

| Option | Description | Pros | Cons |
|---|---|---|---|
| **B1: Pure BSDF** | Standard BSDF importance sampling at bounce 0, same as deeper bounces | Simple. Already implemented. | May under-sample some directions. |
| **B2: Stratified BSDF** | Stratify the hemisphere into bins, ensure at least one photon per bin | Better coverage guarantee. | Complex. Requires many photons per hitpoint. |
| **B3: Guided + BSDF mixture** | Mix guided (from prior photon data) with BSDF | Best variance reduction. | Requires prior photon data (chicken-and-egg). Works from 2nd frame onward. |

**Recommendation: B2** — since the photon pass runs as an offline
precomputation (potentially minutes to hours), we can afford the extra
complexity of hemisphere stratification.  With millions of photons and
unlimited precompute time, B2 gives deterministic coverage guarantees
that pure BSDF sampling (B1) only achieves statistically.  B3 (guided)
can be layered on top of B2 for iterative refinement passes.

### PROPOSAL C: Adaptive Gather Radius

| Option | Description | Pros | Cons |
|---|---|---|---|
| **C1: Fixed radius** | Same radius for all pixels (current approach) | Simple. Predictable. | Over-smooths sparse regions, under-smooths dense ones. |
| **C2: k-NN radius** | Radius = distance to k-th nearest photon | Adapts to local density perfectly. | Requires k-NN query (slower than range query). k must be tuned. |
| **C3: SPPM progressive** | Per-pixel radius that shrinks over iterations | Provably convergent. Handles all densities. | Requires multiple iterations. |
| **C4: Hybrid** | Start with k-NN to set initial radius, then use SPPM to refine | Best of both worlds. | Most complex implementation. |

**Recommendation: C2 (k-NN) for both CPU and GPU** — since the photon
map is precomputed and the KD-tree is stored in the binary file, k-NN
queries at render time add negligible cost compared to the benefit of
adaptive per-pixel radius.  C3 (SPPM) remains available as a progressive
mode for iterative refinement.  C1 (fixed) is a fast-preview fallback.

### PROPOSAL D: Caustic Handling

| Option | Description | Pros | Cons |
|---|---|---|---|
| **D1: Separate caustic map** (current) | Global + caustic maps with different radii | Sharp caustics. Proven. | Two maps, two gathers per pixel. |
| **D2: Single unified map** | One map with adaptive radius (k-NN adapts to caustic density automatically) | Simpler code. k-NN naturally sharpens where density is high. | May smooth caustics in sparse regions. |

**Recommendation: D1** initially (already implemented).  Consider D2 if k-NN
adaptive radius proves sufficient.

### PROPOSAL E: Camera Ray Strategy

| Option | Description | Pros | Cons |
|---|---|---|---|
| **E1: Pure first-hit** | Camera ray stops at first surface. Mirror/glass surfaces bounce the camera ray to find the first *diffuse* hit. | Correct for mirrors/glass. Photon map handles the rest. | Need to handle specular chain in camera pass. |
| **E2: First-hit diffuse only** | Camera ray stops at first hit regardless of material type. Specular surfaces get no indirect lighting. | Simplest possible camera pass. | Mirrors/glass won't show indirect lighting correctly. |
| **E3: Bounded camera bounces** | Camera bounces up to N times (for specular chains only), stops at first diffuse. | Correct rendering of mirrors/glass/caustics. | Slightly more complex than E2 but essential for quality. |

**Recommendation: E3** with N=8 (specular-only bounces).  Camera ray follows
specular bounces (mirror, glass) until it hits a diffuse surface, then does
NEE + photon gather.  This is essential for correct mirror reflections and
glass refraction views.  Only specular bounces are allowed — no diffuse
continuation.

---

## 19) Questions & Decisions for the Project Lead

### Decided (from Proposal Selections)

| Proposal | Decision | Choice |
|---|---|---|
| **A: GPU Spatial Index** | **A2: GPU KD-tree** | Upload flat KD-tree to GPU, traverse in device code. True arbitrary per-pixel radius. ~500 lines new CUDA. |
| **B: Photon Bounce Strategy** | **B2: Stratified BSDF** | Stratify the hemisphere at bounce 0, ensure at least one photon per bin. Affordable with offline precomputation. |
| **C: Adaptive Gather Radius** | **C2: k-NN radius** | Radius = distance to k-th nearest photon. Adapts to local density. k-NN via KD-tree (A2 provides this). Default k ≈ 100. |
| **D: Caustic Handling** | **D1: Separate caustic map** | Global + caustic maps with different gather radii. Already implemented. |
| **E: Camera Ray Strategy** | **E3: Bounded camera bounces** | Camera bounces up to N=8 times for specular chains only (mirror, glass), stops at first diffuse hit → NEE + photon gather. Essential for correct reflections/refractions. |

### All Questions Decided

| Question | Decision | Detail |
|---|---|---|
| **Q4: Photon Budget Split** | **Separate config params** | Separate `--global-photons` and `--caustic-photons` CLI flags + matching `config.h` constants. Gives independent control over global and caustic map quality. |
| **Q5: Cell-Bin Grid Fate** | **(a) Remove entirely** | Delete CellBinGrid code (~800 lines across 4 files). Camera still bounces for specular chains (E3), but hash grid / KD-tree handle spatial queries — no cell-bin grid needed. Switch fully to KD-tree. |
| **Q6: Test Resolution** | **64×64** | Keep 64×64 for integration tests (already ~40 min with OpenMP). No higher-res suite. |
| **Q7: Existing Tests** | **(a) Remove obsolete** | Delete tests that no longer apply (multi-bounce camera, MIS weights, guided bouncing). New tests replace their coverage. Clean codebase over dead references. |
| **Q8: Tone Mapping** | **ACES Filmic** | Switch from Reinhard to ACES Filmic. Better highlight handling. Both CPU and GPU use the same ACES pipeline for fair PSNR comparison. |
| **Q9: Volume Rendering** | **(b) Temporarily disabled** | Disable volume rendering during the rewrite. Focus on surface transport first. Re-enable once surface transport is validated and tested. |
| **Q10: Default Render Mode** | **(c) SPPM progressive** | Default to SPPM with radius shrinkage (Hachisuka & Jensen 2009). Single-shot and simple progressive modes remain available via CLI flags. |
| **Q11: Photon Persistence** | **(c) Auto-save + hash invalidation** | Auto-save photon binary after computation. Auto-invalidate when scene hash (geometry + materials + lights) changes. Cache stored in scene folder as `photon_cache.bin`. |

---

## 20) Photon Map Persistence (Binary Save/Load)

### 20.1 Design Rationale

The photon pass and camera pass are **fully independent** (§2.2).  The
photon map is a static light-field snapshot that can take minutes to hours
to compute at high quality.  Saving it to a binary file allows:

1. **Instant startup** — skip the expensive photon tracing on subsequent
   launches for the same scene
2. **Quality iteration** — precompute with 10M+ photons, many bounces,
   stratified coverage, then render interactively
3. **Reproducibility** — binary file is a deterministic snapshot of the
   indirect light field; two users see the exact same render
4. **Camera independence** — move the camera freely without recomputing
   photons (photon map is view-independent)

### 20.2 Binary Format

```
┌─── Header (64 bytes) ────────────────────────────────┐
│  magic:        uint32   = 0x50484F54 ("PHOT")        │
│  version:      uint32   = 1                           │
│  scene_hash:   uint64   (hash of .obj + .mtl files)  │
│  num_photons:  uint32   (total photon count)          │
│  num_global:   uint32   (global map photons)          │
│  num_caustic:  uint32   (caustic map photons)         │
│  num_lambda:   uint32   (= NUM_LAMBDA, must match)    │
│  gather_radius: float32 (default gather radius used)  │
│  max_bounces:  uint32   (max bounces during trace)    │
│  rng_seed:     uint32   (seed used for generation)    │
│  flags:        uint32   (bit 0: has KD-tree,          │
│                          bit 1: has hash grid,        │
│                          bit 2: Epanechnikov kernel)  │
│  reserved:     12 bytes (zero-filled for future use)  │
├─── Photon Data (SoA, tightly packed) ────────────────┤
│  pos_x[]:      float32 × num_photons                 │
│  pos_y[]:      float32 × num_photons                 │
│  pos_z[]:      float32 × num_photons                 │
│  wi_x[]:       float32 × num_photons                 │
│  wi_y[]:       float32 × num_photons                 │
│  wi_z[]:       float32 × num_photons                 │
│  norm_x[]:     float32 × num_photons                 │
│  norm_y[]:     float32 × num_photons                 │
│  norm_z[]:     float32 × num_photons                 │
│  flux[]:       float32 × num_photons                 │
│  lambda_bin[]: uint8   × num_photons                 │
│  map_type[]:   uint8   × num_photons (0=global,1=cst)│
├─── Spatial Index (KD-tree) ──────────────────────────┤
│  num_nodes:    uint32                                 │
│  nodes[]:      KDNode × num_nodes                    │
│    KDNode = { split_axis: int8, split_pos: float32,  │
│               left: uint32, right: uint32 }  (13B)   │
│  leaf_indices[]: uint32 × num_photons                │
├─── Spatial Index (Hash Grid, optional) ──────────────┤
│  cell_size:    float32                                │
│  table_size:   uint32                                 │
│  cell_start[]: uint32 × table_size                   │
│  cell_end[]:   uint32 × table_size                   │
│  sorted_idx[]: uint32 × num_photons                  │
└──────────────────────────────────────────────────────┘
```

**Estimated file sizes** (1M photons):
- Photon data: 1M × 41 bytes ≈ 39 MB
- KD-tree: ~2M nodes × 13 bytes ≈ 25 MB
- Hash grid: ~3 MB
- **Total: ~67 MB** per million photons

### 20.3 Scene Hash

The scene hash determines whether a cached photon file is still valid.
Compute as:

```cpp
uint64_t compute_scene_hash(const Scene& scene) {
    // Hash inputs: geometry + materials + emitters + light scale
    // Changes to ANY of these invalidate the photon map
    // Camera position does NOT invalidate (photon map is view-independent)
    xxhash64 h;
    h.update(obj_file_contents);
    h.update(mtl_file_contents);
    h.update(&scene.light_scale, sizeof(float));
    return h.digest();
}
```

### 20.4 API

```cpp
// src/photon/photon_io.h (NEW FILE)
#pragma once
#include "photon/photon.h"
#include "photon/kd_tree.h"
#include "photon/hash_grid.h"
#include <filesystem>

struct PhotonCacheInfo {
    uint64_t scene_hash;
    uint32_t num_photons;
    float    gather_radius;
    uint32_t max_bounces;
    bool     has_kdtree;
    bool     has_hashgrid;
};

// Returns true if cached file exists AND scene_hash matches
bool photon_cache_valid(const std::filesystem::path& cache_path,
                        uint64_t current_scene_hash,
                        PhotonCacheInfo* info_out = nullptr);

// Save photon map + spatial indices to binary file
bool save_photon_cache(const std::filesystem::path& cache_path,
                       uint64_t scene_hash,
                       const PhotonSoA& global_photons,
                       const PhotonSoA& caustic_photons,
                       const KDTree* kdtree,        // may be nullptr
                       const HashGrid* hashgrid,    // may be nullptr
                       float gather_radius,
                       int max_bounces,
                       uint32_t rng_seed);

// Load photon map + spatial indices from binary file
bool load_photon_cache(const std::filesystem::path& cache_path,
                       uint64_t expected_scene_hash,
                       PhotonSoA& global_photons,
                       PhotonSoA& caustic_photons,
                       KDTree& kdtree,
                       HashGrid& hashgrid,
                       float& gather_radius);
```

### 20.5 Cache File Location

The binary is saved **in the scene folder** alongside the `.obj` / `.mtl`:

```
scenes/cornell_box/
    CornellBox-Original.obj
    CornellBox-Original.mtl
    photon_cache.bin          ← auto-generated
```

This keeps the cache co-located with the scene and makes it easy to
distribute a pre-lit scene (just include the `.bin` file).

### 20.6 CLI Flags

```
--photon-file <path>     Explicit photon cache file path (overrides default)
--force-recompute        Ignore cached photon file, always recompute
--no-save-photons        Do not save photon cache after computation
--photon-budget <N>      Number of photons to emit (default: 1000000)
```

### 20.7 Startup Logic

```
on_launch(scene_path, args):
    scene = load_scene(scene_path)
    scene_hash = compute_scene_hash(scene)
    cache_path = scene_path.parent / "photon_cache.bin"
    
    if args.force_recompute OR NOT photon_cache_valid(cache_path, scene_hash):
        // Full photon pass
        photon_map = trace_photons(scene, args.photon_budget, args.max_bounces)
        kdtree.build(photon_map)
        if NOT args.no_save_photons:
            save_photon_cache(cache_path, scene_hash, ...)
        status_bar("Photons computed and saved")
    else:
        load_photon_cache(cache_path, scene_hash, ...)
        status_bar("Photons loaded from cache")
    
    // Camera pass — immediate, interactive
    render(scene, photon_map, kdtree)
```

### 20.8 UI Recompute Button

The **P key** (§10.3) triggers a full photon recomputation:

```
on_key_press(P):
    status_bar("Recomputing photons...")
    photon_map = trace_photons(scene, photon_budget, max_bounces)
    kdtree.build(photon_map)
    save_photon_cache(cache_path, scene_hash, ...)
    upload_to_gpu(photon_map, kdtree)
    status_bar("Photons recomputed and saved")
    trigger_rerender()
```

This allows the user to:
1. Adjust light intensity (+/−) and see old photons, then press P to
   recompute with the new lighting
2. Increase photon budget via CLI, press P to recompute
3. Switch scenes (1–4) — photons auto-reload from cache if available;
   press P to force fresh computation

### 20.9 Impact on Integration Tests

Integration tests must support both paths:
- `TEST(Integration, PhotonCacheRoundTrip)` — save, load, verify photon
  data is bit-identical
- `TEST(Integration, SceneHashInvalidation)` — modify scene, verify cache
  is invalidated
- `TEST(Integration, CachedVsFreshRender)` — render from cache vs fresh
  photons, verify identical output

### 20.10 New Files

| File | Purpose | Effort |
|---|---|---|
| `src/photon/photon_io.h` | Binary save/load API | Medium (~200 lines) |
| `src/photon/photon_io.cpp` | Implementation (header/body split for large binary I/O) | Medium (~300 lines) |

---

## 21) Photon Directional Bins (From v1 §15)

The original guideline (v1) contains a detailed implementation plan for
**photon directional bins** — a fixed-size cache of directionally-binned
photon flux per pixel, using a Fibonacci sphere layout.  This plan remains
valuable in the v2 architecture for two purposes:

1. **Gather cache**: At each camera first-hit, bins pre-aggregate photon
   flux into N directional slots.  Multi-SPP renders read from the cache
   instead of re-querying the spatial index per sample (§15.1 of v1).

2. **Photon-guided photon bouncing**: Bins from a prior photon pass can
   guide the bounce strategy in subsequent passes (iterative refinement).

Since the photon map is precomputed and saved to binary (§20), the
directional bins can also be precomputed and included in the binary file
as an additional data block.

**Full specification:** See `revised_guideline.md` §15 (15.1–15.12) for
data structures (`PhotonBinDirs`, `PhotonBin`), config constants,
population kernel, guided bounce strategy, and unit tests.  That plan
should be implemented as a Phase 7 optimization after the core v2
architecture is stable.

---

## 22) Summary of Architectural Differences

| Aspect | Current (v1) | Proposed (v2) |
|---|---|---|
| Camera ray depth | Full path tracing (N bounces) | First hit only (specular chain N≤8 to first diffuse) |
| Indirect lighting source | Photon map + camera BSDF bouncing | Photon map only |
| MIS | 3-way (NEE + BSDF + photon) | None at camera; standard BSDF in photon rays |
| Photon bounce sophistication | Standard BSDF | Stratified BSDF (B2) — hemisphere strata at bounce 0 |
| Spatial index | Hash grid only | KD-tree primary (CPU + GPU). Hash grid as GPU fallback |
| Gather radius | Fixed or SPPM per-pixel (bounded) | k-NN adaptive (C2) per hitpoint via KD-tree |
| Photon precomputation | Every launch recomputes | Binary save/load with scene hash auto-invalidation |
| Photon budget | Single budget for all | Separate `global_photon_budget` / `caustic_photon_budget` |
| Caustic handling | Separate caustic map (D1) | Separate caustic map (D1, retained) |
| Tone mapping | Reinhard | ACES Filmic |
| Volume rendering | Rayleigh + Beer-Lambert + volume photons | Temporarily disabled (re-enable after surface validated) |
| Default render mode | Progressive accumulation | SPPM progressive (radius shrinkage) |
| CPU reference | Partial (not production quality) | Full, physically identical to GPU |
| Integration tests | None (unit tests only) | CPU↔GPU comparison suite (64×64) |
| Cell-bin grid | Used for guided camera bouncing | **Deleted** (~800 lines removed) |
| Obsolete tests | N/A | **Removed** (not DISABLED_, fully deleted) |
| Complexity | Spread across camera + photon | Concentrated in photon pass |
