# Spectral Photon-Centric Renderer — Revised Guideline v2.1

Audience: **GitHub Copilot** implementing/maintaining this renderer.

**Revision rationale (v2):** The previous approach (hybrid path tracer + photon
density estimation at every camera bounce) is replaced by a *photon-centric*
architecture where photon rays carry the full transport burden and camera rays
stop at the first hit.  This simplifies the camera pass dramatically and
concentrates algorithmic sophistication in the physically-motivated photon
tracing, where coverage and convergence matter most.

**Revision rationale (v2.1 — Surface-Aware Gather Update):** Uniform dense
3D grid experiments revealed planar blocking artifacts and energy leakage
between nearby surfaces.  Root cause: treating photons as volumetric samples
instead of surface samples.  The issue is **not** the spatial data structure —
it is the gather kernel metric.  v2.1 mandates a **tangential disk kernel**
that replaces the 3D spherical distance metric with a 2D tangent-plane metric
(§6.3).  This also revises the GPU spatial structure recommendation: the GPU
should use uniform grid / hash grid (GPU-friendly, O(1) build) with shell-
expansion k-NN and tangential metric, while the KD-tree remains the CPU
reference implementation.  KD-tree on GPU is no longer required.

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
- **CPU and GPU must produce distributionally equivalent results.**
  In normal mode, CPU and GPU renders of the same scene must pass PSNR /
  tolerance integration thresholds (§12).  Bitwise identical output is NOT
  required — GPU parallelism changes photon deposit ordering, reduction
  order, and k-NN candidate updates.  An optional **deterministic debug
  mode** (sort photon deposits by key before building indices, force
  deterministic reductions) is available for bisecting CPU↔GPU divergence.
- **KD-tree is the CPU reference spatial query structure.**  Hash grid /
  uniform grid is the GPU primary structure.  Both must use the same
  tangential disk kernel and surface consistency filters (§6.3, §6.4).
- **Gather kernel must use tangential (surface) distance, not 3D Euclidean
  distance.**  Photon mapping estimates surface irradiance, not volumetric
  radiance.  A spherical kernel gathers photons from unrelated surfaces.
  The tangential disk kernel (§6.3) is mandatory for all gather operations.
- **Single-shot density estimation is biased but consistent.**  Only SPPM
  progressive shrinking guarantees asymptotic convergence (unbiased in the
  limit).  Single-shot mode is acceptable for preview but not for ground
  truth.

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

### 1.1 Bias Clarification

Single-pass photon density estimation with fixed radius:

$$
L_{\text{indirect}}(x) = \frac{1}{\pi r^2}\sum_{i} \Phi_i\, f_s
$$

This estimator is **biased** for fixed $r$.  The bias decreases as $r \to 0$
with $N \to \infty$ (consistency), but for any finite $r$ there is smoothing
bias.

**Invariant:**
- **Single-shot mode**: biased but consistent (acceptable for preview)
- **SPPM mode**: asymptotically unbiased (Hachisuka & Jensen 2009)
  — only SPPM progressive radius shrinking guarantees convergence to the
  true solution

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
| Spatial index | KD-tree (arbitrary radius) | Uniform grid / hash grid (GPU-friendly) |
| Photon transport | Identical algorithm | Identical algorithm |
| Gather | KD-tree range query + tangential kernel | Grid shell-expansion k-NN + tangential kernel |
| Allowed tweaks | None — exact physics | Approximate kernels, capped photon counts |
| RNG | PCG, deterministic seed | PCG, same seed → same result |

**Parity requirement:** CPU and GPU must use the same **gather metric**
(tangential disk kernel, §6.3) and **surface consistency filters** (§6.4).
The spatial data structures may differ (KD-tree vs grid) but the estimator
behavior must match.

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
    // Distance metric: tangential (§6.3), NOT 3D Euclidean.
    template<typename Callback>
    void query(float3 pos, float3 normal, float radius,
               const PhotonSoA& photons, Callback callback) const;

    // k-nearest-neighbor query (for adaptive radius)
    // Distance metric: tangential (§6.3), NOT 3D Euclidean.
    void knn(float3 pos, float3 normal, int k, const PhotonSoA& photons,
             std::vector<uint32_t>& out_indices, float& out_max_dist2) const;
};
```

The KD-tree enables **per-hitpoint adaptive radius** — each camera hit can
search with whatever radius is locally appropriate (density-adaptive,
SPPM-shrinking, or k-NN bounded).

### 3.3 Existing: Hash Grid / Uniform Grid (GPU primary)

The hash grid (or uniform grid) is the **GPU primary** spatial index.  Its
O(1)-build, O(cells)-query design is ideal for GPU parallelism.  Cell size
is set to `2 × max_radius`, so it works for any query radius ≤ `max_radius`.
The GPU gather kernel handles per-pixel variable radii within this bound
(see SPPM implementation).  GPU k-NN is implemented via grid shell expansion
(§6.5) — no tree traversal needed.

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

### 4.1 Sample Point Reuse (NEE ↔ Photon Emission) — Optional Optimization

**Baseline (required for correctness):** Sample light points **directly**
per shading event from the emitter distribution.  Each photon emission and
each NEE shadow ray independently samples a triangle, a point on it, and
a direction using the canonical PDFs from §5.1.1.  This is the simplest
correct approach.

**Optional optimization — pre-generated sample pool:**

Each frame, pre-generate a set of **light sample points** by sampling the
emitter distribution:
1. Sample triangle `(tri_id, p_tri)` from distribution
2. Sample uniform point on triangle → `(position, normal)`
3. Store `(position, normal, tri_id, p_tri, Le)` in a sample pool

These sample points may serve double duty:
- **Photon emission**: each photon starts from one of these points
- **NEE shadow rays**: camera pass sends shadow rays to these same points

**Correctness constraint:** If the pool is used, the PDF of selecting pool
element $j$ must be well-defined and divided by in the estimator.  Pool
selection is **with replacement** (same point may be picked multiple times).
The effective per-sample PDF is:
$$
p_{\text{pool}}(j) = \frac{1}{N_{\text{pool}}}
$$
and the combined PDF for using pool element $j$ (which was itself drawn
from the emitter distribution) is:
$$
p_{\text{combined}} = p_{\text{pool}}(j) \cdot N_{\text{pool}} \cdot p_{tri} \cdot p_{pos} = p_{tri} \cdot p_{pos}
$$
i.e., the pool cancels out and we recover the direct sampling PDF, provided
the pool is drawn from the same distribution.

**When NOT to use the pool:** If the pool size is too small relative to
the number of emitter samples needed per frame, correlation between samples
increases variance.  Default: sample directly.  Pool is a variance-reduction
/ caching layer, not a semantics change.

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

The photon flux packet represents the share of the scene's total emitted
power carried by this single photon.  The canonical formula is:

$$
\Phi = \frac{\Phi_{\text{total},\lambda}}{N_{\text{photons}}} \cdot \frac{1}{p(\text{tri}, x, \omega, \lambda)}
$$

where the combined sampling PDF is:
$$
p(\text{tri}, x, \omega, \lambda) = p_{tri} \cdot p_{pos} \cdot p_{dir} \cdot p_{\lambda}
$$

and $\Phi_{\text{total},\lambda}$ is the scene's total emitted flux in
wavelength bin $\lambda$ (see §5.1.1 below).

**Correctness note:** Sampling triangle weights by $A \cdot \bar{L_e}$
is fine; unbiasedness is preserved because $\Phi$ divides by the *actual*
sampling PDF.  The critical requirement is that $N_{\text{photons}}$ appears
in the denominator — without it, the image brightness scales with photon
count.

### 5.1.1 Emitter Power Normalization (Canonical PDFs)

This subsection defines the exact quantities used in photon emission,
NEE (§7.2), and coverage-aware sampling (§7.2.1).  All code must use
these definitions consistently.

**Conventions:**
- $L_e(x, \lambda)$ is **spectral radiance** $[W/(sr \cdot m^2 \cdot nm)]$
  emitted by the surface at point $x$ in wavelength bin $\lambda$.  For
  Lambertian emitters, $L_e$ is independent of direction.
- `light_scale` is a scene-level scalar multiplier applied to all emissive
  materials: $L_e^{\text{scaled}} = \text{light\_scale} \times L_e^{\text{material}}$.
- The **total emitted flux per wavelength bin** for a single emissive
  triangle $t$ is:
$$
\Phi_{t,\lambda} = L_e(t, \lambda) \cdot A_t \cdot \pi
$$
  (for Lambertian emitters: integrating $L_e \cos\theta$ over the
  hemisphere gives $\pi L_e$, times the triangle area $A_t$).
- The **scene total emitted flux per bin** is:
$$
\Phi_{\text{total},\lambda} = \sum_t \Phi_{t,\lambda} = \pi \sum_t L_e(t,\lambda) \cdot A_t
$$

**Canonical PDFs (reference for all estimators):**

| PDF | Symbol | Formula | Units |
|-----|--------|---------|-------|
| Triangle selection | $p_{tri}$ | $\frac{A_t \cdot \bar{L}_{e,t}}{\sum_s A_s \cdot \bar{L}_{e,s}}$ | dimensionless |
| Point on triangle (area-uniform) | $p_{pos}$ | $1 / A_t$ | $m^{-2}$ |
| Direction (Lambertian cosine hemisphere) | $p_{dir}(\omega)$ | $\cos\theta / \pi$ | $sr^{-1}$ |
| Wavelength bin | $p_{\lambda}(i \mid x)$ | $\frac{L_e(x,\lambda_i)}{\sum_j L_e(x,\lambda_j)}$ | dimensionless |
| Combined (product) | $p_{\text{combined}}$ | $p_{tri} \cdot p_{pos} \cdot p_{dir} \cdot p_{\lambda}$ | $m^{-2} \cdot sr^{-1}$ |
| NEE: area→solid-angle Jacobian | $p_\omega$ | $p_A(y) \cdot \frac{\|y-x\|^2}{|\cos\theta_y|}$ | $sr^{-1}$ |
| NEE: coverage mixture (§7.2.1) | $p_{\text{select}}$ | $(1-c)\,p_{\text{power}} + c\,p_{\text{area}}$ | dimensionless |

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

**Delta BSDF clarification:** "non-delta" means **deposit only at non-delta
surfaces**, not "terminate at delta surfaces."  When a photon hits a delta
(mirror/glass) surface, it does NOT deposit, but it MUST continue bouncing.
Terminating at delta surfaces would prevent caustic formation.  Delta
surfaces are pass-through events for photon transport.

#### 5.2.4 Separate Maps

- **Global photon map**: diffuse deposits with `hasSpecularChain == false`
- **Caustic photon map**: diffuse deposits with `hasSpecularChain == true`
  (uses smaller gather radius for sharper features)

**Precise `hasSpecularChain` definition:** A photon has
`hasSpecularChain == true` if **at least one delta BSDF interaction**
(perfect mirror reflection or glass refraction/reflection) occurred along
its path from the light source to the current diffuse deposit point.
Once set to `true`, it remains `true` for all subsequent deposits on that
photon's path.  A photon that bounces only off diffuse/glossy surfaces
always has `hasSpecularChain == false`.

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
   cosine distribution.  **This is stratified sampling of the cosine-
   weighted hemisphere; each stratum has equal probability mass under
   $\cos\theta/\pi$; the estimator remains unbiased because we are still
   sampling from the same target PDF, just with reduced variance.**
   Implementers must NOT change the support or use a per-stratum PDF that
   differs from $\cos\theta / \pi$.

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

## 6) Spatial Index & Surface-Aware Gather

### 6.1 KD-Tree (CPU Reference)

**Build**: Median-split construction O(N log N).  Split axis cycles x→y→z
or is chosen by largest-extent heuristic.  Leaf nodes contain ≤ `MAX_LEAF`
photons (8–16).

**Range query**: Recursive descent, pruning subtrees whose bounding box is
farther than `radius` from the query point.  No cell-size constraint →
**any radius works identically**.  Distance metric is **tangential** (§6.3).

**k-NN query**: Standard priority-queue k-NN on the KD-tree.  Returns the
k closest photons and the **tangential** distance to the k-th (= adaptive
radius).  See §6.3 for the mandatory tangential metric.

The KD-tree is the **CPU reference implementation only**.  It is not
required on the GPU.  Its value is exact physics, arbitrary radius, and
deterministic behavior for validation.

### 6.2 Uniform Grid / Hash Grid (GPU Primary)

The uniform grid (or hash grid) is the **GPU primary** spatial index.
Its properties make it ideal for GPU gather:
- O(1) build time (sort + prefix sum)
- O(cells) query time
- No recursion, no priority queues
- No warp divergence from tree traversal

Cell size is set to `2 × max_radius`, so it works for any query radius
≤ `max_radius`.  GPU k-NN is implemented via **shell expansion** (§6.5).

**The GPU does NOT need a KD-tree.**  The uniform grid with tangential
kernel (§6.3) and shell-expansion k-NN (§6.5) achieves the same surface
irradiance estimation quality as the CPU KD-tree.

### 6.3 Tangential Disk Kernel (Mandatory)

**Root cause of planar blocking artifacts:** Using a full 3D spherical
distance metric for surface irradiance estimation.  Photon mapping estimates
irradiance **on surfaces** — not in volume.  A pure 3D sphere kernel gathers
photons from unrelated surfaces if they fall within $r$.  This produces:

- Discontinuities across planar walls
- Cross-surface photon leakage
- Blocking patterns at surface boundaries

**This is NOT a spatial structure problem.**  KD-tree, hash grid, and
uniform grid all exhibit the same artifact if the gather metric is spherical.
The fix is the kernel metric, not the data structure.

**Solution: Replace spherical gather with tangent-plane metric.**

Given:
- Query point $x$ with surface normal $n_x$
- Photon position $x_i$

Compute:
$$
v = x_i - x
$$
$$
d_{\text{plane}} = n_x \cdot v
$$
$$
v_{\text{tan}} = v - n_x \, d_{\text{plane}}, \quad d_{\text{tan}}^2 = \|v_{\text{tan}}\|^2
$$

The tangential distance $d_{\text{tan}}$ replaces the 3D Euclidean distance
$\|v\|$ in **all** gather operations: range queries, k-NN, density
estimation kernels.

This converts the gather kernel from a **3D sphere** to a **2D disk embedded
in the tangent plane** of the query surface.  It eliminates:

- Wall leakage (photons on opposite side of thin walls)
- Parallel surface contamination (floor photons leaking to nearby table)
- Planar blocking patterns

**Implementation:**

```cpp
// Surface-aware tangential distance computation
inline float tangential_distance2(float3 query_pos, float3 query_normal,
                                   float3 photon_pos) {
    float3 v = photon_pos - query_pos;
    float d_plane = dot(query_normal, v);
    float3 v_tan = v - query_normal * d_plane;
    return dot(v_tan, v_tan);  // tangential distance squared
}
```

**Config:**
```cpp
// config.h
// Maximum plane distance for photon acceptance.
// Photons farther than τ from the query tangent plane are rejected.
// τ interacts with ray offset epsilon, geometric vs shading normals,
// and mesh scale.  Too small → rejects valid same-surface photons;
// too large → leakage returns.
//
// Robust default: tau = max(user_tau, PLANE_TAU_EPSILON_FACTOR * ray_epsilon)
// This ensures τ is never smaller than the ray offset, which would
// reject photons deposited at the exact same surface due to floating-
// point offset.  Optionally also scale by local triangle edge length
// if available: tau = max(user_tau, k * ray_epsilon, 0.01 * avg_edge_len).
constexpr float DEFAULT_PLANE_DISTANCE_THRESHOLD = 1e-3f;
constexpr float PLANE_TAU_EPSILON_FACTOR = 10.0f;  // tau >= 10 * ray_epsilon
```

**Effective τ at runtime:**
```cpp
float effective_tau = std::max(config.plane_distance_threshold,
                               PLANE_TAU_EPSILON_FACTOR * config.ray_epsilon);
```

**Which normal to use:** The tangential metric and plane distance filter
must use **geometric normals** (face normals), not shading normals.
Shading normals from smooth interpolation can break the plane distance
test near triangle edges, causing inconsistent accept/reject behavior.
Store geometric normal per photon at deposit time.

### 6.4 Surface Consistency Filter

Reject photon $i$ unless ALL conditions pass:
1. **Tangential distance**: $d_{\text{tan},i}^2 < r^2$ (replaces spherical $\|x_i - x\|^2 < r^2$)
2. **Plane distance**: $|n_x \cdot (x_i - x)| < \tau$
3. **Normal compatibility**: $\text{dot}(n_{\text{photon}}, n_{\text{query}}) > 0$
4. **Direction consistency**: $\text{dot}(\omega_i, n_{\text{query}}) > 0$

**Critical:** Condition 1 uses **tangential** distance, not 3D Euclidean.
This is the primary fix for planar blocking artifacts.  Conditions 2–4 are
additional safety filters.

### 6.5 GPU-Friendly Adaptive k-NN via Grid Shell Expansion

Instead of tree-based k-NN (which requires recursion and priority queues —
hostile to GPU execution), use **grid shell expansion**:

```
layer = 0
while found < k:
    visit all cells at Manhattan distance == layer from query cell
    for each photon in visited cells:
        compute tangential distance (§6.3)
        apply surface consistency filter (§6.4)
        if passes: update k-NN candidate list
    layer++

radius = sqrt(max tangential distance in k-NN list)
```

**Properties:**
- No recursion
- No priority queues
- No warp divergence
- Fully GPU compatible
- Same result as tree-based k-NN (with tangential metric)

**k-NN must use tangential distance.**  If k-NN uses 3D Euclidean distance,
a nearby surface below the query point inflates the radius incorrectly (e.g.,
a table surface 2 cm above the floor: floor photons count as "near" in 3D
but are on a completely different surface).

**GPU k-NN radius policy (max_radius bound):**

The grid has a finite `max_radius` (= `cell_size / 2`).  In sparse photon
regions, k-NN may need a radius larger than `max_radius` to find k photons.
Policy: **clamp to `max_radius` and accept fewer than k photons.**

- If k-NN finds fewer than k photons within `max_radius`, the gather
  proceeds with the available photons and `radius = max_radius`.
- This introduces **bounded bias** in sparse regions (the density estimate
  uses a radius that may be smaller than the "true" k-NN radius), which
  is acceptable for the GPU interactive path.
- For ground-truth results, use the CPU KD-tree path (unbounded radius).
- `max_radius` should be chosen from scene scale heuristics:
  `max_radius = config.gather_radius` (user-specified) or
  `max_radius = 3.0 * median_photon_spacing` (density-adaptive).

```cpp
// config.h
// Maximum gather radius for GPU grid k-NN.  k-NN shell expansion
// will not exceed this radius.  In sparse regions, fewer than k
// photons may be gathered.  CPU KD-tree is unbounded.
constexpr float DEFAULT_GPU_MAX_GATHER_RADIUS = 0.5f;
```

### 6.6 Density Estimation

At a diffuse camera hitpoint $x$ with outgoing direction $\omega_o$:

$$
L_{\text{photon}}(x,\omega_o,\lambda) = \frac{1}{\pi r^2}\sum_{i\in N(x)} W(d_{\text{tan},i}, r)\, \Phi_i(\lambda)\, f_s(x,\omega_i,\omega_o,\lambda)
$$

Where $d_{\text{tan},i}$ is the tangential distance (§6.3) and $W(d, r)$
is the kernel weight:
- **Box kernel**: $W = 1$ (flat, used by GPU fast path)
- **Epanechnikov kernel**: $W = 1 - d_{\text{tan}}^2/r^2$, normalisation
  denominator becomes $\frac{\pi}{2} r^2$ instead of $\pi r^2$

**Note:** The area denominator $\pi r^2$ is the area of the tangential disk,
which is the correct normalisation for a surface irradiance estimator.

### 6.7 Optional: Triangle ID Filtering

For scenes with clean, watertight geometry, an even stronger isolation can
be achieved by storing `triangle_id` per photon:

```cpp
// In PhotonSoA, add:
std::vector<uint32_t> triangle_id;  // triangle where photon was deposited
```

At gather time, accept only photons deposited on:
- The **same triangle** as the query point, OR
- The same **smoothing group** (for smooth-shaded meshes)

This eliminates cross-wall leakage entirely, even for infinitely thin walls.
However, it can cause discontinuities at triangle boundaries on smooth
surfaces if smoothing groups are not set correctly.

**Recommendation:** Use triangle ID filtering only when geometry is clean
and watertight.  The tangential kernel (§6.3) + surface consistency filter
(§6.4) is sufficient for most scenes.

### 6.8 Spatial Structure Comparison

| Structure | Dynamic Radius | GPU Friendly | Artifact Fix | Recommendation |
|-----------|---------------|--------------|--------------|----------------|
| KD-tree | Yes | Poor (recursion, divergence) | No (needs tangential kernel) | **CPU reference only** |
| Uniform Grid | Yes (bounded by max_radius) | **Excellent** | **Yes** (with tangential kernel) | **GPU primary** |
| Hash Grid | Yes (bounded by max_radius) | **Excellent** | **Yes** (with tangential kernel) | **GPU primary** |
| LBVH | Yes | Good | Yes (with tangential kernel) | Future research option |

**Conclusion:** Kernel fix > structure change.  The tangential disk kernel
(\u00a76.3) fixes the artifact regardless of spatial structure.  The GPU should
use the most GPU-friendly structure (uniform grid / hash grid), not the most
flexible one (KD-tree).

---

## 7) Camera Pass (First-Hit Only)

Camera rays are simple probes.  They follow **specular-only bounces**
(mirror, glass) until the first diffuse surface, then evaluate NEE +
photon gather.  No diffuse continuation.

### 7.1 Per Pixel

1. Generate camera ray (with DOF if enabled, stratified sub-pixel jitter)
2. Trace ray → find first intersection
3. If miss: background color / environment
4. If hit emissive: add emission directly (multiplied by specular chain
   throughput if applicable)
5. If hit **specular (delta) surface** (mirror or glass):
   - Bounce deterministically (reflect or refract via Fresnel)
   - Multiply specular chain throughput by BSDF factor
   - Delta BSDF handling: delta PDFs cancel with the delta measure —
     treat as deterministic events, do NOT divide by a PDF
   - Continue until hitting a diffuse surface or exceeding max specular
     bounces (N=8)
6. If hit **diffuse/glossy non-emissive surface**:
   - Evaluate NEE (§7.2) and photon gather (§7.3)
   - Multiply result by accumulated specular chain throughput

### 7.1.1 Specular Chain Throughput

When the camera ray bounces through specular surfaces before reaching a
diffuse endpoint, an accumulated throughput $T$ tracks the energy
attenuation along the specular chain:

```
T = 1.0  // initial throughput
for each specular bounce:
    T *= f_s(x, wi, wo, λ) / |pdf|   // but delta PDF cancels:
    // For perfect mirror:  T *= reflectance(λ)
    // For glass refract:   T *= transmittance(λ) * (n_i/n_t)^2  (if non-symmetric)
    // For glass reflect:   T *= F(θ, λ)  (Fresnel reflectance)

// At diffuse endpoint:
L_pixel = T * (L_emission + L_direct + L_indirect)
```

**Critical:** If the specular chain throughput $T$ is omitted, mirrors and
glass views will have **wrong brightness** — the most common bug in
specular chain implementations.

**Spectral dispersion:** For glass materials with wavelength-dependent IOR,
the Fresnel reflectance, refraction direction, and transmittance are all
$\lambda$-dependent.  The BSDF sampling and PDF must be computed per
wavelength bin.  Do NOT use RGB-era BSDF code that assumes a single IOR.

### 7.2 NEE (Direct Lighting)

At the first-hit diffuse/glossy surface point $x$ with outgoing direction
$\omega_o$ and surface normal $n_x$:

1. Sample M light points (directly from emitter distribution, or from
   optional shared pool — see §4.1)
2. For each sampled light point $y_j$ on emissive triangle $t_j$:
   cast shadow ray from $x$ to $y_j$, evaluate BSDF, compute contribution
3. Average over M samples

**Canonical area-sampling form (mandatory):**

We sample points $y$ on emitter surfaces **by area** with PDF $p_A(y)$.
The area→solid-angle Jacobian and geometry term are:

$$
p_\omega(\omega) = p_A(y) \cdot \frac{\|y - x\|^2}{|\cos\theta_y|}
$$

$$
G(x, y) = \frac{|\cos\theta_x| \cdot |\cos\theta_y|}{\|x - y\|^2}
$$

where $\cos\theta_x = \omega_j \cdot n_x$ (receiver side) and
$\cos\theta_y = (-\omega_j) \cdot n_y$ (emitter side).

The direct lighting estimator in area-sampling form:

$$
L_{\text{direct}} = \frac{1}{M}\sum_{j=1}^{M} \frac{f_s(x,\omega_j,\omega_o,\lambda)\, L_e(y_j,-\omega_j,\lambda)\, G(x, y_j)}{p_A(y_j)} \cdot V(x, y_j)
$$

Equivalently, expanding $G$ and $p_A$:

$$
L_{\text{direct}} = \frac{1}{M}\sum_{j=1}^{M} \frac{f_s(x,\omega_j,\omega_o,\lambda)\, L_e(y_j,-\omega_j,\lambda)\, |\cos\theta_x|\, |\cos\theta_y|}{p_A(y_j)\, \|x - y_j\|^2} \cdot V(x, y_j)
$$

where $p_A(y_j) = p_{tri}(t_j) \cdot p_{pos}(y_j \mid t_j) = p_{tri}(t_j) / A_{t_j}$
(see §5.1.1 for $p_{tri}$).

**Common implementation error:** Dividing by $p_\omega$ (solid-angle PDF)
but using the area-form integrand, or vice versa.  Pick ONE form and be
consistent.  The area form above is recommended because we sample by area.

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

**v2.1 note:** SPPM gather must use the tangential disk kernel (§6.3).
The shrinking radius $r_i$ applies to the tangential distance, not the 3D
Euclidean distance.  This ensures SPPM convergence respects surface geometry.

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
| `KDTree` build + query | `src/photon/kd_tree.h` | Yes (CPU reference) |
| `HashGrid` build | `src/photon/hash_grid.h` | Yes (CPU build); GPU query in CUDA |
| BSDF evaluate/sample/pdf | `src/bsdf/bsdf.h` | Yes (HD annotated) |
| Emitter sampling | `src/photon/emitter.h` | Yes |
| Density estimator | `src/photon/density_estimator.h` | Yes (tangential kernel, CPU reference) |
| Surface consistency filter | `src/photon/density_estimator.h` | Yes (shared between CPU and GPU) |
| SPPM update/reconstruct | `src/core/sppm.h` | Yes |

### 11.2 CPU Reference Renderer (`src/renderer/`)

- `Renderer::build_photon_maps()` — trace photons, build KD-tree
- `Renderer::render_frame()` — simplified: first-hit → NEE + photon gather
- `Renderer::render_sppm()` — iterate: camera pass, photon pass, gather, update

### 11.3 GPU Renderer (`src/optix/`)

- `OptixRenderer::trace_photons()` — GPU photon emission + bounce
- `OptixRenderer::render_one_spp()` — first-hit, NEE, photon gather via hash grid + tangential kernel
- `OptixRenderer::render_sppm()` — 3-pass loop

### 11.4 Allowed GPU Tweaks

The GPU implementation may:
- Use hash grid / uniform grid instead of KD-tree for gather (recommended)
- Use shell-expansion k-NN instead of tree-based k-NN (§6.5)
- Cap photons-per-cell to limit kernel divergence
- Use box kernel instead of Epanechnikov for speed
- Use lower photon counts for interactive preview

The GPU implementation must NOT:
- Change the deposition rule (lightPathDepth ≥ 2)
- Change the BSDF formulas or PDFs
- Mix wavelength bins
- Include direct lighting in the photon map
- Use 3D Euclidean distance instead of tangential distance in gather
- Omit surface consistency filters (§6.4)

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
8. **No planar blocking**: tangential kernel produces smooth irradiance
   across adjacent coplanar walls (no visible discontinuities)
9. **No cross-surface leakage**: photons from floor do not contribute to
   table surface 2 cm above (tangential + plane distance filter)

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
- **Using 3D Euclidean distance instead of tangential distance in gather kernels** — causes planar blocking artifacts and cross-surface leakage
- **k-NN using 3D distance** — nearby surfaces inflate radius incorrectly (e.g. floor photons counted as "near" to table surface 2 cm above)
- **Missing plane distance filter** ($|d_{\text{plane}}| < \tau$) — allows photons from opposite side of thin walls
- CPU vs GPU: different floating-point order → use tolerance in comparisons
- CPU vs GPU: different spatial structures are OK, but **gather metric must match** (tangential kernel + surface filters)
- **Forgetting specular chain throughput** in camera E3 pass — mirrors/glass views have wrong brightness
- **Terminating photon paths at delta surfaces** instead of continuing — prevents caustic formation
- **Omitting $N_{\text{photons}}$ in photon flux denominator** — image brightness scales with photon count
- **Using NEE solid-angle PDF with area-form integrand** (or vice versa) — wrong Jacobian
- **Using shading normals for tangential metric / plane filter** — use geometric normals for stability

---

## 15) Unit Test Coverage

Existing tests (~268) remain.  New tests required:

| Category | Tests |
|---|---|
| KDTree | Build, range query, k-NN, empty, single photon, boundary |
| KDTree vs HashGrid | Same query results for same photons and radius |
| Tangential kernel | Tangential distance matches analytic for planar geometry |
| Surface filter | Cross-wall photons rejected, same-surface photons accepted |
| Shell expansion k-NN | GPU k-NN matches CPU k-NN for same photon set |
| CPU Renderer (first-hit) | NEE correctness, photon gather correctness |
| Integration (CPU↔GPU) | All tests from §12.3 |
| Adaptive radius | k-NN tangential radius matches expected density |
| SPPM + KD-tree | Progressive convergence identical to hash grid |

---

## 15.1) Implementation Conventions (Copilot Reference)

This section locks down sign conventions, data layout choices, and subtle
physics details that are easy to get subtly wrong.  All code must follow
these conventions.

### 15.1.1 `Photon.wi` Sign Convention

`wi` stored per photon is the **incoming direction toward the surface** —
i.e., the direction FROM which light arrives (pointing toward the deposit
point).  This is the convention used by BSDF evaluation:
```cpp
f_s(x, photon.wi, camera_wo, lambda)
```
where `photon.wi` is the light's incoming direction and `camera_wo` is the
outgoing direction toward the camera.

**Sign rule:** `dot(photon.wi, surface_normal) > 0` must be true for
correctly deposited photons (light arrives from the same hemisphere as the
normal).  This is enforced by surface consistency filter condition 4 (§6.4).

### 15.1.2 Stored Normal Convention

Photons store the **geometric normal** (face normal) of the surface where
they were deposited, NOT the interpolated shading normal.  Rationale:

- **Tangential metric** (§6.3) and **plane distance filter** (§6.4) require
  a stable, consistent normal.  Shading normals from smooth interpolation
  can differ significantly from the geometric normal near triangle edges,
  causing inconsistent accept/reject behavior.
- The query point also uses its **geometric normal** for the metric/filter.
- Shading normals are still used for BSDF evaluation at the query point
  (not for the spatial filter).

### 15.1.3 Kernel Normalization Constants

| Kernel | Normalization denominator | Weight $W(d, r)$ |
|--------|--------------------------|------------------|
| Box | $\pi r^2$ | $1$ |
| Epanechnikov | $\frac{\pi}{2} r^2$ | $1 - d_{\text{tan}}^2 / r^2$ |

These must be **identical** between CPU and GPU implementations.  The BSDF
$f_s$ at the query point is applied **per-photon** inside the sum, not as
an overall multiplier outside.  Ensure the kernel weight is applied to the
flux-weighted BSDF product, not separately.

### 15.1.4 Spectral Dispersion in BSDF

If IOR depends on wavelength $\lambda$ (glass, water, diamond), then:
- BSDF evaluation, sampling, AND PDF are all $\lambda$-dependent
- Fresnel reflectance $F(\theta, \lambda)$ varies per bin
- Refraction angle $\theta_t$ varies per bin (dispersion)
- Do NOT use a single scalar IOR from RGB-era code

Since photons carry a single wavelength bin, each photon's glass
interaction uses the IOR for its specific bin.  This naturally produces
spectral dispersion (rainbow caustics).

### 15.1.5 CPU/GPU Parity Contract

| Mode | Contract | How to verify |
|------|----------|---------------|
| **Normal** | Distributionally equivalent results | Integration tests: PSNR thresholds (§12.3) |
| **Deterministic debug** | Bitwise-reproducible within same platform | Sort photon deposits by key before building index; force deterministic reductions; single-threaded GPU launch |

Normal mode is the default.  Deterministic mode is opt-in via CLI flag
`--deterministic` for debugging CPU↔GPU divergence.  It is NOT required
for production renders.

---

# PART II — IMPLEMENTATION PLAN

## 16) Change Impact Analysis

### 16.1 New Files

| File | Purpose | Effort |
|---|---|---|
| `src/photon/kd_tree.h` | KD-tree build + range query + k-NN (CPU reference) | **Large** (300–500 lines) |
| ~~`src/photon/kd_tree_device.h`~~ | ~~GPU-friendly flat KD-tree layout~~ | **Removed** (v2.1: GPU uses grid, not KD-tree) |
| `src/photon/photon_io.h` | Binary save/load API (§20) | **Medium** (200 lines) |
| `src/photon/photon_io.cpp` | Binary I/O implementation | **Medium** (300 lines) |
| `tests/test_kd_tree.cpp` | KD-tree unit tests | **Medium** (300 lines) |
| `tests/test_integration.cpp` | CPU↔GPU integration tests | **Large** (500–800 lines) |

### 16.2 Modified Files

| File | Changes | Effort |
|---|---|---|
| **`src/renderer/renderer.h`** | Remove multi-bounce `trace_path`, add `render_first_hit()`. Add `KDTree` member. | Small |
| **`src/renderer/renderer.cpp`** | Rewrite `trace_path()` → single-hit `render_pixel()`. Replace hash grid queries with KD-tree. Remove BSDF continuation. ~50% rewrite of 608 lines. | **Large** |
| **`src/optix/optix_device.cu`** | Simplify `__raygen__render`: remove bounce loop, keep only first-hit + NEE + photon gather. Update gather kernel to use tangential distance. ~30% rewrite of 2194 lines. | **Large** |
| **`src/optix/optix_renderer.h`** | Remove multi-bounce render paths. Keep hash grid as primary GPU spatial index. | Medium |
| **`src/optix/optix_renderer.cpp`** | Simplify `render_one_spp()`. Update gather to tangential kernel. ~20% change of 1790 lines. | **Large** |
| **`src/optix/launch_params.h`** | Remove unused bounce params. Add plane distance threshold. | Small |
| **`src/photon/density_estimator.h`** | Update to tangential distance metric. Add surface consistency filter. | Medium |
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
| KD-tree implementation + tests (CPU only) | 2–3 days |
| Tangential kernel + surface filters | 1–2 days |
| GPU shell-expansion k-NN | 1–2 days |
| CPU renderer rewrite (first-hit only) | 1–2 days |
| GPU renderer simplification | 2–3 days |
| Integration test framework | 2–3 days |
| Photon map persistence (binary I/O) | 1–2 days |
| Wiring + config + CLI + UI | 1 day |
| Testing + debugging | 2–3 days |
| **Total** | **~13–21 days** |

---

## 17) Phased Implementation Plan

### Phase 1: KD-Tree + Tangential Kernel (Foundation)
1. Implement `KDTree` in `src/photon/kd_tree.h` (header-only, CPU reference)
2. Implement tangential disk kernel (§6.3) in density estimator
3. Implement surface consistency filter (§6.4) with tangential distance
4. Unit tests: build, range query, k-NN, edge cases
5. Unit tests: tangential distance correctness, cross-wall rejection
6. Validate: tangential kernel eliminates planar blocking on test scenes

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
2. Update GPU gather kernel to use tangential distance metric (§6.3)
3. Implement grid shell-expansion k-NN for GPU (§6.5)
4. Keep hash grid as GPU primary gather backend
5. Update `render_one_spp()` and `fill_common_params()`

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
3. Validate planar blocking fix on all test scenes
4. Re-run CPU vs GPU parity tests with tangential kernel
5. Documentation update
6. Performance profiling

---

## 18) Proposals (Choose One Per Category)

### PROPOSAL A: Spatial Index for GPU

| Option | Description | Pros | Cons |
|---|---|---|---|
| **A1: Hash grid only** (current) | Keep existing hash grid for GPU | Zero GPU work. Proven fast. | Fixed max radius; doesn't scale to arbitrary per-pixel radii. |
| **A2: GPU KD-tree** | Upload flat KD-tree array to GPU, traverse in device code | True arbitrary radius. Matches CPU reference. | More complex device code. Potential warp divergence. ~500 lines new CUDA. |
| **A3: Hybrid** | KD-tree for CPU reference; hash grid for GPU with `max_radius` ceiling | Simple GPU code. CPU validates physics. | GPU gather has a radius cap — not truly arbitrary. |

**Recommendation (v2.1 update): A3 (Hybrid)** — KD-tree for CPU reference,
uniform grid / hash grid for GPU.  The v2 recommendation of A2 (GPU KD-tree)
is superseded.  Experiments showed that planar blocking artifacts are caused
by the gather **kernel metric** (3D Euclidean vs tangential), not the spatial
structure.  With the mandatory tangential disk kernel (§6.3), the hash grid
achieves the same irradiance estimation quality as the KD-tree.  The GPU
benefits from the grid's O(1) build, no recursion, and no warp divergence.
GPU k-NN is implemented via shell expansion (§6.5) instead of tree traversal.

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
map is precomputed and the spatial index is stored in the binary file, k-NN
queries at render time add negligible cost compared to the benefit of
adaptive per-pixel radius.  **v2.1 note:** k-NN must use **tangential
distance** (§6.3), not 3D Euclidean.  On GPU, k-NN is implemented via
grid shell expansion (§6.5).  C3 (SPPM) remains available as a progressive
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
| **A: GPU Spatial Index** | **A3: Hybrid (v2.1 update)** | KD-tree for CPU reference; uniform grid / hash grid for GPU with shell-expansion k-NN (§6.5) and tangential kernel (§6.3). GPU KD-tree no longer required. |
| **B: Photon Bounce Strategy** | **B2: Stratified BSDF** | Stratify the hemisphere at bounce 0, ensure at least one photon per bin. Affordable with offline precomputation. |
| **C: Adaptive Gather Radius** | **C2: k-NN radius** | Radius = distance to k-th nearest photon (tangential distance, §6.3). CPU: k-NN via KD-tree. GPU: k-NN via grid shell expansion (§6.5). Default k ≈ 100. |
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
| Spatial index | Hash grid only | KD-tree (CPU reference). Uniform grid / hash grid (GPU primary). Tangential disk kernel mandatory (§6.3) |
| Gather radius | Fixed or SPPM per-pixel (bounded) | k-NN adaptive (C2) per hitpoint. Tangential distance metric. GPU: shell expansion (§6.5) |
| Gather kernel | 3D spherical distance | **Tangential disk kernel** (§6.3) — 2D disk on tangent plane. Fixes planar blocking artifacts |
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
