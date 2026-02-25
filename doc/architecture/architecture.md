# Architecture — Spectral Photon-Centric Renderer (v2.3)

This document describes the complete rendering pipeline, its design
rationale, mathematical foundations, and implementation details.

**Architecture version:** v2.3 (photon-centric, adaptive kNN gather)

---

## 1. Overview

The renderer implements a **photon-centric** architecture where photon rays
carry the full transport burden and camera rays stop at the first diffuse hit.
All light transport is computed over 32 discrete wavelength bins spanning
380–780 nm. The pipeline runs on the GPU via **NVIDIA OptiX 9.x** for ray
tracing and **CUDA** for auxiliary kernels, with a full **CPU reference
renderer** for validation.

### 1.1 Design Philosophy

Camera rays are **cheap probes**: they find the first visible surface point
(following specular bounces through mirrors/glass) and evaluate direct
lighting via next-event estimation (NEE). All global illumination — indirect
diffuse, caustics, colour bleeding — is carried entirely by the photon map,
which is queried at the camera hit-point.

Photon rays are the **real path tracers**: they start from light sources and
bounce through the scene with full spectral transport, sophisticated
importance sampling, and Russian roulette. The photon distribution encodes
the complete indirect light field.

### 1.2 Priority Order

1. **Physical correctness** — unbiased estimators, correct PDFs, no double counting
2. **Deterministic + debuggable** — component outputs, clear invariants
3. **Simplicity over performance** — explicit code, correct first

### 1.3 Intended Use

This is a **private research renderer** focused on physical correctness and
mathematical clarity. It is not designed for production workloads or
real-time applications.

### 1.4 Requirements

| Component       | Minimum                              | Notes                              |
|-----------------|--------------------------------------|------------------------------------|
| NVIDIA GPU      | Turing (sm_75) or newer              | Required for GPU path              |
| CUDA Toolkit    | 12.x                                 |                                    |
| NVIDIA OptiX    | 7.x or 9.x                          | `OptiX_INSTALL_DIR` must be set    |
| CMake           | 3.24                                 |                                    |
| C++ Standard    | C++17                                |                                    |
| OS              | Windows 10+ (MSVC 2022)              |                                    |
| VRAM            | 8 GB recommended                     |                                    |

---

## 2. Non-Negotiable Invariants

- **Never double-count direct lighting.** Direct illumination at camera
  hit-points is estimated by NEE only. Photon maps must NOT contain
  first-bounce deposits (`lightPathDepth < 2`).
- **Every Monte Carlo estimator must divide by its exact sampling PDF.**
- **Spectral bins never mix during transport.** Convert spectral → RGB only
  at output.
- **Photons store radiant flux per wavelength bin** (a power packet), not
  radiance.
- **CPU and GPU must produce distributionally equivalent results.** Normal
  mode uses PSNR / tolerance thresholds; bitwise identity is NOT required.
- **KD-tree is the CPU reference spatial structure.** Hash grid / uniform
  grid is the GPU primary structure.
- **Gather kernel must use tangential (surface) distance, not 3D Euclidean
  distance.** Photon mapping estimates surface irradiance, not volumetric
  radiance.
- **Single-shot density estimation is biased but consistent.** Only SPPM
  progressive shrinking guarantees asymptotic convergence.

---

## 3. Physical Units & Definitions

Rendering equation (spectral):

$$
L_o(x, \omega_o, \lambda) = L_e(x, \omega_o, \lambda) + \int_{\Omega} L_i(x, \omega_i, \lambda)\, f_s(x, \omega_i, \omega_o, \lambda)\, \cos\theta_i\, d\omega_i
$$

| Quantity     | Symbol   | Units                          |
|------------- |----------|--------------------------------|
| Radiance     | $L$      | $W/(sr \cdot m^2 \cdot nm)$    |
| Flux         | $\Phi$   | $W/nm$                         |
| Irradiance   | $E$      | $W/(m^2 \cdot nm)$             |

A stored photon represents **radiant flux** in one wavelength bin.

### 3.1 Bias Clarification

Single-pass photon density estimation with fixed radius:

$$
L_{\text{indirect}}(x) = \frac{1}{\pi r^2}\sum_{i} \Phi_i\, f_s
$$

This estimator is **biased** for fixed $r$. The bias decreases as $r \to 0$
with $N \to \infty$ (consistency).

- **Single-shot mode**: biased but consistent (acceptable for preview)
- **SPPM mode**: asymptotically unbiased (Hachisuka & Jensen 2009)

---

## 4. Pipeline Stages

```
1. Load scene → build BVH / OptiX GAS
2. Build emitter distribution (alias table / CDF over emissive triangles)
3. ═══ PHOTON PASS ═══  (may be precomputed)
   a. IF cached photon binary exists AND scene unchanged:
        Load photon map + spatial index from binary file
        SKIP steps b–d
   b. Emit N photons from lights (shared emitter distribution)
   c. Trace each photon through scene with full bounce logic:
      - Bounce 0: guided hemisphere/sphere coverage
      - Bounce 1+: BSDF importance sampling + Russian roulette
      - Glass: wavelength-dependent IOR (Cauchy dispersion), Tf filter,
        IOR stack for nested dielectrics, path flag tagging
      - Deposit at each qualifying diffuse hit (lightPathDepth ≥ 2)
   d. Build spatial index (KD-tree on CPU, hash grid on GPU)
   e. Build CellInfoCache (per-cell photon statistics, §5.5)
   f. Adaptive caustic shooting: re-emit photons toward high-CV
      caustic hotspot cells (§5.6), rebuild grids + cache if augmented
   g. Save photon map + spatial index to binary file (optional)
4. ═══ CAMERA PASS ═══  (first-hit only, independent of photon pass)
   a. For each pixel: trace ONE camera ray → find first hit
   b. Follow specular bounces (mirrors, glass) up to N=8
   c. At first diffuse hit:
      - NEE: shadow rays to light sample points
      - Photon gather: query KD-tree / hash grid within adaptive radius
        (radius from CellInfoCache, §5.5)
      - Caustic gather: skip if CellInfoCache reports zero caustics
      - Combine: L = L_direct(NEE) + L_indirect(photon density)
   d. Multiply by specular chain throughput
5. Spectral → RGB conversion (CIE XYZ), ACES filmic tone mapping, output
```

**Key design principle:** The photon pass and camera pass are **fully
independent**. The photon map is a static light-field snapshot that can be
precomputed to any desired quality (millions of photons, many bounces) and
saved to disk. The camera pass loads this map and renders interactively.

### 4.1 Scene Loading

Wavefront OBJ with MTL materials. Materials are mapped to the internal
spectral representation using `rgb_to_spectrum_reflectance()` for
diffuse/specular albedos and `blackbody_spectrum()` for emissive surfaces.

**Scene normalisation:** All non-reference scenes are scaled and translated
to fit inside the Cornell Box reference frame $([-0.5, 0.5]^3)$.

Supported material types:

| Type        | MTL Cue                | Internal Enum       |
|-------------|------------------------|---------------------|
| Lambertian  | default                | `Lambertian`        |
| Mirror      | `illum 3`, Ks > 0.99   | `Mirror`            |
| Glass       | `illum 4`, Ni > 1      | `Glass`             |
| GlossyMetal | `illum 2`, Ns > 0      | `GlossyMetal`       |
| GlossyDiel. | `illum 7`              | `GlossyDielectric`  |
| Emissive    | `Ke` present           | `Emissive`          |

### 4.2 Acceleration Structure

A single bottom-level **Geometry Acceleration Structure (GAS)** is built
from the triangle soup using `optixAccelBuild`. No instancing or multi-level
IAS is used.

### 4.3 Emitter Distribution (Shared by Photon Emission + NEE)

One distribution over all emissive triangles. Reused for both photon
emission and NEE shadow ray sampling. Triangle weight:

$$
w_t = A_t \cdot \bar{L}_{e,t}
$$

Implementation: alias table or CDF. Returns `(tri_id, p_tri)`.

---

## 5. Photon Pass (The Real Path Tracer)

### 5.1 Photon Emission

Per photon:

**A) Sample emissive triangle:** `(tri_id, p_tri)` from the shared
emitter distribution.

**B) Sample uniform point on triangle:**

$$
\alpha = 1-\sqrt{u},\quad \beta = v\sqrt{u},\quad \gamma = 1-\alpha-\beta
$$

PDF: $p_{pos} = 1/A_{tri}$

**C) Sample emission direction:** Cosine-weighted hemisphere around
triangle normal: $p_{dir}(\omega) = \cos\theta / \pi$

**D) Sample wavelength bin** proportional to emission spectrum:

$$
p_{\lambda}(i \mid x) = \frac{L_e(x,\lambda_i)}{\sum_j L_e(x,\lambda_j)}
$$

**E) Compute initial photon flux:**

$$
\Phi = \frac{\Phi_{\text{total},\lambda}}{N_{\text{photons}}} \cdot \frac{1}{p(\text{tri}, x, \omega, \lambda)}
$$

where $\Phi_{\text{total},\lambda} = \pi \sum_t L_e(t,\lambda) \cdot A_t$
and the combined PDF is $p_{tri} \cdot p_{pos} \cdot p_{dir} \cdot p_{\lambda}$.

### 5.1.1 Canonical PDFs

| PDF | Symbol | Formula | Units |
|-----|--------|---------|-------|
| Triangle selection | $p_{tri}$ | $\frac{A_t \cdot \bar{L}_{e,t}}{\sum_s A_s \cdot \bar{L}_{e,s}}$ | dimensionless |
| Point on triangle | $p_{pos}$ | $1 / A_t$ | $m^{-2}$ |
| Direction (Lambertian) | $p_{dir}(\omega)$ | $\cos\theta / \pi$ | $sr^{-1}$ |
| Wavelength bin | $p_{\lambda}$ | $\frac{L_e(x,\lambda_i)}{\sum_j L_e(x,\lambda_j)}$ | dimensionless |
| Combined | $p_{\text{combined}}$ | $p_{tri} \cdot p_{pos} \cdot p_{dir} \cdot p_{\lambda}$ | $m^{-2} \cdot sr^{-1}$ |
| NEE: area→solid-angle | $p_\omega$ | $p_A(y) \cdot \frac{\|y-x\|^2}{|\cos\theta_y|}$ | $sr^{-1}$ |
| NEE: coverage mixture | $p_{\text{select}}$ | $(1-c)\,p_{\text{power}} + c\,p_{\text{area}}$ | dimensionless |

### 5.2 Photon Bounce Logic

#### 5.2.1 Bounce 0 (First Interaction After Emission)

Full coverage of the hemisphere (diffuse/glossy) or full sphere
(glass/translucent). Strategy: cosine-weighted hemisphere sampling with
optional guided importance sampling. For glass/translucent surfaces:
Fresnel-weighted reflection/refraction with full spherical coverage.
Wavelength-dependent IOR for dispersion (Cauchy equation, §10.1).

#### 5.2.2 Bounce 1+ (Deeper Bounces)

Standard BSDF importance sampling:
- Lambertian: cosine hemisphere, throughput *= albedo
- Mirror: perfect reflection
- Glass: Fresnel-weighted reflect/refract
- Glossy: GGX/VNDF sampling

Russian roulette after `MIN_BOUNCES_RR` (default 3):
$p_{continue} = \min(\max_\lambda(T(\lambda)),\, \text{RR\_THRESHOLD})$

#### 5.2.3 Photon Deposition Rule (No Double Counting)

Deposit when:
- Hit material is **non-delta** (diffuse/glossy)
- AND `lightPathDepth >= 2` (skip first hit from light)

Delta (mirror/glass) surfaces: do NOT deposit, but MUST continue bouncing.
Terminating at delta surfaces prevents caustic formation.

#### 5.2.4 Separate Maps

- **Global photon map**: deposits with `hasSpecularChain == false`
- **Caustic photon map**: deposits with `hasSpecularChain == true`
  (uses smaller gather radius for sharper caustics)

`hasSpecularChain == true` means at least one delta BSDF interaction
occurred along the photon's path from the light source.

### 5.2.5 Photon Path Flags

Each photon carries a `path_flags` bitmask and `bounce_count`, set during
tracing and stored in the photon SoA. These flags enable downstream
optimisations (CellInfoCache, adaptive caustic shooting).

| Flag | Bit | Meaning |
|------|-----|---------|
| `PHOTON_FLAG_TRAVERSED_GLASS` | 0x01 | Path passed through at least one glass surface |
| `PHOTON_FLAG_CAUSTIC_GLASS` | 0x02 | Path is a glass caustic (specular chain → diffuse) |
| `PHOTON_FLAG_VOLUME_SEGMENT` | 0x04 | Deposited inside a participating medium |
| `PHOTON_FLAG_DISPERSION` | 0x08 | Path went through a dispersive glass material |

Flags are accumulated along the path: glass detection sets `TRAVERSED_GLASS`;
if the photon later deposits on a diffuse surface, `CAUSTIC_GLASS` is set.
Materials with `dispersion == true` trigger `DISPERSION`.

### 5.2.6 IOR Stack (Nested Dielectrics)

An `IORStack` (4 deep) tracks the current refractive index during photon
tracing. When entering glass, the material's IOR is pushed; when exiting,
it is popped. The stack returns `current_ior()` for Fresnel and Snell
calculations, handling nested glass objects correctly.

```cpp
struct IORStack {
    float stack[4] = {1.0f, 0, 0, 0};
    int   depth    = 1;
    void  push(float n);
    void  pop();
    float current_ior() const { return stack[depth - 1]; }
};
```

### 5.3 Photon Path Decorrelation

#### 5.3.1 Cell-Stratified Bouncing

Divide each hash grid cell's bounce hemisphere into strata. Assign each
photon landing in the cell to a stratum based on an arrival counter, so
successive photons in the same cell bounce in different directions.

```cpp
my_index = atomicAdd(&cell_photon_count[hash(C)], 1);
stratum  = my_index % num_strata;
// Fibonacci hemisphere stratification within stratum
golden_angle = π × (3 - √5)
θ = acos(sqrt((stratum + ξ₁) / num_strata))
φ = golden_angle × stratum + ξ₂ × (2π / num_strata)
```

The PDF remains $\cos\theta / \pi$ — this is stratified sampling of the
same distribution. The estimator remains unbiased.

**Config:** `DEFAULT_PHOTON_BOUNCE_STRATA = 64` (set to 0 to disable).

#### 5.3.2 RNG Decorrelation by Spatial Hash

Seed each photon's bounce RNG from a hash of photon index AND hit cell
coordinate:

```cpp
uint32_t cell_key = hash_cell(floor(hit_pos / cell_size));
rng.advance(cell_key * 0x9E3779B9u);
```

| Method | Complexity | Memory | Coverage guarantee | Recommended |
|--------|-----------|--------|-------------------|-------------|
| Cell-stratified bouncing | Medium | 1 atomic int / cell | Yes — deterministic | CPU + GPU |
| RNG spatial hash | Trivial | Zero | Statistical only | GPU fallback |

### 5.4 Photon Transport Throughput Update

$$
\text{flux} \leftarrow \text{flux} \cdot \frac{f_s \cos\theta}{p(\omega)}
$$

For Lambertian with cosine sampling: factor = albedo $\rho$.

### 5.5 CellInfoCache (Precomputed Per-Cell Statistics)

After the initial photon pass and spatial index build, a **CellInfoCache**
is constructed. It divides the scene into a hashed 3D grid
(`CELL_CACHE_TABLE_SIZE` = 65 536 cells) and precomputes per-cell
statistics using a single pass over all photons with Welford's online
algorithm.

#### 5.5.1 CellInfo Fields

| Field | Description |
|-------|-------------|
| `irradiance` | Sum of photon flux in cell |
| `flux_variance` | Welford variance of flux values |
| `photon_count` | Number of photons in cell |
| `density` | Photons per unit volume |
| `avg_wi` | Average incoming direction |
| `directional_spread` | `1 - |avg_wi|` (0 = collimated, 1 = isotropic) |
| `caustic_count` | Photons with `PHOTON_FLAG_CAUSTIC_GLASS` |
| `caustic_flux` | Total caustic photon flux |
| `is_caustic_hotspot` | True if caustic CV exceeds threshold |
| `glass_fraction` | Fraction of photons with `TRAVERSED_GLASS` flag |
| `avg_normal` | Average surface normal |
| `normal_variance` | Variance of normals (non-planarity) |
| `adaptive_radius` | Precomputed per-cell gather radius |
| `caustic_cv` | Coefficient of variation of caustic flux |

#### 5.5.2 Adaptive Radius

The CellInfoCache provides a **per-cell adaptive gather radius** that
scales inversely with local photon density:

$$
r_{\text{adaptive}} = r_{\text{base}} \cdot \sqrt{\frac{k_{\text{target}}}{\max(n, 1)}}
$$

Clamped to $[r_{\text{base}} \times 0.25,\; r_{\text{base}} \times 2.0]$.

#### 5.5.3 Caustic Hotspot Detection

Cells with `caustic_count ≥ 4` AND `caustic_cv > CAUSTIC_CV_THRESHOLD`
(default 0.50) are flagged as caustic hotspots. These cells are targeted
by the adaptive caustic shooting pass (§5.6).

#### 5.5.4 Usage in Camera Pass

- **Adaptive gather radius**: `get_adaptive_radius(pos)` replaces the
  fixed `config.gather_radius`.
- **Caustic skip**: If `query(pos).caustic_count == 0`, skip the caustic
  photon gather entirely (avoids wasted hash grid queries in empty regions).

Implementation: `src/core/cell_cache.h`.

### 5.6 Adaptive Caustic Shooting

After the initial photon pass and CellInfoCache build, the renderer can
optionally trace **additional targeted caustic photons** toward regions
with high caustic variance (CV). This two-phase approach concentrates
photon budget where it matters most — sharp caustic edges.

#### 5.6.1 Algorithm

```
1. Identify caustic hotspot cells from CellInfoCache
   (caustic_cv > CAUSTIC_CV_THRESHOLD)
2. For up to MAX_CAUSTIC_ITERATIONS:
   a. Emit CAUSTIC_TARGETED_FRACTION × N_caustic additional photons
   b. Append to existing caustic photon SoA
   c. Rebuild caustic hash grid and CellInfoCache
   d. If no hotspots remain (all CV < CAUSTIC_CV_TARGET), stop
```

#### 5.6.2 Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAUSTIC_TARGETED_FRACTION` | 0.30 | Extra photons per iteration (fraction of caustic budget) |
| `CAUSTIC_CV_THRESHOLD` | 0.50 | Minimum CV to qualify as hotspot |
| `CAUSTIC_CV_TARGET` | 0.20 | Target CV for convergence |
| `MAX_CAUSTIC_ITERATIONS` | 3 | Maximum refinement iterations |

Implementation: `trace_targeted_caustic_photons()` in `src/photon/emitter.h`,
integrated in `Renderer::build_photon_maps()` in `src/renderer/renderer.cpp`.

---

## 6. Spatial Index & Surface-Aware Gather

### 6.1 KD-Tree (CPU Reference)

**Build**: Median-split construction $O(N \log N)$. Split axis cycles or is
chosen by largest-extent heuristic. Leaf nodes contain $\leq$ `MAX_LEAF`
photons (8–16).

**k-NN query**: Priority-queue k-NN with tangential distance (§6.3).
Returns k closest photons and the tangential distance to the k-th
(= adaptive radius).

Implementation: `src/photon/kd_tree.h` (header-only).

### 6.2 Uniform Grid / Hash Grid (GPU Primary)

O(1) build (sort + prefix sum), O(cells) query. Cell size =
`2 × max_radius`. GPU k-NN via **max-heap in registers** (§6.5).

The GPU does NOT need a KD-tree. The uniform grid with tangential metric
and register-allocated k-NN achieves the same surface irradiance estimation
quality as the CPU KD-tree path.

Hash function:
$$
h(c_x, c_y, c_z) = (c_x \cdot 73856093 \oplus c_y \cdot 19349663 \oplus c_z \cdot 83492791) \bmod T
$$

Implementation: `src/photon/hash_grid.h` + `src/photon/hash_grid.cu`.

### 6.3 Tangential Distance Metric (Mandatory)

**Root cause of planar blocking artifacts:** Using a full 3D spherical
distance metric for surface irradiance estimation. Photon mapping estimates
irradiance **on surfaces** — not in volume. A pure 3D sphere kernel gathers
photons from unrelated surfaces if they fall within $r$.

This is NOT a spatial structure problem. KD-tree, hash grid, and uniform
grid all exhibit the same artifact if the gather metric is spherical. The
fix is the distance metric, not the data structure.

**Solution: Replace spherical distance with tangent-plane metric.**

Given query point $x$ with surface normal $n_x$ and photon position $x_i$:

$$
v = x_i - x, \quad d_{\text{plane}} = n_x \cdot v
$$
$$
v_{\text{tan}} = v - n_x \, d_{\text{plane}}, \quad d_{\text{tan}}^2 = \|v_{\text{tan}}\|^2
$$

The tangential distance $d_{\text{tan}}$ replaces the 3D Euclidean distance
in **all** gather operations: k-NN, range queries, density estimation.

```cpp
inline float tangential_distance2(float3 query_pos, float3 query_normal,
                                   float3 photon_pos) {
    float3 v = photon_pos - query_pos;
    float d_plane = dot(query_normal, v);
    float3 v_tan = v - query_normal * d_plane;
    return dot(v_tan, v_tan);
}
```

**Which normal:** The tangential metric and plane distance filter use
**geometric normals** (face normals), not shading normals, for stability
near triangle edges.

**Effective τ at runtime:**
```cpp
float effective_tau = std::max(config.plane_distance_threshold,
                               PLANE_TAU_EPSILON_FACTOR * config.ray_epsilon);
```

Implementation: `src/photon/surface_filter.h` + `src/photon/density_estimator.h`.

### 6.4 Surface Consistency Filter

Reject photon $i$ unless ALL conditions pass:
1. **Tangential distance**: $d_{\text{tan},i}^2 < r_k^2$ (kNN adaptive radius)
2. **Plane distance**: $|n_x \cdot (x_i - x)| < \tau$
3. **Normal compatibility**: $\text{dot}(n_{\text{photon}}, n_{\text{query}}) > 0$
4. **Direction consistency**: $\text{dot}(\omega_i, n_{\text{query}}) > 0$

Condition 1 uses tangential distance (the primary fix for planar blocking).
Conditions 2–4 are additional safety filters.

### 6.5 Adaptive k-NN Gather (Primary Algorithm)

The k-NN (k-nearest neighbours) gather is the **primary density estimation
algorithm** for both CPU and GPU.  It eliminates boundary bias at 90° edges
that afflicted fixed-radius approaches.

#### 6.5.1 Why k-NN?

With a **fixed gather radius** $r$, a query point near a 90° edge includes
a full tangential disk of area $\pi r^2$ as denominator, but photons only
occupy the half of the disk that corresponds to the surface.  This halves
the apparent density → dark line along the edge.

With **k-NN**, the algorithm finds exactly $K$ nearest photons.  Near an
edge, the adaptive radius $r_k$ grows automatically to compensate for the
missing half-disk, keeping the $K / (\pi r_k^2)$ density ratio
correct.

#### 6.5.2 Three-Phase Algorithm

The gather runs in three phases:

```
Phase 1 — Collect candidates
    For each photon within max_search_radius:
        compute tangential distance d_tan² (§6.3)
        apply surface consistency filter (§6.4)
        if passes: insert into max-heap of size K
                   (evict farthest if heap is full)

Phase 2 — Determine adaptive radius
    r_k² = tangential distance to the K-th nearest photon
    if fewer than K photons found: r_k² = max_search_radius²

Phase 3 — Accumulate
    For each of the K nearest photons (d_tan² < r_k²):
        w = 1                            (box weight)
        L += w · f_s(x, ωi→ωo, λ) · Φi(λ) / (N · π r_k²)
```

#### 6.5.3 Worked Example

Consider a flat floor with 500 photons uniformly distributed.  A query
point $x$ lies at the centre.  Parameters: $K = 5$, $N = 500$.

| Phase | Action | Result |
|-------|--------|--------|
| 1 | Scan photons in hash-grid cells near $x$, insert closest into max-heap | heap = $\{0.001, 0.002, 0.003, 0.005, 0.008\}$ (tangential $d^2$) |
| 2 | Read heap-top $\Rightarrow r_k^2 = 0.008$ | Adaptive radius $r_k = 0.089$ |
| 3 | Normalise: $A_k = \pi \times 0.008 = 0.0251$ | $L = \sum_{i=1}^{5} f_s \cdot \Phi_i / (N \cdot A_k)$ |

If the same query point were near a 90° wall, the photons in the wall
direction are filtered out by the surface consistency check.  Only floor
photons survive, the heap needs to reach further ($r_k$ grows), and the
larger denominator $\pi r_k^2$ compensates exactly — **no dark edge**.

#### 6.5.4 GPU Implementation (Register Max-Heap)

The GPU implementation in `optix_device.cu` uses a **register-allocated
max-heap** of size $K$ (default 100).  Key details:

- **No dynamic allocation**: `knn_d2[KNN_K]` and `knn_idx[KNN_K]` arrays
  live in registers/local memory.
- **Max-heap property**: the root (`knn_d2[0]`) is always the farthest
  photon.  New photons are accepted only if `d_tan² < knn_d2[0]`.
- **Sift-down**: standard binary heap sift-down after eviction.
- Hash-grid cell traversal is flat (no recursion, no warp divergence).
- Capped at `DEFAULT_GPU_MAX_GATHER_RADIUS` to prevent pathological search.

#### 6.5.5 CPU Implementation (std::nth_element)

The CPU path in `density_estimator.h` collects all candidates passing the
surface filter into a `std::vector`, then uses `std::nth_element` for
$O(N)$ selection of the $K$-th nearest.  This is simpler than a heap
and optimal for the CPU's sequential access pattern.

The KD-tree path uses `knn_tangential()` with a priority-queue heap for
efficient pruning during tree traversal.

#### 6.5.6 Configuration

| Parameter | Default | Role |
|-----------|---------|------|
| `DEFAULT_KNN_K` | 100 | Number of nearest neighbours per query |
| `DEFAULT_GATHER_RADIUS` | 0.05 | Max search radius — global map |
| `DEFAULT_CAUSTIC_RADIUS` | 0.025 | Max search radius — caustic map |
| `DEFAULT_GPU_MAX_GATHER_RADIUS` | 0.5 | Hard upper bound for GPU shell expansion |

The gather radii are **not** the estimation radius — they are caps that
prevent expensive search in sparse regions.  The actual estimation radius
is always $r_k$, the distance to the $K$-th nearest photon.

### 6.6 Density Estimation

At a diffuse camera hitpoint $x$ with outgoing direction $\omega_o$:

$$
L_{\text{photon}}(x,\omega_o,\lambda) = \frac{1}{\pi r_k^2}\sum_{i=1}^{K} f_s(x,\omega_i,\omega_o,\lambda)\, \Phi_i(\lambda) \,/\, N
$$

Where $r_k$ is the tangential distance to the $K$-th nearest photon
(adaptive radius from §6.5) and $N$ is the total number of emitted photons.

The BSDF $f_s$ is applied **per-photon** inside the sum, not as an
overall multiplier.

### 6.7 Gather Radius Strategy

| Mode | Radius per hitpoint |
|------|---------------------|
| k-NN adaptive (default) | Find K nearest → radius = tangential distance to K-th |
| SPPM progressive | Per-pixel shrinking radius (Hachisuka & Jensen 2009) |

Default: **k-NN adaptive** ($K = 100$) for both CPU and GPU.

### 6.8 Spatial Structure Comparison

| Structure | Dynamic Radius | GPU Friendly | Recommendation |
|-----------|---------------|--------------|----------------|
| KD-tree | Yes (unbounded) | Poor (recursion, divergence) | **CPU reference only** |
| Uniform/Hash Grid | Yes (bounded by max_radius) | **Excellent** | **GPU primary** |

**Conclusion:** Metric fix > structure change. The tangential distance
metric fixes planar blocking regardless of spatial structure. The GPU uses
the most GPU-friendly structure (uniform/hash grid with register-allocated
k-NN heap), not the most flexible one (KD-tree).

### 6.9 Optional: Triangle ID Filtering

For scenes with clean, watertight geometry, store `triangle_id` per photon
and accept only photons deposited on the same triangle or smoothing group.
The tangential metric + surface consistency filter is sufficient for most
scenes.

---

## 7. Camera Pass (First-Hit Only)

Camera rays follow **specular-only bounces** (mirror, glass) until the
first diffuse surface, then evaluate NEE + photon gather. No diffuse
continuation.

### 7.1 Per Pixel

1. Generate camera ray (with DOF if enabled, stratified sub-pixel jitter)
2. Trace ray → find first intersection
3. If miss: background colour / environment
4. If hit emissive: add emission directly (× specular chain throughput)
5. If hit **specular (delta) surface** (mirror/glass):
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

```
T = 1.0  // initial throughput
for each specular bounce:
    // For perfect mirror:  T *= reflectance(λ)
    // For glass refract:   T *= transmittance(λ) * (n_i/n_t)²
    // For glass reflect:   T *= F(θ, λ)  (Fresnel reflectance)

// At diffuse endpoint:
L_pixel = T * (L_emission + L_direct + L_indirect)
```

**Critical:** If the specular chain throughput $T$ is omitted, mirrors and
glass views will have **wrong brightness**.

**Spectral dispersion:** For glass materials with wavelength-dependent IOR,
the Fresnel reflectance, refraction direction, and transmittance are all
$\lambda$-dependent. The BSDF sampling and PDF must be computed per
wavelength bin.

### 7.2 NEE (Direct Lighting)

At the first-hit diffuse/glossy surface point $x$ with outgoing direction
$\omega_o$ and surface normal $n_x$:

1. Sample M light points from emitter distribution
2. For each sampled light point $y_j$: cast shadow ray, evaluate BSDF,
   compute contribution
3. Average over M samples

**Coverage-aware stratified sampling:**

$$
p_{\text{select}}(i) = (1 - c) \cdot p_{\text{power}}(i) + c \cdot p_{\text{area}}(i)
$$

Where $c = 0.3$ (default). Two CDFs: `power_cdf` ($A_t \cdot \bar{L}_{e,t}$)
and `area_cdf` ($A_t$ only). The mixture PDF is always positive for any
emissive triangle (at $c > 0$).

**Area-sampling estimator (mandatory):**

$$
L_{\text{direct}} = \frac{1}{M}\sum_{j=1}^{M} \frac{f_s(x,\omega_j,\omega_o,\lambda)\, L_e(y_j,-\omega_j,\lambda)\, |\cos\theta_x|\, |\cos\theta_y|}{p_A(y_j)\, \|x - y_j\|^2} \cdot V(x, y_j)
$$

where $p_A(y_j) = p_{tri}(t_j) / A_{t_j}$.

### 7.3 Photon Density Estimation (Indirect Lighting)

Query the photon map (KD-tree or hash grid) at the hit position:
$L_{\text{indirect}} = L_{\text{photon}}(x, \omega_o, \lambda)$

### 7.4 Final Pixel Radiance

$$
L_{\text{pixel}} = L_{\text{emission}} + L_{\text{direct}} + L_{\text{indirect}}
$$

No camera ray continuation. All multi-bounce transport is in the photon map.

---

## 8. SPPM (Progressive Photon Mapping)

SPPM (Hachisuka & Jensen 2009) is the **default render mode**. Each
iteration:

1. **Camera pass**: trace eye ray to first diffuse hit → visible point.
   Evaluate NEE at visible point for $L_{\text{direct}}$.
2. **Photon pass**: emit $N_p$ photons, trace with full bounce logic,
   build spatial index.
3. **Gather pass**: for each visible point, query photons within $r_i$
   using tangential disk kernel. Count $M_i$ photons and accumulate
   Epanechnikov-kernel-weighted BSDF flux.
4. **Progressive update** — per pixel:

$$
N_{\text{new}} = N_i + \alpha \cdot M_i
$$
$$
r_{\text{new}} = r_i \cdot \sqrt{\frac{N_{\text{new}}}{N_i + M_i}}
$$
$$
\tau_{\text{new}} = (\tau_i + \Phi_{\text{new}}) \cdot \left(\frac{r_{\text{new}}}{r_i}\right)^2
$$

5. **Reconstruct** (after $k$ iterations):

$$
L(x, \omega_o, \lambda) = \frac{\tau(\lambda)}{k_w \cdot r^2 \cdot k \cdot N_p}
    + \frac{L_{\text{direct}}(\lambda)}{k}
$$

   The $k_w$ is $\pi$ for box kernel, $\pi/2$ for Epanechnikov.

The shrinking radius $r_i$ applies to **tangential distance**, not 3D
Euclidean. The shrinkage parameter $\alpha \in (0,1)$ defaults to $2/3$.

### 8.1 Configuration

| Field | Default | Notes |
|-------|---------|-------|
| `sppm_iterations` | 64 | Camera+photon+gather cycles |
| `sppm_alpha` | $2/3$ | Shrinkage factor |
| `sppm_initial_radius` | 0.1 | Starting gather radius |
| `sppm_min_radius` | $10^{-5}$ | Floor clamp |

### 8.2 Implementation

- **Core types** (`src/core/sppm.h`): `SPPMPixel`, `SPPMBuffer`,
  `sppm_progressive_update()`, `sppm_reconstruct()`.
- **CPU gather** (`src/photon/density_estimator.h`): `sppm_gather()`
  — with tangential disk kernel and surface consistency filters.
- **CPU render loop** (`src/renderer/renderer.cpp`): `render_sppm()`.
- **GPU camera pass** (`src/optix/optix_device.cu`): `sppm_camera_pass()`.
- **GPU gather pass** (`src/optix/optix_device.cu`): `sppm_gather_pass()`.
- **GPU render loop** (`src/optix/optix_renderer.cpp`):
  `OptixRenderer::render_sppm()`.

### 8.3 Hash Grid Sizing for SPPM

In SPPM mode, `trace_photons()` receives a `grid_radius_override` equal to
`sppm_initial_radius`. Cell size = `2 × sppm_initial_radius`, ensuring the
neighbour search always covers the maximum per-pixel gather radius.

---

## 9. Spectral → RGB Output

1. Integrate spectrum against CIE XYZ curves (Wyman Gaussian fit)
2. Convert XYZ → linear sRGB
3. Tone map: **ACES Filmic** (replaces Reinhard)
4. Gamma correct (sRGB transfer function)

Same pipeline for all component buffers. Both CPU and GPU use the identical
ACES pipeline for fair PSNR comparison.

---

## 10. BSDF Models

| Model      | $f_s$                            | Sampling PDF                     |
|------------|----------------------------------|----------------------------------|
| Lambertian | $K_d / \pi$                      | $\cos\theta / \pi$               |
| Mirror     | ideal specular reflection         | delta distribution                |
| Glass      | Fresnel-weighted reflect/refract  | Schlick approximation             |
| Glossy     | GGX microfacet (Cook-Torrance)   | VNDF sampling (Heitz 2018)        |

All BSDF evaluations are spectral: albedo $K_d(\lambda)$ or $K_s(\lambda)$
is evaluated per-bin. For glass with wavelength-dependent IOR, Fresnel
reflectance and refraction angle vary per bin (spectral dispersion).

### 10.1 Chromatic Dispersion (Cauchy Equation)

Glass materials optionally support wavelength-dependent index of refraction
via the **Cauchy dispersion model**:

$$
n(\lambda) = A + \frac{B}{\lambda^2}
$$

where $\lambda$ is in nanometers. Default constants (crown glass):
$A = 1.5046$, $B = 4200\,\text{nm}^2$.

#### Material Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `Tf` | `Spectrum` | 1.0 (all bins) | Spectral transmittance filter (glass colour) |
| `cauchy_A` | `float` | 1.5046 | Cauchy A coefficient |
| `cauchy_B` | `float` | 4200.0 | Cauchy B coefficient (nm²) |
| `dispersion` | `bool` | false | Enable wavelength-dependent IOR |

#### `ior_at_lambda(float lambda_nm)`

Returns `cauchy_A + cauchy_B / (lambda_nm * lambda_nm)` when `dispersion`
is true; otherwise returns the constant `ior` field.

#### Per-Wavelength Fresnel in Glass BSDF

When `mat.dispersion` is true, `glass_sample()` evaluates Fresnel
reflectance **per wavelength bin** using `mat.ior_at_lambda(lambda_of_bin(b))`.
This produces wavelength-dependent reflection/refraction splitting — the
physical basis of prismatic rainbows and chromatic caustics.

The glass BSDF also applies the material's **transmittance filter** `Tf` to
both reflected and refracted flux, enabling coloured glass (e.g., stained
glass windows).

```cpp
// Per-bin Fresnel when dispersion is enabled:
for (int b = 0; b < NUM_LAMBDA; ++b) {
    float n_b = mat.ior_at_lambda(lambda_of_bin(b));
    float F_b = fresnel_schlick(cos_theta, n_b);
    sample.weight.value[b] *= mat.Tf.value[b] * (reflect ? F_b : (1 - F_b));
}
```

The legacy `glass_sample(float3 wo, float ior, PCGRng& rng)` overload is
preserved for backward compatibility.

---

## 11. Debug / Component Outputs

### 11.1 Output Files

| File | Contents |
|------|----------|
| `out_nee_direct.png` | NEE-only direct component |
| `out_photon_indirect.png` | Photon density only |
| `out_combined.png` | Sum of above |
| `out_photon_caustic.png` | Caustic map contribution |

Multi-frame naming: `frame_NNNN_out_*.png`

### 11.2 Render Modes

| Mode | Description |
|------|-------------|
| First-hit debug | Normals / material ID / depth (no photon gather, no NEE) |
| DirectOnly | NEE direct lighting only (no photon gather) |
| IndirectOnly | Photon density terms only (no NEE) |
| Combined | Direct + indirect |
| PhotonMap | Raw photon density visualisation (heat map) |

### 11.3 Debug Viewer Key Bindings

| Key | Action |
|-----|--------|
| F1 | Toggle photon point visualisation |
| F2 | Toggle global photon map |
| F3 | Toggle caustic photon map |
| F4 | Toggle hash grid / KD-tree cell visualisation |
| F5 | Toggle photon direction arrows |
| F6 | Toggle PDF display |
| F7 | Toggle gather radius sphere |
| F8 | Toggle MIS weights |
| F9 | Toggle spectral colouring mode |
| TAB | Cycle render modes |
| **P** | Recompute photons (re-run photon pass, rebuild spatial index, save binary) |
| 1–4 | Switch scene |
| +/− | Adjust light intensity |
| W/A/S/D | Camera movement |
| Space/Ctrl | Camera up/down |
| Mouse | Look around |
| M | Toggle mouse capture |
| ESC | Cancel render → release mouse → quit (3-tier) |
| Q | Quit immediately |

### 11.4 Hover Cell Overlay

When hovering over a spatial cell with mouse released and map toggle enabled
(F2/F3): cell coordinate, photon count, sum/average flux, dominant
wavelength, gather radius, map type (global / caustic).

---

## 12. CPU vs GPU: Dual Implementation

### 12.1 Two Implementations

| | CPU Reference | GPU (OptiX) |
|---|---|---|
| Purpose | Ground truth, validation | Interactive, production |
| Spatial index | KD-tree (arbitrary radius) | Uniform grid / hash grid |
| Photon transport | Identical algorithm | Identical algorithm |
| Gather | KD-tree range query + tangential kernel | Grid shell-expansion k-NN + tangential kernel |
| Allowed tweaks | None — exact physics | Approximate kernels, capped photon counts |
| RNG | PCG, deterministic seed | PCG, same seed → same result |

### 12.2 Shared Code (Header-Only or Templated)

| Component | Location | Shared? |
|---|---|---|
| `Spectrum`, `PhotonSoA`, `SPPMPixel` | `src/core/` | Yes |
| KD-tree build + query | `src/photon/kd_tree.h` | Yes (CPU reference) |
| Hash grid build | `src/photon/hash_grid.h` | Yes (CPU build); GPU query in CUDA |
| BSDF evaluate/sample/pdf | `src/bsdf/bsdf.h` | Yes |
| Emitter sampling | `src/photon/emitter.h` | Yes |
| Density estimator + surface filter | `src/photon/density_estimator.h` | Yes |
| SPPM update/reconstruct | `src/core/sppm.h` | Yes |

### 12.3 CPU Reference Renderer

- `Renderer::build_photon_maps()` — trace photons, build KD-tree
- `Renderer::render_frame()` — first-hit → NEE + photon gather
- `Renderer::render_sppm()` — iterate: camera pass, photon pass, gather, update

### 12.4 GPU Renderer

- `OptixRenderer::trace_photons()` — GPU photon emission + bounce
- `OptixRenderer::render_one_spp()` — first-hit, NEE, photon gather via hash grid + tangential kernel
- `OptixRenderer::render_sppm()` — 3-pass loop

### 12.5 Parity Contract

| Mode | Contract | Verification |
|------|----------|--------------|
| **Normal** | Distributionally equivalent results | Integration tests: PSNR thresholds |
| **Deterministic debug** | Bitwise-reproducible within same platform | `--deterministic` flag |

### 12.6 Allowed GPU Tweaks

The GPU implementation **may**:
- Use hash grid / uniform grid instead of KD-tree
- Use shell-expansion k-NN instead of tree-based k-NN
- Cap photons-per-cell to limit kernel divergence
- Use box kernel instead of Epanechnikov
- Use lower photon counts for interactive preview

The GPU implementation **must NOT**:
- Change the deposition rule (`lightPathDepth ≥ 2`)
- Change the BSDF formulas or PDFs
- Mix wavelength bins
- Include direct lighting in the photon map
- Use 3D Euclidean distance instead of tangential distance in gather
- Omit surface consistency filters

---

## 13. Integration Tests (CPU ↔ GPU)

### 13.1 Test Scene

Binary Cornell Box (5 grey/coloured walls, 1 emissive quad). Resolution:
64×64 for fast testing.

### 13.2 Test Harness

```cpp
struct IntegrationTestConfig {
    int    width         = 64;
    int    height        = 64;
    int    num_photons   = 100000;
    float  gather_radius = 0.1f;
    int    nee_samples   = 16;
    int    spp           = 16;
    int    max_bounces   = 4;
    uint32_t rng_seed    = 42;
};
```

### 13.3 Test Cases

| Test | What it validates | Threshold |
|---|---|---|
| `CPU_GPU.DirectLightingMatch` | NEE-only renders agree | PSNR > 40 dB |
| `CPU_GPU.PhotonIndirectMatch` | Photon density agreement | PSNR > 30 dB |
| `CPU_GPU.CombinedMatch` | Full render agreement | PSNR > 30 dB |
| `CPU_GPU.SPPMConvergence` | SPPM after 16 iterations | PSNR > 25 dB |
| `CPU_GPU.CausticMapMatch` | Caustic photon contribution | PSNR > 25 dB |
| `CPU_GPU.AdaptiveRadiusMatch` | k-NN adaptive radius | PSNR > 25 dB |
| `CPU_GPU.EnergyConservation` | Total energy within 5% | ratio ∈ [0.95, 1.05] |
| `CPU_GPU.NoNegativeValues` | No negative radiance | max(min_pixel) ≥ 0 |
| `CPU_GPU.SpectralBinIsolation` | No cross-bin contamination | exact match |
| `CPU_GPU.DifferenceImage` | Save diff image for inspection | always pass |

### 13.4 Output Artifacts

Each run saves to `tests/output/integration/`:
- `cpu_combined.png`, `gpu_combined.png`
- `cpu_nee.png`, `gpu_nee.png`
- `cpu_indirect.png`, `gpu_indirect.png`
- `diff_combined.png` (amplified difference)
- `integration_report.txt`

---

## 14. Acceptance Tests

1. **Direct-only**: soft shadows converge, brightness stable
2. **Indirect-only**: no direct-lit hotspot patterns
3. **Combined**: `combined ≈ nee_direct + photon_indirect + caustic`
4. **Cornell box**: indirect colour bleeding visible
5. **Glass caustics**: concentrated patterns in caustic component
6. **CPU = GPU**: integration tests pass (PSNR thresholds)
7. **Energy conservation**: within 5% of analytic for simple scenes
8. **No planar blocking**: smooth irradiance across coplanar walls
9. **No cross-surface leakage**: tangential + plane distance filter working

---

## 15. Adaptive Sampling

Screen-noise adaptive sampling concentrates samples in high-variance regions
and skips converged pixels once estimated noise falls below a configurable
threshold.

### 15.1 Noise Metric

Per-pixel relative standard error of CIE Y (luminance) of the NEE
direct-only signal:

$$
r_i = \frac{\mathrm{se}_i}{|\mu_i| + \varepsilon}, \quad \varepsilon = 10^{-4}
$$

The active mask uses a neighbourhood maximum over a $(2R+1) \times (2R+1)$
window. A pixel is active if $r_i^{\text{nbr}} > \tau$.

### 15.2 Sampling Policy

| Phase | Pass range | Behaviour |
|-------|------------|-----------|
| Warmup | 0 to `min_spp - 1` | All pixels active |
| Adaptive | `min_spp` to `max_spp` | Mask recomputed; converged pixels skipped |

### 15.3 Configuration

| Field | Default | Notes |
|-------|---------|-------|
| `adaptive_sampling` | `false` | Opt-in |
| `adaptive_min_spp` | 4 | Warmup passes |
| `adaptive_max_spp` | 0 | 0 → inherits `samples_per_pixel` |
| `adaptive_update_interval` | 1 | Mask refresh period |
| `adaptive_threshold` | 0.02 | 2% relative noise target |
| `adaptive_radius` | 1 | Neighbourhood half-width |

### 15.4 Implementation

- **GPU**: Three per-pixel buffers (`d_lum_sum_`, `d_lum_sum2_`,
  `d_active_mask_`) in `OptixRenderer`. The `k_update_mask` CUDA kernel in
  `src/optix/adaptive_sampling.cu` computes the mask.
- **CPU**: Equivalent buffers in `FrameBuffer` with matching logic.

---

## 16. Photon Map Persistence (Binary Save/Load)

### 16.1 Design

The photon pass and camera pass are fully independent. The photon map is a
static light-field snapshot that can take minutes to hours to compute.
Saving it enables:

1. **Instant startup** — skip expensive tracing on subsequent launches
2. **Quality iteration** — precompute with 10M+ photons, render interactively
3. **Reproducibility** — deterministic snapshot of indirect light field
4. **Camera independence** — move camera freely without recomputing

### 16.2 Binary Format

```
┌─── Header (64 bytes) ──────────────────────────────────────────┐
│  magic: "PHOT", version, scene_hash, num_photons,              │
│  num_global, num_caustic, num_lambda, gather_radius,           │
│  max_bounces, rng_seed, flags, reserved                        │
├─── Photon Data (SoA) ─────────────────────────────────────────┤
│  pos_x/y/z, wi_x/y/z, norm_x/y/z, flux, lambda_bin, map_type │
├─── KD-tree Index ───────────────────────────────────────────────┤
│  nodes[], leaf_indices[]                                        │
├─── Hash Grid Index (optional) ──────────────────────────────────┤
│  cell_size, table_size, cell_start[], cell_end[], sorted_idx[] │
└─────────────────────────────────────────────────────────────────┘
```

~67 MB per million photons.

### 16.3 Scene Hash

Hash inputs: geometry + materials + emitters + `light_scale`. Camera
position does NOT invalidate. Uses xxhash64 over .obj + .mtl file contents.

### 16.4 Cache File Location

Saved in the scene folder alongside `.obj` / `.mtl`:
```
scenes/cornell_box/
    CornellBox-Original.obj
    CornellBox-Original.mtl
    photon_cache.bin          ← auto-generated
```

### 16.5 CLI Flags

```
--photon-file <path>     Explicit photon cache file path
--force-recompute        Ignore cached photon file
--no-save-photons        Do not save photon cache
--photon-budget <N>      Number of photons to emit
```

### 16.6 Startup Logic

```
load scene → compute scene hash → check cache validity
  if valid: load from cache
  else: trace photons → build index → save cache
camera pass → render interactively
```

**P key** triggers full photon recomputation at any time.

Implementation: `src/photon/photon_io.h` + `src/photon/photon_io.cpp`.

---

## 17. OptiX Program Structure

All device code in `src/optix/optix_device.cu`, compiled to PTX via `nvcc`
with `--use_fast_math`.

### Programs

| Program                    | Type        | Purpose                              |
|----------------------------|-------------|--------------------------------------|
| `__raygen__render`         | Ray Gen     | First-hit camera pass + NEE + gather |
| `__raygen__photon_trace`   | Ray Gen     | GPU photon emission + tracing        |
| `__closesthit__radiance`   | Closest-Hit | Unpack geometry at hit point         |
| `__closesthit__shadow`     | Closest-Hit | Set "occluded" flag                  |
| `__miss__radiance`         | Miss        | Return zero radiance                 |
| `__miss__shadow`           | Miss        | Return "not occluded"                |

### Payload Layout (14 values)

| Slot  | Contents               |
|-------|------------------------|
| p0-p2 | Hit position (float3)  |
| p3-p5 | Shading normal (float3)|
| p6    | Hit distance t         |
| p7    | Material ID            |
| p8    | Triangle (primitive) ID|
| p9    | Hit flag               |
| p10-p12| Geometric normal      |
| p13   | Reserved               |

### SBT (Shader Binding Table)

Two ray types:
- **Type 0 — Radiance**: closest-hit writes full geometry payload.
- **Type 1 — Shadow**: closest-hit writes occluded flag; uses
  `OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT`.

### Pipeline Configuration

| Parameter                    | Value   |
|------------------------------|---------|
| `OPTIX_NUM_PAYLOAD_VALUES`   | 14      |
| `OPTIX_NUM_ATTRIBUTE_VALUES` | 2       |
| `OPTIX_MAX_TRACE_DEPTH`      | 2       |
| `OPTIX_STACK_SIZE`           | 16,384  |

---

## 18. Configuration

All tunable constants in `src/core/config.h`:

### Sampling & Bounces

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_LAMBDA` | 32 | Wavelength bins (380–780 nm) |
| `DEFAULT_SPP` | 16 | Samples per pixel |
| `DEFAULT_MAX_BOUNCES` | 8 | Maximum path bounces |
| `DEFAULT_MIN_BOUNCES_RR` | 3 | Bounces before Russian roulette |
| `DEFAULT_RR_THRESHOLD` | 0.95 | Max RR survival probability |

### Photon Mapping

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_NUM_PHOTONS` | 1,000,000 | Photons emitted per trace |
| `DEFAULT_GATHER_RADIUS` | 0.05 | Photon gather radius |
| `DEFAULT_CAUSTIC_RADIUS` | 0.02 | Caustic map gather radius |
| `HASHGRID_CELL_FACTOR` | 2.0 | cell_size = factor × radius |
| `DEFAULT_SURFACE_TAU` | 0.02 | Plane-distance filter thickness |
| `DEFAULT_PLANE_DISTANCE_THRESHOLD` | 1e-3 | Max plane distance for tangential kernel |
| `PLANE_TAU_EPSILON_FACTOR` | 10.0 | tau ≥ 10 × ray_epsilon |
| `DEFAULT_PHOTON_BOUNCE_STRATA` | 64 | Max hemisphere strata for decorrelation |
| `DEFAULT_GPU_MAX_GATHER_RADIUS` | 0.5 | Max gather radius for GPU grid k-NN |

### NEE

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_NEE_LIGHT_SAMPLES` | 4 | Shadow rays at bounce 0 |
| `DEFAULT_NEE_COVERAGE_FRACTION` | 0.3 | Coverage-aware sampling factor $c$ |

### SPPM

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sppm_iterations` | 64 | Iteration count |
| `sppm_alpha` | 2/3 | Shrinkage factor |
| `sppm_initial_radius` | 0.1 | Starting gather radius |

### Chromatic Dispersion

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_CAUCHY_A` | 1.5046 | Cauchy A (crown glass) |
| `DEFAULT_CAUCHY_B` | 4200.0 | Cauchy B (nm²) |

### CellInfoCache & Adaptive Caustics

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CELL_CACHE_TABLE_SIZE` | 65 536 | Hash table size for cell cache |
| `CAUSTIC_TARGETED_FRACTION` | 0.30 | Extra caustic photons per iteration |
| `CAUSTIC_CV_THRESHOLD` | 0.50 | Min CV to flag as hotspot |
| `CAUSTIC_CV_TARGET` | 0.20 | Target CV for convergence |
| `MAX_CAUSTIC_ITERATIONS` | 3 | Max adaptive shooting rounds |
| `ADAPTIVE_RADIUS_MIN_FACTOR` | 0.25 | Min adaptive radius scale |
| `ADAPTIVE_RADIUS_MAX_FACTOR` | 2.0 | Max adaptive radius scale |
| `ADAPTIVE_RADIUS_TARGET_K` | 100 | Target photon count per cell |

### Image & Window

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_IMAGE_WIDTH` | 1024 | Output image width |
| `DEFAULT_IMAGE_HEIGHT` | 768 | Output image height |

### OptiX

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OPTIX_NUM_PAYLOAD_VALUES` | 14 | Payloads per trace call |
| `OPTIX_MAX_TRACE_DEPTH` | 2 | Ray types (radiance + shadow) |
| `OPTIX_STACK_SIZE` | 16,384 | GPU thread stack (bytes) |
| `OPTIX_SCENE_EPSILON` | 1e-4 | Shadow/continuation offset |

---

## 19. Key Data Structures

### Spectrum
```cpp
struct Spectrum {
    float value[NUM_LAMBDA];  // 32 bins, 380-780 nm
};
```

### Photon (SoA)
```cpp
struct PhotonSoA {
    vector<float> pos_x, pos_y, pos_z;
    vector<float> wi_x, wi_y, wi_z;       // incoming direction toward surface
    vector<float> norm_x, norm_y, norm_z;  // geometric surface normal at deposit
    vector<uint16_t> lambda_bin;
    vector<float> flux;
    vector<uint8_t> path_flags;            // PHOTON_FLAG_* bitmask (§5.2.5)
    vector<uint8_t> bounce_count;          // number of bounces before deposit
};
```

**`wi` convention:** Incoming direction **toward the surface** — the
direction FROM which light arrives. `dot(photon.wi, surface_normal) > 0`.

**Normal convention:** **Geometric normals** (face normals), not shading
normals. Shading normals are used for BSDF evaluation at the query point,
not for spatial filtering.

### KDTree
```cpp
struct KDTree {
    struct Node {
        int   split_axis;    // 0=x, 1=y, 2=z; -1 = leaf
        float split_pos;
        int   left, right;
    };
    std::vector<Node>     nodes;
    std::vector<uint32_t> indices;  // photon indices in leaf order

    void build(const PhotonSoA& photons);
    // All queries use tangential distance, NOT 3D Euclidean
    template<typename Callback>
    void query(float3 pos, float3 normal, float radius,
               const PhotonSoA& photons, Callback callback) const;
    void knn(float3 pos, float3 normal, int k, const PhotonSoA& photons,
             std::vector<uint32_t>& out_indices, float& out_max_dist2) const;
};
```

### HashGrid
```cpp
// Hashed uniform grid with cellStart/cellEnd/sortedIndices
// Hash: (cx*73856093 ^ cy*19349663 ^ cz*83492791) % table_size
```

### SPPMPixel
```cpp
struct SPPMPixel {
    float3 pos, normal, wo;
    int material_id;
    float radius;          // current gather radius (tangential)
    float N;               // accumulated photon count
    Spectrum tau;           // accumulated flux
    Spectrum L_direct;      // NEE direct component
    bool valid;
};
```

### LaunchParams
Contains all device pointers: framebuffer, scene geometry, materials, photon
map, hash grid, emitter CDF, camera, rendering flags (`is_final_render`,
`render_mode`), SPPM state, and adaptive sampling pointers.

---

## 20. Source Layout

```
src/
  main.cpp                      Entry point, GLFW loop, arg parsing
  core/
    types.h                     float3, Ray, HitRecord, ONB
    spectrum.h                  Spectral arithmetic, CIE XYZ, blackbody
    config.h                    All tunable constants
    random.h                    PCG RNG
    alias_table.h               Alias method for O(1) discrete sampling
    cdf.h                       Generic CDF build + sample
    sppm.h                      SPPM types, progressive update, reconstruction
    photon_bins.h               PhotonBin struct, Fibonacci sphere (Phase 7)
    photon_density_cache.h      Per-pixel cached spectral density
    cell_cache.h                CellInfoCache — per-cell precomputed statistics (§5.5)
    guided_nee.h                Bin-flux-weighted NEE CDF
    nee_sampling.h              NEE sampling helpers
    medium.h                    Participating medium (temporarily disabled)
    phase_function.h            Henyey-Greenstein phase function (temporarily disabled)
    font_overlay.h              Debug text overlay rendering
    test_data_io.h              Ground-truth data helpers for tests
  bsdf/
    bsdf.h                      Lambertian, mirror, glass, GGX VNDF
  scene/
    scene.h                     Scene graph, emitter list
    material.h                  Material definitions
    triangle.h                  Triangle + BVH primitives
    obj_loader.h / .cpp         Wavefront OBJ + MTL parser
  renderer/
    renderer.h / .cpp           CPU reference renderer, RenderConfig, FrameBuffer
    camera.h                    Perspective camera, ray generation
    mis.h                       MIS weight utilities
    direct_light.h / .cu        Direct lighting kernel
    path_tracer.cu              CPU/CUDA path tracing kernels
  photon/
    photon.h                    Photon and PhotonSoA structs
    kd_tree.h                   KD-tree build + range query + k-NN (CPU reference)
    hash_grid.h / .cu           Hashed uniform grid build + query
    emitter.h / .cu             Emitter sampling, CDF construction
    density_estimator.h         Tangential disk kernel density estimation
    surface_filter.h            Surface consistency filter (tangential metric)
    photon_io.h / .cpp          Binary save/load for photon map persistence
  optix/
    optix_renderer.h / .cpp     Host pipeline: SBT, GAS, launch params
    optix_device.cu             All OptiX raygen / closesthit programs
    launch_params.h             GPU/CPU shared launch parameter struct
    adaptive_sampling.h / .cu   Per-pixel noise metric + convergence mask
  debug/
    debug.h                     Visualisation mode state, key bindings
tests/
  test_main.cpp                 Core unit tests (GoogleTest)
  test_kd_tree.cpp              KD-tree unit tests
  test_tangential_gather.cpp    Tangential kernel tests
  test_surface_filter.cpp       Surface consistency filter tests
  test_integration.cpp          CPU↔GPU integration tests
  test_ground_truth.cpp         Reference image ground-truth comparisons
  test_per_ray_validation.cpp   Individual ray correctness checks
  test_pixel_comparison.cpp     Per-pixel render validation
  test_medium.cpp               Participating medium tests
  test_speed_tweaks.cpp         Dispersion, CellInfoCache, caustic tracing, IOR stack (29 tests)
  feature_speed_test.cpp        Performance benchmarks
```

---

## 21. Strengths

1. **Full spectral transport.** 32 wavelength bins; dispersion, metamerism,
   and spectral emission naturally captured without RGB approximations.

2. **Photon-centric simplicity.** Camera pass is trivial (first-hit + NEE +
   gather). All algorithmic complexity concentrated in the photon pass.

3. **Precomputable photon map.** Binary save/load with scene hash — compute
   once, render interactively with arbitrary camera positions.

4. **Surface-aware gather.** Tangential disk kernel eliminates planar
   blocking artifacts and cross-surface leakage.

5. **Dual CPU/GPU implementation.** CPU reference renderer provides ground
   truth; GPU provides interactive speed. Integration tests verify parity.

6. **Adaptive gather radius.** k-NN per hitpoint adapts to local photon
   density automatically.

7. **SPPM convergence.** Progressive radius shrinking guarantees asymptotic
   convergence to the correct solution.

8. **Photon path decorrelation.** Cell-stratified bouncing ensures efficient
   coverage of the hemisphere.

9. **Coverage-aware NEE.** Mixture of power-weighted and area-weighted
   sampling ensures all emitters get shadow rays.

10. **Interactive debug viewer.** Multiple render modes and overlays for
    inspecting every intermediate quantity.

11. **Comprehensive test suite.** Unit tests, integration tests (CPU↔GPU),
    ground-truth comparisons, and per-ray validation.

12. **Chromatic dispersion.** Cauchy-equation wavelength-dependent IOR with
    per-bin Fresnel produces physically correct spectral splitting
    (prismatic rainbows, chromatic caustics).

13. **CellInfoCache precomputation.** Per-cell photon statistics (density,
    variance, caustic count, directional spread) enable adaptive gather
    radius and empty-region skip without per-query overhead.

14. **Adaptive caustic shooting.** Two-phase photon emission concentrates
    budget on high-variance caustic cells, reducing caustic noise.

---

## 22. Weaknesses and Limitations

1. **No diffuse camera continuation.** All indirect lighting comes from the
   photon map. Photon map quality directly determines GI quality.

2. **Single GAS, no instancing.** Flat triangle soup only.

3. **No texture mapping in OptiX.** Per-material properties, not per-texel.

4. **Hash grid built on CPU.** Photon data downloaded → grid built → reuploaded.

5. **No denoising.** Raw Monte Carlo output.

6. **Windows-centric build.** Tested primarily with MSVC on Windows.

7. **Volume rendering temporarily disabled.** Surface transport only.

8. **GPU k-NN bounded by max_radius.** Sparse regions may not find k photons.

---

## 23. Common Bugs to Avoid

- Forgetting area→solid-angle Jacobian (`dist² / cos_y`) in NEE
- Using wrong cosine in Jacobian (must be emitter-side `cos_y`)
- Depositing photons at first diffuse hit from light (double counts with NEE)
- Using triangle-uniform instead of power/area-weighted emission
- Not offsetting shadow/new rays with epsilon
- Mixing wavelength bins during transport
- **Using 3D Euclidean distance instead of tangential distance in gather** — causes planar blocking
- **k-NN using 3D distance** — nearby surfaces inflate radius incorrectly
- **Missing plane distance filter** — allows cross-wall leakage
- **Forgetting specular chain throughput** — mirrors/glass views wrong brightness
- **Terminating photon paths at delta surfaces** — prevents caustic formation
- **Omitting $N_{\text{photons}}$ in photon flux denominator** — brightness scales with count
- **Using NEE solid-angle PDF with area-form integrand** (or vice versa)
- **Using shading normals for tangential metric** — use geometric normals
- CPU vs GPU: different structures OK, but gather metric must match

---

## 24. Architectural Differences from v1

| Aspect | v1 (Previous) | v2.1 (Current) |
|--------|---------------|----------------|
| Camera ray depth | Full path tracing (N bounces) | First hit only (specular chain N≤8 to first diffuse) |
| Indirect lighting | Photon map + camera BSDF bouncing | Photon map only |
| MIS | 3-way (NEE + BSDF + photon) | None at camera; standard BSDF in photon rays |
| Photon bounce strategy | Standard BSDF | Stratified BSDF at bounce 0 |
| Spatial index | Hash grid only | KD-tree (CPU) + Hash grid (GPU) + tangential kernel |
| Gather radius | Fixed or SPPM (bounded) | k-NN adaptive per hitpoint (tangential distance) |
| Gather kernel | 3D spherical | Tangential disk kernel (fixes planar blocking) |
| Photon precomputation | Recomputes every launch | Binary save/load with scene hash invalidation |
| Photon budget | Single budget | Separate `global_photon_budget` / `caustic_photon_budget` |
| Caustic handling | Separate caustic map | Separate caustic map (retained) |
| Tone mapping | Reinhard | ACES Filmic |
| Volume rendering | Rayleigh + Beer-Lambert | Temporarily disabled |
| Default render mode | Progressive accumulation | SPPM progressive (radius shrinkage) |
| CPU reference | Partial | Full, physically identical to GPU |
| Integration tests | None (unit only) | CPU↔GPU comparison suite |
| Cell-bin grid | Used for guided camera bouncing | Deleted (~800 lines) |

---

## 25. Removed / Deprecated Functionality

| Feature | Reason |
|---------|--------|
| Camera ray bouncing (bounce > 0) | Replaced by photon-only indirect |
| MIS 3-way weighting | Camera has no BSDF continuation |
| Photon-guided camera BSDF sampling | No camera BSDF bounces |
| `RenderMode::Full` with multi-bounce | Replaced by first-hit + NEE + photon |
| `dev_sample_guided_bounce()` in camera pass | Photon bounces use standard BSDF |
| Volume rendering | Temporarily disabled for surface validation |
| Dense cell-bin grid (`CellBinGrid`) | Deleted. KD-tree handles spatial queries |
| Obsolete tests (multi-bounce camera, MIS weights, guided bouncing) | Fully deleted, replaced by new tests |
