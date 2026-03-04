# Architecture — Spectral Photon-Guided Path Tracer (v3)

This document describes the complete rendering pipeline, its design
rationale, mathematical foundations, and implementation details.

**Architecture version:** v3 (guided sub-path sidecar, full camera path tracing, dense-grid photon sampling)

---

## 1. Overview

The renderer implements a **photon-guided path tracing** architecture.
Camera rays run a full iterative bounce loop; at each non-delta bounce the
renderer fires a **guided sub-path sidecar** in a photon direction sampled
from a dense 3D grid. The sidecar returns irradiance that is mixed with
next-event estimation (NEE) to form the bounce contribution. Photon rays
carry the transport information that seeds the guide; the camera path
carries the actual radiance integration. All light transport is computed
over 32 discrete wavelength bins spanning 380–780 nm. The pipeline runs on
the GPU via **NVIDIA OptiX 9.x** for ray tracing and **CUDA** for
auxiliary kernels, with a full **CPU reference renderer** for validation.

### 1.1 Design Philosophy

Camera rays are **full path tracers** with a photon-guided sidecar:

1. They traverse specular chains (mirror/glass) deterministically.
2. At each non-delta (diffuse/glossy) surface they:
   - Evaluate **NEE** (1 shadow ray, MIS-weighted against BSDF).
   - Fire a **guided sub-path** in a direction drawn from nearby photons
     stored in a dense grid. The sub-path returns irradiance that is mixed
     with NEE.
   - Continue via **BSDF importance sampling** (always — no coin flip).
3. Russian roulette terminates low-throughput paths.

The photon map is an **irradiance sidecar** — an input signal to the
path tracer, analogous to how NEE is an input signal from the emitter
distribution. The photon map does NOT replace camera-side transport;
it augments BSDF-sampled paths with photon-guided directions.

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
- **Single-shot density estimation is biased but consistent.** Bias
  decreases as gather radius shrinks with increasing photon count.

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

- **Single-shot mode**: biased but consistent; bias decreases with
  increasing photon count and decreasing gather radius.

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
      - Glass/Translucent: wavelength-dependent IOR (Cauchy dispersion),
        Tf filter, IOR stack for nested dielectrics (§5.2.6),
        geometric normals for entering test (§5.2.7), path flag tagging
      - Hero wavelength system (§5.1.2): PBRT v4 style, 4 stratified
        wavelengths per photon, direction from hero bin,
        per-wavelength Cauchy dispersion filter for companions
      - Deposit at each qualifying diffuse hit (lightPathDepth ≥ 2)
   d. Build spatial index (KD-tree on CPU, hash grid on GPU)
   e. Build CellInfoCache (per-cell photon statistics, §5.5)
   f. Adaptive caustic shooting: re-emit photons toward high-CV
      caustic hotspot cells (§5.6), rebuild grids + cache if augmented
   g. Save photon map + spatial index to binary file (optional)
4. ═══ CAMERA PASS ═══  (full path trace with guided sidecar)
   a. For each pixel: trace ONE camera ray
   b. Iterative bounce loop:
      - If hit emissive: add emission (MIS-weighted against NEE)
      - If hit delta surface (mirror/glass/translucent): bounce
        deterministically, update throughput, continue
      - If hit non-delta surface:
        i.  NEE: 1 shadow ray, MIS-weighted against BSDF
        ii. Guide sidecar: sample photon direction from dense grid 3×3×3
            neighbourhood → evaluate BSDF → trace full sub-path
            → compute guided irradiance G → mix with NEE (N):
              caustic photon → max(N, G) by brightness
              diffuse photon → (N + G) / 2
        iii. BSDF continuation: sample new direction, always continue
             (no coin flip between guide and BSDF)
        iv. Russian roulette after MIN_BOUNCES_RR guaranteed bounces
   c. Sub-paths use the same loop but with EnableGuideSidecar=false
      (no recursive sidecar) and skip_bounce0_emission=true
      (avoid double-counting with parent NEE)
5. Spectral → RGB conversion (CIE XYZ), ACES filmic tone mapping, output
```

**Key design principle:** The photon pass and camera pass are **loosely
coupled**. The photon map is an irradiance sidecar — a cached light-field
signal that seeds guided directions for the camera path tracer. Camera rays
perform full transport (NEE + guided sub-path + BSDF continuation); the
photon map augments this transport rather than replacing it.

### 4.1 Scene Loading

Wavefront OBJ with MTL materials. Materials are mapped to the internal
spectral representation using `rgb_to_spectrum_reflectance()` for
diffuse/specular albedos and `blackbody_spectrum()` for emissive surfaces.

**Scene normalisation:** All non-reference scenes are scaled and translated
to fit inside the Cornell Box reference frame $([-0.5, 0.5]^3)$.

Supported material types:

| Type        | MTL Cue / pb_brdf       | Internal Enum       | Delta? |
|-------------|-------------------------|---------------------|--------|
| Lambertian  | default / `lambert`     | `Lambertian`        | No     |
| Mirror      | `illum 3/5` / `conductor` | `Mirror`          | Yes    |
| Glass       | `illum 4` / `dielectric`  | `Glass`           | Yes    |
| GlossyMetal | `illum 2` / `conductor`   | `GlossyMetal`     | No     |
| Emissive    | `Ke` present / `emissive` | `Emissive`        | —      |
| GlossyDiel. | `illum 7` / `dielectric`  | `GlossyDielectric`| No     |
| Translucent | `dielectric` + medium     | `Translucent`     | Yes    |
| Clearcoat   | `clearcoat`               | `Clearcoat`       | No     |
| Fabric      | `fabric`                  | `Fabric`           | No     |

**Translucent** = Glass-like Fresnel bounce + optional interior
participating medium (subsurface scattering via Beer–Lambert). Created
when `pb_brdf = dielectric` with `pb_medium_enabled = true`.

**`pb_tf_spectrum` — direct spectral transmittance override (§4.1.1):**
For glass materials requiring spectrally-narrow colour filters that
cannot be represented via RGB → spectrum conversion, the MTL keyword
`pb_tf_spectrum b0 b1 … bN` writes per-bin transmittance directly into
`Material::Tf`, bypassing the lossy pseudoinverse matrix entirely.

### 4.1.1 `pb_tf_spectrum` — Direct Spectral Transmittance Override

The standard RGB→spectrum path uses a pseudoinverse matrix
(`rgb_to_spectrum_reflectance`) to convert `Tf` RGB values into `NUM_LAMBDA`
spectral bins. Because the matrix must be non-negative and cover a wide
gamut, spectrally narrow colours (e.g. deep green glass with near-zero
transmission at 430 and 730 nm) are impossible to represent accurately.

The `pb_tf_spectrum` MTL keyword bypasses this conversion entirely:

```
pb_tf_spectrum 0.04 0.92 0.04 0.01
```

Each value maps directly to a wavelength bin (430, 530, 630, 730 nm for
`NUM_LAMBDA = 4`). The parser reads exactly `NUM_LAMBDA` floats into
`Material::pb_tf_spectrum`. During material finalization, if
`pb_tf_spectrum_set` is true, `mat.Tf = mat.pb_tf_spectrum` overrides
whatever the RGB conversion produced.

**Limitations:**
- Coupled to `NUM_LAMBDA` — the MTL file must provide exactly as many
  values as there are spectral bins.
- Not artist-friendly — requires knowledge of the bin layout.
- No round-trip guarantee — the override is one-way.
- Still coarse for 4-bin configurations.

**Opacity fix:** Glass and Translucent materials always have
`opacity = 1.0`. Their transmission is handled by the specular bounce
shader (Fresnel refraction), NOT by the anyhit alpha cutout. Without
this, `pb_transmission 1.0` would set `opacity = 0.0`, making the
material invisible to all ray intersections.

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

### 5.1.2 Hero Wavelength System (PBRT v4 Style)

Each photon traces `HERO_WAVELENGTHS` (= 4) wavelength channels
simultaneously. The **hero** (primary) wavelength determines the
refraction direction; **companion** wavelengths are carried along with
independent throughput weights.

**Selection:** The hero bin is sampled from the emission spectrum CDF.
Companions are deterministic stratified offsets:

```
companion[h] = (hero_bin + h * NUM_LAMBDA / HERO_WAVELENGTHS) % NUM_LAMBDA
```

This produces 4 evenly-spaced bins across the 32-bin spectrum, maximising
wavelength diversity per photon.

**Flux per hero:** Each hero channel carries:

$$
\Phi_h = \frac{L_e(\lambda_h) \cos\theta}{p_{tri} \cdot p_{pos} \cdot p_{dir} \cdot p_{\lambda_h}} \cdot \frac{1}{H}
$$

where $H$ = `HERO_WAVELENGTHS`.

**Dispersion filter:** At each Glass/Translucent bounce, the hero
wavelength determines the refraction direction. Companion wavelengths
receive a per-bin Cauchy dispersion filter:

- Refraction: `filter[b] = Tf[b] * (1 - F_b) / (1 - F_hero)`, where
  `F_b = fresnel(cos_i, outside_ior / ior_at_lambda(b))`
- Reflection: `filter[b] = F_b / F_hero`
- TIR bins: `filter[b] = 0` (total internal reflection at that wavelength)

This correctly handles spectral splitting: red and blue companions diverge
in weight as they would physically separate at different refraction angles.

**Camera path:** Uses D-line (~589 nm, `DLINE_BIN`) as hero instead of
bin 0 (380 nm), because the D-line gives a typical glass IOR (1.52 for
crown glass) rather than the UV-extreme IOR that bin 0 would produce.

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

Russian roulette after `DEFAULT_PHOTON_MIN_BOUNCES_RR` (default 10):
$p_{continue} = \min(\max_\lambda(T(\lambda)),\, \text{RR\_THRESHOLD})$

**Russian roulette is skipped for specular/translucent bounces** — these
are deterministic events that must continue to enable caustic formation.

**Caustic caster materials:** Mirror (1), Glass (2), and Translucent (6)
are all classified as `caustic_caster` in `classify_for_photons_by_type()`.
When a photon hits a caustic caster, `on_caustic_path` is set to true.
Mirror surfaces produce purely reflective caustics (no refraction, no
chromatic dispersion, no IOR stack modification). Glass and Translucent
surfaces may additionally trigger IOR stack push/pop and dispersion.

#### 5.2.3 Photon Deposition Rule (No Double Counting)

Deposit when:
- Hit material is **non-delta** (diffuse/glossy)
- AND `lightPathDepth >= 2` (skip first hit from light)

Delta (mirror/glass) surfaces: do NOT deposit, but MUST continue bouncing.
Terminating at delta surfaces prevents caustic formation.

#### 5.2.4 Separate Maps & Three-Valued Tag System

Photons are separated into two maps with a three-valued tag for
density estimation normalization:

| Tag | Name | Source | Gather bucket | Normalization |
|-----|------|--------|---------------|---------------|
| 0 | Non-caustic | Global pass, diffuse-only path | `L_global` | `1 / N_global` |
| 1 | Global-caustic | Global pass, specular chain path (L→S+→D) | `L_global` | `1 / N_global` |
| 2 | Targeted-caustic | Dedicated caustic-targeted pass | `L_caustic` | `1 / N_caustic` |

**All photons are gathered — none are skipped.** Tags 0 and 1 both
contribute to `L_global` with `1/N_global` normalization. Tag 2
contributes to `L_caustic` with `1/N_caustic` normalization. The final
result is `L = L_global + L_caustic`.

**Tag assignment:**
- **Global photon pass** (`__raygen__photon_trace`): writes tag 0 (non-caustic)
  or 1 (caustic) based on `on_caustic_path` at deposition.
- **Targeted caustic pass** (`__raygen__targeted_photon_trace`): host-side
  fills tag 2 for all targeted photons via
  `std::fill(caustic_pass_flags_.begin() + n_global, end, 2)`.

**Why dual normalization?** Global and targeted passes emit different
numbers of photons (`N_global` vs `N_caustic`). Each photon's flux was
scaled by its pass budget at emission time. Using the correct per-budget
normalizer prevents brightness artefacts in overlapping regions.

The caustic map uses a smaller gather radius (`DEFAULT_CAUSTIC_RADIUS`)
for sharper caustic features.

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
| `PHOTON_FLAG_CAUSTIC_SPECULAR` | 0x10 | Path is a mirror/reflective caustic (L→Mirror→D) |

Flags are accumulated along the path: glass detection sets `TRAVERSED_GLASS`;
if the photon later deposits on a diffuse surface, `CAUSTIC_GLASS` is set.
Materials with `dispersion == true` trigger `DISPERSION`. Mirror bounces set
`CAUSTIC_SPECULAR` — these are purely reflective caustics with no IOR stack
changes and no chromatic dispersion.

### 5.2.6 IOR Stack (Nested Dielectrics)

An `IORStack` (4 deep) tracks the current surrounding medium's IOR during
both photon tracing and camera specular chains. The stack starts empty
(= air, IOR 1.0). When entering a dielectric, the material's nominal IOR
is pushed; when exiting via refraction, the top is popped. **Reflection
does not modify the stack** — the ray stays in the same medium.

```cuda
struct IORStack {
    static constexpr int MAX_DEPTH = 4;
    float iors[MAX_DEPTH];
    int   depth;                          // 0 = empty = air
    __forceinline__ __device__ IORStack() : depth(0) {}
    __forceinline__ __device__ float top() const {
        return depth > 0 ? iors[depth - 1] : 1.0f;   // air fallback
    }
    __forceinline__ __device__ void push(float ior) {
        if (depth < MAX_DEPTH) iors[depth++] = ior;
    }
    __forceinline__ __device__ void pop() {
        if (depth > 0) --depth;
    }
};
```

**Usage in `dev_specular_bounce`:**
- `outside_ior = ior_stack ? ior_stack->top() : 1.0f`
- `eta = entering ? (outside_ior / hero_ior) : (hero_ior / outside_ior)`
- On **refraction**: `push(hero_ior)` when entering, `pop()` when exiting.
- On **reflection**: no stack change.
- Per-wavelength Cauchy filter also uses `outside_ior` for all eta
  computations (companion wavelengths).

**Where IORStack is instantiated:**
- `__raygen__photon_trace` bounce loop
- `__raygen__targeted_photon_trace` bounce loop
- `full_path_trace` (camera specular chain + glossy continuation)
- `debug_first_hit` uses `nullptr` (no IOR tracking in debug mode).

### 5.2.7 Geometric vs Shading Normals in Specular Bounces

`dev_specular_bounce` receives **both** normals:

| Normal | Parameter | Used For |
|--------|-----------|----------|
| Shading normal | `normal` | Fresnel `cos_i`, reflection direction, refraction direction (smooth surface appearance) |
| Geometric normal | `geo_normal` | Entering/exiting test (`dot(dir, geo_normal) < 0`), epsilon offset (`pos ± outward_geo * EPSILON`) |

**Why geometric for entering test?** Shading normals are interpolated
and can flip sign near triangle edges, causing incorrect inside/outside
classification. Geometric normals are flat per-triangle and always
consistent with the actual surface orientation.

**Why shading for direction?** Smooth surface appearance requires
interpolated normals for reflection and refraction directions. Using
geometric normals would produce faceted reflections.

This split is applied consistently in all callers across photon trace,
targeted trace, camera path trace, and debug first-hit.

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

Phase 3 — Accumulate (dual-budget)
    For each of the K nearest photons (d_tan² < r_k²):
        w = 1 - d_tan²/r_k²              (Epanechnikov weight)
        tag = photon_is_caustic_pass[i]   (0, 1, or 2)
        N_norm = (tag == 2) ? N_caustic : N_global
        L_target = (tag == 2) ? L_caustic : L_global
        L_target += w · f_s(x, ωi→ωo, λ) · Φi(λ) / (N_norm · ½π r_k²)
    L = L_global + L_caustic
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
L_{\text{photon}}(x,\omega_o,\lambda) = L_{\text{global}} + L_{\text{caustic}}
$$

where each budget is estimated independently:

$$
L_{\text{budget}}(x,\omega_o,\lambda) = \frac{1}{\pi r_k^2}\sum_{i \in \text{budget}} w_i \cdot f_s(x,\omega_i,\omega_o,\lambda)\, \Phi_i(\lambda) \,/\, N_{\text{budget}}
$$

- $r_k$ = tangential distance to the $K$-th nearest photon (adaptive, §6.5)
- $w_i = 1 - d_{\text{tan},i}^2 / r_k^2$ (Epanechnikov kernel weight)
- $N_{\text{budget}}$ = `N_global` for tag 0/1 photons, `N_caustic` for tag 2 photons
- The BSDF $f_s$ is applied **per-photon** inside the sum, not as an
  overall multiplier

When the dual-budget system is inactive (no targeted caustics), all
photons receive `1/N_global` normalization.

### 6.7 Gather Radius Strategy

| Mode | Radius per hitpoint |
|------|---------------------|
| k-NN adaptive (default) | Find K nearest → radius = tangential distance to K-th |

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

## 7. Camera Pass (Full Path Trace + Guided Sidecar)

Camera rays run an iterative bounce loop (`full_path_trace_v3`) that
evaluates NEE, fires a photon-guided sub-path, and continues via BSDF
at every non-delta bounce. The function is templatized:

| Instantiation | Role |
|--------------|------|
| `full_path_trace_v3<true>` | Main camera path — fires guided sub-paths, inits Fibonacci sphere |
| `full_path_trace_v3<false>` | Sub-path — no sidecar, no Fibonacci init (prevents 2^N recursion) |

The template uses `if constexpr` to compile out the sidecar in sub-paths,
giving them an independent register frame and avoiding infinite recursion.

### 7.1 Per Pixel

1. Generate camera ray (with DOF if enabled, stratified sub-pixel jitter)
2. Enter iterative bounce loop (up to `DEFAULT_MAX_BOUNCES_CAMERA`)
3. **Trace ray** → find intersection
4. If miss: break
5. **Medium transport** (if inside per-material or atmospheric medium):
   free-flight sampling, scatter NEE, volume photon gather
6. If hit **emissive**: add Le × MIS weight (`mis_weight_2(pdf_bsdf, pdf_nee)`)
   and break. At bounce 0 no MIS is needed (camera sees the light directly).
   Sub-paths skip bounce-0 emission (`skip_bounce0_emission = true`) to
   avoid double-counting with the parent NEE.
7. If hit **delta surface** (mirror/glass/translucent): deterministic
   specular bounce, update throughput, IOR stack, medium stack; set
   `pdf_combined_prev = 0` (Dirac); continue loop.
8. If hit **non-delta surface**: enter shading block (§7.2–7.5).

### 7.2 NEE at Non-Delta Surfaces

Standard Veach-style MIS: sample one light point, cast shadow ray, evaluate
BSDF at the light direction, weight against BSDF PDF. The NEE contribution
`nee_L` is computed but **deferred** — it is not added to the result
directly. Instead it enters the mixing stage (§7.4).

### 7.3 Guided Sub-Path Sidecar (§4a)

At each non-delta bounce (only in `<true>` instantiation):

1. **Sample photon direction**: `dev_guide_sample_and_pdf_full()` performs
   a reservoir-1 pick over photons in the 3×3×3 dense-grid neighbourhood
   centred on the shading point. The picked photon's `wi` is jittered
   by a cone half-angle (`DEFAULT_PHOTON_GUIDE_CONE_HALF_ANGLE = 0.15 rad`).
   Returns the guided direction, the photon's `path_flags`, and PDF.

2. **Evaluate BSDF** at the guided direction:
   `bsdf_weight = f_eval(wo, wi_guided) × cos_theta`

3. **Trace full sub-path**: `full_path_trace_v3<false>(sub_origin, wi_guided, …, skip_bounce0_emission=true)`.
   The sub-path runs the same bounce loop (NEE + BSDF continuation +
   Russian roulette) but without a sidecar and with bounce-0 emission
   suppressed.

4. **Compute guided irradiance**:
   `G = bsdf_weight × sub_path.combined`

### 7.4 Mixing Rule

The NEE contribution (N) and guided contribution (G) are combined based
on the picked photon's caustic flags:

| Photon type | Flag test | Mixing rule | Rationale |
|------------|-----------|-------------|-----------|
| **Caustic** | `picked_flags & 0x12` (CAUSTIC_GLASS \| CAUSTIC_SPECULAR) | `max(N, G)` by `Spectrum::sum()` | Caustics are concentrated; take the brighter signal |
| **Diffuse** | otherwise | `(N + G) / 2` | Smooth lighting; average reduces variance |

The mixed result `combined_irradiance` is multiplied by throughput and
added to the result.

### 7.5 BSDF Continuation (Always)

After the sidecar, the BSDF **always** samples a continuation direction:

```
bs = bsdf_sample(mat_id, wo_local, uv, rng)
throughput *= bs.f * cos_theta / bs.pdf
```

There is no coin flip between guided and BSDF directions — both fire.
The BSDF direction is the main path; the guided direction is a sidecar
that adds information without replacing the main transport.

`pdf_combined_prev = bs.pdf` is stored for emission MIS at the next bounce.

### 7.6 Russian Roulette

After `min_bounces_rr` guaranteed bounces:

$$
p_{\text{survive}} = \min\bigl(\text{RR\_THRESHOLD},\; \max_\lambda(T(\lambda))\bigr)
$$

If $p_{\text{survive}} < 10^{-4}$: terminate. Otherwise, survive with
probability $p_{\text{survive}}$; boost throughput by $1/p_{\text{survive}}$.

### 7.7 Specular Chain Throughput

The camera maintains an `IORStack ior_stack` and `MediumStack medium_stack`
across all bounces (specular and non-specular). Each specular bounce calls
`dev_specular_bounce()` with the stacks. The throughput is updated
per-wavelength:

```
T[i] *= sb.filter[i]     // Fresnel × Tf × dispersion correction
```

### 7.8 Emission MIS

When the BSDF continuation ray hits an emissive surface at bounce > 0:

$$
w_{\text{BSDF}} = \frac{p_{\text{BSDF}}^2}{p_{\text{BSDF}}^2 + p_{\text{NEE}}^2}
$$

When the previous bounce was delta (`pdf_combined_prev = 0`), NEE cannot
sample that path, so `w_BSDF = 1`. This is the standard power-heuristic
one-sample MIS.

---

## 8. Spectral → RGB Output

1. Integrate spectrum against CIE XYZ curves (Wyman Gaussian fit)
2. Convert XYZ → linear sRGB
3. Tone map: **ACES Filmic** (replaces Reinhard)
4. Gamma correct (sRGB transfer function)

Same pipeline for all component buffers. Both CPU and GPU use the identical
ACES pipeline for fair PSNR comparison.

---

## 9. BSDF Models

| Model      | $f_s$                            | Sampling PDF                     |
|------------|----------------------------------|----------------------------------|
| Lambertian | $K_d / \pi$                      | $\cos\theta / \pi$               |
| Mirror     | ideal specular reflection         | delta distribution                |
| Glass      | Fresnel-weighted reflect/refract  | Schlick approximation             |
| Translucent| Glass-like Fresnel + interior medium | delta (surface) + Beer–Lambert |
| Glossy     | GGX microfacet (Cook-Torrance)   | VNDF sampling (Heitz 2018)        |

All BSDF evaluations are spectral: albedo $K_d(\lambda)$ or $K_s(\lambda)$
is evaluated per-bin. For glass with wavelength-dependent IOR, Fresnel
reflectance and refraction angle vary per bin (spectral dispersion).

### 9.0.1 `dev_specular_bounce` (Unified Glass/Mirror/Translucent)

All delta BSDF interactions (Glass, Mirror, Translucent) go through a
single device function:

```cuda
SpecularBounceResult dev_specular_bounce(
    float3 dir, float3 pos,
    float3 normal,        // shading normal (direction computation)
    float3 geo_normal,    // geometric normal (enter/exit + offset)
    uint32_t mat_id, float2 uv, PCGRng& rng,
    const int* hero_bins = nullptr,  // photon hero channels (nullptr = camera)
    int num_hero = 0,
    IORStack* ior_stack = nullptr);   // nested dielectric tracking
```

**Returns** `SpecularBounceResult { new_dir, new_pos, filter }` where
`filter` is a per-wavelength throughput multiplier.

**Glass/Translucent logic:**
1. **Entering test** uses `geo_normal`: `entering = dot(dir, geo_normal) < 0`
2. **Outside IOR** from stack: `outside_ior = ior_stack->top()` (air = 1.0)
3. **Hero IOR**: `hero_ior = ior_at_lambda(hero_bin)` (D-line for camera,
   `hero_bins[0]` for photons)
4. **Eta**: `entering ? outside_ior/hero_ior : hero_ior/outside_ior`
5. **Fresnel**: `F_hero = fresnel_dielectric(cos_i, eta_hero)`
6. **Stochastic choice**: reflect if `rng < F_hero`, else refract
7. **IOR stack update**: push on enter-refract, pop on exit-refract,
   no change on reflect
8. **Offset**: `pos ± outward_geo * EPSILON` (geometric normal)
9. **Dispersion filter**: per-wavelength Cauchy correction (§5.1.2)

**Mirror logic:** Pure reflection using shading normal for direction,
geometric normal for epsilon offset. `filter = Ks`.

### 9.1 Chromatic Dispersion (Cauchy Equation)

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

## 10. Debug / Component Outputs

### 10.1 Output Files

| File | Contents |
|------|----------|
| `out_nee_direct.png` | NEE-only direct component |
| `out_photon_indirect.png` | Photon density only |
| `out_combined.png` | Sum of above |
| `out_photon_caustic.png` | Caustic map contribution |

Multi-frame naming: `frame_NNNN_out_*.png`

### 10.2 Render Modes

| Mode | Description |
|------|-------------|
| First-hit debug | Normals / material ID / depth (no photon gather, no NEE) |
| DirectOnly | NEE direct lighting only (no photon gather) |
| IndirectOnly | Photon density terms only (no NEE) |
| Combined | Direct + indirect |
| PhotonMap | Raw photon density visualisation (heat map) |

### 10.3 Debug Viewer Key Bindings

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

### 10.4 Hover Cell Overlay

When hovering over a spatial cell with mouse released and map toggle enabled
(F2/F3): cell coordinate, photon count, sum/average flux, dominant
wavelength, gather radius, map type (global / caustic).

---

## 11. CPU vs GPU: Dual Implementation

### 11.1 Two Implementations

| | CPU Reference | GPU (OptiX) |
|---|---|---|
| Purpose | Ground truth, validation | Interactive, production |
| Spatial index | KD-tree (arbitrary radius) | Uniform grid / hash grid |
| Photon transport | Identical algorithm | Identical algorithm |
| Gather | KD-tree range query + tangential kernel | Grid shell-expansion k-NN + tangential kernel |
| Allowed tweaks | None — exact physics | Approximate kernels, capped photon counts |
| RNG | PCG, deterministic seed | PCG, same seed → same result |

### 11.2 Shared Code (Header-Only or Templated)

| Component | Location | Shared? |
|---|---|---|
| `Spectrum`, `PhotonSoA` | `src/core/` | Yes |
| KD-tree build + query | `src/photon/kd_tree.h` | Yes (CPU reference) |
| Hash grid build | `src/photon/hash_grid.h` | Yes (CPU build); GPU query in CUDA |
| BSDF evaluate/sample/pdf | `src/bsdf/bsdf.h` | Yes |
| Emitter sampling | `src/photon/emitter.h` | Yes |
| Density estimator + surface filter | `src/photon/density_estimator.h` | Yes |

### 11.3 CPU Reference Renderer

- `Renderer::build_photon_maps()` — trace photons, build KD-tree
- `Renderer::render_frame()` — first-hit → NEE + photon gather

### 11.4 GPU Renderer

- `OptixRenderer::trace_photons()` — GPU photon emission + bounce
- `OptixRenderer::render_one_spp()` — first-hit, NEE, photon gather via hash grid + tangential kernel

### 11.5 Parity Contract

| Mode | Contract | Verification |
|------|----------|--------------|
| **Normal** | Distributionally equivalent results | Integration tests: PSNR thresholds |
| **Deterministic debug** | Bitwise-reproducible within same platform | `--deterministic` flag |

### 11.6 Allowed GPU Tweaks

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

## 12. Integration Tests (CPU ↔ GPU)

### 12.1 Test Scene

Binary Cornell Box (5 grey/coloured walls, 1 emissive quad). Resolution:
64×64 for fast testing.

### 12.2 Test Harness

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

### 12.3 Test Cases

| Test | What it validates | Threshold |
|---|---|---|
| `CPU_GPU.DirectLightingMatch` | NEE-only renders agree | PSNR > 40 dB |
| `CPU_GPU.PhotonIndirectMatch` | Photon density agreement | PSNR > 30 dB |
| `CPU_GPU.CombinedMatch` | Full render agreement | PSNR > 30 dB |
| `CPU_GPU.CausticMapMatch` | Caustic photon contribution | PSNR > 25 dB |
| `CPU_GPU.AdaptiveRadiusMatch` | k-NN adaptive radius | PSNR > 25 dB |
| `CPU_GPU.EnergyConservation` | Total energy within 5% | ratio ∈ [0.95, 1.05] |
| `CPU_GPU.NoNegativeValues` | No negative radiance | max(min_pixel) ≥ 0 |
| `CPU_GPU.SpectralBinIsolation` | No cross-bin contamination | exact match |
| `CPU_GPU.DifferenceImage` | Save diff image for inspection | always pass |

### 12.4 Output Artifacts

Each run saves to `tests/output/integration/`:
- `cpu_combined.png`, `gpu_combined.png`
- `cpu_nee.png`, `gpu_nee.png`
- `cpu_indirect.png`, `gpu_indirect.png`
- `diff_combined.png` (amplified difference)
- `integration_report.txt`

---

## 13. Acceptance Tests

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

## 14. Adaptive Sampling

Screen-noise adaptive sampling concentrates samples in high-variance regions
and skips converged pixels once estimated noise falls below a configurable
threshold.

### 14.1 Noise Metric

Per-pixel relative standard error of CIE Y (luminance) of the NEE
direct-only signal:

$$
r_i = \frac{\mathrm{se}_i}{|\mu_i| + \varepsilon}, \quad \varepsilon = 10^{-4}
$$

The active mask uses a neighbourhood maximum over a $(2R+1) \times (2R+1)$
window. A pixel is active if $r_i^{\text{nbr}} > \tau$.

### 14.2 Sampling Policy

| Phase | Pass range | Behaviour |
|-------|------------|-----------|
| Warmup | 0 to `min_spp - 1` | All pixels active |
| Adaptive | `min_spp` to `max_spp` | Mask recomputed; converged pixels skipped |

### 14.3 Configuration

| Field | Default | Notes |
|-------|---------|-------|
| `adaptive_sampling` | `false` | Opt-in |
| `adaptive_min_spp` | 4 | Warmup passes |
| `adaptive_max_spp` | 0 | 0 → inherits `samples_per_pixel` |
| `adaptive_update_interval` | 1 | Mask refresh period |
| `adaptive_threshold` | 0.02 | 2% relative noise target |
| `adaptive_radius` | 1 | Neighbourhood half-width |

### 14.4 Implementation

- **GPU**: Three per-pixel buffers (`d_lum_sum_`, `d_lum_sum2_`,
  `d_active_mask_`) in `OptixRenderer`. The `k_update_mask` CUDA kernel in
  `src/optix/adaptive_sampling.cu` computes the mask.
- **CPU**: Equivalent buffers in `FrameBuffer` with matching logic.

---

## 15. Photon Map Persistence (Binary Save/Load)

### 15.1 Design

The photon pass and camera pass are fully independent. The photon map is a
static light-field snapshot that can take minutes to hours to compute.
Saving it enables:

1. **Instant startup** — skip expensive tracing on subsequent launches
2. **Quality iteration** — precompute with 10M+ photons, render interactively
3. **Reproducibility** — deterministic snapshot of indirect light field
4. **Camera independence** — move camera freely without recomputing

### 15.2 Binary Format

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

### 15.3 Scene Hash

Hash inputs: geometry + materials + emitters + `light_scale`. Camera
position does NOT invalidate. Uses xxhash64 over .obj + .mtl file contents.

### 15.4 Cache File Location

Saved in the scene folder alongside `.obj` / `.mtl`:
```
scenes/cornell_box/
    CornellBox-Original.obj
    CornellBox-Original.mtl
    photon_cache.bin          ← auto-generated
```

### 15.5 CLI Flags

```
--photon-file <path>     Explicit photon cache file path
--force-recompute        Ignore cached photon file
--no-save-photons        Do not save photon cache
--photon-budget <N>      Number of photons to emit
```

### 15.6 Startup Logic

```
load scene → compute scene hash → check cache validity
  if valid: load from cache
  else: trace photons → build index → save cache
camera pass → render interactively
```

**P key** triggers full photon recomputation at any time.

Implementation: `src/photon/photon_io.h` + `src/photon/photon_io.cpp`.

---

## 16. OptiX Program Structure

All device code in `src/optix/optix_device.cu`, compiled to PTX via `nvcc`
with `--use_fast_math`.

### Programs

| Program                           | Type        | Purpose                              |
|-----------------------------------|-------------|--------------------------------------|
| `__raygen__render`                | Ray Gen     | Camera path trace (`full_path_trace_v3<true>`) + accumulation |
| `__raygen__photon_trace`          | Ray Gen     | GPU photon emission + tracing        |
| `__raygen__targeted_photon_trace` | Ray Gen     | Targeted caustic photon emission     |
| `__closesthit__radiance`          | Closest-Hit | Unpack geometry at hit point         |
| `__closesthit__shadow`            | Closest-Hit | Set "occluded" flag                  |
| `__miss__radiance`                | Miss        | Return zero radiance                 |
| `__miss__shadow`                  | Miss        | Return "not occluded"                |

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

## 17. Configuration

All tunable constants in `src/core/config.h`:

### Sampling & Bounces

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_LAMBDA` | 32 | Wavelength bins (380–780 nm) |
| `HERO_WAVELENGTHS` | 4 | Hero wavelengths per photon (PBRT v4 style) |
| `DEFAULT_SPP` | 16 | Samples per pixel |
| `DEFAULT_PHOTON_MAX_BOUNCES` | 12 | Maximum photon bounce depth |
| `DEFAULT_MAX_SPECULAR_CHAIN` | 12 | Camera ray specular bounce limit |
| `DEFAULT_MAX_GLOSSY_BOUNCES` | 2 | Glossy continuation after first diffuse hit |
| `DEFAULT_PHOTON_MIN_BOUNCES_RR` | 10 | Guaranteed bounces before Russian roulette |
| `DEFAULT_PHOTON_RR_THRESHOLD` | 0.90 | Max RR survival probability |
| `DEFAULT_LIGHT_CONE_HALF_ANGLE_DEG` | 90.0 | Emission cone half-angle (90° = full hemisphere) |

### Photon Mapping

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_GLOBAL_PHOTON_BUDGET` | 2,000,000 | Diffuse indirect photon budget |
| `DEFAULT_CAUSTIC_PHOTON_BUDGET` | 2,000,000 | Caustic photon budget |
| `DEFAULT_GATHER_RADIUS` | 0.05 | Global map max kNN radius |
| `DEFAULT_CAUSTIC_RADIUS` | 0.025 | Caustic map max kNN radius |
| `HASHGRID_CELL_FACTOR` | 2.0 | cell_size = factor × radius |
| `DEFAULT_SURFACE_TAU` | 0.02 | Plane-distance filter thickness |
| `DEFAULT_PLANE_DISTANCE_THRESHOLD` | 1e-3 | Max plane distance for tangential kernel |
| `PLANE_TAU_EPSILON_FACTOR` | 10.0 | tau ≥ 10 × ray_epsilon |
| `DEFAULT_PHOTON_BOUNCE_STRATA` | 64 | Max hemisphere strata for decorrelation |
| `DEFAULT_GPU_MAX_GATHER_RADIUS` | 0.5 | Max gather radius for GPU grid k-NN |
| `DEFAULT_KNN_K` | 100 | k-NN neighbour count |
| `DEFAULT_TARGETED_CAUSTIC_MIX` | 1.0 | Fraction of caustic budget targeted |
| `PHOTON_MAP_POOL_SIZE` | 4 | Pre-build this many maps at render start |

### Guided Sub-Path Sidecar (§4a)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_USE_GUIDE` | `true` | Master switch for the guided sidecar |
| `DEFAULT_GUIDE_FRACTION` | 0.5 | Legacy enable flag (> 0 enables sidecar at runtime) |
| `MAX_GUIDE_PDF_PHOTONS` | 32 | Max photons considered in reservoir sampling |
| `DENSE_GRID_CELL_SIZE` | 0.01 | Dense grid cell side-length (metres) |
| `DEFAULT_PHOTON_GUIDE_CONE_HALF_ANGLE` | 0.15 | Cone jitter half-angle (radians, ~8.6°) |
| `GUIDE_USE_NEIGHBOURHOOD` | `true` | Use 3×3×3 cell neighbourhood (false = single cell) |
| `DEV_PHOTON_FLAG_CAUSTIC_MASK` | 0x12 | CAUSTIC_GLASS \| CAUSTIC_SPECULAR — triggers max(N,G) mixing |

### NEE

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_NEE_LIGHT_SAMPLES` | 4 | Shadow rays at bounce 0 |
| `DEFAULT_NEE_COVERAGE_FRACTION` | 0.3 | Coverage-aware sampling factor $c$ |

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

## 18. Key Data Structures

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

### LaunchParams
Contains all device pointers: framebuffer, scene geometry, materials, photon
map, hash grid, emitter CDF, camera, rendering flags (`is_final_render`,
`render_mode`), and adaptive sampling pointers.

---

## 19. Source Layout

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
    optix_path_trace_v3.cuh     Full path trace kernel (template <bool EnableGuideSidecar>)
    optix_guided.cuh            Dense-grid reservoir sampling + guide PDF evaluation
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

## 20. GPU Performance Optimizations

### 20.1 DeviceBuffer Allocation Amortization (`ensure_alloc`)

The original code called `DeviceBuffer::alloc()` (→ `cudaFree` + `cudaMalloc`)
on every SPP for `LaunchParams` and auxiliary buffers. For a 64-SPP render
this produced ~450 unnecessary CUDA API round-trips.

`DeviceBuffer::ensure_alloc(size_t n)` replaces `alloc()` at all hot sites:

```cpp
void ensure_alloc(size_t n) {
    if (bytes >= n) return;   // already large enough
    alloc(n);
}
```

After the first SPP the buffer is already the correct size, so every
subsequent call is a no-op pointer comparison. Seven allocation sites in
`optix_renderer.cpp` were converted (`d_launch_params_`, framebuffer arrays,
accumulation buffers, active mask, and luminance statistics).

### 20.2 Emissive Inverse-Index Table

`dev_light_pdf()` originally performed an $O(N)$ linear scan over
`emissive_tri_indices[]` to map an arbitrary `tri_id` to its position in
the emissive CDF. With 400+ emissive triangles and ~4 shadow rays per
pixel, this dominated instruction count.

A precomputed inverse-index array `emissive_local_idx[num_triangles]` is
now uploaded alongside the emitter CDF in `upload_emitter_data()`:

```cpp
std::vector<int> local_idx(scene.triangles.size(), -1);
for (size_t i = 0; i < n; ++i)
    local_idx[scene.emissive_tri_indices[i]] = (int)i;
d_emissive_local_idx_.upload(local_idx);
```

`dev_light_pdf()` uses `params.emissive_local_idx[tri_id]` for $O(1)$
lookup with a linear-scan fallback when the pointer is `nullptr`.

Memory overhead: 4 bytes per triangle (negligible).

### 20.3 Photon Map Pool (Amortization)

Multi-map decorrelation (`MULTI_MAP_SPP_GROUP`) previously re-traced the
full photon pass at every SPP group boundary (every 4 camera samples).
For a 64-SPP render this meant 16 full photon traces — the dominant cost.

The **photon map pool** pre-builds several complete photon maps at the start
of `render_final()` and cycles through them during accumulation:

```
┌─────────────────── render_final() ───────────────────┐
│  Pre-build PHOTON_MAP_POOL_SIZE maps (different seeds)│
│  for s in 0..total_spp:                               │
│      if (s % MULTI_MAP_SPP_GROUP == 0):               │
│          activate map[s/group % pool_size]  ← O(1)    │
│      render_one_spp()                                  │
│  Release pool buffers                                  │
└───────────────────────────────────────────────────────┘
```

`PhotonMapSlot` holds a complete snapshot of the device-side photon SoA,
hash grid, and associated scalars. `swap_photon_map()` exchanges all 17+
device buffers and 4 scalar fields between the main members and a slot
via `std::swap` — no GPU copies, just pointer rotation.

| Configuration | Re-traces (64 SPP) | VRAM per map |
|---------------|---------------------|--------------|
| Legacy (pool=1) | 16 | 1× |
| Pool=4 | 4 (upfront) | 4× |

Trade-off: 4× VRAM for photon data, but ~4× faster wall-clock due to
eliminated redundant photon tracing during accumulation.

Configuration: `PHOTON_MAP_POOL_SIZE` in `config.h`.

---

## 21. Strengths

1. **Full spectral transport.** 32 wavelength bins; dispersion, metamerism,
   and spectral emission naturally captured without RGB approximations.

2. **Guided sub-path sidecar.** Photon directions seed full sub-paths at
   every non-delta bounce. The irradiance sidecar augments BSDF-sampled
   transport without replacing it — both the guide and the BSDF fire.

3. **Caustic-aware mixing.** Two mixing rules adapt to photon type:
   `max(N, G)` for caustics (take the stronger signal), `(N + G) / 2`
   for diffuse (average reduces variance).

4. **No firefly clamps needed.** The mixing rules naturally bound the
   guide contribution relative to NEE, eliminating the need for ad-hoc
   energy clamps.

5. **Precomputable photon map.** Binary save/load with scene hash — compute
   once, render interactively with arbitrary camera positions.

6. **Surface-aware gather.** Tangential disk kernel eliminates planar
   blocking artifacts and cross-surface leakage.

7. **Dual CPU/GPU implementation.** CPU reference renderer provides ground
   truth; GPU provides interactive speed. Integration tests verify parity.

8. **Adaptive gather radius.** k-NN per hitpoint adapts to local photon
   density automatically.

9. **Photon path decorrelation.** Cell-stratified bouncing ensures efficient
   coverage of the hemisphere.

10. **Coverage-aware NEE.** Mixture of power-weighted and area-weighted
    sampling ensures all emitters get shadow rays.

11. **Interactive debug viewer.** Multiple render modes and overlays for
    inspecting every intermediate quantity.

12. **Comprehensive test suite.** Unit tests, integration tests (CPU↔GPU),
    ground-truth comparisons, and per-ray validation.

13. **Chromatic dispersion.** Cauchy-equation wavelength-dependent IOR with
    per-bin Fresnel produces physically correct spectral splitting
    (prismatic rainbows, chromatic caustics).

14. **CellInfoCache precomputation.** Per-cell photon statistics (density,
    variance, caustic count, directional spread) enable adaptive gather
    radius and empty-region skip without per-query overhead.

13. **Adaptive caustic shooting.** Two-phase photon emission concentrates
    budget on high-variance caustic cells, reducing caustic noise.

14. **IOR stack for nested dielectrics.** 4-deep stack tracks surrounding
    medium IOR through nested glass objects, enabling correct Fresnel and
    Snell calculations for glass-inside-glass configurations.

15. **Geometric/shading normal split.** Entering test and epsilon offset
    use face (geometric) normals for robustness; refraction/reflection
    directions use interpolated (shading) normals for smooth appearance.

17. **Hero wavelength system.** PBRT v4-style 4-wavelength tracing per
    photon with stratified companion offsets maximises wavelength diversity
    and enables per-wavelength Cauchy dispersion filtering.

18. **Three-valued tag system.** Dual-budget photon gather with correct
    per-budget normalization. No photons are discarded.

19. **Direct spectral transmittance override.** `pb_tf_spectrum` MTL keyword
    bypasses the lossy RGB→spectrum matrix for precise glass colour control.

20. **GPU allocation amortization.** `ensure_alloc` avoids cudaFree/cudaMalloc
    churn; emissive inverse-index replaces O(N) scan in `dev_light_pdf()`;
    photon map pool eliminates redundant re-tracing during SPP accumulation.

---

## 22. Weaknesses and Limitations

1. **Sub-path cost per bounce.** The guided sidecar traces a full sub-path
   at every non-delta bounce. This multiplies the per-bounce ray cost but
   converges faster than BSDF-only paths in complex lighting.

2. **Single GAS, no instancing.** Flat triangle soup only.

3. **No texture mapping in OptiX.** Per-material properties, not per-texel.

4. **Hash grid built on CPU.** Photon data downloaded → grid built → reuploaded.

5. **No denoising.** Raw Monte Carlo output.

6. **Windows-centric build.** Tested primarily with MSVC on Windows.

7. **Volume rendering temporarily disabled.** Surface transport only.

8. **GPU k-NN bounded by max_radius.** Sparse regions may not find k photons.

9. **Translucent camera shading.** Translucent materials now have full
   interior medium transport (Beer–Lambert, free-flight, HG phase function,
   `MediumStack`). Dual-hemisphere NEE + photon gather at translucent
   surfaces is partially covered (NEE at medium scatter events); full
   surface-level dual-gather is future work.

10. **Dense grid resolution.** The guided sidecar depends on local photon
    density in the dense grid. In very sparse regions the sidecar may find
    no photons, falling back to NEE-only for that bounce.

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
- **Using shading normals for entering test** — use geometric normals (§5.2.7)
- **Not updating IOR stack on refract** — nested glass gets wrong eta
- **Changing IOR stack on reflect** — ray stays in same medium
- **Hardcoding outside_ior = 1.0** — fails for glass-inside-glass
- **Using bin 0 (380 nm) IOR for camera refraction** — use D-line (~589 nm)
- **Skipping tag=1 photons in gather** — discards valid global caustics
- **Mixing N_global and N_caustic normalizers** — wrong brightness in dual-budget
- CPU vs GPU: different structures OK, but gather metric must match
- **Firing sidecar at delta bounces** — sidecar is for non-delta only; delta bounces are deterministic
- **Allowing sub-paths to fire their own sidecars** — causes 2^N recursion; use `<false>` template
- **Not skipping bounce-0 emission in sub-paths** — double-counts with parent NEE
- **Using coin flip between guide and BSDF** — both must fire; BSDF always continues

---

## 24. Architectural Differences from v1

| Aspect | v1 | v2.x | v3 (Current) |
|--------|-----|------|---------------|
| Camera ray depth | Full path tracing (N bounces) | First-hit only (specular chain → first diffuse) | Full path tracing with guided sub-path sidecar at each non-delta bounce |
| Indirect lighting | Photon map + camera BSDF bouncing | Photon map density estimation only | Camera BSDF continuation + photon-guided sub-path irradiance |
| Guide role | None | Direct density estimate at hitpoint | Irradiance sidecar — photon directions seed sub-paths that return radiance |
| MIS | 3-way (NEE + BSDF + photon) | None at camera | 2-way NEE↔BSDF for emission; NEE↔guide mixing (max or average) |
| Mixing rule | N/A | N/A | Caustic photon → max(N,G); Diffuse photon → (N+G)/2 |
| Template recursion guard | N/A | N/A | `<bool EnableGuideSidecar>`: `<true>` = main, `<false>` = sub-path |
| Firefly clamps | N/A | MAX_BOUNCE_CONTRIBUTION, MAX_PATH_THROUGHPUT | Removed — mixing rules handle energy naturally |
| Dense grid | N/A | N/A | 3D uniform grid over photon AABB; 3×3×3 neighbourhood reservoir sampling |
| Photon bounce strategy | Standard BSDF | Stratified BSDF at bounce 0 | Stratified BSDF at bounce 0 |
| Spectral sampling | One wavelength per photon | Hero wavelength system: 4 stratified bins | Hero wavelength system: 4 stratified bins |
| Spatial index | Hash grid only | KD-tree (CPU) + Hash grid (GPU) + tangential kernel | Same + dense grid for guided sampling |
| Gather radius | Fixed or SPPM | k-NN adaptive per hitpoint (tangential) | k-NN adaptive per hitpoint (tangential) |
| Gather kernel | 3D spherical | Tangential disk kernel | Tangential disk kernel |
| Photon tag system | Boolean caustic flag | Three-valued tag (0/1/2) dual-budget | Three-valued tag + per-photon path_flags for sidecar mixing |
| Nested dielectrics | Hardcoded outside_ior = 1.0 | IOR stack (4-deep) | IOR stack (4-deep) |
| Photon precomputation | Recomputes every launch | Binary save/load | Binary save/load |
| Tone mapping | Reinhard | ACES Filmic | ACES Filmic |
| Cell-bin grid | Used for guided camera bouncing | Deleted (~800 lines) | Deleted; replaced by dense grid for sidecar |
| GPU buffer allocation | alloc() every SPP | ensure_alloc() | ensure_alloc() |

---

## 25. Removed / Deprecated Functionality

| Feature | Reason |
|---------|--------|
| First-hit-only camera pass (v2.x) | Replaced by full path trace with guided sidecar |
| Photon density estimation at camera hitpoint (v2.x) | Replaced by guided sub-path irradiance |
| `MAX_BOUNCE_CONTRIBUTION` clamp | Removed — mixing rules handle energy naturally |
| `MAX_PATH_THROUGHPUT` clamp | Removed — mixing rules handle energy naturally |
| MIS 3-way weighting (v1) | Camera uses 2-way NEE↔BSDF for emission MIS |
| Coin-flip guide selection | BSDF always continues; guide is an additive sidecar |
| `dev_sample_guided_bounce()` (v1) | Replaced by `dev_guide_sample_and_pdf_full()` reservoir sampling |
| Dense cell-bin grid (`CellBinGrid`) | Deleted; replaced by dense uniform grid |
| Volume rendering | Temporarily disabled for surface validation |
| Obsolete tests (multi-bounce camera, MIS weights, guided bouncing) | Fully deleted, replaced by new tests |
