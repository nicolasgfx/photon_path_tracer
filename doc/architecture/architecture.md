# Architecture — GPU Path Tracer (v5)

> **Scope.** This document describes the v5 renderer: an RGB GPU path
> tracer with NEE, caustic light tracing, and PBRT v4 scene support.
> It is reverse-engineered from the implementation in `src/`.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Non-Negotiable Invariants](#2-non-negotiable-invariants)
3. [Physical Units & Definitions](#3-physical-units--definitions)
4. [Pipeline Stages](#4-pipeline-stages)
5. [Path Tracer (Camera Pass)](#5-path-tracer-camera-pass)
6. [Caustic Light Tracing](#6-caustic-light-tracing)
7. [Post-Processing Pipeline](#7-post-processing-pipeline)
8. [BSDF Models](#8-bsdf-models)
9. [Scene Loading](#9-scene-loading)
10. [OptiX Program Structure](#10-optix-program-structure)
11. [Acceleration & SBT](#11-acceleration--sbt)
12. [Camera Model](#12-camera-model)
13. [Configuration](#13-configuration)
14. [Debug & Diagnostics](#14-debug--diagnostics)
15. [Analysis Pipeline](#15-analysis-pipeline)
16. [Test Suite](#16-test-suite)
17. [Key Data Structures](#17-key-data-structures)
18. [Source Layout](#18-source-layout)
19. [GPU Performance Notes](#19-gpu-performance-notes)

---

## 1. Overview

The renderer is a **physically-based GPU path tracer** built on
NVIDIA OptiX 9, CUDA 12, and C++17 for Windows/MSVC.

### 1.1 Core Design

- **RGB colour pipeline** — all light transport uses `Color3 {r, g, b}`.
  No spectral binning. Cauchy dispersion uses 3 representative
  wavelengths but routes each to an independent RGB channel.
- **Iterative bounce loop** — camera rays bounce through the scene
  collecting radiance via BSDF importance sampling and NEE.
- **Caustic light tracing** — dedicated forward pass tracing photons
  from emitters through mirror/glass delta chains, splatting onto the
  camera sensor.
- **Post-processing** — GPU firefly filter, optional OptiX AI denoiser,
  bloom, ACES filmic tone mapping.
- **PBRT v4 scenes** — native loader with 13+ material types mapped to
  9 internal GPU shading models.

### 1.2 Priority Order

1. **Physical correctness** — unbiased light transport, energy
   conservation.
2. **Image quality** — low variance at practical sample counts.
3. **GPU throughput** — SoA memory layout, warp-coherent paths, minimal
   divergence.

### 1.3 Requirements

| Component | Minimum |
|---|---|
| GPU | NVIDIA Turing (sm_75+) |
| CUDA Toolkit | 12.x |
| OptiX SDK | 9.x (`OptiX_INSTALL_DIR` env var) |
| CMake | 3.24+ |
| Compiler | MSVC (Visual Studio 2022) |
| C++ Standard | C++17 |
| OS | Windows 10+ |

---

## 2. Non-Negotiable Invariants

These invariants must hold after every code change.

1. **Energy conservation** — no surface interaction may create energy.
   BSDF `f·cos/pdf` must not exceed 1 in expectation for any
   (ωi, ωo) ensemble (white furnace test).
2. **Unbiased accumulation** — the progressive mean over N frames must
   converge to the true rendering equation solution. No non-zero bias
   terms outside optional clamping.
3. **MIS correctness** — NEE and BSDF weights from `mis_weight_2()`
   (power heuristic β=2) must use matching PDFs. A mismatch introduces
   bias.
4. **Delta surfaces** — mirror, glass, and translucent materials skip
   NEE (their BSDF PDF is zero for any finite-area sample). Emission
   MIS uses full weight (`w_bsdf = 1`) after a delta bounce.
5. **Adjoint symmetry** — light-side paths (caustic tracer) and
   camera-side paths produce the same radiance in the limit. The η²
   correction at refractive interfaces must be applied consistently.
6. **Normal consistency** — the shading normal is used for BSDF
   evaluation; the geometric normal defines the offset direction.
   Throughput is never negative.

---

## 3. Physical Units & Definitions

| Symbol | Name | Unit | Notes |
|---|---|---|---|
| $L$ | Radiance | W·m⁻²·sr⁻¹ | Per-channel (R, G, B) |
| $\Phi$ | Flux / power | W | Photon throughput in caustic tracer |
| $f_r$ | BSDF | sr⁻¹ | Bidirectional scattering distribution |
| $p(\omega)$ | PDF | sr⁻¹ | Solid-angle probability density |
| $L_e$ | Emission | W·m⁻²·sr⁻¹ | Emissive surface radiance |
| $T_f$ | Transmittance filter | – | Per-channel [0,1] glass colour |
| $\eta$ | Refractive index | – | External re: vacuum = 1.0 |
| $\alpha$ | GGX roughness² | – | Minimum 0.001 |

### 3.1 Bias Sources

The renderer is unbiased when clamping is disabled
(`DEFAULT_CLAMPING_ENABLED = false`). Optional clamping introduces
controlled bias to suppress fireflies:

| Clamp | LaunchParams field | Default | Effect |
|---|---|---|---|
| Per-bounce f·cos/pdf | `max_bounce_contribution` | 10⁴ | Caps extreme BSDF weights |
| Path throughput | `max_path_throughput` | 10⁴ | Caps cumulative throughput |
| NEE contribution | `max_nee_contribution` | 10⁴ | Caps per-bounce NEE |
| Sample luminance | `max_sample_luminance` | 10⁴ | Caps final sample radiance |
| Caustic splat | `caustic_max_splat_luminance` | 100 | Caps per-photon splat |

All clamps preserve hue (uniform per-channel scaling when triggered).

---

## 4. Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Scene Load                                                   │
│     OBJ/MTL or PBRT v4 → triangles, materials, textures         │
│     Emissive distribution: power-weighted CDF                   │
│     Delta surface distribution: area-weighted CDF (for caustics)│
├─────────────────────────────────────────────────────────────────┤
│  2. OptiX Setup                                                  │
│     GAS (flat) or IAS (instanced) acceleration structure        │
│     4 raygen + 2 miss + 2 hitgroup program groups               │
│     Per-raygen SBT                                              │
├─────────────────────────────────────────────────────────────────┤
│  3. Render Loop (per frame, progressive accumulation)            │
│     a. Camera path tracing  ── §5                               │
│     b. Caustic light tracing ── §6  (optional)                  │
│     c. Post-processing       ── §7                              │
│        Firefly → [OptiX denoiser] → Bloom → ACES → sRGB        │
│     d. Display (GLFW viewer / snapshot export)                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.1 Scene Loading

See [§9. Scene Loading](#9-scene-loading) for full details.

**Summary:** The loader reads OBJ/MTL or PBRT v4, produces flat triangle
arrays with per-vertex (position, normal, tangent, UV) and per-triangle
material ID. PBRT instancing is supported via IAS. Textures are uploaded
as hardware-sampled CUDA texture objects.

**Emissive distribution:** `area × mean_emission` → power-weighted CDF
for NEE sampling (binary search on GPU, O(log N) per sample). An inverse
index `emissive_local_idx[tri]` maps triangle ID → local emitter index
for emission MIS weight computation.

**Delta surface distribution:** All mirror/glass triangles →
area-weighted CDF used by the caustic tracer's Strategy B (targeted
delta-surface emission).

### 4.2 Acceleration Structure

- **GAS (single geometry):** One bottom-level structure containing all
  triangles. Compacted for minimum GPU memory.
- **IAS (instanced):** Per-mesh GAS stored separately; instance
  transforms applied via `OptixInstance`. The closest-hit shader applies
  object→world transforms when `has_instances == 1`.

### 4.3 Emitter Sampling

Power-weighted CDF built on the host from
`area(tri) × max_component(Le)`. The GPU performs binary search on
`emissive_cdf[]` to select a triangle, then uniform barycentric sampling
for a point on the triangle. PDF:

$$p_\text{tri} = \frac{\text{power}(i)}{\sum_j \text{power}(j)}
\qquad
p_\text{point} = \frac{1}{\text{area}(i)}$$

Area-to-solid-angle conversion for NEE:

$$p_\omega = p_\text{tri} \cdot \frac{1}{\text{area}} \cdot \frac{d^2}{\cos\theta_\text{emitter}}$$

---

## 5. Path Tracer (Camera Pass)

**Source:** `src/renderer/integrator/path_tracer.cuh`
**Entry:** `__raygen__render` in `optix_programs.cu`

### 5.1 Per-Pixel Ray Generation

For each pixel, each sample:

1. **Sub-pixel jitter:** `(px + ξ_x, py + ξ_y)` where ξ ∈ [0,1) via
   PCG RNG.
2. **RNG seeding:** `PCGRng::seed(frame * 65537 + sample * 104729 + 42,
   pixel_idx + 1)` — decorrelated across frame, sample, and pixel.
3. **Camera ray:** Pinhole:
   `dir = normalize(cam_w + cam_u·(2u−1) + cam_v·(2v−1))`.
   When DOF is active (`cam_lens_radius > 0`), thin-lens disk sampling
   is applied (see [§12](#12-camera-model)).
4. **Progressive accumulation:** Frame 0 writes directly. Frame N > 0:
   `w_old = old_count / new_count`, `w_new = spp / new_count`.

### 5.2 Bounce Loop

```
trace_path(origin, direction, rng):
    throughput = (1, 1, 1)
    radiance   = (0, 0, 0)
    pdf_prev   = 0             // 0 signals "camera ray" or "delta bounce"
    had_nondelta = false

    for bounce in 0 .. max_bounces:
        trace_radiance(origin, dir) → hit
        if miss: break

        apply normal_map / bump_map perturbation

        ── Emissive hit ──
        if Le > 0:
            if bounce == 0 or pdf_prev == 0:
                w_bsdf = 1
            else:
                p_nee = light_pdf(hit)
                w_bsdf = pdf_prev² / (pdf_prev² + p_nee²)    // MIS
            radiance += throughput * Le * w_bsdf
            break

        ── Delta surface (Mirror / Glass / Translucent) ──
        if is_delta(material):
            if suppress_camera_caustics and had_nondelta: break
            sample BSDF (deterministic reflect or refract)
            throughput *= f·|cos| / pdf
            clamp throughput (if enabled)
            pdf_prev = 0                 // delta → full weight on next Le
            continue

        had_nondelta = true
        write AOV (first non-specular: albedo + shading normal)

        ── NEE: 1 shadow ray ──
        select emitter from CDF
        sample point on emitter triangle
        if unoccluded:
            f_nee = bsdf_evaluate(wo, wi_light)
            w_nee = p_light² / (p_light² + p_bsdf²)
            radiance += throughput * Le * f_nee * cos / p_light * w_nee
        clamp NEE contribution (if enabled)

        ── BSDF importance sample ──
        (wi, f_over_pdf, pdf) = bsdf_sample(wo)
        throughput *= f_over_pdf
        clamp bounce / path throughput (if enabled)

        ── Russian roulette ──
        if bounce >= min_bounces_rr:
            p_survive = min(rr_threshold, max_component(throughput))
            if ξ >= p_survive: break
            throughput /= p_survive

        origin = offset(hit_pos, geo_normal, wi)
        dir = wi
        pdf_prev = pdf

    clamp_sample_luminance(radiance)   // if enabled
    return radiance
```

### 5.3 NEE Implementation

**Source:** `src/renderer/integrator/optix_nee.cuh`

| Property | Detail |
|---|---|
| Emitter selection | Binary search on power-weighted CDF |
| Point sampling | Uniform barycentric on selected triangle |
| PDF | `p_tri / area · d² / cos_emitter` (area → solid angle) |
| MIS weight | `mis_weight_2(p_light, p_bsdf)` — power heuristic β=2 |
| Shadow ray | `TERMINATE_ON_FIRST_HIT + DISABLE_CLOSESTHIT`, tmin=1e-4, tmax=dist−1e-4 |
| Delta surfaces | `bsdf_evaluate()` returns (0,0,0); `bsdf_pdf()` returns 0 — correctly produces zero NEE contribution |
| Shadow anyhit | Stochastic alpha test (TEA4 hash) — glass/translucent are opaque in shadow |

### 5.4 Emission MIS

When a BSDF-sampled ray hits an emissive surface:

- **After delta bounce** (`pdf_prev == 0`): full weight `w_bsdf = 1`
  (NEE could not have estimated this path — the previous surface was a
  delta distribution).
- **After non-delta bounce** (`pdf_prev > 0`): power heuristic
  `w_bsdf = pdf_prev² / (pdf_prev² + p_nee²)`.
- **Bounce 0** (direct camera ray hits emitter): full weight.

### 5.5 Russian Roulette

**Source:** `src/renderer/integrator/russian_roulette.h`

```
p_survive = min(rr_threshold, max_component(throughput))
if p_survive < 1e-4: terminate
if ξ >= p_survive:   terminate
else: throughput /= p_survive
```

Defaults: `min_bounces_rr = 3`, `rr_threshold = 0.95`.
Specular bounces do NOT exempt from RR — all bounces past `min_bounces_rr`
are subject to termination.

### 5.6 Sample Clamping

**Source:** `src/renderer/integrator/sample_clamping.h`

Three-level firefly safety net, gated by `clamping_enabled`
(default: **off**):

1. **Per-bounce** — `clamp_bounce_contribution(f_over_pdf, limit)`:
   per-channel clamp.
2. **Path throughput** — `clamp_path_throughput(throughput, limit)`:
   uniform scale so `max_component ≤ limit` (preserves hue).
3. **Sample luminance** — `clamp_sample_luminance(L, limit)`: uniform
   scale on final radiance (preserves hue).

### 5.7 Caustic De-duplication

When the caustic tracer is active, the camera pass terminates paths that
hit a delta surface after a non-delta surface (`had_nondelta && is_delta`).
These L…D→S⁺→D paths are handled more efficiently by the light-side
tracer (§6). Toggle: **F3** key.

---

## 6. Caustic Light Tracing

**Source:** `src/renderer/integrator/caustic_tracer.cuh`
**Entry:** `__raygen__caustic` in `optix_programs.cu`
**Grid:** `(num_photons, 1, 1)` — one thread per photon.

### 6.1 Emission

Each photon selects an emitting triangle via the power-weighted CDF,
then samples a surface point and cosine-hemisphere direction. Two
emission strategies alternate per photon (50/50 deterministic split):

| Strategy | Selection | Throughput |
|---|---|---|
| **A**: Cosine hemisphere | Random direction from emitter surface | `Le · π · emit_area / pdf_tri` |
| **B**: Targeted delta | Aim at random point on a mirror/glass triangle (area-weighted `delta_cdf`) | `Le · cos_emit · emit_area · cos_target · delta_total_area / (pdf_tri · d²)` |

Strategy B increases the probability of reaching delta surfaces,
reducing variance for concentrated caustic patterns.

### 6.2 Delta Chain Tracing

```
for bounce in 0 .. max_bounces:
    trace ray → hit
    if miss or emissive: return

    if delta surface:
        had_delta = true
        BSDF sample (with dispersive IOR for glass)
        throughput *= f · |cos| / pdf
        Russian roulette (same as §5.5)
        continue

    if non-delta AND had_delta:
        camera connection → splat
        return

    if non-delta AND NOT had_delta:
        BSDF sample, continue  (searching for delta surface)
```

The photon is discarded if it never passes through a delta surface.
Only one camera connection per photon (at the first non-delta hit after
a delta chain).

### 6.3 Cauchy Chromatic Dispersion

Each photon randomly selects one of three wavelength channels:

| Channel | Wavelength | Index |
|---|---|---|
| R | 630 nm | 0 |
| G | 532 nm | 1 |
| B | 465 nm | 2 |

At glass/translucent bounces, if the material has `cauchy_B ≠ 0`:

$$n(\lambda) = A + \frac{B}{\lambda^2}$$

`dev_glass_sample_dispersive()` computes per-wavelength IOR and
Fresnel weights. The splat is weighted 3× into the selected channel,
zero in the others — producing physically motivated spectral separation.

### 6.4 Camera Projection

When a photon exits the delta chain and strikes a diffuse surface:

1. **Inverse pinhole** (`dev_world_to_pixel`): Solve
   `P − cam_pos = λ(cam_w + a·cam_u + b·cam_v)` via 3×3 Cramer's rule.
   Reject if behind camera (λ ≤ 0) or outside NDC [0,1)².
2. **BSDF evaluation**: `f_bsdf = bsdf_evaluate(wo_photon, wi_camera)`
   at the hit surface — ensures correct angular distribution.
3. **Visibility test**: Shadow ray from hit point to camera position.
4. **Sensor weighting**:
   $W_e = \frac{1}{\cos^3\theta \cdot \Omega_\text{pixel}}$
   where $\Omega_\text{pixel} = \frac{4 \|cam_u\| \|cam_v\|}{W \times H}$.
5. **Splat**: `atomicAdd(caustic_r/g/b[pixel], L)` into dedicated
   additive buffers. Per-splat luminance clamped at
   `caustic_max_splat_luminance`.

### 6.5 Caustic Compositing

Caustic buffers (`caustic_r/g/b`) are additive across frames and
composited into the main image during post-processing (§7). The buffers
are normalised by the total photon count accumulated so far.

---

## 7. Post-Processing Pipeline

**Source:** `src/renderer/postfx/postfx_pipeline.h` / `.cpp`

### 7.1 Execution Order

```
RGB SoA accum → HDR float4 normalize → Firefly Filter
                                            │
                          ┌─────────────────┤
                          ▼                 ▼
                   [OptiX Denoiser]    (passthrough)
                          │                 │
                          └────────┬────────┘
                                   ▼
                            Bloom → ACES → sRGB RGBA8
```

Two-phase API for external denoiser integration:

- **Phase 1** (`apply_phase1`): SoA → float4 → firefly → returns HDR
  buffer pointer.
- *Denoiser runs externally on the HDR buffer.*
- **Phase 2** (`apply_phase2`): Bloom → tonemap → sRGB.

### 7.2 Stages

#### 7.2.1 Firefly Filter

**Source:** `src/renderer/postfx/firefly_filter.h` / `.cu`

Median + MAD (median absolute deviation) outlier suppression.

- Kernel radius: 1 pixel (3×3 neighbourhood).
- Threshold: 4.0 MAD multipliers.
- Operates on linear HDR data, per-channel.
- Default: **enabled**.

#### 7.2.2 OptiX AI Denoiser

**Source:** `src/renderer/postfx/optix_denoiser.h` / `.cpp`

- Uses albedo and shading-normal guide layers for edge-aware denoising.
- Blend parameter: 0 = full denoise, 1 = passthrough.
- Default: **disabled**.

#### 7.2.3 Bloom

**Source:** `src/renderer/postfx/bloom.h` / `.cu`

Mip-chain Gaussian blur with luminance threshold.

- 5 mip levels, separable 1D horizontal/vertical passes.
- Luminance threshold derived from scene max luminance.
- Configurable intensity (default 0.5) and radius (default 15).
- Default: **disabled**.

#### 7.2.4 Tone Mapping

**Source:** `src/renderer/postfx/tonemap.h` / `.cu`

ACES filmic curve (Narkowicz 2015) + sRGB gamma (2.2).
Configurable exposure multiplier (default 1.0).

$$L_\text{ACES}(x) = \frac{x(2.51x + 0.03)}{x(2.43x + 0.59) + 0.14}$$

Toggle: **F2** key.

### 7.3 Buffers

| Buffer | Size | Purpose |
|---|---|---|
| `d_hdr_` | W×H×4 float | Working HDR (normalised + firefly output) |
| `d_firefly_temp_` | W×H×4 float | Scratch for firefly filter |
| `d_max_lum_` | 1 float | Bloom max luminance |
| `d_mip_[0..4]` | Per-level | Bloom mip chain |
| `d_mip_tmp_[0..4]` | Per-level | Bloom separable pass scratch |

---

## 8. BSDF Models

**Source:** `src/renderer/material/bsdf.h`, `bsdf_shared.h`,
`specular.h`

### 8.1 Material Types

```cpp
enum MaterialType : uint8_t {
    Lambertian          = 0,
    Mirror              = 1,
    Glass               = 2,
    GlossyMetal         = 3,
    Emissive            = 4,
    GlossyDielectric    = 5,
    Translucent         = 6,
    Clearcoat           = 7,
    Fabric              = 8,
    DiffuseTransmission = 9
};
```

`Emissive` (4) uses Lambertian shading for indirect hits; emission is
handled separately via the `Le` field.

### 8.2 Material Details

| Type | Sampling | BSDF |
|---|---|---|
| **Lambertian** | Cosine-weighted hemisphere | $f = K_d / \pi$ |
| **Mirror** | Perfect reflection: $(-\omega_o.x, -\omega_o.y, \omega_o.z)$ | Delta, $f = K_s / |\cos\theta|$, pdf = 1 |
| **Glass** | Stochastic Fresnel: reflect with $P = F$, refract with $P = 1-F$ | Delta, exact dielectric Fresnel, $T_f$ filter on transmission. TIR fallback to reflection. Thin dielectric variant (straight-through). |
| **GlossyMetal** | MIS: GGX specular + cosine diffuse | Metallic Fresnel: $F_0 = K_s$ (per-channel Schlick). Lobe weights from `max(Ks)` vs `max(Kd)`, clamped [0.05, 0.95]. VNDF half-vector sampling. |
| **GlossyDielectric** | MIS: GGX specular + cosine diffuse | Dielectric Fresnel: $F_0 = ((\eta-1)/(\eta+1))^2$. Specular weight = `max(Ks) × F0`. |
| **Clearcoat** | MIS: GGX coat + cosine base | Layered: dielectric GGX coat over diffuse base. Base energy attenuated by coat Fresnel: $(1 - w_c \cdot F_r) \cdot K_d / \pi$. `coat_α = max(roughness², 0.001)`. |
| **Fabric** | Cosine-weighted | Diffuse + sheen lobe: $\text{sheen}_w \cdot ((1-\text{tint}) + \text{tint} \cdot K_d) \cdot (1-\cos\theta)^5 / \pi$. Sheen is **not** importance-sampled (evaluated in `bsdf_evaluate`, contributes via NEE). |
| **Translucent** | Same as Glass | Glass BSDF + interior participating medium (Beer–Lambert). |
| **DiffuseTransmission** | Stochastic: reflect $P = p_r$, transmit $P = 1-p_r$ | Two-sided Lambert: $K_d/\pi$ for reflection, $T_f/\pi$ for transmission. $p_r = \text{max}(K_d) / (\text{max}(K_d) + \text{max}(T_f))$, clamped [0.1, 0.9]. Both lobes cosine-sampled. |

### 8.3 Microfacet Shared Helpers

**Source:** `src/renderer/material/bsdf_shared.h`

| Function | Description |
|---|---|
| `ggx_D(h, α)` | GGX / Trowbridge-Reitz NDF (isotropic) |
| `ggx_G(wo, wi, α)` | Smith height-correlated masking-shadowing: $G = 4|\omega_o \cdot n||\omega_i \cdot n| / (\Lambda_o \cdot \Lambda_i)$ |
| `ggx_sample_halfvector(wo, α, ξ1, ξ2)` | VNDF sampling (Heitz 2018) |
| `fresnel_dielectric(cos_i, η_i, η_t)` | Exact dielectric Fresnel reflectance |
| `fresnel_schlick(cos, F0)` | Scalar Schlick approximation |
| `fresnel_schlick3(cos, F0)` | Per-channel Schlick (metallic Fresnel) |
| `bsdf_roughness_to_alpha(r)` | `max(r², 0.001)` — floor prevents numerical singularity |
| `mis_weight_2(pdf_a, pdf_b)` | Power heuristic: $pdf_a^2 / (pdf_a^2 + pdf_b^2)$ |

### 8.4 Chromatic Dispersion (Cauchy Equation)

Materials with `cauchy_B ≠ 0` support per-wavelength refractive index:

$$n(\lambda) = A + \frac{B}{\lambda^2}$$

Used exclusively in the caustic tracer's Glass/Translucent bounces via
`dev_glass_sample_dispersive()`. Three representative wavelengths (R =
630 nm, G = 532 nm, B = 465 nm) produce physically motivated spectral
separation.

**Material fields:** `cauchy_A` (base IOR), `cauchy_B` (dispersion
coefficient). When `cauchy_B == 0`, standard non-dispersive IOR from
the `ior` field is used.

---

## 9. Scene Loading

### 9.1 Scene Structure

**Source:** `src/scene/scene.h`

```cpp
struct Scene {
    vector<Triangle>           triangles;
    vector<Material>           materials;
    vector<Texture>            textures;
    vector<HomogeneousMedium>  media;

    // Instancing
    vector<MeshDescriptor>     meshes;       // {tri_offset, tri_count}
    vector<InstanceDescriptor> instances;     // {mesh_id, transform[12]}

    // Emissive distribution
    vector<uint32_t>  emissive_tri_indices;
    AliasTable        emissive_alias_table;
    float             total_emissive_power;

    AABB  scene_bounds;
};
```

Meshes are stored as flat triangle arrays with per-vertex position,
normal, tangent, UV, and per-triangle material ID.

### 9.2 OBJ/MTL Loader

**Source:** `src/scene/obj_loader.h` / `.cpp`

Standard Wavefront OBJ + MTL parser. Maps MTL properties to internal
`Material` struct. Supports diffuse/specular/emission/bump/normal
textures.

### 9.3 PBRT v4 Loader

**Source:** `src/scene/pbrt/pbrt_loader.h` / `.cpp`

Native PBRT v4 text format parser supporting:

| Feature | Detail |
|---|---|
| **Shapes** | `plymesh`, `trianglemesh`, `sphere`, `disk`, `bilinearmesh` |
| **Materials** | 13+ PBRT v4 types with explicit fallback mapping |
| **Instancing** | `ObjectBegin` / `ObjectInstance` → IAS |
| **Includes** | `Include` / `Import` directives |
| **Coordinate systems** | `CoordinateSystem` / `CoordSysTransform` |
| **Attributes** | Hierarchical attribute inheritance |
| **Lights** | `AreaLightSource`, point, spot, distant, infinite portal proxy |
| **Textures** | Texture graph resolution (subset) |
| **Camera** | Extraction from PBRT camera definition |

**Material mapping** (`src/scene/pbrt/pbrt_material_mapper.h`): PBRT
material types (diffuse, coateddiffuse, conductor, dielectric,
diffusetransmission, etc.) are mapped to the 9 internal GPU material
types. Extended `pb_*` fields on `Material` carry PBRT-specific
properties (conductor complex IOR, subsurface, clearcoat layering,
fabric sheen, Cauchy dispersion).

### 9.4 Texture System

CUDA texture objects with hardware sampling. Texture slots per material:

| Slot | Usage |
|---|---|
| `diffuse_tex` | Albedo / Kd |
| `specular_tex` | Specular / Ks |
| `emission_tex` | Emissive Le |
| `normal_tex` | Tangent-space normal map |
| `bump_tex` | Height-field bump (finite differences → perturbed normal) |
| `alpha_tex` | Stochastic opacity (anyhit) |
| `displacement_tex` | (Reserved) |

---

## 10. OptiX Program Structure

**Source:** `src/renderer/accel/optix_programs.cu`

Single `.cu` file that `#include`s all device code:

```cpp
#include "optix_utils.cuh"    // trace_radiance, trace_shadow
#include "optix_nee.cuh"      // NEE + material accessors
#include "path_tracer.cuh"    // trace_path + __raygen__render
#include "caustic_tracer.cuh" // trace_caustic + __raygen__caustic
```

### 10.1 Programs

| Program | Purpose |
|---|---|
| `__raygen__render` | Camera path tracing (§5) — progressive accumulation |
| `__raygen__caustic` | Caustic light tracing (§6) — 1 thread per photon |
| `__raygen__test_normals` | Diagnostic: primary ray → abs(shading normal) as RGB |
| `__raygen__test_nee` | Diagnostic: primary ray → 16-sample NEE direct lighting |
| `__closesthit__radiance` | Interpolate position/normal/tangent/UV; instance transform; pack 18 payload values |
| `__closesthit__shadow` | Set `occluded = 1` |
| `__miss__radiance` | Set `hit = 0` (background) |
| `__miss__shadow` | Set `occluded = 0` (visible) |
| `__anyhit__radiance` | Stochastic alpha test: TEA4 hash → if `ξ >= opacity`, ignore hit |
| `__anyhit__shadow` | Same stochastic alpha test |

### 10.2 Payload Layout

18 `unsigned int` values (p0–p17):

| Payload | Contents |
|---|---|
| p0–p2 | Hit position (float3) |
| p3–p5 | Shading normal (float3) |
| p6 | Distance t |
| p7 | Material ID |
| p8 | Triangle ID |
| p9 | Hit flag (0 = miss) |
| p10–p12 | Geometric normal (float3) |
| p13–p14 | UV coordinates (float2) |
| p15–p17 | Tangent vector (float3) |

---

## 11. Acceleration & SBT

**Source:** `src/renderer/accel/accel_builder.h` / `.cpp`

### 11.1 Acceleration Structure

| Mode | Configuration |
|---|---|
| **Non-instanced** | Single compacted GAS, all triangles |
| **Instanced** | Per-mesh GAS + single IAS with instance transforms |

`has_instances` flag in LaunchParams signals closest-hit to apply
object→world transforms.

### 11.2 Shader Binding Table

4 separate SBTs, one per raygen program:

| SBT | Raygen program |
|---|---|
| `sbt_default_` | `test_normals` |
| `sbt_nee_` | `test_nee` |
| `sbt_render_` | `render` |
| `sbt_caustic_` | `caustic` |

Each SBT shares the same 2 miss records (radiance + shadow) and 2
hitgroup records (radiance with anyhit, shadow with anyhit).

### 11.3 Pipeline Configuration

| Parameter | Value |
|---|---|
| `OPTIX_MAX_TRACE_DEPTH` | 2 (radiance + shadow) |
| `OPTIX_STACK_SIZE` | 16384 bytes |
| Module | Single PTX from `optix_programs.cu` |
| Program groups | 8 total (4 raygen + 2 miss + 2 hitgroup) |

### 11.4 Launch Methods

| Method | Grid | Use |
|---|---|---|
| `launch_progressive(w, h, lp)` | (W, H, 1) | Camera path tracing |
| `launch_caustic(n, lp)` | (N, 1, 1) | Caustic photon tracing |
| `launch_test_normals(w, h, lp)` | (W, H, 1) | Normal diagnostic |
| `launch_test_nee(w, h, lp)` | (W, H, 1) | NEE diagnostic |

---

## 12. Camera Model

**Source:** `src/core/camera.h`

### 12.1 Thin-Lens Model

| Parameter | Default | Description |
|---|---|---|
| `dof_enabled` | false | Enable depth of field |
| `dof_f_number` | 8.0 | f-stop (controls bokeh size) |
| `dof_focus_dist` | 0.1 | Focus distance (metres) |
| `sensor_height` | 0.024 | Sensor height (metres, 24mm = full frame) |
| `dof_focus_range` | 0.05 | Focus jitter range |

Derived: `focal_length = sensor_height / (2·tan(fov/2))`,
`lens_radius = focal_length / (2·f_number)`.

### 12.2 Ray Generation

1. Sub-pixel jitter on pixel coordinates.
2. Compute focus target on the image plane.
3. If thin lens (`lens_radius > 0`):
   - Sample disk: `lens_offset = (u·dx + v·dy) · lens_radius`
     (concentric disk mapping).
   - `origin = cam_pos + lens_offset`.
   - `direction = normalize(focus_target − origin)`.
4. Otherwise: pinhole from `cam_pos`.

### 12.3 GPU vs CPU

The CPU `Camera::generate_ray()` implements the full thin-lens model.
The GPU raygen uses a simplified pinhole
`dir = normalize(cam_w + cam_u·(2u−1) + cam_v·(2v−1))` with DOF
applied when `cam_lens_radius > 0` in LaunchParams.

---

## 13. Configuration

**Source:** `src/core/config.h`

### 13.1 Compile-Time Constants

| Constant | Value | Purpose |
|---|---|---|
| `OPTIX_MAX_TRACE_DEPTH` | 2 | Radiance + shadow |
| `OPTIX_STACK_SIZE` | 16384 | OptiX stack (bytes) |
| `MAX_AOV_BOUNCES` | 4 | Debug AOV bounce limit |

### 13.2 Rendering Defaults

| Constant | Value | Purpose |
|---|---|---|
| `DEFAULT_IMAGE_WIDTH` / `HEIGHT` | 1440 × 1440 | Resolution |
| `DEFAULT_SPP` | 256 | Samples per pixel |
| `DEFAULT_MAX_BOUNCES_CAMERA` | 8 | Camera max bounces |
| `DEFAULT_MIN_BOUNCES_RR` | 3 | Bounces before Russian roulette |
| `DEFAULT_RR_THRESHOLD` | 0.95 | Max survival probability |
| `DEFAULT_MAX_SPECULAR_CHAIN` | 8 | Max specular chain |
| `PREVIEW_MAX_BOUNCES` | 2 | Preview mode |
| `IDLE_TIMEOUT_SEC` | 1.0 | Idle timeout |
| `DEFAULT_EXPOSURE` | 1.0 | Exposure multiplier |
| `DEFAULT_LIGHT_SCALE` | 1.0 | Light intensity scale |

### 13.3 Clamping Defaults (All Disabled by Default)

| Constant | Value |
|---|---|
| `DEFAULT_CLAMPING_ENABLED` | false |
| `DEFAULT_MAX_BOUNCE_CONTRIBUTION` | 10⁴ |
| `DEFAULT_MAX_PATH_THROUGHPUT` | 10⁴ |
| `DEFAULT_MAX_NEE_CONTRIBUTION` | 10⁴ |
| `DEFAULT_MAX_SAMPLE_LUMINANCE` | 10000 |

### 13.4 Caustic Defaults

| Constant | Value |
|---|---|
| `DEFAULT_CAUSTIC_ENABLED` | true |
| `DEFAULT_CAUSTIC_PHOTONS_PER_FRAME` | 262144 (256K) |
| `DEFAULT_CAUSTIC_MAX_SPLAT_LUMINANCE` | 100 |

### 13.5 Post-Processing Defaults

| Constant | Value |
|---|---|
| `DEFAULT_FIREFLY_FILTER_ENABLED` | true |
| `FIREFLY_FILTER_RADIUS` | 1 |
| `FIREFLY_FILTER_THRESHOLD` | 4.0 |
| `DEFAULT_DENOISER_ENABLED` | false |
| `DEFAULT_BLOOM_ENABLED` | false |
| `DEFAULT_BLOOM_INTENSITY` | 0.5 |
| `DEFAULT_BLOOM_RADIUS_H` / `V` | 15.0 |

### 13.6 DOF Defaults

| Constant | Value |
|---|---|
| `DEFAULT_DOF_ENABLED` | false |
| `DEFAULT_DOF_F_NUMBER` | 8.0 |
| `DEFAULT_DOF_FOCUS_DIST` | 0.1 |
| `DEFAULT_SENSOR_HEIGHT` | 0.024 |

### 13.7 Scene Presets

12 built-in presets, selectable via hotkeys:

| Key | Preset |
|---|---|
| **1** | Cornell Box |
| **2** | Veach Bidir |
| **3** | Conference |
| **4** | Bathroom |
| **5** | Bedroom |
| **6** | Living Room |
| **7** | Spaceship |
| **8** | Coffee |
| **9** | Staircase |
| **0** | Staircase 2 |
| **Shift+1** | Spheres |
| **Shift+2** | Water Caustic |

---

## 14. Debug & Diagnostics

### 14.1 Render Modes

Selectable via `RenderMode` enum:

| Mode | Description |
|---|---|
| Combined | Full path tracing (direct + indirect + caustics) |
| DirectOnly | NEE contribution only |
| IndirectOnly | BSDF-sampled indirect only |
| NormalsOnly | Shading normal visualisation |
| NEEOnly | 16-sample NEE diagnostic |

### 14.2 Debug Overlays

| Key | Overlay |
|---|---|
| **S** | Stats: FPS, SPP, resolution, throughput |
| **N** | Noise heatmap — per-pixel variance |
| **C** | Convergence display |
| **F1** | Caustic buffer isolation |

### 14.3 Test Raygen Programs

- `__raygen__test_normals` — primary ray → absolute shading normal as
  false colour.
- `__raygen__test_nee` — primary ray → 16-sample NEE-only radiance
  (useful for verifying emitter sampling and shadow rays).

### 14.4 Viewer Hotkeys

**Source:** `src/renderer/app/viewer.cpp`

#### Camera

| Key | Action |
|---|---|
| **W / A / S / D** | Move forward / left / backward / right |
| **Space / Ctrl** | Move up / down |
| **Q / E** | Roll CCW / CW |
| **Shift** | 3× movement speed |
| **Mouse** | Look (when captured) |
| **Left Alt** | Toggle mouse capture |

#### Rendering

| Key | Action |
|---|---|
| **R** | Save snapshot (PNG + EXR) |
| **+ / −** | Light brightness up / down |
| **1–9, 0** | Scene preset 0–9 |
| **Shift+1–9** | Extended presets 10–18 |

#### Caustics & Tone Mapping

| Key | Action |
|---|---|
| **F1** | Caustic-only isolation |
| **F2** | Toggle ACES tone mapping |
| **F3** | Suppress camera-side caustics (de-duplication) |

#### Depth of Field

| Key | Action |
|---|---|
| **O** | Toggle depth of field on/off |
| **F** | Auto-focus (set focus distance to look-at point) |
| **[** | Widen aperture (lower f-number, more blur) |
| **]** | Narrow aperture (higher f-number, less blur) |

#### Debug

| Key | Action |
|---|---|
| **S** | Toggle stats overlay |
| **N** | Toggle noise heatmap |
| **C** | Toggle convergence display |
| **Esc** | Quit |

---

## 15. Analysis Pipeline

Three executables form an automated profiling and diagnostics pipeline:

```
ppt_analyze → photon_tracer → ppt_diagnose
```

### 15.1 ppt_analyze

**Source:** `src/analyze/`

Offline scene analysis — no GPU required.

- **Input:** `.pbrt` or `.obj` scene file.
- **Process:** Load scene → classify geometry complexity
  (Simple/Moderate/Complex/Dense), emitter distribution
  (SingleDominant/HighVariance/Uniform), delta surface presence,
  emitter-to-delta coupling, open geometry detection.
- **Output:** JSON `SceneProfile` to stdout.

### 15.2 ppt_diagnose

**Source:** `src/diagnose/`

Post-render image quality analysis — no GPU required.

- **Input:** Rendered `.exr` or `.png` image, optional render log.
- **Components:**
  - `ImageOracle` — no-reference analysis, reference comparison, A/B
    differential.
  - `NoiseAnalyzer` — noise level estimation.
  - `ArtifactDetector` — firefly/artifact detection.
  - `ConvergenceAnalyzer` — convergence rate tracking.
  - `VarianceTracker` — per-pixel variance statistics.
  - `BottleneckAnalyzer` — actionable `BottleneckReport`.
- **Output:** JSON quality report.

### 15.3 Runtime Diagnostics

**Source:** `src/diagnose/diagnostics.h`

`RenderDiagnostics` receives per-frame `FrameMetrics` and per-pixel
variance data from GPU buffers, feeds `VarianceTracker` and
`ConvergenceAnalyzer`, and generates `BottleneckReport` with actionable
recommendations.

---

## 16. Test Suite

Tests use GoogleTest, organised into four categories:

| Category | Path | Description |
|---|---|---|
| **Unit** | `tests/unit/` | BSDF, NEE, camera, scene loading, alias table |
| **Integration** | `tests/integration/` | Per-stage validation (core → postfx) |
| **Convergence** | `tests/convergence/` | Multi-SPP regression (direct, full, caustic) |
| **Diagnostics** | `tests/diagnostics/` | Variance tracking, bottleneck detection, image quality |

### Build & Run

```bat
run.bat test          :: Fast tests (skip integration / speed)
run.bat test-all      :: Full suite (may take hours)
```

### Key Test Areas

| Area | Tests |
|---|---|
| BSDF | Fresnel, GGX VNDF reciprocity, energy conservation (white furnace), lobe weights |
| NEE | PDF area→solid-angle, MIS weights, emitter sampling correctness |
| Materials | Per-type sample/evaluate/pdf consistency |
| Convergence | Direct lighting, full path, caustic at multiple SPP checkpoints |

---

## 17. Key Data Structures

### 17.1 LaunchParams

**Source:** `src/renderer/accel/launch_params.h`

Flat GPU-shared struct (`extern "C" __constant__`). Sections:

| Section | Key Fields |
|---|---|
| **Output** | `color_r/g/b` (SoA float accum), `sample_counts`, `srgb_buffer` (RGBA8), `albedo_buffer`, `normal_buffer` |
| **Camera** | `cam_pos`, `cam_u/v/w`, `cam_lens_radius`, `cam_focus_dist` |
| **Rendering** | `samples_per_pixel`, `max_bounces`, `min_bounces_rr`, `rr_threshold`, `frame_number`, `render_mode`, `exposure` |
| **Clamping** | `clamping_enabled`, `max_bounce_contribution`, `max_path_throughput`, `max_nee_contribution`, `max_sample_luminance` |
| **Geometry** | `vertices`, `normals`, `tangents`, `texcoords`, `material_ids` (SoA, per-triangle) |
| **Materials** | `Kd/Ks/Le/Tf` (float×3 per mat), `roughness/ior/opacity` (scalar), `mat_type`, texture IDs, `cauchy_A/B`, coat/sheen/thin |
| **Textures** | `textures` (array of `cudaTextureObject_t`) |
| **Emitters** | `emissive_tri_indices`, `emissive_cdf`, `emissive_local_idx` (inverse index), `total_emissive_power` |
| **Caustics** | `caustic_r/g/b`, `caustic_num_photons`, `caustic_frame_number`, `caustic_max_splat_luminance`, `suppress_camera_caustics`, `delta_tri_indices/cdf` |
| **Adaptive** | `lum_sum`, `lum_sum2`, `active_mask` |
| **Pre-pass** | `prepass_nee_attempts/hits/zero_paths/bounce_sum/total_paths`, `prepass_active` |
| **OptiX** | `traversable`, `has_instances` |

### 17.2 Material

**Source:** `src/scene/material.h`

```
Kd, Ks, Le, Tf                // RGB colour properties
roughness, ior, opacity        // Scalar properties
type: MaterialType             // Enum (0–9)
thin: bool                     // Thin dielectric flag
cauchy_A, cauchy_B             // Cauchy dispersion
clearcoat_weight, clearcoat_roughness
sheen, sheen_tint
medium_id                      // Interior medium reference
texture IDs × 7               // Per-slot texture binding
pb_*                           // Extended PBRT properties
```

### 17.3 IORStack

```cpp
struct IORStack {
    float iors[4];         // Up to 4 nested dielectrics
    int   depth = 0;       // Empty = air (IOR 1.0)
    float top();           // Current IOR (1.0 if empty)
    void  push(float ior);
    void  pop();
};
```

The GPU path tracer infers entering/exiting from local-frame `wo.z`
sign rather than maintaining an explicit stack during tracing.

### 17.4 Scene

See [§9.1](#91-scene-structure).

---

## 18. Source Layout

```
src/
  core/
    config.h                     Compile-time + runtime defaults
    types.h                      float3, Ray, HitRecord, ONB, ShadingFrame
    color.h                      Color3 (linear RGB), luminance, tone mapping
    camera.h                     Thin-lens perspective camera
    random.h                     PCG RNG (CPU + GPU)
    alias_table.h                Vose O(1) alias sampling (host build)
    ior_stack.h                  Nested dielectric IOR tracking

  scene/
    scene.h                      Triangle mesh, instances, emitter distribution
    material.h                   Material struct (10 types + extended PBRT fields)
    medium.h                     Homogeneous participating medium
    obj_loader.h / .cpp          Wavefront OBJ + MTL parser
    texture.h                    CUDA texture wrapper
    pbrt/
      pbrt_loader.h / .cpp       PBRT v4 scene parser
      pbrt_material_mapper.h     PBRT → internal material mapping
      ply_reader.h               PLY mesh loader

  renderer/
    main.cpp                     Entry point, CLI, execution modes

    accel/
      accel_builder.h / .cpp     OptiX context, GAS/IAS, SBT, pipeline
      launch_params.h            GPU launch parameter struct
      optix_programs.cu          Raygen, closesthit, miss, anyhit programs

    integrator/
      path_tracer.cuh            Camera bounce loop + BSDF sampling
      optix_nee.cuh              Next-event estimation + MIS
      caustic_tracer.cuh         Light-side delta tracing + camera splat
      russian_roulette.h         Probabilistic path termination
      sample_clamping.h          Per-bounce / path / sample clamp

    material/
      bsdf.h                     10-type BSDF library (CPU + GPU)
      bsdf_shared.h              GGX, Fresnel, VNDF, MIS helpers
      specular.h                 Delta surface sampling

    postfx/
      postfx_pipeline.h / .cpp   GPU post-processing orchestration
      postfx_params.h            Post-processing configuration
      firefly_filter.h / .cu     Median + MAD outlier filter
      bloom.h / .cu              Mip-chain Gaussian bloom
      tonemap.h / .cu            ACES filmic + sRGB gamma
      optix_denoiser.h / .cpp    OptiX AI denoiser wrapper

    app/
      viewer.h / .cpp            Interactive GLFW window + camera controls
      render_session.h / .cpp    Progressive render loop + buffer management
      render_config.h            Runtime configuration (JSON-serialisable)

  analyze/                       Pre-render scene analysis (ppt_analyze)
  diagnose/                      Post-render quality diagnostics (ppt_diagnose)

tests/
  unit/                          BSDF, NEE, camera, scene loading
  integration/                   Per-stage validation
  convergence/                   Multi-SPP regression
  diagnostics/                   Variance tracking, bottleneck detection

scenes/                          OBJ / MTL / PBRT scene files
doc/                             Architecture, Materials Handbook
tools/                           Build scripts, utilities
```

---

## 19. GPU Performance Notes

### 19.1 SoA Memory Layout

All per-pixel buffers use Structure of Arrays: separate `float*` for R,
G, and B channels (not interleaved float3/float4). This gives coalesced
global memory access when adjacent threads process adjacent pixels.

### 19.2 DeviceBuffer Allocation Amortisation

`DeviceBuffer::ensure_alloc(size)` — only reallocates when the
requested size exceeds the current allocation. Avoids per-frame
CUDA malloc/free overhead during progressive rendering.

### 19.3 Emissive Inverse-Index Table

`emissive_local_idx[tri_id]` maps any triangle index to its local
emitter index (−1 for non-emissive). This allows O(1) emission PDF
lookup during emission MIS weight computation, avoiding a search
through `emissive_tri_indices[]`.

### 19.4 Stochastic Alpha Testing

The anyhit shader uses TEA4 hashing (not RNG state) for stochastic
opacity decisions. This avoids polluting the path-tracing RNG state and
provides deterministic results for a given ray.

### 19.5 Atomic Caustic Splatting

Caustic photons use `atomicAdd` on per-channel float buffers. The SoA
layout (`caustic_r`, `caustic_g`, `caustic_b` separate) reduces
atomic contention compared to interleaved float3 splatting.

### 19.6 Pre-pass

Low-resolution path trace (`full_resolution / 4`) with atomic counters
for NEE hit rate, zero-path fraction, and average bounce depth. Returns
`PrePassMetrics` for adaptive parameter tuning before the full render
begins.
