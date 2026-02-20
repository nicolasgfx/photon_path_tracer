# Architecture -- Spectral Photon + Path Tracing Renderer

This document describes the complete rendering pipeline, its design
rationale, mathematical foundations, strengths, weaknesses, and
intended use.

---

## 1. Overview

The renderer combines **photon mapping** with **Monte Carlo path
tracing** in a fully spectral framework. All light transport is
computed over 32 discrete wavelength bins spanning 380--780 nm. The
entire pipeline runs on the GPU via **NVIDIA OptiX 9.x** for ray
tracing and **CUDA** for auxiliary kernels. There is no CPU fallback.

A key feature is the **photon directional bin cache**: a per-pixel
Fibonacci-sphere discretisation of the photon flux distribution that
guides both BSDF bounce directions and next-event estimation (NEE)
light selection, significantly reducing variance in indirect
illumination.

### Intended Use

This is a **private research renderer** focused on physical
correctness and mathematical clarity. It is not designed for
production workloads or real-time applications. The emphasis is on:

- Explicit, auditable estimators.
- Spectral correctness over speed.
- Debugging and visualisation of every intermediate quantity.

### Requirements

| Component               | Minimum Version       |
|-------------------------|-----------------------|
| NVIDIA GPU              | Turing (sm_75) or newer |
| CUDA Toolkit            | 12.x                 |
| NVIDIA OptiX SDK        | 7.x / 9.x            |
| CMake                   | 3.24                  |
| C++ Standard            | C++17                 |
| OS                      | Windows 10+ (MSVC) or Linux (GCC/Clang) |

The `OptiX_INSTALL_DIR` environment variable (or CMake cache
variable) **must** point to the OptiX SDK. The build will fail with a
fatal error if it is not set.

---

## 2. Pipeline Stages

The renderer executes the following stages in order:

```
Scene Load (OBJ + MTL)
        |
        v
  Build BVH (CPU)
        |
        v
  Build Emissive Distribution (CPU)
        |
        v
  OptiX init / build_accel / upload_scene_data / upload_emitter_data
        |
        v
  GPU Photon Trace  (__raygen__photon_trace)
        |
        v
  Download photons -> Build Hash Grid (CPU) -> Upload grid + photons
        |
        v
  Interactive Debug Viewer (first-hit OptiX, GLFW window)
        |
        v  (R key)
  NEE Debug PNG (single-frame shadow ray preview)
        |
        v
  Populate Photon Directional Bin Cache (populate_bins_mode = 1)
        |
        v
  Full Path Tracing  (__raygen__render, is_final_render=1)
    * Two-pass: bin population pass, then SPP render passes
    * Guided NEE (B2) + Guided BSDF bounce (B1)
    * Stratified sub-pixel sampling (B3)
    * Cached spectral density estimation (A1)
        |
        v
  PNG Output (render, NEE direct, photon indirect, combined)
```

### 2.1 Scene Loading

Wavefront OBJ with MTL materials. Materials are mapped to the
internal spectral representation using `rgb_to_spectrum_reflectance()`
for diffuse/specular albedos and `blackbody_spectrum()` for emissive
surfaces whose emission is defined by `Ke` in the MTL file.

**Scene normalisation:** All non-reference scenes are scaled and
translated to fit inside the Cornell Box reference frame
$([-0.5, 0.5]^3)$, so camera defaults, gather radii, and light
placement transfer across scenes.

Supported material types:

| Type        | MTL Cue                | Internal Enum      |
|-------------|------------------------|--------------------|
| Lambertian  | default                | `Lambertian`       |
| Mirror      | `illum 3`, Ks > 0.99  | `Mirror`           |
| Glass       | `illum 4`, Ni > 1     | `Glass`            |
| Glossy      | `illum 2`, Ns > 0     | `Glossy`           |
| Emissive    | `Ke` present           | `Emissive`         |

### 2.2 Acceleration Structure

A single bottom-level **Geometry Acceleration Structure (GAS)** is
built from the triangle soup using `optixAccelBuild`. No Instance
Acceleration Structure (IAS) is used -- all geometry lives in a
single GAS with `maxTraversableGraphDepth = 1`.

Vertex positions and per-vertex normals are stored in SoA layout on
the device. Triangle indices are implicit (primitive index * 3).

### 2.3 Emitter Data

Before photon tracing, a cumulative distribution function (CDF) over
emissive triangles is built on the CPU and uploaded to the device.
Each triangle's weight is:

$$
w_t = A_t \cdot \bar{L}_{e,t}
$$

where $A_t$ is the triangle area and $\bar{L}_{e,t}$ is the mean
spectral emission power across all wavelength bins. The CDF allows
$O(\log n)$ importance sampling of emissive geometry on the device.

---

## 3. GPU Photon Trace

Photon tracing is implemented as the OptiX raygen program
`__raygen__photon_trace`. It is launched as a **1-D grid** of
`num_photons` threads (default: 1,000,000).

### 3.1 Emission Sampling

Each thread:

1. Samples an emissive triangle from the CDF.
2. Samples a point on that triangle via uniform barycentric
   coordinates:

$$
\alpha = 1 - \sqrt{u}, \quad
\beta  = v\sqrt{u},    \quad
\gamma = 1 - \alpha - \beta
$$

3. Samples a wavelength bin proportional to the triangle's spectral
   emission:

$$
p(\lambda_i | x) = \frac{L_e(x,\lambda_i)}{\sum_j L_e(x,\lambda_j)}
$$

4. Samples a cosine-weighted hemisphere direction above the emitting
   surface.

### 3.2 Photon Flux

The initial flux carried by each photon is:

$$
\Phi = \frac{L_e(x,\omega,\lambda)\cos\theta}
            {p(t) \cdot p(x|t) \cdot p(\omega|x) \cdot p(\lambda|x)}
$$

where $p(t)$ is the triangle selection PDF (from the CDF), $p(x|t) =
1/A_t$, $p(\omega|x) = \cos\theta / \pi$, and $p(\lambda|x)$ is the
spectral PDF above.

### 3.3 Tracing and Storage

Photons are traced through the scene via `optixTrace`. At each
intersection:

- **Emissive surfaces**: tracing terminates.
- **Specular surfaces** (mirror/glass): photon continues via ideal
  reflection; no storage.
- **Diffuse surfaces**: the photon is stored (at bounce > 0, since
  direct illumination is handled by NEE), then scattered via
  cosine-weighted hemisphere sampling. The flux is attenuated by the
  surface albedo $K_d(\lambda)$.

Storage uses `atomicAdd` on a global counter to append photons into
pre-allocated SoA output buffers (position, direction, wavelength
bin, flux).

**Russian roulette** is applied after `MIN_BOUNCES_RR` (3) bounces:

$$
p_{rr} = \min(0.95,\; 0.5)
$$

If the photon survives, its flux is divided by $p_{rr}$.

### 3.4 Hash Grid Construction

After the GPU photon trace completes, the photon data is downloaded
to the CPU, a **hashed uniform grid** is built, and the result is
uploaded back to the device.

Cell coordinate:
```
cellCoord = floor(position / cellSize)
```

The cell size is `gather_radius * HASHGRID_CELL_FACTOR` (default
factor 2.0). Neighbour lookup scans $3 \times 3 \times 3$ cells. The
grid stores sorted indices plus `cellStart` / `cellEnd` arrays
indexed by hash.

Hash function:
$$
h(c_x, c_y, c_z) = (c_x \cdot 73856093 \oplus c_y \cdot 19349663 \oplus c_z \cdot 83492791) \bmod T
$$

where $T$ is the hash table size.

---

## 4. Photon Directional Bin Cache

The photon directional bin cache is the central optimisation of the
renderer. It pre-computes per-pixel photon flux distributions on the
unit sphere and uses them for three purposes:

1. **Guided BSDF bounce (B1):** First-bounce directions are sampled
   proportional to cached photon flux.
2. **Guided NEE (B2):** Shadow rays are steered toward lights that
   contribute the most flux at each pixel.
3. **Cached density estimation (A1):** The first-sample spectral
   density is cached and reused across subsequent SPP frames.

### 4.1 Fibonacci Sphere Binning

The unit sphere $S^2$ is discretised into $N = 32$ quasi-uniform
directions using the **Fibonacci sphere** construction:

$$
\theta_k = \arccos\!\left(1 - \frac{2(k + 0.5)}{N}\right), \quad
\phi_k = \varphi_g \cdot k
$$

where $\varphi_g = \pi(3 - \sqrt{5})$ is the golden angle. This
produces a nearly uniform point distribution on $S^2$ without
clustering at the poles.

Each bin direction $\mathbf{d}_k$ stores a `PhotonBin` (24 bytes):

| Field   | Type  | Description                                   |
|---------|-------|-----------------------------------------------|
| `flux`  | float | Total Epanechnikov-weighted scalar flux        |
| `dir_x` | float | Flux-weighted centroid direction $x$           |
| `dir_y` | float | Flux-weighted centroid direction $y$           |
| `dir_z` | float | Flux-weighted centroid direction $z$           |
| `weight`| float | Total Epanechnikov weight (for normalisation)  |
| `count` | int   | Number of photons accumulated in this bin      |

**Memory:** 24 bytes × 32 bins × 786,432 pixels (1024×768) = **604 MB**.

### 4.2 Bin Population (`dev_populate_bins_for_pixel`)

For each pixel, the population pass:

1. Traces the center ray, following specular bounces to the first
   diffuse hit.
2. Gathers all photons from the hash grid within the gather radius,
   applying the Epanechnikov kernel and plane-distance filter.
3. For each accepted photon, finds the nearest Fibonacci bin via
   brute-force dot product scan ($O(N)$ for $N \le 32$) and
   accumulates:
   - `flux += photon_flux * w` (Epanechnikov weight)
   - `dir += wi_world * photon_flux * w` (flux-weighted centroid)
   - `weight += w`, `count += 1`
4. Normalises centroid directions to unit length (falls back to
   Fibonacci center if degenerate).

The population is launched as a separate OptiX pass
(`populate_bins_mode = 1`) before the render loop.

### 4.3 Guided BSDF Bounce (B1)

At bounce 0 of `full_path_trace()`, the continuation direction is
sampled from the bin flux CDF instead of the cosine hemisphere:

1. **Build hemisphere CDF:** For each bin $k$ with $\cos\theta_{n,k}
   > -\epsilon$, compute:

$$
\text{cdf}[k] = \sum_{j \le k} \Phi_j \cdot \max(\cos\theta_{n,j}, 0)
$$

   where $\Phi_j$ is the bin's flux and $\theta_{n,j}$ is the angle
   between the bin centroid and the surface normal.

2. **Select bin:** Sample $\xi \sim U(0, \text{total})$ and find the
   bin via linear scan of the CDF.

3. **Jitter within bin:** Sample a direction uniformly within a cone
   of half-angle $\alpha = \arccos(1 - 2/N)$ centered on the
   selected bin's normalised centroid.

4. **Hemisphere clamp:** If the sampled direction falls below the
   tangent plane, reflect it across the normal.

5. **Fallback:** If no bins have flux in the positive hemisphere,
   fall back to cosine-weighted hemisphere sampling.

### 4.4 Guided NEE (B2)

Instead of sampling emissive triangles from the power-based CDF
alone, the guided NEE re-weights each triangle using the photon flux
distribution:

For each emissive triangle $i$:

$$
w_i^{\text{guided}} = p_i^{\text{orig}} \cdot \left(1 + \alpha \cdot
\frac{\Phi_{\text{bin}(i)}}{\Phi_{\text{total}}}\right)
$$

where:
- $p_i^{\text{orig}}$ is the original power-proportional CDF weight
- $\alpha = 5.0$ (`NEE_GUIDED_ALPHA`) controls the flux-boost
  strength
- $\Phi_{\text{bin}(i)}$ is the flux of the nearest bin to the
  direction from the hitpoint to the triangle centroid
- $\Phi_{\text{total}} = \sum_k \Phi_k$ normalises the bin flux

A temporary CDF is built on the GPU stack (up to 128 emissive
triangles, `NEE_GUIDED_MAX_EMISSIVE`), sampled, and the modified PDF
is used in the estimator for correct importance-sampling weighting.

**Fallback conditions** (reverts to standard `dev_nee_direct`):
- Bins have zero total flux
- Scene has more than 128 emissive triangles

The remaining NEE logic (bounce-dependent sample count, shadow ray,
BSDF evaluation, solid-angle PDF conversion) is identical to the
standard path.

### 4.5 Cached Spectral Density (A1)

On the first SPP frame (`frame_number == 0`), the photon density
gather result at bounce 0 is written to a per-pixel spectral cache
(`photon_density_cache`, `float[W*H*NUM_LAMBDA]`). On all subsequent
frames, the cache is read directly, skipping the full hash-grid
gather. This amortises the $O(k)$ photon gather cost across SPP.

---

## 5. Debug Viewer (Interactive)

The debug viewer is an interactive GLFW window rendering at 1 spp per
frame via OptiX. It launches the same `__raygen__render` program with
`is_final_render = 0`, which calls `debug_first_hit()`.

### 5.1 First-Hit Rendering

For each pixel:

1. Trace a primary ray via OptiX.
2. If the hit is emissive, return $L_e$.
3. If the hit is specular, follow up to 4 specular bounces.
4. At the first diffuse hit, perform **next-event estimation** (NEE):
   sample one emissive triangle, cast a shadow ray, compute direct
   illumination.

This gives an interactive preview with direct lighting but no global
illumination, at full frame rate on the GPU.

### 5.2 NEE Debug PNG

When `R` is pressed, before the full render begins, a single-frame
NEE debug PNG is rendered with full $M$ shadow-ray samples and BSDF
evaluation. This is saved to `output/out_debug_nee.png` and provides
a quick direct-lighting preview.

### 5.3 Debug Modes

Switchable via keyboard (TAB to cycle, F-keys for overlays):

| Mode        | Key | Description                          |
|-------------|-----|--------------------------------------|
| Full        | TAB | Direct lighting (default debug mode) |
| Direct Only | TAB | NEE direct only                      |
| Indirect Only | TAB | Photon indirect only               |
| Photon Map  | TAB | Photon density visualisation         |
| Normals     | TAB | Surface normals as RGB               |
| Material ID | TAB | Distinct colour per material         |
| Depth       | TAB | Distance to camera (greyscale)       |

### 5.4 Debug Overlays

| Key  | Overlay                             |
|------|-------------------------------------|
| F1   | Photon points                       |
| F2   | Global photon map                   |
| F3   | Caustic map selector (not yet impl.)|
| F4   | Hash grid debug                     |
| F5   | Photon directions                   |
| F6   | PDFs                                |
| F7   | Gather radius sphere                |
| F8   | MIS weights                         |
| F9   | Spectral colouring                  |
| H    | Help overlay                        |

### 5.5 Hover-Cell Inspection

With mouse released (M key) and a map toggle enabled (F2/F3), moving
the cursor over the image displays a panel with: cell index, photon
count, flux sum/average, and dominant wavelength.

---

## 6. Full Path Tracing (R Key)

Pressing **R** triggers the progressive render pipeline:

1. **NEE debug PNG** (single frame, shadow rays enabled)
2. **Bin population pass** (`populate_photon_bins()`)
3. **Progressive SPP loop** (`render_one_spp()`, 1 spp per iteration)
4. **PNG output** (render, components, timing)

### 6.1 Stratified Sub-Pixel Sampling (B3)

When `is_final_render = 1` and `STRATA_X * STRATA_Y > 1`, sub-pixel
offsets are generated from stratified jittered sampling:

$$
j_x = \frac{s_x + \xi_x}{S_x}, \quad
j_y = \frac{s_y + \xi_y}{S_y}
$$

where $s_x = \text{sample\_index} \bmod S_x$,
$s_y = (\text{sample\_index} / S_x) \bmod S_y$, and
$\xi_x, \xi_y \sim U(0,1)$.

Default: $S_x = S_y = 4$, giving $4 \times 4 = 16$ strata matching
the default 16 SPP.

### 6.2 Path Tracing Algorithm (`full_path_trace`)

For each sample:

1. Trace the camera ray.
2. At each **specular** hit, follow the ideal reflection and
   continue.
3. At each **diffuse** hit:
   - **NEE (direct lighting):** If bins are valid, use guided NEE
     (`dev_nee_guided`, §4.4); otherwise, use standard NEE
     (`dev_nee_direct`). Bounce-dependent sample count: $M=4$ at
     bounce 0, $M=1$ at bounce $\ge 1$.
   - **Photon density estimate (indirect):** If bins valid and
     bounce 0 and frame > 0, read from cached spectral density
     (§4.5). Otherwise, do full hash-grid gather. Result is
     attenuated by $\max(v_{\text{NEE}}, \text{PHOTON\_SHADOW\_FLOOR})$
     where $v_{\text{NEE}}$ is the NEE visibility fraction and
     `PHOTON_SHADOW_FLOOR = 0.1` prevents fully killing indirect
     light in deep shadow.
   - **BSDF continuation:** At bounce 0 with valid bins, use guided
     bounce (§4.3). At deeper bounces, use cosine-weighted hemisphere
     sampling.
4. **Throughput update:** $T_{k+1}(\lambda) = T_k(\lambda) \cdot K_d(\lambda)$

5. **Russian roulette** after `MIN_BOUNCES_RR` (3):

$$
p_{rr} = \min(0.95,\; \max_\lambda T(\lambda))
$$

### 6.3 NEE Direct Lighting (`dev_nee_direct`)

Standard power-proportional CDF sampling of emissive triangles:

1. Sample triangle from CDF (binary search).
2. Sample uniform point on triangle.
3. Compute direction, cosines, shadow ray.
4. Evaluate BSDF and emission.
5. Convert area PDF to solid-angle PDF:

$$
p(\omega_i) = p_{\text{tri}} \cdot \frac{1}{A_{\text{tri}}} \cdot
\frac{d^2}{\cos\theta_o}
$$

6. Accumulate:

$$
\hat{L}_{\text{NEE}} = \frac{1}{M} \sum_{s=1}^{M}
\frac{f_s \cdot L_e \cdot \cos\theta_x}{p(\omega_i)}
$$

The visibility fraction (unoccluded samples / total samples) is
returned alongside the radiance estimate.

### 6.4 Density Estimator

At each diffuse hit, the photon map contribution is:

$$
L_o(x,\omega_o,\lambda) \approx
    \frac{1.5}{\pi r^2 \cdot N_{\text{photons}}}
    \sum_{i} \Phi_i(\lambda) \,
    f_s(x,\omega_i,\omega_o,\lambda) \,
    W(\|x - x_i\|)
$$

where $r$ is the gather radius, $N_{\text{photons}}$ is the total
number of emitted photons, 1.5 is the Epanechnikov kernel correction
factor, and $W$ is the Epanechnikov kernel:

$$
W(d) = 1 - \frac{d^2}{r^2}
$$

### 6.5 Surface Consistency Filtering

To avoid cross-surface contamination, a photon is accepted only if:

- $|\langle \mathbf{n}_x, \mathbf{x}_i - \mathbf{x} \rangle| < \tau$
  (plane-distance filter, $\tau = 0.02$)
- Distance $\|\mathbf{x} - \mathbf{x}_i\| < r$

### 6.6 Progressive Accumulation

The spectrum buffer and sample count are accumulated across frames.
The sRGB output is the running average, tone-mapped per frame via
CIE XYZ -> sRGB with gamma correction.

### 6.7 Component Outputs

After the full render completes, four PNG files are saved:

| File                        | Contents                           |
|-----------------------------|------------------------------------|
| `output/render.png`         | Final combined render              |
| `output/out_nee_direct.png` | NEE direct lighting only           |
| `output/out_photon_indirect.png` | Photon indirect only          |
| `output/out_combined.png`   | NEE + photon (spectral sum)        |
| `output/out_debug_nee.png`  | Single-frame NEE debug preview     |

### 6.8 GPU Kernel Profiling

Per-pixel `clock64()` timers accumulate across all bounces and
samples. After the render, `print_kernel_profiling()` prints a
breakdown of time spent in:

- Ray tracing (`optixTrace`)
- NEE (shadow rays + light sampling)
- Photon gather (density estimation)
- BSDF evaluation + continuation

---

## 7. Spectral Framework

### 7.1 Representation

All colour computation uses a `Spectrum` struct with `NUM_LAMBDA =
32` float bins covering 380--780 nm (bin width ~12.5 nm). No RGB
shortcuts are used in the light transport.

### 7.2 CIE Colour Matching

The final spectrum-to-sRGB conversion uses an analytic approximation
of the CIE 1931 colour matching functions (Wyman et al. Gaussian
fit). The pipeline is:

$$
\text{Spectrum} \xrightarrow{\text{CIE XYZ}} (X,Y,Z)
\xrightarrow{\text{sRGB matrix}} (R,G,B)_{\text{linear}}
\xrightarrow{\gamma} (R,G,B)_{\text{sRGB}}
$$

Normalisation: $X, Y, Z$ are divided by $\sum \bar{y}(\lambda)$ so
that a flat-1.0 spectrum maps to $Y = 1$.

### 7.3 Blackbody Emission

Emissive materials with `Ke` in the MTL file are converted to a
Planck blackbody spectrum at a configurable colour temperature,
scaled by the luminance of the `Ke` value.

---

## 8. BSDF Models

| Model      | $f_s$                            | Sampling PDF                     |
|------------|----------------------------------|----------------------------------|
| Lambertian | $K_d / \pi$                       | $\cos\theta / \pi$              |
| Mirror     | ideal specular reflection         | delta distribution               |
| Glass      | Fresnel-weighted reflect/refract  | Schlick approximation            |
| Glossy     | GGX microfacet (Cook-Torrance)   | VNDF sampling (Heitz 2018)       |

All BSDF evaluations and PDFs are spectral: the albedo $K_d(\lambda)$
or $K_s(\lambda)$ is evaluated per-bin.

**Runtime note:** The OptiX device code currently evaluates only
**Lambertian** and **mirror** BSDFs. Glossy and glass material types
fall through to specular bounces (mirror reflection) but do not use
the full GGX microfacet model on the device.

---

## 9. OptiX Program Structure

All device code lives in a single compilation unit
`src/optix/optix_device.cu`, compiled to PTX via a custom `nvcc`
command with `--use_fast_math` (not CMake's CUDA object compilation,
to avoid config-dependent output paths on MSVC).

### Programs

| Program                    | Type       | Purpose                             |
|----------------------------|------------|-------------------------------------|
| `__raygen__render`         | Ray Gen    | Debug first-hit / full path trace / bin population |
| `__raygen__photon_trace`   | Ray Gen    | GPU photon emission + tracing       |
| `__closesthit__radiance`   | Closest-Hit| Unpack geometry at hit point         |
| `__closesthit__shadow`     | Closest-Hit| Set "occluded" flag                  |
| `__miss__radiance`         | Miss       | Return zero radiance                 |
| `__miss__shadow`           | Miss       | Return "not occluded"                |

### Payload Layout (14 values)

| Slot  | Contents               |
|-------|------------------------|
| p0-p2 | Hit position (float3)  |
| p3-p5 | Shading normal (float3)|
| p6    | Hit distance t         |
| p7    | Material ID            |
| p8    | Triangle (primitive) ID|
| p9    | Hit flag (0=miss, 1=hit)|
| p10-p12| Geometric normal (float3)|
| p13   | Reserved               |

### SBT (Shader Binding Table)

Two ray types are used:

- **Type 0 -- Radiance**: closest-hit writes full geometry payload.
- **Type 1 -- Shadow**: closest-hit writes occluded flag; uses
  `OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT`.

The SBT has two raygen records: the default one for rendering and a
separate one for photon tracing. The photon raygen record is swapped
in temporarily during `trace_photons()`.

### OptiX Pipeline Configuration

| Parameter                    | Value   |
|------------------------------|---------|
| `OPTIX_NUM_PAYLOAD_VALUES`   | 14      |
| `OPTIX_NUM_ATTRIBUTE_VALUES` | 2       |
| `OPTIX_MAX_TRACE_DEPTH`      | 2       |
| `OPTIX_STACK_SIZE`           | 16,384  |

The stack size was increased from 2,048 to 16,384 to accommodate
`PhotonBin[32]` and `PhotonBinDirs` arrays on the GPU thread stack
during path tracing.

---

## 10. Source Layout

```
src/
  main.cpp                    Entry point, GLFW loop, arg parsing
  core/
    types.h                   float3, Ray, HitRecord, ONB
    spectrum.h                Spectrum struct, CIE matching, blackbody
    config.h                  All tunable constants
    random.h                  PCG32 RNG (host + device)
    alias_table.h             Alias method for discrete sampling
    photon_bins.h             PhotonBin struct, PhotonBinDirs (Fibonacci sphere)
    font_overlay.h            Watermark/overlay stamping for PNG output
  bsdf/
    bsdf.h                    BSDF evaluation, sampling, PDF
  scene/
    scene.h                   Scene struct (triangles, materials, BVH)
    obj_loader.h / .cpp       Wavefront OBJ + MTL loader
  renderer/
    renderer.h / .cpp         CPU renderer, RenderConfig, FrameBuffer
    camera.h                  Camera (position, FOV, ray generation)
    mis.h                     MIS weight utilities
    path_tracer.cu            CPU/CUDA path tracing kernels
    direct_light.cu / .h      CPU/CUDA direct lighting kernels
  photon/
    photon.h                  Photon and PhotonSoA structs
    hash_grid.h / .cu         Hashed uniform grid (build + query)
    emitter.h / .cu           CPU photon emission + tracing
    density_estimator.h       Density estimation utilities
  optix/
    optix_renderer.h / .cpp   Host-side OptiX pipeline management
    optix_device.cu           All OptiX device programs (PTX source)
    launch_params.h           Shared host/device LaunchParams struct
  debug/
    debug.h                   Debug key bindings, DebugState
tests/
  test_main.cpp               163 unit tests (GoogleTest)
```

---

## 11. Strengths

1. **Full spectral transport.** All light transport is computed over
   32 wavelength bins. Dispersion, metamerism, and spectral emission
   are naturally captured without RGB approximations.

2. **GPU-accelerated everywhere.** Photon tracing, path tracing, bin
   population, and the debug viewer all run on the GPU via OptiX. No
   CPU fallback means the GPU is always utilised.

3. **Photon-guided importance sampling.** The directional bin cache
   steers both BSDF bounce directions (B1) and NEE light selection
   (B2) toward directions where photon flux is concentrated. This
   significantly reduces variance for indirect illumination.

4. **Cached density estimation.** The first-sample photon gather is
   cached per pixel, eliminating redundant hash-grid lookups on
   subsequent SPP frames.

5. **Combined photon + path tracing.** The photon density estimate
   provides indirect illumination (including colour bleeding and
   caustics) while NEE handles direct lighting. This converges faster
   than pure path tracing for scenes with difficult light paths.

6. **Interactive debug viewer.** First-hit OptiX rendering gives
   instant scene feedback with direct lighting, normals, material
   IDs, depth visualisation, and photon overlay -- all at GPU speed.

7. **Explicit math.** Every PDF, every Monte Carlo estimator, and
   every throughput update is written out explicitly. No hidden
   normalisation, no implicit conventions.

8. **Comprehensive test suite.** 163 unit tests cover vector math,
   spectral operations, BSDFs, sampling distributions, the hash grid,
   density estimation, Fibonacci sphere binning, guided bounce,
   stratified sampling, scene loading, and OptiX integration.

9. **Stratified sampling.** $4 \times 4$ stratified sub-pixel jitter
   reduces clumping artifacts at 16 SPP.

10. **Visibility-weighted photon attenuation.** NEE shadow-ray
    visibility modulates the photon density contribution, preserving
    contact shadows while allowing some indirect illumination via
    `PHOTON_SHADOW_FLOOR`.

---

## 12. Weaknesses and Limitations

1. **Fixed photon gather radius.** No progressive photon mapping or
   adaptive radius. The bias from the fixed kernel width does not
   vanish as sample count increases.

2. **No MIS between NEE and photon estimate.** Direct and indirect
   contributions are currently summed without multi-strategy MIS
   weighting, which can introduce variance at transition boundaries.

3. **Lambertian-only in OptiX device code.** The OptiX path tracer
   currently evaluates only Lambertian BSDFs. Glossy and glass
   materials fall through to specular bounces (mirror reflection)
   but do not use the full GGX microfacet model on the device.

4. **Single GAS, no instancing.** The scene is a flat triangle soup
   with no support for instanced geometry or multi-level acceleration
   structures.

5. **No texture mapping in OptiX.** Material properties in the OptiX
   device code are per-material (constant), not per-texel. The OBJ
   loader supports UV coordinates but they are not sampled on the
   device.

6. **Hash grid built on CPU.** After the GPU photon trace, photon
   data is downloaded, the grid is built on the CPU, then re-uploaded.
   This is a pragmatic choice (grid build is fast) but adds a
   synchronisation point.

7. **No denoising.** The raw Monte Carlo output is displayed without
   any AI or bilateral denoiser.

8. **Windows-centric build.** The build system and `run.bat` are
   tested primarily on Windows with MSVC. Linux builds should work
   but are not actively validated.

9. **Large VRAM for bin cache.** At 1024x768 with 32 bins, the bin
   cache alone uses ~604 MB. Higher resolutions or bin counts will
   require proportionally more VRAM.

10. **Guided NEE stack budget.** The guided NEE builds a temporary
    CDF on the GPU stack, limited to 128 emissive triangles. Scenes
    with more than 128 emissive triangles fall back to standard
    (non-guided) NEE.

---

## 13. Configuration

All tunable constants are centralised in `src/core/config.h`:

### Sampling & Bounces

| Parameter                | Default     | Description                       |
|--------------------------|-------------|-----------------------------------|
| `NUM_LAMBDA`             | 32          | Wavelength bins (380--780 nm)     |
| `DEFAULT_SPP`            | 16          | Samples per pixel (final render)  |
| `DEFAULT_MAX_BOUNCES`    | 8           | Maximum path bounces              |
| `DEFAULT_MIN_BOUNCES_RR` | 3           | Bounces before Russian roulette   |
| `DEFAULT_RR_THRESHOLD`   | 0.95        | Max RR survival probability       |

### Photon Mapping

| Parameter                | Default     | Description                       |
|--------------------------|-------------|-----------------------------------|
| `DEFAULT_NUM_PHOTONS`    | 1,000,000   | Photons emitted per trace         |
| `DEFAULT_GATHER_RADIUS`  | 0.05        | Photon gather radius              |
| `DEFAULT_CAUSTIC_RADIUS` | 0.02        | Caustic map gather radius         |
| `HASHGRID_CELL_FACTOR`   | 2.0         | cell_size = factor * radius       |
| `DEFAULT_SURFACE_TAU`    | 0.02        | Plane-distance filter thickness   |
| `PHOTON_SHADOW_FLOOR`    | 0.1         | Min visibility weight for photon gather |

### Photon Directional Bins

| Parameter                | Default     | Description                       |
|--------------------------|-------------|-----------------------------------|
| `PHOTON_BIN_COUNT`       | 32          | Fibonacci sphere bins per pixel   |
| `MAX_PHOTON_BIN_COUNT`   | 32          | Compile-time upper bound          |
| `PHOTON_BIN_HORIZON_EPS` | 0.05        | Bins below $-\epsilon$ of horizon skipped |
| `PHOTON_BIN_NEE_TOP_K`   | 4           | Top-K bins for guided NEE bias    |
| `NEE_GUIDED_MAX_EMISSIVE`| 128         | Max emissive tris for stack CDF   |
| `NEE_GUIDED_ALPHA`       | 5.0         | Flux-boost strength for guided NEE|

### Stratified Sampling

| Parameter                | Default     | Description                       |
|--------------------------|-------------|-----------------------------------|
| `STRATA_X`               | 4           | Horizontal strata                 |
| `STRATA_Y`               | 4           | Vertical strata                   |

### NEE

| Parameter                | Default     | Description                       |
|--------------------------|-------------|-----------------------------------|
| `DEFAULT_NEE_LIGHT_SAMPLES` | 4        | Shadow rays at bounce 0           |
| `DEFAULT_NEE_DEEP_SAMPLES`  | 1        | Shadow rays at bounce >= 1        |

### Image & Window

| Parameter                    | Default     | Description                   |
|------------------------------|-------------|-------------------------------|
| `DEFAULT_IMAGE_WIDTH`        | 1024        | Output image width            |
| `DEFAULT_IMAGE_HEIGHT`       | 768         | Output image height           |
| `DEFAULT_WINDOW_WIDTH`       | 1024        | Debug window width            |
| `DEFAULT_WINDOW_HEIGHT`      | 768         | Debug window height           |

### OptiX

| Parameter                    | Default     | Description                   |
|------------------------------|-------------|-------------------------------|
| `OPTIX_NUM_PAYLOAD_VALUES`   | 14          | Payloads per trace call       |
| `OPTIX_MAX_TRACE_DEPTH`      | 2           | Ray types (radiance + shadow) |
| `OPTIX_STACK_SIZE`           | 16,384      | GPU thread stack size (bytes) |
| `OPTIX_SCENE_EPSILON`       | 1e-4        | Shadow/continuation offset    |

---

## 14. Key Data Structures

### Spectrum
```cpp
struct Spectrum {
    float value[NUM_LAMBDA];  // 32 bins, 380-780 nm
};
```

### PhotonBin (24 bytes)
```cpp
struct PhotonBin {
    float flux;       // total Epanechnikov-weighted scalar flux
    float dir_x;      // flux-weighted centroid direction x
    float dir_y;      // flux-weighted centroid direction y
    float dir_z;      // flux-weighted centroid direction z
    float weight;     // total Epanechnikov weight
    int   count;      // number of photons in this bin
};
```

### PhotonBinDirs (Fibonacci sphere)
```cpp
struct PhotonBinDirs {
    float3 dirs[MAX_PHOTON_BIN_COUNT];
    int    count;
    void init(int n);                // Fibonacci sphere construction
    int  find_nearest(float3 wi);    // brute-force dot product scan O(N)
};
```

### Photon (SoA)
```cpp
struct PhotonSoA {
    vector<float> pos_x, pos_y, pos_z;
    vector<float> wi_x, wi_y, wi_z;
    vector<uint16_t> lambda_bin;
    vector<float> flux;
};
```

### LaunchParams (host <-> device)
Contains all device pointers: framebuffer, scene geometry, materials,
photon map, hash grid, emitter CDF, camera, rendering flags
(`is_final_render`, `render_mode`, `samples_per_pixel`),
profiling timers, and photon bin cache pointers
(`photon_bin_cache`, `photon_density_cache`, `photon_bin_count`,
`photon_bins_valid`, `populate_bins_mode`).

### HashGrid
Hashed uniform grid with `cellStart` / `cellEnd` / `sortedIndices`
arrays. Hash function uses three large primes:
```
hash(cx,cy,cz) = (cx*73856093 ^ cy*19349663 ^ cz*83492791) % table_size
```

---

## 15. Rendering Equation Reference

The rendering equation solved by the path tracer:

$$
L_o(x, \omega_o, \lambda) = L_e(x, \omega_o, \lambda) +
    \int_{\mathcal{H}^2} f_s(x, \omega_i, \omega_o, \lambda) \,
    L_i(x, \omega_i, \lambda) \, \cos\theta_i \, d\omega_i
$$

The Monte Carlo estimator with throughput $T$ at bounce $k$:

$$
\hat{L} = \sum_{k=0}^{K} T_k \cdot
    \left[ L_e^{(k)} + \hat{L}_{\text{NEE}}^{(k)} +
           \hat{L}_{\text{photon}}^{(k)} \right]
$$

where $\hat{L}_{\text{NEE}}$ is the next-event estimation
contribution (guided or standard) and $\hat{L}_{\text{photon}}$ is
the photon density estimate contribution at each diffuse hit along
the path. The photon contribution is attenuated by
$\max(v_{\text{NEE}}, 0.1)$ to preserve contact shadows.

The guided NEE estimator (B2):

$$
\hat{L}_{\text{NEE,guided}} = \frac{1}{M} \sum_{s=1}^{M}
\frac{f_s(x, \omega_s, \omega_o, \lambda) \cdot L_e(y_s, \lambda) \cdot
\cos\theta_x}{p_{\text{guided}}(\omega_s)}
$$

where $p_{\text{guided}}$ is the bin-flux-weighted solid-angle PDF.

## Ground Truth Testing

```
 ══════════════════════════════════════════════════════════════════════
  COMPARISON WORKFLOW: Ground Truth vs Optimized
 ══════════════════════════════════════════════════════════════════════

                        ┌──────────────────────────┐
                        │  tests/data/cornell_box.bin  │  Real photon data
                        │  (binary, ~40MB)             │  from disk
                        │                              │
                        │  Created by:                 │
                        │  1) --save-test-data (GPU)   │
                        │  2) CPU fallback (bootstrap) │
                        └──────────────┬───────────────┘
                                       │
               ┌───────────────────────┼───────────────────────┐
               │                       │                       │
               ▼                       ▼                       ▼
    ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
    │ src/core/         │   │ src/scene/        │   │ src/photon/       │
    │  test_data_io.h   │   │  scene.h          │   │  hash_grid.h     │
    │                   │   │  obj_loader.h     │   │                  │
    │ load_test_data()  │   │                   │   │ grid.build()     │
    │ save_test_data()  │   │ load_obj()        │   │ grid.query()     │
    └────────┬─────────┘   │ scene.build_bvh() │   └────────┬─────────┘
             │              │ scene.intersect() │            │
             │              └────────┬─────────┘            │
             │                       │                      │
             ▼                       ▼                      ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                CornellBoxDataset (singleton)                 │
    │  tests/test_ground_truth.cpp  &  tests/test_per_ray_validation.cpp
    │                                                              │
    │  ┌─ Loaded from binary: ───────────────────────────────────┐ │
    │  │  PhotonSoA photons       (1.3M photons, 9 SoA arrays)  │ │
    │  │  TestDataHeader header   (radii, bounces, scene path)   │ │
    │  └─────────────────────────────────────────────────────────┘ │
    │  ┌─ Rebuilt deterministically: ────────────────────────────┐ │
    │  │  Scene scene             (OBJ → BVH → emissive dist)   │ │
    │  │  HashGrid grid           (from photons + radius)        │ │
    │  │  PhotonBinDirs bin_dirs  (32 Fibonacci directions)      │ │
    │  │  vector<uint8_t> bin_idx (precomputed per-photon)       │ │
    │  │  Camera camera           (64×64 Cornell box preset)     │ │
    │  └─────────────────────────────────────────────────────────┘ │
    └──────────────────────────┬───────────────────────────────────┘
                               │
                          pick N random rays
                          (deterministic RNG seeds)
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
 ┌─────────────────────────────┐  ┌─────────────────────────────────┐
 │  GROUND TRUTH PATH TRACER   │  │  OPTIMIZED PATH TRACER          │
 │  ground_truth_path_trace()  │  │  optimized_path_trace()         │
 │  ground_truth_trace_steps() │  │  optimized_trace_steps()        │
 │                             │  │                                 │
 │  ┌─ Per Bounce: ──────────┐ │  │  ┌─ Per Bounce: ──────────────┐ │
 │  │                        │ │  │  │                            │ │
 │  │ 1. scene.intersect()   │ │  │  │ 1. scene.intersect()       │ │
 │  │    ↓                   │ │  │  │    ↓                       │ │
 │  │ 2. Material dispatch   │ │  │  │ 2. Material dispatch       │ │
 │  │    emissive → Le       │ │  │  │    emissive → Le           │ │
 │  │    specular → reflect  │ │  │  │    specular → reflect      │ │
 │  │    diffuse → ↓         │ │  │  │    diffuse → ↓             │ │
 │  │                        │ │  │  │                            │ │
 │  │ 3. NEE (standard)     ◄├─┤──┤─►3. NEE (GUIDED by bins)    │ │
 │  │    sample_direct_light │ │  │  │    opt_nee_guided_step()   │ │
 │  │    1 shadow ray        │ │  │  │    photon-biased CDF       │ │
 │  │    alias table PDF     │ │  │  │    4 shadow rays @ b=0     │ │
 │  │    no MIS              │ │  │  │                            │ │
 │  │                        │ │  │  │                            │ │
 │  │ 4. Photon density     ◄├─┤──┤─►4. Photon density + BINS    │ │
 │  │    estimate_photon_    │ │  │  │    gather_with_bins_step() │ │
 │  │      density()         │ │  │  │    SAME gather loop        │ │
 │  │    full hash grid query│ │  │  │    + populate local_bins[] │ │
 │  │    Epanechnikov kernel │ │  │  │    + shadow floor weight   │ │
 │  │    per-λ BSDF eval     │ │  │  │                            │ │
 │  │                        │ │  │  │                            │ │
 │  │ 5. BSDF bounce        ◄├─┤──┤─►5. BSDF bounce (GUIDED)    │ │
 │  │    bsdf::sample()      │ │  │  │    opt_guided_bounce_step()│ │
 │  │    cosine hemisphere   │ │  │  │    flux-proportional CDF   │ │
 │  │    T *= f·cos/pdf      │ │  │  │    cone jitter + clamp    │ │
 │  │                        │ │  │  │    T *= Kd (simplified)    │ │
 │  │ 6. Russian roulette    │ │  │  │ 6. Russian roulette        │ │
 │  │    p_rr = min(0.95,    │ │  │  │    (same)                  │ │
 │  │      T.max_component())│ │  │  │                            │ │
 │  └────────────────────────┘ │  │  └────────────────────────────┘ │
 │                             │  │                                 │
 │  Output:                    │  │  Output:                        │
 │    result.combined          │  │    result.combined              │
 │    result.nee_direct        │  │    result.nee_direct            │
 │    result.photon_indirect   │  │    result.photon_indirect       │
 │                             │  │                                 │
 │  (test_per_ray_validation   │  │  (test_per_ray_validation       │
 │   also records per-bounce   │  │   also records per-bounce       │
 │   BounceStep with validity  │  │   BounceStep with validity      │
 │   flags at each step)       │  │   flags at each step)           │
 └──────────────┬──────────────┘  └──────────────┬──────────────────┘
                │                                 │
                └────────────┬────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────┐
              │         COMPARISON ENGINE         │
              │  diagnose_ray() / compare_methods()
              │                                   │
              │  Per-ray checks:                  │
              │   • combined relErr < 0.70        │
              │   • NEE relErr < 0.80             │
              │   • photon relErr < 0.50          │
              │   • combined ≈ nee + photon       │
              │   • all values finite & ≥ 0       │
              │   • throughput < 1e6 (no explode) │
              │   • spectral bins don't mix       │
              │                                   │
              │  Aggregate checks:                │
              │   • mean relErr < 0.30            │
              │   • energy ratio 0.65–1.50        │
              │   • < 40% rays exceed thresholds  │
              │                                   │
              │  Output: warnings for deviations  │
              │  + per-bounce breakdown of worst   │
              └──────────────────────────────────┘


 ══════════════════════════════════════════════════════════════════════
  SOURCE FILES
 ══════════════════════════════════════════════════════════════════════

  Test files (CPU-side comparison):
    tests/test_ground_truth.cpp       16 tests, aggregate stats
    tests/test_per_ray_validation.cpp 15 tests, per-ray step-by-step

  Data I/O:
    src/core/test_data_io.h           Binary save/load PhotonSoA

  Shared physics (header-only, HD = host+device):
    src/renderer/direct_light.h       sample_direct_light()      [NEE]
    src/photon/density_estimator.h    estimate_photon_density()   [Gather]
    src/bsdf/bsdf.h                   bsdf::evaluate/sample()    [BSDF]
    src/core/photon_bins.h            PhotonBinDirs, PhotonBin   [Bins]
    src/core/guided_nee.h             guided_nee_bin_boost()      [Guided NEE]
    src/core/nee_sampling.h           nee_shadow_sample_count()
    src/photon/hash_grid.h            HashGrid::query()
    src/scene/scene.h                 Scene::intersect()          [BVH]
    src/core/spectrum.h               Spectrum[32 bins]
    src/core/random.h                 PCGRng, cosine hemisphere

  Entry point (data generation):
    src/main.cpp                      --save-test-data flag
    
```

## Light bounces

```
Camera ray
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  BOUNCE 0  (first diffuse hit)                      │
│                                                     │
│  ┌──────────────┐   ┌─────────────────────────────┐ │
│  │  NEE (direct)│   │  NO photon density term     │ │
│  │  shadow rays │   │  (intentionally omitted)    │ │
│  │  → light src │   │                             │ │
│  └──────────────┘   └─────────────────────────────┘ │
│                                                     │
│  ┌──────────────────────────────────────────────────┐│
│  │  BSDF continuation → pick next bounce direction ││
│  │  80% guided (by photon bins)                    ││
│  │  20% cosine hemisphere (random)                 ││
│  └──────────────────────────────────────────────────┘│
└──────────────────────────┬──────────────────────────┘
                           │ throughput *= f·cos/pdf
                           ▼
┌─────────────────────────────────────────────────────┐
│  BOUNCE 1  (second diffuse hit)                     │
│                                                     │
│  ┌──────────────┐   ┌─────────────────────────────┐ │
│  │  NEE (direct)│   │  NO photon density term     │ │
│  │  guided by   │   │  (intentionally omitted)    │ │
│  │  local bins  │   │                             │ │
│  └──────────────┘   └─────────────────────────────┘ │
│                                                     │
│  ┌──────────────────────────────────────────────────┐│
│  │  BSDF continuation → pick next bounce direction ││
│  │  80% guided / 20% cosine                       ││
│  └──────────────────────────────────────────────────┘│
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
                    ... (up to max_bounces) ...
                           │
                           ▼
                    Hit emissive? → Le × throughput × MIS weight
                    Miss / RR kill? → path dies with 0

   CEILING (hitpoint)
       │
       │  Bounce 0: NEE shadow rays → aim at lights
       │            (guided bins may point toward exit sign
       │             if photons from sign landed here)
       │
       │  BSDF bounce: 80% chance → bin CDF picks direction
       │                            toward EXIT SIGN
       │                20% chance → random cosine direction
       ▼
   WALL / FLOOR (bounce 1 hitpoint)
       │
       │  NEE → shadow rays to lights (including exit sign)
       │  If ray hits exit sign: Le_green × throughput → GREEN
       │
       │  BSDF bounce continues deeper...
       ▼


Camera → Ceiling → (guided bounce toward sign) → EXIT SIGN (emissive)
                                                      │
                                                      ▼
                                              Le = (0, 1, 0) green
                                              × throughput (albedo of ceiling)
                                              × MIS weight
                                              ─────────────────────
                                              = green contribution


┌──────────────────────────────────────────────────────┐
│  p_guided = 0.80  (from DEFAULT_GUIDED_BSDF_MIX)    │
│                                                      │
│  roll random ξ ∈ [0,1)                               │
│  if ξ < 0.80:                                        │
│     dev_sample_guided_bounce():                      │
│       1. Build CDF over bins weighted by flux×cos    │
│       2. Select bin k ~ CDF                          │
│       3. Jitter within cone half-angle               │
│       4. Clamp to hemisphere                         │
│  else:                                               │
│     sample_cosine_hemisphere():                      │
│       random direction ~ cos(θ)/π                    │
│                                                      │
│  pdf_mix = 0.80 × pdf_guided + 0.20 × pdf_cosine    │
│  throughput *= f × cos / pdf_mix                     │
└──────────────────────────────────────────────────────┘

```
