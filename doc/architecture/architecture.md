# Architecture -- Spectral Photon + Path Tracing Renderer

This document describes the complete rendering pipeline, its design
rationale, mathematical foundations, strengths, weaknesses, and
intended use.

---

## 1. Overview

The renderer combines **photon mapping** with **Monte Carlo path
tracing** in a fully spectral framework. All light transport is
computed over 32 discrete wavelength bins spanning 380--780 nm. The
entire pipeline runs on the GPU via **NVIDIA OptiX 7+** for ray
tracing and **CUDA** for auxiliary kernels. There is no CPU fallback.

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
  Full Path Tracing  (__raygen__render, is_final_render=1)
        |
        v
  PNG Output
```

### 2.1 Scene Loading

Wavefront OBJ with MTL materials. Materials are mapped to the
internal spectral representation using `rgb_to_spectrum_reflectance()`
for diffuse/specular albedos and `blackbody_spectrum()` for emissive
surfaces whose emission is defined by `Ke` in the MTL file.

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
`num_photons` threads.

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
- **Diffuse surfaces**: the photon is stored, then scattered via
  cosine-weighted hemisphere sampling. The flux is attenuated by the
  surface albedo $K_d(\lambda)$.

Storage uses `atomicAdd` on a global counter to append photons into
pre-allocated SoA output buffers (position, direction, wavelength
bin, flux).

**Russian roulette** is applied after `MIN_BOUNCES_RR` bounces:

$$
p_{rr} = \min(0.95,\; \max_\lambda T(\lambda))
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
factor 2.0). Neighbour lookup scans 3x3x3 cells. The grid stores
sorted indices plus `cellStart` / `cellEnd` arrays indexed by hash.

---

## 4. Debug Viewer (Interactive)

The debug viewer is an interactive GLFW window rendering at 1 spp per
frame via OptiX. It launches the same `__raygen__render` program with
`is_final_render = 0`, which calls `debug_first_hit()`.

### 4.1 First-Hit Rendering

For each pixel:

1. Trace a primary ray via OptiX.
2. If the hit is emissive, return $L_e$.
3. If the hit is specular, follow up to 4 specular bounces.
4. At the first diffuse hit, perform **next-event estimation** (NEE):
   sample one emissive triangle, cast a shadow ray, compute direct
   illumination.

This gives an interactive preview with direct lighting but no global
illumination, at full frame rate on the GPU.

### 4.2 Debug Modes

Switchable via keyboard (TAB to cycle, F-keys for overlays):

| Mode        | Key | Description                          |
|-------------|-----|--------------------------------------|
| Full        | TAB | Direct lighting (default debug mode) |
| Normals     | TAB | Surface normals as RGB               |
| Material ID | TAB | Distinct colour per material         |
| Depth       | TAB | Distance to camera (greyscale)       |

---

## 5. Full Path Tracing (R Key)

Pressing **R** triggers `render_final()`, which launches
`__raygen__render` with `is_final_render = 1`. This calls
`full_path_trace()`.

### 5.1 Algorithm

For each sample:

1. Trace the camera ray.
2. At each **specular** hit, follow the ideal reflection and
   continue.
3. At each **diffuse** hit:
   - **Direct lighting** via NEE (sample emissive triangle, shadow
     ray, MIS-ready structure).
   - **Photon density estimate** from the hash grid (indirect
     lighting contribution).
   - **BSDF sampling** for the next bounce direction.
4. Throughput update:

$$
T_{k+1}(\lambda) = T_k(\lambda) \cdot
    \frac{f_s(x,\omega_i,\omega_o,\lambda)\cos\theta}{p(\omega_i)}
$$

5. **Russian roulette** after `MIN_BOUNCES_RR`:

$$
p_{rr} = \min(0.95,\; \max_\lambda T(\lambda))
$$

### 5.2 Density Estimator

At each diffuse hit, the contribution from the photon map is:

$$
L_o(x,\omega_o,\lambda) \approx
    \frac{1}{\pi r^2 \cdot N}
    \sum_{i} \Phi_i(\lambda) \,
    f_s(x,\omega_i,\omega_o,\lambda) \,
    W(\|x - x_i\|)
$$

where $r$ is the gather radius, $N$ is the total number of emitted
photons, and $W$ is the Epanechnikov kernel:

$$
W(d) = 1 - \frac{d^2}{r^2}
$$

### 5.3 Surface Consistency Filtering

To avoid cross-surface contamination, a photon is accepted only if:

- $|\langle n_x, x_i - x \rangle| < \tau$ (plane-distance filter,
  $\tau = 0.02$).
- Distance $\|x - x_i\| < r$.

### 5.4 Progressive Accumulation

The spectrum buffer and sample count are accumulated across frames.
The sRGB output is the running average, tone-mapped per frame via
CIE XYZ -> sRGB with gamma correction.

---

## 6. Spectral Framework

### 6.1 Representation

All colour computation uses a `Spectrum` struct with 32 float bins
covering 380--780 nm (bin width ~12.5 nm). No RGB shortcuts are used
in the light transport.

### 6.2 CIE Colour Matching

The final spectrum-to-sRGB conversion uses an analytic approximation
of the CIE 1931 colour matching functions (Wyman et al. Gaussian
fit). The pipeline is:

$$
\text{Spectrum} \xrightarrow{\text{CIE XYZ}} (X,Y,Z)
\xrightarrow{\text{sRGB matrix}} (R,G,B)_{\text{linear}}
\xrightarrow{\gamma} (R,G,B)_{\text{sRGB}}
$$

### 6.3 Blackbody Emission

Emissive materials with `Ke` in the MTL file are converted to a
Planck blackbody spectrum at a configurable colour temperature,
scaled by the luminance of the `Ke` value.

---

## 7. BSDF Models

| Model      | $f_s$                            | Sampling PDF                     |
|------------|----------------------------------|----------------------------------|
| Lambertian | $K_d / \pi$                       | $\cos\theta / \pi$              |
| Mirror     | ideal specular reflection         | delta distribution               |
| Glass      | Fresnel-weighted reflect/refract  | Schlick approximation            |
| Glossy     | GGX microfacet (Cook-Torrance)   | VNDF sampling (Heitz 2018)       |

All BSDF evaluations and PDFs are spectral: the albedo $K_d(\lambda)$
or $K_s(\lambda)$ is evaluated per-bin.

---

## 8. OptiX Program Structure

All device code lives in a single compilation unit
`src/optix/optix_device.cu`, compiled to PTX via a custom `nvcc`
command (not CMake's CUDA object compilation, to avoid
config-dependent output paths on MSVC).

### Programs

| Program                    | Type       | Purpose                           |
|----------------------------|------------|-----------------------------------|
| `__raygen__render`         | Ray Gen    | Debug first-hit / full path trace |
| `__raygen__photon_trace`   | Ray Gen    | GPU photon emission + tracing     |
| `__closesthit__radiance`   | Closest-Hit| Unpack geometry at hit point      |
| `__closesthit__shadow`     | Closest-Hit| Set "occluded" flag               |
| `__miss__radiance`         | Miss       | Return zero radiance              |
| `__miss__shadow`           | Miss       | Return "not occluded"             |

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

---

## 9. Source Layout

```
src/
  main.cpp                    Entry point, GLFW loop, arg parsing
  core/
    types.h                   float3, Ray, HitRecord, ONB
    spectrum.h                Spectrum struct, CIE matching, blackbody
    config.h                  All tunable constants
    random.h                  PCG32 RNG (host + device)
    alias_table.h             Alias method for discrete sampling
  bsdf/
    bsdf.h                   BSDF evaluation, sampling, PDF
  scene/
    scene.h                  Scene struct (triangles, materials, BVH)
    obj_loader.h / .cpp      Wavefront OBJ + MTL loader
  renderer/
    renderer.h / .cpp        CPU renderer, RenderConfig, FrameBuffer
    camera.h                 Camera (position, FOV, ray generation)
    path_tracer.cu           CPU/CUDA path tracing kernels
    direct_light.cu          CPU/CUDA direct lighting kernels
  photon/
    photon.h                 Photon and PhotonSoA structs
    hash_grid.h / .cu        Hashed uniform grid (build + query)
    emitter.h / .cu          CPU photon emission + tracing
  optix/
    optix_renderer.h / .cpp  Host-side OptiX pipeline management
    optix_device.cu          All OptiX device programs (PTX source)
    launch_params.h          Shared host/device LaunchParams struct
  debug/
    debug.h                  Debug key bindings, DebugState
tests/
  test_main.cpp              152 unit tests (GoogleTest)
```

---

## 10. Strengths

1. **Full spectral transport.** All light transport is computed over
   32 wavelength bins. Dispersion, metamerism, and spectral emission
   are naturally captured without RGB approximations.

2. **GPU-accelerated everywhere.** Photon tracing, path tracing, and
   the debug viewer all run on the GPU via OptiX. No CPU fallback
   means the GPU is always utilised.

3. **Combined photon + path tracing.** The photon density estimate
   provides indirect illumination (including colour bleeding and
   caustics) while NEE handles direct lighting. This converges faster
   than pure path tracing for scenes with difficult light paths.

4. **Interactive debug viewer.** First-hit OptiX rendering gives
   instant scene feedback with direct lighting, normals, material
   IDs, and depth visualisation -- all at GPU speed.

5. **Explicit math.** Every PDF, every Monte Carlo estimator, and
   every throughput update is written out explicitly. No hidden
   normalisation, no implicit conventions.

6. **Comprehensive test suite.** 152 unit tests cover vector math,
   spectral operations, BSDFs, sampling distributions, the hash grid,
   density estimation, scene loading, and OptiX integration.

---

## 11. Weaknesses and Limitations

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

---

## 12. Configuration

All tunable constants are centralised in `src/core/config.h`:

| Parameter              | Default       | Description                       |
|------------------------|---------------|-----------------------------------|
| `NUM_LAMBDA`           | 32            | Wavelength bins (380--780 nm)     |
| `DEFAULT_SPP`          | 16            | Samples per pixel (final render)  |
| `DEFAULT_NUM_PHOTONS`  | 500,000       | Photons emitted per trace         |
| `DEFAULT_GATHER_RADIUS`| 0.05          | Photon gather radius              |
| `DEFAULT_CAUSTIC_RADIUS`| 0.02         | Caustic map gather radius         |
| `DEFAULT_MAX_BOUNCES`  | 8             | Maximum path bounces              |
| `DEFAULT_MIN_BOUNCES_RR`| 3            | Bounces before Russian roulette   |
| `DEFAULT_RR_THRESHOLD` | 0.95          | Max RR survival probability       |
| `DEFAULT_IMAGE_WIDTH/HEIGHT`| 512x512  | Default output resolution         |
| `OPTIX_NUM_PAYLOAD_VALUES`| 14         | OptiX payload slots               |
| `OPTIX_MAX_TRACE_DEPTH`| 2             | Ray types (radiance + shadow)     |

---

## 13. Key Data Structures

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
    vector<float> wi_x, wi_y, wi_z;
    vector<uint16_t> lambda_bin;
    vector<float> flux;
};
```

### LaunchParams (host <-> device)
Contains all device pointers: framebuffer, scene geometry, materials,
photon map, hash grid, emitter CDF, camera, and rendering flags
(`is_final_render`, `render_mode`, `samples_per_pixel`, etc.).

### HashGrid
Hashed uniform grid with `cellStart` / `cellEnd` / `sortedIndices`
arrays. Hash function uses three large primes:
```
hash(cx,cy,cz) = (cx*73856093 ^ cy*19349663 ^ cz*83492791) % table_size
```

---

## 14. Rendering Equation Reference

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
contribution and $\hat{L}_{\text{photon}}$ is the photon density
estimate contribution at each diffuse hit along the path.
