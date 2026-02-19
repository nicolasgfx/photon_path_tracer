# Spectral Photon + Path Tracing Renderer

*(Private Research Renderer -- Focus on Physical Correctness &
Mathematical Clarity)*

------------------------------------------------------------------------

# 1. Physical Units & Definitions

All quantities are defined explicitly to avoid ambiguity.

-   Radiance: ( L(x,`\omega`{=tex},`\lambda`{=tex}) ) \[W / (sr·m²·nm)\]
-   Flux: ( `\Phi `{=tex}) \[W / nm\]
-   Irradiance: ( E(x,`\lambda`{=tex}) ) \[W / (m²·nm)\]

Each stored photon represents **radiant flux per wavelength bin**.

Photon flux is defined such that the density estimator approximates
irradiance.

------------------------------------------------------------------------

# 2. Tech Stack

-   **C++17**
-   **CUDA 12.x**
-   **NVIDIA OptiX 7+/9.x** -- mandatory for all rendering (debug
    viewer and final path tracing). There is **no CPU fallback**.
-   Structure-of-Arrays layout
-   Simple GPU radix sort
-   **GoogleTest v1.14.0** for unit testing
-   **GLFW + OpenGL** for display-only debug viewer
-   No scalability requirements --- clarity over optimization

------------------------------------------------------------------------

# 2b. Build Setup

## Required Environment

| Component            | Version        |
|----------------------|----------------|
| CMake                | >= 3.24        |
| CUDA Toolkit         | 12.x           |
| NVIDIA OptiX SDK     | 7.x or 9.x    |
| C++ Standard         | C++17          |
| GPU                  | sm_75+ (Turing or newer) |

## OptiX Configuration

OptiX is **mandatory**. The CMake build will `FATAL_ERROR` if
`OptiX_INSTALL_DIR` is not set. Set it as an environment variable or
pass it to CMake:

```bash
set OptiX_INSTALL_DIR=C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0
cmake -B build
cmake --build build --config Debug
```

PTX is compiled via a custom `nvcc` command to
`build/ptx/optix_device.ptx` (configuration-independent path).

## Dependencies (auto-fetched)

-   GLFW (FetchContent)
-   stb_image / stb_image_write (FetchContent)
-   GoogleTest v1.14.0 (FetchContent)

------------------------------------------------------------------------

# 3. High-Level Architecture

## 3.1 Photon Pass (Light → Scene)

-   Emit spectral photons
-   Store diffuse interactions
-   Separate:
    -   Global Photon Map
    -   Caustic Photon Map

## 3.2 Camera Pass (Path Tracing)

At each hit:

-   Direct light sampling (next-event estimation)
-   BSDF sampling
-   Photon density estimate
-   Photon-guided sampling (no SH/SG approximation)
-   Multiple importance sampling between all strategies

------------------------------------------------------------------------

# 4. Photon Emission

## 4.1 Emissive Triangle Sampling

Triangle selection weight:

\[ w_t = A_t `\cdot `{=tex}`\bar`{=tex}{L}\_{e,t} \]

CDF built over emissive triangles.

Uniform sampling on triangle:

\[ `\alpha `{=tex}= 1 - `\sqrt{u}`{=tex}, `\quad`{=tex} `\beta `{=tex}=
v`\sqrt{u}`{=tex}, `\quad`{=tex} `\gamma `{=tex}= 1 - `\alpha `{=tex}-
`\beta`{=tex} \]

\[ x = `\alpha `{=tex}v_0 + `\beta `{=tex}v_1 + `\gamma `{=tex}v_2 \]

------------------------------------------------------------------------

## 4.2 Spectral Sampling

Emission depends on:

\[ L_e(x,`\lambda`{=tex}) \]

Wavelength PDF:

\[ p(`\lambda`{=tex}\_i\|x) =
`\frac{L_e(x,\lambda_i)}{\sum_j L_e(x,\lambda_j)}`{=tex} \]

Photon flux:

\[ `\Phi `{=tex}= `\frac{L_e(x,\omega,\lambda)\cos\theta}`{=tex}
{p(t),p(x\|t),p(`\omega`{=tex}\|x),p(`\lambda`{=tex}\|x)} \]

------------------------------------------------------------------------

## 4.3 Photon Tracing

-   Trace via OptiX
-   Store only diffuse hits
-   Continue through specular/refractive surfaces
-   Russian roulette after minimum depth

RR probability:

\[ p\_{rr} = `\min`{=tex}(0.95,
`\max`{=tex}\_`\lambda `{=tex}T(`\lambda`{=tex})) \]

------------------------------------------------------------------------

# 5. Spatial Data Structure

## Hashed Uniform Grid

cellCoord = floor(position / cellSize)

-   cellSize ≈ searchRadius
-   Scan 3×3×3 neighbors
-   Simple fixed-size implementation (no overflow strategy needed for
    private project)

------------------------------------------------------------------------

# 6. Density Estimator

Diffuse radiance:

\[ L_o(x,`\omega`{=tex}\_o,`\lambda`{=tex}) `\approx`{=tex}
`\frac{1}{\pi r^2}`{=tex} `\sum`{=tex}\_i
`\Phi`{=tex}\_i(`\lambda`{=tex})
f_s(x,`\omega`{=tex}\_i,`\omega`{=tex}\_o,`\lambda`{=tex}) \]

Kernel can optionally include:

\[ W(\|\|x - x_i\|\|) \]

------------------------------------------------------------------------

# 7. Surface Consistency Filtering

To avoid cross-surface contamination:

Accept photon only if:

-   ( `\omega`{=tex}\_i `\cdot `{=tex}n_x \< 0 )
-   ( \|n_x `\cdot `{=tex}(x_i - x)\| \< `\tau `{=tex})
-   Optional: normal similarity test

------------------------------------------------------------------------

# 8. Path Tracing

## Throughput Update

\[ T\_{k+1}(`\lambda`{=tex}) = T_k(`\lambda`{=tex})
`\frac{f_s(x,\omega_i,\omega_o,\lambda)\cos\theta}`{=tex}
{p(`\omega`{=tex}\_i)} \]

Spectral throughput carried per wavelength bin.

------------------------------------------------------------------------

## Direct Lighting (Next Event Estimation)

Always performed.

MIS between:

-   Light sampling
-   BSDF sampling
-   Photon-guided sampling

------------------------------------------------------------------------

# 9. Photon-Guided Sampling (Exact)

Discrete proposal:

\[ q(`\omega`{=tex}\_i) `\propto `{=tex}`\Phi`{=tex}\_i \]

Optional radial kernel.

------------------------------------------------------------------------

## MIS (Power Heuristic)

\[ w = `\frac{(p_a)^2}`{=tex} {(p_a)\^2 + (p_b)\^2 + (p_c)\^2} \]

Supports 3-way MIS: - Light sampling - BSDF sampling - Photon-guided

------------------------------------------------------------------------

# 10. Radius Strategy

Fixed radius per photon map.

No progressive shrinking for simplicity.

Radius chosen experimentally to balance bias/variance.

------------------------------------------------------------------------

# 11. Caustic Handling

Separate caustic map:

-   Smaller radius
-   Combined additively with global map
-   Independent density estimate

------------------------------------------------------------------------

# 12. Debug & Visualization

## Key Bindings

  Key   Action
  ----- --------------------------
  F1    Toggle photon points
  F2    Toggle global map
  F3    Toggle caustic map
  F4    Toggle grid cells
  F5    Toggle photon directions
  F6    Toggle PDF values
  F7    Toggle radius sphere
  F8    Toggle MIS weights
  F9    Toggle spectral mode
  TAB   Cycle render modes

------------------------------------------------------------------------

## Hover Cell Overlay

On mouse hover:

Display:

-   Cell coordinate
-   Photon count
-   Total flux
-   Average flux
-   Dominant wavelength bin
-   Radius
-   Global/Caustic label

Optional small histograms:

-   λ distribution
-   Photon direction scatter

------------------------------------------------------------------------

## Photon Debug

-   Render photons as points
-   Color = wavelength
-   Brightness = flux

------------------------------------------------------------------------

## Direction Debug

At hitpoint:

-   Render arrows for each photon direction
-   Length proportional to flux

------------------------------------------------------------------------

## Sampling Debug Overlay

Display:

-   Selected direction
-   PDFs (light / bsdf / photon)
-   MIS weights
-   Final contribution
-   Path throughput

------------------------------------------------------------------------

# 13. UI

-   First-hit preview
-   Photon debug
-   Global-only
-   Caustic-only
-   Final full render

------------------------------------------------------------------------

# 14. Unit Test Coverage

Full coverage of all modules is required. Every core component must
have unit tests exercising its public interface and edge cases.

## Required Coverage Areas

-   **Math utilities** -- vector ops, ONB, coordinate transforms
-   **Spectral framework** -- arithmetic, CIE matching, blackbody
-   **Random number generation** -- distribution quality
-   **Sampling** -- cosine/uniform hemisphere, triangle, alias table
-   **MIS** -- power heuristic (2-way and 3-way)
-   **Geometry** -- ray-triangle, AABB intersection
-   **Fresnel** -- Schlick, dielectric, TIR
-   **GGX microfacet** -- normalisation, Smith G, VNDF sampling
-   **BSDF** -- evaluation, sampling, reciprocity, energy conservation
-   **Hash grid** -- build, query, distance filtering
-   **Density estimator** -- kernel evaluation, surface consistency
-   **Camera** -- ray generation
-   **Scene loading** -- OBJ/MTL parsing, BVH construction
-   **Photon tracing** -- position bounds, flux validation
-   **OptiX pipeline** -- init, accel build, scene upload, debug frame,
    normals mode, final render, framebuffer resize

Tests use GoogleTest v1.14.0. Run via `run.bat test` or
`build\Debug\ppt_tests.exe`.

------------------------------------------------------------------------

# 15. Summary

Private research renderer emphasizing:

-   Spectral correctness
-   Explicit mathematical estimators
-   Unbiased MIS
-   Exact photon usage
-   Clear debugging tools
-   Simple implementation
-   Full unit test coverage
