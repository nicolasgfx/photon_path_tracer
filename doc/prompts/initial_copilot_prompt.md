# Spectral Photon + Path Tracing Renderer

## Copilot-Optimized Implementation Specification

*(Private Research Renderer -- Physically Correct, Simple, Explicit)*

This document is structured for AI-assisted implementation. All modules
are decomposed into concrete components, data structures, and required
functions.

------------------------------------------------------------------------

# 1. Global Design Goals

-   Spectral correctness (30--60 wavelength bins)
-   Explicit physical units
-   Unbiased Monte Carlo estimators
-   **NVIDIA OptiX is mandatory** for all rendering -- debug viewer
    and final path tracing. There is **no CPU fallback**.
-   No scalability optimization required
-   Prefer clarity over micro-optimizations
-   Deterministic, debuggable structure
-   Full unit test coverage for every module

------------------------------------------------------------------------

# 1b. Build Setup

## Required Environment

| Component            | Version        |
|----------------------|----------------|
| CMake                | >= 3.24        |
| CUDA Toolkit         | 12.x           |
| NVIDIA OptiX SDK     | 7.x or 9.x    |
| C++ Standard         | C++17          |
| GPU                  | sm_75+ (Turing or newer) |

## OptiX Configuration

OptiX is **mandatory**. CMake will `FATAL_ERROR` if `OptiX_INSTALL_DIR`
is not set. Set the environment variable or pass to CMake:

```bash
set OptiX_INSTALL_DIR=C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0
cmake -B build
cmake --build build --config Debug
```

PTX is compiled via a custom `nvcc` command producing
`build/ptx/optix_device.ptx` (configuration-independent path).

## Dependencies (auto-fetched via FetchContent)

-   GLFW -- display-only debug viewer backend
-   stb_image / stb_image_write -- image I/O
-   GoogleTest v1.14.0 -- unit testing framework

Copilot must never introduce a CPU rendering fallback. All ray tracing
(debug and production) goes through OptiX.

------------------------------------------------------------------------

# 2. Core Data Structures

## 2.1 Spectral Types

``` cpp
constexpr int NUM_LAMBDA = 32; // configurable

struct Spectrum {
    float value[NUM_LAMBDA];
};
```

## 2.2 Photon

``` cpp
struct Photon {
    float3 position;
    float3 wi;             // incoming direction
    uint16_t lambda_bin;
    float flux;            // radiant flux for that bin
};
```

Stored in Structure-of-Arrays layout for GPU.

## 2.3 Spatial Hash Grid

``` cpp
struct HashGrid {
    float cellSize;
    thrust::device_vector<uint32_t> cellStart;
    thrust::device_vector<uint32_t> cellEnd;
    thrust::device_vector<uint32_t> sortedIndices;
};
```

------------------------------------------------------------------------

# 3. Rendering Pipeline Overview

    Build Scene
    Build Emissive Triangle Distribution
    Photon Pass
    Build Photon Hash Grid
    Camera Path Tracing Pass
    Debug Rendering

------------------------------------------------------------------------

# 4. Photon Pass -- Detailed Steps

## 4.1 Build Emissive Triangle Distribution

For each emissive triangle:

\[ w_t = A_t `\cdot `{=tex}`\bar`{=tex}{L}\_{e,t} \]

Implementation:

``` cpp
float computeTriangleWeight(const Triangle& t);
AliasTable buildAliasTable(const std::vector<float>& weights);
```

## 4.2 Sample Emission Position

Uniform barycentric sampling:

``` cpp
float u = rand();
float v = rand();
float alpha = 1 - sqrt(u);
float beta  = v * sqrt(u);
float gamma = 1 - alpha - beta;
float3 x = alpha*v0 + beta*v1 + gamma*v2;
```

## 4.3 Sample Wavelength

``` cpp
uint16_t sampleLambdaBin(const Spectrum& Le);
```

PDF:

\[ p(`\lambda`{=tex}\_i\|x) =
`\frac{L_e(x,\lambda_i)}{\sum_j L_e(x,\lambda_j)}`{=tex} \]

## 4.4 Photon Flux Definition

\[ `\Phi `{=tex}= `\frac{L_e(x,\omega,\lambda)\cos\theta}`{=tex}
{p(t),p(x\|t),p(`\omega`{=tex}\|x),p(`\lambda`{=tex}\|x)} \]

Copilot must implement PDF tracking explicitly.

## 4.5 Photon Tracing

-   Use OptiX for intersection
-   Continue through specular/refractive
-   Store only diffuse hits

Russian Roulette:

\[ p\_{rr} = `\min`{=tex}(0.95,
`\max`{=tex}\_`\lambda `{=tex}T(`\lambda`{=tex})) \]

------------------------------------------------------------------------

# 5. Spatial Hash Grid Implementation

For each photon:

``` cpp
int3 cellCoord = floor(position / cellSize);
uint32_t key = hash(cellCoord);
```

Pipeline:

1.  Compute keys
2.  Radix sort
3.  Prefix sum to build cellStart/cellEnd

Neighbor lookup scans 3×3×3 cells.

------------------------------------------------------------------------

# 6. Density Estimator

Diffuse estimate:

\[ L_o(x,`\omega`{=tex}\_o,`\lambda`{=tex}) = `\frac{1}{\pi r^2}`{=tex}
`\sum`{=tex}\_i `\Phi`{=tex}\_i(`\lambda`{=tex})
f_s(x,`\omega`{=tex}\_i,`\omega`{=tex}\_o,`\lambda`{=tex}) \]

Optional radial kernel:

\[ W(\|\|x-x_i\|\|) \]

Surface consistency filter:

-   dot(wi, n_x) \< 0
-   \|dot(n_x, x_i - x)\| \< tau

------------------------------------------------------------------------

# 7. Path Tracing Core

## 7.1 Throughput Update

``` cpp
T *= (bsdf * cosTheta) / pdf;
```

Per wavelength bin.

## 7.2 Direct Light Sampling

Always implemented.

## 7.3 Photon-Guided Sampling

Discrete proposal:

\[ q(`\omega`{=tex}\_i) `\propto `{=tex}`\Phi`{=tex}\_i \]

Implementation:

``` cpp
int samplePhotonIndex(const std::vector<Photon>& neighbors);
```

------------------------------------------------------------------------

# 8. Multiple Importance Sampling

3-way MIS:

-   Light sampling
-   BSDF sampling
-   Photon sampling

Power heuristic:

\[ w = `\frac{p_a^2}`{=tex} {p_a\^2 + p_b\^2 + p_c\^2} \]

Copilot must compute all PDFs explicitly.

------------------------------------------------------------------------

# 9. Caustic Map

Separate photon list and grid.

-   Smaller radius
-   Added to final radiance independently

------------------------------------------------------------------------

# 10. Debug System -- Required Features

## Key Bindings

  Key   Function
  ----- --------------------------
  F1    Toggle photon points
  F2    Toggle global map
  F3    Toggle caustic map
  F4    Toggle hash grid
  F5    Toggle photon directions
  F6    Show PDFs
  F7    Show radius sphere
  F8    Show MIS weights
  F9    Spectral coloring
  TAB   Cycle render modes

## Hover Cell Overlay

When mouse intersects grid cell:

Display:

-   Cell index
-   Photon count
-   Sum flux
-   Average flux
-   Dominant lambda bin
-   Map type

Optional mini histograms.

------------------------------------------------------------------------

# 11. Implementation Order (Copilot Guidance)

1.  Scene loading
2.  BSDF implementation
3.  Spectral representation
4.  Photon emission (no grid yet)
5.  Hash grid build
6.  Density estimator
7.  Direct light sampling
8.  MIS integration
9.  Debug visualization
10. Caustic map separation

------------------------------------------------------------------------

# 12. Acceptance Criteria

Renderer is considered correct if:

-   Energy conserved in diffuse-only scene
-   White Lambertian Cornell Box matches reference
-   Caustics visible through glass
-   Spectral light produces colored indirect bounce
-   Debug overlays consistent with photon distribution

------------------------------------------------------------------------

# 13. Unit Test Coverage

Full unit test coverage for all modules is mandatory. Copilot must
write or maintain tests for every core component.

## Required Coverage

-   **Math** -- vector ops, ONB, coordinate transforms
-   **Spectral** -- arithmetic, CIE colour matching, blackbody
-   **RNG** -- distribution quality checks
-   **Sampling** -- cosine/uniform hemisphere, triangle, alias table
-   **MIS** -- power heuristic (2-way, 3-way)
-   **Geometry** -- ray-triangle, AABB intersection
-   **Fresnel** -- Schlick, dielectric, TIR
-   **GGX** -- normalisation, Smith geometry, VNDF sampling
-   **BSDF** -- evaluate, sample, reciprocity, energy conservation
-   **Hash grid** -- build, query, distance filtering
-   **Density estimator** -- kernel evaluation, surface consistency
-   **Camera** -- ray generation
-   **Scene** -- OBJ/MTL load, BVH build, material parse
-   **Photon trace** -- position bounds, flux validation
-   **OptiX** -- init, accel build, upload, debug frame, normals,
    final render, framebuffer resize

Framework: GoogleTest v1.14.0. Run: `run.bat test`.

## Acceptance

All tests must pass (`152/152` or more) before any feature is
considered complete.

------------------------------------------------------------------------

# 14. Summary

This specification is structured for AI code generation:

-   Explicit modules
-   Explicit math
-   Explicit data structures
-   Minimal abstraction
-   Clear implementation sequence
-   Full unit test coverage

Focus: correctness, clarity, debuggability.
