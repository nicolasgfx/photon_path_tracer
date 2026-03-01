<p align="center">
  <img src="doc/gallery/render.png" alt="Spectral Photon-Centric Renderer" width="100%"/>
</p>

<h1 align="center">Spectral Photon-Centric Renderer</h1>

<p align="center">
  A physically-based GPU renderer built around a <b>photon-centric</b> architecture
  with <b>full spectral light transport</b> — 32 wavelength bins, no RGB
  approximations anywhere in the path.
</p>

<p align="center">
  Built on <b>NVIDIA OptiX 9</b> and <b>CUDA 12</b>.
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#pipeline">Pipeline</a> •
  <a href="#build">Build</a> •
  <a href="#usage">Usage</a> •
  <a href="#interactive-viewer">Viewer</a> •
  <a href="#tests">Tests</a> •
  <a href="#license">License</a>
</p>

---

## Features

| Property | Detail |
|---|---|
| **Light transport** | 32 wavelength bins (380–780 nm), full spectral throughout |
| **Architecture** | Photon-centric — photon rays carry all indirect GI |
| **Camera ray role** | First diffuse hit only (specular chain ≤ 12 bounces) |
| **Direct lighting** | NEE with coverage-aware stratified area sampling |
| **Indirect lighting** | Photon density estimation, tangential disk kernel |
| **Spatial index** | KD-tree (CPU reference), hash grid (GPU primary) |
| **Chromatic dispersion** | Cauchy equation: n(λ) = A + B/λ², per-bin Fresnel |
| **Glass colour** | Spectral transmittance filter (Tf) + direct per-bin override (`pb_tf_spectrum`) |
| **Gather kernel** | Tangential disk — 2D surface distance, eliminates cross-surface leakage |
| **Adaptive gather** | k-NN per hitpoint from CellInfoCache photon density |
| **Adaptive caustics** | Targeted two-point emission toward specular geometry |
| **Guided path tracing** | Per-cell Fibonacci-sphere directional histograms, MIS-combined with BSDF |
| **Glossy continuation** | Multi-bounce glossy reflections with guided + BSDF mixture sampling |
| **Adjoint correction** | η² transport-mode tagging at refractive interfaces (photon vs camera) |
| **Progressive mode** | SPPM (Hachisuka & Jensen 2009) |
| **Hero wavelengths** | PBRT v4-style, 4 stratified wavelengths per photon |
| **Tone mapping** | ACES Filmic |
| **Sub-pixel sampling** | Stratified jittered (16 SPP default) |
| **Adaptive sampling** | Screen-noise adaptive with per-pixel convergence mask |
| **Photon map pool** | Pre-built maps cycled during accumulation (no re-trace) |
| **CPU reference** | Physically identical dual implementation, PSNR-tested |
| **Denoiser** | OptiX AI denoiser with albedo + normal guide layers |
| **Tests** | ~340 unit + integration tests (GoogleTest) |

### What Makes This Different

Most renderers work camera-first: each pixel fires a ray, bounces through
the scene, and accumulates energy. This renderer **inverts that
relationship**:

- **The photon pass is the path tracer.** Photon rays start from lights,
  bounce with full BSDF importance sampling and Russian roulette, and
  deposit spectral flux packets at diffuse surfaces.
- **Camera rays are cheap probes.** They find the first visible surface
  and query the precomputed photon map — no further bouncing needed.
- **Photon-guided glossy continuation.** At glossy surfaces, camera
  bounces are steered by per-cell directional histograms built from
  photon incident directions — an O(1) cell lookup replaces expensive
  per-bounce kNN queries.
- **The photon map is view-independent.** Compute once, save to disk,
  reload for interactive camera exploration.
- **The tangential disk kernel** replaces 3D spherical gather with a 2D
  surface-projected metric, eliminating planar blocking artifacts
  regardless of the spatial data structure.

---

## Pipeline

```
STARTUP
  ┌───────────────────────────────────────────────────────────────┐
  │  Load OBJ / MTL scene                                        │
  │  Normalise geometry to [-0.5, 0.5]³                          │
  │  Build CPU BVH + OptiX GAS                                   │
  │  Build emitter distribution (alias table over emissive tris)  │
  │  Upload geometry, materials, emitters to GPU                  │
  └───────────────────────────────────────────────────────────────┘
                              │
                              ▼
PHOTON PASS  ── run once, reuse across camera views ──────────────
  ┌───────────────────────────────────────────────────────────────┐
  │  Emit N photons from lights (power-proportional sampling)     │
  │  Per photon (4 hero wavelengths):                             │
  │    Bounce 0: cosine-weighted hemisphere from emitter          │
  │    Bounce 1+: BSDF importance sampling + Russian roulette     │
  │    Glass: Cauchy dispersion, Tf filter, IOR stack, path flags │
  │    Deposit flux at each diffuse hit (lightPathDepth ≥ 2)      │
  │    Separate global map (diffuse) vs caustic map (specular)    │
  │  Build spatial index: KD-tree (CPU) / hash grid (GPU)         │
  │  Build CellInfoCache — per-cell density, variance, caustics   │
  │  Targeted caustic emission toward specular geometry           │
  │  Build directional histograms (CellBinGrid, 32 Fibonacci     │
  │    bins per cell) for photon-guided path tracing              │
  │  Optional: save photon map to binary cache                    │
  └───────────────────────────────────────────────────────────────┘
                              │
                              ▼
CAMERA PASS  ── per frame ────────────────────────────────────────
  ┌───────────────────────────────────────────────────────────────┐
  │  For each pixel (stratified jittered sub-pixel samples):      │
  │    Trace camera ray → follow specular chain (≤ 12 bounces)    │
  │    At first diffuse hit:                                      │
  │      NEE: shadow rays to sampled emitter points               │
  │      Gather: query photon map (tangential disk, k-NN)         │
  │      L = L_emission + L_direct(NEE) + L_indirect(photon)     │
  │    Glossy continuation (≤ 2 bounces):                         │
  │      MIS mixture sampling: 50% BSDF + 50% guided direction   │
  │      Guided direction sampled from per-cell photon histogram  │
  │      NEE + photon gather at each glossy hit                   │
  └───────────────────────────────────────────────────────────────┘
                              │
                              ▼
OUTPUT
  ┌───────────────────────────────────────────────────────────────┐
  │  OptiX AI Denoiser (optional, albedo + normal guide layers)   │
  │  Spectral → CIE XYZ → linear sRGB → ACES Filmic → gamma     │
  │  Write timestamped PNGs (combined, direct, indirect)          │
  └───────────────────────────────────────────────────────────────┘
```

---

## Requirements

| Component | Minimum | Notes |
|---|---|---|
| NVIDIA GPU | Turing (sm_75) or newer | Required for OptiX |
| CUDA Toolkit | 12.x | |
| NVIDIA OptiX SDK | 9.x | `OptiX_INSTALL_DIR` env var must be set |
| CMake | 3.24+ | |
| C++ Standard | C++17 | |
| MSVC | Visual Studio 2022 | Community / Professional / Enterprise |
| OS | Windows 10+ | |
| VRAM | 8 GB recommended | |

---

## Build

### 1. Set OptiX SDK path

```bat
set OptiX_INSTALL_DIR=C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0
```

### 2. Build

```bat
run.bat build          :: Configure + Release build (Ninja, auto-detects MSVC)
```

Or manually:

```bat
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### 3. Run

```bat
run.bat                :: Build + launch viewer (default scene)
run.bat -- --spp 64 --photons 4000000
```

### Build Scripts

| Command | Action |
|---|---|
| `run.bat` | Build + run (pass extra args after `--`) |
| `run.bat build` | Build only |
| `run.bat test` | Build + run fast tests |
| `run.bat test-all` | Build + run full test suite |
| `run.bat clean` | Delete build directory |
| `build.bat` | Release build (`photon_tracer` only) |
| `build.bat test` | Release build with test target |
| `build.bat rebuild` | Clean rebuild |

Build artifacts: `build/photon_tracer.exe`, `build/ppt_tests.exe`,
`build/ptx/optix_device.ptx`.

---

## Usage

### Command-Line Options

| Option | Default | Description |
|---|---|---|
| `--width W` | 1024 | Image width (pixels) |
| `--height H` | 768 | Image height (pixels) |
| `--spp N` | 16 | Samples per pixel |
| `--photons N` | 1 000 000 | Photon count |
| `--global-photons N` | 1 000 000 | Global photon budget |
| `--caustic-photons N` | 1 000 000 | Caustic photon budget |
| `--radius R` | 0.05 | Photon gather radius (scene units) |
| `--output FILE` | `output/render.png` | Output PNG path |
| `--mode MODE` | `combined` | `combined` · `direct` · `indirect` · `photon` · `normals` · `material` · `depth` |
| `--spatial MODE` | `kdtree` | `kdtree` · `hashgrid` |
| `--adaptive-radius` | off | Enable k-NN adaptive gather radius |
| `--knn-k N` | 100 | k-NN neighbour count |
| `--max-specular-chain N` | 12 | Camera specular bounce limit |
| `--sppm` | off | Enable SPPM progressive mode |
| `--sppm-iterations N` | 64 | SPPM iteration count |
| `--sppm-radius R` | 0.1 | SPPM initial radius |

### Output Files

Each render writes timestamped files to `output/`:

| File | Contents |
|---|---|
| `render_YYYYMMDD_HHMMSS.png` | Final combined render |
| `render_YYYYMMDD_HHMMSS_nee_direct.png` | Direct lighting only |
| `render_YYYYMMDD_HHMMSS_photon_indirect.png` | Photon indirect only |
| `out_debug_nee.png` | Quick NEE preview at render start |

SPPM mode writes per-iteration:
`output/sppm_…_iter0001.png` … `_final.png`.

---

## Interactive Viewer

The executable opens a real-time GLFW window at 1 spp/frame. Use it to
position the camera and inspect the scene before committing to a full render.

### Camera

| Input | Action |
|---|---|
| **W / A / S / D** | Move forward / left / back / right (polled) |
| **Space** | Move up |
| **Left Ctrl** | Move down |
| **Left Shift** | 3× speed (hold) |
| **Mouse** | Look around (when captured) |
| **M** | Toggle mouse capture |
| **Left click** | Re-capture mouse |

### Rendering

| Key | Action |
|---|---|
| **R** | Save timestamped snapshot (PNG + JSON stats + analysis report) |
| **P** | Re-trace photons + rebuild spatial index |
| **TAB** | Cycle render mode: Full → Direct → Indirect → Photon → Normals → Material → Depth → GuideMap → CausticOnly |
| **ESC** | Release mouse → quit (2-tier) |
| **Q** | Release mouse → quit |

### Statistics & Guidance

| Key | Action |
|---|---|
| **T** | Toggle guided / unguided path tracing (sets `guide_fraction` to 0 or default) |
| **C** | Toggle histogram-only conclusions (only when guided is ON) |
| **S** | Toggle live statistics overlay (top-right corner) |

### Depth of Field

| Key | Action |
|---|---|
| **O** | Toggle DoF on / off |
| **[** | Widen aperture (lower f-number, more blur, ⅓ stop per press) |
| **]** | Narrow aperture (higher f-number, less blur, ⅓ stop per press) |
| **,** | Focus nearer (−10%) |
| **.** | Focus farther (+10%) |
| **F** | Auto-focus on screen centre |
| **Middle mouse** | Auto-focus on cursor position |

### Light & Scene

| Key | Action |
|---|---|
| **+ / =** | Increase light brightness (×1.25 step) |
| **− / _** | Decrease light brightness (÷1.25 step) |
| **1 – 8** | Switch scene (see [Scenes](#scenes)) |
| **V** | Toggle participating medium *(state tracked; volume currently disabled)* |
| **G** | Toggle dense cell-bin grid gather |

### Debug Overlays

| Key | Overlay |
|---|---|
| **F1** | Photon point visualisation (all photons, disables F2 filter) |
| **F2** | Cycle photon filter: Off → All → TraversedGlass → CausticGlass → Volume → Dispersion → CausticSpecular → Off |
| **F3** | Toggle Guide Map visualisation (photon-guided direction overlay) |
| **F4** | Hash grid cell visualisation *(planned)* |
| **F5** | Photon direction arrows *(planned)* |
| **F6** | PDF visualisation *(planned)* |
| **F7** | Gather radius sphere *(planned)* |
| **F8** | MIS weight display *(planned)* |
| **F9** | Spectral wavelength colouring (photon overlay) |
| **F10** | Save camera position to active scene folder (`saved_camera.json`) |
| **F11** | Photon heatmap (per-triangle irradiance, GPU false-colour) |
| **H** | Toggle help overlay |

**Hover-cell inspector** (F2 active, mouse released): shows cell
coordinate, photon count, flux statistics, dominant wavelength, and gather
radius for the cell under the cursor.

---

## Scenes

Eight bundled scenes, switchable via hotkeys **1–8**:

| Key | Scene | Complexity | Description |
|---|---|---|---|
| **1** | Cornell Box | Low | Classic Cornell Box with diamond |
| **2** | Cornell Sphere | Low | Mirror sphere + deep-green glass sphere |
| **3** | Cornell Mirror | Low | Tall mirror box |
| **4** | Cornell Water | Low | Refractive water surface |
| **5** | Living Room | Medium | Furnished interior (Jay-Artist) |
| **6** | Conference Room | Medium | Measured room (Grynberg & Ward) |
| **7** | Salle de Bain | Medium | Bathroom scene (nacimus) |
| **8** | Mori Knob | Medium | Reflective ornamental knob |

Additional scenes (not bound to hotkeys): Interior, Sibenik Cathedral,
Sponza, Fireplace Room, Hairball.

Scene files live in `scenes/<name>/`. All scenes use emissive triangles
defined in their MTL files (`Ke` or `pb_brdf emissive`).

---

## Architecture

Full architecture documentation:
[doc/architecture/architecture.md](doc/architecture/architecture.md)

### Key Design Decisions

| Decision | Detail |
|---|---|
| **Photon-centric** | Photon rays carry all indirect transport; camera rays stop at first diffuse hit |
| **Full spectral** | 32 wavelength bins, no RGB anywhere in transport; hero wavelength system (4 bins/photon) |
| **Guided path tracing** | Per-cell Fibonacci-sphere directional histograms from photon wi vectors; O(1) lookup, MIS-combined with BSDF |
| **Tangential disk kernel** | 2D surface-distance gather; eliminates cross-surface leakage and planar blocking |
| **Adjoint-correct transport** | TransportMode tag (Radiance/Importance) with η² correction at refractive interfaces |
| **Dual CPU/GPU** | CPU reference for ground truth; GPU (OptiX + CUDA) for speed; integration tests verify parity |
| **Photon map persistence** | Binary save/load with scene hash — compute once, explore interactively |
| **Photon map pool** | Pre-built maps cycled during accumulation; no re-tracing during render |
| **Direct spectral Tf** | `pb_tf_spectrum` bypasses RGB→spectrum for precise glass colour control |

### Project Layout

```
src/
  main.cpp                       Entry point, CLI, GLFW loop
  app/
    viewer.h / .cpp              GLFW window, event loop, overlays
  core/
    config.h                     Tunable constants, scene profiles
    types.h                      Vec3, Ray, HitRecord, ONB
    spectrum.h                   Spectral arithmetic, CIE XYZ, blackbody
    random.h                     PCG RNG
    sppm.h                       SPPM progressive update & reconstruction
    alias_table.h / cdf.h        O(1) alias sampling, CDF build
    cell_cache.h                 CellInfoCache — per-cell statistics
    nee_sampling.h               Coverage-aware NEE helpers
    font_overlay.h               Debug text rendering
  bsdf/
    bsdf.h                       Lambertian, mirror, glass, GGX VNDF
  scene/
    scene.h                      Scene graph, emitter list, BVH
    material.h                   Material types + pb_* extensions
    triangle.h                   Triangle primitives
    obj_loader.h / .cpp          Wavefront OBJ + MTL parser
  renderer/
    renderer.h / .cpp            CPU reference renderer
    camera.h                     Perspective camera, thin-lens DoF
    direct_light.h / .cu         Direct lighting kernel
  photon/
    photon.h                     Photon, PhotonSoA (structure-of-arrays)
    kd_tree.h                    KD-tree build + k-NN (CPU reference)
    hash_grid.h / .cu            Hash grid build + query (GPU primary)
    emitter.h / .cu              Emitter sampling, alias table
    density_estimator.h          Tangential disk kernel
    surface_filter.h             Surface consistency filter
    cell_bin_grid.h              Per-cell directional histograms (Fibonacci sphere)
    specular_target.h            Targeted caustic emission (two-point sampling)
    photon_io.h / .cpp           Binary photon map save/load
  optix/
    optix_renderer.h / .cpp      Host pipeline: SBT, GAS, launch params
    optix_device.cu              OptiX raygen / closesthit programs
    optix_guided.cuh             Photon-guided direction sampling (§4 MIS)
    launch_params.h              Shared GPU/CPU launch parameter struct
    adaptive_sampling.h / .cu    Per-pixel noise metric + convergence mask
  debug/
    debug.h                      Visualisation state, key bindings
tests/                           ~340 GoogleTest unit + integration tests
scenes/                          OBJ / MTL scene files
doc/
    architecture/                Full architecture document
    gallery/                     Render gallery images
```

---

## Tests

Unit and integration tests via GoogleTest 1.14.0.

```bat
run.bat test              :: Fast tests (skip integration, speed, GPU parity)
run.bat test-all          :: Full suite (~340 tests)
```

Or manually:

```bat
build.bat test
build\ppt_tests.exe --gtest_filter="KDTree*"
```

### Test Coverage

| Area | What is tested |
|---|---|
| Math & geometry | Vector ops, ONB, ray–triangle, AABB |
| Spectral | Arithmetic, CIE XYZ colour matching, blackbody |
| Sampling | PCG RNG, hemisphere sampling, alias table, NEE |
| BSDF | Fresnel, GGX VNDF, reciprocity, energy conservation |
| KD-tree | Build, range query, k-NN, boundary cases |
| Tangential kernel | Distance matches analytic for coplanar geometry |
| Surface filter | Cross-wall rejection, same-surface acceptance |
| Hash grid | Build, query, Epanechnikov kernel, surface consistency |
| Photon flags | Path flag tagging, glass/caustic/dispersion |
| CellInfoCache | Build, query, adaptive radius, hotspot detection |
| Dispersion | Cauchy IOR, per-bin Fresnel, spectral splitting |
| IOR stack | Push/pop/overflow, nested dielectric tracking |
| SPPM | Progressive convergence, radius shrinkage |
| CPU↔GPU parity | Direct PSNR > 40 dB; indirect PSNR > 30 dB; energy ≤ 5% |
| OptiX integration | Init, GAS build, scene upload, debug frame, resize |

---

## Third-Party Scene Credits

All test scenes were obtained from
[Morgan McGuire's Computer Graphics Archive](https://casual-effects.com/data)
and are subject to their respective licenses.

| Scene | License | Copyright |
|---|---|---|
| Cornell Box | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | © 2009 Morgan McGuire |
| Conference Room | Credit required (custom) | © Anat Grynberg & Greg Ward |
| Interior | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | — |
| Mori Knob | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | © Yasutoshi "Mirage" Mori |
| Living Room | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | © 2012 Jay |
| Salle de Bain | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | © Nacimus Ait Cherif |
| Sibenik Cathedral | [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/) | © 2002 Marko Dabrovic |
| Sponza (Crytek) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | © 2010 Frank Meinl, Crytek |

> **Note:** The Sibenik Cathedral scene is licensed **CC BY-NC** and may only
> be used for non-commercial purposes.

Full per-model attributions in [CREDITS.md](CREDITS.md).

---

## License

MIT — see [LICENSE](LICENSE).

Copyright © 2026 nicolasgfx
