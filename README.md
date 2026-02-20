# Spectral Photon + Path Tracing Renderer

A physically-based spectral renderer combining **photon mapping** and
**path tracing**, running entirely on the GPU via **NVIDIA OptiX** and
**CUDA**. Features **photon-guided importance sampling** with
Fibonacci-sphere directional bins for variance reduction.

![Render Result](output/render.png)

---

## Features

- **Full spectral transport** -- 32 wavelength bins (380--780 nm), no
  RGB shortcuts in light transport
- **GPU photon tracing** -- photon emission, scattering, and storage
  via OptiX raygen programs
- **GPU path tracing** -- multi-bounce path tracing with
  next-event estimation and photon density estimation
- **Photon directional bin cache** -- per-pixel Fibonacci-sphere
  binning of photon flux distribution (32 bins, 24 bytes each)
- **Guided BSDF bounce** -- first-bounce directions sampled
  proportional to cached photon flux (CDF + cone jitter)
- **Guided NEE** -- shadow rays steered toward lights matching the
  photon flux distribution (bin-flux-weighted emissive CDF)
- **Cached density estimation** -- first-sample photon gather cached
  per pixel, reused across subsequent SPP frames
- **Stratified sub-pixel sampling** -- 4x4 jittered strata for
  reduced clumping at 16 SPP
- **Interactive debug viewer** -- real-time first-hit rendering with
  normals, material ID, depth, and photon overlay modes
- **Hashed uniform grid** -- fast photon lookup with Epanechnikov
  kernel and surface consistency filtering
- **Visibility-weighted photon attenuation** -- NEE shadow-ray
  visibility modulates photon density, preserving contact shadows
- **GPU kernel profiling** -- per-pixel clock64() timers for
  ray trace, NEE, photon gather, and BSDF breakdown
- **Material system** -- Lambertian, mirror, glass, glossy
  definitions present; OptiX runtime focuses on Lambertian + mirror
- **Comprehensive test suite** -- 163 unit tests covering all core
  components including photon bins, guided bounce, and stratified
  sampling

### Component Outputs

Each full render produces:

| File                             | Contents                    |
|----------------------------------|-----------------------------|
| `output/render.png`              | Final combined render       |
| `output/out_nee_direct.png`      | NEE direct lighting only    |
| `output/out_photon_indirect.png` | Photon indirect only        |
| `output/out_combined.png`        | NEE + photon (spectral sum) |
| `output/out_debug_nee.png`       | Single-frame NEE preview    |

---

## Requirements

| Component            | Minimum Version                   |
|----------------------|-----------------------------------|
| **NVIDIA GPU**       | Turing architecture (sm_75) or newer |
| **CUDA Toolkit**     | 12.x                             |
| **NVIDIA OptiX SDK** | 7.x or 9.x                       |
| **CMake**            | 3.24                              |
| **C++ Standard**     | C++17                             |
| **OS**               | Windows 10+ (MSVC 2022) or Linux  |
| **VRAM**             | 12 GB recommended (bin cache ~604 MB at 1024x768) |

> **OptiX is mandatory.** There is no CPU fallback. The build will
> fail if `OptiX_INSTALL_DIR` is not set.

---

## Build

### 1. Set the OptiX SDK path

```bash
# Environment variable (recommended)
set OptiX_INSTALL_DIR=C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0

# Or pass directly to CMake
cmake -B build -DOptiX_INSTALL_DIR="C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0"
```

### 2. Configure and build

```bash
cmake -B build
cmake --build build --config Debug
```

### 3. Run

```bash
# Interactive debug viewer (default scene from config.h)
build\Debug\photon_tracer.exe

# Custom scene with options
build\Debug\photon_tracer.exe scenes/my_scene.obj --spp 64 --photons 1000000
```

### Quick script (Windows)

```bat
run.bat              # Build & run interactive viewer
run.bat test         # Build & run unit tests
run.bat release      # Build & run in Release mode
run.bat clean        # Delete build directory
```

---

## Usage

The renderer starts in an **interactive debug window**. Press keys to
switch visualisation modes and trigger the final render.

### Controls

#### Camera / window controls

| Input | Action |
|---|---|
| **W/A/S/D** | Move camera forward / left / back / right |
| **SPACE / Left Ctrl** | Move camera up / down |
| **Mouse move** | Look around (when mouse is captured) |
| **Left Shift** | Faster movement (3x speed) |
| **M** | Toggle mouse capture/release |
| **Left click** | Re-capture mouse when released |
| **ESC** or **Q** | If mouse captured: release it. If already released: quit |

#### Render controls

| Key | Action |
|---|---|
| **R** | Start full path tracing render (NEE debug PNG -> bin population -> progressive SPP -> PNG output) |
| **TAB** | Cycle render mode: `Full` -> `DirectOnly` -> `IndirectOnly` -> `PhotonMap` -> `Normals` -> `MaterialID` -> `Depth` |
| **H** | Toggle help overlay |

#### Debug toggles

| Key | Action |
|---|---|
| **F1** | Toggle photon points overlay |
| **F2** | Toggle global-map overlay |
| **F3** | Toggle caustic-map selector (separate caustic map not yet implemented) |
| **F4** | Toggle hash-grid debug flag |
| **F5** | Toggle photon-direction debug flag |
| **F6** | Toggle PDF debug flag |
| **F7** | Toggle gather-radius debug flag |
| **F8** | Toggle MIS-weight debug flag |
| **F9** | Toggle spectral coloring for photon overlay |

#### Hover-cell inspection

- Release mouse capture with **M**.
- Enable a map toggle (**F2** global or **F3** caustic selector).
- Move cursor over the image to view hover panel data:
  cell index, photon count, flux sum/avg, dominant wavelength.

### Command-Line Options

| Option         | Description                          | Default          |
|----------------|--------------------------------------|------------------|
| `--width W`    | Image width                          | 1024             |
| `--height H`   | Image height                         | 768              |
| `--spp N`      | Samples per pixel (final render)     | 16               |
| `--photons N`  | Number of photons                    | 1000000          |
| `--radius R`   | Photon gather radius                 | 0.05             |
| `--output FILE`| Output file path                     | output/render.png|
| `--mode MODE`  | Render mode: full, direct, indirect, photon, normals, material, depth | full |

---

## Rendering Pipeline

```
1. Scene Load (OBJ + MTL) -> normalise to [-0.5, 0.5]^3
2. Build BVH + emissive distribution
3. OptiX init -> build GAS -> upload scene + emitter data
4. GPU photon trace (1M photons) -> hash grid (CPU) -> upload
5. Interactive debug viewer (GLFW, 1 spp first-hit)
6. (R key) -> NEE debug PNG -> populate bin cache -> progressive render
7. PNG output (render + component decomposition + profiling)
```

### Photon Directional Bins

The key innovation: per-pixel Fibonacci-sphere bins cache the photon
flux distribution at each pixel's first diffuse hit. This enables:

- **Guided BSDF bounce (B1):** Sample continuation direction from
  bin flux CDF with cone jitter, steering paths toward bright regions.
- **Guided NEE (B2):** Re-weight emissive triangle CDF using bin
  flux alignment (alpha=5.0), steering shadow rays toward lights that
  actually contribute indirect illumination.
- **Cached density (A1):** Cache the first-sample photon density
  gather per pixel, eliminating redundant hash-grid lookups.
- **Stratified sampling (B3):** 4x4 sub-pixel strata reduce
  clumping at 16 SPP.

---

## Project Structure

```
src/
  main.cpp                     Entry point, GLFW loop
  core/                        Types, spectrum, config, RNG, photon bins
  bsdf/                        BSDF models (Lambertian, mirror, glass, GGX)
  scene/                       OBJ/MTL loader, scene structure, BVH
  renderer/                    CPU renderer, camera, MIS, path tracer kernels
  photon/                      Photon structures, hash grid, emitter, density
  optix/                       OptiX device programs and host pipeline
  debug/                       Debug visualisation state and key bindings
tests/
  test_main.cpp                163 unit tests (GoogleTest)
scenes/                        Test scenes (Cornell box, conference, etc.)
doc/
  architecture/                Detailed architecture documentation
  prompts/                     Design specification documents
```

---

## Scene Selection

Scenes are selected at compile time via `#define` in `src/core/config.h`:

| Scene             | Define               | Notes                          |
|-------------------|----------------------|--------------------------------|
| Cornell Box       | `SCENE_CORNELL_BOX`  | Reference frame (no normalise) |
| Conference Room   | `SCENE_CONFERENCE`   | Default scene                  |
| Living Room       | `SCENE_LIVING_ROOM`  | Complex indoor scene           |
| Breakfast Room    | `SCENE_BREAKFAST_ROOM` | Complex indoor scene         |
| Sibenik Cathedral | `SCENE_SIBENIK`      | Large architectural scene      |

All non-reference scenes are automatically normalised to the Cornell
Box coordinate frame at load time.

---

## Tests

The project includes 163 unit tests built with GoogleTest v1.14.0:

```bash
# Build and run tests
run.bat test

# Or manually
cmake --build build --config Debug
build\Debug\ppt_tests.exe
```

Test coverage includes:

- Vector math, ONB, coordinate transforms
- Spectral arithmetic, CIE colour matching, blackbody emission
- RNG distribution quality
- Cosine/uniform hemisphere sampling, triangle sampling
- MIS power heuristic (2-way and 3-way)
- Alias table construction and sampling
- Ray-triangle intersection (hit, miss, edge cases)
- AABB ray intersection
- Fresnel equations (Schlick, dielectric, TIR)
- GGX microfacet model (normalisation, Smith geometry, VNDF)
- BSDF evaluation, sampling, reciprocity, energy conservation
- Hash grid build/query, distance filtering
- Density estimator with surface consistency filtering
- Epanechnikov and box kernel evaluation
- Camera ray generation
- Cornell box scene loading and BVH validation
- Photon tracing, position bounds, flux validation
- Photon density on known geometry
- Fibonacci sphere coverage (N=8, 16, 32) and nearest-bin queries
- Hemisphere coverage and bin solid angle distribution
- Bin population and centroid normalisation
- Stratified sub-pixel coverage (4x4 strata)
- PhotonBin struct size validation (24 bytes)
- OptiX initialisation, accel build, scene upload
- OptiX debug frame rendering (non-zero output)
- OptiX normals mode visualisation
- OptiX final render validation
- OptiX framebuffer resize

---

## Architecture

See [doc/architecture/architecture.md](doc/architecture/architecture.md)
for a detailed description of the rendering pipeline, mathematical
foundations (rendering equation, Monte Carlo estimators, guided NEE,
density estimation), strengths, and limitations.

---

## License

See [LICENSE](LICENSE).
