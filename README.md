# Spectral Photon + Path Tracing Renderer

> A physically-based GPU renderer combining **photon mapping** and **path tracing**
> over a full spectral representation — no RGB shortcuts in the light transport.

Built on **NVIDIA OptiX** and **CUDA**. The central innovation is a per-pixel
**Fibonacci-sphere directional bin cache** that guides both BSDF bounce sampling
and next-event estimation, measurably reducing variance in indirect illumination.

![Render Result](output/render.png)

---

## At a Glance

| What                        | How                                                             |
|-----------------------------|-----------------------------------------------------------------|
| Light transport             | Full spectral — 32 wavelength bins, 380–780 nm                 |
| Photon tracing              | OptiX raygen program, 1 M photons, cosine-weighted emission     |
| Path tracing                | Multi-bounce NEE + photon density estimation, Russian roulette  |
| Variance reduction          | Fibonacci-sphere bin cache guides BSDF bounce and NEE           |
| Photon lookup               | Hashed uniform grid, Epanechnikov kernel, plane-distance filter |
| Sub-pixel sampling          | 4×4 stratified jittered strata (16 SPP default)                 |
| Debug / preview             | Interactive GLFW viewer, 7 render modes, 9 overlay toggles      |
| Test coverage               | 163 GoogleTest unit tests across all core components            |

---

## How It Works

### Rendering Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  STARTUP                                                        │
│                                                                 │
│  OBJ/MTL scene ──► Normalise to [-0.5,0.5]³                    │
│       │                                                         │
│       ▼                                                         │
│  Build BVH (GAS) + emitter CDF                                  │
│       │                                                         │
│       ▼                                                         │
│  OptiX pipeline init ──► upload geometry, materials, emitters   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  PHOTON PASS  (GPU → CPU → GPU)                                 │
│                                                                 │
│  OptiX raygen: emit 1M photons ──► scatter / store at diffuse   │
│       │                                                         │
│       ▼                                                         │
│  Download positions + flux ──► build hashed uniform grid (CPU)  │
│       │                                                         │
│       ▼                                                         │
│  Upload grid to device                                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  INTERACTIVE VIEWER  (GLFW loop, 1 spp/frame)                   │
│                                                                 │
│  first-hit + NEE (direct lighting only)                         │
│  TAB: cycle render modes   F1-F9: toggle overlays               │
│                                                                 │
│  Press R ──────────────────────────────────────────────────────►│
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  FULL RENDER  (R key)                                           │
│                                                                 │
│  1. NEE debug PNG (single frame, all shadow rays)               │
│  2. Bin population pass ──► per-pixel Fibonacci-sphere bins     │
│  3. Progressive SPP loop:                                       │
│       camera ray                                                │
│       ├─ specular bounce (up to max depth)                      │
│       └─ diffuse hit ──► guided NEE  +  photon density est.     │
│              │                    │                             │
│              │              [frame 0] cache density → reuse     │
│              ▼                                                  │
│         accumulate spectral radiance                            │
│  4. PNG output  (render + NEE direct + photon indirect +        │
│                  combined + per-pixel profiling data)           │
└─────────────────────────────────────────────────────────────────┘
```

### Photon Bin Cache — The Key Optimisation

After the photon pass, a **per-pixel cache** is built by tracing center rays
to their first diffuse hit and gathering nearby photons into 32 Fibonacci-sphere
directional bins (24 bytes each, ~604 MB at 1024×768).

```
  Per-pixel hemisphere
  ┌──────────────────────────────────┐
  │  Fibonacci sphere (N=32 bins)    │
  │                                  │
  │   · · ·  ┌─── bin k ───┐ · · ·  │
  │          │ flux  float  │        │
  │          │ dir_x float  │        │
  │          │ dir_y float  │        │
  │          │ dir_z float  │        │
  │          │ weight float │        │
  │          │ count int    │        │
  │          └──────────────┘        │
  └──────────────────────────────────┘
```

---

## Requirements

| Component       | Minimum                              | Notes                              |
|-----------------|--------------------------------------|------------------------------------|
| NVIDIA GPU      | Turing (sm_75) or newer             | Required — no CPU fallback         |
| CUDA Toolkit    | 12.x                                |                                    |
| NVIDIA OptiX    | 7.x or 9.x                          | `OptiX_INSTALL_DIR` must be set    |
| CMake           | 3.24                                |                                    |
| C++ Standard    | C++17                               |                                    |
| OS              | Windows 10+ (MSVC 2022)             |                                    |
| VRAM            | 12 GB recommended                   | Bin cache alone is ~604 MB         |

> **OptiX is mandatory.** The build will fail immediately if `OptiX_INSTALL_DIR`
> is not set as an environment variable or CMake cache entry.

---

## Build

### Step 1 — Point CMake at the OptiX SDK

```bat
:: Set once in your environment (persists across sessions)
set OptiX_INSTALL_DIR=C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0
```

Or pass it directly on the CMake command line:

```bat
cmake -B build -DOptiX_INSTALL_DIR="C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0"
```

### Step 2 — Configure and Build

```bat
:: Configure
cmake -B build

:: Debug build (default for development)
cmake --build build --config Debug

:: Release build (full optimisation)
cmake --build build --config Release
```

Build artifacts land in `build\Debug\` or `build\Release\`.  
The CUDA PTX file is compiled via the `optix_ptx` target and placed in `build\ptx\`.

### Step 3 — Run

```bat
:: Interactive debug viewer (active scene set via #define in src\core\config.h)
build\Debug\photon_tracer.exe

:: Override defaults at the command line
build\Debug\photon_tracer.exe --spp 32 --photons 2000000 --radius 0.04 --output output\my_render.png
```

### Windows Quick Script

```bat
run.bat              # Configure, build Debug, and launch viewer
run.bat test         # Build Debug and run the full test suite
run.bat release      # Build Release and launch viewer
run.bat clean        # Delete the build directory
```

---

## Command-Line Options

| Option          | Description                                                    | Default            |
|-----------------|----------------------------------------------------------------|--------------------|
| `--width W`     | Output image width (pixels)                                    | 1024               |
| `--height H`    | Output image height (pixels)                                   | 768                |
| `--spp N`       | Samples per pixel for the final render                         | 16                 |
| `--photons N`   | Number of photons to trace                                     | 1000000            |
| `--radius R`    | Photon gather radius (scene units)                             | 0.05               |
| `--output FILE` | Output PNG path                                                | output/render.png  |
| `--mode MODE`   | `full` \| `direct` \| `indirect` \| `photon` \| `normals` \| `material` \| `depth` | full |

---

## Output Files

A completed render writes:

| File                           | Contents                              |
|--------------------------------|---------------------------------------|
| `output/render.png`            | Final combined render                 |
| `output/out_nee_direct.png`    | Direct lighting (NEE) only            |
| `output/out_photon_indirect.png` | Photon indirect illumination only   |
| `output/out_combined.png`      | NEE + photon spectral sum             |
| `output/out_debug_nee.png`     | Single-frame NEE preview (pre-render) |

---

## Interactive Viewer

The renderer opens in a **real-time debug window** running at 1 spp/frame with
direct lighting. Use it to position the camera, inspect scene geometry, and
verify photon map quality before committing to a full render.

### Camera Controls

| Input              | Action                                        |
|--------------------|-----------------------------------------------|
| **W / A / S / D**  | Move forward / left / back / right            |
| **Space / Ctrl**   | Move up / down                                |
| **Mouse**          | Look around (when captured)                   |
| **Shift**          | 3× movement speed                             |
| **M**              | Toggle mouse capture                          |
| **Left click**     | Re-capture mouse when released                |
| **ESC**             | Cancel render → release mouse → quit (3-tier)  |
| **Q**               | Quit immediately                               |

### Render Controls

| Key        | Action                                                                 |
|------------|------------------------------------------------------------------------|
| **R**      | Launch full path tracing render                                        |
| **ESC**    | Cancel an in-progress render and return to preview                     |
| **TAB**    | Cycle mode: `Full` → `DirectOnly` → `IndirectOnly` → `PhotonMap` → `Normals` → `MaterialID` → `Depth` |
| **H**      | Toggle help overlay                                                    |

### Scene Switching

| Key | Scene         |
|-----|---------------|
| **1** | Cornell Box |
| **2** | Conference  |
| **3** | Living Room |
| **4** | Sibenik     |

Switching scenes rebuilds the acceleration structure, re-traces photons, and
resets the camera to the scene's default viewpoint.

### Light Brightness

| Key             | Action                                 |
|-----------------|----------------------------------------|
| **+ / =**       | Increase light brightness (+0.1×)      |
| **- / _**       | Decrease light brightness (−0.1×)      |

Scales all emissive materials uniformly. Photons are automatically re-traced
after each adjustment. Range: 0.1× – 10.0× (default 1.0×).

### Effects Toggles

| Key           | Action                                                             |
|---------------|--------------------------------------------------------------------|
| **V**         | Toggle volumetric scattering (crepuscular rays)                    |
| **O**         | Toggle depth of field (thin-lens)                                  |
| **[ / ]**     | DOF — widen / narrow aperture (⅓-stop steps; lower f = more blur) |
| **, / .**     | DOF — focus closer / farther (10 % steps)                          |

### Debug Overlays (F-keys)

| Key  | Overlay                      | Key  | Overlay                  |
|------|------------------------------|------|--------------------------|
| F1   | Photon points                | F6   | PDFs                     |
| F2   | Global photon map            | F7   | Gather radius sphere     |
| F3   | Caustic map selector         | F8   | MIS weights              |
| F4   | Hash grid debug              | F9   | Spectral colouring       |
| F5   | Photon directions            |      |                          |

**Hover-cell inspection:** Release the mouse (M), enable F2 or F3, then move
the cursor over the image to see cell index, photon count, flux sum/average,
and dominant wavelength in the overlay panel.

---

## Tests

163 unit tests via GoogleTest v1.14.0, covering every core subsystem.

```bat
run.bat test                             :: Quick script

:: Or manually
cmake --build build --config Debug
build\Debug\ppt_tests.exe
build\Debug\ppt_tests.exe --gtest_filter="Photon*"   :: run a subset
```

**Coverage includes:**

```
Math & Geometry        Vector math, ONB, coordinate transforms, ray-triangle
                       intersection, AABB intersection
Spectral               Arithmetic, CIE XYZ colour matching, blackbody emission
Sampling               RNG quality, cosine/uniform hemisphere, triangle,
                       MIS power heuristic (2-way + 3-way), alias table
BSDF                   Fresnel (Schlick, dielectric, TIR), GGX VNDF,
                       evaluation, sampling, reciprocity, energy conservation
Photon Map             Hash grid build/query, Epanechnikov kernel,
                       surface consistency filter, density estimator
Photon Bins            Fibonacci sphere coverage (N=8,16,32), nearest-bin
                       queries, hemisphere coverage, bin solid angle,
                       population, centroid normalisation, struct size (24 B)
Stratified Sampling    4×4 sub-pixel strata coverage
OptiX Integration      Init, GAS build, scene upload, debug frame output,
                       normals mode, final render validation, framebuffer resize
```

---

## Project Structure

```
photon_path_tracer/
├── src/
│   ├── main.cpp                      Entry point, argument parsing, GLFW event loop
│   ├── core/
│   │   ├── types.h                   Vec3, Ray, and shared POD types
│   │   ├── spectrum.h                Spectral arithmetic, CIE XYZ, blackbody
│   │   ├── config.h                  Compile-time scene selection + render constants
│   │   ├── random.h                  PCG RNG
│   │   ├── photon_bins.h             PhotonBin struct, Fibonacci sphere utilities
│   │   ├── photon_density_cache.h    Per-pixel cached spectral density (A1)
│   │   ├── guided_nee.h              Bin-flux-weighted emissive CDF (B2)
│   │   ├── nee_sampling.h            Standard NEE sampling helpers
│   │   ├── cell_bin_grid.h           Cell-level photon binning grid
│   │   ├── cdf.h                     Generic CDF build + sample
│   │   ├── alias_table.h             Alias method for O(1) discrete sampling
│   │   ├── medium.h                  Participating medium (volumetric) data
│   │   ├── phase_function.h          Henyey-Greenstein phase function
│   │   ├── font_overlay.h            Debug text overlay rendering
│   │   └── test_data_io.h            Ground-truth data helpers for tests
│   ├── bsdf/
│   │   └── bsdf.h                    Lambertian, mirror, glass, GGX VNDF
│   ├── scene/
│   │   ├── scene.h                   Scene graph, emitter list
│   │   ├── material.h                Material definitions
│   │   ├── triangle.h                Triangle + BVH primitives
│   │   ├── obj_loader.h / .cpp       Wavefront OBJ + MTL parser
│   ├── renderer/
│   │   ├── renderer.h / .cpp         Host render loop, framebuffer management
│   │   ├── camera.h                  Perspective camera, ray generation
│   │   ├── mis.h                     MIS power heuristic (2-way + 3-way)
│   │   ├── direct_light.h / .cu      Direct lighting kernel
│   │   └── path_tracer.cu            Full path tracing CUDA kernel
│   ├── photon/
│   │   ├── photon.h                  Photon SoA storage layout
│   │   ├── emitter.h / .cu           Emitter sampling, CDF construction
│   │   ├── hash_grid.h / .cu         Hashed uniform grid build + query
│   │   └── density_estimator.h       Epanechnikov kernel density estimation
│   ├── optix/
│   │   ├── optix_device.cu           All OptiX raygen / closesthit programs
│   │   ├── optix_renderer.h / .cpp   Host pipeline: SBT, GAS, launch params
│   │   ├── launch_params.h           GPU/CPU shared launch parameter struct
│   │   └── adaptive_sampling.h / .cu Per-pixel noise metric + convergence mask
│   └── debug/
│       └── debug.h                   Visualisation mode state, key bindings
├── tests/
│   ├── test_main.cpp                 Core unit tests (163 cases, GoogleTest)
│   ├── test_medium.cpp               Participating medium tests
│   ├── test_ground_truth.cpp         Reference image ground-truth comparisons
│   ├── test_pixel_comparison.cpp     Per-pixel render validation
│   ├── test_per_ray_validation.cpp   Individual ray correctness checks
│   └── feature_speed_test.cpp        Feature timing / performance benchmarks
├── scenes/                           Cornell box, conference, living room, etc.
├── doc/
│   └── architecture/                 Full architecture documentation
├── CMakeLists.txt
└── run.bat                           Build, run, test, and clean helper
```

---

## Architecture

[doc/architecture/architecture.md](doc/architecture/architecture.md) covers the
full rendering pipeline in detail: rendering equation, Monte Carlo estimators,
guided NEE derivation, Fibonacci-sphere binning, density estimation, and the
rationale behind each design decision.

---

## License

See [LICENSE](LICENSE).
