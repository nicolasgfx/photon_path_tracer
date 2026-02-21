# Spectral Photon-Centric Renderer

> A physically-based GPU renderer using a **photon-centric** architecture
> over a full spectral representation — no RGB shortcuts in the light transport.

Built on **NVIDIA OptiX** and **CUDA**. Camera rays stop at the first diffuse
hit; all global illumination is carried by the photon map. A **tangential disk
kernel** eliminates planar blocking artifacts. Photon maps can be precomputed,
saved to disk, and loaded instantly for interactive camera exploration.

![Render Result](output/render.png)

---

## At a Glance

| What                        | How                                                                  |
|-----------------------------|----------------------------------------------------------------------|
| Light transport             | Full spectral — 32 wavelength bins, 380–780 nm                      |
| Architecture                | Photon-centric: photon rays carry all indirect transport             |
| Camera pass                 | First-hit only (specular chain ≤8 bounces to first diffuse)         |
| Direct lighting             | NEE with coverage-aware stratified sampling                          |
| Indirect lighting           | Photon density estimation with tangential disk kernel                |
| Spatial index (CPU)         | KD-tree (arbitrary radius, k-NN adaptive)                           |
| Spatial index (GPU)         | Hash grid with shell-expansion k-NN                                 |
| Gather kernel               | Tangential disk (surface distance, not 3D Euclidean)                |
| Default render mode         | SPPM progressive (Hachisuka & Jensen 2009)                          |
| Photon persistence          | Binary save/load; filename encodes photon budget + radius; scene & param hashes auto-invalidate |
| Tone mapping                | ACES Filmic                                                          |
| Sub-pixel sampling          | 4×4 stratified jittered strata (16 SPP default)                     |
| Debug / preview             | Interactive GLFW viewer, 5 render modes, 9 overlay toggles          |
| CPU reference               | Full renderer for validation (physically identical to GPU)           |
| Test coverage               | Unit tests + CPU↔GPU integration tests (GoogleTest)                 |

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
│  Build BVH (GAS) + emitter distribution (CDF / alias table)    │
│       │                                                         │
│       ▼                                                         │
│  OptiX pipeline init ──► upload geometry, materials, emitters   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  PHOTON PASS  (precomputed or cached)                           │
│                                                                 │
│  IF photon_cache.bin exists AND scene unchanged:                │
│     Load photon map + spatial index from cache                  │
│  ELSE:                                                          │
│     Emit N photons from lights                                  │
│     Trace each photon: full BSDF bouncing, Russian roulette     │
│     Deposit at diffuse hits (lightPathDepth ≥ 2 only)           │
│     Build spatial index: KD-tree (CPU) / hash grid (GPU)        │
│     Save to photon_cache.bin                                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  CAMERA PASS  (first-hit only)                                  │
│                                                                 │
│  For each pixel:                                                │
│    Trace camera ray → follow specular chain (mirrors, glass)    │
│    At first diffuse hit:                                        │
│      NEE  ──► shadow rays to emitters (coverage-aware)          │
│      Photon gather ──► tangential disk kernel query              │
│      L = L_direct(NEE) + L_indirect(photon density)             │
│                                                                 │
│  Spectral → CIE XYZ → sRGB → ACES tone map → output            │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

- **Photon rays are the real path tracers.** They carry all indirect
  illumination: diffuse inter-reflection, caustics, colour bleeding.
  Camera rays are cheap probes that only find the first visible surface.

- **Tangential disk kernel.** Photon gather uses surface-projected distance
  instead of 3D Euclidean distance. This eliminates planar blocking artifacts
  and cross-surface photon leakage. The fix is in the kernel metric, not
  the spatial data structure.

- **Precomputable photon maps.** The photon map is view-independent and
  saved as a binary file whose name encodes the photon budget, caustic
  budget, bounce depth, and gather radius
  (`photon_cache_<N>g_<N>c_<N>b_<R>r.bin`). Both a scene hash and a
  parameter hash are embedded in the header; changing any render
  parameter automatically invalidates the cache. Change the camera
  freely without recomputing photons. Press **P** to recompute on demand.

- **Dual CPU/GPU renderer.** CPU KD-tree provides ground truth; GPU hash
  grid provides interactive speed. Both use the same tangential kernel and
  surface consistency filters. Integration tests verify PSNR parity.

---

## Requirements

| Component       | Minimum                              | Notes                              |
|-----------------|--------------------------------------|------------------------------------|
| NVIDIA GPU      | Turing (sm_75) or newer              | Required for GPU path              |
| CUDA Toolkit    | 12.x                                 |                                    |
| NVIDIA OptiX    | 7.x or 9.x                          | `OptiX_INSTALL_DIR` must be set    |
| CMake           | 3.24                                 |                                    |
| C++ Standard    | C++17                                |                                    |
| OS              | Windows 10+ (MSVC 2022)              |                                    |
| VRAM            | 8 GB recommended                     |                                    |

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
:: Interactive debug viewer
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
| `--photons N`   | Number of photons to trace (global map)                        | 1000000            |
| `--global-photons N` | Global photon budget (separate from caustic)              | 1000000            |
| `--caustic-photons N` | Caustic photon budget                                    | 1000000            |
| `--radius R`    | Photon gather radius (scene units)                             | 0.05               |
| `--output FILE` | Output PNG path                                                | output/render.png  |
| `--mode MODE`   | `combined` \| `direct` \| `indirect` \| `photon` \| `normals` \| `material` \| `depth` | combined |
| `--photon-file PATH` | Explicit photon cache file path                          | scene_dir/photon_cache.bin |
| `--force-recompute`  | Ignore cached photon file, always recompute              |                    |
| `--no-save-photons`  | Do not save photon cache after computation               |                    |
| `--spatial MODE`      | `kdtree` \| `hashgrid` — spatial index for CPU           | kdtree             |
| `--adaptive-radius`   | Enable k-NN adaptive gather radius                      |                    |
| `--deterministic`     | Enable deterministic debug mode (for CPU↔GPU bisection) |                    |
| `--sppm`              | Enable SPPM mode                                        | enabled by default |
| `--sppm-iterations N` | SPPM iteration count                                    | 64                 |
| `--sppm-radius R`     | SPPM initial radius                                     | 0.1                |

---

## Output Files

A completed render writes:

| File                             | Contents                              |
|----------------------------------|---------------------------------------|
| `output/render.png`              | Final combined render                 |
| `output/out_nee_direct.png`      | Direct lighting (NEE) only            |
| `output/out_photon_indirect.png` | Photon indirect illumination only     |
| `output/out_photon_caustic.png`  | Caustic map contribution              |
| `output/out_combined.png`        | NEE + photon spectral sum             |

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
| **ESC**            | Cancel render → release mouse → quit (3-tier) |
| **Q**              | Quit immediately                              |

### Render Controls

| Key        | Action                                                                 |
|------------|------------------------------------------------------------------------|
| **R**      | Launch full render (SPPM by default)                                   |
| **P**      | Recompute photons (re-trace, rebuild index, save cache)                |
| **ESC**    | Cancel an in-progress render and return to preview                     |
| **TAB**    | Cycle mode: `Combined` → `DirectOnly` → `IndirectOnly` → `PhotonMap` → `Normals` → `MaterialID` → `Depth` |
| **H**      | Toggle help overlay                                                    |

### Scene Switching

| Key   | Scene             | Lighting          | Complexity |
|-------|-------------------|-------------------|------------|
| **1** | Cornell Box       | MTL emitters      | Low        |
| **2** | Conference Room   | MTL emitters      | Medium     |
| **3** | Living Room       | MTL emitters      | Medium     |
| **4** | Fireplace Room    | MTL emitters      | Medium     |
| **5** | Breakfast Room    | MTL emitters      | Medium     |
| **6** | Salle de Bain     | MTL emitters      | Medium     |
| **7** | Sibenik Cathedral | Directional (sun) | High       |
| **8** | Sponza            | Spherical env     | High       |
| **9** | Hairball          | Spherical env     | High       |

Switching scenes rebuilds the acceleration structure, applies a
complexity-based parameter preset (photon budget, gather radius, SPP,
bounces), sets the scene's lighting mode, loads the photon cache if a
matching binary exists, and resets the camera.

### Light Brightness

| Key             | Action                                 |
|-----------------|----------------------------------------|
| **+ / =**       | Increase light brightness (+0.1×)      |
| **- / _**       | Decrease light brightness (−0.1×)      |

Scales all emissive materials uniformly. Press **P** to recompute photons
after adjustment. Range: 0.1× – 10.0× (default 1.0×).

### Effects Toggles

| Key           | Action                                                             |
|---------------|--------------------------------------------------------------------|
| **O**         | Toggle depth of field (thin-lens)                                  |
| **[ / ]**     | DOF — widen / narrow aperture (⅓-stop steps)                      |
| **, / .**     | DOF — focus closer / farther (10% steps)                           |

### Debug Overlays (F-keys)

| Key  | Overlay                      | Key  | Overlay                  |
|------|------------------------------|------|--------------------------|
| F1   | Photon points                | F6   | PDFs                     |
| F2   | Global photon map            | F7   | Gather radius sphere     |
| F3   | Caustic map selector         | F8   | MIS weights              |
| F4   | Hash grid / KD-tree cells    | F9   | Spectral colouring       |
| F5   | Photon directions            |      |                          |

**Hover-cell inspection:** Release the mouse (M), enable F2 or F3, then move
the cursor over the image to see cell index, photon count, flux sum/average,
and dominant wavelength in the overlay panel.

---

## Tests

Unit tests and integration tests via GoogleTest v1.14.0.

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
                       alias table, coverage-aware NEE
BSDF                   Fresnel (Schlick, dielectric, TIR), GGX VNDF,
                       evaluation, sampling, reciprocity, energy conservation
KD-Tree                Build, range query, k-NN, empty, single photon, boundary
Tangential Kernel      Tangential distance matches analytic for planar geometry
Surface Filter         Cross-wall photons rejected, same-surface accepted
Hash Grid              Build/query, Epanechnikov kernel, surface consistency
Density Estimator      Tangential disk kernel correctness
Photon Map             Deposition rules, separate global/caustic maps
SPPM                   Progressive convergence, radius shrinkage
Integration (CPU↔GPU)  Direct lighting match, photon indirect match,
                       combined match, energy conservation, PSNR thresholds
OptiX Integration      Init, GAS build, scene upload, debug frame output,
                       normals mode, final render validation, framebuffer resize
Stratified Sampling    4×4 sub-pixel strata coverage
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
│   │   ├── sppm.h                    SPPM types, progressive update, reconstruction
│   │   ├── photon_bins.h             PhotonBin struct, Fibonacci sphere (Phase 7)
│   │   ├── photon_density_cache.h    Per-pixel cached spectral density
│   │   ├── guided_nee.h             Bin-flux-weighted emissive CDF
│   │   ├── nee_sampling.h           NEE sampling helpers
│   │   ├── cdf.h                     Generic CDF build + sample
│   │   ├── alias_table.h            Alias method for O(1) discrete sampling
│   │   ├── medium.h                  Participating medium (temporarily disabled)
│   │   ├── phase_function.h          Henyey-Greenstein phase function (temporarily disabled)
│   │   ├── font_overlay.h           Debug text overlay rendering
│   │   └── test_data_io.h           Ground-truth data helpers for tests
│   ├── bsdf/
│   │   └── bsdf.h                    Lambertian, mirror, glass, GGX VNDF
│   ├── scene/
│   │   ├── scene.h                   Scene graph, emitter list
│   │   ├── material.h               Material definitions
│   │   ├── triangle.h               Triangle + BVH primitives
│   │   └── obj_loader.h / .cpp      Wavefront OBJ + MTL parser
│   ├── renderer/
│   │   ├── renderer.h / .cpp        CPU reference renderer, RenderConfig, FrameBuffer
│   │   ├── camera.h                  Perspective camera, ray generation
│   │   ├── mis.h                     MIS weight utilities
│   │   ├── direct_light.h / .cu     Direct lighting kernel
│   │   └── path_tracer.cu           CPU/CUDA path tracing kernels
│   ├── photon/
│   │   ├── photon.h                  Photon and PhotonSoA structs
│   │   ├── kd_tree.h                KD-tree build + range query + k-NN (CPU reference)
│   │   ├── hash_grid.h / .cu        Hashed uniform grid build + query
│   │   ├── emitter.h / .cu          Emitter sampling, CDF construction
│   │   ├── density_estimator.h      Tangential disk kernel density estimation
│   │   ├── surface_filter.h         Surface consistency filter (tangential metric)
│   │   └── photon_io.h / .cpp       Binary save/load for photon map persistence
│   ├── optix/
│   │   ├── optix_device.cu          All OptiX raygen / closesthit programs
│   │   ├── optix_renderer.h / .cpp  Host pipeline: SBT, GAS, launch params
│   │   ├── launch_params.h          GPU/CPU shared launch parameter struct
│   │   └── adaptive_sampling.h / .cu Per-pixel noise metric + convergence mask
│   └── debug/
│       └── debug.h                   Visualisation mode state, key bindings
├── tests/
│   ├── test_main.cpp                Core unit tests (GoogleTest)
│   ├── test_kd_tree.cpp             KD-tree unit tests
│   ├── test_tangential_gather.cpp   Tangential kernel tests
│   ├── test_surface_filter.cpp      Surface consistency filter tests
│   ├── test_integration.cpp         CPU↔GPU integration tests
│   ├── test_ground_truth.cpp        Reference image ground-truth comparisons
│   ├── test_per_ray_validation.cpp  Individual ray correctness checks
│   ├── test_pixel_comparison.cpp    Per-pixel render validation
│   ├── test_medium.cpp              Participating medium tests
│   └── feature_speed_test.cpp       Performance benchmarks
├── scenes/                           Cornell box, conference, living room, etc.
├── doc/
│   └── architecture/                Full architecture documentation
├── CMakeLists.txt
└── run.bat                          Build, run, test, and clean helper
```

---

## Architecture

[doc/architecture/architecture.md](doc/architecture/architecture.md) covers the
full rendering pipeline in detail: photon-centric design, tangential disk
kernel, surface-aware gather, NEE with coverage-aware sampling, SPPM
progressive convergence, KD-tree vs hash grid spatial indices, binary
photon map persistence, and CPU↔GPU parity testing.

---

## License

See [LICENSE](LICENSE).
