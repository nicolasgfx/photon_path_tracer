# Spectral Photon-Centric Renderer

A physically-based GPU renderer built around a **photon-centric** architecture
with **full spectral light transport** — 32 wavelength bins, no RGB approximations
anywhere in the path.

Built on **NVIDIA OptiX** and **CUDA**.

---

## What Makes This Different

Most renderers, including production path tracers, work camera-first: each pixel
fires a ray, bounces it through the scene, and accumulates energy. This renderer
inverts that relationship.

| Property | This renderer | Typical path tracer |
|---|---|---|
| **Light transport carrier** | Photon rays carry all indirect GI | Camera ray bounces carry GI |
| **Camera ray role** | Probe to first diffuse surface only | Full recursive path |
| **Spectral representation** | 32 bins, 380–780 nm throughout | sRGB or hero wavelength |
| **Chromatic dispersion** | Cauchy-equation per-bin IOR + Fresnel | Rarely modelled |
| **Gather kernel** | Tangential disk (2D surface distance) | N/A |
| **Indirect lighting** | Decoupled from camera, precomputable | Recomputed every frame |
| **Caustics** | Separate caustic photon map + adaptive shooting | Expensive or approximated |
| **Cell cache** | Precomputed per-cell statistics, adaptive radius | N/A |
| **CPU reference** | Full dual implementation, PSNR-tested | Rarely present |

**The photon pass is the path tracer.** Photon rays start from lights and
bounce through the scene with full BSDF importance sampling and Russian roulette,
depositing spectral flux packets at diffuse surfaces. Camera rays find the first
visible surface and query the precomputed photon map — no further bouncing needed.

**The tangential disk kernel** replaces the standard 3D spherical gather with a
2D surface-projected metric. This eliminates planar blocking artifacts and
cross-surface photon leakage that affect every renderer using a 3D Euclidean
gather metric, regardless of spatial data structure.

**The photon map is view-independent.** It can be computed once, saved to disk,
and reloaded instantly for interactive camera exploration of the same scene.

---

## At a Glance

| Component | Detail |
|---|---|
| Light transport | Full spectral — 32 wavelength bins, 380–780 nm |
| Architecture | Photon-centric: photon rays carry all indirect transport |
| Camera pass | First diffuse hit only (specular chain ≤ 8 bounces) |
| Direct lighting | NEE with coverage-aware stratified area sampling |
| Indirect lighting | Photon density estimation, tangential disk kernel |
| Spatial index (CPU) | KD-tree, arbitrary radius, k-NN adaptive |
| Spatial index (GPU) | Hash grid, shell-expansion k-NN |
| Chromatic dispersion | Cauchy equation: $n(\lambda) = A + B/\lambda^2$, per-bin Fresnel |
| Glass colour | Spectral transmittance filter (Tf) per material |
| Photon path flags | Per-photon glass/caustic/dispersion/volume tags |
| Cell cache | CellInfoCache — 65K-cell precomputed density, variance, caustic stats |
| Adaptive gather radius | Per-cell radius from CellInfoCache photon density |
| Adaptive caustic shooting | Multi-iteration targeted emission for high-CV cells |
| Gather kernel | Tangential disk — 2D surface distance, not 3D Euclidean |
| Global illumination mode | SPPM progressive (Hachisuka & Jensen 2009) |
| Tone mapping | ACES Filmic |
| Sub-pixel sampling | Stratified jittered (16 SPP default) |
| Viewer | Interactive GLFW window, 7 render modes |
| CPU reference | Physically identical to GPU path, used for PSNR validation |
| Tests | ~500 unit + integration tests (GoogleTest) |

---

## Rendering Pipeline

```
STARTUP
  Load OBJ/MTL scene
  Normalize scene to [-0.5, 0.5]³
  Build CPU BVH + OptiX GAS
  Build emitter distribution (alias table over emissive triangles)
  Upload geometry, materials, emitters to GPU

PHOTON PASS  ── run once, reuse across camera views ──
  Emit N photons from lights (power-proportional sampling)
  Per photon:
    Bounce 0: cosine-weighted hemisphere coverage from emitter
    Bounce 1+: BSDF importance sampling + Russian roulette
    Glass: Cauchy dispersion (per-λ IOR), Tf filter, IOR stack, path flags
    Deposit flux packet at each diffuse hit where lightPathDepth ≥ 2
    (first-bounce deposits skipped — would double-count with NEE)
    Separate global map (diffuse paths) vs caustic map (specular chain)
  Build spatial index: KD-tree (CPU) or hash grid (GPU)
  Build CellInfoCache: per-cell density, variance, caustic stats
  Adaptive caustic shooting: re-emit toward high-CV hotspot cells

CAMERA PASS  ── executed per frame ──
  For each pixel (stratified jittered sub-pixel samples):
    Trace camera ray
    Follow specular chain (mirror / glass) up to 8 bounces
    At first diffuse hit:
      NEE: shadow rays to sampled emitter points
      Gather: query photon map with tangential disk kernel
        (adaptive gather radius from CellInfoCache)
        (caustic gather skipped if cell has zero caustics)
      L = L_emission + L_direct(NEE) + L_indirect(photon density)
    Spectral → CIE XYZ → linear sRGB → ACES → gamma → PNG
```

---

## Tangential Disk Kernel

Standard photon mapping uses 3D Euclidean distance to gather photons. This
causes well-known artifacts: photons deposited on a floor "bleed through"
into a table 2 cm above, and planar boundaries produce visible blocking patterns.
The artifacts appear regardless of whether the spatial structure is a KD-tree,
hash grid, or BVH — the problem is the metric, not the structure.

This renderer replaces the spherical gather with a **tangent-plane disk kernel**:

```
v       = photon_pos − query_pos
d_plane = dot(surface_normal, v)          // distance off the surface
v_tan   = v − surface_normal × d_plane    // project onto tangent plane
d_tan²  = dot(v_tan, v_tan)               // 2D disk distance
```

Photon accepted if: `d_tan² < r²` AND `|d_plane| < τ` AND normals compatible.

This turns the gather region from a 3D sphere into a 2D disk flush with the
queried surface, eliminating cross-surface leakage entirely. Both the CPU
KD-tree and GPU hash grid use the same kernel.

---

## Requirements

| Component | Minimum | Notes |
|---|---|---|
| NVIDIA GPU | Turing (sm_75) or newer | Required for GPU path |
| CUDA Toolkit | 12.x | |
| NVIDIA OptiX | 7.x or 9.x | `OptiX_INSTALL_DIR` must be set |
| CMake | 3.24 | |
| C++ Standard | C++17 | |
| OS | Windows 10+ (MSVC 2022) | |
| VRAM | 8 GB recommended | |

> **OptiX is mandatory.** The build fails immediately if `OptiX_INSTALL_DIR`
> is not set as an environment variable or CMake cache entry.

---

## Build

### Step 1 — Set OptiX SDK path

```bat
set OptiX_INSTALL_DIR=C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0
```

Or pass directly to CMake:

```bat
cmake -B build -DOptiX_INSTALL_DIR="C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0"
```

### Step 2 — Configure and build

```bat
cmake -B build
cmake --build build --config Debug    :: development build
cmake --build build --config Release  :: optimised build
```

Build artifacts land in `build\Debug\` or `build\Release\`.
The CUDA PTX file (`optix_ptx` target) is placed in `build\ptx\`.

### Step 3 — Run

```bat
build\Debug\photon_tracer.exe
build\Debug\photon_tracer.exe --spp 32 --photons 2000000 --radius 0.04
```

### Windows quick script

```bat
run.bat              :: configure, build Debug, launch viewer
run.bat test         :: build Debug, run full test suite
run.bat release      :: build Release, launch viewer
run.bat clean        :: delete build directory
```

---

## Command-Line Options

| Option | Description | Default |
|---|---|---|
| `--width W` | Output image width (pixels) | 1024 |
| `--height H` | Output image height (pixels) | 768 |
| `--spp N` | Samples per pixel | 16 |
| `--photons N` | Photon count (global map) | 1 000 000 |
| `--global-photons N` | Global photon budget | 1 000 000 |
| `--caustic-photons N` | Caustic photon budget | 1 000 000 |
| `--radius R` | Photon gather radius (scene units) | 0.05 |
| `--output FILE` | Output PNG path | output/render.png |
| `--mode MODE` | `combined` \| `direct` \| `indirect` \| `photon` \| `normals` \| `material` \| `depth` | combined |
| `--spatial MODE` | `kdtree` \| `hashgrid` — CPU spatial index | kdtree |
| `--adaptive-radius` | Enable k-NN adaptive gather radius | off |
| `--knn-k N` | k for k-NN adaptive radius query | 32 |
| `--sppm` | Enable SPPM progressive mode | off |
| `--sppm-iterations N` | SPPM iteration count | 64 |
| `--sppm-radius R` | SPPM initial radius | 0.1 |

---

## Output Files

Each completed render writes timestamped files to `output/`:

| File | Contents |
|---|---|
| `render_YYYYMMDD_HHMMSS.png` | Final combined render (NEE + photon) |
| `render_YYYYMMDD_HHMMSS_nee_direct.png` | Direct lighting (NEE) only |
| `render_YYYYMMDD_HHMMSS_photon_indirect.png` | Photon indirect only |
| `out_debug_nee.png` | Quick NEE preview written at render start |

SPPM mode writes a PNG after every iteration:
`output/sppm_YYYYMMDD_HHMMSS_iter0001.png … _final.png`

---

## Interactive Viewer

Launching the executable opens a real-time GLFW window running at 1 spp/frame
using direct lighting only. Use it to position the camera, inspect the photon
distribution, and verify geometry before committing to a full render.

### Camera

| Input | Action |
|---|---|
| **W / A / S / D** | Move forward / left / back / right |
| **Space / Ctrl** | Move up / down |
| **Mouse** | Look around (when captured) |
| **Shift** | 3× movement speed |
| **M** | Toggle mouse capture / release |
| **Left click** | Re-capture mouse |

### Render

| Key | Action |
|---|---|
| **R** | Launch full render (progressive SPP or SPPM depending on `--sppm`) |
| **P** | Re-trace photon maps and rebuild spatial index |
| **ESC** | Cancel in-progress render → release mouse → quit (3-step) |
| **TAB** | Cycle render mode: `Full` → `Direct Only` → `Indirect Only` → `Photon Map` → `Normals` → `Material ID` → `Depth` |
| **H** | Toggle help overlay |

### Depth of Field (thin-lens)

| Key | Action |
|---|---|
| **O** | Toggle DOF on / off |
| **[ / ]** | Widen / narrow aperture (⅓-stop per press) |
| **, / .** | Focus closer / farther (10% per press) |
| **F** | Auto-focus on screen centre (ray cast) |
| **Middle mouse** | Auto-focus on cursor position |

### Light Brightness

| Key | Action |
|---|---|
| **+ / =** | Increase brightness, re-traces photons automatically |
| **-** | Decrease brightness, re-traces photons automatically |

Range: 0.1×–10.0×, default 1.0×.

### Volume / Gather Mode

| Key | Action |
|---|---|
| **V** | Toggle participating medium (currently disabled; toggle state tracked) |
| **G** | Toggle dense cell-bin grid gather mode |

### Scene Switching (keys 1–8)

| Key | Scene | Light mode | Complexity |
|---|---|---|---|
| **1** | Cornell Box | MTL emitters | Low |
| **2** | Cornell Sphere | MTL emitters | Low |
| **3** | Cornell Mirror | MTL emitters | Low |
| **4** | Cornell Water | MTL emitters | Low |
| **5** | Living Room | MTL emitters | Medium |
| **6** | Conference Room | MTL emitters | Medium |
| **7** | Salle de Bain | MTL emitters | Medium |
| **8** | Mori Knob | MTL emitters | Medium |

Switching loads the new scene, rebuilds the BVH and OptiX GAS, re-traces
photons, and resets the camera to the scene's default position.

### Debug Overlays (F-keys)

| Key | Overlay | Status |
|---|---|---|
| **F1** | Photon point visualization (projected onto framebuffer) | Implemented |
| **F2** | Global photon map (with hover-cell inspector) | Implemented |
| **F3** | Caustic photon map | UI wired; separate map pending |
| **F4** | Hash grid cell visualization | Planned |
| **F5** | Photon direction arrows | Planned |
| **F6** | PDF visualization | Planned |
| **F7** | Gather radius sphere | Planned |
| **F8** | MIS weight display | Planned |
| **F9** | Spectral wavelength coloring | Planned |

**Hover-cell inspector:** with F2 active and mouse released (M), move the cursor
over the image to display a panel showing cell coordinate, photon count, total
and average flux, dominant wavelength bin and nm value for the photon cell under
the cursor.

---

## Tests

Unit and integration tests via GoogleTest 1.14.0.

```bat
run.bat test

:: Or manually
cmake --build build --config Debug
build\Debug\ppt_tests.exe
build\Debug\ppt_tests.exe --gtest_filter="KDTree*"
```

**Test coverage:**

| Area | What is tested |
|---|---|
| Math & geometry | Vector math, ONB, ray–triangle intersection, AABB |
| Spectral | Arithmetic, CIE XYZ colour matching, blackbody radiation |
| Sampling | PCG RNG, cosine/uniform hemisphere, alias table, coverage-aware NEE |
| BSDF | Fresnel (Schlick, dielectric, TIR), GGX VNDF, reciprocity, energy conservation |
| KD-tree | Build, range query, k-NN, empty tree, single photon, boundary cases |
| Tangential kernel | Distance matches analytic for coplanar geometry |
| Surface filter | Cross-wall photons rejected; same-surface photons accepted |
| Hash grid | Build, query, Epanechnikov kernel, surface consistency |
| Density estimator | Tangential disk kernel correctness |
| Photon flags | Path flag tagging, glass/caustic/dispersion classification |
| CellInfoCache | Build, query, adaptive radius, caustic hotspot detection |
| Dispersion | Cauchy IOR, per-bin Fresnel, spectral splitting |
| Adaptive caustics | Targeted emission convergence, photon budget distribution |
| IOR stack | Push/pop/overflow, nested dielectric tracking |
| Photon map | Deposition rules (lightPathDepth ≥ 2), global/caustic separation |
| SPPM | Progressive convergence, radius shrinkage |
| CPU↔GPU integration | Direct lighting PSNR > 40 dB; indirect PSNR > 30 dB; energy conservation within 5% |
| OptiX integration | Init, GAS build, scene upload, debug frame, normals mode, framebuffer resize |

---

## Project Structure

```
photon_path_tracer/
├── src/
│   ├── main.cpp                     Entry point, CLI parsing, GLFW event loop
│   ├── core/
│   │   ├── config.h                 Scene profiles, render constants, complexity presets
│   │   ├── types.h                  Vec3, Ray, shared POD types
│   │   ├── spectrum.h               Spectral arithmetic, CIE XYZ, blackbody
│   │   ├── random.h                 PCG RNG
│   │   ├── sppm.h                   SPPM types, progressive radius update, reconstruction
│   │   ├── cdf.h / alias_table.h    CDF build and O(1) alias sampling
│   │   ├── nee_sampling.h           Coverage-aware NEE helpers
│   │   ├── cell_cache.h             CellInfoCache — precomputed per-cell statistics
│   │   └── font_overlay.h           Debug text rendering (stb_easy_font)
│   ├── bsdf/
│   │   └── bsdf.h                   Lambertian, mirror, glass, GGX VNDF
│   ├── scene/
│   │   ├── scene.h                  Scene graph, emitter list, BVH
│   │   ├── material.h               Material types and properties
│   │   ├── triangle.h               Triangle primitives
│   │   └── obj_loader.h / .cpp      Wavefront OBJ + MTL parser
│   ├── renderer/
│   │   ├── renderer.h / .cpp        CPU reference renderer, RenderConfig, FrameBuffer
│   │   ├── camera.h                 Perspective camera with thin-lens DOF
│   │   └── direct_light.h           Direct lighting kernel (CPU)
│   ├── photon/
│   │   ├── photon.h                 Photon and PhotonSoA (structure-of-arrays)
│   │   ├── kd_tree.h                KD-tree: build, range query, k-NN (CPU reference)
│   │   ├── hash_grid.h / .cu        Hash grid: build + query (GPU primary)
│   │   ├── emitter.h / .cu          Emitter sampling, alias table construction
│   │   ├── density_estimator.h      Tangential disk kernel density estimation
│   │   └── surface_filter.h         Surface consistency filter
│   ├── optix/
│   │   ├── optix_device.cu          OptiX raygen / closesthit programs
│   │   ├── optix_renderer.h / .cpp  Host pipeline: SBT, GAS, launch params
│   │   └── launch_params.h          Shared GPU/CPU launch parameter struct
│   └── debug/
│       └── debug.h                  Debug overlay state, key bindings, photon projection
├── tests/
│   ├── test_main.cpp                Core unit tests
│   ├── test_kd_tree.cpp             KD-tree tests
│   ├── test_tangential_gather.cpp   Tangential kernel tests
│   ├── test_surface_filter.cpp      Surface consistency filter tests
│   ├── test_integration.cpp         CPU↔GPU integration tests
│   ├── test_ground_truth.cpp        Reference image comparisons
│   ├── test_medium.cpp              Participating medium tests
│   └── feature_speed_test.cpp       Performance benchmarks
│   └── test_speed_tweaks.cpp     Dispersion, CellInfoCache, adaptive caustics, IOR stack
├── scenes/                          OBJ scene files
├── doc/architecture/                Architecture documentation
├── CREDITS.md                       Third-party scene attributions
├── CMakeLists.txt
└── run.bat                          Build / run / test / clean helper
```

---

## Third-Party Scene Credits

All test scenes are from **Morgan McGuire's Computer Graphics Archive**
(https://casual-effects.com/data). Full per-model attributions, copyright
notices, and license terms are in [CREDITS.md](CREDITS.md).

> **Note:** The Sibenik Cathedral scene is licensed **CC BY-NC** and may only
> be used for non-commercial purposes.

---

## License

See [LICENSE](LICENSE).
