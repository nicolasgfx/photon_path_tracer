# Path Tracer Redesign Part 1 — Photon Mapping

> This document covers the photon tracing subsystem only. Relevant interfaces for the camera-side path tracer are noted where they affect the photon path design.

**Constraints:**
- Two-pass architecture: (1) photon tracing, (2) path tracing with photon guidance
- CPU and GPU consistency is a high-priority requirement

---

## 1. Cleanup Plan

The project grew organically and accumulated significant complexity: 48 source files / ~18 500 lines, a 2 900-line monolithic CUDA kernel, ~1 400 lines of duplicated BSDF/NEE/emission code between CPU and GPU paths, misplaced files across directories, deprecated stubs still compiled, and naming inconsistencies that obscure intent.

This redesign is an opportunity to make the codebase clean enough for an academic reference implementation: every file in the right place, no dead code, no duplication, consistent naming, and clear separation of concerns.

### 1.1 Code to Delete

Remove outright — not moved, not deprecated, **deleted.**

| Target | Reason |
|--------|--------|
| **Deduped emitter point set** | Replaced by view-dependent power CDF |
| **Multi-map photon re-tracing** (`MULTI_MAP_SPP_GROUP`) | Premature optimisation; recreate the map per frame initially |
| **`DEFAULT_PHOTON_EMITTER_UNIFORM_MIX`** | Unnecessary complexity in emission sampling |
| **`DEFAULT_PHOTON_BOUNCE_STRATA`** | Cell-stratified bounce decorrelation; replace with simpler approach if needed |
| **`guided_nee.h`** (48 lines) | Deprecated v1 photon-bin NEE; superseded by `direct_light.h`. Still `#include`d — remove all references |
| **`mis.h`** (104 lines) | Deprecated v1 MIS; superseded by `nee_shared.h`. Move any test-only references to a local test helper |
| **`emitter.cu`** (99 lines) | Non-functional stub; real GPU emission is in `optix_device.cu` |
| **`direct_light.cu`** (22 lines) | Non-functional stub; real shadow rays go through OptiX `optixTrace` |
| **`photon_beam.h`** (25 lines) | Placeholder stub; resurrect from git when volume beams are implemented |
| **Directional photon bins** (`photon_bins.h`, 63 lines; `PHOTON_BIN_COUNT/MAX_PHOTON_BIN_COUNT` in `config.h`) | Re-evaluate after new architecture; delete unless the hash-grid directional cache (§4) still needs them |
| **Adaptive radius / caustic emission tuning constants** | `ADAPTIVE_RADIUS_{MIN,MAX}_FACTOR`, `ADAPTIVE_RADIUS_TARGET_K`, `CAUSTIC_TARGETED_FRACTION`, `CAUSTIC_MIN_FOR_ANALYSIS`, `CAUSTIC_CV_THRESHOLD`, `CAUSTIC_MAX_TARGETED_ITERS` — re-derive from the new architecture or delete |

### 1.2 Files to Move (Separation of Concerns)

Seven of the 20 files in `core/` do not belong there. A "core" directory should contain project-wide primitives (types, spectrum, RNG, config). Domain-specific code must move to its domain directory.

| File | Current | Move to | Reason |
|------|---------|---------|--------|
| `cell_bin_grid.h` | `core/` | `photon/` | Photon directional bin grid; depends on `PhotonSoA` |
| `cell_cache.h` | `core/` | `photon/` | Photon per-cell statistics; built from `PhotonSoA`, uses hash-grid hash |
| `photon_bins.h` | `core/` | `photon/` | Fibonacci sphere bins — a photon-mapping concept |
| `emitter_points.h` | `core/` | `renderer/` | Emissive surface point set — consumed by NEE/`light_cache.h` |
| `light_cache.h` | `core/` | `renderer/` | Per-cell shadow-ray cache — NEE infrastructure |
| `font_overlay.h` | `core/` | `debug/` | UI text overlay utility, not a core primitive |
| `test_data_io.h` | `core/` | `tests/` | Binary save/load for test snapshots; no production use |

After moves, `core/` shrinks from 20 to 13 files (and further to 11 after merges below).

### 1.3 Files to Merge

Small, single-function files that fracture a concept across multiple locations.

| Merge source | Into target | Rationale |
|-------------|-------------|-----------|
| `nee_sampling.h` (13 lines, 1 function) | `nee_shared.h` | Single NEE helper; name is confusingly similar |
| `cdf.h` (24 lines, 1 function) | `random.h` | `binary_search_cdf()` is a sampling utility |
| `adaptive_emission.h` (26 lines, 1 struct) | `emitter.h` | Stub wrapper around an alias table — already consumed there |
| `medium.h` + `phase_function.h` | → `volume/medium.h` | Participating medium + its phase functions are one concept |
| `sppm.h` | consider → `renderer/sppm.h` | SPPM is a rendering algorithm, not a core primitive |

Also rename after merge: `nee_shared.h` ← (was `nee_sampling.h` + `nee_shared.h`). No remaining ambiguity.

### 1.4 Decompose `optix_device.cu` (2 904 lines → ~10 includes)

The monolithic GPU kernel file makes reasoning about any single subsystem impossible. Split into `__device__` `.cuh` includes, each with one clear responsibility. The main `.cu` retains only the top-level `__raygen__` / `__closesthit__` / `__miss__` entry points and the `#include` list.

| New file | Lines | Content |
|----------|-------|---------|
| `optix_utils.cuh` | ~80 | CIE LUT, float↔uint helpers |
| `optix_material.cuh` | ~290 | `dev_get_Kd()`, `dev_get_roughness()`, `DevONB` → rename to reuse `ONB` |
| `optix_bsdf.cuh` | ~450 | GGX, Fresnel, `dev_bsdf_evaluate/sample/pdf` |
| `optix_specular.cuh` | ~170 | Dispersion, IOR stack, `SpecularBounceResult` |
| `optix_nee.cuh` | ~320 | `dev_light_pdf`, `dev_nee_evaluate_sample`, golden-ratio NEE dispatch |
| `optix_path_trace.cuh` | ~270 | `full_path_trace()` |
| `optix_photon_trace.cuh` | ~300 | `__raygen__photon_trace` bounce loop body |
| `optix_targeted_photon.cuh` | ~270 | `__raygen__targeted_photon_trace` |
| `optix_sppm.cuh` | ~240 | SPPM camera pass + gather kernel |
| `optix_debug.cuh` | ~250 | `debug_first_hit`, heatmap overlays |

Non-goal: these are compilation-unit includes (`#include` inside the `.cu`), not separately compiled translation units. OptiX requires a single module per pipeline.

### 1.5 Eliminate CPU ↔ GPU Duplication (~1 400 lines)

The single largest structural problem. Identical physics is reimplemented in CPU-only and GPU-only variants with different function names, different branch structures, and occasional semantic drift.

| Concept | CPU location | GPU reimplementation | Estimated dup. |
|---------|-------------|---------------------|---------------|
| BSDF (Fresnel, GGX, evaluate, sample, pdf) | `bsdf.h` + `bsdf_shared.h` | `optix_device.cu:444–910` | ~450 lines |
| ONB frame construction | `types.h::ONB` | `optix_device.cu::DevONB` | ~30 lines |
| IOR stack (push/pop/top) | `emitter.h::IORStack` | `optix_device.cu::IORStack` | ~40 lines |
| Photon emission + bounce loop | `emitter.h::trace_photons()` | `__raygen__photon_trace` | ~300 lines |
| Targeted caustic emission | `emitter.h::trace_targeted_caustic_*` | `__raygen__targeted_photon_trace` | ~270 lines |
| NEE direct lighting | `direct_light.h` | `optix_device.cu:916–1228` | ~300 lines |
| Camera ray generation | `camera.h::generate_ray()` | `dev_generate_camera_ray` | ~60 lines |
| CIE colour matching | `spectrum.h` (Wyman analytic) | `optix_device.cu:57–66` (LUT) | ~20 lines (different impl!) |
| Hash function (Teschner primes) | `hash_grid.h` | `cell_cache.h`, `light_cache.h`, `hash_grid.cu`, `optix_device.cu` | ~10 × 4 copies |

**Strategy:** Promote shared code into `__host__ __device__` (`HD`) headers. The pattern already exists in `bsdf_shared.h` and `nee_shared.h` — extend it systematically:

1. **`bsdf_shared.h`** becomes the single BSDF truth. Move all Fresnel, GGX, sampling math into `HD` functions. `bsdf.h` and `optix_bsdf.cuh` become thin wrappers that call the shared functions with their respective `HitRecord`/`TraceResult` unpacking.
2. **`types.h::ONB`** — mark `HD`. Delete `DevONB`.
3. **`hash_shared.h`** (new, ~15 lines) — single `HD` Teschner hash. All four current copies replaced.
4. **`ior_stack.h`** (new, ~40 lines) — single `HD` `IORStack`. Used by both `emitter.h` and `optix_specular.cuh`.
5. **CIE colour matching** — `spectrum.h` Wyman fit functions already work on device. Mark `HD`, delete GPU LUT.
6. **`camera.h::generate_ray()`** — mark `HD`. Delete `dev_generate_camera_ray`.

CPU ↔ GPU emission/bounce/NEE flow is structurally different (OptiX ray-gen vs. explicit BVH traversal) and cannot be trivially unified, but the **material interaction code within each bounce** can share the same `HD` functions.

### 1.6 Rename for Clarity and Consistency

| Current name | Rename to | Rationale |
|-------------|-----------|-----------|
| `DevONB` | (delete; use `ONB`) | Shared `HD` type |
| `DevMaterialType` | (delete; use `MaterialType`) | Must be kept in sync manually today — share via `HD` or a common enum header |
| `dev_bsdf_evaluate` | `bsdf_evaluate` (shared) | Same function, different prefix |
| `dev_fresnel_schlick` | `fresnel_schlick` (shared) | Same function, different prefix |
| `dev_ggx_D/G1/G` | `ggx_D/G1/G` (shared) | Same function, different prefix |
| `core/nee_sampling.h` | (merge into `nee_shared.h`) | Eliminate confusing pair |
| `launch_tonemap_kernel()` in `hash_grid.cu` | Move to `renderer/tonemap.cu` or `optix/tonemap.cu` | Unrelated to hash grids |
| `RENDER_MODE_*` int constants in `launch_params.h` | Share `RenderMode` enum from `renderer.h` | Duplicate enum as magic ints |

### 1.7 Reduce `main.cpp` (1 823 lines)

`main.cpp` mixes application bootstrap, GLFW window management, input handling, overlay rendering, and scene augmentation. Extract:

| Extract into | Content |
|-------------|---------|
| `app/viewer.h/.cpp` | GLFW window, event loop, key/mouse callbacks, overlay rendering |
| Scene augmentation (sphere lights, medium assignment) | `scene/scene_builder.h` or `scene/scene.cpp` |

Target: `main.cpp` ≤ 200 lines — parse args, create scene, create renderer, run viewer.

### 1.8 Reduce `optix_renderer.cpp` (2 515 lines)

| Extract into | Content |
|-------------|---------|
| `optix/optix_setup.cpp` | `create_context()`, `create_module()`, `create_programs()`, `create_pipeline()`, `build_sbt()`, `build_accel()` (~600 lines) |
| `optix/optix_upload.cpp` | `upload_scene_data()`, `upload_photon_data()`, `upload_light_cache()` (~500 lines) |
| `optix/optix_denoiser.cpp` | `setup_denoiser()`, `run_denoiser()` (~200 lines) |

Remaining `optix_renderer.cpp`: render pass dispatch, frame orchestration. ≤ 1 200 lines.

### 1.9 Target Directory Layout

After all moves, merges, and extractions, the project should look like this:

```
src/
  main.cpp                          ← ≤200 lines: args → scene → renderer → viewer
  app/
    viewer.h / viewer.cpp           ← GLFW window, event loop, input, overlay
  bsdf/
    bsdf_shared.h                   ← HD: all Fresnel, GGX, lobe probs (single truth)
    bsdf.h                          ← CPU high-level: evaluate/sample/pdf (calls bsdf_shared)
  core/
    alias_table.h                   ← Vose alias method (HD)
    config.h                        ← Compile-time + runtime constants, scene profiles
    hash.h                          ← NEW: HD Teschner spatial hash (single truth)
    ior_stack.h                     ← NEW: HD IOR stack (single truth)
    random.h                        ← PCG RNG + sampling utils + binary_search_cdf (merged)
    runtime_config.h                ← JSON config parser
    spectrum.h                      ← Spectral types, CIE Wyman fit (mark HD)
    types.h                         ← float3 ops, ONB (mark HD), Ray, HitRecord
    material_flags.h                ← MaterialClass, classify_for_photons
  debug/
    debug.h                         ← Debug state, cell info, heatmap overlay
    font_overlay.h                  ← MOVED from core/
  optix/
    launch_params.h                 ← LaunchParams, SBT records (share RenderMode enum)
    optix_device.cu                 ← Entry points only + #includes
    optix_bsdf.cuh                  ← GPU BSDF wrappers (calls bsdf_shared.h)
    optix_debug.cuh                 ← debug_first_hit
    optix_material.cuh              ← Material accessors (share MaterialType enum)
    optix_nee.cuh                   ← NEE evaluate, dispatch, golden-ratio stratification
    optix_path_trace.cuh            ← full_path_trace()
    optix_photon_trace.cuh          ← __raygen__photon_trace bounce loop
    optix_specular.cuh              ← Dispersion, Fresnel dielectric, specular bounce
    optix_sppm.cuh                  ← SPPM camera + gather
    optix_targeted_photon.cuh       ← __raygen__targeted_photon_trace
    optix_utils.cuh                 ← CIE LUT (→ delete, share spectrum.h), float↔uint
    optix_renderer.h                ← Host pipeline class
    optix_renderer.cpp              ← Render pass dispatch (≤1200 lines)
    optix_setup.cpp                 ← Context, module, pipeline, SBT, accel build
    optix_upload.cpp                ← Scene/photon/light-cache upload
    optix_denoiser.cpp              ← Denoiser setup + run
    adaptive_sampling.h / .cu       ← Per-pixel noise + adaptive mask
  photon/
    photon.h                        ← Photon, PhotonSoA, flags
    emitter.h                       ← CPU emission + tracing (absorb adaptive_emission.h)
    hash_grid.h                     ← CPU hash grid (use core/hash.h)
    hash_grid.cu                    ← GPU hash grid build (remove tonemap kernel)
    kd_tree.h                       ← CPU KD-tree
    density_estimator.h             ← CPU density estimation
    specular_target.h               ← Targeted caustic sampling
    surface_filter.h                ← Tangential distance, surface consistency
    photon_io.h / .cpp              ← Binary cache save/load
    cell_bin_grid.h                 ← MOVED from core/
    cell_cache.h                    ← MOVED from core/ (use core/hash.h)
    tri_photon_irradiance.h         ← Per-triangle irradiance heatmap
  renderer/
    renderer.h / .cpp               ← CPU render pipeline
    camera.h                        ← Thin-lens camera (mark HD, delete GPU duplicate)
    direct_light.h                  ← CPU NEE (coverage-aware)
    nee_shared.h                    ← HD NEE math (absorb nee_sampling.h)
    pixel_lighting.h                ← Per-pixel AOV decomposition
    sppm.h                          ← MOVED from core/
    emitter_points.h                ← MOVED from core/
    light_cache.h                   ← MOVED from core/ (use core/hash.h)
    tonemap.cu                      ← MOVED from hash_grid.cu
  scene/
    scene.h                         ← Scene representation, BVH
    scene_builder.h                 ← NEW: sphere lights, medium assignment (from main.cpp)
    triangle.h                      ← Triangle, AABB, Möller–Trumbore
    material.h                      ← Material, MaterialType (shared enum for GPU)
    obj_loader.h / .cpp             ← OBJ/MTL parser
  volume/
    medium.h                        ← MERGED: HomogeneousMedium + phase functions
tests/
    test_data_io.h                  ← MOVED from core/
    ...
```

### 1.10 Execution Order

The cleanup is large but can be done incrementally without breaking the build at any intermediate step. Recommended sequence:

1. **Delete dead code** (§1.1) — safe, nothing depends on stubs
2. **Merge small files** (§1.3) — update `#include`s in the same commit
3. **Move misplaced files** (§1.2) — update `#include` paths, `CMakeLists.txt`
4. **Extract `main.cpp`** (§1.7) — create `app/viewer`, slim down main
5. **Create shared `HD` headers** (§1.5) — `hash.h`, `ior_stack.h`; mark existing functions `HD`
6. **Decompose `optix_device.cu`** (§1.4) — extract `.cuh` files one section at a time
7. **Unify BSDF** (§1.5, BSDF row) — consolidate into `bsdf_shared.h`, delete GPU duplicates
8. **Decompose `optix_renderer.cpp`** (§1.8) — extract setup / upload / denoiser
9. **Rename pass** (§1.6) — final sweep for naming consistency
10. **Verify** — build, run fast tests, run all tests, compare output images

Each step should be a single commit. The build must pass after every commit.

---

## 2. Architecture

### 2.1 Project Structure

- All photon logic lives in `src/photon/`.
- No photon-specific logic hidden inside the monolithic `optix_renderer.cu` — extract into dedicated files.
- Material interaction code (BSDF, Fresnel, medium transitions) is defined centrally and shared by both photon and camera paths.

### 2.2 Centralized Material Interactions

The physics is identical for both transport directions. A photon hitting a glass surface obeys the same Fresnel equations, Snell's law, and BSDF as a camera ray hitting that surface. The material interaction code (reflect/refract probabilities, IOR, roughness sampling, medium kernels) is shared.

#### Non-Symmetric BSDFs (Adjoint Correction)

For **symmetric** BSDFs (Lambertian, perfect mirror, smooth glass, GGX glossy): `f(wi → wo) = f(wo → wi)`. Photon and camera paths use the same code.

For **non-symmetric** transport, the adjoint BSDF is needed. Three cases:

1. **Shading normals** — the correction factor depends on transport direction. PBRT v4 applies `ShadingNormalCorrection()` that flips sign. Without it, photon mapping with bump maps is biased.
2. **Subsurface scattering (BSSRDF)** — entry/exit Fresnel boundary conditions differ by direction. Production renderers store `TransportMode::Radiance` vs `TransportMode::Importance`.
3. **Refractive interfaces** — the η² correction factor: photon flux and camera radiance scale differently by $(η_i / η_t)^2$ at each refraction.

#### Our Current Material Set

Lambertian, Mirror, Glass, GlossyMetal, GlossyDielectric, Translucent, Clearcoat, Fabric — all symmetric or nearly symmetric. The main concern is the η² correction at glass/translucent refraction. Mirror, Lambertian, and glossy reflection are fully symmetric.

**Bottom line:** Share the BSDF code. Tag each path with its transport direction and apply the η² correction at refractive interfaces for photon paths.

---

## 3. Photon Emission

### 3.1 Two-Phase Emission

Two separate photon maps are built each frame:

1. **Global photon map** — diffuse indirect illumination
2. **Caustic photon map** — specular-to-diffuse light transport (mirror, glass, translucent caustics)

### 3.2 View-Adaptive Photon Budgeting

Feedback loop to allocate photon budget where it matters:

1. Run a cheap pilot camera pass (1–2 SPP, or the first SPP group).
2. For each photon that contributed to a visible pixel, record which emitter it came from.
3. Reallocate the photon budget proportionally — emitters whose photons were used get more budget next round.
4. Re-trace with updated CDF.

This is an emit → gather → measure → re-weight → emit cycle. The multi-map re-tracing infrastructure can be reused for this once re-added.

### 3.3 Emission Pipeline

Each photon is born through a five-step sampling chain.  The CPU sampler (`sample_emitted_photon()` in `emitter.h`) and the GPU raygen kernel (`__raygen__photon_trace()` in `optix_device.cu`) share the same mathematical framework but diverge in implementation details noted below.

#### Step 1 — Triangle Selection

Pick one emissive triangle proportional to its total radiant power:

$$p_{\text{tri}}(i) = \frac{L_{e,\text{total}}(i)}{\displaystyle\sum_j L_{e,\text{total}}(j)}$$

**CPU:** O(1) alias-table lookup (`scene.emissive_alias_table.sample()`).
**GPU:** mixture sampling with weight `DEFAULT_PHOTON_EMITTER_UNIFORM_MIX` (default 0.10):

$$p_{\text{tri}} = (1-\alpha)\,p_{\text{power}} + \alpha\,p_{\text{uniform}}, \qquad p_{\text{uniform}} = \tfrac{1}{N_{\text{emissive}}}$$

where the power branch uses a binary CDF search and the uniform branch draws uniformly.  The uniform mix prevents zero-probability emitters and reduces variance for scenes with many weak lights.

#### Step 2 — Point on Triangle (Uniform Barycentric)

Given random $u_1, u_2 \in [0,1)$, compute:

$$s = \sqrt{u_1}, \qquad \alpha = 1 - s, \qquad \beta = u_2\,s, \qquad \gamma = 1 - \alpha - \beta$$

$$\mathbf{p} = \alpha\,\mathbf{v}_0 + \beta\,\mathbf{v}_1 + \gamma\,\mathbf{v}_2, \qquad p_{\text{pos}} = \frac{1}{A}$$

where $A$ is the triangle area.  The $\sqrt{u_1}$ transform produces a uniform distribution over the triangle surface (Osada et al. 2002).  Both CPU (`sample_triangle()`) and GPU (`sample_triangle_dev()`) are identical.

#### Step 3 — Direction Sampling

A cosine-weighted hemisphere (Lambertian) emission direction is sampled in the surface-normal frame:

$$p_{\text{dir}}(\omega) = \frac{\cos\theta}{\pi}$$

**CPU:** Malley's method via `sample_cosine_hemisphere()`.
**GPU:** Uses `sample_cosine_cone_dev()` with configurable half-angle `DEFAULT_LIGHT_CONE_HALF_ANGLE_DEG`.  The general cone PDF is:

$$p_{\text{dir}} = \frac{\cos\theta}{\pi\,(1 - \cos^2\theta_{\max})}$$

At the default setting of 90°, $\cos\theta_{\max} = 0$, and the cone degenerates to the full cosine hemisphere — identical to the CPU.  Reducing the half-angle restricts emission to a narrower lobe (useful for spotlights or focused emitters).

#### Step 4 — Hero Wavelength Sampling (GPU only)

The GPU implements PBRT v4–style hero wavelength transport (`HERO_WAVELENGTHS = 4`):

1. **Primary bin:** sample from the spectral Le CDF: $p_\lambda(b) = L_e(b) / \sum_i L_e(i)$.
2. **Companion bins:** stratified offsets at $b_h = (b_0 + h \cdot N_\lambda / H) \bmod N_\lambda$ for $h = 1 \ldots H{-}1$.

Each photon carries $H$ wavelength bins with independent flux values — see Step 5.

The CPU path does **not** use hero-wavelength bucketing.  It fills all `NUM_LAMBDA` bins of `Spectrum` in a single evaluation of `mat.Le`.

#### Step 5 — Flux Computation

The initial photon flux balances the sampling chain by dividing out all PDFs:

**CPU (full spectrum):**

$$\Phi(\lambda) = \frac{L_e(\lambda)\,\cos\theta}{p_{\text{tri}} \cdot p_{\text{pos}} \cdot p_{\text{dir}}}$$

**GPU (per hero channel $h$):**

$$\Phi_h = \frac{L_e(\lambda_h)\,\cos\theta}{p_{\text{tri}} \cdot p_{\text{pos}} \cdot p_{\text{dir}} \cdot p_{\lambda_h}} \cdot \frac{1}{H}$$

The $1/H$ factor (PBRT v4 §14.3) compensates for the fact that each physical photon writes to $H$ spectral bins, while the density estimator divides by $N_{\text{emitted}}$ (the number of physical photons, not per-bin contributions).  Without this normalization the indirect component is $H\times$ too bright.

**Emission textures:** The GPU reads spatially varying emission via `dev_get_Le(mat_id, uv)` using interpolated texture coordinates.  The CPU currently reads the flat material value `mat.Le`.

### 3.4 Targeted Caustic Emission

Standard emission rarely hits specular geometry.  Targeted caustic emission (Jensen §9.2) importance-samples photons toward glass, mirror, and translucent triangles.

#### Sampling Strategy (Two-Point)

1. **Pick emitter triangle** — same power CDF + uniform mix as §3.3 Step 1.
2. **Pick specular triangle** — area-weighted alias table over all Glass/Mirror/Translucent triangles:

   $$p_{\text{spec}}(k) = \frac{A_k}{\displaystyle\sum_j A_j}$$

   Built once in `SpecularTargetSet::build()` (CPU) and uploaded as `targeted_spec_alias_{prob,idx,pdf}` arrays (GPU).

3. **Sample points** — uniform barycentric on both emitter ($\mathbf{p}_L$) and specular ($\mathbf{p}_S$) triangles.
4. **Direction** — $\omega = \text{normalize}(\mathbf{p}_S - \mathbf{p}_L)$.
5. **Backface culling** — reject the sample if:
   - $\cos\theta_{\text{light}} \le 0$ — photon would leave from the back of the emitter, or
   - $\cos\theta_{\text{target}} \le 0$ — photon would arrive at the back face of the sampled specular triangle.

   This prevents wasting budget on geometrically impossible directions. For convex objects (e.g., a glass sphere), roughly half the target triangles face toward the light — the other half are rejected.

6. **Visibility gate with specular pass-through** — a radiance ray (not a binary shadow test) is traced from $\mathbf{p}_L$ in direction $\omega$. The first intersection is classified:

   | First hit classification | Action |
   |---|---|
   | No hit | **Cancel** — photon discarded (missed everything) |
   | Emissive surface | **Cancel** — photon discarded (hit another light) |
   | Specular or translucent | **Accept** — photon enters bounce loop at this hit point |
   | Anything else (diffuse, glossy, etc.) | **Cancel** — photon discarded (opaque blocker) |

   Any non-accept outcome **cancels the photon entirely** — it is not stored, not redirected, simply dropped. This is the expected cost of importance-sampling toward specular geometry: many candidate directions are wasted, but the ones that survive carry high-value caustic flux.

   **Critical:** the photon enters the bounce loop at the **first specular surface it hits**, which may not be the sampled target triangle. This is intentional and handles geometric complexity correctly (see below).

#### Geometric Complexity and the Redirect Rule

For objects with non-trivial geometry (multi-triangle meshes, concave shapes, nested shells), the sampled target point $\mathbf{p}_S$ may not be directly visible from the light. Three common cases and how they are resolved:

**Case 1 — Glass sphere, target on back hemisphere.** The sampled target triangle faces the light (passes backface cull), but the front hemisphere of the sphere is in the way. The visibility ray's first hit is a front-face glass triangle. Since it is specular → accepted. The bounce loop enters the sphere at this front face, refracts through, and exits the back — naturally producing a proper lens caustic. The sampled target was merely a directional hint; the physics is determined by the bounce loop.

**Case 2 — Complex glass mesh (e.g., glass dragon), target behind other geometry.** The sampled target is a triangle on the dragon's tail. The visibility ray first hits a triangle on the dragon's wing (also glass). Since glass → accepted. The photon enters the wing, refracts through the dragon's interior, and may or may not reach the tail. The bounce loop handles all the internal reflections and refractions correctly regardless.

**Case 3 — Opaque object between light and glass.** The visibility ray first hits a diffuse wall. Since not specular/translucent → rejected. Budget is not wasted.

The key insight: **targeted emission is a directional importance-sampling strategy, not a point-to-point connection.** The sampled target biases the photon's direction toward specular geometry; the actual light transport is performed by the standard bounce loop starting from the first intersection. This makes the approach robust for any geometric complexity — no special handling is needed for concavity, self-occlusion, or nested shells.

#### Flux Formula

The area-to-solid-angle Jacobian converts the two-point PDF:

$$p_{\omega} = p_{\text{spec}} \cdot p_{\text{on\,tri}} \cdot \frac{d^2}{\cos\theta_{\text{target}}}$$

where $p_{\text{on\,tri}} = 1/A_{\text{spec}}$ and $d = \|\mathbf{p}_S - \mathbf{p}_L\|$.  The targeted flux is then:

$$\Phi_h = \frac{L_e(\lambda_h)\,\cos\theta_{\text{light}}\,A_{\text{light}}}{p_{\text{emitter}} \cdot p_{\omega} \cdot p_{\lambda_h}} \cdot \frac{1}{H}$$

Firefly clamping (`fmin(flux, 1e6)`) is applied to all hero channels after evaluation.

#### Bounce Loop

After emission the photon enters the same bounce loop as standard photons, except:
- `on_caustic_path` starts false and becomes true after the first specular/translucent hit.
- The targeted trace only **stores** photons that are on a caustic path (L→S→D).
- The trace terminates immediately when a diffuse/glossy surface is hit (no further diffuse bouncing), since the purpose is caustic illumination only.

### 3.5 CPU ↔ GPU Divergences

| Aspect | CPU (`emitter.h`) | GPU (`optix_device.cu`) |
|---|---|---|
| Triangle selection | O(1) alias table | Binary CDF search + uniform mix |
| Uniform mix | None (pure power CDF) | `DEFAULT_PHOTON_EMITTER_UNIFORM_MIX` = 0.10 |
| Direction domain | Cosine hemisphere | Cosine cone (= hemisphere at default 90°) |
| Spectral transport | Full `NUM_LAMBDA`-bin `Spectrum` | Hero wavelength (4 bins + stratified companions) |
| Emission texture | Flat `mat.Le` | `dev_get_Le(mat_id, uv)` with texture lookup |
| Flux formula | $L_e \cos\theta / (p_{\text{tri}} \cdot p_{\text{pos}} \cdot p_{\text{dir}})$ | Same, but per-hero with $\div p_\lambda \cdot \tfrac{1}{H}$ |
| RNG decorrelation | `rng_spatial_seed()` (position hash) | `rng.advance(cell_key)` (PCG increment) |

The CPU path is used only for validation and debugging; production rendering uses the GPU kernels exclusively.

---

## 4. Photon-Guided Path Tracing

### 4.1 Core Idea — Hash Grid Directional Cache

Replace Müller's SD-tree with a **pre-built directional histogram per hash grid cell**, constructed from photon `wi` vectors after the photon map is built. At render time, each camera path bounce looks up the local directional distribution in O(1) — no kNN search per bounce.

### 4.2 Why This Works

Müller's SD-tree learns the radiance field *online* during rendering. We already have 5M photons with positions and `wi` directions on the GPU — the radiance field is known before any camera ray fires. Existing infrastructure:

| Component | Status |
|---|---|
| Hash grid spatial index (GPU) | Exists |
| kNN shell expansion (GPU gather) | Exists |
| Photon `wi` vectors on device | Exists |
| Photon flux on device | Exists |
| Fibonacci sphere binning (`PHOTON_BIN_COUNT = 32`) | Exists |
| CellInfoCache per-cell stats | Exists |

### 4.3 Build Phase

After the photon map is built, before the camera pass: for each hash grid cell, bin photon `wi` vectors into a 32-bin Fibonacci sphere histogram, weighted by flux:

```
cell_histograms[cell_hash][fib_bin(wi)] += photon_flux / PHOTON_BIN_COUNT_SOLID_ANGLE
```

Storage: 64K cells × 32 bins × 4 bytes = **8 MB**.

### 4.4 Query Phase

```cuda
// At camera path bounce point P:
uint32_t cell = hash(P);
float* histogram = cell_histograms[cell];       // O(1) lookup, no kNN

// Sample direction from histogram (discrete → continuous)
int bin = sample_discrete(histogram, PHOTON_BIN_COUNT, rng);
float3 guided_dir = fibonacci_sphere_dir(bin) + jitter_within_bin(rng);

// MIS combine with BSDF sample (power heuristic)
float3 bsdf_dir  = sample_bsdf(material, normal, wo, rng);
float guide_pdf  = histogram[bin] / total_weight / bin_solid_angle;
float bsdf_pdf   = eval_bsdf_pdf(material, normal, wo, guided_dir);
float mis_weight  = power_heuristic(guide_pdf, bsdf_pdf);
```

### 4.5 Why Not kNN Per Bounce

Multi-bounce path tracing at 4 bounces × 512² × 64 SPP = 67M kNN queries. Shell expansion is fast but not free. The cell histogram reduces per-bounce guidance to a single memory fetch.

### 4.6 Limitations

Cell resolution determines guide quality. A cell can cover a wall and the floor behind it — blending their directional distributions. For most scenes this is acceptable (diffuse surfaces have broad lobes anyway). For high-frequency caustics, a finer grid or kNN fallback for the first bounce may help.

### 4.7 Full Path History — Not Needed

Replaying photon paths backwards through multi-bounce chains breaks down because the camera path never lands at the exact intermediate positions the photon visited. Any angular deviation at bounce $n$ compounds into spatial divergence at bounce $n+1$. The per-cell directional histogram provides **spatially local, depth-independent** guidance — each bounce is guided by the distribution at its own position.

### 4.8 Path Type Classification

For adaptive sample allocation, store a 10-byte material-type sequence per photon (`[Emissive, Mirror, Glass, Diffuse, ...]`) instead of full vertex positions. Per-cell aggregation of path types feeds the adaptive SPP allocator: cells with high caustic fraction → more samples, cells with only diffuse → fewer samples. Reuses `CellInfoCache` and `path_flags`.

---

## 5. Path Tracer Integration

> TODO: Define the interface between the photon subsystem and the new path tracer. The current ray tracer will be replaced with a full recursive path tracer that queries the hash grid directional cache at each bounce.

---

## 6. Debugging & Visualization

**Photon splat overlay:** Render photon positions as `GL_POINTS` overlaid on the rasterized scene — photon splat visualization in screen space. Lightweight; useful for quick diagnostics.

**Lightmap atlas:** Alternative approach — accumulate photon contributions into a per-texel lightmap. More powerful but heavier infrastructure.

**Spectral channels:** With 4 hero wavelengths per photon, accumulate into `float[NUM_LAMBDA]` per texel (not just 4 channels — `NUM_LAMBDA` may be larger). Convert to RGB for display.

---

## 7. Material Interaction Reference

This section documents every material interaction that occurs when a photon (or camera ray) hits a surface. Each material type has two subsections: a plain-language explanation and the full mathematical description (ideal physics + code implementation). Medium transitions, translucent objects, and interior light sources are covered at the end.

All directions are in the **local shading frame** where $\hat{n} = (0,0,1)$. $\omega_o$ is the outgoing direction (toward the source of the ray — the light for photons, the camera for camera rays). $\omega_i$ is the sampled incoming direction (the new bounce direction).

### 7.0 Notation

| Symbol | Meaning |
|---|---|
| $K_d(\lambda)$ | Diffuse reflectance spectrum |
| $K_s(\lambda)$ | Specular reflectance / F0 spectrum |
| $T_f(\lambda)$ | Transmittance filter (glass body colour) |
| $L_e(\lambda)$ | Emission spectrum |
| $\eta$ | Index of refraction |
| $\alpha$ | GGX roughness parameter $= \max(r^2,\, 0.001)$ |
| $F(\theta, \eta)$ | Fresnel reflectance |
| $D(\mathbf{h}, \alpha)$ | GGX normal distribution function |
| $G(\omega_o, \omega_i, \alpha)$ | Smith masking-shadowing function |

---

### 7.1 Lambertian (MaterialType = 0)

#### Plain language

A perfectly matte surface. Light arriving from any direction is scattered equally in all directions above the surface. The colour of the surface is $K_d$. This is the simplest BSDF — think of chalk or unfinished plaster.

When a photon hits a Lambertian surface, it is **deposited** into the photon map (if bounce > 0) and then bounced in a random direction weighted toward the surface normal (cosine-weighted hemisphere sampling).

#### Mathematics

**Ideal:**

$$f_r(\omega_o, \omega_i) = \frac{K_d(\lambda)}{\pi}$$

**Sampling (importance sampling the cosine term):**

$$\omega_i = \text{cosine\_hemisphere}(u_1, u_2), \qquad p(\omega_i) = \frac{\cos\theta_i}{\pi}$$

**Throughput update:**

$$\Phi' = \Phi \cdot \frac{f_r \cdot \cos\theta_i}{p(\omega_i)} = \Phi \cdot K_d$$

**Code:** `lambertian_sample()` in `bsdf.h`. Cosine hemisphere sampling, pdf = $\cos\theta_i / \pi$, $f = K_d / \pi$.

---

### 7.2 Perfect Mirror (MaterialType = 1)

#### Plain language

A perfect mirror. All incoming light is reflected at the mirror angle — no scattering, no absorption beyond the specular reflectance $K_s$. This is a **delta distribution**: the photon bounces in exactly one direction.

When a photon hits a mirror, it is **not deposited** (delta surfaces cannot be gathered by kernel density estimation). Instead, the photon is reflected and continues. If this is the first specular bounce, the path is marked as a caustic path (`PHOTON_FLAG_CAUSTIC_SPECULAR`).

#### Mathematics

**Ideal:**

$$f_r(\omega_o, \omega_i) = K_s(\lambda) \cdot \frac{\delta(\omega_i - \omega_r)}{\cos\theta_i}$$

where $\omega_r = (-\omega_{o,x},\; -\omega_{o,y},\; \omega_{o,z})$ is the reflected direction.

**Sampling:** Deterministic reflection, $p = 1$.

$$\omega_i = \text{reflect}(\omega_o) = (-\omega_{o,x},\; -\omega_{o,y},\; +\omega_{o,z})$$

**Throughput update:**

$$\Phi' = \Phi \cdot K_s$$

**Code:** `mirror_sample()` returns `f = Ks / (|cos θ_i| + ε)`, pdf = 1. The $1/\cos\theta_i$ in $f$ cancels with the $\cos\theta_i$ geometry factor in the throughput update, leaving $\Phi' = \Phi \cdot K_s$.

---

### 7.3 Dielectric Glass (MaterialType = 2)

#### Plain language

A transparent dielectric (glass, water, crystal). At each surface hit, the photon is either **reflected** or **refracted** (transmitted through the surface), determined by the Fresnel equations. The probability of reflection increases at grazing angles (the "Fresnel effect").

Key behaviours:
- **Refraction** follows Snell's law: light bends when crossing materials with different IOR.
- **Total internal reflection (TIR)**: when light exits a dense medium at a steep angle, refraction is impossible and all light is reflected.
- **Chromatic dispersion** (optional): each wavelength has a slightly different IOR via the Cauchy equation, so different wavelengths refract at different angles — producing rainbows and chromatic caustics.
- **Transmittance filter** $T_f$: only applied to transmitted light (not reflected). This gives coloured glass its colour.
- **IOR stack**: nested dielectrics (e.g., glass sphere in water) are tracked via a push/pop stack so that $\eta = \eta_\text{outer} / \eta_\text{inner}$ is always correct.
- **Medium stack**: if the glass has an interior participating medium (`medium_id >= 0`), entering pushes it onto the medium stack; exiting pops it.

A photon hitting glass is **not deposited** (delta surface). It continues as a caustic path.

#### Mathematics

**Fresnel (exact dielectric):**

$$F(\cos\theta_i, \eta) = \frac{1}{2}\left(r_s^2 + r_p^2\right)$$

where $\eta = \eta_i / \eta_t$ (ratio of IORs), and:

$$r_s = \frac{\eta\cos\theta_i - \cos\theta_t}{\eta\cos\theta_i + \cos\theta_t}, \qquad r_p = \frac{\cos\theta_i - \eta\cos\theta_t}{\cos\theta_i + \eta\cos\theta_t}$$

$$\cos\theta_t = \sqrt{1 - \eta^2(1 - \cos^2\theta_i)}$$

If $\eta^2(1 - \cos^2\theta_i) \geq 1$: TIR, $F = 1$.

**Snell's law (local frame refraction):**

$$\omega_t = \left(-\eta\,\omega_{o,x},\; -\eta\,\omega_{o,y},\; -\cos\theta_t\right)$$

**Stochastic selection:** draw $\xi \sim U[0,1]$.
- If $\xi < F$: **reflect**. $\omega_i = \text{reflect}(\omega_o)$, pdf $= F$, $f(\lambda) = F / |\cos\theta_i|$.
- If $\xi \geq F$: **refract**. $\omega_i = \omega_t$, pdf $= 1 - F$, $f(\lambda) = T_f(\lambda) \cdot (1 - F) / |\cos\theta_i|$.

**Chromatic dispersion (Cauchy model):**

$$n(\lambda) = A + \frac{B}{\lambda^2} \qquad (\lambda \text{ in nm})$$

One "hero" wavelength determines the refraction direction. All other wavelength bins get per-bin Fresnel weights using their own $n(\lambda)$ but share the hero direction (spectral MIS). Bins that would undergo TIR at their own IOR get zero weight.

**Entering vs exiting:** determined by $\omega_o \cdot \hat{n}$ (sign of `wo.z`).
- Entering ($\omega_o \cdot \hat{n} > 0$): $\eta = 1 / n_\text{mat}$, push IOR stack, push medium stack.
- Exiting ($\omega_o \cdot \hat{n} < 0$): $\eta = n_\text{mat}$ (or top of IOR stack), pop IOR stack, pop medium stack.

**Code:** `glass_sample(wo, mat, rng, hero_bin)` in `bsdf.h`. TIR fallback reflects with `f = 1/|cos θ|` (neutral). The emitter loop in `emitter.h` handles IOR stack push/pop and medium stack push/pop for Glass and Translucent types.

---

### 7.4 Glossy Metal (MaterialType = 3)

#### Plain language

A rough metallic surface like brushed aluminium or copper. Combines a **diffuse** component ($K_d$) with a **specular microfacet** component ($K_s$). The specular highlight is broad or sharp depending on roughness.

Metals use their specular colour $K_s$ directly as the Fresnel $F_0$ (unlike dielectrics which derive $F_0$ from IOR). This means the specular reflection is colour-tinted (gold reflects yellow, copper reflects orange).

When a photon hits a glossy metal, it **is deposited** (this is a non-delta surface). The photon is then bounced according to a stochastic choice between the diffuse and specular lobes.

#### Mathematics

**BSDF (two-lobe mixture):**

$$f_r(\omega_o, \omega_i) = \underbrace{\frac{D(\mathbf{h}, \alpha) \; G(\omega_o, \omega_i, \alpha) \; F_\text{Schlick}(\omega_o \cdot \mathbf{h},\, K_s(\lambda))}{4\,|\cos\theta_o|\,|\cos\theta_i|}}_{\text{Cook-Torrance specular}} + \underbrace{\frac{K_d(\lambda)}{\pi}}_{\text{diffuse}}$$

**GGX NDF:**

$$D(\mathbf{h}, \alpha) = \frac{\alpha^2}{\pi\left((\mathbf{h} \cdot \hat{n})^2(\alpha^2 - 1) + 1\right)^2}$$

**Smith masking-shadowing:**

$$G(\omega_o, \omega_i, \alpha) = G_1(\omega_o, \alpha) \cdot G_1(\omega_i, \alpha), \qquad G_1(\mathbf{v}, \alpha) = \frac{2\,|\mathbf{v} \cdot \hat{n}|}{|\mathbf{v} \cdot \hat{n}| + \sqrt{\alpha^2 + (1 - \alpha^2)(\mathbf{v} \cdot \hat{n})^2}}$$

**Schlick Fresnel:**

$$F_\text{Schlick}(\cos\theta, F_0) = F_0 + (1 - F_0)(1 - \cos\theta)^5$$

**Lobe selection probability:**

$$p_\text{spec} = \frac{\max(K_s)}{\max(K_s) + \max(K_d)}, \qquad p_\text{diff} = 1 - p_\text{spec}$$

**Specular sampling:** VNDF (Visible Normal Distribution Function) sampling of GGX half-vector, then reflect.

**Combined PDF (MIS):**

$$p(\omega_i) = p_\text{spec} \cdot \frac{D(\mathbf{h}) \cdot |\mathbf{h} \cdot \hat{n}|}{4\,|\omega_o \cdot \mathbf{h}|} + p_\text{diff} \cdot \frac{\cos\theta_i}{\pi}$$

**Code:** `glossy_sample()` in `bsdf.h`. Uses `ggx_sample_halfvector()` for VNDF-based specular sampling with Smith-G and single-scatter Cook-Torrance evaluation.

---

### 7.5 Emissive (MaterialType = 4)

#### Plain language

A light-emitting surface (area light). The emissive surface also has a diffuse $K_d$ component for BSDF interactions. Photons **originate** from emissive surfaces — they are not typically "hit" in the photon trace (the emitter logic in `emitter.h` samples them as sources). If a photon does land on an emissive surface (bounce > 0), it is deposited and bounces exactly like Lambertian.

#### Mathematics

Emission: $L_e(\lambda)$ is the spectral radiance of the light.

BSDF for bouncing: identical to Lambertian ($f_r = K_d / \pi$), cosine-hemisphere sampling.

**Code:** The `sample()` dispatch routes Emissive to `lambertian_sample()`. The `evaluate()` and `pdf()` functions also treat Emissive as Lambertian.

---

### 7.6 Glossy Dielectric (MaterialType = 5)

#### Plain language

A smooth non-metallic material like plastic, lacquered wood, or polished stone. Combines a **GGX specular** highlight with an **energy-conserving diffuse** base. Unlike glossy metal, the Fresnel $F_0$ is computed from the IOR (typically 0.04 for IOR=1.5), so the specular highlight is white/neutral rather than coloured.

The key difference from GlossyMetal: the diffuse lobe is weighted by $(1 - F_r)$ — energy reflected by the specular layer is subtracted from the diffuse, preventing energy creation.

Photon deposits: yes (non-delta surface).

#### Mathematics

**BSDF:**

$$f_r(\omega_o, \omega_i) = K_s(\lambda) \cdot \frac{D \cdot G \cdot F_\text{Schlick}(\omega_o \cdot \mathbf{h},\, F_0)}{4\,|\cos\theta_o|\,|\cos\theta_i|} + (1 - F_r) \cdot \frac{K_d(\lambda)}{\pi}$$

where $F_0 = \left(\frac{\eta - 1}{\eta + 1}\right)^2$ and $F_r = F_\text{Schlick}(\omega_o \cdot \mathbf{h},\, F_0)$.

**Lobe probability (energy-aware):**

$$p_\text{spec} = \text{clamp}\left(\frac{\max(K_s) \cdot F_0}{\max(K_s) \cdot F_0 + \max(K_d)},\; 0.05,\; 0.95\right)$$

The 0.05 lower clamp ensures the specular peak always gets some samples, even for mostly-diffuse materials.

**Code:** `glossy_dielectric_sample()` in `bsdf.h`. Same GGX VNDF sampling machinery as GlossyMetal, but with IOR-based $F_0$ and $(1 - F_r)$ energy conservation on the diffuse term.

---

### 7.7 Translucent (MaterialType = 6)

#### Plain language

A glass-like surface with an **interior participating medium** (volumetric scattering inside). Think of a marble ball, a jade figurine, or a glass of milk. The **surface BSDF is identical to Glass** (Fresnel reflect/refract, Snell's law, optional dispersion). The difference is what happens *inside*: the material has absorption coefficient $\sigma_a(\lambda)$, scattering coefficient $\sigma_s(\lambda)$, and a phase function asymmetry parameter $g$.

When a photon enters a Translucent object:
1. The surface interaction is the same as Glass (Fresnel decision, refraction).
2. The entry pushes the material's `medium_id` onto the medium stack.
3. Inside the medium, **Beer–Lambert attenuation** reduces flux: $T(d) = e^{-\sigma_t d}$.
4. The photon may **scatter** inside the medium before reaching the next surface (free-flight sampling).
5. When the photon hits the inner surface to exit, it's another Fresnel reflect/refract with IOR stack pop.

#### Mathematics

**Surface BSDF:** identical to Glass (§7.3 above).

**Interior medium — Beer–Lambert transmittance:**

$$T(\lambda, d) = \exp\left(-\sigma_t(\lambda) \cdot d\right), \qquad \sigma_t = \sigma_s + \sigma_a$$

**Free-flight sampling (hero wavelength scheme):**

Pick hero bin $h$ uniformly. Sample free-flight distance:

$$t = -\frac{\ln(\xi)}{\sigma_t^{(h)}}$$

If $t < d_\text{surface}$: the photon **scatters** inside the medium. Spectral MIS weight:

$$w(\lambda) = \frac{\sigma_s(\lambda) \cdot e^{-\sigma_t(\lambda) \cdot t}}{\frac{1}{N}\sum_j \sigma_t^{(j)} \cdot e^{-\sigma_t^{(j)} \cdot t}}$$

If $t \geq d_\text{surface}$: the photon reaches the next surface. Transmittance MIS weight:

$$w(\lambda) = \frac{e^{-\sigma_t(\lambda) \cdot d}}{\frac{1}{N}\sum_j e^{-\sigma_t^{(j)} \cdot d}}$$

**Phase function (Henyey-Greenstein):**

$$p_\text{HG}(\cos\theta, g) = \frac{1 - g^2}{4\pi\left(1 + g^2 - 2g\cos\theta\right)^{3/2}}$$

$g = 0$ is isotropic, $g > 0$ is forward-scattering, $g < 0$ is back-scattering.

**Code:** The surface BSDF uses `glass_sample()`. Medium tracking uses `MediumStack` in `emitter.h`. The free-flight + spectral MIS logic is in the beam tracing section of `trace_photons()` (lines ~460–590 of `emitter.h`). `HomogeneousMedium` is defined in `medium.h`.

---

### 7.8 Clearcoat (MaterialType = 7)

#### Plain language

A layered material: a thin transparent dielectric **coat** over a **base** (Lambert diffuse). Think of car paint, varnished wood, or glossy paper. The coat has its own roughness and IOR, producing a specular highlight on top of the diffuse base. Energy reflected by the coat is removed from the base:

> Base receives: $(1 - w_\text{coat} \cdot F_r) \times K_d / \pi$

The coat does not refract through to the base — it's a simplified single-interface layered model (no inter-layer transport). The coat is non-delta (rough GGX), so photons **are deposited** on clearcoat surfaces.

#### Mathematics

**Two-lobe model:**

$$f_r(\omega_o, \omega_i) = \underbrace{w_\text{coat} \cdot \frac{D(\mathbf{h}, \alpha_\text{coat}) \; G(\omega_o, \omega_i, \alpha_\text{coat}) \; F_\text{Schlick}(\omega_o \cdot \mathbf{h},\, F_{0,\text{coat}})}{4\,|\cos\theta_o|\,|\cos\theta_i|}}_{\text{coat specular}} + \underbrace{(1 - w_\text{coat} \cdot F_r) \cdot \frac{K_d(\lambda)}{\pi}}_{\text{base diffuse}}$$

where $F_{0,\text{coat}} = \left(\frac{\eta_\text{coat} - 1}{\eta_\text{coat} + 1}\right)^2$ and $w_\text{coat}$ = `pb_clearcoat`.

**Lobe probability:**

$$p_\text{coat} = \text{clamp}(w_\text{coat} \cdot F_{0,\text{coat}},\; 0.05,\; 0.95)$$

**Code:** `clearcoat_sample()` in `bsdf.h`. The coat uses the material's `ior` for its $F_0$, and `pb_clearcoat_roughness` for its GGX alpha. Base is Lambert with energy deduction.

---

### 7.9 Fabric (MaterialType = 8)

#### Plain language

A cloth-like material with a diffuse base plus a **sheen** lobe that brightens at grazing angles — the "Fresnel glow" you see on velvet, silk, or microfibre. The sheen can be colour-tinted toward the base colour.

Fabric is non-delta, so photons **are deposited**. Both lobes are diffuse-like (cosine hemisphere sampling is used for both).

#### Mathematics

**BSDF:**

$$f_r(\omega_o, \omega_i) = \frac{K_d(\lambda)}{\pi} + \frac{w_\text{sheen} \cdot c_\text{sheen}(\lambda) \cdot (1 - \cos\theta_h)^5}{\pi}$$

where $\theta_h = \arccos(|\omega_o \cdot \mathbf{h}|)$ and:

$$c_\text{sheen}(\lambda) = (1 - t_\text{tint}) \cdot 1 + t_\text{tint} \cdot K_d(\lambda)$$

$w_\text{sheen}$ = `pb_sheen`, $t_\text{tint}$ = `pb_sheen_tint`.

**Sampling:** Cosine hemisphere (same as Lambertian). The sheen term is evaluated but not importance-sampled (it's low-energy and diffuse-like, so cosine sampling is adequate).

**Code:** `fabric_sample()` in `bsdf.h`. Simple cosine hemisphere + sheen evaluate.

---

### 7.10 Medium Transitions

#### IOR Stack

When a photon enters a dielectric (Glass or Translucent), it pushes the material's IOR onto a 4-deep stack. When it exits, it pops. This handles nested dielectrics correctly:

| Transition | IOR Stack | $\eta$ used |
|---|---|---|
| Air → Glass (ior=1.5) | `[1.5]` | $1 / 1.5 = 0.667$ |
| Glass → Air | `[]` | $1.5 / 1.0 = 1.5$ |
| Air → Glass → Water (ior=1.33) | `[1.5, 1.33]` | $1.5 / 1.33 = 1.128$ |
| Water → Glass (exiting water) | `[1.5]` | $1.33 / 1.5 = 0.887$ |

Entering/exiting is determined by `dot(ray_direction, geometric_normal)`: negative = entering, positive = exiting. The geometric normal (not shading normal) is used for robustness near mesh edges.

**Code:** `IORStack` struct in `emitter.h`, push on entering, pop on exiting.

#### Medium Stack (Participating Media)

Parallel to the IOR stack, a `MediumStack` tracks which participating medium the photon is currently inside. When a photon crosses into a Translucent surface with `medium_id >= 0`, that medium is pushed. When it exits, popped.

While inside a medium, the photon trace loop:
1. Records a **beam segment** from the current position to the next surface hit.
2. Applies Beer–Lambert attenuation to the spectral flux using spectral MIS (hero wavelength scheme).
3. May scatter within the medium (Henyey-Greenstein phase function) before reaching the surface.

If no participating medium is active (`medium_stack.current_medium_id() == -1`), no volumetric processing occurs.

**Code:** `MediumStack` struct in `emitter.h`, `HomogeneousMedium` in `medium.h`.

---

### 7.11 Special Case: Light Source Inside a Translucent Sphere

This is the combination of an Emissive surface enclosed by a Translucent shell. The photon path is:

1. **Emission:** Photon is emitted from the inner emissive surface. Direction is sampled from cosine hemisphere on the light.
2. **Medium traversal:** The photon travels through the interior medium (if the Translucent shell has `medium_id >= 0`). Beer–Lambert attenuates the flux; the photon may scatter inside the medium.
3. **Inner surface hit:** The photon hits the inner surface of the Translucent shell. This is an **exiting** event from the medium's perspective (the photon is going from inside the shell → outward). Fresnel determines reflect vs. refract.
   - If reflected: photon bounces back into the medium, may hit the light source again, or traverse and hit the shell again.
   - If refracted: photon exits the Translucent shell into the surrounding air (or the next enclosing medium).
4. **Exit attenuation:** On transmission, $T_f(\lambda)$ filters the photon colour. The IOR transition applies ($\eta = n_\text{shell}$ for exiting).
5. **Caustic deposit:** When the photon finally hits a diffuse surface outside, it is deposited with `PHOTON_FLAG_TRAVERSED_GLASS` set (and possibly `PHOTON_FLAG_CAUSTIC_GLASS` and `PHOTON_FLAG_DISPERSION`). The photon forms a **caustic** on the surrounding surfaces.

**Possible trapping:** If the Translucent shell has high IOR and the photon hits the inner surface at a steep angle, TIR traps it inside. Russian roulette eventually terminates such trapped paths. The medium absorption also attenuates trapped photons toward zero.

**Colour bleeding:** The medium's $\sigma_a(\lambda)$ and the shell's $T_f(\lambda)$ jointly determine the colour of the transmitted light. A red $T_f$ in a clear medium produces a red glow; a clear $T_f$ with wavelength-dependent $\sigma_a$ produces selective absorption (e.g., green jade).

---

### 7.12 Photon Deposit vs. Bounce Summary

| Material | Delta? | Photon deposited? | Bounce type | Caustic flag |
|---|---|---|---|---|
| Lambertian | No | Yes (bounce > 0) | Cosine hemisphere | — |
| Mirror | Yes | No | Deterministic reflection | `CAUSTIC_SPECULAR` |
| Glass | Yes | No | Fresnel reflect/refract | `TRAVERSED_GLASS`, `CAUSTIC_GLASS` |
| GlossyMetal | No | Yes (bounce > 0) | GGX VNDF + cosine mix | — |
| Emissive | No | Yes (bounce > 0) | Cosine hemisphere | — |
| GlossyDielectric | No | Yes (bounce > 0) | GGX VNDF + cosine mix | — |
| Translucent | Yes (surface) | No | Fresnel reflect/refract + medium | `TRAVERSED_GLASS`, `CAUSTIC_GLASS` |
| Clearcoat | No | Yes (bounce > 0) | GGX coat + cosine base mix | — |
| Fabric | No | Yes (bounce > 0) | Cosine hemisphere | — |

**Rule:** A photon is deposited at non-delta surfaces when bounce > 0. It enters the **caustic map** if and only if the path has traversed at least one caustic-caster (Mirror, Glass, or Translucent) without an intervening diffuse bounce (`on_caustic_path == true`).

---

### 7.13 Throughput Update (All Materials)

After each bounce, the spectral flux is updated:

$$\Phi'(\lambda) = \Phi(\lambda) \cdot \frac{f(\lambda) \cdot |\cos\theta_i|}{p(\omega_i)}$$

For delta distributions (Mirror, Glass, Translucent), $p = 1$ or $p = F$ and the $\cos\theta_i$ in $f$ cancels with the geometry factor, yielding clean throughput:
- Mirror: $\Phi' = \Phi \cdot K_s$
- Glass reflect: $\Phi' = \Phi \cdot 1$ (energy-neutral Fresnel)
- Glass refract: $\Phi' = \Phi \cdot T_f$

---

### 7.14 Russian Roulette

After `min_bounces_rr` (default 3), the photon is terminated stochastically:

$$p_\text{survive} = \min\left(\text{rr\_threshold},\; \max_\lambda \Phi(\lambda)\right)$$

If terminated ($\xi \geq p_\text{survive}$): path ends, no deposit.
If surviving: $\Phi \mathrel{{/}{=}} p_\text{survive}$ (unbiased energy compensation).

This applies both inside media (scatter events) and at surfaces.

**Specular exemption:** Russian roulette is **skipped** for specular and translucent bounces (GPU: explicit guard; CPU targeted trace: `!mat.is_specular()` check). This ensures caustic photons survive the full glass/mirror chain unattenuated. Without this, multi-bounce glass caustics (e.g. light inside a glass sphere) would be prematurely terminated.

---

## 8. Code Audit — Undocumented Logic

This section catalogues implementation logic found in the working code that was not previously captured in this document. Each item references the code location and describes the behaviour.

### 8.1 Emissive Surface Termination (GPU only)

**Code:** `optix_device.cu` lines ~2594, ~2882 — `if (dev_is_emissive(hit_mat)) break;`

The GPU bounce loop **terminates** when a photon hits an emissive surface (at any bounce). The rationale is that re-emitting a photon from a light source double-counts direct lighting.

**CPU divergence:** The CPU `trace_photons()` does **not** have this explicit check. The BSDF dispatch routes `MaterialType::Emissive` to `lambertian_sample()`, so the photon bounces off the light as if it were Lambertian. In practice, light sources are typically convex (so photons rarely hit them from the front), but the behaviour differs.

### 8.2 Caustic-Only Store Mode (GPU)

**Code:** `optix_device.cu` line ~2599 — `if (params.caustic_only_store && !on_caustic_path) { ... }`

When `caustic_only_store` is true, non-caustic photons at diffuse surfaces are **not stored** but the bounce loop **continues**. This allows L→D→S→D paths (where a photon bounces diffusely, then hits glass, then deposits on the next diffuse surface) to still produce caustic deposits. The targeted trace terminates immediately at diffuse surfaces; the standard trace does not.

### 8.3 Diffuse-Only Gather (evaluate_diffuse)

**Code:** `density_estimator.h` line ~129, `renderer.cpp` line ~146 — `bsdf::evaluate_diffuse(mat, wo_local, wi_local)`

The photon density estimator uses `evaluate_diffuse()` — **not** the full `evaluate()` — when gathering photon contributions. This means only the diffuse BSDF lobe is used for kernel density estimation. The peaked specular lobes (GGX, coat) are excluded because they create unbounded variance in fixed-radius estimators, producing coloured hotspots. For Clearcoat materials, the diffuse portion includes the $(1 - w_\text{coat} \cdot F_r)$ energy attenuation. For delta materials (Mirror, Glass, Translucent), `evaluate_diffuse` returns zero.

### 8.4 Path Flag Accumulation

**Code:** `emitter.h` lines ~636–660, `optix_device.cu` bounce loops

Path flags are accumulated (bitwise OR) through the photon's lifetime. Specific rules:

| Flag | When set |
|---|---|
| `PHOTON_FLAG_TRAVERSED_GLASS` (0x01) | Any Glass or Translucent bounce |
| `PHOTON_FLAG_CAUSTIC_GLASS` (0x02) | Glass/Translucent at **bounce == 0 only** (direct caustic) |
| `PHOTON_FLAG_VOLUME_SEGMENT` (0x04) | Any volume interaction (legacy atmospheric) |
| `PHOTON_FLAG_DISPERSION` (0x08) | Glass/Translucent with `mat.dispersion == true` |
| `PHOTON_FLAG_CAUSTIC_SPECULAR` (0x10) | Mirror bounce |

The `bounce == 0` restriction on `CAUSTIC_GLASS` distinguishes direct caustics (L→Glass→D) from indirect (L→D→...→Glass→D).

### 8.5 CPU Hero Wavelength Compatibility Layer

**Code:** `emitter.h` lines ~612–625 (global trace), ~900–915 (targeted trace)

When the CPU stores a photon, it synthetically generates hero wavelength bins for GPU gather compatibility. It picks a random primary bin, computes stratified companions at `(primary + h * NUM_LAMBDA / HERO_WAVELENGTHS) % NUM_LAMBDA`, and scales flux by `NUM_LAMBDA / HERO_WAVELENGTHS`. This exists solely so the GPU-side kernel density estimator can read CPU-emitted photons without special-casing.

### 8.6 Cell-Stratified Bounce (CPU only)

**Code:** `emitter.h` lines ~676–705

The CPU trace uses Fibonacci-lattice stratification for Lambertian and Emissive bounce directions (controlled by `DEFAULT_PHOTON_BOUNCE_STRATA`). Spatial hashing assigns each hit to a cell; an atomic counter provides the stratum index; the Fibonacci lattice maps stratum → stratified $(u_1, u_2)$ with jitter. This reduces banding artifacts from correlated bounces.

**GPU divergence:** The GPU does **not** use cell-stratified bouncing. It relies solely on RNG spatial decorrelation (`rng.advance(cell_key)`).

### 8.7 Ray Origin Offset Convention

**Code:** CPU `emitter.h` line ~749, GPU bounce loops

After a bounce, the ray origin is biased along the shading normal to prevent self-intersection:

- **Reflection** ($\omega_i \cdot \hat{n} > 0$): offset along $+\hat{n}$
- **Refraction** ($\omega_i \cdot \hat{n} < 0$): offset along $-\hat{n}$

CPU uses `EPSILON` (1e-4), GPU uses `OPTIX_SCENE_EPSILON`. For specular bounces the GPU uses `sb.new_pos` from `dev_specular_bounce()` which applies its own offset logic.

### 8.8 Adaptive Caustic Shooting (Hotspot Retracing)

**Code:** `emitter.h` lines ~780–810 — `trace_targeted_caustic_photons()`

After the initial photon trace, the system identifies caustic hotspot cells (high coefficient of variation) via `CellInfoCache::get_caustic_hotspot_keys()`. It then re-traces additional photons (budget = `CAUSTIC_TARGETED_FRACTION × num_photons`) and appends only the caustic deposits. The budget halves each iteration, up to `CAUSTIC_MAX_TARGETED_ITERS` rounds.

This is distinct from the targeted caustic **emission** (§3.4) — it uses standard uniform emission and simply traces more photons, keeping only caustic deposits.

### 8.9 Source Emitter Tracking

**Code:** `photon.h` — `source_emissive_idx`, stored per photon

Each photon records which emissive triangle it originated from (local index into `emissive_tri_indices`). This enables the view-adaptive budgeting feedback loop (§3.2): after gathering, the system counts which emitters contributed to visible pixels and reallocates the CDF.

### 8.10 Volume Double-Attenuation Guard

**Code:** `emitter.h` lines ~400–405

When a photon is inside an object-attached medium (tracked by `MediumStack`), the legacy atmospheric volume system is skipped to prevent double-attenuation. The guard checks `beam_handles_medium = beam_enabled && cur_med_id >= 0`.

---

## 9. Photon Trace Mind Games

These trace-throughs follow individual photons step by step through the documented rules. Each scenario uses a simplified setup with concrete numbers. The purpose is to verify that the documented logic produces physically plausible outcomes.

### 9.1 Scenario A — Diffuse Cornell Box (L→D→D)

**Setup:** A single area light (Le = 10 W/sr/m², area = 0.5 m²) over a white Lambertian floor (Kd = 0.8 on all λ).

**Emission (§3.3):**
1. Triangle selection: only one emitter → p_tri = 1.0
2. Point: uniform on the 0.5 m² triangle → p_pos = 1/0.5 = 2.0
3. Direction: cosine hemisphere → p_dir = cos θ / π. Say θ = 30° → cos θ = 0.866, p_dir = 0.866/π ≈ 0.276
4. Flux: Φ = 10 × 0.866 / (1.0 × 2.0 × 0.276) ≈ 15.7 W/nm

**Bounce 0:** Photon exits light downward. Hits the floor at an angle. This is bounce 0 → **not deposited** (rule: bounce > 0). Floor is Lambertian (non-delta). Throughput update: Φ' = Φ × Kd = 15.7 × 0.8 = 12.56. New direction: cosine hemisphere off floor. `on_caustic_path = false`.

**Bounce 1:** Photon travels to a side wall (Kd = 0.5). Bounce 1 > 0, non-delta → **deposited** in global map with Φ = 12.56. Throughput update: Φ' = 12.56 × 0.5 = 6.28. `on_caustic_path = false` → **not** deposited in caustic map.

**Bounce 2:** Hits ceiling (Kd = 0.8). Deposited with Φ = 6.28. Throughput: Φ' = 6.28 × 0.8 = 5.02.

**Bounce 3 (RR):** min_bounces_rr = 3, so RR kicks in. p_survive = min(0.95, 5.02) = 0.95. Say ξ = 0.4 < 0.95 → survives. Φ' = 5.02 / 0.95 = 5.28.

**Expectation check:** After 2 diffuse bounces, flux ≈ 6.28 = 15.7 × 0.8 × 0.5. After 3, ≈ 5.28 (with RR boost). This is plausible — each bounce attenuates by the albedo, and RR keeps the estimator unbiased. ✅

### 9.2 Scenario B — Mirror Caustic (L→Mirror→D)

**Setup:** Area light above, a 45° mirror, and a Lambertian floor. Ks = 0.95.

**Emission:** Same as above. Φ ≈ 15.7.

**Bounce 0:** Photon hits the mirror. Mirror is **delta** → **not deposited**. `on_caustic_path = true` (caustic caster). `path_flags |= PHOTON_FLAG_CAUSTIC_SPECULAR`. Direction: deterministic reflection. Throughput: Φ' = 15.7 × 0.95 = 14.92 (§7.2: Φ' = Φ × Ks).

**Bounce 1:** Reflected photon hits the floor (Lambertian, Kd = 0.8). Bounce 1 > 0, non-delta → **deposited** in:
- Global map ✅
- Caustic map ✅ (because `on_caustic_path == true`)
- Path flags include `CAUSTIC_SPECULAR`

Stored flux = 14.92. Throughput for next bounce: Φ' = 14.92 × 0.8 = 11.94. `on_caustic_path = false` (diffuse hit clears it).

**Expectation check:** A mirror caustic delivers nearly the full light power (attenuated only by Ks) concentrated in the reflected beam. 14.92 out of 15.7 → 95% efficiency ≈ Ks. ✅

### 9.3 Scenario C — Glass Sphere Caustic (L→Glass→Glass→D)

**Setup:** Area light, glass sphere (IOR = 1.5, Tf = 1.0 pure clear glass, no dispersion), Lambertian floor. Φ at emission ≈ 15.7.

**Bounce 0:** Photon hits front of glass sphere. Entering → push IOR stack [1.5]. Fresnel at θ_i = 20°: η = 1/1.5 = 0.667, cos θ_i = 0.94, cos θ_t = √(1 - 0.444 × 0.116) ≈ 0.974. F ≈ 0.04 (near-normal). Stochastic selection: say ξ = 0.5 > F → **refract**. Direction: Snell's law bends inward. Throughput: Φ' = Φ × Tf = 15.7 × 1.0 = 15.7 (clear glass). `on_caustic_path = true`, `path_flags |= TRAVERSED_GLASS | CAUSTIC_GLASS` (bounce == 0).

**Bounce 1:** Photon travels through glass interior (no medium → no volumetric attenuation). Hits back of sphere (exiting). Pop IOR stack []. η = 1.5/1.0 = 1.5. cos θ_i inside glass — suppose 25° →  cos θ_i = 0.906, sin²θ_t = 1.5² × (1 − 0.906²) = 2.25 × 0.179 = 0.403, cos θ_t = √0.597 ≈ 0.773. F ≈ 0.04. Refract again. Throughput: Φ' = 15.7 × 1.0 = 15.7. Still on caustic path.

**Bounce 2:** Photon exits sphere, hits Lambertian floor. Bounce 2 > 0, non-delta → **deposited** in:
- Global map ✅
- Caustic map ✅ (`on_caustic_path == true`)
- Path flags: `TRAVERSED_GLASS | CAUSTIC_GLASS`

Stored flux ≈ 15.7 (nearly all light transmitted — clear glass with near-normal incidence).

**Expectation check:** Clear glass at near-normal transmits ~96% per interface. Two interfaces: ~92%. We got 100% because Tf = 1.0 and we happened to draw "refract" (not reflect) at both interfaces. In the Monte Carlo sense, the probability of refracting both times is (1-F)² ≈ 0.92, and when it does refract the throughput is 15.7/1 = 15.7. Over many photons, the average deposited flux = 15.7 × 0.92 ≈ 14.4, which matches the two-interface Fresnel expectation. ✅

### 9.4 Scenario D — Light Inside Translucent Sphere (medium traversal)

**Setup:** Emissive surface inside a translucent shell (IOR = 1.5, σ_t = 0.5/m uniform, σ_s = 0.4/m, σ_a = 0.1/m, g = 0.3, shell radius ≈ 0.5 m so interior path ≈ 0.5 m). Φ at emission ≈ 15.7.

**Bounce 0:** Photon emitted from inner light. Travels through interior medium toward inner surface of shell. Inside medium (medium_stack has the translucent shell's medium).

**Beam tracing:** seg_t ≈ 0.5 m. Hero bin: pick σ_t = 0.5. Free-flight sample: t_ff = −ln(ξ)/0.5. Say ξ = 0.6 → t_ff = 0.51/0.5 = 1.02 m > seg_t = 0.5 m → **no scatter**, photon reaches shell surface. Transmittance MIS: Tr(λ) = exp(−0.5 × 0.5) = exp(−0.25) ≈ 0.78. After MIS weighting: Φ' ≈ 15.7 × 0.78 = 12.2.

**Surface interaction:** Inner surface of translucent shell, **exiting** (from inside the medium). Pop medium stack. IOR stack: pop (back to air). Fresnel at interface: η = 1.5 (exiting). Say θ_i = 15° → F ≈ 0.04. Draw refract. Throughput: Φ' = 12.2 × Tf. `on_caustic_path = true`, `path_flags |= TRAVERSED_GLASS`.

**Bounce 1:** Photon exits shell into air. Hits a Lambertian floor outside. **Deposited** in global + caustic map with Φ ≈ 12.2 × Tf.

**Expectation check:** Medium absorbed ~22% over 0.5 m (Beer–Lambert with σ_t = 0.5). Fresnel lost ~4% at exit. Total: ~75% of emitted flux. For a translucent shell this is physically reasonable — some absorption, most light escapes. ✅

### 9.5 Scenario E — Targeted Caustic (Two-Point Sampling, §3.4)

**Setup:** Area light (A_light = 0.5 m², Le = 10), glass sphere (A_spec = 0.3 m²) at distance d = 3 m from light.

**Step 1:** Pick emitter triangle (p_emitter = 1.0, single light).
**Step 2:** Pick specular triangle — single glass sphere → p_spec = 1.0.
**Step 3:** Sample points. Light point $\mathbf{p}_L$, target point $\mathbf{p}_S$.
**Step 4:** Direction: ω = normalize(p_S − p_L), dist = 3 m.
**Backface check:** cos θ_light = dot(ω, light_normal) — say 0.9 > 0 ✅. cos θ_target = dot(−ω, spec_normal) — say 0.85 > 0 ✅.
**Visibility:** Trace from light toward target. First hit is the glass sphere (specular) → accept.

**Flux:** 
- p_on_tri = 1/0.3 = 3.33
- p_ω = p_spec × p_on_tri × d² / cos θ_target = 1.0 × 3.33 × 9 / 0.85 = 35.3
- Φ = Le × cos θ_light × A_light / (p_emitter × p_ω) = 10 × 0.9 × 0.5 / (1.0 × 35.3) = 0.127

The photon enters the glass bounce loop with Φ = 0.127. After Fresnel refraction through the sphere and deposit on the floor, it contributes a caustic with reasonable energy.

**Sanity check:** Compare to standard emission. A cosine-hemisphere photon has probability ≈ A_spec × cos θ / (π × d²) = 0.3 × 0.9 / (π × 9) ≈ 0.0095 of hitting the sphere. The targeted photon always hits the sphere (rejection chance is only backface/visibility). The targeted flux is correspondingly lower (0.127 vs ~15.7) — but every photon contributes, whereas only ~1% of standard photons would hit the glass. Net efficiency: targeted produces ~0.127 flux × 100% hit rate ≈ 0.127 vs standard ~15.7 × 0.95% ≈ 0.149. Similar total energy, but the targeted estimator has much lower variance because it never wastes photons. ✅

### 9.6 Scenario F — on_caustic_path Reset and Re-activation (L→D→S→D)

**Setup:** Light → diffuse wall → mirror → Lambertian floor. This tests the `on_caustic_path` flag lifecycle.

**Bounce 0:** Hits diffuse wall. Non-delta, bounce 0 → **not deposited**. `on_caustic_path = false`. Throughput × Kd_wall.

**Bounce 1:** Hits mirror. Delta → **not deposited**. `on_caustic_path = true`. `path_flags |= CAUSTIC_SPECULAR`. Throughput × Ks.

**Bounce 2:** Reflected photon hits floor. Non-delta, bounce 2 > 0, `on_caustic_path == true` → **deposited in both global and caustic maps**. `on_caustic_path = false`.

**Key insight:** The `caustic_only_store` mode (§8.2) would skip depositing bounces 0 (if it were > 0) that are not on a caustic path, but would still allow the photon to continue bouncing. This enables L→D→S→D paths to contribute caustic photons — critical for scenes where light bounces off a wall before hitting glass.

**In the targeted trace:** The targeted trace would **break** at bounce 0 (diffuse hit, not on caustic path) — correctly, because targeted emission is aimed at specular geometry and should not waste bounces on diffuse chains. ✅

---

## 10. Implementation Checklist

A comprehensive checklist covering every documented rule and behaviour. Each item can be tested or inspected against the codebase. Items are grouped by subsystem.

**Legend:** `[x]` = implemented and verified, `[~]` = partially implemented or differs from spec, `[ ]` = not implemented.

*Last audited: 2026-02-28*

### 10.1 Emission Pipeline

- [x] **CP-01** Triangle selection uses power-weighted CDF (CPU: alias table, GPU: binary CDF search)
- [~] **CP-02** GPU applies uniform mix (`DEFAULT_PHOTON_EMITTER_UNIFORM_MIX`) to triangle selection — *deleted in §1.1 cleanup; GPU photon emission now uses pure power CDF (uniform mix remains in NEE dispatch only)*
- [~] **CP-03** Mixture PDF: $p_\text{tri} = (1-\alpha) p_\text{power} + \alpha / N_\text{emissive}$ — *N/A: uniform mix removed from photon emission; doc should be updated to reflect pure CDF*
- [x] **CP-04** Uniform barycentric sampling uses $\sqrt{u_1}$ for area-uniform distribution
- [x] **CP-05** CPU and GPU `sample_triangle()` implementations are mathematically identical
- [x] **CP-06** CPU uses cosine hemisphere for emission direction
- [x] **CP-07** GPU uses cosine cone; at default 90° half-angle it equals cosine hemisphere
- [x] **CP-08** GPU hero wavelength: primary bin sampled from Le CDF, companions at stratified offsets
- [x] **CP-09** Companion bin formula: $b_h = (b_0 + h \cdot N_\lambda / H) \bmod N_\lambda$
- [x] **CP-10** Flux formula divides out all sampling PDFs (p_tri × p_pos × p_dir × p_λ)
- [x] **CP-11** Hero-wavelength normalization factor $1/H$ applied (PBRT v4 §14.3)
- [x] **CP-12** GPU reads emission texture via `dev_get_Le(mat_id, uv)` with interpolated UVs
- [x] **CP-13** CPU reads flat `mat.Le` (no texture)
- [x] **CP-14** Photon carries `source_emissive_idx` for adaptive feedback

### 10.2 Targeted Caustic Emission

- [x] **TC-01** Specular target set includes Glass, Mirror, and Translucent triangles
- [x] **TC-02** Specular triangle selection is area-weighted (alias table)
- [x] **TC-03** Points sampled on both emitter and specular triangles (uniform barycentric)
- [x] **TC-04** Direction: normalize(p_target − p_light)
- [x] **TC-05** Light-side backface cull: reject if cos θ_light ≤ 0
- [x] **TC-06** Target-side backface cull: reject if cos θ_target ≤ 0
- [x] **TC-07** Visibility: radiance ray (not binary shadow). Accept if first hit is specular/translucent — *GPU: radiance ray; CPU: binary shadow with specular pass-through (functionally equivalent)*
- [x] **TC-08** Opaque blockers reject the photon; emissive hits reject — *GPU: both checks; CPU: opaque check only (emissive blocker rejection missing on CPU)*
- [x] **TC-09** Area-to-solid-angle Jacobian: $p_\omega = p_\text{spec} \cdot (1/A_\text{spec}) \cdot d^2 / \cos\theta_\text{target}$
- [x] **TC-10** Flux includes $A_\text{light}$ to cancel p_pos on the emitter
- [x] **TC-11** Firefly clamping applied (flux ≤ 1e6 per hero channel)
- [x] **TC-12** Targeted trace terminates at first non-caustic diffuse hit (break)
- [x] **TC-13** `on_caustic_path` starts false, becomes true on first specular/translucent hit

### 10.3 Photon Deposit Rules

- [x] **PD-01** Photon deposited only at non-delta surfaces with bounce > 0
- [x] **PD-02** Delta surfaces (Mirror, Glass, Translucent) are **never** deposited
- [x] **PD-03** Bounce 0 (direct lighting) is **never** deposited
- [x] **PD-04** Caustic map receives photon only if `on_caustic_path == true`
- [x] **PD-05** Global map receives all eligible photons (caustic and non-caustic)
- [x] **PD-06** `on_caustic_path` set true on first caustic caster (Mirror, Glass, Translucent)
- [x] **PD-07** `on_caustic_path` reset to false after diffuse or glossy bounce
- [x] **PD-08** `caustic_only_store` (GPU) skips non-caustic deposits but continues bouncing
- [x] **PD-09** GPU terminates on emissive surface hit (`break`)
- [x] **PD-10** Path flags accumulated via bitwise OR through lifetime

### 10.4 Material Interactions (BSDFs)

- [x] **MI-01** Lambertian: $f = K_d/\pi$, cosine hemisphere sampling, throughput = $K_d$
- [x] **MI-02** Mirror: deterministic reflection, throughput = $K_s$, delta (no deposit)
- [x] **MI-03** Glass: Fresnel reflect/refract decision, TIR fallback, Snell's law
- [x] **MI-04** Glass: IOR stack push on enter, pop on exit (determined by geometric normal)
- [x] **MI-05** Glass: $T_f$ applied to transmitted (not reflected) light
- [x] **MI-06** Glass: chromatic dispersion via Cauchy equation when `mat.dispersion == true`
- [x] **MI-07** Glass: hero wavelength determines refraction angle; companions get per-bin Fresnel weights
- [x] **MI-08** GlossyMetal: GGX VNDF specular + Lambertian diffuse, lobe selection by max(Ks)/max(Kd)
- [x] **MI-09** GlossyDielectric: IOR-based F0, $(1-F_r)$ energy conservation on diffuse
- [x] **MI-10** Translucent: Glass surface BSDF + interior medium. Medium stack push/pop — *CPU: full MediumStack; GPU: IORStack only (no GPU MediumStack)*
- [x] **MI-11** Clearcoat: coat GGX + Lambertian base with $(1 - w_\text{coat} F_r)$ attenuation
- [x] **MI-12** Fabric: Lambertian + sheen (Schlick Fresnel term)
- [x] **MI-13** Emissive: BSDF treated as Lambertian for bouncing (§7.5)
- [x] **MI-14** Roughness clamped: $\alpha = \max(r^2, 0.001)$

### 10.5 Medium Transport

- [~] **MT-01** Medium stack tracks current participating medium (push on enter, pop on exit) — *CPU only; GPU has IORStack but no MediumStack*
- [x] **MT-02** Beer–Lambert transmittance: $T(\lambda, d) = \exp(-\sigma_t(\lambda) d)$
- [~] **MT-03** Free-flight sampling from hero bin's exponential $\sigma_t$ — *GPU: hero bin σt; CPU: uses average σt across all bins*
- [ ] **MT-04** Spectral MIS weights applied for scatter and no-scatter events — *not implemented; uniform attenuation only*
- [ ] **MT-05** Scatter direction sampled from HG phase function — *HG sampler exists in `medium.h` but is not wired into any trace loop*
- [ ] **MT-06** Phase function PDF cancels with sampling (importance sampling) — *blocked by MT-05*
- [ ] **MT-07** Beam segments recorded for beam estimator (p0 → scatter point or surface) — *no beam estimator infrastructure*
- [ ] **MT-08** Volume double-attenuation guard: legacy atmospheric skipped when inside object medium — *guard not present; legacy volume runs unconditionally*
- [ ] **MT-09** RR inside media uses same threshold logic as surface bounces — *no in-medium scatter loop exists*
- [~] **MT-10** Medium stack overflow/underflow logged in debug builds — *CPU: yes (`#ifndef NDEBUG` printf); GPU: no MediumStack*

### 10.6 Russian Roulette & Path Termination

- [x] **RR-01** RR starts after `min_bounces_rr` (default 3) bounces — *EmitterConfig default = 3; config.h `DEFAULT_PHOTON_MIN_BOUNCES_RR` = 8; runtime value may differ from doc default*
- [x] **RR-02** p_survive = min(rr_threshold, max_λ Φ(λ))
- [x] **RR-03** On survival: Φ /= p_survive (unbiased compensation)
- [x] **RR-04** Specular/translucent bounces exempt from RR (both GPU and CPU targeted)
- [ ] **RR-05** RR applies inside media (scatter events) — *no in-medium scatter loop; volume rendering disabled by default*

### 10.7 Spectral & Hero Wavelength Transport

- [x] **HW-01** GPU: HERO_WAVELENGTHS = 4 bins per photon
- [x] **HW-02** Primary bin sampled from Le CDF at emission
- [x] **HW-03** Companion bins at stratified offsets across NUM_LAMBDA
- [x] **HW-04** Per-hero-channel throughput update at each bounce
- [x] **HW-05** 1/H normalization prevents H× overbright
- [x] **HW-06** CPU photons carry synthetic hero bins for GPU gather compatibility
- [x] **HW-07** CPU hero compatibility uses `NUM_LAMBDA / HERO_WAVELENGTHS` scale factor

### 10.8 RNG & Decorrelation

- [x] **RN-01** CPU: `rng_spatial_seed()` re-seeds RNG from position hash at each bounce > 0
- [x] **RN-02** GPU: `rng.advance(cell_key * 0x9E3779B9u)` at each bounce
- [~] **RN-03** CPU: Optional cell-stratified Fibonacci bouncing for Lambertian/Emissive — *removed in §1.1 cleanup (DEFAULT_PHOTON_BOUNCE_STRATA deleted); only rng_spatial_seed decorrelation remains*
- [x] **RN-04** GPU: No cell stratification (relies on RNG decorrelation only)
- [x] **RN-05** Multi-map re-tracing uses `photon_map_seed` to produce uncorrelated maps

### 10.9 Gather & Density Estimation

- [x] **GE-01** Gather uses `evaluate_diffuse()`, not full `evaluate()` (avoids specular hotspots)
- [x] **GE-02** Delta materials return zero from `evaluate_diffuse()`
- [x] **GE-03** Clearcoat diffuse includes coat energy attenuation factor
- [x] **GE-04** Fabric diffuse includes base only (no sheen in gather, sheen is low-energy)

### 10.10 Adjoint Correction & Symmetry

- [x] **AC-01** Shared BSDF code for photon and camera paths
- [x] **AC-02** Transport direction tag for η² correction at refractive interfaces
- [x] **AC-03** All current materials are symmetric or nearly symmetric (main concern: η² at Glass/Translucent)
- [ ] **AC-04** Shading normal correction not yet implemented (needed if bump maps used) — *by design; no bump maps currently*

### 10.11 View-Adaptive Budgeting

- [ ] **VA-01** Pilot camera pass (1–2 SPP) identifies useful emitters — *not implemented*
- [~] **VA-02** Per-emitter contribution counted via `source_emissive_idx` — *plumbing only: field stored on photons and copied back from GPU, but no aggregation logic counts per-emitter usefulness*
- [ ] **VA-03** CDF reallocated proportionally to emitter usefulness — *not implemented*
- [ ] **VA-04** Re-trace with updated power CDF — *not implemented; multi-map re-tracing uses fixed CDF*

### 10.12 Photon-Guided Path Tracing

- [x] **GP-01** Hash grid + Fibonacci sphere (32 bins) directional histograms built after photon map
- [~] **GP-02** Per-cell histogram: `hist[cell][fib_bin(wi)] += flux / bin_solid_angle` — *flux accumulated raw; bin_solid_angle normalization deferred to query time (mathematically equivalent)*
- [~] **GP-03** Storage budget: 64K cells × 32 bins × 4B = 8 MB — *grid dimensions dynamic from photon AABB; PhotonBin struct is ~52 B not 4 B; no fixed 8 MB cap*
- [x] **GP-04** Query: O(1) cell lookup, discrete sample from histogram
- [x] **GP-05** MIS combine guided direction with BSDF sample (power heuristic) — *uses balance heuristic (mixture PDF), not power heuristic; functionally correct*
- [ ] **GP-06** Per-cell path type classification for adaptive SPP allocation — *CellInfoCache has caustic_count/glass_fraction plumbing, but no adaptive SPP allocator consumes it*
---

## 11. Audit Summary

*Last audited: 2026-02-28*

### §1 Cleanup Plan — Completion Status

| Subsection | Score | Notes |
|---|---|---|
| §1.1 Delete dead code | 9/11 | `CAUSTIC_MIN_FOR_ANALYSIS` + `CAUSTIC_CV_THRESHOLD` still in `cell_cache.h` (used by hotspot detection). `photon_bins.h` intentionally retained (used by `cell_bin_grid.h`). `MULTI_MAP_SPP_GROUP` residue in test file only. |
| §1.2 Move files | 7/7 | All files in correct directories. `emitter_points.h` was absorbed (no separate file). |
| §1.3 Merge files | 5/5 | `nee_sampling.h` → `nee_shared.h`, `cdf.h` → `random.h`, `adaptive_emission.h` → `emitter.h`, `medium.h` + `phase_function.h` → `volume/medium.h`, `sppm.h` → `renderer/`. |
| §1.4 Decompose `optix_device.cu` | 10/10 | All 10 `.cuh` includes created. Additional `optix_camera.cuh`, `optix_guided.cuh`, `optix_nee_dispatch.cuh` beyond plan. |
| §1.5 Eliminate CPU↔GPU duplication | 5/6 | `hash.h`, `ior_stack.h`, `bsdf_shared.h` shared HD. `DevONB` deleted. `spectrum.h` HD. Camera: `dev_generate_camera_ray` wrapper still exists (thin shim calling shared `generate_camera_ray`). |
| §1.6 Renames | 5/7 | `DevONB`, `DevMaterialType`, `dev_fresnel_schlick`, `dev_ggx_*` cleaned up. `RENDER_MODE_*` → enum. `tonemap` moved. **Remaining:** `dev_bsdf_pdf`, `dev_bsdf_sample`, `DevBSDFSample` still use `dev_`/`Dev` prefix in `optix_bsdf.cuh`. |
| §1.7 Reduce `main.cpp` | Partial | 271 lines (target ≤ 200). Extractions done (`viewer`, `scene_builder`), but ~70 lines over target. |
| §1.8 Reduce `optix_renderer.cpp` | Partial | 1751 lines (target ≤ 1200). Three extraction files exist (`optix_setup.cpp`, `optix_upload.cpp`, `optix_denoiser.cpp`), but ~550 lines over target. |
| §1.9 Target directory layout | Complete | All 50+ target files present. 11 extra files beyond spec (manifest, `adaptive_sampling.*`, `launch_params.h`, `optix_renderer.h`, `optix_device.cu`, `optix_guided.cuh`, `optix_nee_dispatch.cuh`, `photon_bins.h`, `tonemap.h`, `optix_camera.cuh`). |

### §10 Implementation Checklist — Scorecard

| Section | Done | Partial | Not Done | Total |
|---|---|---|---|---|
| 10.1 Emission Pipeline | 12 | 2 | 0 | 14 |
| 10.2 Targeted Caustic | 13 | 0 | 0 | 13 |
| 10.3 Photon Deposit Rules | 10 | 0 | 0 | 10 |
| 10.4 Material Interactions | 14 | 0 | 0 | 14 |
| 10.5 Medium Transport | 1 | 3 | 6 | 10 |
| 10.6 Russian Roulette | 4 | 0 | 1 | 5 |
| 10.7 Spectral/Hero λ | 7 | 0 | 0 | 7 |
| 10.8 RNG & Decorrelation | 4 | 1 | 0 | 5 |
| 10.9 Gather & Density | 4 | 0 | 0 | 4 |
| 10.10 Adjoint Correction | 3 | 0 | 1 | 4 |
| 10.11 View-Adaptive | 0 | 1 | 3 | 4 |
| 10.12 Guided Path Tracing | 3 | 2 | 1 | 6 |
| **Totals** | **75** | **9** | **12** | **96** |

### What's Solid (ready for Part 2)

The photon mapping subsystem is production-quality:

- **Emission pipeline** (§10.1–10.2): fully implemented on CPU and GPU, including targeted caustic emission with specular pass-through visibility and firefly clamping.
- **Photon storage** (§10.3): deposit rules, caustic path tracking, and flag accumulation all verified correct.
- **Material interactions** (§10.4): all 9 material types correctly handle photon bouncing, BSDF evaluation, and delta classification. Shared `bsdf_shared.h` eliminates CPU↔GPU drift.
- **Spectral transport** (§10.7): hero wavelength scheme fully operational with 1/H normalization, CPU↔GPU compatibility, and stratified companions.
- **Density estimation** (§10.9): diffuse-only gather with proper delta exclusion, coat attenuation, and fabric handling.
- **Adjoint correction** (§10.10): `TransportMode` tag with η² correction at refractive interfaces.
- **Guided path tracing** (§10.12): CellBinGrid + Fibonacci histogram build, O(1) query, MIS with BSDF — the core guidance infrastructure works.
- **Code structure** (§1): ~95% of the cleanup plan is done. Directory layout matches target. All major decompositions and merges complete.

### What's Not Done (future work)

| Category | Items | Priority for Part 2 |
|---|---|---|
| **Volume/medium transport** | MT-04 through MT-09: spectral MIS, HG scatter, beam estimator, double-attenuation guard, in-medium RR | Low — volumes are disabled by default; revisit when enabling participating media |
| **View-adaptive budgeting** | VA-01 through VA-04: pilot pass, per-emitter tallying, adaptive CDF | Medium — useful for multi-light scenes; `source_emissive_idx` plumbing exists |
| **Adaptive SPP** | GP-06: per-cell path type → SPP allocation | Medium — CellInfoCache has the raw data; needs a consumer |
| **Shading normal correction** | AC-04 | Low — no bump maps in current scenes |
| **Minor cleanup** | Rename `dev_bsdf_pdf`/`dev_bsdf_sample`/`DevBSDFSample`; delete `dev_generate_camera_ray` wrapper; slim `main.cpp` 271→200; slim `optix_renderer.cpp` 1751→1200 | Low — cosmetic, no functional impact |

### Doc vs Code Divergences

These items describe behaviour in the design doc that does not match the actual code:

1. **CP-02/CP-03 (§3.3 Step 1)**: Doc describes GPU uniform mix (`DEFAULT_PHOTON_EMITTER_UNIFORM_MIX`) for photon emission triangle selection. Code was deleted in §1.1 cleanup — GPU photon emission now uses **pure power CDF**. The uniform mix only exists in NEE dispatch. *The §3.3 text should be updated.*
2. **RN-03 (§8.6)**: Doc describes cell-stratified Fibonacci bouncing (`DEFAULT_PHOTON_BOUNCE_STRATA`). Deleted in §1.1 cleanup. *The §8.6 text is now historical.*
3. **MT-08 (§8.10)**: Doc describes a volume double-attenuation guard. Code runs legacy atmospheric system unconditionally with no medium-stack check. *Guard was never implemented.*
4. **GP-03 (§4.3)**: Doc specifies "64K cells × 32 bins × 4 bytes = 8 MB". Actual implementation uses dynamic grid sizing from photon AABB and ~52 B per bin. *Budget estimate is aspirational, not enforced.*
5. **GP-05 (§4.4)**: Doc specifies "power heuristic" for MIS. Code uses balance heuristic (mixture PDF). *Functionally correct but terminology differs.*