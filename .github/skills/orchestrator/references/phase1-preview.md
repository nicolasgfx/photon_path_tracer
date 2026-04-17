# Phase 1 — Preview Renderer

Load a PBRTv4 scene and display it in a real-time interactive preview window with simplified path tracing (low bounces, no photons, no guide).

## Skills Required (in order)

1. **scene-pipeline** — Parse PBRT, build `Scene`, classify `SceneProfile`
2. **acceleration** — OptiX GAS/IAS, SBT, device uploads
3. **camera-system** — Primary ray generation from PBRT camera params
4. **material-system** — BSDF sampling/eval for all hit surfaces
5. **direct-lighting** — NEE + emissive CDF + envmap (if present)
6. **path-integrator** — 2-bounce render loop (preview mode)
7. **post-processing** — Tonemap only (+ denoiser + firefly filter)

## Config Overrides (vs defaults in config.h)

```cpp
// Preview-mode overrides applied in RenderConfig or Viewer
config.max_bounces        = PREVIEW_MAX_BOUNCES;    // 2 (config.h §3)
config.guide_enabled      = false;                   // no SD-tree
config.num_global_photons = 0;                       // no photon tracing
config.num_caustic_photons = 0;                      // no caustics
config.denoiser_enabled   = true;                    // smooth low-SPP noise
config.samples_per_pixel  = 1;                       // progressive accumulation
config.rr_threshold       = 0.90f;                   // aggressive RR for speed
```

## Execution Steps

### Step 1 — Scene Loading (scene-pipeline skill)

**Goal**: Load a `.pbrt` file into the `Scene` struct with all geometry, materials, textures, emitters, and camera parameters populated.

**Entry point**: `src/main.cpp` lines 80-115

**Call sequence**:
```
load_pbrt(scene_file, scene)   → parse + build Scene
scene.compute_bounds()          → AABB for acceleration
scene.build_emissive_distribution()  → alias table for NEE
scene.compute_envmap_selection_prob() → envmap vs area light weight (if envmap present)
```

**Files**:
- `src/scene/pbrt/pbrt_loader.h` — `load_pbrt()` entry
- `src/scene/pbrt/pbrt_parser.h` — tokenizer
- `src/scene/pbrt/pbrt_material_mapper.h` — PBRT material → v5 Material
- `src/scene/scene.h` — `Scene` struct
- `src/scene/scene_builder.h` — `build_cornell_box()` fallback

**Outputs**:
- `scene.triangles[]` — all geometry
- `scene.materials[]` — all materials with Kd, Ks, Le, roughness, IOR
- `scene.envmap` — environment map (optional)
- `scene.pbrt_cam_*` — camera position/look_at/up/fov/flip_x
- `scene.scene_bounds` — AABB
- `scene.total_emissive_power` — for emitter sampling

**Gate check**:
```
[Scene] Triangles: N  Emissive: M  Power: P
[Scene] Bounds: (lx,ly,lz)-(hx,hy,hz)
```
N > 0, bounds non-degenerate.

**Known gotchas**:
- PBRT `Camera "perspective" "float fov" [F]` uses **horizontal** FoV in v4 — our loader converts to vertical
- `pbrt_cam_flip_x` flag for left-handed→right-handed coordinate conversion
- Missing envmap is OK — NEE falls back to area lights only
- Empty emitter list → dark scene (check material `Le > 0` mapping)

---

### Step 2 — Acceleration Structure (acceleration skill)

**Goal**: Build OptiX GAS from triangles, upload geometry/materials to GPU, populate `LaunchParams`.

**Entry point**: `RenderSession::init()` in `src/app/render_session.h`

**Call sequence**:
```
builder_.init()                  → OptiX context + pipeline
builder_.build_gas(scene)        → GAS from triangles
builder_.upload_geometry(scene)  → device buffers in LaunchParams
builder_.upload_materials(scene) → device material buffer
```

**Files**:
- `src/accel/accel_builder.h/.cpp` — `AccelBuilder` class
- `src/accel/accel_types.h` — `AccelStructure` wrapper
- `src/accel/launch_params.h` — `LaunchParams` struct (§3: geometry, §4: materials)
- `src/accel/optix_programs.cu` — hit/miss programs

**Outputs**:
- `OptixTraversableHandle` in `LaunchParams`
- Device buffers: vertices, normals, tangents, texcoords, material_ids
- SBT records populated

**Gate check**: `launch_test_normals()` produces a colored normal visualization — geometry is correctly uploaded and intersectable.

**Known gotchas**:
- `SELF_INTERSECTION_EPSILON = 1e-4` — too small causes speckles, too large causes shadow leaks
- Compacted GAS reduces memory ~2× — enabled by default
- SBT must match 1 hit record per mesh for instancing

---

### Step 3 — Camera Setup (camera-system skill)

**Goal**: Generate camera rays matching the PBRT scene's viewpoint.

**Entry point**: `src/main.cpp` lines 128-165

**Call sequence**:
```
camera.position = scene.pbrt_cam_position
camera.look_at  = scene.pbrt_cam_look_at
camera.fov_deg  = scene.pbrt_cam_fov
camera.update()  → compute u/v/w basis, lower_left, horizontal, vertical

// If PBRT flagged left-handed coords:
if (scene.pbrt_cam_flip_x)
    camera.u *= -1;  camera.horizontal *= -1;  recompute lower_left
```

**Files**:
- `src/camera/camera.h` — `Camera` struct + `generate_camera_ray()`
- `src/app/viewer.h` — `load_camera_json()` for saved camera overrides

**Outputs**:
- `Camera` with valid `position`, `u`, `v`, `w`, `lower_left`, `horizontal`, `vertical`
- For preview: `dof_enabled = false` (no depth of field)

**Gate check**: Normal visualization shows scene from expected viewpoint — geometry fills the frame, not clipped or inverted.

**Known gotchas**:
- Saved `camera.json` in scene folder overrides PBRT camera — check for stale saves
- `flip_x` must be applied AFTER `camera.update()` — order matters

---

### Step 4 — Material Upload (material-system skill)

**Goal**: All 9 material types compiled and available on GPU for BSDF eval during path tracing.

**Entry point**: Materials are uploaded in Step 2 (`upload_materials()`), but BSDF evaluation happens in the integrator.

**Files**:
- `src/material/bsdf.h` — `dev_bsdf_sample()`, `dev_bsdf_evaluate()`, `dev_bsdf_pdf()`
- `src/material/bsdf_shared.h` — `DevBSDFSample` struct
- `src/material/specular.h` — Glass/mirror bounce logic

**Material types (PbBrdf enum)**:
1. Diffuse (Lambertian)
2. Mirror (perfect specular)
3. Glass (dielectric, Fresnel + refraction)
4. GlossyMetal (GGX VNDF)
5. Plastic (diffuse + specular coat)
6. CoatedDiffuse (layered)
7. ThinDielectric (single-surface glass)
8. Conductor (Fresnel metal)
9. Substrate (multi-layer)

**Gate check**: Material-ID false-color visualization (if available) or render with flat shading — each material type returns valid, non-black colors.

---

### Step 5 — Direct Lighting / NEE (direct-lighting skill)

**Goal**: Upload emissive distribution + envmap to GPU, enable next-event estimation in path tracer.

**Entry point**: `RenderSession::init()` → `lighting_.upload(scene, lp_)`

**Call sequence**:
```
lighting_.upload(scene, launch_params)
  → uploads emissive triangle CDF (power-weighted)
  → uploads envmap data + marginal/conditional CDFs
  → sets envmap_selection_prob in LaunchParams
```

**Files**:
- `src/lighting/lighting_upload.h` — `LightingUploader::upload()`
- `src/lighting/optix_nee.cuh` — GPU NEE kernel (`dev_nee_evaluate_sample()`)
- `src/lighting/optix_envmap.cuh` — envmap importance sampling
- `src/lighting/nee_shared.h` — MIS weight, PDF conversion

**Outputs**:
- `LaunchParams` populated with: emissive CDF, envmap texture, envmap CDFs, `envmap_selection_prob`
- NEE kernel available for `trace_path()` to call

**Gate check**: `RenderMode::DirectOnly` (NEE only, no bounces) → scene is lit, shadows visible.

**Known gotchas**:
- Zero emissive power → NEE disabled, no direct lighting
- `envmap_selection_prob` balances envmap vs area light sampling — defaults to 0.9 if envmap present
- Shadow epsilon must match acceleration epsilon for consistent results

---

### Step 6 — Path Integrator / Preview Loop (path-integrator skill)

**Goal**: Run the 2-bounce path tracing loop with NEE per bounce, progressive accumulation.

**Entry point**: `RenderSession::render_frame()` → `builder_.launch(w, h)` → `__raygen__render`

**Call sequence (GPU, per pixel)**:
```
ray = generate_camera_ray(pixel, jitter)
for bounce = 0..1:  // PREVIEW_MAX_BOUNCES = 2
    hit = optixTrace(ray)
    if miss: radiance += throughput * background; break
    radiance += throughput * nee_evaluate(hit)     // direct lighting
    sample   = bsdf_sample(hit)                     // bounce direction
    throughput *= sample.f * cos / sample.pdf
    ray = spawn_ray(hit.pos, sample.wi)
    // No guide sampling, no photon gather in preview
accumulate(pixel, radiance)
```

**Files**:
- `src/integrator/path_tracer.cuh` — `trace_path()` + `__raygen__render`
- `src/integrator/russian_roulette.h` — RR (minimal impact at 2 bounces)
- `src/integrator/sample_clamping.h` — per-bounce + per-sample clamping

**Preview-specific behavior**:
- `guide_enabled = false` → skip all SD-tree sampling branches
- `max_bounces = 2` → loop exits early
- Clamping at `MAX_SAMPLE_LUMINANCE = 10000` catches fireflies
- Progressive accumulation: frame_number increments each call, radiance accumulates

**Gate check**: 2-bounce preview renders with visible direct + 1-bounce indirect lighting. Scene is recognizable, not entirely black or white.

---

### Step 7 — Post-Processing + Display (post-processing skill)

**Goal**: Tonemap HDR accumulator → sRGB, display in GLFW window.

**Entry point**: `RenderSession::apply_postfx(params)` → `Viewer::display_frame()`

**Call sequence**:
```
postfx_.apply(d_color_, d_sample_counts_, d_srgb_, d_hdr_,
              width_, height_, params)
  → rgb_to_hdr: divide by sample count, apply exposure
  → firefly_filter: median+MAD outlier suppression
  → tonemap_hdr: ACES + sRGB gamma → uint8

download_srgb(srgb_buffer)
display via OpenGL texture
```

**Files**:
- `src/postfx/postfx_pipeline.h/.cpp` — `PostFxPipeline` orchestrator
- `src/postfx/tonemap.h` — ACES tonemap kernel
- `src/postfx/firefly_filter.h` — outlier suppression
- `src/app/viewer.h` — `display_frame()` → OpenGL quad

**Preview-specific**:
- Bloom OFF (not needed for preview)
- Firefly filter ON (catches low-SPP outliers)
- Denoiser ON at `blend = 0.0` (full denoise for smooth preview)
- Exposure = `DEFAULT_EXPOSURE * light_scale`

**Gate check**: GLFW window shows a tonemapped image. Colors are reasonable (not washed out or pure black). Mouse/keyboard camera controls work, accumulation resets on camera move.

## End-to-End Verification

After all 7 steps complete:

1. **Launch**: `photon_tracer.exe scenes/bathroom/scene-v4.pbrt` (or any PBRT scene)
2. **Window**: GLFW window appears with progressive preview rendering
3. **Interaction**: Mouse look + WASD movement resets accumulation, re-renders
4. **Quality**: After ~10-30 SPP idle, scene is recognizable with direct + indirect lighting
5. **Stability**: No crashes, NaN pixels (black/white splotches), or GPU errors

## Transition to Phase 2

When preview is working, Phase 2 adds:
- `train_guide()` call before render loop (photon-system + path-guide skills)
- `guide_enabled = true` in config
- Photon budgets restored to defaults (2M global + 2M caustic)
- Max bounces increased to 8

These changes wire into the existing `RenderSession` and `Viewer` without architectural changes.
