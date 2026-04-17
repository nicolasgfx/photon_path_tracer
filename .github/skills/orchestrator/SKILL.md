---
name: orchestrator
description: 'Multi-stage pipeline orchestration using 3 executables (ppt_analyze → photon_tracer → ppt_diagnose). Use when: wiring the analyze→render→diagnose pipeline, setting up preview or production rendering, coordinating skill invocation order, configuring render parameters, or debugging stage-to-stage data flow.'
---

# Orchestrator

Coordinates the rendering pipeline across 3 executables. Owns no source files — determines **which tools to invoke, in what order, with what config, and how to verify**.

## 3-Executable Pipeline

```
ppt_analyze scene.pbrt           → stdout: profile.json
    ↓
photon_tracer --config profile.json scene.pbrt → output.exr + render_log.json
    ↓
ppt_diagnose output.exr render_log.json → stdout: report.json
```

### Feedback Loop

```
for iteration in 1..3:
    profile = ppt_analyze(scene)
    render(scene, profile)
    report = ppt_diagnose(output)
    if report.artifacts.empty(): break
    apply report.corrections to profile
```

### Data Flow

| Producer | Output | Consumer |
|----------|--------|----------|
| `ppt_analyze` | `profile.json` (SceneProfile + RenderConfig) | `photon_tracer` (via --config) |
| `photon_tracer` | `output.exr` + `render_log.json` | `ppt_diagnose` |
| `ppt_diagnose` | `report.json` (artifacts + corrections) | orchestrator (feedback) |

## Skills → Executables

| Skill | Executable | Invocation |
|-------|-----------|------------|
| scene-analysis | `ppt_analyze` | `ppt_analyze <scene>` |
| renderer | `photon_tracer` | `photon_tracer [--config X] <scene>` |
| post-processing | `photon_tracer` | Built into renderer; params via config |
| quality | `ppt_diagnose` | `ppt_diagnose <output.exr> [render_log.json]` |
| test-harness | `ppt_tests` | Test runner |
| scene-editor | `scene_editor` | `scene_editor <scene>` |

## Config Override Cascade

CLI args **>** JSON config (`--config`) **>** SceneProfile auto-tune **>** compile-time defaults (`config.h`)

## CLI Reference (essentials)

See `photon_tracer --help` for the full list (25+ flags). Most-used flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--config FILE` | — | Load JSON config |
| `--spp N` | 256 | Samples per pixel |
| `--bounces N` | 8 | Max camera bounces |
| `--exposure F` | 1.0 | Tonemap exposure multiplier |
| `--light-scale F` | 1.0 | Light intensity multiplier |
| `--output FILE` | output/render.png | Output path |
| `--mode MODE` | full | full\|direct\|indirect\|photon\|normals\|material\|depth\|guide |
| `--headless` | — | No window, batch render + save |
| `--sweep [SPP,...]` | 16,64,256,512,1024,2048 | Convergence sweep (implies --headless) |
| `--no-aces` | — | Disable ACES tonemapping (linear output) |
| `--no-guide` | — | Disable SD-tree guided path tracing |
| `--guide-fraction F` | 0.5 | Bilateral MIS weight (0 = pure BSDF, 1 = pure guide) |
| `--no-denoiser` | — | Disable OptiX AI denoiser |
| `--no-bloom` | — | Disable bloom post-FX |
| `--no-firefly` | — | Disable firefly filter |
| `--no-path-clamp` | — | Disable per-path throughput clamping (sets limit to 1e30) |
| `--no-light-tree` | — | Disable light tree (use flat power-weighted CDF) |
| `--light-tree-leaf N` | 4 | Max triangles per light tree leaf node |

All 60 `RenderConfig` fields can be set via `--config` JSON files. See `src/renderer/app/render_config_json.cpp` for the full list. Compile-time defaults in `src/core/config.h`.

## Gate Checks

| Stage | Gate | Verification |
|-------|------|-------------|
| `ppt_analyze` | Valid JSON on stdout | Parse output, check `scene_metrics.num_triangles > 0` |
| `photon_tracer` | `output.exr` written | File exists, dimensions match config |
| `ppt_diagnose` | Valid JSON on stdout | Parse output, check `quality.noise_level` exists |

## Key Source Files

| File | Role |
|------|------|
| `src/renderer/main.cpp` | Renderer entry: CLI → scene load → analyze → render → output |
| `src/renderer/app/render_session.h` | GPU pipeline: `init()`, `render_frame()`, `train_guide()`, `apply_postfx()` |
| `src/renderer/app/render_config.h` | `RenderConfig` struct — all runtime params |
| `src/core/config.h` | Compile-time defaults |
| `src/analyze/scene_profile_applicator.h` | SceneProfile → RenderConfig mapping |

## Convergence Sweep Strategy

Render at multiple SPP checkpoints to measure how noise decreases. Reveals whether a change improves convergence or just masks noise at a single sample count.

**When to use:** A/B tests, diagnosing slow convergence, validating that guide/photon/clamping changes genuinely improve convergence rate, checking for energy loss from over-clamping.

```
photon_tracer --sweep [16,64,256,1024,4096] --config profile.json scene.pbrt
```

**Outputs:** `output/render_sppNNNN.png` + `output/render_sweep.json` with per-level noise + mean_lum.

| Convergence rate α | Interpretation |
|--------------------|---------------|
| 0.8–1.2 | Normal MC convergence |
| < 0.3 | Stalled — check guide/photon/NEE |
| > 1.2 | Suspiciously fast — verify energy conservation |

**Energy conservation check:** If `mean_lum` varies > 5% across SPP levels, over-clamping or bias is likely.

## Single-Variable Isolation

**Never change multiple parameters at once.** One variable at a time → render → measure → compare → next variable.

```
1. Establish baseline with --sweep
2. Pick ONE parameter, sweep candidate values:
   photon_tracer --sweep --guide-fraction 0.0  scene.pbrt --output output/gf_00.png
   photon_tracer --sweep --guide-fraction 0.5  scene.pbrt --output output/gf_05.png
   photon_tracer --sweep --guide-fraction 0.7  scene.pbrt --output output/gf_07.png
3. Compare sweep reports → identify best value
4. Lock that parameter, pick NEXT parameter
5. After individual effects understood, combine best values into final config
```

**Parameter investigation order** (high to low impact):
1. `guide_enabled` / `guide_fraction` — largest variance reduction lever
2. `num_global_photons` — dominant for caustics and SDS paths
3. `max_bounce_contribution` / `max_sample_luminance` — firefly vs energy trade-off
4. `max_bounces` / `rr_threshold` — path length vs bias
5. `nee_light_samples` — direct lighting quality
6. `denoiser_blend` / postfx — cosmetic, only after underlying quality is understood

**Anti-patterns:** Changing multiple params together; skipping baseline; testing only at one SPP.

## Guided vs Unguided A/B Comparison

Compare SD-tree path guiding against pure BSDF sampling at the same SPP:

```
photon_tracer --sweep --no-guide  scene.pbrt --output output/unguided.png
photon_tracer --sweep             scene.pbrt --output output/guided.png
ppt_diagnose output/unguided_sweep.json output/guided_sweep.json
```

Expected: guided render shows lower noise at ≥64 SPP, same `mean_lum` (energy conservation). The convergence rate α should be higher with guiding on scenes with complex indirect illumination. `--mode guide` visualizes the trained SD-tree PDF as a heatmap.

## Diagnosing "Overall Too Dark" Images

Decision tree — work top-to-bottom, stop when root cause found:

```
Image too dark?
│
├─ Is HDR mean_lum low?
│   ├─ NO → Exposure/tonemapping issue
│   │       Test: --exposure 2.0 or --no-aces
│   │
│   └─ YES → Light transport issue, continue:
│       │
│       ├─ Does --light-scale 10 produce correct brightness?
│       │   └─ YES → Scene Le values ~10× too low
│       │           Check: normalize_to_reference(), PBRT Le parsing
│       │           Files: src/scene/scene.h, src/scene/pbrt/pbrt_loader.cpp
│       │
│       ├─ Is --mode direct very dark?
│       │   └─ YES → NEE broken
│       │           Common: one-sided emitter normals (use fabsf for cos_light),
│       │           shadow ray self-intersection, emissive triangle not registered
│       │           Files: src/renderer/lighting/optix_nee.cuh
│       │           Check: [PrePass] NEE hit=X% (should be >1% for visible area lights)
│       │
│       ├─ Does mean_lum increase significantly bounces=4 → bounces=16?
│       │   └─ YES → Multi-bounce scene, increase --bounces
│       │           Typical: most scenes converge by 8; glass needs 12+
│       │           Also check: aggressive RR (low rr_threshold) = early termination
│       │
│       └─ All above OK → Material BSDF energy loss
│               Check: Lambertian returns Kd/π (not Kd/2π)
│               Check: GGX VNDF sampling, Smith G term includes both G1
│               Check: Glass refract_local() copysignf(cos_t, -wo.z)
│               File: src/renderer/integrator/path_tracer.cuh
│               Diagnostic: --mode material (false-color material IDs)
```

## PTX Rebuild Caveat

Ninja tracks only `optix_programs.cu` — header changes in `.cuh` files are NOT detected. Always touch the .cu file:

```
copy /b src\renderer\accel\optix_programs.cu +,, src\renderer\accel\optix_programs.cu
ninja photon_tracer
```
