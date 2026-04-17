# Phase 0 — Brute-Force Baseline

Simplest possible starting point: unguided path tracing with NEE, no photons, no guide, no denoiser, no clamping, no firefly filter. Pure BSDF sampling + next-event estimation.

## Purpose

Establish a ground truth baseline before adding complexity. Every subsequent phase adds ONE component and verifies improvement against this baseline.

## Config

**File**: `configs/phase0_brute_force.json`

```json
{
  "max_bounces": 8,
  "min_bounces_rr": 3,
  "rr_threshold": 0.95,
  "guide_enabled": false,
  "guide_fraction": 0.0,
  "num_global_photons": 0,
  "num_caustic_photons": 0,
  "denoiser_enabled": false,
  "max_bounce_contribution": 1e30,
  "max_path_throughput": 1e30,
  "max_nee_contribution": 1e30,
  "max_sample_luminance": 1e30,
  "firefly_enabled": false,
  "bloom_enabled": false,
  "exposure": 1.0
}
```

## What's Active

| Component | Status | Notes |
|-----------|--------|-------|
| Scene loading (OBJ/PBRT) | ON | Full geometry + materials |
| Acceleration (OptiX GAS) | ON | Single GAS, triangle intersection |
| Camera (thin-lens) | ON | DOF from saved_camera.json if present |
| Materials (all 9 types) | ON | Full BSDF sample/eval/PDF |
| NEE (direct lighting) | ON | Power-weighted emitter sampling + shadow rays |
| Path integrator (8 bounces) | ON | BSDF-only sampling, no guide MIS |
| Russian roulette | ON | After bounce 3, threshold 0.95 |
| Tonemap (ACES + sRGB) | ON | Only post-processing applied |
| Clamping | **OFF** | All limits set to 1e30 (effectively infinite) |
| Firefly filter | **OFF** | |
| Bloom | **OFF** | |
| Denoiser | **OFF** | |
| Photon tracing | **OFF** | 0 photons emitted |
| SD-tree guide | **OFF** | guide_enabled=false, guide_fraction=0 |

## Launch Command

```
photon_tracer.exe --config configs/phase0_brute_force.json scenes/cornell_box/cornellbox.obj
```

## Expected Behavior

- Window opens immediately (no guide training delay)
- Progressive accumulation: noisy at 1 SPP, cleans up over time
- Direct lighting visible from first frame (NEE)
- Indirect lighting (color bleeding) visible after a few SPP
- May have occasional fireflies (no clamping or filter)
- Converges slowly compared to guided rendering

## Gate Checks

| Check | Expected | How to Verify |
|-------|----------|---------------|
| Scene loads | 13056 tris, 7 mats, 128 emissive | Console output |
| No guide training | No `[Guide]` messages | Console output |
| No denoiser init | No `[Denoiser]` messages | Console output |
| Direct lighting | Scene lit from ceiling light | Visual — first frame |
| Color bleeding | Red/green on white walls | Visual — after ~10 SPP |
| Progressive accumulation | Noise decreases over time | Visual — idle for 30s |

## Transition to Phase 1

Phase 1 adds denoiser + firefly filter + clamping on top of Phase 0:
- `denoiser_enabled: true` — smooth low-SPP noise
- `firefly_enabled: true` — suppress outlier pixels
- Restore clamping: `max_sample_luminance: 10000`, etc.
- Everything else stays the same (still no guide, no photons)
