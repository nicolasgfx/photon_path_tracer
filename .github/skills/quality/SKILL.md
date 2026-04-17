---
name: quality
description: 'Image quality analysis, artifact detection, convergence diagnostics, noise decomposition, and parameter corrections. Use when: diagnosing image quality problems, running ppt_diagnose, analyzing convergence, identifying bottlenecks, tuning render parameters via feedback, or comparing A/B renders.'
---

# Quality

The `ppt_diagnose` executable and in-render diagnostics system.

## Source Map

| Location | Files | Purpose |
|----------|-------|---------|
| `src/diagnose/` | diagnose_main.cpp | ppt_diagnose entry point: loads EXR, runs analysis, prints JSON |
| `src/diagnose/` | image_oracle.h/.cpp | Main analysis engine: no-reference and reference-based |
| `src/diagnose/` | artifact_detector.h | 6 artifact types with severity scoring |
| `src/diagnose/` | noise_analyzer.h | Laplacian variance, spatial autocorrelation |
| `src/diagnose/` | image_metrics.h | PSNR, SSIM, RMSE computation |
| `src/diagnose/` | image_verdict.h | Verdict + ArtifactReport + Correction structs |
| `src/diagnose/` | diagnostics.h/.cpp | In-render convergence analysis |
| `src/diagnose/` | variance_tracker.h/.cpp | Per-pixel variance tracking from GPU buffers |
| `src/diagnose/` | convergence_analyzer.h | Convergence rate estimation (log-log fit) |
| `src/diagnose/` | bottleneck_report.h | Bottleneck identification + parameter recommendations |

## Tool: ppt_diagnose

```
ppt_diagnose <render.exr> [render_log.json]
```

Loads rendered EXR, runs image oracle, prints JSON to stdout:
```json
{
  "quality": { "noise_level", "convergence_rate", "summary" },
  "artifacts": [{ "type", "severity", "description" }],
  "corrections": [{ "target", "parameter", "current", "recommended", "rationale" }]
}
```

## Artifact Detection (6 Types)

| Type | Detection | Threshold | Correction Target |
|------|-----------|-----------|-------------------|
| **Firefly** | Pixel > K × local median | K=3–6 (adaptive per SPP) | path-integrator (clamp), post-processing (filter) |
| **Splotch** | Spatial autocorrelation ρ(lag=8) > 0.15 | 0.15 | path-guide (more training iters, lower split threshold) |
| **HighNoise** | Laplacian variance (Immerkaer 1996) | σ > scene-dependent | guide_fraction increase, more SPP |
| **EnergyLoss** | Mean luminance < expected (over-clamping) | 15% deviation | raise clamping thresholds |
| **Banding** | Gradient quantization | histogram peaks | increase precision / dithering |
| **SlowConvergence** | Rate α < 0.3 in log-log fit MSE∝N^(-α) | α < 0.3 | guide, photon, or scene-specific tuning |

## Convergence Analysis

- **Rate α**: Log-log fit of MSE vs SPP; α≈1.0 = ideal MC; α<0.3 = stalled; α>1.0 = biased
- **Variance decomposition**: direct% / indirect% / photon% / firefly% — bottleneck = max fraction
- **Bottleneck mapping**: direct>50%→NEE tuning; indirect>50%→guide tuning; photon>30%→budget; firefly>5%→filter
- **Per-bounce analysis**: Identifies noisiest bounce level
- **SPP predictor**: Given rate α and target quality, estimate SPP needed
- **Convergence sweep**: `photon_tracer --sweep` renders at multiple SPP levels (default: 16,64,256,512,1024,2048), saves checkpoint images and a `_sweep.json` report with per-level noise + mean luminance. Use to diagnose noise reduction effectiveness across sample counts. See orchestrator skill for full details.

## Parameter Isolation

When diagnosing artifacts or evaluating parameter changes, **vary one parameter at a time**. Render multiple values of that single parameter (ideally as `--sweep` runs), compare metrics, then lock the best value before moving to the next parameter. Changing multiple parameters simultaneously prevents attributing improvements or regressions to their cause. Only combine parameters after their individual effects are understood. See the orchestrator skill's "Single-Variable Isolation Strategy" for the full workflow.

## Feedback Loop

```
ppt_analyze scene.pbrt > profile.json
photon_tracer --config profile.json scene.pbrt → output.exr + render_log.json
ppt_diagnose output.exr render_log.json > report.json
# Apply corrections from report.json → re-render (up to 3 iterations)
```

## Image Metrics

- **PSNR**: Peak signal-to-noise ratio vs reference (dB)
- **SSIM**: Structural similarity index (0–1)
- **RMSE**: Root mean squared error
- **Noise level**: Normalized Laplacian variance [0–1]
