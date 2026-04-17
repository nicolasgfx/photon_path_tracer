# Post-Processing — Reactive Denoising Strategy

## When to Apply What

Post-processing is reactive — it applies after rendering completes each frame. The optimal configuration depends on how much rendering noise remains.

### Low SPP (< 64 samples)
High noise level. Denoiser and firefly filter are essential:
- `firefly_threshold = 3.0` (aggressive: more outliers at low SPP)
- Denoiser blend = 1.0 (full strength)
- Bloom disabled (noise would amplify)

### Medium SPP (64-256 samples)
Moderate noise. Balance between denoiser and detail preservation:
- `firefly_threshold = 4.0` (moderate)
- Denoiser blend = 0.5-0.8
- Bloom optional

### High SPP (> 256 samples)
Low noise. Minimize post-processing to preserve detail:
- `firefly_threshold = 6.0` (gentle: few remaining outliers)
- Denoiser blend = 0.0-0.3 (or off entirely)
- Bloom optional for artistic effect

## Denoiser Artifacts

The OptiX AI denoiser can introduce artifacts:
- **Smearing**: Fine geometric detail gets blurred
- **Temporal instability**: Flickering in progressive rendering
- **Over-smoothing**: Textured surfaces lose detail

### Mitigation
1. Use AOV guides (albedo + normal buffers) for structure-preserving denoising
2. Reduce denoiser blend at high SPP
3. Use `apply_pre_denoise()` to clean outliers before denoiser sees them
4. The firefly filter prevents outlier pixels from confusing the denoiser

## Firefly Filter vs Integrator Clamping

Two complementary approaches to outlier control:

| Method | Where | When | Trade-off |
|---|---|---|---|
| Sample clamping | Integrator (per-sample) | During rendering | Introduces bias |
| Firefly filter | PostFx (image-space) | After rendering | May miss some outliers |

Recommended: Use moderate clamping + moderate firefly filter. Neither too aggressive alone.
