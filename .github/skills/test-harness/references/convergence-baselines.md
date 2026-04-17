# Convergence Baselines Reference

## Overview

Convergence baselines store expected MSE-at-N-SPP for each reference scene. They are the regression detection mechanism — any code change that degrades convergence rate by more than 15% triggers a test failure.

## Baseline Storage

- **Location**: `tests/data/convergence_baselines/<scene_name>.baseline`
- **Format**: Plain text. First line = scene name. Subsequent lines = `spp mse` pairs.
- **Persistence**: Checked into version control. Updated only when intentional algorithm changes improve convergence.

## Creating a New Baseline

```cpp
#include "testing/convergence_test.h"

// After rendering at standard SPP levels and measuring MSE:
auto baseline = convergence_test::create_baseline(
    "my_new_scene",
    {16, 64, 256, 1024},
    {0.20f, 0.05f, 0.013f, 0.004f}
);

// Rate is auto-computed via log_log_fit
printf("Convergence rate: %.3f\n", baseline.convergence_rate);

convergence_test::save_baseline(baseline, "tests/data/convergence_baselines/my_new_scene.baseline");
```

## Validating Against a Baseline

```cpp
convergence_test::ConvergenceBaseline baseline;
convergence_test::load_baseline("tests/data/convergence_baselines/cornell_box.baseline", baseline);

auto result = convergence_test::compare_to_baseline(
    measured_spp_levels, measured_mse_values, baseline,
    /*rate_tolerance=*/0.15f,    // 15% rate degradation allowed
    /*mse_scale_tolerance=*/3.0f // 3× per-level MSE degradation allowed
);

if (!result.passed) {
    result.report();
    // Will print measured vs baseline rates + per-level MSE comparison
}
```

## Reference Scene Baselines

### Cornell Box
- **Scene**: 12 triangles, 2 emissive surfaces, diffuse walls
- **Expected rate**: α ≈ -0.9
- **Characteristics**: Standard MC convergence, no caustics, no envmap

### Glass Sphere
- **Scene**: Cornell box + specular glass sphere at center
- **Expected rate**: α ≈ -0.5
- **Characteristics**: Caustics from refracted light → slower convergence
- **Photon system test**: Caustic region should brighten with photon budget

### Envmap Outdoor
- **Scene**: Ground plane + environment map lighting
- **Expected rate**: α ≈ -0.8
- **Characteristics**: Tests envmap importance sampling quality

### Furnace
- **Scene**: All-white lambertian room, emissive ceiling, albedo=1.0
- **Expected rate**: α ≈ -1.0
- **Expected luminance**: 1.0 (energy conservation)
- **Characteristics**: Not compared by MSE but by mean luminance deviation

## When to Update Baselines

Update baselines when:
1. Algorithm improvement is **intentional** and **verified** (e.g., better importance sampling)
2. New baseline is strictly better (lower MSE at all SPP levels, steeper rate)
3. Scene geometry changed (different reference images needed)

**Never** update baselines to mask a regression. If a code change degrades convergence, fix the code, not the baseline.

## Debugging Convergence Failures

If `compare_to_baseline` fails:

1. **Rate regression** (`rate_ok = false`):
   - Measured rate is less negative than baseline (converging slower)
   - Check recent changes to importance sampling, MIS weights, RR
   - Use `render-diagnostics` to identify which stage's variance increased

2. **MSE regression** (`all_mse_ok = false`):
   - A specific SPP level has 3× worse MSE than baseline
   - Could be energy loss (dark bias) or gain (bright bias)
   - Check clamp limits, material changes, emitter distribution

3. **Low R²** (`r_squared < 0.8`):
   - Non-monotonic MSE reduction (MSE increases at some SPP level)
   - Usually indicates a bug: correlation between samples, broken accumulation
   - Should never happen with correct implementation
