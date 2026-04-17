# Test Taxonomy

## 1. Unit Tests (per-stage, fast, CPU-only where possible)

Each pipeline stage has a dedicated test executable and CMake target.

| Phase | Target | Tests | Key Validations |
|---|---|---|---|
| 1 | core_test | 10 | Header compilation, config constants, type traits |
| 2 | scene_test | 10 | Material, Triangle, Texture, EnvMap, PBRT parsing, SceneProfile |
| 3 | accel_test | 4 | GAS/IAS build, OptixInstance pack, SBT layout |
| 4 | material_test | 34 | Furnace, reciprocity, PDF chi-squared, all 9 MaterialTypes |
| 5 | lighting_test | 4 | NEE shared types, emitter CDF, MIS weights |
| 6 | integrator_test | 5 | PathState, RR thresholds, clamping, path tracer types |
| 7 | photon_test | 6 | Photon struct, storage, flags, flux conservation |
| 8 | guide_test | 6 | SDTree init/deposit/refine/flatten/reset |
| 9 | postfx_test | 7 | Tonemap (ACES+sRGB), firefly filter, bloom, pipeline |
| 10 | diagnostics_test | 8 | Variance tracker, convergence analyzer, bottleneck report |
| 11 | oracle_test | 10 | Image metrics, noise analysis, artifact detection, verdict |
| 12 | harness_test | 15 | Framework, statistical tests, convergence, reference scenes |
| 13 | app_test | 12 | RenderConfig, CLI, FrameBuffer, font overlay, debug state |

### Unit Test Rules
- Must compile without GPU (CPU-only headers). Use `CUDAToolkit_INCLUDE_DIRS` for `vector_types.h`.
- Use `TestCounters` from `test_framework.h` for consistent reporting.
- Each test: `TF_BEGIN → TF_EXPECT (assertions) → TF_PASS`. Return from test function on first failure.
- `main()` returns `counters.report("Suite Name")` — exit code 0 = all pass.

## 2. Integration Tests (multi-stage, GPU required)

Integration tests render actual images and validate against quality thresholds.

| Test | Stages Exercised | SPP | Quality Gate |
|---|---|---|---|
| Direct-only render | scene → accel → lighting → integrator → postfx | 64 | RMSE < 0.15 vs reference |
| Full path-traced render | All stages | 256 | RMSE < 0.08 vs reference |
| Guide-assisted render | All + guide | 256 | Variance < unguided at same SPP |
| Photon caustics | scene → accel → photon → integrator | 256 | Caustic region energy > threshold |

### Integration Test Naming
- Prefix with `Integration` in test name (or Google Test filter pattern)
- Excluded from fast test runs: `--gtest_filter=-*Integration*:*PixelComparison*:*GroundTruth*`

## 3. Convergence Regression Tests

The key innovation. Tracks MSE-at-N-SPP over time to catch convergence regressions.

### Standard SPP Levels
`{16, 64, 256, 1024}` — defined in `convergence_test.h`

### Process
1. Render scene at each SPP level
2. Compute MSE vs reference image at each level
3. Fit log-log line: `log(MSE) = a + b * log(SPP)`
4. Slope `b` is the convergence rate (should be ≈ -1.0 for unbiased MC)
5. Compare against stored baseline (`tests/data/convergence_baselines/`)

### Failure Conditions
- **Rate regression**: `measured_rate > baseline_rate * (1 - tolerance)` where tolerance = 0.15 (15%)
- **MSE regression**: Any SPP level MSE exceeds baseline by 3× scale factor
- **R² too low**: Fit quality R² < 0.8 suggests non-monotonic convergence (bug)

### Expected Convergence Rates by Scene
| Scene | Expected Rate α | Notes |
|---|---|---|
| Cornell box | ≈ -0.9 | Standard diffuse, near-ideal 1/N |
| Glass sphere | ≈ -0.5 | Caustics slow convergence |
| Envmap outdoor | ≈ -0.8 | Envmap importance sampling helps |
| Furnace | ≈ -1.0 | Perfect energy conservation target |

## 4. Cross-Stage Diagnostic Tests

Validate that the diagnostics pipeline correctly identifies noise sources.

| Scenario | Expected Bottleneck | Verified Output |
|---|---|---|
| Direct-lit scene, no indirect | direct-lighting | `direct_variance_fraction > 0.8` |
| Complex indirect scene | path-guide | `indirect_variance_fraction > 0.6` |
| Caustic scene, no photons | photon-system | `missing_caustics = true` |
| All converged | none (balanced) | All fractions < 0.4 |

## Test Execution

### Fast Tests (< 30 seconds, no GPU, pre-commit safe)
```
ppt_tests.exe --gtest_filter=-*Integration*:*PixelComparison*:*GroundTruth*:*PerRay*:*SpeedTest*:*SpeedTweaks*:*CpuGpu*
```

### All Tests (includes GPU integration)
```
ppt_tests.exe --gtest_print_time=1
```

### Individual Stage Tests
```
build/<stage>_test.exe
```
