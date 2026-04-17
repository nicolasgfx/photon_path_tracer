---
name: test-harness
description: 'Convergence regression testing, reference comparison, statistical validation, and cross-stage test orchestration. Use when: writing tests for any pipeline stage, setting up convergence baselines, running furnace/reciprocity/chi-squared tests, validating render quality against reference images, creating mock test scenes, or tracking test coverage across the pipeline.'
---

# Test Harness (Meta-Skill C)

## Role

Orchestrate testing across all pipeline stages. Convergence regression testing. Statistical validation. Provides shared utilities that all per-stage tests build upon.

## Key Interfaces

### Source Files
- `src/testing/test_framework.h` — TestCounters, BufferStats, RMSE, luminance helpers
- `src/testing/statistical_tests.h` — chi-squared, furnace, reciprocity, KS tests
- `src/testing/convergence_test.h` — log-log fit, ConvergenceBaseline, save/load
- `src/testing/reference_scenes.h` — Cornell box, glass sphere, envmap, furnace scenes

### Critical Types

```cpp
// ── Test framework ──────────────────────────────────────────────
struct TestCounters {
    int tests_run, tests_passed, tests_failed;
    void begin(const char* name);
    void pass(const char* name);
    void fail(const char* name, const char* cond);
    int report(const char* suite_name) const;
};

// Macros: TF_BEGIN(counters, name), TF_PASS(counters, name),
//         TF_EXPECT(counters, cond, msg)

namespace test_utils {
    BufferStats compute_buffer_stats(const float*, int);
    float compute_rmse(const float* a, const float* b, int num_pixels);
    float mean_luminance(const float* rgb, int num_pixels);
    bool  buffer_is_clean(const float*, int);
    bool  buffer_has_nonzero(const float*, int, float threshold);
    std::vector<float> normalize_by_spp(const std::vector<float>&, int spp);
}

// ── Statistical tests ───────────────────────────────────────────
struct ChiSquaredResult { double chi2, p_value; int df; bool passed; };
ChiSquaredResult chi_squared_test(const int* observed, const double* expected,
                                   int bins, double significance = 0.01);
FurnaceResult furnace_test(const float* rgb, int num_pixels,
                           float expected = 1.0f, float tolerance = 0.05f);
template<typename EvalFunc>
ReciprocityResult reciprocity_test(EvalFunc eval_bsdf, int pairs, float tol);
double ks_test_uniform(const float* samples, int n);

// ── Convergence testing ─────────────────────────────────────────
struct ConvergenceBaseline {
    std::string scene_name;
    std::vector<int> spp_levels;
    std::vector<float> mse_values;
    float convergence_rate;
};
LinFitResult log_log_fit(const int* spp, const float* mse, int n);
ConvergenceResult compare_to_baseline(spp, mse, baseline, rate_tol, mse_tol);
bool save_baseline(const ConvergenceBaseline&, const std::string& filepath);
bool load_baseline(const std::string& filepath, ConvergenceBaseline& out);

// ── Reference scenes ────────────────────────────────────────────
struct ReferenceScene {
    std::string name;
    Scene scene;
    TestCamera camera;
    float expected_mean_luminance;
    float expected_convergence_rate;
    bool  has_caustics, has_envmap, is_furnace;
};
ReferenceScene cornell_box();
ReferenceScene glass_sphere();
ReferenceScene envmap_outdoor();
ReferenceScene furnace();      // white room for energy conservation
```

## Test Taxonomy

### 1. Unit Tests (per-stage, fast)
Each pipeline stage has its own test file and CMake target:

| Stage | Test Target | Key Tests |
|---|---|---|
| Core | core_test | Header compilation, config constants |
| Scene | scene_test | Material, Triangle, PBRT parsing, SceneProfile |
| Accel | accel_test | GAS/IAS build, OptixInstance pack, SBT |
| Material | material_test | Furnace, reciprocity, PDF chi-squared (34 tests) |
| Lighting | lighting_test | NEE shared types, emitter CDF, MIS weights |
| Integrator | integrator_test | PathState, RR, clamping, path tracer types |
| Photon | photon_test | Photon struct, storage, flags, flux conservation |
| Guide | guide_test | SDTree init/deposit/refine/flatten/reset |
| PostFx | postfx_test | Tonemap, firefly, bloom, pipeline |
| Diagnostics | diagnostics_test | Variance, convergence, bottleneck |
| Oracle | oracle_test | Metrics, noise, artifacts, verdict |
| Test Harness | harness_test | Framework, stats tests, convergence, scenes |
| App | app_test | Config, CLI, framebuffer, font, debug |

### 2. Integration Tests (multi-stage, GPU required)
- Direct-only render at 64 SPP → RMSE < threshold vs reference
- Full render at 256 SPP → RMSE < threshold vs reference
- Guide training → variance reduction vs unguided

### 3. Convergence Regression Tests
- Render at [16, 64, 256, 1024] SPP → measure MSE at each level
- Fit log-log line → convergence rate α
- Compare against stored baseline: fail if rate regresses by > 15%
- Per-level MSE check: fail if any level is 3× worse than baseline

### 4. Cross-Stage Diagnostic Tests
- Render → collect ConvergenceReport → verify bottleneck identification
- Known-noisy scene → verify diagnostics correctly identifies source

## Development Procedures

### Writing a Test for a New Stage
1. Create `src/stage_test.cpp` (or appropriate location)
2. Include `testing/test_framework.h` for macros
3. Use `TF_BEGIN(counters, "test name")`, `TF_EXPECT(counters, cond, msg)`, `TF_PASS(counters, "test name")`
4. Add CMake target with `CUDAToolkit_INCLUDE_DIRS` (needed for `types.h`)
5. Return `counters.report("Suite Name")` from main
6. All tests must pass with 0 regressions across existing targets

### Setting Up a Convergence Baseline
```cpp
// Render at multiple SPP levels, record MSE
auto baseline = convergence_test::create_baseline(
    "cornell_box", {16, 64, 256, 1024}, {0.15, 0.04, 0.01, 0.003});
convergence_test::save_baseline(baseline, "tests/data/cornell_box.baseline");
```

### Running Convergence Regression
```cpp
ConvergenceBaseline baseline;
convergence_test::load_baseline("tests/data/cornell_box.baseline", baseline);
auto result = convergence_test::compare_to_baseline(
    spp_levels, measured_mse, baseline);
result.report();  // prints rate comparison + pass/fail
```

### Statistical Test Guide

**Chi-squared PDF test** — Validate that sampled directions match expected PDF:
```cpp
// Bin N samples, compare against expected distribution
auto result = chi_squared_test(observed_bins, expected_bins, num_bins);
// p_value >= 0.01 means samples are consistent with the expected PDF
```

**Furnace test** — Energy conservation (white room → radiance = 1.0):
```cpp
auto result = furnace_test(rgb_buffer, num_pixels, 1.0f, 0.05f);
// Mean luminance should be within 5% of 1.0
```

**Reciprocity test** — BSDF symmetry f(wi→wo) == f(wo→wi):
```cpp
auto result = reciprocity_test([&](float3 wi, float3 wo) {
    return evaluate_bsdf(material, wi, wo);
}, 100, 0.01f);
```

**KS test** — Verify samples are uniformly distributed:
```cpp
double ks_stat = ks_test_uniform(samples, num_samples);
// ks_stat < 1.36/√N for 95% confidence
```

## Reference Scenes

| Scene | Key property | Expected α | Tests |
|-------|-------------|-----------|-------|
| Cornell Box | 12 tris, 2 emissive | 0.9 | Basic convergence |
| Glass Sphere | Cornell + glass sphere | 0.5 | Caustics, photon system, guide |
| Envmap Outdoor | Ground plane + envmap | 0.8 | Envmap importance sampling |
| Furnace | White room, emissive ceiling | 1.0 (luminance) | Energy conservation |

## Testing Procedures

### Harness Self-Test (harness_test)
1. TestCounters tracking: begin/pass/fail counts
2. BufferStats: known data → correct mean/max/min/nan/inf
3. RMSE: identical → 0, known difference → expected value
4. Chi-squared: uniform → high p-value (pass), skewed → low p-value (fail)
5. Furnace: perfect 1.0 → pass, biased → fail
6. Log-log fit: synthetic 1/N data → slope ≈ -1.0
7. Convergence baseline: save → load roundtrip; within-tolerance → pass; too-bad → fail
8. Reference scenes: Cornell 12+ tris, glass has_caustics, envmap has_envmap, furnace luminance=1.0
9. KS test: uniform → pass, biased → fail
