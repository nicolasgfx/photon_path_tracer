# Statistical Tests Reference

## Chi-Squared PDF Test

**Purpose**: Validate that sampled directions match the expected PDF distribution.

**Implementation**: `statistical_tests.h :: chi_squared_test()`

### How It Works
1. Bin N samples from the sampler into `num_bins` histogram bins
2. Compute expected count per bin from the analytical PDF
3. Compute chi-squared statistic: `χ² = Σ (observed_i - expected_i)² / expected_i`
4. Compute p-value via regularized upper incomplete gamma: `Q(df/2, χ²/2)`
5. Pass if `p_value >= significance_level` (default 0.01)

### Usage Pattern
```cpp
int num_bins = 64;
int N = 10000;
std::vector<int> observed(num_bins, 0);
std::vector<double> expected(num_bins);

// Sample and bin
for (int i = 0; i < N; ++i) {
    float3 dir = sample_bsdf(material, wi, u1, u2);
    int bin = direction_to_bin(dir, num_bins);
    observed[bin]++;
}

// Expected from PDF
for (int b = 0; b < num_bins; ++b) {
    float3 dir_center = bin_to_direction(b, num_bins);
    float pdf = evaluate_pdf(material, wi, dir_center);
    expected[b] = N * pdf * solid_angle_per_bin;
}

auto result = statistical_tests::chi_squared_test(observed, expected);
result.report("Lambertian PDF");  // chi2=58.3, df=63, p=0.64 → PASS
```

### Interpretation
- **p ≥ 0.01**: Samples consistent with expected PDF (PASS)
- **p < 0.01**: Statistically significant deviation (FAIL), indicates sampling bug
- Skip bins where `expected[i] < 1e-12` (empty bins cause division by zero)

### Common Failure Causes
- Wrong PDF normalization (not integrating to 1 over hemisphere)
- Sample/PDF mismatch (sampling from one distribution, evaluating another)
- Hemisphere orientation bug (sampling above normal when should be below)

---

## Furnace Test

**Purpose**: Energy conservation. A perfect white lambertian room converges to uniform radiance.

**Implementation**: `statistical_tests.h :: furnace_test()`

### Setup
- All surfaces: albedo = 1.0 (perfect white lambertian)
- Emissive ceiling: Le = 1.0
- No absorption → energy neither created nor destroyed
- Expected result: every pixel converges to luminance ≈ 1.0

### Usage
```cpp
auto result = statistical_tests::furnace_test(
    rgb_buffer, num_pixels, /*expected=*/1.0f, /*tolerance=*/0.05f);
result.report("Energy conservation");
```

### Interpretation
- **|mean_luminance - 1.0| ≤ 0.05**: Energy conserved (PASS)
- **mean_luminance < 0.95**: Energy loss → check RR threshold, clamp limits, material albedo
- **mean_luminance > 1.05**: Energy gain → check emitter double-counting, MIS weights

### Reference Scene
Use `reference_scenes::furnace()` which provides:
- White box: 5 walls (material 0, albedo 1.0) + emissive ceiling (material 1, Le 1.0)
- Camera at origin, `is_furnace = true`, `expected_mean_luminance = 1.0`

---

## Reciprocity Test

**Purpose**: BSDF symmetry — `f(ωi → ωo) ≈ f(ωo → ωi)`.

**Implementation**: `statistical_tests.h :: reciprocity_test()`

### How It Works
1. Generate `num_pairs` direction pairs (wi, wo) via stratified hemisphere sampling
2. Evaluate BSDF in both directions: `f_fwd = eval(wi, wo)`, `f_rev = eval(wo, wi)`
3. Compute relative error: `|f_fwd - f_rev| / max(|f_fwd|, |f_rev|, ε)`
4. Pass if max relative error < tolerance (default 0.01)

### Usage
```cpp
auto result = statistical_tests::reciprocity_test(
    [&](float3 wi, float3 wo) {
        return evaluate_bsdf(material, wi, wo);
    }, /*num_pairs=*/100, /*tolerance=*/0.01f);
result.report("GGX reciprocity");
```

### Materials That Should Pass
All physically-based BSDFs: Lambertian, GGX conductor, GGX dielectric, thin-film, plastic

### Known Exceptions
- Fresnel-weighted BSDFs with extreme grazing angles may show numerical precision issues
- Use tolerance ≈ 0.05 for glass/specular at grazing geometry

---

## Kolmogorov-Smirnov Test

**Purpose**: Verify samples follow uniform distribution U[0,1].

**Implementation**: `statistical_tests.h :: ks_test_uniform()`

### How It Works
1. Sort N samples
2. Compute KS statistic: `D = max_i |F_empirical(x_i) - F_uniform(x_i)|`
3. Critical value at 95% confidence: `1.36 / √N`
4. Pass if `D < critical_value`

### Usage
```cpp
float samples[1000];
for (int i = 0; i < 1000; ++i)
    samples[i] = rng.uniform_float();

double ks = statistical_tests::ks_test_uniform(samples, 1000);
bool ok = ks < 1.36 / std::sqrt(1000.0);  // ≈ 0.043
```

### Applications
- Validate RNG quality (uniform output)
- Check stratified jitter doesn't break uniformity
- Verify importance sampling CDF inversion

---

## Convergence Rate Validation

**Purpose**: Verify that MSE decreases at expected rate as SPP increases.

**Implementation**: `convergence_test.h :: log_log_fit()`, `compare_to_baseline()`

### Theory
For an unbiased Monte Carlo estimator, MSE ∝ 1/N where N = SPP.
In log-log space: `log(MSE) = a + b * log(SPP)` with slope `b ≈ -1.0`.

### Usage
```cpp
// Measure MSE at standard levels
int spp[] = {16, 64, 256, 1024};
float mse[] = {0.15f, 0.04f, 0.01f, 0.003f};

auto fit = convergence_test::log_log_fit(spp, mse, 4);
// fit.slope ≈ -0.94, fit.r_squared ≈ 0.998

// Compare against stored baseline
ConvergenceBaseline baseline;
convergence_test::load_baseline("tests/data/cornell_box.baseline", baseline);
auto result = convergence_test::compare_to_baseline(
    {16,64,256,1024}, {0.15f,0.04f,0.01f,0.003f}, baseline);
result.report();
```

### Convergence Rate Guide
| Rate | Meaning | Typical Cause |
|---|---|---|
| ≈ -1.0 | Ideal MC convergence | Well-importance-sampled paths |
| -0.7 to -0.9 | Acceptable | Minor variance from complex illumination |
| -0.4 to -0.7 | Slow | Caustics, poor importance sampling, missing guide |
| > -0.4 | Very slow | Bug or fundamentally hard light transport |
| > 0.0 | Diverging | Bug — energy growth, broken RR, bad PDF |

### Baseline File Format
```
cornell_box
16 0.1500
64 0.0400
256 0.0100
1024 0.0030
```
One header line (scene name), then `spp mse` pairs per line.
