# Path Tracer Redesign Part 2 — Path Tracing

> This document covers the camera-side path tracer that replaces the current first-hit NEE architecture.  Part 1 (Photon Mapping) establishes the photon subsystem; this document builds on top of it.

**Motivation:** The current ray tracer gathers photon information via kNN photon search from the hash grid, with visible limitations and weak convergence properties.  We need a switch to full path tracing.  Instead of random direction sampling, the photon information is used to sample relevant directions.

That means at any hitpoint, we sample the kNN neighbourhood for photons and analyse them.  The results of the analysis are:
1. **Directional signal** — which directions are the photons coming from?
2. **Spatial density** — many photons close together can mean a high-frequency area, where many rays might be needed to address this complexity.
3. **Spatial variance** — few photons with high spatial variance indicate an undersampled area, leading to noise.  Probably need to shoot more photons.  Vice versa, many photons in a small space might reduce the need for many rays.
4. **Photon type** — presence of caustic photons means that we have to cast more rays than normal into the caustic photons' direction in order to get a clean, high-frequency caustic.
5. **Photon energy** — full saturation of a pixel means that we can reduce the number of rays for this pixel.  We also know that this light source triangle (stored in photon) has a dominating local effect.

**Constraints:**
- Two-pass architecture: (1) photon tracing, (2) camera path tracing with photon guidance
- CPU and GPU consistency remains a high-priority requirement
- Part 1 photon subsystem is unchanged — only the camera-side code is replaced

**Current state (v2.3):** Camera rays are first-hit probes.  Specular surfaces are followed through a chain; at the first non-delta hit, NEE captures direct light and a hash-grid photon gather captures indirect light.  Glossy surfaces get a short BSDF continuation loop (up to 2 bounces).  There is no multi-bounce path tracing — indirect illumination beyond the photon map's kNN estimate is lost.

**Target state (v3):** A photon-guided path tracer.  Every camera bounce performs:
1. **NEE** — 1 shadow ray toward a light source, MIS-weighted against the BSDF (standard practice in all production path tracers; Veach 1997)
2. **Photon-guided direction sampling** — the cell-bin histogram provides a directional PDF of incoming light; the next-bounce direction is drawn from a **BSDF + photon-guide mixture PDF**.  This is the core innovation: the photon map tells the path tracer *where light is coming from*, replacing blind random-walk exploration with informed importance sampling.
3. **Photon analysis** — per-cell density, variance, type, and energy drive adaptive decisions (sample count, guide fraction, caustic handling)

The guide fraction (probability of picking the photon PDF vs the BSDF PDF) is always active.  In cells with no photons, it degrades gracefully to pure BSDF sampling ($\alpha = 0$).  The photon map shifts from being the *sole* indirect answer to being a **guide and fallback**: guide at every bounce, final gather at path termination, and direct density estimate for caustics.

---

## 1. Cleanup Plan

This redesign is an opportunity to make the codebase clean enough for an academic reference implementation: every file in the right place, no dead code, no duplication, consistent naming, and clear separation of concerns.

We cleaned up the project in Part 1.  Take this as basis and continue with the same value system (separation of concerns, no code duplication, shared logic).  Use the rewriting to remove ALL gates, parameters and complexities coming from the existing NEE ray tracer.  We want a clean, fresh parametrisation.

### 1.1 NEE Legacy Removal

The current NEE-centric architecture introduces parameters, code paths, and complexity that do not belong in a multi-bounce path tracer.  Remove or repurpose:

| Item | File | Action |
|---|---|---|
| `DEFAULT_NEE_LIGHT_SAMPLES` | `config.h` | **Remove.** Path tracer uses 1 NEE sample per bounce (standard). |
| `DEFAULT_NEE_DEEP_SAMPLES` | `config.h` | **Remove.** No separate deep-bounce NEE count needed. |
| `DEFAULT_NEE_COVERAGE_FRACTION` | `config.h` | **Remove.** Replace with single power-weighted alias table (no coverage mix). |
| `nee_shadow_sample_count()` | `nee_shared.h` | **Remove.** Always 1 shadow ray per bounce. |
| `dev_nee_direct()` (multi-sample M loop) | `optix_nee_dispatch.cuh` | **Replace** with single-sample NEE (1 shadow ray, 1 emitter pick). |
| `dev_nee_golden_stratified()` | `optix_nee_dispatch.cuh` | **Remove.** Golden-ratio stratification is for multi-sample NEE only. |
| `dev_nee_select_global()` (coverage mix) | `optix_nee_dispatch.cuh` | **Simplify** to pure power CDF alias table sample. |
| `sample_direct_light()` (coverage mix) | `direct_light.h` | **Simplify** to pure power CDF alias table sample. |
| `direct_light_pdf()` (coverage mix) | `direct_light.h` | **Simplify** to pure power CDF. |
| `emissive_area_alias_table` | `scene.h` | **Remove.** Only power-weighted table needed. |
| `DEFAULT_MAX_SPECULAR_CHAIN` | `config.h` | **Keep** but rename to `DEFAULT_MAX_BOUNCES_CAMERA`. Specular chain is now part of the general bounce loop. |
| `DEFAULT_MAX_GLOSSY_BOUNCES` | `config.h` | **Remove.** No separate glossy continuation loop — the path tracer handles all bounces uniformly. |
| `nee_light_samples`, `nee_deep_samples` | `LaunchParams` | **Remove** both fields. |
| `light_cache_*`, `shadow_ray_*` | `LaunchParams` | **Remove.** Light importance cache and coverage-based shadow targets are NEE-specific. |
| `LightCache`, `light_cache.h` | `renderer/` | **Remove** entirely. |
| `EmitterPointSet` | `renderer/` | **Remove** entirely. |
| `lobe_balance` (Bresenham) | `LaunchParams` | **Remove.** No per-pixel lobe balance in the path tracer. |
| `DEFAULT_ENABLE_BRESENHAM_BSDF` | `config.h` | **Remove.** |
| `debug_shadow_rays` | `LaunchParams` | **Remove.** First-hit debug mode removed. |
| `is_final_render` | `LaunchParams` | **Remove.** Single rendering path — no debug-first-hit vs final-render split. |
| `PixelLighting` (6-channel decomposition) | `pixel_lighting.h` | **Simplify.** Keep `combined` + `direct` + `indirect` only. Remove `glossy_indirect`, `translucency`, `indirect_caustic` as separate channels — the path tracer accumulates everything into one throughput. |

### 1.2 Config.h Redesign

Replace the NEE-centric §2/§4 with path-tracer parameters:

```cpp
// =====================================================================
//  §2  CORE RENDERING (v3 — Photon-Guided Path Tracing)
// =====================================================================

constexpr int DEFAULT_SPP = 64;                        // [R]
constexpr int STRATA_X = 8;
constexpr int STRATA_Y = 8;

// ── Path depth ──────────────────────────────────────────────────────
constexpr int DEFAULT_MAX_BOUNCES_CAMERA = 12;         // [R]  max camera path depth
constexpr int DEFAULT_MIN_BOUNCES_RR     = 3;          // [R]  guaranteed bounces before RR
constexpr float DEFAULT_RR_THRESHOLD     = 0.95f;      // [R]  max survival probability

// ── Photon-guided sampling ──────────────────────────────────────────
constexpr float DEFAULT_GUIDE_FRACTION   = 0.5f;       // [R]  probability of guided vs BSDF sample
constexpr bool  DEFAULT_USE_GUIDE        = true;        // [K]  enable/disable guided sampling

// ── Photon density fallback ─────────────────────────────────────────
constexpr int  DEFAULT_GUIDE_FALLBACK_BOUNCE = 3;       // [R]  switch to photon gather after this bounce
constexpr bool DEFAULT_PHOTON_FINAL_GATHER   = true;    // [K]  use photon map as final gather at terminal bounces
```

### 1.3 Files to Create

| File | Purpose |
|---|---|
| `optix_path_trace_v3.cuh` | New photon-guided path tracer kernel (replaces `optix_path_trace.cuh`) |
| `src/renderer/path_trace.h` | CPU reference path tracer (replaces `render_pixel()`) |
| `src/photon/photon_analysis.h` | Photon neighbourhood analysis (density, variance, type, energy) |

### 1.4 Files to Delete

| File | Reason |
|---|---|
| `renderer/light_cache.h` | NEE-specific light importance cache |
| `optix_nee_dispatch.cuh` | Multi-sample NEE dispatch (collapse into `optix_nee.cuh`) |

### 1.5 Remaining Part 1 Cleanup (carry-forward)

These items from Part 1 §11 should be completed as part of this work:

| Item | Action |
|---|---|
| `dev_bsdf_pdf`, `dev_bsdf_sample`, `DevBSDFSample` | Rename to `bsdf_pdf`, `bsdf_sample`, `BSDFSample` |
| `dev_generate_camera_ray` | Inline and remove wrapper |
| `main.cpp` (271 → ≤200 lines) | Extract remaining setup logic |
| `optix_renderer.cpp` (1751 → ≤1200) | Move upload logic to `optix_upload.cpp`, denoiser to `optix_denoiser.cpp` |

---

## 2. Architecture — Two-Pass Photon-Guided Path Tracing

### 2.1 Pipeline Overview

```
Pass 1: Photon Tracing (updated — see §9.6 for Part 1 changes)
  ├─ Emit global photons    → global_photons (PhotonSoA)
  ├─ Emit caustic photons   → caustic_photons (PhotonSoA)
  ├─ Volume photon deposits  → volume_photons (PhotonSoA)     [NEW]
  ├─ Build hash grids       → global_grid, caustic_grid, volume_grid
  ├─ Build CellBinGrid      → cell_bin_grid (directional histograms)
  ├─ Build VolumeCellBinGrid → vol_cell_bin_grid (3D volume)   [NEW]
  └─ Photon analysis        → per-cell density/variance/type maps  [NEW]

Pass 2: Camera Path Tracing (NEW — replaces NEE first-hit)
  ├─ For each pixel:
  │   ├─ Generate camera ray
  │   ├─ Iterative bounce loop:
  │   │   ├─ Trace ray → hit (or medium scatter event)
  │   │   ├─ If inside medium → free-flight sampling:
  │   │   │   ├─ Sample distance t from exponential(σ_t)
  │   │   │   ├─ If t < t_surface → scatter inside medium:
  │   │   │   │   ├─ NEE toward light (with Beer-Lambert shadow)
  │   │   │   │   ├─ Volume photon guide (vol_cell_bin_grid)
  │   │   │   │   ├─ HG phase function direction sample
  │   │   │   │   └─ Continue bounce loop from scatter point
  │   │   │   └─ If t ≥ t_surface → apply transmittance, continue to surface
  │   │   ├─ If emissive → add emission (MIS-weighted)
  │   │   ├─ If delta (mirror/glass) → specular bounce, continue
  │   │   ├─ If translucent → Fresnel R/T, push/pop medium, continue
  │   │   ├─ NEE: 1 shadow ray → direct light (MIS with BSDF)
  │   │   ├─ Sample next direction:
  │   │   │   ├─ With probability p_guide: guided sample (cell histogram)
  │   │   │   └─ With probability 1-p_guide: BSDF sample
  │   │   │   └─ MIS-combine both PDFs
  │   │   ├─ Russian roulette (after min bounces)
  │   │   └─ Optional: photon map final gather at terminal bounce
  │   └─ Accumulate spectral radiance
  └─ Tonemap + denoiser
```

### 2.2 Key Design Decision: When to Use the Photon Map

The photon map serves three distinct roles in the new architecture:

| Role | When | How |
|---|---|---|
| **Guide** | Every non-delta surface bounce | Cell-bin histogram provides a directional PDF for importance sampling. Cheap (O(1) lookup). |
| **Volume guide** | Every medium scatter event | Volume cell-bin grid provides directional PDF inside participating media. Same mechanism, 3D grid instead of surface grid. |
| **Final gather** | Terminal bounce (max depth or RR termination) | Instead of returning black at path termination, query the photon map for an estimate of remaining indirect light. Reduces bias from path truncation. |
| **Caustic capture** | Any bounce hitting a surface with caustic photons | Caustic photons carry L→S→D transport that the camera path cannot efficiently reconstruct. The photon density estimate at caustic-flagged cells is added directly. |
| **Volume density** | Medium scatter events + final gather inside media | Volume photon kNN density estimate for in-medium radiance. Used at terminal scatter events and to supplement guided sampling. |

### 2.3 What Changes vs Current Architecture

| Aspect | Current (v2.3) | New (v3) |
|---|---|---|
| Camera bounce depth | 1 (first hit) + 2 glossy | 12 (iterative loop) |
| NEE shadow rays | M per hit (4–64) | 1 per hit |
| Indirect light | Photon gather only | Path tracing + photon guide + photon final gather |
| Glossy reflections | Separate continuation loop | Part of main bounce loop |
| Caustics in reflection | Not captured | Naturally traced |
| Multi-bounce colour bleeding | Missing | Captured by iterative bounces |
| Translucent/volume scattering | Beer-Lambert only (no in-medium scatter) | Full volumetric path tracing with photon-guided scatter |
| Photon map role | Sole indirect source | Guide + final gather + caustic backup + volume guide |

---

## 3. Photon Analysis at Each Hit Point

> This section implements the four analysis dimensions from the project specification.

At every non-delta camera bounce, the path tracer reads the local photon neighbourhood to make sampling decisions.  The analysis is performed on the **cell-bin grid** (O(1)) not kNN (too expensive per bounce).

### 3.1 Directional Analysis — "Where is the light coming from?"

**Source:** CellBinGrid `bin_flux[32]` for the cell containing the hit point.

**Use:** The Fibonacci-sphere histogram already encodes the incoming radiance distribution.  Bins with high flux indicate dominant light directions.  The existing guided sampling (Part 1 §4) directly uses this histogram.

**Enhancement for v3:**  Weight the guide fraction `p_guide` by histogram quality:
- If `total_flux > 0` and the histogram has ≥3 non-zero bins: `p_guide = DEFAULT_GUIDE_FRACTION`
- If only 1–2 bins non-zero (highly directional, likely caustic): `p_guide = 0.7` (trust the guide more)
- If `total_flux == 0` (no photons in cell): `p_guide = 0` (fall back to BSDF only)

```cuda
float compute_guide_fraction(const GuidedHistogram& h) {
    if (!h.valid) return 0.f;
    int active_bins = 0;
    for (int k = 0; k < h.num_bins; ++k)
        if (h.bin_flux[k] > 0.f) active_bins++;
    if (active_bins <= 2) return 0.7f;  // strong directional signal
    return DEFAULT_GUIDE_FRACTION;       // general guidance
}
```

### 3.2 Spatial Density — "How complex is the lighting here?"

**Source:** Cell photon count from `CellInfoCache`.

**Use:** High photon density = high-frequency lighting (e.g., caustic focus, shadow boundary).  The adaptive SPP allocator (§7) uses this to assign more samples to complex regions.

**Metric:** Photon count per unit area in the cell:

$$\rho_\text{cell} = \frac{N_\text{photons}}{A_\text{cell}}$$

where $A_\text{cell} = (\text{cell\_size})^2$ (projected area on the surface).

### 3.3 Spatial Variance — "Is this region undersampled?"

**Source:** Flux variance across neighbouring cells.  `CellCacheInfo.flux_variance` (Welford) and `CellCacheInfo.caustic_cv` are already computed.

**Use:** High variance between adjacent cells indicates an undersampled region — the photon map has not converged here.  Two responses:
1. **More camera samples** (adaptive SPP) to average out the noise
2. **More photons** in the next re-trace (view-adaptive budgeting, §8)

**Metric:** Coefficient of variation across the 3×3×3 cell neighbourhood:

$$\text{CV} = \frac{\sigma_\text{flux}}{\bar{\mu}_\text{flux} + \epsilon}$$

### 3.4 Photon Type — "Are there caustics here?"

**Source:** `path_flags` of photons in the cell (stored in `PhotonSoA`).  `CellCacheInfo.caustic_count`, `.caustic_flux`, `.caustic_cv`, `.glass_fraction`, and `.is_caustic_hotspot` are already computed.

**Use:** Cells with a high fraction of caustic photons (`CAUSTIC_GLASS`, `CAUSTIC_SPECULAR`) carry high-frequency light transport that the path tracer cannot efficiently reproduce.

**Response:**
- At render time: **Add the photon caustic estimate directly** to the pixel, rather than relying on the path tracer to rediscover it.  This is a density estimate of only the caustic-flagged photons.
- At sample allocation: boost SPP for pixels that contain caustic cells (§7).

### 3.5 Photon Energy — "Is this pixel saturated?"

**Source:** Total flux in the cell.

**Use:** If the cell has very high total flux (saturated highlights, direct light hotspots), additional camera samples yield diminishing returns — the pixel is already bright and noise is perceptually less visible.

**Response:** Reduce SPP for saturated pixels in the adaptive allocator.  The `source_emissive_idx` stored on photons identifies which light source dominates.

### 3.6 Conclusions → Measures

All five analysis dimensions produce conclusions that converge on a small set of actionable measures.  Grouping by measure avoids duplicate logic and keeps the implementation focused on the primary goal: **reduce noise as fast as possible**.

#### M1 — Guide Fraction (`p_guide`)

Controls how much the path tracer trusts the photon histogram vs. the BSDF.

| # | Conclusion | Source | Effect on `p_guide` |
|---|------------|--------|---------------------|
| C1 | Few active bins → strong directional signal | §3.1 | Increase to 0.7 |
| C2 | Low concentration ($c = f_\text{max}/f_\text{total} < 0.1$) → diffuse, guide unhelpful | §3.1 | Decrease toward 0 |
| C3 | `directional_spread > 0.9` → isotropic, skip guide entirely | §3.1 | Set to 0 |
| C4 | Low photon count → histogram unreliable | §3.2 | Scale down proportionally |
| C5 | High flux CV → photon map not converged | §3.3 | Scale down proportionally |
| C6 | Caustic edge (high `caustic_cv`, 1–2 bins) → guide is precise | §3.4 | Increase to 0.7+ |

**Combined formula:**

```cuda
float compute_guide_fraction(const CellCacheInfo& cell,
                             const GuidedHistogram& h) {
    if (!h.valid || cell.photon_count == 0) return 0.f;

    // Base: histogram quality
    float p = DEFAULT_GUIDE_FRACTION;
    if (h.active_bins <= 2) p = 0.7f;          // C1/C6

    // Reliability: scale by photon count (C4)
    p *= fminf(1.f, cell.photon_count / 30.f);

    // Convergence: reduce when variance is high (C5)
    float cv = cell.flux_variance / (cell.irradiance + 1e-6f);
    p *= fmaxf(0.1f, 1.f - cv / 2.f);

    // Skip when too diffuse to guide (C3)
    if (cell.directional_spread > 0.9f) p = 0.f;

    return p;
}
```

#### M2 — Sample Budget per Pixel (Adaptive SPP)

Controls how many camera rays each pixel receives.  §7 builds a per-pixel weight map from these signals.

| # | Conclusion | Source | Effect on SPP |
|---|------------|--------|---------------|
| C1 | High photon density → complex lighting | §3.2 | ↑ more samples |
| C2 | High flux CV → noisy region | §3.3 | ↑ more samples |
| C3 | Caustic cell → high-frequency transport | §3.4 | ↑ more samples |
| C4 | Caustic edge (high `caustic_cv`) → steep gradient | §3.4 | ↑↑ most samples |
| C5 | High total flux → saturated, noise invisible | §3.5 | ↓ fewer samples |

#### M3 — Gather Radius

Controls the kernel size for the final-bounce density estimate.

| # | Conclusion | Source | Effect on radius |
|---|------------|--------|------------------|
| C1 | `adaptive_radius` already scales with local density | §3.2 | Use as base |
| C2 | Caustic photons need tighter kernel | §3.4 | Shrink: $r_\text{caustic} = r_\text{adaptive} \cdot (N_\text{reliable} / N_\text{caustic})^{0.25}$ |
| C3 | Caustic edges need smallest radius | §3.4 | Minimum clamp at edges |

#### M4 — Photon Re-trace Budget

Controls the view-adaptive feedback loop (§8).

| # | Conclusion | Source | Effect |
|---|------------|--------|--------|
| C1 | High-CV visible cells need more photons | §3.3 | Re-weight emitter CDF toward contributing lights |
| C2 | Mean CV across visible cells as convergence criterion | §3.3 | Stop re-tracing when $\overline{\text{CV}} < 0.15$ |

#### M5 — Caustic Additive Contribution

Caustic photon density estimate is always added directly to the pixel — never folded into the path throughput.  L→S→D transport is nearly impossible for camera paths to discover; the photon estimate is the authority.  Double-counting is self-correcting via MIS (probability of randomly finding the caustic path ≈ 0).

#### Deferred Optimisations

The following conclusions are valid but add complexity for marginal gain.  Implement only after M1–M5 are working:

- **Skip NEE at indirect-only cells** (§3.1) — saves shadow rays but requires per-cell `indirect_dominated` tracking.
- **Path depth scaling with density** (§3.2) — saves 1–2 bounces in bright regions but interacts with MIS weights.
- **Variance map as denoiser noise hint** (§3.3) — helps OptiX denoiser but requires auxiliary buffer plumbing.
- **Glass fraction as spectral resolution hint** (§3.4) — informational; no action until spectral transport is implemented.

### 3.7 Photon Analysis Data Structure

```cpp
// photon_analysis.h — Per-cell analysis results
struct CellAnalysis {
    float  total_flux;       // sum of all photon flux in cell
    float  flux_density;     // total_flux / cell_area
    float  flux_cv;          // coefficient of variation (neighbourhood)
    float  caustic_fraction; // fraction of photons with caustic flags
    float  guide_fraction;   // recommended p_guide for this cell
    int    dominant_emitter;  // source_emissive_idx with highest flux
    int    active_bins;      // number of non-zero directional bins
    bool   has_photons;      // any photons at all

    static CellAnalysis empty() {
        return {0, 0, 0, 0, 0, -1, 0, false};
    }
};
```

This is computed once after the photon map build and before the camera pass.  It reuses and extends the existing `CellInfoCache`.

---

## 4. Photon-Guided Path Tracer — Core Algorithm

### 4.1 Pseudocode

```
function path_trace(ray, rng) → Spectrum:
    L = 0                          // accumulated radiance
    throughput = 1                  // spectral throughput
    ior_stack = []                 // nested dielectric tracking

    for bounce = 0 .. MAX_BOUNCES_CAMERA:
        hit = trace(ray)
        if miss: break

        // ── Emission (MIS-weighted) ─────────────────────────────────
        if hit.is_emissive:
            if bounce == 0:
                L += throughput * Le          // camera sees light directly
            else:
                w_bsdf = mis_weight(pdf_bsdf_prev, pdf_nee(hit))
                L += throughput * Le * w_bsdf // MIS with NEE
            break

        // ── Medium interaction (if currently inside a medium) ────────
        if in_medium:
            t_ff = -ln(rng) / sigma_t_hero
            if t_ff < hit.t:                   // scatter before surface
                scatter_pos = ray.origin + t_ff * ray.dir
                throughput *= spectral_mis_scatter_weight(t_ff)
                // NEE at scatter point (with Beer-Lambert shadow attenuation)
                L += throughput * volume_nee(scatter_pos, medium)
                // Volume photon guide for scatter direction
                vol_guide = read_vol_cell_histogram(scatter_pos)
                wi = sample_HG_or_guided(vol_guide, medium.g, rng)
                ray = Ray(scatter_pos, wi)
                continue                       // stay in medium, next iteration
            else:
                throughput *= spectral_mis_transmittance(hit.t)
                // fall through to surface interaction below

        // ── Delta surfaces (mirror, glass) ──────────────────────────
        if hit.is_delta and not hit.is_translucent:
            sb = specular_bounce(ray.dir, hit, ior_stack)
            throughput *= sb.filter
            ray = Ray(sb.new_pos, sb.new_dir)
            continue                           // no NEE, no guide at delta

        // ── Translucent surface (delta + medium transition) ─────────
        if hit.is_translucent:
            sb = specular_bounce(ray.dir, hit, ior_stack)
            throughput *= sb.filter
            if sb.entering:
                medium_stack.push(hit.medium_id)
                in_medium = true
            else:
                medium_stack.pop()
                in_medium = medium_stack.has_medium()
            ray = Ray(sb.new_pos, sb.new_dir)
            continue                           // no NEE at delta surface

        // ── Photon analysis ─────────────────────────────────────────
        analysis = cell_analysis(hit.position)
        guide = read_cell_histogram(hit.position, hit.normal)

        // ── Caustic photon contribution (additive) ──────────────────
        if analysis.caustic_fraction > 0:
            L += throughput * photon_caustic_estimate(hit)

        // ── NEE: 1 shadow ray ───────────────────────────────────────
        (Le_nee, pdf_nee) = nee_sample(hit)
        if Le_nee > 0:
            f = bsdf_evaluate(hit.mat, wo, wi_nee)
            pdf_bsdf_at_nee = bsdf_pdf(hit.mat, wo, wi_nee)
            w_nee = mis_weight(pdf_nee, pdf_bsdf_at_nee)
            L += throughput * f * Le_nee * cos_theta / pdf_nee * w_nee

        // ── Next direction: guided or BSDF ──────────────────────────
        p_guide = analysis.guide_fraction
        if rng < p_guide and guide.valid:
            wi = guided_sample(guide)
            pdf_guide = guided_pdf(guide, wi)
            pdf_bsdf  = bsdf_pdf(hit.mat, wo, wi)
            pdf_combined = p_guide * pdf_guide + (1 - p_guide) * pdf_bsdf
        else:
            wi = bsdf_sample(hit.mat, wo)
            pdf_bsdf  = bsdf_pdf(hit.mat, wo, wi)
            pdf_guide = guided_pdf(guide, wi)
            pdf_combined = (1 - p_guide) * pdf_bsdf + p_guide * pdf_guide

        f = bsdf_evaluate(hit.mat, wo, wi)
        throughput *= f * cos_theta / pdf_combined

        // ── Russian roulette ────────────────────────────────────────
        if bounce >= MIN_BOUNCES_RR:
            p_survive = min(RR_THRESHOLD, max_component(throughput))
            if rng >= p_survive: break
            throughput /= p_survive

        // ── Photon final gather at terminal bounce ──────────────────
        if bounce == MAX_BOUNCES_CAMERA - 1 and PHOTON_FINAL_GATHER:
            L += throughput * photon_density_estimate(hit)
            break

        ray = Ray(hit.pos + offset, wi)
        pdf_bsdf_prev = pdf_combined    // store for next-bounce emission MIS

    return L
```

### 4.2 Key Differences from Current Code

1. **Single bounce loop** replaces specular_chain + glossy_continuation.  Delta surfaces (`is_delta`) just continue the loop like any other bounce.

2. **NEE is always 1 sample.**  No multi-sample averaging, no golden-ratio stratification.  The path tracer's own bounces provide the variance reduction that multi-sample NEE was compensating for.

3. **Emission MIS.**  When a BSDF-sampled direction hits an emissive surface, the contribution is weighted by MIS(pdf_bsdf, pdf_nee).  This requires storing `pdf_bsdf_prev` from the previous bounce.

4. **Caustic photons are additive.**  At every non-delta hit, if the cell contains caustic photons, their density estimate is added.  The path tracer does not try to reproduce L→S→D transport — the photon map handles it.

5. **Photon final gather.**  At the terminal bounce (depth limit or RR kill), instead of returning black, the photon map provides an estimate of the remaining indirect light.  This dramatically reduces low-frequency bias from path truncation.

### 4.3 MIS Formulation

Three sampling strategies compete at each non-delta bounce:

| Strategy | Samples | PDF notation |
|---|---|---|
| NEE (light sampling) | 1 shadow ray | $p_\text{nee}$ |
| BSDF sampling | Via next-bounce direction | $p_\text{bsdf}$ |
| Guided sampling | Via cell-bin histogram | $p_\text{guide}$ |

**NEE vs BSDF** is the standard 2-way power heuristic:

$$w_\text{nee} = \frac{p_\text{nee}^2}{p_\text{nee}^2 + p_\text{bsdf}^2}$$

**Guided vs BSDF** for the next-bounce direction uses a **mixture PDF**:

$$p_\text{combined}(\omega_i) = \alpha \cdot p_\text{guide}(\omega_i) + (1 - \alpha) \cdot p_\text{bsdf}(\omega_i)$$

where $\alpha = p_\text{guide\_fraction}$ (from photon analysis).  This is the one-sample MIS approach: select one strategy stochastically, evaluate the sample under the combined PDF.

When the chosen next-bounce direction hits an emitter, the emission MIS uses $p_\text{combined}$ (not just $p_\text{bsdf}$) against $p_\text{nee}$:

$$w_\text{bsdf} = \frac{p_\text{combined}^2}{p_\text{combined}^2 + p_\text{nee}^2}$$

### 4.4 Specular and Translucent Handling

**Mirror and Glass** have Dirac-delta BSDFs — the direction is deterministic, $p = 1$.  These surfaces:
- Are **not** shaded (no NEE, no photon gather, no guide)
- Update throughput by $K_s$ (mirror) or $T_f$ (glass refract)
- Track IOR stack for nested dielectrics
- Simply continue the bounce loop

**Translucent** surfaces are also delta (Fresnel R/T), but additionally manage a **medium transition**.  When a camera ray enters a Translucent object:
1. Fresnel determines reflect vs refract (same as Glass).
2. On refraction **into** the object: push `medium_id` onto `MediumStack`; set `in_medium = true`.
3. On refraction **out of** the object: pop `MediumStack`.
4. While `in_medium`, each subsequent trace iteration performs **free-flight sampling** before the surface hit test — the ray may scatter inside the medium before reaching the next surface.

At each medium scatter event, the path tracer:
- Queries the **volume photon map** (kNN density estimate or volume cell-bin grid) for photon-guided direction sampling
- Performs **NEE** toward a light source (with Beer-Lambert transmittance along the shadow ray)
- Samples a scatter direction from a mixture of Henyey-Greenstein phase function and volume photon guide
- Applies spectral MIS transmittance weights (hero wavelength scheme)

This is standard **volumetric path tracing** (PBRT-v4 §14, Novák et al. 2018).  No beam tracing or beam estimators are needed.

The photon map's role at delta surfaces themselves is zero — but the *next* non-delta hit benefits from caustic photons deposited by Part 1's photon tracing.  Inside translucent media, **volume photons** provide the guide signal.

---

## 5. GPU Kernel Design

### 5.1 `full_path_trace_v3()` — Device Function

Replaces `full_path_trace()` in `optix_path_trace.cuh`.

**Register pressure management:**  The current kernel uses ~80 registers.  The v3 kernel adds:
- `CellAnalysis` read (1 hash lookup, ~10 regs)
- `pdf_bsdf_prev` carry-forward (1 float)
- Removal of the inner glossy loop (saves ~20 regs)

Net register delta: approximately neutral.

**Memory access pattern:**  The critical new access is the cell-bin histogram read at each non-delta bounce.  This is a contiguous 32-float array per cell — 128 bytes.  On SM 8.0+, this fits in L1 cache.  For 5M photons across 64K cells at 32 bins, the total grid is 8 MB — fits comfortably in L2.

### 5.2 Launch Configuration

Unchanged from current: 1 thread per pixel, 1 wave per SPP group.  The inner bounce loop is entirely within-thread — no inter-thread communication.

### 5.3 New LaunchParams Fields

```cpp
// ── v3 Path Tracer ──────────────────────────────────────────────────
int    max_bounces_camera;       // max bounce depth
int    min_bounces_rr;           // guaranteed bounces before RR
float  rr_threshold;             // max survival probability
float  guide_fraction;           // base guided/BSDF mix ratio
int    guide_fallback_bounce;    // switch to photon gather after this
int    photon_final_gather;      // 1 = use photon map at terminal bounce

// ── Photon Analysis (per-cell, precomputed) ─────────────────────────
float* cell_caustic_fraction;    // [grid_total_cells]
float* cell_flux_density;        // [grid_total_cells]
float* cell_guide_fraction;      // [grid_total_cells]  precomputed p_guide
```

### 5.4 Removed LaunchParams Fields

```
nee_light_samples, nee_deep_samples,
debug_shadow_rays, is_final_render,
lobe_balance,
light_cache_entries, light_cache_count, light_cache_total_importance,
light_cache_cell_size, light_cache_valid, use_light_cache,
shadow_ray_targets, shadow_ray_cell_offset, shadow_ray_cell_count,
shadow_ray_cache_valid, coverage_max_value
```

---

## 6. CPU Reference Path Tracer

The CPU path tracer in `renderer.cpp` must be updated to match the iterative bounce algorithm.  The current `render_pixel()` does first-hit + NEE + gather.  Replace with:

```cpp
// path_trace.h — CPU photon-guided path tracer
Spectrum path_trace_cpu(
    Ray ray, PCGRng& rng,
    const Scene& scene,
    const RenderConfig& cfg,
    const PhotonSoA& global_photons,
    const PhotonSoA& caustic_photons,
    const HashGrid& global_grid,
    const HashGrid& caustic_grid,
    const CellBinGrid& cell_bin_grid);
```

The CPU version shares:
- `bsdf.h` (BSDF evaluation, sampling, PDF)
- `nee_shared.h` (NEE math — PDF conversion, geometry checks)
- `spectrum.h` (spectral operations)
- `photon_analysis.h` (cell analysis)
- `density_estimator.h` (photon gather for final gather + caustic estimate)

The CPU path tracer is used for:
1. Unit test validation (ground truth comparison)
2. CPU↔GPU consistency tests
3. Reference renders for debugging

---

## 7. Adaptive Sample Allocation

### 7.1 Per-Pixel SPP Budget

The photon analysis from §3 feeds an adaptive SPP allocator that distributes the total SPP budget across pixels non-uniformly:

$$\text{spp}(x, y) = \text{spp}_\text{base} \cdot w(x, y), \qquad w = \frac{\text{cost}(x, y)}{\overline{\text{cost}}}$$

where `cost(x, y)` combines:

| Factor | Weight | High → | Low → |
|---|---|---|---|
| Flux CV (§3.3) | 0.4 | More samples (noisy region) | Fewer samples |
| Caustic fraction (§3.4) | 0.3 | More samples (sharp caustic) | Fewer samples |
| Luminance variance (from previous pass) | 0.2 | More samples (high noise) | Fewer samples |
| Saturated flux (§3.5) | 0.1 | Fewer samples (diminishing returns) | Normal samples |

### 7.2 Implementation

1. **Pilot pass** (4 SPP, uniform): accumulates luminance moments `lum_sum`, `lum_sum2`.
2. **Analysis pass**: reads photon cell data + pilot variance → computes per-pixel cost map.
3. **Full pass** (remaining SPP): each pixel draws `spp(x,y)` samples.

The existing `adaptive_sampling.cu` infrastructure (active mask, lum_sum, lum_sum2) is reused and extended.

### 7.3 GPU Implementation

```cuda
// Per-pixel adaptive iteration count
__forceinline__ __device__
int compute_pixel_spp(int px, int py, int base_spp) {
    int cell = dev_cell_grid_index(hit_position(px, py));
    float w = params.cell_guide_fraction[cell];  // proxy for complexity
    // Scale by pilot-pass variance
    int idx = py * params.width + px;
    float var = pilot_variance(idx);
    float cost = 0.4f * var + 0.3f * params.cell_caustic_fraction[cell]
               + 0.2f * w + 0.1f * (1.f - saturated_factor(idx));
    return clamp((int)(base_spp * cost / avg_cost), 1, base_spp * 4);
}
```

---

## 8. View-Adaptive Photon Budgeting

> Picks up Part 1 open items VA-01 through VA-04.

### 8.1 Feedback Loop

After the pilot camera pass, we know which photon cells were actually visible and useful.  Use this to reallocate the photon budget:

```
Iteration i:
  1. Trace N photons with current emitter CDF
  2. Build photon map + hash grid + cell-bin grid
  3. Run pilot camera pass (4 SPP)
  4. For each visible pixel, record which cells were queried
  5. For each queried cell, record source_emissive_idx of its photons
  6. Count per-emitter visibility:
       emitter_usefulness[k] = Σ (flux from emitter k in visible cells)
  7. Update emitter CDF:
       p_emitter[k] ∝ emitter_usefulness[k] + ε (floor to prevent starvation)
  8. Re-trace with updated CDF
```

### 8.2 Budget Distribution

The re-weighted CDF is a mixture:

$$p_\text{emitter}(k) = (1 - \beta) \cdot p_\text{power}(k) + \beta \cdot p_\text{view}(k)$$

where $\beta$ starts at 0.0 (first iteration, no feedback) and increases to 0.7 over iterations.  $p_\text{view}(k)$ is proportional to `emitter_usefulness[k]`.  The power-weighted floor $(1 - \beta)$ ensures that emitters contributing to off-screen indirect light are not starved.

### 8.3 When to Re-trace

Re-tracing the photon map is expensive ($O(N_\text{photons})$).  Two strategies:

1. **Single re-trace:** Trace → pilot → reweight → re-trace → full render.  One extra photon pass.
2. **Progressive:** Re-trace every `N` SPP groups.  Smoothly converges the CDF.  Higher quality, more cost.

Default: single re-trace (strategy 1) for balanced mode.

---

## 9. Volumetric Path Tracing in Translucent Media

> Picks up Part 1 open items MT-04 through MT-09.
> Replaces beam tracing with the standard **volumetric photon mapping** approach (Jensen 2001, PBRT-v4 §14, Novák et al. 2018).

Translucent objects (marble, jade, wax, milk) have a participating medium inside a dielectric shell.  Both photons and camera rays enter the medium, scatter, and are absorbed — using the same physics.

### 9.1 Approach: Volumetric Photon Mapping (not beam tracing)

The state-of-the-art approach for homogeneous participating media in photon-guided path tracers is:

1. **Photon side (Pass 1):** When a photon scatters inside a medium, **deposit a volume photon** at the scatter point.  This is a regular photon with a 3D position, direction, flux, and a `VOLUME_SCATTER` flag.  Volume photons are stored in a dedicated `volume_photons` SoA and indexed by a 3D hash grid (`volume_grid`).  No beams, no segments — just point deposits at scatter events.

2. **Camera side (Pass 2):** When a camera ray is inside a medium, perform free-flight sampling.  At each scatter event, query the volume photon map for:
   - **Directional guide:** volume cell-bin grid histogram → photon-guided scatter direction
   - **Density estimate:** kNN volume photon gather → in-medium radiance estimate (for final gather at path termination inside media)
   - **NEE:** shadow ray toward a light with Beer-Lambert transmittance

This is simpler and more robust than beam tracing.  Beam tracing requires storing line segments, computing line-point distances, and a custom beam estimator — all of which add complexity with no advantage for homogeneous media.  Volumetric photon mapping uses the same infrastructure as surface photon mapping (SoA, hash grid, cell-bin grid), just in 3D.

### 9.2 In-Medium Camera Path

When a camera path enters a participating medium (through a Translucent surface), each iteration of the bounce loop starts with a free-flight decision:

1. **Free-flight sampling:** Sample a distance $t = -\ln(\xi) / \sigma_t^{(h)}$ using the hero wavelength's extinction coefficient.
2. **If $t < t_\text{surface}$ → medium scatter event:**
   - Move to scatter point: $\mathbf{x}_s = \mathbf{x} + t \cdot \omega$
   - Apply spectral MIS scatter weight (see §9.4)
   - Perform NEE toward a light (§9.6)
   - Query volume cell-bin grid for photon-guided scatter direction
   - Sample new direction from HG + volume guide mixture
   - Increment bounce counter (scatter events count as bounces for RR)
   - Continue bounce loop from scatter point (still inside medium)
3. **If $t \geq t_\text{surface}$ → ray reaches next surface:**
   - Apply spectral MIS transmittance weight (see §9.4)
   - Process surface hit normally (may be exiting Translucent → pop medium stack)

### 9.3 Phase Function Sampling

At a scatter event inside a medium, the phase function determines the scatter direction:

$$p_\text{HG}(\cos\theta, g) = \frac{1 - g^2}{4\pi(1 + g^2 - 2g\cos\theta)^{3/2}}$$

**Sampling:**
$$\cos\theta = \frac{1}{2g}\left(1 + g^2 - \left(\frac{1 - g^2}{1 - g + 2g\xi}\right)^2\right), \quad g \neq 0$$

For $g = 0$ (isotropic): $\cos\theta = 1 - 2\xi$.

**Photon-guided mixture:** At each scatter event, the direction is sampled from a mixture of HG and the volume photon guide:

$$p_\text{scatter}(\omega_i) = \alpha_v \cdot p_\text{vol\_guide}(\omega_i) + (1 - \alpha_v) \cdot p_\text{HG}(\omega_i)$$

where $\alpha_v$ is the volume guide fraction (from volume cell analysis, analogous to the surface guide fraction).  If no volume photons exist in the cell, $\alpha_v = 0$ and pure HG is used.

### 9.4 Spectral MIS Transmittance (Hero Wavelength)

The hero wavelength scheme from Part 1 (§7.7) applies identically to camera paths.  At a scatter event with free-flight distance $t$:

$$w_\text{scatter}(\lambda) = \frac{\sigma_s(\lambda) \cdot e^{-\sigma_t(\lambda) \cdot t}}{\frac{1}{N}\sum_j \sigma_t^{(j)} \cdot e^{-\sigma_t^{(j)} \cdot t}}$$

At a surface hit with segment distance $d$:

$$w_\text{transmit}(\lambda) = \frac{e^{-\sigma_t(\lambda) \cdot d}}{\frac{1}{N}\sum_j e^{-\sigma_t^{(j)} \cdot d}}$$

### 9.5 Russian Roulette Inside Media

Scatter events count as bounces for the RR budget.  The survival probability uses the single-scatter albedo:

$$p_\text{survive} = \min\left(\texttt{RR\_THRESHOLD},\; \max_\lambda\left(\frac{\sigma_s(\lambda)}{\sigma_t(\lambda)}\right)\right)$$

Highly absorptive media ($\sigma_a \gg \sigma_s$) naturally terminate paths quickly via low survival probability.

### 9.6 NEE at Medium Scatter Events

At each medium scatter event, sample a direction toward a light source.  The shadow ray must account for Beer-Lambert attenuation:

$$T_\text{shadow}(\lambda) = \exp\left(-\sigma_t(\lambda) \cdot d_\text{shadow}\right)$$

The NEE contribution at a scatter point is:

$$L_\text{nee} = \frac{p_\text{HG}(\cos\theta) \cdot L_e \cdot T_\text{shadow}}{p_\text{nee}}$$

MIS-weighted against the phase function PDF (same power heuristic as surface NEE vs BSDF).

### 9.7 Volume Photon Guide

The volume cell-bin grid (`vol_cell_bin_grid`) is a 3D hash grid of directional histograms, built from volume photon deposits.  It is the volumetric analogue of the surface `CellBinGrid`:

| Surface guide | Volume guide |
|---|---|
| Indexed by surface cell (2.5D) | Indexed by 3D cell |
| Built from surface photon `wi` | Built from volume photon `wi` at scatter points |
| Used at non-delta surface bounces | Used at medium scatter events |
| Histogram: `bin_flux[32]` | Histogram: `bin_flux[32]` (same Fibonacci sphere binning) |

### 9.8 Part 1 Update: Volume Photon Deposits

> **This section requires changes to the photon tracing pass (Part 1).**

The current Part 1 photon tracer records **beam segments** when a photon traverses a medium.  This must be changed to **point deposits at scatter events** — simpler, and consistent with the volumetric photon mapping approach.

**Change in photon bounce loop:**

When a photon is inside a participating medium and scatters (free-flight distance $t < d_\text{surface}$):
1. Deposit a **volume photon** at the scatter point: position $\mathbf{x}_s$, direction $\omega_i$ (incoming), spectral flux, `VOLUME_SCATTER` flag.
2. Store in `volume_photons` SoA (new, parallel to `global_photons` and `caustic_photons`).
3. Continue bouncing from the scatter point with a new direction sampled from HG.

**New data structures:**

| Structure | Purpose |
|---|---|
| `volume_photons` (PhotonSoA) | Volume photon deposits at medium scatter events |
| `volume_grid` (HashGrid) | 3D spatial index for volume photons |
| `vol_cell_bin_grid` (CellBinGrid) | Directional histogram per 3D cell (for volume guide) |

**What to remove from Part 1:**

| Item | Action |
|---|---|
| Beam segment recording | Remove — no beam estimator needed |
| `photon_beam.h` (placeholder) | Delete — volume photon point deposits replace beams |
| `PHOTON_FLAG_VOLUME_SEGMENT` | Replace with `PHOTON_FLAG_VOLUME_SCATTER` |
| Beam estimator infrastructure (MT-07 from Part 1) | Not needed — kNN gather on volume photons instead |

### 9.9 Double-Attenuation Guard

When a photon or camera ray is inside an object-attached medium (tracked by `MediumStack`), any legacy atmospheric volume system must be skipped to prevent double-attenuation.  The guard checks: `in_object_medium = medium_stack.current_medium_id() >= 0`.

---

## 10. Debugging & Visualization

### 10.1 Render Mode Selector (TAB)

Update the render mode cycle for v3:

| Mode | What it shows |
|---|---|
| Combined | Full path-traced result |
| Direct Only | NEE contribution only (1-bounce) |
| Indirect Only | All bounces after first (BSDF + guide) |
| Photon Map | Raw photon density estimate (no path tracing) |
| Guide Map | Visualise the cell-bin histogram as a false-colour directional map |
| Normals | Shading normals |
| Depth | Depth buffer |
| Caustic Only | Caustic photon density estimate |

### 10.2 Photon Overlay (F1/F2)

Keep the F1/F2 photon visualisation from Part 1 as-is.

### 10.3 Per-Bounce AOVs

New debug output: write per-bounce radiance contributions to separate buffers.  This helps diagnose where energy is gained or lost along the path.

```
bounce_0_buffer[pixel] = throughput_0 * (Le + L_nee_0 + L_caustic_0)
bounce_1_buffer[pixel] = throughput_1 * (Le + L_nee_1 + L_caustic_1)
...
```

### 10.4 Guide Visualisation (new F-key)

Assign F3 (currently reserved) to **visualise the guided direction PDF**:
- At each pixel, trace to the first non-delta hit
- Read the cell-bin histogram
- Render the histogram as a colour-coded hemisphere (hot = high probability, cold = low)

---

## 11. Implementation Checklist

### 11.1 Cleanup (§1)

| ID | Task | Status |
|---|---|---|
| CL-01 | Remove `DEFAULT_NEE_LIGHT_SAMPLES`, `DEFAULT_NEE_DEEP_SAMPLES` from `config.h` | [x] |
| CL-02 | Remove `DEFAULT_NEE_COVERAGE_FRACTION` from `config.h` | [x] |
| CL-03 | Remove `DEFAULT_MAX_GLOSSY_BOUNCES` from `config.h` | [x] |
| CL-04 | Remove `DEFAULT_ENABLE_BRESENHAM_BSDF` from `config.h` | [x] |
| CL-05 | Remove `nee_shadow_sample_count()` from `nee_shared.h` | [x] |
| CL-06 | Remove `dev_nee_golden_stratified()` from `optix_nee_dispatch.cuh` | [x] |
| CL-07 | Collapse `optix_nee_dispatch.cuh` into `optix_nee.cuh` | [x] |
| CL-08 | Remove `emissive_area_alias_table` from `scene.h` | [x] |
| CL-09 | Remove `light_cache.h` and all `light_cache_*` from `LaunchParams` | [x] |
| CL-10 | Remove `lobe_balance` from `LaunchParams` | [x] |
| CL-11 | Remove `is_final_render`, `debug_shadow_rays` from `LaunchParams` | [x] |
| CL-12 | Simplify `PixelLighting` to 3 channels (combined, direct, indirect) | [x] |
| CL-13 | Add new config.h parameters (§1.2) | [x] |
| CL-14 | Rename `dev_bsdf_pdf` → `bsdf_pdf`, `dev_bsdf_sample` → `bsdf_sample`, `DevBSDFSample` → `BSDFSample` | [x] |
| CL-15 | Remove `dev_generate_camera_ray` wrapper | [x] |
| CL-16 | Slim `main.cpp` to ≤ 200 lines | [ ] |
| CL-17 | Slim `optix_renderer.cpp` to ≤ 1200 lines | [ ] |

### 11.2 Photon Analysis (§3)

| ID | Task | Status |
|---|---|---|
| PA-01 | Create `photon_analysis.h` with `CellAnalysis` struct | [x] |
| PA-02 | Implement directional analysis (active bin count, guide fraction) | [x] |
| PA-03 | Implement spatial density metric (flux per cell area) | [x] |
| PA-04 | Implement spatial variance (cross-cell CV) | [x] |
| PA-05 | Implement caustic fraction per cell | [x] |
| PA-06 | Implement photon energy / saturation metric | [x] |
| PA-07 | Build `CellAnalysis` array after photon map build | [x] |
| PA-08 | Upload `cell_caustic_fraction`, `cell_flux_density`, `cell_guide_fraction` to GPU | [x] |

### 11.3 Path Tracer Core (§4)

| ID | Task | Status |
|---|---|---|
| PT-01 | Create `optix_path_trace_v3.cuh` with iterative `full_path_trace_v3()` | [x] |
| PT-02 | Single bounce loop: delta surfaces, NEE, guided/BSDF direction | [x] |
| PT-03 | Emission MIS (carry forward `pdf_bsdf_prev`) | [x] |
| PT-04 | Photon caustic additive contribution at non-delta hits | [x] |
| PT-05 | Photon final gather at terminal bounce | [x] |
| PT-06 | Adaptive guide fraction from photon analysis (§3.1) | [x] |
| PT-07 | Russian roulette with spectral throughput | [x] |
| PT-08 | IOR stack integration for nested dielectrics | [x] |
| PT-09 | Remove old `full_path_trace()` and `debug_first_hit()` | [x] |
| PT-10 | Update `__raygen__render` to call `full_path_trace_v3()` | [x] |

### 11.4 CPU Reference (§6)

| ID | Task | Status |
|---|---|---|
| CP-01 | Create `path_trace.h` with `path_trace_cpu()` | [x] |
| CP-02 | Mirror GPU algorithm exactly (same bounce loop, same MIS, same RR) | [x] |
| CP-03 | Connect to `render_frame()` in `renderer.cpp` | [x] |
| CP-04 | CPU↔GPU consistency test | [x] |

### 11.5 Adaptive Sampling (§7)

| ID | Task | Status |
|---|---|---|
| AS-01 | Pilot pass (4 SPP uniform) with luminance moment accumulation | [ ] |
| AS-02 | Per-pixel cost map from photon analysis + pilot variance | [ ] |
| AS-03 | Variable SPP per pixel (clamp to [1, 4×base]) | [x] |
| AS-04 | GPU implementation of `compute_pixel_spp()` | [x] |
| AS-05 | Progress-aware SPP tracking (partial pixels) | [ ] |

### 11.6 View-Adaptive Budgeting (§8)

| ID | Task | Status |
|---|---|---|
| VA-01 | Per-emitter visibility counting from pilot pass | [ ] |
| VA-02 | `emitter_usefulness[]` aggregation from visible cells | [ ] |
| VA-03 | Mixture CDF: $(1-\beta) \cdot p_\text{power} + \beta \cdot p_\text{view}$ | [ ] |
| VA-04 | Re-trace photon map with updated CDF | [ ] |
| VA-05 | Progressive re-trace option | [ ] |

### 11.7 Medium Transport — Camera Side (§9)

| ID | Task | Status |
|---|---|---|
| MT-01 | GPU MediumStack (parallel to IORStack) | [ ] |
| MT-02 | Free-flight sampling with hero-wavelength spectral MIS | [ ] |
| MT-03 | HG phase function scatter direction sampling | [ ] |
| MT-04 | HG + volume photon guide mixture sampling | [ ] |
| MT-05 | Russian roulette inside media scatter loop | [ ] |
| MT-06 | NEE at medium scatter events (with Beer-Lambert attenuation) | [ ] |
| MT-07 | Volume kNN density estimate at terminal scatter events | [ ] |
| MT-08 | Volume double-attenuation guard | [ ] |

### 11.8 Medium Transport — Photon Side (Part 1 update, §9.8)

| ID | Task | Status |
|---|---|---|
| VP-01 | Volume photon deposit at scatter events (new `volume_photons` SoA) | [x] |
| VP-02 | 3D `volume_grid` hash grid for volume photons | [ ] |
| VP-03 | `vol_cell_bin_grid` directional histograms from volume photons | [ ] |
| VP-04 | Remove beam segment recording from photon bounce loop | [ ] |
| VP-05 | Delete `photon_beam.h` placeholder | [ ] |
| VP-06 | Replace `PHOTON_FLAG_VOLUME_SEGMENT` → `PHOTON_FLAG_VOLUME_SCATTER` | [ ] |
| VP-07 | Upload `volume_grid`, `vol_cell_bin_grid` to GPU | [ ] |

### 11.9 Debug & Visualisation (§10)

| ID | Task | Status |
|---|---|---|
| DB-01 | Update TAB render modes for v3 | [ ] |
| DB-02 | Add "Guide Map" render mode | [ ] |
| DB-03 | Add "Caustic Only" render mode | [ ] |
| DB-04 | Per-bounce AOV buffers | [ ] |
| DB-05 | F3 guide visualisation (hemisphere heatmap) | [ ] |
| DB-06 | Console printout of photon analysis statistics | [x] |

---

## 12. Implementation Order

The work is sequenced to maintain a working renderer at every step, with the test suite passing.

### Phase 1 — Cleanup and Foundation ✅

1. **CL-01 through CL-15:** ✅ Remove NEE multi-sample infrastructure, add new config parameters, naming/size cleanup.
2. **CL-16, CL-17:** Remaining carry-forward cleanup (not yet started).
3. **PA-01 through PA-08:** ✅ Photon analysis build + upload.
4. **Build + test** — ✅ 337 tests pass, renderer works with `full_path_trace_v3()`.

### Phase 2 — Core Path Tracer ✅

5. **PT-01 through PT-03:** ✅ Created `full_path_trace_v3()` with single bounce loop, delta surface handling, emission MIS.
6. **PT-04, PT-05:** ✅ Caustic contribution and photon final gather.
7. **PT-06, PT-07, PT-08:** ✅ Adaptive guide fraction, RR, IOR stack.
8. **PT-09, PT-10:** ✅ Removed old path, updated raygen.
9. **CP-01 through CP-04:** ✅ CPU reference + consistency test.
10. **Build + test** — ✅ Renderer produces correct images.

### Phase 3 — Adaptive Sampling (in progress)

11. **AS-03, AS-04:** ✅ Variable SPP with active mask + lum_sum2 variance.
12. **AS-01, AS-02, AS-05:** Not yet started (pilot pass, cost map, progress-aware tracking).
13. **VA-01 through VA-05:** Not yet started (view-adaptive photon budgeting).

### Phase 4 — Medium Transport (not started)

14. **VP-01 through VP-07:** Partially started — VP-01 (volume photon SoA) done.
15. **MT-01 through MT-08:** Not yet started (camera-side volumetric path tracing).
16. **Build + test** — participating media not yet testable.

### Phase 5 — Polish (in progress)

17. **DB-06:** ✅ Console printout of photon analysis statistics.
18. **DB-01 through DB-05:** Not yet started (render modes, AOV buffers, guide vis).
19. Validation renders: Cornell Box, Cornell Sphere, Salle de Bain etc.
20. Performance profiling and register pressure tuning.

---

## 13. Material Handling Reference

No changes to the material interaction code from Part 1 (§7).  The path tracer calls the same BSDF functions:

| Material | Path tracer action | NEE? | Guide? |
|---|---|---|---|
| Lambertian | Evaluate + sample | Yes | Yes |
| Mirror | Specular bounce, continue | No | No |
| Glass | Fresnel R/T, continue | No | No |
| GlossyMetal | GGX VNDF + diffuse mix | Yes | Yes |
| Emissive | Add Le (MIS-weighted) | — | — |
| GlossyDielectric | GGX VNDF + diffuse mix | Yes | Yes |
| Translucent | Fresnel R/T + medium push/pop + volumetric scatter loop | No (surface), Yes (scatter) | No (surface), Yes (scatter) |
| Clearcoat | Coat GGX + base diffuse | Yes | Yes |
| Fabric | Cosine + sheen | Yes | Yes |

---

## 14. Expected Quality Improvements

| Issue | Current | After v3 |
|---|---|---|
| Multi-bounce colour bleeding | Missing (indirect is single-gather) | Full multi-bounce transport |
| Glossy-to-glossy reflections | Limited to 2 continuation bounces | Unlimited (up to max_bounces) |
| Caustics in reflections | Not captured | Mirror/glass chain → caustic photon at hit |
| Dark corners | Photon gather has boundary bias | Path tracing + final gather reduces bias |
| Convergence rate | Slow (high variance from photon kNN) | Guided sampling reduces variance |
| Translucent objects | Beer-Lambert only, no interior scattering visible | Full volumetric path tracing with photon-guided scatter |
| Fireflies | Multi-sample NEE with high M | 1-sample NEE + MIS eliminates double-counting |
