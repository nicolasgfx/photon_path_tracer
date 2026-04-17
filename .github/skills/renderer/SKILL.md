---
name: renderer
description: 'GPU render pipeline internals. Use when: modifying renderer source code, fixing material/intersection/energy bugs, or debugging noise/fireflies in implementation. NOT for simply running the executable (use orchestrator skill).'
---

# Renderer

Photon-guided RGB path tracer: OptiX 9 acceleration, Müller SD-tree guidance, progressive 1 SPP/frame accumulation, one-sample bilateral MIS. Single-pass architecture:

- **Camera path tracing:** iterative bounce loop with bilateral MIS (SD-tree guide + BSDF); NEE with power-heuristic MIS; Russian roulette; three-level clamping
- **SD-tree training:** geometric doubling from camera-path samples only (Li · cos θ weighting, Müller 2017). No photon system — camera rays are the sole training signal.

All 9 material types implemented. RGB 3-channel float3. Post-FX: firefly filter → bloom → ACES tonemap → OptiX denoiser.

**Not implemented:** participating media, chromatic dispersion, stochastic opacity, photon mapping (photon stubs exist but are not wired in).

## Source Map

| Subsystem | Path | Key Files |
|-----------|------|-----------|
| Acceleration | `src/renderer/accel/` | `accel_builder.cpp/.h`, `optix_programs.cu`, `launch_params.h` |
| Camera | `src/renderer/camera/` | `camera.h` |
| Materials | `src/renderer/material/` | `bsdf.h`, `bsdf_shared.h`, `specular.h` |
| Lighting | `src/renderer/lighting/` | `optix_nee.cuh`, `nee_shared.h`, `lighting_upload.h/.cpp` |
| Integrator | `src/renderer/integrator/` | `path_tracer.cuh`, `path_state.h`, `russian_roulette.h`, `sample_clamping.h` |
| Photon | `src/renderer/photon/` | `photon.h`, `photon_storage.h`, `photon_tracer.cuh` |
| Guide | `src/renderer/guide/` | `sd_tree.h/.cpp`, `dquad_gpu.h`, `sd_tree_gpu.h`, `sd_tree_device.cuh` |
| PostFX | `src/renderer/postfx/` | `postfx_pipeline.h/.cpp`, `bloom.h/.cu`, `tonemap.h/.cu`, `firefly_filter.h/.cu` |
| App | `src/renderer/app/` | `viewer.h/.cpp`, `render_session.h/.cpp`, `render_config.h`, `cli_args.h` |
| Core | `src/core/` | `config.h`, `types.h`, `material_flags.h`, `scene_profile.h` |

## Key Invariants

These are mathematically mandated. Violating any one causes double-counting, energy loss, or infinite density:

1. **Every MC estimator divides by its exact sampling PDF** — no approximations (Veach 1997)
2. **NEE + emission MIS mutual exclusivity** — emission on BSDF-hit is MIS-weighted against NEE; first bounce and post-specular take full weight (PBRT v4 §13.4)
3. **Delta BSDFs: no NEE, pdf_prev = 0** — delta distributions have zero measure; NEE at delta surfaces = infinite variance (Veach 1997 §9.2)
4. **Photon deposit only when bounce > 0** — prevents double-counting direct lighting already computed by NEE (Jensen 1996 decomposition)
5. **Never deposit at delta surfaces** — would produce infinite density; photons pass through and deposit at next non-delta surface (this creates caustics: L→S+→D)
6. **Adjoint η² correction at refractive interfaces** — photon throughput must include η² factor; camera paths must not (Veach 1997 §5.2)
7. **Geometric doubling discards training samples** — final image uses only frozen-phase samples to avoid bias from evolving guide (Müller 2017 §5)
8. **Bilateral MIS combined PDF uses both terms** — even when one strategy generated the sample: `p_combined = α·p_guide + (1−α)·p_bsdf` (Müller 2017 §4)
9. **Diffuse-only gather** — density estimator uses `evaluate_diffuse()`, NOT full `evaluate()`. GGX specular lobes create firefly hotspots in density estimates. Specular transport is handled by the path tracer's BSDF continuation.

## Materials (9 types)

| ID | Type | Delta? | Sampling | Key detail |
|----|------|--------|----------|------------|
| 0 | Lambertian | No | Cosine hemisphere | `f = Kd/π`, `pdf = cosθ/π` |
| 1 | Mirror | **Yes** | Perfect reflection | `f = Ks/|cosθ|` (built-in δ/pdf) |
| 2 | Glass | **Yes** | Fresnel stochastic reflect/refract | Scalar IOR, Color3 Tf filter, TIR fallback |
| 3 | GlossyMetal | No | Coin flip: GGX VNDF vs cosine | Cook-Torrance, Ks = Fresnel F0 |
| 4 | Emissive | — | — | Emits Le; path terminates |
| 5 | GlossyDielectric | No | Coin flip: GGX vs cosine | F0 from IOR: `((ior−1)/(ior+1))²` |
| 6 | Translucent | **Yes** | Same as Glass + medium stack | IOR stack push/pop for nested dielectrics |
| 7 | Clearcoat | No | Coin flip: coat GGX vs base | `f_base *= (1 − coat_weight·Fr_coat)` |
| 8 | Fabric | No | Cosine + sheen | `sheen_w·sheen_col·(1−VoH)⁵` |

**Pitfalls:**
- Lobe probability clamped to [0.05, 0.95] on GlossyMetal/GlossyDielectric/Clearcoat — prevents one lobe from starving
- Glass `refract_local()`: refracted z must use `copysignf(cos_t, -wo.z)` to always cross the surface
- Roughness floor: `alpha = max(roughness², BSDF_MIN_ALPHA)` prevents mirror singularity in GGX

## Trace Path Loop

Condensed from `path_tracer.cuh` — the core bounce loop:

```
for bounce in [0, max_bounces):
    hit = trace_radiance(origin, direction)
    if NO HIT: break

    if EMISSIVE:
        if bounce == 0: L += throughput * Le                    // camera ray direct
        else if pdf_prev > 0: L += throughput * Le * mis(pdf_prev, light_pdf)  // MIS
        else: L += throughput * Le                              // post-delta: full weight
        break

    if DELTA: sample specular; throughput *= f*cosθ; pdf_prev = 0; continue

    // NON-DELTA: NEE + BSDF sample
    L += clamp(throughput * nee_dispatch(...))                  // direct lighting
    sample direction (bilateral MIS when guide active: α·p_guide + (1−α)·p_bsdf)
    throughput *= f*cosθ / combined_pdf                         // per-bounce clamp applied
    throughput = clamp_path_throughput(throughput)               // per-path clamp

    // Russian roulette (bounce >= min_bounces_rr)
    p_survive = min(rr_threshold, max_component(throughput))
    if random >= p_survive: break
    throughput /= p_survive                                     // unbiased compensation

L = clamp_sample_luminance(L)                                   // per-sample clamp
if !isfinite(L): L = 0
```

## Three-Level Clamping

| Level | Where | Config field |
|-------|-------|-------------|
| Per-bounce | `min(f·cos/pdf, limit)` each channel | `max_bounce_contribution` |
| Per-path | `if max(tp) > limit: tp *= limit/max` | `max_path_throughput` |
| Per-sample | `if lum > limit: L *= limit/lum` | `max_sample_luminance` |

See `config.h` for current defaults.

**Pitfall:** Over-clamping causes energy loss visible as darkening. Check `mean_lum` stability across SPP levels.

## Photon System

**Deposit rules** (see §Key Invariants #4, #5):
- Deposit at non-delta surfaces when `bounce > 0`
- Delta surfaces: continue bouncing, don't deposit

**Caustic lifecycle** (`on_caustic_path` flag):
1. Starts `false` at emission
2. Set `true` on delta material hit (Mirror/Glass/Translucent)
3. Reset `false` on diffuse/glossy deposit
4. While `true`: deposit enters **both** global and caustic maps

**Hash grid:** Teschner spatial hash, cell size = `2 × gather_radius`. k-NN via register-allocated max-heap with shell expansion (2 layers max). Tangential distance metric mandatory — prevents cross-surface bleed at edges.

**Density estimate:** Epanechnikov kernel: `w = 1 − d_tan²/r_k²`. When count < 5 within max radius → return zero.

**Emissive termination:** Photon bounce loop terminates on emissive hit (re-bounce wastes budget).

## SD-Tree Path Guide

**Training (geometric doubling, camera-only, Müller 2017 §5):** For iteration k = 0..NUM_TRAINING_ITERS: render 2^k SPP, record camera-path bounce samples (pos, dir, Li·cosθ weight) into GPU training buffer → download to CPU → deposit into SD-tree → refine spatial (median split, threshold ∝ √2^k) → refine directional → reset flux. Iter 0 uses pure BSDF (guide_fraction=0); iters k>0 use bilateral MIS (guide_fraction=0.5). All training samples discarded. Then freeze structure, render remaining SPP.

**Training weight:** Li · cos θ (incident radiance at each bounce, Müller 2017). Deposited immediately at each non-delta bounce — no backward bookkeeping.

**Periodic rebuilds (interactive):** When viewer enters idle mode, train_guide() is called before full-quality rendering starts. No periodic rebuilds during accumulation.

**Bilateral MIS:** `p_combined(ω) = α·p_guide(ω) + (1−α)·p_bsdf(ω)` with `α = guide_fraction` (default 0.5). One sample from the mixture via coin flip. One-sample is cheaper than two-sample and lets you trace 2× more paths. When guide sample falls below hemisphere, falls back to BSDF.

**DQuad:** Fixed-depth 5 directional quadtree (1024 bins, 1365 nodes/leaf). Cylindrical equal-area mapping: u = φ/(2π), v = (1 − cos θ)/2.

**GPU integration:** CPU builds tree → flattens to `sd_nodes[]` + `sd_dquad_nodes[]` (8B each) → cudaMemcpy. Device functions: `dev_sd_tree_sample(pos, rng, &pdf)`, `dev_sd_tree_pdf(pos, dir)`. Falls back to uniform sphere when `sd_tree_valid == 0`.

**CLI flags:** `--no-guide` (disable guide for A/B comparison), `--guide-fraction F` (override bilateral MIS weight), `--mode guide` (visualize guide PDF as heatmap).

## Render Loop Sync Points

```
render_frame():
    optixLaunch(1 SPP) → cudaDeviceSynchronize()       // SYNC 1: CPU blocks on GPU
    postfx:
        firefly → bloom_find_max → cudaSync()           // SYNC 2: bloom threshold
        cudaMemcpy D2H (1 float max_lum)                // SYNC 3
        bloom chain → cudaSync()                         // SYNC 4
        tonemap → cudaSync()                             // SYNC 5
    cudaMemcpy D2H (srgb + rgb buffers)                  // SYNC 6-7
    glfwSwapBuffers() (VSync = 60 Hz)
```

## PTX Rebuild Caveat

Ninja tracks only `optix_programs.cu` as source — header changes in `.cuh` files are NOT detected. Always touch the .cu file after modifying headers used by OptiX programs:

```
copy /b src\renderer\accel\optix_programs.cu +,, src\renderer\accel\optix_programs.cu
ninja photon_tracer
```

## References

| Paper | Key concepts for this codebase |
|-------|-------------------------------|
| Jensen 1996, 2001 | Photon mapping: deposit rules, density estimation, caustic maps, rendering equation decomposition |
| Müller et al. 2017 (EGSR Best Paper) | SD-tree, bilateral MIS, geometric doubling, camera training samples |
| Veach 1997 (PhD) | MIS balance/power heuristic, adjoint η² correction, delta BSDF handling |
| PBRT v4 (Pharr et al. 2023) | PathIntegrator patterns, RR η²-correction, path regularization, BVH light sampling |
| Walter et al. 2007 | GGX NDF, microfacet BTDF |
| Heitz 2014 | Smith G masking-shadowing, VNDF importance sampling |
| Silverman 1986 | Epanechnikov kernel optimality for density estimation |
