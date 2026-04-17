---
name: post-processing
description: 'GPU post-processing pipeline inside photon_tracer: firefly filter (median+MAD), bloom (mip-chain Gaussian), tonemap (ACES+sRGB gamma), OptiX denoiser. Use when: fixing firefly artifacts, tuning bloom thresholds, adjusting tonemapping, debugging output corruption, integrating OptiX denoiser, or optimizing post-FX performance.'
---

# Post-Processing

Part of the `photon_tracer` executable. Source in `src/renderer/postfx/`.

## Pipeline Position

```
Integrator (RGB accum) → PostFx Pipeline → sRGB uint8 output
                      ↗
  PostFxParams (per-scene configuration)

Chain: rgb_to_hdr → firefly_filter → bloom → tonemap_hdr
```

- **Stage**: 9 of 9
- **Upstream**: Integrator (float3 RGB accumulator + per-pixel sample counts), PostFxParams
- **Downstream**: Display (sRGB RGBA8), Denoiser (HDR float4)

## Key Interfaces

### Source Files
- `src/postfx/postfx_params.h` — `PostFxParams` configuration struct
- `src/postfx/firefly_filter.h` — `launch_firefly_filter()` kernel
- `src/postfx/bloom.h` — Multi-scale mip-chain bloom kernels
- `src/postfx/tonemap.h` — `launch_rgb_to_hdr()`, `launch_tonemap_hdr()` kernels
- `src/postfx/postfx_pipeline.h/.cpp` — `PostFxPipeline` orchestrator class

### Critical Types

```cpp
struct PostFxParams {
    // Bloom
    bool  bloom_enabled     = DEFAULT_BLOOM_ENABLED;
    float bloom_intensity   = DEFAULT_BLOOM_INTENSITY;
    float bloom_radius_h    = DEFAULT_BLOOM_RADIUS_H;
    float bloom_radius_v    = DEFAULT_BLOOM_RADIUS_V;
    float bloom_scene_min_Le, bloom_scene_max_Le;

    // Firefly filter
    bool  firefly_enabled   = DEFAULT_FIREFLY_FILTER_ENABLED;
    int   firefly_radius    = FIREFLY_FILTER_RADIUS;
    float firefly_threshold = FIREFLY_FILTER_THRESHOLD;

    // Tonemap
    float exposure          = DEFAULT_EXPOSURE;
    bool  use_aces          = USE_ACES_TONEMAPPING;
};

class PostFxPipeline {
    void init(int width, int height);
    void apply(const float* d_color_buf, const float* d_sample_cnt,
               uint8_t* d_srgb_out, float* d_hdr_out,
               int width, int height, const PostFxParams& params);
    void apply_pre_denoise(float* d_hdr, int w, int h, const PostFxParams&);
    float* hdr_buffer() const;
    void cleanup();
};
```

### Pipeline Stages (in order)

1. **rgb_to_hdr**: RGB accumulator → HDR float4 (divide by sample count, apply exposure)
2. **firefly_filter**: Median + MAD outlier suppression (3×3 or 5×5 window)
3. **bloom**: Multi-scale mip-chain blur
   - Find max luminance (parallel reduction)
   - Bright-pass extract (adaptive ramp threshold)
   - 5-level mip chain: downsample → blur H → blur V → upsample-accumulate
   - Final composite onto full-res HDR
4. **tonemap_hdr**: ACES tonemapping + sRGB gamma → uint8 RGBA

## Convergence Role

Post-processing is **reactive** — it compensates for remaining noise after rendering.

| Component | Convergence Impact |
|---|---|
| Firefly filter | Removes outlier pixels that clamping missed |
| Bloom | Purely artistic — no convergence impact |
| Denoiser | Trades detail for smoothness (most effective at low SPP) |
| Tonemap | Preserves HDR information into displayable range |

**Scene-aware post-processing**:

| Condition | Firefly threshold | Denoiser blend | Bloom |
|---|---|---|---|
| Low SPP (< 64) | aggressive (K=3) | 1.0 (full) | off |
| High SPP (> 256) | gentle (K=6) | 0.5 or off | optional |
| Bright envmap | K=2 | 1.0 | threshold=2.0 |

## Parameters

| Parameter | Location | Default | Role |
|---|---|---|---|
| `bloom_enabled` | config.h | varies | Enable/disable bloom |
| `bloom_intensity` | config.h | varies | Bloom mix strength |
| `bloom_radius_h/v` | config.h | varies | Gaussian blur radius |
| `firefly_enabled` | config.h | true | Enable outlier filter |
| `firefly_radius` | config.h | 1 | Filter window (1=3×3, 2=5×5) |
| `firefly_threshold` | config.h | ~4.0 | MAD multiplier K |
| `exposure` | config.h | 1.0 | Exposure adjustment (pre-tonemap) |
| `use_aces` | config.h | true | ACES vs simple Reinhard |

## Development Procedures

### Bloom Pipeline Detail
```
1. launch_bloom_find_max_luminance() → d_max_lum
2. Compute adaptive thresholds: lo = max_lum * ramp_lo, hi = max_lum * ramp_hi
3. launch_bloom_bright_extract(hdr, mip[0], lo, hi)
4. For each mip level 1..4:
   launch_bloom_downsample(mip[i-1], mip[i])
5. For each mip level 4..0:
   launch_bloom_blur_h(mip[i], tmp[i], radius)
   launch_bloom_blur_v(tmp[i], mip[i], radius)
   if i > 0: launch_bloom_upsample_accumulate(mip[i], mip[i-1])
6. launch_bloom_composite(hdr, mip[0], intensity)
```

### Adding a New Post-FX Stage
1. Add kernel declaration in new header (e.g., `vignette.h`)
2. Implement CUDA kernel in `.cu` file
3. Add parameter to `PostFxParams`
4. Wire into `PostFxPipeline::apply()` at the correct position
5. Add test case verifying identity pass (no-op when disabled)

### GPU Memory Management
- `PostFxPipeline` pre-allocates all scratch buffers in `init()`
- 5 mip levels + temp buffers for blur passes
- Total GPU memory: ~10× input buffer size (for all mip + scratch)
- `cleanup()` frees all device memory

## Debugging via Diagnostics

- Compare HDR buffer before/after firefly filter: count changed pixels
- Check bloom contribution: render with `bloom_intensity = 0` vs default
- Verify tonemap range: all output pixels should be [0, 255]
- `hdr_buffer()` gives post-bloom, pre-tonemap HDR for denoiser input

## Testing Procedures

### Unit Tests (postfx_test)
1. Known HDR → verified tonemap output (ACES formula)
2. Firefly filter: inject outlier → verify it's removed
3. Bloom: identity test (zero intensity → unchanged)
4. Pipeline: end-to-end with known input
5. PostFxParams defaults match config.h constants

### Validation
- Tonemap preserves relative luminance ordering
- Firefly filter doesn't affect clean images (below threshold)
- Bloom energy conservation: total luminance should increase only by bloom amount
