# Post-Processing — Pipeline Stage Details

## Firefly Filter

Median-based outlier detection using Median Absolute Deviation (MAD):
```
1. For each pixel, gather neighbors in window (3×3 or 5×5)
2. Compute median luminance of window
3. Compute MAD = median(|lum_i - median|)
4. If pixel luminance > median + K × MAD × 1.4826: replace with median
5. The 1.4826 constant normalizes MAD to match standard deviation for Gaussian
```

### Parameters
- `radius = 1` → 3×3 window (9 samples), `radius = 2` → 5×5 (25 samples)
- `threshold (K)` = 3-5 typical. Lower = more aggressive filtering.

### GPU Implementation
- Two-pass: original → temp buffer (compute median), then temp → original (replace)
- In-place result via double buffering with `d_firefly_temp_`

## Bloom Architecture

Multi-scale mip-chain approach (similar to Unreal Engine 4):

```
Full-res HDR → Bright Extract → Mip 0 (half-res)
                                   ↓ downsample
                                Mip 1 (quarter-res)
                                   ↓
                                Mip 2 (⅛)
                                   ↓
                                Mip 3 (1/16)
                                   ↓
                                Mip 4 (1/32)
                                   ↑ blur + upsample
                                ...accumulate back up...
                                   ↑
                                Mip 0 (post-bloom)
                                   ↓
                         Composite onto full-res
```

### Bright-Pass Extract
Adaptive ramp based on scene max luminance:
```
lo_threshold = max_lum × 0.7    (start of ramp)
hi_threshold = max_lum × 0.9    (full bloom)
extract = smoothstep(lo, hi, pixel_lum) × pixel_color
```

### Separable Gaussian Blur
Each mip gets horizontal then vertical Gaussian blur:
- Radius parameter controls kernel width
- Sigma derived from radius
- Applied per-channel (RGB)

## Tonemap Pipeline

### Stage 1: RGB → HDR
```cuda
float3 accum = color_buf[pixel_idx * 3 + {0,1,2}];
float  count = sample_cnt ? sample_cnt[pixel_idx] : 1.0f;
float3 hdr   = accum / count * exposure;
d_hdr[pixel_idx] = float4(hdr.r, hdr.g, hdr.b, 1.0f);
```

### Stage 2: HDR → sRGB (ACES)
```cuda
// ACES Filmic Tonemap (Krzysztof Narkowicz approximation)
float3 aces_tonemap(float3 x) {
    float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// sRGB gamma
float linear_to_srgb(float c) {
    return (c <= 0.0031308f)
        ? c * 12.92f
        : 1.055f * pow(c, 1.0f/2.4f) - 0.055f;
}
```

## Denoiser Integration Point

The `apply_pre_denoise()` method runs only the firefly filter stage on an HDR buffer, producing a cleaned HDR suitable for an external denoiser (e.g., OptiX AI denoiser). The denoiser operates between firefly filter and bloom/tonemap.

```
Pipeline with denoiser:
  rgb_to_hdr → firefly_filter → [DENOISER] → bloom → tonemap
```
