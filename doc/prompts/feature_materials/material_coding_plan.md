# Material System Coding Plan

## Comprehensive Analysis & Implementation Roadmap

**Date:** 2025-01-XX  
**Scope:** Full MTL property support for all 10 scene files  
**Constraint:** All changes must respect the spectral photon-centric architecture (revised_guideline_v2.md)

---

## Table of Contents

1. [Scene MTL Audit — What Our Scenes Actually Need](#1-scene-mtl-audit)
2. [Current Code Capabilities](#2-current-code-capabilities)
3. [Gap Analysis](#3-gap-analysis)
4. [Implementation Plan](#4-implementation-plan)
   - Phase 1: MTL Parser Expansion
   - Phase 2: Transmission Filter (Colored Glass)
   - Phase 3: Alpha Masking (map_d / d cutouts)
   - Phase 4: Emission Texture Maps (map_Ke)
   - Phase 5: Extended Texture Support (map_Ks, map_Ka, map_bump)
   - Phase 6: Rough Glass / Translucency
5. [Per-File Impact Assessment](#5-per-file-impact-assessment)
6. [Guideline Compliance Checklist](#6-guideline-compliance-checklist)
7. [Testing Strategy](#7-testing-strategy)
8. [Priority & Dependencies](#8-priority--dependencies)

---

## 1) Scene MTL Audit

### 1.1 Complete Property Inventory

Every MTL keyword used across all 9 `.mtl` files, with which scenes use it:

| MTL Keyword   | Scenes Using It | Count | Currently Parsed? |
|---------------|----------------|-------|-------------------|
| `newmtl`      | All 9           | ~180  | ✅ Yes             |
| `Kd`          | All 9           | ~180  | ✅ Yes             |
| `Ks`          | All 9           | ~180  | ✅ Yes             |
| `Ka`          | conference, mori_knob, fireplace_room, interior, sponza | ~150 | ❌ No |
| `Ke`          | cornell_box, living_room, sibenik, interior | ~20 | ✅ Yes |
| `Ns`          | All 9           | ~180  | ✅ Yes             |
| `Ni`          | conference, mori_knob, fireplace_room, interior, sponza | ~130 | ✅ Yes |
| `d`           | living_room (0.5), conference, mori_knob, sponza | ~130 | ❌ No |
| `Tr`          | sponza, interior (Glass: 0.8, 0.9) | ~20 | ❌ No |
| `Tf`          | mori_knob, fireplace_room, interior, sponza | ~10 | ❌ No |
| `illum`       | All 9           | ~180  | ✅ Yes (partial)   |
| `map_Kd`      | sibenik, living_room, fireplace_room, interior, sponza | ~80 | ✅ Yes |
| `map_Ka`      | sibenik, interior, sponza | ~50 | ❌ No |
| `map_Ks`      | interior, sponza | ~30 | ❌ No |
| `map_d`       | interior (plants, plates, cutlery) | 3 | ❌ No |
| `map_Ke`      | interior (lanterns, lamps, fan, wall light) | 5 | ❌ No |
| `map_bump`    | sibenik, sponza, interior | ~15 | ❌ No |
| `bump`        | sponza (alternate syntax) | ~10 | ❌ No |
| `sharpness`   | mori_knob (LTELogo: 100) | 1 | ❌ No |

### 1.2 Illumination Models Used

| illum | Meaning (MTL Spec) | Scenes Using It | Current Mapping |
|-------|-------------------|-----------------|-----------------|
| 0     | Color on, ambient off | — | Lambertian |
| 1     | Color on, ambient on | cornell_box, mori_knob | Lambertian |
| 2     | Highlight on (Blinn-Phong) | conference, living_room, salle_de_bain, sponza, interior, fireplace_room | GlossyDielectric (if Ks>0.01) |
| 3     | Reflection on, ray trace | salle_de_bain (Mirror, Bin), fireplace_room (Mirror), mori_knob (InnerMat, LTELogo) | Mirror ✅ |
| 4     | Glass on, ray trace (Tf, Ni) | mori_knob (OuterMat) | Glass ✅ |
| 5     | Fresnel mirror, ray trace | — | Mirror |
| 7     | Refraction on, Fresnel, ray trace | fireplace_room (Glass), interior (6 glass materials, lambert2SG/3SG) | Glass ✅ |

### 1.3 Critical Materials Per Scene

#### Cornell Box (cornellbox.mtl)
- 6 simple materials. All `illum 1`. Only Kd/Ka/Ks/Ke.
- `light` has Ke emission. All others are Lambertian.
- **Status: Fully supported.** No changes needed.

#### Conference (conference.mtl)
- 33 materials. All `illum 2`, Ns/Ka/Kd/Ks/Ke/Ni/d.
- No textures. No glass/mirrors. Pure GlossyDielectric+Lambertian.
- Ka ignored (ambient unnecessary for global illumination).
- `d 1.0` everywhere (fully opaque).
- **Status: Functionally supported.** Ka/d parsing is cosmetic.

#### Mori Knob (testObj.mtl)
- 5 materials. illum 1/3/4.
- `OuterMat`: illum 4, Tf 0.75 0.92 0.99, Ni 1.5 → **colored glass** (warm blue tint)
- `InnerMat`: illum 3 → Mirror
- `LTELogo`: illum 3, sharpness 100 → Mirror
- **Status: Partially supported.** Glass works but Tf color is lost — glass renders as neutral instead of blue-tinted.

#### Sibenik Cathedral (sibnek.mtl)
- 9 custom materials with `map_Ka`, `map_Kd`, `map_bump`, `bump`.
- Stained glass uses Ke for emission (already supported).
- Bump/normal maps not parsed.
- **Status: Partially supported.** Diffuse textures work. Bump maps missing.

#### Living Room (living_room.mtl)
- 37+ materials. `illum 2` throughout.
- `CeilingLampshade` has `d 0.500000` → semi-transparent lampshade.
- 5+ lights with `Ke` emission.
- Many `map_Kd` textures.
- **Status: Mostly supported.** Semi-transparent lampshade (d=0.5) not handled — will render opaque.

#### Salle de Bain (salle_de_bain.mtl)
- 14 materials. illum 2/3.
- `Mirror`: Ks 0.99, illum 3 → Mirror ✅
- `Bin`: illum 3, Ks ~0.18 → Mirror (but intended as mildly reflective metal)
- **Status: Supported.** Bin renders as mirror which is specced correctly per illum 3.

#### Sponza (sponza.mtl)
- 20+ materials. illum 2.
- `Tr 0.000` on most materials (fully opaque, harmless).
- `Tf 1.0 1.0 1.0` on most materials (neutral filter, harmless).
- `map_Ka`, `map_Kd`, `map_bump`, `bump` used extensively.
- **Status: Mostly supported.** No actual transparency used. Bump maps missing.

#### Fireplace Room (fireplace_room.mtl)
- 22 materials. illum 1/2/3/7.
- `Glass`: illum 7, Tf 0.1 0.1 0.1 → **dark-tinted glass**.
- `mirror01`: illum 3 → Mirror.
- Many `map_Kd` textures.
- **Status: Partially supported.** Glass Tf tinting is lost — renders as neutral glass.

#### Interior (interior.mtl) — **Most Complex**
- 60+ materials. illum 2/7.
- 6 glass materials with illum 7:
  - `LiquorBottle_GlassA`: Tf 0.1 0.1 0.05, Tr 0.8 → dark amber glass
  - `LiquorBottle_GlassB/C`: Tf 0.2 0.2 0.2 / 0.05 0.1 0.05, Tr 0.8 → tinted glass
  - `CookieJar_Glass`: Tf 0.2 0.2 0.2, Tr 0.9 → clear-ish glass
  - `Paris_Cashregister_Glass`: Tf 0.2 0.2 0.2, Tr 0.8
  - `lambert2SG_light`/`lambert3SG`: illum 7, Tr 0.8 → glass light covers
- 5 emission maps (`map_Ke`):
  - `Paris_Lantern`, `Paris_Ceiling_Lamp`, `Paris_Wall_Light`, `Paris_CeilingFan`, `Wall_Lamp`
- 3 alpha masks (`map_d`):
  - `Plants_plants`, `Plates_Details`, `Cutlery_details`
- Extensive `map_Ka`, `map_Ks` usage
- Several `map_bump` entries
- **Status: Significantly under-supported.** Colored glass is neutral. Emission maps not loaded (5 light fixtures have zero emission). Alpha masks not loaded (plants render as solid rectangles). Specular maps not used.

---

## 2) Current Code Capabilities

### 2.1 Material Struct (`material.h`)

```
Fields:  name, type, Kd(Spectrum), Ks(Spectrum), Le(Spectrum),
         roughness(float), ior(float), diffuse_tex(int), specular_tex(int)
```

Missing fields for full MTL support:
- `Tf` (transmission filter spectrum) — needed for colored glass
- `d` / `Tr` (opacity/transparency) — needed for translucent surfaces
- `bump_tex` (int) — needed for bump/normal maps
- `alpha_tex` (int) — needed for alpha mask cutouts (map_d)
- `emission_tex` (int) — needed for spatially-varying emission (map_Ke)
- `Ka` (Spectrum) — ambient; low priority for GI renderer

### 2.2 OBJ Loader (`obj_loader.cpp`)

**Parsed:** Kd, Ks, Ke, Ns→roughness, Ni→ior, illum, map_Kd  
**Not parsed:** Ka, d, Tr, Tf, map_Ka, map_Ks, map_d, map_Ke, map_bump, bump, sharpness

The illum→MaterialType mapping is correct for the models used. The deferred processing pattern (store illum, apply after all keywords parsed) is well-designed and should be preserved.

### 2.3 BSDF (`bsdf.h`)

- **Glass** (`glass_sample`): Color-neutral. Uses `Spectrum::constant()` for both reflection and transmission throughput. **Does not accept Tf.** The refracted ray carries `(1-F) / cos_t` as a scalar replicated across all wavelength bins. This is physically correct for clear glass but wrong for tinted glass.
- **Mirror** (`mirror_sample`): Uses `Ks` as spectral reflectance. Correct.
- **GlossyMetal/GlossyDielectric**: Full spectral, correct.
- **No texture lookup inside BSDF**: Textures are resolved before calling `bsdf::sample()`. This is the correct design — keep it.

### 2.4 Texture System (`scene.h`)

- `Texture::sample(float2 uv)` returns `float3` (RGB). Already loads RGBA (4 channels) but discards alpha.
- No `sample_alpha(float2 uv)` method.
- GPU path uploads `diffuse_tex` per material and a texture atlas. Only diffuse textures are uploaded.

### 2.5 Renderer Texture Usage (`renderer.cpp`)

Texture lookup happens at 3 places in the renderer:
1. Camera specular chain (line ~194): overwrites `mat.Kd` with texture color
2. Photon tracing bounce (line ~634): overwrites `mat.Kd` with texture color
3. Photon tracing photon deposit path (line ~722): overwrites `mat.Kd` with texture color

Pattern is always: if `mat.diffuse_tex >= 0`, sample texture, convert RGB→spectrum, assign to `mat.Kd`. The `specular_tex` field is never read anywhere.

---

## 3) Gap Analysis

### 3.1 Critical Gaps (Break Scenes)

| Gap | Impact | Scenes Affected |
|-----|--------|----------------|
| **No Tf (transmission filter)** | Glass is color-neutral; amber bottles, blue-tinted knob, dark windows render as clear glass | mori_knob, fireplace_room, interior |
| **No map_Ke (emission texture)** | 5 light fixtures in Interior emit zero light; scene is too dark | interior |
| **No map_d (alpha mask)** | Plants, cutlery, plate details render as solid opaque rectangles | interior |
| **No d/Tr (dissolve/transparency)** | Semi-transparent lampshade renders opaque | living_room, interior |

### 3.2 Visual Quality Gaps (Degrade Appearance)

| Gap | Impact | Scenes Affected |
|-----|--------|----------------|
| **No map_Ks** | Per-texel specular variation missing; surfaces look uniformly shiny or dull | interior, sponza |
| **No map_bump/bump** | Stone, brick, fabric surfaces look flat; no fine detail | sibenik, sponza, interior |
| **No map_Ka** | Minor; ambient maps sometimes double as AO or secondary diffuse | sibenik, sponza, interior |

### 3.3 Non-Gaps (Safe to Ignore)

| Property | Reason to Skip |
|----------|---------------|
| `Ka` (ambient color) | Irrelevant for global illumination renderer; ambient is a hack for rasterizers |
| `sharpness` | Only used once (mori_knob LTELogo=100). MTL spec says it's for reflection map sampling. Our mirror is delta; no effect. |
| `illum 4` vs `illum 7` | Both already map to Glass. The distinction (local vs global ray trace) is irrelevant for a full path tracer. |
| `Tr 0.000` / `Tf 1.0 1.0 1.0` | These are default/neutral values (fully opaque / no filter). Parsing them is bookkeeping but won't change any output. |

---

## 4) Implementation Plan

### Phase 1: MTL Parser Expansion

**Goal:** Parse all used MTL keywords and store them in the Material struct, even if the rendering pipeline doesn't use them yet. This decouples data ingestion from rendering.

#### 1a. Extend Material Struct

**File:** `src/scene/material.h`

Add fields to `struct Material`:

```cpp
// Transmission filter: spectral multiplier per refraction event
// Tf = (1,1,1) means clear glass. Tf = (0.1, 0.1, 0.05) means dark amber.
Spectrum      Tf           = Spectrum::constant(1.0f);

// Opacity: 1.0 = fully opaque, 0.0 = fully transparent
// Maps to MTL "d" (dissolve). "Tr" = 1 - d.
float         opacity      = 1.0f;

// Additional texture slots (−1 = none)
int           alpha_tex    = -1;  // map_d: alpha mask texture
int           emission_tex = -1;  // map_Ke: emission texture
int           bump_tex     = -1;  // map_bump / bump: bump/normal map
int           specular_tex_ks = -1; // renamed: map_Ks specular map
                                    // (specular_tex field exists but unused)
```

**Design note on `specular_tex`:** The existing `specular_tex` field in Material is never assigned or read. Repurpose it for `map_Ks` rather than adding a new field. If `specular_tex` was intended for something else, document the change.

**Spectral conversion of Tf:** The MTL file gives Tf as RGB. Convert using `rgb_to_spectrum_reflectance()` — this is technically a transmittance filter, not a reflectance, but the conversion math is the same (it's a [0,1] color that modulates per-wavelength). The name is slightly misleading but the function is correct: it maps an RGB triplet to a smooth spectral curve whose integral matches the luminance.

#### 1b. Extend OBJ Loader

**File:** `src/scene/obj_loader.cpp`

Add parsing for all missing keywords. The structure follows the existing pattern:

```
else if (keyword == "Ka")     → parse RGB, convert to spectrum, store (or skip)
else if (keyword == "d")      → parse float → mat.opacity = value
else if (keyword == "Tr")     → parse float → mat.opacity = 1.0f - value
else if (keyword == "Tf")     → parse RGB → mat.Tf = rgb_to_spectrum_reflectance(r,g,b)
else if (keyword == "map_Ka") → load texture, store index (or skip if treating as diffuse alias)
else if (keyword == "map_Ks") → load texture, store as specular_tex
else if (keyword == "map_d")  → load texture, store as alpha_tex (IMPORTANT: load grayscale)
else if (keyword == "map_Ke") → load texture, store as emission_tex
else if (keyword == "map_bump" || keyword == "bump") → load texture, store as bump_tex
```

**Texture loading concerns:**
- `map_d`: This is a grayscale alpha mask. Load with `stbi_load(..., 1)` for single channel, or load RGBA and use the red channel. The Texture struct currently assumes 4 channels; either:
  - (a) Always load as RGBA=4 and read the R channel as alpha, or
  - (b) Add a `TextureFormat` enum to distinguish RGB vs grayscale textures.
  - **Recommended: (a)** — simpler, minimal struct changes.

- `map_Ke`: This is an RGB emission map. The RGB values need to be converted to spectral emission at evaluation time, same as Ke → Le conversion. Load as standard RGBA texture.

- `map_bump` / `bump`: These are grayscale height maps (or RGB normal maps if the file is a normal map). Load as RGBA. Usage is deferred to Phase 5.

- **Texture deduplication**: The existing code already deduplicates by `full_path`. The same dedup logic should apply to all new texture types (use a shared helper function for texture loading + dedup).

**Refactor suggestion:** Extract the texture loading + dedup logic into a helper:

```cpp
int load_or_reuse_texture(Scene& scene, const std::string& full_path, int desired_channels = 4);
```

This reduces code duplication across map_Kd, map_Ks, map_d, map_Ke, map_bump.

#### 1c. Post-Processing Updates

**File:** `src/scene/obj_loader.cpp` (illum post-processing)

Update the `illum` post-processing to account for the new Tf:

```
case 4: case 6: case 7:
    mat.type = MaterialType::Glass;
    // If Tf is not neutral (i.e., != 1.0), mark for colored glass
    // (No new MaterialType needed — Glass + non-neutral Tf is sufficient)
    break;
```

No new `MaterialType` is needed. The existing `Glass` type combined with a `Tf` field naturally represents colored glass. The BSDF code will branch on `Tf` being non-constant.

---

### Phase 2: Transmission Filter (Colored Glass)

**Goal:** Make glass BSDF wavelength-dependent via Tf, producing colored caustics and tinted glass appearance.

**Guideline constraint (§0, §7.1.1):** "Spectral bins never mix during transport." This is naturally satisfied because Tf is a per-wavelength multiplier — each bin is scaled independently.

#### 2a. Modify glass_sample

**File:** `src/bsdf/bsdf.h`

Current signature:
```cpp
BSDFSample glass_sample(float3 wo, float ior, PCGRng& rng);
```

New signature:
```cpp
BSDFSample glass_sample(float3 wo, float ior, const Spectrum& Tf, PCGRng& rng);
```

**Implementation change in the refraction branch:**

Currently:
```cpp
float factor = (1.f - F) / (fabsf(s.wi.z) + EPSILON);
s.f = Spectrum::constant(factor);  // ← Color-neutral
```

After:
```cpp
float factor = (1.f - F) / (fabsf(s.wi.z) + EPSILON);
// Apply transmission filter per wavelength bin
for (int i = 0; i < NUM_LAMBDA; ++i) {
    s.f.value[i] = factor * Tf.value[i];
}
```

**Physics rationale:** When light refracts through a glass surface, the transmission filter attenuates each wavelength differently. Amber glass (Tf ≈ 0.8, 0.5, 0.1) passes long wavelengths (red) and absorbs short wavelengths (blue). This is a surface property in the MTL model (not volumetric absorption), so it's applied once per refraction event.

**Reflection branch:** The reflection branch should NOT apply Tf — Fresnel reflection is wavelength-independent for dielectric surfaces (the Fresnel equations depend on IOR which is a scalar in our model). The reflected throughput remains `Spectrum::constant(F / cos)`.

**If we later add dispersion (wavelength-dependent IOR):** Each wavelength bin would have a different IOR, making Fresnel itself spectral. This is a future extension mentioned in the guideline (§7.1.1). For now, IOR is scalar.

#### 2b. Update bsdf::sample() Dispatch

**File:** `src/bsdf/bsdf.h`

```cpp
case MaterialType::Glass:
    return glass_sample(wo, mat.ior, mat.Tf, rng);
    //                              ^^^^^^ new parameter
```

#### 2c. Update Renderer Throughput

**File:** `src/renderer/renderer.cpp`

The camera specular chain and photon tracing bounce code already multiply throughput by `bsdf_sample.f`. Since `bsdf_sample.f` now carries the Tf tinting, no renderer changes are needed for the basic case.

However, verify that the specular chain throughput update (§7.1.1 of guideline) correctly applies:
```
T *= f_s(x, wi, wo, λ)
```
Where `f_s` is now Tf-tinted for refraction. This should work naturally.

#### 2d. GPU / OptiX Path

**Files:** `src/optix/launch_params.h`, `src/optix/optix_renderer.cpp`, OptiX device code

The GPU path needs:
1. Upload `Tf` per material (as a device-side Spectrum array or as raw float arrays)
2. The device-side glass BSDF kernel must accept Tf
3. Texture atlas updates if any Tf-related textures exist (unlikely in our scenes)

**Approach:** Add a `d_Tf_` DeviceBuffer in `OptixRenderer`, upload `mat.Tf` for each material. In the OptiX closest-hit program, read Tf from the material and pass to `glass_sample`.

#### 2e. Photon Map Impact — Colored Caustics

This is where Tf becomes most visually impactful. When a photon refracts through colored glass:

1. Photon carries spectral flux Φ(λ)
2. After refraction: Φ'(λ) = Φ(λ) × Tf(λ) × (1-F) / cos_t
3. When deposited on a diffuse surface beyond the glass → caustic is colored

The photon tracing code already multiplies photon flux by `bsdf_sample.f`, so colored caustics will emerge automatically once `glass_sample` returns Tf-tinted `f`.

**Caustic map separation:** The guideline (§5.2.4) requires caustic photons (`hasSpecularChain == true`) to go into a separate caustic map with smaller gather radius. This is already implemented and doesn't need changes — glass refraction sets `hasSpecularChain = true`.

---

### Phase 3: Alpha Masking (map_d / d cutouts)

**Goal:** Support alpha-tested geometry (plants, cutlery details, plate patterns).

**Guideline constraint:** Alpha masking affects ray intersection, not BSDF evaluation. It's a geometric operation: "is there material here, or is it a hole?"

#### 3a. Design Decision: Alpha Testing vs Alpha Blending

Two approaches for handling `d < 1` or `map_d` alpha:

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Alpha testing** (stochastic) | At each intersection, if `alpha(uv) < random()`, treat as miss and continue ray | Simple, no material type changes, works with all BSDFs | Adds noise, requires continuation rays |
| **Alpha blending** | New MaterialType or special BSDF that blends throughput | Physically motivated | Complex, changes photon deposition, changes NEE |

**Recommended: Stochastic alpha testing.** It's standard in production renderers, adds minimal noise at high sample counts, and doesn't require new material types or BSDF changes.

#### 3b. Implementation — CPU Path

**File:** `src/renderer/renderer.cpp`

At every intersection point (camera trace, photon trace, NEE shadow rays), after computing hit and before evaluating the material:

```cpp
// Alpha test: stochastic transparency
float alpha = mat.opacity;  // default from "d" keyword
if (mat.alpha_tex >= 0 && mat.alpha_tex < (int)scene_->textures.size()) {
    float3 tex_alpha = scene_->textures[mat.alpha_tex].sample(hit.uv);
    alpha *= tex_alpha.x;  // Use R channel as alpha
}
if (alpha < 1.0f && rng.next_float() > alpha) {
    // Treat as transparent — continue ray through this surface
    ray.origin = hit.position + ray.direction * EPSILON;
    // Don't count as a bounce (it's a pass-through)
    continue; // or re-trace
}
```

**Where to add this check:**
1. Camera specular chain loop (line ~190)
2. Photon trace bounce loop (line ~630)
3. NEE shadow ray test (must not incorrectly occlude through alpha-masked geometry)

**Critical: NEE shadow rays.** The `any_intersection()` function for shadow rays must also do alpha testing. If the shadow ray hits an alpha-masked triangle and fails the alpha test, it must continue checking for further intersections along the ray. This may require modifying the intersection code or using a loop.

#### 3c. Implementation — GPU Path

**Files:** OptiX any-hit program

OptiX has a natural mechanism for this: the **any-hit program**. When a ray hits alpha-masked geometry, the any-hit program samples the alpha texture and calls `optixIgnoreIntersection()` if the alpha test fails. This is the standard approach for GPU alpha masking.

1. Upload `alpha_tex` per material to device
2. Upload opacity per material
3. In the any-hit program: sample alpha texture, compare with random threshold, ignore if transparent

**OptiX best practice:** Use a separate GAS (Geometry Acceleration Structure) for alpha-masked geometry to limit the any-hit overhead to only those triangles that need it.

#### 3d. Dissolve (d) Without Texture

For materials like CeilingLampshade (`d 0.5`) that don't have map_d but have `d < 1`:
- The stochastic alpha test works the same way: `alpha = mat.opacity = 0.5`
- 50% of rays pass through, 50% interact with the surface
- This naturally creates a semi-transparent appearance

**Photon deposition on semi-transparent surfaces:** When a photon passes through (alpha test fails), it doesn't deposit — it continues. When it interacts (alpha test passes), it deposits normally. Throughput adjustment: divide photon flux by `alpha` to maintain energy conservation:

```cpp
if (alpha < 1.0f) {
    throughput /= alpha;  // Compensate for stochastic acceptance
}
```

Wait — actually, this is importanced-sampling a Bernoulli: accept with probability `alpha`, reject with probability `1-alpha`. On acceptance, the estimator is flux / alpha to remain unbiased. On rejection, the ray continues. This is correct Monte Carlo.

---

### Phase 4: Emission Texture Maps (map_Ke)

**Goal:** Support spatially-varying emission for light fixtures (lanterns, lamps, ceiling fans).

**Impact:** Without this, 5 light fixtures in the Interior scene emit zero light because their Ke base color is (0,0,0) and all emission comes from the map_Ke texture.

#### 4a. Emission Map Evaluation

**File:** `src/renderer/renderer.cpp`

At every point where emission is evaluated (camera direct hit, NEE evaluation, photon emission sampling):

```cpp
if (mat.emission_tex >= 0 && mat.emission_tex < (int)scene_->textures.size()) {
    float3 rgb = scene_->textures[mat.emission_tex].sample(hit.uv);
    // Convert texture RGB to spectral emission
    Spectrum Le_tex = rgb_to_spectrum_emission(rgb.x, rgb.y, rgb.z);
    // Modulate: if Ke is set, multiply (intensity control); if Ke is zero, use texture directly
    if (mat.Le.max_component() > 0.f) {
        // Ke acts as intensity scale for the emission map
        mat.Le = mat.Le * Le_tex;  // per-wavelength multiply? Or Ke as scalar scale?
    } else {
        mat.Le = Le_tex;
    }
}
```

**Design decision:** How to combine Ke (base) and map_Ke (texture)?

Looking at the Interior MTL file:
```
Ke 0.000000 0.000000 0.000000
map_Ke textures/Paris_Lantern_Light.png
```

Ke is (0,0,0) and map_Ke provides the actual emission. So the texture replaces, not modulates. But materials with non-zero Ke and a map_Ke should multiply (standard MTL convention: map modulates the base value).

**Rule:**
- If `Ke == (0,0,0)` and `map_Ke` exists: `Le = rgb_to_spectrum_emission(map_Ke_sample)`
- If `Ke != (0,0,0)` and `map_Ke` exists: `Le = base_Le * rgb_to_spectrum_emission(map_Ke_sample)` (multiply)
- If `Ke != (0,0,0)` and no `map_Ke`: `Le = base_Le` (current behavior)

**Simpler approach:** Always treat map_Ke as the emission color (override), since that matches how all our scenes use it. If Ke is non-zero, use it as a scalar multiplier on the texture luminance.

#### 4b. Emissive Triangle Detection

**File:** `src/scene/scene.h` (emissive alias table construction)

Currently, `Material::is_emissive()` checks `Le.max_component() > 0`. For materials with `Ke=(0,0,0)` but `emission_tex >= 0`, the material needs to be flagged as emissive at load time, not at per-pixel evaluation time.

**Solution:** After loading all textures and materials, do a post-processing pass:

```cpp
for (auto& mat : scene.materials) {
    if (mat.emission_tex >= 0 && mat.Le.max_component() == 0.f) {
        // Sample the emission texture at a few points to estimate average emission
        // Or simply set a flag / sentinel Le value to mark as emissive
        mat.Le = Spectrum::constant(1.0f);  // Placeholder — will be overwritten per-pixel
        mat.type = MaterialType::Emissive;
    }
}
```

**Better approach:** Add an `emissive_flag` bool to Material, separate from Le magnitude:

```cpp
bool has_emission() const { return Le.max_component() > 0.f || emission_tex >= 0; }
```

Then use `has_emission()` instead of `is_emissive()` for emissive alias table construction. The alias table needs per-triangle emission power; for textured emitters, compute average emission by sampling the texture over the triangle (or use a precomputed average).

**Precomputed average emission for alias table:**

At load time, for each emission-textured material, compute the texture's average RGB:
```cpp
float3 avg_rgb = compute_texture_average(scene.textures[mat.emission_tex]);
mat.Le = rgb_to_spectrum_emission(avg_rgb.x, avg_rgb.y, avg_rgb.z);
```

This gives a reasonable per-triangle power estimate for the alias table, while the per-pixel Le is evaluated from the texture at render time.

#### 4c. NEE Considerations

When NEE samples a point on an emission-textured triangle, the emission at that point comes from the texture, not from the uniform Le. The NEE code must:

1. Sample a triangle from the alias table (using the precomputed average power)
2. Sample a point on the triangle
3. Look up the emission texture at that UV coordinate
4. Use the actual texture emission value (not the average) for the radiance estimate
5. The PDF is still based on the alias table (power-proportional) — no change

This naturally reduces variance: bright spots on the texture get their actual brightness, dark spots contribute less.

#### 4d. Photon Emission Sampling

When emitting photons from textured-emission triangles:
1. Photon is emitted from a random point on the triangle
2. The photon's initial flux should be proportional to `Le_texture(uv)`, not the average
3. Since photon sampling uses the alias table (power-proportional per triangle), and the point is uniformly sampled on the triangle, the flux should be:
   ```
   Φ = Le_texture(uv) × A_triangle × π / N_photons_for_this_triangle
   ```

This is important for correctness: if a lamp has a bright filament and a dark housing, photons should carry more flux when emitted from the filament region.

---

### Phase 5: Extended Texture Support

#### 5a. Specular Map (map_Ks)

**Goal:** Per-texel specular reflectance variation.

**Implementation:** At every point where `mat.Ks` is used in BSDF evaluation:

```cpp
if (mat.specular_tex >= 0 && mat.specular_tex < (int)scene_->textures.size()) {
    float3 rgb = scene_->textures[mat.specular_tex].sample(hit.uv);
    mat.Ks = rgb_to_spectrum_reflectance(rgb.x, rgb.y, rgb.z);
}
```

This mirrors the existing diffuse texture pattern. Apply in the same 3 locations where diffuse textures are applied (camera trace, photon trace bounce, photon deposit).

**Guideline impact:** None. Ks is already spectral; texture lookup just provides per-texel values. Spectral bins don't mix.

#### 5b. Ambient Map (map_Ka)

**Decision: Skip or alias to diffuse.**

Ka is the ambient reflectance. In a global illumination renderer, there is no ambient term — indirect illumination replaces it. However, in some MTL files (especially older ones), map_Ka textures are actually ambient occlusion maps or simply copies of the diffuse texture.

**Recommended approach:** Parse and store `map_Ka` but don't use it in rendering. If users report missing detail, selectively treat map_Ka as a secondary diffuse modulator:
```cpp
mat.Kd *= ka_texture_sample;  // AO-like darkening
```

This is a creative interpretation, not physically accurate. For now: parse and ignore.

#### 5c. Bump Map (map_bump / bump)

**Goal:** Perturb shading normals for surface detail.

**This is the most complex texture feature.** Bump mapping modifies the shading normal at each hit point, which affects:
- BSDF evaluation (normal determines local frame)
- NEE geometric term (cos_theta_i uses shading normal)
- Photon deposition normal consistency filter (§6.4 — uses geometric normal, not shading normal!)

**Implementation outline:**

1. **Texture loading:** Load bump map as grayscale heightfield.

2. **Normal perturbation:** At each hit point, compute the perturbed normal:
   ```
   height(u,v) = bump_texture.sample(u, v).x
   dh/du = (height(u+du, v) - height(u-du, v)) / (2*du)
   dh/dv = (height(u, v+dv) - height(u, v-dv)) / (2*dv)
   perturbed_normal = normalize(N - dh/du * T - dh/dv * B)
   ```
   Where T, B are the tangent/bitangent at the hit point.

3. **Tangent computation:** Requires per-triangle tangent vectors. Currently not stored or computed. Would need:
   - Compute tangent/bitangent from UV coordinates at triangle vertices
   - Store per-vertex or per-triangle tangents
   - Interpolate at hit point

4. **Critical constraint (§6.3, §6.4 of guideline):**
   > "Which normal to use: The tangential metric and plane distance filter must use geometric normals (face normals), not shading normals."
   
   Bump-mapped normals are shading normals. They must be used for BSDF evaluation but NOT for the photon gather distance metric or surface consistency filter. The code must distinguish between `hit.geometric_normal` and `hit.shading_normal`.

5. **Frame construction:** The BSDF operates in a local frame where z = normal. With bump mapping, this frame must use the perturbed shading normal, not the geometric normal.

**Recommendation: Defer to a later phase.** Bump mapping requires tangent vectors, per-fragment normal perturbation, and careful separation of geometric vs shading normals. It's a significant undertaking, and the scenes that use it (sibenik, sponza) still look acceptable without it. Implement after Phases 1–4 are stable.

---

### Phase 6: Rough Glass / Translucency (Future)

**Goal:** Support non-delta glass (microfacet refraction) for frosted glass, semi-transparent materials.

**Context:** None of our scenes explicitly require rough glass. However, combining `illum 7` (glass) with `Ns < 1000` (low Phong exponent → high roughness) could be interpreted as rough glass. Currently, Glass is always delta (perfect specular).

#### 6a. Design

A rough glass BSDF combines GGX microfacet sampling with Fresnel-weighted reflect/refract:

- Sample microfacet half-vector from GGX distribution
- Compute Fresnel reflectance at the microfacet
- Reflect or refract through the microfacet
- Weight by GGX geometry term
- Apply Tf for transmission

This is essentially Walter et al. (2007) "Microfacet Models for Refraction through Rough Surfaces."

#### 6b. MaterialType Consideration

Options:
1. Add `MaterialType::RoughGlass` — explicit new type
2. Reuse `Glass` with `roughness > threshold` to select rough vs delta

**Recommended: Option 2.** If `mat.type == Glass && mat.roughness < 0.01`, use delta glass. Otherwise use rough glass. This avoids adding a new enum value and keeps the material classification based on illum.

#### 6c. BSDF Implementation

```cpp
BSDFSample rough_glass_sample(float3 wo, float ior, float roughness,
                               const Spectrum& Tf, PCGRng& rng);
```

Key steps:
1. Sample GGX half-vector h
2. F = fresnel_dielectric(dot(wo, h), eta)
3. If reflect: wi = reflect(wo, h), f = F * D(h) * G(wo, wi) / (4 * cos_o * cos_i)
4. If refract: wi = refract(wo, h, eta), f = (1-F) * Tf * D(h) * G(wo, wi) * |dot(wi,h)| / (denominator with Jacobian)

The refraction Jacobian from half-vector to refracted direction is:
$$
\left|\frac{\partial \omega_h}{\partial \omega_i}\right| = \frac{\eta^2 |\omega_i \cdot \omega_h|}{(\eta (\omega_o \cdot \omega_h) + \omega_i \cdot \omega_h)^2}
$$

This is non-trivial to implement correctly. Numerical stability near grazing angles and TIR boundaries requires careful handling.

#### 6d. Guideline Impact

- Rough glass is NOT delta → photons DO deposit at rough glass surfaces (§5.2.3: "deposit only at non-delta surfaces")
- `is_specular()` must return `false` for rough glass
- Camera rays stop at rough glass (it's non-delta → treated like a glossy/diffuse surface)
- This changes the `hasSpecularChain` logic: rough glass refraction does NOT set it to true

**This is a significant behavioral change.** Currently, all glass is delta, so photons never deposit on glass and camera rays bounce through glass. Rough glass breaks this invariant.

**Recommendation: Only implement if scenes need it.** Currently our scenes don't have explicit rough glass. Defer.

---

## 5) Per-File Impact Assessment

### Files That Need Changes

| File | Phase | Changes |
|------|-------|---------|
| `src/scene/material.h` | 1 | Add Tf, opacity, alpha_tex, emission_tex, bump_tex fields; add `has_emission()` method |
| `src/scene/obj_loader.cpp` | 1 | Parse Ka, d, Tr, Tf, map_Ka, map_Ks, map_d, map_Ke, map_bump, bump; extract texture loading helper |
| `src/bsdf/bsdf.h` | 2 | Add Tf parameter to `glass_sample`; update `bsdf::sample()` dispatch |
| `src/renderer/renderer.cpp` | 2,3,4,5 | Alpha test at all intersection points; emission texture lookup; specular texture lookup; frame-level texture binding |
| `src/scene/scene.h` | 3,4 | Add `sample_alpha()` to Texture; update `has_emission()` for alias table; alpha-aware intersection |
| `src/optix/launch_params.h` | 2,3,4 | Add device pointers for Tf, opacity, alpha_tex, emission_tex |
| `src/optix/optix_renderer.cpp` | 2,3,4 | Upload new material data to device |
| OptiX device code (`.cu` files) | 2,3,4 | Glass BSDF with Tf; any-hit alpha test; emission texture lookup |

### Files That Do NOT Need Changes

| File | Reason |
|------|--------|
| `src/core/spectrum.h` | Spectrum arithmetic already supports per-wavelength multiply |
| `src/core/types.h` | No new types needed |
| `src/core/config.h` | No new config parameters needed (alpha threshold could be added but 0.5 default is fine) |
| `src/photon/*` | Photon storage/gather unchanged; Tf tinting flows through throughput naturally |
| `src/debug/*` | Debug visualizations don't need changes |

---

## 6) Guideline Compliance Checklist

For each change, verify these invariants from `revised_guideline_v2.md`:

| Invariant | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
|-----------|---------|---------|---------|---------|---------|
| **Spectral bins never mix** | ✅ Parsing only | ✅ Tf is per-λ multiply | ✅ Alpha test is scalar | ✅ Le_tex goes through rgb_to_spectrum per pixel | ✅ Ks_tex per-λ |
| **Photons store flux, not radiance** | N/A | ✅ Throughput multiply preserves flux units | ✅ Pass-through preserves flux | ✅ Emission flux from texture | N/A |
| **lightPathDepth ≥ 2 for deposit** | N/A | ✅ Glass is delta; no deposit | ✅ Alpha pass-through doesn't count as bounce | ✅ No change to deposit rule | N/A |
| **NEE never double-counted** | N/A | ✅ NEE unchanged | ✅ Shadow rays respect alpha | ✅ NEE samples texture Le | N/A |
| **Tangential kernel uses geometric normal** | N/A | N/A | N/A | N/A | ⚠️ Phase 5c bump: must separate geometric/shading normal |
| **CPU/GPU distributional equivalence** | ✅ Same data | ⚠️ Must implement on both | ⚠️ Must implement on both | ⚠️ Must implement on both | ⚠️ Must implement on both |

---

## 7) Testing Strategy

### 7.1 Unit Tests

| Test | Phase | Description |
|------|-------|-------------|
| `test_tf_parsing` | 1 | Load mori_knob MTL, verify Tf spectrum is non-constant |
| `test_opacity_parsing` | 1 | Load interior MTL, verify Glass materials have opacity < 1 |
| `test_map_d_loading` | 1 | Load interior MTL, verify alpha_tex >= 0 for Plants_plants |
| `test_map_ke_loading` | 1 | Load interior MTL, verify emission_tex >= 0 for Paris_Lantern |
| `test_colored_glass_bsdf` | 2 | Call glass_sample with non-neutral Tf; verify f is non-constant across λ |
| `test_clear_glass_unchanged` | 2 | Call glass_sample with Tf=(1,1,1); verify identical to old behavior |
| `test_alpha_mask_passthrough` | 3 | Trace ray at alpha-masked geometry; verify some rays pass through |
| `test_emission_texture` | 4 | Evaluate Le at textured emitter; verify non-zero emission |

### 7.2 Integration Tests

| Test | Phase | Description |
|------|-------|-------------|
| **Mori Knob render** | 2 | Render mori_knob; verify outer shell has blue tint (not clear) |
| **Interior render** | 3,4 | Render interior; verify plants are leaf-shaped (not rectangles), lamps emit light |
| **Energy conservation** | 2,3 | Furnace test: uniform scene with colored glass; verify energy is conserved ±1% |
| **CPU↔GPU match** | All | PSNR comparison between CPU and GPU renders for each affected scene |

### 7.3 Visual Regression

For each scene, compare renders before/after:
- Cornell Box: should be identical (no new features used)
- Mori Knob: glass should be blue-tinted
- Interior: plants should have alpha cutouts, lamps should glow
- Fireplace Room: glass should be dark-tinted
- Living Room: lampshade should be semi-transparent

---

## 8) Priority & Dependencies

```
Phase 1 (Parser)  ──────────────────→  Required by all subsequent phases
    │
    ├── Phase 2 (Colored Glass Tf) ──→  Highest visual impact, simplest BSDF change
    │
    ├── Phase 4 (Emission Maps) ─────→  Critical for Interior scene (dark without it)
    │
    ├── Phase 3 (Alpha Masking) ─────→  Critical for Interior scene (plants)
    │
    └── Phase 5a (Specular Maps) ────→  Visual quality improvement
         │
         └── Phase 5c (Bump Maps) ──→  Complex; defer until pipeline is stable
              │
              └── Phase 6 (Rough Glass) → No current scene requires it; future work
```

### Recommended Execution Order

1. **Phase 1** (Parser) — 1 day. Foundation for everything.
2. **Phase 2** (Tf) — 1 day. Small BSDF change, big visual payoff.
3. **Phase 4** (map_Ke) — 1–2 days. Makes Interior renderable.
4. **Phase 3** (map_d + dissolve) — 2 days. Intersection changes are delicate.
5. **Phase 5a** (map_Ks) — 0.5 day. Drop-in parallel to map_Kd.
6. **Phase 5c** (Bump) — 3–5 days. Needs tangent vectors, normal perturbation, geometric/shading normal split.
7. **Phase 6** (Rough glass) — 3–5 days. Complex BSDF, changes photon deposition semantics.

**Total estimated effort (Phases 1–5a):** 5–6 days for core functionality.  
**Total with bump + rough glass:** 11–16 days.

---

## Appendix A: MTL Properties Reference

From the Wavefront MTL specification (paulbourke.net / loc.gov fdd000508):

| Property | Type | Range | Description |
|----------|------|-------|-------------|
| Ka | RGB | 0–1 | Ambient reflectance |
| Kd | RGB | 0–1 | Diffuse reflectance |
| Ks | RGB | 0–1 | Specular reflectance |
| Ke | RGB | 0–∞ | Emission |
| Ns | float | 0–1000 | Specular exponent (Phong) |
| Ni | float | 0.001–10 | Index of refraction |
| d | float | 0–1 | Dissolve (opacity). 1 = opaque |
| Tr | float | 0–1 | Transparency. Tr = 1 - d |
| Tf | RGB | 0–1 | Transmission filter color |
| illum | int | 0–10 | Illumination model |
| map_Ka | path | — | Ambient texture map |
| map_Kd | path | — | Diffuse texture map |
| map_Ks | path | — | Specular texture map |
| map_Ns | path | — | Specular exponent map |
| map_d | path | — | Alpha/dissolve texture map |
| map_Ke | path | — | Emission texture map |
| map_bump / bump | path | — | Bump (height) map |

## Appendix B: Per-Scene Compatibility Matrix

After implementing Phases 1–5a:

| Scene | Diffuse | Specular | Glass Color | Emission | Alpha | Bump | Status |
|-------|---------|----------|-------------|----------|-------|------|--------|
| Cornell Box | ✅ | N/A | N/A | ✅ | N/A | N/A | **Complete** |
| Conference | ✅ | ✅ | N/A | N/A | N/A | N/A | **Complete** |
| Mori Knob | ✅ | ✅ | ✅ Phase 2 | N/A | N/A | N/A | **Complete** |
| Sibenik | ✅ | N/A | N/A | ✅ | N/A | ❌ bumps | **Functional** |
| Living Room | ✅ | N/A | N/A | ✅ | ✅ Phase 3 (d=0.5) | N/A | **Complete** |
| Salle de Bain | ✅ | ✅ | N/A | N/A | N/A | N/A | **Complete** |
| Sponza | ✅ | ✅ Phase 5a | N/A | N/A | N/A | ❌ bumps | **Functional** |
| Fireplace Room | ✅ | N/A | ✅ Phase 2 | N/A | N/A | N/A | **Complete** |
| Interior | ✅ | ✅ Phase 5a | ✅ Phase 2 | ✅ Phase 4 | ✅ Phase 3 | ❌ bumps | **Complete** |

"Functional" = renders correctly but missing bump detail.  
"Complete" = all MTL properties respected.
