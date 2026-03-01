# Photon-Beam Material Extensions for Wavefront MTL (`pb_*`)

**Status:** Draft spec for renderer integration (MTL remains backward compatible)

This document defines how to interpret the custom `pb_*` tokens added to standard Wavefront `.mtl` files.

- **Backwards compatible:** classic MTL loaders should ignore unknown keys.
- **Renderer-facing:** values are defined in physical terms (or clear heuristics) and include defaults.
- **Spectral-friendly:** where applicable, parameters are RGB triples intended to be converted into your internal spectral representation.

---

## 1) Design goals

1. **Expressive enough** for: translucency (milk/ceramic/water), glossy metals, basic fabric, wallpaper/paint, wood (natural & painted), stone.
2. **Minimal disruption** to Wavefront workflows.
3. **Deterministic parsing:** no name-based hacks required, but optional `pb_semantic` provides strong hints.
4. **Unit-aware media:** volumetric coefficients require a length unit; we support explicit scene-scale hints.

---

## 2) Compatibility with classic MTL

### 2.1 Parsing rule

- Parse classic MTL keywords as usual (`newmtl`, `Kd`, `Ks`, `Ka`, `Ke`, `Ns`, `Ni`, `d`, `Tr`, `Tf`, `illum`, `map_*`).
- Parse any `pb_*` tokens you recognize.
- **Ignore unknown keys** (both classic and `pb_*`) without failing.

### 2.2 Precedence rule

- If a `pb_*` token provides a value, it **overrides** any derived value from classic MTL.
- Otherwise derive reasonable defaults from classic MTL.

Example:
- If `pb_eta` is present, use it even if `Ni` exists.
- If `pb_roughness` is present, use it even if `Ns` exists.

---

## 3) Scene scale and units

Many `pb_*` values are dimensionless (e.g., roughness). Some are *per-length* coefficients (participating media).

### 3.1 Length unit

We define “**scene length unit**” as the unit used by geometry coordinates (OBJ positions).

To interpret volumetric coefficients consistently across scenes, the `.mtl` may include one optional header hint:

- `pb_meters_per_unit <float>`
  - Example: `1.0` if geometry coordinates are meters.
  - Example: `0.01` if geometry coordinates are centimeters.

Optional debug hints:
- `pb_scene_bbox_min x y z`
- `pb_scene_bbox_max x y z`

### 3.2 Converting coefficients

If volumetric coefficients are authored in **per scene unit**:

- $\sigma_{\text{per meter}} = \sigma_{\text{per unit}} / \text{pb\_meters\_per\_unit}$

If `pb_meters_per_unit` is missing:
- Prefer an `.obj` unit declaration if your pipeline provides one.
- Otherwise default to `1.0` and optionally warn.

---

## 4) Token reference (surface)

All floats are assumed finite. Clamp ranges where noted.

### 4.1 `pb_brdf`

`pb_brdf <lambert|dielectric|conductor|clearcoat|emissive|fabric>`

Selects the surface scattering model.

Recommended renderer interpretations:
- `lambert`: diffuse only
- `dielectric`: microfacet dielectric (GGX) with Fresnel
- `conductor`: microfacet conductor (GGX) with complex IOR
- `clearcoat`: layered model (coat microfacet dielectric over a base BRDF)
- `emissive`: purely emissive (still may have scattering if you choose, but simplest is emission-only)
- `fabric`: diffuse + sheen lobe (simple cloth)

Default if omitted:
- If `Ke` is non-zero → `emissive`
- Else if `illum` indicates glass/refraction (4/6/7) or `Ni` present and transparency suggests transmission → `dielectric`
- Else if `illum` indicates mirror/reflection (3/5) → `conductor` or “mirror” special-case (delta)
- Else if `Ks` is very low → `lambert`
- Else → `dielectric` (glossy dielectric)

### 4.2 `pb_semantic`

`pb_semantic <tag>`

Optional classification hint for choosing defaults or specialized parameter mapping.

Suggested tags (non-exhaustive):
- `subsurface`, `glass`, `metal`, `fabric`, `leather`, `wood_natural`, `wood_painted`, `wallpaper`, `stone`, `plastic`

Renderer guidance:
- Treat this as a **hint**, not a hard override.
- Use it to pick default roughness/coat strength when explicit parameters are missing.

### 4.3 Roughness and anisotropy

- `pb_roughness <0..1>`
  - Microfacet roughness parameter $\alpha$ (GGX recommended).
  - **Convention:** interpret `pb_roughness` as **GGX alpha** directly (not “perceptual roughness”).
    - If you prefer perceptual roughness internally, convert consistently (e.g., $\alpha = r^2$).
  - Clamp to `[0.001, 1.0]` for numerical stability.

- `pb_anisotropy <0..1>`
  - 0 = isotropic, 1 = strong anisotropy.

- `pb_roughness_x <0..1>` / `pb_roughness_y <0..1>`
  - Explicit anisotropic roughness parameters.
  - If provided, they override `pb_roughness` for anisotropic BRDFs.

Default mapping from classic MTL `Ns` (Phong exponent) if no `pb_roughness` is provided:

A common conversion for GGX is:
- $\alpha = \sqrt{\frac{2}{Ns + 2}}$

Clamp $Ns$ to a reasonable range (e.g., `[1, 10000]`) before converting.

### 4.4 Index of refraction

- `pb_eta <float>`
  - IOR for dielectrics.

Defaults:
- If `Ni` exists → `pb_eta = Ni`
- Else for common dielectrics → `1.5`

### 4.4b Chromatic dispersion

- `pb_dispersion <cauchy_b>`
  - Enables wavelength-dependent IOR via the **Cauchy equation**: $n(\lambda) = A + B / \lambda^2$ ($\lambda$ in nm).
  - The single parameter is the **Cauchy B coefficient** (nm²).
  - **A is auto-derived** so that $n(589\text{nm}) = \text{pb\_eta}$ (sodium D-line anchor).
  - Larger B → more separation between red and blue light → more visible chromatic aberration.

Typical values:
- Crown glass (subtle): `pb_dispersion 4200`
- Flint glass (moderate): `pb_dispersion 12000`
- Super-flint / artistic (dramatic rainbows): `pb_dispersion 50000`

Notes:
- Requires `pb_brdf dielectric` (glass material).
- On the GPU, the hero-wavelength strategy is used: `hero_bins[0]` determines refraction direction; all other bins share that direction with per-wavelength Fresnel weights.
- Omitting `pb_dispersion` uses a constant IOR (no chromatic aberration).

### 4.5 Conductor optical constants

- `pb_conductor_eta r g b`
- `pb_conductor_k r g b`

Represents a **complex IOR** per channel. Renderer uses these to compute spectral Fresnel for conductors.

Defaults if missing:
- If `Ks` exists, you may approximate with a “tint” approach (not physically exact):
  - Use `Ks` as F0 and derive an equivalent conductor (implementation-dependent).
- Otherwise use a generic metal preset.

### 4.6 Transmission and thin materials

- `pb_transmission <0..1>`
  - Controls how much light is transmitted through the surface (in addition to Fresnel effects).
  - For **solid glass**, transmission is usually implied by the dielectric model; this parameter is mainly for artist control and thin sheets.

- `pb_thin <0|1>`
  - If `1`, treat as a **thin sheet** material: no interior medium is required and refraction may be approximated as “thin transmission”.

- `pb_thickness <float>`
  - Thickness in **meters** for thin materials.
  - Used to convert absorption/scattering for thin sheets (Beer–Lambert through thickness).

Deriving defaults from classic MTL:
- If `d < 1` or `Tr > 0` → set `pb_transmission` accordingly.
  - Suggested: `pb_transmission = clamp(1 - d, 0, 1)` if using `d` as dissolve.
  - If `Tr` exists, prefer `pb_transmission = clamp(Tr, 0, 1)`.

If neither `pb_transmission` nor transparency hints are present:
- For `pb_brdf dielectric` intended as **glass** (e.g., `illum` 4/6/7 and `Kd≈0`) you may default to `pb_transmission = 1`.
- For `pb_brdf dielectric` intended as **plastic/paint** (typical `Kd>0`) default to `pb_transmission = 0`.

### 4.7 Token summary (quick reference)

| Token | Type | Meaning | Typical use |
|------|------|---------|-------------|
| `pb_brdf` | enum | BRDF/BSDF model selection | all |
| `pb_semantic` | string | material class hint | all |
| `pb_roughness` | float | GGX $\alpha$ | most non-matte |
| `pb_anisotropy` | float | anisotropy strength | brushed metals |
| `pb_roughness_x/y` | float | anisotropic $\alpha_x/\alpha_y$ | brushed metals |
| `pb_eta` | float | dielectric IOR | glass, varnish |
| `pb_dispersion` | float (nm²) | Cauchy B coefficient; enables chromatic dispersion | glass prisms |
| `pb_conductor_eta/k` | vec3 | complex IOR | metals |
| `pb_transmission` | float | transmission weight | glass, thin sheets |
| `pb_thin` | bool | thin sheet mode | paper, lampshades |
| `pb_thickness` | float (m) | physical thickness | thin sheets |
| `pb_sheen` | float | cloth sheen strength | fabric |
| `pb_sheen_tint` | float | sheen tinting | fabric |
| `pb_clearcoat*` | float | clearcoat layer controls | varnish/paint |
| `pb_medium` | enum | participating medium enable | translucency |
| `pb_density` | float | medium scale | translucency |
| `pb_sigma_a/s` | vec3 | absorption/scattering coefficients | translucency |
| `pb_g` | float | HG anisotropy | translucency |

---

## 5) Token reference (layering)

When `pb_brdf = clearcoat`:

- Base layer:
  - `pb_base_brdf <lambert|dielectric|conductor>`
  - `pb_base_roughness <0..1>`

- Coat controls:
  - `pb_clearcoat <0..1>` (weight)
  - `pb_clearcoat_roughness <0..1>`

Interpretation guidance:
- Coat is a dielectric microfacet lobe (GGX).
- Coat IOR is `pb_eta` if provided, else 1.5.
- Base BRDF uses the usual parameters (`Kd`, `Ks`, etc.) and `pb_base_*` overrides.

Defaults:
- If `pb_clearcoat` missing → `1.0`
- If `pb_clearcoat_roughness` missing → use `pb_roughness`
- If `pb_base_brdf` missing → `lambert`

---

## 6) Token reference (volumetric / translucency)

These parameters define a **homogeneous participating medium**, typically used for:
- fruit translucency
- milky liquids
- foggy glass
- ceramic/porcelain-like bulk scattering (approx.)

Tokens:

- `pb_medium homogeneous`
  - Enables a homogeneous medium inside a closed mesh.

- `pb_density <float>`
  - Multiplier applied to both $\sigma_a$ and $\sigma_s$.

- `pb_sigma_a r g b`
  - Absorption coefficient (per scene unit by default).

- `pb_sigma_s r g b`
  - Scattering coefficient (per scene unit by default).

- `pb_g <float>`
  - HG anisotropy in `[-1, 1]`.

Interpretation guidance:
- Convert RGB → spectrum.
- Compute:
  - $\sigma_a = pb\_density \cdot pb\_sigma\_a$
  - $\sigma_s = pb\_density \cdot pb\_sigma\_s$
  - $\sigma_t = \sigma_a + \sigma_s$
  - $albedo = \sigma_s / \sigma_t$ (component-wise; guard zeros)

Phase function:
- Henyey–Greenstein with parameter `pb_g`.

When to apply the medium:
- Only if `pb_medium homogeneous` is present.
- If `pb_thin=1`, **do not** treat as a volume by default; use thin-sheet absorption/scattering through `pb_thickness`.

Defaults:
- If `pb_density` is missing → `1.0`.
- If `pb_sigma_a` is missing → `(0,0,0)`.
- If `pb_sigma_s` is missing → `(0,0,0)`.
- If `pb_g` is missing → `0.0`.

---

## 7) How classic MTL fields map into the models

### 7.1 Color terms

- `Kd`: base/albedo for diffuse or base layer
- `Ks`: specular tint / F0 tint (legacy) and fallback for conductor reflectance
- `Ke`: emission (radiance-like; unit system is renderer-defined)

Implementation note for a spectral renderer:
- Treat `Kd/Ks/Ke/pb_sigma_*` RGB triples as **artist-space RGB** that must be converted to your spectral representation.
- Avoid guessing the transfer function implicitly. A practical convention is:
  - `map_Kd` textures are sRGB unless specified otherwise.
  - numeric triples in `.mtl` are linear (common in exporters), but this should be validated for your content pipeline.

### 7.2 Illum models

Classic `illum` is inconsistent across exporters. Suggested interpretation:
- Use it as a **hint** only.
- Prefer explicit `pb_brdf`.

However, for backwards compatibility:
- `illum 3/5` → reflection-enabled (mirror-ish)
- `illum 4/6/7` → refraction-enabled (glass-ish)

### 7.3 Opacity

- `d` (dissolve): 1 = opaque, 0 = invisible
- `Tr`: 0 = opaque, 1 = fully transparent (often the opposite of `d`)

Guidance:
- If both present, prefer `d` unless your exporter uses `Tr` consistently.
- If you support cutouts, `map_d` is the alpha mask.

## 7.4 Per-`pb_brdf` interpretation of classic MTL fields

This section is the **renderer contract**: given a chosen `pb_brdf`, how to use `Kd/Ks/Ns/Ni/d/Tr/Ke`.

### `pb_brdf lambert`

- Use `Kd` (or `map_Kd`) as diffuse albedo.
- Ignore `Ks` and `Ns`.
- Opacity may still come from `d/Tr/map_d` if you support cutouts.

### `pb_brdf emissive`

- Use `Ke` (or `map_Ke`) as emitted radiance.
- Simplest model: **no scattering** (black BSDF), purely emissive.
- If you want “emissive + visible surface”, combine emission with a diffuse BRDF.

### `pb_brdf dielectric`

Two common regimes:

1) **Glass / transmissive dielectric**
  - Typically: `Kd ≈ 0`, `illum` 4/6/7, `Ni` present, and/or transparency present.
  - Use Fresnel from `pb_eta`/`Ni`.
  - Use transmission controlled by `pb_transmission` if present, else default to 1.
  - If `pb_medium homogeneous` is present, apply the interior medium for refracted paths.

2) **Plastic / paint / non-transmissive dielectric**
  - Typically: `Kd > 0`, mild `Ks`, `illum 1/2`.
  - Use a microfacet dielectric specular lobe (Fresnel from `pb_eta`, usually ~1.5).
  - Use diffuse base from `Kd`.
  - `Ks` may be treated as a **specular intensity/tint multiplier** (legacy control). If you want stricter energy behavior, clamp it and/or ignore it when `pb_eta` is explicitly set.

Roughness:
- Prefer `pb_roughness`; else derive from `Ns`.

### `pb_brdf conductor`

- Prefer complex IOR from `pb_conductor_eta/k`.
- If complex IOR is missing, approximate conductor Fresnel using `Ks` as a reflectance/F0-like control (implementation-dependent).
- Ignore `Ni`.
- `Kd` is typically ignored for conductors.

Roughness:
- Prefer `pb_roughness` and anisotropy parameters; else derive from `Ns`.

### `pb_brdf clearcoat`

- Coat: dielectric microfacet lobe with IOR `pb_eta` (or default 1.5).
- Coat weight/roughness from `pb_clearcoat` / `pb_clearcoat_roughness`.
- Base BRDF from `pb_base_brdf` and its parameters.

Practical mapping suggestions:
- Wallpaper/paint/varnish: base is `lambert` using `Kd/map_Kd`, coat adds highlights.

### `pb_brdf fabric`

- Base: diffuse lobe using `Kd/map_Kd`.
- Sheen: grazing lobe controlled by `pb_sheen` and tinted via `pb_sheen_tint`.
- `Ks` may be ignored by default for fabric.

---

## 8) Recommended BSDF recipes by semantic category

These are *defaults* when explicit parameters are missing.

### 8.1 Translucency (milk, ceramic, water)

- **Milk / cloudy liquids:** dielectric + homogeneous medium
  - high scattering (`pb_sigma_s` large)
  - low absorption (`pb_sigma_a` small)
  - `pb_g` around 0.7–0.9

- **Water (clear):** dielectric, no medium or extremely low absorption

- **Ceramic/porcelain:** often looks like a dielectric with *some* subsurface scattering.
  - Use medium only if you want that look; otherwise treat as dielectric with higher roughness.

### 8.2 Metals (including glossy)

- Use `pb_brdf conductor`.
- For brushed metals, use anisotropy:
  - set `pb_anisotropy` and/or `pb_roughness_x/y`.

### 8.3 Fabric (simple)

- Use `pb_brdf fabric`.
- Implement as:
  - diffuse lobe (Lambert or Oren–Nayar if you choose)
  - plus a sheen lobe (grazing highlight) controlled by `pb_sheen`.

### 8.4 Wallpaper / painted walls

- Use `pb_brdf clearcoat` with low coat weight and moderately rough coat.
- Base = lambert with the wall color/texture.

### 8.5 Wood

- **Natural wood:** clearcoat/varnish is common.
  - coat roughness moderate; base diffuse carries texture.

- **Painted wood:** stronger clearcoat and smoother coat.

### 8.6 Stone

- Usually dielectric with high roughness.
- If you later add normal/bump support, stone benefits greatly.

---

## 9) Renderer integration checklist

When you implement this in the renderer later:

1. **Parse:** store both classic MTL fields and `pb_*` overrides.
2. **Normalize:** apply defaulting/derivation rules.
3. **Convert colors:** RGB → spectrum consistently.
4. **Pick BRDF:** from `pb_brdf` (or derived).
5. **Layering:** if `clearcoat`, build layered model.
6. **Media:** if `pb_medium homogeneous`, attach interior medium.
7. **Units:** apply `pb_meters_per_unit` conversion for volumetric coefficients.
8. **Energy sanity:** clamp extreme values; avoid NaNs.

---

## 10) Appendix: Minimal examples

### 10.1 Subsurface fruit

```
newmtl Apple
Kd 1 1 1
map_Kd textures\\apple.jpg
pb_brdf dielectric
pb_semantic subsurface
pb_roughness 0.35
pb_medium homogeneous
pb_density 0.25
pb_sigma_s 25 25 25
pb_sigma_a 0.2 1.0 2.0
pb_g 0.8
```

### 10.2 Brushed metal

```
newmtl DullSteel
Kd 0.02 0.02 0.02
Ks 0.5 0.5 0.5
pb_brdf conductor
pb_semantic metal
pb_anisotropy 0.8
pb_roughness_x 0.12
pb_roughness_y 0.35
pb_conductor_eta 2.5 2.5 2.5
pb_conductor_k 3.0 3.0 3.0
```

### 10.3 Wallpaper

```
newmtl Walls
Kd 0.6 0.6 0.6
pb_brdf clearcoat
pb_semantic wallpaper
pb_eta 1.5
pb_clearcoat 0.2
pb_clearcoat_roughness 0.35
pb_base_brdf lambert
pb_base_roughness 1.0
```
