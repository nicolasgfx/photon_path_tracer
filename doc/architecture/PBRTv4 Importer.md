# PBRTv4 Importer — Architecture Reference

The PBRTv4 importer loads scenes authored in the [PBRT v4 file format](https://pbrt.org/fileformat-v4) into the renderer's internal `Scene` representation. It is implemented as a four-stage pipeline across six source files in `src/scene/pbrt/`.

---

## File Overview

| File | Role |
|---|---|
| `pbrt_parser.h` | Data structures (IR) and parser class declaration |
| `pbrt_parser.cpp` | Tokenizer, parameter parser, recursive-descent dispatch |
| `pbrt_loader.h/.cpp` | Orchestrates loading: shapes → triangles, lights, camera, media |
| `pbrt_material_mapper.h/.cpp` | Maps PBRT materials/textures to renderer `Material` structs |
| `ply_reader.h/.cpp` | Binary + ASCII PLY mesh loader with gzip support |

---

## Pipeline

```
.pbrt file
    │
    ▼
┌──────────────────────────────────────────────┐
│  1. PbrtParser::parse_file()                 │
│     Tokenize → parse params → dispatch       │
│     directives → build PbrtScene IR          │
└─────────────────────┬────────────────────────┘
                      │ PbrtScene
                      ▼
┌──────────────────────────────────────────────┐
│  2. MaterialMapper::map_all_named_materials()│
│     Walk textures + materials → Scene.mats   │
└─────────────────────┬────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│  3. load_pbrt() orchestration                │
│     • process_media()                        │
│     • process_shape() for each shape         │
│     • Instance template loading              │
│     • extract_lights() → emissive + portals   │
│     • extract_camera() → Scene camera fields │
│     • write_saved_camera()                   │
│     • finalize_pb_materials()                │
└─────────────────────┬────────────────────────┘
                      │
                      ▼
                  Scene struct
```

---

## Stage 1 — Parser (`pbrt_parser.h` / `pbrt_parser.cpp`)

### Intermediate Representation

All parsed data is stored in `PbrtScene`, a plain-data intermediate representation that decouples parsing from scene construction.

**Core Structs:**

| Struct | Key Fields |
|---|---|
| `Mat4` | `double m[4][4]` row-major. Static factories: `identity()`, `translate()`, `scale()`, `rotate()`, `from_column_major()`. Operators: `operator*`, `inverse()`. |
| `Param` | `type` (string), `name`, `floats`, `ints`, `strings`, `boolean` — polymorphic parameter storage |
| `PbrtTextureDecl` | `tex_name`, `tex_type` ("spectrum"/"float"), `tex_class` ("imagemap"/"scale"/"constant"/"mix"/...), `params` |
| `PbrtMaterial` | `mat_name`, `mat_type`, `params` |
| `PbrtShape` | `shape_type`, `params`, `material_name`, `inline_mat`, `transform`, `has_area_light`, `area_light_params`, `reverse_orientation`, `medium_interior` |
| `PbrtLight` | `light_type`, `params`, `transform` |
| `PbrtCamera` | `cam_type`, `params`, `has_lookat`, `eye[3]`, `target[3]`, `up[3]`, `pre_transform` |
| `PbrtFilm` | `film_type`, `params` |
| `PbrtObjectTemplate` | `name`, `shapes` (vector) |
| `PbrtInstanceRef` | `template_name`, `transform` |
| `PbrtMediumDecl` | `name`, `type`, `params` |

**`PbrtScene` aggregates:**

| Field | Type |
|---|---|
| `camera` | `PbrtCamera` |
| `film` | `PbrtFilm` |
| `global_transform` | `Mat4` (pre-WorldBegin accumulated transform) |
| `source_dir` | `std::string` (directory of root `.pbrt` file) |
| `lights` | `vector<PbrtLight>` |
| `textures` | `unordered_map<string, PbrtTextureDecl>` |
| `named_materials` | `unordered_map<string, PbrtMaterial>` |
| `shapes` | `vector<PbrtShape>` |
| `instance_refs` | `vector<PbrtInstanceRef>` |
| `object_templates` | `unordered_map<string, PbrtObjectTemplate>` |
| `named_media` | `unordered_map<string, PbrtMediumDecl>` |

**Free param helpers:** `get_param()`, `get_float()`, `get_int()`, `get_string()`, `get_bool()`, `get_rgb()`, `get_texture_ref()`, `get_param_type()`.

### Tokenizer

`PbrtParser::tokenize()` splits input text into tokens:
- Skips whitespace and `#`-comments
- Quoted strings `"..."` kept intact (including quotes)
- `[` and `]` as individual tokens
- Bare words/numbers delimited by whitespace, quotes, brackets, or `#`

Helpers: `is_number_token()` validates numeric strings; `unquote()` strips outer quotes.

### Parameter Parser

`parse_params()` recognises `"type name" [ values... ]` pairs.

**Strict v4 parameter types (`PBRT_PARAM_TYPES`):**

```
float, integer, string, bool, boolean, rgb, spectrum, blackbody,
point, point2, point3, vector, vector2, vector3, normal, texture, color
```

**Type coercion (`coerce_param`):**

| Type | Storage |
|---|---|
| `float` | `→ p.floats` |
| `integer` | `→ p.ints` |
| `string`, `texture` | `→ p.strings` (unquoted) |
| `rgb`, `color` | `→ p.floats` |
| `spectrum` | Quoted first token → `p.strings` (spectrum file); else `→ p.floats` (wavelength/value pairs) |
| `blackbody` | `→ p.floats` |
| `bool`, `boolean` | `→ p.boolean` |
| `point`/`point3`/`vector`/`vector3`/`normal` | `→ p.floats` |
| `point2`, `vector2` | `→ p.floats` |
| Unknown | Try `stod` → floats; else → strings |

Values may be bracketed (`[ ... ]`) or single unbracketed tokens. The parser distinguishes trailing bare strings from the start of a new `"type name"` parameter.

### Dispatch

`dispatch()` performs a linear scan of tokens, matching directive keywords:

| Directive | Behaviour |
|---|---|
| **`Film`** | Stores type + params in `scene_.film` |
| **`Camera`** | Stores type + params + `pre_transform` in `scene_.camera` |
| **`Sampler`** | Parsed and discarded |
| **`Integrator`** | Parsed and discarded |
| **`PixelFilter`** | Parsed and discarded |
| **`ColorSpace`** | Parsed and discarded |
| **`Option`** | Parsed and discarded |
| **`LookAt`** | 9 doubles → `camera.eye`, `camera.target`, `camera.up`, sets `has_lookat` |
| **`WorldBegin`** | Stores accumulated `pre_world_transform` as `global_transform`, resets CTM to identity, sets `in_world_` |
| **`Scale`** | Multiplies scale matrix into CTM (or `pre_world_transform` if pre-world) |
| **`Translate`** | Multiplies translation into CTM |
| **`Rotate`** | Axis-angle rotation multiplied into CTM |
| **`ConcatTransform`** | 16 column-major doubles → `from_column_major()` → multiplied into CTM |
| **`Transform`** | 16 column-major doubles → **replaces** CTM |
| **`Identity`** | Resets CTM to identity |
| **`TransformBegin`** | Pushes CTM onto `transform_stack_` |
| **`TransformEnd`** | Pops CTM from `transform_stack_` |
| **`AttributeBegin`** | Pushes CTM + `GraphicsState` (material, area light, reverse orientation) |
| **`AttributeEnd`** | Pops CTM + `GraphicsState` |
| **`ObjectBegin`** | Creates `PbrtObjectTemplate`, sets `in_object_` |
| **`ObjectEnd`** | Clears `in_object_` |
| **`ObjectInstance`** | Emissive templates → flattens shapes with combined transform. Non-emissive → `PbrtInstanceRef` with current CTM |
| **`Include`** | Resolves relative path → recursive `parse_file_recursive()` |
| **`Texture`** | Reads name, return-type, class + params → stored in `scene_.textures` |
| **`MakeNamedMaterial`** | Reads name + params (extracts `"type"` param) → stored in `scene_.named_materials` |
| **`NamedMaterial`** | Sets `current_material_` reference, clears `current_inline_mat_` |
| **`Material`** | Creates inline `PbrtMaterial` → `current_inline_mat_`, clears `current_material_` |
| **`LightSource`** | Creates `PbrtLight` with current transform → appended to `scene_.lights` |
| **`AreaLightSource`** | Sets `has_area_light_`, stores params for subsequent shapes |
| **`Shape`** | Creates `PbrtShape` with full current state; routed to `in_object_` template or `scene_.shapes` |
| **`ReverseOrientation`** | Toggles `reverse_orientation_` |
| **`MakeNamedMedium`** | Reads name + params → stored in `scene_.named_media` |
| **`MediumInterface`** | Reads interior + exterior medium names |
| Unknown | Silently advances past the token |

**Parser State:**

`PbrtParser` maintains:
- **Transform state**: `current_transform_` (CTM), `pre_world_transform_`, `transform_stack_`
- **Material state**: `current_material_` (named ref), `current_inline_mat_` (inline Material)
- **Area light state**: `has_area_light_`, `current_area_light_params_`
- **Medium state**: `current_medium_interior_`, `current_medium_exterior_`
- **Flags**: `reverse_orientation_`, `in_world_`, `in_object_`
- **Graphics stack**: `graphics_stack_` saves/restores material + area light + reverse orientation on `AttributeBegin`/`End`

---

## Stage 2 — Material Mapping (`pbrt_material_mapper.h/.cpp`)

### Class `MaterialMapper`

**Public methods:**
- `map_all_named_materials()` — iterates all `PbrtScene::named_materials`, calls `map_one_material()` for each
- `resolve_shape_material(shape)` — priority: (1) `shape.material_name` → lookup or map on-demand; (2) `shape.inline_mat` → synthetic name `"_inline_{type}_{counter}"`; (3) fallback `"__default__"` (grey Lambert, Kd=0.5)
- `create_emissive_material(base_name, params)` — deduplicates by name; handles blackbody `L` → `blackbody_to_rgb()`, RGB `L` × `scale`, or white fallback

### Material Type Dispatch

`map_one_material()` dispatches on `mat_type`:

| PBRT Type | Renderer BRDF | Mapper Method | Notes |
|---|---|---|---|
| `coateddiffuse` | `Clearcoat` | `map_coated_diffuse()` | Pearl detection: smooth coat + bright base → Dielectric with synthetic scattering medium |
| `diffuse` | `Lambert` | `map_diffuse()` | `reflectance` → Kd |
| `dielectric` | `Dielectric` | `map_dielectric()` | Smooth (roughness ≤ 0.05) → Glass; rough → GlossyDielectric |
| `thindielectric` | `Dielectric` (thin) | `map_thin_dielectric()` | `pb_thin=true`, full transmission |
| `conductor` | `Conductor` | `map_conductor()` | Named spectrum lookup or direct RGB η/k |
| `coatedconductor` | `Conductor` (GlossyMetal) | `map_coated_conductor()` | Sacrifices thin coat for correct metallic appearance |
| `measured` | `Clearcoat` | `map_measured()` | Approximated: white Lambert + clearcoat (roughness=0.05) |
| `diffusetransmission` | `Lambert` (thin) | `map_diffuse_transmission()` | `pb_thin=true`, transmission = avg(transmittance) × scale |
| `subsurface` | `Lambert` + Subsurface semantic | `map_subsurface()` | Fallback Kd = skin-like {0.8, 0.6, 0.5} |
| `mix` | Delegates to sub-material | `map_mix()` | Selects sub-material by `amount` threshold (>0.5 → material[1]) |
| `hair` | `Lambert` (tinted) | `map_hair()` | Reflectance, sigma_a → exp(-σa), or melanin model |
| `interface` | `Dielectric` (invisible) | `map_interface()` | IOR=1.0, transmission=1, opacity=0 — medium boundary |
| Unknown | `Lambert` (warning) | — | Fallback with stderr warning |

### Conductor η/k Resolution

`resolve_conductor_eta_k()` resolves complex conductance values:
1. Default: aluminum
2. Named spectrum strings (e.g. `"metal-Cu-eta"`) → fuzzy match against 13 built-in `CONDUCTOR_PRESETS` (Al, Cu, Au, Ag, Fe, Ti, Cr, W, Ni, Pt, Co, Pd, Zn)
3. Direct RGB (3-float arrays)
4. Separate η and k params with `-k` suffix matching

### Roughness Conversion

`pbrt_roughness_to_ours(roughness, remap)`:
- `remap=true` (PBRT default): `roughness^0.25`
- `remap=false`: `sqrt(roughness)`

### Texture Resolution

**`resolve_texture_path(tex_name)`** — walks the PBRT texture graph:
- `imagemap` → returns `filename` param
- `scale` → follows `tex`/`scale` children
- `mix` / `directionmix` → follows `tex1`, `tex2` children
- Others (constant, etc.) → empty string

**`resolve_texture(tex_name)`** — full resolution pipeline:
1. Call `resolve_texture_path()` for image path
2. If no image path → try **procedural baking** via `bake_procedural_texture()`
3. If image path found → build absolute path from `source_dir_`
4. Deduplicate by path
5. Load: `.exr` via TinyEXR (`LoadEXR()`), all others via `stbi_load()` (forced RGBA)
6. Store as `Texture` in `scene_.textures`

### Procedural Texture Baking

`bake_procedural_texture()` rasterises procedural textures to 256×256 RGBA float:

| Texture Class | Method |
|---|---|
| `constant` | Flat fill from `value` (float or RGB) |
| `checkerboard` | 2-color check pattern; params: `tex1`, `tex2` (RGB or scalar), `uscale` (frequency) |
| `bilerp` | Bilinear interpolation of 4 corner colors: `v00`, `v01`, `v10`, `v11` |
| `dots` | Polka dots (4×4 grid); params: `inside`, `outside` RGB |
| `fbm`, `wrinkled`, `windy` | Hash-based value noise approximation |
| `marble` | `sin(6u + 5·noise)` marble veining pattern |

---

## Stage 3 — Scene Loading (`pbrt_loader.cpp`)

### Public API

```cpp
bool load_pbrt(const std::string& filepath, Scene& scene);
```

### Loading Pipeline

The `load_pbrt()` function executes these steps in order:

1. **Parse** — `PbrtParser::parse_file()` → `PbrtScene`
2. **Map materials** — `MaterialMapper::map_all_named_materials()`
3. **Process media** — `process_media()` → `HomogeneousMedium` entries
4. **Process shapes** — `process_shape()` per shape → flat triangle soup
   - 4b: Process instanced templates → `MeshDescriptor` + `InstanceDescriptor` entries
5. **Process lights** — `extract_lights()` → point/spot emissive spheres, portals (envmap skipped)
6. **Extract camera** — `extract_camera()` → `Scene.pbrt_cam_*` fields + `saved_camera.json`
7. **Finalize materials** — `finalize_pb_materials()` (external, in `obj_loader.cpp`)
8. **Report** — final triangle/material/texture counts

### Transform Helpers

| Function | Purpose |
|---|---|
| `mat4_mul_point(m, x, y, z)` | 4×4 × point with perspective divide |
| `transform_point(m, p)` | Transform a position |
| `transform_normal(m, n)` | Normal transform via cofactor matrix; negates if det < 0 |
| `det3x3(m)` | Determinant of upper-left 3×3 |
| `blackbody_to_rgb(temp_K, rgb)` | McCamy blackbody-to-RGB approximation |

### Shape Processing

`process_shape()` converts each `PbrtShape` into triangles appended to `scene.triangles`.

**Shape types handled:**

| Type String | Handling | Key Params |
|---|---|---|
| `plymesh` | Load via `load_ply()`, compute normals if missing | `filename` |
| `trianglemesh` | Inline vertex data | `P`, `N`, `uv`/`st`, `indices` |
| `sphere` | Tessellated UV-sphere (32θ × 64φ) | `radius` (default 1.0) |
| `disk` | Fan tessellation (64 segments) | `radius`, `height` |
| `bilinearmesh` | 4-vertex quads → 2 triangles each | `P`, `indices`, `uv`, `N` |
| Other | Warning, skipped | — |

**Shape processing pipeline per shape:**

1. **Alpha=0 check** — if fully transparent with area light → creates power-scaled small emissive sphere proxy at shape translation, then skips
2. **Material resolution** — `mapper.resolve_shape_material(shape)`
3. **Area light merge** — if `has_area_light`, creates emissive material and merges base material's textures/Kd into it
4. **Winding** — `flip_winding = (det3x3 < 0) XOR reverse_orientation`
5. **Geometry collection** — type-specific loading (see table above)
6. **Normal generation** — if missing, computed as per-face geometric normals after transform
7. **World transform** — positions transformed by `shape.transform`; normals by cofactor matrix
8. **Triangle construction** — vertex/normal/UV assignment with winding flip; material_id assigned

**Statistics tracking (`ShapeStats`):** plymesh, trianglemesh, sphere, disk, bilinearmesh, skipped, tri_count.

### Light Processing

`extract_lights()` processes `PbrtScene::lights` → `LightInfo`:

| Type | Handling |
|---|---|
| `infinite` | Skipped (envmap not yet supported). Portal quads are still extracted for directed photon emission. |
| `point` | Position from transform column 3. Params: `I` (RGB), `scale`. → Small emissive sphere (radius 0.01). |
| `spot` | Treated as point light (simplified). Same params as `point`. |
| `distant` | Direction computed from `from`/`to` params. Logged and skipped. |

**Portal handling:** Portal quads (12 floats = 4 vertices each, transformed from light-local to world space) are converted to emissive triangles with flipped winding (inward normals) and `opacity=0` for directed photon emission through windows.

**Point light geometry** (`add_point_light_geometry()`): Creates a small 8θ×16φ tessellated emissive sphere at the light position with `MaterialType::Emissive` and `PbBrdf::Emissive`.

### Camera Extraction

`extract_camera()` returns `CameraInfo` (position, look_at, up, fov, flip_x):

- **With `LookAt`**: Uses eye/target/up directly. Detects pre-world mirror transforms (det < 0 → `flip_x`)
- **Without `LookAt`**: Inverts `pre_transform` (camera-from-world → camera-to-world); extracts position from column 3, forward from column 2 (+Z, PBRT left-handed convention), up from column 1
- **`fov`** from `"fov"` param (default 90)
- Writes `saved_camera.json` (won't overwrite existing files)

### Medium Processing

`process_media()` handles `PbrtScene::named_media` → `HomogeneousMedium` entries:

| Type | Handling |
|---|---|
| `homogeneous` | Params: `sigma_a` (RGB), `sigma_s` (RGB), `scale`, `g` (asymmetry). Creates `HomogeneousMedium` with σ_a, σ_s, σ_t = σ_a + σ_s. |
| Other | Warning, skipped |

### Instancing Pipeline

1. **Emissive templates** — flattened at parse time: shapes appended to `scene_.shapes` with combined transforms (cannot be hardware-instanced since emissive geometry needs unique materials)
2. **Non-emissive templates** — each unique `PbrtObjectTemplate` is loaded once → `MeshDescriptor`. Each `PbrtInstanceRef` maps to an `InstanceDescriptor` with a 3×4 row-major float transform
3. **Medium linkage** — tracks `(mat_id, medium_name)` pairs. Clones materials when the same named material is used with different media. Sets `pb_medium_enabled=true`, promotes `Glass` → `Translucent`

### Material-Medium Linkage

When a shape has `medium_interior` set:
1. Looks up the medium by name in `medium_name_to_id`
2. Checks if the material already has a different medium → clones the material
3. Sets `medium_id`, `pb_medium_enabled = true`
4. Glass materials are promoted to Translucent (since they now participate in volumetric transport)

---

## Stage 4 — PLY Reader (`ply_reader.h/.cpp`)

### Supported Formats

- `binary_little_endian 1.0` (dominant for PBRT scenes)
- `ascii 1.0`
- Transparent gzip decompression (`.gz` extension via RFC 1952 header parsing + `stbi_zlib_decode_noheader_malloc`)

### Type System

PLY types: `float`/`float32`, `double`/`float64`, `char`/`int8`, `uchar`/`uint8`, `short`/`int16`, `ushort`/`uint16`, `int`/`int32`, `uint`/`uint32`, plus `list` for face indices.

### Vertex Property Recognition

| Property Names | Maps To |
|---|---|
| `x`, `y`, `z` | positions (required) |
| `nx`, `ny`, `nz` | normals (optional) |
| `u`/`s`/`texture_u`, `v`/`t`/`texture_v` | texcoords (optional) |

### Reading Strategy

- **Binary**: Computes vertex stride and per-property offsets; reads all vertex data in a single bulk `file.read()`; reads faces one-by-one (count → indices → fan triangulation for n-gons)
- **ASCII**: Reads elements in header-declared order; parses floats/ints per-line; fan triangulation for polygons

### Normal Computation

`compute_face_normals()`: Accumulates area-weighted face normals per vertex (via `cross(e1, e2)`, magnitude = 2×area). Normalises; zero-length normals fall back to `(0, 0, 1)`.

---

## Known Limitations

### Unsupported Directives
- `ActiveTransform` (Start/End/All) — parsed but not tracked
- `CoordinateSystem` / `CoordSysTransform` — not implemented
- `Accelerator` — ignored (renderer uses its own BVH)
- `ImportInstancedGeometry` — not handled

### Shape Gaps
- `cylinder`, `curve` (Bézier/B-spline), `loopsubdiv` — not yet tessellated; will produce "Unsupported shape type" warnings

### Light Gaps
- `distant` — logged and skipped
- `projection`, `goniometric` — not handled; silently ignored

### Camera Gaps
- Only `perspective` cameras have meaningful extraction
- `orthographic`, `spherical`, `realistic` — type is stored but falls through to the same perspective extraction path

### Medium Gaps
- Only `homogeneous` media are loaded
- `uniformgrid`, `rgbgrid`, `cloud`, `nanovdb` — skipped with a warning

### Material Approximations
- `measured` — approximated as clearcoated white Lambert (no BRDF fitting)
- `hair` — approximated as tinted Lambert (no anisotropic fibre scattering)
- `interface` — modelled as invisible dielectric (IOR=1.0)
- `mix` — selects one sub-material by `amount` threshold rather than blending
- `subsurface` — approximated as Lambert (no actual SSS evaluation)
- `coatedconductor` — mapped as GlossyMetal; thin coat highlight not represented

### Texture Approximations
- Procedural textures are baked to 256×256 at load time (no runtime evaluation)
- `fbm`/`wrinkled`/`windy` use simplified hash-based noise, not full Perlin noise
- Texture `scale` chains and `mix`/`directionmix` graphs are walked but only the first image leaf is resolved
