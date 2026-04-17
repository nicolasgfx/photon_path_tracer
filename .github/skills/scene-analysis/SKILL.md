---
name: scene-analysis
description: 'Scene loading, PBRT/OBJ parsing, material mapping, scene classification, and auto-tuning parameter recommendation. Use when: loading scenes, fixing parsing bugs, adjusting scene classification, tuning emitter detection, modifying SceneProfile, or running ppt_analyze for pre-render analysis.'
---

# Scene Analysis

The `ppt_analyze` executable and shared scene loading infrastructure.

## Source Map

| Location | Files | Purpose |
|----------|-------|---------|
| `src/analyze/` | analyze_main.cpp, scene_analyzer.h, scene_profile_applicator.h, analyze_json_output.h | ppt_analyze entry point, analysis engine, config mapping, JSON output |
| `src/scene/` | scene.h, triangle.h, material.h, texture.h, envmap.h, medium.h | Shared scene representation |
| `src/scene/pbrt/` | pbrt_loader.h/.cpp, pbrt_parser.h/.cpp, pbrt_material_mapper.h/.cpp, ply_reader.h/.cpp | PBRT v4 parser and loader |
| `src/scene/` | obj_loader.h/.cpp | Wavefront OBJ/MTL loader |
| `src/core/` | scene_profile.h | SceneProfile struct (shared by all exes) |

## Tool: ppt_analyze

```
ppt_analyze <scene.pbrt|scene.obj>
```

Loads scene, runs analysis, prints JSON to stdout:
```json
{
  "scene_metrics": { "num_triangles", "geometry_complexity", "dominant_lighting", ... },
  "recommended_config": { "max_bounces", "photon_budget", "guide_fraction", ... }
}
```

## Scene Classification

### Enums

| Enum | Values | Driven by |
|------|--------|-----------|
| `LightingType` | Envmap, LargeArea, SmallPoint, Mixed | envmap presence, emitter_size_ratio |
| `EmitterDistribution` | Uniform, HighVariance, SingleDominant | max/min emitter area ratio |
| `GeometryComplexity` | Simple (<10K), Moderate, Complex, Dense (>5M) | triangle count |
| `EnvironmentType` | None, Uniform, HighDynamic (DR>100), Interior | envmap dynamic range, portals |

### Analysis Pipeline

1. **Geometry**: Count triangles, instances; classify complexity; detect degenerate/open geometry
2. **Environment**: Load envmap, compute dynamic range (max/median luminance), classify type
3. **Emitters**: Count emissive triangles, compute size ratio vs scene bbox, classify distribution; accumulate per-emitter `luminance(Le) × area` → `total_emissive_flux`, `max_emitter_radiance`
4. **Emitter grouping**: Group emissive triangles by material → per-light centroid, normal, flux. Shared by visibility probe and coupling estimation.
5. **Visibility probe**: CPU shadow rays from 512 surface probes toward emitter centroids → `emitter_direct_visibility`, `mostly_indirect_emitters`
6. **Materials**: Census of 9 material types; count delta materials (mirror+glass); avg roughness
7. **Caustics**: `has_glass && num_emitters > 0` → `has_caustic_paths`
8. **Delta surface geometry**: Count delta triangles, compute `delta_area_fraction`, set `caustic_geometry_favorable` (area > 0.001)
9. **Emitter-to-delta coupling**: Approximate solid angle of delta surfaces as seen from each emitter centroid: `Σ(delta_area × |cos_angle| / dist²) / (2π)`. Flux-weighted average across emitters → `emitter_delta_coupling` [0,1]. `caustic_difficulty = clamp(0.10 / coupling, 1, 100)` serves as budget multiplier. Uses reservoir subsampling for >1000 delta tris.
10. **Classification**: Combine above into `LightingType`
11. **Convergence hints**: Map classification → recommended render parameters

### Caustic Tuning

The caustic system uses **direction biasing** in the GPU kernel: 50% of photons emit via cosine-weighted hemisphere (standard), 50% aim directly at delta surfaces (using the delta CDF). This dramatically improves coupling for scenes where emitters don't directly face glass objects (e.g., living room with small glass decorations vs. Veach benchmark).

Budget and splat clamp are driven by the coupling metric:

| Metric | Description | Formula |
|--------|-------------|---------|
| `emitter_delta_coupling` | Flux-weighted solid angle fraction of delta surfaces | `Σ(A_delta × |cos| / d²) / (2π)`, averaged over emitter groups |
| `caustic_difficulty` | Budget multiplier (1 = easy, 100 = very hard) | `clamp(0.10 / max(coupling, 0.001), 1, 100)` |
| `recommended_caustic_photons_per_frame` | Photon budget per frame | `256K × min(difficulty, 8) × (2 if indirect)`, clamped to [256K, 2M] |
| `recommended_caustic_max_splat_luminance` | Per-splat clamp | `clamp(total_flux × 2, 50, 500)` |

### SceneProfile → RenderConfig Mapping

`apply_scene_profile()` writes recommended values to `RenderConfig`, but ONLY for fields still at their compile-time defaults (respects override cascade).

| Scene Characteristic | Parameter | Logic |
|---------------------|-----------|-------|
| has_glass | max_bounces | 12 (vs default 8) |
| has_translucent | max_bounces | 16 |
| has_caustic_paths | photon_budget | 4M; caustic_budget = 1-2M |
| SmallPoint lighting | photon_budget | 2M |
| Envmap lighting | guide_training_iters | 15 (vs default 10) |
| LargeArea lighting | guide_fraction | 0.3 (vs default 0.5) |
| Scene diagonal | gather_radius | diagonal × 0.005 |
| caustic_difficulty | caustic_photons_per_frame | 256K × min(difficulty, 8), clamped [256K, 2M] |
| total_emissive_flux | caustic_max_splat_luminance | clamp(flux × 2, 50, 500) |
| num_emitters ≤ 1 | light_tree_enabled | false (flat CDF is O(1) for 1 emitter) |

## PBRT v4 Parsing

Supported shapes: `plymesh`, `trianglemesh`, `sphere` (tessellated), `disk`, `bilinearmesh`
Material mapping: PBRT→v5 type conversion (e.g., `glass`→Glass, `conductor`→GlossyMetal, `coateddiffuse`→Clearcoat)
Textures: TGA, PNG, EXR float; image textures referenced by materials
