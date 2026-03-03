# Dense Grid Rewrite — Hash Grid Removal

## Goal

Replace the Teschner hash grid photon lookup with a collision-free **dense 3D grid**.
At each camera bounce the new logic is:

1. **Pick a random photon** in the dense grid cell containing the hit point → bounce the ray in that photon's incoming direction (`wi`).
2. **Sample the BSDF** as usual.

Two-strategy MIS (photon-guided vs BSDF) replaces the old kNN histogram guide.
All kNN gather, Fibonacci-bin histograms, density estimation, caustic additive,
final gather, cell analysis, and the hash grid itself are **removed entirely**.
The volume photon system (volume grid, volume kNN, volume guide) is **kept unchanged**.

---

## Reference implementation

`tools/photon_map_analysis/main.cpp` contains a working CPU `DenseGrid` struct
that serves as the blueprint.  The key data structure:

```cpp
struct DenseGrid {
    float min_x, min_y, min_z;
    float cell_size;
    int dim_x, dim_y, dim_z;
    std::vector<uint32_t> sorted_indices; // photon indices sorted by cell
    std::vector<uint32_t> cell_start;     // [total_cells] first photon index
    std::vector<uint32_t> cell_end;       // [total_cells] one-past-last
};
// flat_index = ix + iy * dim_x + iz * dim_x * dim_y
```

Cell size = `DEFAULT_CAUSTIC_RADIUS` (0.025 m).  For salle_de_bain this gives
dims ≈ 27×15×37 = ~15 K cells, ~120 KB memory — trivial on a GPU.

---

## Files to modify (in order)

### Step 1 — `src/core/config.h`

**Add** (after the `DEFAULT_USE_DENSE_GRID` line, around L392):

```cpp
constexpr float DENSE_GRID_CELL_SIZE       = DEFAULT_CAUSTIC_RADIUS;  // 0.025 m
constexpr float DENSE_GRID_GUIDE_FRACTION  = 0.5f;                   // photon vs BSDF mix
```

**Remove later** (after all references are gone):
- `DEFAULT_KNN_K` (L258)
- `CELL_CACHE_TABLE_SIZE` (L262)
- `PHOTON_BIN_COUNT` / `MAX_PHOTON_BIN_COUNT` (L265–L266)
- `ADAPTIVE_RADIUS_*` constants (L268–L270)
- `CAUSTIC_*` adaptive emission constants (L273–L276) — only if caustic system is also removed

Items to keep:
- `DEFAULT_GATHER_RADIUS` / `DEFAULT_CAUSTIC_RADIUS` — still used for volume gather and dense cell size
- `DEFAULT_GUIDE_FRACTION` — reused as the photon vs BSDF MIS weight
- `DEFAULT_SURFACE_TAU` — keep for normal/plane filtering in the dense cell lookup

---

### Step 2 — `src/optix/launch_params.h`

**Remove** the surface hash grid fields (L131–L135):
```cpp
// DELETE these 5 lines:
uint32_t* grid_sorted_indices;
uint32_t* grid_cell_start;
uint32_t* grid_cell_end;
float     grid_cell_size;
uint32_t  grid_table_size;
```

**Remove** the per-cell analysis fields (L82–L85):
```cpp
// DELETE these 3 lines:
float* cell_guide_fraction;
float* cell_caustic_fraction;
float* cell_flux_density;
```

**Remove** (L243, in "Dense 3D cell-bin grid" section):
```cpp
// DELETE:
int  photon_bin_count;
```

**Remove** (L136):
```cpp
// DELETE:
uint8_t* photon_bin_idx;
```

**Add** (in place of the removed hash grid block) the dense grid fields:
```cpp
// ── Dense 3D grid (surface photon lookup) ────────────────────────
uint32_t* dense_sorted_indices;     // [num_photons] sorted by cell
uint32_t* dense_cell_start;         // [total_cells]
uint32_t* dense_cell_end;           // [total_cells]
int       dense_valid;              // 1 = built, 0 = not available
float     dense_min_x, dense_min_y, dense_min_z;  // AABB min
float     dense_cell_size;          // cell edge length
int       dense_dim_x, dense_dim_y, dense_dim_z;  // grid resolution
```

Keep all volume grid fields (L245–L258) **unchanged**.

---

### Step 3 — New file: `src/photon/dense_grid.h`

Pure-header, host-only dense grid builder.  Mirrors the analysis tool's
`DenseGrid` but uses a `DenseGridData` struct compatible with GPU upload.

```cpp
#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>

struct DenseGridData {
    float min_x, min_y, min_z;
    float cell_size;
    int dim_x, dim_y, dim_z;
    std::vector<uint32_t> sorted_indices;
    std::vector<uint32_t> cell_start;
    std::vector<uint32_t> cell_end;

    int total_cells() const { return dim_x * dim_y * dim_z; }

    int cell_index(float x, float y, float z) const {
        int ix = (int)std::floor((x - min_x) / cell_size);
        int iy = (int)std::floor((y - min_y) / cell_size);
        int iz = (int)std::floor((z - min_z) / cell_size);
        ix = std::max(0, std::min(ix, dim_x - 1));
        iy = std::max(0, std::min(iy, dim_y - 1));
        iz = std::max(0, std::min(iz, dim_z - 1));
        return ix + iy * dim_x + iz * dim_x * dim_y;
    }
};

/// Build a dense grid from photon positions.
/// cell_size should be DENSE_GRID_CELL_SIZE (0.025 m).
inline DenseGridData build_dense_grid(
    const float* pos_x, const float* pos_y, const float* pos_z,
    int n, float cell_size)
{
    DenseGridData g;
    g.cell_size = cell_size;

    // Compute AABB with half-cell padding
    float pad = cell_size * 0.5f;
    g.min_x = g.min_y = g.min_z =  1e30f;
    float max_x = -1e30f, max_y = -1e30f, max_z = -1e30f;
    for (int i = 0; i < n; ++i) {
        if (pos_x[i] < g.min_x) g.min_x = pos_x[i];
        if (pos_y[i] < g.min_y) g.min_y = pos_y[i];
        if (pos_z[i] < g.min_z) g.min_z = pos_z[i];
        if (pos_x[i] > max_x) max_x = pos_x[i];
        if (pos_y[i] > max_y) max_y = pos_y[i];
        if (pos_z[i] > max_z) max_z = pos_z[i];
    }
    g.min_x -= pad; g.min_y -= pad; g.min_z -= pad;
    max_x += pad; max_y += pad; max_z += pad;

    g.dim_x = std::max(1, (int)std::ceil((max_x - g.min_x) / cell_size));
    g.dim_y = std::max(1, (int)std::ceil((max_y - g.min_y) / cell_size));
    g.dim_z = std::max(1, (int)std::ceil((max_z - g.min_z) / cell_size));

    int total = g.total_cells();

    // Assign each photon to a cell, count per cell
    std::vector<uint32_t> cell_ids(n);
    std::vector<uint32_t> counts(total, 0);
    for (int i = 0; i < n; ++i) {
        cell_ids[i] = (uint32_t)g.cell_index(pos_x[i], pos_y[i], pos_z[i]);
        counts[cell_ids[i]]++;
    }

    // Prefix sum → cell_start
    g.cell_start.resize(total);
    g.cell_end.resize(total);
    uint32_t offset = 0;
    for (int c = 0; c < total; ++c) {
        g.cell_start[c] = offset;
        offset += counts[c];
        g.cell_end[c] = offset;
    }

    // Scatter photon indices into sorted order
    g.sorted_indices.resize(n);
    std::vector<uint32_t> write_pos = g.cell_start; // copy
    for (int i = 0; i < n; ++i) {
        uint32_t c = cell_ids[i];
        g.sorted_indices[write_pos[c]++] = (uint32_t)i;
    }

    return g;
}
```

---

### Step 4 — `src/optix/optix_guided.cuh`

**Remove entirely** the surface guide functions:
- `DevPhotonBinDirs` struct (L18–L48)
- `bin_solid_angle()` (L52–L54)
- `dev_cell_cache_index()` (L66–L76)
- `GuidedHistogram` struct (L80–L88)
- `dev_knn_guide_sample()` (L100–L194)
- `GuidedSample` struct (L206–L215)
- `dev_sample_guided_direction()` (L217–L260)
- `dev_guided_pdf()` (L267–L278)

**Keep** the volume guide functions (L283 onwards):
- `dev_vol_cell_grid_index()` (L285–L294)
- `dev_read_vol_cell_histogram()` (L296–L376)

Note: `DevPhotonBinDirs` and `bin_solid_angle()` are also used by the volume
guide code.  If the volume guide still references them, keep those two items
and only remove the surface-specific functions.  Check whether
`dev_read_vol_cell_histogram()` uses `GuidedHistogram` — if yes, keep that
struct too but rename the file header comment to say "volume guide only".

**Add** (at the top, before the volume section) the new dense grid lookup:
```cuda
// ── Dense grid cell lookup ──────────────────────────────────────────
__forceinline__ __device__
int dev_dense_cell_index(float3 pos) {
    int ix = (int)floorf((pos.x - params.dense_min_x) / params.dense_cell_size);
    int iy = (int)floorf((pos.y - params.dense_min_y) / params.dense_cell_size);
    int iz = (int)floorf((pos.z - params.dense_min_z) / params.dense_cell_size);
    ix = max(0, min(ix, params.dense_dim_x - 1));
    iy = max(0, min(iy, params.dense_dim_y - 1));
    iz = max(0, min(iz, params.dense_dim_z - 1));
    return ix + iy * params.dense_dim_x
             + iz * params.dense_dim_x * params.dense_dim_y;
}

// ── Pick a random photon from the dense cell at `pos` ───────────────
// Returns the photon index, or -1 if the cell is empty.
// Also applies normal gate and surface tau filter.
__forceinline__ __device__
int dev_random_photon_in_cell(float3 pos, float3 filter_normal, PCGRng& rng) {
    if (!params.dense_valid || params.num_photons == 0)
        return -1;

    int cell = dev_dense_cell_index(pos);
    uint32_t start = params.dense_cell_start[cell];
    uint32_t end   = params.dense_cell_end[cell];
    if (start >= end) return -1;

    // Try up to 8 random picks to find one passing the normal/tau gate
    for (int attempt = 0; attempt < 8; ++attempt) {
        uint32_t pick = start + (uint32_t)(rng.next_float() * (float)(end - start));
        if (pick >= end) pick = end - 1;
        uint32_t idx = params.dense_sorted_indices[pick];

        // Normal gate
        float3 photon_n = make_f3(
            params.photon_norm_x[idx],
            params.photon_norm_y[idx],
            params.photon_norm_z[idx]);
        if (dot(photon_n, filter_normal) <= 0.0f) continue;

        // Surface tau (plane-distance) gate
        float3 pp = make_f3(
            params.photon_pos_x[idx],
            params.photon_pos_y[idx],
            params.photon_pos_z[idx]);
        float d_plane = fabsf(dot(pos - pp, filter_normal));
        if (d_plane > DEFAULT_SURFACE_TAU) continue;

        // Wi direction gate: photon must arrive from upper hemisphere
        float3 wi = make_f3(
            params.photon_wi_x[idx],
            params.photon_wi_y[idx],
            params.photon_wi_z[idx]);
        if (dot(wi, filter_normal) <= 0.f) continue;

        return (int)idx;
    }
    return -1;
}
```

---

### Step 5 — `src/optix/optix_nee.cuh`

**Remove** these surface photon gather functions:
- `dev_estimate_photon_density()` (L46–L257) — terminal gather
- `dev_estimate_caustic_only()` (L271–L417) — caustic additive

**Keep** everything else:
- `dev_light_pdf()` (L7–L43) — emitter PDF for MIS
- `sample_triangle_dev()` (L422–L430) — triangle sampling
- All NEE functions: `dev_nee_direct()`, `dev_nee_dispatch()`, etc.
- `dev_estimate_volume_photon_density()` (L577–L694) — volume gather
- `dev_nee_volume_scatter()` (L715+) — volume NEE

---

### Step 6 — `src/optix/optix_path_trace_v3.cuh`

This is the main rewrite.  The bounce loop (inside `full_path_trace_v3()`)
currently does:

```
L376–L384: kNN guide histogram query          → REMOVE
L388–L402: Adaptive guide fraction (cell_*)    → REMOVE
L411–L425: Caustic additive (dev_estimate_caustic_only)  → REMOVE
L445–L459: Terminal gather (dev_estimate_photon_density)  → REMOVE
L463–L487: Guided direction (dev_sample_guided_direction) → REPLACE
L489–L512: BSDF sample with guide MIS                     → REPLACE
```

**Replace** the entire guided-direction + BSDF block (L376–L512) with:

```cuda
        // ── Dense-grid photon guide: pick random photon in cell ─────
        float p_guide = (params.dense_valid && !params.preview_mode)
                        ? params.guide_fraction : 0.f;

        float3 wi_world;
        float combined_pdf = 0.f;
        Spectrum f_over_pdf = Spectrum::zero();
        bool sample_valid = false;

        if (p_guide > 0.f && rng.next_float() < p_guide) {
            // ── Photon-guided sample ────────────────────────────────
            int pidx = dev_random_photon_in_cell(
                hit.position, hit.geo_normal, rng);
            if (pidx >= 0) {
                float3 wi_photon = make_f3(
                    params.photon_wi_x[pidx],
                    params.photon_wi_y[pidx],
                    params.photon_wi_z[pidx]);
                // The guided direction is the photon's incoming direction
                wi_world = wi_photon;
                float3 wi_local = frame.world_to_local(wi_world);
                if (wi_local.z > 0.f) {
                    Spectrum f_eval = bsdf_evaluate(mat_id, wo_local, wi_local, hit.uv);
                    float pdf_bsdf = bsdf_pdf(mat_id, wo_local, wi_local);

                    // Guide PDF = 1 / (2π) (uniform hemisphere — we picked
                    // a random photon direction, effectively importance-
                    // sampled by the photon map).
                    float pdf_guide = INV_TWO_PI;
                    combined_pdf = p_guide * pdf_guide + (1.f - p_guide) * pdf_bsdf;
                    if (combined_pdf > 1e-8f) {
                        float cos_theta = wi_local.z;
                        for (int i = 0; i < NUM_LAMBDA; ++i)
                            f_over_pdf.value[i] = f_eval.value[i] * cos_theta / combined_pdf;
                        sample_valid = true;
                    }
                }
            }
        }

        if (!sample_valid) {
            // ── BSDF sample ─────────────────────────────────────────
            long long t_bsdf = clock64();
            BSDFSample bs = bsdf_sample(mat_id, wo_local, hit.uv, rng, pixel_idx);
            result.clk_bsdf += clock64() - t_bsdf;

            if (bs.pdf < 1e-8f || bs.wi.z <= 0.f) break;

            wi_world = frame.local_to_world(bs.wi);
            float cos_theta = bs.wi.z;

            // MIS: include guide PDF when guide is available
            if (p_guide > 0.f) {
                float pdf_guide = INV_TWO_PI;
                combined_pdf = (1.f - p_guide) * bs.pdf + p_guide * pdf_guide;
            } else {
                combined_pdf = bs.pdf;
            }
            if (combined_pdf < 1e-8f) break;

            for (int i = 0; i < NUM_LAMBDA; ++i)
                f_over_pdf.value[i] = bs.f.value[i] * cos_theta / combined_pdf;
            sample_valid = true;
        }
```

**Remove** the `DevPhotonBinDirs fib;` initialization near the top of the
function (around L53) — it will no longer be needed for the surface path.
If volume guide still uses it from within this file, keep it but guard
with `if (params.volume_enabled)`.

**Remove** the caustic additive block (`#ifndef PPT_DISABLE_CAUSTIC_GATHER` ... `#endif`).

**Remove** the terminal gather block (`#ifndef PPT_DISABLE_FINAL_GATHER` ... `#endif`).

The remaining bounce loop keeps:
- NEE shadow rays (unchanged)
- Emission MIS at emitter hits (unchanged)
- Russian roulette (unchanged)
- Volume scattering (unchanged)

---

### Step 7 — `src/optix/optix_renderer.h`

**Remove** hash grid DeviceBuffers (L490):
```cpp
// DELETE:
DeviceBuffer d_grid_sorted_indices_, d_grid_cell_start_, d_grid_cell_end_;
```

**Remove** GPU hash grid build scratch (L493–L495):
```cpp
// DELETE:
DeviceBuffer d_grid_keys_in_, d_grid_keys_out_;
DeviceBuffer d_grid_indices_in_;
DeviceBuffer d_grid_cub_temp_;
```

**Remove** cell analysis DeviceBuffers (L557–L561):
```cpp
// DELETE:
DeviceBuffer d_cell_guide_fraction_;
DeviceBuffer d_cell_caustic_fraction_;
DeviceBuffer d_cell_flux_density_;
int          cell_analysis_count_ = 0;
ConclusionCounters cell_conclusions_;
```

**Remove** cell bin grid (L554):
```cpp
// DELETE:
DeviceBuffer d_cell_bin_grid_;
```

**Remove** stored_grid_ (L544):
```cpp
// DELETE:
HashGrid  stored_grid_;
```

**Remove** the `grid()` accessor (L353):
```cpp
// DELETE:
const HashGrid&  grid()    const { return stored_grid_; }
```

**Remove** `upload_cell_analysis()` declaration (L162–L163).

**Add** dense grid DeviceBuffers (where hash grid was):
```cpp
// Dense 3D grid (surface photon lookup)
DeviceBuffer d_dense_sorted_indices_;
DeviceBuffer d_dense_cell_start_;
DeviceBuffer d_dense_cell_end_;
DenseGridData stored_dense_grid_;   // CPU-side copy for snapshot/analysis
```

**Update** `upload_photon_data()` signature (L152–L158): remove `HashGrid` params,
add `DenseGridData` param (or build the dense grid inside the function).

---

### Step 8 — `src/optix/optix_renderer.cpp`

**In `fill_common_params()`** (L44–L195):

Remove the hash grid parameters from the function signature:
```cpp
// DELETE from signature:
const DeviceBuffer& grid_sorted, const DeviceBuffer& grid_start,
const DeviceBuffer& grid_end,
```
Remove the cell analysis parameters:
```cpp
// DELETE from signature:
const DeviceBuffer& cell_guide, const DeviceBuffer& cell_caustic,
const DeviceBuffer& cell_density
```

Remove the hash grid wiring (L159–L167):
```cpp
// DELETE:
p.grid_sorted_indices = ...
p.grid_cell_start     = ...
p.grid_cell_end       = ...
p.grid_cell_size      = ...
p.grid_table_size     = ...
```

Remove the cell analysis wiring (L170–L175):
```cpp
// DELETE:
p.cell_guide_fraction   = ...
p.cell_caustic_fraction = ...
p.cell_flux_density     = ...
```

**Add** dense grid wiring (in fill_common_params or a new helper):
```cpp
// Dense grid wiring — called after fill_common_params
void fill_dense_grid_params(LaunchParams& p,
    const DeviceBuffer& sorted, const DeviceBuffer& start,
    const DeviceBuffer& end, const DenseGridData& grid)
{
    p.dense_sorted_indices = sorted.d_ptr
        ? const_cast<uint32_t*>(sorted.as<uint32_t>()) : nullptr;
    p.dense_cell_start = start.d_ptr
        ? const_cast<uint32_t*>(start.as<uint32_t>()) : nullptr;
    p.dense_cell_end = end.d_ptr
        ? const_cast<uint32_t*>(end.as<uint32_t>()) : nullptr;
    p.dense_valid    = (sorted.d_ptr != 0) ? 1 : 0;
    p.dense_min_x    = grid.min_x;
    p.dense_min_y    = grid.min_y;
    p.dense_min_z    = grid.min_z;
    p.dense_cell_size = grid.cell_size;
    p.dense_dim_x    = grid.dim_x;
    p.dense_dim_y    = grid.dim_y;
    p.dense_dim_z    = grid.dim_z;
}
```

**Update all 5 launch call sites** (~L290, ~L368, ~L532, ~L845, ~L936)
to remove hash grid / cell analysis arguments from `fill_common_params()`
and add a call to `fill_dense_grid_params()` after it.

**In `fill_cell_grid_params()`** (L200–L259):
Remove the surface `photon_bin_count` wiring.  Keep the volume section.

---

### Step 9 — `src/optix/optix_upload.cpp`

**In `upload_photon_data()`** (L206–L275):

Remove hash grid upload (L239–L241):
```cpp
// DELETE:
d_grid_sorted_indices_.upload(global_grid.sorted_indices);
d_grid_cell_start_.upload(global_grid.cell_start);
d_grid_cell_end_.upload(global_grid.cell_end);
```

Remove bin_idx precompute + CellInfoCache build + upload_cell_analysis
(L252–L271):
```cpp
// DELETE entire block: bin_idx, CellInfoCache, upload_cell_analysis
```

**Add** dense grid build + upload:
```cpp
#include "photon/dense_grid.h"

// Build dense grid on CPU
DenseGridData dense = build_dense_grid(
    global_photons.pos_x.data(),
    global_photons.pos_y.data(),
    global_photons.pos_z.data(),
    (int)global_photons.size(),
    DENSE_GRID_CELL_SIZE);

// Upload to GPU
d_dense_sorted_indices_.upload(dense.sorted_indices);
d_dense_cell_start_.upload(dense.cell_start);
d_dense_cell_end_.upload(dense.cell_end);
stored_dense_grid_ = std::move(dense);

std::printf("[Dense Grid] %d×%d×%d = %d cells  photons=%d\n",
    stored_dense_grid_.dim_x, stored_dense_grid_.dim_y,
    stored_dense_grid_.dim_z, stored_dense_grid_.total_cells(),
    (int)global_photons.size());
```

**Remove** `upload_cell_analysis()` function entirely (L323–L360).

---

### Step 10 — `src/optix/optix_photon_trace.cpp`

**Remove** the GPU hash grid build block (L653–L720):
```cpp
// DELETE the entire "GPU-side hash grid build" section
// from d_grid_sorted_indices_.alloc(...) through the logging printf
```

**Replace** with dense grid build on GPU (or CPU — same logic as Step 9).
If photon tracing already builds on GPU, the simplest approach is to build
the dense grid on CPU after downloading the photon SoA, then upload:

```cpp
// After downloading photon SoA from GPU output buffers:
DenseGridData dense = build_dense_grid(
    stored_photons_.pos_x.data(),
    stored_photons_.pos_y.data(),
    stored_photons_.pos_z.data(),
    (int)stored_photons_.size(),
    DENSE_GRID_CELL_SIZE);

d_dense_sorted_indices_.upload(dense.sorted_indices);
d_dense_cell_start_.upload(dense.cell_start);
d_dense_cell_end_.upload(dense.cell_end);
stored_dense_grid_ = std::move(dense);
```

**Remove** GPU hash grid scratch buffers (d_grid_keys_in_ etc.) — they
are no longer allocated.

**Keep** the volume grid build and volume photon handling unchanged.

---

### Step 11 — `src/optix/optix_renderer.h` includes

**Remove** includes that are no longer needed:
```cpp
// If no other code uses these:
// #include "photon/hash_grid.h"   — remove if hash_grid.h is not used for volume
// #include "photon/cell_cache.h"
```

**Add**:
```cpp
#include "photon/dense_grid.h"
```

Note: `hash_grid.h` may still be needed for the volume hash grid (`volume_grid_`).
Only remove if no volume code references it.

---

### Step 12 — Cleanup

After all the above compiles and runs:

1. **Remove** `#include "photon/cell_cache.h"` from all files that no longer use it.
2. **Remove** `photon/cell_cache.h` and `photon/cell_cache.cpp` if they exist and
   are only used by the removed surface pipeline.
3. **Remove** the `photon_bin_idx` field from `PhotonSoA` (in `photon/photon_io.h`
   or wherever it's defined) if nothing else uses it.
4. **Remove** the `#ifndef PPT_DISABLE_CAUSTIC_GATHER` / `#ifndef PPT_DISABLE_FINAL_GATHER`
   guard macros and their use sites.
5. **Remove** the `CellBinGrid` / `cell_bin_grid_` from `optix_renderer.h` (L549–L553)
   if it was only used for the surface guide.
6. **Remove** the `photon_bin_count` from LaunchParams if only used for surface guide.
7. Update tests that reference the hash grid or kNN gather.

---

## Constants summary

| Constant | Value | Use |
|---|---|---|
| `DENSE_GRID_CELL_SIZE` | `0.025f` (= `DEFAULT_CAUSTIC_RADIUS`) | Dense cell edge length |
| `DEFAULT_GUIDE_FRACTION` | `0.5f` | Photon vs BSDF MIS mix |
| `DEFAULT_SURFACE_TAU` | `0.02f` | Plane-distance gate in cell lookup |
| `INV_TWO_PI` | `1/(2π)` | Guide PDF for uniform hemisphere |

---

## Build & test

```
build.bat           # release build (no tests)
build.bat test      # build + run fast tests
```

After the rewrite:
1. Build must succeed with zero errors.
2. Run the renderer, visually confirm convergence (expect noisier at low SPP
   but correct at high SPP).
3. Take a snapshot → run `photon_map_analysis` tool to verify the dense grid
   is populated and photon directions are coherent.

---

## What is NOT changed

- Volume photon system: `volume_grid_`, `vol_cell_bin_grid_`, all `d_vol_*`
  DeviceBuffers, `dev_estimate_volume_photon_density()`,
  `dev_read_vol_cell_histogram()`, `dev_vol_cell_grid_index()`.
- NEE shadow rays: `dev_nee_dispatch()`, `dev_nee_direct()`, etc.
- Emission MIS at emitter hits.
- Russian roulette.
- Camera specular chain / DOF / adaptive sampling.
- Photon tracing kernel (`__raygen__photon_trace`) — photons are still
  traced and deposited the same way; only the acceleration structure
  for lookup changes.
- SPPM system (if used).
