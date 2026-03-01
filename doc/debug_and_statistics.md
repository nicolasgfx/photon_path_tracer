# Hotkeys, Gates and Statistics

## Implementation Status

All features below are **implemented**. Build passes, fast tests green.

---

## §1 Compile-Time Statistics Gate

Statistics collection is gated by `constexpr bool ENABLE_STATS` in
[src/core/config.h](../src/core/config.h).  When `false`, the compiler
eliminates all statistics code (zero overhead).  Collection uses
`if constexpr (ENABLE_STATS)` throughout.

**Files:**
- `src/core/config.h` — gate definition (§10 DEBUG & STATISTICS)
- `src/debug/stats_collector.h` — all stats structs + `print_stats_console()`

---

## §2 Hotkeys

| Key | Action | Details |
|-----|--------|---------|
| **T** | Toggle guided/unguided path tracing | Sets `guide_fraction_` to 0 (unguided) or `DEFAULT_GUIDE_FRACTION` (guided). Resets accumulation. |
| **C** | Toggle histogram-only conclusions | Only active when guided is ON. Skips expensive analysis; keeps histogram conclusions only. Resets accumulation. |
| **S** | Toggle stats overlay | Shows live renderer statistics in top-right corner (SPP, photon counts, GPU info, timing). |
| **R** | Snapshot + console stats | Existing PNG/JSON export now also prints grouped `RendererStats` to console (gated by `ENABLE_STATS`). |

**Files:**
- `src/app/viewer.cpp` — key handlers in `key_callback()`, overlay in `render_stats_overlay()`
- `src/app/viewer.h` — `AppState` fields: `guided_enabled`, `histogram_only`, `show_stats_overlay`
- `src/optix/optix_renderer.h` — `guide_fraction_`, `histogram_only_`, setters/getters

---

## §3 Statistics Output

Statistics are displayed in two places:
1. **Console** — full grouped output on R-key snapshot (box-drawing table)
2. **Overlay** — live summary on S-key toggle (top-right corner)

### Groups

#### Hardware
- GPU name, VRAM (MB), SM count, compute capability

#### Geometry
- Triangle count, emissive triangle count, material count

#### Photon Mapping
- Emitted / stored counts
- Per-tag breakdown: global (tag 0), global-caustic (tag 1), targeted (tag 2)
- Photon flag tallies: traversed_glass, caustic_glass, caustic_specular, dispersion, volume_segment
- Gather and caustic radii
- Hash grid: total cells, populated cells, occupancy %, photons/cell (min/max/avg)

#### Path Tracing
- SPP: min, max, avg
- Guide state: enabled/disabled, fraction, histogram-only mode
- Conclusion counters (per `build_cell_analysis()` call):
  - C1/C6: histogram quality (bins ≤ 2)
  - C3: too diffuse (spread > 0.9)
  - C4: low photon count (< 30)
  - C5: high variance (CV attenuation)
- Guide fraction distribution: 10-bin histogram [0.0–1.0)
- Cells with photons / cells with caustics

#### Rendering
- Total render time (ms/s)
- Per-phase timing (when available): photon trace, caustic trace, targeted trace, hash grid build, cell grid build, camera pass, denoiser

**Files:**
- `src/debug/stats_collector.h` — `RendererStats`, `ConclusionCounters`, `PhotonFlagCounts`, `TimingStats`, `GridOccupancy`, `GuideFractionDist`, `print_stats_console()`
- `src/photon/photon_analysis.h` — `build_cell_analysis()` now accepts optional `ConclusionCounters*`

---

## §4 Runtime Guide Fraction

`guide_fraction_` is a runtime member on `OptixRenderer` (was hardcoded as
`DEFAULT_GUIDE_FRACTION`).  All 6 launch sites in `optix_renderer.cpp` now
read `guide_fraction_` instead of the constant.  The T hotkey sets it to 0
(unguided) or `DEFAULT_GUIDE_FRACTION` (guided).

---

## §5 GPU Device Info

`OptixRenderer::init()` calls `cudaGetDeviceProperties()` and stores:
- `gpu_name_`, `gpu_vram_total_`, `gpu_sm_count_`, `gpu_cc_major_`, `gpu_cc_minor_`

Printed at startup (`[GPU]` line) and available via public accessors for
overlay and console stats.

---

## §6 Debug Output Gating

All debug file output is subordinate to `ENABLE_STATS`.  When the gate is
`false`, no stats JSON, no progress snapshots, and no debug PNGs are
emitted — the snapshot R-key still saves the PNG but skips all statistics.

### Gated flags (config.h)

| Flag | Default | Notes |
|------|---------|-------|
| `PROGRESS_SNAPSHOT_ENABLED` | `ENABLE_STATS && true` | Power-of-2 SPP snapshots |
| `DEBUG_COMPONENT_PNGS` | `ENABLE_STATS && false` | per-component PNGs |
| `DEBUG_PHOTON_INDIRECT_PNG` | `ENABLE_STATS && false` | photon-indirect preview |
| `DEBUG_CAUSTIC_PNG` | `ENABLE_STATS && false` | caustic-only debug |
| `DEBUG_COVERAGE_PNG` | `ENABLE_STATS && false` | coverage debug |

### Gated code paths

- **JSON stats block** (`viewer.cpp`, R-key handler) — wrapped in
  `if constexpr (ENABLE_STATS)`.  Includes the snapshot JSON, console
  RendererStats, and analysis report.
- **`print_kernel_profiling()`** (`optix_renderer.cpp`) — early return when
  `ENABLE_STATS == false` (heavy D2H copy).

**Files:**
- `src/core/config.h` — flag subordination
- `src/app/viewer.cpp` — JSON + stats gating
- `src/optix/optix_renderer.cpp` — profiling gating

---

## §7 Analysis Report & GPU Expert Prompt

On R-key snapshot, an `_analysis.json` file is written alongside the
normal snapshot JSON.  This file follows schema `photon_tracer_analysis_v1`
and contains the full `RendererStats` (hardware, geometry, photon mapping,
path tracing, timing, config) plus camera state.

### Usage

1. Render to desired SPP, press **R**
2. Find `output/snapshot_*_analysis.json`
3. Paste contents into [doc/prompts/gpu_expert_analysis.md](prompts/gpu_expert_analysis.md)
4. Send to LLM for automated performance / quality analysis

### Schema top-level keys

| Key | Contents |
|-----|----------|
| `hardware` | GPU name, VRAM, SM count, compute capability |
| `image` | Resolution, accumulated SPP |
| `camera` | Position, look_at, FOV, light scale |
| `geometry` | Triangles, emissive tris, materials |
| `photon_map` | Budgets, radii, flags, grid occupancy |
| `path_tracing` | Guide state, conclusions, histogram, SPP range |
| `timing_ms` | Per-phase breakdown |
| `config` | Compile-time constants snapshot |

**Files:**
- `src/debug/stats_collector.h` — `AnalysisReport`, `write_analysis_json()`
- `doc/prompts/gpu_expert_analysis.md` — reusable LLM prompt template