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