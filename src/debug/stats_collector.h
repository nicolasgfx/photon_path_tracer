#pragma once
// ─────────────────────────────────────────────────────────────────────
// stats_collector.h – Runtime statistics collection (gated by ENABLE_STATS)
// ─────────────────────────────────────────────────────────────────────
// Collects per-render statistics grouped by subsystem:
//   • Photon mapping: counts by flag, emission distribution, hash occupancy
//   • Path tracing:   SPP range, conclusion/measure counters
//   • Geometry:       triangle/emissive/material breakdown
//   • Hardware:       GPU name and capabilities
//   • Rendering:      timing breakdown
//
// All collection is gated by `if constexpr (ENABLE_STATS)` so the
// compiler eliminates everything when ENABLE_STATS == false.
// ─────────────────────────────────────────────────────────────────────

#include "core/config.h"
#include "core/types.h"
#include "photon/photon.h"

#include <cstdint>
#include <cstdio>
#include <array>
#include <string>
#include <algorithm>
#include <cmath>
#include <vector>

// ── Conclusion counters (§3 of architecture doc) ────────────────────
// C1/C6: histogram quality (few active bins)
// C3:    skipped — too diffuse (spread > 0.9)
// C4:    reliability scaled by photon count
// C5:    reduced by high variance
struct ConclusionCounters {
    int c1_c6_histogram = 0;   // bins ≤ 2 → boosted to 0.7
    int c3_too_diffuse  = 0;   // spread > 0.9 → guide = 0
    int c4_low_count    = 0;   // photon_count < 30 → scaled down
    int c5_high_var     = 0;   // CV attenuated guide
    int total_cells     = 0;   // cells evaluated

    void reset() { *this = ConclusionCounters{}; }
};

// ── Photon flag tallies ─────────────────────────────────────────────
struct PhotonFlagCounts {
    int traversed_glass    = 0;
    int caustic_glass      = 0;
    int volume_segment     = 0;
    int dispersion         = 0;
    int caustic_specular   = 0;
    int total              = 0;

    void reset() { *this = PhotonFlagCounts{}; }
};

// ── Timing breakdown (milliseconds) ─────────────────────────────────
struct TimingStats {
    double photon_trace_ms       = 0.0;
    double caustic_trace_ms      = 0.0;
    double targeted_trace_ms     = 0.0;
    double hash_grid_build_ms    = 0.0;
    double cell_grid_build_ms    = 0.0;
    double camera_pass_ms        = 0.0;
    double denoiser_ms           = 0.0;
    double total_render_ms       = 0.0;

    void reset() { *this = TimingStats{}; }
};

// ── Hash grid occupancy ─────────────────────────────────────────────
struct GridOccupancy {
    int  populated_cells     = 0;
    int  total_cells         = 0;
    int  min_photons_per_cell = 0;
    int  max_photons_per_cell = 0;
    float avg_photons_per_cell = 0.f;

    float occupancy_pct() const {
        return total_cells > 0 ? 100.f * (float)populated_cells / (float)total_cells : 0.f;
    }

    void reset() { *this = GridOccupancy{}; }
};

// ── Guide fraction distribution (10 bins, 0.0–1.0) ─────────────────
struct GuideFractionDist {
    std::array<int, 10> bins = {};   // [0..0.1), [0.1..0.2), ... [0.9..1.0]
    int cells_with_photons  = 0;
    int cells_with_caustics = 0;

    void reset() { *this = GuideFractionDist{}; }

    void add(float gf, bool has_caustic) {
        int b = (std::min)(9, (int)(gf * 10.f));
        bins[b]++;
        cells_with_photons++;
        if (has_caustic) cells_with_caustics++;
    }
};

// ── Full statistics snapshot ────────────────────────────────────────
struct RendererStats {
    // Photon mapping
    PhotonFlagCounts photon_flags;
    int   photons_global         = 0;   // tag 0
    int   photons_global_caustic = 0;   // tag 1
    int   photons_targeted       = 0;   // tag 2
    int   photons_emitted        = 0;
    int   photons_stored         = 0;
    float gather_radius          = 0.f;
    float caustic_radius         = 0.f;
    GridOccupancy grid_occupancy;

    // Path tracing
    int   spp_min = 0, spp_max = 0;
    float spp_avg = 0.f;
    ConclusionCounters conclusions;
    GuideFractionDist  guide_dist;

    // Geometry
    int   num_triangles       = 0;
    int   num_emissive_tris   = 0;
    int   num_materials       = 0;
    std::array<int, 9> tris_per_material_type = {}; // indexed by MaterialType

    // Hardware
    std::string gpu_name;
    size_t      gpu_vram_bytes = 0;
    int         gpu_sm_count   = 0;
    int         gpu_cc_major   = 0;
    int         gpu_cc_minor   = 0;

    // Rendering timing
    TimingStats timing;

    // Guide state
    float guide_fraction      = 0.f;
    bool  guided_enabled      = true;
    bool  histogram_only      = false;

    void reset() { *this = RendererStats{}; }
};

// ── Tally photon flags from a PhotonSoA ─────────────────────────────
inline PhotonFlagCounts tally_photon_flags(const PhotonSoA& photons) {
    PhotonFlagCounts c;
    if constexpr (!ENABLE_STATS) return c;

    c.total = (int)photons.size();
    for (size_t i = 0; i < photons.size(); ++i) {
        uint8_t f = photons.path_flags[i];
        if (f & PHOTON_FLAG_TRAVERSED_GLASS)  c.traversed_glass++;
        if (f & PHOTON_FLAG_CAUSTIC_GLASS)    c.caustic_glass++;
        if (f & PHOTON_FLAG_VOLUME_SCATTER)   c.volume_segment++;
        if (f & PHOTON_FLAG_DISPERSION)       c.dispersion++;
        if (f & PHOTON_FLAG_CAUSTIC_SPECULAR) c.caustic_specular++;
    }
    return c;
}

// ── Compute hash grid occupancy from HashGrid ───────────────────────
inline GridOccupancy compute_grid_occupancy(const HashGrid& grid) {
    GridOccupancy g;
    if constexpr (!ENABLE_STATS) return g;
    if (grid.table_size == 0) return g;

    g.total_cells = (int)grid.table_size;
    g.min_photons_per_cell = INT_MAX;
    long long sum = 0;

    for (uint32_t i = 0; i < grid.table_size; ++i) {
        if (grid.cell_start[i] == 0xFFFFFFFFu) continue;
        int count = (int)(grid.cell_end[i] - grid.cell_start[i]);
        g.populated_cells++;
        g.min_photons_per_cell = (std::min)(g.min_photons_per_cell, count);
        g.max_photons_per_cell = (std::max)(g.max_photons_per_cell, count);
        sum += count;
    }

    if (g.populated_cells == 0) g.min_photons_per_cell = 0;
    g.avg_photons_per_cell = g.populated_cells > 0
        ? (float)sum / (float)g.populated_cells : 0.f;
    return g;
}

// ── Print grouped statistics to console ─────────────────────────────
inline void print_stats_console(const RendererStats& s) {
    if constexpr (!ENABLE_STATS) return;

    std::printf("\n");
    std::printf("╔══════════════════════════════════════════════════════╗\n");
    std::printf("║              RENDERER STATISTICS                    ║\n");
    std::printf("╠══════════════════════════════════════════════════════╣\n");

    // ── Hardware ─────────────────────────────────────────────────────
    std::printf("║ HARDWARE                                            ║\n");
    std::printf("║  GPU: %-46s ║\n", s.gpu_name.c_str());
    std::printf("║  VRAM: %.0f MB  |  SMs: %d  |  CC: %d.%d            \n",
                (double)s.gpu_vram_bytes / (1024.0 * 1024.0),
                s.gpu_sm_count, s.gpu_cc_major, s.gpu_cc_minor);

    // ── Geometry ─────────────────────────────────────────────────────
    std::printf("╠──────────────────────────────────────────────────────╣\n");
    std::printf("║ GEOMETRY                                            ║\n");
    std::printf("║  Triangles: %d  |  Emissive: %d  |  Materials: %d\n",
                s.num_triangles, s.num_emissive_tris, s.num_materials);

    // ── Photon Mapping ───────────────────────────────────────────────
    std::printf("╠──────────────────────────────────────────────────────╣\n");
    std::printf("║ PHOTON MAPPING                                      ║\n");
    std::printf("║  Emitted: %d  |  Stored: %d\n",
                s.photons_emitted, s.photons_stored);
    std::printf("║  Global: %d  |  Global-caustic: %d  |  Targeted: %d\n",
                s.photons_global, s.photons_global_caustic, s.photons_targeted);
    std::printf("║  Flags  trav_glass: %d  caustic_glass: %d  specular: %d\n",
                s.photon_flags.traversed_glass,
                s.photon_flags.caustic_glass,
                s.photon_flags.caustic_specular);
    std::printf("║         dispersion: %d  volume: %d\n",
                s.photon_flags.dispersion,
                s.photon_flags.volume_segment);
    std::printf("║  Radii  gather: %.5f  caustic: %.5f\n",
                s.gather_radius, s.caustic_radius);
    std::printf("║  Grid   cells: %d  populated: %d (%.1f%%)\n",
                s.grid_occupancy.total_cells,
                s.grid_occupancy.populated_cells,
                s.grid_occupancy.occupancy_pct());
    std::printf("║         photons/cell  min: %d  max: %d  avg: %.1f\n",
                s.grid_occupancy.min_photons_per_cell,
                s.grid_occupancy.max_photons_per_cell,
                s.grid_occupancy.avg_photons_per_cell);

    // ── Path Tracing ────────────────────────────────────────────────
    std::printf("╠──────────────────────────────────────────────────────╣\n");
    std::printf("║ PATH TRACING                                        ║\n");
    std::printf("║  SPP: min %d  max %d  avg %.1f\n",
                s.spp_min, s.spp_max, s.spp_avg);
    std::printf("║  Guide: %s  fraction: %.2f  histogram-only: %s\n",
                s.guided_enabled ? "ON" : "OFF",
                s.guide_fraction,
                s.histogram_only ? "YES" : "no");
    if (s.conclusions.total_cells > 0) {
        std::printf("║  Conclusions (%d cells analysed):\n", s.conclusions.total_cells);
        std::printf("║    C1/C6 histogram: %d  C3 too-diffuse: %d\n",
                    s.conclusions.c1_c6_histogram,
                    s.conclusions.c3_too_diffuse);
        std::printf("║    C4 low-count:    %d  C5 high-var:    %d\n",
                    s.conclusions.c4_low_count,
                    s.conclusions.c5_high_var);
    }
    if (s.guide_dist.cells_with_photons > 0) {
        std::printf("║  Guide fraction distribution:\n║    ");
        for (int b = 0; b < 10; ++b)
            std::printf("[%.1f]: %d  ", b * 0.1f, s.guide_dist.bins[b]);
        std::printf("\n");
        std::printf("║    Cells w/photons: %d  w/caustics: %d\n",
                    s.guide_dist.cells_with_photons,
                    s.guide_dist.cells_with_caustics);
    }

    // ── Rendering ───────────────────────────────────────────────────
    std::printf("╠──────────────────────────────────────────────────────╣\n");
    std::printf("║ RENDERING                                           ║\n");
    if (s.timing.total_render_ms > 0.0)
        std::printf("║  Total render time: %.1f ms (%.2f s)\n",
                    s.timing.total_render_ms,
                    s.timing.total_render_ms / 1000.0);
    if (s.timing.photon_trace_ms > 0.0)
        std::printf("║  Photon trace: %.1f ms\n", s.timing.photon_trace_ms);
    if (s.timing.caustic_trace_ms > 0.0)
        std::printf("║  Caustic trace: %.1f ms\n", s.timing.caustic_trace_ms);
    if (s.timing.targeted_trace_ms > 0.0)
        std::printf("║  Targeted trace: %.1f ms\n", s.timing.targeted_trace_ms);
    if (s.timing.hash_grid_build_ms > 0.0)
        std::printf("║  Hash grid build: %.1f ms\n", s.timing.hash_grid_build_ms);
    if (s.timing.cell_grid_build_ms > 0.0)
        std::printf("║  Cell grid build: %.1f ms\n", s.timing.cell_grid_build_ms);
    if (s.timing.camera_pass_ms > 0.0)
        std::printf("║  Camera pass: %.1f ms\n", s.timing.camera_pass_ms);
    if (s.timing.denoiser_ms > 0.0)
        std::printf("║  Denoiser: %.1f ms\n", s.timing.denoiser_ms);

    std::printf("╚══════════════════════════════════════════════════════╝\n\n");
}
