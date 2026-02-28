
#pragma once
// ─────────────────────────────────────────────────────────────────────
// debug.h – Debug visualization overlay system (Section 10)
// ─────────────────────────────────────────────────────────────────────
// Key bindings:
//   F1  – Show ALL photons as GL_POINTS (no ray tracing)
//   F2  – Cycle photon type filter (all → glass → caustic_glass →
//          volume → dispersion → mirror_caustic → off)
//   F3  – (reserved)
//   F4  – Toggle hash grid
//   F5  – Toggle photon directions
//   F6  – Show PDFs
//   F7  – Show radius sphere
//   F8  – Show MIS weights
//   F9  – Spectral coloring
//   F11 – Photon heatmap (per-triangle irradiance)
//   TAB – Cycle render modes
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"
#include "photon/photon.h"
#include "photon/hash_grid.h"
#include "renderer/renderer.h"

#include <cstdint>
#include <vector>
#include <iostream>

// ── Photon filter modes (F2 cycles through these) ──────────────────

enum class PhotonFilterMode : int {
    Off             = 0,  // No filter active
    All             = 1,  // All photons (same as F1)
    TraversedGlass  = 2,  // PHOTON_FLAG_TRAVERSED_GLASS
    CausticGlass    = 3,  // PHOTON_FLAG_CAUSTIC_GLASS
    Volume          = 4,  // PHOTON_FLAG_VOLUME_SEGMENT
    Dispersion      = 5,  // PHOTON_FLAG_DISPERSION
    CausticSpecular = 6,  // PHOTON_FLAG_CAUSTIC_SPECULAR
    Count_          = 7
};

inline const char* photon_filter_name(PhotonFilterMode m) {
    switch (m) {
        case PhotonFilterMode::Off:             return "Off";
        case PhotonFilterMode::All:             return "All";
        case PhotonFilterMode::TraversedGlass:  return "Traversed glass";
        case PhotonFilterMode::CausticGlass:    return "Caustic (glass)";
        case PhotonFilterMode::Volume:          return "Volume segment";
        case PhotonFilterMode::Dispersion:      return "Dispersion";
        case PhotonFilterMode::CausticSpecular: return "Caustic (mirror)";
        default:                                return "?";
    }
}

inline uint8_t photon_filter_flag(PhotonFilterMode m) {
    switch (m) {
        case PhotonFilterMode::TraversedGlass:  return PHOTON_FLAG_TRAVERSED_GLASS;
        case PhotonFilterMode::CausticGlass:    return PHOTON_FLAG_CAUSTIC_GLASS;
        case PhotonFilterMode::Volume:          return PHOTON_FLAG_VOLUME_SEGMENT;
        case PhotonFilterMode::Dispersion:      return PHOTON_FLAG_DISPERSION;
        case PhotonFilterMode::CausticSpecular: return PHOTON_FLAG_CAUSTIC_SPECULAR;
        default:                                return 0;  // All / Off → no bit filter
    }
}

// ── Debug overlay state ─────────────────────────────────────────────

struct DebugState {
    bool show_photon_points   = false;  // F1: show ALL photons
    PhotonFilterMode photon_filter = PhotonFilterMode::Off;  // F2: cycle filter
    bool show_hash_grid       = false;  // F4
    bool show_photon_dirs     = false;  // F5
    bool show_pdfs            = false;  // F6
    bool show_radius_sphere   = false;  // F7
    bool show_mis_weights     = false;  // F8
    bool spectral_coloring    = false;  // F9
    bool show_photon_heatmap  = false;  // F11

    bool show_help_overlay    = false;  // H to toggle

    RenderMode current_mode   = RenderMode::Full;

    // Mouse hover state
    int   hover_x = -1;
    int   hover_y = -1;

    // Returns true if any photon overlay is active (F1 or F2 with filter)
    bool photon_overlay_active() const {
        return show_photon_points || photon_filter != PhotonFilterMode::Off;
    }

    void toggle_photon_points()  { show_photon_points  = !show_photon_points;
                                   if (show_photon_points) photon_filter = PhotonFilterMode::Off; }
    void cycle_photon_filter()   {
        int m = (int)photon_filter + 1;
        if (m >= (int)PhotonFilterMode::Count_) m = 0;
        photon_filter = (PhotonFilterMode)m;
        if (photon_filter != PhotonFilterMode::Off) show_photon_points = false;
    }
    void toggle_hash_grid()      { show_hash_grid      = !show_hash_grid; }
    void toggle_photon_dirs()    { show_photon_dirs    = !show_photon_dirs; }
    void toggle_pdfs()           { show_pdfs           = !show_pdfs; }
    void toggle_radius_sphere()  { show_radius_sphere  = !show_radius_sphere; }
    void toggle_mis_weights()    { show_mis_weights    = !show_mis_weights; }
    void toggle_spectral()       { spectral_coloring   = !spectral_coloring; }
    void toggle_photon_heatmap() { show_photon_heatmap = !show_photon_heatmap; }
    void toggle_help_overlay()   { show_help_overlay   = !show_help_overlay; }

    void cycle_render_mode() {
        int mode = (int)current_mode;
        mode = (mode + 1) % 7;
        current_mode = (RenderMode)mode;
    }

    static const char* render_mode_name(RenderMode m) {
        switch (m) {
            case RenderMode::Full:         return "Full";
            case RenderMode::DirectOnly:   return "Direct Only";
            case RenderMode::IndirectOnly: return "Indirect Only";
            case RenderMode::PhotonMap:    return "Photon Map";
            case RenderMode::Normals:      return "Normals";
            case RenderMode::MaterialID:   return "Material ID";
            case RenderMode::Depth:        return "Depth";
            default:                       return "Unknown";
        }
    }
};

// ── Cell overlay information ────────────────────────────────────────

struct CellInfo {
    int3     cell_index;
    uint32_t photon_count;
    float    sum_flux;
    float    avg_flux;
    int      dominant_lambda_bin;
    float    dominant_lambda_nm;
    PhotonMapType map_type;
};

// Query cell info at a world position
inline CellInfo query_cell_info(
    float3 world_pos,
    const PhotonSoA& photons,
    const HashGrid& grid,
    PhotonMapType map_type)
{
    CellInfo info;
    info.cell_index = grid.cell_coord(world_pos);
    info.map_type   = map_type;
    info.photon_count = 0;
    info.sum_flux = 0.f;

    Spectrum flux_spectrum = Spectrum::zero();

    grid.query(world_pos, grid.cell_size, photons,
        [&](uint32_t idx, float /*dist2*/) {
            info.photon_count++;
            float f = photons.total_flux(idx);
            info.sum_flux += f;
            Spectrum pf = photons.get_flux(idx);
            for (int b = 0; b < NUM_LAMBDA; ++b)
                flux_spectrum.value[b] += pf.value[b];
        });

    info.avg_flux = (info.photon_count > 0)
        ? info.sum_flux / info.photon_count : 0.f;
    info.dominant_lambda_bin = flux_spectrum.dominant_bin();
    info.dominant_lambda_nm  = lambda_of_bin(info.dominant_lambda_bin);

    return info;
}

// ── Photon visualization overlay ────────────────────────────────────

// Draw photon points into the framebuffer (projected via camera)
// filter_flag: if non-zero, only photons with that flag bit set are drawn.
inline void overlay_photon_points(
    FrameBuffer& fb,
    const Camera& camera,
    const PhotonSoA& photons,
    bool spectral_color,
    uint8_t filter_flag = 0,
    float point_brightness = 2.0f)
{
    for (size_t i = 0; i < photons.size(); ++i) {
        // Flag filter: skip photons that don't match
        if (filter_flag != 0) {
            if (photons.path_flags.size() <= i) continue;
            if ((photons.path_flags[i] & filter_flag) == 0) continue;
        }
        float3 pos = make_f3(photons.pos_x[i], photons.pos_y[i], photons.pos_z[i]);

        // Project to screen: compute the pixel coordinates
        float3 to_photon = pos - camera.position;
        float depth = dot(to_photon, camera.w * (-1.f)); // Depth along view dir
        if (depth <= 0.f) continue;

        // Project onto camera plane
        float u_coord = dot(to_photon, camera.u);
        float v_coord = dot(to_photon, camera.v);

        // Convert to normalized [0,1] screen coords
        float aspect = (float)fb.width / (float)fb.height;
        float theta  = camera.fov_deg * PI / 180.0f;
        float half_h = tanf(theta * 0.5f);
        float half_w = half_h * aspect;

        float sx = (u_coord / depth + half_w) / (2.f * half_w);
        float sy = (v_coord / depth + half_h) / (2.f * half_h);

        int px = (int)(sx * fb.width);
        int py = (int)(sy * fb.height);

        if (px < 0 || px >= fb.width || py < 0 || py >= fb.height) continue;

        // Color by wavelength or simple white
        int idx = py * fb.width + px;
        if (spectral_color) {
            // Map spectral flux to visible color
            Spectrum s = photons.get_flux(i) * point_brightness;
            float3 rgb = spectrum_to_srgb(s);
            fb.srgb[idx * 4 + 0] = (uint8_t)(fminf(rgb.x * 255.f, 255.f));
            fb.srgb[idx * 4 + 1] = (uint8_t)(fminf(rgb.y * 255.f, 255.f));
            fb.srgb[idx * 4 + 2] = (uint8_t)(fminf(rgb.z * 255.f, 255.f));
        } else {
            float intensity = fminf(photons.total_flux(i) * point_brightness, 1.f);
            uint8_t v = (uint8_t)(intensity * 255.f);
            fb.srgb[idx * 4 + 0] = v;
            fb.srgb[idx * 4 + 1] = v;
            fb.srgb[idx * 4 + 2] = v;
        }
    }
}

// ── Process keyboard input for debug toggles ────────────────────────

// Returns true if the key was handled
inline bool handle_debug_key(int key, DebugState& state) {
    // GLFW key codes
    constexpr int KEY_F1  = 290;
    constexpr int KEY_F2  = 291;
    constexpr int KEY_F3  = 292;
    constexpr int KEY_F4  = 293;
    constexpr int KEY_F5  = 294;
    constexpr int KEY_F6  = 295;
    constexpr int KEY_F7  = 296;
    constexpr int KEY_F8  = 297;
    constexpr int KEY_F9  = 298;
    constexpr int KEY_F11 = 300;
    constexpr int KEY_TAB = 258;
    constexpr int KEY_H   = 72;

    auto on_off = [](bool v) -> const char* { return v ? "ON" : "OFF"; };

    switch (key) {
        case KEY_F1:
            state.toggle_photon_points();
            std::cout << "[Debug] F1  All photons: " << on_off(state.show_photon_points) << "\n";
            return true;
        case KEY_F2:
            state.cycle_photon_filter();
            std::cout << "[Debug] F2  Photon filter: " << photon_filter_name(state.photon_filter) << "\n";
            return true;
        case KEY_F3:
            std::cout << "[Debug] F3  (reserved)\n";
            return true;
        case KEY_F4:
            state.toggle_hash_grid();
            std::cout << "[Debug] F4  Hash grid: " << on_off(state.show_hash_grid) << "  (planned)\n";
            return true;
        case KEY_F5:
            state.toggle_photon_dirs();
            std::cout << "[Debug] F5  Photon dirs: " << on_off(state.show_photon_dirs) << "  (planned)\n";
            return true;
        case KEY_F6:
            state.toggle_pdfs();
            std::cout << "[Debug] F6  PDFs: " << on_off(state.show_pdfs) << "  (planned)\n";
            return true;
        case KEY_F7:
            state.toggle_radius_sphere();
            std::cout << "[Debug] F7  Radius sphere: " << on_off(state.show_radius_sphere) << "  (planned)\n";
            return true;
        case KEY_F8:
            state.toggle_mis_weights();
            std::cout << "[Debug] F8  MIS weights: " << on_off(state.show_mis_weights) << "  (planned)\n";
            return true;
        case KEY_F9:
            state.toggle_spectral();
            std::cout << "[Debug] F9  Spectral coloring: " << on_off(state.spectral_coloring) << "\n";
            return true;
        case KEY_F11:
            state.toggle_photon_heatmap();
            std::cout << "[Debug] F11 Photon heatmap: " << on_off(state.show_photon_heatmap) << "\n";
            return true;
        case KEY_TAB:
            state.cycle_render_mode();
            std::cout << "[Debug] TAB Render mode: " << DebugState::render_mode_name(state.current_mode) << "\n";
            return true;
        case KEY_H:
            state.toggle_help_overlay();
            return true;
        default:
            return false;
    }
}
