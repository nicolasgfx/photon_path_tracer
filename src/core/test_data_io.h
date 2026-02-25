#pragma once
// ─────────────────────────────────────────────────────────────────────
// test_data_io.h – Binary save / load of photon data for unit tests
// ─────────────────────────────────────────────────────────────────────
// Provides functions to:
//   1. save_test_data()  – dump PhotonSoA + config to a binary file
//   2. load_test_data()  – read them back
//
// The scene itself is NOT serialized — it is loaded from the OBJ at
// test time (deterministic).  Only the stochastic photon data that
// cannot be reproduced without matching RNG states is serialized.
//
// Binary layout (little-endian):
//   [Header]
//     magic           : uint32  = 0x50505444 ("PPTD")
//     version         : uint32  = 3
//     num_photons_cfg : uint32  (number of photons emitted)
//     gather_radius   : float
//     caustic_radius  : float
//     max_bounces     : uint32
//     min_bounces_rr  : uint32
//     rr_threshold    : float
//     algo_version    : uint32  (v3+; PHOTON_ALGORITHM_VERSION at save time)
//     scene_path_len  : uint32
//     scene_path      : char[scene_path_len]  (relative, e.g. "cornell_box/cornellbox.obj")
//   [Global photons]
//     count           : uint64
//     pos_x           : float[count]
//     pos_y           : float[count]
//     pos_z           : float[count]
//     wi_x            : float[count]
//     wi_y            : float[count]
//     wi_z            : float[count]
//     lambda_bin      : uint16[count]   (primary hero only)
//     flux            : float[count]    (primary hero only)
//     norm_x          : float[count]  (v2+)
//     norm_y          : float[count]  (v2+)
//     norm_z          : float[count]  (v2+)
//   [Caustic photons]  (same layout, preceded by uint64 count)
// ─────────────────────────────────────────────────────────────────────

#include "photon/photon.h"
#include <string>
#include <fstream>
#include <iostream>
#include <cstdint>

// Magic number: "PPTD" = Photon Path Tracer Data
constexpr uint32_t PPTD_MAGIC   = 0x50505444u;
constexpr uint32_t PPTD_VERSION = 3u;  // v3: added algo_version for staleness detection

// Photon algorithm version — bump this whenever trace_photons() semantics
// change (emission sampling, deposit rules, material classification, etc.).
// A stale binary snapshot will be auto-regenerated on the next test run.
constexpr uint32_t PHOTON_ALGORITHM_VERSION = 1u;  // v2.2: cosine hemisphere + classify_for_photons

// ── Header stored in the binary ─────────────────────────────────────
struct TestDataHeader {
    uint32_t magic          = PPTD_MAGIC;
    uint32_t version        = PPTD_VERSION;
    uint32_t num_photons_cfg = 0;   // photons emitted (config)
    float    gather_radius  = 0.f;
    float    caustic_radius = 0.f;
    uint32_t max_bounces    = 0;
    uint32_t min_bounces_rr = 0;
    float    rr_threshold   = 0.f;
    uint32_t algo_version   = PHOTON_ALGORITHM_VERSION; // v3+
    std::string scene_path;         // relative to scenes/ dir
};

// ── Helper: write raw bytes ─────────────────────────────────────────
namespace detail {

template<typename T>
inline void write_val(std::ofstream& f, T v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template<typename T>
inline void write_vec(std::ofstream& f, const std::vector<T>& v) {
    if (!v.empty())
        f.write(reinterpret_cast<const char*>(v.data()),
                static_cast<std::streamsize>(v.size() * sizeof(T)));
}

template<typename T>
inline T read_val(std::ifstream& f) {
    T v{};
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    return v;
}

template<typename T>
inline void read_vec(std::ifstream& f, std::vector<T>& v, size_t count) {
    v.resize(count);
    if (count > 0)
        f.read(reinterpret_cast<char*>(v.data()),
               static_cast<std::streamsize>(count * sizeof(T)));
}

inline void write_photons(std::ofstream& f, const PhotonSoA& p) {
    uint64_t n = (uint64_t)p.size();
    write_val(f, n);
    write_vec(f, p.pos_x);
    write_vec(f, p.pos_y);
    write_vec(f, p.pos_z);
    write_vec(f, p.wi_x);
    write_vec(f, p.wi_y);
    write_vec(f, p.wi_z);

    // Binary format stores 1 lambda_bin + 1 flux per photon (primary hero).
    // In memory, these arrays may hold HERO_WAVELENGTHS entries per photon
    // (from trace_photons) or just 1 per photon (from a previous load).
    if (n > 0 && p.lambda_bin.size() == n * HERO_WAVELENGTHS && HERO_WAVELENGTHS > 1) {
        std::vector<uint16_t> lb(n);
        std::vector<float>    fl(n);
        for (uint64_t i = 0; i < n; ++i) {
            lb[i] = p.lambda_bin[i * HERO_WAVELENGTHS];
            fl[i] = p.flux       [i * HERO_WAVELENGTHS];
        }
        write_vec(f, lb);
        write_vec(f, fl);
    } else {
        write_vec(f, p.lambda_bin);
        write_vec(f, p.flux);
    }

    write_vec(f, p.norm_x);
    write_vec(f, p.norm_y);
    write_vec(f, p.norm_z);
}

inline bool read_photons(std::ifstream& f, PhotonSoA& p, uint32_t version = PPTD_VERSION) {
    uint64_t n = read_val<uint64_t>(f);
    if (n == 0) { p.clear(); return true; }
    read_vec(f, p.pos_x,      (size_t)n);
    read_vec(f, p.pos_y,      (size_t)n);
    read_vec(f, p.pos_z,      (size_t)n);
    read_vec(f, p.wi_x,       (size_t)n);
    read_vec(f, p.wi_y,       (size_t)n);
    read_vec(f, p.wi_z,       (size_t)n);
    read_vec(f, p.lambda_bin, (size_t)n);
    read_vec(f, p.flux,       (size_t)n);
    if (version >= 2) {
        read_vec(f, p.norm_x, (size_t)n);
        read_vec(f, p.norm_y, (size_t)n);
        read_vec(f, p.norm_z, (size_t)n);
    }

    // ── Convert legacy lambda_bin + flux → spectral_flux ────────────
    // Old format stored one wavelength bin + scalar flux per photon.
    // New v2 code expects spectral_flux (interleaved, NUM_LAMBDA per photon).
    p.spectral_flux.assign(n * NUM_LAMBDA, 0.f);
    for (size_t i = 0; i < n; ++i) {
        int bin = static_cast<int>(p.lambda_bin[i]);
        if (bin >= 0 && bin < NUM_LAMBDA)
            p.spectral_flux[i * NUM_LAMBDA + bin] = p.flux[i];
    }

    return f.good();
}

} // namespace detail

// ── Public API ──────────────────────────────────────────────────────

/// Save photon data + config to a binary file.
/// Returns true on success.
inline bool save_test_data(
    const std::string&  filepath,
    const PhotonSoA&    global_photons,
    const PhotonSoA&    caustic_photons,
    const TestDataHeader& hdr)
{
    std::ofstream f(filepath, std::ios::binary);
    if (!f) {
        std::cerr << "[TestDataIO] Cannot open " << filepath << " for writing\n";
        return false;
    }

    // Header
    detail::write_val(f, hdr.magic);
    detail::write_val(f, hdr.version);
    detail::write_val(f, hdr.num_photons_cfg);
    detail::write_val(f, hdr.gather_radius);
    detail::write_val(f, hdr.caustic_radius);
    detail::write_val(f, hdr.max_bounces);
    detail::write_val(f, hdr.min_bounces_rr);
    detail::write_val(f, hdr.rr_threshold);
    detail::write_val(f, hdr.algo_version);

    uint32_t path_len = (uint32_t)hdr.scene_path.size();
    detail::write_val(f, path_len);
    f.write(hdr.scene_path.data(), path_len);

    // Photons
    detail::write_photons(f, global_photons);
    detail::write_photons(f, caustic_photons);

    f.close();
    if (!f) {
        std::cerr << "[TestDataIO] Error writing " << filepath << "\n";
        return false;
    }

    std::cout << "[TestDataIO] Saved " << filepath << " ("
              << global_photons.size() << " global + "
              << caustic_photons.size() << " caustic photons)\n";
    return true;
}

/// Load photon data + config from a binary file.
/// Returns true on success.
inline bool load_test_data(
    const std::string&  filepath,
    PhotonSoA&          global_photons,
    PhotonSoA&          caustic_photons,
    TestDataHeader&     hdr)
{
    std::ifstream f(filepath, std::ios::binary);
    if (!f) {
        std::cerr << "[TestDataIO] Cannot open " << filepath << " for reading\n";
        return false;
    }

    hdr.magic   = detail::read_val<uint32_t>(f);
    hdr.version = detail::read_val<uint32_t>(f);

    if (hdr.magic != PPTD_MAGIC) {
        std::cerr << "[TestDataIO] Bad magic number (expected 0x"
                  << std::hex << PPTD_MAGIC << ", got 0x" << hdr.magic
                  << std::dec << ")\n";
        return false;
    }
    if (hdr.version != PPTD_VERSION && hdr.version != 2 && hdr.version != 1) {
        std::cerr << "[TestDataIO] Version mismatch (expected "
                  << PPTD_VERSION << " or 1-2, got " << hdr.version << ")\n";
        return false;
    }
    if (hdr.version == 1) {
        std::cout << "[TestDataIO] Loading v1 file (no norm arrays; normal visibility filter disabled)\n";
    }

    hdr.num_photons_cfg = detail::read_val<uint32_t>(f);
    hdr.gather_radius   = detail::read_val<float>(f);
    hdr.caustic_radius  = detail::read_val<float>(f);
    hdr.max_bounces     = detail::read_val<uint32_t>(f);
    hdr.min_bounces_rr  = detail::read_val<uint32_t>(f);
    hdr.rr_threshold    = detail::read_val<float>(f);

    // v3+: read and validate photon algorithm version
    if (hdr.version >= 3) {
        hdr.algo_version = detail::read_val<uint32_t>(f);
        if (hdr.algo_version != PHOTON_ALGORITHM_VERSION) {
            std::cerr << "[TestDataIO] Stale snapshot (algo v"
                      << hdr.algo_version << " vs current v"
                      << PHOTON_ALGORITHM_VERSION << ") — will regenerate\n";
            return false;
        }
    } else {
        // Pre-v3 files have no algo_version → always stale
        std::cerr << "[TestDataIO] Pre-v3 snapshot has no algo_version — will regenerate\n";
        return false;
    }

    uint32_t path_len = detail::read_val<uint32_t>(f);
    hdr.scene_path.resize(path_len);
    f.read(hdr.scene_path.data(), path_len);

    if (!detail::read_photons(f, global_photons,  hdr.version)) return false;
    if (!detail::read_photons(f, caustic_photons, hdr.version)) return false;

    if (!f.good() && !f.eof()) {
        std::cerr << "[TestDataIO] Error reading " << filepath << "\n";
        return false;
    }

    std::cout << "[TestDataIO] Loaded " << filepath << " ("
              << global_photons.size() << " global + "
              << caustic_photons.size() << " caustic photons, scene="
              << hdr.scene_path << ")\n";
    return true;
}
