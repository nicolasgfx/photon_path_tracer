// ─────────────────────────────────────────────────────────────────────
// photon_io.cpp – Binary photon map persistence (§20)
// ─────────────────────────────────────────────────────────────────────
#include "photon/photon_io.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <cstdio>
#include <filesystem>

namespace fs = std::filesystem;

// ── File format constants ───────────────────────────────────────────
static constexpr char     MAGIC[4]       = {'P', 'P', 'T', '\0'};
static constexpr uint32_t FORMAT_VERSION = 3;  // v3: adds param_hash

// ── FNV-1a hash ─────────────────────────────────────────────────────
static uint64_t fnv1a_64(const void* data, size_t len) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    uint64_t hash = 14695981039346656037ULL;  // FNV offset basis
    for (size_t i = 0; i < len; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;  // FNV prime
    }
    return hash;
}

static uint64_t fnv1a_combine(uint64_t h, const void* data, size_t len) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) {
        h ^= bytes[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// ── PhotonCacheParams ───────────────────────────────────────────────

std::string PhotonCacheParams::cache_filename() const {
    // Pattern: photon_cache_<global>g_<caustic>c_<bounces>b_<radius>r.bin
    char buf[256];
    snprintf(buf, sizeof(buf), "photon_cache_%dg_%dc_%db_%.3fr.bin",
             global_photon_budget, caustic_photon_budget,
             max_bounces, gather_radius);
    return std::string(buf);
}

uint64_t PhotonCacheParams::param_hash() const {
    uint64_t h = 14695981039346656037ULL;
    h = fnv1a_combine(h, &global_photon_budget, sizeof(global_photon_budget));
    h = fnv1a_combine(h, &caustic_photon_budget, sizeof(caustic_photon_budget));
    h = fnv1a_combine(h, &max_bounces, sizeof(max_bounces));
    h = fnv1a_combine(h, &gather_radius, sizeof(gather_radius));
    h = fnv1a_combine(h, &caustic_radius, sizeof(caustic_radius));
    return h;
}

// ── Scene hash ──────────────────────────────────────────────────────

uint64_t compute_scene_hash(const std::string& scene_path,
                            uint32_t triangle_count) {
    uint64_t h = fnv1a_64(scene_path.data(), scene_path.size());
    h ^= fnv1a_64(&triangle_count, sizeof(triangle_count));
    h *= 1099511628211ULL;
    return h;
}

// ── Cache path builder ──────────────────────────────────────────────

std::string photon_cache_path(const std::string& scene_obj_path,
                              const PhotonCacheParams& params) {
    // Place cache file alongside the .obj in the scene folder
    fs::path obj(scene_obj_path);
    fs::path scene_dir = obj.parent_path();
    return (scene_dir / params.cache_filename()).string();
}

// ── Write/Read helpers ──────────────────────────────────────────────

template<typename T>
static bool write_val(std::ofstream& f, const T& v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(T));
    return f.good();
}

template<typename T>
static bool write_vec(std::ofstream& f, const std::vector<T>& v) {
    if (v.empty()) return true;
    f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
    return f.good();
}

template<typename T>
static bool read_val(std::ifstream& f, T& v) {
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    return f.good();
}

template<typename T>
static bool read_vec(std::ifstream& f, std::vector<T>& v, size_t count) {
    v.resize(count);
    if (count == 0) return true;
    f.read(reinterpret_cast<char*>(v.data()), count * sizeof(T));
    return f.good();
}

// ── Write/Read one photon map ───────────────────────────────────────

static bool write_photon_soa(std::ofstream& f, const PhotonSoA& p) {
    return write_vec(f, p.pos_x)     && write_vec(f, p.pos_y)     &&
           write_vec(f, p.pos_z)     && write_vec(f, p.wi_x)      &&
           write_vec(f, p.wi_y)      && write_vec(f, p.wi_z)      &&
           write_vec(f, p.norm_x)    && write_vec(f, p.norm_y)    &&
           write_vec(f, p.norm_z)    &&
           write_vec(f, p.spectral_flux);
}

static bool read_photon_soa(std::ifstream& f, PhotonSoA& p, size_t count) {
    p.clear();
    if (count == 0) return true;
    return read_vec(f, p.pos_x, count)     && read_vec(f, p.pos_y, count)     &&
           read_vec(f, p.pos_z, count)     && read_vec(f, p.wi_x, count)      &&
           read_vec(f, p.wi_y, count)      && read_vec(f, p.wi_z, count)      &&
           read_vec(f, p.norm_x, count)    && read_vec(f, p.norm_y, count)    &&
           read_vec(f, p.norm_z, count)    &&
           read_vec(f, p.spectral_flux, count * NUM_LAMBDA);
}

// ── Validate cache ──────────────────────────────────────────────────

bool photon_cache_valid(const std::string& filepath,
                        uint64_t scene_hash,
                        const PhotonCacheParams& params) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open()) return false;

    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, MAGIC, 4) != 0) return false;

    uint32_t version;
    if (!read_val(f, version) || version != FORMAT_VERSION) return false;

    uint64_t stored_scene_hash;
    if (!read_val(f, stored_scene_hash)) return false;
    if (stored_scene_hash != scene_hash) return false;

    uint64_t stored_param_hash;
    if (!read_val(f, stored_param_hash)) return false;
    if (stored_param_hash != params.param_hash()) return false;

    return true;
}

// ── Save (v3 format) ────────────────────────────────────────────────

bool save_photon_maps(const std::string& filepath,
                      uint64_t scene_hash,
                      const PhotonCacheParams& params,
                      const PhotonSoA& global_photons,
                      const PhotonSoA& caustic_photons)
{
    // Ensure parent directory exists
    fs::path p(filepath);
    if (p.has_parent_path())
        fs::create_directories(p.parent_path());

    std::ofstream f(filepath, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[PhotonIO] Failed to open " << filepath << " for writing\n";
        return false;
    }

    // Header
    f.write(MAGIC, 4);
    if (!write_val(f, FORMAT_VERSION))              return false;
    if (!write_val(f, scene_hash))                  return false;
    uint64_t ph = params.param_hash();
    if (!write_val(f, ph))                          return false;
    uint32_t emitted = (uint32_t)params.global_photon_budget;
    if (!write_val(f, emitted))                     return false;

    uint32_t n_global  = (uint32_t)global_photons.size();
    uint32_t n_caustic = (uint32_t)caustic_photons.size();
    if (!write_val(f, n_global))                    return false;
    if (!write_val(f, n_caustic))                   return false;

    // Store params for informational purposes
    uint32_t max_b = (uint32_t)params.max_bounces;
    if (!write_val(f, max_b))                       return false;
    if (!write_val(f, params.gather_radius))        return false;
    if (!write_val(f, params.caustic_radius))       return false;

    // Photon data
    if (!write_photon_soa(f, global_photons))       return false;
    if (!write_photon_soa(f, caustic_photons))      return false;

    std::cout << "[PhotonIO] Saved " << n_global << " global + "
              << n_caustic << " caustic photons to " << filepath << "\n";
    return true;
}

// ── Load (v3 format) ────────────────────────────────────────────────

bool load_photon_maps(const std::string& filepath,
                      uint64_t expected_scene_hash,
                      const PhotonCacheParams& params,
                      PhotonSoA& global_photons,
                      PhotonSoA& caustic_photons)
{
    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open()) {
        return false;  // No cache file — not an error
    }

    // Validate magic
    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, MAGIC, 4) != 0) {
        std::cerr << "[PhotonIO] Invalid file format: " << filepath << "\n";
        return false;
    }

    // Validate version
    uint32_t version;
    if (!read_val(f, version)) return false;
    if (version < 3) {
        std::cerr << "[PhotonIO] Old v" << version
                  << " photon cache — re-tracing required\n";
        return false;
    }
    if (version != FORMAT_VERSION) {
        std::cerr << "[PhotonIO] Unsupported version " << version
                  << " (expected " << FORMAT_VERSION << ")\n";
        return false;
    }

    // Validate scene hash
    uint64_t stored_hash;
    if (!read_val(f, stored_hash)) return false;
    if (stored_hash != expected_scene_hash) {
        std::cerr << "[PhotonIO] Scene hash mismatch — photon file is stale\n";
        return false;
    }

    // Validate param hash
    uint64_t stored_param_hash;
    if (!read_val(f, stored_param_hash)) return false;
    if (stored_param_hash != params.param_hash()) {
        std::cerr << "[PhotonIO] Param hash mismatch — different config\n";
        return false;
    }

    uint32_t emitted;
    if (!read_val(f, emitted)) return false;

    uint32_t n_global, n_caustic;
    if (!read_val(f, n_global))  return false;
    if (!read_val(f, n_caustic)) return false;

    // Read stored params (informational, already validated via hash)
    uint32_t max_b; float gr, cr;
    if (!read_val(f, max_b)) return false;
    if (!read_val(f, gr))    return false;
    if (!read_val(f, cr))    return false;

    // Sanity check
    if (n_global > 100000000 || n_caustic > 100000000) {
        std::cerr << "[PhotonIO] Unreasonable photon counts: "
                  << n_global << " / " << n_caustic << "\n";
        return false;
    }

    if (!read_photon_soa(f, global_photons, n_global))    return false;
    if (!read_photon_soa(f, caustic_photons, n_caustic))  return false;

    std::cout << "[PhotonIO] Loaded " << n_global << " global + "
              << n_caustic << " caustic photons from " << filepath << "\n";
    return true;
}

// ── Legacy overloads (backward compat) ──────────────────────────────

bool save_photon_maps(const std::string& filepath,
                      uint64_t scene_hash,
                      uint32_t num_photons_emitted,
                      const PhotonSoA& global_photons,
                      const PhotonSoA& caustic_photons)
{
    PhotonCacheParams p;
    p.global_photon_budget = (int)num_photons_emitted;
    return save_photon_maps(filepath, scene_hash, p,
                            global_photons, caustic_photons);
}

bool load_photon_maps(const std::string& filepath,
                      uint64_t expected_scene_hash,
                      uint32_t& num_photons_emitted,
                      PhotonSoA& global_photons,
                      PhotonSoA& caustic_photons)
{
    PhotonCacheParams p;
    bool ok = load_photon_maps(filepath, expected_scene_hash, p,
                               global_photons, caustic_photons);
    if (ok)
        num_photons_emitted = (uint32_t)p.global_photon_budget;
    return ok;
}
