#pragma once
// ─────────────────────────────────────────────────────────────────────
// photon_io.h – Binary photon map persistence (§20)
// ─────────────────────────────────────────────────────────────────────
// Save / load photon maps to/from binary files with scene hash
// validation.  The binary filename encodes the photon-map parameters
// (budget, bounces, radius) so different parameter sets produce
// different cache files.  This allows reusing expensive photon traces
// across multiple renders of the same scene without recomputing.
//
// File format (v3):
//   - Magic: 4 bytes "PPT\0"
//   - Version: uint32_t (3)
//   - Scene hash: uint64_t (FNV-1a of scene OBJ path + triangle count)
//   - Param hash: uint64_t (FNV-1a of photon config params)
//   - Num photons emitted: uint32_t (config.num_photons)
//   - Global photon count: uint32_t
//   - Caustic photon count: uint32_t
//   - Max bounces: uint32_t
//   - Gather radius: float
//   - Caustic radius: float
//   - Global photon data: SoA arrays
//   - Caustic photon data: SoA arrays
//
// Cache filename pattern:
//   photon_cache_<global>g_<caustic>c_<bounces>b_<radius>.bin
//   Example: photon_cache_1000000g_250000c_4b_0.050r.bin
// ─────────────────────────────────────────────────────────────────────
#include "photon/photon.h"
#include <string>
#include <cstdint>
#include <filesystem>

// ── Photon cache parameters (encoded in filename + validated in header) ──
struct PhotonCacheParams {
    int    global_photon_budget  = 1000000;
    int    caustic_photon_budget = 250000;
    int    max_bounces           = 4;
    float  gather_radius         = 0.05f;
    float  caustic_radius        = 0.02f;

    // Build a deterministic filename encoding these params
    std::string cache_filename() const;

    // Hash of the parameters for in-file validation
    uint64_t param_hash() const;
};

// Compute a scene hash for cache validation.
// Uses FNV-1a on the scene file path and triangle count.
uint64_t compute_scene_hash(const std::string& scene_path,
                            uint32_t triangle_count);

// Build the full cache file path:  <scene_dir>/photon_cache_<params>.bin
std::string photon_cache_path(const std::string& scene_obj_path,
                              const PhotonCacheParams& params);

// Check if a valid cache file exists for this scene + param set.
// Returns true if file exists, magic/version match, and both
// scene_hash and param_hash match.
bool photon_cache_valid(const std::string& filepath,
                        uint64_t scene_hash,
                        const PhotonCacheParams& params);

// Save global and caustic photon maps to a binary file.
// Returns true on success.
bool save_photon_maps(const std::string& filepath,
                      uint64_t scene_hash,
                      const PhotonCacheParams& params,
                      const PhotonSoA& global_photons,
                      const PhotonSoA& caustic_photons);

// Load global and caustic photon maps from a binary file.
// Validates the scene hash and param hash — returns false if mismatch.
// On success, global_photons and caustic_photons are populated.
bool load_photon_maps(const std::string& filepath,
                      uint64_t expected_scene_hash,
                      const PhotonCacheParams& params,
                      PhotonSoA& global_photons,
                      PhotonSoA& caustic_photons);

// ── Legacy overloads (backward compat) ──────────────────────────────
// These wrap the new API with default PhotonCacheParams.
bool save_photon_maps(const std::string& filepath,
                      uint64_t scene_hash,
                      uint32_t num_photons_emitted,
                      const PhotonSoA& global_photons,
                      const PhotonSoA& caustic_photons);

bool load_photon_maps(const std::string& filepath,
                      uint64_t expected_scene_hash,
                      uint32_t& num_photons_emitted,
                      PhotonSoA& global_photons,
                      PhotonSoA& caustic_photons);
