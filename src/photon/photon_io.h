#pragma once
// ─────────────────────────────────────────────────────────────────────
// photon_io.h – Photon map cache save/load (§20)
// ─────────────────────────────────────────────────────────────────────
// Binary format (little-endian):
//   Offset 0   : PhotonCacheHeader (64 bytes)
//   Offset 64  : PhotonSoA arrays in this order:
//                pos_x, pos_y, pos_z, wi_x, wi_y, wi_z,
//                norm_x, norm_y, norm_z, spectral_flux,
//                lambda_bin, flux, num_hero, bin_idx
//   After SoA  : HashGrid vectors: sorted_indices, cell_start, cell_end
// ─────────────────────────────────────────────────────────────────────
#include "photon.h"
#include "hash_grid.h"
#include <string>
#include <cstdint>

// ── Cache file header (exactly 64 bytes) ─────────────────────────────
#pragma pack(push, 1)
struct PhotonCacheHeader {
    uint32_t magic;             // 0x50484F54 = "PHOT"
    uint32_t version;           // format version = 1
    uint64_t scene_hash;        // hash of scene path + render config
    uint64_t num_photons;       // number of photons stored
    int32_t  hero_wavelengths;  // HERO_WAVELENGTHS at save time
    int32_t  num_lambda;        // NUM_LAMBDA at save time
    float    gather_radius;     // gather radius used to build the grid
    float    cell_size;         // hash grid cell_size
    uint32_t table_size;        // hash grid table_size
    uint8_t  reserved[20];      // pad to 64 bytes
};
#pragma pack(pop)

static_assert(sizeof(PhotonCacheHeader) == 64,
              "PhotonCacheHeader must be exactly 64 bytes");

constexpr uint32_t PHOTON_CACHE_MAGIC   = 0x50484F54u; // "PHOT"
constexpr uint32_t PHOTON_CACHE_VERSION = 1u;

// ── API ───────────────────────────────────────────────────────────────

/// Compute a scene hash from the OBJ path, number of triangles, and
/// key config parameters.  Used to validate that the cache matches the
/// current scene and render configuration.
uint64_t compute_scene_hash(const std::string& scene_path,
                            int   num_triangles,
                            int   num_photons,
                            float gather_radius);

/// Return true if the cache file at \p path exists and its header matches
/// \p scene_hash, HERO_WAVELENGTHS, and NUM_LAMBDA.
bool photon_cache_valid(const std::string& path, uint64_t scene_hash);

/// Serialise \p photons and \p grid to \p path.
/// Returns true on success, false on I/O error.
bool save_photon_cache(const std::string& path,
                       const PhotonSoA&  photons,
                       const HashGrid&   grid,
                       uint64_t          scene_hash,
                       float             gather_radius);

/// Deserialise \p photons and \p grid from \p path.
/// \p gather_radius_out receives the value stored in the header.
/// Returns true on success, false if the file is missing, version mismatch,
/// or any data corruption is detected.
bool load_photon_cache(const std::string& path,
                       PhotonSoA&  photons,
                       HashGrid&   grid,
                       float&      gather_radius_out);