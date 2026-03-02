// photon_io.cpp – Photon map cache save / load (§20)
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS  // suppress fopen/fread/fwrite MSVC deprecation
#endif
#include "photon_io.h"
#include <cstdio>
#include <cstring>
#include <iostream>

// ── Helpers ──────────────────────────────────────────────────────────

// FNV-1a 64-bit hash (fast, good distribution for file-name keys)
static uint64_t fnv1a_64(const void* data, size_t len, uint64_t seed = 0xcbf29ce484222325ull) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint64_t h = seed;
    for (size_t i = 0; i < len; ++i) {
        h ^= (uint64_t)p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

template<typename T>
static bool write_vec(FILE* f, const std::vector<T>& v) {
    if (v.empty()) return true;
    return fwrite(v.data(), sizeof(T), v.size(), f) == v.size();
}

template<typename T>
static bool read_vec(FILE* f, std::vector<T>& v, size_t count) {
    v.resize(count);
    if (count == 0) return true;
    return fread(v.data(), sizeof(T), count, f) == count;
}

// ── Public API ────────────────────────────────────────────────────────

uint64_t compute_scene_hash(const std::string& scene_path,
                            int   num_triangles,
                            int   num_photons,
                            float gather_radius)
{
    uint64_t h = fnv1a_64(scene_path.data(), scene_path.size());
    h = fnv1a_64(&num_triangles,  sizeof(int),   h);
    h = fnv1a_64(&num_photons,    sizeof(int),   h);
    h = fnv1a_64(&gather_radius,  sizeof(float), h);
    // Bake compile-time constants so cache is invalidated if recompiled
    // with different spectral resolution.
    int nl = NUM_LAMBDA;
    int nh = HERO_WAVELENGTHS;
    h = fnv1a_64(&nl, sizeof(int), h);
    h = fnv1a_64(&nh, sizeof(int), h);
    return h;
}

bool photon_cache_valid(const std::string& path, uint64_t scene_hash) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    PhotonCacheHeader hdr;
    bool ok = (fread(&hdr, sizeof(hdr), 1, f) == 1);
    fclose(f);

    if (!ok)                                return false;
    if (hdr.magic   != PHOTON_CACHE_MAGIC)  return false;
    if (hdr.version < 2u || hdr.version > PHOTON_CACHE_VERSION) return false;
    if (hdr.scene_hash != scene_hash)        return false;
    if (hdr.hero_wavelengths != HERO_WAVELENGTHS) return false;
    if (hdr.num_lambda != NUM_LAMBDA)        return false;
    return true;
}

bool save_photon_cache(const std::string& path,
                       const PhotonSoA&  photons,
                       const HashGrid&   grid,
                       uint64_t          scene_hash,
                       float             gather_radius)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        std::cerr << "[photon_io] Cannot open for write: " << path << "\n";
        return false;
    }

    const uint64_t N = (uint64_t)photons.size();

    // Write header
    PhotonCacheHeader hdr{};
    hdr.magic            = PHOTON_CACHE_MAGIC;
    hdr.version          = PHOTON_CACHE_VERSION;
    hdr.scene_hash       = scene_hash;
    hdr.num_photons      = N;
    hdr.hero_wavelengths = HERO_WAVELENGTHS;
    hdr.num_lambda       = NUM_LAMBDA;
    hdr.gather_radius    = gather_radius;
    hdr.cell_size        = grid.cell_size;
    hdr.table_size       = grid.table_size;
    hdr.has_path_data    = 1u;  // v3: path metadata present

    bool ok = (fwrite(&hdr, sizeof(hdr), 1, f) == 1);

    // Write SoA photon arrays
    ok = ok && write_vec(f, photons.pos_x);
    ok = ok && write_vec(f, photons.pos_y);
    ok = ok && write_vec(f, photons.pos_z);
    ok = ok && write_vec(f, photons.wi_x);
    ok = ok && write_vec(f, photons.wi_y);
    ok = ok && write_vec(f, photons.wi_z);
    ok = ok && write_vec(f, photons.norm_x);
    ok = ok && write_vec(f, photons.norm_y);
    ok = ok && write_vec(f, photons.norm_z);
    ok = ok && write_vec(f, photons.spectral_flux);
    ok = ok && write_vec(f, photons.lambda_bin);
    ok = ok && write_vec(f, photons.flux);
    ok = ok && write_vec(f, photons.num_hero);
    ok = ok && write_vec(f, photons.bin_idx);
    ok = ok && write_vec(f, photons.source_emissive_idx);

    // Write hash grid arrays
    ok = ok && write_vec(f, grid.sorted_indices);
    ok = ok && write_vec(f, grid.cell_start);
    ok = ok && write_vec(f, grid.cell_end);

    // v3: path metadata
    ok = ok && write_vec(f, photons.path_flags);
    ok = ok && write_vec(f, photons.bounce_count);
    ok = ok && write_vec(f, photons.tri_id);

    fclose(f);

    if (!ok) {
        std::cerr << "[photon_io] Write error: " << path << "\n";
        return false;
    }

    std::cout << "[photon_io] Saved " << N << " photons to " << path << "\n";
    return true;
}

bool load_photon_cache(const std::string& path,
                       PhotonSoA&  photons,
                       HashGrid&   grid,
                       float&      gather_radius_out)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        std::cerr << "[photon_io] Cache not found: " << path << "\n";
        return false;
    }

    PhotonCacheHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fclose(f);
        std::cerr << "[photon_io] Failed to read header: " << path << "\n";
        return false;
    }

    if (hdr.magic != PHOTON_CACHE_MAGIC ||
        hdr.version < 2u || hdr.version > PHOTON_CACHE_VERSION ||
        hdr.hero_wavelengths != HERO_WAVELENGTHS || hdr.num_lambda != NUM_LAMBDA) {
        fclose(f);
        std::cerr << "[photon_io] Header mismatch (magic/version/spectral): " << path << "\n";
        return false;
    }

    const size_t N  = (size_t)hdr.num_photons;
    const size_t TS = (size_t)hdr.table_size;

    gather_radius_out    = hdr.gather_radius;
    grid.cell_size       = hdr.cell_size;
    grid.table_size      = hdr.table_size;

    bool ok = true;
    ok = ok && read_vec(f, photons.pos_x,         N);
    ok = ok && read_vec(f, photons.pos_y,         N);
    ok = ok && read_vec(f, photons.pos_z,         N);
    ok = ok && read_vec(f, photons.wi_x,          N);
    ok = ok && read_vec(f, photons.wi_y,          N);
    ok = ok && read_vec(f, photons.wi_z,          N);
    ok = ok && read_vec(f, photons.norm_x,        N);
    ok = ok && read_vec(f, photons.norm_y,        N);
    ok = ok && read_vec(f, photons.norm_z,        N);
    ok = ok && read_vec(f, photons.spectral_flux, N * (size_t)NUM_LAMBDA);
    ok = ok && read_vec(f, photons.lambda_bin,    N * (size_t)HERO_WAVELENGTHS);
    ok = ok && read_vec(f, photons.flux,          N * (size_t)HERO_WAVELENGTHS);
    ok = ok && read_vec(f, photons.num_hero,      N);
    ok = ok && read_vec(f, photons.bin_idx,       N);
    ok = ok && read_vec(f, photons.source_emissive_idx, N);
    ok = ok && read_vec(f, grid.sorted_indices,   N);
    ok = ok && read_vec(f, grid.cell_start,       TS);
    ok = ok && read_vec(f, grid.cell_end,         TS);

    // v3: path metadata
    if (ok && hdr.version >= 3u && hdr.has_path_data) {
        ok = ok && read_vec(f, photons.path_flags,   N);
        ok = ok && read_vec(f, photons.bounce_count, N);
        ok = ok && read_vec(f, photons.tri_id,       N);
    } else {
        // Pre-v3 cache: fill defaults
        photons.path_flags.assign(N, 0u);
        photons.bounce_count.assign(N, 0u);
        photons.tri_id.assign(N, 0xFFFFFFFFu);
    }

    fclose(f);

    if (!ok) {
        std::cerr << "[photon_io] Read error in photon data: " << path << "\n";
        photons.clear();
        return false;
    }

    std::cout << "[photon_io] Loaded " << N << " photons from " << path << "\n";
    return true;
}
