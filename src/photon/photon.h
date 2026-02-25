#pragma once
// ─────────────────────────────────────────────────────────────────────
// photon.h – Photon data structure (AoS and SoA layouts)
// ─────────────────────────────────────────────────────────────────────
// Full spectral flux: each photon carries a Spectrum (NUM_LAMBDA bins)
// to eliminate single-wavelength chromatic noise artifacts.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/config.h"
#include "core/spectrum.h"
#include <vector>
#include <cstdint>

// ── Photon path flag bits ────────────────────────────────────────────
constexpr uint8_t PHOTON_FLAG_TRAVERSED_GLASS = 0x01;  // bit 0: passed through ≥1 glass interface
constexpr uint8_t PHOTON_FLAG_CAUSTIC_GLASS   = 0x02;  // bit 1: caustic path starts with glass
constexpr uint8_t PHOTON_FLAG_VOLUME_SEGMENT  = 0x04;  // bit 2: had a volume interaction
constexpr uint8_t PHOTON_FLAG_DISPERSION      = 0x08;  // bit 3: dispersion was active

// ── Single photon (AoS, for host-side convenience) ──────────────────
struct Photon {
    float3   position;
    float3   wi;            // Incoming direction at the hit point (away from surface)
    float3   geom_normal;   // Geometric normal of the surface where the photon was deposited
    Spectrum spectral_flux; // Full spectral radiant flux [W/nm] for all wavelength bins

    // ── GPU hero-wavelength fields ───────────────────────────────────
    // The GPU emitter writes HERO_WAVELENGTHS wavelength bins per photon.
    // CPU emitter fills spectral_flux; these are kept in sync by the
    // SoA push_back / get helpers so both pipelines work.
    uint16_t lambda_bin[HERO_WAVELENGTHS] = {};  // Wavelength bin indices (GPU path)
    float    flux[HERO_WAVELENGTHS]       = {};  // Scalar flux per hero bin (GPU path)
    int      num_hero = 1;                       // number of valid hero channels (1..HERO_WAVELENGTHS)

    // ── Light source tracking (for NEE light importance cache) ───────
    // Local index into emissive_tri_indices[] identifying which light
    // emitted/caused this photon.  0xFFFF = unknown/unset.
    uint16_t source_emissive_idx = 0xFFFFu;

    // ── Hit triangle ─────────────────────────────────────────────────────
    uint32_t triangle_id = 0xFFFFFFFFu; // scene triangle index at deposit

    // ── Path metadata ────────────────────────────────────────────────────
    uint8_t  path_flags   = 0;     // PHOTON_FLAG_* bit field
    uint8_t  bounce_count = 0;     // total bounces at deposit
};

// ── SoA layout for GPU storage ──────────────────────────────────────
// Each array has length num_photons.
// Spectral flux is stored interleaved: spectral_flux[i * NUM_LAMBDA + b].
struct PhotonSoA {
    // Position (x, y, z)
    std::vector<float> pos_x;
    std::vector<float> pos_y;
    std::vector<float> pos_z;

    // Incoming direction
    std::vector<float> wi_x;
    std::vector<float> wi_y;
    std::vector<float> wi_z;

    // Geometric surface normal at the photon hit
    std::vector<float> norm_x;
    std::vector<float> norm_y;
    std::vector<float> norm_z;

    // Full spectral flux: interleaved [photon_0_bin_0, ..., photon_0_bin_N, photon_1_bin_0, ...]
    std::vector<float> spectral_flux;

    // ── GPU hero-wavelength fields (HERO_WAVELENGTHS per photon) ────
    // GPU emitter produces HERO_WAVELENGTHS wavelength bins + fluxes
    // per photon.  Interleaved: [photon0_hero0, photon0_hero1, ...,
    //                            photon1_hero0, photon1_hero1, ...]
    // CPU emitter fills spectral_flux; these are maintained in parallel
    // so that both the v2 CPU renderer and the GPU renderer work.
    std::vector<uint16_t> lambda_bin;   // [N * HERO_WAVELENGTHS] bin indices
    std::vector<float>    flux;         // [N * HERO_WAVELENGTHS] per-hero flux
    std::vector<uint8_t>  num_hero;     // [N] valid hero count per photon

    // Source emissive triangle index (for NEE light importance cache)
    // 0xFFFF = unknown/unset.
    std::vector<uint16_t> source_emissive_idx;

    // Hit triangle index (scene tri ID at photon deposit site).
    // 0xFFFFFFFF = unknown/unset.
    std::vector<uint32_t> tri_id;

    // Path metadata
    std::vector<uint8_t>  path_flags;      // PHOTON_FLAG_* bit field
    std::vector<uint8_t>  bounce_count;     // total bounces at deposit

    // Directional bin index (for cell-bin grid)
    std::vector<uint8_t>  bin_idx;          // precomputed directional bin

    size_t size() const { return pos_x.size(); }

    // ── Spectral flux accessors ─────────────────────────────────────
    Spectrum get_flux(size_t i) const {
        Spectrum s;
        const float* base = &spectral_flux[i * NUM_LAMBDA];
        for (int b = 0; b < NUM_LAMBDA; ++b) s.value[b] = base[b];
        return s;
    }

    void set_flux(size_t i, const Spectrum& s) {
        float* base = &spectral_flux[i * NUM_LAMBDA];
        for (int b = 0; b < NUM_LAMBDA; ++b) base[b] = s.value[b];
    }

    float total_flux(size_t i) const {
        float sum = 0.f;
        const float* base = &spectral_flux[i * NUM_LAMBDA];
        for (int b = 0; b < NUM_LAMBDA; ++b) sum += base[b];
        return sum;
    }

    void reserve(size_t n) {
        pos_x.reserve(n);      pos_y.reserve(n);      pos_z.reserve(n);
        wi_x.reserve(n);       wi_y.reserve(n);       wi_z.reserve(n);
        norm_x.reserve(n);     norm_y.reserve(n);     norm_z.reserve(n);
        spectral_flux.reserve(n * NUM_LAMBDA);
        lambda_bin.reserve(n * HERO_WAVELENGTHS);
        flux.reserve(n * HERO_WAVELENGTHS);
        num_hero.reserve(n);
        source_emissive_idx.reserve(n);
        tri_id.reserve(n);
        path_flags.reserve(n);
        bounce_count.reserve(n);
        bin_idx.reserve(n);
    }

    // Append all photons from another PhotonSoA into this one.
    void append(const PhotonSoA& other) {
        if (other.size() == 0) return;
        pos_x.insert(pos_x.end(), other.pos_x.begin(), other.pos_x.end());
        pos_y.insert(pos_y.end(), other.pos_y.begin(), other.pos_y.end());
        pos_z.insert(pos_z.end(), other.pos_z.begin(), other.pos_z.end());
        wi_x.insert(wi_x.end(), other.wi_x.begin(), other.wi_x.end());
        wi_y.insert(wi_y.end(), other.wi_y.begin(), other.wi_y.end());
        wi_z.insert(wi_z.end(), other.wi_z.begin(), other.wi_z.end());
        norm_x.insert(norm_x.end(), other.norm_x.begin(), other.norm_x.end());
        norm_y.insert(norm_y.end(), other.norm_y.begin(), other.norm_y.end());
        norm_z.insert(norm_z.end(), other.norm_z.begin(), other.norm_z.end());
        spectral_flux.insert(spectral_flux.end(), other.spectral_flux.begin(), other.spectral_flux.end());
        lambda_bin.insert(lambda_bin.end(), other.lambda_bin.begin(), other.lambda_bin.end());
        flux.insert(flux.end(), other.flux.begin(), other.flux.end());
        num_hero.insert(num_hero.end(), other.num_hero.begin(), other.num_hero.end());
        source_emissive_idx.insert(source_emissive_idx.end(), other.source_emissive_idx.begin(), other.source_emissive_idx.end());
        tri_id.insert(tri_id.end(), other.tri_id.begin(), other.tri_id.end());
        path_flags.insert(path_flags.end(), other.path_flags.begin(), other.path_flags.end());
        bounce_count.insert(bounce_count.end(), other.bounce_count.begin(), other.bounce_count.end());
        bin_idx.insert(bin_idx.end(), other.bin_idx.begin(), other.bin_idx.end());
    }

    void resize(size_t n) {
        pos_x.resize(n);       pos_y.resize(n);       pos_z.resize(n);
        wi_x.resize(n);        wi_y.resize(n);        wi_z.resize(n);
        norm_x.resize(n);      norm_y.resize(n);      norm_z.resize(n);
        spectral_flux.resize(n * NUM_LAMBDA);
        lambda_bin.resize(n * HERO_WAVELENGTHS);
        flux.resize(n * HERO_WAVELENGTHS);
        num_hero.resize(n, 1);
        source_emissive_idx.resize(n, 0xFFFFu);
        tri_id.resize(n, 0xFFFFFFFFu);
        path_flags.resize(n, 0);
        bounce_count.resize(n, 0);
        bin_idx.resize(n, 0);
    }

    void push_back(const Photon& p) {
        pos_x.push_back(p.position.x);
        pos_y.push_back(p.position.y);
        pos_z.push_back(p.position.z);
        wi_x.push_back(p.wi.x);
        wi_y.push_back(p.wi.y);
        wi_z.push_back(p.wi.z);
        norm_x.push_back(p.geom_normal.x);
        norm_y.push_back(p.geom_normal.y);
        norm_z.push_back(p.geom_normal.z);
        for (int b = 0; b < NUM_LAMBDA; ++b)
            spectral_flux.push_back(p.spectral_flux.value[b]);
        for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
            lambda_bin.push_back(p.lambda_bin[h]);
            flux.push_back(p.flux[h]);
        }
        num_hero.push_back((uint8_t)p.num_hero);
        source_emissive_idx.push_back(p.source_emissive_idx);
        tri_id.push_back(p.triangle_id);
        path_flags.push_back(p.path_flags);
        bounce_count.push_back(p.bounce_count);
        bin_idx.push_back(0);  // default bin; caller can overwrite
    }

    Photon get(size_t i) const {
        Photon p;
        p.position      = make_f3(pos_x[i],  pos_y[i],  pos_z[i]);
        p.wi            = make_f3(wi_x[i],   wi_y[i],   wi_z[i]);
        p.geom_normal   = make_f3(norm_x[i], norm_y[i], norm_z[i]);
        p.spectral_flux = get_flux(i);
        p.num_hero      = (num_hero.size() > i) ? (int)num_hero[i] : 1;
        for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
            size_t idx = i * HERO_WAVELENGTHS + h;
            p.lambda_bin[h] = (lambda_bin.size() > idx) ? lambda_bin[idx] : 0;
            p.flux[h]       = (flux.size() > idx) ? flux[idx] : 0.f;
        }
        p.source_emissive_idx = (source_emissive_idx.size() > i)
                              ? source_emissive_idx[i] : (uint16_t)0xFFFFu;
        p.triangle_id  = (tri_id.size() > i)        ? tri_id[i]        : 0xFFFFFFFFu;
        p.path_flags   = (path_flags.size() > i)   ? path_flags[i]   : (uint8_t)0;
        p.bounce_count = (bounce_count.size() > i) ? bounce_count[i] : (uint8_t)0;
        return p;
    }

    void clear() {
        pos_x.clear();  pos_y.clear();  pos_z.clear();
        wi_x.clear();   wi_y.clear();   wi_z.clear();
        norm_x.clear(); norm_y.clear(); norm_z.clear();
        spectral_flux.clear();
        lambda_bin.clear();
        flux.clear();
        num_hero.clear();
        source_emissive_idx.clear();
        tri_id.clear();
        path_flags.clear();
        bounce_count.clear();
        bin_idx.clear();
    }
};

// ── Photon maps ─────────────────────────────────────────────────────
enum class PhotonMapType {
    Global,     // Indirect diffuse illumination
    Caustic     // Specular-to-diffuse caustic illumination
};
