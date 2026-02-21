#pragma once
// ─────────────────────────────────────────────────────────────────────
// photon.h – Photon data structure (AoS and SoA layouts)
// ─────────────────────────────────────────────────────────────────────
// Full spectral flux: each photon carries a Spectrum (NUM_LAMBDA bins)
// to eliminate single-wavelength chromatic noise artifacts.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/spectrum.h"
#include <vector>
#include <cstdint>

// ── Single photon (AoS, for host-side convenience) ──────────────────
struct Photon {
    float3   position;
    float3   wi;            // Incoming direction at the hit point (away from surface)
    float3   geom_normal;   // Geometric normal of the surface where the photon was deposited
    Spectrum spectral_flux; // Full spectral radiant flux [W/nm] for all wavelength bins

    // ── Legacy fields (GPU pipeline & compat) ───────────────────────
    // The GPU emitter writes a single wavelength bin per photon.
    // CPU emitter fills spectral_flux; these are kept in sync by the
    // SoA push_back / get helpers so both pipelines work.
    uint16_t lambda_bin = 0;  // Wavelength bin index (GPU path)
    float    flux       = 0.f;// Scalar flux for that bin (GPU path)
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

    // ── Legacy per-photon fields (GPU pipeline) ─────────────────────
    // GPU emitter produces one wavelength bin + scalar flux per photon.
    // CPU emitter fills spectral_flux; these are maintained in parallel
    // so that both the v2 CPU renderer and the GPU renderer work.
    std::vector<uint16_t> lambda_bin;   // Wavelength bin index per photon
    std::vector<float>    flux;         // Scalar flux per photon

    // Precomputed directional bin index (Fibonacci sphere nearest bin)
    // Computed on CPU after photon trace, used on device for O(1) bin lookup.
    std::vector<uint8_t> bin_idx;

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
        lambda_bin.reserve(n);
        flux.reserve(n);
        bin_idx.reserve(n);
    }

    void resize(size_t n) {
        pos_x.resize(n);       pos_y.resize(n);       pos_z.resize(n);
        wi_x.resize(n);        wi_y.resize(n);        wi_z.resize(n);
        norm_x.resize(n);      norm_y.resize(n);      norm_z.resize(n);
        spectral_flux.resize(n * NUM_LAMBDA);
        lambda_bin.resize(n);
        flux.resize(n);
        bin_idx.resize(n);
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
        lambda_bin.push_back(p.lambda_bin);
        flux.push_back(p.flux);
    }

    Photon get(size_t i) const {
        Photon p;
        p.position      = make_f3(pos_x[i],  pos_y[i],  pos_z[i]);
        p.wi            = make_f3(wi_x[i],   wi_y[i],   wi_z[i]);
        p.geom_normal   = make_f3(norm_x[i], norm_y[i], norm_z[i]);
        p.spectral_flux = get_flux(i);
        p.lambda_bin    = lambda_bin.size() > i ? lambda_bin[i] : 0;
        p.flux          = flux.size() > i ? flux[i] : 0.f;
        return p;
    }

    void clear() {
        pos_x.clear();  pos_y.clear();  pos_z.clear();
        wi_x.clear();   wi_y.clear();   wi_z.clear();
        norm_x.clear(); norm_y.clear(); norm_z.clear();
        spectral_flux.clear();
        lambda_bin.clear();
        flux.clear();
        bin_idx.clear();
    }
};

// ── Photon maps ─────────────────────────────────────────────────────
enum class PhotonMapType {
    Global,     // Indirect diffuse illumination
    Caustic     // Specular-to-diffuse caustic illumination
};
