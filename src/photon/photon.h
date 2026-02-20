#pragma once
// ─────────────────────────────────────────────────────────────────────
// photon.h – Photon data structure (AoS and SoA layouts)
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include <vector>
#include <cstdint>

// ── Single photon (AoS, for host-side convenience) ──────────────────
struct Photon {
    float3   position;
    float3   wi;            // Incoming direction at the hit point
    uint16_t lambda_bin;    // Wavelength bin index [0, NUM_LAMBDA)
    float    flux;          // Radiant flux for that bin [W/nm]
};

// ── SoA layout for GPU storage ──────────────────────────────────────
// Each array has length num_photons.
struct PhotonSoA {
    // Position (x, y, z)
    std::vector<float> pos_x;
    std::vector<float> pos_y;
    std::vector<float> pos_z;

    // Incoming direction
    std::vector<float> wi_x;
    std::vector<float> wi_y;
    std::vector<float> wi_z;

    // Wavelength bin
    std::vector<uint16_t> lambda_bin;

    // Flux
    std::vector<float> flux;

    // Precomputed directional bin index (Fibonacci sphere nearest bin)
    // Computed on CPU after photon trace, used on device for O(1) bin lookup.
    std::vector<uint8_t> bin_idx;

    size_t size() const { return pos_x.size(); }

    void reserve(size_t n) {
        pos_x.reserve(n);      pos_y.reserve(n);      pos_z.reserve(n);
        wi_x.reserve(n);       wi_y.reserve(n);       wi_z.reserve(n);
        lambda_bin.reserve(n);
        flux.reserve(n);
        bin_idx.reserve(n);
    }

    void resize(size_t n) {
        pos_x.resize(n);       pos_y.resize(n);       pos_z.resize(n);
        wi_x.resize(n);        wi_y.resize(n);        wi_z.resize(n);
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
        lambda_bin.push_back(p.lambda_bin);
        flux.push_back(p.flux);
    }

    Photon get(size_t i) const {
        Photon p;
        p.position = make_f3(pos_x[i], pos_y[i], pos_z[i]);
        p.wi       = make_f3(wi_x[i],  wi_y[i],  wi_z[i]);
        p.lambda_bin = lambda_bin[i];
        p.flux     = flux[i];
        return p;
    }

    void clear() {
        pos_x.clear();  pos_y.clear();  pos_z.clear();
        wi_x.clear();   wi_y.clear();   wi_z.clear();
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
