#pragma once
// ─────────────────────────────────────────────────────────────────────
// alias_table.h – Vose's Alias Method for O(1) discrete sampling
// ─────────────────────────────────────────────────────────────────────
// Used for:
//   - Emissive triangle selection
//   - Wavelength bin selection
//   - Photon-guided directional sampling
// ─────────────────────────────────────────────────────────────────────
#include "types.h"
#include <vector>
#include <numeric>
#include <algorithm>

struct AliasEntry {
    float    prob;      // probability of choosing this entry directly
    uint32_t alias;     // redirect index
};

struct AliasTable {
    std::vector<AliasEntry> entries;
    std::vector<float>      pdf_values; // original normalized PDF
    float                   total_weight;
    int                     n;

    // Build from unnormalized weights
    static AliasTable build(const std::vector<float>& weights) {
        AliasTable table;
        table.n = (int)weights.size();
        if (table.n == 0) return table;

        table.total_weight = 0.f;
        for (float w : weights) table.total_weight += w;

        table.pdf_values.resize(table.n);
        table.entries.resize(table.n);

        float inv_total = (table.total_weight > 0.f) ? 1.f / table.total_weight : 0.f;
        std::vector<float> scaled(table.n);
        for (int i = 0; i < table.n; ++i) {
            table.pdf_values[i] = weights[i] * inv_total;
            scaled[i] = table.pdf_values[i] * table.n;
        }

        // Classify into small/large
        std::vector<int> small, large;
        small.reserve(table.n);
        large.reserve(table.n);
        for (int i = 0; i < table.n; ++i) {
            if (scaled[i] < 1.0f) small.push_back(i);
            else                  large.push_back(i);
        }

        // Build alias table (Vose's algorithm)
        while (!small.empty() && !large.empty()) {
            int s = small.back(); small.pop_back();
            int l = large.back(); large.pop_back();

            table.entries[s].prob  = scaled[s];
            table.entries[s].alias = l;

            scaled[l] = (scaled[l] + scaled[s]) - 1.0f;
            if (scaled[l] < 1.0f) small.push_back(l);
            else                  large.push_back(l);
        }

        // Handle numerical imprecision
        while (!large.empty()) {
            int l = large.back(); large.pop_back();
            table.entries[l].prob  = 1.0f;
            table.entries[l].alias = l;
        }
        while (!small.empty()) {
            int s = small.back(); small.pop_back();
            table.entries[s].prob  = 1.0f;
            table.entries[s].alias = s;
        }

        return table;
    }

    // Sample: returns chosen index
    // u1, u2 are uniform [0,1)
    int sample(float u1, float u2) const {
        int idx = (int)(u1 * n);
        idx = std::min(idx, n - 1);
        if (u2 < entries[idx].prob) return idx;
        return entries[idx].alias;
    }

    // PDF of returning index i
    float pdf(int i) const {
        if (i < 0 || i >= n) return 0.f;
        return pdf_values[i];
    }
};

// ── Device-friendly alias table (flat arrays) ──────────────────────
struct DeviceAliasTable {
    float*    d_prob;       // [n]
    uint32_t* d_alias;      // [n]
    float*    d_pdf_values; // [n]
    int       n;
    float     total_weight;

    HD int sample(float u1, float u2) const {
        int idx = (int)(u1 * n);
        if (idx >= n) idx = n - 1;
        if (u2 < d_prob[idx]) return idx;
        return d_alias[idx];
    }

    HD float pdf(int i) const {
        return d_pdf_values[i];
    }
};
