#pragma once
// ─────────────────────────────────────────────────────────────────────
// hash.h – Spatial hashing (Teschner et al.)
//
// Single HD implementation shared by CPU hash grids, GPU kernels,
// cell cache, light cache, and emitter-point filtering.
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include <cstdint>

// Raw Teschner hash (no modulo) — use when you need a map key or want
// to apply your own table-size modulo.
inline HD uint32_t teschner_hash_raw(int3 cell) {
    return (uint32_t)(cell.x * 73856093u)
         ^ (uint32_t)(cell.y * 19349663u)
         ^ (uint32_t)(cell.z * 83492791u);
}

// Teschner hash with table-size modulo — direct replacement for all
// grid hash functions.
inline HD uint32_t teschner_hash(int3 cell, uint32_t table_size) {
    return teschner_hash_raw(cell) % table_size;
}
