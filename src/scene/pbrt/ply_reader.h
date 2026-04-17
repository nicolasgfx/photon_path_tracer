#pragma once
// ─────────────────────────────────────────────────────────────────────
// ply_reader.h – Binary PLY mesh reader (binary little-endian + ASCII)
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include <vector>
#include <string>

namespace pbrt {

struct PlyMesh {
    std::vector<float3> positions;
    std::vector<float3> normals;
    std::vector<float2> texcoords;
    std::vector<int3>   faces;

    bool has_normals()   const { return !normals.empty(); }
    bool has_texcoords() const { return !texcoords.empty(); }
};

// Load a PLY file.  Supported formats:
//   - binary_little_endian 1.0  (dominant for PBRT scenes)
//   - ascii 1.0
// Vertex properties: x,y,z  (required); nx,ny,nz (optional); u/s,v/t (optional)
// Face properties: vertex_indices as list (triangles or quads → split to tris)
bool load_ply(const std::string& filepath, PlyMesh& out);

// Compute flat (per-face) normals from positions+faces when normals are missing
void compute_face_normals(PlyMesh& mesh);

} // namespace pbrt
