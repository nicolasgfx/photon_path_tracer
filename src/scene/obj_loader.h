#pragma once
// ─────────────────────────────────────────────────────────────────────
// obj_loader.h – Wavefront OBJ/MTL scene loader
// ─────────────────────────────────────────────────────────────────────
#include "scene/scene.h"
#include <string>

// Load an OBJ file (optionally references MTL files) into the scene.
// Supports:
//   - v x y z [r g b]    (vertices with optional per-vertex color)
//   - vn nx ny nz         (normals)
//   - vt u v              (texture coords)
//   - f v/vt/vn ...       (faces – triangulated automatically)
//   - mtllib file.mtl
//   - usemtl name
bool load_obj(const std::string& filepath, Scene& scene);

// Apply pb_* material extensions (shared between OBJ and PBRT loaders).
// Resolves pb_brdf → MaterialType, pb_roughness, pb_eta, conductor
// complex IOR, creates HomogeneousMedium entries, etc.
void finalize_pb_materials(Scene& scene);
