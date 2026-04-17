#pragma once
// ─────────────────────────────────────────────────────────────────────
// pbrt_loader.h – Load a PBRT v4 scene directly into the renderer
// ─────────────────────────────────────────────────────────────────────
#include "scene/scene.h"
#include <string>

// Load a PBRT v4 scene file (text format) into the renderer's Scene struct.
// Supports: plymesh, trianglemesh, sphere, disk, bilinearmesh shapes;
//           major PBRT v4 material types with explicit fallbacks;
//           Include and Import directives; CoordinateSystem / CoordSysTransform;
//           Attribute inheritance for shape/light/material/texture/medium;
//           ObjectBegin/ObjectInstance instancing (baked to flat triangles);
//           AreaLightSource plus point/spot/distant/infinite portal proxy lights;
//           Texture graph resolution for the currently supported texture subset;
//           camera extraction.
// Deferred: image-based infinite lights / envmaps and full texture fidelity.
//
// Returns false on failure.
bool load_pbrt(const std::string& filepath, Scene& scene);
