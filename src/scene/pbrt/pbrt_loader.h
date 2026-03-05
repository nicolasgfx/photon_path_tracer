#pragma once
// ─────────────────────────────────────────────────────────────────────
// pbrt_loader.h – Load a PBRT v4 scene directly into the renderer
// ─────────────────────────────────────────────────────────────────────
#include "scene/scene.h"
#include <string>

// Load a PBRT v4 scene file (text format) into the renderer's Scene struct.
// Supports: plymesh, trianglemesh, sphere, disk, bilinearmesh shapes;
//           all major PBRT v4 material types; Include directives;
//           ObjectBegin/ObjectInstance instancing (baked to flat triangles);
//           AreaLightSource, LightSource "infinite"/"point"/"spot";
//           Texture graph resolution; Camera extraction.
//
// Returns false on failure.
bool load_pbrt(const std::string& filepath, Scene& scene);
