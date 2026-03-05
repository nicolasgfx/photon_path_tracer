// ─────────────────────────────────────────────────────────────────────
// pbrt_loader.cpp – Orchestrate loading a PBRT v4 scene
// ─────────────────────────────────────────────────────────────────────
// 1. Parse the .pbrt file → PbrtScene (intermediate representation)
// 2. Map PBRT materials → renderer Material structs (via MaterialMapper)
// 3. Process shapes: load PLY meshes / inline triangle meshes / sphere
//    / disk / bilinearmesh → flat Triangle soup
// 4. Apply per-shape world transforms (positions and normals)
// 5. Handle AreaLightSource → emissive material
// 6. Handle ReverseOrientation → flip normals + winding
// 7. Extract camera → populate camera_info for main.cpp
// 8. Extract lights (infinite → envmap, point/spot → small emissive geo)
// 9. Handle NamedMedium → HomogeneousMedium
// 10. Call finalize_pb_materials() to complete material processing
// ─────────────────────────────────────────────────────────────────────

#include "scene/pbrt/pbrt_loader.h"
#include "scene/pbrt/pbrt_parser.h"
#include "scene/pbrt/ply_reader.h"
#include "scene/pbrt/pbrt_material_mapper.h"
#include "scene/material.h"
#include "scene/triangle.h"
#include "scene/scene.h"
#include "core/types.h"
#include "core/spectrum.h"
#include "core/config.h"

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cassert>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Forward-declare finalize_pb_materials (defined in obj_loader.cpp)
extern void finalize_pb_materials(Scene& scene);

namespace {

// ─────────────────────────────────────────────────────────────────────
// Transform helpers
// ─────────────────────────────────────────────────────────────────────

// Multiply Mat4 * (x,y,z,w) → return (rx, ry, rz, rw)
struct Vec4 { double x, y, z, w; };

inline Vec4 mat4_mul_point(const pbrt::Mat4& m, double x, double y, double z) {
    double rx = m.m[0][0]*x + m.m[0][1]*y + m.m[0][2]*z + m.m[0][3];
    double ry = m.m[1][0]*x + m.m[1][1]*y + m.m[1][2]*z + m.m[1][3];
    double rz = m.m[2][0]*x + m.m[2][1]*y + m.m[2][2]*z + m.m[2][3];
    double rw = m.m[3][0]*x + m.m[3][1]*y + m.m[3][2]*z + m.m[3][3];
    if (std::abs(rw) > 1e-12 && std::abs(rw - 1.0) > 1e-12) {
        rx /= rw; ry /= rw; rz /= rw;
    }
    return {rx, ry, rz, rw};
}

inline float3 transform_point(const pbrt::Mat4& m, float3 p) {
    auto r = mat4_mul_point(m, p.x, p.y, p.z);
    return make_f3((float)r.x, (float)r.y, (float)r.z);
}

// Transform normal by inverse-transpose of the upper-left 3×3
// We compute the cofactor matrix (transpose of adjugate) directly.
inline float3 transform_normal(const pbrt::Mat4& m, float3 n) {
    // Cofactor of 3×3 (row i, col j) = (-1)^(i+j) * det of 2×2 minor
    double a = m.m[0][0], b = m.m[0][1], c = m.m[0][2];
    double d = m.m[1][0], e = m.m[1][1], f = m.m[1][2];
    double g = m.m[2][0], h = m.m[2][1], k = m.m[2][2];

    // inv-transpose row 0 = cofactors of column 0 of original
    // Actually: normal transform = (M^{-1})^T * n
    // = adjugate(M)^T^T / det * n = adjugate(M) / det * n
    // Since we only need direction (will normalize), skip det.
    // cofactor matrix C(i,j):
    double c00 = e*k - f*h;
    double c01 = -(d*k - f*g);
    double c02 = d*h - e*g;
    double c10 = -(b*k - c*h);
    double c11 = a*k - c*g;
    double c12 = -(a*h - b*g);
    double c20 = b*f - c*e;
    double c21 = -(a*f - c*d);
    double c22 = a*e - b*d;

    // Normal transform = transpose(cofactor) * n  (= adjugate * n)
    // transpose of cofactor: row i = cofactor column i
    // So (cofactor^T * n)_i = C[0][i]*n.x + C[1][i]*n.y + C[2][i]*n.z
    float rx = (float)(c00*n.x + c10*n.y + c20*n.z);
    float ry = (float)(c01*n.x + c11*n.y + c21*n.z);
    float rz = (float)(c02*n.x + c12*n.y + c22*n.z);
    return normalize(make_f3(rx, ry, rz));
}

// Determinant of upper-left 3×3
inline double det3x3(const pbrt::Mat4& m) {
    double a = m.m[0][0], b = m.m[0][1], c = m.m[0][2];
    double d = m.m[1][0], e = m.m[1][1], f = m.m[1][2];
    double g = m.m[2][0], h = m.m[2][1], k = m.m[2][2];
    return a*(e*k - f*h) - b*(d*k - f*g) + c*(d*h - e*g);
}

// ─────────────────────────────────────────────────────────────────────
// Sphere tessellation
// ─────────────────────────────────────────────────────────────────────
void tessellate_sphere(float radius, int n_theta, int n_phi,
                       std::vector<float3>& positions,
                       std::vector<float3>& normals,
                       std::vector<int3>& faces) {
    positions.clear();
    normals.clear();
    faces.clear();

    // Generate vertices
    for (int t = 0; t <= n_theta; ++t) {
        float theta = (float)M_PI * t / n_theta;
        float sin_t = sinf(theta);
        float cos_t = cosf(theta);
        for (int p = 0; p <= n_phi; ++p) {
            float phi = 2.f * (float)M_PI * p / n_phi;
            float x = sin_t * cosf(phi);
            float y = cos_t;
            float z = sin_t * sinf(phi);
            normals.push_back(make_f3(x, y, z));
            positions.push_back(make_f3(x * radius, y * radius, z * radius));
        }
    }

    // Generate faces
    for (int t = 0; t < n_theta; ++t) {
        for (int p = 0; p < n_phi; ++p) {
            int i0 = t * (n_phi + 1) + p;
            int i1 = i0 + 1;
            int i2 = i0 + (n_phi + 1);
            int i3 = i2 + 1;
            if (t > 0)
                faces.push_back(make_i3(i0, i2, i1));
            if (t < n_theta - 1)
                faces.push_back(make_i3(i1, i2, i3));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Disk tessellation
// ─────────────────────────────────────────────────────────────────────
void tessellate_disk(float radius, float height, int segments,
                     std::vector<float3>& positions,
                     std::vector<float3>& normals,
                     std::vector<int3>& faces) {
    positions.clear();
    normals.clear();
    faces.clear();

    // Center vertex
    positions.push_back(make_f3(0, height, 0));
    normals.push_back(make_f3(0, 1, 0));

    for (int i = 0; i <= segments; ++i) {
        float angle = 2.f * (float)M_PI * i / segments;
        float x = radius * cosf(angle);
        float z = radius * sinf(angle);
        positions.push_back(make_f3(x, height, z));
        normals.push_back(make_f3(0, 1, 0));
    }

    for (int i = 0; i < segments; ++i) {
        faces.push_back(make_i3(0, i + 1, i + 2));
    }
}

// ─────────────────────────────────────────────────────────────────────
// Process one shape → append Triangles to scene
// ─────────────────────────────────────────────────────────────────────
struct ShapeStats {
    int plymesh      = 0;
    int trianglemesh = 0;
    int sphere       = 0;
    int disk         = 0;
    int bilinearmesh = 0;
    int skipped      = 0;
    size_t tri_count = 0;
};

void process_shape(const pbrt::PbrtShape& shape,
                   const pbrt::PbrtScene& pbrt_scene,
                   pbrt::MaterialMapper& mapper,
                   Scene& scene,
                   ShapeStats& stats) {
    using namespace pbrt;

    // ── Resolve material ────────────────────────────────────────────
    uint32_t mat_id = mapper.resolve_shape_material(shape);

    // ── Handle AreaLightSource → create/merge emissive material ─────
    if (shape.has_area_light) {
        uint32_t emissive_id = mapper.create_emissive_material(
            shape.material_name.empty() ? "area_light" : shape.material_name,
            shape.area_light_params);

        // Merge: take the base material's textures/surface properties,
        // but override Le from the area light.
        if (mat_id < scene.materials.size()) {
            Material& base = scene.materials[mat_id];
            Material& emit = scene.materials[emissive_id];
            emit.Kd          = base.Kd;
            emit.diffuse_tex = base.diffuse_tex;
            emit.roughness   = base.roughness;
            emit.opacity     = base.opacity;
        }
        mat_id = emissive_id;
    }

    // ── Determine if winding needs flipping ─────────────────────────
    double det = det3x3(shape.transform);
    bool flip_winding = (det < 0.0) ^ shape.reverse_orientation;

    // ── Collect geometry ────────────────────────────────────────────
    std::vector<float3> positions;
    std::vector<float3> normals;
    std::vector<float2> texcoords;
    std::vector<int3>   faces;

    if (shape.shape_type == "plymesh") {
        std::string filename = get_string(shape.params, "filename");
        if (filename.empty()) {
            std::fprintf(stderr, "[PBRT] plymesh missing 'filename'\n");
            ++stats.skipped;
            return;
        }
        // Resolve relative to source directory
        std::string ply_path = pbrt_scene.source_dir + "/" + filename;

        PlyMesh ply;
        if (!load_ply(ply_path, ply)) {
            std::fprintf(stderr, "[PBRT] Failed to load PLY: %s\n", ply_path.c_str());
            ++stats.skipped;
            return;
        }
        if (ply.faces.empty()) {
            ++stats.skipped;
            return;
        }
        if (!ply.has_normals())
            compute_face_normals(ply);

        positions = std::move(ply.positions);
        normals   = std::move(ply.normals);
        texcoords = std::move(ply.texcoords);
        faces     = std::move(ply.faces);
        ++stats.plymesh;

    } else if (shape.shape_type == "trianglemesh") {
        // Inline vertex data from params
        const Param* p_P  = get_param(shape.params, "P");
        const Param* p_N  = get_param(shape.params, "N");
        const Param* p_uv = get_param(shape.params, "uv");
        const Param* p_st = get_param(shape.params, "st");
        const Param* p_idx = get_param(shape.params, "indices");

        if (!p_P || p_P->floats.size() < 9) {
            std::fprintf(stderr, "[PBRT] trianglemesh missing 'P' data\n");
            ++stats.skipped;
            return;
        }

        size_t nv = p_P->floats.size() / 3;
        positions.resize(nv);
        for (size_t i = 0; i < nv; ++i) {
            positions[i] = make_f3((float)p_P->floats[i*3+0],
                                   (float)p_P->floats[i*3+1],
                                   (float)p_P->floats[i*3+2]);
        }

        if (p_N && p_N->floats.size() >= nv * 3) {
            normals.resize(nv);
            for (size_t i = 0; i < nv; ++i) {
                normals[i] = make_f3((float)p_N->floats[i*3+0],
                                     (float)p_N->floats[i*3+1],
                                     (float)p_N->floats[i*3+2]);
            }
        }

        // Try "uv" first, then "st"
        const Param* p_tex = p_uv ? p_uv : p_st;
        if (p_tex && p_tex->floats.size() >= nv * 2) {
            texcoords.resize(nv);
            for (size_t i = 0; i < nv; ++i) {
                texcoords[i] = make_f2((float)p_tex->floats[i*2+0],
                                       (float)p_tex->floats[i*2+1]);
            }
        }

        if (p_idx && !p_idx->ints.empty()) {
            size_t nf = p_idx->ints.size() / 3;
            faces.resize(nf);
            for (size_t i = 0; i < nf; ++i) {
                faces[i] = make_i3(p_idx->ints[i*3+0],
                                   p_idx->ints[i*3+1],
                                   p_idx->ints[i*3+2]);
            }
        } else {
            // No indices → sequential triangles
            size_t nf = nv / 3;
            faces.resize(nf);
            for (size_t i = 0; i < nf; ++i)
                faces[i] = make_i3((int)(i*3+0), (int)(i*3+1), (int)(i*3+2));
        }
        ++stats.trianglemesh;

    } else if (shape.shape_type == "sphere") {
        float radius = (float)get_float(shape.params, "radius", 1.0);
        // Tessellation resolution based on size
        int n_theta = 32, n_phi = 64;
        tessellate_sphere(radius, n_theta, n_phi, positions, normals, faces);
        ++stats.sphere;

    } else if (shape.shape_type == "disk") {
        float radius = (float)get_float(shape.params, "radius", 1.0);
        float height = (float)get_float(shape.params, "height", 0.0);
        tessellate_disk(radius, height, 64, positions, normals, faces);
        ++stats.disk;

    } else if (shape.shape_type == "bilinearmesh") {
        // Bilinear patches: 4 vertices per patch → 2 triangles
        const Param* p_P   = get_param(shape.params, "P");
        const Param* p_idx = get_param(shape.params, "indices");
        const Param* p_uv  = get_param(shape.params, "uv");
        const Param* p_N   = get_param(shape.params, "N");

        if (!p_P || p_P->floats.size() < 12) {
            std::fprintf(stderr, "[PBRT] bilinearmesh missing 'P' data\n");
            ++stats.skipped;
            return;
        }

        size_t nv = p_P->floats.size() / 3;
        positions.resize(nv);
        for (size_t i = 0; i < nv; ++i) {
            positions[i] = make_f3((float)p_P->floats[i*3+0],
                                   (float)p_P->floats[i*3+1],
                                   (float)p_P->floats[i*3+2]);
        }
        if (p_N && p_N->floats.size() >= nv * 3) {
            normals.resize(nv);
            for (size_t i = 0; i < nv; ++i) {
                normals[i] = make_f3((float)p_N->floats[i*3+0],
                                     (float)p_N->floats[i*3+1],
                                     (float)p_N->floats[i*3+2]);
            }
        }
        if (p_uv && p_uv->floats.size() >= nv * 2) {
            texcoords.resize(nv);
            for (size_t i = 0; i < nv; ++i) {
                texcoords[i] = make_f2((float)p_uv->floats[i*2+0],
                                       (float)p_uv->floats[i*2+1]);
            }
        }

        // Each quad (4 indices) → 2 triangles
        if (p_idx && p_idx->ints.size() >= 4) {
            size_t nq = p_idx->ints.size() / 4;
            faces.reserve(nq * 2);
            for (size_t i = 0; i < nq; ++i) {
                int i0 = p_idx->ints[i*4+0];
                int i1 = p_idx->ints[i*4+1];
                int i2 = p_idx->ints[i*4+2];
                int i3 = p_idx->ints[i*4+3];
                faces.push_back(make_i3(i0, i1, i2));
                faces.push_back(make_i3(i0, i2, i3));
            }
        } else {
            // Sequential quads
            size_t nq = nv / 4;
            faces.reserve(nq * 2);
            for (size_t i = 0; i < nq; ++i) {
                int b = (int)(i * 4);
                faces.push_back(make_i3(b, b+1, b+2));
                faces.push_back(make_i3(b, b+2, b+3));
            }
        }
        ++stats.bilinearmesh;

    } else {
        std::fprintf(stderr, "[PBRT] Unsupported shape type: %s\n",
                     shape.shape_type.c_str());
        ++stats.skipped;
        return;
    }

    if (faces.empty()) return;

    // ── Generate normals if missing ─────────────────────────────────
    bool per_face_normals = normals.empty();
    if (per_face_normals) {
        // Will compute per-face normals after transform
        normals.resize(positions.size(), make_f3(0, 0, 0));
    }

    // ── Apply world transform ───────────────────────────────────────
    for (auto& p : positions)
        p = transform_point(shape.transform, p);
    if (!per_face_normals) {
        for (auto& n : normals)
            n = transform_normal(shape.transform, n);
    }

    // ── Build triangles ─────────────────────────────────────────────
    size_t base_tri = scene.triangles.size();
    scene.triangles.reserve(base_tri + faces.size());

    bool has_uv = !texcoords.empty();
    int nv = (int)positions.size();

    for (const auto& f : faces) {
        // Bounds check
        if (f.x < 0 || f.y < 0 || f.z < 0 ||
            f.x >= nv || f.y >= nv || f.z >= nv)
            continue;

        Triangle tri;
        if (flip_winding) {
            tri.v0 = positions[f.x];
            tri.v1 = positions[f.z];
            tri.v2 = positions[f.y];
        } else {
            tri.v0 = positions[f.x];
            tri.v1 = positions[f.y];
            tri.v2 = positions[f.z];
        }

        if (per_face_normals) {
            float3 gn = tri.geometric_normal();
            tri.n0 = tri.n1 = tri.n2 = gn;
        } else {
            if (flip_winding) {
                tri.n0 = normals[f.x];
                tri.n1 = normals[f.z];
                tri.n2 = normals[f.y];
            } else {
                tri.n0 = normals[f.x];
                tri.n1 = normals[f.y];
                tri.n2 = normals[f.z];
            }
        }

        if (has_uv) {
            if (flip_winding) {
                tri.uv0 = texcoords[f.x];
                tri.uv1 = texcoords[f.z];
                tri.uv2 = texcoords[f.y];
            } else {
                tri.uv0 = texcoords[f.x];
                tri.uv1 = texcoords[f.y];
                tri.uv2 = texcoords[f.z];
            }
        } else {
            tri.uv0 = tri.uv1 = tri.uv2 = make_f2(0, 0);
        }

        tri.material_id = mat_id;
        scene.triangles.push_back(tri);
    }

    stats.tri_count += scene.triangles.size() - base_tri;
}

// ─────────────────────────────────────────────────────────────────────
// Process lights
// ─────────────────────────────────────────────────────────────────────
struct LightInfo {
    // Infinite light (envmap)
    bool has_envmap = false;
    std::string envmap_path;
    float envmap_scale = 1.0f;
    float3 envmap_rotation = {0, 0, 0};
    float3 envmap_constant = {0, 0, 0};  // constant-colour envmap (no EXR)

    // Point / spot lights → small emissive spheres
    struct PointLightGeo {
        float3 position;
        float3 color;     // RGB pre-multiplied by intensity
        float  radius;    // small sphere radius
    };
    std::vector<PointLightGeo> point_lights;
};

LightInfo extract_lights(const pbrt::PbrtScene& pbrt_scene) {
    using namespace pbrt;
    LightInfo info;

    for (const auto& light : pbrt_scene.lights) {
        if (light.light_type == "infinite") {
            info.has_envmap = true;

            std::string filename = get_string(light.params, "filename");
            if (filename.empty())
                filename = get_string(light.params, "mapname");

            if (!filename.empty()) {
                info.envmap_path = pbrt_scene.source_dir + "/" + filename;
            }

            // Scale
            auto L = get_rgb(light.params, "L");
            float scale_val = (float)get_float(light.params, "scale", 1.0);
            if (!L.empty() && L.size() >= 3) {
                info.envmap_scale = scale_val;
                // If L is not white, encode it as constant colour + file
                float lr = (float)L[0], lg = (float)L[1], lb = (float)L[2];
                if (filename.empty()) {
                    // No texture → constant colour envmap
                    info.envmap_constant = make_f3(lr * scale_val,
                                                   lg * scale_val,
                                                   lb * scale_val);
                } else {
                    // Texture present — apply L as tint via scale
                    // For simplicity, use average of L as scale multiplier
                    info.envmap_scale = scale_val * (lr + lg + lb) / 3.0f;
                }
            } else {
                info.envmap_scale = scale_val;
            }

            // Rotation from transform
            // PBRT applies the light transform to the envmap directions.
            // We don't easily extract Euler angles from an arbitrary matrix,
            // so we leave rotation at 0 and let transform_point handle it
            // in the environment map evaluation.  For a future enhancement,
            // we could decompose the rotation matrix.

        } else if (light.light_type == "point") {
            auto I = get_rgb(light.params, "I");
            float scale = (float)get_float(light.params, "scale", 1.0);
            float3 color = make_f3(1, 1, 1);
            if (I.size() >= 3)
                color = make_f3((float)I[0], (float)I[1], (float)I[2]);
            color = color * scale;

            // Extract position from transform
            float3 pos = make_f3((float)light.transform.m[0][3],
                                 (float)light.transform.m[1][3],
                                 (float)light.transform.m[2][3]);

            info.point_lights.push_back({pos, color, 0.01f});

        } else if (light.light_type == "spot") {
            auto I = get_rgb(light.params, "I");
            float scale = (float)get_float(light.params, "scale", 1.0);
            float3 color = make_f3(1, 1, 1);
            if (I.size() >= 3)
                color = make_f3((float)I[0], (float)I[1], (float)I[2]);
            color = color * scale;

            float3 pos = make_f3((float)light.transform.m[0][3],
                                 (float)light.transform.m[1][3],
                                 (float)light.transform.m[2][3]);

            info.point_lights.push_back({pos, color, 0.01f});
        }
        // distant lights not commonly used; skip for now
    }
    return info;
}

// ─────────────────────────────────────────────────────────────────────
// Create small emissive sphere to represent a point/spot light
// ─────────────────────────────────────────────────────────────────────
void add_point_light_geometry(const LightInfo::PointLightGeo& pl,
                              Scene& scene) {
    // Create emissive material
    Material mat;
    mat.name = "__point_light";
    mat.type = MaterialType::Emissive;
    mat.Le = rgb_to_spectrum_emission(pl.color.x, pl.color.y, pl.color.z);
    mat.Kd = Spectrum::zero();
    mat.pb_brdf = PbBrdf::Emissive;
    uint32_t mat_id = (uint32_t)scene.materials.size();
    scene.materials.push_back(mat);

    // Tessellate a small sphere at the light position
    std::vector<float3> positions, normals;
    std::vector<int3> faces;
    tessellate_sphere(pl.radius, 8, 16, positions, normals, faces);

    // Offset to light position
    for (auto& p : positions)
        p = p + pl.position;

    for (const auto& f : faces) {
        Triangle tri;
        tri.v0 = positions[f.x]; tri.v1 = positions[f.y]; tri.v2 = positions[f.z];
        tri.n0 = normals[f.x];   tri.n1 = normals[f.y];   tri.n2 = normals[f.z];
        tri.uv0 = tri.uv1 = tri.uv2 = make_f2(0, 0);
        tri.material_id = mat_id;
        scene.triangles.push_back(tri);
    }
}

// ─────────────────────────────────────────────────────────────────────
// Extract camera information
// ─────────────────────────────────────────────────────────────────────
struct CameraInfo {
    bool valid = false;
    float3 position = {0, 0, 0};
    float3 look_at  = {0, 0, -1};
    float3 up       = {0, 1, 0};
    float  fov      = 70.0f;
};

CameraInfo extract_camera(const pbrt::PbrtScene& pbrt_scene) {
    using namespace pbrt;
    CameraInfo info;

    const auto& cam = pbrt_scene.camera;
    if (!cam.has_lookat) return info;

    info.valid = true;
    info.position = make_f3((float)cam.eye[0], (float)cam.eye[1], (float)cam.eye[2]);
    info.look_at  = make_f3((float)cam.target[0], (float)cam.target[1], (float)cam.target[2]);
    info.up       = make_f3((float)cam.up[0], (float)cam.up[1], (float)cam.up[2]);
    info.fov      = (float)get_float(cam.params, "fov", 90.0);

    return info;
}

// ─────────────────────────────────────────────────────────────────────
// Write saved_camera.json for PBRT scene
// ─────────────────────────────────────────────────────────────────────
bool write_saved_camera(const std::string& folder, const CameraInfo& cam,
                        const LightInfo& lights) {
    namespace fs = std::filesystem;
    fs::create_directories(folder);
    std::string path = folder + "/saved_camera.json";

    // Only write if file doesn't already exist (don't overwrite user tweaks)
    if (fs::exists(path)) return true;

    std::ofstream f(path);
    if (!f.is_open()) return false;

    f << "{\n";
    f << "  \"position\": [" << cam.position.x << ", " << cam.position.y << ", " << cam.position.z << "],\n";
    f << "  \"look_at\": [" << cam.look_at.x << ", " << cam.look_at.y << ", " << cam.look_at.z << "],\n";
    f << "  \"up\": [" << cam.up.x << ", " << cam.up.y << ", " << cam.up.z << "],\n";
    f << "  \"fov\": " << cam.fov << ",\n";
    f << "  \"yaw\": 0.0,\n";
    f << "  \"pitch\": 0.0,\n";
    f << "  \"roll\": 0.0,\n";
    f << "  \"light_scale\": 1.0";

    // Add envmap if present
    if (lights.has_envmap && !lights.envmap_path.empty()) {
        f << ",\n  \"envmap\": \"" << lights.envmap_path << "\"";
        f << ",\n  \"envmap_scale\": " << lights.envmap_scale;
    }

    f << "\n}\n";
    f.close();
    return true;
}

// ─────────────────────────────────────────────────────────────────────
// Handle NamedMedium → HomogeneousMedium
// ─────────────────────────────────────────────────────────────────────
void process_media(const pbrt::PbrtScene& pbrt_scene, Scene& scene) {
    using namespace pbrt;

    for (const auto& [name, decl] : pbrt_scene.named_media) {
        if (decl.type != "homogeneous") {
            std::fprintf(stderr, "[PBRT] Unsupported medium type: %s (%s)\n",
                         decl.type.c_str(), name.c_str());
            continue;
        }

        HomogeneousMedium med;
        auto sigma_a_rgb = get_rgb(decl.params, "sigma_a");
        auto sigma_s_rgb = get_rgb(decl.params, "sigma_s");
        float scale = (float)get_float(decl.params, "scale", 1.0);
        float g_val = (float)get_float(decl.params, "g", 0.0);

        float sa_r = 0, sa_g = 0, sa_b = 0;
        float ss_r = 0, ss_g = 0, ss_b = 0;
        if (sigma_a_rgb.size() >= 3) {
            sa_r = (float)sigma_a_rgb[0]; sa_g = (float)sigma_a_rgb[1]; sa_b = (float)sigma_a_rgb[2];
        }
        if (sigma_s_rgb.size() >= 3) {
            ss_r = (float)sigma_s_rgb[0]; ss_g = (float)sigma_s_rgb[1]; ss_b = (float)sigma_s_rgb[2];
        }

        Spectrum sa = rgb_to_spectrum_reflectance(sa_r * scale, sa_g * scale, sa_b * scale);
        Spectrum ss = rgb_to_spectrum_reflectance(ss_r * scale, ss_g * scale, ss_b * scale);

        for (int b = 0; b < NUM_LAMBDA; ++b) {
            med.sigma_a.value[b] = sa.value[b];
            med.sigma_s.value[b] = ss.value[b];
            med.sigma_t.value[b] = sa.value[b] + ss.value[b];
        }
        med.g = g_val;

        int medium_id = (int)scene.media.size();
        scene.media.push_back(med);

        std::printf("[PBRT] Medium '%s': id=%d, g=%.2f, scale=%.1f\n",
                    name.c_str(), medium_id, g_val, scale);
    }
}

} // anonymous namespace


// =====================================================================
// Public API
// =====================================================================

bool load_pbrt(const std::string& filepath, Scene& scene) {
    std::printf("[PBRT] Loading: %s\n", filepath.c_str());

    // ── 1. Parse ────────────────────────────────────────────────────
    pbrt::PbrtParser parser;
    pbrt::PbrtScene pbrt_scene;
    try {
        pbrt_scene = parser.parse_file(filepath);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[PBRT] Parse error: %s\n", e.what());
        return false;
    }

    std::printf("[PBRT] Parsed: %zu shapes, %zu named materials, %zu textures, "
                "%zu lights, %zu object templates, %zu media\n",
                pbrt_scene.shapes.size(),
                pbrt_scene.named_materials.size(),
                pbrt_scene.textures.size(),
                pbrt_scene.lights.size(),
                pbrt_scene.object_templates.size(),
                pbrt_scene.named_media.size());

    // ── 2. Map materials ────────────────────────────────────────────
    pbrt::MaterialMapper mapper(pbrt_scene, scene, pbrt_scene.source_dir);
    mapper.map_all_named_materials();

    std::printf("[PBRT] Mapped %zu materials, %zu textures loaded\n",
                scene.materials.size(), scene.textures.size());
    std::fflush(stdout);

    // ── 3. Process media ────────────────────────────────────────────
    process_media(pbrt_scene, scene);

    // ── 4. Process shapes ───────────────────────────────────────────
    std::fprintf(stderr, "[PBRT-DBG] About to process %zu shapes\n", pbrt_scene.shapes.size());
    ShapeStats stats;
    int shape_idx = 0;
    for (const auto& shape : pbrt_scene.shapes) {
        std::fprintf(stderr, "[PBRT-DBG] shape %d/%zu type=%s mat=%s\n",
                     shape_idx, (size_t)pbrt_scene.shapes.size(),
                     shape.shape_type.c_str(), shape.material_name.c_str());
        process_shape(shape, pbrt_scene, mapper, scene, stats);
        std::fprintf(stderr, "[PBRT-DBG] shape %d done, tris=%zu\n",
                     shape_idx, scene.triangles.size());
        ++shape_idx;
    }

    std::printf("[PBRT] Shapes: plymesh=%d trianglemesh=%d sphere=%d disk=%d "
                "bilinear=%d skipped=%d\n",
                stats.plymesh, stats.trianglemesh, stats.sphere, stats.disk,
                stats.bilinearmesh, stats.skipped);
    std::printf("[PBRT] Total triangles: %zu\n", scene.triangles.size());
    std::fflush(stdout);

    // ── 5. Process lights ───────────────────────────────────────────
    LightInfo lights = extract_lights(pbrt_scene);

    for (const auto& pl : lights.point_lights) {
        add_point_light_geometry(pl, scene);
        std::printf("[PBRT] Point light at (%.2f, %.2f, %.2f) → emissive sphere\n",
                    pl.position.x, pl.position.y, pl.position.z);
    }

    // ── 6. Extract camera, write saved_camera.json if needed ────────
    CameraInfo cam_info = extract_camera(pbrt_scene);
    if (cam_info.valid) {
        std::printf("[PBRT] Camera: pos=(%.2f,%.2f,%.2f) look_at=(%.2f,%.2f,%.2f) fov=%.1f\n",
                    cam_info.position.x, cam_info.position.y, cam_info.position.z,
                    cam_info.look_at.x, cam_info.look_at.y, cam_info.look_at.z,
                    cam_info.fov);

        // Derive scene folder for saved_camera.json
        // The PBRT file lives e.g. in "scenes/kroken/" or "tools/pbrtv4_scenes/..."
        // We write saved_camera.json next to the PBRT file.
        namespace fs = std::filesystem;
        std::string folder = fs::path(filepath).parent_path().string();
        write_saved_camera(folder, cam_info, lights);
    }

    // ── 7. Finalize pb_* material extensions ────────────────────────
    finalize_pb_materials(scene);

    // ── 8. Report envmap info (main.cpp handles actual loading) ─────
    if (lights.has_envmap) {
        if (!lights.envmap_path.empty())
            std::printf("[PBRT] Envmap: %s (scale=%.2f)\n",
                        lights.envmap_path.c_str(), lights.envmap_scale);
        else
            std::printf("[PBRT] Constant envmap: (%.3f, %.3f, %.3f)\n",
                        lights.envmap_constant.x, lights.envmap_constant.y,
                        lights.envmap_constant.z);
    }

    std::printf("[PBRT] Loading complete: %zu triangles, %zu materials, "
                "%zu textures, %zu media\n",
                scene.triangles.size(), scene.materials.size(),
                scene.textures.size(), scene.media.size());
    std::fflush(stdout);

    return true;
}
