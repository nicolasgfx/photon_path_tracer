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
// 8. Extract lights (point/spot → small emissive geo, portals → emitter quads)
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
#include "core/color.h"
#include "core/config.h"
#include "core/thread_pool.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cassert>
#include <future>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

constexpr float PI_F = 3.14159265358979323846f;

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

// ── Blackbody → linear sRGB (McCamy approximation) ──────────────────
static void blackbody_to_rgb(float temp_K, float rgb[3]) {
    float t = temp_K / 100.f;
    float r, g, b;
    if (t <= 66.f) {
        r = 255.f;
        g = std::max(0.f, 99.4708f * std::log(t) - 161.1196f);
        b = (t <= 19.f) ? 0.f : std::max(0.f, 138.5177f * std::log(t - 10.f) - 305.0448f);
    } else {
        r = std::max(0.f, 329.6987f * std::pow(t - 60.f, -0.1332f));
        g = std::max(0.f, 288.1222f * std::pow(t - 60.f, -0.0755f));
        b = 255.f;
    }
    r = std::min(255.f, r) / 255.f;
    g = std::min(255.f, g) / 255.f;
    b = std::min(255.f, b) / 255.f;
    float mx = std::max({r, g, b, 1e-10f});
    rgb[0] = r / mx; rgb[1] = g / mx; rgb[2] = b / mx;
}

inline float mean_rgb(const Color3& color) {
    return color.sum() / 3.0f;
}

Color3 extract_color_param(const std::vector<pbrt::Param>& params,
                          const std::string& name,
                          const Color3& fallback = Color3::one()) {
    using namespace pbrt;

    auto* param = get_param(params, name);
    if (!param) return fallback;

    if (param->type == "blackbody" && !param->floats.empty()) {
        float rgb[3];
        blackbody_to_rgb((float)param->floats[0], rgb);
        return Color3::from_rgb(rgb[0], rgb[1], rgb[2]);
    }

    if (param->floats.size() >= 3)
        return Color3::from_rgb((float)param->floats[0],
                                (float)param->floats[1],
                                (float)param->floats[2]);

    if (param->type == "spectrum" && param->floats.size() >= 2) {
        double sum = 0.0;
        int count = 0;
        for (size_t i = 1; i < param->floats.size(); i += 2) {
            sum += param->floats[i];
            ++count;
        }
        float value = count > 0 ? (float)(sum / count) : 1.0f;
        return Color3::constant(value);
    }

    if (!param->floats.empty())
        return Color3::constant((float)param->floats[0]);

    if (param->type == "spectrum" && !param->strings.empty()) {
        std::fprintf(stderr,
                     "[PBRT] Deferred spectrum/string parameter '%s'='%s' uses fallback color\n",
                     name.c_str(), param->strings[0].c_str());
    }

    return fallback;
}

Color3 scale_color_to_power(Color3 base_color, float target_power, float geometry_power_unit) {
    if (target_power <= 0.0f || geometry_power_unit <= 0.0f) return base_color;

    float base_mean = mean_rgb(base_color);
    if (base_mean <= 1e-6f) {
        base_color = Color3::one();
        base_mean = 1.0f;
    }

    return base_color * (target_power / (geometry_power_unit * base_mean));
}

float get_param_scale(const std::vector<pbrt::Param>& params) {
    return (float)pbrt::get_float(params, "scale", 1.0);
}

float compute_area_power_unit(float area, bool twosided) {
    return PI_F * area * (twosided ? 2.0f : 1.0f);
}

float compute_sphere_power_unit(float radius) {
    return 4.0f * PI_F * PI_F * radius * radius;
}

float compute_spot_solid_angle(const std::vector<pbrt::Param>& params) {
    float cone_angle_deg = (float)pbrt::get_float(params, "coneangle", 30.0);
    float cone_delta_deg = (float)pbrt::get_float(params, "conedeltaangle", 5.0);
    float outer_deg = std::max(0.0f, cone_angle_deg + cone_delta_deg);
    float inner_deg = std::max(0.0f, cone_angle_deg);
    float outer = outer_deg * PI_F / 180.0f;
    float inner = inner_deg * PI_F / 180.0f;
    float full_cone = 2.0f * PI_F * (1.0f - std::cos(inner));
    float falloff_cone = 2.0f * PI_F * (std::cos(inner) - std::cos(outer));
    return full_cone + 0.5f * std::max(0.0f, falloff_cone);
}

Color3 compute_area_light_emission(const std::vector<pbrt::Param>& params,
                                   float area,
                                   bool twosided) {
    Color3 base = extract_color_param(params, "L", Color3::one());
    float scale = get_param_scale(params);
    const pbrt::Param* power_param = pbrt::get_param(params, "power");

    if (power_param)
        return scale_color_to_power(base,
                                    (float)power_param->floats[0] * scale,
                                    compute_area_power_unit(area, twosided));

    return base * scale;
}

Color3 compute_point_light_emission(const std::vector<pbrt::Param>& params, float radius) {
    Color3 base = extract_color_param(params, "I", Color3::one());
    float scale = get_param_scale(params);
    const pbrt::Param* power_param = pbrt::get_param(params, "power");
    float target_power = 0.0f;

    if (power_param)
        target_power = (float)power_param->floats[0] * scale;
    else
        target_power = 4.0f * PI_F * mean_rgb(base) * scale;

    return scale_color_to_power(base, target_power, compute_sphere_power_unit(radius));
}

Color3 compute_spot_light_emission(const std::vector<pbrt::Param>& params, float radius) {
    Color3 base = extract_color_param(params, "I", Color3::one());
    float scale = get_param_scale(params);
    const pbrt::Param* power_param = pbrt::get_param(params, "power");
    float target_power = 0.0f;

    if (power_param)
        target_power = (float)power_param->floats[0] * scale;
    else
        target_power = mean_rgb(base) * scale * compute_spot_solid_angle(params);

    return scale_color_to_power(base, target_power, compute_sphere_power_unit(radius));
}

Color3 compute_infinite_portal_emission(const std::vector<pbrt::Param>& params) {
    Color3 base = extract_color_param(params, "L", Color3::one());
    float scale = get_param_scale(params);
    const pbrt::Param* illuminance_param = pbrt::get_param(params, "illuminance");

    if (illuminance_param)
        return scale_color_to_power(base,
                                    (float)illuminance_param->floats[0] * scale,
                                    PI_F);

    if (pbrt::get_param(params, "filename")) {
        std::fprintf(stderr,
                     "[PBRT] Infinite light texture deferred; portal proxy uses constant fallback emission\n");
    }

    return base * scale;
}

void build_orthonormal_basis(float3 n, float3& tangent, float3& bitangent) {
    float3 up = std::fabs(n.z) < 0.999f ? make_f3(0, 0, 1) : make_f3(0, 1, 0);
    tangent = normalize(cross(up, n));
    bitangent = normalize(cross(n, tangent));
}

float compute_triangle_area_sum(const std::vector<float3>& positions,
                                const std::vector<int3>& faces) {
    float area = 0.0f;
    for (const auto& face : faces) {
        float3 e0 = positions[face.y] - positions[face.x];
        float3 e1 = positions[face.z] - positions[face.x];
        area += 0.5f * length(cross(e0, e1));
    }
    return area;
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
    // adj(M) = det(M) * M^{-T}, so when det<0 the adjugate flips the
    // normal direction.  Negate to keep the normal on the correct side.
    double det = m.m[0][0]*c00 + m.m[0][1]*c01 + m.m[0][2]*c02;
    if (det < 0.0) { rx = -rx; ry = -ry; rz = -rz; }
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

    // PBRT disk lies in the XY plane at z=height, normal along +Z.
    // Center vertex
    positions.push_back(make_f3(0, 0, height));
    normals.push_back(make_f3(0, 0, 1));

    for (int i = 0; i <= segments; ++i) {
        float angle = 2.f * (float)M_PI * i / segments;
        float x = radius * cosf(angle);
        float y = radius * sinf(angle);
        positions.push_back(make_f3(x, y, height));
        normals.push_back(make_f3(0, 0, 1));
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

// Pre-loaded PLY mesh cache: path → PlyMesh (populated by parallel loader)
using PlyCache = std::unordered_map<std::string, pbrt::PlyMesh>;

void process_shape(const pbrt::PbrtShape& shape,
                   const pbrt::PbrtScene& pbrt_scene,
                   pbrt::MaterialMapper& mapper,
                   Scene& scene,
                   ShapeStats& stats,
                   PlyCache& ply_cache) {
    using namespace pbrt;

    // ── Skip fully-transparent shapes (alpha = 0) ───────────────────
    float shape_alpha = (float)get_float(shape.params, "alpha", 1.0f);
    if (shape_alpha <= 0.f) {
        if (shape.has_area_light) {
            // Alpha=0 emissive: invisible geometry that still emits.
            // Approximate as small emissive sphere (point light proxy).
            float3 pos = make_f3((float)shape.transform.m[0][3],
                                 (float)shape.transform.m[1][3],
                                 (float)shape.transform.m[2][3]);

            uint32_t emissive_id = mapper.create_emissive_material(
                "alpha0_light", shape.area_light_params);
            const Material& emat = scene.materials[emissive_id];

            // Estimate original area for power scaling
            float radius = 1.f;
            if (shape.shape_type == "sphere")
                radius = (float)get_float(shape.params, "radius", 1.0);
            float big_area = 4.f * (float)M_PI * radius * radius;
            float small_r  = 0.5f;
            float small_area = 4.f * (float)M_PI * small_r * small_r;
            float power_ratio = big_area / std::max(small_area, 1e-6f);

            Material proxy_mat;
            proxy_mat.name    = "__alpha0_proxy";
            proxy_mat.type    = MaterialType::Emissive;
            proxy_mat.pb_brdf = PbBrdf::Emissive;
            proxy_mat.Le = emat.Le * power_ratio;
            proxy_mat.Kd = Color3::zero();
            uint32_t proxy_id = (uint32_t)scene.materials.size();
            scene.materials.push_back(proxy_mat);

            // Tessellate small sphere
            std::vector<float3> sph_pos, sph_nrm;
            std::vector<int3>   sph_faces;
            tessellate_sphere(small_r, 8, 16, sph_pos, sph_nrm, sph_faces);
            for (auto& p : sph_pos) p = p + pos;
            for (const auto& f : sph_faces) {
                Triangle tri;
                tri.v0 = sph_pos[f.x]; tri.v1 = sph_pos[f.y]; tri.v2 = sph_pos[f.z];
                tri.n0 = sph_nrm[f.x]; tri.n1 = sph_nrm[f.y]; tri.n2 = sph_nrm[f.z];
                tri.uv0 = tri.uv1 = tri.uv2 = make_f2(0, 0);
                tri.material_id = proxy_id;
                scene.triangles.push_back(tri);
            }

            std::printf("[PBRT] Alpha=0 emissive %s → point light proxy at "
                        "(%.1f, %.1f, %.1f) power_ratio=%.1f\n",
                        shape.shape_type.c_str(), pos.x, pos.y, pos.z,
                        power_ratio);
        }
        ++stats.skipped;
        return;
    }

    // ── Resolve material ────────────────────────────────────────────
    uint32_t mat_id = mapper.resolve_shape_material(shape);

    // ── Handle AreaLightSource → create/merge emissive material ─────
    if (shape.has_area_light) {
        std::string emit_key = "__emit_shape_" + std::to_string(scene.materials.size());
        uint32_t emissive_id = mapper.create_emissive_material(emit_key, shape.area_light_params);

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
        std::string ply_path = pbrt_scene.source_dir + "/" + filename;

        // Use pre-loaded cache if available, otherwise load on demand
        auto cache_it = ply_cache.find(ply_path);
        PlyMesh ply;
        if (cache_it != ply_cache.end()) {
            ply = std::move(cache_it->second);
            ply_cache.erase(cache_it);
        } else {
            if (!load_ply(ply_path, ply)) {
                std::fprintf(stderr, "[PBRT] Failed to load PLY: %s\n", ply_path.c_str());
                ++stats.skipped;
                return;
            }
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
        std::string tm_filename = get_string(shape.params, "filename");
        if (!tm_filename.empty()) {
            std::string ply_path = pbrt_scene.source_dir + "/" + tm_filename;

            auto cache_it = ply_cache.find(ply_path);
            PlyMesh ply;
            if (cache_it != ply_cache.end()) {
                ply = std::move(cache_it->second);
                ply_cache.erase(cache_it);
            } else {
                if (!load_ply(ply_path, ply)) {
                    std::fprintf(stderr, "[PBRT] trianglemesh failed to load PLY: %s\n",
                                 ply_path.c_str());
                    ++stats.skipped;
                    return;
                }
            }
            if (ply.faces.empty()) { ++stats.skipped; return; }
            if (!ply.has_normals()) compute_face_normals(ply);
            positions = std::move(ply.positions);
            normals   = std::move(ply.normals);
            texcoords = std::move(ply.texcoords);
            faces     = std::move(ply.faces);
            ++stats.trianglemesh;
        } else {
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
        } // end else (inline trianglemesh)

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

    if (shape.has_area_light && mat_id < scene.materials.size()) {
        bool twosided = get_bool(shape.area_light_params, "twosided", false);
        float area_sum = compute_triangle_area_sum(positions, faces);
        Material& emit = scene.materials[mat_id];
        emit.Le = compute_area_light_emission(shape.area_light_params, area_sum, twosided);

        if (get_param(shape.area_light_params, "filename")) {
            std::fprintf(stderr,
                         "[PBRT] Deferred area-light emission texture on shape '%s'\n",
                         shape.shape_type.c_str());
        }
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

    if (shape.has_area_light && get_bool(shape.area_light_params, "twosided", false)) {
        size_t tri_end = scene.triangles.size();
        scene.triangles.reserve(tri_end + (tri_end - base_tri));
        for (size_t i = base_tri; i < tri_end; ++i) {
            Triangle back = scene.triangles[i];
            std::swap(back.v1, back.v2);
            back.n0 = back.n0 * -1.0f;
            back.n1 = back.n1 * -1.0f;
            back.n2 = back.n2 * -1.0f;
            std::swap(back.uv1, back.uv2);
            scene.triangles.push_back(back);
        }
    }

    stats.tri_count += scene.triangles.size() - base_tri;
}

// ─────────────────────────────────────────────────────────────────────
// Process lights
// ─────────────────────────────────────────────────────────────────────
struct LightInfo {
    struct PortalQuad {
        float3 v[4];
        Color3 emission = Color3::one();
    };
    std::vector<PortalQuad> portal_quads;

    struct SphereLightGeo {
        float3 position;
        Color3 emission = Color3::one();
        float radius = 0.01f;
        const char* name = "__point_light";
    };

    struct DiskLightGeo {
        float3 center;
        float3 normal;
        float radius = 1.0f;
        Color3 emission = Color3::one();
        const char* name = "__distant_light";
    };

    std::vector<SphereLightGeo> sphere_lights;
    std::vector<DiskLightGeo> disk_lights;
};

LightInfo extract_lights(const pbrt::PbrtScene& pbrt_scene,
                         const float3& scene_center,
                         float scene_radius) {
    using namespace pbrt;
    LightInfo info;
    scene_radius = std::max(scene_radius, 1.0f);

    for (const auto& light : pbrt_scene.lights) {
        if (light.light_type == "infinite") {
            if (get_param(light.params, "filename")) {
                std::printf("[PBRT] Infinite light texture deferred — using portals only if present\n");
            }

            Color3 portal_emission = compute_infinite_portal_emission(light.params);

            // Portal quads (emissive geometry for directed illumination)
            for (const auto& param : light.params) {
                if (param.name == "portal" && param.floats.size() >= 12) {
                    LightInfo::PortalQuad pq;
                    for (int vi = 0; vi < 4; ++vi) {
                        float3 lp = make_f3((float)param.floats[vi * 3 + 0],
                                            (float)param.floats[vi * 3 + 1],
                                            (float)param.floats[vi * 3 + 2]);
                        pq.v[vi] = transform_point(light.transform, lp);
                    }
                    pq.emission = portal_emission;
                    info.portal_quads.push_back(pq);
                    std::printf("[PBRT] Portal quad extracted: (%.1f,%.1f,%.1f)-(%.1f,%.1f,%.1f)-(%.1f,%.1f,%.1f)-(%.1f,%.1f,%.1f)\n",
                             pq.v[0].x, pq.v[0].y, pq.v[0].z,
                             pq.v[1].x, pq.v[1].y, pq.v[1].z,
                             pq.v[2].x, pq.v[2].y, pq.v[2].z,
                             pq.v[3].x, pq.v[3].y, pq.v[3].z);
                }
            }

        } else if (light.light_type == "point") {
            // Extract position from transform
            float3 pos = make_f3((float)light.transform.m[0][3],
                                 (float)light.transform.m[1][3],
                                 (float)light.transform.m[2][3]);

            LightInfo::SphereLightGeo pl;
            pl.position = pos;
            pl.radius = 0.01f;
            pl.emission = compute_point_light_emission(light.params, pl.radius);
            pl.name = "__point_light";
            info.sphere_lights.push_back(pl);

        } else if (light.light_type == "spot") {
            float3 pos = make_f3((float)light.transform.m[0][3],
                                 (float)light.transform.m[1][3],
                                 (float)light.transform.m[2][3]);

            LightInfo::SphereLightGeo sl;
            sl.position = pos;
            sl.radius = 0.01f;
            sl.emission = compute_spot_light_emission(light.params, sl.radius);
            sl.name = "__spot_light";
            info.sphere_lights.push_back(sl);
        } else if (light.light_type == "distant") {
            double from_arr[3] = {0, 0, 0};
            double to_arr[3] = {0, 0, 1};
            get_float3(light.params, "from", from_arr);
            get_float3(light.params, "to", to_arr);
            float3 from_pt = make_f3((float)from_arr[0], (float)from_arr[1], (float)from_arr[2]);
            float3 to_pt = make_f3((float)to_arr[0], (float)to_arr[1], (float)to_arr[2]);
            float3 dir = normalize(to_pt - from_pt);

            Color3 base = extract_color_param(light.params, "L", Color3::one());
            float scale = get_param_scale(light.params);
            const Param* illum_param = get_param(light.params, "illuminance");
            Color3 emission = base * scale;
            if (illum_param)
                emission = scale_color_to_power(base,
                                                (float)illum_param->floats[0] * scale,
                                                PI_F);

            LightInfo::DiskLightGeo dl;
            dl.normal = dir;
            dl.radius = scene_radius;
            dl.center = scene_center - dir * (scene_radius * 2.0f);
            dl.emission = emission;
            info.disk_lights.push_back(dl);
            std::printf("[PBRT] Distant light dir=(%.3f,%.3f,%.3f) → disk proxy radius=%.3f\n",
                        dir.x, dir.y, dir.z, dl.radius);
        }
    }
    return info;
}

// ─────────────────────────────────────────────────────────────────────
// Create small emissive sphere to represent a point/spot light
// ─────────────────────────────────────────────────────────────────────
void add_sphere_light_geometry(const LightInfo::SphereLightGeo& pl,
                               Scene& scene) {
    // Create emissive material
    Material mat;
    mat.name = pl.name;
    mat.type = MaterialType::Emissive;
    mat.Le = pl.emission;
    mat.Kd = Color3::zero();
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

void add_disk_light_geometry(const LightInfo::DiskLightGeo& dl, Scene& scene) {
    Material mat;
    mat.name = dl.name;
    mat.type = MaterialType::Emissive;
    mat.Le = dl.emission;
    mat.Kd = Color3::zero();
    mat.pb_brdf = PbBrdf::Emissive;
    uint32_t mat_id = (uint32_t)scene.materials.size();
    scene.materials.push_back(mat);

    float3 tangent, bitangent;
    build_orthonormal_basis(dl.normal, tangent, bitangent);
    float3 u = tangent * dl.radius;
    float3 v = bitangent * dl.radius;

    Triangle t0;
    t0.v0 = dl.center - u - v;
    t0.v1 = dl.center + u - v;
    t0.v2 = dl.center + u + v;
    t0.n0 = t0.n1 = t0.n2 = dl.normal;
    t0.uv0 = make_f2(0, 0); t0.uv1 = make_f2(1, 0); t0.uv2 = make_f2(1, 1);
    t0.material_id = mat_id;

    Triangle t1;
    t1.v0 = dl.center - u - v;
    t1.v1 = dl.center + u + v;
    t1.v2 = dl.center - u + v;
    t1.n0 = t1.n1 = t1.n2 = dl.normal;
    t1.uv0 = make_f2(0, 0); t1.uv1 = make_f2(1, 1); t1.uv2 = make_f2(0, 1);
    t1.material_id = mat_id;

    scene.triangles.push_back(t0);
    scene.triangles.push_back(t1);
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
    bool   flip_x   = false;
};

CameraInfo extract_camera(const pbrt::PbrtScene& pbrt_scene) {
    using namespace pbrt;
    CameraInfo info;

    const auto& cam = pbrt_scene.camera;
    info.fov = (float)get_float(cam.params, "fov", 90.0);

    if (cam.has_lookat) {
        info.valid = true;
        info.position = make_f3((float)cam.eye[0], (float)cam.eye[1], (float)cam.eye[2]);
        info.look_at  = make_f3((float)cam.target[0], (float)cam.target[1], (float)cam.target[2]);
        info.up       = make_f3((float)cam.up[0], (float)cam.up[1], (float)cam.up[2]);
    } else {
        // No LookAt — pre_transform is the camera-from-world (world-to-camera)
        // matrix per PBRT spec.  Invert to get camera-to-world, then extract
        // position / forward / up.
        const auto& m = cam.pre_transform;
        bool is_identity = true;
        for (int r = 0; r < 4 && is_identity; ++r)
            for (int c = 0; c < 4 && is_identity; ++c)
                if (std::abs(m.m[r][c] - (r == c ? 1.0 : 0.0)) > 1e-8)
                    is_identity = false;
        if (is_identity) return info;  // no camera info at all

        const Mat4 c2w = m.inverse();
        info.valid = true;
        info.position = make_f3((float)c2w.m[0][3], (float)c2w.m[1][3], (float)c2w.m[2][3]);
        // PBRT cameras are left-handed: forward = +Z in camera space.
        // Column 2 of c2w maps camera +Z to world, so forward = +column2.
        float3 fwd = make_f3((float)c2w.m[0][2], (float)c2w.m[1][2], (float)c2w.m[2][2]);
        info.look_at = info.position + fwd;
        info.up = make_f3((float)c2w.m[0][1], (float)c2w.m[1][1], (float)c2w.m[2][1]);
        std::printf("[PBRT] Camera from Transform matrix: pos=(%.3f,%.3f,%.3f) fwd=(%.3f,%.3f,%.3f)\n",
                    info.position.x, info.position.y, info.position.z,
                    fwd.x, fwd.y, fwd.z);
    }

    // Detect pre-world mirror (e.g. explicit Scale -1 1 1 before LookAt).
    // Only check when has_lookat: a bare Transform camera-to-world matrix
    // naturally has det≈-1 (PBRT left-handed camera convention) and should
    // NOT trigger a flip.
    if (cam.has_lookat) {
        const auto& pt = cam.pre_transform;
        double det = pt.m[0][0] * (pt.m[1][1]*pt.m[2][2] - pt.m[1][2]*pt.m[2][1])
                   - pt.m[0][1] * (pt.m[1][0]*pt.m[2][2] - pt.m[1][2]*pt.m[2][0])
                   + pt.m[0][2] * (pt.m[1][0]*pt.m[2][1] - pt.m[1][1]*pt.m[2][0]);
        info.flip_x = (det < 0);
        if (info.flip_x)
            std::printf("[PBRT] Pre-world mirror detected (det=%.2f), camera flip_x enabled\n", det);
    }

    return info;
}

// ─────────────────────────────────────────────────────────────────────
// Handle NamedMedium → HomogeneousMedium
// Returns map of medium name → index in scene.media[]
// ─────────────────────────────────────────────────────────────────────
std::unordered_map<std::string, int> process_media(
    const pbrt::PbrtScene& pbrt_scene, Scene& scene) {
    using namespace pbrt;

    std::unordered_map<std::string, int> name_to_id;

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

        med.sigma_a = Color3::from_rgb(sa_r * scale, sa_g * scale, sa_b * scale);
        med.sigma_s = Color3::from_rgb(ss_r * scale, ss_g * scale, ss_b * scale);
        med.sigma_t = med.sigma_a + med.sigma_s;
        med.g = g_val;

        int medium_id = (int)scene.media.size();
        scene.media.push_back(med);
        name_to_id[name] = medium_id;

        std::printf("[PBRT] Medium '%s': id=%d, g=%.2f, scale=%.1f\n",
                    name.c_str(), medium_id, g_val, scale);
    }

    return name_to_id;
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
                "%zu lights, %zu object templates, %zu instance refs, %zu media\n",
                pbrt_scene.shapes.size(),
                pbrt_scene.named_materials.size(),
                pbrt_scene.textures.size(),
                pbrt_scene.lights.size(),
                pbrt_scene.object_templates.size(),
                pbrt_scene.instance_refs.size(),
                pbrt_scene.named_media.size());

    // ── 2. Map materials ────────────────────────────────────────────
    pbrt::MaterialMapper mapper(pbrt_scene, scene, pbrt_scene.source_dir);
    mapper.map_all_named_materials();

    std::printf("[PBRT] Mapped %zu materials, %zu textures loaded\n",
                scene.materials.size(), scene.textures.size());
    std::fflush(stdout);

    // ── 3. Process media ────────────────────────────────────────────
    auto medium_name_to_id = process_media(pbrt_scene, scene);

    // ── 4. Pre-load PLY meshes in parallel ────────────────────────
    auto t_ply_start = std::chrono::steady_clock::now();

    // Collect all unique PLY paths from shapes + templates
    std::unordered_set<std::string> ply_paths_set;
    auto collect_ply = [&](const pbrt::PbrtShape& shape) {
        using namespace pbrt;
        std::string filename;
        if (shape.shape_type == "plymesh")
            filename = get_string(shape.params, "filename");
        else if (shape.shape_type == "trianglemesh")
            filename = get_string(shape.params, "filename");
        if (!filename.empty())
            ply_paths_set.insert(pbrt_scene.source_dir + "/" + filename);
    };
    for (const auto& shape : pbrt_scene.shapes) collect_ply(shape);
    for (const auto& [name, tpl] : pbrt_scene.object_templates)
        for (const auto& shape : tpl.shapes) collect_ply(shape);

    // Submit parallel loads
    PlyCache ply_cache;
    if (!ply_paths_set.empty()) {
        std::vector<std::string> ply_paths(ply_paths_set.begin(), ply_paths_set.end());
        int num_threads = std::min((int)ply_paths.size(),
                                   std::max(1, (int)std::thread::hardware_concurrency()));
        ThreadPool pool(num_threads);

        struct PlyResult { std::string path; pbrt::PlyMesh mesh; bool ok; };
        std::vector<std::future<PlyResult>> futures;
        futures.reserve(ply_paths.size());

        for (auto& path : ply_paths) {
            futures.push_back(pool.submit([p = path]() -> PlyResult {
                PlyResult r;
                r.path = p;
                r.ok = pbrt::load_ply(p, r.mesh);
                return r;
            }));
        }

        int loaded = 0, failed = 0;
        for (auto& f : futures) {
            auto r = f.get();
            if (r.ok) {
                ply_cache[std::move(r.path)] = std::move(r.mesh);
                ++loaded;
            } else {
                ++failed;
            }
        }

        auto t_ply_end = std::chrono::steady_clock::now();
        double ply_ms = std::chrono::duration<double, std::milli>(t_ply_end - t_ply_start).count();
        std::printf("[PBRT] Parallel PLY pre-load: %d meshes in %.0f ms (%d threads, %d failed)\n",
                    loaded, ply_ms, num_threads, failed);
    }

    // ── 5. Process shapes (sequential — uses pre-loaded cache) ──────
    std::unordered_map<std::string, uint32_t> mat_medium_clone_map;

    ShapeStats stats;
    int shape_idx = 0;
    for (const auto& shape : pbrt_scene.shapes) {
        process_shape(shape, pbrt_scene, mapper, scene, stats, ply_cache);

        // Link MediumInterface to material for shapes that have one
        if (!shape.medium_interior.empty()) {
            auto med_it = medium_name_to_id.find(shape.medium_interior);
            if (med_it != medium_name_to_id.end()) {
                // Find the mat_id on the last batch of triangles this shape added
                if (!scene.triangles.empty()) {
                    uint32_t mat_id = scene.triangles.back().material_id;
                    std::string clone_key = std::to_string(mat_id) + "|" + shape.medium_interior;

                    auto clone_it = mat_medium_clone_map.find(clone_key);
                    if (clone_it != mat_medium_clone_map.end()) {
                        // Already cloned this (material, medium) pair — reuse
                        uint32_t cloned_id = clone_it->second;
                        // Patch all triangles from this shape
                        for (size_t t = scene.triangles.size(); t > 0; --t) {
                            if (scene.triangles[t-1].material_id == mat_id)
                                scene.triangles[t-1].material_id = cloned_id;
                            else
                                break;
                        }
                    } else {
                        // Check if the material already has a medium assigned
                        Material& existing = scene.materials[mat_id];
                        if (existing.medium_id >= 0 && existing.medium_id != med_it->second) {
                            // Different medium — need to clone the material
                            Material cloned = existing;
                            cloned.medium_id = med_it->second;
                            cloned.pb_medium_enabled = true;
                            if (cloned.type == MaterialType::Glass)
                                cloned.type = MaterialType::Translucent;
                            uint32_t cloned_id = (uint32_t)scene.materials.size();
                            scene.materials.push_back(std::move(cloned));
                            mat_medium_clone_map[clone_key] = cloned_id;
                            // Patch triangles
                            for (size_t t = scene.triangles.size(); t > 0; --t) {
                                if (scene.triangles[t-1].material_id == mat_id)
                                    scene.triangles[t-1].material_id = cloned_id;
                                else
                                    break;
                            }
                        } else {
                            // First medium assignment or same medium — set directly
                            existing.medium_id = med_it->second;
                            existing.pb_medium_enabled = true;
                            if (existing.type == MaterialType::Glass)
                                existing.type = MaterialType::Translucent;
                            mat_medium_clone_map[clone_key] = mat_id;
                        }
                    }
                    std::printf("[PBRT] Linked medium '%s' (id=%d) to material '%s'\n",
                                shape.medium_interior.c_str(), med_it->second,
                                scene.materials[scene.triangles.back().material_id].name.c_str());
                }
            }
        }
        ++shape_idx;
    }

    // World-space geometry becomes meshes[0] + instances[0] (identity)
    if (!scene.triangles.empty()) {
        MeshDescriptor m0;
        m0.tri_offset = 0;
        m0.tri_count  = (uint32_t)scene.triangles.size();
        scene.meshes.push_back(m0);

        InstanceDescriptor inst0;
        inst0.mesh_id = 0;
        // Identity 3×4 row-major
        std::memset(inst0.transform, 0, sizeof(inst0.transform));
        inst0.transform[0] = 1.f; inst0.transform[5] = 1.f; inst0.transform[10] = 1.f;
        scene.instances.push_back(inst0);
    }

    // ── 4b. Process instanced templates ─────────────────────────────
    // Collect unique templates referenced by instance_refs
    std::unordered_map<std::string, uint32_t> template_mesh_id; // template name → mesh index

    for (const auto& iref : pbrt_scene.instance_refs) {
        if (template_mesh_id.count(iref.template_name)) continue; // already loaded

        auto it = pbrt_scene.object_templates.find(iref.template_name);
        if (it == pbrt_scene.object_templates.end()) continue;

        uint32_t tri_start = (uint32_t)scene.triangles.size();

        // Load all shapes in this template ONCE (in template object space)
        for (const auto& tpl_shape : it->second.shapes) {
            size_t tri_before = scene.triangles.size();
            process_shape(tpl_shape, pbrt_scene, mapper, scene, stats, ply_cache);

            // Link MediumInterface for instanced shapes too
            if (!tpl_shape.medium_interior.empty() && scene.triangles.size() > tri_before) {
                auto med_it = medium_name_to_id.find(tpl_shape.medium_interior);
                if (med_it != medium_name_to_id.end()) {
                    uint32_t mat_id = scene.triangles.back().material_id;
                    std::string clone_key = std::to_string(mat_id) + "|" + tpl_shape.medium_interior;
                    auto clone_it = mat_medium_clone_map.find(clone_key);
                    if (clone_it != mat_medium_clone_map.end()) {
                        uint32_t cloned_id = clone_it->second;
                        for (size_t t = scene.triangles.size(); t > tri_before; --t) {
                            if (scene.triangles[t-1].material_id == mat_id)
                                scene.triangles[t-1].material_id = cloned_id;
                        }
                    } else {
                        Material& existing = scene.materials[mat_id];
                        if (existing.medium_id >= 0 && existing.medium_id != med_it->second) {
                            Material cloned = existing;
                            cloned.medium_id = med_it->second;
                            cloned.pb_medium_enabled = true;
                            if (cloned.type == MaterialType::Glass)
                                cloned.type = MaterialType::Translucent;
                            uint32_t cloned_id = (uint32_t)scene.materials.size();
                            scene.materials.push_back(std::move(cloned));
                            mat_medium_clone_map[clone_key] = cloned_id;
                            for (size_t t = scene.triangles.size(); t > tri_before; --t) {
                                if (scene.triangles[t-1].material_id == mat_id)
                                    scene.triangles[t-1].material_id = cloned_id;
                            }
                        } else {
                            existing.medium_id = med_it->second;
                            existing.pb_medium_enabled = true;
                            if (existing.type == MaterialType::Glass)
                                existing.type = MaterialType::Translucent;
                            mat_medium_clone_map[clone_key] = mat_id;
                        }
                    }
                }
            }
        }

        uint32_t tri_end = (uint32_t)scene.triangles.size();
        if (tri_end > tri_start) {
            uint32_t mesh_id = (uint32_t)scene.meshes.size();
            MeshDescriptor md;
            md.tri_offset = tri_start;
            md.tri_count  = tri_end - tri_start;
            scene.meshes.push_back(md);
            template_mesh_id[iref.template_name] = mesh_id;
        }
    }

    // Create InstanceDescriptor for each instance ref
    for (const auto& iref : pbrt_scene.instance_refs) {
        auto it = template_mesh_id.find(iref.template_name);
        if (it == template_mesh_id.end()) continue;

        InstanceDescriptor inst;
        inst.mesh_id = it->second;
        // Convert Mat4 (4×4 row-major double) → float[12] (3×4 row-major)
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c)
                inst.transform[r * 4 + c] = (float)iref.transform.m[r][c];
        scene.instances.push_back(inst);
    }

    std::printf("[PBRT] Instancing: %zu meshes, %zu instances (%zu unique templates)\n",
                scene.meshes.size(), scene.instances.size(), template_mesh_id.size());

    std::printf("[PBRT] Shapes: plymesh=%d trianglemesh=%d sphere=%d disk=%d "
                "bilinear=%d skipped=%d\n",
                stats.plymesh, stats.trianglemesh, stats.sphere, stats.disk,
                stats.bilinearmesh, stats.skipped);
    std::printf("[PBRT] Total triangles: %zu\n", scene.triangles.size());
    std::fflush(stdout);

    // ── 5. Process lights ───────────────────────────────────────────
    scene.compute_bounds();
    LightInfo lights = extract_lights(pbrt_scene,
                                      scene.scene_bounding_center(),
                                      scene.scene_bounding_radius());

    scene.has_portals = !lights.portal_quads.empty();

    for (const auto& pl : lights.sphere_lights) {
        add_sphere_light_geometry(pl, scene);
        std::printf("[PBRT] Sphere light proxy at (%.2f, %.2f, %.2f) → emissive sphere\n",
                    pl.position.x, pl.position.y, pl.position.z);
    }

    for (const auto& dl : lights.disk_lights) {
        add_disk_light_geometry(dl, scene);
        std::printf("[PBRT] Distant light proxy at (%.2f, %.2f, %.2f) radius=%.2f\n",
                    dl.center.x, dl.center.y, dl.center.z, dl.radius);
    }

    // Convert portal quads → emissive triangles (opacity=0)
    if (!lights.portal_quads.empty()) {
        for (const auto& pq : lights.portal_quads) {
            Material mat;
            mat.name    = "__portal_emitter";
            mat.type    = MaterialType::Emissive;
            mat.Le      = pq.emission;
            mat.Kd      = Color3::zero();
            mat.pb_brdf = PbBrdf::Emissive;
            mat.opacity = 0.0f;  // transparent to camera/shadow rays
            uint32_t mat_id = (uint32_t)scene.materials.size();
            scene.materials.push_back(mat);

            float3 edge0 = pq.v[1] - pq.v[0];
            float3 edge1 = pq.v[3] - pq.v[0];
            float3 cr = cross(edge0, edge1);
            float  area = length(cr);
            // Negate: PBRT portal normal is outward-facing; we need the
            // triangle geo_n to point INWARD so photons emit into the room.
            float3 n = (area > 1e-8f) ? cr * (-1.0f / area) : make_f3(0, 0, -1);

            // Triangle 0: v0-v2-v1  (flipped winding → inward geo_n)
            Triangle t0;
            t0.v0 = pq.v[0]; t0.v1 = pq.v[2]; t0.v2 = pq.v[1];
            t0.n0 = t0.n1 = t0.n2 = n;
            t0.uv0 = t0.uv1 = t0.uv2 = make_f2(0, 0);
            t0.material_id = mat_id;
            scene.triangles.push_back(t0);

            // Triangle 1: v0-v3-v2  (flipped winding → inward geo_n)
            Triangle t1;
            t1.v0 = pq.v[0]; t1.v1 = pq.v[3]; t1.v2 = pq.v[2];
            t1.n0 = t1.n1 = t1.n2 = n;
            t1.uv0 = t1.uv1 = t1.uv2 = make_f2(0, 0);
            t1.material_id = mat_id;
            scene.triangles.push_back(t1);

            std::printf("[PBRT] Portal → emissive triangles: area=%.2f normal=(%.3f,%.3f,%.3f)\n",
                        area, n.x, n.y, n.z);
        }
    }

    // ── 6. Extract camera ────────────────────────────────────────────
    CameraInfo cam_info = extract_camera(pbrt_scene);
    if (cam_info.valid) {
        std::printf("[PBRT] Camera: pos=(%.2f,%.2f,%.2f) look_at=(%.2f,%.2f,%.2f) fov=%.1f\n",
                    cam_info.position.x, cam_info.position.y, cam_info.position.z,
                    cam_info.look_at.x, cam_info.look_at.y, cam_info.look_at.z,
                    cam_info.fov);

        // Populate Scene camera fields so normalize_to_reference() can
        // transform them alongside geometry.
        scene.scene_cam_valid    = true;
        scene.scene_cam_position = cam_info.position;
        scene.scene_cam_look_at  = cam_info.look_at;
        scene.scene_cam_up       = cam_info.up;
        scene.scene_cam_fov      = cam_info.fov;
        scene.scene_cam_flip_x   = cam_info.flip_x;
    }

    // ── 7. Finalize pb_* material extensions ────────────────────────
    finalize_pb_materials(scene);

    std::printf("[PBRT] Loading complete: %zu triangles, %zu materials, "
                "%zu textures, %zu media\n",
                scene.triangles.size(), scene.materials.size(),
                scene.textures.size(), scene.media.size());
    std::fflush(stdout);

    return true;
}
