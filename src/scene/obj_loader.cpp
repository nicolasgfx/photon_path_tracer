// ─────────────────────────────────────────────────────────────────────
// obj_loader.cpp – Wavefront OBJ/MTL parser implementation
// ─────────────────────────────────────────────────────────────────────
#include "scene/obj_loader.h"
#include "core/spectrum.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <filesystem>
#include <algorithm>
#include <cstring>

namespace fs = std::filesystem;

// ── Helper: trim whitespace ─────────────────────────────────────────
static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// ── Parse face index: v, v/vt, v/vt/vn, v//vn ──────────────────────
struct FaceVertex {
    int v  = -1;   // vertex index (0-based)
    int vt = -1;   // texcoord index
    int vn = -1;   // normal index
};

static FaceVertex parse_face_vertex(const std::string& token, int nv, int nvt, int nvn) {
    FaceVertex fv;
    // Formats: v, v/vt, v/vt/vn, v//vn
    auto parts = std::vector<std::string>{};
    std::istringstream ss(token);
    std::string part;
    while (std::getline(ss, part, '/')) {
        parts.push_back(part);
    }

    if (parts.size() >= 1 && !parts[0].empty()) {
        fv.v = std::stoi(parts[0]);
        fv.v = (fv.v > 0) ? fv.v - 1 : nv + fv.v;
    }
    if (parts.size() >= 2 && !parts[1].empty()) {
        fv.vt = std::stoi(parts[1]);
        fv.vt = (fv.vt > 0) ? fv.vt - 1 : nvt + fv.vt;
    }
    if (parts.size() >= 3 && !parts[2].empty()) {
        fv.vn = std::stoi(parts[2]);
        fv.vn = (fv.vn > 0) ? fv.vn - 1 : nvn + fv.vn;
    }
    return fv;
}

// ── Load MTL file ───────────────────────────────────────────────────
static bool load_mtl(const std::string& filepath, Scene& scene,
                     std::unordered_map<std::string, uint32_t>& mat_map,
                     const std::string& base_dir) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[MTL] Cannot open: " << filepath << "\n";
        return false;
    }

    std::cout << "[MTL] Loading: " << filepath << "\n";
    Material* current = nullptr;

    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        std::string keyword;
        ss >> keyword;

        if (keyword == "newmtl") {
            std::string name;
            std::getline(ss >> std::ws, name);
            name = trim(name);

            Material mat;
            mat.name = name;
            mat_map[name] = (uint32_t)scene.materials.size();
            scene.materials.push_back(mat);
            current = &scene.materials.back();
        }
        else if (!current) {
            continue;
        }
        else if (keyword == "Kd") {
            float r, g, b;
            ss >> r >> g >> b;
            current->Kd = rgb_to_spectrum_reflectance(r, g, b);
        }
        else if (keyword == "Ks") {
            float r, g, b;
            ss >> r >> g >> b;
            current->Ks = rgb_to_spectrum_reflectance(r, g, b);
        }
        else if (keyword == "Ke") {
            // Emission
            float r, g, b;
            ss >> r >> g >> b;
            if (r > 0.f || g > 0.f || b > 0.f) {
                // Convert RGB emission to spectral using emission basis
                // (preserves absolute intensity, no white-normalisation)
                current->Le = rgb_to_spectrum_emission(r, g, b);
                current->type = MaterialType::Emissive;
            }
        }
        else if (keyword == "Ns") {
            float ns;
            ss >> ns;
            // Convert Phong exponent to roughness
            current->roughness = sqrtf(2.0f / (ns + 2.0f));
        }
        else if (keyword == "Ni") {
            ss >> current->ior;
        }
        else if (keyword == "illum") {
            int illum;
            ss >> illum;
            // Don't let illum override an already-set Emissive type
            // (Ke line may have been parsed before illum)
            if (current->type == MaterialType::Emissive) continue;
            switch (illum) {
                case 0: case 1:
                    current->type = MaterialType::Lambertian;
                    break;
                case 2:
                    // Diffuse + specular
                    if (current->Ks.max_component() > 0.01f)
                        current->type = MaterialType::GlossyMetal;
                    break;
                case 3:
                    current->type = MaterialType::Mirror;
                    break;
                case 4: case 6: case 7:
                    current->type = MaterialType::Glass;
                    break;
                case 5:
                    current->type = MaterialType::Mirror;
                    break;
                default:
                    break;
            }
        }
        else if (keyword == "map_Kd") {
            std::string tex_path;
            std::getline(ss >> std::ws, tex_path);
            tex_path = trim(tex_path);
            // Texture loading would go here (stb_image)
            // For now, store the path reference
            (void)base_dir;
            (void)tex_path;
        }
    }

    std::cout << "[MTL] Loaded " << scene.materials.size() << " materials\n";
    return true;
}

// ── Load OBJ ────────────────────────────────────────────────────────
bool load_obj(const std::string& filepath, Scene& scene) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[OBJ] Cannot open: " << filepath << "\n";
        return false;
    }

    std::cout << "[OBJ] Loading: " << filepath << "\n";
    std::string base_dir = fs::path(filepath).parent_path().string();

    std::vector<float3> positions;
    std::vector<float3> vertex_colors;    // Optional per-vertex colors
    std::vector<float3> normals;
    std::vector<float2> texcoords;

    std::unordered_map<std::string, uint32_t> mat_map;
    uint32_t current_material = 0;
    bool has_vertex_colors = false;

    // Add a default material
    {
        Material default_mat;
        default_mat.name = "__default__";
        default_mat.Kd   = Spectrum::constant(0.5f);
        mat_map["__default__"] = 0;
        scene.materials.push_back(default_mat);
    }

    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        line_num++;
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        std::string keyword;
        ss >> keyword;

        if (keyword == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            positions.push_back(make_f3(x, y, z));

            // Check for per-vertex colors (v x y z r g b)
            float r, g, b;
            if (ss >> r >> g >> b) {
                vertex_colors.push_back(make_f3(r, g, b));
                has_vertex_colors = true;
            } else if (has_vertex_colors) {
                // Maintain alignment if some verts have colors and some don't
                vertex_colors.push_back(make_f3(1, 1, 1));
            }
        }
        else if (keyword == "vn") {
            float x, y, z;
            ss >> x >> y >> z;
            normals.push_back(make_f3(x, y, z));
        }
        else if (keyword == "vt") {
            float u, v;
            ss >> u >> v;
            texcoords.push_back(make_f2(u, v));
        }
        else if (keyword == "f") {
            // Parse face vertices
            std::vector<FaceVertex> face_verts;
            std::string token;
            while (ss >> token) {
                face_verts.push_back(parse_face_vertex(
                    token, (int)positions.size(), (int)texcoords.size(), (int)normals.size()));
            }

            // Triangulate (fan triangulation for convex polygons)
            for (size_t i = 2; i < face_verts.size(); ++i) {
                Triangle tri;

                auto get_pos = [&](const FaceVertex& fv) -> float3 {
                    if (fv.v >= 0 && fv.v < (int)positions.size())
                        return positions[fv.v];
                    return make_f3(0,0,0);
                };

                auto get_normal = [&](const FaceVertex& fv) -> float3 {
                    if (fv.vn >= 0 && fv.vn < (int)normals.size())
                        return normals[fv.vn];
                    return make_f3(0,0,0);
                };

                auto get_uv = [&](const FaceVertex& fv) -> float2 {
                    if (fv.vt >= 0 && fv.vt < (int)texcoords.size())
                        return texcoords[fv.vt];
                    return make_f2(0,0);
                };

                tri.v0 = get_pos(face_verts[0]);
                tri.v1 = get_pos(face_verts[i-1]);
                tri.v2 = get_pos(face_verts[i]);

                tri.uv0 = get_uv(face_verts[0]);
                tri.uv1 = get_uv(face_verts[i-1]);
                tri.uv2 = get_uv(face_verts[i]);

                // Normals: use provided or compute geometric
                bool has_normals = (face_verts[0].vn >= 0);
                if (has_normals) {
                    tri.n0 = get_normal(face_verts[0]);
                    tri.n1 = get_normal(face_verts[i-1]);
                    tri.n2 = get_normal(face_verts[i]);
                } else {
                    float3 gn = tri.geometric_normal();
                    tri.n0 = tri.n1 = tri.n2 = gn;
                }

                tri.material_id = current_material;

                // If the cornell box has per-vertex colors and no materials,
                // create a material from the average vertex color
                if (has_vertex_colors && current_material == 0) {
                    float3 c0 = (face_verts[0].v < (int)vertex_colors.size())
                        ? vertex_colors[face_verts[0].v] : make_f3(0.5f, 0.5f, 0.5f);
                    float3 c1 = (face_verts[i-1].v < (int)vertex_colors.size())
                        ? vertex_colors[face_verts[i-1].v] : make_f3(0.5f, 0.5f, 0.5f);
                    float3 c2 = (face_verts[i].v < (int)vertex_colors.size())
                        ? vertex_colors[face_verts[i].v] : make_f3(0.5f, 0.5f, 0.5f);

                    // Average vertex color for this triangle
                    float3 avg = (c0 + c1 + c2) / 3.f;
                    // Use the default material but we'll set per-triangle Kd later
                    // For now, update the default material Kd
                    // (A proper implementation would use per-vertex color interpolation)
                    (void)avg;
                }

                scene.triangles.push_back(tri);
            }
        }
        else if (keyword == "mtllib") {
            std::string mtl_file;
            std::getline(ss >> std::ws, mtl_file);
            mtl_file = trim(mtl_file);
            std::string mtl_path = base_dir + "/" + mtl_file;
            load_mtl(mtl_path, scene, mat_map, base_dir);
        }
        else if (keyword == "usemtl") {
            std::string name;
            std::getline(ss >> std::ws, name);
            name = trim(name);
            auto it = mat_map.find(name);
            if (it != mat_map.end()) {
                current_material = it->second;
            } else {
                std::cerr << "[OBJ] Unknown material: " << name << " (line " << line_num << ")\n";
            }
        }
        else if (keyword == "o" || keyword == "g" || keyword == "s") {
            // Object/group/smoothing group – ignore for now
        }
    }

    std::cout << "[OBJ] Loaded " << positions.size() << " vertices, "
              << scene.triangles.size() << " triangles, "
              << scene.materials.size() << " materials\n";

    if (has_vertex_colors) {
        std::cout << "[OBJ] Per-vertex colors detected (radiosity data)\n";
    }

    return true;
}
