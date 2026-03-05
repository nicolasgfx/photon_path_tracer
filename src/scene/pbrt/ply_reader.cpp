// ─────────────────────────────────────────────────────────────────────
// ply_reader.cpp – Binary PLY mesh reader
// ─────────────────────────────────────────────────────────────────────
#include "scene/pbrt/ply_reader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cstring>

namespace pbrt {

// ── Property descriptor from PLY header ─────────────────────────────
enum class PlyPropType { Float, Double, Int8, UInt8, Int16, UInt16, Int32, UInt32, List };

struct PlyProperty {
    std::string  name;
    PlyPropType  type;
    PlyPropType  list_count_type;  // for List
    PlyPropType  list_value_type;  // for List

    int byte_size() const {
        switch (type) {
            case PlyPropType::Float:  return 4;
            case PlyPropType::Double: return 8;
            case PlyPropType::Int8:
            case PlyPropType::UInt8:  return 1;
            case PlyPropType::Int16:
            case PlyPropType::UInt16: return 2;
            case PlyPropType::Int32:
            case PlyPropType::UInt32: return 4;
            default: return 0;
        }
    }
};

static PlyPropType parse_ply_type(const std::string& s) {
    if (s == "float" || s == "float32") return PlyPropType::Float;
    if (s == "double" || s == "float64") return PlyPropType::Double;
    if (s == "char" || s == "int8") return PlyPropType::Int8;
    if (s == "uchar" || s == "uint8") return PlyPropType::UInt8;
    if (s == "short" || s == "int16") return PlyPropType::Int16;
    if (s == "ushort" || s == "uint16") return PlyPropType::UInt16;
    if (s == "int" || s == "int32") return PlyPropType::Int32;
    if (s == "uint" || s == "uint32") return PlyPropType::UInt32;
    return PlyPropType::Int32;  // fallback
}

static int type_byte_size(PlyPropType t) {
    switch (t) {
        case PlyPropType::Float:  return 4;
        case PlyPropType::Double: return 8;
        case PlyPropType::Int8:
        case PlyPropType::UInt8:  return 1;
        case PlyPropType::Int16:
        case PlyPropType::UInt16: return 2;
        case PlyPropType::Int32:
        case PlyPropType::UInt32: return 4;
        default: return 0;
    }
}

// Read a float from binary data at arbitrary PLY type
static float read_float_from(const char* data, PlyPropType t) {
    switch (t) {
        case PlyPropType::Float: {
            float v; std::memcpy(&v, data, 4); return v;
        }
        case PlyPropType::Double: {
            double v; std::memcpy(&v, data, 8); return (float)v;
        }
        case PlyPropType::Int8: return (float)*(int8_t*)data;
        case PlyPropType::UInt8: return (float)*(uint8_t*)data;
        case PlyPropType::Int16: { int16_t v; std::memcpy(&v, data, 2); return (float)v; }
        case PlyPropType::UInt16: { uint16_t v; std::memcpy(&v, data, 2); return (float)v; }
        case PlyPropType::Int32: { int32_t v; std::memcpy(&v, data, 4); return (float)v; }
        case PlyPropType::UInt32: { uint32_t v; std::memcpy(&v, data, 4); return (float)v; }
        default: return 0.f;
    }
}

static int read_int_from(const char* data, PlyPropType t) {
    switch (t) {
        case PlyPropType::Float: { float v; std::memcpy(&v, data, 4); return (int)v; }
        case PlyPropType::Double: { double v; std::memcpy(&v, data, 8); return (int)v; }
        case PlyPropType::Int8: return (int)*(int8_t*)data;
        case PlyPropType::UInt8: return (int)*(uint8_t*)data;
        case PlyPropType::Int16: { int16_t v; std::memcpy(&v, data, 2); return (int)v; }
        case PlyPropType::UInt16: { uint16_t v; std::memcpy(&v, data, 2); return (int)v; }
        case PlyPropType::Int32: { int32_t v; std::memcpy(&v, data, 4); return (int)v; }
        case PlyPropType::UInt32: { uint32_t v; std::memcpy(&v, data, 4); return (int)v; }
        default: return 0;
    }
}

struct PlyElement {
    std::string name;
    int count;
    std::vector<PlyProperty> properties;
};

bool load_ply(const std::string& filepath, PlyMesh& out) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[PLY] Cannot open: " << filepath << "\n";
        return false;
    }

    // ── Parse header ────────────────────────────────────────────────
    std::string line;
    std::getline(file, line);
    if (line.find("ply") == std::string::npos) {
        std::cerr << "[PLY] Not a PLY file: " << filepath << "\n";
        return false;
    }

    bool is_binary_le = false;
    bool is_ascii = false;
    std::vector<PlyElement> elements;

    while (std::getline(file, line)) {
        // Trim \r
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line == "end_header") break;

        std::istringstream ss(line);
        std::string keyword;
        ss >> keyword;

        if (keyword == "format") {
            std::string fmt;
            ss >> fmt;
            if (fmt == "binary_little_endian") is_binary_le = true;
            else if (fmt == "ascii") is_ascii = true;
            else {
                std::cerr << "[PLY] Unsupported format: " << fmt
                          << " in " << filepath << "\n";
                return false;
            }
        }
        else if (keyword == "element") {
            std::string name; int count;
            ss >> name >> count;
            elements.push_back({name, count, {}});
        }
        else if (keyword == "property") {
            if (elements.empty()) continue;
            std::string next;
            ss >> next;
            if (next == "list") {
                std::string count_type_str, value_type_str, prop_name;
                ss >> count_type_str >> value_type_str >> prop_name;
                PlyProperty prop;
                prop.name = prop_name;
                prop.type = PlyPropType::List;
                prop.list_count_type = parse_ply_type(count_type_str);
                prop.list_value_type = parse_ply_type(value_type_str);
                elements.back().properties.push_back(prop);
            } else {
                std::string prop_name;
                ss >> prop_name;
                PlyProperty prop;
                prop.name = prop_name;
                prop.type = parse_ply_type(next);
                elements.back().properties.push_back(prop);
            }
        }
    }

    // ── Find vertex and face elements ───────────────────────────────
    PlyElement* vertex_elem = nullptr;
    PlyElement* face_elem = nullptr;
    for (auto& elem : elements) {
        if (elem.name == "vertex") vertex_elem = &elem;
        else if (elem.name == "face") face_elem = &elem;
    }

    if (!vertex_elem) {
        std::cerr << "[PLY] No vertex element in " << filepath << "\n";
        return false;
    }

    // Find property indices for vertex
    int idx_x = -1, idx_y = -1, idx_z = -1;
    int idx_nx = -1, idx_ny = -1, idx_nz = -1;
    int idx_u = -1, idx_v = -1;
    for (int i = 0; i < (int)vertex_elem->properties.size(); ++i) {
        auto& name = vertex_elem->properties[i].name;
        if (name == "x") idx_x = i;
        else if (name == "y") idx_y = i;
        else if (name == "z") idx_z = i;
        else if (name == "nx") idx_nx = i;
        else if (name == "ny") idx_ny = i;
        else if (name == "nz") idx_nz = i;
        else if (name == "u" || name == "s" || name == "texture_u") idx_u = i;
        else if (name == "v" || name == "t" || name == "texture_v") idx_v = i;
    }

    bool has_normals = (idx_nx >= 0 && idx_ny >= 0 && idx_nz >= 0);
    bool has_uvs = (idx_u >= 0 && idx_v >= 0);

    int num_verts = vertex_elem->count;
    int num_faces = face_elem ? face_elem->count : 0;

    out.positions.resize(num_verts);
    if (has_normals) out.normals.resize(num_verts);
    if (has_uvs)     out.texcoords.resize(num_verts);
    out.faces.reserve(num_faces);

    // ── Read binary data ────────────────────────────────────────────
    if (is_binary_le) {
        // Compute vertex stride
        int vertex_stride = 0;
        for (auto& prop : vertex_elem->properties) {
            if (prop.type == PlyPropType::List) {
                std::cerr << "[PLY] List property in vertex element not supported\n";
                return false;
            }
            vertex_stride += prop.byte_size();
        }

        // Compute offsets for each property
        std::vector<int> offsets(vertex_elem->properties.size());
        {
            int off = 0;
            for (int i = 0; i < (int)vertex_elem->properties.size(); ++i) {
                offsets[i] = off;
                off += vertex_elem->properties[i].byte_size();
            }
        }

        // Read all vertex data at once
        std::vector<char> vertex_data(vertex_stride * num_verts);
        file.read(vertex_data.data(), vertex_data.size());

        for (int vi = 0; vi < num_verts; ++vi) {
            const char* row = vertex_data.data() + vi * vertex_stride;
            if (idx_x >= 0) out.positions[vi].x = read_float_from(row + offsets[idx_x], vertex_elem->properties[idx_x].type);
            if (idx_y >= 0) out.positions[vi].y = read_float_from(row + offsets[idx_y], vertex_elem->properties[idx_y].type);
            if (idx_z >= 0) out.positions[vi].z = read_float_from(row + offsets[idx_z], vertex_elem->properties[idx_z].type);
            if (has_normals) {
                out.normals[vi].x = read_float_from(row + offsets[idx_nx], vertex_elem->properties[idx_nx].type);
                out.normals[vi].y = read_float_from(row + offsets[idx_ny], vertex_elem->properties[idx_ny].type);
                out.normals[vi].z = read_float_from(row + offsets[idx_nz], vertex_elem->properties[idx_nz].type);
            }
            if (has_uvs) {
                out.texcoords[vi].x = read_float_from(row + offsets[idx_u], vertex_elem->properties[idx_u].type);
                out.texcoords[vi].y = read_float_from(row + offsets[idx_v], vertex_elem->properties[idx_v].type);
            }
        }

        // Read faces
        if (face_elem && num_faces > 0) {
            for (int fi = 0; fi < num_faces; ++fi) {
                // Each face: count_type count, then count × value_type indices
                auto& face_prop = face_elem->properties[0]; // assume first is the index list
                int count_bytes = type_byte_size(face_prop.list_count_type);
                char count_buf[8];
                file.read(count_buf, count_bytes);
                int nv = read_int_from(count_buf, face_prop.list_count_type);

                int val_bytes = type_byte_size(face_prop.list_value_type);
                std::vector<char> idx_buf(nv * val_bytes);
                file.read(idx_buf.data(), idx_buf.size());

                // Read indices
                std::vector<int> indices(nv);
                for (int k = 0; k < nv; ++k)
                    indices[k] = read_int_from(idx_buf.data() + k * val_bytes,
                                               face_prop.list_value_type);

                // Skip non-face properties (if any extra properties after the list)
                for (size_t pi = 1; pi < face_elem->properties.size(); ++pi) {
                    auto& fp = face_elem->properties[pi];
                    if (fp.type == PlyPropType::List) {
                        char cb[8];
                        file.read(cb, type_byte_size(fp.list_count_type));
                        int cnt = read_int_from(cb, fp.list_count_type);
                        file.ignore(cnt * type_byte_size(fp.list_value_type));
                    } else {
                        file.ignore(fp.byte_size());
                    }
                }

                // Fan triangulation
                for (int t = 2; t < nv; ++t) {
                    out.faces.push_back(make_i3(indices[0], indices[t-1], indices[t]));
                }
            }
        }
    }
    else if (is_ascii) {
        // ── ASCII mode ──────────────────────────────────────────────
        // Read elements in order
        for (auto& elem : elements) {
            for (int ei = 0; ei < elem.count; ++ei) {
                std::getline(file, line);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                std::istringstream ss(line);

                if (elem.name == "vertex") {
                    std::vector<float> vals;
                    for (auto& prop : elem.properties) {
                        if (prop.type == PlyPropType::List) continue;
                        float v; ss >> v;
                        vals.push_back(v);
                    }
                    if (idx_x >= 0 && idx_x < (int)vals.size()) out.positions[ei].x = vals[idx_x];
                    if (idx_y >= 0 && idx_y < (int)vals.size()) out.positions[ei].y = vals[idx_y];
                    if (idx_z >= 0 && idx_z < (int)vals.size()) out.positions[ei].z = vals[idx_z];
                    if (has_normals) {
                        if (idx_nx >= 0 && idx_nx < (int)vals.size()) out.normals[ei].x = vals[idx_nx];
                        if (idx_ny >= 0 && idx_ny < (int)vals.size()) out.normals[ei].y = vals[idx_ny];
                        if (idx_nz >= 0 && idx_nz < (int)vals.size()) out.normals[ei].z = vals[idx_nz];
                    }
                    if (has_uvs) {
                        if (idx_u >= 0 && idx_u < (int)vals.size()) out.texcoords[ei].x = vals[idx_u];
                        if (idx_v >= 0 && idx_v < (int)vals.size()) out.texcoords[ei].y = vals[idx_v];
                    }
                }
                else if (elem.name == "face") {
                    int nv; ss >> nv;
                    std::vector<int> indices(nv);
                    for (int k = 0; k < nv; ++k) ss >> indices[k];
                    for (int t = 2; t < nv; ++t)
                        out.faces.push_back(make_i3(indices[0], indices[t-1], indices[t]));
                }
                // else: skip other elements
            }
        }
    }

    return true;
}

void compute_face_normals(PlyMesh& mesh) {
    mesh.normals.resize(mesh.positions.size(), make_f3(0, 0, 0));

    // Accumulate area-weighted face normals per vertex
    for (auto& face : mesh.faces) {
        float3 v0 = mesh.positions[face.x];
        float3 v1 = mesh.positions[face.y];
        float3 v2 = mesh.positions[face.z];
        float3 e1 = v1 - v0;
        float3 e2 = v2 - v0;
        float3 fn = cross(e1, e2);
        // fn magnitude = 2*area (area-weighted)
        mesh.normals[face.x] = mesh.normals[face.x] + fn;
        mesh.normals[face.y] = mesh.normals[face.y] + fn;
        mesh.normals[face.z] = mesh.normals[face.z] + fn;
    }

    // Normalize
    for (auto& n : mesh.normals) {
        float len = length(n);
        if (len > 1e-8f)
            n = n / len;
        else
            n = make_f3(0, 0, 1);
    }
}

} // namespace pbrt
