// ─────────────────────────────────────────────────────────────────────
// ply_reader.cpp – Binary PLY mesh reader (memory-mapped I/O)
// ─────────────────────────────────────────────────────────────────────
#include "scene/pbrt/ply_reader.h"
#include "core/mapped_file.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cstring>

// stb zlib decompression (implementation lives in obj_loader.cpp)
extern "C" {
    extern char* stbi_zlib_decode_noheader_malloc(const char* buffer, int len, int* outlen);
    extern void  stbi_image_free(void* retval_from_stbi_load);
}

namespace pbrt {

// ── Gzip decompression helper ───────────────────────────────────────
// Returns decompressed data, or empty vector on failure.
static std::vector<char> decompress_gzip(const std::string& filepath) {
    MappedFile mf;
    if (!mf.open(filepath)) return {};
    if (mf.size() < 18) return {};

    const auto* hdr = reinterpret_cast<const unsigned char*>(mf.data());
    if (hdr[0] != 0x1f || hdr[1] != 0x8b || hdr[2] != 0x08)
        return {};

    // Skip gzip header (RFC 1952)
    size_t pos = 10;
    uint8_t flags = hdr[3];
    if (flags & 0x04) {
        if (pos + 2 > mf.size()) return {};
        uint16_t xlen;
        std::memcpy(&xlen, mf.data() + pos, 2);
        pos += 2 + xlen;
    }
    if (flags & 0x08)
        while (pos < mf.size() && mf.data()[pos++] != '\0') {}
    if (flags & 0x10)
        while (pos < mf.size() && mf.data()[pos++] != '\0') {}
    if (flags & 0x02) pos += 2;

    if (pos >= mf.size()) return {};

    int deflate_len = (int)(mf.size() - pos - 8);
    if (deflate_len <= 0) return {};

    int out_len = 0;
    char* decompressed = stbi_zlib_decode_noheader_malloc(
        mf.data() + pos, deflate_len, &out_len);
    if (!decompressed) return {};

    std::vector<char> result(decompressed, decompressed + out_len);
    stbi_image_free(decompressed);
    return result;
}

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

// ── Buffer-based line reader (replaces std::getline) ────────────────
// Advances `pos` past the next newline, returns the line without \r\n.
static std::string_view buf_getline(const char* data, size_t size, size_t& pos) {
    size_t start = pos;
    while (pos < size && data[pos] != '\n') ++pos;
    size_t end = pos;
    if (pos < size) ++pos;  // skip \n
    if (end > start && data[end - 1] == '\r') --end;
    return {data + start, end - start};
}

// Internal implementation that parses from a contiguous memory buffer.
static bool load_ply_from_buffer(const char* data, size_t size,
                                 const std::string& filepath, PlyMesh& out);

bool load_ply(const std::string& filepath, PlyMesh& out) {
    // Detect .gz extension
    bool is_gz = false;
    {
        auto len = filepath.size();
        if (len >= 3 && filepath.compare(len - 3, 3, ".gz") == 0)
            is_gz = true;
    }

    if (is_gz) {
        auto decompressed = decompress_gzip(filepath);
        if (decompressed.empty()) {
            std::cerr << "[PLY] Failed to decompress gzip: " << filepath << "\n";
            return false;
        }
        return load_ply_from_buffer(decompressed.data(), decompressed.size(),
                                    filepath, out);
    }

    MappedFile mf;
    if (!mf.open(filepath)) {
        std::cerr << "[PLY] Cannot open: " << filepath << "\n";
        return false;
    }
    return load_ply_from_buffer(mf.data(), mf.size(), filepath, out);
}

static bool load_ply_from_buffer(const char* data, size_t size,
                                 const std::string& filepath, PlyMesh& out) {

    size_t pos = 0;

    // ── Parse header ────────────────────────────────────────────────
    auto first_line = buf_getline(data, size, pos);
    if (first_line.find("ply") == std::string_view::npos) {
        std::cerr << "[PLY] Not a PLY file: " << filepath << "\n";
        return false;
    }

    bool is_binary_le = false;
    bool is_ascii = false;
    std::vector<PlyElement> elements;

    while (pos < size) {
        auto line = buf_getline(data, size, pos);
        if (line == "end_header") break;

        // Simple word extraction from string_view
        size_t i = 0, n = line.size();
        auto skip_ws = [&]() { while (i < n && (line[i] == ' ' || line[i] == '\t')) ++i; };
        auto next_word = [&]() -> std::string {
            skip_ws();
            size_t start = i;
            while (i < n && line[i] != ' ' && line[i] != '\t') ++i;
            return std::string(line.substr(start, i - start));
        };

        std::string keyword = next_word();

        if (keyword == "format") {
            std::string fmt = next_word();
            if (fmt == "binary_little_endian") is_binary_le = true;
            else if (fmt == "ascii") is_ascii = true;
            else {
                std::cerr << "[PLY] Unsupported format: " << fmt
                          << " in " << filepath << "\n";
                return false;
            }
        }
        else if (keyword == "element") {
            std::string name = next_word();
            std::string count_str = next_word();
            int count = count_str.empty() ? 0 : std::stoi(count_str);
            elements.push_back({name, count, {}});
        }
        else if (keyword == "property") {
            if (elements.empty()) continue;
            std::string next = next_word();
            if (next == "list") {
                std::string count_type_str = next_word();
                std::string value_type_str = next_word();
                std::string prop_name = next_word();
                PlyProperty prop;
                prop.name = prop_name;
                prop.type = PlyPropType::List;
                prop.list_count_type = parse_ply_type(count_type_str);
                prop.list_value_type = parse_ply_type(value_type_str);
                elements.back().properties.push_back(prop);
            } else {
                std::string prop_name = next_word();
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

        // Bounds check for vertex block
        size_t vertex_block_size = (size_t)vertex_stride * num_verts;
        if (pos + vertex_block_size > size) {
            std::cerr << "[PLY] Unexpected end of binary vertex data in " << filepath << "\n";
            return false;
        }

        // Parse vertices directly from the buffer (zero-copy)
        const char* vdata = data + pos;
        for (int vi = 0; vi < num_verts; ++vi) {
            const char* row = vdata + vi * vertex_stride;
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
        pos += vertex_block_size;

        // Read faces — walk the buffer with pointer arithmetic (no per-face read calls)
        if (face_elem && num_faces > 0) {
            const char* fptr = data + pos;
            const char* fend = data + size;

            for (int fi = 0; fi < num_faces; ++fi) {
                auto& face_prop = face_elem->properties[0];
                int count_bytes = type_byte_size(face_prop.list_count_type);
                if (fptr + count_bytes > fend) break;
                int nv = read_int_from(fptr, face_prop.list_count_type);
                fptr += count_bytes;

                int val_bytes = type_byte_size(face_prop.list_value_type);
                int idx_block = nv * val_bytes;
                if (fptr + idx_block > fend) break;

                // Fan triangulation directly from buffer
                if (nv == 3) {
                    int i0 = read_int_from(fptr, face_prop.list_value_type);
                    int i1 = read_int_from(fptr + val_bytes, face_prop.list_value_type);
                    int i2 = read_int_from(fptr + 2 * val_bytes, face_prop.list_value_type);
                    out.faces.push_back(make_i3(i0, i1, i2));
                } else {
                    // General polygon — read indices, fan triangulate
                    int i0 = read_int_from(fptr, face_prop.list_value_type);
                    for (int t = 2; t < nv; ++t) {
                        int i_prev = read_int_from(fptr + (t-1) * val_bytes, face_prop.list_value_type);
                        int i_curr = read_int_from(fptr + t * val_bytes, face_prop.list_value_type);
                        out.faces.push_back(make_i3(i0, i_prev, i_curr));
                    }
                }
                fptr += idx_block;

                // Skip non-face properties (if any extra properties after the list)
                for (size_t pi = 1; pi < face_elem->properties.size(); ++pi) {
                    auto& fp = face_elem->properties[pi];
                    if (fp.type == PlyPropType::List) {
                        if (fptr + type_byte_size(fp.list_count_type) > fend) break;
                        int cnt = read_int_from(fptr, fp.list_count_type);
                        fptr += type_byte_size(fp.list_count_type);
                        fptr += cnt * type_byte_size(fp.list_value_type);
                    } else {
                        fptr += fp.byte_size();
                    }
                }
            }
        }
    }
    else if (is_ascii) {
        // ── ASCII mode — parse from buffer ──────────────────────────
        for (auto& elem : elements) {
            for (int ei = 0; ei < elem.count; ++ei) {
                auto line = buf_getline(data, size, pos);
                // Parse line manually
                size_t li = 0, ln = line.size();
                auto skip_ws = [&]() { while (li < ln && (line[li] == ' ' || line[li] == '\t')) ++li; };
                auto next_float = [&]() -> float {
                    skip_ws();
                    size_t start = li;
                    while (li < ln && line[li] != ' ' && line[li] != '\t') ++li;
                    return std::stof(std::string(line.substr(start, li - start)));
                };
                auto next_int = [&]() -> int {
                    skip_ws();
                    size_t start = li;
                    while (li < ln && line[li] != ' ' && line[li] != '\t') ++li;
                    return std::stoi(std::string(line.substr(start, li - start)));
                };

                if (elem.name == "vertex") {
                    std::vector<float> vals;
                    for (auto& prop : elem.properties) {
                        if (prop.type == PlyPropType::List) continue;
                        vals.push_back(next_float());
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
                    int nv = next_int();
                    std::vector<int> indices(nv);
                    for (int k = 0; k < nv; ++k) indices[k] = next_int();
                    for (int t = 2; t < nv; ++t)
                        out.faces.push_back(make_i3(indices[0], indices[t-1], indices[t]));
                }
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
