// ─────────────────────────────────────────────────────────────────────
// pbrt_parser.cpp – PBRT v4 tokenizer and recursive-descent parser
// ─────────────────────────────────────────────────────────────────────
#include "scene/pbrt/pbrt_parser.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <regex>
#include <cassert>
#include <unordered_set>

namespace fs = std::filesystem;

namespace pbrt {

// =====================================================================
//  Mat4 implementation
// =====================================================================

Mat4 Mat4::identity() {
    Mat4 m{};
    m.m[0][0] = m.m[1][1] = m.m[2][2] = m.m[3][3] = 1.0;
    return m;
}

Mat4 Mat4::translate(double tx, double ty, double tz) {
    Mat4 m = identity();
    m.m[0][3] = tx;
    m.m[1][3] = ty;
    m.m[2][3] = tz;
    return m;
}

Mat4 Mat4::scale(double sx, double sy, double sz) {
    Mat4 m = identity();
    m.m[0][0] = sx;
    m.m[1][1] = sy;
    m.m[2][2] = sz;
    return m;
}

Mat4 Mat4::rotate(double angle_deg, double ax, double ay, double az) {
    double a = angle_deg * 3.14159265358979323846 / 180.0;
    double c = std::cos(a), s = std::sin(a);
    double len = std::sqrt(ax*ax + ay*ay + az*az);
    if (len < 1e-12) return identity();
    ax /= len; ay /= len; az /= len;

    Mat4 m = identity();
    m.m[0][0] = ax*ax + (1 - ax*ax)*c;
    m.m[0][1] = ax*ay*(1 - c) - az*s;
    m.m[0][2] = ax*az*(1 - c) + ay*s;
    m.m[1][0] = ax*ay*(1 - c) + az*s;
    m.m[1][1] = ay*ay + (1 - ay*ay)*c;
    m.m[1][2] = ay*az*(1 - c) - ax*s;
    m.m[2][0] = ax*az*(1 - c) - ay*s;
    m.m[2][1] = ay*az*(1 - c) + ax*s;
    m.m[2][2] = az*az + (1 - az*az)*c;
    return m;
}

Mat4 Mat4::from_column_major(const double vals[16]) {
    // PBRT stores column-major: val[col*4+row]
    // We store row-major:     m[row][col]
    Mat4 r{};
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row)
            r.m[row][col] = vals[col * 4 + row];
    return r;
}

Mat4 Mat4::operator*(const Mat4& rhs) const {
    Mat4 r{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            double sum = 0;
            for (int k = 0; k < 4; ++k)
                sum += m[i][k] * rhs.m[k][j];
            r.m[i][j] = sum;
        }
    return r;
}

// =====================================================================
//  Param helpers
// =====================================================================

const Param* get_param(const std::vector<Param>& params, const std::string& name) {
    for (auto& p : params)
        if (p.name == name) return &p;
    return nullptr;
}

double get_float(const std::vector<Param>& params, const std::string& name, double def) {
    auto* p = get_param(params, name);
    if (!p) return def;
    if (!p->floats.empty()) return p->floats[0];
    return def;
}

int get_int(const std::vector<Param>& params, const std::string& name, int def) {
    auto* p = get_param(params, name);
    if (!p) return def;
    if (!p->ints.empty()) return p->ints[0];
    if (!p->floats.empty()) return (int)p->floats[0];
    return def;
}

std::string get_string(const std::vector<Param>& params, const std::string& name,
                       const std::string& def) {
    auto* p = get_param(params, name);
    if (!p) return def;
    if (!p->strings.empty()) return p->strings[0];
    return def;
}

bool get_bool(const std::vector<Param>& params, const std::string& name, bool def) {
    auto* p = get_param(params, name);
    if (!p) return def;
    return p->boolean;
}

std::vector<double> get_rgb(const std::vector<Param>& params, const std::string& name,
                            const std::vector<double>& def) {
    auto* p = get_param(params, name);
    if (!p) return def;
    if (p->floats.size() >= 3)
        return {p->floats[0], p->floats[1], p->floats[2]};
    return def;
}

std::string get_texture_ref(const std::vector<Param>& params, const std::string& name) {
    auto* p = get_param(params, name);
    if (!p) return "";
    if (p->type == "texture" && !p->strings.empty())
        return p->strings[0];
    return "";
}

std::string get_param_type(const std::vector<Param>& params, const std::string& name) {
    auto* p = get_param(params, name);
    if (!p) return "";
    return p->type;
}

// =====================================================================
//  Tokenizer
// =====================================================================

static bool is_number_token(const std::string& s) {
    if (s.empty()) return false;
    size_t i = 0;
    if (s[i] == '-' || s[i] == '+') ++i;
    if (i >= s.size()) return false;
    bool has_digit = false;
    while (i < s.size() && (std::isdigit((unsigned char)s[i]) || s[i] == '.')) {
        if (std::isdigit((unsigned char)s[i])) has_digit = true;
        ++i;
    }
    if (!has_digit) return false;
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
        ++i;
        if (i < s.size() && (s[i] == '-' || s[i] == '+')) ++i;
        while (i < s.size() && std::isdigit((unsigned char)s[i])) ++i;
    }
    return i == s.size();
}

static std::string unquote(const std::string& s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
        return s.substr(1, s.size() - 2);
    return s;
}

std::vector<std::string> PbrtParser::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    tokens.reserve(text.size() / 4);  // rough estimate

    size_t i = 0, n = text.size();
    while (i < n) {
        char c = text[i];

        // Skip whitespace
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            ++i; continue;
        }

        // Comment: skip to end of line
        if (c == '#') {
            while (i < n && text[i] != '\n') ++i;
            continue;
        }

        // Quoted string
        if (c == '"') {
            size_t start = i;
            ++i;
            while (i < n && text[i] != '"') ++i;
            if (i < n) ++i;  // skip closing quote
            tokens.push_back(text.substr(start, i - start));
            continue;
        }

        // Brackets
        if (c == '[' || c == ']') {
            tokens.push_back(std::string(1, c));
            ++i; continue;
        }

        // Bare word / number
        size_t start = i;
        while (i < n && text[i] != ' ' && text[i] != '\t' && text[i] != '\r'
               && text[i] != '\n' && text[i] != '"' && text[i] != '['
               && text[i] != ']' && text[i] != '#') {
            ++i;
        }
        tokens.push_back(text.substr(start, i - start));
    }
    return tokens;
}

// =====================================================================
//  Parameter parser
// =====================================================================

// Valid PBRT v4 parameter type keywords
static const std::unordered_set<std::string> PBRT_PARAM_TYPES = {
    "float", "integer", "string", "bool", "boolean",
    "rgb", "spectrum", "blackbody",
    "point", "point2", "point3", "vector", "vector2", "vector3", "normal",
    "texture", "color"
};

static void coerce_param(Param& p, const std::vector<std::string>& raw) {
    if (p.type == "float") {
        for (auto& s : raw) p.floats.push_back(std::stod(unquote(s)));
    }
    else if (p.type == "integer") {
        for (auto& s : raw) p.ints.push_back((int)std::stod(unquote(s)));
    }
    else if (p.type == "string" || p.type == "texture") {
        for (auto& s : raw) p.strings.push_back(unquote(s));
    }
    else if (p.type == "rgb" || p.type == "color") {
        for (auto& s : raw) p.floats.push_back(std::stod(unquote(s)));
    }
    else if (p.type == "spectrum") {
        if (!raw.empty() && raw[0].front() == '"') {
            p.strings.push_back(unquote(raw[0]));
        } else {
            for (auto& s : raw) p.floats.push_back(std::stod(unquote(s)));
        }
    }
    else if (p.type == "blackbody") {
        for (auto& s : raw) p.floats.push_back(std::stod(unquote(s)));
    }
    else if (p.type == "bool" || p.type == "boolean") {
        if (!raw.empty()) {
            std::string val = unquote(raw[0]);
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);
            p.boolean = (val == "true" || val == "1");
        }
    }
    else if (p.type == "point" || p.type == "point3" || p.type == "vector"
             || p.type == "vector3" || p.type == "normal") {
        for (auto& s : raw) p.floats.push_back(std::stod(unquote(s)));
    }
    else if (p.type == "point2" || p.type == "vector2") {
        for (auto& s : raw) p.floats.push_back(std::stod(unquote(s)));
    }
    else {
        // Fallback: try numbers, else strings
        for (auto& s : raw) {
            std::string u = unquote(s);
            if (is_number_token(u))
                p.floats.push_back(std::stod(u));
            else
                p.strings.push_back(u);
        }
    }
}

size_t PbrtParser::parse_params(const std::vector<std::string>& tokens, size_t pos,
                                std::vector<Param>& out) {
    while (pos < tokens.size()) {
        const auto& tok = tokens[pos];

        // A quoted "type name" starts a parameter
        if (tok.size() >= 2 && tok.front() == '"' && tok.back() == '"') {
            std::string inner = tok.substr(1, tok.size() - 2);
            // Split into type + name
            auto sp = inner.find(' ');
            if (sp == std::string::npos) sp = inner.find('\t');
            if (sp == std::string::npos) break;  // not "type name" → stop

            std::string ptype = inner.substr(0, sp);
            std::string pname = inner.substr(sp + 1);
            // Trim pname
            while (!pname.empty() && (pname.front() == ' ' || pname.front() == '\t'))
                pname.erase(pname.begin());

            // Validate type keyword (to distinguish from bare string values)
            if (PBRT_PARAM_TYPES.find(ptype) == PBRT_PARAM_TYPES.end()) break;

            ++pos;

            // Collect value(s)
            std::vector<std::string> raw;
            if (pos < tokens.size() && tokens[pos] == "[") {
                ++pos;  // skip [
                while (pos < tokens.size() && tokens[pos] != "]") {
                    raw.push_back(tokens[pos]);
                    ++pos;
                }
                if (pos < tokens.size()) ++pos;  // skip ]
            }
            else if (pos < tokens.size()) {
                // Single value (unbracketed)
                const auto& next = tokens[pos];
                if (next.front() == '"' && next.back() == '"') {
                    // Could be a bare string value or next "type name"
                    std::string candidate = next.substr(1, next.size() - 2);
                    auto csp = candidate.find(' ');
                    if (csp != std::string::npos) {
                        std::string ctype = candidate.substr(0, csp);
                        if (PBRT_PARAM_TYPES.find(ctype) != PBRT_PARAM_TYPES.end()) {
                            // Next param → current has no value
                            // (empty raw)
                        } else {
                            // Multi-word string value (e.g. "textures/Sky 19.pfm")
                            raw.push_back(next);
                            ++pos;
                        }
                    } else {
                        // Single-word quoted string value
                        raw.push_back(next);
                        ++pos;
                    }
                }
                else if (next != "[" && next != "]") {
                    // Bare numeric
                    if (is_number_token(next)) {
                        raw.push_back(next);
                        ++pos;
                    }
                }
            }

            Param p;
            p.type = ptype;
            p.name = pname;
            coerce_param(p, raw);
            out.push_back(std::move(p));
        }
        else {
            break;  // next keyword / directive
        }
    }
    return pos;
}

// =====================================================================
//  File I/O
// =====================================================================

PbrtScene PbrtParser::parse_file(const std::string& filepath) {
    std::string abs_path = fs::absolute(filepath).string();
    scene_.source_dir = fs::path(abs_path).parent_path().string();
    parse_file_recursive(abs_path);
    return std::move(scene_);
}

void PbrtParser::parse_file_recursive(const std::string& filepath) {
    std::string abs_path = fs::absolute(filepath).string();
    std::string base_dir = fs::path(abs_path).parent_path().string();

    std::ifstream file(abs_path);
    if (!file.is_open()) {
        std::cerr << "[PBRT] Cannot open: " << abs_path << "\n";
        return;
    }

    // Read entire file
    std::string text((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
    file.close();

    auto tokens = tokenize(text);
    dispatch(tokens, base_dir);
}

// =====================================================================
//  Main dispatch loop
// =====================================================================

void PbrtParser::dispatch(const std::vector<std::string>& tokens, const std::string& base_dir) {
    size_t pos = 0;
    while (pos < tokens.size()) {
        std::string word = unquote(tokens[pos]);

        // ── Pre-WorldBegin directives ───────────────────────────────
        if (word == "Film") {
            ++pos;
            scene_.film.film_type = unquote(tokens[pos]); ++pos;
            pos = parse_params(tokens, pos, scene_.film.params);
        }
        else if (word == "Camera") {
            ++pos;
            scene_.camera.cam_type = unquote(tokens[pos]); ++pos;
            pos = parse_params(tokens, pos, scene_.camera.params);
            scene_.camera.pre_transform = pre_world_transform_;
        }
        else if (word == "Sampler" || word == "Integrator" || word == "PixelFilter"
                 || word == "ColorSpace" || word == "Option") {
            ++pos;
            if (pos < tokens.size()) { unquote(tokens[pos]); ++pos; }
            std::vector<Param> skip_params;
            pos = parse_params(tokens, pos, skip_params);
        }
        else if (word == "LookAt") {
            ++pos;
            double vals[9];
            for (int i = 0; i < 9 && pos < tokens.size(); ++i, ++pos)
                vals[i] = std::stod(unquote(tokens[pos]));
            scene_.camera.has_lookat = true;
            scene_.camera.eye[0] = vals[0]; scene_.camera.eye[1] = vals[1]; scene_.camera.eye[2] = vals[2];
            scene_.camera.target[0] = vals[3]; scene_.camera.target[1] = vals[4]; scene_.camera.target[2] = vals[5];
            scene_.camera.up[0] = vals[6]; scene_.camera.up[1] = vals[7]; scene_.camera.up[2] = vals[8];
        }
        else if (word == "WorldBegin") {
            ++pos;
            in_world_ = true;
            scene_.global_transform = pre_world_transform_;
            current_transform_ = Mat4::identity();
        }

        // ── Transform directives ────────────────────────────────────
        else if (word == "Scale") {
            ++pos;
            double sx = std::stod(unquote(tokens[pos])); ++pos;
            double sy = std::stod(unquote(tokens[pos])); ++pos;
            double sz = std::stod(unquote(tokens[pos])); ++pos;
            Mat4 m = Mat4::scale(sx, sy, sz);
            if (in_world_)
                current_transform_ = current_transform_ * m;
            else
                pre_world_transform_ = pre_world_transform_ * m;
        }
        else if (word == "Translate") {
            ++pos;
            double tx = std::stod(unquote(tokens[pos])); ++pos;
            double ty = std::stod(unquote(tokens[pos])); ++pos;
            double tz = std::stod(unquote(tokens[pos])); ++pos;
            Mat4 m = Mat4::translate(tx, ty, tz);
            if (in_world_)
                current_transform_ = current_transform_ * m;
            else
                pre_world_transform_ = pre_world_transform_ * m;
        }
        else if (word == "Rotate") {
            ++pos;
            double angle = std::stod(unquote(tokens[pos])); ++pos;
            double ax = std::stod(unquote(tokens[pos])); ++pos;
            double ay = std::stod(unquote(tokens[pos])); ++pos;
            double az = std::stod(unquote(tokens[pos])); ++pos;
            Mat4 m = Mat4::rotate(angle, ax, ay, az);
            if (in_world_)
                current_transform_ = current_transform_ * m;
            else
                pre_world_transform_ = pre_world_transform_ * m;
        }
        else if (word == "ConcatTransform") {
            ++pos;
            if (pos < tokens.size() && tokens[pos] == "[") ++pos;
            double vals[16];
            for (int i = 0; i < 16 && pos < tokens.size(); ++i, ++pos)
                vals[i] = std::stod(unquote(tokens[pos]));
            if (pos < tokens.size() && tokens[pos] == "]") ++pos;
            Mat4 m = Mat4::from_column_major(vals);
            if (in_world_)
                current_transform_ = current_transform_ * m;
            else
                pre_world_transform_ = pre_world_transform_ * m;
        }
        else if (word == "Transform") {
            ++pos;
            if (pos < tokens.size() && tokens[pos] == "[") ++pos;
            double vals[16];
            for (int i = 0; i < 16 && pos < tokens.size(); ++i, ++pos)
                vals[i] = std::stod(unquote(tokens[pos]));
            if (pos < tokens.size() && tokens[pos] == "]") ++pos;
            Mat4 m = Mat4::from_column_major(vals);
            if (in_world_)
                current_transform_ = m;
            else
                pre_world_transform_ = m;
        }
        else if (word == "Identity") {
            ++pos;
            if (in_world_)
                current_transform_ = Mat4::identity();
            else
                pre_world_transform_ = Mat4::identity();
        }
        else if (word == "TransformBegin") {
            ++pos;
            transform_stack_.push_back(current_transform_);
        }
        else if (word == "TransformEnd") {
            ++pos;
            if (!transform_stack_.empty()) {
                current_transform_ = transform_stack_.back();
                transform_stack_.pop_back();
            }
        }

        // ── Attribute stack ─────────────────────────────────────────
        else if (word == "AttributeBegin") {
            ++pos;
            transform_stack_.push_back(current_transform_);
            graphics_stack_.push_back({
                current_material_, current_inline_mat_, has_area_light_,
                current_area_light_params_, reverse_orientation_
            });
        }
        else if (word == "AttributeEnd") {
            ++pos;
            if (!transform_stack_.empty()) {
                current_transform_ = transform_stack_.back();
                transform_stack_.pop_back();
            }
            if (!graphics_stack_.empty()) {
                auto& gs = graphics_stack_.back();
                current_material_ = gs.material_name;
                current_inline_mat_ = gs.inline_mat;
                has_area_light_ = gs.has_area_light;
                current_area_light_params_ = gs.area_light_params;
                reverse_orientation_ = gs.reverse_orientation;
                graphics_stack_.pop_back();
            }
        }

        // ── Object instancing ───────────────────────────────────────
        else if (word == "ObjectBegin") {
            ++pos;
            std::string obj_name = unquote(tokens[pos]); ++pos;
            // Store as a temporary template (allocated in scene_)
            scene_.object_templates[obj_name] = PbrtObjectTemplate{obj_name, {}};
            in_object_ = &scene_.object_templates[obj_name];
        }
        else if (word == "ObjectEnd") {
            ++pos;
            in_object_ = nullptr;
        }
        else if (word == "ObjectInstance") {
            ++pos;
            std::string obj_name = unquote(tokens[pos]); ++pos;
            auto it = scene_.object_templates.find(obj_name);
            if (it != scene_.object_templates.end()) {
                Mat4 xform = current_transform_;
                for (auto& tpl_shape : it->second.shapes) {
                    PbrtShape s;
                    s.shape_type = tpl_shape.shape_type;
                    s.params = tpl_shape.params;
                    s.material_name = tpl_shape.material_name;
                    s.inline_mat = tpl_shape.inline_mat;
                    s.transform = xform * tpl_shape.transform;
                    s.has_area_light = tpl_shape.has_area_light;
                    s.area_light_params = tpl_shape.area_light_params;
                    s.group_name = tpl_shape.group_name;
                    s.from_instance = true;
                    s.reverse_orientation = tpl_shape.reverse_orientation;
                    scene_.shapes.push_back(std::move(s));
                }
            } else {
                std::cerr << "[PBRT] ObjectInstance references unknown object: "
                          << obj_name << "\n";
            }
        }

        // ── Include ─────────────────────────────────────────────────
        else if (word == "Include") {
            ++pos;
            std::string inc_path = unquote(tokens[pos]); ++pos;
            std::string full_path = (fs::path(base_dir) / inc_path).string();
            full_path = fs::weakly_canonical(full_path).string();
            parse_file_recursive(full_path);
        }

        // ── Texture ─────────────────────────────────────────────────
        else if (word == "Texture") {
            ++pos;
            std::string tex_name = unquote(tokens[pos]); ++pos;
            std::string tex_type = unquote(tokens[pos]); ++pos;
            std::string tex_class = unquote(tokens[pos]); ++pos;
            std::vector<Param> params;
            pos = parse_params(tokens, pos, params);
            PbrtTextureDecl td{tex_name, tex_type, tex_class, std::move(params)};
            scene_.textures[tex_name] = std::move(td);
        }

        // ── MakeNamedMaterial ───────────────────────────────────────
        else if (word == "MakeNamedMaterial") {
            ++pos;
            std::string mat_name = unquote(tokens[pos]); ++pos;
            std::vector<Param> params;
            pos = parse_params(tokens, pos, params);
            std::string mat_type = get_string(params, "type", "unknown");
            PbrtMaterial mat{mat_name, mat_type, std::move(params)};
            scene_.named_materials[mat_name] = std::move(mat);
        }

        // ── NamedMaterial (reference) ───────────────────────────────
        else if (word == "NamedMaterial") {
            ++pos;
            current_material_ = unquote(tokens[pos]); ++pos;
            current_inline_mat_ = nullptr;
        }

        // ── Inline Material ─────────────────────────────────────────
        else if (word == "Material") {
            ++pos;
            std::string mat_type = unquote(tokens[pos]); ++pos;
            std::vector<Param> params;
            pos = parse_params(tokens, pos, params);
            current_inline_mat_ = std::make_shared<PbrtMaterial>(
                PbrtMaterial{"", mat_type, std::move(params)});
            current_material_.clear();
        }

        // ── LightSource ─────────────────────────────────────────────
        else if (word == "LightSource") {
            ++pos;
            std::string lt_type = unquote(tokens[pos]); ++pos;
            std::vector<Param> params;
            pos = parse_params(tokens, pos, params);
            PbrtLight light{lt_type, std::move(params), current_transform_};
            scene_.lights.push_back(std::move(light));
        }

        // ── AreaLightSource ─────────────────────────────────────────
        else if (word == "AreaLightSource") {
            ++pos;
            std::string lt_type = unquote(tokens[pos]); ++pos;
            std::vector<Param> params;
            pos = parse_params(tokens, pos, params);
            has_area_light_ = true;
            current_area_light_params_ = std::move(params);
        }

        // ── Shape ───────────────────────────────────────────────────
        else if (word == "Shape") {
            ++pos;
            std::string shape_type = unquote(tokens[pos]); ++pos;
            std::vector<Param> params;
            pos = parse_params(tokens, pos, params);
            PbrtShape shape;
            shape.shape_type = shape_type;
            shape.params = std::move(params);
            shape.material_name = current_material_;
            shape.inline_mat = current_inline_mat_;
            shape.transform = current_transform_;
            shape.has_area_light = has_area_light_;
            shape.area_light_params = current_area_light_params_;
            shape.reverse_orientation = reverse_orientation_;

            if (in_object_)
                in_object_->shapes.push_back(std::move(shape));
            else
                scene_.shapes.push_back(std::move(shape));
        }

        // ── ReverseOrientation ──────────────────────────────────────
        else if (word == "ReverseOrientation") {
            ++pos;
            reverse_orientation_ = !reverse_orientation_;
        }

        // ── MakeNamedMedium ─────────────────────────────────────────
        else if (word == "MakeNamedMedium") {
            ++pos;
            std::string med_name = unquote(tokens[pos]); ++pos;
            std::vector<Param> params;
            pos = parse_params(tokens, pos, params);
            std::string med_type = get_string(params, "type", "homogeneous");
            scene_.named_media[med_name] = PbrtMediumDecl{med_name, med_type, std::move(params)};
        }

        // ── MediumInterface ─────────────────────────────────────────
        else if (word == "MediumInterface") {
            ++pos;
            current_medium_interior_ = unquote(tokens[pos]); ++pos;
            current_medium_exterior_ = unquote(tokens[pos]); ++pos;
        }

        // ── Unknown: skip ───────────────────────────────────────────
        else {
            ++pos;
        }
    }
}

} // namespace pbrt
