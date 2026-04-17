// ─────────────────────────────────────────────────────────────────────
// pbrt_parser.cpp – PBRT v4 tokenizer and recursive-descent parser
// ─────────────────────────────────────────────────────────────────────
#include "scene/pbrt/pbrt_parser.h"
#include "core/mapped_file.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <regex>
#include <cassert>
#include <unordered_set>
#include <stdexcept>
#include <string_view>

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

Mat4 Mat4::inverse() const {
    const double (&a)[4][4] = m;
    double s0 = a[0][0]*a[1][1] - a[1][0]*a[0][1];
    double s1 = a[0][0]*a[1][2] - a[1][0]*a[0][2];
    double s2 = a[0][0]*a[1][3] - a[1][0]*a[0][3];
    double s3 = a[0][1]*a[1][2] - a[1][1]*a[0][2];
    double s4 = a[0][1]*a[1][3] - a[1][1]*a[0][3];
    double s5 = a[0][2]*a[1][3] - a[1][2]*a[0][3];
    double c5 = a[2][2]*a[3][3] - a[3][2]*a[2][3];
    double c4 = a[2][1]*a[3][3] - a[3][1]*a[2][3];
    double c3 = a[2][1]*a[3][2] - a[3][1]*a[2][2];
    double c2 = a[2][0]*a[3][3] - a[3][0]*a[2][3];
    double c1 = a[2][0]*a[3][2] - a[3][0]*a[2][2];
    double c0 = a[2][0]*a[3][1] - a[3][0]*a[2][1];
    double det = s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0;
    if (std::fabs(det) < 1e-30) return identity();
    double inv = 1.0 / det;
    Mat4 r{};
    r.m[0][0] = ( a[1][1]*c5 - a[1][2]*c4 + a[1][3]*c3) * inv;
    r.m[0][1] = (-a[0][1]*c5 + a[0][2]*c4 - a[0][3]*c3) * inv;
    r.m[0][2] = ( a[3][1]*s5 - a[3][2]*s4 + a[3][3]*s3) * inv;
    r.m[0][3] = (-a[2][1]*s5 + a[2][2]*s4 - a[2][3]*s3) * inv;
    r.m[1][0] = (-a[1][0]*c5 + a[1][2]*c2 - a[1][3]*c1) * inv;
    r.m[1][1] = ( a[0][0]*c5 - a[0][2]*c2 + a[0][3]*c1) * inv;
    r.m[1][2] = (-a[3][0]*s5 + a[3][2]*s2 - a[3][3]*s1) * inv;
    r.m[1][3] = ( a[2][0]*s5 - a[2][2]*s2 + a[2][3]*s1) * inv;
    r.m[2][0] = ( a[1][0]*c4 - a[1][1]*c2 + a[1][3]*c0) * inv;
    r.m[2][1] = (-a[0][0]*c4 + a[0][1]*c2 - a[0][3]*c0) * inv;
    r.m[2][2] = ( a[3][0]*s4 - a[3][1]*s2 + a[3][3]*s0) * inv;
    r.m[2][3] = (-a[2][0]*s4 + a[2][1]*s2 - a[2][3]*s0) * inv;
    r.m[3][0] = (-a[1][0]*c3 + a[1][1]*c1 - a[1][2]*c0) * inv;
    r.m[3][1] = ( a[0][0]*c3 - a[0][1]*c1 + a[0][2]*c0) * inv;
    r.m[3][2] = (-a[3][0]*s3 + a[3][1]*s1 - a[3][2]*s0) * inv;
    r.m[3][3] = ( a[2][0]*s3 - a[2][1]*s1 + a[2][2]*s0) * inv;
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

std::vector<double> get_floats(const std::vector<Param>& params, const std::string& name,
                               const std::vector<double>& def) {
    auto* p = get_param(params, name);
    if (!p || p->floats.empty()) return def;
    return p->floats;
}

bool get_float3(const std::vector<Param>& params, const std::string& name, double out[3]) {
    auto* p = get_param(params, name);
    if (!p || p->floats.size() < 3) return false;
    out[0] = p->floats[0];
    out[1] = p->floats[1];
    out[2] = p->floats[2];
    return true;
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

static bool is_number_token(std::string_view s) {
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

static std::string_view unquote(std::string_view s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
        return s.substr(1, s.size() - 2);
    return s;
}

std::vector<std::string_view> PbrtParser::tokenize(const char* text, size_t len) {
    std::vector<std::string_view> tokens;
    tokens.reserve(len / 4);

    size_t i = 0;
    while (i < len) {
        char c = text[i];

        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            ++i; continue;
        }

        if (c == '#') {
            while (i < len && text[i] != '\n') ++i;
            continue;
        }

        if (c == '"') {
            size_t start = i;
            ++i;
            while (i < len && text[i] != '"') ++i;
            if (i < len) ++i;
            tokens.push_back(std::string_view(text + start, i - start));
            continue;
        }

        if (c == '[' || c == ']') {
            tokens.push_back(std::string_view(text + i, 1));
            ++i; continue;
        }

        size_t start = i;
        while (i < len && text[i] != ' ' && text[i] != '\t' && text[i] != '\r'
               && text[i] != '\n' && text[i] != '"' && text[i] != '['
               && text[i] != ']' && text[i] != '#') {
            ++i;
        }
        tokens.push_back(std::string_view(text + start, i - start));
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

static void coerce_param(Param& p, const std::vector<std::string_view>& raw) {
    if (p.type == "float") {
        for (auto& s : raw) p.floats.push_back(std::stod(std::string(unquote(s))));
    }
    else if (p.type == "integer") {
        for (auto& s : raw) p.ints.push_back((int)std::stod(std::string(unquote(s))));
    }
    else if (p.type == "string" || p.type == "texture") {
        for (auto& s : raw) p.strings.push_back(std::string(unquote(s)));
    }
    else if (p.type == "rgb" || p.type == "color") {
        for (auto& s : raw) p.floats.push_back(std::stod(std::string(unquote(s))));
    }
    else if (p.type == "spectrum") {
        if (!raw.empty() && raw[0].front() == '"') {
            p.strings.push_back(std::string(unquote(raw[0])));
        } else {
            for (auto& s : raw) p.floats.push_back(std::stod(std::string(unquote(s))));
        }
    }
    else if (p.type == "blackbody") {
        for (auto& s : raw) p.floats.push_back(std::stod(std::string(unquote(s))));
    }
    else if (p.type == "bool" || p.type == "boolean") {
        if (!raw.empty()) {
            std::string val(unquote(raw[0]));
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);
            p.boolean = (val == "true" || val == "1");
        }
    }
    else if (p.type == "point" || p.type == "point3" || p.type == "vector"
             || p.type == "vector3" || p.type == "normal") {
        for (auto& s : raw) p.floats.push_back(std::stod(std::string(unquote(s))));
    }
    else if (p.type == "point2" || p.type == "vector2") {
        for (auto& s : raw) p.floats.push_back(std::stod(std::string(unquote(s))));
    }
    else {
        for (auto& s : raw) {
            std::string u(unquote(s));
            if (is_number_token(u))
                p.floats.push_back(std::stod(u));
            else
                p.strings.push_back(u);
        }
    }
}

static void upsert_param(std::vector<Param>& params, const Param& param) {
    for (auto& existing : params) {
        if (existing.name == param.name) {
            existing = param;
            return;
        }
    }
    params.push_back(param);
}

static std::vector<Param> merge_params(const std::vector<Param>& inherited,
                                       const std::vector<Param>& local) {
    std::vector<Param> merged = inherited;
    for (const auto& param : local)
        upsert_param(merged, param);
    return merged;
}

size_t PbrtParser::parse_params(const std::vector<std::string_view>& tokens, size_t pos,
                                std::vector<Param>& out) {
    while (pos < tokens.size()) {
        const auto& tok = tokens[pos];

        // A quoted "type name" starts a parameter
        if (tok.size() >= 2 && tok.front() == '"' && tok.back() == '"') {
            auto inner = tok.substr(1, tok.size() - 2);
            auto sp = inner.find(' ');
            if (sp == std::string_view::npos) sp = inner.find('\t');
            if (sp == std::string_view::npos) break;

            std::string ptype(inner.substr(0, sp));
            std::string_view pname_sv = inner.substr(sp + 1);
            // Trim pname
            while (!pname_sv.empty() && (pname_sv.front() == ' ' || pname_sv.front() == '\t'))
                pname_sv.remove_prefix(1);
            std::string pname(pname_sv);

            if (PBRT_PARAM_TYPES.find(ptype) == PBRT_PARAM_TYPES.end()) break;

            ++pos;

            // Collect value(s)
            std::vector<std::string_view> raw;
            if (pos < tokens.size() && tokens[pos] == "[") {
                ++pos;
                while (pos < tokens.size() && tokens[pos] != "]") {
                    raw.push_back(tokens[pos]);
                    ++pos;
                }
                if (pos < tokens.size()) ++pos;
            }
            else if (pos < tokens.size()) {
                const auto& next = tokens[pos];
                if (next.front() == '"' && next.back() == '"') {
                    auto candidate = next.substr(1, next.size() - 2);
                    auto csp = candidate.find(' ');
                    if (csp != std::string_view::npos) {
                        std::string ctype(candidate.substr(0, csp));
                        if (PBRT_PARAM_TYPES.find(ctype) != PBRT_PARAM_TYPES.end()) {
                            // Next param — current has no value
                        } else {
                            raw.push_back(next);
                            ++pos;
                        }
                    } else {
                        raw.push_back(next);
                        ++pos;
                    }
                }
                else if (next != "[" && next != "]") {
                    std::string lowered(next);
                    std::transform(lowered.begin(), lowered.end(), lowered.begin(), ::tolower);
                    if (is_number_token(next)
                        || ((ptype == "bool" || ptype == "boolean")
                            && (lowered == "true" || lowered == "false"))) {
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
            break;
        }
    }
    return pos;
}

// =====================================================================
//  File I/O
// =====================================================================

PbrtScene PbrtParser::parse_file(const std::string& filepath) {
    scene_ = {};
    transform_stack_.clear();
    graphics_stack_.clear();
    current_transform_ = Mat4::identity();
    current_material_.clear();
    current_inline_mat_.reset();
    has_area_light_ = false;
    current_area_light_params_.clear();
    reverse_orientation_ = false;
    in_world_ = false;
    in_object_ = nullptr;
    pre_world_transform_ = Mat4::identity();
    coordinate_systems_.clear();
    file_stack_.clear();
    shape_attributes_.clear();
    light_attributes_.clear();
    material_attributes_.clear();
    texture_attributes_.clear();
    medium_attributes_.clear();

    std::string abs_path = fs::absolute(filepath).string();
    scene_.source_dir = fs::path(abs_path).parent_path().string();
    parse_file_recursive(abs_path);
    return std::move(scene_);
}

void PbrtParser::parse_file_recursive(const std::string& filepath) {
    std::string abs_path = fs::absolute(filepath).string();
    std::string base_dir = fs::path(abs_path).parent_path().string();

    if (std::find(file_stack_.begin(), file_stack_.end(), abs_path) != file_stack_.end()) {
        throw std::runtime_error("PBRT include/import cycle detected at: " + abs_path);
    }

    file_stack_.push_back(abs_path);

    MappedFile mf;
    if (!mf.open(abs_path)) {
        file_stack_.pop_back();
        throw std::runtime_error("Cannot open PBRT file: " + abs_path);
    }

    // Tokenize directly from the memory-mapped buffer (zero-copy string_views)
    auto tokens = tokenize(mf.data(), mf.size());
    dispatch(tokens, base_dir);
    // MappedFile stays alive through dispatch — string_views point into it
    file_stack_.pop_back();
}

// =====================================================================
//  Main dispatch loop
// =====================================================================

void PbrtParser::dispatch(const std::vector<std::string_view>& tokens, const std::string& base_dir) {
    size_t pos = 0;
    while (pos < tokens.size()) {
        auto word = unquote(tokens[pos]);

        // ── Pre-WorldBegin directives ───────────────────────────────
        if (word == "Film") {
            ++pos;
            scene_.film.film_type = std::string(unquote(tokens[pos])); ++pos;
            pos = parse_params(tokens, pos, scene_.film.params);
        }
        else if (word == "Camera") {
            ++pos;
            scene_.camera.cam_type = std::string(unquote(tokens[pos])); ++pos;
            pos = parse_params(tokens, pos, scene_.camera.params);
            scene_.camera.pre_transform = pre_world_transform_;
            coordinate_systems_["camera"] = pre_world_transform_;
        }
        else if (word == "Sampler" || word == "Integrator" || word == "PixelFilter"
                 || word == "ColorSpace" || word == "Option") {
            ++pos;
            if (pos < tokens.size()) { ++pos; }
            std::vector<Param> skip_params;
            pos = parse_params(tokens, pos, skip_params);
        }
        else if (word == "LookAt") {
            ++pos;
            double vals[9];
            for (int i = 0; i < 9 && pos < tokens.size(); ++i, ++pos)
                vals[i] = std::stod(std::string(unquote(tokens[pos])));
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
        else if (word == "CoordinateSystem") {
            ++pos;
            std::string name(unquote(tokens[pos])); ++pos;
            coordinate_systems_[name] = in_world_ ? current_transform_ : pre_world_transform_;
        }
        else if (word == "CoordSysTransform") {
            ++pos;
            std::string name(unquote(tokens[pos])); ++pos;
            auto it = coordinate_systems_.find(name);
            if (it == coordinate_systems_.end())
                throw std::runtime_error("Unknown coordinate system: " + name);
            if (in_world_)
                current_transform_ = it->second;
            else
                pre_world_transform_ = it->second;
        }
        else if (word == "TransformTimes" || word == "ActiveTransform") {
            throw std::runtime_error(
                "Unsupported PBRT animated transform directive: " + std::string(word));
        }

        // ── Transform directives ────────────────────────────────────
        else if (word == "Scale") {
            ++pos;
            double sx = std::stod(std::string(unquote(tokens[pos]))); ++pos;
            double sy = std::stod(std::string(unquote(tokens[pos]))); ++pos;
            double sz = std::stod(std::string(unquote(tokens[pos]))); ++pos;
            Mat4 m = Mat4::scale(sx, sy, sz);
            if (in_world_)
                current_transform_ = current_transform_ * m;
            else
                pre_world_transform_ = pre_world_transform_ * m;
        }
        else if (word == "Translate") {
            ++pos;
            double tx = std::stod(std::string(unquote(tokens[pos]))); ++pos;
            double ty = std::stod(std::string(unquote(tokens[pos]))); ++pos;
            double tz = std::stod(std::string(unquote(tokens[pos]))); ++pos;
            Mat4 m = Mat4::translate(tx, ty, tz);
            if (in_world_)
                current_transform_ = current_transform_ * m;
            else
                pre_world_transform_ = pre_world_transform_ * m;
        }
        else if (word == "Rotate") {
            ++pos;
            double angle = std::stod(std::string(unquote(tokens[pos]))); ++pos;
            double ax = std::stod(std::string(unquote(tokens[pos]))); ++pos;
            double ay = std::stod(std::string(unquote(tokens[pos]))); ++pos;
            double az = std::stod(std::string(unquote(tokens[pos]))); ++pos;
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
                vals[i] = std::stod(std::string(unquote(tokens[pos])));
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
                vals[i] = std::stod(std::string(unquote(tokens[pos])));
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
                current_area_light_params_, reverse_orientation_,
                current_medium_interior_, current_medium_exterior_,
                shape_attributes_, light_attributes_, material_attributes_,
                texture_attributes_, medium_attributes_
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
                current_medium_interior_ = gs.medium_interior;
                current_medium_exterior_ = gs.medium_exterior;
                shape_attributes_ = gs.shape_attributes;
                light_attributes_ = gs.light_attributes;
                material_attributes_ = gs.material_attributes;
                texture_attributes_ = gs.texture_attributes;
                medium_attributes_ = gs.medium_attributes;
                graphics_stack_.pop_back();
            }
        }
        else if (word == "Attribute") {
            ++pos;
            std::string target(unquote(tokens[pos])); ++pos;
            std::vector<Param> params;
            pos = parse_params(tokens, pos, params);

            if (target == "shape")
                shape_attributes_ = merge_params(shape_attributes_, params);
            else if (target == "light")
                light_attributes_ = merge_params(light_attributes_, params);
            else if (target == "material")
                material_attributes_ = merge_params(material_attributes_, params);
            else if (target == "texture")
                texture_attributes_ = merge_params(texture_attributes_, params);
            else if (target == "medium")
                medium_attributes_ = merge_params(medium_attributes_, params);
            else
                std::cerr << "[PBRT] Unsupported Attribute target: " << target << "\n";
        }

        // ── Object instancing ───────────────────────────────────────
        else if (word == "ObjectBegin") {
            ++pos;
            std::string obj_name(unquote(tokens[pos])); ++pos;
            transform_stack_.push_back(current_transform_);
            current_transform_ = Mat4::identity();
            scene_.object_templates[obj_name] = PbrtObjectTemplate{obj_name, {}};
            in_object_ = &scene_.object_templates[obj_name];
        }
        else if (word == "ObjectEnd") {
            ++pos;
            in_object_ = nullptr;
            if (!transform_stack_.empty()) {
                current_transform_ = transform_stack_.back();
                transform_stack_.pop_back();
            }
        }
        else if (word == "ObjectInstance") {
            ++pos;
            std::string obj_name(unquote(tokens[pos])); ++pos;
            auto it = scene_.object_templates.find(obj_name);
            if (it != scene_.object_templates.end()) {
                bool has_emissive = false;
                for (const auto& s : it->second.shapes)
                    if (s.has_area_light) { has_emissive = true; break; }

                if (has_emissive) {
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
                    scene_.instance_refs.push_back(PbrtInstanceRef{obj_name, current_transform_});
                }
            } else {
                std::cerr << "[PBRT] ObjectInstance references unknown object: "
                          << obj_name << "\n";
            }
        }

        // ── Include ─────────────────────────────────────────────────
        else if (word == "Include" || word == "Import") {
            ++pos;
            std::string inc_path(unquote(tokens[pos])); ++pos;
            std::string full_path = (fs::path(base_dir) / inc_path).string();
            full_path = fs::weakly_canonical(full_path).string();
            parse_file_recursive(full_path);
        }

        // ── Texture ─────────────────────────────────────────────────
        else if (word == "Texture") {
            ++pos;
            std::string tex_name(unquote(tokens[pos])); ++pos;
            std::string tex_type(unquote(tokens[pos])); ++pos;
            std::string tex_class(unquote(tokens[pos])); ++pos;
            std::vector<Param> params;
            pos = parse_params(tokens, pos, params);
            params = merge_params(texture_attributes_, params);
            PbrtTextureDecl td{tex_name, tex_type, tex_class, std::move(params)};
            scene_.textures[tex_name] = std::move(td);
        }

        // ── MakeNamedMaterial ───────────────────────────────────────
        else if (word == "MakeNamedMaterial") {
            ++pos;
            std::string mat_name(unquote(tokens[pos])); ++pos;
            std::vector<Param> params;
            pos = parse_params(tokens, pos, params);
            params = merge_params(material_attributes_, params);
            std::string mat_type = get_string(params, "type", "unknown");
            PbrtMaterial mat{mat_name, mat_type, std::move(params)};
            scene_.named_materials[mat_name] = std::move(mat);
        }

        // ── NamedMaterial (reference) ───────────────────────────────
        else if (word == "NamedMaterial") {
            ++pos;
            current_material_ = std::string(unquote(tokens[pos])); ++pos;
            current_inline_mat_ = nullptr;
        }

        // ── Inline Material ─────────────────────────────────────────
        else if (word == "Material") {
            ++pos;
            std::string mat_type(unquote(tokens[pos])); ++pos;
            std::vector<Param> params;
            pos = parse_params(tokens, pos, params);
            params = merge_params(material_attributes_, params);
            current_inline_mat_ = std::make_shared<PbrtMaterial>(
                PbrtMaterial{"", mat_type, std::move(params)});
            current_material_.clear();
        }

        // ── LightSource ─────────────────────────────────────────────
        else if (word == "LightSource") {
            ++pos;
            std::string lt_type(unquote(tokens[pos])); ++pos;
            std::vector<Param> params;
            pos = parse_params(tokens, pos, params);
            params = merge_params(light_attributes_, params);
            PbrtLight light{lt_type, std::move(params), current_transform_};
            scene_.lights.push_back(std::move(light));
        }

        // ── AreaLightSource ─────────────────────────────────────────
        else if (word == "AreaLightSource") {
            ++pos;
            std::string lt_type(unquote(tokens[pos])); ++pos;
            std::vector<Param> params;
            pos = parse_params(tokens, pos, params);
            has_area_light_ = true;
            current_area_light_params_ = merge_params(light_attributes_, params);
        }

        // ── Shape ───────────────────────────────────────────────────
        else if (word == "Shape") {
            ++pos;
            std::string shape_type(unquote(tokens[pos])); ++pos;
            std::vector<Param> params;
            pos = parse_params(tokens, pos, params);
            params = merge_params(shape_attributes_, params);
            PbrtShape shape;
            shape.shape_type = shape_type;
            shape.params = std::move(params);
            shape.material_name = current_material_;
            shape.inline_mat = current_inline_mat_;
            shape.transform = current_transform_;
            shape.has_area_light = has_area_light_;
            shape.area_light_params = current_area_light_params_;
            shape.reverse_orientation = reverse_orientation_;
            shape.medium_interior = current_medium_interior_;

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
            std::string med_name(unquote(tokens[pos])); ++pos;
            std::vector<Param> params;
            pos = parse_params(tokens, pos, params);
            params = merge_params(medium_attributes_, params);
            std::string med_type = get_string(params, "type", "homogeneous");
            scene_.named_media[med_name] = PbrtMediumDecl{med_name, med_type, std::move(params)};
        }

        // ── MediumInterface ─────────────────────────────────────────
        else if (word == "MediumInterface") {
            ++pos;
            current_medium_interior_ = std::string(unquote(tokens[pos])); ++pos;
            current_medium_exterior_ = std::string(unquote(tokens[pos])); ++pos;
        }

        // ── Unknown: skip ───────────────────────────────────────────
        else {
            ++pos;
        }
    }
}

} // namespace pbrt
