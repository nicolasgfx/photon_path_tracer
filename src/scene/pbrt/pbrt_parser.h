#pragma once
// ─────────────────────────────────────────────────────────────────────
// pbrt_parser.h – PBRT v4 tokenizer and recursive-descent parser
// ─────────────────────────────────────────────────────────────────────
// Parses PBRT v4 text-format scene files into an intermediate PbrtScene
// structure.  Handles: Include, AttributeBegin/End, TransformBegin/End,
// ObjectBegin/End + ObjectInstance, transform stack, Texture/Material/
// Shape/Light directives, Camera/Film/LookAt.
// ─────────────────────────────────────────────────────────────────────

#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <variant>
#include <memory>
#include <cstdint>

namespace pbrt {

// ── 4×4 column-major transform ──────────────────────────────────────
struct Mat4 {
    double m[4][4];  // row, col

    static Mat4 identity();
    static Mat4 translate(double tx, double ty, double tz);
    static Mat4 scale(double sx, double sy, double sz);
    static Mat4 rotate(double angle_deg, double ax, double ay, double az);
    static Mat4 from_column_major(const double vals[16]);

    Mat4 operator*(const Mat4& rhs) const;  // this * rhs
};

// ── Typed parameter value ───────────────────────────────────────────
// PBRT parameters: "type name" [values]
struct Param {
    std::string type;   // "float","integer","string","rgb","spectrum",
                        // "texture","bool","blackbody","point3","normal","point2"
    std::string name;   // "reflectance","filename","fov", …

    // Value storage: always stored as one of these
    std::vector<double>      floats;   // float, rgb, spectrum, blackbody, point, normal
    std::vector<int>         ints;     // integer
    std::vector<std::string> strings;  // string, texture
    bool                     boolean = false; // bool
};

// Helper: get param by name
const Param* get_param(const std::vector<Param>& params, const std::string& name);
double       get_float(const std::vector<Param>& params, const std::string& name, double def);
int          get_int(const std::vector<Param>& params, const std::string& name, int def);
std::string  get_string(const std::vector<Param>& params, const std::string& name,
                        const std::string& def = "");
bool         get_bool(const std::vector<Param>& params, const std::string& name, bool def);
std::vector<double> get_rgb(const std::vector<Param>& params, const std::string& name,
                            const std::vector<double>& def = {});
std::string  get_texture_ref(const std::vector<Param>& params, const std::string& name);
std::string  get_param_type(const std::vector<Param>& params, const std::string& name);

// ── Parsed data structures ──────────────────────────────────────────

struct PbrtTextureDecl {
    std::string tex_name;   // declared name
    std::string tex_type;   // "spectrum" | "float"
    std::string tex_class;  // "imagemap" | "scale" | "constant" | "mix"
    std::vector<Param> params;
};

struct PbrtMaterial {
    std::string mat_name;   // empty for inline Material
    std::string mat_type;   // "coateddiffuse","diffuse","conductor", …
    std::vector<Param> params;
};

struct PbrtShape {
    std::string shape_type;  // "plymesh","trianglemesh","sphere","disk","bilinearmesh"
    std::vector<Param> params;
    std::string material_name;                  // NamedMaterial reference
    std::shared_ptr<PbrtMaterial> inline_mat;   // inline Material (may be null)
    Mat4 transform = Mat4::identity();
    bool has_area_light = false;
    std::vector<Param> area_light_params;       // AreaLightSource params
    std::string group_name;
    bool from_instance = false;
    bool reverse_orientation = false;
    std::string medium_interior;                // MediumInterface interior name (may be empty)
};

struct PbrtLight {
    std::string light_type;  // "infinite","point","spot","distant","diffuse"
    std::vector<Param> params;
    Mat4 transform = Mat4::identity();
};

struct PbrtCamera {
    std::string cam_type = "perspective";
    std::vector<Param> params;
    bool has_lookat = false;
    double eye[3]    = {0, 0, 0};
    double target[3] = {0, 0, -1};
    double up[3]     = {0, 1, 0};
    Mat4 pre_transform = Mat4::identity();
};

struct PbrtFilm {
    std::string film_type = "rgb";
    std::vector<Param> params;
};

struct PbrtObjectTemplate {
    std::string name;
    std::vector<PbrtShape> shapes;
};

struct PbrtInstanceRef {
    std::string template_name;
    Mat4 transform;
};

struct PbrtMediumDecl {
    std::string name;
    std::string type;
    std::vector<Param> params;
};

struct PbrtScene {
    PbrtCamera camera;
    PbrtFilm   film;
    Mat4       global_transform = Mat4::identity();
    std::string source_dir;

    std::vector<PbrtLight> lights;
    std::unordered_map<std::string, PbrtTextureDecl>  textures;
    std::unordered_map<std::string, PbrtMaterial>      named_materials;
    std::vector<PbrtShape> shapes;
    std::vector<PbrtInstanceRef> instance_refs;
    std::unordered_map<std::string, PbrtObjectTemplate> object_templates;
    std::unordered_map<std::string, PbrtMediumDecl>     named_media;
};

// ── Parser ──────────────────────────────────────────────────────────

class PbrtParser {
public:
    PbrtScene parse_file(const std::string& filepath);

private:
    PbrtScene scene_;

    // State
    std::vector<Mat4> transform_stack_;
    Mat4 current_transform_ = Mat4::identity();
    std::string current_material_;
    std::shared_ptr<PbrtMaterial> current_inline_mat_;
    bool has_area_light_ = false;
    std::vector<Param> current_area_light_params_;
    bool reverse_orientation_ = false;
    bool in_world_ = false;
    PbrtObjectTemplate* in_object_ = nullptr;
    Mat4 pre_world_transform_ = Mat4::identity();

    // Material/area-light/reverseOrientation stacked state
    struct GraphicsState {
        std::string material_name;
        std::shared_ptr<PbrtMaterial> inline_mat;
        bool has_area_light;
        std::vector<Param> area_light_params;
        bool reverse_orientation;
    };
    std::vector<GraphicsState> graphics_stack_;

    // MediumInterface
    std::string current_medium_interior_;
    std::string current_medium_exterior_;

    void parse_file_recursive(const std::string& filepath);
    void dispatch(const std::vector<std::string>& tokens, const std::string& base_dir);

    // Tokenizer
    static std::vector<std::string> tokenize(const std::string& text);

    // Parameter parser: parses "type name" [values] pairs
    static size_t parse_params(const std::vector<std::string>& tokens, size_t pos,
                               std::vector<Param>& out);
};

} // namespace pbrt
