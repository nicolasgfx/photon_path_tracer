// ─────────────────────────────────────────────────────────────────────
// pbrt_material_mapper.cpp – Map PBRT v4 materials → renderer Materials
// ─────────────────────────────────────────────────────────────────────
#include "scene/pbrt/pbrt_material_mapper.h"
#include "core/spectrum.h"

#include "stb_image.h"

#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cmath>

namespace fs = std::filesystem;

namespace pbrt {

// ── Well-known conductor spectra (R≈650nm, G≈550nm, B≈450nm) ────────
struct ConductorPreset { const char* name; float eta[3]; float k[3]; };
static const ConductorPreset CONDUCTOR_PRESETS[] = {
    {"metal-Al-eta",  {1.34f, 0.96f, 0.50f}, {7.47f, 6.40f, 5.30f}},  // aluminum
    {"metal-Cu-eta",  {0.21f, 0.92f, 1.16f}, {3.58f, 2.60f, 2.30f}},  // copper
    {"metal-Au-eta",  {0.16f, 0.42f, 1.47f}, {3.98f, 2.38f, 1.60f}},  // gold
    {"metal-Ag-eta",  {0.05f, 0.06f, 0.05f}, {4.28f, 3.52f, 2.73f}},  // silver
    {"metal-Fe-eta",  {2.87f, 2.95f, 2.65f}, {3.12f, 2.93f, 2.77f}},  // iron
    {"metal-Ti-eta",  {2.16f, 1.93f, 1.72f}, {2.56f, 2.37f, 2.18f}},  // titanium
    {"metal-Cr-eta",  {3.11f, 3.18f, 2.17f}, {3.31f, 3.32f, 3.20f}},  // chromium
    {"metal-W-eta",   {4.37f, 3.31f, 2.99f}, {3.27f, 2.69f, 2.54f}},  // tungsten
    {"metal-Ni-eta",  {1.98f, 1.70f, 1.67f}, {3.74f, 3.01f, 2.50f}},  // nickel
    {"metal-Pt-eta",  {2.38f, 2.04f, 1.69f}, {4.26f, 3.72f, 3.13f}},  // platinum
    {"metal-Co-eta",  {2.18f, 2.00f, 1.55f}, {4.09f, 3.59f, 3.36f}},  // cobalt
    {"metal-Pd-eta",  {1.66f, 1.27f, 0.82f}, {4.33f, 3.55f, 2.88f}},  // palladium
    {"metal-Zn-eta",  {1.10f, 0.64f, 1.21f}, {5.55f, 4.76f, 3.57f}},  // zinc
};

// ── Blackbody → linear sRGB ─────────────────────────────────────────
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

// ── Safe float extraction ───────────────────────────────────────────
static float safe_float(const Param* p, float def) {
    if (!p) return def;
    if (!p->floats.empty()) return (float)p->floats[0];
    return def;
}

static float safe_float_or_lum(const Param* p, float def) {
    if (!p) return def;
    if (p->floats.size() >= 3)
        return 0.2126f * (float)p->floats[0] + 0.7152f * (float)p->floats[1]
             + 0.0722f * (float)p->floats[2];
    if (!p->floats.empty()) return (float)p->floats[0];
    return def;
}

// =====================================================================
//  MaterialMapper
// =====================================================================

MaterialMapper::MaterialMapper(const PbrtScene& pbrt_scene, Scene& scene,
                               const std::string& pbrt_source_dir)
    : pbrt_scene_(pbrt_scene), scene_(scene), source_dir_(pbrt_source_dir) {}

void MaterialMapper::map_all_named_materials() {
    for (auto& [name, pbrt_mat] : pbrt_scene_.named_materials) {
        map_one_material(name, pbrt_mat);
    }
    std::cout << "[PBRT] Mapped " << mat_index_.size() << " named materials\n";
}

uint32_t MaterialMapper::resolve_shape_material(const PbrtShape& shape) {
    // Named material reference
    if (!shape.material_name.empty()) {
        auto it = mat_index_.find(shape.material_name);
        if (it != mat_index_.end()) return it->second;
        // Try to find in named_materials and map on demand
        auto mit = pbrt_scene_.named_materials.find(shape.material_name);
        if (mit != pbrt_scene_.named_materials.end())
            return map_one_material(shape.material_name, mit->second);
        std::cerr << "[PBRT] Unknown material reference: " << shape.material_name << "\n";
    }

    // Inline material
    if (shape.inline_mat) {
        std::string syn_name = "_inline_" + shape.inline_mat->mat_type
                             + "_" + std::to_string(inline_counter_++);
        return map_one_material(syn_name, *shape.inline_mat);
    }

    // No material — return or create default
    auto dit = mat_index_.find("__default__");
    if (dit != mat_index_.end()) return dit->second;
    Material def;
    def.name = "__default__";
    def.Kd = Spectrum::constant(0.5f);
    uint32_t idx = (uint32_t)scene_.materials.size();
    scene_.materials.push_back(def);
    mat_index_["__default__"] = idx;
    return idx;
}

uint32_t MaterialMapper::create_emissive_material(
    const std::string& base_name,
    const std::vector<Param>& area_light_params)
{
    // Deduplicate by name
    auto it = mat_index_.find(base_name);
    if (it != mat_index_.end()) return it->second;

    Material mat;
    mat.name = base_name;
    mat.pb_brdf = PbBrdf::Emissive;

    // Check for blackbody
    auto* L_param = get_param(area_light_params, "L");
    float light_scale = (float)get_float(area_light_params, "scale", 1.0);

    bool is_blackbody = false;
    if (L_param && L_param->type == "blackbody") {
        is_blackbody = true;
        float temp_K = L_param->floats.empty() ? 6500.f
                                                : (float)L_param->floats[0];
        float rgb[3];
        blackbody_to_rgb(temp_K, rgb);
        mat.Le = rgb_to_spectrum_emission(
            rgb[0] * light_scale, rgb[1] * light_scale, rgb[2] * light_scale);
    }

    if (!is_blackbody) {
        if (L_param && L_param->floats.size() >= 3) {
            float r = (float)L_param->floats[0] * light_scale;
            float g = (float)L_param->floats[1] * light_scale;
            float b = (float)L_param->floats[2] * light_scale;
            mat.Le = rgb_to_spectrum_emission(r, g, b);
        } else {
            mat.Le = rgb_to_spectrum_emission(light_scale, light_scale, light_scale);
        }
    }

    mat.type = MaterialType::Emissive;

    uint32_t idx = (uint32_t)scene_.materials.size();
    scene_.materials.push_back(std::move(mat));
    mat_index_[base_name] = idx;
    return idx;
}

// =====================================================================
//  Single material mapping
// =====================================================================

uint32_t MaterialMapper::map_one_material(const std::string& name,
                                          const PbrtMaterial& pbrt_mat) {
    auto it = mat_index_.find(name);
    if (it != mat_index_.end()) return it->second;

    Material mat;
    mat.name = name;

    const auto& mt = pbrt_mat.mat_type;
    const auto& params = pbrt_mat.params;

    if (mt == "coateddiffuse")        map_coated_diffuse(mat, params);
    else if (mt == "diffuse")         map_diffuse(mat, params);
    else if (mt == "dielectric")      map_dielectric(mat, params);
    else if (mt == "thindielectric")  map_thin_dielectric(mat, params);
    else if (mt == "conductor")       map_conductor(mat, params);
    else if (mt == "coatedconductor") map_coated_conductor(mat, params);
    else if (mt == "measured")        map_measured(mat, params);
    else if (mt == "diffusetransmission") map_diffuse_transmission(mat, params);
    else if (mt == "mix")             map_mix(mat, pbrt_mat);
    else {
        // Unknown → default lambert
        mat.pb_brdf = PbBrdf::Lambert;
        std::cerr << "[PBRT] Unknown material type '" << mt << "' for " << name << "\n";
    }

    uint32_t idx = (uint32_t)scene_.materials.size();
    scene_.materials.push_back(std::move(mat));
    mat_index_[name] = idx;
    return idx;
}

// =====================================================================
//  Per-type mappers
// =====================================================================

void MaterialMapper::map_diffuse(Material& mat, const std::vector<Param>& params) {
    mat.pb_brdf = PbBrdf::Lambert;
    mat.pb_roughness = 1.0f;
    mat.pb_roughness_set = true;

    std::string ref_type = get_param_type(params, "reflectance");
    if (ref_type == "texture") {
        std::string tex_name = get_texture_ref(params, "reflectance");
        int tid = resolve_texture(tex_name);
        if (tid >= 0) mat.diffuse_tex = tid;
    } else {
        auto rgb = get_rgb(params, "reflectance", {0.5, 0.5, 0.5});
        mat.Kd = rgb_to_spectrum_reflectance((float)rgb[0], (float)rgb[1], (float)rgb[2]);
    }
}

void MaterialMapper::map_coated_diffuse(Material& mat, const std::vector<Param>& params) {
    mat.pb_brdf = PbBrdf::Clearcoat;
    mat.pb_clearcoat = 1.0f;
    mat.pb_clearcoat_set = true;
    mat.pb_base_brdf = PbBrdf::Lambert;

    // Reflectance (diffuse base)
    std::string ref_type = get_param_type(params, "reflectance");
    if (ref_type == "texture") {
        std::string tex_name = get_texture_ref(params, "reflectance");
        int tid = resolve_texture(tex_name);
        if (tid >= 0) mat.diffuse_tex = tid;
        mat.Kd = Spectrum::constant(0.5f);
    } else {
        auto rgb = get_rgb(params, "reflectance", {0.5, 0.5, 0.5});
        mat.Kd = rgb_to_spectrum_reflectance((float)rgb[0], (float)rgb[1], (float)rgb[2]);
    }

    // Roughness
    bool remap = get_bool(params, "remaproughness", true);
    auto* ur_p = get_param(params, "uroughness");
    auto* vr_p = get_param(params, "vroughness");
    auto* r_p  = get_param(params, "roughness");

    if (ur_p && vr_p) {
        float ur = pbrt_roughness_to_ours(safe_float(ur_p, 0.f), remap);
        float vr = pbrt_roughness_to_ours(safe_float(vr_p, 0.f), remap);
        mat.pb_roughness_x = ur;
        mat.pb_roughness_y = vr;
        mat.pb_roughness_xy_set = true;
        mat.pb_clearcoat_roughness = std::sqrt(ur * vr);
    } else if (r_p) {
        mat.pb_clearcoat_roughness = pbrt_roughness_to_ours(safe_float(r_p, 0.f), remap);
    } else {
        mat.pb_clearcoat_roughness = 0.0f;
    }
    mat.pb_clearcoat_set = true;

    // Coat IOR
    float eta = (float)get_float(params, "eta", 1.5);
    mat.pb_eta = eta;
    mat.pb_eta_set = true;
    mat.ior = eta;

    // Displacement/bump
    std::string disp_type = get_param_type(params, "displacement");
    if (disp_type == "texture") {
        std::string tex_name = get_texture_ref(params, "displacement");
        int tid = resolve_texture(tex_name);
        if (tid >= 0) mat.bump_tex = tid;
    }
}

void MaterialMapper::map_conductor(Material& mat, const std::vector<Param>& params) {
    mat.pb_brdf = PbBrdf::Conductor;
    mat.Kd = Spectrum::zero();

    float eta_rgb[3], k_rgb[3];
    resolve_conductor_eta_k(params, eta_rgb, k_rgb);

    mat.pb_conductor_eta_rgb[0] = eta_rgb[0]; mat.pb_conductor_eta_rgb[1] = eta_rgb[1]; mat.pb_conductor_eta_rgb[2] = eta_rgb[2];
    mat.pb_conductor_k_rgb[0] = k_rgb[0]; mat.pb_conductor_k_rgb[1] = k_rgb[1]; mat.pb_conductor_k_rgb[2] = k_rgb[2];
    mat.pb_conductor_set = true;

    // Roughness
    bool remap = get_bool(params, "remaproughness", true);
    auto* ur_p = get_param(params, "uroughness");
    auto* vr_p = get_param(params, "vroughness");
    auto* r_p  = get_param(params, "roughness");

    if (ur_p && vr_p) {
        float ur = pbrt_roughness_to_ours(safe_float(ur_p, 0.f), remap);
        float vr = pbrt_roughness_to_ours(safe_float(vr_p, 0.f), remap);
        mat.pb_roughness_x = ur;
        mat.pb_roughness_y = vr;
        mat.pb_roughness_xy_set = true;
        mat.pb_roughness = std::sqrt(ur * vr);
    } else if (r_p) {
        mat.pb_roughness = pbrt_roughness_to_ours(safe_float(r_p, 0.f), remap);
    } else {
        mat.pb_roughness = 0.0f;
    }
    mat.pb_roughness_set = true;
}

void MaterialMapper::map_coated_conductor(Material& mat, const std::vector<Param>& params) {
    mat.pb_brdf = PbBrdf::Clearcoat;
    mat.pb_base_brdf = PbBrdf::Conductor;
    mat.Kd = Spectrum::zero();

    // Conductor base
    float eta_rgb[3], k_rgb[3];
    // Try conductor.eta / conductor.k, fallback to eta/k
    auto* ce_p = get_param(params, "conductor.eta");
    auto* ck_p = get_param(params, "conductor.k");
    if (!ce_p) ce_p = get_param(params, "eta");
    if (!ck_p) ck_p = get_param(params, "k");

    // Build a temporary params list for resolve_conductor_eta_k
    std::vector<Param> cond_params;
    if (ce_p) { Param p = *ce_p; p.name = "eta"; cond_params.push_back(p); }
    if (ck_p) { Param p = *ck_p; p.name = "k"; cond_params.push_back(p); }
    resolve_conductor_eta_k(cond_params, eta_rgb, k_rgb);

    mat.pb_conductor_eta_rgb[0] = eta_rgb[0]; mat.pb_conductor_eta_rgb[1] = eta_rgb[1]; mat.pb_conductor_eta_rgb[2] = eta_rgb[2];
    mat.pb_conductor_k_rgb[0] = k_rgb[0]; mat.pb_conductor_k_rgb[1] = k_rgb[1]; mat.pb_conductor_k_rgb[2] = k_rgb[2];
    mat.pb_conductor_set = true;

    // Reflectance texture
    std::string ref_type = get_param_type(params, "reflectance");
    if (ref_type == "texture") {
        int tid = resolve_texture(get_texture_ref(params, "reflectance"));
        if (tid >= 0) mat.diffuse_tex = tid;
    } else if (ref_type == "rgb") {
        auto rgb = get_rgb(params, "reflectance", {0, 0, 0});
        mat.Kd = rgb_to_spectrum_reflectance((float)rgb[0], (float)rgb[1], (float)rgb[2]);
    }

    // Base roughness
    bool remap = get_bool(params, "remaproughness", true);
    auto* br_p = get_param(params, "conductor.roughness");
    if (!br_p) br_p = get_param(params, "roughness");
    mat.pb_base_roughness = br_p ? pbrt_roughness_to_ours(safe_float(br_p, 0.f), remap) : 0.0f;

    // Coat
    float coat_eta = (float)get_float(params, "interface.eta", 1.5);
    mat.pb_eta = coat_eta;
    mat.pb_eta_set = true;
    mat.ior = coat_eta;

    float coat_roughness = (float)get_float(params, "interface.roughness", 0.0);
    mat.pb_clearcoat_roughness = pbrt_roughness_to_ours(coat_roughness, remap);
    mat.pb_clearcoat = 1.0f;
    mat.pb_clearcoat_set = true;
}

void MaterialMapper::map_dielectric(Material& mat, const std::vector<Param>& params) {
    mat.pb_brdf = PbBrdf::Dielectric;
    mat.pb_transmission = 1.0f;
    mat.pb_transmission_set = true;
    mat.Kd = Spectrum::zero();
    mat.Ks = Spectrum::constant(1.0f);

    float eta = (float)get_float(params, "eta", 1.5);
    mat.pb_eta = eta;
    mat.pb_eta_set = true;
    mat.ior = eta;

    auto* r_p = get_param(params, "roughness");
    bool remap = get_bool(params, "remaproughness", true);
    if (r_p) {
        mat.pb_roughness = pbrt_roughness_to_ours(safe_float(r_p, 0.f), remap);
        mat.pb_roughness_set = true;
    }

    // Displacement/bump
    std::string disp_type = get_param_type(params, "displacement");
    if (disp_type == "texture") {
        int tid = resolve_texture(get_texture_ref(params, "displacement"));
        if (tid >= 0) mat.bump_tex = tid;
    }
}

void MaterialMapper::map_thin_dielectric(Material& mat, const std::vector<Param>& params) {
    mat.pb_brdf = PbBrdf::Dielectric;
    mat.pb_transmission = 1.0f;
    mat.pb_transmission_set = true;
    mat.pb_thin = true;
    mat.Kd = Spectrum::zero();
    mat.Ks = Spectrum::constant(1.0f);

    float eta = (float)get_float(params, "eta", 1.5);
    mat.pb_eta = eta;
    mat.pb_eta_set = true;
    mat.ior = eta;

    mat.pb_roughness = 0.0f;
    mat.pb_roughness_set = true;
}

void MaterialMapper::map_diffuse_transmission(Material& mat, const std::vector<Param>& params) {
    mat.pb_brdf = PbBrdf::Lambert;
    mat.pb_thin = true;

    // Reflectance
    std::string ref_type = get_param_type(params, "reflectance");
    if (ref_type == "texture") {
        int tid = resolve_texture(get_texture_ref(params, "reflectance"));
        if (tid >= 0) mat.diffuse_tex = tid;
    } else {
        auto rgb = get_rgb(params, "reflectance", {0.25, 0.25, 0.25});
        mat.Kd = rgb_to_spectrum_reflectance((float)rgb[0], (float)rgb[1], (float)rgb[2]);
    }

    // Transmittance
    auto trans_rgb = get_rgb(params, "transmittance", {0.25, 0.25, 0.25});
    float trans_avg = ((float)trans_rgb[0] + (float)trans_rgb[1] + (float)trans_rgb[2]) / 3.f;
    float scale = (float)get_float(params, "scale", 1.0);
    mat.pb_transmission = trans_avg * scale;
    mat.pb_transmission_set = true;

    // Alpha texture
    std::string alpha_type = get_param_type(params, "alpha");
    if (alpha_type == "texture") {
        int tid = resolve_texture(get_texture_ref(params, "alpha"));
        if (tid >= 0) mat.alpha_tex = tid;
    }
}

void MaterialMapper::map_mix(Material& mat, const PbrtMaterial& pbrt_mat) {
    auto& params = pbrt_mat.params;

    // Get sub-material names
    auto* mat_param = get_param(params, "materials");
    float amount = (float)get_float(params, "amount", 0.5);

    std::vector<std::string> mat_names;
    if (mat_param) {
        for (auto& s : mat_param->strings) mat_names.push_back(s);
    }

    std::string chosen_name;
    if (mat_names.size() >= 2)
        chosen_name = (amount > 0.5f) ? mat_names[1] : mat_names[0];
    else if (mat_names.size() == 1)
        chosen_name = mat_names[0];

    if (!chosen_name.empty()) {
        auto it = pbrt_scene_.named_materials.find(chosen_name);
        if (it != pbrt_scene_.named_materials.end()) {
            const auto& sub_mt = it->second.mat_type;
            const auto& sub_params = it->second.params;
            if (sub_mt == "coateddiffuse")        map_coated_diffuse(mat, sub_params);
            else if (sub_mt == "diffuse")         map_diffuse(mat, sub_params);
            else if (sub_mt == "dielectric")      map_dielectric(mat, sub_params);
            else if (sub_mt == "thindielectric")  map_thin_dielectric(mat, sub_params);
            else if (sub_mt == "conductor")       map_conductor(mat, sub_params);
            else if (sub_mt == "coatedconductor") map_coated_conductor(mat, sub_params);
            else if (sub_mt == "measured")        map_measured(mat, sub_params);
            else if (sub_mt == "diffusetransmission") map_diffuse_transmission(mat, sub_params);
            else                                  map_diffuse(mat, sub_params);
            return;
        }
    }
    // Fallback
    mat.pb_brdf = PbBrdf::Lambert;
}

void MaterialMapper::map_measured(Material& mat, const std::vector<Param>& params) {
    // Approximate measured BSDF as clearcoated white diffuse
    mat.pb_brdf = PbBrdf::Clearcoat;
    mat.pb_clearcoat = 1.0f;
    mat.pb_clearcoat_set = true;
    mat.pb_clearcoat_roughness = 0.05f;
    mat.pb_base_brdf = PbBrdf::Lambert;
    mat.pb_eta = 1.5f;
    mat.pb_eta_set = true;
    mat.ior = 1.5f;
    mat.Kd = Spectrum::constant(0.85f);
    mat.Ks = Spectrum::constant(0.15f);
}

// =====================================================================
//  Texture resolution
// =====================================================================

// Walk the PBRT texture graph: imagemap, scale, mix, constant
std::string MaterialMapper::resolve_texture_path(const std::string& tex_name) {
    auto it = pbrt_scene_.textures.find(tex_name);
    if (it == pbrt_scene_.textures.end()) return "";

    auto& td = it->second;
    if (td.tex_class == "imagemap") {
        return get_string(td.params, "filename");
    }
    if (td.tex_class == "scale") {
        // Follow 'tex' child
        std::string child = get_texture_ref(td.params, "tex");
        if (!child.empty()) {
            std::string result = resolve_texture_path(child);
            if (!result.empty()) return result;
        }
        // Fallback: 'scale' might be a texture ref
        std::string scale_ref = get_texture_ref(td.params, "scale");
        if (!scale_ref.empty())
            return resolve_texture_path(scale_ref);
        return "";
    }
    if (td.tex_class == "mix" || td.tex_class == "directionmix") {
        for (const char* key : {"tex1", "tex2"}) {
            std::string child = get_texture_ref(td.params, key);
            if (!child.empty()) {
                std::string result = resolve_texture_path(child);
                if (!result.empty()) return result;
            }
        }
        return "";
    }
    return "";  // constant, etc.
}

int MaterialMapper::resolve_texture(const std::string& tex_name) {
    if (tex_name.empty()) return -1;

    std::string rel_path = resolve_texture_path(tex_name);
    if (rel_path.empty()) return -1;

    // Build absolute path relative to PBRT source dir
    std::string full_path = (fs::path(source_dir_) / rel_path).string();
    full_path = fs::weakly_canonical(full_path).string();

    // Deduplicate
    for (int i = 0; i < (int)scene_.textures.size(); ++i) {
        if (scene_.textures[i].path == full_path)
            return i;
    }

    // Load via stb_image
    int w, h, c;
    unsigned char* img = stbi_load(full_path.c_str(), &w, &h, &c, 4);
    if (!img) {
        std::cerr << "[PBRT] Failed to load texture: " << full_path << "\n";
        return -1;
    }

    Texture tex;
    tex.width = w;
    tex.height = h;
    tex.channels = 4;
    tex.path = full_path;
    tex.data.resize(w * h * 4);
    for (int p = 0; p < w * h * 4; ++p)
        tex.data[p] = img[p] / 255.0f;
    stbi_image_free(img);

    int idx = (int)scene_.textures.size();
    scene_.textures.push_back(std::move(tex));
    std::cout << "[PBRT] Loaded texture: " << rel_path << " (" << w << "x" << h << ")\n";
    return idx;
}

// =====================================================================
//  Roughness conversion
// =====================================================================

float MaterialMapper::pbrt_roughness_to_ours(float roughness, bool remap) {
    if (roughness < 1e-8f) return 0.0f;
    if (remap)
        return std::pow(roughness, 0.25f);    // our_r = roughness^(1/4)
    else
        return std::sqrt(roughness);          // our_r = sqrt(roughness)
}

// =====================================================================
//  Conductor complex IOR
// =====================================================================

void MaterialMapper::resolve_conductor_eta_k(const std::vector<Param>& params,
                                             float eta_rgb[3], float k_rgb[3]) {
    // Default: aluminum
    eta_rgb[0] = 1.34f; eta_rgb[1] = 0.96f; eta_rgb[2] = 0.50f;
    k_rgb[0]   = 7.47f; k_rgb[1]   = 6.40f; k_rgb[2]   = 5.30f;

    auto* eta_p = get_param(params, "eta");
    auto* k_p   = get_param(params, "k");

    // Try named spectrum
    if (eta_p && !eta_p->strings.empty()) {
        std::string spec_name = eta_p->strings[0];
        bool found = false;
        for (auto& preset : CONDUCTOR_PRESETS) {
            std::string pn(preset.name);
            if (spec_name.find(pn) != std::string::npos || pn.find(spec_name) != std::string::npos) {
                eta_rgb[0] = preset.eta[0]; eta_rgb[1] = preset.eta[1]; eta_rgb[2] = preset.eta[2];
                k_rgb[0] = preset.k[0]; k_rgb[1] = preset.k[1]; k_rgb[2] = preset.k[2];
                found = true;
                break;
            }
        }
        if (!found) {
            std::cerr << "[PBRT] Unknown conductor spectrum '" << spec_name
                      << "', using aluminum defaults\n";
        }
    } else if (eta_p && eta_p->floats.size() >= 3) {
        eta_rgb[0] = (float)eta_p->floats[0];
        eta_rgb[1] = (float)eta_p->floats[1];
        eta_rgb[2] = (float)eta_p->floats[2];
    }

    if (k_p && !k_p->strings.empty()) {
        std::string spec_name = k_p->strings[0];
        for (auto& preset : CONDUCTOR_PRESETS) {
            std::string pn(preset.name);
            // k spectra have "-k" suffix
            std::string k_name = pn;
            auto epos = k_name.find("-eta");
            if (epos != std::string::npos)
                k_name.replace(epos, 4, "-k");
            if (spec_name.find(k_name) != std::string::npos || k_name.find(spec_name) != std::string::npos) {
                k_rgb[0] = preset.k[0]; k_rgb[1] = preset.k[1]; k_rgb[2] = preset.k[2];
                break;
            }
        }
    } else if (k_p && k_p->floats.size() >= 3) {
        k_rgb[0] = (float)k_p->floats[0];
        k_rgb[1] = (float)k_p->floats[1];
        k_rgb[2] = (float)k_p->floats[2];
    }
}

} // namespace pbrt
