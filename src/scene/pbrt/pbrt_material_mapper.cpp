// ─────────────────────────────────────────────────────────────────────
// pbrt_material_mapper.cpp – Map PBRT v4 materials → renderer Materials
// ─────────────────────────────────────────────────────────────────────
#include "scene/pbrt/pbrt_material_mapper.h"
#include "core/color.h"

#include "stb_image.h"
#include "tinyexr.h"

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

// ── Spectrum-aware eta extraction ───────────────────────────────────
// PBRT "spectrum eta" stores wavelength/value pairs: [λ0 v0 λ1 v1 ...].
// A plain "float eta" stores a single scalar.
// get_float() returns floats[0] which is the wavelength for spectrum
// params — wrong.  This helper averages the value entries instead.
static float get_spectrum_eta(const std::vector<Param>& params,
                              const std::string& name, float def) {
    auto* p = get_param(params, name);
    if (!p) return def;
    if (p->type == "spectrum" && p->floats.size() >= 2) {
        // Wavelength/value pairs: average the values (odd indices)
        double sum = 0;
        int count = 0;
        for (size_t i = 1; i < p->floats.size(); i += 2) {
            sum += p->floats[i];
            ++count;
        }
        return count > 0 ? (float)(sum / count) : def;
    }
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
    def.Kd = Color3::constant(0.5f);
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
        mat.Le = Color3::from_rgb(
            rgb[0] * light_scale, rgb[1] * light_scale, rgb[2] * light_scale);
    }

    if (!is_blackbody) {
        if (L_param && L_param->floats.size() >= 3) {
            float r = (float)L_param->floats[0] * light_scale;
            float g = (float)L_param->floats[1] * light_scale;
            float b = (float)L_param->floats[2] * light_scale;
            mat.Le = Color3::from_rgb(r, g, b);
        } else {
            mat.Le = Color3::from_rgb(light_scale, light_scale, light_scale);
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
    else if (mt == "subsurface")      map_subsurface(mat, params);
    else if (mt == "mix")             map_mix(mat, pbrt_mat);
    else if (mt == "hair")            map_hair(mat, params);
    else if (mt == "interface")       map_interface(mat, params);
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
        mat.Kd = Color3::constant(1.0f);  // texture IS the reflectance; Kd=1 so dev_get_Kd() passes it through
    } else {
        auto rgb = get_rgb(params, "reflectance", {0.5, 0.5, 0.5});
        mat.Kd = Color3::from_rgb((float)rgb[0], (float)rgb[1], (float)rgb[2]);
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
        mat.Kd = Color3::constant(1.0f);  // texture IS the reflectance; Kd=1 so dev_get_Kd() passes it through
    } else {
        auto rgb = get_rgb(params, "reflectance", {0.5, 0.5, 0.5});
        mat.Kd = Color3::from_rgb((float)rgb[0], (float)rgb[1], (float)rgb[2]);
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
    float eta = get_spectrum_eta(params, "eta", 1.5f);
    mat.pb_eta = eta;
    mat.pb_eta_set = true;
    mat.ior = eta;

    // Displacement/bump
    std::string disp_type = get_param_type(params, "displacement");
    if (disp_type == "texture") {
        std::string tex_name = get_texture_ref(params, "displacement");
        int tid = resolve_texture(tex_name, /*is_color=*/false);
        if (tid >= 0) {
            mat.bump_tex = tid;
            mat.displacement_tex = tid;
        }
    }

    // Normal map
    std::string nmap_type = get_param_type(params, "normalmap");
    if (nmap_type == "texture") {
        int tid = resolve_texture(get_texture_ref(params, "normalmap"), /*is_color=*/false);
        if (tid >= 0) mat.normal_tex = tid;
    }

    // Pearl-like translucency: very smooth coat over bright diffuse base
    // → add synthetic scattering medium for sub-surface translucency.
    float coat_rough = mat.pb_clearcoat_roughness;
    auto ref_rgb = get_rgb(params, "reflectance", {0.5, 0.5, 0.5});
    float lum = 0.2126f * (float)ref_rgb[0] + 0.7152f * (float)ref_rgb[1]
              + 0.0722f * (float)ref_rgb[2];
    if (coat_rough >= 0.f && coat_rough < 0.01f && lum > 0.4f) {
        mat.pb_brdf = PbBrdf::Dielectric;
        mat.pb_transmission = 1.0f;
        mat.pb_transmission_set = true;
        mat.pb_medium_enabled = true;
        mat.pb_sigma_s_rgb[0] = 20.f; mat.pb_sigma_s_rgb[1] = 20.f; mat.pb_sigma_s_rgb[2] = 20.f;
        mat.pb_sigma_s_set = true;
        mat.pb_sigma_a_rgb[0] = 0.5f; mat.pb_sigma_a_rgb[1] = 0.3f; mat.pb_sigma_a_rgb[2] = 0.3f;
        mat.pb_sigma_a_set = true;
        mat.pb_g = 0.8f;
        mat.Kd = Color3::from_rgb((float)ref_rgb[0], (float)ref_rgb[1], (float)ref_rgb[2]);
    }
}

void MaterialMapper::map_conductor(Material& mat, const std::vector<Param>& params) {
    mat.pb_brdf = PbBrdf::Conductor;
    mat.Kd = Color3::zero();

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
    // GPU Clearcoat BSDF base is always Lambert — conductor base would be
    // black (Kd=0).  Map to GlossyMetal instead: sacrifices the thin coat
    // highlight (~4% at normal incidence for IOR 1.5) but gives correct
    // metallic appearance with angular colour shift.
    mat.pb_brdf = PbBrdf::Conductor;
    mat.Kd = Color3::zero();

    // Conductor base
    float eta_rgb[3], k_rgb[3];
    auto* ce_p = get_param(params, "conductor.eta");
    auto* ck_p = get_param(params, "conductor.k");
    if (!ce_p) ce_p = get_param(params, "eta");
    if (!ck_p) ck_p = get_param(params, "k");

    std::vector<Param> cond_params;
    if (ce_p) { Param p = *ce_p; p.name = "eta"; cond_params.push_back(p); }
    if (ck_p) { Param p = *ck_p; p.name = "k"; cond_params.push_back(p); }
    resolve_conductor_eta_k(cond_params, eta_rgb, k_rgb);

    mat.pb_conductor_eta_rgb[0] = eta_rgb[0]; mat.pb_conductor_eta_rgb[1] = eta_rgb[1]; mat.pb_conductor_eta_rgb[2] = eta_rgb[2];
    mat.pb_conductor_k_rgb[0] = k_rgb[0]; mat.pb_conductor_k_rgb[1] = k_rgb[1]; mat.pb_conductor_k_rgb[2] = k_rgb[2];
    mat.pb_conductor_set = true;

    // Reflectance texture (modulates Kd for subtle tinting)
    std::string ref_type = get_param_type(params, "reflectance");
    if (ref_type == "texture") {
        int tid = resolve_texture(get_texture_ref(params, "reflectance"));
        if (tid >= 0) mat.diffuse_tex = tid;
    } else if (ref_type == "rgb") {
        auto rgb = get_rgb(params, "reflectance", {0, 0, 0});
        mat.Kd = Color3::from_rgb((float)rgb[0], (float)rgb[1], (float)rgb[2]);
    }

    // Use conductor roughness (not interface roughness) for GlossyMetal
    bool remap = get_bool(params, "remaproughness", true);
    auto* br_p = get_param(params, "conductor.roughness");
    if (!br_p) br_p = get_param(params, "roughness");
    mat.pb_roughness = br_p ? pbrt_roughness_to_ours(safe_float(br_p, 0.f), remap) : 0.0f;
    mat.pb_roughness_set = true;
}

void MaterialMapper::map_dielectric(Material& mat, const std::vector<Param>& params) {
    float eta = get_spectrum_eta(params, "eta", 1.5f);

    // PBRT scenes sometimes encode reciprocal IOR (eta < 1) for
    // interior surfaces like air pockets in ice.  With our IOR stack
    // tracking, the absolute interior IOR should be used instead.
    // No real optical medium has n < 1; normalise to air (1.0).
    if (eta < 1.0f) {
        std::cout << "[PBRT] " << mat.name << ": eta=" << eta
                  << " < 1 -> normalised to 1.0 (air)\n";
        eta = 1.0f;
    }

    mat.pb_eta = eta;
    mat.pb_eta_set = true;
    mat.ior = eta;

    auto* r_p  = get_param(params, "roughness");
    auto* ur_p = get_param(params, "uroughness");
    auto* vr_p = get_param(params, "vroughness");
    bool remap = get_bool(params, "remaproughness", true);
    float our_roughness = 0.f;
    float pbrt_alpha    = 0.f;   // GGX alpha in PBRT convention
    if (ur_p || vr_p) {
        float raw_u = ur_p ? safe_float(ur_p, 0.f) : (r_p ? safe_float(r_p, 0.f) : 0.f);
        float raw_v = vr_p ? safe_float(vr_p, 0.f) : (r_p ? safe_float(r_p, 0.f) : 0.f);
        float raw_avg = std::sqrt(raw_u * raw_v);
        our_roughness = pbrt_roughness_to_ours(raw_avg, remap);
        pbrt_alpha = remap ? std::sqrt(raw_avg) : raw_avg;
        mat.pb_roughness = our_roughness;
        mat.pb_roughness_set = true;
    } else if (r_p) {
        float raw = safe_float(r_p, 0.f);
        our_roughness = pbrt_roughness_to_ours(raw, remap);
        pbrt_alpha = remap ? std::sqrt(raw) : raw;
        mat.pb_roughness = our_roughness;
        mat.pb_roughness_set = true;
    }

    // Rough dielectrics: no microfacet refraction model exists, so map to
    // GlossyDielectric (opaque Fresnel specular + no diffuse body).
    // Smooth dielectrics: map to Glass (delta BSDF with Fresnel refraction).
    // Only truly near-specular surfaces (alpha <= 0.001) become transparent
    // Glass; anything rougher is intentionally textured and should stay opaque
    // (e.g. TV screens, frosted panels).
    if (pbrt_alpha > 0.001f) {
        mat.pb_brdf = PbBrdf::Dielectric;
        mat.pb_transmission = 0.f;
        mat.pb_transmission_set = true;
        mat.Kd = Color3::zero();
        mat.Ks = Color3::constant(1.0f);
    } else {
        mat.pb_brdf = PbBrdf::Dielectric;
        mat.pb_transmission = 1.0f;
        mat.pb_transmission_set = true;
        mat.Kd = Color3::zero();
        mat.Ks = Color3::constant(1.0f);
    }

    // Displacement/bump
    std::string disp_type = get_param_type(params, "displacement");
    if (disp_type == "texture") {
        int tid = resolve_texture(get_texture_ref(params, "displacement"), /*is_color=*/false);
        if (tid >= 0) {
            mat.bump_tex = tid;
            mat.displacement_tex = tid;
        }
    }

    // Normal map
    std::string nmap_type2 = get_param_type(params, "normalmap");
    if (nmap_type2 == "texture") {
        int tid = resolve_texture(get_texture_ref(params, "normalmap"), /*is_color=*/false);
        if (tid >= 0) mat.normal_tex = tid;
    }
}

void MaterialMapper::map_thin_dielectric(Material& mat, const std::vector<Param>& params) {
    mat.pb_brdf = PbBrdf::Dielectric;
    mat.pb_transmission = 1.0f;
    mat.pb_transmission_set = true;
    mat.pb_thin = true;
    mat.Kd = Color3::zero();
    mat.Ks = Color3::constant(1.0f);

    float eta = get_spectrum_eta(params, "eta", 1.5f);
    mat.pb_eta = eta;
    mat.pb_eta_set = true;
    mat.ior = eta;

    mat.pb_roughness = 0.0f;
    mat.pb_roughness_set = true;
}

uint32_t MaterialMapper::get_or_create_thin_variant(uint32_t base_mat_id) {
    auto it = thin_variant_cache_.find(base_mat_id);
    if (it != thin_variant_cache_.end()) return it->second;

    // Copy the base material and flip to thin
    Material thin = scene_.materials[base_mat_id];
    thin.name += "__thin_auto";
    thin.pb_thin = true;

    uint32_t idx = (uint32_t)scene_.materials.size();
    scene_.materials.push_back(thin);
    thin_variant_cache_[base_mat_id] = idx;
    return idx;
}

void MaterialMapper::map_diffuse_transmission(Material& mat, const std::vector<Param>& params) {
    mat.type = MaterialType::DiffuseTransmission;  // Two-sided Lambert: Kd/π reflect + Tf/π transmit

    // Reflectance (used as diffuse texture for Kd / Tf modulation)
    std::string ref_type = get_param_type(params, "reflectance");
    if (ref_type == "texture") {
        int tid = resolve_texture(get_texture_ref(params, "reflectance"));
        if (tid >= 0) mat.diffuse_tex = tid;
    } else {
        auto rgb = get_rgb(params, "reflectance", {0.25, 0.25, 0.25});
        mat.Kd = Color3::from_rgb((float)rgb[0], (float)rgb[1], (float)rgb[2]);
    }

    // Transmittance → store as Tf spectrum so dev_get_Tf() returns colorful filter
    auto trans_rgb = get_rgb(params, "transmittance", {0.25, 0.25, 0.25});
    float scale = (float)get_float(params, "scale", 1.0);
    mat.Tf = Color3::from_rgb(
        (float)trans_rgb[0] * scale,
        (float)trans_rgb[1] * scale,
        (float)trans_rgb[2] * scale);
    float trans_avg = ((float)trans_rgb[0] + (float)trans_rgb[1] + (float)trans_rgb[2]) / 3.f;
    mat.pb_transmission = trans_avg * scale;
    mat.pb_transmission_set = true;

    // Alpha texture
    std::string alpha_type = get_param_type(params, "alpha");
    if (alpha_type == "texture") {
        int tid = resolve_texture(get_texture_ref(params, "alpha"), /*is_color=*/false);
        if (tid >= 0) mat.alpha_tex = tid;
    }

    // IOR for thin dielectric Fresnel (typical plant leaf ~1.5)
    mat.ior = 1.5f;
}

void MaterialMapper::map_subsurface(Material& mat, const std::vector<Param>& params) {
    // PBRT "subsurface" — approximate as Lambert diffuse.
    // Renderer has no true SSS, so use reflectance or named preset colour.
    mat.pb_brdf = PbBrdf::Lambert;
    mat.pb_semantic = PbSemantic::Subsurface;

    // Try explicit reflectance first
    std::string ref_type = get_param_type(params, "reflectance");
    if (ref_type == "texture") {
        int tid = resolve_texture(get_texture_ref(params, "reflectance"));
        if (tid >= 0) mat.diffuse_tex = tid;
        mat.Kd = Color3::constant(1.0f);  // texture IS the reflectance
    } else {
        auto rgb = get_rgb(params, "reflectance", {-1, -1, -1});
        if (rgb[0] >= 0) {
            mat.Kd = Color3::from_rgb((float)rgb[0], (float)rgb[1], (float)rgb[2]);
        } else {
            // Fallback: skin-like colour for named presets
            mat.Kd = Color3::from_rgb(0.8f, 0.6f, 0.5f);
        }
    }

    // Eta (index of refraction) — store but not critical for Lambert
    float eta = (float)get_float(params, "eta", 1.5);
    mat.pb_eta = eta;
    mat.pb_eta_set = true;
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
            else if (sub_mt == "subsurface")      map_subsurface(mat, sub_params);
            else if (sub_mt == "hair")            map_hair(mat, sub_params);
            else if (sub_mt == "interface")       map_interface(mat, sub_params);
            else                                  map_diffuse(mat, sub_params);
            return;
        }
    }
    // Fallback
    mat.pb_brdf = PbBrdf::Lambert;
}

void MaterialMapper::map_measured(Material& mat, const std::vector<Param>& /*params*/) {
    // Approximate measured BSDF as clearcoated white diffuse
    mat.pb_brdf = PbBrdf::Clearcoat;
    mat.pb_clearcoat = 1.0f;
    mat.pb_clearcoat_set = true;
    mat.pb_clearcoat_roughness = 0.05f;
    mat.pb_base_brdf = PbBrdf::Lambert;
    mat.pb_eta = 1.5f;
    mat.pb_eta_set = true;
    mat.ior = 1.5f;
    mat.Kd = Color3::constant(0.85f);
    mat.Ks = Color3::constant(0.15f);
}

void MaterialMapper::map_hair(Material& mat, const std::vector<Param>& params) {
    // Map PBRT hair BSDF to diffuse approximation.
    // Hair scattering is too complex for a standard BSDF; approximate as
    // tinted diffuse matching the hair colour.
    mat.pb_brdf = PbBrdf::Lambert;
    mat.pb_roughness = 0.8f;
    mat.pb_roughness_set = true;

    // Try reflectance first (explicit color)
    auto rgb = get_rgb(params, "reflectance", {-1, -1, -1});
    if (rgb[0] >= 0) {
        mat.Kd = Color3::from_rgb((float)rgb[0], (float)rgb[1], (float)rgb[2]);
        return;
    }

    // sigma_a (absorption coefficient) → darker = more absorption
    auto sa = get_rgb(params, "sigma_a", {-1, -1, -1});
    if (sa[0] >= 0) {
        // Approximate: reflectance ≈ exp(-sigma_a) for unit-diameter hair
        float r = std::exp(-(float)sa[0]);
        float g = std::exp(-(float)sa[1]);
        float b = std::exp(-(float)sa[2]);
        mat.Kd = Color3::from_rgb(r, g, b);
        return;
    }

    // Melanin model: eumelanin + pheomelanin
    float eumelanin  = (float)get_float(params, "eumelanin", 1.3f);
    float pheomelanin = (float)get_float(params, "pheomelanin", 0.0f);
    // Approximate melanin → color (simplified Donner & Jensen model)
    float sa_r = eumelanin * 0.419f + pheomelanin * 0.187f;
    float sa_g = eumelanin * 0.697f + pheomelanin * 0.4f;
    float sa_b = eumelanin * 1.37f  + pheomelanin * 1.05f;
    mat.Kd = Color3::from_rgb(
        std::exp(-sa_r), std::exp(-sa_g), std::exp(-sa_b));
}

void MaterialMapper::map_interface(Material& mat, const std::vector<Param>& /*params*/) {
    // Interface material: purely transparent boundary for media transitions.
    // No surface scattering — acts as an invisible medium boundary.
    mat.pb_brdf = PbBrdf::Dielectric;
    mat.pb_transmission = 1.0f;
    mat.pb_transmission_set = true;
    mat.pb_roughness = 0.0f;
    mat.pb_roughness_set = true;
    mat.pb_eta = 1.0f;  // no refraction at interface
    mat.pb_eta_set = true;
    mat.ior = 1.0f;
    mat.Kd = Color3::zero();
    mat.Ks = Color3::zero();
    mat.opacity = 0.0f;
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

// Bake a procedural texture to RGBA float pixels.
// Returns true if baking succeeded, false if texture type unknown.
static bool bake_procedural_texture(const PbrtTextureDecl& td,
                                    const PbrtScene& /*pbrt_scene*/,
                                    int w, int h,
                                    std::vector<float>& pixels) {
    pixels.resize(w * h * 4);

    if (td.tex_class == "constant") {
        float val = (float)get_float(td.params, "value", 0.5);
        auto rgb = get_rgb(td.params, "value", {val, val, val});
        float r = (rgb.size() >= 3) ? (float)rgb[0] : val;
        float g = (rgb.size() >= 3) ? (float)rgb[1] : val;
        float b = (rgb.size() >= 3) ? (float)rgb[2] : val;
        for (int i = 0; i < w * h; ++i) {
            pixels[i*4+0] = r; pixels[i*4+1] = g;
            pixels[i*4+2] = b; pixels[i*4+3] = 1.f;
        }
        return true;
    }

    if (td.tex_class == "checkerboard") {
        auto tex1_rgb = get_rgb(td.params, "tex1", {1, 1, 1});
        auto tex2_rgb = get_rgb(td.params, "tex2", {0, 0, 0});
        float r1 = (float)tex1_rgb[0], g1 = (float)tex1_rgb[1], b1 = (float)tex1_rgb[2];
        float r2 = (float)tex2_rgb[0], g2 = (float)tex2_rgb[1], b2 = (float)tex2_rgb[2];
        // If tex1/tex2 are scalar values
        float v1 = (float)get_float(td.params, "tex1", -1.0);
        float v2 = (float)get_float(td.params, "tex2", -1.0);
        if (tex1_rgb.empty() && v1 >= 0) { r1 = g1 = b1 = v1; }
        if (tex2_rgb.empty() && v2 >= 0) { r2 = g2 = b2 = v2; }

        int freq = std::max(1, get_int(td.params, "uscale", 2));
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int cu = (x * freq / w) % 2;
                int cv = (y * freq / h) % 2;
                bool check = (cu ^ cv) != 0;
                int idx = (y * w + x) * 4;
                pixels[idx+0] = check ? r1 : r2;
                pixels[idx+1] = check ? g1 : g2;
                pixels[idx+2] = check ? b1 : b2;
                pixels[idx+3] = 1.f;
            }
        }
        return true;
    }

    if (td.tex_class == "bilerp") {
        auto v00 = get_rgb(td.params, "v00", {0, 0, 0});
        auto v01 = get_rgb(td.params, "v01", {1, 0, 0});
        auto v10 = get_rgb(td.params, "v10", {0, 1, 0});
        auto v11 = get_rgb(td.params, "v11", {1, 1, 1});
        for (int y = 0; y < h; ++y) {
            float t = (float)y / std::max(h - 1, 1);
            for (int x = 0; x < w; ++x) {
                float s = (float)x / std::max(w - 1, 1);
                int idx = (y * w + x) * 4;
                for (int c = 0; c < 3; ++c) {
                    float top    = (float)v00[c] * (1.f - s) + (float)v10[c] * s;
                    float bottom = (float)v01[c] * (1.f - s) + (float)v11[c] * s;
                    pixels[idx+c] = top * (1.f - t) + bottom * t;
                }
                pixels[idx+3] = 1.f;
            }
        }
        return true;
    }

    if (td.tex_class == "dots") {
        auto inside_rgb  = get_rgb(td.params, "inside", {1, 1, 1});
        auto outside_rgb = get_rgb(td.params, "outside", {0, 0, 0});
        for (int y = 0; y < h; ++y) {
            float v = (float)y / h;
            for (int x = 0; x < w; ++x) {
                float u = (float)x / w;
                // Simple polka dot pattern (4×4 grid)
                float cu = fmodf(u * 4.f, 1.f) - 0.5f;
                float cv = fmodf(v * 4.f, 1.f) - 0.5f;
                bool inside = (cu*cu + cv*cv) < 0.15f;
                int idx = (y * w + x) * 4;
                pixels[idx+0] = inside ? (float)inside_rgb[0] : (float)outside_rgb[0];
                pixels[idx+1] = inside ? (float)inside_rgb[1] : (float)outside_rgb[1];
                pixels[idx+2] = inside ? (float)inside_rgb[2] : (float)outside_rgb[2];
                pixels[idx+3] = 1.f;
            }
        }
        return true;
    }

    // fbm, wrinkled, windy, marble → noise-based grayscale
    if (td.tex_class == "fbm" || td.tex_class == "wrinkled" ||
        td.tex_class == "windy" || td.tex_class == "marble") {
        // Simple value-noise approximation (no full Perlin implementation)
        int octaves = get_int(td.params, "octaves", 8);
        (void)octaves;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                // Pseudo-noise based on position hash
                float u = (float)x / w * 8.f;
                float v = (float)y / h * 8.f;
                int ix = (int)floorf(u), iy = (int)floorf(v);
                float fx = u - ix, fy = v - iy;
                // Simple hash-based noise
                auto hash = [](int x, int y) -> float {
                    int n = x * 73856093 ^ y * 19349663;
                    n = (n << 13) ^ n;
                    return (float)((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff)
                           / (float)0x7fffffff;
                };
                float n00 = hash(ix, iy), n10 = hash(ix+1, iy);
                float n01 = hash(ix, iy+1), n11 = hash(ix+1, iy+1);
                float n0 = n00 * (1.f-fx) + n10 * fx;
                float n1 = n01 * (1.f-fx) + n11 * fx;
                float val = n0 * (1.f-fy) + n1 * fy;
                if (td.tex_class == "marble")
                    val = 0.5f + 0.5f * sinf(6.f * u + 5.f * val);
                int idx = (y * w + x) * 4;
                pixels[idx+0] = pixels[idx+1] = pixels[idx+2] = val;
                pixels[idx+3] = 1.f;
            }
        }
        return true;
    }

    return false;
}

int MaterialMapper::resolve_texture(const std::string& tex_name, bool is_color) {
    if (tex_name.empty()) return -1;

    std::string rel_path = resolve_texture_path(tex_name);
    if (rel_path.empty()) {
        // Try procedural baking
        auto it = pbrt_scene_.textures.find(tex_name);
        if (it != pbrt_scene_.textures.end()) {
            int bake_res = 256;
            std::vector<float> pixels;
            if (bake_procedural_texture(it->second, pbrt_scene_, bake_res, bake_res, pixels)) {
                Texture tex;
                tex.width = bake_res;
                tex.height = bake_res;
                tex.channels = 4;
                tex.path = "__procedural_" + tex_name;
                tex.data = std::move(pixels);
                int idx = (int)scene_.textures.size();
                scene_.textures.push_back(std::move(tex));
                std::cout << "[PBRT] Baked procedural texture: " << tex_name
                          << " (" << it->second.tex_class << ", " << bake_res << "x" << bake_res << ")\n";
                return idx;
            }
        }
        return -1;
    }

    // Build absolute path relative to PBRT source dir
    std::string full_path = (fs::path(source_dir_) / rel_path).string();
    full_path = fs::weakly_canonical(full_path).string();

    // Deduplicate
    for (int i = 0; i < (int)scene_.textures.size(); ++i) {
        if (scene_.textures[i].path == full_path)
            return i;
    }

    // Load image data (EXR via tinyexr, everything else via stb_image)
    int w = 0, h = 0;
    std::vector<float> pixels;  // RGBA float

    std::string ext = fs::path(full_path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".exr") {
        float* rgba = nullptr;
        const char* err = nullptr;
        int ret = LoadEXR(&rgba, &w, &h, full_path.c_str(), &err);
        if (ret != TINYEXR_SUCCESS || !rgba) {
            std::cerr << "[PBRT] Failed to load EXR texture: " << full_path;
            if (err) { std::cerr << " (" << err << ")"; FreeEXRErrorMessage(err); }
            std::cerr << "\n";
            return -1;
        }
        pixels.assign(rgba, rgba + w * h * 4);
        free(rgba);
    } else {
        int c;
        unsigned char* img = stbi_load(full_path.c_str(), &w, &h, &c, 4);
        if (!img) {
            std::cerr << "[PBRT] Failed to load texture: " << full_path << "\n";
            return -1;
        }
        pixels.resize(w * h * 4);
        for (int p = 0; p < w * h * 4; ++p) {
            float v = img[p] / 255.0f;
            if (is_color && (p % 4) != 3)
                v = srgb_to_linear(v);
            pixels[p] = v;
        }
        stbi_image_free(img);
    }

    Texture tex;
    tex.width = w;
    tex.height = h;
    tex.channels = 4;
    tex.path = full_path;
    tex.data = std::move(pixels);

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
