// ─────────────────────────────────────────────────────────────────────
// obj_loader.cpp – Wavefront OBJ/MTL parser implementation
// ─────────────────────────────────────────────────────────────────────
#include "scene/obj_loader.h"
#include "core/spectrum.h"
#include "core/medium.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <cmath>

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

// ── Helper: load (or reuse) a texture by path ──────────────────────
// Returns the texture index in scene.textures, or -1 on failure.
static int load_or_reuse_texture(Scene& scene, const std::string& full_path,
                                  const std::string& display_name,
                                  int desired_channels = 4) {
    // Deduplicate: if already loaded, return existing index
    for (size_t ti = 0; ti < scene.textures.size(); ++ti) {
        if (scene.textures[ti].path == full_path)
            return (int)ti;
    }
    int w, h, c;
    unsigned char* img = stbi_load(full_path.c_str(), &w, &h, &c, desired_channels);
    if (!img) {
        std::cerr << "[MTL] Failed to load texture: " << full_path << "\n";
        return -1;
    }
    Texture tex;
    tex.width    = w;
    tex.height   = h;
    tex.channels = desired_channels;
    tex.path     = full_path;
    tex.data.resize(w * h * desired_channels);
    for (int p = 0; p < w * h * desired_channels; ++p)
        tex.data[p] = img[p] / 255.0f;
    stbi_image_free(img);
    int idx = (int)scene.textures.size();
    scene.textures.push_back(std::move(tex));
    std::cout << "[MTL] Loaded texture: " << display_name
              << " (" << w << "x" << h << ")\n";
    return idx;
}

// ── Helper: resolve a texture path from the MTL line ────────────────
static std::string resolve_tex_path(std::istringstream& ss,
                                     const std::string& base_dir) {
    std::string tex_path;
    std::getline(ss >> std::ws, tex_path);
    tex_path = trim(tex_path);
    std::replace(tex_path.begin(), tex_path.end(), '\\', '/');
    return base_dir + "/" + tex_path;
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
    // Use an index instead of a raw pointer so reallocation of scene.materials
    // never leaves us with a dangling pointer.
    int current_idx = -1;

    // Store raw illum values so we can apply type assignment AFTER all
    // keywords are parsed (Ks may appear after illum in the file).
    std::unordered_map<int, int> illum_values;

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
            current_idx = (int)scene.materials.size();
            mat_map[name] = (uint32_t)current_idx;
            scene.materials.push_back(mat);
        }
        else if (current_idx < 0) {
            continue;
        }
        else if (keyword == "Kd") {
            float r, g, b;
            ss >> r >> g >> b;
            scene.materials[current_idx].Kd = rgb_to_spectrum_reflectance(r, g, b);
        }
        else if (keyword == "Ks") {
            float r, g, b;
            ss >> r >> g >> b;
            scene.materials[current_idx].Ks = rgb_to_spectrum_reflectance(r, g, b);
        }
        else if (keyword == "Ke") {
            // Emission
            float r, g, b;
            ss >> r >> g >> b;
            if (r > 0.f || g > 0.f || b > 0.f) {
                // Convert RGB emission to spectral using emission basis
                // (preserves absolute intensity, no white-normalisation)
                scene.materials[current_idx].Le = rgb_to_spectrum_emission(r, g, b);
                scene.materials[current_idx].type = MaterialType::Emissive;
            }
        }
        else if (keyword == "Ns") {
            float ns;
            ss >> ns;
            // Convert Phong exponent to roughness
            scene.materials[current_idx].roughness = sqrtf(2.0f / (ns + 2.0f));
        }
        else if (keyword == "Ni") {
            ss >> scene.materials[current_idx].ior;
        }
        else if (keyword == "illum") {
            int illum;
            ss >> illum;
            // Store for deferred processing (Ks may not be parsed yet)
            illum_values[current_idx] = illum;
        }
        else if (keyword == "Tf") {
            float r, g, b;
            ss >> r >> g >> b;
            scene.materials[current_idx].Tf = rgb_to_spectrum_reflectance(r, g, b);
        }
        else if (keyword == "d") {
            float d_val;
            ss >> d_val;
            scene.materials[current_idx].opacity = d_val;
        }
        else if (keyword == "Tr") {
            float tr_val;
            ss >> tr_val;
            scene.materials[current_idx].opacity = 1.0f - tr_val;
        }
        else if (keyword == "Ka") {
            // Ambient: irrelevant for GI renderer — parse and skip.
            float r, g, b;
            ss >> r >> g >> b;
            (void)r; (void)g; (void)b;
        }
        else if (keyword == "map_Kd") {
            std::string full_path = resolve_tex_path(ss, base_dir);
            int ti = load_or_reuse_texture(scene, full_path, full_path);
            if (ti >= 0) scene.materials[current_idx].diffuse_tex = ti;
        }
        else if (keyword == "map_Ks") {
            std::string full_path = resolve_tex_path(ss, base_dir);
            int ti = load_or_reuse_texture(scene, full_path, full_path);
            if (ti >= 0) scene.materials[current_idx].specular_tex = ti;
        }
        else if (keyword == "map_d") {
            std::string full_path = resolve_tex_path(ss, base_dir);
            // Alpha mask — load as RGBA (R channel used as alpha)
            int ti = load_or_reuse_texture(scene, full_path, full_path, 4);
            if (ti >= 0) scene.materials[current_idx].alpha_tex = ti;
        }
        else if (keyword == "map_Ke") {
            std::string full_path = resolve_tex_path(ss, base_dir);
            int ti = load_or_reuse_texture(scene, full_path, full_path);
            if (ti >= 0) {
                scene.materials[current_idx].emission_tex = ti;
                // Ensure material is flagged as emissive even if Ke is (0,0,0)
                if (scene.materials[current_idx].Le.max_component() == 0.f) {
                    scene.materials[current_idx].Le = Spectrum::constant(1.0f);
                    scene.materials[current_idx].type = MaterialType::Emissive;
                }
            }
        }
        else if (keyword == "map_bump" || keyword == "bump") {
            std::string full_path = resolve_tex_path(ss, base_dir);
            int ti = load_or_reuse_texture(scene, full_path, full_path);
            if (ti >= 0) scene.materials[current_idx].bump_tex = ti;
        }
        else if (keyword == "map_Ka") {
            // Ambient texture: parse path but don't use in rendering.
            // Kept for future AO-like darkening if desired.
            resolve_tex_path(ss, base_dir);
        }
        // ── Photon-Beam material extensions (pb_*) ──────────────────
        else if (keyword == "pb_brdf") {
            std::string tag;
            ss >> tag;
            auto& m = scene.materials[current_idx];
            if      (tag == "lambert")    m.pb_brdf = PbBrdf::Lambert;
            else if (tag == "dielectric") m.pb_brdf = PbBrdf::Dielectric;
            else if (tag == "conductor")  m.pb_brdf = PbBrdf::Conductor;
            else if (tag == "clearcoat")  m.pb_brdf = PbBrdf::Clearcoat;
            else if (tag == "emissive")   m.pb_brdf = PbBrdf::Emissive;
            else if (tag == "fabric")     m.pb_brdf = PbBrdf::Fabric;
        }
        else if (keyword == "pb_semantic") {
            std::string tag;
            ss >> tag;
            auto& m = scene.materials[current_idx];
            if      (tag == "subsurface")    m.pb_semantic = PbSemantic::Subsurface;
            else if (tag == "glass")         m.pb_semantic = PbSemantic::Glass;
            else if (tag == "metal")         m.pb_semantic = PbSemantic::Metal;
            else if (tag == "fabric")        m.pb_semantic = PbSemantic::Fabric;
            else if (tag == "leather")       m.pb_semantic = PbSemantic::Leather;
            else if (tag == "wood_natural")  m.pb_semantic = PbSemantic::WoodNatural;
            else if (tag == "wood_painted")  m.pb_semantic = PbSemantic::WoodPainted;
            else if (tag == "wallpaper")     m.pb_semantic = PbSemantic::Wallpaper;
            else if (tag == "stone")         m.pb_semantic = PbSemantic::Stone;
            else if (tag == "plastic")       m.pb_semantic = PbSemantic::Plastic;
        }
        else if (keyword == "pb_roughness") {
            auto& m = scene.materials[current_idx];
            ss >> m.pb_roughness;
            m.pb_roughness = fmaxf(0.001f, fminf(1.0f, m.pb_roughness));
            m.pb_roughness_set = true;
        }
        else if (keyword == "pb_anisotropy") {
            auto& m = scene.materials[current_idx];
            ss >> m.pb_anisotropy;
            m.pb_anisotropy = fmaxf(0.f, fminf(1.0f, m.pb_anisotropy));
            m.pb_anisotropy_set = true;
        }
        else if (keyword == "pb_roughness_x") {
            auto& m = scene.materials[current_idx];
            ss >> m.pb_roughness_x;
            m.pb_roughness_x = fmaxf(0.001f, fminf(1.0f, m.pb_roughness_x));
            m.pb_roughness_xy_set = true;
        }
        else if (keyword == "pb_roughness_y") {
            auto& m = scene.materials[current_idx];
            ss >> m.pb_roughness_y;
            m.pb_roughness_y = fmaxf(0.001f, fminf(1.0f, m.pb_roughness_y));
            m.pb_roughness_xy_set = true;
        }
        else if (keyword == "pb_eta") {
            auto& m = scene.materials[current_idx];
            ss >> m.pb_eta;
            m.pb_eta_set = true;
        }
        else if (keyword == "pb_conductor_eta") {
            auto& m = scene.materials[current_idx];
            ss >> m.pb_conductor_eta_rgb[0] >> m.pb_conductor_eta_rgb[1] >> m.pb_conductor_eta_rgb[2];
            m.pb_conductor_set = true;
        }
        else if (keyword == "pb_conductor_k") {
            auto& m = scene.materials[current_idx];
            ss >> m.pb_conductor_k_rgb[0] >> m.pb_conductor_k_rgb[1] >> m.pb_conductor_k_rgb[2];
            m.pb_conductor_set = true;
        }
        else if (keyword == "pb_transmission") {
            auto& m = scene.materials[current_idx];
            ss >> m.pb_transmission;
            m.pb_transmission = fmaxf(0.f, fminf(1.0f, m.pb_transmission));
            m.pb_transmission_set = true;
        }
        else if (keyword == "pb_thin") {
            int v; ss >> v;
            scene.materials[current_idx].pb_thin = (v != 0);
        }
        else if (keyword == "pb_thickness") {
            ss >> scene.materials[current_idx].pb_thickness;
        }
        else if (keyword == "pb_clearcoat") {
            auto& m = scene.materials[current_idx];
            ss >> m.pb_clearcoat;
            m.pb_clearcoat = fmaxf(0.f, fminf(1.0f, m.pb_clearcoat));
            m.pb_clearcoat_set = true;
        }
        else if (keyword == "pb_clearcoat_roughness") {
            auto& m = scene.materials[current_idx];
            ss >> m.pb_clearcoat_roughness;
            m.pb_clearcoat_roughness = fmaxf(0.001f, fminf(1.0f, m.pb_clearcoat_roughness));
            m.pb_clearcoat_set = true;
        }
        else if (keyword == "pb_base_brdf") {
            std::string tag;
            ss >> tag;
            auto& m = scene.materials[current_idx];
            if      (tag == "lambert")    m.pb_base_brdf = PbBrdf::Lambert;
            else if (tag == "dielectric") m.pb_base_brdf = PbBrdf::Dielectric;
            else if (tag == "conductor")  m.pb_base_brdf = PbBrdf::Conductor;
        }
        else if (keyword == "pb_base_roughness") {
            auto& m = scene.materials[current_idx];
            ss >> m.pb_base_roughness;
            m.pb_base_roughness = fmaxf(0.001f, fminf(1.0f, m.pb_base_roughness));
        }
        else if (keyword == "pb_sheen") {
            auto& m = scene.materials[current_idx];
            ss >> m.pb_sheen;
            m.pb_sheen = fmaxf(0.f, fminf(1.0f, m.pb_sheen));
            m.pb_sheen_set = true;
        }
        else if (keyword == "pb_sheen_tint") {
            auto& m = scene.materials[current_idx];
            ss >> m.pb_sheen_tint;
            m.pb_sheen_tint = fmaxf(0.f, fminf(1.0f, m.pb_sheen_tint));
        }
        else if (keyword == "pb_medium") {
            std::string tag;
            ss >> tag;
            if (tag == "homogeneous")
                scene.materials[current_idx].pb_medium_enabled = true;
        }
        else if (keyword == "pb_density") {
            ss >> scene.materials[current_idx].pb_density;
        }
        else if (keyword == "pb_sigma_a") {
            auto& m = scene.materials[current_idx];
            ss >> m.pb_sigma_a_rgb[0] >> m.pb_sigma_a_rgb[1] >> m.pb_sigma_a_rgb[2];
            m.pb_sigma_a_set = true;
        }
        else if (keyword == "pb_sigma_s") {
            auto& m = scene.materials[current_idx];
            ss >> m.pb_sigma_s_rgb[0] >> m.pb_sigma_s_rgb[1] >> m.pb_sigma_s_rgb[2];
            m.pb_sigma_s_set = true;
        }
        else if (keyword == "pb_g") {
            auto& m = scene.materials[current_idx];
            ss >> m.pb_g;
            m.pb_g = fmaxf(-1.0f, fminf(1.0f, m.pb_g));
        }
        else if (keyword == "pb_tf_spectrum") {
            auto& m = scene.materials[current_idx];
            for (int b = 0; b < NUM_LAMBDA; ++b)
                ss >> m.pb_tf_spectrum.value[b];
            m.pb_tf_spectrum_set = true;
        }
        else if (keyword == "pb_dispersion") {
            auto& m = scene.materials[current_idx];
            ss >> m.pb_dispersion_B;
            if (m.pb_dispersion_B >= 0.f)
                m.pb_dispersion_set = true;
        }
        else if (keyword == "pb_meters_per_unit") {
            ss >> scene.materials[current_idx].pb_meters_per_unit;
        }
        // Ignore any unknown pb_* key (backward compatible)
    }

    // ── Post-processing: apply illum → MaterialType now that all
    //    keywords (Kd, Ks, Ns, Ni, Ke, …) are fully parsed. ──────────
    for (auto& [idx, illum] : illum_values) {
        auto& mat = scene.materials[idx];
        // Don't let illum override an already-set Emissive type
        if (mat.type == MaterialType::Emissive) continue;
        switch (illum) {
            case 0: case 1:
                mat.type = MaterialType::Lambertian;
                break;
            case 2:
                // OBJ spec: illum 2 = Blinn-Phong diffuse + specular
                // Always dielectric Fresnel (F0 from IOR).  Ks is the
                // specular highlight colour/intensity, NOT metallic F0.
                if (mat.Ks.max_component() > 0.01f)
                    mat.type = MaterialType::GlossyDielectric;
                break;
            case 3:
                mat.type = MaterialType::Mirror;
                break;
            case 4: case 6: case 7:
                mat.type = MaterialType::Glass;
                break;
            case 5:
                mat.type = MaterialType::Mirror;
                break;
            default:
                break;
        }
        // Glass/Translucent must stay opaque to the anyhit test.
        // 'd' (dissolve) is for stochastic alpha cutouts — not for 
        // refractive materials whose transmission is handled in the
        // specular bounce shader.
        if (mat.type == MaterialType::Glass ||
            mat.type == MaterialType::Translucent) {
            mat.opacity = 1.0f;
        }
    }

    std::cout << "[MTL] Loaded " << scene.materials.size() << " materials\n";
    return true;
}

// ── Apply pb_* overrides to all materials ───────────────────────────
// Called once after all MTL files are loaded.  Resolves pb_brdf → 
// MaterialType, applies pb_roughness/pb_eta overrides, creates
// HomogeneousMedium entries for pb_medium, etc.
static void finalize_pb_materials(Scene& scene) {
    int pb_count = 0;
    int media_created = 0;

    for (auto& mat : scene.materials) {
        // ── Skip materials that have no pb_* extensions ─────────────
        bool has_any_pb = (mat.pb_brdf != PbBrdf::None)
                       || mat.pb_roughness_set
                       || mat.pb_eta_set
                       || mat.pb_conductor_set
                       || mat.pb_transmission_set
                       || mat.pb_clearcoat_set
                       || mat.pb_sheen_set
                       || mat.pb_medium_enabled
                       || mat.pb_dispersion_set
                       || mat.pb_tf_spectrum_set;
        if (!has_any_pb) continue;
        ++pb_count;

        // ── 1. Roughness override ───────────────────────────────────
        // pb_roughness is GGX alpha directly
        if (mat.pb_roughness_set) {
            mat.roughness = mat.pb_roughness;
        }

        // ── 2. IOR override ─────────────────────────────────────────
        if (mat.pb_eta_set) {
            mat.ior = mat.pb_eta;
        }

        // ── 2b. Chromatic dispersion from pb_dispersion ────────────
        // pb_dispersion <B> enables Cauchy dispersion with that B.
        // A is derived so that n(589nm) == mat.ior (sodium D-line anchor).
        if (mat.pb_dispersion_set) {
            constexpr float lambda_d = 589.0f; // sodium D-line, nm
            mat.cauchy_B   = mat.pb_dispersion_B;
            mat.cauchy_A   = mat.ior - mat.cauchy_B / (lambda_d * lambda_d);
            mat.dispersion = true;
        }

        // ── 3. Conductor complex IOR → spectral ────────────────────
        if (mat.pb_conductor_set) {
            mat.pb_conductor_eta_spec = rgb_to_spectrum_reflectance(
                mat.pb_conductor_eta_rgb[0],
                mat.pb_conductor_eta_rgb[1],
                mat.pb_conductor_eta_rgb[2]);
            mat.pb_conductor_k_spec = rgb_to_spectrum_reflectance(
                mat.pb_conductor_k_rgb[0],
                mat.pb_conductor_k_rgb[1],
                mat.pb_conductor_k_rgb[2]);
        }

        // ── 3b. Direct spectral transmittance override ──────────────
        // pb_tf_spectrum bypasses the RGB→spectrum conversion for Tf,
        // allowing spectrally-narrow colour filters (e.g. deep green glass).
        if (mat.pb_tf_spectrum_set) {
            mat.Tf = mat.pb_tf_spectrum;
        }

        // ── 4. Transmission override ────────────────────────────────
        if (mat.pb_transmission_set) {
            mat.opacity = 1.0f - mat.pb_transmission;
        }

        // ── 5. Map pb_brdf → MaterialType ───────────────────────────
        // pb_brdf takes precedence over illum-derived type.
        switch (mat.pb_brdf) {
            case PbBrdf::Lambert:
                mat.type = MaterialType::Lambertian;
                break;

            case PbBrdf::Emissive:
                mat.type = MaterialType::Emissive;
                break;

            case PbBrdf::Dielectric: {
                // Glass-like vs plastic-like heuristic
                bool is_glass = mat.pb_transmission_set
                                ? (mat.pb_transmission > 0.5f)
                                : (mat.Kd.max_component() < 0.05f);
                if (is_glass) {
                    // If medium is also needed → Translucent, else Glass
                    if (mat.pb_medium_enabled && !mat.pb_thin)
                        mat.type = MaterialType::Translucent;
                    else
                        mat.type = MaterialType::Glass;
                } else {
                    mat.type = MaterialType::GlossyDielectric;
                }
                break;
            }

            case PbBrdf::Conductor:
                mat.type = MaterialType::GlossyMetal;
                // If complex IOR is provided, store F0 approximation in Ks
                // for the existing GlossyMetal BSDF path.
                if (mat.pb_conductor_set) {
                    // Normal-incidence reflectance from complex IOR:
                    // F0 = ((n-1)² + k²) / ((n+1)² + k²)
                    for (int b = 0; b < NUM_LAMBDA; ++b) {
                        float n = mat.pb_conductor_eta_spec.value[b];
                        float k = mat.pb_conductor_k_spec.value[b];
                        float num = (n - 1.f) * (n - 1.f) + k * k;
                        float den = (n + 1.f) * (n + 1.f) + k * k;
                        mat.Ks.value[b] = (den > 0.f) ? num / den : 0.5f;
                    }
                    // Conductors have no diffuse
                    mat.Kd = Spectrum::zero();
                }
                break;

            case PbBrdf::Clearcoat:
                mat.type = MaterialType::Clearcoat;
                // Resolve clearcoat roughness default
                if (mat.pb_clearcoat_roughness < 0.f)
                    mat.pb_clearcoat_roughness = mat.roughness;
                // Resolve base roughness default
                if (mat.pb_base_roughness < 0.f)
                    mat.pb_base_roughness = 1.0f; // diffuse base
                break;

            case PbBrdf::Fabric:
                mat.type = MaterialType::Fabric;
                // Ensure sheen has a value
                if (!mat.pb_sheen_set)
                    mat.pb_sheen = 0.5f;
                break;

            case PbBrdf::None:
                // No explicit pb_brdf — leave type from illum processing
                break;
        }

        // ── 6. Create HomogeneousMedium from pb_medium ──────────────
        if (mat.pb_medium_enabled && !mat.pb_thin) {
            float scale = mat.pb_density / mat.pb_meters_per_unit;

            HomogeneousMedium med;
            Spectrum sa = rgb_to_spectrum_reflectance(
                mat.pb_sigma_a_rgb[0], mat.pb_sigma_a_rgb[1], mat.pb_sigma_a_rgb[2]);
            Spectrum ss_coeff = rgb_to_spectrum_reflectance(
                mat.pb_sigma_s_rgb[0], mat.pb_sigma_s_rgb[1], mat.pb_sigma_s_rgb[2]);

            for (int b = 0; b < NUM_LAMBDA; ++b) {
                med.sigma_a.value[b] = sa.value[b] * scale;
                med.sigma_s.value[b] = ss_coeff.value[b] * scale;
                med.sigma_t.value[b] = med.sigma_a.value[b] + med.sigma_s.value[b];
            }

            med.g = mat.pb_g;

            mat.medium_id = (int)scene.media.size();
            scene.media.push_back(med);
            ++media_created;

            // Ensure material type supports media
            if (mat.type != MaterialType::Translucent
                && mat.type != MaterialType::Glass) {
                mat.type = MaterialType::Translucent;
            }
        }

        // ── 7. Anisotropic roughness: store as base roughness ──────
        // The current BSDF uses isotropic GGX; we store the geometric
        // mean of pb_roughness_x/y so the average surface look is
        // preserved.  When anisotropic GGX is added later, the full
        // x/y values are already in the Material.
        if (mat.pb_roughness_xy_set) {
            float rx = (mat.pb_roughness_x > 0.f) ? mat.pb_roughness_x : mat.roughness;
            float ry = (mat.pb_roughness_y > 0.f) ? mat.pb_roughness_y : mat.roughness;
            mat.roughness = sqrtf(rx * ry);
        }

        // ── 8. Glass/Translucent must be opaque to the anyhit test ──
        // Their transparency is handled by the specular bounce shader
        // (Fresnel refraction), NOT by stochastic alpha.  Without this,
        // pb_transmission 1.0 sets opacity=0.0, which makes the anyhit
        // programs ignore every intersection — effectively invisible.
        if (mat.type == MaterialType::Glass ||
            mat.type == MaterialType::Translucent) {
            mat.opacity = 1.0f;
        }
    }

    if (pb_count > 0) {
        std::cout << "[MTL] Applied pb_* extensions to " << pb_count
                  << " material(s)";
        if (media_created > 0)
            std::cout << ", created " << media_created << " medium(s)";
        std::cout << "\n";
    }
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
    std::unordered_set<std::string> loaded_mtls;   // track already-loaded MTL paths
    uint32_t current_material = 0;
    bool has_vertex_colors = false;

    // Smooth normal accumulation: triangles that had no OBJ normals
    struct SmoothTriRef { uint32_t tri_idx; int vi0, vi1, vi2; };
    std::vector<SmoothTriRef> smooth_tris;

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

                // Normals: use provided or defer to smooth-normal post-pass
                bool has_normals = (face_verts[0].vn >= 0);
                if (has_normals) {
                    tri.n0 = get_normal(face_verts[0]);
                    tri.n1 = get_normal(face_verts[i-1]);
                    tri.n2 = get_normal(face_verts[i]);
                } else {
                    // Placeholder – will be filled by area-weighted post-pass below
                    float3 gn = tri.geometric_normal();
                    tri.n0 = tri.n1 = tri.n2 = gn;
                    smooth_tris.push_back({
                        (uint32_t)scene.triangles.size(),
                        face_verts[0].v,
                        face_verts[i-1].v,
                        face_verts[i].v
                    });
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
            // Deduplicate: many OBJ files repeat mtllib once per group/object.
            std::string canonical = fs::weakly_canonical(mtl_path).string();
            if (loaded_mtls.count(canonical) == 0) {
                loaded_mtls.insert(canonical);
                load_mtl(mtl_path, scene, mat_map, base_dir);
            }
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

    // ── Area-weighted smooth normals post-pass ───────────────────────
    // For triangles that had no per-vertex normals in the OBJ file,
    // accumulate area-weighted geometric normals per vertex position and
    // normalise, giving smooth shading across the mesh.
    if (!smooth_tris.empty()) {
        std::vector<float3> accum(positions.size(), make_f3(0.f, 0.f, 0.f));

        for (const auto& ref : smooth_tris) {
            const Triangle& tri = scene.triangles[ref.tri_idx];
            float3 gn = tri.geometric_normal();
            float  w  = tri.area();
            if (ref.vi0 >= 0 && ref.vi0 < (int)accum.size())
                accum[ref.vi0] = accum[ref.vi0] + gn * w;
            if (ref.vi1 >= 0 && ref.vi1 < (int)accum.size())
                accum[ref.vi1] = accum[ref.vi1] + gn * w;
            if (ref.vi2 >= 0 && ref.vi2 < (int)accum.size())
                accum[ref.vi2] = accum[ref.vi2] + gn * w;
        }

        // Normalise the per-vertex sums
        for (auto& n : accum) {
            float l = length(n);
            n = (l > 1e-8f) ? normalize(n) : make_f3(0.f, 0.f, 1.f);
        }

        // Write smooth normals back into the triangles
        for (const auto& ref : smooth_tris) {
            Triangle& tri = scene.triangles[ref.tri_idx];
            if (ref.vi0 >= 0 && ref.vi0 < (int)accum.size()) tri.n0 = accum[ref.vi0];
            if (ref.vi1 >= 0 && ref.vi1 < (int)accum.size()) tri.n1 = accum[ref.vi1];
            if (ref.vi2 >= 0 && ref.vi2 < (int)accum.size()) tri.n2 = accum[ref.vi2];
        }

        std::cout << "[OBJ] Computed area-weighted smooth normals for "
                  << smooth_tris.size() << " triangles\n";
    }

    if (has_vertex_colors) {
        std::cout << "[OBJ] Per-vertex colors detected (radiosity data)\n";
    }

    // ── Apply pb_* material extensions ──────────────────────────────
    finalize_pb_materials(scene);

    return true;
}
