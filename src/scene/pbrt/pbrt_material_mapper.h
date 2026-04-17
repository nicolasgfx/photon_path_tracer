#pragma once
// ─────────────────────────────────────────────────────────────────────
// pbrt_material_mapper.h – Map PBRT v4 materials to renderer Materials
// ─────────────────────────────────────────────────────────────────────
#include "scene/pbrt/pbrt_parser.h"
#include "scene/material.h"
#include "scene/scene.h"
#include <string>
#include <unordered_map>

namespace pbrt {

// Resolve PBRT textures and materials into renderer Material structs.
// Returns a map of material_name → material_index in scene.materials[].
class MaterialMapper {
public:
    MaterialMapper(const PbrtScene& pbrt_scene, Scene& scene,
                   const std::string& pbrt_source_dir);

    // Map all named materials.  Call before processing shapes.
    void map_all_named_materials();

    // Get or create a material index for a shape.
    // Uses material_name (NamedMaterial) or inline_mat.
    uint32_t resolve_shape_material(const PbrtShape& shape);

    // Return a thin-dielectric variant of the given material.
    // Creates and caches a copy with pb_thin=true if one doesn't exist yet.
    uint32_t get_or_create_thin_variant(uint32_t base_mat_id);

    // Create an emissive material from AreaLightSource params.
    // Returns a new or existing material index.
    uint32_t create_emissive_material(const std::string& base_name,
                                      const std::vector<Param>& area_light_params);

private:
    const PbrtScene& pbrt_scene_;
    Scene& scene_;
    std::string source_dir_;

    // material_name → index in scene.materials
    std::unordered_map<std::string, uint32_t> mat_index_;

    // base_mat_id → thin-variant mat_id (for open-mesh auto-promotion)
    std::unordered_map<uint32_t, uint32_t> thin_variant_cache_;

    // inline material dedup
    int inline_counter_ = 0;

    uint32_t map_one_material(const std::string& name, const PbrtMaterial& pbrt_mat);

    void map_diffuse(Material& mat, const std::vector<Param>& params);
    void map_coated_diffuse(Material& mat, const std::vector<Param>& params);
    void map_conductor(Material& mat, const std::vector<Param>& params);
    void map_coated_conductor(Material& mat, const std::vector<Param>& params);
    void map_dielectric(Material& mat, const std::vector<Param>& params);
    void map_thin_dielectric(Material& mat, const std::vector<Param>& params);
    void map_diffuse_transmission(Material& mat, const std::vector<Param>& params);
    void map_mix(Material& mat, const PbrtMaterial& pbrt_mat);
    void map_measured(Material& mat, const std::vector<Param>& params);
    void map_subsurface(Material& mat, const std::vector<Param>& params);
    void map_hair(Material& mat, const std::vector<Param>& params);
    void map_interface(Material& mat, const std::vector<Param>& params);

    // Texture resolution
    int resolve_texture(const std::string& tex_name, bool is_color = true);
    std::string resolve_texture_path(const std::string& tex_name);

    // Roughness conversion: PBRT roughness → renderer roughness
    static float pbrt_roughness_to_ours(float roughness, bool remap = true);

    // Conductor spectra
    static void resolve_conductor_eta_k(const std::vector<Param>& params,
                                        float eta_rgb[3], float k_rgb[3]);
};

} // namespace pbrt
