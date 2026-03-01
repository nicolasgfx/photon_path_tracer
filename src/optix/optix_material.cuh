#pragma once

// optix_material.cuh – Material type checks, texture sampling, accessors,
//                       cosine hemisphere/cone sampling, spectrum→sRGB

// MaterialType enum is shared from core/types.h (included via launch_params.h).

// == Material helpers (device-side) ===================================

__forceinline__ __device__
bool dev_is_emissive(uint32_t mat_id) {
    return params.mat_type[mat_id] == Emissive;
}

__forceinline__ __device__
bool dev_is_specular(uint32_t mat_id) {
    uint8_t t = params.mat_type[mat_id];
    return t == Mirror || t == Glass;
}

__forceinline__ __device__
bool dev_is_glass(uint32_t mat_id) {
    return params.mat_type[mat_id] == Glass;
}

// Translucent: dielectric surface + interior participating medium.
// NOT a pure delta specular; eligible for NEE and photon gather,
// but still uses Fresnel reflect/refract at the boundary.
__forceinline__ __device__
bool dev_is_translucent(uint32_t mat_id) {
    return params.mat_type[mat_id] == Translucent;
}

__forceinline__ __device__
bool dev_is_mirror(uint32_t mat_id) {
    return params.mat_type[mat_id] == Mirror;
}

__forceinline__ __device__
float dev_get_ior(uint32_t mat_id) {
    return params.ior[mat_id];
}

// Return the interior medium_id for a material (-1 = no medium).
__forceinline__ __device__
int dev_get_medium_id(uint32_t mat_id) {
    return (params.mat_medium_id) ? params.mat_medium_id[mat_id] : -1;
}

// Return true if the material has an interior participating medium.
__forceinline__ __device__
bool dev_has_medium(uint32_t mat_id) {
    return dev_get_medium_id(mat_id) >= 0;
}

// Load a HomogeneousMedium from the device media table.
__forceinline__ __device__
HomogeneousMedium dev_get_medium(int medium_id) {
    return params.media[medium_id];
}

// Sample any texture by texture ID at the given UV.
// Returns linear RGB (0-1).  Falls back to (1,1,1) when no texture.
__forceinline__ __device__
float3 dev_sample_tex_by_id(int tex_id, float2 uv) {
    if (tex_id < 0 || tex_id >= params.num_textures || params.tex_atlas == nullptr)
        return make_f3(1.f, 1.f, 1.f);

    GpuTexDesc desc = params.tex_descs[tex_id];
    float u = uv.x - floorf(uv.x);
    float v = uv.y - floorf(uv.y);
    v = 1.f - v;
    int ix = __float2int_rd(u * (float)desc.width)  % desc.width;
    int iy = __float2int_rd(v * (float)desc.height) % desc.height;
    if (ix < 0) ix += desc.width;
    if (iy < 0) iy += desc.height;
    int pixel = iy * desc.width + ix;
    int base  = desc.offset + pixel * 4;
    return make_f3(params.tex_atlas[base + 0],
                   params.tex_atlas[base + 1],
                   params.tex_atlas[base + 2]);
}

// Sample the flat texture atlas at the given UV for material mat_id.
// Returns linear RGB (0-1).  Falls back to (0,0,0) when no texture.
__forceinline__ __device__
float3 dev_sample_diffuse_tex(uint32_t mat_id, float2 uv) {
    int tex_id = params.diffuse_tex[mat_id];
    if (tex_id < 0 || tex_id >= params.num_textures || params.tex_atlas == nullptr)
        return make_f3(0.f, 0.f, 0.f);

    GpuTexDesc desc = params.tex_descs[tex_id];
    // Wrap UVs to [0,1)
    float u = uv.x - floorf(uv.x);
    float v = uv.y - floorf(uv.y);
    // Flip V (OBJ convention: V=0 at bottom)
    v = 1.f - v;
    int ix = __float2int_rd(u * (float)desc.width)  % desc.width;
    int iy = __float2int_rd(v * (float)desc.height) % desc.height;
    if (ix < 0) ix += desc.width;
    if (iy < 0) iy += desc.height;
    int pixel = iy * desc.width + ix;
    int base  = desc.offset + pixel * 4;
    return make_f3(params.tex_atlas[base + 0],
                   params.tex_atlas[base + 1],
                   params.tex_atlas[base + 2]);
}

__forceinline__ __device__
Spectrum dev_get_Kd(uint32_t mat_id, float2 uv) {
    // If the material has a diffuse texture, sample it and convert to spectrum
    if (params.diffuse_tex != nullptr && params.diffuse_tex[mat_id] >= 0) {
        float3 rgb = dev_sample_diffuse_tex(mat_id, uv);
        return rgb_to_spectrum_reflectance(rgb.x, rgb.y, rgb.z);
    }
    // Fallback: pre-converted spectral Kd
    Spectrum s;
    for (int i = 0; i < NUM_LAMBDA; ++i)
        s.value[i] = params.Kd[mat_id * NUM_LAMBDA + i];
    return s;
}

// Glass transmittance filter: base Tf (stored in Kd buffer) × diffuse texture.
// When a glass material has a diffuse texture the texture colour modulates the
// flat transmittance so that every texel can have its own spectral filter.
__forceinline__ __device__
Spectrum dev_get_Tf(uint32_t mat_id, float2 uv) {
    // Base Tf lives in the Kd GPU buffer (uploaded from mat.Tf for glass)
    Spectrum Tf;
    for (int i = 0; i < NUM_LAMBDA; ++i)
        Tf.value[i] = params.Kd[mat_id * NUM_LAMBDA + i];

    // Modulate by diffuse texture when present
    if (params.diffuse_tex != nullptr && params.diffuse_tex[mat_id] >= 0) {
        float3 rgb = dev_sample_diffuse_tex(mat_id, uv);
        Spectrum tex = rgb_to_spectrum_reflectance(rgb.x, rgb.y, rgb.z);
        for (int i = 0; i < NUM_LAMBDA; ++i)
            Tf.value[i] *= tex.value[i];
    }
    return Tf;
}

__forceinline__ __device__
Spectrum dev_get_Le(uint32_t mat_id, float2 uv = make_float2(0.f, 0.f)) {
    Spectrum s;
    for (int i = 0; i < NUM_LAMBDA; ++i)
        s.value[i] = params.Le[mat_id * NUM_LAMBDA + i];
    // Modulate by emission texture (map_Ke) when present.
    // The texture acts as a spatial mask: bright texels emit, dark texels don't.
    if (params.emission_tex != nullptr && params.emission_tex[mat_id] >= 0) {
        float3 emi_rgb = dev_sample_tex_by_id(params.emission_tex[mat_id], uv);
        float emi_lum = 0.2126f * emi_rgb.x + 0.7152f * emi_rgb.y + 0.0722f * emi_rgb.z;
        for (int i = 0; i < NUM_LAMBDA; ++i)
            s.value[i] *= emi_lum;
    }
    return s;
}

// == Cosine-weighted hemisphere sampling ==============================
__forceinline__ __device__
float3 sample_cosine_hemisphere_dev(PCGRng& rng) {
    float u1 = rng.next_float();
    float u2 = rng.next_float();
    float r = sqrtf(u1);
    float phi = 2.0f * PI * u2;
    return make_f3(r * cosf(phi), r * sinf(phi), sqrtf(fmaxf(0.f, 1.f - u1)));
}

// Cosine-weighted cone sampling (device version)
// Samples within a cone of half-angle defined by cos_theta_max.
__forceinline__ __device__
float3 sample_cosine_cone_dev(PCGRng& rng, float cos_theta_max) {
    float u1 = rng.next_float();
    float u2 = rng.next_float();
    float cos2_max = cos_theta_max * cos_theta_max;
    float cos2_theta = 1.0f - u1 * (1.0f - cos2_max);
    float cos_theta  = sqrtf(fmaxf(0.f, cos2_theta));
    float sin_theta  = sqrtf(fmaxf(0.f, 1.0f - cos2_theta));
    float phi = 2.0f * PI * u2;
    return make_f3(sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta);
}

// ONB is now shared HD in core/types.h — DevONB struct deleted.

// == Spectrum -> sRGB (device-side) ===================================
// Uses shared HD spectrum_to_xyz() from core/spectrum.h.
__forceinline__ __device__
float3 dev_spectrum_to_srgb(const Spectrum& s) {
    float3 xyz = spectrum_to_xyz(s);
    float X = xyz.x, Y = xyz.y, Z = xyz.z;

    // Apply exposure (runtime — set via render_config.json and R key)
    X *= params.exposure; Y *= params.exposure; Z *= params.exposure;

    float r =  3.2406f*X - 1.5372f*Y - 0.4986f*Z;
    float g = -0.9689f*X + 1.8758f*Y + 0.0415f*Z;
    float b =  0.0557f*X - 0.2040f*Y + 1.0570f*Z;

    // Tone mapping (§14 guideline): ACES Filmic or clamp-only
    if (USE_ACES_TONEMAPPING) {
        // Narkowicz 2015 fitted ACES curve: maps [0,∞) → [0,1)
        auto aces = [](float x) -> float {
            x = fmaxf(x, 0.f);
            return (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f);
        };
        r = aces(r);
        g = aces(g);
        b = aces(b);
    } else {
        r = fmaxf(r, 0.f);
        g = fmaxf(g, 0.f);
        b = fmaxf(b, 0.f);
    }

    auto gamma = [](float c) -> float {
        c = fmaxf(c, 0.f);
        return (c <= 0.0031308f) ? 12.92f*c : 1.055f*powf(c, 1.f/2.4f) - 0.055f;
    };

    return make_f3(gamma(r), gamma(g), gamma(b));
}

// == Material data accessors (GPU) ====================================

__forceinline__ __device__
Spectrum dev_get_Ks(uint32_t mat_id) {
    Spectrum s;
    for (int i = 0; i < NUM_LAMBDA; ++i)
        s.value[i] = params.Ks[mat_id * NUM_LAMBDA + i];
    return s;
}

__forceinline__ __device__
float dev_get_roughness(uint32_t mat_id) {
    return params.roughness[mat_id];
}

__forceinline__ __device__
float dev_get_clearcoat_weight(uint32_t mat_id) {
    return params.clearcoat_weight ? params.clearcoat_weight[mat_id] : 1.0f;
}

__forceinline__ __device__
float dev_get_clearcoat_roughness(uint32_t mat_id) {
    return params.clearcoat_roughness ? params.clearcoat_roughness[mat_id] : 0.03f;
}

__forceinline__ __device__
float dev_get_sheen(uint32_t mat_id) {
    return params.sheen ? params.sheen[mat_id] : 0.0f;
}

__forceinline__ __device__
float dev_get_sheen_tint(uint32_t mat_id) {
    return params.sheen_tint ? params.sheen_tint[mat_id] : 0.0f;
}

__forceinline__ __device__
bool dev_is_glossy(uint32_t mat_id) {
    return params.mat_type[mat_id] == GlossyMetal;
}

__forceinline__ __device__
bool dev_is_dielectric_glossy(uint32_t mat_id) {
    return params.mat_type[mat_id] == GlossyDielectric;
}

__forceinline__ __device__
bool dev_is_clearcoat(uint32_t mat_id) {
    return params.mat_type[mat_id] == Clearcoat;
}

__forceinline__ __device__
bool dev_is_fabric(uint32_t mat_id) {
    return params.mat_type[mat_id] == Fabric;
}

// Returns true for any glossy surface (metallic, dielectric, or clearcoat).
// Use this for glossy continuation gates; use dev_is_glossy() /
// dev_is_dielectric_glossy() only when differentiating Fresnel model.
__forceinline__ __device__
bool dev_is_any_glossy(uint32_t mat_id) {
    uint8_t t = params.mat_type[mat_id];
    return t == GlossyMetal || t == GlossyDielectric || t == Clearcoat;
}
