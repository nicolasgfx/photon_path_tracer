// ---------------------------------------------------------------------
// optix_device.cu - All OptiX device programs (single compilation unit)
// ---------------------------------------------------------------------
//
// Programs:
//   __raygen__render                  - debug first-hit OR full path tracing
//   __raygen__photon_trace            - GPU photon emission + tracing
//   __raygen__targeted_photon_trace   - GPU targeted caustic emission (Jensen §9.2)
//   __closesthit__radiance            - closest-hit for radiance rays
//   __closesthit__shadow    - closest-hit for shadow rays
//   __miss__radiance        - miss for radiance rays
//   __miss__shadow          - miss for shadow rays
//
// Payload layout (14 values):
//   p0-p2  : hit position (float3)
//   p3-p5  : shading normal (float3)
//   p6     : hit distance t (float)
//   p7     : material ID (uint32_t)
//   p8     : triangle ID (uint32_t)
//   p9     : hit flag (0 = miss, 1 = hit)
//   p10-p12: geometric normal (float3)
//   p13    : (reserved)
// ---------------------------------------------------------------------
#include <optix.h>
#include <cstdint>
#include "optix/launch_params.h"
#include "core/random.h"
#include "core/spectrum.h"
#include "core/config.h"
#include "core/cdf.h"
#include "core/nee_sampling.h"
#include "core/medium.h"
#include "core/phase_function.h"

// ---------------------------------------------------------------------
// Module-local implementation constants
// (kept out of core/config.h to avoid clutter)
// ---------------------------------------------------------------------
namespace {
    // Ray epsilon to avoid self-intersections.
    constexpr float OPTIX_SCENE_EPSILON = 1e-4f;

    // Large tmax avoids clipping long rays in normalized scenes.
    constexpr float DEFAULT_RAY_TMAX = 1e20f;

    // Spatial hashing primes (Teschner et al.). CPU hash grid uses the same values.
    constexpr uint32_t HASHGRID_PRIME_1 = 73856093u;
    constexpr uint32_t HASHGRID_PRIME_2 = 19349663u;
    constexpr uint32_t HASHGRID_PRIME_3 = 83492791u;
}

// ── Precomputed CIE 1931 colour-matching LUTs (Wyman et al. 2013) ───
// Evaluated at bin centres: λ_i = 380 + (i + 0.5) × 12.5 nm, i = 0..31.
__device__ const float CIE_XBAR[NUM_LAMBDA] = {
    0.00085280f, 0.00948721f, 0.05744076f, 0.18924914f,
    0.33911000f, 0.34859709f, 0.27100800f, 0.15654349f,
    0.05396928f, 0.00357322f, 0.02105945f, 0.09939486f,
    0.23283069f, 0.41356647f, 0.62452582f, 0.83222873f,
    0.99022381f, 1.05553252f, 0.98618610f, 0.78291581f,
    0.52804653f, 0.30257481f, 0.14729811f, 0.06092078f,
    0.02140610f, 0.00639019f, 0.00162067f, 0.00034920f,
    0.00006392f, 0.00000994f, 0.00000131f, 0.00000015f,
};
__device__ const float CIE_YBAR[NUM_LAMBDA] = {
    0.00042785f, 0.00116282f, 0.00294402f, 0.00694358f,
    0.01525596f, 0.03122631f, 0.05956898f, 0.10636988f,
    0.18173774f, 0.31073917f, 0.52575729f, 0.77789306f,
    0.92738762f, 0.99189733f, 0.98786222f, 0.92528632f,
    0.80659971f, 0.65076114f, 0.48388622f, 0.33010563f,
    0.20584746f, 0.11703822f, 0.06058265f, 0.02852724f,
    0.01221501f, 0.00475528f, 0.00168297f, 0.00054148f,
    0.00015838f, 0.00004211f, 0.00001018f, 0.00000224f,
};
__device__ const float CIE_ZBAR[NUM_LAMBDA] = {
    0.01360229f, 0.05277060f, 0.23976472f, 0.92140354f,
    1.67860453f, 1.78376062f, 1.64162010f, 1.10660676f,
    0.57342498f, 0.28956506f, 0.14509460f, 0.06643900f,
    0.02705018f, 0.00976209f, 0.00312229f, 0.00088504f,
    0.00022233f, 0.00004950f, 0.00000977f, 0.00000171f,
    0.00000026f, 0.00000004f, 0.00000000f, 0.00000000f,
    0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
    0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
};
__device__ const float CIE_YBAR_SUM_INV = 1.0f / 8.55521603f;

extern "C" {
    __constant__ LaunchParams params;
}

// == Float / uint reinterpret helpers =================================
__forceinline__ __device__ unsigned int f2u(float f) {
    return __float_as_uint(f);
}
__forceinline__ __device__ float u2f(unsigned int u) {
    return __uint_as_float(u);
}

// == Trace result struct ==============================================
struct TraceResult {
    float3   position;
    float3   shading_normal;
    float3   geo_normal;
    float2   uv;
    float    t;
    uint32_t material_id;
    uint32_t triangle_id;
    bool     hit;
};

// == Trace radiance ray and unpack payload ============================
__forceinline__ __device__
TraceResult trace_radiance(float3 origin, float3 direction,
                           float tmin = OPTIX_SCENE_EPSILON,
                           float tmax = DEFAULT_RAY_TMAX) {
    unsigned int p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14;
    p0=p1=p2=p3=p4=p5=p6=p7=p8=p9=p10=p11=p12=p13=p14=0;

    optixTrace(
        params.traversable,
        origin, direction,
        tmin, tmax, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,
        p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14);

    TraceResult r;
    r.position       = make_f3(u2f(p0), u2f(p1), u2f(p2));
    r.shading_normal = make_f3(u2f(p3), u2f(p4), u2f(p5));
    r.t              = u2f(p6);
    r.material_id    = p7;
    r.triangle_id    = p8;
    r.hit            = (p9 != 0);
    r.geo_normal     = make_f3(u2f(p10), u2f(p11), u2f(p12));
    r.uv             = make_f2(u2f(p13), u2f(p14));
    return r;
}

// == Trace shadow ray =================================================
__forceinline__ __device__
bool trace_shadow(float3 origin, float3 direction, float max_dist) {
    // Start occluded=1; closesthit is disabled so payload stays 1 on hit.
    // Only __miss__shadow sets it to 0 (visible).
    unsigned int occluded = 1;
    optixTrace(
        params.traversable,
        origin, direction,
        OPTIX_SCENE_EPSILON, max_dist - OPTIX_SCENE_EPSILON, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
        OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        1, 1, 1,
        occluded);
    return (occluded == 0); // 0 = visible
}

// == Material helpers (device-side) ===================================
enum DevMaterialType : uint8_t {
    DEV_LAMBERTIAN = 0,
    DEV_MIRROR     = 1,
    DEV_GLASS      = 2,
    DEV_GLOSSY     = 3,
    DEV_EMISSIVE   = 4,
    DEV_GLOSSY_DIELECTRIC = 5,
    DEV_TRANSLUCENT = 6,
    DEV_CLEARCOAT   = 7,
    DEV_FABRIC      = 8
};

__forceinline__ __device__
bool dev_is_emissive(uint32_t mat_id) {
    return params.mat_type[mat_id] == DEV_EMISSIVE;
}

__forceinline__ __device__
bool dev_is_specular(uint32_t mat_id) {
    uint8_t t = params.mat_type[mat_id];
    return t == DEV_MIRROR || t == DEV_GLASS;
}

__forceinline__ __device__
bool dev_is_glass(uint32_t mat_id) {
    return params.mat_type[mat_id] == DEV_GLASS;
}

// Translucent: dielectric surface + interior participating medium.
// NOT a pure delta specular; eligible for NEE and photon gather,
// but still uses Fresnel reflect/refract at the boundary.
__forceinline__ __device__
bool dev_is_translucent(uint32_t mat_id) {
    return params.mat_type[mat_id] == DEV_TRANSLUCENT;
}

__forceinline__ __device__
bool dev_is_mirror(uint32_t mat_id) {
    return params.mat_type[mat_id] == DEV_MIRROR;
}

__forceinline__ __device__
float dev_get_ior(uint32_t mat_id) {
    return params.ior[mat_id];
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

// == Device-side ONB ==================================================
struct DevONB {
    float3 u, v, w;

    __forceinline__ __device__
    static DevONB from_normal(float3 n) {
        DevONB onb;
        onb.w = n;
        float3 a = (fabsf(n.x) > 0.9f) ? make_f3(0,1,0) : make_f3(1,0,0);
        onb.v = normalize(cross(n, a));
        onb.u = cross(onb.w, onb.v);
        return onb;
    }

    __forceinline__ __device__
    float3 local_to_world(float3 d) const {
        return u * d.x + v * d.y + w * d.z;
    }

    __forceinline__ __device__
    float3 world_to_local(float3 d) const {
        return make_f3(dot(d, u), dot(d, v), dot(d, w));
    }
};

// == Spectrum -> sRGB (device-side) ===================================
// Normalised by the integral of ybar so that a flat-1.0 spectrum -> Y=1.
__forceinline__ __device__
float3 dev_spectrum_to_srgb(const Spectrum& s) {
    float X = 0.f, Y = 0.f, Z = 0.f;
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        X += s.value[i] * CIE_XBAR[i];
        Y += s.value[i] * CIE_YBAR[i];
        Z += s.value[i] * CIE_ZBAR[i];
    }

    // Normalise: divide by sum(ybar) so flat-1.0 -> Y=1
    X *= CIE_YBAR_SUM_INV; Y *= CIE_YBAR_SUM_INV; Z *= CIE_YBAR_SUM_INV;

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
    return params.mat_type[mat_id] == DEV_GLOSSY;
}

__forceinline__ __device__
bool dev_is_dielectric_glossy(uint32_t mat_id) {
    return params.mat_type[mat_id] == DEV_GLOSSY_DIELECTRIC;
}

__forceinline__ __device__
bool dev_is_clearcoat(uint32_t mat_id) {
    return params.mat_type[mat_id] == DEV_CLEARCOAT;
}

__forceinline__ __device__
bool dev_is_fabric(uint32_t mat_id) {
    return params.mat_type[mat_id] == DEV_FABRIC;
}

// Returns true for any glossy surface (metallic, dielectric, or clearcoat).
// Use this for glossy continuation gates; use dev_is_glossy() /
// dev_is_dielectric_glossy() only when differentiating Fresnel model.
__forceinline__ __device__
bool dev_is_any_glossy(uint32_t mat_id) {
    uint8_t t = params.mat_type[mat_id];
    return t == DEV_GLOSSY || t == DEV_GLOSSY_DIELECTRIC || t == DEV_CLEARCOAT;
}

// == GGX microfacet distribution (device-side) ========================

__forceinline__ __device__
float dev_ggx_D(float3 h, float alpha) {
    float NdotH = h.z;  // local frame: N = (0,0,1)
    if (NdotH <= 0.f) return 0.f;
    float a2 = alpha * alpha;
    float d  = NdotH * NdotH * (a2 - 1.f) + 1.f;
    return a2 / (PI * d * d);
}

__forceinline__ __device__
float dev_ggx_G1(float3 v, float alpha) {
    float NdotV = fabsf(v.z);
    float a2 = alpha * alpha;
    return 2.f * NdotV / (NdotV + sqrtf(a2 + (1.f - a2) * NdotV * NdotV));
}

__forceinline__ __device__
float dev_ggx_G(float3 wo, float3 wi, float alpha) {
    return dev_ggx_G1(wo, alpha) * dev_ggx_G1(wi, alpha);
}

__forceinline__ __device__
float dev_fresnel_schlick(float cos_theta, float f0) {
    float t = 1.f - cos_theta;
    float t2 = t * t;
    return f0 + (1.f - f0) * t2 * t2 * t;
}

// Sample GGX visible normal (VNDF) — device version
__forceinline__ __device__
float3 dev_ggx_sample_halfvector(float3 wo, float alpha, float u1, float u2) {
    // Stretch
    float3 wh = normalize(make_f3(alpha * wo.x, alpha * wo.y, wo.z));

    // Orthonormal basis
    float3 t1 = (wh.z < 0.9999f) ? normalize(cross(make_f3(0,0,1), wh))
                                   : make_f3(1,0,0);
    float3 t2 = cross(wh, t1);

    // Uniform disk sample
    float r   = sqrtf(u1);
    float phi = TWO_PI * u2;
    float p1  = r * cosf(phi);
    float p2  = r * sinf(phi);
    float s   = 0.5f * (1.f + wh.z);
    p2 = (1.f - s) * sqrtf(fmaxf(0.f, 1.f - p1*p1)) + s * p2;

    // Project onto hemisphere
    float3 nh = t1 * p1 + t2 * p2 + wh * sqrtf(fmaxf(0.f, 1.f - p1*p1 - p2*p2));

    // Unstretch
    return normalize(make_f3(alpha * nh.x, alpha * nh.y, fmaxf(0.f, nh.z)));
}

// == BSDF evaluate / pdf / sample (Lambertian + Cook-Torrance glossy) =

// Diffuse-only BSDF for photon density estimation (§6 standard practice).
// The full Cook-Torrance specular lobe produces unbounded variance
// in fixed-radius kernel estimators, creating coloured hotspots.
// Use the Lambertian component only for density estimation;
// NEE (direct lighting) still uses the full BSDF.
__forceinline__ __device__
Spectrum dev_bsdf_evaluate_diffuse(uint32_t mat_id, float3 wo, float3 wi, float2 uv) {
    if (wi.z <= 0.f || wo.z <= 0.f) return Spectrum::zero();
    Spectrum Kd = dev_get_Kd(mat_id, uv);

    // Clearcoat: diffuse portion attenuated by coat energy loss
    if (dev_is_clearcoat(mat_id)) {
        float coat_w = dev_get_clearcoat_weight(mat_id);
        float ior = dev_get_ior(mat_id);
        float coat_f0t = (ior - 1.f) / (ior + 1.f);
        float coat_F0  = coat_f0t * coat_f0t;
        float cos_o = fabsf(wo.z);
        float Fr = dev_fresnel_schlick(cos_o, coat_F0);
        Spectrum f;
        for (int i = 0; i < NUM_LAMBDA; ++i)
            f.value[i] = (1.f - coat_w * Fr) * Kd.value[i] * INV_PI;
        return f;
    }

    // Fabric: diffuse base only (no sheen for photon gather)
    return Kd * INV_PI;
}

__forceinline__ __device__
Spectrum dev_bsdf_evaluate(uint32_t mat_id, float3 wo, float3 wi, float2 uv) {
    if (wi.z <= 0.f || wo.z <= 0.f) return Spectrum::zero();

    Spectrum Kd = dev_get_Kd(mat_id, uv);

    // ── Clearcoat: dielectric coat GGX + attenuated Lambert base ────
    if (dev_is_clearcoat(mat_id)) {
        float coat_w = dev_get_clearcoat_weight(mat_id);
        float coat_r = dev_get_clearcoat_roughness(mat_id);
        float coat_alpha = fmaxf(coat_r * coat_r, 0.001f);
        float ior = dev_get_ior(mat_id);
        float coat_f0t = (ior - 1.f) / (ior + 1.f);
        float coat_F0  = coat_f0t * coat_f0t;

        float3 h = normalize(wo + wi);
        float ndf_c = dev_ggx_D(h, coat_alpha);
        float geo_c = dev_ggx_G(wo, wi, coat_alpha);
        float VdotH = fabsf(dot(wo, h));
        float Fr = dev_fresnel_schlick(VdotH, coat_F0);
        float denom = 4.f * fabsf(wo.z) * fabsf(wi.z) + EPSILON;
        float coat_spec = coat_w * (ndf_c * geo_c * Fr) / denom;
        Spectrum f;
        for (int i = 0; i < NUM_LAMBDA; ++i)
            f.value[i] = coat_spec + (1.f - coat_w * Fr) * Kd.value[i] * INV_PI;
        return f;
    }

    // ── Fabric: diffuse + sheen lobe ────────────────────────────────
    if (dev_is_fabric(mat_id)) {
        float sheen_w = dev_get_sheen(mat_id);
        float tint    = dev_get_sheen_tint(mat_id);
        float3 h = normalize(wo + wi);
        float cos_theta_h = fabsf(dot(wo, h));
        float t = 1.f - cos_theta_h;
        float t5 = t * t * t * t * t;
        Spectrum f;
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            float sheen_col = (1.f - tint) * 1.0f + tint * Kd.value[i];
            f.value[i] = Kd.value[i] * INV_PI + sheen_w * sheen_col * t5 * INV_PI;
        }
        return f;
    }

    if (!dev_is_glossy(mat_id) && !dev_is_dielectric_glossy(mat_id)) {
        // Pure Lambertian
        return Kd * INV_PI;
    }

    // Cook-Torrance glossy: specular lobe + diffuse
    Spectrum Ks = dev_get_Ks(mat_id);
    float roughness = dev_get_roughness(mat_id);
    float alpha = fmaxf(roughness * roughness, 0.001f);

    float3 h = normalize(wo + wi);
    float ndf = dev_ggx_D(h, alpha);
    float geo = dev_ggx_G(wo, wi, alpha);
    float VdotH = fabsf(dot(wo, h));
    float denom = 4.f * fabsf(wo.z) * fabsf(wi.z) + EPSILON;

    Spectrum f;
    if (dev_is_dielectric_glossy(mat_id)) {
        // Dielectric Fresnel: F0 from IOR, Ks scales specular color
        float ior = dev_get_ior(mat_id);
        float F0 = ((ior - 1.f) / (ior + 1.f)) * ((ior - 1.f) / (ior + 1.f));
        float Fr = dev_fresnel_schlick(VdotH, F0);
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            float spec = (ndf * geo * Fr * Ks.value[i]) / denom;
            float diff = (1.f - Fr) * Kd.value[i] * INV_PI;
            f.value[i] = spec + diff;
        }
    } else {
        // Metallic: Ks is the Fresnel F0 per wavelength
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            float Fr = dev_fresnel_schlick(VdotH, Ks.value[i]);
            f.value[i] = (ndf * geo * Fr) / denom + Kd.value[i] * INV_PI;
        }
    }
    return f;
}

__forceinline__ __device__
float dev_bsdf_pdf(uint32_t mat_id, float3 wo, float3 wi) {
    if (wi.z <= 0.f || wo.z <= 0.f) return 0.f;

    float diff_pdf = fmaxf(0.f, wi.z) * INV_PI;

    // ── Clearcoat: mixed coat-GGX + cosine pdf ─────────────────────
    if (dev_is_clearcoat(mat_id)) {
        float coat_w = dev_get_clearcoat_weight(mat_id);
        float coat_r = dev_get_clearcoat_roughness(mat_id);
        float coat_alpha = fmaxf(coat_r * coat_r, 0.001f);
        float ior = dev_get_ior(mat_id);
        float coat_f0t = (ior - 1.f) / (ior + 1.f);
        float coat_F0  = coat_f0t * coat_f0t;
        float p_coat = fmaxf(fminf(coat_w * coat_F0, 0.95f), 0.05f);

        float3 h = normalize(wo + wi);
        float ndf_c = dev_ggx_D(h, coat_alpha);
        float VdotH = fabsf(dot(wo, h));
        float spec_pdf = ndf_c * fabsf(h.z) / (4.f * VdotH + EPSILON);
        return p_coat * spec_pdf + (1.f - p_coat) * diff_pdf;
    }

    // ── Fabric: cosine hemisphere only ──────────────────────────────
    if (dev_is_fabric(mat_id)) {
        return diff_pdf;
    }

    if (!dev_is_glossy(mat_id) && !dev_is_dielectric_glossy(mat_id)) {
        return diff_pdf;
    }

    // Glossy: mixed PDF = p_spec * spec_pdf + (1-p_spec) * diff_pdf
    Spectrum Ks = dev_get_Ks(mat_id);
    Spectrum Kd = dev_get_Kd(mat_id, make_float2(0.f, 0.f));
    float roughness = dev_get_roughness(mat_id);
    float alpha = fmaxf(roughness * roughness, 0.001f);

    float spec_weight, diff_weight;
    if (dev_is_dielectric_glossy(mat_id)) {
        float ior = dev_get_ior(mat_id);
        float F0 = ((ior - 1.f) / (ior + 1.f)) * ((ior - 1.f) / (ior + 1.f));
        spec_weight = fmaxf(Ks.max_component() * F0, 0.05f);
        diff_weight = Kd.max_component();
    } else {
        spec_weight = Ks.max_component();
        diff_weight = Kd.max_component();
    }
    // Must match the roughness boost in dev_bsdf_sample()
    float roughness_boost = 1.f / (1.f + 10.f * alpha);
    spec_weight = fmaxf(spec_weight, roughness_boost);

    float total = spec_weight + diff_weight;
    float p_spec = (total > 0.f) ? spec_weight / total : 0.5f;

    float3 h = normalize(wo + wi);
    float ndf = dev_ggx_D(h, alpha);
    float VdotH = fabsf(dot(wo, h));
    float spec_pdf = ndf * fabsf(h.z) / (4.f * VdotH + EPSILON);

    return p_spec * spec_pdf + (1.f - p_spec) * diff_pdf;
}

// Overload for backward compatibility (Lambertian-only call sites)
__forceinline__ __device__
float dev_bsdf_pdf(float3 wi) {
    return fmaxf(0.f, wi.z) * INV_PI;
}

// Result struct for device BSDF sampling
struct DevBSDFSample {
    float3   wi;          // sampled direction (local frame)
    float    pdf;         // probability density
    Spectrum f;           // BSDF value f(wo, wi) per wavelength
    bool     is_specular; // true for delta distributions
};

// Sample BSDF: handles Lambertian, GlossyMetal (Cook-Torrance + diffuse)
// pixel_idx >= 0 enables the per-pixel Bresenham lobe balance accumulator
// (pass -1 to fall back to a random coin flip, e.g. for photon tracing).
__forceinline__ __device__
DevBSDFSample dev_bsdf_sample(uint32_t mat_id, float3 wo, float2 uv,
                              PCGRng& rng, int pixel_idx = -1) {
    DevBSDFSample s;
    s.is_specular = false;

    Spectrum Kd = dev_get_Kd(mat_id, uv);

    // ── Clearcoat: sample coat GGX or base Lambert ──────────────────
    if (dev_is_clearcoat(mat_id)) {
        float coat_w = dev_get_clearcoat_weight(mat_id);
        float coat_r = dev_get_clearcoat_roughness(mat_id);
        float coat_alpha = fmaxf(coat_r * coat_r, 0.001f);
        float ior = dev_get_ior(mat_id);
        float coat_f0t = (ior - 1.f) / (ior + 1.f);
        float coat_F0  = coat_f0t * coat_f0t;
        float p_coat = fmaxf(fminf(coat_w * coat_F0, 0.95f), 0.05f);

        if (rng.next_float() < p_coat) {
            // Sample coat GGX lobe
            float3 h = dev_ggx_sample_halfvector(wo, coat_alpha,
                                                  rng.next_float(), rng.next_float());
            s.wi = make_f3(2.f * dot(wo, h) * h.x - wo.x,
                           2.f * dot(wo, h) * h.y - wo.y,
                           2.f * dot(wo, h) * h.z - wo.z);
            if (s.wi.z <= 0.f) { s.pdf = 0.f; s.f = Spectrum::zero(); return s; }

            float VdotH = fabsf(dot(wo, h));
            float Fr = dev_fresnel_schlick(VdotH, coat_F0);
            float ndf_c = dev_ggx_D(h, coat_alpha);
            float geo_c = dev_ggx_G(wo, s.wi, coat_alpha);
            float spec_pdf = ndf_c * fabsf(h.z) / (4.f * VdotH + EPSILON);
            float diff_pdf = fmaxf(0.f, s.wi.z) * INV_PI;
            s.pdf = p_coat * spec_pdf + (1.f - p_coat) * diff_pdf;

            float denom = 4.f * fabsf(wo.z) * fabsf(s.wi.z) + EPSILON;
            float coat_spec = coat_w * (ndf_c * geo_c * Fr) / denom;
            for (int i = 0; i < NUM_LAMBDA; ++i)
                s.f.value[i] = coat_spec + (1.f - coat_w * Fr) * Kd.value[i] * INV_PI;
        } else {
            // Sample base Lambert
            s.wi = sample_cosine_hemisphere_dev(rng);
            if (s.wi.z <= 0.f) { s.pdf = 0.f; s.f = Spectrum::zero(); return s; }

            float diff_pdf = fmaxf(0.f, s.wi.z) * INV_PI;
            float3 h = normalize(wo + s.wi);
            float ndf_c = dev_ggx_D(h, coat_alpha);
            float VdotH = fabsf(dot(wo, h));
            float spec_pdf = ndf_c * fabsf(h.z) / (4.f * VdotH + EPSILON);
            s.pdf = p_coat * spec_pdf + (1.f - p_coat) * diff_pdf;

            float Fr = dev_fresnel_schlick(VdotH, coat_F0);
            float geo_c = dev_ggx_G(wo, s.wi, coat_alpha);
            float denom = 4.f * fabsf(wo.z) * fabsf(s.wi.z) + EPSILON;
            float coat_spec = coat_w * (ndf_c * geo_c * Fr) / denom;
            for (int i = 0; i < NUM_LAMBDA; ++i)
                s.f.value[i] = coat_spec + (1.f - coat_w * Fr) * Kd.value[i] * INV_PI;
        }
        return s;
    }

    // ── Fabric: cosine hemisphere + sheen ───────────────────────────
    if (dev_is_fabric(mat_id)) {
        s.wi = sample_cosine_hemisphere_dev(rng);
        if (s.wi.z <= 0.f) { s.pdf = 0.f; s.f = Spectrum::zero(); return s; }
        s.pdf = fmaxf(0.f, s.wi.z) * INV_PI;

        float sheen_w = dev_get_sheen(mat_id);
        float tint    = dev_get_sheen_tint(mat_id);
        float3 h = normalize(wo + s.wi);
        float cos_theta_h = fabsf(dot(wo, h));
        float t = 1.f - cos_theta_h;
        float t5 = t * t * t * t * t;
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            float sheen_col = (1.f - tint) * 1.0f + tint * Kd.value[i];
            s.f.value[i] = Kd.value[i] * INV_PI + sheen_w * sheen_col * t5 * INV_PI;
        }
        return s;
    }

    if (!dev_is_glossy(mat_id) && !dev_is_dielectric_glossy(mat_id)) {
        // Pure Lambertian: cosine hemisphere
        s.wi = sample_cosine_hemisphere_dev(rng);
        s.pdf = fmaxf(0.f, s.wi.z) * INV_PI;
        s.f = Kd * INV_PI;
        return s;
    }

    // Cook-Torrance glossy with diffuse+specular lobe selection
    Spectrum Ks = dev_get_Ks(mat_id);
    float roughness = dev_get_roughness(mat_id);
    float alpha = fmaxf(roughness * roughness, 0.001f);
    bool is_diel = dev_is_dielectric_glossy(mat_id);
    float ior_val = is_diel ? dev_get_ior(mat_id) : 1.5f;
    float F0_diel = ((ior_val - 1.f) / (ior_val + 1.f)) * ((ior_val - 1.f) / (ior_val + 1.f));

    float spec_weight, diff_weight;
    if (is_diel) {
        spec_weight = fmaxf(Ks.max_component() * F0_diel, 0.05f);
        diff_weight = Kd.max_component();
    } else {
        spec_weight = Ks.max_component();
        diff_weight = Kd.max_component();
    }

    // Boost specular sampling probability for low-roughness (near-mirror)
    // surfaces.  Without this, a shiny dielectric with large Kd and small
    // Ks*F0 sends ~94 % of samples to the diffuse cosine lobe, wasting
    // almost all of them — they evaluate to near-zero specular BSDF at
    // random directions far from the narrow GGX peak, adding pure noise.
    // The boost smoothly fades to zero for rough surfaces (alpha → 1).
    float roughness_boost = 1.f / (1.f + 10.f * alpha);
    spec_weight = fmaxf(spec_weight, roughness_boost);

    float total_w = spec_weight + diff_weight;
    float p_spec = (total_w > 0.f) ? spec_weight / total_w : 0.5f;

    // ── Lobe selection via Bresenham accumulator ─────────────────────
    // When pixel_idx >= 0 and the lobe_balance buffer is available we
    // use a Bresenham error accumulator instead of a random coin flip.
    // Positive balance = specular deficit → choose specular this sample.
    // This guarantees that over K samples the specular count is within
    // 1 of K * p_spec, eliminating binomial variance in lobe counts.
    bool choose_specular;
    if (pixel_idx >= 0 && params.lobe_balance) {
        float balance = params.lobe_balance[pixel_idx];
        choose_specular = (balance >= 0.f);
        // Update balance: positive means we owe more specular samples
        if (choose_specular)
            balance -= (1.f - p_spec);  // paid specular cost
        else
            balance += p_spec;          // paid diffuse cost
        params.lobe_balance[pixel_idx] = balance;
    } else {
        choose_specular = (rng.next_float() < p_spec);
    }

    if (choose_specular) {
        // Sample GGX specular lobe
        float u1 = rng.next_float();
        float u2 = rng.next_float();
        float3 h = dev_ggx_sample_halfvector(wo, alpha, u1, u2);

        s.wi = make_f3(2.f * dot(wo, h) * h.x - wo.x,
                       2.f * dot(wo, h) * h.y - wo.y,
                       2.f * dot(wo, h) * h.z - wo.z);

        if (s.wi.z <= 0.f) {
            s.pdf = 0.f;
            s.f = Spectrum::zero();
            return s;
        }

        float ndf = dev_ggx_D(h, alpha);
        float geo = dev_ggx_G(wo, s.wi, alpha);
        float VdotH = fabsf(dot(wo, h));

        float spec_pdf = ndf * fabsf(h.z) / (4.f * VdotH + EPSILON);
        float diff_pdf = fmaxf(0.f, s.wi.z) * INV_PI;
        s.pdf = p_spec * spec_pdf + (1.f - p_spec) * diff_pdf;

        float denom = 4.f * fabsf(wo.z) * fabsf(s.wi.z) + EPSILON;
        if (is_diel) {
            float Fr = dev_fresnel_schlick(VdotH, F0_diel);
            for (int i = 0; i < NUM_LAMBDA; ++i) {
                float spec = (ndf * geo * Fr * Ks.value[i]) / denom;
                float diff = (1.f - Fr) * Kd.value[i] * INV_PI;
                s.f.value[i] = spec + diff;
            }
        } else {
            for (int i = 0; i < NUM_LAMBDA; ++i) {
                float Fr = dev_fresnel_schlick(VdotH, Ks.value[i]);
                s.f.value[i] = (ndf * geo * Fr) / denom + Kd.value[i] * INV_PI;
            }
        }
    } else {
        // Sample diffuse lobe
        s.wi = sample_cosine_hemisphere_dev(rng);

        if (s.wi.z <= 0.f) {
            s.pdf = 0.f;
            s.f = Spectrum::zero();
            return s;
        }

        float diff_pdf = fmaxf(0.f, s.wi.z) * INV_PI;

        float3 h = normalize(wo + s.wi);
        float ndf = dev_ggx_D(h, alpha);
        float VdotH = fabsf(dot(wo, h));
        float spec_pdf = ndf * fabsf(h.z) / (4.f * VdotH + EPSILON);

        s.pdf = p_spec * spec_pdf + (1.f - p_spec) * diff_pdf;

        float geo = dev_ggx_G(wo, s.wi, alpha);
        float denom = 4.f * fabsf(wo.z) * fabsf(s.wi.z) + EPSILON;
        if (is_diel) {
            float Fr = dev_fresnel_schlick(VdotH, F0_diel);
            for (int i = 0; i < NUM_LAMBDA; ++i) {
                float spec = (ndf * geo * Fr * Ks.value[i]) / denom;
                float diff = (1.f - Fr) * Kd.value[i] * INV_PI;
                s.f.value[i] = spec + diff;
            }
        } else {
            for (int i = 0; i < NUM_LAMBDA; ++i) {
                float Fr = dev_fresnel_schlick(VdotH, Ks.value[i]);
                s.f.value[i] = (ndf * geo * Fr) / denom + Kd.value[i] * INV_PI;
            }
        }
    }

    return s;
}

// == MIS weight (2-way power heuristic, device-side) ==================
__forceinline__ __device__
float mis_weight_2_dev(float pdf_a, float pdf_b) {
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    return a2 / fmaxf(a2 + b2, 1e-30f);
}

// == Compute light sampling PDF for a direction that hit a light =======
__forceinline__ __device__
float dev_light_pdf(uint32_t tri_id, float3 geo_normal, float3 wi, float t) {
    if (params.num_emissive == 0) return 0.f;

    // Find this triangle in the emissive list
    float pdf_tri = 0.f;
    for (int i = 0; i < params.num_emissive; ++i) {
        if (params.emissive_tri_indices[i] == tri_id) {
            if (i == 0) pdf_tri = params.emissive_cdf[0];
            else pdf_tri = params.emissive_cdf[i] - params.emissive_cdf[i - 1];
            break;
        }
    }
    if (pdf_tri <= 0.f) return 0.f;

    float3 v0 = params.vertices[tri_id * 3 + 0];
    float3 v1 = params.vertices[tri_id * 3 + 1];
    float3 v2 = params.vertices[tri_id * 3 + 2];
    float area = length(cross(v1 - v0, v2 - v0)) * 0.5f;
    if (area <= 0.f) return 0.f;

    float cos_o = fabsf(dot(wi * (-1.f), geo_normal));
    if (cos_o <= 0.f) return 0.f;

    float dist2 = t * t;
    return pdf_tri * (1.f / area) * dist2 / cos_o;
}

// == Debug render modes ===============================================

__forceinline__ __device__
Spectrum render_normals_dev(const TraceResult& hit) {
    Spectrum s;
    float r = hit.shading_normal.x * 0.5f + 0.5f;
    float g = hit.shading_normal.y * 0.5f + 0.5f;
    float b = hit.shading_normal.z * 0.5f + 0.5f;
    for (int i = 0; i < NUM_LAMBDA; ++i) s.value[i] = 0.f;
    for (int i = 0; i < NUM_LAMBDA/3; ++i) s.value[i] = r;
    for (int i = NUM_LAMBDA/3; i < 2*NUM_LAMBDA/3; ++i) s.value[i] = g;
    for (int i = 2*NUM_LAMBDA/3; i < NUM_LAMBDA; ++i) s.value[i] = b;
    return s;
}

__forceinline__ __device__
Spectrum render_material_id_dev(uint32_t mat_id) {
    float hue = fmodf(mat_id * 0.618033988f, 1.f);
    float r = fabsf(hue*6.f - 3.f) - 1.f;
    float g = 2.f - fabsf(hue*6.f - 2.f);
    float b = 2.f - fabsf(hue*6.f - 4.f);
    r = fminf(fmaxf(r, 0.f), 1.f);
    g = fminf(fmaxf(g, 0.f), 1.f);
    b = fminf(fmaxf(b, 0.f), 1.f);
    Spectrum s;
    for (int i = 0; i < NUM_LAMBDA; ++i) s.value[i] = 0.f;
    for (int i = 0; i < NUM_LAMBDA/3; ++i) s.value[i] = r;
    for (int i = NUM_LAMBDA/3; i < 2*NUM_LAMBDA/3; ++i) s.value[i] = g;
    for (int i = 2*NUM_LAMBDA/3; i < NUM_LAMBDA; ++i) s.value[i] = b;
    return s;
}

__forceinline__ __device__
Spectrum render_depth_dev(float t, float max_depth) {
    float d = fminf(t / max_depth, 1.f);
    return Spectrum::constant(1.f - d);
}

// == Hash grid photon lookup (device-side) ============================

__forceinline__ __device__
uint32_t dev_hash_cell(int cx, int cy, int cz, uint32_t table_size) {
    uint32_t h = (uint32_t)(cx * (int)HASHGRID_PRIME_1)
               ^ (uint32_t)(cy * (int)HASHGRID_PRIME_2)
               ^ (uint32_t)(cz * (int)HASHGRID_PRIME_3);
    return h % table_size;
}

__forceinline__ __device__
Spectrum dev_estimate_photon_density(
    float3 pos, float3 normal,       // shading normal (ONB / BSDF frame)
    float3 filter_normal,            // geometric normal (surface filtering §15.1.2)
    float3 wo_local, uint32_t mat_id,
    float radius, float2 uv)
{
    Spectrum L = Spectrum::zero();
    if (params.num_photons == 0 || params.grid_table_size == 0) return L;

    // ── Per-photon hash-grid walk O(N_cell) ──────────────────────────
    // Dual-budget caustic gather (Jensen 1996 two-budget approach).
    // Three-tag system (0=global noncaustic, 1=global caustic, 2=caustic pass):
    //   tag 0 → L_global    normalised by 1/N_global  with gather_radius
    //   tag 1 → SKIPPED     (superseded by dedicated caustic map)
    //   tag 2 → L_caustic   normalised by 1/N_caustic with caustic_gather_radius
    // Result = L_global + L_caustic (additive, no double-counting).

    const bool dual_budget = (params.photon_is_caustic_pass != nullptr
                              && params.num_caustic_emitted > 0);

    // Radii and normalisation for each budget
    float r_global  = radius;
    float r_caustic = dual_budget ? params.caustic_gather_radius : radius;
    float r_max     = fmaxf(r_global, r_caustic);  // search radius for cell traversal

    float r2_global  = r_global * r_global;
    float r2_caustic = r_caustic * r_caustic;
    float inv_area_global  = 2.f / (PI * r2_global);
    float inv_area_caustic = dual_budget ? 2.f / (PI * r2_caustic) : inv_area_global;
    float norm_global  = 1.f / (float)params.num_photons_emitted;
    float norm_caustic = dual_budget ? 1.f / (float)params.num_caustic_emitted : norm_global;

    Spectrum L_global  = Spectrum::zero();
    Spectrum L_caustic = Spectrum::zero();

    float cell_size = params.grid_cell_size;
    int cx0 = (int)floorf((pos.x - r_max) / cell_size);
    int cy0 = (int)floorf((pos.y - r_max) / cell_size);
    int cz0 = (int)floorf((pos.z - r_max) / cell_size);
    int cx1 = (int)floorf((pos.x + r_max) / cell_size);
    int cy1 = (int)floorf((pos.y + r_max) / cell_size);
    int cz1 = (int)floorf((pos.z + r_max) / cell_size);

    float r2_search = r_max * r_max;  // broad search, per-photon radius check below

    // Track visited bucket keys to avoid double-processing hash-colliding cells
    // (same bucket can be reached from multiple distinct (ix,iy,iz) coordinates).
    uint32_t visited_keys[27];
    int num_visited = 0;

    // Build ONB once outside the photon loop (reused for every wi transform)
    DevONB frame = DevONB::from_normal(normal);

    for (int iz = cz0; iz <= cz1; ++iz)
    for (int iy = cy0; iy <= cy1; ++iy)
    for (int ix = cx0; ix <= cx1; ++ix) {
        uint32_t key = dev_hash_cell(ix, iy, iz, params.grid_table_size);

        // Skip if this hash bucket was already processed
        bool already_visited = false;
        for (int v = 0; v < num_visited; ++v) {
            if (visited_keys[v] == key) { already_visited = true; break; }
        }
        if (already_visited) continue;
        visited_keys[num_visited++] = key;

        uint32_t start = params.grid_cell_start[key];
        uint32_t end   = params.grid_cell_end[key];
        if (start == 0xFFFFFFFF) continue;

        for (uint32_t j = start; j < end; ++j) {
            uint32_t idx = params.grid_sorted_indices[j];
            float3 pp = make_f3(
                params.photon_pos_x[idx],
                params.photon_pos_y[idx],
                params.photon_pos_z[idx]);

            float3 diff = pos - pp;

            // Tangential-disk distance metric (§7.1 guideline)
            float d_plane = dot(diff, filter_normal);
            float3 v_tan = diff - filter_normal * d_plane;
            float d_tan2 = dot(v_tan, v_tan);

            // Broad-phase: reject beyond max radius
            if (d_tan2 > r2_search) continue;

            float plane_dist = fabsf(d_plane);
            if (plane_dist > DEFAULT_SURFACE_TAU) continue;

            // Visibility term: photon must be on the same side of the surface
            // as the query point.  This rejects photons deposited on the back
            // face of a wall (stored normal faces away → dot < 0) and prevents
            // irradiance leaking through thin geometry even when surface_tau is
            // too loose to catch the discrepancy via plane distance alone.
            float3 photon_n = make_f3(
                params.photon_norm_x[idx],
                params.photon_norm_y[idx],
                params.photon_norm_z[idx]);
            if (dot(photon_n, filter_normal) <= 0.0f) continue;

            float3 wi_world = make_f3(
                params.photon_wi_x[idx],
                params.photon_wi_y[idx],
                params.photon_wi_z[idx]);

            // photon_wi points away from surface toward light
            // (stored as -direction in photon tracer) — do NOT negate
            if (dot(wi_world, filter_normal) <= 0.f) continue;
            float3 wi_local = frame.world_to_local(wi_world);

            // Diffuse-only BSDF for density estimation (§6 standard practice).
            // Full Cook-Torrance creates 50x+ variance on glossy surfaces.
            Spectrum f = dev_bsdf_evaluate_diffuse(mat_id, wo_local, wi_local, uv);

            // Route to appropriate accumulator based on 3-value tag:
            //   0 = global noncaustic → L_global   (1/N_global, r_global)
            //   1 = global caustic    → SKIP       (superseded by caustic map)
            //   2 = caustic pass      → L_caustic  (1/N_caustic, r_caustic)
            uint8_t tag = dual_budget ? params.photon_is_caustic_pass[idx] : 0;
            if (tag == 1) continue;  // skip global-pass caustic photons
            bool is_caustic_pass = (tag == 2);
            float my_r2       = is_caustic_pass ? r2_caustic      : r2_global;
            float my_inv_area = is_caustic_pass ? inv_area_caustic : inv_area_global;
            float my_norm     = is_caustic_pass ? norm_caustic     : norm_global;

            // Per-budget radius check
            if (d_tan2 > my_r2) continue;

            // Epanechnikov kernel weight (§6.3 guideline)
            float w = 1.0f - d_tan2 / my_r2;

            Spectrum& L_target = is_caustic_pass ? L_caustic : L_global;

            // Accumulate HERO_WAVELENGTHS bins per photon (multi-hero transport)
            int n_hero = params.photon_num_hero ? (int)params.photon_num_hero[idx] : 1;
            for (int h = 0; h < n_hero; ++h) {
                float p_flux = params.photon_flux[idx * HERO_WAVELENGTHS + h];
                int bin = (int)params.photon_lambda[idx * HERO_WAVELENGTHS + h];
                if (bin >= 0 && bin < NUM_LAMBDA)
                    L_target.value[bin] += f.value[bin] * p_flux * w * my_inv_area * my_norm;
            }
        }
    }

    // Additive combination: non-caustic indirect from global map +
    // caustic contribution from dedicated caustic map (no double-counting
    // because global-pass caustic photons are tagged=1 and skipped above).
    if (dual_budget) {
        for (int lam = 0; lam < NUM_LAMBDA; ++lam)
            L.value[lam] = L_global.value[lam] + L_caustic.value[lam];
    } else {
        L = L_global;
    }
    return L;
}

// NOTE: CDF binary search lives in core/cdf.h (host+device).

// == Sample triangle barycentric (device) =============================
__forceinline__ __device__
float3 sample_triangle_dev(float u1, float u2) {
    float su = sqrtf(u1);
    float alpha = 1.f - su;
    float beta  = u2 * su;
    float gamma = 1.f - alpha - beta;
    return make_f3(alpha, beta, gamma);
}

// NOTE: CDF binary search lives in core/cdf.h (host+device).

// NeeResult: returned by all NEE variants
struct NeeResult {
    Spectrum L;                // direct lighting contribution
    float    visibility;       // fraction of unoccluded shadow samples [0,1]
};

// =====================================================================
// dev_nee_evaluate_sample -- shared inner loop for all NEE variants
//
// Given a selected emissive triangle (local_idx) and its selection
// probability (p_tri), samples a point on the triangle, casts a shadow
// ray, evaluates emission × BSDF × MIS, and returns the single-sample
// contribution.  Each NEE variant only needs to implement its own
// emitter selection strategy and PDF computation, then call this helper.
// =====================================================================
struct NeeSampleResult {
    Spectrum L;       // MIS-weighted contribution (zero if occluded/backfacing)
    bool     visible; // shadow ray unoccluded?
};

__forceinline__ __device__
NeeSampleResult dev_nee_evaluate_sample(
    int local_idx, float p_tri,
    float3 pos, float3 normal, float3 wo_local,
    uint32_t mat_id, const DevONB& frame, float2 uv, PCGRng& rng)
{
    NeeSampleResult r;
    r.L = Spectrum::zero();
    r.visible = false;

    uint32_t light_tri = params.emissive_tri_indices[local_idx];

    // Sample uniform point on triangle
    float3 bary = sample_triangle_dev(rng.next_float(), rng.next_float());
    float3 lv0 = params.vertices[light_tri * 3 + 0];
    float3 lv1 = params.vertices[light_tri * 3 + 1];
    float3 lv2 = params.vertices[light_tri * 3 + 2];
    float3 light_pos = lv0 * bary.x + lv1 * bary.y + lv2 * bary.z;

    float3 le1 = lv1 - lv0;
    float3 le2 = lv2 - lv0;
    float3 light_normal = normalize(cross(le1, le2));
    float  light_area   = length(cross(le1, le2)) * 0.5f;

    // Direction, distance, cosines
    float3 to_light = light_pos - pos;
    float dist2 = dot(to_light, to_light);
    float dist  = sqrtf(dist2);
    float3 wi   = to_light * (1.f / dist);

    float cos_x = dot(wi, normal);
    float cos_y = -dot(wi, light_normal);
    if (cos_x <= 0.f || cos_y <= 0.f) return r;

    // Shadow ray
    if (!trace_shadow(pos + normal * OPTIX_SCENE_EPSILON, wi, dist))
        return r;
    r.visible = true;

    if (p_tri <= 0.f) return r;

    // Emission and BSDF
    uint32_t light_mat = params.material_ids[light_tri];
    float2 luv0 = params.texcoords[light_tri * 3 + 0];
    float2 luv1 = params.texcoords[light_tri * 3 + 1];
    float2 luv2 = params.texcoords[light_tri * 3 + 2];
    float2 light_uv = make_float2(
        luv0.x * bary.x + luv1.x * bary.y + luv2.x * bary.z,
        luv0.y * bary.x + luv1.y * bary.y + luv2.y * bary.z);
    Spectrum Le = dev_get_Le(light_mat, light_uv);

    float3 wi_local = frame.world_to_local(wi);
    Spectrum f = dev_bsdf_evaluate(mat_id, wo_local, wi_local, uv);

    // PDF conversion: area → solid angle
    float p_y_area = p_tri / light_area;
    float p_wi     = p_y_area * dist2 / cos_y;

    // MIS vs BSDF sampling
    float w_mis = 1.0f;
    if (DEFAULT_USE_MIS) {
        float pdf_bsdf = dev_bsdf_pdf(mat_id, wo_local, wi_local);
        w_mis = mis_weight_2_dev(p_wi, pdf_bsdf);
    }

    // MIS-weighted: f * Le * cos_x / p_wi
    for (int i = 0; i < NUM_LAMBDA; ++i)
        r.L.value[i] = w_mis * f.value[i] * Le.value[i]
                      * cos_x / fmaxf(p_wi, 1e-8f);
    return r;
}

// =====================================================================
// Cauchy dispersion helpers (§10.1)
// n(λ) = A + B / λ²   (λ in nm).
// Falls back to constant IOR when dispersion is disabled.
// =====================================================================
__forceinline__ __device__
float dev_ior_at_lambda(uint32_t mat_id, float lambda_nm) {
    if (params.mat_dispersion && params.mat_dispersion[mat_id])
        return params.cauchy_A[mat_id] + params.cauchy_B[mat_id] / (lambda_nm * lambda_nm);
    return dev_get_ior(mat_id);
}

__forceinline__ __device__
bool dev_has_dispersion(uint32_t mat_id) {
    return params.mat_dispersion && params.mat_dispersion[mat_id];
}

// =====================================================================
// dev_fresnel_dielectric — Fresnel reflectance for a dielectric boundary.
// eta = n_i / n_t (ratio of IORs).  cos_i = |cosine of incident angle|.
// Returns F (reflectance), 0..1.   F == 1.0 when TIR.
// =====================================================================
__forceinline__ __device__
float dev_fresnel_dielectric(float cos_i, float eta) {
    float sin2_t = eta * eta * (1.f - cos_i * cos_i);
    if (sin2_t >= 1.f) return 1.f;       // total internal reflection
    float cos_t = sqrtf(1.f - sin2_t);
    float rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    float rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    return 0.5f * (rs * rs + rp * rp);
}

// =====================================================================
// dev_specular_bounce — Shared helper for glass (Fresnel dielectric) and
// mirror reflection.  Returns the new ray direction, offset position, and
// a spectral throughput filter.
//
// The filter already accounts for the stochastic Fresnel choice: the
// hero wavelength decides reflect vs refract (probability F_hero for
// reflection, 1-F_hero for refraction), and the filter re-weights every
// wavelength bin by the ratio of that bin's Fresnel to the hero's.
//
// This ensures identical energy accounting to the CPU BSDF
// (bsdf.h glass_sample), where:
//   reflect: throughput *= (F_b / F_hero)                 — no Tf
//   refract: throughput *= Tf[b] * (1 - F_b) / (1 - F_hero) — with Tf
//   (bins with TIR get zero weight)
//
// Chromatic dispersion (Cauchy equation, §10.1):
// When dispersion is enabled, the hero wavelength (hero_bins[0] if
// provided, else bin 0) determines the refraction direction.  All other
// wavelength bins share that direction but receive per-wavelength Fresnel
// weights in `filter`.
//
// Without dispersion: all bins share the same IOR/Fresnel, the ratio
// is 1 everywhere: reflect → filter = 1.0, refract → filter = Tf.
// This recovers the original non-dispersive behaviour exactly.
// =====================================================================
struct SpecularBounceResult {
    float3   new_dir;
    float3   new_pos;
    Spectrum filter;   // throughput multiplier for the chosen path
};

__forceinline__ __device__
SpecularBounceResult dev_specular_bounce(
    float3 dir, float3 pos, float3 normal,
    uint32_t mat_id, float2 uv, PCGRng& rng,
    const int* hero_bins = nullptr, int num_hero = 0)
{
    SpecularBounceResult r;
    r.filter = Spectrum::constant(1.0f);

    if (dev_is_glass(mat_id) || dev_is_translucent(mat_id)) {
        bool entering = dot(dir, normal) < 0.f;
        float3 outward_n = entering ? normal : normal * (-1.f);
        float cos_i = fabsf(dot(dir, outward_n));

        // Hero wavelength determines direction when dispersion is enabled.
        // When called from the camera path (no hero_bins), use the D-line
        // bin (~589 nm) so the refraction direction uses the nominal IOR
        // instead of the extreme UV bin 0 (380 nm, IOR much too high).
        constexpr int DLINE_BIN = (int)((589.0f - LAMBDA_MIN) / LAMBDA_STEP);
        int hero_bin = (hero_bins && num_hero > 0) ? hero_bins[0] : DLINE_BIN;
        float hero_ior = dev_ior_at_lambda(mat_id, lambda_of_bin(hero_bin));
        float eta_hero = entering ? (1.f / hero_ior) : hero_ior;

        // Fresnel reflectance at hero wavelength
        float F_hero = dev_fresnel_dielectric(cos_i, eta_hero);

        Spectrum Tf = dev_get_Tf(mat_id, uv);
        bool do_reflect = (rng.next_float() < F_hero);

        if (do_reflect) {
            r.new_dir = dir - outward_n * (2.f * dot(dir, outward_n));
            r.new_pos = pos + outward_n * OPTIX_SCENE_EPSILON;

            if (dev_has_dispersion(mat_id)) {
                // Reflection filter: F_b / F_hero  (Tf NOT applied)
                float inv_F_hero = 1.f / fmaxf(F_hero, 1e-8f);
                for (int b = 0; b < NUM_LAMBDA; ++b) {
                    float n_b   = dev_ior_at_lambda(mat_id, lambda_of_bin(b));
                    float eta_b = entering ? (1.f / n_b) : n_b;
                    float F_b   = dev_fresnel_dielectric(cos_i, eta_b);
                    r.filter.value[b] = F_b * inv_F_hero;
                }
            }
            // else: non-dispersive → F_b == F_hero for all bins,
            //       ratio = 1.0, filter stays constant(1.0).  Correct.
        } else {
            // Refract — direction from hero wavelength IOR
            float sin2_hero = eta_hero * eta_hero * (1.f - cos_i * cos_i);
            float3 refracted = dir * eta_hero +
                outward_n * (eta_hero * cos_i - sqrtf(fmaxf(0.f, 1.f - sin2_hero)));
            r.new_dir = normalize(refracted);
            r.new_pos = pos - outward_n * OPTIX_SCENE_EPSILON;

            if (dev_has_dispersion(mat_id)) {
                // Refraction filter: Tf[b] * (1 - F_b) / (1 - F_hero)
                // Bins with TIR (F_b == 1) get zero weight.
                float inv_one_minus_F = 1.f / fmaxf(1.f - F_hero, 1e-8f);
                for (int b = 0; b < NUM_LAMBDA; ++b) {
                    float n_b   = dev_ior_at_lambda(mat_id, lambda_of_bin(b));
                    float eta_b = entering ? (1.f / n_b) : n_b;
                    float F_b   = dev_fresnel_dielectric(cos_i, eta_b);
                    if (F_b >= 1.f) {
                        r.filter.value[b] = 0.f;  // TIR at this wavelength
                    } else {
                        r.filter.value[b] = Tf.value[b] * (1.f - F_b) * inv_one_minus_F;
                    }
                }
            } else {
                // Non-dispersive: (1-F)/(1-F) = 1 for all bins → filter = Tf
                r.filter = Tf;
            }
        }
    } else {
        // Mirror: pure reflection
        r.new_dir = dir - normal * (2.f * dot(dir, normal));
        r.new_pos = pos + normal * OPTIX_SCENE_EPSILON;
    }
    return r;
}

// =====================================================================
// DEBUG FIRST-HIT RENDERING
// Simple direct-lighting: one ray, one shadow test, one frame
// =====================================================================
__forceinline__ __device__
Spectrum debug_first_hit(float3 origin, float3 direction, PCGRng& rng) {
    TraceResult hit = trace_radiance(origin, direction);
    if (!hit.hit) return Spectrum::zero();

    uint32_t mat_id = hit.material_id;

    // Debug visualisation modes
    if (params.render_mode == RENDER_MODE_NORMALS)
        return render_normals_dev(hit);
    if (params.render_mode == RENDER_MODE_MATERIAL_ID)
        return render_material_id_dev(mat_id);
    if (params.render_mode == RENDER_MODE_DEPTH)
        return render_depth_dev(hit.t, 5.0f);

    // Per-triangle photon irradiance heatmap (§ preview photon map)
    if (params.show_photon_heatmap && params.tri_photon_irradiance
        && hit.triangle_id < (uint32_t)params.num_triangles) {
        float density = params.tri_photon_irradiance[hit.triangle_id];
        // Log-scale heatmap: blue(cold) → green → red(hot)
        float t = fminf(log2f(density * 1e4f + 1.f) / 15.f, 1.f);
        // Simple 3-stop gradient: blue → green → red
        float r = fminf(fmaxf(t * 2.f - 1.f, 0.f), 1.f);
        float g = 1.f - fabsf(t * 2.f - 1.f);
        float b = fminf(fmaxf(1.f - t * 2.f, 0.f), 1.f);
        Spectrum s;
        for (int i = 0; i < NUM_LAMBDA; ++i) s.value[i] = 0.f;
        for (int i = 0; i < NUM_LAMBDA/3; ++i) s.value[i] = r;
        for (int i = NUM_LAMBDA/3; i < 2*NUM_LAMBDA/3; ++i) s.value[i] = g;
        for (int i = 2*NUM_LAMBDA/3; i < NUM_LAMBDA; ++i) s.value[i] = b;
        return s;
    }

    // Emission
    if (dev_is_emissive(mat_id))
        return dev_get_Le(mat_id, hit.uv);

    // Specular: one bounce then direct lighting
    float3 cur_pos = hit.position;
    float3 cur_dir = direction;
    float3 cur_normal = hit.shading_normal;
    uint32_t cur_mat = mat_id;
    float2 cur_uv = hit.uv;
    Spectrum throughput_s = Spectrum::constant(1.0f);

    for (int bounce = 0; bounce < DEFAULT_MAX_SPECULAR_CHAIN; ++bounce) {
        if (!dev_is_specular(cur_mat) && !dev_is_translucent(cur_mat)) break;

        SpecularBounceResult sb = dev_specular_bounce(
            cur_dir, cur_pos, cur_normal, cur_mat, cur_uv, rng);
        for (int i = 0; i < NUM_LAMBDA; ++i)
            throughput_s.value[i] *= sb.filter.value[i];
        cur_dir = sb.new_dir;
        cur_pos = sb.new_pos;

        TraceResult hit2 = trace_radiance(cur_pos, cur_dir);
        if (!hit2.hit) return Spectrum::zero();
        if (dev_is_emissive(hit2.material_id))
            return throughput_s * dev_get_Le(hit2.material_id, hit2.uv);
        cur_pos = hit2.position;
        cur_normal = hit2.shading_normal;
        cur_mat = hit2.material_id;
        cur_uv = hit2.uv;
    }

    // Diffuse/glossy hit: fast single-sample unshadowed direct lighting.
    // When params.debug_shadow_rays is on, __raygen__render routes to
    // full_path_trace instead, so this path is always the fast preview.
    Spectrum L = Spectrum::zero();

    {
        // Real-time debug: fast single-sample, no shadow ray
        // Supports both diffuse and glossy materials with reflection bounces
        Spectrum glossy_tp = Spectrum::constant(1.0f);

        for (int gb = 0; gb <= DEFAULT_MAX_GLOSSY_BOUNCES; ++gb) {
            DevONB frame = DevONB::from_normal(cur_normal);
            float3 wo_local = frame.world_to_local(normalize(-cur_dir));
            if (wo_local.z <= 0.f) break;

            // ── Fast unshadowed direct lighting at this hit ──
            if (params.num_emissive > 0) {
                float xi = rng.next_float();
                int local_idx = binary_search_cdf(
                    params.emissive_cdf, params.num_emissive, xi);
                if (local_idx >= params.num_emissive) local_idx = params.num_emissive - 1;
                uint32_t light_tri = params.emissive_tri_indices[local_idx];

                float3 bary = sample_triangle_dev(rng.next_float(), rng.next_float());
                float3 lv0 = params.vertices[light_tri * 3 + 0];
                float3 lv1 = params.vertices[light_tri * 3 + 1];
                float3 lv2 = params.vertices[light_tri * 3 + 2];
                float3 light_pos = lv0 * bary.x + lv1 * bary.y + lv2 * bary.z;

                float3 le1 = lv1 - lv0;
                float3 le2 = lv2 - lv0;
                float3 light_normal = normalize(cross(le1, le2));
                float  light_area   = length(cross(le1, le2)) * 0.5f;

                float3 to_light = light_pos - cur_pos;
                float dist2 = dot(to_light, to_light);
                float dist  = sqrtf(dist2);
                float3 wi   = to_light * (1.f / dist);

                float cos_i = dot(wi, cur_normal);
                float cos_o = -dot(wi, light_normal);

                if (cos_i > 0.f && cos_o > 0.f) {
                    float pdf_tri;
                    if (local_idx == 0)
                        pdf_tri = params.emissive_cdf[0];
                    else
                        pdf_tri = params.emissive_cdf[local_idx] - params.emissive_cdf[local_idx - 1];

                    float pdf_area = 1.f / light_area;
                    float geom = cos_o / dist2;
                    float pdf_solid_angle = pdf_tri * pdf_area / geom;

                    uint32_t light_mat = params.material_ids[light_tri];
                    // Interpolate UV at sampled light point for emission texture
                    float2 luv0 = params.texcoords[light_tri * 3 + 0];
                    float2 luv1 = params.texcoords[light_tri * 3 + 1];
                    float2 luv2 = params.texcoords[light_tri * 3 + 2];
                    float2 light_uv = make_float2(
                        luv0.x * bary.x + luv1.x * bary.y + luv2.x * bary.z,
                        luv0.y * bary.x + luv1.y * bary.y + luv2.y * bary.z);
                    Spectrum Le = dev_get_Le(light_mat, light_uv);

                    // Use full BSDF for glossy surfaces, Lambertian otherwise
                    float3 wi_local = frame.world_to_local(wi);
                    Spectrum bsdf_val = dev_bsdf_evaluate(cur_mat, wo_local, wi_local, cur_uv);

                    for (int i = 0; i < NUM_LAMBDA; ++i) {
                        L.value[i] += glossy_tp.value[i] * Le.value[i]
                                     * bsdf_val.value[i]
                                     * cos_i / fmaxf(pdf_solid_angle, 1e-8f);
                    }
                }
            }

            // ── Glossy continuation: specular reflection for glossy surfaces ──
            // Trace a deterministic mirror-direction ray weighted by the correct
            // Monte Carlo weight G(wo,wi) × F(cosθ).  The GGX NDF D cancels
            // with the importance-sampling PDF, so we never evaluate D directly.
            // Only for near-mirror surfaces (roughness < 0.1) where one ray
            // at the specular peak is a reasonable approximation of the lobe.
            if (!dev_is_any_glossy(cur_mat)) break;

            // For clearcoat, use the coat roughness for continuation check
            float cont_roughness = dev_is_clearcoat(cur_mat)
                ? dev_get_clearcoat_roughness(cur_mat)
                : dev_get_roughness(cur_mat);
            if (cont_roughness >= 0.1f) break;

            // Mirror reflection direction (used as the half-vector = normal)
            float3 refl_dir = cur_dir - cur_normal * (2.f * dot(cur_dir, cur_normal));
            refl_dir = normalize(refl_dir);

            // Glossy continuation weight:
            // We trace one ray at the mirror direction.  The correct Monte
            // Carlo weight for a GGX-importance-sampled mirror direction is
            //   weight = G(wo,wi) · F(cosθ)
            // (the NDF D in the BRDF numerator cancels with the GGX sampling
            // PDF).  This avoids the old D·G·F/(4cos²) blowout for smooth
            // surfaces where D(n) → ∞.
            float cos_view = fabsf(dot(normalize(-cur_dir), cur_normal));
            float alpha_r = fmaxf(cont_roughness * cont_roughness, 0.001f);

            // Smith G for both wo and wi (same angle for mirror direction)
            float G_val = dev_ggx_G1(wo_local, alpha_r);
            G_val *= G_val;  // G(wo,wi) ≈ G1(wo)·G1(wi), same angle

            if (dev_is_clearcoat(cur_mat)) {
                // Clearcoat: dielectric coat Fresnel, weight = coat_w × G × Fr
                float coat_w = dev_get_clearcoat_weight(cur_mat);
                float ior_r = dev_get_ior(cur_mat);
                float F0_r = ((ior_r - 1.f) / (ior_r + 1.f)) * ((ior_r - 1.f) / (ior_r + 1.f));
                float Fr_r = dev_fresnel_schlick(cos_view, F0_r);
                for (int i = 0; i < NUM_LAMBDA; ++i)
                    glossy_tp.value[i] *= G_val * coat_w * Fr_r;
            } else if (dev_is_dielectric_glossy(cur_mat)) {
                Spectrum Ks_r = dev_get_Ks(cur_mat);
                float ior_r = dev_get_ior(cur_mat);
                float F0_r = ((ior_r - 1.f) / (ior_r + 1.f)) * ((ior_r - 1.f) / (ior_r + 1.f));
                float Fr_r = dev_fresnel_schlick(cos_view, F0_r);
                for (int i = 0; i < NUM_LAMBDA; ++i)
                    glossy_tp.value[i] *= G_val * Fr_r * Ks_r.value[i];
            } else {
                // Metallic: per-channel Fresnel
                Spectrum Ks_r = dev_get_Ks(cur_mat);
                for (int i = 0; i < NUM_LAMBDA; ++i) {
                    float Fr_r = dev_fresnel_schlick(cos_view, Ks_r.value[i]);
                    glossy_tp.value[i] *= G_val * Fr_r;
                }
            }

            float max_tp = glossy_tp.max_component();
            if (max_tp < 0.001f) break;

            cur_pos = cur_pos + cur_normal * OPTIX_SCENE_EPSILON;
            TraceResult hit_g = trace_radiance(cur_pos, refl_dir);
            if (!hit_g.hit) break;

            cur_mat = hit_g.material_id;
            cur_dir = refl_dir;

            if (dev_is_emissive(cur_mat)) {
                L += glossy_tp * dev_get_Le(cur_mat, hit_g.uv);
                break;
            }

            // Follow specular chain if we hit mirror/glass/translucent
            if (dev_is_specular(cur_mat) || dev_is_translucent(cur_mat)) {
                for (int s = 0; s < DEFAULT_MAX_SPECULAR_CHAIN; ++s) {
                    SpecularBounceResult sb = dev_specular_bounce(
                        cur_dir, hit_g.position, hit_g.shading_normal,
                        cur_mat, hit_g.uv, rng);
                    for (int i = 0; i < NUM_LAMBDA; ++i)
                        glossy_tp.value[i] *= sb.filter.value[i];
                    cur_dir = sb.new_dir;
                    cur_pos = sb.new_pos;

                    hit_g = trace_radiance(cur_pos, cur_dir);
                    if (!hit_g.hit) goto debug_done;
                    cur_mat = hit_g.material_id;
                    if (dev_is_emissive(cur_mat)) {
                        L += glossy_tp * dev_get_Le(cur_mat, hit_g.uv);
                        goto debug_done;
                    }
                    if (!dev_is_specular(cur_mat) && !dev_is_translucent(cur_mat)) break;
                }
            }

            cur_pos    = hit_g.position;
            cur_normal = hit_g.shading_normal;
            cur_uv     = hit_g.uv;
        }
    }

debug_done:
    return throughput_s * L;
}

// =====================================================================
// NEE DIRECT LIGHTING (with shadow ray)
// M light samples per hitpoint, averaged.  Coverage-aware CDF selection.
// =====================================================================

// Helper: coverage-aware emitter selection + PDF
__forceinline__ __device__
int dev_nee_select_global(PCGRng& rng, float& p_tri_out) {
    const float c = DEFAULT_NEE_COVERAGE_FRACTION;
    int local_idx;
    if (c > 0.f && rng.next_float() < c) {
        local_idx = (int)(rng.next_float() * (float)params.num_emissive);
        if (local_idx >= params.num_emissive) local_idx = params.num_emissive - 1;
    } else {
        float xi = rng.next_float();
        local_idx = binary_search_cdf(
            params.emissive_cdf, params.num_emissive, xi);
        if (local_idx >= params.num_emissive) local_idx = params.num_emissive - 1;
    }
    float p_power = (local_idx == 0)
        ? params.emissive_cdf[0]
        : params.emissive_cdf[local_idx] - params.emissive_cdf[local_idx - 1];
    float p_uniform = 1.0f / (float)params.num_emissive;
    p_tri_out = (1.0f - c) * p_power + c * p_uniform;
    return local_idx;
}

// Helper: global coverage-aware PDF for a given emissive index
__forceinline__ __device__
float dev_nee_global_pdf(int local_idx) {
    const float c = DEFAULT_NEE_COVERAGE_FRACTION;
    float p_power = (local_idx == 0)
        ? params.emissive_cdf[0]
        : params.emissive_cdf[local_idx] - params.emissive_cdf[local_idx - 1];
    float p_uniform = 1.0f / (float)params.num_emissive;
    return (1.0f - c) * p_power + c * p_uniform;
}

__forceinline__ __device__
NeeResult dev_nee_direct(float3 pos, float3 normal, float3 wo_local,
                         uint32_t mat_id, PCGRng& rng, int bounce,
                         float2 uv)
{
    NeeResult result;
    result.L = Spectrum::zero();
    result.visibility = 0.f;
    if (params.num_emissive <= 0) return result;

    const int M = nee_shadow_sample_count(
        bounce, params.nee_light_samples, params.nee_deep_samples);
    int visible_count = 0;
    DevONB frame = DevONB::from_normal(normal);

    for (int s = 0; s < M; ++s) {
        float p_tri;
        int local_idx = dev_nee_select_global(rng, p_tri);

        NeeSampleResult sr = dev_nee_evaluate_sample(
            local_idx, p_tri, pos, normal, wo_local, mat_id, frame, uv, rng);
        if (sr.visible) visible_count++;
        result.L += sr.L;
    }

    if (M > 1) {
        float inv_M = 1.f / (float)M;
        for (int i = 0; i < NUM_LAMBDA; ++i)
            result.L.value[i] *= inv_M;
    }
    result.visibility = (float)visible_count / (float)M;
    return result;
}

// =====================================================================
// Golden-Ratio Stratified CDF NEE (§7.2.2)
//
// Low-discrepancy shadow ray placement using the golden ratio
// φ = (√5−1)/2 ≈ 0.618.  A random base offset (from RNG) provides
// spatial decorrelation, while successive samples s = 0..M−1 are
// spaced by φ across the power-weighted emissive CDF, guaranteeing
// each shadow ray targets a different region of the distribution.
// Cost: O(M log N_e) per shading point — no precomputation needed.
// =====================================================================

static constexpr float GOLDEN_RATIO_CONJ = 0.6180339887498949f; // (√5−1)/2

__forceinline__ __device__
NeeResult dev_nee_golden_stratified(float3 pos, float3 normal, float3 wo_local,
                                    uint32_t mat_id, PCGRng& rng, int bounce,
                                    float2 uv = make_float2(0.f, 0.f))
{
    NeeResult result;
    result.L = Spectrum::zero();
    result.visibility = 0.f;
    if (params.num_emissive <= 0) return result;

    const int M = nee_shadow_sample_count(
        bounce, params.nee_light_samples, params.nee_deep_samples);
    int visible_count = 0;
    DevONB frame = DevONB::from_normal(normal);

    // Random base offset — Cranley-Patterson rotation for decorrelation
    float base = rng.next_float();

    for (int s = 0; s < M; ++s) {
        // Golden-ratio stratified CDF sample
        float xi = base + s * GOLDEN_RATIO_CONJ;
        xi = xi - floorf(xi);  // fract — wrap to [0,1)

        int local_idx = binary_search_cdf(
            params.emissive_cdf, params.num_emissive, xi);
        if (local_idx >= params.num_emissive) local_idx = params.num_emissive - 1;

        // PDF for this emitter under the power-weighted CDF
        float p_tri = (local_idx == 0)
            ? params.emissive_cdf[0]
            : params.emissive_cdf[local_idx] - params.emissive_cdf[local_idx - 1];

        NeeSampleResult sr = dev_nee_evaluate_sample(
            local_idx, p_tri, pos, normal, wo_local, mat_id, frame, uv, rng);
        if (sr.visible) visible_count++;
        result.L += sr.L;
    }

    if (M > 1) {
        float inv_M = 1.f / (float)M;
        for (int i = 0; i < NUM_LAMBDA; ++i)
            result.L.value[i] *= inv_M;
    }
    result.visibility = (float)visible_count / (float)M;
    return result;
}

// =====================================================================
// dev_nee_dispatch -- route to golden-ratio stratified NEE
// =====================================================================
__forceinline__ __device__
NeeResult dev_nee_dispatch(float3 pos, float3 normal, float3 wo_local,
                           uint32_t mat_id, PCGRng& rng, int bounce,
                           float2 uv)
{
    return dev_nee_golden_stratified(pos, normal, wo_local, mat_id, rng, bounce, uv);
}

// =====================================================================
// FULL PATH TRACING (final render only)
// Hybrid: NEE (direct) + Photon density estimation (indirect)
// Returns: combined, nee_direct, photon_indirect components separately
//
// =====================================================================
// full_path_trace — v2 Architecture (first-hit + specular chain + glossy continuation)
//
// Camera rays are first-hit probes.  Specular surfaces (mirror/glass)
// are followed through a chain of up to DEFAULT_MAX_SPECULAR_CHAIN bounces.
// At the first non-delta hit:
//   1. NEE captures direct illumination
//   2. Photon hash-grid gather captures indirect illumination
//   3. If glossy: BSDF-sample a continuation ray and repeat (up to
//      DEFAULT_MAX_GLOSSY_BOUNCES) — this produces scene reflections
//   4. If diffuse: stop (photon map has the rest)
// Volume integration is disabled in v2 (§Q9).
// =====================================================================
struct PathTraceResult {
    Spectrum combined;
    Spectrum nee_direct;
    Spectrum photon_indirect;
    // Kernel profiling clocks (accumulated across bounces)
    long long clk_ray_trace;
    long long clk_nee;
    long long clk_photon_gather;
    long long clk_bsdf;
};

__forceinline__ __device__
PathTraceResult full_path_trace(float3 origin, float3 direction, PCGRng& rng,
                                int pixel_idx,
                                int sample_index, int total_spp) {
    PathTraceResult result;
    result.combined        = Spectrum::zero();
    result.nee_direct      = Spectrum::zero();
    result.photon_indirect = Spectrum::zero();
    result.clk_ray_trace     = 0;
    result.clk_nee           = 0;
    result.clk_photon_gather = 0;
    result.clk_bsdf          = 0;

    Spectrum throughput = Spectrum::constant(1.0f);

    const int max_spec = DEFAULT_MAX_SPECULAR_CHAIN;

    for (int bounce = 0; bounce <= max_spec; ++bounce) {
        long long t0 = clock64();
        TraceResult hit = trace_radiance(origin, direction);
        result.clk_ray_trace += clock64() - t0;

        if (!hit.hit) break;

        uint32_t mat_id = hit.material_id;

        // Emission: only on first bounce (camera sees a light)
        if (dev_is_emissive(mat_id) && bounce == 0) {
            result.combined  += throughput * dev_get_Le(mat_id, hit.uv);
            result.nee_direct += throughput * dev_get_Le(mat_id, hit.uv);
            break;
        }
        if (dev_is_emissive(mat_id)) break;  // specular chain hit a light

        // Specular bounce: follow the chain
        if (dev_is_specular(mat_id)) {
            SpecularBounceResult sb = dev_specular_bounce(
                direction, hit.position, hit.shading_normal,
                mat_id, hit.uv, rng);
            for (int i = 0; i < NUM_LAMBDA; ++i)
                throughput.value[i] *= sb.filter.value[i];
            direction = sb.new_dir;
            origin    = sb.new_pos;
            continue;
        }

        // Translucent: Fresnel boundary bounce, then continue.
        // Unlike glass, translucent is eligible for NEE and photon gather.
        if (dev_is_translucent(mat_id)) {
            SpecularBounceResult sb = dev_specular_bounce(
                direction, hit.position, hit.shading_normal,
                mat_id, hit.uv, rng);
            for (int i = 0; i < NUM_LAMBDA; ++i)
                throughput.value[i] *= sb.filter.value[i];
            direction = sb.new_dir;
            origin    = sb.new_pos;
            continue;
        }
        DevONB frame = DevONB::from_normal(hit.shading_normal);
        float3 wo_local = frame.world_to_local(direction * (-1.f));
        if (wo_local.z <= 0.f) break;

        // 1. NEE: direct lighting via shadow ray
        if (params.render_mode != RENDER_MODE_INDIRECT_ONLY) {
            long long t_nee = clock64();
            NeeResult nee = dev_nee_dispatch(
                hit.position, hit.shading_normal, wo_local,
                mat_id, rng, bounce, hit.uv);
            result.clk_nee += clock64() - t_nee;

            Spectrum nee_contrib = throughput * nee.L;
            result.combined   += nee_contrib;
            result.nee_direct += nee_contrib;
        }

        // 2. Photon hash-grid gather: indirect lighting
        if (params.render_mode != RENDER_MODE_DIRECT_ONLY) {
            long long t_pg = clock64();
            Spectrum L_photon = dev_estimate_photon_density(
                hit.position, hit.shading_normal, hit.geo_normal,
                wo_local, mat_id, params.gather_radius, hit.uv);
            result.clk_photon_gather += clock64() - t_pg;

            Spectrum photon_contrib = throughput * L_photon;
            result.combined         += photon_contrib;
            result.photon_indirect  += photon_contrib;
        }

        // ── Glossy BSDF continuation (§7.1.1) ──────────────────────
        // If the surface is glossy, sample the BSDF to trace a
        // reflection ray and continue gathering at subsequent hits.
        // This is what produces visible scene reflections on glossy
        // surfaces (not just specular highlights of light sources).
        if (!dev_is_any_glossy(mat_id)) break;  // pure diffuse: stop

        // BSDF continuation loop for glossy surfaces
        for (int g_bounce = 0; g_bounce < DEFAULT_MAX_GLOSSY_BOUNCES; ++g_bounce) {
            // Sample BSDF for the reflection direction
            long long t_bsdf = clock64();
            DevBSDFSample bs = dev_bsdf_sample(mat_id, wo_local, hit.uv, rng, pixel_idx);
            result.clk_bsdf += clock64() - t_bsdf;

            if (bs.pdf < 1e-8f || bs.wi.z <= 0.f) break;

            // Update throughput: f * cos_theta / pdf
            float cos_theta = bs.wi.z;
            for (int i = 0; i < NUM_LAMBDA; ++i)
                throughput.value[i] *= bs.f.value[i] * cos_theta / bs.pdf;

            // Russian roulette on throughput to avoid tracing negligible paths
            float max_tp = throughput.max_component();
            if (max_tp < 0.01f) break;
            if (g_bounce >= 1) {
                float survive = fminf(max_tp, 0.95f);
                if (rng.next_float() > survive) break;
                float inv_survive = 1.f / survive;
                for (int i = 0; i < NUM_LAMBDA; ++i)
                    throughput.value[i] *= inv_survive;
            }

            // Transform sampled direction to world space and trace
            float3 wi_world = frame.local_to_world(bs.wi);
            origin = hit.position + hit.shading_normal * OPTIX_SCENE_EPSILON;

            long long t_rt = clock64();
            hit = trace_radiance(origin, wi_world);
            result.clk_ray_trace += clock64() - t_rt;

            if (!hit.hit) break;

            mat_id = hit.material_id;
            direction = wi_world;

            // If we hit an emissive surface, add its contribution.
            // This stochastic BSDF sample competes with the NEE shadow rays
            // fired at the previous diffuse/glossy hit.  Apply 2-way power-
            // heuristic MIS so both estimators contribute without double
            // counting.  NEE's matching weight is applied in dev_nee_direct().
            //
            // Note: throughput already contains (f * cosθ / pdf_bsdf) from
            // the BSDF sample above, so the raw contribution is
            //   throughput * Le  =  (f cosθ / pdf_bsdf) * Le.
            // The MIS weight scales this down when NEE would have had high
            // probability to reach this same emitter (p_nee >> pdf_bsdf).
            if (dev_is_emissive(mat_id)) {
                Spectrum Le = dev_get_Le(mat_id, hit.uv);
                float w_bsdf = 1.0f;
                if (DEFAULT_USE_MIS) {
                    // p_nee: solid-angle PDF that the NEE strategy would assign
                    // to this direction (power-weighted CDF × area → solid-angle).
                    float p_nee = dev_light_pdf(
                        hit.triangle_id, hit.geo_normal, direction, hit.t);
                    w_bsdf = mis_weight_2_dev(bs.pdf, p_nee);
                }
                result.combined   += throughput * Le * w_bsdf;
                result.nee_direct += throughput * Le * w_bsdf;
                break;
            }

            // If we hit a specular or translucent surface, follow it
            if (dev_is_specular(mat_id) || dev_is_translucent(mat_id)) {
                int spec_remain = max_spec - bounce;
                for (int s = 0; s < spec_remain; ++s) {
                    SpecularBounceResult sb = dev_specular_bounce(
                        direction, hit.position, hit.shading_normal,
                        mat_id, hit.uv, rng);
                    for (int i = 0; i < NUM_LAMBDA; ++i)
                        throughput.value[i] *= sb.filter.value[i];
                    direction = sb.new_dir;
                    origin    = sb.new_pos;

                    long long t_rt2 = clock64();
                    hit = trace_radiance(origin, direction);
                    result.clk_ray_trace += clock64() - t_rt2;
                    if (!hit.hit) goto done;
                    mat_id = hit.material_id;
                    if (dev_is_emissive(mat_id)) {
                        result.combined   += throughput * dev_get_Le(mat_id, hit.uv);
                        result.nee_direct += throughput * dev_get_Le(mat_id, hit.uv);
                        goto done;
                    }
                    if (!dev_is_specular(mat_id) && !dev_is_translucent(mat_id)) break;
                }
            }

            // NEE + photon gather at this bounce's hit
            frame = DevONB::from_normal(hit.shading_normal);
            wo_local = frame.world_to_local(direction * (-1.f));
            if (wo_local.z <= 0.f) break;

            if (params.render_mode != RENDER_MODE_INDIRECT_ONLY) {
                long long t_nee2 = clock64();
                NeeResult nee = dev_nee_dispatch(
                    hit.position, hit.shading_normal, wo_local,
                    mat_id, rng, bounce + g_bounce + 1, hit.uv);
                result.clk_nee += clock64() - t_nee2;

                Spectrum nee_c = throughput * nee.L;
                result.combined   += nee_c;
                result.nee_direct += nee_c;
            }

            if (params.render_mode != RENDER_MODE_DIRECT_ONLY) {
                long long t_pg2 = clock64();
                Spectrum L_ph = dev_estimate_photon_density(
                    hit.position, hit.shading_normal, hit.geo_normal,
                    wo_local, mat_id, params.gather_radius, hit.uv);
                result.clk_photon_gather += clock64() - t_pg2;

                Spectrum ph_c = throughput * L_ph;
                result.combined        += ph_c;
                result.photon_indirect += ph_c;
            }

            // If the continuation hit is not glossy, stop
            if (!dev_is_any_glossy(mat_id)) break;
        }

        break;  // exit outer specular-chain loop
    }

done:
    return result;
}

// ── Forward declarations for SPPM device functions ──────────────────
static __forceinline__ __device__ void sppm_camera_pass(
    int px, int py, int pixel_idx,
    float3 origin, float3 direction, PCGRng& rng);
static __forceinline__ __device__ void sppm_gather_pass(int px, int py, int pixel_idx);

// =====================================================================
// Camera ray helper — shared between SPPM camera dispatch and normal
// render loop.  Includes sub-pixel jitter, optional stratification,
// focus-range jitter, and thin-lens DOF (Shirley concentric disk).
//
// sample_index < 0 → simple random jitter (SPPM / debug).
// sample_index >= 0 → stratified sub-pixel when is_final_render.
// =====================================================================
__forceinline__ __device__
void dev_generate_camera_ray(int px, int py, PCGRng& rng,
                             float3& origin, float3& direction,
                             int sample_index = -1)
{
    // Sub-pixel jitter (optionally stratified)
    float jx, jy;
    if (sample_index >= 0 && params.is_final_render
        && STRATA_X > 1 && STRATA_Y > 1) {
        int stratum_x = sample_index % STRATA_X;
        int stratum_y = (sample_index / STRATA_X) % STRATA_Y;
        jx = ((float)stratum_x + rng.next_float()) / (float)STRATA_X;
        jy = ((float)stratum_y + rng.next_float()) / (float)STRATA_Y;
    } else {
        jx = rng.next_float();
        jy = rng.next_float();
    }

    float u = ((float)px + jx) / (float)params.width;
    float v = ((float)py + jy) / (float)params.height;

    float3 focus_target = params.cam_lower_left
                          + params.cam_horizontal * u
                          + params.cam_vertical * v;

    // Focus-range jitter
    if (params.cam_focus_range > 0.f && params.cam_focus_dist > 0.f) {
        float range_jitter = (rng.next_float() - 0.5f) * params.cam_focus_range;
        float jittered_dist = fmaxf(params.cam_focus_dist + range_jitter, 1e-4f);
        float scale = jittered_dist / params.cam_focus_dist;
        focus_target = params.cam_pos
                     + (focus_target - params.cam_pos) * scale;
    }

    // Thin-lens DOF (Shirley concentric disk)
    if (params.cam_lens_radius > 0.f) {
        float lu1 = rng.next_float();
        float lu2 = rng.next_float();
        float a = 2.f * lu1 - 1.f;
        float b = 2.f * lu2 - 1.f;
        float dr, dphi;
        if (a == 0.f && b == 0.f) { dr = 0.f; dphi = 0.f; }
        else if (a * a > b * b) { dr = a; dphi = (PI / 4.f) * (b / a); }
        else                    { dr = b; dphi = (PI / 2.f) - (PI / 4.f) * (a / b); }
        float dx = dr * cosf(dphi);
        float dy = dr * sinf(dphi);
        float3 lens_offset = (params.cam_u * dx + params.cam_v * dy)
                             * params.cam_lens_radius;
        origin    = params.cam_pos + lens_offset;
        direction = normalize(focus_target - origin);
    } else {
        origin    = params.cam_pos;
        direction = normalize(focus_target - params.cam_pos);
    }
}

// =====================================================================
// __raygen__render
// =====================================================================
extern "C" __global__ void __raygen__render() {
    const uint3 idx = optixGetLaunchIndex();
    int px = idx.x;
    int py = idx.y;
    int pixel_idx = py * params.width + px;

    // ── SPPM mode dispatch ──────────────────────────────────────────
    if (params.sppm_mode == 1) {
        // SPPM camera pass: trace to first diffuse hit, store visible point
        PCGRng rng = PCGRng::seed(
            (uint64_t)pixel_idx * 1000
                + (uint64_t)params.sppm_iteration * 100000,
            (uint64_t)pixel_idx + 1);

        float3 origin, direction;
        dev_generate_camera_ray(px, py, rng, origin, direction);

        // Reset valid flag before camera pass
        params.sppm_vp_valid[pixel_idx] = 0;

        sppm_camera_pass(px, py, pixel_idx, origin, direction, rng);
        return;
    }

    if (params.sppm_mode == 2) {
        // SPPM gather pass: density estimation + progressive update
        sppm_gather_pass(px, py, pixel_idx);
        return;
    }

    // ── Adaptive sampling: skip inactive pixels ──────────────────────
    // active_mask is nullptr when adaptive sampling is disabled.
    if (params.active_mask && params.active_mask[pixel_idx] == 0)
        return;

    // ── NORMAL RENDER PATH ──────────────────────────────────────

    Spectrum L_accum = Spectrum::zero();
    Spectrum L_nee_accum = Spectrum::zero();
    Spectrum L_photon_accum = Spectrum::zero();

    long long prof_total_start = clock64();
    long long prof_ray  = 0, prof_nee  = 0;
    long long prof_pg   = 0, prof_bsdf = 0;

    for (int s = 0; s < params.samples_per_pixel; ++s) {
        PCGRng rng = PCGRng::seed(
            (uint64_t)pixel_idx * 1000
                + (uint64_t)params.frame_number * 100000 + s,
            (uint64_t)pixel_idx + 1);

        // Stratified sub-pixel sampling (when SPP = STRATA_X * STRATA_Y)
        int sample_index = params.frame_number * params.samples_per_pixel + s;

        float3 origin, direction;
        dev_generate_camera_ray(px, py, rng, origin, direction, sample_index);

        if (params.is_final_render || params.debug_shadow_rays) {
            PathTraceResult ptr = full_path_trace(origin, direction, rng, pixel_idx, s, params.samples_per_pixel);
            L_accum        += ptr.combined;
            L_nee_accum    += ptr.nee_direct;
            L_photon_accum += ptr.photon_indirect;
            prof_ray  += ptr.clk_ray_trace;
            prof_nee  += ptr.clk_nee;
            prof_pg   += ptr.clk_photon_gather;
            prof_bsdf += ptr.clk_bsdf;
        } else {
            Spectrum L = debug_first_hit(origin, direction, rng);
            L_accum += L;
        }
    }

    long long prof_total_clk = clock64() - prof_total_start;

    // Progressive accumulation (combined)
    for (int i = 0; i < NUM_LAMBDA; ++i)
        params.spectrum_buffer[pixel_idx * NUM_LAMBDA + i] += L_accum.value[i];
    params.sample_counts[pixel_idx] += (float)params.samples_per_pixel;

    // Adaptive sampling: accumulate luminance moments (if enabled)
    if (params.lum_sum && params.lum_sum2) {
        float Y = 0.f;
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            Y += L_accum.value[i] * CIE_YBAR[i];
        }
        Y *= CIE_YBAR_SUM_INV;
        Y = fmaxf(Y, 0.f);
        params.lum_sum[pixel_idx]  += Y;
        params.lum_sum2[pixel_idx] += Y * Y;
    }

    // Component accumulation (only during final render)
    if (params.is_final_render && params.nee_direct_buffer) {
        for (int i = 0; i < NUM_LAMBDA; ++i)
            params.nee_direct_buffer[pixel_idx * NUM_LAMBDA + i] += L_nee_accum.value[i];
    }
    if (params.is_final_render && params.photon_indirect_buffer) {
        for (int i = 0; i < NUM_LAMBDA; ++i)
            params.photon_indirect_buffer[pixel_idx * NUM_LAMBDA + i] += L_photon_accum.value[i];
    }

    // Profiling accumulation (final render only, if buffers exist)
    if (params.is_final_render && params.prof_total) {
        params.prof_total[pixel_idx]         += prof_total_clk;
        params.prof_ray_trace[pixel_idx]     += prof_ray;
        params.prof_nee[pixel_idx]           += prof_nee;
        params.prof_photon_gather[pixel_idx] += prof_pg;
        params.prof_bsdf[pixel_idx]          += prof_bsdf;
    }

    // Tonemap to sRGB
    float n_samples = params.sample_counts[pixel_idx];
    Spectrum avg;
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        avg.value[i] = (n_samples > 0.f)
            ? params.spectrum_buffer[pixel_idx * NUM_LAMBDA + i] / n_samples
            : 0.f;
    }

    float3 rgb = dev_spectrum_to_srgb(avg);
    rgb.x = fminf(fmaxf(rgb.x, 0.f), 1.f);
    rgb.y = fminf(fmaxf(rgb.y, 0.f), 1.f);
    rgb.z = fminf(fmaxf(rgb.z, 0.f), 1.f);

    params.srgb_buffer[pixel_idx * 4 + 0] = (uint8_t)(rgb.x * 255.f);
    params.srgb_buffer[pixel_idx * 4 + 1] = (uint8_t)(rgb.y * 255.f);
    params.srgb_buffer[pixel_idx * 4 + 2] = (uint8_t)(rgb.z * 255.f);
    params.srgb_buffer[pixel_idx * 4 + 3] = 255;
}

// =====================================================================
// __raygen__photon_trace  -  GPU photon emission and tracing
//   Multi-hero wavelength transport (PBRT v4 style, §1.1 config.h):
//   Each photon carries HERO_WAVELENGTHS bins with stratified offsets.
// =====================================================================
extern "C" __global__ void __raygen__photon_trace() {
    const uint3 idx = optixGetLaunchIndex();
    int photon_idx = idx.x;
    if (photon_idx >= params.num_photons) return;
    if (params.num_emissive <= 0) return;

    // Incorporate photon_map_seed for multi-map re-tracing (§1.2)
    PCGRng rng = PCGRng::seed(
        (uint64_t)photon_idx * 7 + 42 + (uint64_t)params.photon_map_seed * 0x100000007ULL,
        (uint64_t)photon_idx + 1);

    // 1. Sample emissive triangle (mixture: power CDF + uniform)
    const float mix_uniform = fminf(fmaxf(DEFAULT_PHOTON_EMITTER_UNIFORM_MIX, 0.0f), 1.0f);

    int local_idx = 0;
    if (mix_uniform > 0.0f && rng.next_float() < mix_uniform) {
        float u = rng.next_float();
        local_idx = (int)(u * (float)params.num_emissive);
        if (local_idx >= params.num_emissive) local_idx = params.num_emissive - 1;
    } else {
        float xi = rng.next_float();
        local_idx = binary_search_cdf(
            params.emissive_cdf, params.num_emissive, xi);
        if (local_idx >= params.num_emissive) local_idx = params.num_emissive - 1;
    }
    uint32_t tri_idx = params.emissive_tri_indices[local_idx];

    float pdf_power;
    if (local_idx == 0) pdf_power = params.emissive_cdf[0];
    else pdf_power = params.emissive_cdf[local_idx] - params.emissive_cdf[local_idx - 1];
    float pdf_uniform = 1.0f / (float)params.num_emissive;
    float pdf_tri = (1.0f - mix_uniform) * pdf_power + mix_uniform * pdf_uniform;

    // 2. Get triangle geometry
    float3 v0 = params.vertices[tri_idx * 3 + 0];
    float3 v1 = params.vertices[tri_idx * 3 + 1];
    float3 v2 = params.vertices[tri_idx * 3 + 2];
    uint32_t mat_id = params.material_ids[tri_idx];

    // 3. Sample point on triangle
    float3 bary = sample_triangle_dev(rng.next_float(), rng.next_float());
    float3 pos = v0 * bary.x + v1 * bary.y + v2 * bary.z;
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 geo_n = normalize(cross(e1, e2));
    float  area  = length(cross(e1, e2)) * 0.5f;
    float  pdf_pos = 1.f / area;

    // 4. Sample HERO_WAVELENGTHS stratified wavelength bins
    //    Hero = sampled from Le CDF; companions at stratified offsets
    //    (Wilkie et al. 2002 / PBRT v4 style)
    // Interpolate UV at emitter sample point for emission texture
    float2 euv0 = params.texcoords[tri_idx * 3 + 0];
    float2 euv1 = params.texcoords[tri_idx * 3 + 1];
    float2 euv2 = params.texcoords[tri_idx * 3 + 2];
    float2 emit_uv = make_float2(
        euv0.x * bary.x + euv1.x * bary.y + euv2.x * bary.z,
        euv0.y * bary.x + euv1.y * bary.y + euv2.y * bary.z);
    Spectrum Le = dev_get_Le(mat_id, emit_uv);
    float Le_sum = 0.f;
    for (int i = 0; i < NUM_LAMBDA; ++i) Le_sum += Le.value[i];
    if (Le_sum <= 0.f) return;

    // Sample primary hero wavelength from Le CDF
    float xi_lambda = rng.next_float() * Le_sum;
    int hero_bin = NUM_LAMBDA - 1;
    float cum = 0.f;
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        cum += Le.value[i];
        if (xi_lambda <= cum) { hero_bin = i; break; }
    }

    // Build stratified companion bins
    int   hero_bins[HERO_WAVELENGTHS];
    float hero_flux[HERO_WAVELENGTHS];
    int   num_hero = HERO_WAVELENGTHS;

    hero_bins[0] = hero_bin;
    for (int h = 1; h < HERO_WAVELENGTHS; ++h) {
        // Stratified offset: evenly spaced across the spectrum
        int offset = (h * NUM_LAMBDA) / HERO_WAVELENGTHS;
        int companion = (hero_bin + offset) % NUM_LAMBDA;
        hero_bins[h] = companion;
    }

    // 5. Sample cosine-weighted direction within emission cone
    constexpr float cone_half_rad = DEFAULT_LIGHT_CONE_HALF_ANGLE_DEG * (PI / 180.0f);
    const float cos_cone_max = cosf(cone_half_rad);
    float3 local_dir = sample_cosine_cone_dev(rng, cos_cone_max);
    DevONB frame = DevONB::from_normal(geo_n);
    float3 world_dir = frame.local_to_world(local_dir);
    float cos_theta = local_dir.z;
    float cone_denom = PI * (1.0f - cos_cone_max * cos_cone_max);
    float pdf_dir = (cone_denom > 0.f) ? cos_theta / cone_denom : cos_theta * INV_PI;

    // 6. Compute initial flux per hero wavelength
    //    Each hero channel: flux_h = Le(λ_h) * cos / (pdf_tri * pdf_pos * pdf_dir * pdf_lambda_h)
    //    pdf_lambda_h for companion channels uses the same Le CDF probability
    //    as if that bin had been directly sampled: pdf_lambda_h = Le(λ_h) / Le_sum
    //
    //    PBRT v4 §14.3 hero-wavelength normalization: divide by
    //    HERO_WAVELENGTHS because each physical photon contributes to
    //    HERO_WAVELENGTHS spectral bins, but the density estimator
    //    divides by N_emitted (the number of physical photons, not
    //    the number of per-bin contributions).  Without this factor
    //    the indirect component is HERO_WAVELENGTHS× too bright.
    float denom_common = pdf_tri * pdf_pos * pdf_dir;
    float inv_hero = 1.0f / (float)HERO_WAVELENGTHS;
    for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
        int bin = hero_bins[h];
        float Le_h = Le.value[bin];
        float pdf_lambda_h = Le_h / Le_sum;
        hero_flux[h] = (denom_common * pdf_lambda_h > 0.f)
                     ? (Le_h * cos_theta) / (denom_common * pdf_lambda_h) * inv_hero
                     : 0.f;
    }

    // 7. Trace through scene
    float3 origin    = pos + geo_n * OPTIX_SCENE_EPSILON;
    float3 direction = world_dir;
    // Caustic tracking (matches CPU emitter.h convention):
    // Starts false; set true on first specular/translucent hit; reset
    // to false on diffuse/glossy.  Only L→S→D (or L→D→S→D) paths
    // are tagged as caustic.
    bool on_caustic_path = false;

    for (int bounce = 0; bounce < params.photon_max_bounces; ++bounce) {
        TraceResult hit = trace_radiance(origin, direction);
        if (!hit.hit) break;

        // RNG spatial decorrelation
        {
            uint32_t cell_key = dev_hash_cell(
                (int)floorf(hit.position.x / params.grid_cell_size),
                (int)floorf(hit.position.y / params.grid_cell_size),
                (int)floorf(hit.position.z / params.grid_cell_size),
                0x7FFFFFFFu);
            rng.advance(cell_key * 0x9E3779B9u);
        }

        uint32_t hit_mat = hit.material_id;

        // ── Volume photon deposit (Beer–Lambert free-flight) ───────
        if (params.volume_enabled && params.volume_density > 0.f && hit.hit) {
            float seg_t = hit.t;
            float mid_y = origin.y + direction.y * (seg_t * 0.5f);
            HomogeneousMedium med = make_rayleigh_medium(
                params.volume_density, params.volume_albedo,
                params.volume_falloff, mid_y);

            // Use hero bin 0 for volume scattering decision
            float sig_t_lam = med.sigma_t.value[hero_bins[0]];
            if (sig_t_lam > 0.f) {
                float u_ff = rng.next_float();
                float t_ff = -logf(fmaxf(1.f - u_ff, 1e-12f)) / sig_t_lam;

                if (t_ff < seg_t) {
                    float3 vol_pos = origin + direction * t_ff;
                    float sig_s_lam = med.sigma_s.value[hero_bins[0]];
                    float vol_flux = hero_flux[0] * (sig_s_lam / fmaxf(sig_t_lam, 1e-20f));

                    uint32_t vslot = atomicAdd(params.out_vol_photon_count, 1u);
                    if (vslot < (uint32_t)params.max_stored_vol_photons) {
                        params.out_vol_photon_pos_x[vslot]  = vol_pos.x;
                        params.out_vol_photon_pos_y[vslot]  = vol_pos.y;
                        params.out_vol_photon_pos_z[vslot]  = vol_pos.z;
                        params.out_vol_photon_wi_x[vslot]   = -direction.x;
                        params.out_vol_photon_wi_y[vslot]   = -direction.y;
                        params.out_vol_photon_wi_z[vslot]   = -direction.z;
                        params.out_vol_photon_lambda[vslot]  = (uint16_t)hero_bins[0];
                        params.out_vol_photon_flux[vslot]    = vol_flux;
                    }
                }

                // Attenuate all hero flux by transmittance over this segment
                for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
                    float sig_t_h = med.sigma_t.value[hero_bins[h]];
                    hero_flux[h] *= expf(-sig_t_h * seg_t);
                }
            }
        }

        // Skip emissive surfaces
        if (dev_is_emissive(hit_mat)) break;

        // Store photon at diffuse surfaces (skip bounce 0 = direct lighting)
        if (!dev_is_specular(hit_mat) && !dev_is_translucent(hit_mat) && bounce > 0) {
            // In caustic-only mode, skip non-caustic photons
            if (params.caustic_only_store && !on_caustic_path) {
                // Don't store — but still bounce (diffuse/glossy may produce
                // future caustic-path photons after further specular bounces?
                // No — once on_caustic_path is false it stays false,
                // so we can skip the store and continue bouncing normally.)
            } else {
            uint32_t slot = atomicAdd(params.out_photon_count, 1u);
            if (slot < (uint32_t)params.max_stored_photons) {
                params.out_photon_pos_x[slot]   = hit.position.x;
                params.out_photon_pos_y[slot]   = hit.position.y;
                params.out_photon_pos_z[slot]   = hit.position.z;
                params.out_photon_wi_x[slot]    = -direction.x;
                params.out_photon_wi_y[slot]    = -direction.y;
                params.out_photon_wi_z[slot]    = -direction.z;
                params.out_photon_norm_x[slot]  = hit.geo_normal.x;
                params.out_photon_norm_y[slot]  = hit.geo_normal.y;
                params.out_photon_norm_z[slot]  = hit.geo_normal.z;
                // Write HERO_WAVELENGTHS bins per photon (interleaved)
                for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
                    params.out_photon_lambda[slot * HERO_WAVELENGTHS + h] = (uint16_t)hero_bins[h];
                    params.out_photon_flux[slot * HERO_WAVELENGTHS + h]   = hero_flux[h];
                }
                params.out_photon_num_hero[slot] = (uint8_t)num_hero;
                params.out_photon_source_emissive[slot] = (uint16_t)local_idx;
                if (params.out_photon_is_caustic)
                    params.out_photon_is_caustic[slot] = on_caustic_path ? (uint8_t)1 : (uint8_t)0;
                if (params.out_photon_tri_id)
                    params.out_photon_tri_id[slot] = hit.triangle_id;
            }
            }
        }

        // Bounce — track per-hero-channel throughput
        float rr_albedo = 1.0f;
        if (dev_is_specular(hit_mat) || dev_is_translucent(hit_mat)) {
            on_caustic_path = true;  // specular → mark caustic (matches CPU emitter.h)
            SpecularBounceResult sb = dev_specular_bounce(
                direction, hit.position, hit.shading_normal,
                hit_mat, hit.uv, rng, hero_bins, HERO_WAVELENGTHS);
            // Apply transmittance filter to hero channels
            for (int h = 0; h < HERO_WAVELENGTHS; ++h)
                hero_flux[h] *= sb.filter.value[hero_bins[h]];
            direction = sb.new_dir;
            origin    = sb.new_pos;
        } else if (dev_is_any_glossy(hit_mat)) {
            DevONB bounce_frame = DevONB::from_normal(hit.shading_normal);
            float3 wo_local = bounce_frame.world_to_local(-direction);
            if (wo_local.z <= 0.f) break;

            DevBSDFSample bs = dev_bsdf_sample(hit_mat, wo_local, hit.uv, rng);
            if (bs.pdf <= 0.f || bs.wi.z <= 0.f) break;

            float cos_theta_b = bs.wi.z;
            // Per-hero-channel throughput
            float max_throughput = 0.f;
            for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
                float throughput_h = bs.f.value[hero_bins[h]] * cos_theta_b / bs.pdf;
                hero_flux[h] *= throughput_h;
                max_throughput = fmaxf(max_throughput, throughput_h);
            }
            rr_albedo = fminf(max_throughput, 1.0f);

            direction = bounce_frame.local_to_world(bs.wi);
            origin = hit.position + hit.shading_normal * OPTIX_SCENE_EPSILON;
            on_caustic_path = false;
        } else {
            // Diffuse: cosine hemisphere sampling, per-hero throughput
            DevONB bounce_frame = DevONB::from_normal(hit.shading_normal);
            float3 wi_local = sample_cosine_hemisphere_dev(rng);
            Spectrum Kd = dev_get_Kd(hit_mat, hit.uv);
            float max_albedo = 0.f;
            for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
                float albedo_h = Kd.value[hero_bins[h]];
                hero_flux[h] *= albedo_h;
                max_albedo = fmaxf(max_albedo, albedo_h);
            }
            rr_albedo = max_albedo;

            direction = bounce_frame.local_to_world(wi_local);
            origin = hit.position + hit.shading_normal * OPTIX_SCENE_EPSILON;
            on_caustic_path = false;
        }

        // Russian roulette — skip for specular (glass/mirror) bounces so
        // that caustic photons survive the full glass path unattenuated.
        if (bounce >= DEFAULT_MIN_BOUNCES_RR && !dev_is_specular(hit_mat) && !dev_is_translucent(hit_mat)) {
            float p_rr = fminf(DEFAULT_RR_THRESHOLD, rr_albedo);
            if (rng.next_float() >= p_rr) break;
            for (int h = 0; h < HERO_WAVELENGTHS; ++h)
                hero_flux[h] /= p_rr;
        }
    }
}

// =====================================================================
// __raygen__targeted_photon_trace  -  Targeted caustic emission (GPU)
//   Jensen §9.2: importance-sample emission toward specular geometry.
//   1. Pick emitter (power CDF + uniform mix)
//   2. Pick specular triangle (area-weighted alias table)
//   3. Sample point on specular triangle
//   4. Compute direction from light → target, visibility check (shadow ray)
//   5. Compute hero-wavelength flux with PDF correction
//   6. Trace through scene (identical bounce loop to standard photon trace)
// =====================================================================
extern "C" __global__ void __raygen__targeted_photon_trace() {
    const uint3 idx = optixGetLaunchIndex();
    int photon_idx = idx.x;
    if (photon_idx >= params.num_photons) return;
    if (params.num_emissive <= 0) return;
    if (params.num_targeted_spec_tris <= 0) return;

    // Decorrelated RNG (different constant from standard trace)
    PCGRng rng = PCGRng::seed(
        (uint64_t)photon_idx * 13 + 0xCA051CULL + (uint64_t)params.photon_map_seed * 0x100000007ULL,
        (uint64_t)photon_idx + 7);

    // ── 1. Pick emitter triangle (same power CDF + uniform mixture) ──
    const float mix_uniform = fminf(fmaxf(DEFAULT_PHOTON_EMITTER_UNIFORM_MIX, 0.0f), 1.0f);

    int local_idx = 0;
    if (mix_uniform > 0.0f && rng.next_float() < mix_uniform) {
        float u = rng.next_float();
        local_idx = (int)(u * (float)params.num_emissive);
        if (local_idx >= params.num_emissive) local_idx = params.num_emissive - 1;
    } else {
        float xi = rng.next_float();
        local_idx = binary_search_cdf(params.emissive_cdf, params.num_emissive, xi);
        if (local_idx >= params.num_emissive) local_idx = params.num_emissive - 1;
    }
    uint32_t emit_tri = params.emissive_tri_indices[local_idx];

    float pdf_power;
    if (local_idx == 0) pdf_power = params.emissive_cdf[0];
    else pdf_power = params.emissive_cdf[local_idx] - params.emissive_cdf[local_idx - 1];
    float pdf_uniform = 1.0f / (float)params.num_emissive;
    float pdf_emitter = (1.0f - mix_uniform) * pdf_power + mix_uniform * pdf_uniform;

    // Get emitter triangle geometry
    float3 ev0 = params.vertices[emit_tri * 3 + 0];
    float3 ev1 = params.vertices[emit_tri * 3 + 1];
    float3 ev2 = params.vertices[emit_tri * 3 + 2];
    uint32_t emit_mat = params.material_ids[emit_tri];

    // Sample point on emitter
    float3 ebary = sample_triangle_dev(rng.next_float(), rng.next_float());
    float3 light_pos = ev0 * ebary.x + ev1 * ebary.y + ev2 * ebary.z;
    float3 ee1 = ev1 - ev0;
    float3 ee2 = ev2 - ev0;
    float3 light_normal = normalize(cross(ee1, ee2));
    float  light_area   = length(cross(ee1, ee2)) * 0.5f;

    // Get Le
    float2 euv0 = params.texcoords[emit_tri * 3 + 0];
    float2 euv1 = params.texcoords[emit_tri * 3 + 1];
    float2 euv2 = params.texcoords[emit_tri * 3 + 2];
    float2 emit_uv = make_float2(
        euv0.x * ebary.x + euv1.x * ebary.y + euv2.x * ebary.z,
        euv0.y * ebary.x + euv1.y * ebary.y + euv2.y * ebary.z);
    Spectrum Le = dev_get_Le(emit_mat, emit_uv);
    float Le_sum = 0.f;
    for (int i = 0; i < NUM_LAMBDA; ++i) Le_sum += Le.value[i];
    if (Le_sum <= 0.f) return;

    // ── 2. Pick specular triangle (area-weighted alias table) ────────
    float s1 = rng.next_float();
    float s2 = rng.next_float();
    int spec_n = params.num_targeted_spec_tris;
    int spec_local = (int)(s1 * spec_n);
    if (spec_local >= spec_n) spec_local = spec_n - 1;
    if (s2 >= params.targeted_spec_alias_prob[spec_local])
        spec_local = (int)params.targeted_spec_alias_idx[spec_local];
    float pdf_spec_tri = params.targeted_spec_pdf[spec_local];
    float spec_area    = params.targeted_spec_areas[spec_local];
    uint32_t spec_global = params.targeted_spec_tri_indices[spec_local];

    // ── 3. Sample point on specular triangle ─────────────────────────
    float3 sv0 = params.vertices[spec_global * 3 + 0];
    float3 sv1 = params.vertices[spec_global * 3 + 1];
    float3 sv2 = params.vertices[spec_global * 3 + 2];
    float3 sbary = sample_triangle_dev(rng.next_float(), rng.next_float());
    float3 target_pos = sv0 * sbary.x + sv1 * sbary.y + sv2 * sbary.z;
    float3 se1 = sv1 - sv0;
    float3 se2 = sv2 - sv0;
    float3 spec_normal = normalize(cross(se1, se2));
    float  pdf_on_tri  = 1.0f / fmaxf(spec_area, 1e-20f);

    // ── 4. Direction from light → target + visibility ────────────────
    float3 to_target = target_pos - light_pos;
    float  dist_sq   = dot(to_target, to_target);
    float  dist      = sqrtf(dist_sq);
    if (dist < OPTIX_SCENE_EPSILON) return;
    float3 dir = to_target / dist;

    // Light-side cosine: photon must leave from the front
    float cos_light = dot(dir, light_normal);
    if (cos_light <= 0.f) return;

    // Visibility: shadow ray from light → specular target
    float3 shadow_origin = light_pos + light_normal * OPTIX_SCENE_EPSILON;
    if (!trace_shadow(shadow_origin, dir, dist)) return;  // occluded

    // Target-side cosine (for area→solid angle Jacobian)
    float cos_target = fabsf(dot(dir * (-1.f), spec_normal));
    if (cos_target < 1e-6f) return;

    // ── 5. Hero wavelength setup + flux with PDF correction ──────────
    // Sample primary hero wavelength from Le CDF
    float xi_lambda = rng.next_float() * Le_sum;
    int hero_bin = NUM_LAMBDA - 1;
    float cum = 0.f;
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        cum += Le.value[i];
        if (xi_lambda <= cum) { hero_bin = i; break; }
    }

    int   hero_bins[HERO_WAVELENGTHS];
    float hero_flux[HERO_WAVELENGTHS];
    int   num_hero = HERO_WAVELENGTHS;

    hero_bins[0] = hero_bin;
    for (int h = 1; h < HERO_WAVELENGTHS; ++h) {
        int offset = (h * NUM_LAMBDA) / HERO_WAVELENGTHS;
        hero_bins[h] = (hero_bin + offset) % NUM_LAMBDA;
    }

    // PDF of target direction in solid angle (from the light):
    //   pdf_target_sa = pdf_spec_tri * pdf_on_tri * dist² / |cos_target|
    float pdf_target_sa = pdf_spec_tri * pdf_on_tri * dist_sq / cos_target;
    if (pdf_target_sa <= 0.f) return;

    // Flux: Φ(λ_h) = Le(λ_h) * cos_light * light_area / (pdf_emitter * pdf_target_sa * pdf_lambda_h)
    //   The light_area accounts for pdf_pos_on_emitter = 1/light_area (uniform
    //   point sampling on the chosen emitter triangle).  Must match the CPU
    //   formula in specular_target.h: scale = cos_light * A / (pdf_emitter * pdf_target_sa).
    //   HERO_WAVELENGTHS normalisation (PBRT v4 §14.3)
    float denom_common = pdf_emitter * pdf_target_sa;
    float inv_hero = 1.0f / (float)HERO_WAVELENGTHS;
    for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
        int bin = hero_bins[h];
        float Le_h = Le.value[bin];
        float pdf_lambda_h = Le_h / Le_sum;
        hero_flux[h] = (denom_common * pdf_lambda_h > 0.f)
                     ? (Le_h * cos_light * light_area) / (denom_common * pdf_lambda_h) * inv_hero
                     : 0.f;
    }

    // Clamp to prevent fireflies from near-miss geometry
    for (int h = 0; h < HERO_WAVELENGTHS; ++h)
        hero_flux[h] = fminf(hero_flux[h], 1e6f);

    // ── 6. Trace through scene (identical bounce loop) ───────────────
    float3 origin    = shadow_origin;
    float3 direction = dir;
    bool on_caustic_path = false;  // becomes true after first specular hit

    for (int bounce = 0; bounce < params.photon_max_bounces; ++bounce) {
        TraceResult hit = trace_radiance(origin, direction);
        if (!hit.hit) break;

        // RNG spatial decorrelation
        {
            uint32_t cell_key = dev_hash_cell(
                (int)floorf(hit.position.x / params.grid_cell_size),
                (int)floorf(hit.position.y / params.grid_cell_size),
                (int)floorf(hit.position.z / params.grid_cell_size),
                0x7FFFFFFFu);
            rng.advance(cell_key * 0x9E3779B9u);
        }

        uint32_t hit_mat = hit.material_id;

        // Skip emissive surfaces
        if (dev_is_emissive(hit_mat)) break;

        // Store photon at diffuse surfaces (only if on caustic path)
        if (!dev_is_specular(hit_mat) && !dev_is_translucent(hit_mat) && bounce > 0) {
            if (on_caustic_path) {
                uint32_t slot = atomicAdd(params.out_photon_count, 1u);
                if (slot < (uint32_t)params.max_stored_photons) {
                    params.out_photon_pos_x[slot]   = hit.position.x;
                    params.out_photon_pos_y[slot]   = hit.position.y;
                    params.out_photon_pos_z[slot]   = hit.position.z;
                    params.out_photon_wi_x[slot]    = -direction.x;
                    params.out_photon_wi_y[slot]    = -direction.y;
                    params.out_photon_wi_z[slot]    = -direction.z;
                    params.out_photon_norm_x[slot]  = hit.geo_normal.x;
                    params.out_photon_norm_y[slot]  = hit.geo_normal.y;
                    params.out_photon_norm_z[slot]  = hit.geo_normal.z;
                    for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
                        params.out_photon_lambda[slot * HERO_WAVELENGTHS + h] = (uint16_t)hero_bins[h];
                        params.out_photon_flux[slot * HERO_WAVELENGTHS + h]   = hero_flux[h];
                    }
                    params.out_photon_num_hero[slot] = (uint8_t)num_hero;
                    params.out_photon_source_emissive[slot] = (uint16_t)local_idx;
                    if (params.out_photon_is_caustic)
                        params.out_photon_is_caustic[slot] = (uint8_t)1;
                    if (params.out_photon_tri_id)
                        params.out_photon_tri_id[slot] = hit.triangle_id;
                }
            }
            // Caustic path ends at diffuse surface — done
            break;
        }

        // Specular/translucent hit: mark caustic path and bounce
        if (dev_is_specular(hit_mat) || dev_is_translucent(hit_mat)) {
            on_caustic_path = true;
            SpecularBounceResult sb = dev_specular_bounce(
                direction, hit.position, hit.shading_normal,
                hit_mat, hit.uv, rng, hero_bins, HERO_WAVELENGTHS);
            for (int h = 0; h < HERO_WAVELENGTHS; ++h)
                hero_flux[h] *= sb.filter.value[hero_bins[h]];
            direction = sb.new_dir;
            origin    = sb.new_pos;
            continue;  // skip RR for specular bounces
        }

        // Glossy bounce (non-caustic after first diffuse interaction)
        if (dev_is_any_glossy(hit_mat)) {
            DevONB bounce_frame = DevONB::from_normal(hit.shading_normal);
            float3 wo_local = bounce_frame.world_to_local(-direction);
            if (wo_local.z <= 0.f) break;
            DevBSDFSample bs = dev_bsdf_sample(hit_mat, wo_local, hit.uv, rng);
            if (bs.pdf <= 0.f || bs.wi.z <= 0.f) break;
            for (int h = 0; h < HERO_WAVELENGTHS; ++h)
                hero_flux[h] *= bs.f.value[hero_bins[h]] * bs.wi.z / bs.pdf;
            direction = bounce_frame.local_to_world(bs.wi);
            origin = hit.position + hit.shading_normal * OPTIX_SCENE_EPSILON;
            on_caustic_path = false;
        } else {
            // Diffuse bounce
            DevONB bounce_frame = DevONB::from_normal(hit.shading_normal);
            float3 wi_local = sample_cosine_hemisphere_dev(rng);
            Spectrum Kd = dev_get_Kd(hit_mat, hit.uv);
            float rr_albedo = 0.f;
            for (int h = 0; h < HERO_WAVELENGTHS; ++h) {
                float albedo_h = Kd.value[hero_bins[h]];
                hero_flux[h] *= albedo_h;
                rr_albedo = fmaxf(rr_albedo, albedo_h);
            }
            direction = bounce_frame.local_to_world(wi_local);
            origin = hit.position + hit.shading_normal * OPTIX_SCENE_EPSILON;
            on_caustic_path = false;
        }

        // Russian roulette (skip for specular, already handled via continue)
        if (bounce >= DEFAULT_MIN_BOUNCES_RR) {
            float max_flux = 0.f;
            for (int h = 0; h < HERO_WAVELENGTHS; ++h)
                max_flux = fmaxf(max_flux, hero_flux[h]);
            float p_rr = fminf(DEFAULT_RR_THRESHOLD, fminf(max_flux, 1.0f));
            if (rng.next_float() >= p_rr) break;
            for (int h = 0; h < HERO_WAVELENGTHS; ++h)
                hero_flux[h] /= p_rr;
        }
    }
}

// =====================================================================
// SPPM camera pass – traces eye paths to first diffuse hit and stores
// the visible-point data per pixel.  Also evaluates NEE at the visible
// point for the direct-lighting component.
// =====================================================================
static __forceinline__ __device__ void sppm_camera_pass(
    int px, int py, int pixel_idx,
    float3 origin, float3 direction, PCGRng& rng)
{
    Spectrum throughput = Spectrum::constant(1.0f);
    Spectrum L_direct   = Spectrum::zero();

    for (int bounce = 0; bounce <= DEFAULT_MAX_SPECULAR_CHAIN; ++bounce) {
        TraceResult hit = trace_radiance(origin, direction);
        if (!hit.hit) break;

        uint32_t mat_id = hit.material_id;

        // Emission seen directly or via specular chain
        if (dev_is_emissive(mat_id) && bounce == 0) {
            Spectrum Le = dev_get_Le(mat_id, hit.uv);
            L_direct += throughput * Le;
        }

        // Specular (glass/mirror/translucent): bounce through
        if (dev_is_specular(mat_id) || dev_is_translucent(mat_id)) {
            SpecularBounceResult sb = dev_specular_bounce(
                direction, hit.position, hit.shading_normal,
                mat_id, hit.uv, rng);
            for (int i = 0; i < NUM_LAMBDA; ++i)
                throughput.value[i] *= sb.filter.value[i];
            direction = sb.new_dir;
            origin    = sb.new_pos;
            continue;
        }

        // ── Diffuse hit: store visible point ────────────────────
        DevONB frame = DevONB::from_normal(hit.shading_normal);
        float3 wo_local = frame.world_to_local(direction * (-1.f));
        if (wo_local.z <= 0.f) break;

        // Store visible-point data
        params.sppm_vp_pos_x[pixel_idx]  = hit.position.x;
        params.sppm_vp_pos_y[pixel_idx]  = hit.position.y;
        params.sppm_vp_pos_z[pixel_idx]  = hit.position.z;
        params.sppm_vp_norm_x[pixel_idx] = hit.geo_normal.x;
        params.sppm_vp_norm_y[pixel_idx] = hit.geo_normal.y;
        params.sppm_vp_norm_z[pixel_idx] = hit.geo_normal.z;
        params.sppm_vp_wo_x[pixel_idx]   = wo_local.x;
        params.sppm_vp_wo_y[pixel_idx]   = wo_local.y;
        params.sppm_vp_wo_z[pixel_idx]   = wo_local.z;
        params.sppm_vp_mat_id[pixel_idx] = mat_id;
        params.sppm_vp_uv_u[pixel_idx]   = hit.uv.x;
        params.sppm_vp_uv_v[pixel_idx]   = hit.uv.y;
        for (int i = 0; i < NUM_LAMBDA; ++i)
            params.sppm_vp_throughput[pixel_idx * NUM_LAMBDA + i] = throughput.value[i];
        params.sppm_vp_valid[pixel_idx] = 1;

        // ── NEE at visible point (via shared dev_nee_dispatch) ────
        NeeResult sppm_nee = dev_nee_dispatch(
            hit.position, hit.shading_normal, wo_local,
            mat_id, rng, bounce, hit.uv);
        L_direct += throughput * sppm_nee.L;

        break;  // stop at first diffuse hit
    }

    // Accumulate direct lighting into persistent buffer
    for (int i = 0; i < NUM_LAMBDA; ++i)
        params.sppm_L_direct[pixel_idx * NUM_LAMBDA + i] += L_direct.value[i];
}

// =====================================================================
// SPPM gather pass – for each pixel with a valid visible point, query
// the hash grid within the pixel's current radius, accumulate BSDF-
// weighted flux, and perform the progressive radius/flux update.
// =====================================================================
static __forceinline__ __device__ void sppm_gather_pass(int px, int py, int pixel_idx) {
    if (params.sppm_vp_valid[pixel_idx] == 0) return;

    // Read visible-point data
    float3 pos     = make_f3(params.sppm_vp_pos_x[pixel_idx],
                             params.sppm_vp_pos_y[pixel_idx],
                             params.sppm_vp_pos_z[pixel_idx]);
    float3 normal  = make_f3(params.sppm_vp_norm_x[pixel_idx],
                             params.sppm_vp_norm_y[pixel_idx],
                             params.sppm_vp_norm_z[pixel_idx]);
    float3 wo_local = make_f3(params.sppm_vp_wo_x[pixel_idx],
                              params.sppm_vp_wo_y[pixel_idx],
                              params.sppm_vp_wo_z[pixel_idx]);
    uint32_t mat_id = params.sppm_vp_mat_id[pixel_idx];
    float2 uv       = make_f2(params.sppm_vp_uv_u[pixel_idx],
                               params.sppm_vp_uv_v[pixel_idx]);

    float radius = params.sppm_radius[pixel_idx];
    float r2     = radius * radius;

    // Build ONB for BSDF evaluation
    DevONB frame = DevONB::from_normal(normal);

    // Read camera throughput
    Spectrum tp;
    for (int i = 0; i < NUM_LAMBDA; ++i)
        tp.value[i] = params.sppm_vp_throughput[pixel_idx * NUM_LAMBDA + i];

    // Hash grid query — gather photons within radius
    Spectrum phi = Spectrum::zero();
    int M = 0;

    float cell_size = params.grid_cell_size;
    int cx0 = (int)floorf((pos.x - radius) / cell_size);
    int cy0 = (int)floorf((pos.y - radius) / cell_size);
    int cz0 = (int)floorf((pos.z - radius) / cell_size);
    int cx1 = (int)floorf((pos.x + radius) / cell_size);
    int cy1 = (int)floorf((pos.y + radius) / cell_size);
    int cz1 = (int)floorf((pos.z + radius) / cell_size);

    uint32_t visited_keys[64];
    int num_visited = 0;

    for (int iz = cz0; iz <= cz1; ++iz)
    for (int iy = cy0; iy <= cy1; ++iy)
    for (int ix = cx0; ix <= cx1; ++ix) {
        uint32_t key = dev_hash_cell(ix, iy, iz, params.grid_table_size);

        bool already = false;
        for (int v = 0; v < num_visited; ++v)
            if (visited_keys[v] == key) { already = true; break; }
        if (already) continue;
        if (num_visited >= 64) break;  // safety: prevent buffer overflow
        visited_keys[num_visited++] = key;

        uint32_t start = params.grid_cell_start[key];
        uint32_t end   = params.grid_cell_end[key];
        if (start == 0xFFFFFFFF) continue;

        for (uint32_t j = start; j < end; ++j) {
            uint32_t idx = params.grid_sorted_indices[j];
            float3 pp = make_f3(params.photon_pos_x[idx],
                                params.photon_pos_y[idx],
                                params.photon_pos_z[idx]);
            float3 diff = pos - pp;

            // Tangential-disk distance metric (§7.1 guideline)
            float d_plane = dot(diff, normal);
            float3 v_tan = diff - normal * d_plane;
            float d_tan2 = dot(v_tan, v_tan);
            if (d_tan2 > r2) continue;

            // Surface consistency
            float plane_dist = fabsf(d_plane);
            if (plane_dist > DEFAULT_SURFACE_TAU) continue;

            // Normal consistency (§15.1.2: threshold = 0)
            float3 photon_n = make_f3(params.photon_norm_x[idx],
                                      params.photon_norm_y[idx],
                                      params.photon_norm_z[idx]);
            if (dot(photon_n, normal) <= 0.0f) continue;

            // Direction consistency
            float3 wi_world = make_f3(params.photon_wi_x[idx],
                                      params.photon_wi_y[idx],
                                      params.photon_wi_z[idx]);
            if (dot(wi_world, normal) <= 0.f) continue;

            // Diffuse-only BSDF for density estimation (§6 standard practice).
            // Full Cook-Torrance creates 50x+ variance on glossy surfaces.
            float3 wi_local = frame.world_to_local(wi_world);
            Spectrum f = dev_bsdf_evaluate_diffuse(mat_id, wo_local, wi_local, uv);

            // Epanechnikov kernel: w = 1 - d_tan²/r² (smooth falloff)
            float w = 1.0f - d_tan2 / r2;

            // Accumulate HERO_WAVELENGTHS bins per photon (multi-hero transport)
            int n_hero = params.photon_num_hero ? (int)params.photon_num_hero[idx] : 1;
            for (int h = 0; h < n_hero; ++h) {
                uint16_t bin = params.photon_lambda[idx * HERO_WAVELENGTHS + h];
                if (bin < NUM_LAMBDA) {
                    phi.value[bin] += w * f.value[bin] * params.photon_flux[idx * HERO_WAVELENGTHS + h];
                }
            }
            ++M;
        }
    }

    // Apply camera throughput
    for (int i = 0; i < NUM_LAMBDA; ++i)
        phi.value[i] *= tp.value[i];

    // ── Progressive update ──────────────────────────────────────
    if (M > 0) {
        float N_old = params.sppm_N[pixel_idx];
        float N_new = N_old + params.sppm_alpha * (float)M;
        float ratio = N_new / (N_old + (float)M);
        float r_old = radius;
        float r_new = r_old * sqrtf(ratio);
        if (r_new < params.sppm_min_radius) r_new = params.sppm_min_radius;

        float area_ratio = (r_new * r_new) / (r_old * r_old);

        for (int i = 0; i < NUM_LAMBDA; ++i) {
            params.sppm_tau[pixel_idx * NUM_LAMBDA + i] =
                (params.sppm_tau[pixel_idx * NUM_LAMBDA + i] + phi.value[i]) * area_ratio;
        }

        params.sppm_N[pixel_idx]      = N_new;
        params.sppm_radius[pixel_idx] = r_new;
    }

    // ── Reconstruct and tonemap ─────────────────────────────────
    // L_indirect = tau / (A_kernel * k * N_p)
    // With Epanechnikov kernel, A_kernel = pi*r^2/2  (not pi*r^2)
    float r_final = params.sppm_radius[pixel_idx];
    float denom = 0.5f * PI * r_final * r_final
                  * (float)(params.sppm_iteration + 1)
                  * (float)params.sppm_photons_per_iter;
    float inv_denom = (denom > 0.f) ? (1.f / denom) : 0.f;
    float inv_k = 1.f / (float)(params.sppm_iteration + 1);

    Spectrum L;
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        float tau_val = params.sppm_tau[pixel_idx * NUM_LAMBDA + i];
        float direct  = params.sppm_L_direct[pixel_idx * NUM_LAMBDA + i];
        L.value[i] = tau_val * inv_denom + direct * inv_k;
    }

    // Write to spectrum buffer for tonemap
    for (int i = 0; i < NUM_LAMBDA; ++i)
        params.spectrum_buffer[pixel_idx * NUM_LAMBDA + i] = L.value[i];
    params.sample_counts[pixel_idx] = 1.f;

    // Tonemap to sRGB
    float3 rgb = dev_spectrum_to_srgb(L);
    rgb.x = fminf(fmaxf(rgb.x, 0.f), 1.f);
    rgb.y = fminf(fmaxf(rgb.y, 0.f), 1.f);
    rgb.z = fminf(fmaxf(rgb.z, 0.f), 1.f);
    params.srgb_buffer[pixel_idx * 4 + 0] = (uint8_t)(rgb.x * 255.f);
    params.srgb_buffer[pixel_idx * 4 + 1] = (uint8_t)(rgb.y * 255.f);
    params.srgb_buffer[pixel_idx * 4 + 2] = (uint8_t)(rgb.z * 255.f);
    params.srgb_buffer[pixel_idx * 4 + 3] = 255;
}

// =====================================================================
// Stochastic opacity helpers (TEA hash for cheap per-intersection RNG)
// =====================================================================
__forceinline__ __device__
uint32_t tea4(uint32_t v0, uint32_t v1) {
    uint32_t s0 = 0;
    for (int n = 0; n < 4; ++n) {
        s0 += 0x9e3779b9u;
        v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4u);
        v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761eu);
    }
    return v0;
}

// =====================================================================
// __anyhit__radiance  –  stochastic alpha test for translucent surfaces
// =====================================================================
extern "C" __global__ void __anyhit__radiance() {
    if (!params.opacity) return;   // no opacity data uploaded

    const int      prim_idx = optixGetPrimitiveIndex();
    const uint32_t mat_id   = params.material_ids[prim_idx];
    const float    opac     = params.opacity[mat_id];

    if (opac >= 1.f) return;       // fully opaque — accept hit
    if (opac <= 0.f) { optixIgnoreIntersection(); return; } // fully transparent

    // Cheap per-intersection random number via TEA hash
    const uint3  idx  = optixGetLaunchIndex();
    const uint32_t pixel_seed = idx.x + idx.y * 65537u;
    const uint32_t h   = tea4(pixel_seed, prim_idx ^ __float_as_uint(optixGetRayTmax()));
    const float    xi  = (float)(h & 0x00FFFFFFu) / (float)0x01000000u;  // [0,1)

    if (xi >= opac) optixIgnoreIntersection();  // stochastic transparency
}

// =====================================================================
// __anyhit__shadow  –  stochastic alpha test for shadow rays
// =====================================================================
extern "C" __global__ void __anyhit__shadow() {
    if (!params.opacity) return;

    const int      prim_idx = optixGetPrimitiveIndex();
    const uint32_t mat_id   = params.material_ids[prim_idx];
    const float    opac     = params.opacity[mat_id];

    if (opac >= 1.f) return;
    if (opac <= 0.f) { optixIgnoreIntersection(); return; }

    const uint3  idx  = optixGetLaunchIndex();
    const uint32_t pixel_seed = idx.x + idx.y * 65537u;
    const uint32_t h   = tea4(pixel_seed, prim_idx ^ __float_as_uint(optixGetRayTmax()));
    const float    xi  = (float)(h & 0x00FFFFFFu) / (float)0x01000000u;

    if (xi >= opac) optixIgnoreIntersection();
}

// =====================================================================
// __closesthit__radiance
// =====================================================================
extern "C" __global__ void __closesthit__radiance() {
    const int    prim_idx = optixGetPrimitiveIndex();
    const float2 bary     = optixGetTriangleBarycentrics();

    float alpha = 1.f - bary.x - bary.y;
    float beta  = bary.x;
    float gamma = bary.y;

    float3 v0 = params.vertices[prim_idx * 3 + 0];
    float3 v1 = params.vertices[prim_idx * 3 + 1];
    float3 v2 = params.vertices[prim_idx * 3 + 2];
    float3 pos = v0 * alpha + v1 * beta + v2 * gamma;
    float3 geo_normal = normalize(cross(v1 - v0, v2 - v0));

    float3 n0 = params.normals[prim_idx * 3 + 0];
    float3 n1 = params.normals[prim_idx * 3 + 1];
    float3 n2 = params.normals[prim_idx * 3 + 2];
    float3 shading_normal = normalize(n0 * alpha + n1 * beta + n2 * gamma);

    // Interpolate texture coordinates
    float2 uv0 = params.texcoords[prim_idx * 3 + 0];
    float2 uv1 = params.texcoords[prim_idx * 3 + 1];
    float2 uv2 = params.texcoords[prim_idx * 3 + 2];
    float2 uv  = make_f2(
        uv0.x * alpha + uv1.x * beta + uv2.x * gamma,
        uv0.y * alpha + uv1.y * beta + uv2.y * gamma);

    float t = optixGetRayTmax();
    uint32_t mat_id = params.material_ids[prim_idx];

    optixSetPayload_0(__float_as_uint(pos.x));
    optixSetPayload_1(__float_as_uint(pos.y));
    optixSetPayload_2(__float_as_uint(pos.z));
    optixSetPayload_3(__float_as_uint(shading_normal.x));
    optixSetPayload_4(__float_as_uint(shading_normal.y));
    optixSetPayload_5(__float_as_uint(shading_normal.z));
    optixSetPayload_6(__float_as_uint(t));
    optixSetPayload_7(mat_id);
    optixSetPayload_8((uint32_t)prim_idx);
    optixSetPayload_9(1u);
    optixSetPayload_10(__float_as_uint(geo_normal.x));
    optixSetPayload_11(__float_as_uint(geo_normal.y));
    optixSetPayload_12(__float_as_uint(geo_normal.z));
    optixSetPayload_13(__float_as_uint(uv.x));
    optixSetPayload_14(__float_as_uint(uv.y));
}

// =====================================================================
// __closesthit__shadow
// =====================================================================
extern "C" __global__ void __closesthit__shadow() {
    optixSetPayload_0(1u); // 1 = occluded
}

// =====================================================================
// __miss__radiance
// =====================================================================
extern "C" __global__ void __miss__radiance() {
    optixSetPayload_0(0u);
    optixSetPayload_1(0u);
}

// =====================================================================
// __miss__shadow
// =====================================================================
extern "C" __global__ void __miss__shadow() {
    optixSetPayload_0(0u); // 0 = not occluded
}
