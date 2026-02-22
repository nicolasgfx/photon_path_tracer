// ---------------------------------------------------------------------
// optix_device.cu - All OptiX device programs (single compilation unit)
// ---------------------------------------------------------------------
//
// Programs:
//   __raygen__render        - debug first-hit OR full path tracing
//   __raygen__photon_trace  - GPU photon emission + tracing
//   __closesthit__radiance  - closest-hit for radiance rays
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
#include "core/photon_bins.h"
#include "core/cdf.h"
#include "core/nee_sampling.h"
#include "core/guided_nee.h"
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
    DEV_GLOSSY_DIELECTRIC = 5
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

__forceinline__ __device__
bool dev_is_mirror(uint32_t mat_id) {
    return params.mat_type[mat_id] == DEV_MIRROR;
}

__forceinline__ __device__
float dev_get_ior(uint32_t mat_id) {
    return params.ior[mat_id];
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

__forceinline__ __device__
Spectrum dev_get_Le(uint32_t mat_id) {
    Spectrum s;
    for (int i = 0; i < NUM_LAMBDA; ++i)
        s.value[i] = params.Le[mat_id * NUM_LAMBDA + i];
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
    float Y_integral = 0.f;
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        float lam = LAMBDA_MIN + (i + 0.5f) * LAMBDA_STEP;
        float x1 = (lam - 442.0f) * ((lam < 442.0f) ? 0.0624f : 0.0374f);
        float x2 = (lam - 599.8f) * ((lam < 599.8f) ? 0.0264f : 0.0323f);
        float x3 = (lam - 501.1f) * ((lam < 501.1f) ? 0.0490f : 0.0382f);
        float xbar =  0.362f*expf(-0.5f*x1*x1)
                     + 1.056f*expf(-0.5f*x2*x2)
                     - 0.065f*expf(-0.5f*x3*x3);

        float y1 = (lam - 568.8f) * ((lam < 568.8f) ? 0.0213f : 0.0247f);
        float y2 = (lam - 530.9f) * ((lam < 530.9f) ? 0.0613f : 0.0322f);
        float ybar = 0.821f*expf(-0.5f*y1*y1) + 0.286f*expf(-0.5f*y2*y2);

        float z1 = (lam - 437.0f) * ((lam < 437.0f) ? 0.0845f : 0.0278f);
        float z2 = (lam - 459.0f) * ((lam < 459.0f) ? 0.0385f : 0.0725f);
        float zbar = 1.217f*expf(-0.5f*z1*z1) + 0.681f*expf(-0.5f*z2*z2);

        X += s.value[i] * xbar;
        Y += s.value[i] * ybar;
        Z += s.value[i] * zbar;
        Y_integral += ybar;
    }

    // Normalise: divide by sum(ybar) so flat-1.0 -> Y=1
    float scale = (Y_integral > 0.f) ? 1.0f / Y_integral : 0.f;
    X *= scale; Y *= scale; Z *= scale;

    // Apply exposure
    X *= DEFAULT_EXPOSURE; Y *= DEFAULT_EXPOSURE; Z *= DEFAULT_EXPOSURE;

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
bool dev_is_glossy(uint32_t mat_id) {
    return params.mat_type[mat_id] == DEV_GLOSSY;
}

__forceinline__ __device__
bool dev_is_dielectric_glossy(uint32_t mat_id) {
    return params.mat_type[mat_id] == DEV_GLOSSY_DIELECTRIC;
}

// Returns true for any glossy surface (metallic or dielectric).
// Use this for glossy continuation gates; use dev_is_glossy() /
// dev_is_dielectric_glossy() only when differentiating Fresnel model.
__forceinline__ __device__
bool dev_is_any_glossy(uint32_t mat_id) {
    uint8_t t = params.mat_type[mat_id];
    return t == DEV_GLOSSY || t == DEV_GLOSSY_DIELECTRIC;
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
    return Kd * INV_PI;
}

__forceinline__ __device__
Spectrum dev_bsdf_evaluate(uint32_t mat_id, float3 wo, float3 wi, float2 uv) {
    if (wi.z <= 0.f || wo.z <= 0.f) return Spectrum::zero();

    Spectrum Kd = dev_get_Kd(mat_id, uv);

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

// == Dense cell-bin grid O(1) lookup (device-side) ====================
// Returns a pointer to the PHOTON_BIN_COUNT precomputed bins for the
// cell that contains pos, or nullptr if the grid is invalid / pos is
// outside the grid.  Moved here so dev_estimate_photon_density can call it.
__forceinline__ __device__
const PhotonBin* dev_cell_grid_lookup(float3 pos) {
    if (params.cell_grid_valid == 0 || params.cell_bin_grid == nullptr)
        return nullptr;
    float cs = params.cell_grid_cell_size;
    if (cs <= 0.0f) return nullptr;
    int ix = (int)floorf((pos.x - params.cell_grid_min_x) / cs);
    int iy = (int)floorf((pos.y - params.cell_grid_min_y) / cs);
    int iz = (int)floorf((pos.z - params.cell_grid_min_z) / cs);
    // Clamp to grid bounds
    if (ix < 0) ix = 0; else if (ix >= params.cell_grid_dim_x) ix = params.cell_grid_dim_x - 1;
    if (iy < 0) iy = 0; else if (iy >= params.cell_grid_dim_y) iy = params.cell_grid_dim_y - 1;
    if (iz < 0) iz = 0; else if (iz >= params.cell_grid_dim_z) iz = params.cell_grid_dim_z - 1;
    int cell = ix + iy * params.cell_grid_dim_x
             + iz * params.cell_grid_dim_x * params.cell_grid_dim_y;
    return &params.cell_bin_grid[cell * params.photon_bin_count];
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

    // ── Fast path: dense cell-bin gather with trilinear blending ──────
    // Enabled when the cell-bin grid was uploaded and the runtime flag is
    // set (DEFAULT_USE_DENSE_GRID / G key).  The Epanechnikov tangential-
    // disk kernel (§7.1) is baked into bin.flux at CPU build time with
    // neighbour scatter (each photon deposited into up to 27 cells).
    //
    // At query time we blend 8 trilinear-neighbour cells to smooth the
    // cell-boundary discontinuities caused by the kernel weight being
    // baked relative to cell centres rather than the actual query point.
    // Total work: O(8 × PHOTON_BIN_COUNT) — still ~30× faster than the
    // per-photon hash-grid walk.
    if (params.use_dense_grid_gather && params.cell_grid_valid) {
        float cs = params.cell_grid_cell_size;
        if (cs > 0.f && params.cell_bin_grid != nullptr) {
            // Continuous cell coordinates (cell centres at 0.5, 1.5, …)
            float fx = (pos.x - params.cell_grid_min_x) / cs - 0.5f;
            float fy = (pos.y - params.cell_grid_min_y) / cs - 0.5f;
            float fz = (pos.z - params.cell_grid_min_z) / cs - 0.5f;

            int ix0 = (int)floorf(fx);  float tx = fx - (float)ix0;
            int iy0 = (int)floorf(fy);  float ty = fy - (float)iy0;
            int iz0 = (int)floorf(fz);  float tz = fz - (float)iz0;

            const int  N        = params.photon_bin_count;
            const float inv_area = 2.f / (PI * radius * radius);
            const float pnorm    = 1.f / (float)params.num_photons_emitted;
            DevONB frame = DevONB::from_normal(normal);

            for (int corner = 0; corner < 8; ++corner) {
                int ix = (corner & 1) ? ix0 + 1 : ix0;
                int iy = (corner & 2) ? iy0 + 1 : iy0;
                int iz = (corner & 4) ? iz0 + 1 : iz0;

                // Clamp to grid bounds
                if (ix < 0) ix = 0;
                else if (ix >= params.cell_grid_dim_x) ix = params.cell_grid_dim_x - 1;
                if (iy < 0) iy = 0;
                else if (iy >= params.cell_grid_dim_y) iy = params.cell_grid_dim_y - 1;
                if (iz < 0) iz = 0;
                else if (iz >= params.cell_grid_dim_z) iz = params.cell_grid_dim_z - 1;

                float wx = (corner & 1) ? tx : (1.f - tx);
                float wy = (corner & 2) ? ty : (1.f - ty);
                float wz = (corner & 4) ? tz : (1.f - tz);
                float tw = wx * wy * wz;
                if (tw <= 0.f) continue;

                int cell_idx = ix
                    + iy * params.cell_grid_dim_x
                    + iz * params.cell_grid_dim_x * params.cell_grid_dim_y;
                const PhotonBin* bins = &params.cell_bin_grid[cell_idx * N];

                for (int k = 0; k < N; ++k) {
                    const PhotonBin& b = bins[k];
                    if (b.count == 0) continue;

                    // Normal gate (§15.1.2)
                    float3 avg_n = make_f3(b.avg_nx, b.avg_ny, b.avg_nz);
                    if (dot(avg_n, filter_normal) <= 0.f) continue;

                    // Hemisphere gate
                    float3 bin_dir = make_f3(b.dir_x, b.dir_y, b.dir_z);
                    if (dot(bin_dir, filter_normal) <= 0.f) continue;

                    float3 wi_local = frame.world_to_local(bin_dir);
                    if (wi_local.z <= 0.f) continue;

                    // Diffuse-only BSDF evaluation (§6)
                    Spectrum f = dev_bsdf_evaluate_diffuse(mat_id, wo_local, wi_local, uv);

                    // Trilinear-weighted accumulation
                    for (int lam = 0; lam < NUM_LAMBDA; ++lam)
                        L.value[lam] += tw * f.value[lam] * b.flux[lam] * inv_area * pnorm;
                }
            }
            return L;
        }
        // cell_bin_grid invalid: fall through to hash-grid path.
    }

    // ── Exact path: per-photon hash-grid walk O(N_cell) ──────────────
    float cell_size = params.grid_cell_size;
    int cx0 = (int)floorf((pos.x - radius) / cell_size);
    int cy0 = (int)floorf((pos.y - radius) / cell_size);
    int cz0 = (int)floorf((pos.z - radius) / cell_size);
    int cx1 = (int)floorf((pos.x + radius) / cell_size);
    int cy1 = (int)floorf((pos.y + radius) / cell_size);
    int cz1 = (int)floorf((pos.z + radius) / cell_size);

    float r2 = radius * radius;
    float inv_area = 2.f / (PI * r2);  // Epanechnikov 2D normalization
    float norm = 1.f / (float)params.num_photons_emitted; // 1/N_emitted (§5.3)

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
            if (d_tan2 > r2) continue;

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

            // Epanechnikov kernel weight (§6.3 guideline)
            float w = 1.0f - d_tan2 / r2;
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
            // Accumulate HERO_WAVELENGTHS bins per photon (multi-hero transport)
            int n_hero = params.photon_num_hero ? (int)params.photon_num_hero[idx] : 1;
            for (int h = 0; h < n_hero; ++h) {
                float p_flux = params.photon_flux[idx * HERO_WAVELENGTHS + h];
                int bin = (int)params.photon_lambda[idx * HERO_WAVELENGTHS + h];
                if (bin >= 0 && bin < NUM_LAMBDA)
                    L.value[bin] += f.value[bin] * p_flux * w * inv_area * norm;
            }
        }
    }
    return L;
}

// == Cone sampling (uniform direction within cone of half-angle) ======
__forceinline__ __device__
float3 sample_cone_dev(float3 axis, float cos_half_angle, PCGRng& rng) {
    // Sample uniformly within a cone aligned to +Z, then rotate to axis
    float u1 = rng.next_float();
    float u2 = rng.next_float();
    float cos_theta = 1.0f - u1 * (1.0f - cos_half_angle);
    float sin_theta = sqrtf(fmaxf(0.f, 1.0f - cos_theta * cos_theta));
    float phi = TWO_PI * u2;
    float3 local = make_f3(sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta);

    // Build ONB around axis and transform
    DevONB frame = DevONB::from_normal(axis);
    return frame.local_to_world(local);
}

// == Guided BSDF bounce from bin flux CDF ============================
// Returns a direction sampled proportional to bin flux, jittered within
// the selected bin's solid angle.  Falls back to cosine hemisphere if
// bins carry no flux in the positive hemisphere.
//
// Bin selection is depth-dependent to maximise per-SPP convergence:
//
//   bounce_depth == 0  (throughput = 1, dominant variance contribution):
//     SORTED CYCLIC SCHEDULE — sample s visits the s-th most important
//     bin by flux weight w_k = flux_k * cos(theta_n_k), cycling through
//     all active hemisphere bins.  Sample 0 always hits the top bin,
//     sample 1 the second, etc.  After n_active samples every bin has
//     been visited at least once; subsequent cycles repeat in the same
//     order.  This is depth-first convergence: the first diffuse bounce
//     (highest throughput) is coverage-optimised before deeper,
//     attenuated bounces.
//
//   bounce_depth >= 1  (throughput <= albedo^b, already attenuated):
//     STOCHASTIC STRATIFICATION — CDF interval [0,W) is split into S
//     strata; sample s uses stratum (s + b*97) % S, mixing in the bounce
//     depth via prime stride 97 > MAX_BOUNCES to break per-bounce
//     correlation within a single path.
//
// In both cases the flux-proportional PDF p(bin k) = w_k/W is used for
// MIS — unchanged from the baseline so dev_guided_bounce_pdf() needs no
// modification.
__forceinline__ __device__
float3 dev_sample_guided_bounce(
    const PhotonBin* bins, int N, float3 normal,
    const PhotonBinDirs& bin_dirs, PCGRng& rng,
    int sample_index, int total_spp, int bounce_depth)
{
    // Build CDF from bin fluxes — ONLY positive-hemisphere bins
    // (bins whose centroid has dot(dir, normal) > 0)
    float cdf[MAX_PHOTON_BIN_COUNT];
    float total = 0.0f;
    for (int k = 0; k < N; ++k) {
        float3 bdir = make_f3(bins[k].dir_x, bins[k].dir_y, bins[k].dir_z);
        float cos_n = dot(bdir, normal);
        if (cos_n <= 0.0f || bins[k].count == 0) {
            cdf[k] = total;
            continue;
        }
        total += bins[k].scalar_flux * cos_n;
        cdf[k] = total;
    }

    // Fallback to cosine hemisphere if no flux in positive hemisphere
    if (total <= 0.0f) {
        return sample_cosine_hemisphere_dev(rng);
    }

    // ── Bin selection — strategy depends on bounce depth ─────────────
    int selected = N - 1;

    if (bounce_depth == 0) {
        // Bounce-0: sorted cyclic schedule.
        // Build sorted index array by w_k (descending).
        // Insertion sort, O(N²/2) — negligible for N <= 32.
        int sorted_k[MAX_PHOTON_BIN_COUNT];
        int n_sorted = 0;
        for (int k = 0; k < N; ++k) {
            float3 bdir = make_f3(bins[k].dir_x, bins[k].dir_y, bins[k].dir_z);
            float cos_n = dot(bdir, normal);
            if (cos_n <= 0.0f || bins[k].count == 0) continue;
            float wk = bins[k].scalar_flux * cos_n;
            // Find insertion position (descending)
            int pos = n_sorted;
            for (int j = 0; j < n_sorted; ++j) {
                float3 bjd = make_f3(bins[sorted_k[j]].dir_x,
                                     bins[sorted_k[j]].dir_y,
                                     bins[sorted_k[j]].dir_z);
                if (wk > bins[sorted_k[j]].scalar_flux * dot(bjd, normal)) {
                    pos = j; break;
                }
            }
            for (int j = n_sorted; j > pos; --j) sorted_k[j] = sorted_k[j - 1];
            sorted_k[pos] = k;
            n_sorted++;
        }
        if (n_sorted > 0) selected = sorted_k[sample_index % n_sorted];
    } else {
        // Deeper bounces: stochastic stratification with per-bounce rotation.
        float strat_u = (total_spp > 1)
            ? ((float)((sample_index + bounce_depth * 97) % total_spp) + rng.next_float()) / (float)total_spp
            : rng.next_float();
        float xi = strat_u * total;
        for (int k = 0; k < N; ++k) {
            if (xi <= cdf[k]) { selected = k; break; }
        }
    }

    // Jitter within bin solid angle (cone around bin centroid)
    float3 bin_dir = make_f3(bins[selected].dir_x,
                             bins[selected].dir_y,
                             bins[selected].dir_z);
    float len = length(bin_dir);
    if (len < 1e-6f) bin_dir = bin_dirs.dirs[selected]; // fallback to Fibonacci center
    else bin_dir = bin_dir * (1.0f / len);

    // Cone half-angle ≈ arccos(1 - 2/N)
    float cos_half = 1.0f - 2.0f / (float)N;

    // Sample cone direction; resample if it falls below hemisphere
    // (only possible for near-horizon bins where cone straddles tangent plane)
    float3 wi;
    const int MAX_RESAMPLE = 8;
    for (int attempt = 0; attempt < MAX_RESAMPLE; ++attempt) {
        wi = sample_cone_dev(bin_dir, cos_half, rng);
        if (dot(wi, normal) > 0.0f) return wi;
    }

    // All resamples went below hemisphere — use bin centroid directly
    // (it is guaranteed positive-hemisphere by the CDF filter above)
    return bin_dir;
}

// == Guided bounce PDF (solid-angle) =================================
// Matches dev_sample_guided_bounce():
//   - Only positive-hemisphere bins participate (dot(dir, normal) > 0)
//   - Select bin with weight flux * dot(bin_dir, normal)
//   - Sample uniformly in a cone about that bin's centroid
//   - Below-hemisphere jitter is rejected (resampled), not reflected
//
// The PDF is simply: sum over all bins that could produce wi
//   p(wi) = Σ_k [ p_select(k) × p_cone(wi | k) ]
// where p_cone is nonzero only if wi falls within bin k's cone.
__forceinline__ __device__
float dev_guided_bounce_pdf(
    float3 wi,
    const PhotonBin* bins, int N, float3 normal,
    const PhotonBinDirs& bin_dirs)
{
    if (bins == nullptr || N <= 0) return 0.0f;
    if (dot(wi, normal) <= 0.0f) return 0.0f;

    // Cone half-angle used in sampling
    float cos_half = 1.0f - 2.0f / (float)N;
    cos_half = fminf(fmaxf(cos_half, -1.0f), 1.0f);

    // Uniform cone PDF = 1 / (2π(1 - cos_half))
    float cone_omega = TWO_PI * (1.0f - cos_half);
    if (cone_omega <= 1e-12f) return 0.0f;
    float pdf_cone = 1.0f / cone_omega;

    // Build selection weights — only positive-hemisphere bins (same as sampler)
    float total = 0.0f;
    for (int k = 0; k < N; ++k) {
        if (bins[k].count == 0) continue;
        float3 axis = make_f3(bins[k].dir_x, bins[k].dir_y, bins[k].dir_z);
        float len = length(axis);
        axis = (len > 1e-6f) ? axis * (1.0f / len) : bin_dirs.dirs[k];
        float cos_n = dot(axis, normal);
        if (cos_n <= 0.0f) continue;
        total += bins[k].scalar_flux * cos_n;
    }
    if (total <= 0.0f) return 0.0f;

    float pdf = 0.0f;
    for (int k = 0; k < N; ++k) {
        if (bins[k].count == 0) continue;
        float3 axis = make_f3(bins[k].dir_x, bins[k].dir_y, bins[k].dir_z);
        float len = length(axis);
        axis = (len > 1e-6f) ? axis * (1.0f / len) : bin_dirs.dirs[k];
        float cos_n = dot(axis, normal);
        if (cos_n <= 0.0f) continue;

        float w = bins[k].scalar_flux * cos_n;
        float p_sel = w / total;

        // In-cone check: wi falls within this bin's cone
        if (dot(axis, wi) >= cos_half)
            pdf += p_sel * pdf_cone;
    }

    return pdf;
}

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

// NeeResult: returned by dev_nee_direct (defined below)
struct NeeResult {
    Spectrum L;                // direct lighting contribution
    float    visibility;       // fraction of unoccluded shadow samples [0,1]
};

// Forward declaration (used by debug_first_hit when shadow rays are on)
__forceinline__ __device__
NeeResult dev_nee_direct(float3 pos, float3 normal, float3 wo_local,
                         uint32_t mat_id, PCGRng& rng, int bounce = 0,
                         float2 uv = make_float2(0.f, 0.f));

// Forward declaration – dispatches to cached or direct NEE
__forceinline__ __device__
NeeResult dev_nee_dispatch(float3 pos, float3 normal, float3 wo_local,
                           uint32_t mat_id, PCGRng& rng, int bounce = 0,
                           float2 uv = make_float2(0.f, 0.f));

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

    // Emission
    if (dev_is_emissive(mat_id))
        return dev_get_Le(mat_id);

    // Specular: one bounce then direct lighting
    float3 cur_pos = hit.position;
    float3 cur_dir = direction;
    float3 cur_normal = hit.shading_normal;
    uint32_t cur_mat = mat_id;
    float2 cur_uv = hit.uv;
    Spectrum throughput_s = Spectrum::constant(1.0f);

    for (int bounce = 0; bounce < DEFAULT_MAX_SPECULAR_CHAIN; ++bounce) {
        if (!dev_is_specular(cur_mat)) break;
        // Mirror reflection
        cur_dir = cur_dir - cur_normal * (2.f * dot(cur_dir, cur_normal));
        cur_pos = cur_pos + cur_normal * OPTIX_SCENE_EPSILON;
        TraceResult hit2 = trace_radiance(cur_pos, cur_dir);
        if (!hit2.hit) return Spectrum::zero();
        if (dev_is_emissive(hit2.material_id))
            return throughput_s * dev_get_Le(hit2.material_id);
        cur_pos = hit2.position;
        cur_normal = hit2.shading_normal;
        cur_mat = hit2.material_id;
        cur_uv = hit2.uv;
        Spectrum Kd = dev_get_Kd(cur_mat, cur_uv);
        for (int i = 0; i < NUM_LAMBDA; ++i) throughput_s.value[i] *= Kd.value[i];
    }

    // Diffuse/glossy hit: direct lighting via next-event estimation
    Spectrum L = Spectrum::zero();

    if (params.debug_shadow_rays) {
        // NEE PNG path: full dev_nee_direct with M shadow rays + BSDF eval
        float3 wo_local = make_float3(0.f, 0.f, 1.f); // approximate wo in local frame
        DevONB frame = DevONB::from_normal(cur_normal);
        wo_local = frame.world_to_local(normalize(-cur_dir));
        NeeResult nee = dev_nee_dispatch(cur_pos, cur_normal, wo_local, cur_mat, rng, 0, cur_uv);
        L = nee.L;

        // v2: also gather photon density for indirect lighting
        if (params.render_mode != RENDER_MODE_DIRECT_ONLY) {
            Spectrum L_photon = dev_estimate_photon_density(
                    cur_pos, cur_normal, cur_normal, wo_local, cur_mat,
                    params.gather_radius, cur_uv);
            L += L_photon;
        }

        // Glossy BSDF continuation: trace reflections for glossy surfaces
        if (dev_is_any_glossy(cur_mat)) {
            Spectrum glossy_tp = Spectrum::constant(1.0f);
            for (int gb = 0; gb < DEFAULT_MAX_GLOSSY_BOUNCES; ++gb) {
                DevBSDFSample bs = dev_bsdf_sample(cur_mat, wo_local, cur_uv, rng);
                if (bs.pdf < 1e-8f || bs.wi.z <= 0.f) break;

                float cos_theta = bs.wi.z;
                for (int i = 0; i < NUM_LAMBDA; ++i)
                    glossy_tp.value[i] *= bs.f.value[i] * cos_theta / bs.pdf;

                float max_tp = glossy_tp.max_component();
                if (max_tp < 0.01f) break;

                float3 wi_world = frame.local_to_world(bs.wi);
                cur_pos = cur_pos + cur_normal * OPTIX_SCENE_EPSILON;
                TraceResult hit_g = trace_radiance(cur_pos, wi_world);
                if (!hit_g.hit) break;

                cur_mat = hit_g.material_id;
                cur_dir = wi_world;

                if (dev_is_emissive(cur_mat)) {
                    L += glossy_tp * dev_get_Le(cur_mat);
                    break;
                }

                // Follow specular chain if we hit mirror/glass
                if (dev_is_specular(cur_mat)) {
                    for (int s = 0; s < DEFAULT_MAX_SPECULAR_CHAIN; ++s) {
                        float3 n = hit_g.shading_normal;
                        cur_dir = cur_dir - n * (2.f * dot(cur_dir, n));
                        cur_pos = hit_g.position + n * OPTIX_SCENE_EPSILON;
                        hit_g = trace_radiance(cur_pos, cur_dir);
                        if (!hit_g.hit) goto debug_done;
                        cur_mat = hit_g.material_id;
                        if (dev_is_emissive(cur_mat)) {
                            L += glossy_tp * dev_get_Le(cur_mat);
                            goto debug_done;
                        }
                        if (!dev_is_specular(cur_mat)) break;
                    }
                }

                cur_pos    = hit_g.position;
                cur_normal = hit_g.shading_normal;
                cur_uv     = hit_g.uv;

                frame = DevONB::from_normal(cur_normal);
                wo_local = frame.world_to_local(normalize(-cur_dir));
                if (wo_local.z <= 0.f) break;

                NeeResult nee_g = dev_nee_dispatch(cur_pos, cur_normal, wo_local, cur_mat, rng, gb + 1, cur_uv);
                L += glossy_tp * nee_g.L;

                if (params.render_mode != RENDER_MODE_DIRECT_ONLY) {
                    Spectrum L_ph = dev_estimate_photon_density(
                        cur_pos, cur_normal, cur_normal, wo_local, cur_mat,
                        params.gather_radius, cur_uv);
                    L += glossy_tp * L_ph;
                }

                if (!dev_is_any_glossy(cur_mat)) break;
            }
        }
    } else {
        // Real-time debug: fast single-sample, no shadow ray
        // Supports both diffuse and glossy materials with reflection bounces

        // Helper lambda-like: approximate unshadowed direct lighting at a
        // surface point using a single emissive sample and full BSDF eval.
        // We do this inline for each bounce to keep the code self-contained.
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
                    Spectrum Le = dev_get_Le(light_mat);

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
            if (dev_get_roughness(cur_mat) >= 0.1f) break;

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
            float roughness_r = dev_get_roughness(cur_mat);
            float alpha_r = fmaxf(roughness_r * roughness_r, 0.001f);

            // Smith G for both wo and wi (same angle for mirror direction)
            float G_val = dev_ggx_G1(wo_local, alpha_r);
            G_val *= G_val;  // G(wo,wi) ≈ G1(wo)·G1(wi), same angle

            Spectrum Ks_r = dev_get_Ks(cur_mat);
            if (dev_is_dielectric_glossy(cur_mat)) {
                float ior_r = dev_get_ior(cur_mat);
                float F0_r = ((ior_r - 1.f) / (ior_r + 1.f)) * ((ior_r - 1.f) / (ior_r + 1.f));
                float Fr_r = dev_fresnel_schlick(cos_view, F0_r);
                for (int i = 0; i < NUM_LAMBDA; ++i)
                    glossy_tp.value[i] *= G_val * Fr_r * Ks_r.value[i];
            } else {
                // Metallic: per-channel Fresnel
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
                L += glossy_tp * dev_get_Le(cur_mat);
                break;
            }

            // Follow specular chain if we hit a perfect mirror
            if (dev_is_specular(cur_mat)) {
                for (int s = 0; s < DEFAULT_MAX_SPECULAR_CHAIN; ++s) {
                    float3 n = hit_g.shading_normal;
                    cur_dir = cur_dir - n * (2.f * dot(cur_dir, n));
                    cur_pos = hit_g.position + n * OPTIX_SCENE_EPSILON;
                    hit_g = trace_radiance(cur_pos, cur_dir);
                    if (!hit_g.hit) goto debug_done;
                    cur_mat = hit_g.material_id;
                    if (dev_is_emissive(cur_mat)) {
                        L += glossy_tp * dev_get_Le(cur_mat);
                        goto debug_done;
                    }
                    if (!dev_is_specular(cur_mat)) break;
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
// GUIDED NEE (B2) — Bin-flux-weighted light selection
//
// Instead of sampling emissive triangles from the power-based CDF,
// we re-weight each emissive triangle by how much photon flux arrives
// from that direction.  This steers shadow rays toward lights that
// actually contribute indirect illumination at this hitpoint.
//
// For each emissive triangle:
//   weight_i = cdf_weight_i × (1 + ALPHA × bin_flux(direction_to_light))
//
// A temporary CDF is built on the stack (num_emissive entries, capped
// at 128) and sampled.  The modified PDF is used in the estimator
// for correct weighting (importance sampling with a different proposal).
//
// Falls back to dev_nee_direct when:
//   - bins are not valid
//   - num_emissive > 128 (stack budget)
//   - no bin has flux
// =====================================================================
constexpr int   NEE_GUIDED_MAX_EMISSIVE = 128;  // stack-budget for temp CDF
constexpr float NEE_GUIDED_ALPHA        = 5.0f; // flux-boost strength

__forceinline__ __device__
NeeResult dev_nee_guided(float3 pos, float3 normal, float3 wo_local,
                         uint32_t mat_id, PCGRng& rng, int bounce,
                         const PhotonBin* bins, int N,
                         const PhotonBinDirs& bin_dirs,
                         float2 uv = make_float2(0.f, 0.f))
{
    NeeResult result;
    result.L = Spectrum::zero();
    result.visibility = 0.f;
    if (params.num_emissive <= 0) return result;

    // Fall back to standard NEE if too many emissive tris for stack CDF
    if (params.num_emissive > NEE_GUIDED_MAX_EMISSIVE) {
        return dev_nee_direct(pos, normal, wo_local, mat_id, rng, bounce, uv);
    }

    // Compute total bin flux (for early-out if bins are empty)
    float total_bin_flux = 0.0f;
    for (int k = 0; k < N; ++k) total_bin_flux += bins[k].scalar_flux;
    if (total_bin_flux <= 0.0f) {
        return dev_nee_direct(pos, normal, wo_local, mat_id, rng, bounce, uv);
    }

    // Build bin-flux-weighted CDF over emissive triangles
    float guided_cdf[NEE_GUIDED_MAX_EMISSIVE];
    float guided_total = 0.0f;

    for (int i = 0; i < params.num_emissive; ++i) {
        // Original CDF weight for this triangle
        float p_orig;
        if (i == 0)
            p_orig = params.emissive_cdf[0];
        else
            p_orig = params.emissive_cdf[i] - params.emissive_cdf[i - 1];

        // Direction from hitpoint to triangle centroid
        uint32_t tri = params.emissive_tri_indices[i];
        float3 v0 = params.vertices[tri * 3 + 0];
        float3 v1 = params.vertices[tri * 3 + 1];
        float3 v2 = params.vertices[tri * 3 + 2];
        float3 centroid = (v0 + v1 + v2) * (1.0f / 3.0f);
        float3 to_light = centroid - pos;
        float d = length(to_light);

        float bin_boost = 0.0f;
        if (d > 1e-6f) {
            float3 wi = to_light * (1.0f / d);
            bin_boost = guided_nee_bin_boost(
                wi, normal, bins, N, bin_dirs, total_bin_flux);
        }

        float w = guided_nee_weight(p_orig, bin_boost, NEE_GUIDED_ALPHA);
        guided_total += w;
        guided_cdf[i] = guided_total;
    }

    if (guided_total <= 0.0f) {
        return dev_nee_direct(pos, normal, wo_local, mat_id, rng, bounce, uv);
    }

    // Normalize CDF
    float inv_total = 1.0f / guided_total;
    for (int i = 0; i < params.num_emissive; ++i)
        guided_cdf[i] *= inv_total;

    // Sample count (same bounce-dependent logic as standard NEE)
    const int M = nee_shadow_sample_count(
        bounce, params.nee_light_samples, params.nee_deep_samples);
    int visible_count = 0;

    DevONB frame = DevONB::from_normal(normal);

    for (int s = 0; s < M; ++s) {
        // Sample from guided CDF
        float xi = rng.next_float();
        int local_idx = binary_search_cdf(guided_cdf, params.num_emissive, xi);

        uint32_t light_tri = params.emissive_tri_indices[local_idx];

        // Sample point on triangle
        float3 bary = sample_triangle_dev(rng.next_float(), rng.next_float());
        float3 lv0 = params.vertices[light_tri * 3 + 0];
        float3 lv1 = params.vertices[light_tri * 3 + 1];
        float3 lv2 = params.vertices[light_tri * 3 + 2];
        float3 light_pos = lv0 * bary.x + lv1 * bary.y + lv2 * bary.z;

        float3 le1 = lv1 - lv0;
        float3 le2 = lv2 - lv0;
        float3 light_normal = normalize(cross(le1, le2));
        float  light_area   = length(cross(le1, le2)) * 0.5f;

        // Direction and cosines
        float3 to_light = light_pos - pos;
        float dist2 = dot(to_light, to_light);
        float dist  = sqrtf(dist2);
        float3 wi   = to_light * (1.f / dist);

        float cos_x = dot(wi, normal);
        float cos_y = -dot(wi, light_normal);
        if (cos_x <= 0.f || cos_y <= 0.f) continue;

        // Shadow ray
        if (!trace_shadow(pos + normal * OPTIX_SCENE_EPSILON, wi, dist))
            continue;
        visible_count++;

        // PDF from guided distribution (NOT original CDF)
        float p_guided;
        if (local_idx == 0)
            p_guided = guided_cdf[0];
        else
            p_guided = guided_cdf[local_idx] - guided_cdf[local_idx - 1];

        uint32_t light_mat = params.material_ids[light_tri];
        Spectrum Le = dev_get_Le(light_mat);

        float3 wi_local = frame.world_to_local(wi);
        Spectrum f = dev_bsdf_evaluate(mat_id, wo_local, wi_local, uv);

        // PDF conversion: area → solid angle
        float p_y_area = p_guided / light_area;
        float p_wi     = p_y_area * dist2 / cos_y;

        // MIS vs BSDF sampling (guided/cosine mixture)
        float w_mis = 1.0f;
        if (DEFAULT_USE_MIS) {
            float p_guided_bsdf = fminf(fmaxf(DEFAULT_GUIDED_BSDF_MIX, 0.0f), 1.0f);
            // BSDF PDF in local space (glossy-aware)
            float pdf_cos = dev_bsdf_pdf(mat_id, wo_local, wi_local);
            // Guided BSDF PDF in world space (only if bins are valid)
            float pdf_guided = (p_guided_bsdf > 0.0f)
                ? dev_guided_bounce_pdf(wi, bins, N, normal, bin_dirs)
                : 0.0f;
            float pdf_bsdf = p_guided_bsdf * pdf_guided + (1.0f - p_guided_bsdf) * pdf_cos;
            w_mis = mis_weight_2_dev(p_wi, pdf_bsdf);
        }

        // Accumulate MIS-weighted: f * Le * cos_x / p_wi
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            result.L.value[i] += w_mis * f.value[i] * Le.value[i]
                          * cos_x / fmaxf(p_wi, 1e-8f);
        }
    }

    // Average over M samples
    if (M > 1) {
        float inv_M = 1.f / (float)M;
        for (int i = 0; i < NUM_LAMBDA; ++i)
            result.L.value[i] *= inv_M;
    }

    result.visibility = (float)visible_count / (float)M;
    return result;
}

// =====================================================================
// NEE DIRECT LIGHTING (with shadow ray)
// M light samples per hitpoint, averaged.  Reuses the same CDF as
// photon emission so we have ONE light distribution for the whole
// renderer (per the spec).
//
// Returns both the direct lighting spectrum AND the fraction of
// shadow-ray samples that were unoccluded (visibility_fraction).
// The visibility fraction is used to attenuate the photon gather
// contribution at the same hitpoint, preserving contact shadows
// near thin occluders.
// =====================================================================

__forceinline__ __device__
NeeResult dev_nee_direct(float3 pos, float3 normal, float3 wo_local,
                         uint32_t mat_id, PCGRng& rng, int bounce,
                         float2 uv)
{
    NeeResult result;
    result.L = Spectrum::zero();
    result.visibility = 0.f;
    if (params.num_emissive <= 0) return result;

    // Bounce-dependent sample count: full samples at bounce 0,
    // reduced at deeper bounces (throughput already attenuated)
    const int M = nee_shadow_sample_count(
        bounce, params.nee_light_samples, params.nee_deep_samples);
    int visible_count = 0;   // count of unoccluded shadow samples

    // Build shading frame once (reused for every sample)
    DevONB frame = DevONB::from_normal(normal);

    for (int s = 0; s < M; ++s) {
        // Step A — Coverage-aware emitter selection (§7.2.1)
        // Mixture of power-weighted CDF and uniform coverage sampling
        const float c = DEFAULT_NEE_COVERAGE_FRACTION;
        float xi = rng.next_float();
        int local_idx;
        if (c > 0.f && rng.next_float() < c) {
            // Uniform selection (coverage component)
            float u = rng.next_float();
            local_idx = (int)(u * (float)params.num_emissive);
            if (local_idx >= params.num_emissive) local_idx = params.num_emissive - 1;
        } else {
            // Power-weighted CDF selection
            local_idx = binary_search_cdf(
                params.emissive_cdf, params.num_emissive, xi);
            if (local_idx >= params.num_emissive)
                local_idx = params.num_emissive - 1;
        }
        uint32_t light_tri = params.emissive_tri_indices[local_idx];

        // Step B — Sample uniform point on that triangle
        float3 bary = sample_triangle_dev(rng.next_float(), rng.next_float());
        float3 lv0 = params.vertices[light_tri * 3 + 0];
        float3 lv1 = params.vertices[light_tri * 3 + 1];
        float3 lv2 = params.vertices[light_tri * 3 + 2];
        float3 light_pos = lv0 * bary.x + lv1 * bary.y + lv2 * bary.z;

        float3 le1 = lv1 - lv0;
        float3 le2 = lv2 - lv0;
        float3 light_normal = normalize(cross(le1, le2));
        float  light_area   = length(cross(le1, le2)) * 0.5f;

        // Step C — Direction, distance, cosines
        float3 to_light = light_pos - pos;
        float dist2 = dot(to_light, to_light);
        float dist  = sqrtf(dist2);
        float3 wi   = to_light * (1.f / dist);

        float cos_x = dot(wi, normal);          // cosine at shading point
        float cos_y = -dot(wi, light_normal);   // cosine at light surface

        if (cos_x <= 0.f || cos_y <= 0.f) continue; // backfacing

        // Step D — Shadow ray visibility test
        if (!trace_shadow(pos + normal * OPTIX_SCENE_EPSILON, wi, dist))
            continue;

        visible_count++;

        // Step E — Evaluate emission and BSDF
        // Mixture PDF: (1-c)*p_power + c*p_uniform  (§7.2.1)
        float p_power;
        if (local_idx == 0)
            p_power = params.emissive_cdf[0];
        else
            p_power = params.emissive_cdf[local_idx]
                    - params.emissive_cdf[local_idx - 1];
        float p_uniform = 1.0f / (float)params.num_emissive;
        float p_tri = (1.0f - c) * p_power + c * p_uniform;

        uint32_t light_mat = params.material_ids[light_tri];
        Spectrum Le = dev_get_Le(light_mat);

        float3 wi_local = frame.world_to_local(wi);
        Spectrum f = dev_bsdf_evaluate(mat_id, wo_local, wi_local, uv);

        // Step F — PDF conversion: area → solid angle
        // p_y_area  = p_tri * (1 / A_tri)
        // p_wi      = p_y_area * (dist2 / cos_y)
        float p_y_area = p_tri / light_area;
        float p_wi     = p_y_area * dist2 / cos_y;

        // MIS vs BSDF sampling (glossy-aware PDF)
        float w_mis = 1.0f;
        if (DEFAULT_USE_MIS) {
            float pdf_bsdf = dev_bsdf_pdf(mat_id, wo_local, wi_local);
            w_mis = mis_weight_2_dev(p_wi, pdf_bsdf);
        }

        // Step G — Accumulate MIS-weighted: f * Le * cos_x / p_wi
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            result.L.value[i] += w_mis * f.value[i] * Le.value[i]
                          * cos_x / fmaxf(p_wi, 1e-8f);
        }
    }

    // Average over M samples
    if (M > 1) {
        float inv_M = 1.f / (float)M;
		for (int i = 0; i < NUM_LAMBDA; ++i)
			result.L.value[i] *= inv_M;
	}

    result.visibility = (float)visible_count / (float)M;
    return result;
	}

// =====================================================================
// NEE WITH LIGHT IMPORTANCE CACHE (§7.2.2)
//
// Instead of sampling ALL emissive triangles via the power-weighted CDF,
// we use the precomputed per-cell light importance cache to steer shadow
// rays toward lights that actually contribute at this location.
//
// Strategy:
//   With probability (1 - epsilon): sample from cell cache (top-K lights)
//   With probability epsilon:       sample from global power CDF (fallback)
//
// The mixture PDF is computed exactly for correct MIS weighting:
//   p_mix(i) = (1-eps) × p_cache(i) + eps × p_global(i)
//
// Falls back to dev_nee_direct when the cache is empty for a cell.
// =====================================================================

__forceinline__ __device__
NeeResult dev_nee_cached(float3 pos, float3 normal, float3 wo_local,
                         uint32_t mat_id, PCGRng& rng, int bounce,
                         float2 uv = make_float2(0.f, 0.f))
{
    // ── Look up light cache for this cell ────────────────────────────
    int cx = (int)floorf(pos.x / params.light_cache_cell_size);
    int cy = (int)floorf(pos.y / params.light_cache_cell_size);
    int cz = (int)floorf(pos.z / params.light_cache_cell_size);
    uint32_t cache_key = dev_hash_cell(cx, cy, cz, LIGHT_CACHE_TABLE_SIZE);

    int cache_count = params.light_cache_count[cache_key];

    // Fall back to standard NEE if cache is empty for this cell
    if (cache_count <= 0) {
        return dev_nee_direct(pos, normal, wo_local, mat_id, rng, bounce, uv);
    }

    float cache_total = params.light_cache_total_importance[cache_key];
    if (cache_total <= 0.f) {
        return dev_nee_direct(pos, normal, wo_local, mat_id, rng, bounce, uv);
    }

    const CellLightEntry* cache_entries =
        &params.light_cache_entries[cache_key * NEE_CELL_TOP_K];

    // ── Shadow ray sampling loop ─────────────────────────────────────
    NeeResult result;
    result.L = Spectrum::zero();
    result.visibility = 0.f;
    if (params.num_emissive <= 0) return result;

    const int M = nee_shadow_sample_count(
        bounce, params.nee_light_samples, params.nee_deep_samples);
    int visible_count = 0;

    DevONB frame = DevONB::from_normal(normal);
    const float eps = NEE_CACHE_FALLBACK_PROB;
    const float c   = DEFAULT_NEE_COVERAGE_FRACTION;

    for (int s = 0; s < M; ++s) {
        // Strategy selection: cache vs global fallback
        bool use_cache = (rng.next_float() >= eps);
        int local_idx;

        if (use_cache) {
            // Sample from per-cell cache: importance-weighted
            float xi = rng.next_float() * cache_total;
            float cum = 0.f;
            int chosen = cache_count - 1;
            for (int j = 0; j < cache_count; ++j) {
                cum += cache_entries[j].importance;
                if (xi <= cum) { chosen = j; break; }
            }
            local_idx = (int)cache_entries[chosen].emissive_idx;
        } else {
            // Global fallback: coverage-aware mixture (same as dev_nee_direct)
            float xi = rng.next_float();
            if (c > 0.f && rng.next_float() < c) {
                local_idx = (int)(rng.next_float() * (float)params.num_emissive);
                if (local_idx >= params.num_emissive) local_idx = params.num_emissive - 1;
            } else {
                local_idx = binary_search_cdf(
                    params.emissive_cdf, params.num_emissive, xi);
                if (local_idx >= params.num_emissive) local_idx = params.num_emissive - 1;
            }
        }

        uint32_t light_tri = params.emissive_tri_indices[local_idx];

        // Sample point on triangle
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
        if (cos_x <= 0.f || cos_y <= 0.f) continue;

        // Shadow ray
        if (!trace_shadow(pos + normal * OPTIX_SCENE_EPSILON, wi, dist))
            continue;
        visible_count++;

        // ── Compute mixture PDF ──────────────────────────────────────
        // p_cache: probability of selecting this light from the cache
        float p_cache = 0.f;
        for (int j = 0; j < cache_count; ++j) {
            if (cache_entries[j].emissive_idx == (uint16_t)local_idx) {
                p_cache = cache_entries[j].importance / cache_total;
                break;
            }
        }

        // p_global: coverage-aware mixture (power + uniform)
        float p_power;
        if (local_idx == 0) p_power = params.emissive_cdf[0];
        else p_power = params.emissive_cdf[local_idx]
                     - params.emissive_cdf[local_idx - 1];
        float p_uniform = 1.f / (float)params.num_emissive;
        float p_global = (1.f - c) * p_power + c * p_uniform;

        // Combined mixture: (1-eps)*cache + eps*global
        float p_tri = (1.f - eps) * p_cache + eps * p_global;
        if (p_tri <= 0.f) continue;  // shouldn't happen, but guard

        // Emission and BSDF
        uint32_t light_mat = params.material_ids[light_tri];
        Spectrum Le = dev_get_Le(light_mat);

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

        // Accumulate MIS-weighted: f * Le * cos_x / p_wi
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            result.L.value[i] += w_mis * f.value[i] * Le.value[i]
                          * cos_x / fmaxf(p_wi, 1e-8f);
        }
    }

    // Average over M samples
    if (M > 1) {
        float inv_M = 1.f / (float)M;
        for (int i = 0; i < NUM_LAMBDA; ++i)
            result.L.value[i] *= inv_M;
    }

    result.visibility = (float)visible_count / (float)M;
    return result;
}

// =====================================================================
// dev_nee_dispatch -- route to cached or direct NEE
// =====================================================================
__forceinline__ __device__
NeeResult dev_nee_dispatch(float3 pos, float3 normal, float3 wo_local,
                           uint32_t mat_id, PCGRng& rng, int bounce,
                           float2 uv)
{
    if (params.use_light_cache && params.light_cache_valid)
        return dev_nee_cached(pos, normal, wo_local, mat_id, rng, bounce, uv);
    return dev_nee_direct(pos, normal, wo_local, mat_id, rng, bounce, uv);
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
            result.combined  += throughput * dev_get_Le(mat_id);
            result.nee_direct += throughput * dev_get_Le(mat_id);
            break;
        }
        if (dev_is_emissive(mat_id)) break;  // specular chain hit a light

        // Specular bounce: follow the chain
        if (dev_is_specular(mat_id)) {
            // Glass (Fresnel dielectric): reflect or refract via Fresnel
            if (dev_is_glass(mat_id)) {
                float eta = dev_get_ior(mat_id);
                float3 n = hit.shading_normal;
                bool entering = dot(direction, n) < 0.f;
                float3 outward_normal = entering ? n : n * (-1.f);
                float ni_over_nt = entering ? (1.f / eta) : eta;

                float cos_i = fabsf(dot(direction, outward_normal));
                float sin2_t = ni_over_nt * ni_over_nt * (1.f - cos_i * cos_i);
                float fresnel = 1.f;
                if (sin2_t < 1.f) {
                    float cos_t = sqrtf(1.f - sin2_t);
                    float rs = (ni_over_nt * cos_i - cos_t) / (ni_over_nt * cos_i + cos_t);
                    float rp = (cos_i - ni_over_nt * cos_t) / (cos_i + ni_over_nt * cos_t);
                    fresnel = 0.5f * (rs * rs + rp * rp);
                }

                Spectrum Kd = dev_get_Kd(mat_id, hit.uv);
                for (int i = 0; i < NUM_LAMBDA; ++i)
                    throughput.value[i] *= Kd.value[i];

                if (rng.next_float() < fresnel) {
                    direction = direction - outward_normal * (2.f * dot(direction, outward_normal));
                    origin = hit.position + outward_normal * OPTIX_SCENE_EPSILON;
                } else {
                    float3 refracted = direction * ni_over_nt +
                        outward_normal * (ni_over_nt * cos_i - sqrtf(fmaxf(0.f, 1.f - sin2_t)));
                    direction = normalize(refracted);
                    origin = hit.position - outward_normal * OPTIX_SCENE_EPSILON;
                }
            } else {
                // Mirror: pure reflection
                float3 n = hit.shading_normal;
                direction = direction - n * (2.f * dot(direction, n));
                origin = hit.position + n * OPTIX_SCENE_EPSILON;
            }
            continue;
        }

        // ── Non-specular hit: NEE + photon gather ────────────────────
        // For glossy materials, continue bouncing via BSDF sampling
        // to capture scene reflections (polished tables, metals, etc.).
        // Pure diffuse surfaces terminate after this gather.
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
                Spectrum Le = dev_get_Le(mat_id);
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

            // If we hit a specular surface, follow it through the
            // remaining specular chain budget, then gather at diffuse
            if (dev_is_specular(mat_id)) {
                int spec_remain = max_spec - bounce;
                for (int s = 0; s < spec_remain; ++s) {
                    if (dev_is_glass(mat_id)) {
                        float eta = dev_get_ior(mat_id);
                        float3 n = hit.shading_normal;
                        bool entering = dot(direction, n) < 0.f;
                        float3 outward_n = entering ? n : n * (-1.f);
                        float ni_nt = entering ? (1.f / eta) : eta;
                        float cos_in = fabsf(dot(direction, outward_n));
                        float sin2_t = ni_nt * ni_nt * (1.f - cos_in * cos_in);
                        float fresnel = 1.f;
                        if (sin2_t < 1.f) {
                            float cos_t = sqrtf(1.f - sin2_t);
                            float rs = (ni_nt * cos_in - cos_t) / (ni_nt * cos_in + cos_t);
                            float rp = (cos_in - ni_nt * cos_t) / (cos_in + ni_nt * cos_t);
                            fresnel = 0.5f * (rs * rs + rp * rp);
                        }
                        Spectrum Kd_g = dev_get_Kd(mat_id, hit.uv);
                        for (int i = 0; i < NUM_LAMBDA; ++i)
                            throughput.value[i] *= Kd_g.value[i];
                        if (rng.next_float() < fresnel) {
                            direction = direction - outward_n * (2.f * dot(direction, outward_n));
                            origin = hit.position + outward_n * OPTIX_SCENE_EPSILON;
                        } else {
                            float3 refr = direction * ni_nt +
                                outward_n * (ni_nt * cos_in - sqrtf(fmaxf(0.f, 1.f - sin2_t)));
                            direction = normalize(refr);
                            origin = hit.position - outward_n * OPTIX_SCENE_EPSILON;
                        }
                    } else {
                        float3 n = hit.shading_normal;
                        direction = direction - n * (2.f * dot(direction, n));
                        origin = hit.position + n * OPTIX_SCENE_EPSILON;
                    }
                    long long t_rt2 = clock64();
                    hit = trace_radiance(origin, direction);
                    result.clk_ray_trace += clock64() - t_rt2;
                    if (!hit.hit) goto done;
                    mat_id = hit.material_id;
                    if (dev_is_emissive(mat_id)) {
                        result.combined   += throughput * dev_get_Le(mat_id);
                        result.nee_direct += throughput * dev_get_Le(mat_id);
                        goto done;
                    }
                    if (!dev_is_specular(mat_id)) break;  // fell through to diffuse/glossy
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

        float jx = rng.next_float();
        float jy = rng.next_float();
        float u = ((float)px + jx) / (float)params.width;
        float v = ((float)py + jy) / (float)params.height;
        float3 focus_target = params.cam_lower_left
                              + params.cam_horizontal * u
                              + params.cam_vertical * v;
        float3 origin    = params.cam_pos;
        float3 direction = normalize(focus_target - params.cam_pos);

        if (params.cam_lens_radius > 0.f) {
            float lu1 = rng.next_float();
            float lu2 = rng.next_float();
            float a = 2.f * lu1 - 1.f, b = 2.f * lu2 - 1.f;
            float dr, dphi;
            if (a == 0.f && b == 0.f) { dr = 0.f; dphi = 0.f; }
            else if (a * a > b * b)   { dr = a; dphi = (PI / 4.f) * (b / a); }
            else                      { dr = b; dphi = (PI / 2.f) - (PI / 4.f) * (a / b); }
            float3 lens_offset = (params.cam_u * dr * cosf(dphi) + params.cam_v * dr * sinf(dphi))
                                 * params.cam_lens_radius;
            origin    = params.cam_pos + lens_offset;
            direction = normalize(focus_target - origin);
        }

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
        float jx, jy;
        int sample_index = params.frame_number * params.samples_per_pixel + s;
        if (params.is_final_render && STRATA_X > 1 && STRATA_Y > 1) {
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

        // Focus-plane target for this pixel sample
        float3 focus_target =
            params.cam_lower_left
            + params.cam_horizontal * u
            + params.cam_vertical * v;

        // Focus-range jitter: widen the razor-thin focus plane into a slab.
        // Per-sample we shift the focus target along the pinhole ray by a
        // random offset within [-range/2, +range/2], so objects inside the
        // slab stay acceptably sharp while distant objects still blur.
        if (params.cam_focus_range > 0.f && params.cam_focus_dist > 0.f) {
            float range_jitter = (rng.next_float() - 0.5f) * params.cam_focus_range;
            float jittered_dist = fmaxf(params.cam_focus_dist + range_jitter, 1e-4f);
            float scale = jittered_dist / params.cam_focus_dist;
            // Rescale the offset from camera to keep the target on the
            // jittered focus plane along the same viewing direction.
            focus_target = params.cam_pos
                         + (focus_target - params.cam_pos) * scale;
        }

        float3 origin;
        float3 direction;
        if (params.cam_lens_radius > 0.f) {
            // Thin-lens DOF: sample circular aperture
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
            // Pinhole
            origin    = params.cam_pos;
            direction = normalize(focus_target - params.cam_pos);
        }

        if (params.is_final_render) {
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
        // Choose which radiance component to measure variance on.
        // ADAPTIVE_NOISE_USE_DIRECT_ONLY=true  → L_nee_accum (explicit shadow rays,
        //   orders-of-magnitude lower per-sample variance; converges quickly).
        // ADAPTIVE_NOISE_USE_DIRECT_ONLY=false → L_accum (full multi-bounce path;
        //   variance is enormous and the noise metric may never converge).
        const Spectrum& lum_proxy = ADAPTIVE_NOISE_USE_DIRECT_ONLY
            ? L_nee_accum : L_accum;

        float Y = 0.f, Y_integral = 0.f;
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            float lam = LAMBDA_MIN + (i + 0.5f) * LAMBDA_STEP;
            float y1  = (lam - 568.8f) * ((lam < 568.8f) ? 0.0213f : 0.0247f);
            float y2  = (lam - 530.9f) * ((lam < 530.9f) ? 0.0613f : 0.0322f);
            float ybar = 0.821f * expf(-0.5f * y1 * y1)
                       + 0.286f * expf(-0.5f * y2 * y2);
            Y          += lum_proxy.value[i] * ybar;
            Y_integral += ybar;
        }
        if (Y_integral > 0.f) Y /= Y_integral;
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
    Spectrum Le = dev_get_Le(mat_id);
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
        if (!dev_is_specular(hit_mat) && bounce > 0) {
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
            }
        }

        // Bounce — track per-hero-channel throughput
        float rr_albedo = 1.0f;
        if (dev_is_specular(hit_mat)) {
            if (dev_is_glass(hit_mat)) {
                float eta = dev_get_ior(hit_mat);
                float3 n = hit.shading_normal;
                bool entering = dot(direction, n) < 0.f;
                float3 outward_normal = entering ? n : n * (-1.f);
                float ni_over_nt = entering ? (1.f / eta) : eta;

                float cos_i = fabsf(dot(direction, outward_normal));
                float sin2_t = ni_over_nt * ni_over_nt * (1.f - cos_i * cos_i);
                float fresnel = 1.f;
                if (sin2_t < 1.f) {
                    float cos_t = sqrtf(1.f - sin2_t);
                    float rs = (ni_over_nt * cos_i - cos_t) / (ni_over_nt * cos_i + cos_t);
                    float rp = (cos_i - ni_over_nt * cos_t) / (cos_i + ni_over_nt * cos_t);
                    fresnel = 0.5f * (rs * rs + rp * rp);
                }

                if (rng.next_float() < fresnel) {
                    direction = direction - outward_normal * (2.f * dot(direction, outward_normal));
                    origin = hit.position + outward_normal * OPTIX_SCENE_EPSILON;
                } else {
                    float3 refracted = direction * ni_over_nt +
                        outward_normal * (ni_over_nt * cos_i - sqrtf(fmaxf(0.f, 1.f - sin2_t)));
                    direction = normalize(refracted);
                    origin = hit.position - outward_normal * OPTIX_SCENE_EPSILON;
                }
                // Glass: throughput unchanged (geometric-only), shared path
            } else {
                // Mirror reflection: throughput unchanged
                float3 n = hit.shading_normal;
                direction = direction - n * (2.f * dot(direction, n));
                origin = hit.position + n * OPTIX_SCENE_EPSILON;
            }
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
        }

        // Russian roulette
        if (bounce >= DEFAULT_MIN_BOUNCES_RR) {
            float p_rr = fminf(DEFAULT_RR_THRESHOLD, rr_albedo);
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
            Spectrum Le = dev_get_Le(mat_id);
            L_direct += throughput * Le;
        }

        // Glass (Fresnel dielectric): bounce through via Fresnel routing
        if (dev_is_glass(mat_id)) {
            float eta = dev_get_ior(mat_id);
            float3 n = hit.shading_normal;
            bool entering = dot(direction, n) < 0.f;
            float3 outward_normal = entering ? n : n * (-1.f);
            float ni_over_nt = entering ? (1.f / eta) : eta;

            float cos_i = fabsf(dot(direction, outward_normal));
            float sin2_t = ni_over_nt * ni_over_nt * (1.f - cos_i * cos_i);
            float fresnel = 1.f;
            if (sin2_t < 1.f) {
                float cos_t = sqrtf(1.f - sin2_t);
                float rs = (ni_over_nt * cos_i - cos_t) / (ni_over_nt * cos_i + cos_t);
                float rp = (cos_i - ni_over_nt * cos_t) / (cos_i + ni_over_nt * cos_t);
                fresnel = 0.5f * (rs * rs + rp * rp);
            }

            if (rng.next_float() < fresnel) {
                direction = direction - outward_normal * (2.f * dot(direction, outward_normal));
                origin = hit.position + outward_normal * OPTIX_SCENE_EPSILON;
            } else {
                float3 refracted = direction * ni_over_nt +
                    outward_normal * (ni_over_nt * cos_i - sqrtf(fmaxf(0.f, 1.f - sin2_t)));
                direction = normalize(refracted);
                origin = hit.position - outward_normal * OPTIX_SCENE_EPSILON;
            }
            continue;
        }

        // Mirror: pure reflection
        if (dev_is_mirror(mat_id)) {
            float3 n = hit.shading_normal;
            direction = direction - n * (2.f * dot(direction, n));
            origin = hit.position + n * OPTIX_SCENE_EPSILON;
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

        // ── NEE at visible point ────────────────────────────────
        // Simple NEE: one shadow ray to a light
        int n_nee = (bounce == 0) ? params.nee_light_samples : params.nee_deep_samples;
        for (int ns = 0; ns < n_nee; ++ns) {
            float xi = rng.next_float();
            int emissive_idx = binary_search_cdf(
                params.emissive_cdf, params.num_emissive, xi);
            if (emissive_idx >= params.num_emissive)
                emissive_idx = params.num_emissive - 1;

            uint32_t tri_id = params.emissive_tri_indices[emissive_idx];
            float u1 = rng.next_float();
            float u2 = rng.next_float();
            float su = sqrtf(u1);
            float bary_a = 1.f - su;
            float bary_b = u2 * su;
            float bary_c = 1.f - bary_a - bary_b;

            float3 lv0 = params.vertices[tri_id * 3 + 0];
            float3 lv1 = params.vertices[tri_id * 3 + 1];
            float3 lv2 = params.vertices[tri_id * 3 + 2];
            float3 light_pos = lv0 * bary_a + lv1 * bary_b + lv2 * bary_c;
            float3 light_normal = normalize(cross(lv1 - lv0, lv2 - lv0));

            float3 to_light = light_pos - hit.position;
            float dist2 = dot(to_light, to_light);
            float dist = sqrtf(dist2);
            float3 wi = to_light * (1.f / dist);

            float cos_recv = dot(wi, hit.shading_normal);
            if (cos_recv <= 0.f) continue;

            float cos_emit = -dot(wi, light_normal);
            if (cos_emit <= 0.f) continue;

            // Shadow test
            if (!trace_shadow(hit.position + hit.shading_normal * OPTIX_SCENE_EPSILON,
                              wi, dist - 2.f * OPTIX_SCENE_EPSILON))
                continue;

            // Triangle area
            float3 e1 = lv1 - lv0, e2 = lv2 - lv0;
            float area = 0.5f * length(cross(e1, e2));
            if (area <= 0.f) continue;

            float pdf_light = dist2 / (cos_emit * area * params.num_emissive);
            if (pdf_light <= 0.f) continue;

            Spectrum Le = dev_get_Le(params.material_ids[tri_id]);
            float3 wi_local = frame.world_to_local(wi);
            Spectrum f_bsdf = dev_bsdf_evaluate(mat_id, wo_local, wi_local, hit.uv);

            for (int i = 0; i < NUM_LAMBDA; ++i) {
                float contrib = throughput.value[i] * Le.value[i] *
                                f_bsdf.value[i] * cos_recv /
                                (pdf_light * (float)n_nee);
                L_direct.value[i] += contrib;
            }
        }

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
