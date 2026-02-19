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
#include "optix/launch_params.h"
#include "core/random.h"
#include "core/spectrum.h"
#include "core/config.h"

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
    unsigned int p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13;
    p0=p1=p2=p3=p4=p5=p6=p7=p8=p9=p10=p11=p12=p13=0;

    optixTrace(
        params.traversable,
        origin, direction,
        tmin, tmax, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,
        p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13);

    TraceResult r;
    r.position       = make_f3(u2f(p0), u2f(p1), u2f(p2));
    r.shading_normal = make_f3(u2f(p3), u2f(p4), u2f(p5));
    r.t              = u2f(p6);
    r.material_id    = p7;
    r.triangle_id    = p8;
    r.hit            = (p9 != 0);
    r.geo_normal     = make_f3(u2f(p10), u2f(p11), u2f(p12));
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
    DEV_EMISSIVE   = 4
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
Spectrum dev_get_Kd(uint32_t mat_id) {
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

    float r =  3.2406f*X - 1.5372f*Y - 0.4986f*Z;
    float g = -0.9689f*X + 1.8758f*Y + 0.0415f*Z;
    float b =  0.0557f*X - 0.2040f*Y + 1.0570f*Z;

    auto gamma = [](float c) -> float {
        c = fmaxf(c, 0.f);
        return (c <= 0.0031308f) ? 12.92f*c : 1.055f*powf(c, 1.f/2.4f) - 0.055f;
    };

    return make_f3(gamma(r), gamma(g), gamma(b));
}

// == BSDF evaluate / pdf (Lambertian only) ============================
__forceinline__ __device__
Spectrum dev_bsdf_evaluate(uint32_t mat_id, float3 /*wo*/, float3 wi) {
    if (wi.z <= 0.f) return Spectrum::zero();
    return dev_get_Kd(mat_id) * INV_PI;
}

__forceinline__ __device__
float dev_bsdf_pdf(float3 wi) {
    return fmaxf(0.f, wi.z) * INV_PI;
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
    float3 pos, float3 normal, float3 wo_local, uint32_t mat_id,
    float radius)
{
    Spectrum L = Spectrum::zero();
    if (params.num_photons == 0 || params.grid_table_size == 0) return L;

    float cell_size = params.grid_cell_size;
    int cx0 = (int)floorf((pos.x - radius) / cell_size);
    int cy0 = (int)floorf((pos.y - radius) / cell_size);
    int cz0 = (int)floorf((pos.z - radius) / cell_size);
    int cx1 = (int)floorf((pos.x + radius) / cell_size);
    int cy1 = (int)floorf((pos.y + radius) / cell_size);
    int cz1 = (int)floorf((pos.z + radius) / cell_size);

    float r2 = radius * radius;
    float inv_area = 1.f / (PI * r2);
    int count = 0;

    for (int iz = cz0; iz <= cz1; ++iz)
    for (int iy = cy0; iy <= cy1; ++iy)
    for (int ix = cx0; ix <= cx1; ++ix) {
        uint32_t key = dev_hash_cell(ix, iy, iz, params.grid_table_size);
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
            float d2 = dot(diff, diff);
            if (d2 > r2) continue;

            float plane_dist = fabsf(dot(diff, normal));
            if (plane_dist > DEFAULT_SURFACE_TAU) continue;

            float w = 1.f - d2 / r2; // Epanechnikov kernel

            float3 wi_world = make_f3(
                params.photon_wi_x[idx],
                params.photon_wi_y[idx],
                params.photon_wi_z[idx]);

            // photon_wi already points away from surface toward light
            // (stored as -direction in photon tracer) – do NOT negate
            DevONB frame = DevONB::from_normal(normal);
            float3 wi_local = frame.world_to_local(wi_world);

            Spectrum f = dev_bsdf_evaluate(mat_id, wo_local, wi_local);
            float flux = params.photon_flux[idx];
            int bin = params.photon_lambda[idx];

            L.value[bin] += f.value[bin] * flux * w * inv_area;
            count++;
        }
    }

    if (count > 0) {
        float norm = 1.5f / (float)params.num_photons; // 1.5 = Epanechnikov kernel correction
        for (int i = 0; i < NUM_LAMBDA; ++i) L.value[i] *= norm;
    }
    return L;
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

// == Binary search in CDF (device) ====================================
__forceinline__ __device__
int dev_binary_search_cdf(const float* cdf, int n, float u) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cdf[mid] <= u) lo = mid + 1;
        else hi = mid;
    }
    return lo;
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

    // Emission
    if (dev_is_emissive(mat_id))
        return dev_get_Le(mat_id);

    // Specular: one bounce then direct lighting
    float3 cur_pos = hit.position;
    float3 cur_dir = direction;
    float3 cur_normal = hit.shading_normal;
    uint32_t cur_mat = mat_id;
    Spectrum throughput_s = Spectrum::constant(1.0f);

    for (int bounce = 0; bounce < 4; ++bounce) {
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
        Spectrum Kd = dev_get_Kd(cur_mat);
        for (int i = 0; i < NUM_LAMBDA; ++i) throughput_s.value[i] *= Kd.value[i];
    }

    // Diffuse hit: direct lighting via next-event estimation
    // (no shadow test in debug mode -- always assume light is visible)
    Spectrum L = Spectrum::zero();
    Spectrum Kd = dev_get_Kd(cur_mat);

    if (params.num_emissive > 0) {
        float xi = rng.next_float();
        int local_idx = dev_binary_search_cdf(
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

            for (int i = 0; i < NUM_LAMBDA; ++i) {
                L.value[i] = Le.value[i] * Kd.value[i] * INV_PI
                             * cos_i / fmaxf(pdf_solid_angle, 1e-8f);
            }
        }
    }

    return throughput_s * L;
}

// =====================================================================
// NEE DIRECT LIGHTING (with shadow ray)
// M light samples per hitpoint, averaged.  Reuses the same CDF as
// photon emission so we have ONE light distribution for the whole
// renderer (per the spec).
// =====================================================================
__forceinline__ __device__
Spectrum dev_nee_direct(float3 pos, float3 normal, float3 wo_local,
                        uint32_t mat_id, PCGRng& rng)
{
    Spectrum L = Spectrum::zero();
    if (params.num_emissive <= 0) return L;

    const int M = (params.nee_light_samples > 0)
                    ? params.nee_light_samples : 1;

    // Build shading frame once (reused for every sample)
    DevONB frame = DevONB::from_normal(normal);

    for (int s = 0; s < M; ++s) {
        // Step A — Sample emissive triangle from shared CDF
        float xi = rng.next_float();
        int local_idx = dev_binary_search_cdf(
            params.emissive_cdf, params.num_emissive, xi);
        if (local_idx >= params.num_emissive)
            local_idx = params.num_emissive - 1;
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

        // Step E — Evaluate emission and BSDF
        // p_tri from CDF (same distribution as photon emission)
        float p_tri;
        if (local_idx == 0)
            p_tri = params.emissive_cdf[0];
        else
            p_tri = params.emissive_cdf[local_idx]
                  - params.emissive_cdf[local_idx - 1];

        uint32_t light_mat = params.material_ids[light_tri];
        Spectrum Le = dev_get_Le(light_mat);

        float3 wi_local = frame.world_to_local(wi);
        Spectrum f = dev_bsdf_evaluate(mat_id, wo_local, wi_local);

        // Step F — PDF conversion: area → solid angle
        // p_y_area  = p_tri * (1 / A_tri)
        // p_wi      = p_y_area * (dist2 / cos_y)
        float p_y_area = p_tri / light_area;
        float p_wi     = p_y_area * dist2 / cos_y;

        // Step G — Accumulate  f * Le * cos_x / p_wi
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            L.value[i] += f.value[i] * Le.value[i]
                          * cos_x / fmaxf(p_wi, 1e-8f);
        }
    }

    // Average over M samples
    if (M > 1) {
        float inv_M = 1.f / (float)M;
        for (int i = 0; i < NUM_LAMBDA; ++i)
            L.value[i] *= inv_M;
    }

    return L;
}

// =====================================================================
// FULL PATH TRACING (final render only)
// Hybrid: NEE (direct) + Photon density estimation (indirect)
// Returns: combined, nee_direct, photon_indirect components separately
// =====================================================================
struct PathTraceResult {
    Spectrum combined;
    Spectrum nee_direct;
    Spectrum photon_indirect;
};

__forceinline__ __device__
PathTraceResult full_path_trace(float3 origin, float3 direction, PCGRng& rng) {
    PathTraceResult result;
    result.combined        = Spectrum::zero();
    result.nee_direct      = Spectrum::zero();
    result.photon_indirect = Spectrum::zero();

    Spectrum throughput = Spectrum::constant(1.0f);
    bool prev_was_specular = true; // treat camera ray as specular for emission

    for (int bounce = 0; bounce <= params.max_bounces; ++bounce) {
        TraceResult hit = trace_radiance(origin, direction);
        if (!hit.hit) break;

        uint32_t mat_id = hit.material_id;

        // Emission: count if camera ray or specular bounce (unoccluded path)
        if (dev_is_emissive(mat_id)) {
            if (prev_was_specular) {
                Spectrum Le_contrib = throughput * dev_get_Le(mat_id);
                result.combined   += Le_contrib;
                result.nee_direct += Le_contrib; // attribute to direct
            }
            break;
        }

        // Specular bounce (mirror)
        if (dev_is_specular(mat_id)) {
            float3 n = hit.shading_normal;
            direction = direction - n * (2.f * dot(direction, n));
            origin = hit.position + n * OPTIX_SCENE_EPSILON;
            prev_was_specular = true;
            continue;
        }

        prev_was_specular = false;

        // Diffuse hit
        DevONB frame = DevONB::from_normal(hit.shading_normal);
        float3 wo_local = frame.world_to_local(direction * (-1.f));
        if (wo_local.z <= 0.f) break;

        // ── NEE: Direct lighting via shadow ray ──────────────────
        if (params.render_mode != RENDER_MODE_INDIRECT_ONLY) {
            Spectrum L_nee = dev_nee_direct(
                hit.position, hit.shading_normal, wo_local, mat_id, rng);
            Spectrum nee_contrib = throughput * L_nee;
            result.combined   += nee_contrib;
            result.nee_direct += nee_contrib;
        }

        // ── Photon density estimation: Indirect only ─────────────
        if (params.render_mode != RENDER_MODE_DIRECT_ONLY) {
            Spectrum L_photon = dev_estimate_photon_density(
                hit.position, hit.shading_normal, wo_local, mat_id,
                params.gather_radius);
            Spectrum photon_contrib = throughput * L_photon;
            result.combined        += photon_contrib;
            result.photon_indirect += photon_contrib;
        }

        // ── BSDF continuation (multi-bounce) ─────────────────────
        float3 wi_local = sample_cosine_hemisphere_dev(rng);
        float cos_theta_b = wi_local.z;
        if (cos_theta_b <= 0.f) break;

        Spectrum Kd = dev_get_Kd(mat_id);
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            throughput.value[i] *= Kd.value[i];
        }

        if (bounce >= DEFAULT_MIN_BOUNCES_RR) {
            float p_rr = fminf(DEFAULT_RR_THRESHOLD, throughput.max_component());
            if (p_rr <= 0.f) break;
            if (rng.next_float() >= p_rr) break;
            throughput *= 1.0f / p_rr;
        }

        direction = frame.local_to_world(wi_local);
        origin = hit.position + hit.shading_normal * OPTIX_SCENE_EPSILON;
    }

    return result;
}

// =====================================================================
// __raygen__render
// =====================================================================
extern "C" __global__ void __raygen__render() {
    const uint3 idx = optixGetLaunchIndex();
    int px = idx.x;
    int py = idx.y;
    int pixel_idx = py * params.width + px;

    Spectrum L_accum = Spectrum::zero();
    Spectrum L_nee_accum = Spectrum::zero();
    Spectrum L_photon_accum = Spectrum::zero();

    for (int s = 0; s < params.samples_per_pixel; ++s) {
        PCGRng rng = PCGRng::seed(
            (uint64_t)pixel_idx * 1000
                + (uint64_t)params.frame_number * 100000 + s,
            (uint64_t)pixel_idx + 1);

        float u = ((float)px + rng.next_float()) / (float)params.width;
        float v = ((float)py + rng.next_float()) / (float)params.height;

        float3 origin    = params.cam_pos;
        float3 direction = normalize(
            params.cam_lower_left
            + params.cam_horizontal * u
            + params.cam_vertical * v
            - params.cam_pos);

        if (params.is_final_render) {
            PathTraceResult ptr = full_path_trace(origin, direction, rng);
            L_accum        += ptr.combined;
            L_nee_accum    += ptr.nee_direct;
            L_photon_accum += ptr.photon_indirect;
        } else {
            Spectrum L = debug_first_hit(origin, direction, rng);
            L_accum += L;
        }
    }

    // Progressive accumulation (combined)
    for (int i = 0; i < NUM_LAMBDA; ++i)
        params.spectrum_buffer[pixel_idx * NUM_LAMBDA + i] += L_accum.value[i];
    params.sample_counts[pixel_idx] += (float)params.samples_per_pixel;

    // Component accumulation (only during final render)
    if (params.is_final_render && params.nee_direct_buffer) {
        for (int i = 0; i < NUM_LAMBDA; ++i)
            params.nee_direct_buffer[pixel_idx * NUM_LAMBDA + i] += L_nee_accum.value[i];
    }
    if (params.is_final_render && params.photon_indirect_buffer) {
        for (int i = 0; i < NUM_LAMBDA; ++i)
            params.photon_indirect_buffer[pixel_idx * NUM_LAMBDA + i] += L_photon_accum.value[i];
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
// =====================================================================
extern "C" __global__ void __raygen__photon_trace() {
    const uint3 idx = optixGetLaunchIndex();
    int photon_idx = idx.x;
    if (photon_idx >= params.num_photons) return;
    if (params.num_emissive <= 0) return;

    PCGRng rng = PCGRng::seed(
        (uint64_t)photon_idx * 7 + 42,
        (uint64_t)photon_idx + 1);

    // 1. Sample emissive triangle via CDF
    float xi = rng.next_float();
    int local_idx = dev_binary_search_cdf(
        params.emissive_cdf, params.num_emissive, xi);
    if (local_idx >= params.num_emissive) local_idx = params.num_emissive - 1;
    uint32_t tri_idx = params.emissive_tri_indices[local_idx];

    // PDF of this triangle
    float pdf_tri;
    if (local_idx == 0) pdf_tri = params.emissive_cdf[0];
    else pdf_tri = params.emissive_cdf[local_idx] - params.emissive_cdf[local_idx - 1];

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

    // 4. Sample wavelength from Le
    Spectrum Le = dev_get_Le(mat_id);
    float Le_sum = 0.f;
    for (int i = 0; i < NUM_LAMBDA; ++i) Le_sum += Le.value[i];
    if (Le_sum <= 0.f) return;

    float xi_lambda = rng.next_float() * Le_sum;
    int lambda_bin = NUM_LAMBDA - 1;
    float cum = 0.f;
    for (int i = 0; i < NUM_LAMBDA; ++i) {
        cum += Le.value[i];
        if (xi_lambda <= cum) { lambda_bin = i; break; }
    }
    float Le_lambda = Le.value[lambda_bin];
    float pdf_lambda = Le_lambda / Le_sum;

    // 5. Sample cosine hemisphere direction
    float3 local_dir = sample_cosine_hemisphere_dev(rng);
    DevONB frame = DevONB::from_normal(geo_n);
    float3 world_dir = frame.local_to_world(local_dir);
    float cos_theta = local_dir.z;
    float pdf_dir = cos_theta * INV_PI;

    // 6. Compute initial photon flux
    float denom = pdf_tri * pdf_pos * pdf_dir * pdf_lambda;
    float flux = (denom > 0.f) ? (Le_lambda * cos_theta) / denom : 0.f;

    // 7. Trace through scene
    float3 origin    = pos + geo_n * OPTIX_SCENE_EPSILON;
    float3 direction = world_dir;

    for (int bounce = 0; bounce < params.photon_max_bounces; ++bounce) {
        TraceResult hit = trace_radiance(origin, direction);
        if (!hit.hit) break;

        uint32_t hit_mat = hit.material_id;

        // Skip emissive surfaces
        if (dev_is_emissive(hit_mat)) break;

        // Store photon at diffuse surfaces (skip bounce 0 = direct lighting,
        // which is handled by NEE in the camera render pass).
        if (!dev_is_specular(hit_mat) && bounce > 0) {
            uint32_t slot = atomicAdd(params.out_photon_count, 1u);
            if (slot < (uint32_t)params.max_stored_photons) {
                params.out_photon_pos_x[slot]  = hit.position.x;
                params.out_photon_pos_y[slot]  = hit.position.y;
                params.out_photon_pos_z[slot]  = hit.position.z;
                params.out_photon_wi_x[slot]   = -direction.x;
                params.out_photon_wi_y[slot]   = -direction.y;
                params.out_photon_wi_z[slot]   = -direction.z;
                params.out_photon_lambda[slot]  = (uint16_t)lambda_bin;
                params.out_photon_flux[slot]    = flux;
            }
        }

        // Bounce
        if (dev_is_specular(hit_mat)) {
            // Mirror reflection
            float3 n = hit.shading_normal;
            direction = direction - n * (2.f * dot(direction, n));
            origin = hit.position + n * OPTIX_SCENE_EPSILON;
        } else {
            // Diffuse: cosine hemisphere sampling
            DevONB bounce_frame = DevONB::from_normal(hit.shading_normal);
            float3 wi_local = sample_cosine_hemisphere_dev(rng);
            // Throughput for Lambertian: f*cos/pdf = Kd (for single wavelength)
            Spectrum Kd = dev_get_Kd(hit_mat);
            flux *= Kd.value[lambda_bin];

            direction = bounce_frame.local_to_world(wi_local);
            origin = hit.position + hit.shading_normal * OPTIX_SCENE_EPSILON;
        }

        // Russian roulette
        if (bounce >= DEFAULT_MIN_BOUNCES_RR) {
            float p_rr = fminf(DEFAULT_RR_THRESHOLD, 0.5f);
            if (rng.next_float() >= p_rr) break;
            flux /= p_rr;
        }
    }
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
