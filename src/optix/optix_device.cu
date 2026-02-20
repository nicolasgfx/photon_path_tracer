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
#include "core/photon_bins.h"
#include "core/cdf.h"
#include "core/nee_sampling.h"
#include "core/photon_density_cache.h"
#include "core/guided_nee.h"

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
__forceinline__ __device__
float3 dev_sample_guided_bounce(
    const PhotonBin* bins, int N, float3 normal,
    const PhotonBinDirs& bin_dirs, PCGRng& rng)
{
    // Build CDF from bin fluxes (hemisphere-only bins)
    float cdf[MAX_PHOTON_BIN_COUNT];
    float total = 0.0f;
    for (int k = 0; k < N; ++k) {
        float3 bdir = make_f3(bins[k].dir_x, bins[k].dir_y, bins[k].dir_z);
        float cos_n = dot(bdir, normal);
        if (cos_n <= -PHOTON_BIN_HORIZON_EPS || bins[k].count == 0) {
            cdf[k] = total;
            continue;
        }
        total += bins[k].flux * fmaxf(cos_n, 0.0f);
        cdf[k] = total;
    }

    // Fallback to cosine hemisphere if no flux
    if (total <= 0.0f) {
        return sample_cosine_hemisphere_dev(rng);
    }

    // Select bin via CDF
    float xi = rng.next_float() * total;
    int selected = N - 1;
    for (int k = 0; k < N; ++k) {
        if (xi <= cdf[k]) { selected = k; break; }
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
    float3 wi = sample_cone_dev(bin_dir, cos_half, rng);

    // Clamp to positive hemisphere
    if (dot(wi, normal) <= 0.0f) {
        // Reflect across tangent plane
        wi = wi - normal * (2.0f * dot(wi, normal));
    }

    return wi;
}

// == Populate photon bins for a single pixel ==========================
// Called from __raygen__render when populate_bins_mode = 1.
// Traces center ray, gathers photons, writes bins to cache.
__forceinline__ __device__
void dev_populate_bins_for_pixel(int pixel_idx, float3 origin, float3 direction) {
    const int N = params.photon_bin_count;

    // Initialize bins to zero
    PhotonBin bins[MAX_PHOTON_BIN_COUNT];
    for (int k = 0; k < N; ++k) {
        bins[k].flux   = 0.0f;
        bins[k].dir_x  = 0.0f;
        bins[k].dir_y  = 0.0f;
        bins[k].dir_z  = 0.0f;
        bins[k].weight = 0.0f;
        bins[k].count  = 0;
    }

    // Build Fibonacci bin directions
    PhotonBinDirs bin_dirs;
    bin_dirs.init(N);

    // Trace center ray — follow specular bounces to first diffuse hit
    float3 pos = origin;
    float3 dir = direction;
    float3 normal;
    uint32_t mat_id = 0;
    bool found_diffuse = false;

    for (int bounce = 0; bounce < 8; ++bounce) {
        TraceResult hit = trace_radiance(pos, dir);
        if (!hit.hit) break;

        mat_id = hit.material_id;

        // Skip emissive
        if (dev_is_emissive(mat_id)) break;

        // Diffuse surface found
        if (!dev_is_specular(mat_id)) {
            pos = hit.position;
            normal = hit.shading_normal;
            found_diffuse = true;
            break;
        }

        // Mirror bounce
        float3 n = hit.shading_normal;
        dir = dir - n * (2.f * dot(dir, n));
        pos = hit.position + n * OPTIX_SCENE_EPSILON;
    }

    if (!found_diffuse) {
        // Write empty bins
        for (int k = 0; k < N; ++k)
            params.photon_bin_cache[pixel_idx * N + k] = bins[k];
        return;
    }

    // Gather photons from hash grid and populate bins
    if (params.num_photons == 0 || params.grid_table_size == 0) {
        for (int k = 0; k < N; ++k)
            params.photon_bin_cache[pixel_idx * N + k] = bins[k];
        return;
    }

    float radius = params.gather_radius;
    float cell_size = params.grid_cell_size;
    float r2 = radius * radius;

    int cx0 = (int)floorf((pos.x - radius) / cell_size);
    int cy0 = (int)floorf((pos.y - radius) / cell_size);
    int cz0 = (int)floorf((pos.z - radius) / cell_size);
    int cx1 = (int)floorf((pos.x + radius) / cell_size);
    int cy1 = (int)floorf((pos.y + radius) / cell_size);
    int cz1 = (int)floorf((pos.z + radius) / cell_size);

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

            float w = 1.0f - d2 / r2; // Epanechnikov kernel

            float3 wi_world = make_f3(
                params.photon_wi_x[idx],
                params.photon_wi_y[idx],
                params.photon_wi_z[idx]);

            int k = bin_dirs.find_nearest(wi_world);

            float flux = params.photon_flux[idx];
            bins[k].flux   += flux * w;
            bins[k].dir_x  += wi_world.x * flux * w;
            bins[k].dir_y  += wi_world.y * flux * w;
            bins[k].dir_z  += wi_world.z * flux * w;
            bins[k].weight += w;
            bins[k].count  += 1;
        }
    }

    // Normalize centroid directions
    for (int k = 0; k < N; ++k) {
        if (bins[k].count > 0) {
            float3 d = make_f3(bins[k].dir_x, bins[k].dir_y, bins[k].dir_z);
            float len = length(d);
            if (len > 1e-8f) {
                bins[k].dir_x = d.x / len;
                bins[k].dir_y = d.y / len;
                bins[k].dir_z = d.z / len;
            } else {
                // Fallback to Fibonacci center
                bins[k].dir_x = bin_dirs.dirs[k].x;
                bins[k].dir_y = bin_dirs.dirs[k].y;
                bins[k].dir_z = bin_dirs.dirs[k].z;
            }
        }
    }

    // Write bins to cache
    for (int k = 0; k < N; ++k)
        params.photon_bin_cache[pixel_idx * N + k] = bins[k];
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
                         uint32_t mat_id, PCGRng& rng, int bounce = 0);

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
    Spectrum L = Spectrum::zero();

    if (params.debug_shadow_rays) {
        // NEE PNG path: full dev_nee_direct with M shadow rays + BSDF eval
        float3 wo_local = make_float3(0.f, 0.f, 1.f); // approximate wo in local frame
        DevONB frame = DevONB::from_normal(cur_normal);
        wo_local = frame.world_to_local(normalize(-cur_dir));
        NeeResult nee = dev_nee_direct(cur_pos, cur_normal, wo_local, cur_mat, rng);
        L = nee.L;
    } else {
        // Real-time debug: fast single-sample, no shadow ray
        Spectrum Kd = dev_get_Kd(cur_mat);

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

                for (int i = 0; i < NUM_LAMBDA; ++i) {
                    L.value[i] = Le.value[i] * Kd.value[i] * INV_PI
                                 * cos_i / fmaxf(pdf_solid_angle, 1e-8f);
                }
            }
        }
    }

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
                         const PhotonBinDirs& bin_dirs)
{
    NeeResult result;
    result.L = Spectrum::zero();
    result.visibility = 0.f;
    if (params.num_emissive <= 0) return result;

    // Fall back to standard NEE if too many emissive tris for stack CDF
    if (params.num_emissive > NEE_GUIDED_MAX_EMISSIVE) {
        return dev_nee_direct(pos, normal, wo_local, mat_id, rng, bounce);
    }

    // Compute total bin flux (for early-out if bins are empty)
    float total_bin_flux = 0.0f;
    for (int k = 0; k < N; ++k) total_bin_flux += bins[k].flux;
    if (total_bin_flux <= 0.0f) {
        return dev_nee_direct(pos, normal, wo_local, mat_id, rng, bounce);
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
        return dev_nee_direct(pos, normal, wo_local, mat_id, rng, bounce);
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
        Spectrum f = dev_bsdf_evaluate(mat_id, wo_local, wi_local);

        // PDF conversion: area → solid angle
        float p_y_area = p_guided / light_area;
        float p_wi     = p_y_area * dist2 / cos_y;

        // Accumulate f * Le * cos_x / p_wi
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            result.L.value[i] += f.value[i] * Le.value[i]
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
                         uint32_t mat_id, PCGRng& rng, int bounce)
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
        // Step A — Sample emissive triangle from shared CDF
        float xi = rng.next_float();
        int local_idx = binary_search_cdf(
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

        visible_count++;

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
            result.L.value[i] += f.value[i] * Le.value[i]
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
// FULL PATH TRACING (final render only)
// Hybrid: NEE (direct) + Photon density estimation (indirect)
// Returns: combined, nee_direct, photon_indirect components separately
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
                                int pixel_idx) {
    PathTraceResult result;
    result.combined        = Spectrum::zero();
    result.nee_direct      = Spectrum::zero();
    result.photon_indirect = Spectrum::zero();
    result.clk_ray_trace     = 0;
    result.clk_nee           = 0;
    result.clk_photon_gather = 0;
    result.clk_bsdf          = 0;

    Spectrum throughput = Spectrum::constant(1.0f);
    bool prev_was_specular = true; // treat camera ray as specular for emission

    // Pre-load bin directions if bins are valid (for guided bounce)
    const bool use_bins = (params.photon_bins_valid == 1 && params.photon_bin_cache != nullptr);
    PhotonBinDirs bin_dirs;
    if (use_bins) bin_dirs.init(params.photon_bin_count);

    for (int bounce = 0; bounce <= params.max_bounces; ++bounce) {
        long long t0 = clock64();
        TraceResult hit = trace_radiance(origin, direction);
        result.clk_ray_trace += clock64() - t0;

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
        float nee_visibility = 1.f;  // default: fully lit (no NEE = no occlusion info)
        if (params.render_mode != RENDER_MODE_INDIRECT_ONLY) {
            long long t_nee = clock64();
            NeeResult nee;
            if (use_bins) {
                const PhotonBin* pbins = &params.photon_bin_cache[pixel_idx * params.photon_bin_count];
                nee = dev_nee_guided(
                    hit.position, hit.shading_normal, wo_local, mat_id, rng, bounce,
                    pbins, params.photon_bin_count, bin_dirs);
            } else {
                nee = dev_nee_direct(
                    hit.position, hit.shading_normal, wo_local, mat_id, rng, bounce);
            }
            result.clk_nee += clock64() - t_nee;

            nee_visibility = nee.visibility;
            Spectrum nee_contrib = throughput * nee.L;
            result.combined   += nee_contrib;
            result.nee_direct += nee_contrib;
        }

        // ── Photon density estimation: Indirect only ─────────────
        // When bins are valid and this is NOT the first sample (frame>0),
        // read from cached spectral density.  On frame 0, do full gather
        // and cache the result for subsequent samples.
        // PHOTON_SHADOW_FLOOR prevents fully killing indirect light
        // in deep shadow — some indirect illumination should survive.
        if (params.render_mode != RENDER_MODE_DIRECT_ONLY) {
            long long t_pg = clock64();
            Spectrum L_photon;

            if (should_read_photon_density_cache(
                    use_bins, bounce, params.photon_density_cache, params.frame_number))
            {
                // Read from cached spectral density (populated on frame 0)
                for (int i = 0; i < NUM_LAMBDA; ++i)
                    L_photon.value[i] = params.photon_density_cache[pixel_idx * NUM_LAMBDA + i];
            } else {
                // Full photon gather (always on frame 0, or when bins not valid)
                L_photon = dev_estimate_photon_density(
                    hit.position, hit.shading_normal, wo_local, mat_id,
                    params.gather_radius);

                // Cache the result for subsequent frames (first bounce only)
                if (should_write_photon_density_cache(
                        use_bins, bounce, params.photon_density_cache, params.frame_number))
                {
                    for (int i = 0; i < NUM_LAMBDA; ++i)
                        params.photon_density_cache[pixel_idx * NUM_LAMBDA + i] = L_photon.value[i];
                }
            }
            result.clk_photon_gather += clock64() - t_pg;

            float vis_weight = fmaxf(nee_visibility, PHOTON_SHADOW_FLOOR);
            Spectrum photon_contrib = throughput * L_photon * vis_weight;
            result.combined        += photon_contrib;
            result.photon_indirect += photon_contrib;
        }

        // ── BSDF continuation (multi-bounce) ─────────────────────
        // When bins are valid at bounce 0, use guided bounce direction
        // proportional to bin flux.  Deeper bounces use cosine hemisphere.
        long long t_bsdf = clock64();
        float3 wi_world;
        if (use_bins && bounce == 0) {
            const PhotonBin* pbins = &params.photon_bin_cache[pixel_idx * params.photon_bin_count];
            wi_world = dev_sample_guided_bounce(
                pbins, params.photon_bin_count, hit.shading_normal, bin_dirs, rng);
            // Transform to local for throughput computation
        } else {
            float3 wi_local = sample_cosine_hemisphere_dev(rng);
            wi_world = frame.local_to_world(wi_local);
        }

        float cos_theta_b = dot(wi_world, hit.shading_normal);
        if (cos_theta_b <= 0.f) { result.clk_bsdf += clock64() - t_bsdf; break; }

        Spectrum Kd = dev_get_Kd(mat_id);
        for (int i = 0; i < NUM_LAMBDA; ++i) {
            throughput.value[i] *= Kd.value[i];
        }

        if (bounce >= DEFAULT_MIN_BOUNCES_RR) {
            float p_rr = fminf(DEFAULT_RR_THRESHOLD, throughput.max_component());
            if (p_rr <= 0.f) { result.clk_bsdf += clock64() - t_bsdf; break; }
            if (rng.next_float() >= p_rr) { result.clk_bsdf += clock64() - t_bsdf; break; }
            throughput *= 1.0f / p_rr;
        }

        direction = wi_world;
        origin = hit.position + hit.shading_normal * OPTIX_SCENE_EPSILON;
        result.clk_bsdf += clock64() - t_bsdf;
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

    // ── BIN POPULATION MODE ─────────────────────────────────────
    // Separate launch: trace center ray, gather photons, fill bins.
    if (params.populate_bins_mode) {
        float u = ((float)px + 0.5f) / (float)params.width;
        float v = ((float)py + 0.5f) / (float)params.height;
        float3 origin    = params.cam_pos;
        float3 direction = normalize(
            params.cam_lower_left
            + params.cam_horizontal * u
            + params.cam_vertical * v
            - params.cam_pos);
        dev_populate_bins_for_pixel(pixel_idx, origin, direction);
        return;
    }

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

        float3 origin    = params.cam_pos;
        float3 direction = normalize(
            params.cam_lower_left
            + params.cam_horizontal * u
            + params.cam_vertical * v
            - params.cam_pos);

        if (params.is_final_render) {
            PathTraceResult ptr = full_path_trace(origin, direction, rng, pixel_idx);
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
    int local_idx = binary_search_cdf(
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
