#ifndef PPT_INTEGRATOR_CAUSTIC_TRACER_CUH
#define PPT_INTEGRATOR_CAUSTIC_TRACER_CUH
// ─────────────────────────────────────────────────────────────────────
// integrator/caustic_tracer.cuh – Caustic light tracing kernel
//
// Traces photons from emitters through delta surface chains (mirror/
// glass/translucent) and splats to the caustic buffer on the first
// non-delta hit via camera connection + visibility test.
//
// Reuses: dev_bsdf_sample, dev_bsdf_evaluate, dev_is_delta,
//         build_shading_frame, trace_radiance, trace_shadow,
//         dev_nee_select_global, sample_triangle_dev,
//         clamp_path_throughput, russian_roulette, PCGRng
//         — all from path_tracer.cuh / optix_nee.cuh / optix_utils.cuh
// ─────────────────────────────────────────────────────────────────────

#include "integrator/sample_clamping.h"
#include "integrator/russian_roulette.h"

// Representative wavelengths (nm) for RGB channels
constexpr float CAUSTIC_LAMBDA_R = 630.f;
constexpr float CAUSTIC_LAMBDA_G = 532.f;
constexpr float CAUSTIC_LAMBDA_B = 465.f;

// =====================================================================
// dev_cauchy_ior — wavelength-dependent IOR from Cauchy coefficients
// =====================================================================
__forceinline__ __device__
float dev_cauchy_ior(uint32_t mat_id, float lambda_nm) {
    if (!params.cauchy_B) return dev_get_ior(mat_id);
    float B = params.cauchy_B[mat_id];
    if (B == 0.f) return dev_get_ior(mat_id);
    float A = params.cauchy_A[mat_id];
    return A + B / (lambda_nm * lambda_nm);
}

// =====================================================================
// dev_glass_sample_dispersive — glass BSDF sample with overridden IOR
//
// Identical to the Glass case in dev_bsdf_sample but uses the
// provided wavelength-dependent IOR instead of the material default.
// =====================================================================
__forceinline__ __device__
DevBSDFSample dev_glass_sample_dispersive(
    uint32_t mat_id, float3 wo, bool entering, float2 uv,
    float ior_override, PCGRng& rng)
{
    DevBSDFSample s;
    s.wi = make_f3(0, 0, 0);
    s.f = make_f3(0, 0, 0);
    s.pdf = 0.f;
    s.is_specular = true;

    float3 Tf = dev_get_Tf(mat_id);
    float cos_i = fabsf(wo.z);

    if (dev_is_thin(mat_id)) {
        float eta = 1.f / ior_override;
        float F = fresnel_dielectric(cos_i, eta);
        if (rng.next_float() < F) {
            s.wi = reflect_local(wo);
            s.pdf = F;
            float factor = F / (fabsf(s.wi.z) + EPSILON);
            s.f = make_f3(factor, factor, factor);
        } else {
            s.wi = transmit_thin_local(wo);
            s.pdf = 1.f - F;
            float factor = (1.f - F) / (fabsf(s.wi.z) + EPSILON);
            s.f = Tf * factor;
        }
        return s;
    }

    float eta = entering ? (1.f / ior_override) : ior_override;
    float F = fresnel_dielectric(cos_i, eta);

    if (rng.next_float() < F) {
        s.wi = reflect_local(wo);
        s.pdf = F;
        float factor = F / (fabsf(s.wi.z) + EPSILON);
        s.f = make_f3(factor, factor, factor);
    } else {
        float3 wt;
        if (refract_local(wo, eta, wt)) {
            s.wi = wt;
            s.pdf = 1.f - F;
            float factor = (1.f - F) / (fabsf(s.wi.z) + EPSILON);
            s.f = Tf * factor;
        } else {
            s.wi = reflect_local(wo);
            s.pdf = 1.f;
            float factor = 1.f / (fabsf(s.wi.z) + EPSILON);
            s.f = make_f3(factor, factor, factor);
        }
    }
    return s;
}

// =====================================================================
// dev_world_to_pixel — inverse pinhole projection
//
// Camera convention (from __raygen__render):
//   ray_dir = cam_w + cam_u * (2*u_ndc - 1) + cam_v * (2*v_ndc - 1)
//
// Given a world point P, solve for (u_ndc, v_ndc) where
//   (P - cam_pos) = lambda * (cam_w + cam_u * a + cam_v * b)
// with a = 2*u_ndc - 1, b = 2*v_ndc - 1.
//
// This is a 3×3 linear system.  We use Cramer's rule (always works
// for a valid camera basis since cam_w, cam_u, cam_v are linearly
// independent).
// =====================================================================

__forceinline__ __device__
bool dev_world_to_pixel(float3 world_pos, int& px_out, int& py_out) {
    float3 d = world_pos - params.cam_pos;

    // Columns of the 3×3 matrix [cam_w | cam_u | cam_v]
    const float3& c0 = params.cam_w;
    const float3& c1 = params.cam_u;
    const float3& c2 = params.cam_v;

    // Determinant via triple product
    float det = c0.x * (c1.y * c2.z - c1.z * c2.y)
              - c1.x * (c0.y * c2.z - c0.z * c2.y)
              + c2.x * (c0.y * c1.z - c0.z * c1.y);

    if (fabsf(det) < 1e-20f) return false;

    float inv_det = 1.f / det;

    // Cramer: lambda = det([d | c1 | c2]) / det
    float lambda = (d.x * (c1.y * c2.z - c1.z * c2.y)
                  - c1.x * (d.y * c2.z - d.z * c2.y)
                  + c2.x * (d.y * c1.z - d.z * c1.y)) * inv_det;

    if (lambda <= 0.f) return false;  // behind camera

    // Cramer gives (lambda*a) and (lambda*b); divide by lambda to get a, b
    float la = (c0.x * (d.y * c2.z - d.z * c2.y)
              - d.x * (c0.y * c2.z - c0.z * c2.y)
              + c2.x * (c0.y * d.z - c0.z * d.y)) * inv_det;

    float lb = (c0.x * (c1.y * d.z - c1.z * d.y)
              - c1.x * (c0.y * d.z - c0.z * d.y)
              + d.x * (c0.y * c1.z - c0.z * c1.y)) * inv_det;

    float a = la / lambda;  // 2*u_ndc - 1
    float b = lb / lambda;  // 2*v_ndc - 1
    float u_ndc = (a + 1.f) * 0.5f;
    float v_ndc = (b + 1.f) * 0.5f;

    if (u_ndc < 0.f || u_ndc >= 1.f || v_ndc < 0.f || v_ndc >= 1.f)
        return false;

    px_out = (int)(u_ndc * (float)params.width);
    py_out = (int)(v_ndc * (float)params.height);

    // Clamp edge pixels
    if (px_out >= params.width)  px_out = params.width  - 1;
    if (py_out >= params.height) py_out = params.height - 1;

    return true;
}

// =====================================================================
// dev_sensor_importance — pinhole We measurement contribution
//
// For a pinhole camera, the measurement function is:
//   We = 1 / (cos^3(theta) * pixel_solid_angle)
// where theta = angle between the connection direction and camera axis.
//
// pixel_solid_angle = 4 * |cam_u| * |cam_v| / (width * height)
//   (since cam_u/cam_v encode half the image-plane extent)
// =====================================================================

__forceinline__ __device__
float dev_sensor_importance(float3 cam_dir) {
    float3 cam_forward = normalize(params.cam_w);
    float cos_theta = dot(cam_dir, cam_forward);
    if (cos_theta <= 1e-6f) return 0.f;

    float pixel_sa = 4.f * length(params.cam_u) * length(params.cam_v)
                   / (float)(params.width * params.height);
    if (pixel_sa <= 0.f) return 0.f;

    float cos3 = cos_theta * cos_theta * cos_theta;
    return 1.f / (cos3 * pixel_sa);
}

// =====================================================================
// DeltaTarget — a point sampled on a delta surface for direction biasing
// =====================================================================

struct DeltaTarget {
    float3 position;
    float3 normal;
};

// =====================================================================
// dev_sample_delta_target — pick a random point on a delta surface
//
// Uses the prebuilt area-weighted CDF over all delta triangles.
// Returns false if there are no delta surfaces.
// =====================================================================

__forceinline__ __device__
bool dev_sample_delta_target(PCGRng& rng, DeltaTarget& out) {
    if (params.num_delta_tris <= 0 || !params.delta_cdf || !params.delta_tri_indices)
        return false;

    // Pick a delta triangle via CDF
    int local_idx = binary_search_cdf(
        params.delta_cdf, params.num_delta_tris, rng.next_float());
    uint32_t tri_id = params.delta_tri_indices[local_idx];

    float3 v0 = params.vertices[tri_id * 3 + 0];
    float3 v1 = params.vertices[tri_id * 3 + 1];
    float3 v2 = params.vertices[tri_id * 3 + 2];

    // Sample a point on the triangle
    float3 bary = sample_triangle_dev(rng.next_float(), rng.next_float());
    out.position = v0 * bary.x + v1 * bary.y + v2 * bary.z;

    // Geometric normal
    float3 cr = cross(v1 - v0, v2 - v0);
    float len = length(cr);
    if (len < 1e-20f) return false;
    out.normal = cr * (1.f / len);

    return true;
}

// =====================================================================
// trace_caustic — single photon: emit → delta chain → camera splat
// =====================================================================

__forceinline__ __device__
void trace_caustic(int photon_idx, PCGRng& rng) {
    if (params.num_emissive <= 0) return;
    if (!params.caustic_r || !params.caustic_g || !params.caustic_b) return;

    int max_bounces = params.max_bounces;

    // ── 1. Sample emitting triangle ─────────────────────────────────
    float pdf_emit_tri;
    int emit_local = dev_nee_select_global(rng, pdf_emit_tri);
    if (pdf_emit_tri <= 0.f) return;

    uint32_t emit_tri_id = params.emissive_tri_indices[emit_local];
    float3 ev0 = params.vertices[emit_tri_id * 3 + 0];
    float3 ev1 = params.vertices[emit_tri_id * 3 + 1];
    float3 ev2 = params.vertices[emit_tri_id * 3 + 2];
    float3 ecross = cross(ev1 - ev0, ev2 - ev0);
    float emit_area = length(ecross) * 0.5f;
    if (emit_area <= 0.f) return;
    float3 emit_normal = ecross * (0.5f / emit_area);

    // Orient to match stored vertex normal (PLY winding can disagree)
    float3 en0 = params.normals[emit_tri_id * 3];
    if (dot(emit_normal, en0) < 0.f)
        emit_normal = emit_normal * (-1.f);

    float3 bary = sample_triangle_dev(rng.next_float(), rng.next_float());
    float3 emit_pos = ev0 * bary.x + ev1 * bary.y + ev2 * bary.z;

    uint32_t emit_mat = params.material_ids[emit_tri_id];
    float2 emit_uv = make_f2(bary.y, bary.z); // approximate
    float3 Le = dev_get_Le(emit_mat, emit_uv);

    // pdf for choosing this point on this emitter
    // (pdf_emit_pos cancels in throughput formula, only pdf_emit_tri remains)

    // ── 2. Direction sampling: 50/50 hemisphere vs delta-target ─────
    //
    // Strategy A (even photons): cosine-weighted hemisphere
    //   pdf_dir = cos_emit / PI
    //   throughput = Le × PI × emit_area / pdf_emit_tri
    //
    // Strategy B (odd photons): aim at a random delta surface point
    //   pdf_dir(area) = 1 / delta_total_area   (uniform over delta area)
    //   pdf_dir(solid angle) = dist² / (|cos_target| × delta_total_area)
    //   throughput = Le × cos_emit × emit_area × delta_total_area × |cos_target|
    //               / (pdf_emit_tri × dist²)
    //
    // Both strategies share the same 1/N normalization at splat time.
    // The 50/50 split is deterministic (no RNG consumed) via photon_idx & 1.

    ONB emit_frame = ONB::from_normal(emit_normal);
    float3 direction;
    float3 throughput;

    bool use_delta_target = (photon_idx & 1) != 0;

    if (use_delta_target) {
        // Strategy B: aim at delta surface
        DeltaTarget dt;
        if (dev_sample_delta_target(rng, dt)) {
            float3 to_target = dt.position - emit_pos;
            float dist2 = dot(to_target, to_target);
            if (dist2 < 1e-12f) use_delta_target = false;
            else {
                float dist = sqrtf(dist2);
                direction = to_target * (1.f / dist);
                float cos_e = dot(direction, emit_normal);
                float cos_t = fabsf(dot(direction, dt.normal));
                if (cos_e <= 0.f || cos_t < 1e-6f)
                    use_delta_target = false;
                else {
                    // throughput = Le * cos_e / (pdf_tri * pdf_pos * pdf_dir_solidangle)
                    //   pdf_pos = 1/emit_area
                    //   pdf_dir_solidangle = dist² / (cos_t * delta_total_area)
                    // = Le * cos_e * emit_area * cos_t * delta_total_area / (pdf_tri * dist²)
                    throughput = Le * (cos_e * emit_area * cos_t *
                                       params.delta_total_area /
                                       (pdf_emit_tri * dist2));
                }
            }
        } else {
            use_delta_target = false;
        }
    }

    if (!use_delta_target) {
        // Strategy A: cosine-weighted hemisphere (original path)
        float u1 = rng.next_float();
        float u2 = rng.next_float();
        float cos_emit = sqrtf(u1);
        float sin_emit = sqrtf(fmaxf(0.f, 1.f - u1));
        float phi = TWO_PI * u2;
        float3 local_dir = make_f3(sin_emit * cosf(phi),
                                    sin_emit * sinf(phi),
                                    cos_emit);
        direction = emit_frame.local_to_world(local_dir);
        throughput = Le * (PI * emit_area / pdf_emit_tri);
    }

    if (params.clamping_enabled)
        throughput = clamp_path_throughput(throughput, params.max_path_throughput);

    // ── Dispersion: pick a wavelength channel per photon ────────────
    // 0=R, 1=G, 2=B.  Throughput is 3× in the chosen channel only.
    int wave_ch = (int)(rng.next_float() * 3.f);
    if (wave_ch > 2) wave_ch = 2;
    constexpr float lambdas[3] = {CAUSTIC_LAMBDA_R, CAUSTIC_LAMBDA_G, CAUSTIC_LAMBDA_B};
    float lambda_nm = lambdas[wave_ch];

    float3 origin = emit_pos + emit_normal * PATH_EPSILON;
    bool had_delta_bounce = false;

    // ── 3. Bounce loop ──────────────────────────────────────────────
    for (int bounce = 0; bounce < max_bounces; ++bounce) {
        TraceResult hit = trace_radiance(origin, direction);
        if (!hit.hit) return;

        uint32_t mat_id = hit.material_id;

        // Normal perturbation (reuse path tracer logic)
        hit.shading_normal = dev_apply_normal_map(
            mat_id, hit.shading_normal, hit.geo_normal, hit.tangent, hit.uv);
        hit.shading_normal = dev_apply_bump_map(
            mat_id, hit.shading_normal, hit.geo_normal, hit.tangent, hit.uv);

        // Hit an emitter → abort (don't double-count direct lighting)
        if (dev_get_mat_type(mat_id) == Emissive) return;

        // ── Delta surface: bounce through ───────────────────────────
        if (dev_is_delta(mat_id)) {
            had_delta_bounce = true;
            float3 wo_world = direction * (-1.f);
            ShadingFrame sf = build_shading_frame(
                hit.shading_normal, hit.geo_normal, wo_world);
            if (!sf.valid) return;
            float3 wo_local = sf.wo_local(wo_world);

            // Use wavelength-dependent IOR for Glass/Translucent
            uint8_t mt = dev_get_mat_type(mat_id);
            DevBSDFSample bs;
            if ((mt == Glass || mt == Translucent)
                && params.cauchy_B && params.cauchy_B[mat_id] != 0.f) {
                float ior_disp = dev_cauchy_ior(mat_id, lambda_nm);
                bs = dev_glass_sample_dispersive(
                    mat_id, wo_local, sf.entering, hit.uv, ior_disp, rng);
            } else {
                bs = dev_bsdf_sample(
                    mat_id, wo_local, sf.entering, hit.uv, rng);
            }
            if (bs.pdf <= 0.f) return;

            float cos_theta = fabsf(bs.wi.z);
            float3 delta_transfer = bs.f * (cos_theta / bs.pdf);
            float3 wi_world = sf.frame.local_to_world(bs.wi);

            throughput = throughput * delta_transfer;
            if (params.clamping_enabled)
                throughput = clamp_path_throughput(throughput, params.max_path_throughput);

            if (bounce >= params.min_bounces_rr) {
                float max_tp = max_component(throughput);
                RRResult rr = russian_roulette(max_tp, params.rr_threshold,
                                               rng.next_float());
                if (rr.terminate) return;
                throughput = throughput * rr.inv_survival;
            }

            float3 offset_n = sf.geo_n;
            if (dot(wi_world, sf.geo_n) < 0.f)
                offset_n = offset_n * (-1.f);
            origin    = hit.position + offset_n * PATH_EPSILON;
            direction = wi_world;
            continue;
        }

        // ── Non-delta surface ───────────────────────────────────────
        float3 wo_world = direction * (-1.f);
        ShadingFrame sf = build_shading_frame(
            hit.shading_normal, hit.geo_normal, wo_world);
        if (!sf.valid) return;
        float3 wo_local = sf.wo_local(wo_world);

        // Camera connection only after at least one delta bounce
        if (had_delta_bounce) {
            float3 to_cam = params.cam_pos - hit.position;
            float cam_dist = length(to_cam);
            if (cam_dist >= 1e-6f) {
                float3 cam_dir = to_cam * (1.f / cam_dist);

                int px, py;
                if (dev_world_to_pixel(hit.position, px, py)) {
                    float3 wi_cam_local = sf.frame.world_to_local(cam_dir);
                    if (wi_cam_local.z > 0.f) {
                        float3 f_d = dev_bsdf_evaluate(mat_id, wo_local, wi_cam_local, hit.uv);

                        float3 shadow_origin = hit.position + sf.geo_n * PATH_EPSILON;
                        if (trace_shadow(shadow_origin, cam_dir, cam_dist)) {
                            float We = dev_sensor_importance(cam_dir * (-1.f));
                            if (We > 0.f) {
                                float cos_hit = wi_cam_local.z;
                                float inv_n = 1.f / (float)params.caustic_num_photons;
                                float3 L = throughput * f_d * (cos_hit * We * inv_n);
                                if (params.clamping_enabled)
                                    L = clamp_sample_luminance(L, params.caustic_max_splat_luminance);
                                if (is_finite_f3(L)) {
                                    int pixel = py * params.width + px;
                                    // Dispersion: 3× weight into chosen channel only
                                    float Lr = (wave_ch == 0) ? L.x * 3.f : 0.f;
                                    float Lg = (wave_ch == 1) ? L.y * 3.f : 0.f;
                                    float Lb = (wave_ch == 2) ? L.z * 3.f : 0.f;
                                    atomicAdd(&params.caustic_r[pixel], Lr);
                                    atomicAdd(&params.caustic_g[pixel], Lg);
                                    atomicAdd(&params.caustic_b[pixel], Lb);
                                }
                            }
                        }
                    }
                }
            }
            return;  // one camera connection per photon
        }

        // No delta bounce yet → BSDF bounce and keep tracing
        DevBSDFSample bs = dev_bsdf_sample(
            mat_id, wo_local, sf.entering, hit.uv, rng);
        if (bs.pdf <= 0.f || bs.wi.z <= 0.f) return;

        float cos_theta = bs.wi.z;
        float3 f_over_pdf = bs.f * (cos_theta / bs.pdf);
        float3 wi_world = sf.frame.local_to_world(bs.wi);

        throughput = throughput * f_over_pdf;
        if (params.clamping_enabled)
            throughput = clamp_path_throughput(throughput, params.max_path_throughput);

        if (bounce >= params.min_bounces_rr) {
            float max_tp = max_component(throughput);
            RRResult rr = russian_roulette(max_tp, params.rr_threshold,
                                           rng.next_float());
            if (rr.terminate) return;
            throughput = throughput * rr.inv_survival;
        }

        origin    = hit.position + sf.geo_n * PATH_EPSILON;
        direction = wi_world;
    }
}

// =====================================================================
// __raygen__caustic — entry point (1 thread per photon)
// =====================================================================

extern "C" __global__ void __raygen__caustic() {
    const uint3 idx = optixGetLaunchIndex();
    int photon_idx = idx.x;
    if (photon_idx >= params.caustic_num_photons) return;

    PCGRng rng = PCGRng::seed(
        (uint64_t)params.caustic_frame_number * 65537ull + 7919ull,
        (uint64_t)photon_idx + 1ull);

    trace_caustic(photon_idx, rng);
}

#endif // PPT_INTEGRATOR_CAUSTIC_TRACER_CUH
