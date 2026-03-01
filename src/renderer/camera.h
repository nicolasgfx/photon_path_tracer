#pragma once
// ─────────────────────────────────────────────────────────────────────
// camera.h – Thin-lens camera model (pinhole when lens_radius == 0)
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/config.h"
#include "core/random.h"

// ─────────────────────────────────────────────────────────────────────
// generate_camera_ray – Unified HD camera ray generation
// ─────────────────────────────────────────────────────────────────────
// Supports sub-pixel jitter, optional stratification, focus-range
// jitter, and thin-lens DOF (Shirley concentric disk).
//
// sample_index >= 0 → stratified sub-pixel jitter.
// ─────────────────────────────────────────────────────────────────────
inline HD Ray generate_camera_ray(
    int px, int py, PCGRng& rng,
    int img_width, int img_height,
    float3 lower_left, float3 horizontal, float3 vertical,
    float3 cam_pos, float3 cam_u, float3 cam_v,
    float lens_radius, float focus_dist, float focus_range,
    int sample_index = -1)
{
    // Sub-pixel jitter (stratified when sample_index >= 0)
    float jx, jy;
    if (sample_index >= 0
        && STRATA_X > 1 && STRATA_Y > 1) {
        int stratum_x = sample_index % STRATA_X;
        int stratum_y = (sample_index / STRATA_X) % STRATA_Y;
        jx = ((float)stratum_x + rng.next_float()) / (float)STRATA_X;
        jy = ((float)stratum_y + rng.next_float()) / (float)STRATA_Y;
    } else {
        jx = rng.next_float();
        jy = rng.next_float();
    }

    float s = ((float)px + jx) / (float)img_width;
    float t = ((float)py + jy) / (float)img_height;

    float3 focus_target = lower_left + horizontal * s + vertical * t;

    // Focus-range jitter
    if (focus_range > 0.f && focus_dist > 0.f) {
        float range_jitter = (rng.next_float() - 0.5f) * focus_range;
        float jittered_dist = fmaxf(focus_dist + range_jitter, 1e-4f);
        float scale = jittered_dist / focus_dist;
        focus_target = cam_pos + (focus_target - cam_pos) * scale;
    }

    // Thin-lens DOF (Shirley concentric disk)
    Ray ray;
    if (lens_radius > 0.f) {
        float2 disk = sample_concentric_disk(rng.next_float(), rng.next_float());
        float3 lens_offset = (cam_u * disk.x + cam_v * disk.y) * lens_radius;
        ray.origin    = cam_pos + lens_offset;
        ray.direction = normalize(focus_target - ray.origin);
    } else {
        ray.origin    = cam_pos;
        ray.direction = normalize(focus_target - cam_pos);
    }
    return ray;
}

struct Camera {
    float3 position;
    float3 look_at;
    float3 up;

    float  fov_deg;    // Vertical field of view in degrees
    int    width;
    int    height;

    // ── Depth of field parameters ───────────────────────────────────
    bool   dof_enabled     = DEFAULT_DOF_ENABLED;
    float  dof_focus_dist  = DEFAULT_DOF_FOCUS_DISTANCE; // scene units
    float  dof_f_number    = DEFAULT_DOF_F_NUMBER;
    float  sensor_height   = DEFAULT_DOF_SENSOR_HEIGHT;  // metres (0.024 = 24 mm)
    float  dof_focus_range = DEFAULT_DOF_FOCUS_RANGE;    // fraction of focus_dist that stays sharp (0.05 = 5%)

    // Derived DOF (computed by update())
    float  focal_length = 0.f;   // computed from sensor_height + fov_deg
    float  lens_radius  = 0.f;   // 0 → pinhole

    // Derived (call update() after modifying parameters)
    float3 u, v, w;   // Camera frame: w = -look_direction
    float3 lower_left;
    float3 horizontal;
    float3 vertical;

    void update() {
        float aspect = (float)width / (float)height;
        float theta  = fov_deg * PI / 180.0f;
        float h = tanf(theta * 0.5f);

        // Compute camera basis
        w = normalize(position - look_at);
        u = normalize(cross(up, w));
        v = cross(w, u);

        // DOF derived values
        focal_length = sensor_height / (2.0f * h);  // f = H / (2 tan(fov/2))
        lens_radius  = dof_enabled ? focal_length / (2.0f * dof_f_number) : 0.f;

        // Build view plane.  When DOF is active the plane sits at the
        // focus distance so that the focus-plane target is used for
        // thin-lens ray construction.  When DOF is off (lens_radius==0)
        // the focus distance factor is 1.0 (unit plane), recovering the
        // original pinhole math.
        float focus_factor = (lens_radius > 0.f) ? dof_focus_dist : 1.0f;
        float viewport_h = 2.0f * h * focus_factor;
        float viewport_w = aspect * viewport_h;

        horizontal = u * viewport_w;
        vertical   = v * viewport_h;
        lower_left = position - horizontal * 0.5f - vertical * 0.5f - w * focus_factor;
    }

    // Generate a ray for pixel (px, py) with jitter for anti-aliasing
    Ray generate_ray(int px, int py, PCGRng& rng,
                     int sample_index = -1) const {
        return generate_camera_ray(
            px, py, rng, width, height,
            lower_left, horizontal, vertical,
            position, u, v,
            lens_radius, dof_focus_dist, dof_focus_range,
            sample_index);
    }

    // Default Cornell Box camera
    static Camera cornell_box_camera(int w, int h) {
        Camera cam;
        cam.position = make_f3(0.0f, 0.0f, 2.5f);
        cam.look_at  = make_f3(0.0f, 0.0f, 0.0f);
        cam.up       = make_f3(0.0f, 1.0f, 0.0f);
        cam.fov_deg  = 40.0f;
        cam.width    = w;
        cam.height   = h;
        cam.update();
        return cam;
    }
};
