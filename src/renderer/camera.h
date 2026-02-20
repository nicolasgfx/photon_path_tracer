#pragma once
// ─────────────────────────────────────────────────────────────────────
// camera.h – Thin-lens camera model (pinhole when lens_radius == 0)
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/config.h"
#include "core/random.h"

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
    float  dof_focus_range = DEFAULT_DOF_FOCUS_RANGE;    // 0 = thin plane, >0 = in-focus slab depth

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
    Ray generate_ray(int px, int py, PCGRng& rng) const {
        float s = ((float)px + rng.next_float()) / (float)width;
        float t = ((float)py + rng.next_float()) / (float)height;

        // Focus-plane target (same whether DOF is on or off)
        float3 target = lower_left + horizontal * s + vertical * t;

        Ray ray;
        if (lens_radius > 0.f) {
            // Thin-lens: sample a point on the circular aperture
            float2 disk = sample_concentric_disk(rng.next_float(), rng.next_float());
            float3 lens_offset = (u * disk.x + v * disk.y) * lens_radius;
            ray.origin    = position + lens_offset;
            ray.direction = normalize(target - ray.origin);
        } else {
            // Pinhole (original behaviour)
            ray.origin    = position;
            ray.direction = normalize(target - position);
        }
        return ray;
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
