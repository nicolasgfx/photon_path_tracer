#pragma once
// ─────────────────────────────────────────────────────────────────────
// camera.h – Pinhole camera model
// ─────────────────────────────────────────────────────────────────────
#include "core/types.h"
#include "core/random.h"

struct Camera {
    float3 position;
    float3 look_at;
    float3 up;

    float  fov_deg;    // Vertical field of view in degrees
    int    width;
    int    height;

    // Derived (call update() after modifying parameters)
    float3 u, v, w;   // Camera frame: w = -look_direction
    float3 lower_left;
    float3 horizontal;
    float3 vertical;

    void update() {
        float aspect = (float)width / (float)height;
        float theta  = fov_deg * PI / 180.0f;
        float h = tanf(theta * 0.5f);
        float viewport_h = 2.0f * h;
        float viewport_w = aspect * viewport_h;

        w = normalize(position - look_at);
        u = normalize(cross(up, w));
        v = cross(w, u);

        horizontal = u * viewport_w;
        vertical   = v * viewport_h;
        lower_left = position - horizontal * 0.5f - vertical * 0.5f - w;
    }

    // Generate a ray for pixel (px, py) with jitter for anti-aliasing
    Ray generate_ray(int px, int py, PCGRng& rng) const {
        float s = ((float)px + rng.next_float()) / (float)width;
        float t = ((float)py + rng.next_float()) / (float)height;

        Ray ray;
        ray.origin    = position;
        ray.direction = normalize(lower_left + horizontal * s + vertical * t - position);
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
