#pragma once

// optix_camera.cuh – camera ray generation from OptiX launch params

// =====================================================================
// Camera ray helper — calls the unified generate_camera_ray() from
// camera.h, populating args from OptiX launch params.
// =====================================================================
__forceinline__ __device__
void generate_camera_ray_from_params(int px, int py, PCGRng& rng,
                                     float3& origin, float3& direction,
                                     int sample_index = -1)
{
    Ray ray = generate_camera_ray(
        px, py, rng,
        params.width, params.height,
        params.cam_lower_left, params.cam_horizontal, params.cam_vertical,
        params.cam_pos, params.cam_u, params.cam_v,
        params.cam_lens_radius, params.cam_focus_dist, params.cam_focus_range,
        sample_index);  // v3: always stratified when sample_index >= 0
    origin    = ray.origin;
    direction = ray.direction;
}
