"""
Camera and lighting extractor.

Extracts camera parameters (position, look-at, FOV, DOF, film) from the
PBRT scene and writes an extended camera.json compatible with the photon-beam
renderer.  Also extracts light sources for the camera JSON.
"""

from __future__ import annotations
import json
import math
import os

import numpy as np

from .pbrt_parser import PbrtScene, PbrtLight, PbrtShape, get_param
from .material_mapper import blackbody_to_rgb


def extract_camera_json(scene: PbrtScene, output_path: str,
                        env_map_path: str | None = None,
                        scene_bounds: tuple[list[float], list[float]] | None = None):
    """Extract camera + lighting metadata and write camera.json.
    
    Parameters
    ----------
    scene        : parsed PBRT scene
    output_path  : path to write camera.json
    env_map_path : relative path to the environment map (or None)
    scene_bounds : (bbox_min, bbox_max) in world space (or None)
    """
    cam = scene.camera
    film = scene.film

    # --- Position and look-at ---
    # Geometry stays in PBRT world space (converter no longer applies the
    # pre-WorldBegin transform).  Derive camera entirely from the scene
    # description rather than baking it into geometry.
    if cam.look_at is not None:
        # LookAt eye/target/up are already in world space.
        eye, target, up = cam.look_at
        up_transformed = up.copy()
    else:
        # No LookAt — derive camera from the pre-WorldBegin Transform.
        # That matrix is the world-to-camera transform M_w2c.
        # Camera-to-world = inv(M_w2c); eye = last column, forward = -Z col.
        M_w2c = cam.pre_transform
        if not np.allclose(M_w2c, np.eye(4)):
            M_c2w = np.linalg.inv(M_w2c)
            eye = M_c2w[:3, 3].copy()
            forward = -M_c2w[:3, 2]
            mag = np.linalg.norm(forward)
            if mag > 1e-12:
                forward /= mag
            up_raw = M_c2w[:3, 1]
            mag2 = np.linalg.norm(up_raw)
            if mag2 > 1e-12:
                up_raw /= mag2
            target = eye + forward
            up_transformed = up_raw
        else:
            eye = np.array([0.0, 0.0, 0.0])
            target = np.array([0.0, 0.0, -1.0])
            up_transformed = np.array([0.0, 1.0, 0.0])

    # --- FOV ---
    fov = get_param(cam.params, 'fov', 45.0)

    # --- Film ---
    xres = get_param(film.params, 'xresolution', 1600)
    yres = get_param(film.params, 'yresolution', 850)
    iso = get_param(film.params, 'iso', 100)

    # --- DOF ---
    lens_radius = get_param(cam.params, 'lensradius', 0.0)
    focal_distance = get_param(cam.params, 'focaldistance', 0.0)

    # --- Yaw / Pitch from direction ---
    direction = target - eye
    dist = np.linalg.norm(direction)
    if dist > 1e-12:
        direction /= dist
    yaw = math.atan2(direction[0], direction[2])   # around Y axis
    pitch = math.asin(max(-1, min(1, direction[1])))  # elevation

    # --- Light scale ---
    # Estimate from PBRT film ISO (relative to 100 base)
    light_scale = 100.0 / max(iso, 1)

    # --- Environment map ---
    env_rotation_deg = [0.0, 0.0, 0.0]
    env_scale = 1.0

    for light in scene.lights:
        if light.light_type == 'infinite':
            env_scale_param = get_param(light.params, 'scale', 1.0)
            if isinstance(env_scale_param, list):
                env_scale = env_scale_param[0]
            else:
                env_scale = float(env_scale_param)
            # Extract rotation from the light's transform
            env_rotation_deg = _extract_env_rotation(light.transform)

    # --- Area lights (night scene) ---
    lights_array = []
    for light in scene.lights:
        if light.light_type != 'infinite':
            continue
    
    # Check shapes with area_light for emissive geometry
    for shape in scene.shapes:
        if shape.area_light:
            light_info = _extract_area_light(shape)
            if light_info:
                lights_array.append(light_info)

    # Also extract standalone area lights from scene lights that have shapes
    # (these are the sphere lights in the night scene)
    for light in scene.lights:
        if light.light_type == 'infinite':
            continue
        # For lights attached to shapes (caught above), skip
        # Light info is embedded in shape.area_light

    # --- Build JSON ---
    camera_data = {
        "position": [round(v, 6) for v in eye.tolist()],
        "look_at": [round(v, 6) for v in target.tolist()],
        "up": [round(v, 6) for v in up_transformed.tolist()],
        "fov_deg": float(fov),
        "yaw": round(float(yaw), 6),
        "pitch": round(float(pitch), 6),
        "resolution": [int(xres), int(yres)],
        "lens_radius": float(lens_radius),
        "focal_distance": float(focal_distance),
        "iso": float(iso),
        "light_scale": round(float(light_scale), 6),
    }

    if env_map_path:
        camera_data["environment_map"] = env_map_path
        camera_data["environment_rotation_deg"] = [round(v, 2) for v in env_rotation_deg]
        camera_data["environment_scale"] = round(float(env_scale), 4)

    if lights_array:
        camera_data["lights"] = lights_array

    if scene_bounds:
        camera_data["scene_bounds_min"] = [round(v, 6) for v in scene_bounds[0]]
        camera_data["scene_bounds_max"] = [round(v, 6) for v in scene_bounds[1]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(camera_data, f, indent=2)

    print(f"  Camera: {output_path}")


def _apply_global(gt: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Apply 4×4 transform to a 3D point."""
    p4 = np.array([point[0], point[1], point[2], 1.0])
    result = gt @ p4
    return result[:3]


def _extract_env_rotation(transform: np.ndarray) -> list[float]:
    """Try to extract Euler rotation angles from the env light transform.
    
    PBRT Barcelona scene uses: Rotate -10 0 0 1; Rotate -160 0 1 0; Rotate -90 1 0 0
    We store these as [Z_deg, Y_deg, X_deg] for the user to interpret.
    
    Rather than decomposing the matrix (error-prone), we record the
    rotation from the transform matrix as best-effort Euler angles.
    """
    # Simple extraction: the rotation matrix is in the upper-left 3x3
    R = transform[:3, :3].copy()
    
    # Check for identity
    if np.allclose(R, np.eye(3), atol=1e-6):
        return [0.0, 0.0, 0.0]
    
    # ZYX Euler decomposition
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    
    return [round(math.degrees(z), 2), round(math.degrees(y), 2), round(math.degrees(x), 2)]


def _extract_area_light(shape: PbrtShape) -> dict | None:
    """Extract area light info from a shape with an attached AreaLightSource."""
    al = shape.area_light
    if al is None:
        return None

    params = al['params']
    xform = al['transform']  # light's world transform

    light_info: dict = {
        "type": shape.shape_type,  # "sphere" or "plymesh"
    }

    # Extract position from transform (already in world space)
    pos = np.array([xform[0, 3], xform[1, 3], xform[2, 3]])
    light_info["position"] = [round(v, 6) for v in pos.tolist()]

    # Sphere radius
    radius = get_param(shape.params, 'radius', None)
    if radius is not None:
        light_info["radius"] = float(radius)

    # Mesh filename
    filename = get_param(shape.params, 'filename', None)
    if filename is not None:
        light_info["mesh"] = filename

    # Temperature and scale
    scale = get_param(params, 'scale', 1.0)
    if isinstance(scale, list):
        scale = scale[0]
    light_info["scale"] = float(scale)

    # Blackbody
    for p in params:
        if p.type == 'blackbody' and p.name == 'L':
            temp_K = p.value[0] if isinstance(p.value, list) else float(p.value)
            light_info["temperature_K"] = float(temp_K)
            light_info["color_rgb"] = [round(v, 4) for v in blackbody_to_rgb(temp_K)]
            break
    else:
        # RGB emission
        L = get_param(params, 'L', None)
        if isinstance(L, list) and len(L) >= 3:
            light_info["color_rgb"] = [round(v, 4) for v in L[:3]]

    return light_info
