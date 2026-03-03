"""
Sphere mesh generator.

Generates UV-sphere meshes for PBRT area-light spheres that don't exist
as PLY files.
"""

from __future__ import annotations
import math
import numpy as np


def generate_sphere(center: tuple[float, float, float],
                    radius: float,
                    n_lat: int = 12,
                    n_lon: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a UV sphere mesh.
    
    Returns (positions, normals, uvs, faces) — all numpy arrays.
    positions: (N, 3) float32
    normals:   (N, 3) float32
    uvs:       (N, 2) float32
    faces:     (M, 3) int32
    """
    verts = []
    norms = []
    texcoords = []
    
    for lat in range(n_lat + 1):
        theta = math.pi * lat / n_lat
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        
        for lon in range(n_lon + 1):
            phi = 2 * math.pi * lon / n_lon
            sin_p = math.sin(phi)
            cos_p = math.cos(phi)
            
            nx = sin_t * cos_p
            ny = cos_t
            nz = sin_t * sin_p
            
            x = center[0] + radius * nx
            y = center[1] + radius * ny
            z = center[2] + radius * nz
            
            u = lon / n_lon
            v = lat / n_lat
            
            verts.append((x, y, z))
            norms.append((nx, ny, nz))
            texcoords.append((u, v))
    
    faces = []
    for lat in range(n_lat):
        for lon in range(n_lon):
            i0 = lat * (n_lon + 1) + lon
            i1 = i0 + 1
            i2 = i0 + (n_lon + 1)
            i3 = i2 + 1
            
            if lat != 0:
                faces.append((i0, i2, i1))
            if lat != n_lat - 1:
                faces.append((i1, i2, i3))
    
    return (
        np.array(verts, dtype=np.float32),
        np.array(norms, dtype=np.float32),
        np.array(texcoords, dtype=np.float32),
        np.array(faces, dtype=np.int32),
    )
