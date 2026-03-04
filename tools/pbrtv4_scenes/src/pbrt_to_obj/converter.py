"""
Converter orchestrator â€” ties together parsing, PLY loading, material mapping,
texture processing, OBJ writing, and camera extraction.

Memory-efficient streaming OBJ writer: reads each PLY, transforms, writes
directly to the output file, then frees the mesh data before processing
the next shape. Peak memory â‰ˆ one mesh at a time.
"""

from __future__ import annotations
import gc
import math
import os
import re
from pathlib import Path

import numpy as np

from .pbrt_parser import (
    PbrtParser, PbrtScene, PbrtShape, PbrtLight,
    get_param, get_param_type,
)
from .ply_reader import read_ply, PlyMesh
from .material_mapper import (
    MtlMaterial, ResolvedTexture, TextureResolver,
    map_materials, map_inline_material, create_emissive_material,
    blackbody_to_rgb,
)
from .texture_processor import process_textures, copy_env_map
from .obj_writer import write_mtl
from .camera_extractor import extract_camera_json
from .sphere_gen import generate_sphere


def convert_scene(scene_pbrt_path: str, output_dir: str, *,
                   verbose: bool = True, include_instances: bool = False,
                   z_up: bool = False):
    """Convert a single PBRT scene file to OBJ + MTL + camera.json.

    Parameters
    ----------
    scene_pbrt_path    : absolute path to the top-level .pbrt file
    output_dir         : absolute path to output folder (will be created)
    verbose            : print progress
    include_instances  : if False (default), skip ObjectInstance-expanded shapes
    z_up               : if True, rotate geometry from Z-up to Y-up
    """
    scene_pbrt_path = os.path.abspath(scene_pbrt_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Converting: {scene_pbrt_path}")
        print(f"Output:     {output_dir}")
        print(f"{'='*60}")

    # --- Step 1: Parse PBRT scene ---
    if verbose:
        print("\n[1/6] Parsing PBRT scene...")
    parser = PbrtParser()
    scene = parser.parse_file(scene_pbrt_path)

    n_direct = sum(1 for s in scene.shapes if not s.from_instance)
    n_instanced = sum(1 for s in scene.shapes if s.from_instance)
    if verbose:
        print(f"  Named materials: {len(scene.named_materials)}")
        print(f"  Textures: {len(scene.textures)}")
        print(f"  Shapes: {n_direct} direct + {n_instanced} instanced"
              f"{'' if include_instances else ' (skipped)'}")
        print(f"  Object templates: {len(scene.object_templates)}")
        print(f"  Lights: {len(scene.lights)}")
        print(f"  Global transform diagonal: [{scene.global_transform[0,0]:.1f}, "
              f"{scene.global_transform[1,1]:.1f}, {scene.global_transform[2,2]:.1f}]")

    # --- Step 2: Map materials ---
    if verbose:
        print("\n[2/6] Mapping materials...")
    materials, all_textures = map_materials(scene)

    # Build texture resolver (needed for alpha textures below)
    resolver = TextureResolver(scene.textures, scene.source_dir)

    # Handle emissive shapes (AreaLightSource)
    _emissive_counter = 0
    for shape in scene.shapes:
        if not include_instances and shape.from_instance:
            continue
        if shape.area_light is not None:
            em_name = f"_emissive_{_emissive_counter}"
            em_mat = create_emissive_material(em_name, shape.area_light['params'])
            # Preserve base material Kd if it exists (emissive shapes often
            # have a base material whose diffuse colour we should keep)
            base_mat_name = shape.material_name
            if base_mat_name and base_mat_name in materials:
                base = materials[base_mat_name]
                if base.map_Kd:
                    em_mat.map_Kd = base.map_Kd
                if any(v > 0.01 for v in base.Kd):
                    em_mat.Kd = list(base.Kd)
            materials[em_name] = em_mat
            shape.material_name = em_name
            _emissive_counter += 1

    # Ensure a "default" material exists for shapes with no material assignment
    _needs_default = any(
        s.material_name is None and s.inline_material is None and s.area_light is None
        for s in scene.shapes
        if include_instances or not s.from_instance
    )
    if _needs_default and 'default' not in materials:
        materials['default'] = MtlMaterial(
            name='default',
            Kd=[0.5, 0.5, 0.5],
            pb_brdf='lambert',
            comments=['# Fallback for shapes with no material assignment'],
        )

    # Resolve alpha textures from shape params (for leaves)
    for shape in scene.shapes:
        if not include_instances and shape.from_instance:
            continue
        alpha_type = get_param_type(shape.params, 'alpha')
        if alpha_type == 'texture':
            alpha_tex_name = get_param(shape.params, 'alpha')
            rt = resolver.resolve(alpha_tex_name)
            if rt:
                # Register the alpha texture
                base = os.path.basename(rt.filepath)
                name_part, ext = os.path.splitext(base)
                out_name = f"{name_part}_alpha{ext}"
                all_textures[out_name] = ResolvedTexture(rt.filepath, rt.scale, is_alpha=True)
                # Set on the material
                mat_name = shape.material_name
                if mat_name and mat_name in materials:
                    if materials[mat_name].map_d is None:
                        materials[mat_name].map_d = f"textures/{out_name}"

    if verbose:
        print(f"  Total materials: {len(materials)}")
        for name, mat in materials.items():
            pb = mat.pb_brdf or '?'
            print(f"    {name}: {pb}")

    # --- Step 3: Process textures ---
    if verbose:
        print("\n[3/6] Processing textures...")
    tex_dir = os.path.join(output_dir, "textures")
    process_textures(all_textures, scene.source_dir, tex_dir)

    # Copy environment map
    env_map_out = None
    for light in scene.lights:
        if light.light_type == 'infinite':
            fn = get_param(light.params, 'filename')
            if fn:
                env_map_out = copy_env_map(fn, scene.source_dir, tex_dir)

    # --- Step 4: Load PLY meshes and write OBJ (streaming, low-memory) ---
    if verbose:
        print("\n[4/6] Loading PLY meshes and writing OBJ (streaming)...")

    scene_name = Path(scene_pbrt_path).stem
    obj_path = os.path.join(output_dir, f"{scene_name}.obj")
    mtl_path = os.path.join(output_dir, f"{scene_name}.mtl")

    global_xform = np.eye(4, dtype=np.float64)
    if z_up:
        # 90° rotation around X: (x,y,z) → (x, z, -y)
        # This converts Z-up (ground = XY) to Y-up (ground = XZ)
        global_xform = np.array([
            [1,  0,  0,  0],
            [0,  0, -1,  0],
            [0,  1,  0,  0],
            [0,  0,  0,  1],
        ], dtype=np.float64)
        if verbose:
            print("  [Z-up] Applying 90 deg X rotation (Z-up -> Y-up)")
    bbox_min = np.array([1e30, 1e30, 1e30])
    bbox_max = np.array([-1e30, -1e30, -1e30])

    _group_names: dict[str, int] = {}

    skipped = 0
    loaded = 0
    total_verts = 0
    total_faces = 0

    # Global vertex offsets for OBJ (1-based)
    v_offset = 0
    vn_offset = 0
    vt_offset = 0

    # Open in text mode with Unix line endings; no BytesIO intermediate buffers.
    with open(obj_path, 'w', newline='\n', buffering=1_048_576) as obj_f:
      try:
        obj_f.write(f"# Generated by pbrt_to_obj converter\nmtllib {scene_name}.mtl\n\n")

        for shape_idx, shape in enumerate(scene.shapes):
            # Skip ObjectInstance-expanded shapes unless opted in
            if not include_instances and shape.from_instance:
                skipped += 1
                continue

            if shape.shape_type == 'plymesh':
                filename = get_param(shape.params, 'filename')
                if not filename:
                    skipped += 1
                    continue

                ply_path = os.path.normpath(os.path.join(scene.source_dir, filename))
                if not os.path.isfile(ply_path):
                    if verbose:
                        print(f"  [WARN] PLY not found: {ply_path}")
                    skipped += 1
                    continue

                # Read PLY fresh each time (no cache â€” keeps peak memory to
                # one mesh).  Disk I/O is fast; the bottleneck is text formatting.
                try:
                    mesh = read_ply(ply_path)
                except Exception as e:
                    if verbose:
                        print(f"  [WARN] Failed to read PLY {ply_path}: {e}")
                    skipped += 1
                    continue

                loaded += 1

                # Transform vertices: shape.transform (instance) then global_xform
                combined_xform = global_xform @ shape.transform
                positions, normals, faces = _transform_mesh(
                    mesh.positions, mesh.normals, mesh.faces, combined_xform,
                    reverse_orientation=shape.reverse_orientation,
                )
                uvs = mesh.uvs          # keep UV reference
                del mesh                 # free raw PLY data immediately

                # Bounding box
                bbox_min = np.minimum(bbox_min, positions.min(axis=0))
                bbox_max = np.maximum(bbox_max, positions.max(axis=0))

                # Group header
                mat_name = shape.material_name or "default"
                base_name = os.path.splitext(os.path.basename(filename))[0]
                group_name = _unique_name(f"{base_name}_{mat_name}", _group_names)

                obj_f.write(f"g {group_name}\nusemtl {mat_name}\n")

                # --- Vertices: write directly to file (no BytesIO) ---
                np.savetxt(obj_f, positions, fmt='v %.6f %.6f %.6f')

                # --- Texture coordinates ---
                has_uv = uvs is not None
                if has_uv:
                    np.savetxt(obj_f, uvs, fmt='vt %.6f %.6f')

                # --- Normals ---
                has_n = normals is not None
                if has_n:
                    np.savetxt(obj_f, normals, fmt='vn %.6f %.6f %.6f')

                # --- Faces ---
                nv = positions.shape[0]
                nf = faces.shape[0]
                fi = faces.astype(np.int64)
                del positions, normals, uvs    # free before building face array

                if has_uv and has_n:
                    v = fi + v_offset + 1
                    t = fi + vt_offset + 1
                    n = fi + vn_offset + 1
                    arr = np.column_stack([v[:,0],t[:,0],n[:,0],
                                           v[:,1],t[:,1],n[:,1],
                                           v[:,2],t[:,2],n[:,2]])
                    np.savetxt(obj_f, arr, fmt='f %d/%d/%d %d/%d/%d %d/%d/%d')
                    del arr, v, t, n
                elif has_n:
                    v = fi + v_offset + 1
                    n = fi + vn_offset + 1
                    arr = np.column_stack([v[:,0],n[:,0],
                                           v[:,1],n[:,1],
                                           v[:,2],n[:,2]])
                    np.savetxt(obj_f, arr, fmt='f %d//%d %d//%d %d//%d')
                    del arr, v, n
                elif has_uv:
                    v = fi + v_offset + 1
                    t = fi + vt_offset + 1
                    arr = np.column_stack([v[:,0],t[:,0],
                                           v[:,1],t[:,1],
                                           v[:,2],t[:,2]])
                    np.savetxt(obj_f, arr, fmt='f %d/%d %d/%d %d/%d')
                    del arr, v, t
                else:
                    v = fi + v_offset + 1
                    np.savetxt(obj_f, v, fmt='f %d %d %d')
                    del v

                del fi, faces

                v_offset += nv
                if has_uv:  vt_offset += nv
                if has_n:   vn_offset += nv
                total_verts += nv
                total_faces += nf
                obj_f.write("\n")

                # Progress every 50 meshes; free unreachable objects
                if loaded % 50 == 0:
                    obj_f.flush()
                    gc.collect()
                    if verbose:
                        print(f"  ... {loaded}/{len(scene.shapes)} meshes, "
                              f"{total_verts:,} verts, {total_faces:,} faces")

            elif shape.shape_type == 'sphere':
                # Generate sphere mesh (for area lights)
                radius = get_param(shape.params, 'radius', 1.0)
                combined_xform = global_xform @ shape.transform
                center = combined_xform[:3, 3]
                # Use geometric mean of axis scales for uniform sphere approximation
                sx = np.linalg.norm(combined_xform[:3, 0])
                sy = np.linalg.norm(combined_xform[:3, 1])
                sz = np.linalg.norm(combined_xform[:3, 2])
                scale = (sx * sy * sz) ** (1.0 / 3.0)

                positions, normals, uvs, faces = generate_sphere(
                    (center[0], center[1], center[2]),
                    float(radius * scale),
                    n_lat=12,
                    n_lon=24,
                )

                # Handle ReverseOrientation for sphere
                if shape.reverse_orientation:
                    faces = faces.copy()
                    faces[:, [1, 2]] = faces[:, [2, 1]]

                bbox_min = np.minimum(bbox_min, positions.min(axis=0))
                bbox_max = np.maximum(bbox_max, positions.max(axis=0))

                mat_name = shape.material_name or "default"
                group_name = _unique_name(f"sphere_{mat_name}", _group_names)

                obj_f.write(f"g {group_name}\nusemtl {mat_name}\n")
                np.savetxt(obj_f, positions, fmt='v %.6f %.6f %.6f')
                np.savetxt(obj_f, uvs, fmt='vt %.6f %.6f')
                np.savetxt(obj_f, normals, fmt='vn %.6f %.6f %.6f')

                nv = positions.shape[0]
                fi = faces.astype(np.int64)
                v = fi + v_offset + 1
                t = fi + vt_offset + 1
                n = fi + vn_offset + 1
                arr = np.column_stack([v[:,0],t[:,0],n[:,0],
                                       v[:,1],t[:,1],n[:,1],
                                       v[:,2],t[:,2],n[:,2]])
                np.savetxt(obj_f, arr, fmt='f %d/%d/%d %d/%d/%d %d/%d/%d')

                nf = faces.shape[0]
                del arr, v, t, n, fi, positions, normals, uvs, faces

                v_offset += nv; vt_offset += nv; vn_offset += nv
                total_verts += nv; total_faces += nf
                obj_f.write("\n")
                loaded += 1

            elif shape.shape_type == 'trianglemesh':
                # Inline triangle mesh (common for emissive area lights)
                raw_P = get_param(shape.params, 'P')
                raw_indices = get_param(shape.params, 'indices')
                if raw_P is None or raw_indices is None:
                    skipped += 1
                    continue

                positions = np.array(raw_P, dtype=np.float32).reshape(-1, 3)
                faces = np.array(raw_indices, dtype=np.int32).reshape(-1, 3)

                raw_N = get_param(shape.params, 'N')
                normals = None
                if raw_N is not None:
                    normals = np.array(raw_N, dtype=np.float32).reshape(-1, 3)

                raw_uv = get_param(shape.params, 'uv')
                if raw_uv is None:
                    raw_uv = get_param(shape.params, 'st')  # PBRT also uses 'st' for UVs
                uvs = None
                if raw_uv is not None:
                    uvs = np.array(raw_uv, dtype=np.float32).reshape(-1, 2)

                # Apply transform
                combined_xform = global_xform @ shape.transform
                positions, normals, faces = _transform_mesh(
                    positions, normals, faces, combined_xform,
                    reverse_orientation=shape.reverse_orientation,
                )

                bbox_min = np.minimum(bbox_min, positions.min(axis=0))
                bbox_max = np.maximum(bbox_max, positions.max(axis=0))

                mat_name = shape.material_name or "default"
                group_name = _unique_name(f"trimesh_{mat_name}", _group_names)

                obj_f.write(f"g {group_name}\nusemtl {mat_name}\n")
                np.savetxt(obj_f, positions, fmt='v %.6f %.6f %.6f')

                has_uv = uvs is not None
                has_n = normals is not None
                if has_uv:
                    np.savetxt(obj_f, uvs, fmt='vt %.6f %.6f')
                if has_n:
                    np.savetxt(obj_f, normals, fmt='vn %.6f %.6f %.6f')

                nv = positions.shape[0]
                nf = faces.shape[0]
                fi = faces.astype(np.int64)
                del positions, normals, uvs

                if has_uv and has_n:
                    v = fi + v_offset + 1
                    t = fi + vt_offset + 1
                    n = fi + vn_offset + 1
                    arr = np.column_stack([v[:,0],t[:,0],n[:,0],
                                           v[:,1],t[:,1],n[:,1],
                                           v[:,2],t[:,2],n[:,2]])
                    np.savetxt(obj_f, arr, fmt='f %d/%d/%d %d/%d/%d %d/%d/%d')
                    del arr, v, t, n
                elif has_n:
                    v = fi + v_offset + 1
                    n = fi + vn_offset + 1
                    arr = np.column_stack([v[:,0],n[:,0],
                                           v[:,1],n[:,1],
                                           v[:,2],n[:,2]])
                    np.savetxt(obj_f, arr, fmt='f %d//%d %d//%d %d//%d')
                    del arr, v, n
                elif has_uv:
                    v = fi + v_offset + 1
                    t = fi + vt_offset + 1
                    arr = np.column_stack([v[:,0],t[:,0],
                                           v[:,1],t[:,1],
                                           v[:,2],t[:,2]])
                    np.savetxt(obj_f, arr, fmt='f %d/%d %d/%d %d/%d')
                    del arr, v, t
                else:
                    v = fi + v_offset + 1
                    np.savetxt(obj_f, v, fmt='f %d %d %d')
                    del v

                del fi, faces
                v_offset += nv
                if has_uv:  vt_offset += nv
                if has_n:   vn_offset += nv
                total_verts += nv
                total_faces += nf
                obj_f.write("\n")
                loaded += 1

      except Exception as e:
        import traceback
        print(f"\n  [ERROR] OBJ writing failed at shape {shape_idx}/{len(scene.shapes)}: {e}")
        traceback.print_exc()

    gc.collect()

    if verbose:
        print(f"  Loaded {loaded} mesh(es), skipped {skipped}")
        print(f"  Total: {total_verts:,} vertices, {total_faces:,} faces")
        print(f"  Bounding box: ({bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}) "
              f"â†’ ({bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f})")
        obj_size = os.path.getsize(obj_path)
        if obj_size > 500_000_000:
            print(f"  OBJ: {obj_size/1e9:.2f} GB â†’ {obj_path}")
        else:
            print(f"  OBJ: {obj_size/1e6:.2f} MB â†’ {obj_path}")

    # --- Step 5: Write MTL ---
    if verbose:
        print("\n[5/6] Writing MTL...")

    write_mtl(mtl_path, materials)

    # --- Step 6: Write camera JSON ---
    if verbose:
        print("\n[6/6] Extracting camera...")

    cam_path = os.path.join(output_dir, "camera.json")
    scene_bounds = (bbox_min.tolist(), bbox_max.tolist())
    extract_camera_json(scene, cam_path, env_map_out, scene_bounds,
                        z_up_rotation=global_xform if z_up else None)

    if verbose:
        print(f"\n{'='*60}")
        print(f"DONE: {output_dir}")
        print(f"{'='*60}\n")


def convert_scene_folder(folder_path: str, output_base: str, *,
                          verbose: bool = True, include_instances: bool = False,
                          z_up: bool = False):
    """Auto-discover scene .pbrt files in a folder and convert each.

    Skips files that are known includes (materials.pbrt, geometry.pbrt).
    Creates one output subfolder per scene file.
    """
    folder_path = os.path.abspath(folder_path)
    output_base = os.path.abspath(output_base)

    # Find all .pbrt files in the folder
    all_pbrt = sorted(Path(folder_path).glob("*.pbrt"))
    
    # Filter out include-only files
    skip_names = {'materials.pbrt', 'geometry.pbrt'}
    scene_files = [p for p in all_pbrt if p.name.lower() not in skip_names]

    if not scene_files:
        print(f"No scene .pbrt files found in {folder_path}")
        return

    folder_name = os.path.basename(folder_path)

    print(f"Found {len(scene_files)} scene(s) in {folder_path}:")
    for sf in scene_files:
        print(f"  - {sf.name}")

    for sf in scene_files:
        stem = sf.stem
        out_dir = os.path.join(output_base, f"{folder_name}-{stem}")
        convert_scene(str(sf), out_dir, verbose=verbose,
                      include_instances=include_instances, z_up=z_up)


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def _transform_mesh(positions: np.ndarray,
                    normals: np.ndarray | None,
                    faces: np.ndarray,
                    xform: np.ndarray,
                    reverse_orientation: bool = False) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Apply 4x4 transform to mesh vertices and normals.
    
    If the transform has a negative determinant (mirror) XOR reverse_orientation
    is True, reverses face winding to preserve correct orientation.
    """
    n_verts = positions.shape[0]

    # Transform positions
    ones = np.ones((n_verts, 1), dtype=np.float32)
    pos4 = np.hstack([positions, ones])  # (N, 4)
    xf = xform.astype(np.float32)
    transformed = (xf @ pos4.T).T  # (N, 4)
    new_positions = transformed[:, :3].copy()

    # Transform normals (using inverse-transpose of upper-left 3x3)
    new_normals = None
    if normals is not None:
        normal_mat = np.linalg.inv(xform[:3, :3]).T.astype(np.float32)
        new_normals = (normal_mat @ normals.T).T.copy()
        # Re-normalise
        lens = np.linalg.norm(new_normals, axis=1, keepdims=True)
        lens = np.maximum(lens, 1e-8)
        new_normals /= lens

    # Check determinant â€” if negative, reverse face winding
    # Also flip if ReverseOrientation was set (XOR logic)
    det = np.linalg.det(xform[:3, :3])
    flip = (det < 0) ^ reverse_orientation
    if flip:
        # Reverse winding: swap columns 1 and 2
        faces = faces.copy()
        faces[:, [1, 2]] = faces[:, [2, 1]]

    return new_positions, new_normals, faces


def _unique_name(base: str, registry: dict[str, int]) -> str:
    """Generate a unique group name."""
    # Sanitise for OBJ (no spaces)
    clean = re.sub(r'[^a-zA-Z0-9_.\-]', '_', base)
    if clean not in registry:
        registry[clean] = 0
        return clean
    registry[clean] += 1
    name = f"{clean}_{registry[clean]}"
    return name
