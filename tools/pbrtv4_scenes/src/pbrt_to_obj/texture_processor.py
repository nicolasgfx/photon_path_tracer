"""
Texture processor — copies and optionally bakes (scales) textures.

When a PBRT texture chain multiplies an imagemap by a scale factor
(e.g., concrete × 0.64), this module loads the source image, applies the
multiplicative scale to all RGB channels, and saves the result as a new PNG.
"""

from __future__ import annotations
import os
import shutil
from pathlib import Path

from .material_mapper import ResolvedTexture

# Attempt PIL import — graceful fallback if not available
try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def process_textures(all_textures: dict[str, ResolvedTexture],
                     source_dir: str,
                     output_tex_dir: str,
                     *,
                     copy_env_maps: bool = True):
    """Copy / bake all referenced textures to the output textures/ folder.

    Parameters
    ----------
    all_textures : dict  mapping output_filename → ResolvedTexture
    source_dir   : str   PBRT scene root directory (for resolving relative paths)
    output_tex_dir : str  absolute path to output textures/ folder
    copy_env_maps : bool  also copy EXR environment maps (passed separately)
    """
    os.makedirs(output_tex_dir, exist_ok=True)

    for out_name, rt in all_textures.items():
        src_path = _resolve_source_path(rt.filepath, source_dir)
        dst_path = os.path.join(output_tex_dir, out_name)

        if not os.path.isfile(src_path):
            print(f"  [WARN] Texture not found: {src_path}")
            continue

        if abs(rt.scale - 1.0) < 0.001:
            # Straight copy
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
        else:
            # Bake scale factor
            _bake_scaled_texture(src_path, dst_path, rt.scale)

    print(f"  Processed {len(all_textures)} texture(s) -> {output_tex_dir}")


def copy_env_map(filepath: str, source_dir: str, output_tex_dir: str) -> str | None:
    """Copy an EXR / HDR environment map to output.  Returns output relative path."""
    src_path = _resolve_source_path(filepath, source_dir)
    if not os.path.isfile(src_path):
        print(f"  [WARN] Env map not found: {src_path}")
        return None
    os.makedirs(output_tex_dir, exist_ok=True)
    out_name = os.path.basename(filepath)
    dst_path = os.path.join(output_tex_dir, out_name)
    if not os.path.exists(dst_path):
        shutil.copy2(src_path, dst_path)
    return f"textures/{out_name}"


def _resolve_source_path(rel_path: str, source_dir: str) -> str:
    """Resolve a PBRT-relative texture path to an absolute path."""
    # Handle ../ references (e.g., ../landscape/textures/foo.png)
    return os.path.normpath(os.path.join(source_dir, rel_path))


def _bake_scaled_texture(src_path: str, dst_path: str, scale: float):
    """Load image, multiply RGB by scale, save as PNG."""
    if not HAS_PIL:
        print(f"  [WARN] PIL not available — copying unscaled: {os.path.basename(src_path)}")
        shutil.copy2(src_path, dst_path)
        return

    if os.path.exists(dst_path):
        return

    try:
        img = Image.open(src_path)
        src_ext = os.path.splitext(src_path)[1].lower()
        dst_ext = os.path.splitext(dst_path)[1].lower()

        # Determine if source is HDR format
        is_hdr = src_ext in ('.exr', '.hdr', '.pfm')

        if is_hdr:
            # For HDR textures, scale in linear float space and save as EXR/HDR
            # PIL can't handle EXR natively — just copy with a warning
            print(f"  [WARN] HDR texture scaling not supported, copying unscaled: {os.path.basename(src_path)}")
            shutil.copy2(src_path, dst_path)
            return

        img_array = np.array(img, dtype=np.float32)

        # Apply scale in linear space (approximate sRGB decode → scale → encode)
        # sRGB decode: linear = srgb^2.2  (approximate)
        if img_array.max() > 0:
            img_linear = np.power(img_array / 255.0, 2.2)

            if img_linear.ndim == 3 and img_linear.shape[2] == 4:
                # RGBA — scale RGB only, preserve alpha
                img_linear[:, :, :3] *= scale
            elif img_linear.ndim == 3:
                img_linear *= scale
            else:
                img_linear *= scale

            # sRGB encode back
            img_linear = np.clip(img_linear, 0.0, 1.0)
            img_array = np.power(img_linear, 1.0 / 2.2) * 255.0
        else:
            img_array *= scale

        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        result = Image.fromarray(img_array)
        result.save(dst_path, 'PNG')
        print(f"  Baked: {os.path.basename(src_path)} x {scale:.3f} -> {os.path.basename(dst_path)}")
    except Exception as e:
        print(f"  [WARN] Failed to bake {src_path}: {e}")
        shutil.copy2(src_path, dst_path)
