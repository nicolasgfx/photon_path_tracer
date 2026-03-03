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

    print(f"  Processed {len(all_textures)} texture(s) → {output_tex_dir}")


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
        img_array = np.array(img, dtype=np.float32)

        if img_array.ndim == 3 and img_array.shape[2] == 4:
            # RGBA — scale RGB only, preserve alpha
            img_array[:, :, :3] *= scale
        elif img_array.ndim == 3:
            img_array *= scale
        else:
            img_array *= scale

        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        result = Image.fromarray(img_array)
        result.save(dst_path, 'PNG')
        print(f"  Baked: {os.path.basename(src_path)} × {scale:.3f} → {os.path.basename(dst_path)}")
    except Exception as e:
        print(f"  [WARN] Failed to bake {src_path}: {e}")
        shutil.copy2(src_path, dst_path)
