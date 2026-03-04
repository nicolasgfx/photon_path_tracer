"""
CLI entry point for the PBRT-to-OBJ converter.

Usage
-----
Convert all scenes in a PBRT folder:
    python -m pbrt_to_obj --scene pbrt-v4-scenes/barcelona-pavilion --output output/

Convert a single .pbrt file:
    python -m pbrt_to_obj --scene pbrt-v4-scenes/barcelona-pavilion/pavilion-day.pbrt --output output/

Options:
    --scene   Path to a PBRT scene folder or a single .pbrt file
    --output  Base output directory (default: ./output)
    --quiet   Suppress progress output
"""

from __future__ import annotations
import argparse
import os
import sys

from .converter import convert_scene, convert_scene_folder


def main():
    parser = argparse.ArgumentParser(
        prog="pbrt_to_obj",
        description="Convert PBRT v4 scenes to Wavefront OBJ + MTL with pb_* extensions",
    )
    parser.add_argument(
        "--scene", "-s",
        required=True,
        help="Path to a PBRT scene folder or a single .pbrt file",
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Base output directory (default: output/)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--include-instances",
        action="store_true",
        default=False,
        help="Include ObjectInstance-expanded shapes (large tree meshes). "
             "Off by default to keep output small.",
    )
    parser.add_argument(
        "--zup",
        action="store_true",
        default=False,
        help="Rotate geometry from Z-up to Y-up (90° around X). "
             "Use for PBRT scenes where the ground plane is XY.",
    )

    args = parser.parse_args()
    scene_path = os.path.abspath(args.scene)
    output_path = os.path.abspath(args.output)
    verbose = not args.quiet
    include_instances = args.include_instances
    z_up = args.zup

    if not os.path.exists(scene_path):
        print(f"Error: path not found: {scene_path}", file=sys.stderr)
        sys.exit(1)

    if os.path.isdir(scene_path):
        convert_scene_folder(scene_path, output_path, verbose=verbose,
                             include_instances=include_instances, z_up=z_up)
    elif os.path.isfile(scene_path) and scene_path.endswith('.pbrt'):
        # Single file — derive output folder name
        folder_name = os.path.basename(os.path.dirname(scene_path))
        stem = os.path.splitext(os.path.basename(scene_path))[0]
        out_dir = os.path.join(output_path, f"{folder_name}-{stem}")
        convert_scene(scene_path, out_dir, verbose=verbose,
                      include_instances=include_instances, z_up=z_up)
    else:
        print(f"Error: not a .pbrt file or directory: {scene_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
