"""
Binary PLY mesh reader.

Supports binary_little_endian, binary_big_endian, and ASCII PLY files.
Extracts vertex positions, normals, texture coordinates, and face indices.
"""

from __future__ import annotations
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np


@dataclass
class PlyMesh:
    """Parsed PLY mesh data."""
    positions: np.ndarray    # (N, 3) float32
    normals: np.ndarray | None   # (N, 3) float32 or None
    uvs: np.ndarray | None       # (N, 2) float32 or None
    faces: np.ndarray        # (M, 3) int32 — triangulated


# PLY property type sizes
_PLY_TYPE_MAP = {
    'char': ('b', 1), 'int8': ('b', 1),
    'uchar': ('B', 1), 'uint8': ('B', 1),
    'short': ('h', 2), 'int16': ('h', 2),
    'ushort': ('H', 2), 'uint16': ('H', 2),
    'int': ('i', 4), 'int32': ('i', 4),
    'uint': ('I', 4), 'uint32': ('I', 4),
    'float': ('f', 4), 'float32': ('f', 4),
    'double': ('d', 8), 'float64': ('d', 8),
}


@dataclass
class _Property:
    name: str
    is_list: bool
    count_type: str  # for lists
    value_type: str


@dataclass
class _Element:
    name: str
    count: int
    properties: list[_Property]


def read_ply(filepath: str) -> PlyMesh:
    """Read a PLY file (binary or ASCII) and return a PlyMesh."""
    filepath = str(filepath)
    with open(filepath, 'rb') as f:
        header_lines, byte_order, elements = _parse_header(f)
        
        if byte_order == 'ascii':
            return _read_ascii(f, elements)
        else:
            endian = '<' if byte_order == 'binary_little_endian' else '>'
            return _read_binary(f, elements, endian)


def _parse_header(f: BinaryIO) -> tuple[list[str], str, list[_Element]]:
    """Parse PLY header, return (header_lines, format, elements)."""
    lines: list[str] = []
    elements: list[_Element] = []
    byte_order = 'binary_little_endian'
    
    while True:
        line = f.readline().decode('ascii', errors='replace').strip()
        lines.append(line)
        if line == 'end_header':
            break
        
        parts = line.split()
        if not parts:
            continue
        
        if parts[0] == 'format':
            byte_order = parts[1]
        elif parts[0] == 'element':
            elements.append(_Element(parts[1], int(parts[2]), []))
        elif parts[0] == 'property':
            if len(elements) == 0:
                continue
            if parts[1] == 'list':
                # property list <count_type> <value_type> <name>
                prop = _Property(parts[4], True, parts[2], parts[3])
            else:
                prop = _Property(parts[2], False, '', parts[1])
            elements[-1].properties.append(prop)
    
    return lines, byte_order, elements


def _read_binary(f: BinaryIO, elements: list[_Element], endian: str) -> PlyMesh:
    """Read binary PLY data."""
    vertex_data = None
    face_data = None
    
    for elem in elements:
        if elem.name == 'vertex':
            vertex_data = _read_binary_vertices(f, elem, endian)
        elif elem.name == 'face':
            face_data = _read_binary_faces(f, elem, endian)
        else:
            # Skip unknown elements
            _skip_binary_element(f, elem, endian)
    
    if vertex_data is None:
        raise ValueError("No vertex element found in PLY file")
    if face_data is None:
        raise ValueError("No face element found in PLY file")
    
    positions, normals, uvs = vertex_data
    return PlyMesh(positions, normals, uvs, face_data)


def _read_binary_vertices(f: BinaryIO, elem: _Element, endian: str
                          ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Read vertex element, extract positions, normals, UVs."""
    # Build struct format and index mapping
    fmt_parts = []
    prop_names = []
    for prop in elem.properties:
        if prop.is_list:
            raise ValueError("List property in vertex element not supported")
        code, _ = _PLY_TYPE_MAP[prop.value_type]
        fmt_parts.append(code)
        prop_names.append(prop.name)
    
    fmt = endian + ''.join(fmt_parts)
    row_size = struct.calcsize(fmt)
    
    # Read all vertices at once
    raw = f.read(row_size * elem.count)
    if len(raw) < row_size * elem.count:
        raise ValueError(f"Truncated PLY: expected {row_size * elem.count} bytes for vertices, got {len(raw)}")
    
    # Unpack
    rows = [struct.unpack_from(fmt, raw, i * row_size) for i in range(elem.count)]
    
    # Map property names to columns
    col = {name: i for i, name in enumerate(prop_names)}
    
    # Positions
    if 'x' not in col or 'y' not in col or 'z' not in col:
        raise ValueError("PLY vertex missing x/y/z properties")
    positions = np.array(
        [(r[col['x']], r[col['y']], r[col['z']]) for r in rows],
        dtype=np.float32,
    )
    
    # Normals
    normals = None
    if 'nx' in col and 'ny' in col and 'nz' in col:
        normals = np.array(
            [(r[col['nx']], r[col['ny']], r[col['nz']]) for r in rows],
            dtype=np.float32,
        )
    
    # UVs — try various naming conventions
    uvs = None
    u_name = next((n for n in ('u', 's', 'texture_u', 'texture_s') if n in col), None)
    v_name = next((n for n in ('v', 't', 'texture_v', 'texture_t') if n in col), None)
    if u_name and v_name:
        uvs = np.array(
            [(r[col[u_name]], r[col[v_name]]) for r in rows],
            dtype=np.float32,
        )
    
    return positions, normals, uvs


def _read_binary_faces(f: BinaryIO, elem: _Element, endian: str) -> np.ndarray:
    """Read face element, fan-triangulate polygons."""
    # Find the list property (vertex_indices / vertex_index)
    list_prop = None
    other_props = []
    for prop in elem.properties:
        if prop.is_list and prop.name in ('vertex_indices', 'vertex_index'):
            list_prop = prop
        elif prop.is_list:
            other_props.append(prop)
        else:
            other_props.append(prop)
    
    if list_prop is None:
        raise ValueError("No vertex_indices list property found in face element")
    
    count_code, count_size = _PLY_TYPE_MAP[list_prop.count_type]
    idx_code, idx_size = _PLY_TYPE_MAP[list_prop.value_type]
    
    count_fmt = endian + count_code
    
    triangles: list[tuple[int, int, int]] = []
    
    for _ in range(elem.count):
        # Read count
        raw_count = f.read(count_size)
        n_verts = struct.unpack(count_fmt, raw_count)[0]
        
        # Read indices
        idx_fmt = endian + idx_code * n_verts
        raw_idx = f.read(idx_size * n_verts)
        indices = struct.unpack(idx_fmt, raw_idx)
        
        # Skip any other properties on this face
        for op in other_props:
            if op.is_list:
                oc_code, oc_size = _PLY_TYPE_MAP[op.count_type]
                ov_code, ov_size = _PLY_TYPE_MAP[op.value_type]
                raw_oc = f.read(oc_size)
                on = struct.unpack(endian + oc_code, raw_oc)[0]
                f.read(ov_size * on)
            else:
                _, ps = _PLY_TYPE_MAP[op.value_type]
                f.read(ps)
        
        # Fan-triangulate
        for i in range(1, n_verts - 1):
            triangles.append((indices[0], indices[i], indices[i + 1]))
    
    return np.array(triangles, dtype=np.int32)


def _read_ascii(f: BinaryIO, elements: list[_Element]) -> PlyMesh:
    """Read ASCII PLY data."""
    # Read remaining text
    text = f.read().decode('ascii', errors='replace')
    lines = text.strip().split('\n')
    line_idx = 0
    
    vertex_data = None
    face_data = None
    
    for elem in elements:
        if elem.name == 'vertex':
            vertex_data, line_idx = _read_ascii_vertices(lines, line_idx, elem)
        elif elem.name == 'face':
            face_data, line_idx = _read_ascii_faces(lines, line_idx, elem)
        else:
            line_idx += elem.count  # skip
    
    if vertex_data is None:
        raise ValueError("No vertex element in ASCII PLY")
    if face_data is None:
        raise ValueError("No face element in ASCII PLY")
    
    positions, normals, uvs = vertex_data
    return PlyMesh(positions, normals, uvs, face_data)


def _read_ascii_vertices(lines: list[str], start: int, elem: _Element):
    prop_names = [p.name for p in elem.properties]
    col = {name: i for i, name in enumerate(prop_names)}
    
    positions_list = []
    normals_list = []
    uvs_list = []
    has_normals = 'nx' in col and 'ny' in col and 'nz' in col
    u_name = next((n for n in ('u', 's', 'texture_u', 'texture_s') if n in col), None)
    v_name = next((n for n in ('v', 't', 'texture_v', 'texture_t') if n in col), None)
    has_uvs = u_name is not None and v_name is not None
    
    for i in range(elem.count):
        vals = lines[start + i].split()
        positions_list.append((float(vals[col['x']]), float(vals[col['y']]), float(vals[col['z']])))
        if has_normals:
            normals_list.append((float(vals[col['nx']]), float(vals[col['ny']]), float(vals[col['nz']])))
        if has_uvs:
            uvs_list.append((float(vals[col[u_name]]), float(vals[col[v_name]])))
    
    positions = np.array(positions_list, dtype=np.float32)
    normals = np.array(normals_list, dtype=np.float32) if has_normals else None
    uvs = np.array(uvs_list, dtype=np.float32) if has_uvs else None
    
    return (positions, normals, uvs), start + elem.count


def _read_ascii_faces(lines: list[str], start: int, elem: _Element):
    triangles = []
    for i in range(elem.count):
        vals = [int(v) for v in lines[start + i].split()]
        n_verts = vals[0]
        indices = vals[1:1 + n_verts]
        for j in range(1, n_verts - 1):
            triangles.append((indices[0], indices[j], indices[j + 1]))
    return np.array(triangles, dtype=np.int32), start + elem.count


def _skip_binary_element(f: BinaryIO, elem: _Element, endian: str):
    """Skip unknown binary elements."""
    has_list = any(p.is_list for p in elem.properties)
    if not has_list:
        # Fixed-size rows
        row_size = sum(_PLY_TYPE_MAP[p.value_type][1] for p in elem.properties)
        f.read(row_size * elem.count)
    else:
        for _ in range(elem.count):
            for prop in elem.properties:
                if prop.is_list:
                    cc, cs = _PLY_TYPE_MAP[prop.count_type]
                    raw = f.read(cs)
                    n = struct.unpack(endian + cc, raw)[0]
                    _, vs = _PLY_TYPE_MAP[prop.value_type]
                    f.read(vs * n)
                else:
                    _, s = _PLY_TYPE_MAP[prop.value_type]
                    f.read(s)
