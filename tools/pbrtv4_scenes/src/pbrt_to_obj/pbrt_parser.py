"""
PBRT v4 tokenizer and scene-graph parser.

Handles the PBRT v4 text format: Include directives, AttributeBegin/End,
ObjectBegin/End + ObjectInstance, Transform stack, Texture/Material/Shape
directives, LookAt/Camera/Film/Sampler/Integrator, LightSource + AreaLightSource.
"""

from __future__ import annotations
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Data classes for parsed scene
# ---------------------------------------------------------------------------

@dataclass
class PbrtParam:
    """A single typed parameter, e.g. ``"float fov" [45]``."""
    type: str          # "float", "integer", "string", "rgb", "spectrum", "texture", "bool", "blackbody", …
    name: str          # "fov", "reflectance", "filename", …
    value: Any         # float | int | str | list[float] | list[str] …


@dataclass
class PbrtTextureDecl:
    """A `Texture` directive."""
    tex_name: str      # declared name
    tex_type: str      # "spectrum" | "float"
    tex_class: str     # "imagemap" | "scale" | "constant" | …
    params: list[PbrtParam] = field(default_factory=list)


@dataclass
class PbrtMaterial:
    """A named or inline material."""
    mat_name: str | None       # None for inline Material directives
    mat_type: str              # "coateddiffuse", "dielectric", "conductor", "diffuse", "measured", "diffusetransmission", …
    params: list[PbrtParam] = field(default_factory=list)


@dataclass
class PbrtShape:
    """A Shape directive with associated world-transform and material."""
    shape_type: str                  # "plymesh", "sphere", "trianglemesh", …
    params: list[PbrtParam] = field(default_factory=list)
    material_name: str | None = None          # NamedMaterial reference, or None
    inline_material: PbrtMaterial | None = None  # inline Material
    transform: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    area_light: dict | None = None   # AreaLightSource params (night-scene emissive shapes)
    group_name: str = ""
    from_instance: bool = False      # True when emitted by ObjectInstance expansion
    reverse_orientation: bool = False # True when ReverseOrientation was active


@dataclass
class PbrtLight:
    """An InfiniteLight or area-light sphere."""
    light_type: str                  # "infinite", "diffuse", …
    params: list[PbrtParam] = field(default_factory=list)
    transform: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    shapes: list[PbrtShape] = field(default_factory=list)   # for area lights with attached shapes


@dataclass
class PbrtCamera:
    """Camera definition."""
    cam_type: str = "perspective"
    params: list[PbrtParam] = field(default_factory=list)
    look_at: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    pre_transform: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))


@dataclass
class PbrtFilm:
    """Film definition."""
    film_type: str = "rgb"
    params: list[PbrtParam] = field(default_factory=list)


@dataclass
class PbrtObjectTemplate:
    """An ObjectBegin … ObjectEnd block."""
    name: str
    # shapes, textures, materials collected inside the block
    shapes: list[PbrtShape] = field(default_factory=list)
    textures: list[PbrtTextureDecl] = field(default_factory=list)
    materials: list[PbrtMaterial] = field(default_factory=list)


@dataclass
class PbrtScene:
    """Complete parsed scene."""
    camera: PbrtCamera = field(default_factory=PbrtCamera)
    film: PbrtFilm = field(default_factory=PbrtFilm)
    sampler_params: list[PbrtParam] = field(default_factory=list)
    integrator: str | None = None
    integrator_params: list[PbrtParam] = field(default_factory=list)
    # Pre-WorldBegin transform (Scale -1 1 1)
    global_transform: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    lights: list[PbrtLight] = field(default_factory=list)
    textures: dict[str, PbrtTextureDecl] = field(default_factory=dict)
    named_materials: dict[str, PbrtMaterial] = field(default_factory=dict)
    shapes: list[PbrtShape] = field(default_factory=list)
    object_templates: dict[str, PbrtObjectTemplate] = field(default_factory=dict)
    source_dir: str = ""


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"""
    \#[^\n]*              |   # comments
    "[^"]*"               |   # quoted strings
    \[                    |   # open bracket
    \]                    |   # close bracket
    [^\s\[\]"#]+              # bare words / numbers
""", re.VERBOSE)


def _tokenize(text: str) -> list[str]:
    """Return list of tokens (strings still quoted, brackets kept)."""
    tokens: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok.startswith('#'):
            continue
        tokens.append(tok)
    return tokens


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _unquote(s: str) -> str:
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    return s


# ---------------------------------------------------------------------------
# Parameter parser
# ---------------------------------------------------------------------------

# Valid PBRT v4 parameter type keywords (used to distinguish "type name"
# declarations from multi-word string values like "textures/Sky 19.pfm").
_PBRT_PARAM_TYPES = frozenset({
    'float', 'integer', 'string', 'bool', 'boolean',
    'rgb', 'spectrum', 'blackbody',
    'point', 'point2', 'point3', 'vector', 'vector2', 'vector3', 'normal',
    'texture', 'color',
})

def _parse_params(tokens: list[str], pos: int) -> tuple[list[PbrtParam], int]:
    """Parse a sequence of ``"type name" value`` or ``"type name" [values]`` pairs.

    Returns (params, new_pos).  Stops when reaching a keyword or EOF.
    """
    params: list[PbrtParam] = []
    while pos < len(tokens):
        tok = tokens[pos]
        # A quoted string that looks like "type name" starts a parameter
        if tok.startswith('"') and tok.endswith('"'):
            inner = tok[1:-1]
            parts = inner.split(None, 1)
            if len(parts) == 2:
                ptype, pname = parts
                pos += 1
                # Collect value(s)
                if pos < len(tokens) and tokens[pos] == '[':
                    # bracketed list
                    pos += 1  # skip '['
                    vals: list[str] = []
                    while pos < len(tokens) and tokens[pos] != ']':
                        vals.append(tokens[pos])
                        pos += 1
                    if pos < len(tokens):
                        pos += 1  # skip ']'
                    value = _coerce(ptype, vals)
                elif pos < len(tokens) and not tokens[pos].startswith('"') and tokens[pos] not in ('[', ']'):
                    # single unbracketed numeric value
                    value = _coerce(ptype, [tokens[pos]])
                    pos += 1
                elif pos < len(tokens) and tokens[pos].startswith('"') and tokens[pos].endswith('"'):
                    # Quoted token — could be a bare string value (e.g. "trilinear")
                    # or the start of the next "type name" parameter.
                    candidate = tokens[pos][1:-1]
                    cparts = candidate.split(None, 1)
                    if ' ' not in candidate and '\t' not in candidate:
                        # Single-word quoted string → value for this param
                        value = _coerce(ptype, [tokens[pos]])
                        pos += 1
                    elif len(cparts) == 2 and cparts[0] in _PBRT_PARAM_TYPES:
                        # Multi-word with a valid PBRT type → next "type name" pair
                        value = _coerce(ptype, [])
                    else:
                        # Multi-word but NOT a valid type → string value with spaces
                        # e.g. "textures/Sky 19.pfm"
                        value = _coerce(ptype, [tokens[pos]])
                        pos += 1
                else:
                    value = _coerce(ptype, [])
                params.append(PbrtParam(ptype, pname, value))
            else:
                # Not a "type name" pair – might be a bare string value; stop
                break
        else:
            break  # next keyword / directive
    return params, pos


def _coerce(ptype: str, raw: list[str]) -> Any:
    """Convert raw string tokens to typed Python values."""
    if ptype in ("float",):
        nums = [float(_unquote(v)) for v in raw]
        return nums[0] if len(nums) == 1 else nums
    if ptype in ("integer",):
        nums = [int(float(_unquote(v))) for v in raw]
        return nums[0] if len(nums) == 1 else nums
    if ptype in ("string", "texture"):
        strs = [_unquote(v) for v in raw]
        return strs[0] if len(strs) == 1 else strs
    if ptype in ("rgb",):
        return [float(_unquote(v)) for v in raw]  # always list of 3
    if ptype in ("spectrum",):
        # could be a filename string or list of floats
        if raw and raw[0].startswith('"'):
            return _unquote(raw[0])
        return [float(_unquote(v)) for v in raw]
    if ptype in ("blackbody",):
        return [float(_unquote(v)) for v in raw]
    if ptype in ("bool",):
        return _unquote(raw[0]).lower() in ("true", "1") if raw else False
    if ptype in ("point", "point3", "vector", "vector3", "normal"):
        return [float(_unquote(v)) for v in raw]
    if ptype in ("point2", "vector2"):
        return [float(_unquote(v)) for v in raw]
    if ptype in ("color",):
        return [float(_unquote(v)) for v in raw]
    # fallback
    if len(raw) == 1:
        return _unquote(raw[0])
    return [_unquote(v) for v in raw]


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def _mat_translate(tx: float, ty: float, tz: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[0, 3] = tx; m[1, 3] = ty; m[2, 3] = tz
    return m


def _mat_scale(sx: float, sy: float, sz: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = sx; m[1, 1] = sy; m[2, 2] = sz
    return m


def _mat_rotate(angle_deg: float, ax: float, ay: float, az: float) -> np.ndarray:
    """Axis-angle rotation matrix (same convention as PBRT)."""
    a = math.radians(angle_deg)
    c = math.cos(a); s = math.sin(a)
    length = math.sqrt(ax*ax + ay*ay + az*az)
    if length < 1e-12:
        return np.eye(4, dtype=np.float64)
    ax /= length; ay /= length; az /= length
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = ax*ax + (1 - ax*ax)*c
    m[0, 1] = ax*ay*(1 - c) - az*s
    m[0, 2] = ax*az*(1 - c) + ay*s
    m[1, 0] = ax*ay*(1 - c) + az*s
    m[1, 1] = ay*ay + (1 - ay*ay)*c
    m[1, 2] = ay*az*(1 - c) - ax*s
    m[2, 0] = ax*az*(1 - c) - ay*s
    m[2, 1] = ay*az*(1 - c) + ax*s
    m[2, 2] = az*az + (1 - az*az)*c
    return m


def _mat_concat(new: np.ndarray, current: np.ndarray) -> np.ndarray:
    """PBRT concatenation: current = current @ new (right-multiply)."""
    return current @ new


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

class PbrtParser:
    """Recursive descent parser for PBRT v4 scene files."""

    def __init__(self):
        self.scene = PbrtScene()
        self._transform_stack: list[np.ndarray] = []     # for AttributeBegin/End
        self._current_transform = np.eye(4, dtype=np.float64)
        self._current_material: str | None = None
        self._current_inline_material: PbrtMaterial | None = None
        self._current_area_light: dict | None = None
        self._reverse_orientation: bool = False
        self._in_world = False
        self._in_object: PbrtObjectTemplate | None = None
        self._material_stack: list[tuple[str | None, PbrtMaterial | None, dict | None, bool]] = []
        self._pre_world_transform = np.eye(4, dtype=np.float64)

    def parse_file(self, filepath: str) -> PbrtScene:
        filepath = os.path.abspath(filepath)
        self.scene.source_dir = os.path.dirname(filepath)
        self._parse_file_recursive(filepath)
        return self.scene

    def _parse_file_recursive(self, filepath: str):
        filepath = os.path.abspath(filepath)
        base_dir = os.path.dirname(filepath)
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        tokens = _tokenize(text)
        self._dispatch(tokens, base_dir)

    def _dispatch(self, tokens: list[str], base_dir: str):
        pos = 0
        while pos < len(tokens):
            tok = tokens[pos]
            word = _unquote(tok)

            # --- Pre-WorldBegin directives ---
            if word == 'Film':
                pos += 1
                self.scene.film.film_type = _unquote(tokens[pos]); pos += 1
                params, pos = _parse_params(tokens, pos)
                self.scene.film.params = params

            elif word == 'Camera':
                pos += 1
                self.scene.camera.cam_type = _unquote(tokens[pos]); pos += 1
                params, pos = _parse_params(tokens, pos)
                self.scene.camera.params = params
                self.scene.camera.pre_transform = self._pre_world_transform.copy()

            elif word == 'Sampler':
                pos += 1
                _unquote(tokens[pos]); pos += 1  # sampler type
                params, pos = _parse_params(tokens, pos)
                self.scene.sampler_params = params

            elif word == 'Integrator':
                pos += 1
                self.scene.integrator = _unquote(tokens[pos]); pos += 1
                params, pos = _parse_params(tokens, pos)
                self.scene.integrator_params = params

            elif word == 'LookAt':
                pos += 1
                vals = []
                for _ in range(9):
                    vals.append(float(_unquote(tokens[pos]))); pos += 1
                eye = np.array(vals[0:3], dtype=np.float64)
                target = np.array(vals[3:6], dtype=np.float64)
                up = np.array(vals[6:9], dtype=np.float64)
                self.scene.camera.look_at = (eye, target, up)

            elif word == 'WorldBegin':
                pos += 1
                self._in_world = True
                # Save the accumulated pre-world transform
                self.scene.global_transform = self._pre_world_transform.copy()
                self._current_transform = np.eye(4, dtype=np.float64)

            # --- Transform directives (work both pre- and post-WorldBegin) ---
            elif word == 'Scale':
                pos += 1
                sx = float(_unquote(tokens[pos])); pos += 1
                sy = float(_unquote(tokens[pos])); pos += 1
                sz = float(_unquote(tokens[pos])); pos += 1
                m = _mat_scale(sx, sy, sz)
                if self._in_world:
                    self._current_transform = _mat_concat(m, self._current_transform)
                else:
                    self._pre_world_transform = _mat_concat(m, self._pre_world_transform)

            elif word == 'Translate':
                pos += 1
                tx = float(_unquote(tokens[pos])); pos += 1
                ty = float(_unquote(tokens[pos])); pos += 1
                tz = float(_unquote(tokens[pos])); pos += 1
                m = _mat_translate(tx, ty, tz)
                if self._in_world:
                    self._current_transform = _mat_concat(m, self._current_transform)
                else:
                    self._pre_world_transform = _mat_concat(m, self._pre_world_transform)

            elif word == 'Rotate':
                pos += 1
                angle = float(_unquote(tokens[pos])); pos += 1
                ax = float(_unquote(tokens[pos])); pos += 1
                ay = float(_unquote(tokens[pos])); pos += 1
                az = float(_unquote(tokens[pos])); pos += 1
                m = _mat_rotate(angle, ax, ay, az)
                if self._in_world:
                    self._current_transform = _mat_concat(m, self._current_transform)
                else:
                    self._pre_world_transform = _mat_concat(m, self._pre_world_transform)

            elif word == 'ConcatTransform':
                pos += 1
                if tokens[pos] == '[':
                    pos += 1
                vals = []
                while len(vals) < 16:
                    vals.append(float(_unquote(tokens[pos]))); pos += 1
                if pos < len(tokens) and tokens[pos] == ']':
                    pos += 1
                # PBRT stores column-major in the file (per PBRT book B.3)
                m = np.array(vals, dtype=np.float64).reshape(4, 4).T
                if self._in_world:
                    self._current_transform = _mat_concat(m, self._current_transform)
                else:
                    self._pre_world_transform = _mat_concat(m, self._pre_world_transform)

            elif word == 'Transform':
                pos += 1
                if tokens[pos] == '[':
                    pos += 1
                vals = []
                while len(vals) < 16:
                    vals.append(float(_unquote(tokens[pos]))); pos += 1
                if pos < len(tokens) and tokens[pos] == ']':
                    pos += 1
                # PBRT stores column-major in the file (per PBRT book B.3)
                m = np.array(vals, dtype=np.float64).reshape(4, 4).T
                if self._in_world:
                    self._current_transform = m
                else:
                    self._pre_world_transform = m

            elif word == 'Identity':
                pos += 1
                if self._in_world:
                    self._current_transform = np.eye(4, dtype=np.float64)
                else:
                    self._pre_world_transform = np.eye(4, dtype=np.float64)

            # --- Attribute stack ---
            elif word == 'AttributeBegin':
                pos += 1
                self._transform_stack.append(self._current_transform.copy())
                self._material_stack.append(
                    (self._current_material, self._current_inline_material, self._current_area_light, self._reverse_orientation)
                )

            elif word == 'AttributeEnd':
                pos += 1
                if self._transform_stack:
                    self._current_transform = self._transform_stack.pop()
                if self._material_stack:
                    self._current_material, self._current_inline_material, self._current_area_light, self._reverse_orientation = self._material_stack.pop()

            # --- Object instancing ---
            elif word == 'ObjectBegin':
                pos += 1
                obj_name = _unquote(tokens[pos]); pos += 1
                self._in_object = PbrtObjectTemplate(name=obj_name)

            elif word == 'ObjectEnd':
                pos += 1
                if self._in_object is not None:
                    self.scene.object_templates[self._in_object.name] = self._in_object
                    self._in_object = None

            elif word == 'ObjectInstance':
                pos += 1
                obj_name = _unquote(tokens[pos]); pos += 1
                # Emit shapes from the template with current transform
                tpl = self.scene.object_templates.get(obj_name)
                if tpl:
                    xform = self._current_transform.copy()
                    for tpl_shape in tpl.shapes:
                        combined = xform @ tpl_shape.transform
                        s = PbrtShape(
                            shape_type=tpl_shape.shape_type,
                            params=tpl_shape.params,
                            material_name=tpl_shape.material_name,
                            inline_material=tpl_shape.inline_material,
                            transform=combined,
                            area_light=tpl_shape.area_light,
                            group_name=tpl_shape.group_name,
                            from_instance=True,
                            reverse_orientation=tpl_shape.reverse_orientation,
                        )
                        self.scene.shapes.append(s)

            # --- Include ---
            elif word == 'Include':
                pos += 1
                inc_path = _unquote(tokens[pos]); pos += 1
                full_path = os.path.normpath(os.path.join(base_dir, inc_path))
                self._parse_file_recursive(full_path)

            # --- Texture ---
            elif word == 'Texture':
                pos += 1
                tex_name = _unquote(tokens[pos]); pos += 1
                tex_type = _unquote(tokens[pos]); pos += 1
                tex_class = _unquote(tokens[pos]); pos += 1
                params, pos = _parse_params(tokens, pos)
                td = PbrtTextureDecl(tex_name, tex_type, tex_class, params)
                if self._in_object is not None:
                    self._in_object.textures.append(td)
                self.scene.textures[tex_name] = td

            # --- MakeNamedMaterial ---
            elif word == 'MakeNamedMaterial':
                pos += 1
                mat_name = _unquote(tokens[pos]); pos += 1
                params, pos = _parse_params(tokens, pos)
                mat_type = "unknown"
                for p in params:
                    if p.name == "type":
                        mat_type = p.value
                        break
                mat = PbrtMaterial(mat_name, mat_type, params)
                self.scene.named_materials[mat_name] = mat
                if self._in_object is not None:
                    self._in_object.materials.append(mat)

            # --- NamedMaterial (reference) ---
            elif word == 'NamedMaterial':
                pos += 1
                self._current_material = _unquote(tokens[pos]); pos += 1
                self._current_inline_material = None

            # --- Inline Material ---
            elif word == 'Material':
                pos += 1
                mat_type = _unquote(tokens[pos]); pos += 1
                params, pos = _parse_params(tokens, pos)
                mat = PbrtMaterial(None, mat_type, params)
                self._current_inline_material = mat
                self._current_material = None
                if self._in_object is not None:
                    self._in_object.materials.append(mat)

            # --- LightSource ---
            elif word == 'LightSource':
                pos += 1
                lt_type = _unquote(tokens[pos]); pos += 1
                params, pos = _parse_params(tokens, pos)
                light = PbrtLight(lt_type, params, self._current_transform.copy())
                self.scene.lights.append(light)

            # --- AreaLightSource ---
            elif word == 'AreaLightSource':
                pos += 1
                lt_type = _unquote(tokens[pos]); pos += 1
                params, pos = _parse_params(tokens, pos)
                self._current_area_light = {
                    'type': lt_type,
                    'params': params,
                    'transform': self._current_transform.copy(),
                }

            # --- Shape ---
            elif word == 'Shape':
                pos += 1
                shape_type = _unquote(tokens[pos]); pos += 1
                params, pos = _parse_params(tokens, pos)
                shape = PbrtShape(
                    shape_type=shape_type,
                    params=params,
                    material_name=self._current_material,
                    inline_material=self._current_inline_material,
                    transform=self._current_transform.copy(),
                    area_light=self._current_area_light,
                    reverse_orientation=self._reverse_orientation,
                )
                if self._in_object is not None:
                    self._in_object.shapes.append(shape)
                else:
                    self.scene.shapes.append(shape)

            # --- ReverseOrientation ---
            elif word == 'ReverseOrientation':
                pos += 1
                self._reverse_orientation = not self._reverse_orientation

            # --- Directives we skip ---
            elif word in ('MakeNamedMedium', 'MediumInterface',
                          'PixelFilter', 'ColorSpace', 'Option'):
                pos += 1
                # skip any following params
                _, pos = _parse_params(tokens, pos)

            else:
                pos += 1  # unknown token, skip


# ---------------------------------------------------------------------------
# Helper to get a param value
# ---------------------------------------------------------------------------

def get_param(params: list[PbrtParam], name: str, default: Any = None) -> Any:
    for p in params:
        if p.name == name:
            return p.value
    return default


def get_param_type(params: list[PbrtParam], name: str) -> str | None:
    for p in params:
        if p.name == name:
            return p.type
    return None
