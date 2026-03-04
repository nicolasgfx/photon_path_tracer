"""
Map PBRT v4 materials to Wavefront MTL + pb_* extensions.

Resolves texture graph chains (imagemap → scale → material) into direct
texture file paths with bake-scale factors.  Outputs MTL-compatible material
descriptions annotated with pb_* tokens that the photon-beam renderer
understands.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any

from .pbrt_parser import (
    PbrtScene, PbrtMaterial, PbrtTextureDecl, PbrtParam,
    get_param, get_param_type,
)


# ---------------------------------------------------------------------------
# Well-known conductor spectra  (R=650nm, G=550nm, B=450nm approximation)
# ---------------------------------------------------------------------------

CONDUCTOR_PRESETS: dict[str, tuple[list[float], list[float]]] = {
    'metal-Al-eta':  ([1.34, 0.96, 0.50],  [7.47, 6.40, 5.30]),   # aluminum
    'metal-Cu-eta':  ([0.21, 0.92, 1.16],  [3.58, 2.60, 2.30]),   # copper
    'metal-Au-eta':  ([0.16, 0.42, 1.47],  [3.98, 2.38, 1.60]),   # gold
    'metal-Ag-eta':  ([0.05, 0.06, 0.05],  [4.28, 3.52, 2.73]),   # silver
    'metal-Fe-eta':  ([2.87, 2.95, 2.65],  [3.12, 2.93, 2.77]),   # iron
    'metal-Ti-eta':  ([2.16, 1.93, 1.72],  [2.56, 2.37, 2.18]),   # titanium
    'metal-Cr-eta':  ([3.11, 3.18, 2.17],  [3.31, 3.32, 3.20]),   # chromium
    'metal-W-eta':   ([4.37, 3.31, 2.99],  [3.27, 2.69, 2.54]),   # tungsten
    'metal-Ni-eta':  ([1.98, 1.70, 1.67],  [3.74, 3.01, 2.50]),   # nickel
    'metal-Pt-eta':  ([2.38, 2.04, 1.69],  [4.26, 3.72, 3.13]),   # platinum
    'metal-Co-eta':  ([2.18, 2.00, 1.55],  [4.09, 3.59, 3.36]),   # cobalt
    'metal-Pd-eta':  ([1.66, 1.27, 0.82],  [4.33, 3.55, 2.88]),   # palladium
    'metal-Zn-eta':  ([1.10, 0.64, 1.21],  [5.55, 4.76, 3.57]),   # zinc
}


# ---------------------------------------------------------------------------
# Blackbody → linear sRGB (chromaticity, peak normalised to 1.0)
# ---------------------------------------------------------------------------

def blackbody_to_rgb(temperature_K: float) -> list[float]:
    """Convert blackbody temperature to linear sRGB (peak=1.0)."""
    # Tanner Helland approximation (sRGB chromaticity)
    t = temperature_K / 100.0
    if t <= 66:
        r = 255.0
        g = max(0.0, 99.4708025861 * math.log(t) - 161.1195681661)
        if t <= 19:
            b = 0.0
        else:
            b = max(0.0, 138.5177312231 * math.log(t - 10.0) - 305.0447927307)
    else:
        r = max(0.0, 329.698727446 * ((t - 60) ** -0.1332047592))
        g = max(0.0, 288.1221695283 * ((t - 60) ** -0.0755148492))
        b = 255.0
    r = min(255, r) / 255.0
    g = min(255, g) / 255.0
    b = min(255, b) / 255.0
    mx = max(r, g, b, 1e-10)
    return [r / mx, g / mx, b / mx]


# ---------------------------------------------------------------------------
# Resolved texture reference
# ---------------------------------------------------------------------------

@dataclass
class ResolvedTexture:
    """A resolved texture file path with optional bake-scale."""
    filepath: str          # original relative filepath from PBRT
    scale: float = 1.0     # multiplicative scale to bake into the texture
    is_alpha: bool = False  # if True, this is used as an alpha/opacity map


# ---------------------------------------------------------------------------
# Resolved MTL material
# ---------------------------------------------------------------------------

@dataclass
class MtlMaterial:
    """Complete MTL + pb_* material ready to write."""
    name: str
    # Classic MTL
    Kd: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    Ks: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    Ke: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    Ns: float = 100.0          # Phong exponent (derived from roughness)
    Ni: float = 1.5
    d: float = 1.0             # opacity
    illum: int = 2
    map_Kd: str | None = None
    map_Ks: str | None = None
    map_Ke: str | None = None
    map_d: str | None = None
    map_bump: str | None = None
    # pb_* extensions
    pb_brdf: str | None = None
    pb_semantic: str | None = None
    pb_roughness: float | None = None
    pb_roughness_x: float | None = None
    pb_roughness_y: float | None = None
    pb_anisotropy: float | None = None
    pb_eta: float | None = None
    pb_conductor_eta: list[float] | None = None
    pb_conductor_k: list[float] | None = None
    pb_transmission: float | None = None
    pb_thin: int | None = None
    pb_thickness: float | None = None
    pb_clearcoat: float | None = None
    pb_clearcoat_roughness: float | None = None
    pb_base_brdf: str | None = None
    pb_base_roughness: float | None = None
    pb_sheen: float | None = None
    pb_sheen_tint: float | None = None
    # Texture bake info (filepath -> scale)
    textures_to_bake: dict[str, float] = field(default_factory=dict)
    # Comments
    comments: list[str] = field(default_factory=list)


def roughness_to_phong(alpha: float) -> float:
    """Convert GGX roughness α to Phong exponent Ns = 2/α² − 2."""
    alpha = max(alpha, 0.001)
    return 2.0 / (alpha * alpha) - 2.0


def _safe_float(val, default: float = 0.0) -> float:
    """Convert a PBRT param value to float, returning *default* for texture refs / lists."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except (ValueError, TypeError):
            return default  # texture reference string
    if isinstance(val, (list, tuple)):
        # RGB → luminance
        if len(val) >= 3:
            return 0.2126 * float(val[0]) + 0.7152 * float(val[1]) + 0.0722 * float(val[2])
        elif len(val) == 1:
            return float(val[0])
    return default


def _pbrt_roughness_to_ours(roughness: float, remaproughness: bool = True) -> float:
    """Convert PBRT roughness value to our renderer's roughness parameter.
    
    PBRT's roughness chain:
      - If remaproughness=true (default):  PBRT alpha = sqrt(roughness)
      - If remaproughness=false:            PBRT alpha = roughness
    
    Our renderer's chain:
      - bsdf_roughness_to_alpha(r) = r * r    (i.e., alpha = r²)
    
    So we need our_roughness² = pbrt_alpha:
      - remap=true:  our_roughness = roughness^(1/4)
      - remap=false: our_roughness = sqrt(roughness)
    """
    roughness = max(roughness, 0.0)
    if roughness < 1e-8:
        return 0.0
    if remaproughness:
        return roughness ** 0.25
    else:
        return math.sqrt(roughness)
    return 2.0 / (alpha * alpha) - 2.0


# ---------------------------------------------------------------------------
# Texture graph resolver
# ---------------------------------------------------------------------------

class TextureResolver:
    """Walks the PBRT texture graph to find the final image file and scale."""

    def __init__(self, textures: dict[str, PbrtTextureDecl], source_dir: str):
        self.textures = textures
        self.source_dir = source_dir

    def resolve(self, tex_name: str) -> ResolvedTexture | None:
        """Follow a texture name through scale/imagemap chains.
        
        Returns the final image path (relative to PBRT source dir) and
        accumulated scale factor, or None if not resolvable.
        """
        return self._resolve_recursive(tex_name, 1.0, depth=0)

    def _resolve_recursive(self, tex_name: str, accum_scale: float, depth: int
                           ) -> ResolvedTexture | None:
        if depth > 10:
            return None
        td = self.textures.get(tex_name)
        if td is None:
            return None

        if td.tex_class == 'imagemap':
            filename = get_param(td.params, 'filename')
            if filename:
                return ResolvedTexture(filename, accum_scale)
            return None

        if td.tex_class == 'scale':
            scale_val = get_param(td.params, 'scale', 1.0)
            inner = get_param(td.params, 'tex')

            # Determine numeric scale factor
            if isinstance(scale_val, (int, float)):
                effective_scale = float(scale_val)
            elif isinstance(scale_val, (list, tuple)) and len(scale_val) >= 3:
                # RGB spectrum → luminance approximation
                effective_scale = (0.2126 * scale_val[0] + 0.7152 * scale_val[1]
                                   + 0.0722 * scale_val[2])
            else:
                # Texture reference string – can't fold into a scalar
                effective_scale = 1.0

            # Try following 'tex' first (primary image source)
            if isinstance(inner, str):
                result = self._resolve_recursive(
                    inner, accum_scale * effective_scale, depth + 1)
                if result:
                    return result

            # Fall back: if 'scale' is itself a texture ref, follow it instead
            if isinstance(scale_val, str):
                result = self._resolve_recursive(
                    scale_val, accum_scale, depth + 1)
                if result:
                    return result

            return None

        if td.tex_class == 'constant':
            # No file to reference
            return None

        if td.tex_class in ('mix', 'directionmix'):
            # Mix two textures — try to resolve each and return first image found
            for key in ('tex1', 'tex2'):
                child = get_param(td.params, key)
                if isinstance(child, str):
                    result = self._resolve_recursive(
                        child, accum_scale, depth + 1)
                    if result:
                        return result
            return None

        return None


# ---------------------------------------------------------------------------
# Material mapper
# ---------------------------------------------------------------------------

def map_materials(scene: PbrtScene) -> tuple[dict[str, MtlMaterial], dict[str, ResolvedTexture]]:
    """Map all PBRT materials to MTL materials.
    
    Returns (materials_dict, all_textures) where all_textures maps
    output tex filename → ResolvedTexture.
    """
    resolver = TextureResolver(scene.textures, scene.source_dir)
    materials: dict[str, MtlMaterial] = {}
    all_textures: dict[str, ResolvedTexture] = {}  # out_name → resolved

    # Process named materials
    for name, pbrt_mat in scene.named_materials.items():
        mtl = _map_one_material(name, pbrt_mat, resolver, all_textures,
                                named_materials=scene.named_materials)
        materials[name] = mtl

    # Process any inline materials from shapes that don't have a named material
    # Deduplicate by (mat_type + resolved texture paths + param values) signature
    inline_counter = 0
    _sig_map: dict[str, str] = {}  # signature → material name
    for shape in scene.shapes:
        if shape.inline_material and shape.material_name is None:
            imat = shape.inline_material
            sig = _inline_material_signature(imat, resolver)
            if sig in _sig_map:
                shape.material_name = _sig_map[sig]
            else:
                syn_name = f"_inline_{imat.mat_type}_{inline_counter}"
                mtl = _map_one_material(syn_name, imat, resolver, all_textures)
                materials[syn_name] = mtl
                shape.material_name = syn_name
                _sig_map[sig] = syn_name
                inline_counter += 1

    return materials, all_textures


def _inline_material_signature(mat: PbrtMaterial, resolver: 'TextureResolver') -> str:
    """Create a deduplication key for an inline material.
    
    Two inline materials with the same type and the same resolved texture files
    (and same parameter values) should map to the same MTL entry.
    """
    parts = [mat.mat_type]
    for p in sorted(mat.params, key=lambda x: x.name):
        if p.type == 'texture':
            # Resolve to the actual image path for dedup
            rt = resolver.resolve(p.value) if isinstance(p.value, str) else None
            if rt:
                parts.append(f"{p.name}={rt.filepath}:{rt.scale}")
            else:
                parts.append(f"{p.name}=tex:{p.value}")
        else:
            parts.append(f"{p.name}={p.value}")
    return "|".join(parts)


def map_inline_material(mat: PbrtMaterial, scene: PbrtScene,
                        materials: dict[str, MtlMaterial],
                        all_textures: dict[str, ResolvedTexture],
                        counter: int) -> str:
    """Map a single inline material and return its MTL name."""
    resolver = TextureResolver(scene.textures, scene.source_dir)
    syn_name = f"_inline_{mat.mat_type}_{counter}"
    if syn_name not in materials:
        mtl = _map_one_material(syn_name, mat, resolver, all_textures)
        materials[syn_name] = mtl
    return syn_name


def _resolve_and_register(tex_name: str, resolver: TextureResolver,
                          all_textures: dict[str, ResolvedTexture],
                          suffix: str = "") -> str | None:
    """Resolve a texture reference and register it. Returns output filename."""
    rt = resolver.resolve(tex_name)
    if rt is None:
        return None
    # Build a clean output filename
    import os
    base = os.path.basename(rt.filepath)
    name, ext = os.path.splitext(base)
    if abs(rt.scale - 1.0) > 0.001:
        out_name = f"{name}_scaled{ext}"
    else:
        out_name = base
    if suffix:
        n2, e2 = os.path.splitext(out_name)
        out_name = f"{n2}_{suffix}{e2}"
    all_textures[out_name] = rt
    return f"textures/{out_name}"


def _map_one_material(name: str, pbrt_mat: PbrtMaterial,
                      resolver: TextureResolver,
                      all_textures: dict[str, ResolvedTexture],
                      named_materials: dict[str, PbrtMaterial] | None = None) -> MtlMaterial:
    """Map a single PBRT material to MtlMaterial."""
    mt = pbrt_mat.mat_type
    params = pbrt_mat.params
    mtl = MtlMaterial(name=name)

    if mt == 'coateddiffuse':
        _map_coated_diffuse(mtl, params, resolver, all_textures)
    elif mt == 'diffuse':
        _map_diffuse(mtl, params, resolver, all_textures)
    elif mt == 'dielectric':
        _map_dielectric(mtl, params, resolver, all_textures)
    elif mt == 'thindielectric':
        _map_thin_dielectric(mtl, params)
    elif mt == 'conductor':
        _map_conductor(mtl, params, resolver, all_textures)
    elif mt == 'coatedconductor':
        _map_coated_conductor(mtl, params, resolver, all_textures)
    elif mt == 'measured':
        _map_measured(mtl, params)
    elif mt == 'diffusetransmission':
        _map_diffuse_transmission(mtl, params, resolver, all_textures)
    elif mt == 'mix':
        _map_mix(mtl, params, pbrt_mat, resolver, all_textures, named_materials)
    else:
        mtl.comments.append(f"# Unknown PBRT material type: {mt}")
        mtl.pb_brdf = 'lambert'

    return mtl


# ---------------------------------------------------------------------------
# Per-type mappers
# ---------------------------------------------------------------------------

def _map_coated_diffuse(mtl: MtlMaterial, params: list[PbrtParam],
                        resolver: TextureResolver,
                        all_textures: dict[str, ResolvedTexture]):
    """coateddiffuse → clearcoat (pb_brdf clearcoat)."""
    mtl.pb_brdf = 'clearcoat'
    mtl.illum = 2

    # Reflectance (diffuse base)
    ref_type = get_param_type(params, 'reflectance')
    if ref_type == 'texture':
        tex_name = get_param(params, 'reflectance')
        out = _resolve_and_register(tex_name, resolver, all_textures)
        if out:
            mtl.map_Kd = out
        mtl.Kd = [0.5, 0.5, 0.5]  # fallback
    elif ref_type == 'rgb':
        rgb = get_param(params, 'reflectance', [0.5, 0.5, 0.5])
        mtl.Kd = list(rgb)
    else:
        # try "color reflectance"
        rgb = get_param(params, 'reflectance', [0.5, 0.5, 0.5])
        if isinstance(rgb, list):
            mtl.Kd = list(rgb)

    # Roughness
    roughness = get_param(params, 'roughness', None)
    uroughness = get_param(params, 'uroughness', None)
    vroughness = get_param(params, 'vroughness', None)
    remap = get_param(params, 'remaproughness', True)

    if uroughness is not None and vroughness is not None:
        ur = _pbrt_roughness_to_ours(_safe_float(uroughness), remap)
        vr = _pbrt_roughness_to_ours(_safe_float(vroughness), remap)
        mtl.pb_roughness_x = ur
        mtl.pb_roughness_y = vr
        mtl.pb_clearcoat_roughness = math.sqrt(ur * vr)
        mtl.Ns = roughness_to_phong(mtl.pb_clearcoat_roughness)
    elif roughness is not None:
        our_r = _pbrt_roughness_to_ours(_safe_float(roughness), remap)
        mtl.pb_clearcoat_roughness = our_r
        mtl.Ns = roughness_to_phong(our_r)
    else:
        mtl.pb_clearcoat_roughness = 0.0
        mtl.Ns = 10000.0

    # Coat IOR
    eta = get_param(params, 'eta', 1.5)
    mtl.pb_eta = _safe_float(eta, 1.5)
    mtl.Ni = _safe_float(eta, 1.5)

    # Coat weight
    mtl.pb_clearcoat = 1.0
    mtl.pb_base_brdf = 'lambert'

    # Displacement/bump
    disp_type = get_param_type(params, 'displacement')
    if disp_type == 'texture':
        tex_name = get_param(params, 'displacement')
        out = _resolve_and_register(tex_name, resolver, all_textures, suffix="bump")
        if out:
            mtl.map_bump = out


def _map_diffuse(mtl: MtlMaterial, params: list[PbrtParam],
                 resolver: TextureResolver,
                 all_textures: dict[str, ResolvedTexture]):
    """diffuse → lambert."""
    mtl.pb_brdf = 'lambert'
    mtl.illum = 1
    mtl.pb_roughness = 1.0
    mtl.Ns = 0.0

    ref_type = get_param_type(params, 'reflectance')
    if ref_type == 'texture':
        tex_name = get_param(params, 'reflectance')
        out = _resolve_and_register(tex_name, resolver, all_textures)
        if out:
            mtl.map_Kd = out
    elif ref_type == 'rgb':
        rgb = get_param(params, 'reflectance', [0.5, 0.5, 0.5])
        mtl.Kd = list(rgb)
    else:
        rgb = get_param(params, 'reflectance', [0.5, 0.5, 0.5])
        if isinstance(rgb, list):
            mtl.Kd = list(rgb)


def _map_dielectric(mtl: MtlMaterial, params: list[PbrtParam],
                    resolver: TextureResolver,
                    all_textures: dict[str, ResolvedTexture]):
    """dielectric → glass / water."""
    mtl.pb_brdf = 'dielectric'
    mtl.illum = 4  # glass
    mtl.pb_transmission = 1.0
    mtl.d = 1.0  # opacity stays 1 for glass (renderer handles refraction)
    mtl.Kd = [0.0, 0.0, 0.0]
    mtl.Ks = [1.0, 1.0, 1.0]

    eta = get_param(params, 'eta', 1.5)
    mtl.pb_eta = _safe_float(eta, 1.5)
    mtl.Ni = _safe_float(eta, 1.5)

    roughness = get_param(params, 'roughness', None)
    remap = get_param(params, 'remaproughness', True)
    if roughness is not None:
        our_r = _pbrt_roughness_to_ours(_safe_float(roughness), remap)
        mtl.pb_roughness = our_r
        mtl.Ns = roughness_to_phong(our_r)

    # Displacement/bump (e.g., water)
    disp_type = get_param_type(params, 'displacement')
    if disp_type == 'texture':
        tex_name = get_param(params, 'displacement')
        out = _resolve_and_register(tex_name, resolver, all_textures, suffix="bump")
        if out:
            mtl.map_bump = out


def _map_conductor(mtl: MtlMaterial, params: list[PbrtParam],
                   resolver: TextureResolver,
                   all_textures: dict[str, ResolvedTexture]):
    """conductor → metal."""
    mtl.pb_brdf = 'conductor'
    mtl.illum = 3  # mirror-like
    mtl.Kd = [0.0, 0.0, 0.0]

    # Resolve conductor optical constants
    eta_val = get_param(params, 'eta')
    k_val = get_param(params, 'k')
    eta_rgb, k_rgb = _resolve_conductor_eta_k(eta_val, k_val)

    if isinstance(eta_val, str) and not any(eta_val in pn for pn in CONDUCTOR_PRESETS):
        mtl.comments.append(f"# Unknown conductor spectrum '{eta_val}', using aluminum defaults")

    mtl.pb_conductor_eta = eta_rgb
    mtl.pb_conductor_k = k_rgb

    # Compute Schlick F0 from eta/k: F0 = ((n-1)²+k²) / ((n+1)²+k²)
    f0 = [((n - 1)**2 + k**2) / ((n + 1)**2 + k**2) for n, k in zip(eta_rgb, k_rgb)]
    mtl.Ks = f0

    roughness = get_param(params, 'roughness', None)
    uroughness = get_param(params, 'uroughness', None)
    vroughness = get_param(params, 'vroughness', None)
    remap = get_param(params, 'remaproughness', True)

    if uroughness is not None and vroughness is not None:
        ur = _pbrt_roughness_to_ours(_safe_float(uroughness), remap)
        vr = _pbrt_roughness_to_ours(_safe_float(vroughness), remap)
        mtl.pb_roughness_x = ur
        mtl.pb_roughness_y = vr
        mtl.pb_roughness = math.sqrt(ur * vr)
        mtl.Ns = roughness_to_phong(mtl.pb_roughness)
    elif roughness is not None:
        our_r = _pbrt_roughness_to_ours(_safe_float(roughness), remap)
        mtl.pb_roughness = our_r
        mtl.Ns = roughness_to_phong(our_r)
    else:
        mtl.pb_roughness = 0.0
        mtl.Ns = 10000.0


def _map_coated_conductor(mtl: MtlMaterial, params: list[PbrtParam],
                          resolver: TextureResolver,
                          all_textures: dict[str, ResolvedTexture]):
    """coatedconductor → clearcoat over conductor base (pb_brdf clearcoat).
    
    PBRT's coatedconductor has a dielectric coat over a conductor base.
    We map it to clearcoat with the conductor's specular colour as base.
    """
    mtl.pb_brdf = 'clearcoat'
    mtl.illum = 3
    mtl.Kd = [0.0, 0.0, 0.0]

    # Conductor base properties
    eta_val = get_param(params, 'conductor.eta')
    k_val = get_param(params, 'conductor.k')
    if eta_val is None:
        eta_val = get_param(params, 'eta')
    if k_val is None:
        k_val = get_param(params, 'k')

    eta_rgb, k_rgb = _resolve_conductor_eta_k(eta_val, k_val)
    mtl.pb_conductor_eta = eta_rgb
    mtl.pb_conductor_k = k_rgb

    # Schlick F0 as Ks for the base specular
    f0 = [((n - 1)**2 + k**2) / ((n + 1)**2 + k**2) for n, k in zip(eta_rgb, k_rgb)]
    mtl.Ks = f0

    # Reflectance (if present, use as diffuse tint)
    ref_type = get_param_type(params, 'reflectance')
    if ref_type == 'texture':
        tex_name = get_param(params, 'reflectance')
        out = _resolve_and_register(tex_name, resolver, all_textures)
        if out:
            mtl.map_Kd = out
    elif ref_type == 'rgb':
        rgb = get_param(params, 'reflectance', [0.0, 0.0, 0.0])
        mtl.Kd = list(rgb)

    # Base roughness
    roughness = get_param(params, 'conductor.roughness',
                          get_param(params, 'roughness', None))
    remap = get_param(params, 'remaproughness', True)
    if roughness is not None:
        our_r = _pbrt_roughness_to_ours(_safe_float(roughness), remap)
        mtl.pb_base_roughness = our_r
    else:
        mtl.pb_base_roughness = 0.0

    # Coat properties
    coat_eta = get_param(params, 'interface.eta', 1.5)
    mtl.pb_eta = _safe_float(coat_eta, 1.5)
    mtl.Ni = _safe_float(coat_eta, 1.5)

    coat_roughness = get_param(params, 'interface.roughness', 0.0)
    if coat_roughness is not None:
        our_cr = _pbrt_roughness_to_ours(_safe_float(coat_roughness), remap)
        mtl.pb_clearcoat_roughness = our_cr
    else:
        mtl.pb_clearcoat_roughness = 0.0

    mtl.pb_clearcoat = 1.0
    mtl.pb_base_brdf = 'conductor'
    mtl.Ns = roughness_to_phong(mtl.pb_clearcoat_roughness or 0.001)

    mtl.comments.append("# PBRT coatedconductor → clearcoat over conductor base")


def _map_thin_dielectric(mtl: MtlMaterial, params: list[PbrtParam]):
    """thindielectric → glass with pb_thin flag.
    
    PBRT's ThinDielectric represents an infinitely thin glass surface
    (no refraction offset, just Fresnel reflection/transmission).
    """
    mtl.pb_brdf = 'dielectric'
    mtl.illum = 4
    mtl.pb_transmission = 1.0
    mtl.pb_thin = 1
    mtl.d = 1.0
    mtl.Kd = [0.0, 0.0, 0.0]
    mtl.Ks = [1.0, 1.0, 1.0]

    eta = get_param(params, 'eta', 1.5)
    mtl.pb_eta = _safe_float(eta, 1.5)
    mtl.Ni = _safe_float(eta, 1.5)

    # ThinDielectric has no roughness — perfectly smooth
    mtl.pb_roughness = 0.0
    mtl.Ns = 10000.0

    mtl.comments.append("# PBRT thindielectric → thin glass (no refraction offset)")


def _map_mix(mtl: MtlMaterial, params: list[PbrtParam],
             pbrt_mat: PbrtMaterial,
             resolver: TextureResolver,
             all_textures: dict[str, ResolvedTexture],
             named_materials: dict[str, PbrtMaterial] | None = None):
    """mix → choose dominant material based on amount.
    
    PBRT's Mix material blends two sub-materials. Since our renderer
    doesn't support material blending, we choose the dominant one
    (amount > 0.5 → material[1], else material[0]).
    """
    materials_param = get_param(params, 'materials')
    amount = get_param(params, 'amount', 0.5)
    if isinstance(amount, list):
        amount = amount[0]
    amount = _safe_float(amount, 0.5)

    mat_names = []
    if isinstance(materials_param, list):
        mat_names = [str(m) for m in materials_param]
    elif isinstance(materials_param, str):
        mat_names = [materials_param]

    # Pick dominant material
    chosen_name = None
    if len(mat_names) >= 2:
        chosen_name = mat_names[1] if amount > 0.5 else mat_names[0]
    elif len(mat_names) == 1:
        chosen_name = mat_names[0]

    if chosen_name and named_materials and chosen_name in named_materials:
        sub_mat = named_materials[chosen_name]
        sub_mtl = _map_one_material(mtl.name, sub_mat, resolver, all_textures,
                                     named_materials=named_materials)
        # Copy all fields from sub-material
        for field_name in vars(sub_mtl):
            setattr(mtl, field_name, getattr(sub_mtl, field_name))
        mtl.name = mtl.name  # restore original name
        mtl.comments.append(f"# PBRT mix material → using dominant '{chosen_name}' (amount={amount:.2f})")
    else:
        mtl.pb_brdf = 'lambert'
        mtl.comments.append(f"# PBRT mix material — could not resolve sub-materials: {mat_names}")


def _resolve_conductor_eta_k(eta_val, k_val) -> tuple[list[float], list[float]]:
    """Resolve conductor eta/k from named spectra or raw RGB values."""
    eta_rgb = None
    k_rgb = None

    if isinstance(eta_val, str):
        for preset_name, (p_eta, p_k) in CONDUCTOR_PRESETS.items():
            if eta_val in preset_name or preset_name in eta_val:
                eta_rgb = p_eta
                k_rgb = p_k
                break
        if eta_rgb is None:
            eta_rgb = [1.34, 0.96, 0.50]  # default aluminum
            k_rgb = [7.47, 6.40, 5.30]
    elif isinstance(eta_val, list):
        eta_rgb = eta_val[:3]

    if isinstance(k_val, str) and k_rgb is None:
        for preset_name, (p_eta, p_k) in CONDUCTOR_PRESETS.items():
            if k_val in preset_name or preset_name in k_val:
                k_rgb = p_k
                if eta_rgb is None:
                    eta_rgb = p_eta
                break
    elif isinstance(k_val, list) and k_rgb is None:
        k_rgb = k_val[:3]

    if eta_rgb is None:
        eta_rgb = [1.34, 0.96, 0.50]
    if k_rgb is None:
        k_rgb = [7.47, 6.40, 5.30]

    return eta_rgb, k_rgb


def _map_measured(mtl: MtlMaterial, params: list[PbrtParam]):
    """measured (BSDF binary) → approximate as clearcoated white diffuse.
    
    The cm_white_spec.bsdf is white specular leather from EPFL RGL database.
    We approximate it as a clearcoat over a white Lambertian base.
    """
    mtl.pb_brdf = 'clearcoat'
    mtl.pb_semantic = 'leather'
    mtl.illum = 2
    mtl.Kd = [0.85, 0.85, 0.85]
    mtl.Ks = [0.15, 0.15, 0.15]
    mtl.pb_clearcoat = 1.0
    mtl.pb_clearcoat_roughness = 0.05
    mtl.pb_base_brdf = 'lambert'
    mtl.pb_eta = 1.5
    mtl.Ni = 1.5
    mtl.Ns = roughness_to_phong(0.05)

    filename = get_param(params, 'filename', '')
    mtl.comments.append(f"# Approximation of measured BSDF: {filename}")
    mtl.comments.append(f"# Original: EPFL RGL tensor-tree (Dupuy & Jakob)")
    mtl.comments.append(f"# Visually: glossy white leather (Barcelona chair)")


def _map_diffuse_transmission(mtl: MtlMaterial, params: list[PbrtParam],
                               resolver: TextureResolver,
                               all_textures: dict[str, ResolvedTexture]):
    """diffusetransmission → translucent fabric/leaf material."""
    mtl.pb_brdf = 'lambert'
    mtl.pb_semantic = 'fabric'
    mtl.pb_thin = 1
    mtl.illum = 1

    # Reflectance
    ref_type = get_param_type(params, 'reflectance')
    if ref_type == 'texture':
        tex_name = get_param(params, 'reflectance')
        out = _resolve_and_register(tex_name, resolver, all_textures)
        if out:
            mtl.map_Kd = out
    elif ref_type == 'rgb':
        rgb = get_param(params, 'reflectance', [0.25, 0.25, 0.25])
        mtl.Kd = list(rgb)

    # Transmittance  
    trans_type = get_param_type(params, 'transmittance')
    if trans_type == 'rgb':
        transmittance = get_param(params, 'transmittance', [0.25, 0.25, 0.25])
        # Average transmittance as transmission factor
        if isinstance(transmittance, list):
            mtl.pb_transmission = sum(transmittance[:3]) / len(transmittance[:3])
        else:
            mtl.pb_transmission = _safe_float(transmittance, 0.25)
    else:
        # Default: PBRT diffusetransmission defaults both reflectance and
        # transmittance to 0.25  
        mtl.pb_transmission = 0.25

    # Scale parameter  
    scale = get_param(params, 'scale', 1.0)
    if isinstance(scale, (int, float)) and float(scale) != 1.0:
        mtl.pb_transmission *= float(scale)

    # Alpha texture (for leaf cutouts)
    alpha_type = get_param_type(params, 'alpha')
    if alpha_type == 'texture':
        tex_name = get_param(params, 'alpha')
        out = _resolve_and_register(tex_name, resolver, all_textures, suffix="alpha")
        if out:
            mtl.map_d = out

    mtl.comments.append("# PBRT diffusetransmission — thin translucent (leaves)")
    mtl.comments.append("# pb_thin + pb_transmission flags for future renderer extension")


# ---------------------------------------------------------------------------
# Emissive material from area light
# ---------------------------------------------------------------------------

def create_emissive_material(name: str, area_light_params: list[PbrtParam],
                             scale: float = 1.0) -> MtlMaterial:
    """Create an emissive MTL material from PBRT AreaLightSource params."""
    mtl = MtlMaterial(name=name)
    mtl.pb_brdf = 'emissive'
    mtl.illum = 0

    # Check for blackbody
    bb = get_param(area_light_params, 'L')
    light_scale = get_param(area_light_params, 'scale', 1.0)
    if isinstance(light_scale, list):
        light_scale = light_scale[0]

    # Detect blackbody type parameter
    for p in area_light_params:
        if p.type == 'blackbody' and p.name == 'L':
            temp_K = p.value[0] if isinstance(p.value, list) else float(p.value)
            rgb = blackbody_to_rgb(temp_K)
            # Apply the PBRT scale factor to Ke so the OBJ/MTL carries the
            # correct absolute emission intensity (matches RGB-emission path).
            mtl.Ke = [v * float(light_scale) for v in rgb]
            mtl.comments.append(f"# Blackbody {temp_K}K, PBRT scale={light_scale}")
            return mtl

    # RGB emission
    if isinstance(bb, list) and len(bb) >= 3:
        mtl.Ke = [v * light_scale for v in bb[:3]]
    else:
        mtl.Ke = [1.0, 1.0, 1.0]

    return mtl
