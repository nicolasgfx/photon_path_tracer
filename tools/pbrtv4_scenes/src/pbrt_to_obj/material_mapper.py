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
            scale = get_param(td.params, 'scale', 1.0)
            inner = get_param(td.params, 'tex')
            if inner:
                return self._resolve_recursive(inner, accum_scale * scale, depth + 1)
            return None

        if td.tex_class == 'constant':
            # No file to reference
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
        mtl = _map_one_material(name, pbrt_mat, resolver, all_textures)
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
                      all_textures: dict[str, ResolvedTexture]) -> MtlMaterial:
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
    elif mt == 'conductor':
        _map_conductor(mtl, params, resolver, all_textures)
    elif mt == 'measured':
        _map_measured(mtl, params)
    elif mt == 'diffusetransmission':
        _map_diffuse_transmission(mtl, params, resolver, all_textures)
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

    if uroughness is not None and vroughness is not None:
        mtl.pb_roughness_x = float(uroughness)
        mtl.pb_roughness_y = float(vroughness)
        mtl.pb_clearcoat_roughness = math.sqrt(float(uroughness) * float(vroughness))
        mtl.Ns = roughness_to_phong(mtl.pb_clearcoat_roughness)
    elif roughness is not None:
        mtl.pb_clearcoat_roughness = float(roughness)
        mtl.Ns = roughness_to_phong(float(roughness))
    else:
        mtl.pb_clearcoat_roughness = 0.0
        mtl.Ns = 10000.0

    # Coat IOR
    eta = get_param(params, 'eta', 1.5)
    mtl.pb_eta = float(eta)
    mtl.Ni = float(eta)

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
    mtl.pb_eta = float(eta)
    mtl.Ni = float(eta)

    roughness = get_param(params, 'roughness', None)
    if roughness is not None:
        mtl.pb_roughness = float(roughness)
        mtl.Ns = roughness_to_phong(float(roughness))

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
    mtl.Ks = [0.9, 0.9, 0.9]

    # Try to resolve named spectra
    eta_val = get_param(params, 'eta')
    k_val = get_param(params, 'k')

    eta_rgb = None
    k_rgb = None

    if isinstance(eta_val, str):
        # Named spectrum lookup
        for preset_name, (p_eta, p_k) in CONDUCTOR_PRESETS.items():
            if eta_val in preset_name or preset_name in eta_val:
                eta_rgb = p_eta
                k_rgb = p_k
                break
        if eta_rgb is None:
            # Default to aluminum
            eta_rgb = [1.34, 0.96, 0.50]
            k_rgb = [7.47, 6.40, 5.30]
            mtl.comments.append(f"# Unknown conductor spectrum '{eta_val}', using aluminum defaults")
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

    mtl.pb_conductor_eta = eta_rgb
    mtl.pb_conductor_k = k_rgb

    roughness = get_param(params, 'roughness', None)
    if roughness is not None:
        mtl.pb_roughness = float(roughness)
        mtl.Ns = roughness_to_phong(float(roughness))
    else:
        mtl.pb_roughness = 0.0
        mtl.Ns = 10000.0


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
    """diffusetransmission → lambert with pb_thin flag for future extension."""
    mtl.pb_brdf = 'lambert'
    mtl.pb_semantic = 'fabric'
    mtl.pb_thin = 1
    mtl.pb_transmission = 0.5
    mtl.illum = 1

    # Reflectance texture
    ref_type = get_param_type(params, 'reflectance')
    if ref_type == 'texture':
        tex_name = get_param(params, 'reflectance')
        out = _resolve_and_register(tex_name, resolver, all_textures)
        if out:
            mtl.map_Kd = out

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
            # Scale by the PBRT scale factor (normalized — actual intensity 
            # will be set through light_scale in camera JSON)
            mtl.Ke = rgb
            mtl.comments.append(f"# Blackbody {temp_K}K, PBRT scale={light_scale}")
            return mtl

    # RGB emission
    if isinstance(bb, list) and len(bb) >= 3:
        mtl.Ke = [v * light_scale for v in bb[:3]]
    else:
        mtl.Ke = [1.0, 1.0, 1.0]

    return mtl
