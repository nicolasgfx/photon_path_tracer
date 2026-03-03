Tier 1 — Fix converter roughness + missing material handlers (biggest visual bang, Python-only changes):

Roughness remap in material_mapper.py
Add coatedconductor, thindielectric, mix handlers
Fix emissive shape material loss
Expand conductor preset table
Tier 2 — Renderer material model upgrades (C++/CUDA, moderate effort):
5. Full complex conductor Fresnel (data already parsed)
6. Bump/normal map application in shaders (texture slot already exists)
7. Bilinear texture filtering
8. Clearcoat with conductor/dielectric base (not just Lambert)
9. Thin dielectric refraction mode
10. Height-correlated Smith G2

Tier 3 — New renderer features (significant effort):
11. HDR environment map lighting
12. Anisotropic GGX on GPU
13. Kulla-Conty multi-scatter energy compensation
14. DiffuseTransmission material type
15. Mitchell-Netravali pixel filter

Tier 4 — Stretch (large architectural work):
16. Tabulated BSSRDF
17. BVH light sampler
18. Heterogeneous media (grid/VDB)
19. Curve geometry (hair)


Gap Analysis: Photon Path Tracer vs PBRT-v4
Critical Gaps (wrong visuals today)
Area	Your Renderer	PBRT-v4	Impact
Roughness chain	Converter passes PBRT roughness raw → shader squares it (
α
=
r
2
α=r 
2
 ). For PBRT roughness=0.01: your 
α
=
0.0001
α=0.0001, PBRT's 
α
=
0.1
α=0.1 — 1000x too smooth	
α
=
r
α= 
r
​
  (remaproughness=true, default)	Dominant visual mismatch for all converted scenes
Conductor Fresnel	Complex IOR (n,k) pre-baked to scalar F0 at load time via Schlick	Full per-shading-point FrComplex(cosθ, η+ik) — preserves angular color shift	Copper/gold lose characteristic edge tinting
Bump/normal mapping	Parsed into bump_tex, never applied at shading time (CPU or GPU)	Full bump mapping + normal maps	Converted scenes with bump maps render flat
Smith masking function	Separable 
G
=
G
1
(
ω
o
)
⋅
G
1
(
ω
i
)
G=G 
1
​
 (ω 
o
​
 )⋅G 
1
​
 (ω 
i
​
 )	Height-correlated 
G
2
G 
2
​
  (less masking, more energy)	Slight energy loss on rough surfaces
Multi-scatter GGX	None — single-scatter only	Kulla-Conty energy compensation table	Rough metals/dielectrics darken at grazing angles
Significant Gaps (missing features)
Area	Your Renderer	PBRT-v4	Impact
HDR environment maps	Only procedural uniform sky domes (HemisphereEnv/SphericalEnv)	InfiniteAreaLight with HDR image, luminance importance sampling	Cannot do outdoor/studio lighting — major scene limitation
Texture filtering	Nearest-neighbor (point) sampling only	MIP maps + EWA anisotropic filtering	Visible aliasing and shimmer at oblique/distant views
Anisotropic roughness	pb_roughness_x/y parsed and stored → finalization computes 
r
x
⋅
r
y
r 
x
​
 ⋅r 
y
​
 
​
  isotropic fallback. Never reaches GPU	Full anisotropic GGX with separate 
α
x
α 
x
​
 , 
α
y
α 
y
​
 	Brushed metals look wrong
Tabulated BSSRDF	Only brute-force volumetric SSS (free-flight in HomogeneousMedium)	TabulatedBSSRDF (fast dipole/searchlight) + volumetric	Dense media (skin, wax, milk) converge very slowly
Layered BSDF model	Clearcoat: analytic coat-over-Lambert only. pb_base_brdf conductor/dielectric parsed but base is always Lambert	Stochastic random-walk LayeredBxDF — coat over any base (diffuse, conductor, etc.) with inter-layer scattering medium	Coated metals (lacquered chrome, car paint) render as coated matte
Thin dielectric	pb_thin parsed, no special refraction handling	ThinDielectricBxDF — infinite internal reflections, straight-through transmission, no IOR stack push	Thin sheets (windows, lampshades) behave like solid glass
Diffuse transmission	No DiffuseTransmission material type	DiffuseTransmissionBxDF — 
f
r
=
R
/
π
f 
r
​
 =R/π, 
f
t
=
T
/
π
f 
t
​
 =T/π	Leaves, paper, thin fabric not renderable as translucent
Light tree / BVH sampler	Power-weighted CDF + binary search	BVHLightSampler (spatial light tree)	Many-light scenes (100+ emitters) have poor convergence
Moderate Gaps (quality-of-life / edge cases)
Area	Your Renderer	PBRT-v4	Impact
Pixel filter	Box filter (uniform accumulate)	Gaussian, Mitchell-Netravali, LanczosSinc	Sub-optimal anti-aliasing
Heterogeneous media	Homogeneous only	GridMedium, NanoVDBMedium	No smoke, fire, clouds
Spectral uplifting	4×3 pseudoinverse matrix (can go negative for saturated colors)	Sigmoid-polynomial (Jakob 2019) — guaranteed non-negative	Subtle metameric inaccuracies
Geometry types	Triangle meshes only	+ analytic spheres, curves (hair), bilinear patches, subdivision surfaces	Hair/fur scenes impossible
Roughness texture	specular_tex parsed, not uploaded/sampled on GPU	Per-pixel roughness via texture	Spatially-varying roughness (weathered surfaces) not possible
Displacement mapping	Not supported (bump map not even functional)	Full tessellation-based displacement	No fine geometric detail
Motion blur	Not supported	Full AnimatedTransform with per-ray shutter sampling	Stills-only limitation
What You Already Match or Exceed
Area	Status
Dielectric Fresnel (delta surfaces)	Exact — matches PBRT for mirror/glass specular bounce
Hero-wavelength MIS	Implemented — equivalent to PBRT §14.3
Chromatic dispersion (Cauchy)	Implemented with D-line anchor — PBRT-equivalent
Per-material interior media	Full Beer-Lambert + free-flight + HG phase + MediumStack on GPU
IOR stack for nested dielectrics	Implemented — correct refraction through glass-in-glass
GGX NDF + VNDF sampling	Matches PBRT's Trowbridge-Reitz + Heitz 2018 sampling
NEE with MIS	2-way power heuristic — standard practice
Photon-guided sampling	Novel feature PBRT doesn't have — directional photon density for importance sampling
Spectral bins (4-channel)	Functional equivalent to PBRT's 4-wavelength SampledSpectrum
Adjoint 
η
2
η 
2
  correction	Correct for importance (photon) transport — matches PBRT
Converter-Specific Gaps (separate from renderer capability)
Issue	Effect
Roughness not remapped	All glossy/metal surfaces 10–1000x too smooth
coatedconductor unmapped	Falls to Lambert — lacquered metals render as flat grey
thindielectric unmapped	Falls to Lambert
mix material unmapped	Blended materials fall to Lambert
diffusetransmission half-mapped	Hardcoded pb_transmission = 0.5, loses reflectance/transmittance colors
Emissive shapes lose base material	Area lights discard their surface BRDF
HDR textures clipped to uint8	EXR/PFM scaled textures destroyed by texture_processor.py
Texture UV transforms lost	PBRT uscale/vscale/udelta/vdelta ignored
Only 5 conductor presets	PBRT ships ~40 named metal spectra — most fall back to aluminum