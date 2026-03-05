PBRT v4 handles translucency/subsurface scattering through volumetric path tracing with participating media:

The subsurface material (like the dragon scene) defines scattering coefficients (
σ
a
σ 
a
​
 , 
σ
s
σ 
s
​
 ) and phase function (typically Henyey-Greenstein). The "name" parameter (e.g. "Skin1") selects from a table of measured scattering coefficients from Jensen et al. 2001. The "scale" parameter adjusts the mean free path.

Rendering mechanism: When a ray enters the surface (refracted via Snell's law using eta), PBRT treats the interior as a participating medium. It uses:

Free-flight sampling: exponentially distributed step distances based on 
σ
t
=
σ
a
+
σ
s
σ 
t
​
 =σ 
a
​
 +σ 
s
​
 
Scattering events: at each scatter point, a new direction is sampled from the phase function
Absorption: Beer-Lambert attenuation 
e
−
σ
a
⋅
d
e 
−σ 
a
​
 ⋅d
  along each segment
The ray eventually exits the surface at a different point → giving the characteristic soft, translucent look
The key equation is the volumetric rendering equation (PBRT v4 §14.1):

L
(
p
,
ω
)
=
∫
0
t
m
a
x
T
r
(
p
,
p
′
)
[
σ
a
L
e
(
p
′
,
ω
)
+
σ
s
∫
S
2
f
p
(
ω
′
,
ω
)
L
i
(
p
′
,
ω
′
)
 
d
ω
′
]
d
t
L(p,ω)=∫ 
0
t 
max
​
 
​
 T 
r
​
 (p,p 
′
 )[σ 
a
​
 L 
e
​
 (p 
′
 ,ω)+σ 
s
​
 ∫ 
S 
2
 
​
 f 
p
​
 (ω 
′
 ,ω)L 
i
​
 (p 
′
 ,ω 
′
 )dω 
′
 ]dt

where 
T
r
T 
r
​
  is the transmittance (Beer-Lambert).

Your renderer already has participating media support (the media vector in Scene, and the sigma_a/sigma_s scaling in normalize_to_reference). The missing piece for the sssdragon is connecting the PBRT "subsurface" material's named presets to your medium system — currently map_subsurface only maps it as Lambert diffuse without creating an interior medium.