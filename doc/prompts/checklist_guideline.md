# Spectral Photon + Path Tracing Renderer

## Extended Execution Sequence Specification

*(Extension of Copilot-Optimized Implementation Specification)*

This section defines the **precise chronological execution flow** from
photon emission to final image synthesis.

All steps are deterministic and mathematically explicit.

------------------------------------------------------------------------

# A. PHOTON PASS -- FULL SEQUENCE

------------------------------------------------------------------------

## A1. Precomputation: Emissive Geometry Analysis

1.  Iterate over all triangles.
2.  If material is emissive:
    -   Compute area:

\[ A_t = `\frac{1}{2}`{=tex} \| (v_1 - v_0) `\times `{=tex}(v_2 - v_0)
\| \]

-   Estimate average emitted radiance ( `\bar`{=tex}{L}\_{e,t} )
-   Compute weight:

\[ w_t = A_t `\cdot `{=tex}`\bar`{=tex}{L}\_{e,t} \]

3.  Build alias table over weights.

------------------------------------------------------------------------

## A2. Photon Emission Loop (for N photons)

For each photon:

### A2.1 Select Emissive Triangle

Sample triangle index using alias table.

### A2.2 Sample Surface Point

Uniform barycentric sampling:

\[ `\alpha `{=tex}= 1 - `\sqrt{u}`{=tex}, `\quad`{=tex} `\beta `{=tex}=
v`\sqrt{u}`{=tex} \]

\[ x = `\alpha `{=tex}v_0 + `\beta `{=tex}v_1 +
(1-`\alpha`{=tex}-`\beta`{=tex})v_2 \]

### A2.3 Sample Emission Direction

For diffuse area light:

\[ `\omega `{=tex}= `\text{cosine-weighted hemisphere}`{=tex}(n_x) \]

### A2.4 Sample Wavelength

Construct discrete spectral PDF:

\[ p(`\lambda`{=tex}\_i\|x) =
`\frac{L_e(x,\lambda_i)}{\sum_j L_e(x,\lambda_j)}`{=tex} \]

Sample bin ( `\lambda`{=tex}\_k ).

### A2.5 Compute Photon Flux

\[ `\Phi `{=tex}= `\frac{L_e(x,\omega,\lambda_k)\cos\theta}`{=tex}
{p(t),p(x\|t),p(`\omega`{=tex}\|x),p(`\lambda`{=tex}\_k\|x)} \]

Store photon throughput T = Φ.

------------------------------------------------------------------------

## A3. Photon Transport

While photon alive:

1.  Trace ray using OptiX.
2.  If miss → terminate.
3.  At hit:
    -   If diffuse → store photon
    -   Evaluate BSDF
    -   Sample next direction
    -   Update throughput:

\[ T `\leftarrow `{=tex}T
`\cdot `{=tex}`\frac{f_s \cos\theta}{p(\omega)}`{=tex} \]

4.  Apply Russian Roulette.

------------------------------------------------------------------------

# B. BUILD PHOTON HASH GRID

1.  Compute cell coordinate:

```{=html}
<!-- -->
```
    cellCoord = floor(position / cellSize)

2.  Hash cellCoord → key.
3.  Radix sort by key.
4.  Build cellStart/cellEnd arrays.
5.  Store compacted photon list.

------------------------------------------------------------------------

# C. CAMERA PATH TRACING -- FULL SEQUENCE

------------------------------------------------------------------------

## C1. Primary Ray Generation

For each pixel:

1.  Generate camera ray.
2.  Initialize spectral throughput T = 1.
3.  Initialize radiance accumulator L = 0.

------------------------------------------------------------------------

## C2. Path Loop (per bounce)

While bounce \< maxDepth:

1.  Intersect scene via OptiX.
2.  If miss:
    -   Add environment radiance.
    -   Terminate.
3.  At hitpoint x:

------------------------------------------------------------------------

### C2.1 Direct Lighting (Soft Shadows)

For each light sample:

1.  Sample point on light.
2.  Compute direction ( `\omega`{=tex}\_l ).
3.  Shadow ray to light.
4.  If visible:

\[ L += T `\cdot `{=tex}f_s `\cdot `{=tex}L_e `\cdot `{=tex}G /
p\_{light} \]

Soft shadowing emerges from area light sampling.

------------------------------------------------------------------------

### C2.2 Photon Density Estimation

1.  Gather photons within radius r.
2.  Apply surface consistency filter.
3.  Compute:

\[ L\_{photon} = `\frac{1}{\pi r^2}`{=tex} `\sum`{=tex}\_i
`\Phi`{=tex}\_i f_s \]

4.  Add to radiance:

```{=html}
<!-- -->
```
    L += T * L_photon;

------------------------------------------------------------------------

### C2.3 Determine Path Continuation

Select strategy via probabilities:

-   Light sampling
-   BSDF sampling
-   Photon-guided sampling

------------------------------------------------------------------------

### C2.4 Sample New Direction

#### Option 1: BSDF

Cosine hemisphere sampling.

#### Option 2: Photon-Guided

1.  Gather neighbors.
2.  Build discrete PDF:

\[ q(`\omega`{=tex}\_i) `\propto `{=tex}`\Phi`{=tex}\_i \]

3.  Sample photon index.
4.  Set next direction = photon.wi.

------------------------------------------------------------------------

### C2.5 MIS Weighting

Compute PDFs for all strategies.

Power heuristic:

\[ w = `\frac{p_a^2}{p_a^2 + p_b^2 + p_c^2}`{=tex} \]

Update contribution accordingly.

------------------------------------------------------------------------

### C2.6 Throughput Update

\[ T `\leftarrow `{=tex}T
`\cdot `{=tex}`\frac{f_s \cos\theta}{p(\omega)}`{=tex} \]

Apply Russian roulette after minimum depth.

------------------------------------------------------------------------

# D. ENERGY FLOW CONSISTENCY

At all stages ensure:

-   Spectral throughput updated per wavelength bin.
-   Flux stored per wavelength bin.
-   No mixing of wavelength bins.
-   Energy conservation of BSDF verified by tests.

------------------------------------------------------------------------

# E. SPECTRAL COLORING

Final pixel color:

1.  Convert spectral radiance to XYZ using CIE curves.
2.  Convert XYZ to linear RGB.
3.  Apply tone mapping.
4.  Gamma correction.

------------------------------------------------------------------------

# F. SOFT SHADOWING

Soft shadows arise from:

-   Area light sampling in direct lighting.
-   Occlusion via shadow rays.
-   No photon involvement required.

------------------------------------------------------------------------

# G. FINAL FRAME ACCUMULATION

For progressive rendering:

\[ L\_{final} = `\frac{1}{N}`{=tex} `\sum`{=tex}\_{s=1}\^N L_s \]

Where each sample includes full photon + path evaluation.

------------------------------------------------------------------------

# H. DEBUG EXECUTION VISIBILITY

At each stage expose:

-   Photon emission positions
-   Photon flux magnitude
-   Gathered photon count
-   Selected sampling strategy
-   MIS weights
-   Throughput evolution per bounce
-   Spectral bin contributions

------------------------------------------------------------------------

# I. END-TO-END EXECUTION SUMMARY

1.  Analyze emissive triangles
2.  Emit photons with correct spectral flux
3.  Build spatial hash grid
4.  For each pixel:
    -   Trace primary ray
    -   Compute direct lighting
    -   Gather photon density
    -   Perform MIS-guided continuation
    -   Accumulate spectral radiance
5.  Convert spectral → RGB
6.  Display image

------------------------------------------------------------------------

This document defines a complete physically consistent light transport
pipeline from photon emission to final spectral image synthesis.
