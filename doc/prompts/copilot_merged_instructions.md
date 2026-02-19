# Spectral Photon + Path Tracing Renderer — Single Copilot Instruction Set

Audience: **GitHub Copilot** implementing/maintaining this renderer.

Priority order:
1. **Physical correctness** (unbiased estimators, correct PDFs, no double counting)
2. **Deterministic + debuggable** (component outputs, clear invariants)
3. **Simplicity over performance** (fixed radii, fixed bins, explicit code)

OptiX is mandatory for all ray tracing in the final product. Do not add a CPU rendering fallback.

---

## 0) Non‑Negotiable Invariants (Read First)

- **Never double count direct lighting**.
  - Direct illumination at camera hitpoints is estimated by **NEE only** (shadow rays).
  - Photon maps must **not** contain “direct-from-light to first diffuse” deposits.
- **Every Monte Carlo estimator must divide by the exact PDF of how it sampled**.
- **Spectral bins never mix during transport**. Convert spectral → RGB only at output.
- **Photons store radiant flux per wavelength bin** (a power packet), not radiance.

---

## 1) Physical Units & Definitions

Rendering equation (spectral):

$$
L_o(x, \omega_o, \lambda) = \int_{\Omega} L_i(x, \omega_i, \lambda)\, f_s(x, \omega_i, \omega_o, \lambda)\, \cos\theta_i\, d\omega_i
$$

- Radiance $L$ : $[W\,/(sr\cdot m^2\cdot nm)]$
- Flux $\Phi$ : $[W\,/nm]$
- Irradiance $E$ : $[W\,/(m^2\cdot nm)]$

A stored photon represents **radiant flux** in one wavelength bin.

---

## 2) Core Data Structures (Conceptual)

- `Spectrum`: fixed-size array of `NUM_LAMBDA` bins.
- `Photon` (SoA on GPU):
  - `position`
  - `wi` (incoming direction at the deposit point, pointing **into** the surface)
  - `lambda_bin`
  - `flux` (radiant flux in that bin)
- `HashGrid`: hashed uniform grid for neighbor queries within fixed radius.

---

## 3) Single Shared Light Distribution (Photon Emission + NEE)

Build ONE distribution over all emissive triangles and reuse it for:
- photon emission (light → scene)
- NEE (surface hitpoint → light)

Triangle weight:

$$
w_t = A_t \cdot \bar{L}_{e,t}
$$

- $A_t$ = triangle area
- $\bar{L}_{e,t}$ = a deterministic “average emission proxy” (e.g., mean of `Le` spectrum; for textured emission, sample a small fixed set of UVs)

Implementation may use an **alias table** or a **CDF**. What matters is:
- sampling returns `tri_id`
- you can query `p_tri` = normalized probability of selecting that triangle.

---

## 4) Photon Pass (Build Photon Maps)

### 4.1 Emit 1 Photon (Per-photon steps)

A) Sample emissive triangle: `(tri_id, p_tri)` from the shared distribution.

B) Sample uniform point on triangle (area-uniform):

$$
\alpha = 1-\sqrt{u},\quad \beta = v\sqrt{u},\quad \gamma = 1-\alpha-\beta
$$

$$
x = \alpha v_0 + \beta v_1 + \gamma v_2
$$

PDF on area of chosen triangle:

$$
p_{pos} = 1/A_{tri}
$$

C) Sample emission direction.
- For a Lambertian emitter: cosine-weighted hemisphere oriented around triangle normal.

$$
p_{dir}(\omega) = \cos\theta/\pi
$$

D) Sample wavelength bin proportional to emission spectrum at the sampled point:

$$
p_{\lambda}(i\mid x) = \frac{L_e(x,\lambda_i)}{\sum_j L_e(x,\lambda_j)}
$$

E) Compute initial photon flux (power packet) for the chosen bin:

$$
\Phi = \frac{L_e(x,\omega,\lambda)\,\cos\theta}{p_{tri}\, p_{pos}\, p_{dir}\, p_{\lambda}}
$$

Correctness note: Sampling triangle weights by $A\cdot\bar{L_e}$ is fine; unbiasedness is preserved because $\Phi$ divides by the *actual* sampling PDF.

### 4.2 Photon Transport (Bounce loop)

Maintain:
- `flux` (scalar for the photon’s single `lambda_bin`)
- `hasSpecularChain` (true if any specular event occurred since emission)
- `lightPathDepth` (number of surface interactions so far after emission; define unambiguously)

Update rule at each bounce (conceptual throughput):

$$
flux \leftarrow flux \cdot \frac{f_s\cos\theta}{p(\omega)}
$$

- For Lambertian sampled with cosine hemisphere, $f_s=\rho/\pi$ and $p(\omega)=\cos\theta/\pi$, so the factor reduces to $\rho$.

### 4.3 Photon Deposition Rule (Critical: avoids double counting)

We must ensure photon maps contain **indirect** (and caustic) energy, not direct.

Define **lightPathDepth** as:
- 1 at the *first* surface hit after emission,
- 2 at the second surface hit, etc.

Deposit photons only when:
- hit material event is **diffuse-like** (non-delta)
- AND `lightPathDepth >= 2`

This is equivalent to “skip the first diffuse hit from the light source”.

### 4.4 Global vs Caustic Maps

- **Global photon map**: deposited diffuse hits with `hasSpecularChain == false`.
- **Caustic photon map**: deposited diffuse hits with `hasSpecularChain == true`.

Caustic photons typically use a smaller gather radius.

---

## 5) Spatial Hash Grid (Fixed-Radius Neighbor Search)

- Choose `cellSize ≈ 2 * radius`.
- Key = hash(floor(pos / cellSize)).
- Sort photons by key; build `cellStart[key] / cellEnd[key]`.
- Query scans neighbor cells covering the radius (commonly 3×3×3 when cellSize=2r).

Correctness note: Because different cells can collide into the same hash bucket, avoid double-processing a bucket during a query.

---

## 6) Photon Density Estimator (Indirect + Caustics)

At a diffuse camera hitpoint $x$ with outgoing direction $\omega_o$:

$$
L_{photon}(x,\omega_o,\lambda) = \frac{1}{\pi r^2}\sum_{i\in N(x)} \Phi_i(\lambda)\, f_s(x,\omega_i,\omega_o,\lambda)
$$

Optional radial kernel (e.g., Epanechnikov):
- Multiply each photon by kernel weight $W(\|x-x_i\|)$.
- If you use Epanechnikov, apply the correct normalization constant.

### 6.1 Surface Consistency Filter (Recommended)

Reject photon $i$ unless:

- hemisphere consistency: photon arrives from above surface
  - `dot(wi_photon, n_x) > 0` if `wi_photon` points into the surface
- plane-distance check:

$$
|n_x \cdot (x_i - x)| < \tau
$$

Purpose: avoid cross-surface contamination through thin walls/gaps.

---

## 7) Camera Pass (Hybrid Path Tracing)

Per camera path:
- `throughput T(λ) = 1`
- `radiance L(λ) = 0`

### 7.1 Direct Lighting: NEE (Soft Shadows)

At a diffuse hitpoint:
1) sample `(tri_id, p_tri)` from shared light distribution
2) sample uniform point `y` on that triangle (PDF `1/A`)
3) compute `wi = normalize(y-x)`, distance
4) cast shadow ray (x→y); if occluded: 0
5) evaluate `Le(y, -wi, λ)` and BSDF `f(x, wi, wo, λ)`

Convert PDF from area to solid angle:

$$
 p_{y,area} = p_{tri}\cdot\frac{1}{A_{tri}},\quad
 p_{\omega} = p_{y,area}\cdot\frac{\|x-y\|^2}{\cos\theta_y}
$$

Contribution per sample:

$$
L_{direct} += \frac{f\,Le\,\cos\theta_x}{p_{\omega}}
$$

Average over $M$ NEE samples.

### 7.2 Indirect Lighting: Photon Density

Add photon-map estimate separately:

$$
L += T \cdot L_{photon}
$$

### 7.3 Path Continuation (Optional; if enabled)

Strategies (conceptual):
- BSDF sampling
- photon-guided sampling
- (optional) light sampling as continuation (if you do full MIS path tracing)

If combining multiple proposals, use power heuristic:

$$
 w_k = \frac{p_k^2}{\sum_j p_j^2}
$$

Correctness note: If photon-guided sampling is a **discrete** proposal over a set of directions, you must be explicit about the measure and how its PDF is compared in MIS against continuous PDFs. If this is unclear, disable photon-guided MIS until the measure is defined unambiguously.

---

## 8) Spectral → RGB Output

After accumulation:
1. integrate spectrum against CIE XYZ curves
2. convert XYZ → linear sRGB
3. tone map
4. gamma correct

Perform the exact same conversion for component buffers so they are visually comparable.

---

## 9) Required Debug / Component Outputs

Every final render iteration/frame must be able to output at least:
- `out_nee_direct.png` (NEE-only direct component)
- `out_photon_indirect.png` (photon density only)
- `out_combined.png` (sum of the above)

Recommended additional outputs (high value for debugging):
- `out_photon_caustic.png`
- `out_indirect_total.png` (= global + caustic)
- `out_photon_count.png` (gathered photon count)

### 9.1 File Naming Convention

When writing multiple frames/iterations, prefix with a frame counter:
- `frame_0001_out_nee_direct.png`
- `frame_0001_out_photon_indirect.png`
- `frame_0001_out_photon_caustic.png`
- `frame_0001_out_indirect_total.png`
- `frame_0001_out_combined.png`

---

## 9b) Debug Viewer UX (Project Requirement)

### Key Bindings

- F1: toggle photon points
- F2: toggle global map
- F3: toggle caustic map
- F4: toggle hash grid cells
- F5: toggle photon directions
- F6: toggle PDF display
- F7: toggle radius sphere
- F8: toggle MIS weights
- F9: toggle spectral coloring mode
- TAB: cycle render modes

### Hover Cell Overlay

When the mouse hovers a hash-grid cell, display at least:
- cell coordinate
- photon count
- sum flux
- average flux
- dominant wavelength bin and wavelength (nm)
- gather radius
- map type (global/caustic)

---

## 9c) Render Modes (Debug + Validation)

Expose a single `RenderMode` enum and ensure each mode is a strict subset of terms:

- **First-hit debug**: normals/material ID/depth (no photon gather, no continuation)
- **DirectOnly**: direct lighting terms only (NEE + specular-chain visible emission if supported)
- **IndirectOnly**: photon density terms only
- **Full/Combined**: direct + indirect
- **PhotonMap**: visualize the photon density estimate itself (not a lit result)

Correctness requirement: if a mode disables NEE, it must also not include any direct-light photon deposits (those must never be stored).

---

## 10) Minimal Acceptance Tests (Must Pass)

1. Direct-only (photons off): soft shadows converge, brightness stable.
2. Indirect-only (NEE off): no “direct-lit” hotspot patterns.
3. Combined: `combined ≈ nee_direct + photon_indirect (+ caustic if present)`.
4. Cornell box: indirect color bleeding visible.
5. Glass caustics: concentrated patterns show up in caustic component.

---

## 11) Complete Execution Order (Single Frame)

1. Load scene (OBJ/MTL).
2. Identify emissive triangles.
3. Build shared light distribution (alias table or CDF) over emissive triangles.
4. Photon pass:
  - emit photons with tracked PDFs and spectral sampling
  - trace/scatter photons
  - deposit photons using the “no-direct-deposits” rule
  - build hash grid(s) (global + caustic)
5. Camera pass:
  - for each pixel: trace camera ray
  - at hits: add NEE direct (if enabled)
  - gather photon density (global + caustic) (if enabled)
  - continue path with BSDF sampling (and MIS if enabled)
6. Accumulate spectral radiance buffers.
7. Convert spectral → RGB and write PNGs (final + components).

---

## 12) Common Bugs to Avoid

- Forgetting area→solid-angle Jacobian (`dist^2 / cos_y`) in NEE.
- Using the wrong cosine in that Jacobian (must be emitter-side `cos_y`).
- Depositing photons at the first diffuse hit from the light (double counts with NEE).
- Using triangle-uniform sampling instead of power/area-weighted emission.
- Not offsetting shadow rays / new rays with an epsilon.
- Mixing wavelength bins during transport.

---

## 13) Unit Test Coverage Expectations

At minimum, keep tests covering:
- alias table / CDF sampling correctness (PDF sums to 1, sampling matches PDF)
- triangle sampling uniformity
- NEE PDF conversion correctness (area → solid angle)
- hash grid build/query correctness (distance filter + collision handling)
- density estimator surface consistency filters
- spectral conversions

---

## 14) “Reality Check” Notes for This Repo

This document is a **target** spec. When implementing, always reconcile with:
- the OptiX device programs (final behavior)
- the host-side OptiX glue (what gets uploaded / which buffers exist)

If any part of the spec conflicts with the running OptiX path, prefer:
1) fixing the code (if it’s a bug), or
2) updating this spec (if the spec was too ambitious/unclear).
