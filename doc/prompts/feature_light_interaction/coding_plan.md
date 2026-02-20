
# Feature Coding Plan: Light interaction with homogeneous media ("crepuscular rays")

## Goal

Add physically-based light scattering/attenuation in air (participating medium) so that bright light sources produce visible shafts (“god rays”) when partially occluded.

Key requirements:

- Medium is **homogeneous by default** (constant density), but includes an optional **fall-off ratio** knob to model realistic haze concentration changes.
- **Particle density** and **fall-off ratio** must be adjustable via [src/core/config.h](src/core/config.h).
- Keep the implementation physically grounded: Beer–Lambert transmittance + single scattering first; then extend to volume/path tracing (multiple scattering) in a second step.

Non-goals (for the first iteration):

- Heterogeneous volumes from textures/VDB
- Volumetric emission (smoke fire) or colored absorption from complex spectra
- New UI panels or interactive controls beyond config defaults

---

## Physical model (what we are simulating)

We model the air as a participating medium with:

- Extinction $\sigma_t = \sigma_a + \sigma_s$ (absorption + scattering)
- Single-scattering albedo $\alpha = \sigma_s/\sigma_t$ (probability that an interaction is scattering)
- Phase function $p(\cos\theta)$ describing angular scattering

For a camera ray segment from $t=0$ to $t=t_{max}$:

1) **Transmittance (Beer–Lambert)**

$$T(t) = \exp(-\sigma_t \; t)$$

2) **In-scattering (single scattering, direct lighting only)**

$$L_{ss}(\omega_o)=\int_0^{t_{max}} T(t)\,\sigma_s\,\int_{\Omega} p(\omega_i\!\to\!\omega_o)\,L_i(x_t,\omega_i)\,d\omega_i\,dt$$

We’ll estimate the integral with Monte Carlo by sampling points along the view ray segment and performing NEE from the medium point to emissive triangles (with shadow ray visibility).

### Phase function choice (physically accurate default)

Rayleigh scattering (molecules, very small particles):

$$p_R(\mu)=\frac{3}{16\pi}(1+\mu^2),\quad \mu=\cos\theta$$

Spectral dependence (Rayleigh): $\sigma_s(\lambda) \propto 1/\lambda^4$.

Note: crepuscular rays in dusty air are often dominated by forward-scattering aerosols (Mie). Rayleigh is still a good first “physically motivated” model and matches the request; we can add Henyey–Greenstein as an optional phase later if needed.

---

## Configuration knobs (in config.h)

Add the minimal set of compile-time defaults in [src/core/config.h](src/core/config.h):

- `constexpr bool  DEFAULT_ENABLE_VOLUME = false;`
- `constexpr float DEFAULT_VOLUME_DENSITY = ...;`  
	Scales the overall interaction strength (acts as a “particles in air” knob).

- `constexpr float DEFAULT_VOLUME_FALLOFF = 0.0f;`  
	Interpreted as an exponential height falloff coefficient (world-space `y`):
	$$\rho(y)=\rho_0\,\exp(-k\,(y-y_0))$$
	- `k = DEFAULT_VOLUME_FALLOFF`
	- with `k=0` the medium is strictly homogeneous.

- `constexpr float DEFAULT_VOLUME_SIGMA_T_SCALE = 1.0f;`  
	A “fall-off ratio” alternative if you prefer distance-based attenuation; if we keep only one fall-off knob, interpret it as **height falloff** (above).

- `constexpr float DEFAULT_VOLUME_ALBEDO = 0.9f;`  
	Controls absorption vs scattering (keeps results stable without requiring separate σa/σs tuning).

- `constexpr float DEFAULT_VOLUME_RAY_MARCH_MAX_T = 5.0f;`  
	Safety cap for rays that miss geometry (normalized scenes).

- `constexpr int DEFAULT_VOLUME_SAMPLES_PER_SEGMENT = 1;`  
	Samples of the medium integral per ray segment (start with 1–2).

Thread these into runtime structs:

- Extend `RenderConfig` in [src/renderer/renderer.h](src/renderer/renderer.h) with:
	- `bool enable_volume`, `float volume_density`, `float volume_falloff`, `float volume_albedo`, `int volume_samples`.

OptiX side:

- Either rely on compile-time constants from `config.h` (fastest initial wiring), or pass them through `LaunchParams` in [src/optix/launch_params.h](src/optix/launch_params.h) if you want per-run tweaking without recompiling.

Recommended: **pass through `LaunchParams`** (keeps CPU/GPU behavior consistent and makes it easy to run tests with volume on/off).

---

## Implementation phases

### Phase 1 (MVP, physically based): single scattering + transmittance

Deliver visible crepuscular rays with correct volumetric shadows.

#### 1) Add core medium utilities

Create new header(s) under `src/core/`:

- `src/core/medium.h`
	- `struct HomogeneousMedium { Spectrum sigma_s; Spectrum sigma_a; Spectrum sigma_t; };`
	- `HD Spectrum transmittance(const HomogeneousMedium&, float distance)`
	- `HD HomogeneousMedium make_rayleigh_air(float density, float albedo, float falloff, float y)`
		- Use `lambda_of_bin(i)` from [src/core/spectrum.h](src/core/spectrum.h).
		- Rayleigh spectral shape: `rayleigh(i) = powf(lambda_ref / lambda, 4)`.
		- Convert density + albedo into σ terms:
			- Choose a base extinction `sigma_t_ref = density * scale` (scale in scene units).
			- `sigma_s = alpha * sigma_t_ref * rayleigh_spectrum`
			- `sigma_a = (1-alpha) * sigma_t_ref` (gray absorption is acceptable initially).

- `src/core/phase_function.h`
	- `HD float rayleigh_phase_pdf(float cos_theta)` returning $\frac{3}{16\pi}(1+\cos^2\theta)$.
	- (No sampling needed for Phase 1, only evaluation.)

Keep these headers host+device friendly (`HD`) so they can be used from:

- CPU integrator: [src/renderer/renderer.cpp](src/renderer/renderer.cpp)
- OptiX device code: [src/optix/optix_device.cu](src/optix/optix_device.cu)

#### 2) Add volumetric NEE estimator (single scattering)

Implement a function for a ray segment `[origin, origin + dir * t_end]`:

- Inputs:
	- `origin`, `dir`, `t_end`
	- medium parameters (or `HomogeneousMedium` at sampled point)
	- scene/light sampling access (same emissive CDF used by direct lighting)
	- RNG

- Estimator (stratified distance sampling):
	- For `j in 0..N-1` sample $t$ uniformly in `[0, t_end]` (or stratified)
	- Compute point `x = origin + dir * t`
	- Evaluate density falloff at `x.y` (optional)
	- Compute `T_cam = exp(-sigma_t * t)` per wavelength
	- Sample a light point using the existing NEE code path (emissive triangle CDF)
	- Shadow test `x -> light` using existing shadow-ray code
	- Compute `T_light = exp(-sigma_t * dist_to_light)` per wavelength
	- Phase term uses `cos_theta = dot(wi_to_light, -dir)` and Rayleigh phase
	- Contribution:
		$$L \mathrel{+}= \frac{t_{end}}{N}\; T_{cam}(t)\,\sigma_s\,p(\cos\theta)\,T_{light}\,L_e\,G\,/\,pdf_{light}$$
		where `G` is the usual geometry term for the sampled emissive triangle point.

Notes:

- This is an unbiased MC estimator of the single-scattering integral (up to numeric precision).
- Start with `N=1` and increase if noise is too high in shafts.

#### 3) Integrate into CPU path tracer

Modify [src/renderer/renderer.cpp](src/renderer/renderer.cpp) in `Renderer::trace_path()`:

- At each bounce before handling the hit shading, we have a segment length:
	- if hit: `t_end = hit.t`
	- if miss: `t_end = DEFAULT_VOLUME_RAY_MARCH_MAX_T`

- If volume enabled:
	1) Accumulate `L_total += throughput * L_single_scatter_segment(...)`.
	2) Apply segment transmittance to throughput: `throughput *= T(t_end)`.

- Then proceed with existing surface logic (NEE/photon/BSDF).

Initial scope suggestion:

- Apply medium only on the **camera ray segment to the first surface** (bounce 0), which already produces crepuscular rays.
- Extend to all segments later (still Phase 1, but more expensive).

#### 4) Integrate into OptiX path tracer

Modify [src/optix/optix_device.cu](src/optix/optix_device.cu) in `full_path_trace()`:

- After `TraceResult hit = trace_radiance(origin, direction)` and before emissive/specular/diffuse handling:
	- Determine `t_end` (hit `t` or max)
	- If volume enabled:
		- Add single-scattering term for this segment into `result.combined` (and optionally into `result.nee_direct` since it’s “direct-from-light”, just volumetric)
		- Multiply `throughput` by transmittance for the segment

- Reuse existing pieces:
	- emissive triangle CDF sampling already exists in `dev_nee_direct` / `dev_nee_guided`
	- shadow visibility exists (`trace_shadow`)

Implementation approach:

- Factor out a device helper `dev_volume_single_scatter_segment(...)` similar to `dev_nee_direct` but operating at a point in space (no surface frame), using Rayleigh phase.

#### 5) Parameter plumbing (RenderConfig → LaunchParams)

If we want runtime control without recompiling:

- Extend `LaunchParams` in [src/optix/launch_params.h](src/optix/launch_params.h) with:
	- `int enable_volume;`
	- `float volume_density;`
	- `float volume_falloff;`
	- `float volume_albedo;`
	- `int volume_samples;`
	- `float volume_max_t;`

- Populate those fields in host code when uploading launch params in [src/optix/optix_renderer.cpp](src/optix/optix_renderer.cpp) (paths: `render_debug_frame`, `render_one_spp`, `render_final`, and bin population launches if needed).

---

### Phase 2 (next step): full volumetric path tracing

Move from “add a volumetric direct term” to true participating-media transport.

Target behavior:

- Medium interactions become path vertices (like surfaces).
- Multiple scattering supported.
- NEE can be performed at medium vertices as well.

Algorithm (homogeneous medium, distance sampling):

For each ray segment:

1) Sample a free-flight distance:
$$s = -\ln(1-u)/\sigma_t$$

2) Compare with surface distance `t_surface`:

- If `s < t_surface`: medium interaction occurs
	- throughput *= `T(s)`
	- choose scatter vs absorb:
		- scatter probability = `alpha`
	- at scatter:
		- NEE from medium point to lights (phase function)
		- sample new direction from phase (Rayleigh or HG)
		- continue

- Else: surface interaction occurs
	- throughput *= `T(t_surface)`
	- continue with existing surface shading

Spectral note:

- Full spectral free-flight sampling is non-trivial when $\sigma_t(\lambda)$ varies.
- Options:
	1) “Gray tracking” for sampling distance (use scalar σt) + per-λ transmittance (approx)
	2) Hero-wavelength paths (sample one λ per path) (bigger refactor)
	3) Delta tracking with a majorant σ (most correct, more code)

Recommendation: keep Phase 2 scoped by starting with (1) and clearly documenting the approximation, then iterate to (3) if you want strict spectral correctness.

---

## Validation & tests

Add small, deterministic tests (fast) and optional render comparisons (slower).

### Unit tests (fast)

Add a new test file, e.g. `tests/test_medium.cpp`:

- `Transmittance_BeerLambertMatchesExpected`
	- check `T(d)=exp(-sigma_t*d)` for a few known values.

- `RayleighPhase_Normalizes`
	- numerically integrate $p_R(\mu)$ over sphere and ensure it’s ~1 (coarse quadrature).

- `RayleighSpectrum_IsBlue`
	- ensure `sigma_s(380nm) > sigma_s(780nm)` and ratio roughly matches $(780/380)^4`.

### Integration tests (existing harness)

Extend per-ray validation / pixel comparison only after Phase 1 stabilizes:

- In `test_per_ray_validation.cpp`, add a “volume on/off” mode that:
	- asserts non-negativity and finiteness of volume contributions
	- checks that enabling volume reduces surface radiance behind long distances (transmittance)

Avoid hard ground-truth images initially; use invariant checks (energy decreases with higher density).

---

## Performance considerations

- Single scattering cost is dominated by shadow rays from medium samples.
- Keep `DEFAULT_VOLUME_SAMPLES_PER_SEGMENT` small (1–2) and rely on progressive sampling (spp) to converge.
- Only do volumetric sampling on bounce 0 at first.
- Reuse existing emissive CDF sampling; do not iterate all lights.

---

## File touch-points (repo-specific)

- Config knobs: [src/core/config.h](src/core/config.h)
- CPU integrator: [src/renderer/renderer.cpp](src/renderer/renderer.cpp)
- GPU integrator: [src/optix/optix_device.cu](src/optix/optix_device.cu)
- OptiX param plumbing: [src/optix/launch_params.h](src/optix/launch_params.h), [src/optix/optix_renderer.cpp](src/optix/optix_renderer.cpp)
- New core utilities: `src/core/medium.h`, `src/core/phase_function.h` (planned)
- Tests: `tests/test_medium.cpp` (planned)

---

## Acceptance criteria

- With volume enabled and non-zero density:
	- Visible crepuscular rays appear where light is partially occluded.
	- Shafts fade correctly with distance (Beer–Lambert) and respond to falloff knob.
	- Output remains stable (no NaNs/negative spectra).

- With volume disabled or density=0:
	- Render matches current baseline (within noise) and existing tests remain valid.

