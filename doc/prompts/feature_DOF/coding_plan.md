# Feature: Physically-based Depth of Field (DOF)

## Goal
Add **physically-based depth of field** using a **thin-lens camera model** (stochastic lens sampling during primary ray generation).

This is **not** a depth-map blur / post-process effect.

### Success criteria (acceptance)
- DOF is visible in the **interactive debug preview** (OptiX `__raygen__render` first-hit path).
- DOF is also applied to the **final progressive render** path.
- When DOF is disabled (aperture = 0), output matches current pinhole behavior (ideally bitwise-identical for a given RNG sequence).
- Configuration is controlled via `src/core/config.h` defaults and flows cleanly through the existing pipeline (camera → launch params → raygen).

## Scope / Non-goals
### In scope
- Circular aperture thin-lens model.
- Configurable focus distance and aperture (via f-number or explicit lens radius).
- Works for both:
	- debug first-hit rendering
	- full path tracing rendering

### Not in scope (for this feature)
- Depth-map based blur / post-process.
- Polygonal bokeh blades, cat-eye bokeh, vignetting.
- Chromatic aberration, sensor response curves.
- Motion blur / rolling shutter.

## Current pipeline touchpoints (repo-specific)
- Camera model lives in `src/renderer/camera.h` (currently pinhole).
- Interactive debug preview and final render both generate primary rays in `src/optix/optix_device.cu` (`__raygen__render`).
- Host packs camera values into `LaunchParams` in `src/optix/optix_renderer.cpp` via `fill_common_params()`.
- OptiX launch params are defined in `src/optix/launch_params.h`.
- Debug overlays / hover queries compute a pinhole ray in `src/main.cpp` (e.g., hover inspector). Keep these as **pinhole** for stable picking unless explicitly changed later.

## Camera model design
### Thin lens (physically-based DOF)
For each pixel sample:
1. Sample a point on the lens aperture (disk).
2. Compute the target point on the **focus plane** corresponding to the pixel sample.
3. Ray origin = lens sample point; ray direction = normalized(target - origin).

This yields physically plausible blur for out-of-focus geometry.

### Parameterization
We already use `fov_deg` + resolution, and scene units are normalized to roughly a unit cube (camera around `z≈2.5`).

To keep parameters intuitive and “physical”, use:
- **Focus distance** $d_f$ (scene units; treat as meters).
- **F-number** $N$ (unitless).
- A chosen **sensor height** $H$ (scene units; default 0.024 for a 24mm full-frame height), which defines a physical focal length for the given FOV:

$$f = \frac{H}{2\tan(\mathrm{fov}/2)}$$

Then lens radius is:

$$r = \frac{f}{2N}$$

Notes:
- This makes **aperture affect blur size**.
- If we also want aperture to affect brightness (more physically photometric), we can optionally scale sensor response by lens area $\pi r^2$. See “Exposure handling” below.

### Exposure handling (decision point)
There are two common choices:
1. **Photometric**: changing aperture changes exposure (smaller aperture → darker image).
2. **Artist-friendly / auto-exposure**: keep overall brightness roughly constant when changing aperture.

For “physically correct” intent, default to (1), but consider adding a config toggle to keep current look stable if needed.

## Configuration (via `src/core/config.h`)
Add new defaults near other “iteration knobs”:
- `constexpr bool  DEFAULT_DOF_ENABLED = false;`
- `constexpr float DEFAULT_DOF_FOCUS_DISTANCE = 2.5f;`  // scene units
- `constexpr float DEFAULT_DOF_F_NUMBER = 2.8f;`        // f/2.8 etc
- `constexpr float DEFAULT_DOF_SENSOR_HEIGHT = 0.024f;` // 24mm full-frame height
- `constexpr bool  DEFAULT_DOF_PHOTOMETRIC_EXPOSURE = true;` // optional

Also consider adding per-scene overrides (optional) if different scenes need different focus distances.

## Implementation plan (step-by-step)

### 1) Add sampling utility: concentric disk
Why: uniform sampling over a circular aperture is required.

Implement in `src/core/random.h` as `inline HD float2 sample_concentric_disk(float u1, float u2)`.
- Use Shirley & Chiu concentric mapping (stable, low distortion).
- Keep it `HD` so it can be used by both host C++ and device CUDA.

Deliverable:
- Unit-tested (optional but nice): basic invariants (radius <= 1, no NaNs).

### 2) Extend `Camera` struct for DOF
Modify `src/renderer/camera.h`:
- Add fields:
	- `bool  dof_enabled;`
	- `float dof_focus_dist;`
	- `float dof_f_number;`
	- `float sensor_height;`
	- Derived: `float focal_length;` and `float lens_radius;`

Update `Camera::update()`:
- Compute `u,v,w` exactly as today.
- Compute `focal_length` from `sensor_height` and `fov_deg`.
- Compute `lens_radius = dof_enabled ? focal_length / (2 * dof_f_number) : 0`.
- Build the view plane at the **focus distance**:
	- `horizontal` and `vertical` should be scaled by `dof_focus_dist`.
	- `lower_left = position - 0.5*horizontal - 0.5*vertical - w*dof_focus_dist`.
- When `dof_enabled==false` or `lens_radius==0`, set `dof_focus_dist = 1` (or leave it but lens_radius=0 is enough), so ray directions match the old pinhole behavior.

Update `Camera::generate_ray()`:
- Keep existing AA jitter.
- If `lens_radius == 0`: use current pinhole origin/direction.
- Else:
	- Sample `disk = sample_concentric_disk(rng.next_float(), rng.next_float())`.
	- `lens_offset = (u * disk.x + v * disk.y) * lens_radius`.
	- `ray.origin = position + lens_offset`.
	- `target = lower_left + horizontal*s + vertical*t`.
	- `ray.direction = normalize(target - ray.origin)`.

### 3) Wire config defaults into camera creation
Modify camera initialization in `src/main.cpp` (where camera is set from `SCENE_CAM_*`):
- Set the new DOF fields from `DEFAULT_DOF_*`.

Important: interactive mode currently sets `camera.look_at = camera.position + forward;` each frame.
- DOF focus distance must **not** rely on `distance(position, look_at)`.
- Keep `camera.dof_focus_dist` as an explicit parameter that persists while moving.

### 4) Add DOF parameters to OptiX launch params
Modify `src/optix/launch_params.h`:
- Add:
	- `float cam_lens_radius;`
	- `float cam_focus_dist;` (optional if lower_left/h/v already embed it, but useful for clarity)
	- Optionally `int cam_dof_enabled;` (or just use `cam_lens_radius > 0`).

Modify `src/optix/optix_renderer.cpp` (`fill_common_params()`):
- Pack the new values from `Camera` into `LaunchParams`.

### 5) Implement thin-lens raygen in OptiX device code
Modify `src/optix/optix_device.cu` in `__raygen__render`:
- Compute the “pinhole target” point for the pixel sample:
	- `target = params.cam_lower_left + params.cam_horizontal*u + params.cam_vertical*v;`
- If `params.cam_lens_radius > 0`:
	- Sample disk (same concentric mapping; either include a small device helper in `optix_device.cu` or reuse from `core/random.h` if it compiles for device).
	- `lens_offset = (params.cam_u * disk.x + params.cam_v * disk.y) * params.cam_lens_radius;`
	- `origin = params.cam_pos + lens_offset;`
	- `direction = normalize(target - origin);`
- Else:
	- Keep existing pinhole `origin=params.cam_pos; direction=normalize(target - params.cam_pos);`

Apply this to both:
- debug first-hit path (`is_final_render == 0`)
- final path tracing path (`is_final_render == 1`)

Special modes:
- `populate_bins_mode`: keep deterministic lens center (`lens_offset=0`) even if DOF is enabled.
	- Rationale: photon-bin cache assumes one stable first-hit surface per pixel.
- `trace_primary_mode` (if used): same deterministic lens center.

### 6) Exposure choice (if implementing photometric aperture)
If `DEFAULT_DOF_PHOTOMETRIC_EXPOSURE == true`:
- Multiply the radiance contribution by a factor proportional to lens area $A=\pi r^2$.
- To avoid breaking existing baselines, normalize by a reference lens area (e.g., f/2.8 at 24mm) or expose a `DEFAULT_EXPOSURE` knob.

If deferring this decision:
- Implement geometry-only DOF first (blur), keep brightness consistent, and document that photometric exposure is a follow-up.

### 7) Validation & regression checks
#### Visual checks
- Place two objects: one near (e.g., z=1.5) and one far (z=0). Focus at one distance; verify:
	- in-focus object sharp
	- out-of-focus object blurred
- Vary f-number:
	- f/1.4 → strong blur
	- f/16  → near-pinhole

#### Deterministic / regression
- With DOF disabled (`lens_radius=0`), ensure:
	- primary ray direction math matches previous implementation
	- debug preview looks identical

#### Noise expectations
- DOF adds 2D lens sampling → more variance. Expect higher spp for same quality.

## File-by-file checklist
- `src/core/config.h`
	- add `DEFAULT_DOF_*` constants
- `src/core/random.h`
	- add `sample_concentric_disk()` (HD)
- `src/renderer/camera.h`
	- add DOF parameters + derived values
	- update `update()` and `generate_ray()`
- `src/optix/launch_params.h`
	- add camera lens fields
- `src/optix/optix_renderer.cpp`
	- pack DOF fields into launch params
- `src/optix/optix_device.cu`
	- implement thin-lens ray generation in `__raygen__render`
- `src/main.cpp`
	- initialize DOF defaults on camera creation

## Rollout plan
1. Land geometry-only DOF (thin lens) with `DEFAULT_DOF_ENABLED=false` to keep current behavior.
2. Verify DOF in interactive debug preview and final progressive render.
3. Decide on photometric exposure scaling; either implement or add a config toggle.

