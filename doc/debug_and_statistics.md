# hotkeys, gates and statistics

- gate statistics which will significantly reduce redering time (via ifndef). I don't want to impact the renderer with statistics at all times.

## Hotkeys

- a hotkey to switch between unguided (regular, brute force) path tracing and photon mode guided path tracing mode
- a hotkey to disable all conclusion measures EXCEPT histogram (to visually assess impact of deeper analysis) WHEN in guided path tracing mode

## statistics output

group statistics which belong together

### photon mapping

- how many photons of each type

### path tracing

- samples per pixel: min, max, avg
- photon analysis: which conclusion was made how often?
- photon analysis: which measure was made how often?

### geometry

- relevant geometry-related numbers
- num light sources

### hardware

- which GPU

## rendering

- rendering time (after last mouse movement stopped to pressing "r")
- denoising before presssing "r" to save screenshot?