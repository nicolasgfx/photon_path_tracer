---
name: code-quality
description: 'C++ code quality standards: style, naming, comments, structure, brevity. Use when: writing new C/C++/CUDA code, reviewing style, refactoring for readability, cleaning up comments, or enforcing project conventions. Applies to all .cpp, .h, .cu, .cuh files.'
---

# Code Quality — C++ Style Guide

Hybrid pragmatic style: Carmack's flat, tight structure with Abrash's "explain the non-obvious" commenting discipline. Optimized for brevity, clarity, and readability.

**Core principle:** Code that needs a comment to explain *what* it does is code that needs rewriting. Comments exist to explain *why*.

---

## Naming Conventions

| Element | Convention | Example |
|---|---|---|
| Local variables | `snake_case` | `float inv_pdf = 1.f / pdf;` |
| Function names | `snake_case` | `void launch_photon_trace(...)` |
| Member variables | trailing `_` | `int width_;` `bool ready_;` |
| File-scope statics | `s_` prefix | `static AppState s_app_state;` |
| Structs / Classes | `PascalCase` | `struct SceneProfile { ... };` |
| Enum classes | `PascalCase` values | `enum class LightingType { Envmap, SmallPoint };` |
| Constants | `UPPER_CASE` | `constexpr float PHOTON_EPSILON = 1e-4f;` |
| Template params | `PascalCase` | `template <typename T>` |
| Macros | `UPPER_CASE` | `#define HD __host__ __device__` |
| CUDA kernels | double-underscore OptiX pattern | `__raygen__photon_trace` |

**Naming rules:**
- Names are descriptive but compact — prefer `inv_pdf` over `inverse_of_probability_density`
- Boolean members/variables: use positive names (`enabled`, `ready`, `has_tris`) — never double negatives
- Short-lived loop variables can be single-letter: `i`, `n`, `w`, `h`
- Abbreviations are fine when domain-standard: `pdf`, `bsdf`, `nee`, `mis`, `rng`, `idx`

---

## File Structure

Every `.cpp` / `.cu` file follows this order:

```cpp
// viewer.cpp — GLFW window, progressive render loop, input handling

#ifdef _WIN32
#define NOMINMAX
#endif

#include "app/viewer.h"       // own header first
#include "core/config.h"      // project headers
#include "camera/camera.h"

#include <GLFW/glfw3.h>       // third-party headers

#include <cstdio>             // standard library
#include <algorithm>

// file-scope constants and statics

constexpr float EPSILON = 1e-5f;
static AppState s_app_state;

// -----------------------------------------------
// major section
// -----------------------------------------------

...
```

**Include order** (separated by blank lines):
1. Own header (the `.h` matching this `.cpp`)
2. Project headers
3. Third-party headers
4. Standard library headers

**Header files:**
- Use `#pragma once` (preferred) or traditional `#ifndef` guards for CUDA headers that need it
- No implementation in headers except: inline functions, templates, constexpr, trivial one-liners

---

## Separators & Sections

Three tiers of visual separation:

### Major section — top-level divisions in a file

```cpp
// -----------------------------------------------
// input callbacks
// -----------------------------------------------
```

Use for: callbacks, initialization, render loop, utilities, etc. Typically 3-6 per file.

### Minor section — sub-groups within a major section

```cpp
// --- frame timing
```

Use for: logical grouping within a major section. Single line, no decoration.

### No separator — blank line only

Between related functions or logical blocks within a section. No comment needed.

**Banned:** Unicode box-drawing characters (`─`), decorative ASCII art, `//====` banners, double-line boxing around function names.

---

## Comment Discipline

### When to comment

| Situation | Comment? | Example |
|---|---|---|
| Magic number / threshold | **Yes** | `constexpr float FIREFLY_SIGMA = 4.0f; // MAD multiplier — 4σ catches outliers without over-clamping` |
| Math derivation | **Yes** | `// flux = Le * disk_area / (pdf_dir * p_select)` |
| Performance tradeoff | **Yes** | `// hash grid O(1) vs kd-tree O(log N) — worth it above 1M photons` |
| Non-obvious invariant | **Yes** | `// must be called after BVH build — depends on traversable handle` |
| "Why not the obvious approach" | **Yes** | `// power heuristic β=2 beats balance for glossy+diffuse (Veach §9.2.4)` |
| Workaround / trap | **Yes** | `// OptiX 9.0 bug: denoiser crashes if intensity buffer is uninitialized` |
| Getter, setter, trivial accessor | **No** | — |
| Early return / guard clause | **No** | — |
| Self-documenting code | **No** | `float cos_theta = dot(n, wi);` needs no comment |
| Cleanup / resource release | **No** | — |
| Struct field with clear name | **No** | `int width;` needs no comment |

### Comment style

```cpp
// single-line comment — lower case, no period unless multi-sentence
float inv_pdf = 1.f / fmaxf(pdf, 1e-10f);  // clamp avoids inf on degenerate tris
```

- Comments are lowercase unless starting with a proper noun or acronym
- Inline comments: 1-2 spaces before `//`, short phrase
- Block comments above a function: 2-4 lines max, explain *approach* not *API*
- Never `/* */` style in regular code — reserve for disabling code blocks during debug
- Academic citations: `(Author YEAR)` or `(Author §section)` inline

### Comment density targets

| Code type | Target density | Rationale |
|---|---|---|
| Algorithm kernels (path tracing, photon, MIS) | ~20-25% | Math and physics need rationale |
| Plumbing (init, cleanup, callbacks, I/O) | ~5% | Self-documenting if named well |
| Data structures (structs, enums, configs) | ~10% | Comment non-obvious fields only |

---

## Braces & Formatting

**K&R brace style** — opening brace on same line:

```cpp
if (action != GLFW_PRESS) return;

for (int i = 0; i < n; ++i) {
    process(items[i]);
}

switch (key) {
case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(w, GLFW_TRUE); break;
case GLFW_KEY_R:      st.snapshot_requested = true;           break;
case GLFW_KEY_G:
    st.guide_enabled ^= 1;
    st.camera_moved = true;
    break;
}
```

**Rules:**
- Omit braces for single-statement `if` / `for` / `while` that fit on one line or the next line
- `switch` cases at same indent level as `switch` (no extra indentation)
- Simple switch cases can be one-liners with aligned `break`
- 4-space indentation, no tabs
- Max line width: ~100 characters (soft limit — don't break readability to hit it)
- Align related assignments vertically when it aids scanning:

```cpp
guide_layer.albedo.data   = (CUdeviceptr)d_albedo;
guide_layer.albedo.width  = (unsigned int)width;
guide_layer.albedo.height = (unsigned int)height;
```

---

## Function Design

**Size targets:**
- Inline utilities: 1-5 lines
- Normal functions: 10-50 lines
- Complex algorithms: up to ~80 lines if they tell a coherent story
- If a function exceeds 80 lines, look for natural extraction points

**Structure:**
- Guard clauses and early returns at the top — never nest when you can return
- One level of nesting is normal, two is a warning, three means extract
- Prefer flat control flow over deep nesting

```cpp
// good: flat with early returns
bool init(const Config& cfg) {
    if (!cfg.valid()) return false;
    if (!create_window(cfg.width, cfg.height)) return false;

    width_  = cfg.width;
    height_ = cfg.height;
    ready_  = true;
    return true;
}

// bad: nested pyramid
bool init(const Config& cfg) {
    if (cfg.valid()) {
        if (create_window(cfg.width, cfg.height)) {
            width_  = cfg.width;
            height_ = cfg.height;
            ready_  = true;
            return true;
        }
    }
    return false;
}
```

---

## Classes & Data

- **Structs** for data aggregation and POD types — public by default
- **Classes** only when encapsulation invariants must be maintained (e.g., `RenderSession` with `init()` / `is_ready()` lifecycle)
- Minimal inheritance — prefer composition
- No getters/setters for simple data — use public members
- Two-phase initialization pattern when needed: default construct + `init()` method

```cpp
struct MeshDescriptor {
    uint32_t tri_offset;
    uint32_t tri_count;
};

class RenderSession {
    bool ready_ = false;
    int  width_  = 0;
    int  height_ = 0;
public:
    bool init(const Scene& scene, const RenderConfig& config);
    bool is_ready() const { return ready_; }
};
```

---

## C++ Feature Usage

**Prefer:**
- `constexpr` over `#define` for constants
- `enum class` over plain `enum`
- `float` literals with `f` suffix: `1.0f`, `0.5f`, `1e-4f`
- Range-for when iterating containers: `for (auto& item : items)`
- `auto` when the type is obvious from context: `auto& st = app_state();`
- `std::printf` / `std::fprintf` for console output with component prefix: `[Viewer]`, `[Scene]`
- Designated initializers (C++20) when initializing structs with many fields

**Avoid:**
- `std::endl` — use `\n` (no flush overhead)
- `using namespace std;` — never, anywhere
- Smart pointers for GPU resources — use explicit RAII wrappers with `alloc()` / `free()`
- Exceptions for control flow — use return codes or early returns
- `std::string` in hot paths — use `const char*` or fixed buffers
- Over-templating — templates for genuine type genericity, not code golf

---

## CUDA / GPU Specifics

```cpp
#ifdef __CUDACC__
#define HD __host__ __device__
#define DEV __device__
#else
#define HD
#define DEV
#endif

inline HD float3 reflect(float3 d, float3 n) {
    return d - 2.f * dot(d, n) * n;
}
```

- `HD` macro for shared host/device functions
- Kernel launches get a brief comment if the grid/block config is non-obvious
- GPU buffer wrappers: `alloc()`, `zero()`, `download()`, `free()` — no raw `cudaMalloc`/`cudaFree` in algorithm code
- Check CUDA errors at API boundaries, not after every call in hot paths

---

## Patterns to Follow

### Console output

```cpp
printf("[Viewer] Guide: %s\n", enabled ? "ON" : "OFF");
fprintf(stderr, "[RenderSession] Warning: denoiser init failed\n");
```

Component prefix in brackets. Warnings/errors to stderr.

### Compact cleanup

```cpp
void cleanup() {
    if (denoiser_) optixDenoiserDestroy(denoiser_);
    cudaFree(d_state_);
    cudaFree(d_scratch_);
    *this = {};   // zero everything
}
```

No per-field null checks when the free function is null-safe. Reset with `*this = {}` or member-by-member if partial.

### Boolean toggles

```cpp
st.guide_enabled ^= 1;     // XOR toggle — compact, idiomatic in input handlers
```

### Operator overloads for math types

```cpp
inline HD float3 operator+(float3 a, float3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline HD float  dot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline HD float  length(float3 v) { return sqrtf(dot(v, v)); }
```

One-line for trivial math. No comments needed.

---

## Anti-Patterns

| Don't | Do instead |
|---|---|
| `// increment counter` before `counter++` | Delete the comment |
| `//====` or `// ──` decorative banners | `// -----------------------------------------------` for major, `// ---` for minor |
| 6-line docstring on a 3-line function | No comment, or 1-line above |
| `if (ptr != nullptr)` | `if (ptr)` |
| `return (value);` | `return value;` |
| `== true` / `== false` | Direct boolean or `!` |
| Blank line after opening brace | Remove it |
| Blank line before closing brace | Remove it |
| Multiple blank lines | Single blank line max |
| Trailing whitespace | Strip it |
| Comments that restate parameter names | Delete them |

---

## Refactoring Checklist

When cleaning up existing code, apply in this order:

1. **Strip noise** — remove redundant comments, decorative separators, blank lines
2. **Flatten** — convert nested if/else to early returns
3. **Name** — rename unclear variables; if the new name makes a comment redundant, delete the comment
4. **Extract** — pull out functions only at natural algorithm boundaries (not for arbitrary line count targets)
5. **Align** — vertically align related assignments if 3+ lines
6. **Verify** — build and test after changes
