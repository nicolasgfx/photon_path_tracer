# Third-Party Dependencies

This directory contains local copies of external libraries and SDKs used by the Radiosity Renderer.

## OptiX SDK 9.1.0

**Source:** NVIDIA OptiX SDK 9.1.0  
**Original Location:** `C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0`  
**Date Copied:** February 1, 2026

### Structure

```
third_party/optix/
├── include/          # OptiX headers (16 files)
│   ├── optix.h
│   ├── optix_device.h
│   ├── optix_types.h
│   └── internal/     # Internal implementation headers
├── lib/              # OptiX libraries (GLFW)
│   ├── glfw3dll.lib
│   └── cmake/        # CMake configuration files
└── bin/              # Runtime DLLs
    └── glfw3.dll
```

### Contents

- **Include files (16)**: All OptiX API headers for compilation
- **Library files (6)**: GLFW library files for window management
- **Binary files (1)**: GLFW runtime DLL

### Why Local Copy?

1. **Self-contained project** - No external SDK path dependencies
2. **Version control** - Headers versioned with the code
3. **Portability** - Works on any machine without OptiX SDK installation
4. **Build reliability** - No "SDK not found" errors
5. **Reproducibility** - Exact SDK version locked to project

### License

These files are part of the NVIDIA OptiX SDK and are subject to NVIDIA's license terms. They are included here for compilation purposes only.

**Important Notes:**
- OptiX runtime is still provided by NVIDIA GPU drivers (not included here)
- Only compile-time dependencies are copied
- Runtime requires compatible NVIDIA GPU and drivers with OptiX support

### Updating OptiX Version

To update to a newer OptiX SDK version:

```powershell
# Update the source path
$optixSDK = "C:\ProgramData\NVIDIA Corporation\OptiX SDK X.X.X"

# Copy files
Copy-Item "$optixSDK\include\*" -Destination "third_party/optix/include\" -Recurse -Force
Copy-Item "$optixSDK\lib\*" -Destination "third_party/optix/lib\" -Recurse -Force
Copy-Item "$optixSDK\bin\*" -Destination "third_party/optix/bin\" -Recurse -Force

# Update CMakeLists.txt if necessary (CUDA version requirements may change)
```

### CMake Integration

The project automatically uses this local OptiX SDK:

```cmake
set(OPTIX_ROOT ${CMAKE_SOURCE_DIR}/third_party/optix)
```

No environment variables or external paths required!

### Size

- **Total size**: ~1.5 MB (compressed text files)
- **Include**: ~500 KB
- **Lib**: ~500 KB
- **Bin**: ~400 KB

This is minimal overhead for a self-contained project.

---

## Future Dependencies

When adding other third-party libraries, follow the same pattern:

```
third_party/
├── optix/           # OptiX SDK (current)
├── library_name/    # Future library
│   ├── include/
│   ├── lib/
│   └── bin/
└── README.md        # This file
```
