cmake_minimum_required(VERSION 3.24)
cmake_policy(SET CMP0141 NEW)
project(photon_path_tracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -- Find packages -----------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# OptiX (MANDATORY)
# 1. Command-line -DOptiX_INSTALL_DIR=...  takes highest priority
# 2. Environment variable OptiX_INSTALL_DIR
# 3. Default Windows install path (NVIDIA Corporation\OptiX SDK 9.x.x)
if(DEFINED ENV{OptiX_INSTALL_DIR} AND NOT OptiX_INSTALL_DIR)
    set(OptiX_INSTALL_DIR $ENV{OptiX_INSTALL_DIR})
endif()
if(NOT OptiX_INSTALL_DIR)
    # Auto-detect: search ProgramData for the latest OptiX SDK
    file(GLOB _optix_candidates "C:/ProgramData/NVIDIA Corporation/OptiX SDK *")
    if(_optix_candidates)
        list(SORT _optix_candidates)
        list(GET _optix_candidates -1 OptiX_INSTALL_DIR)   # pick latest version
    endif()
endif()
if(NOT OptiX_INSTALL_DIR)
    message(FATAL_ERROR
        "OptiX_INSTALL_DIR not set. Set the environment variable or pass "
        "-DOptiX_INSTALL_DIR=<path> to CMake. OptiX is mandatory.")
endif()
set(OptiX_INCLUDE_DIR "${OptiX_INSTALL_DIR}/include")
if(NOT EXISTS "${OptiX_INCLUDE_DIR}/optix.h")
    message(FATAL_ERROR "optix.h not found in ${OptiX_INCLUDE_DIR}")
endif()
message(STATUS "OptiX include dir: ${OptiX_INCLUDE_DIR}")

# GLFW for interactive window
find_package(glfw3 3.3 QUIET)
if(NOT glfw3_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        3.3.9
    )
    set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(glfw)
endif()

# stb_image for texture loading
include(FetchContent)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        f1c79c02822848a9bed4315b12c8c8f3761e1296  # pinned
)
FetchContent_MakeAvailable(stb)

# -- CUDA architecture -------------------------------------------------
# Dev builds: compile only for the local GPU (fast).  Set PPT_CUDA_ARCH_ALL=ON
# for release builds targeting multiple GPU generations.
option(PPT_CUDA_ARCH_ALL "Build for all supported CUDA architectures (75;86;89)" OFF)
if(PPT_CUDA_ARCH_ALL)
    set(PPT_CUDA_ARCHS "75;86;89")
else()
    # native = detect GPU at configure time (CMake 3.24+)
    set(PPT_CUDA_ARCHS "native")
endif()

# -- Source files ------------------------------------------------------
set(CORE_SOURCES
    src/main.cpp
    src/scene/obj_loader.cpp
    src/renderer/renderer.cpp
    src/optix/optix_renderer.cpp
)

set(CUDA_SOURCES
    src/photon/hash_grid.cu
    src/photon/emitter.cu
    # src/renderer/path_tracer.cu  -- removed (v1 stub, superseded by OptiX)
    src/renderer/direct_light.cu
    src/optix/adaptive_sampling.cu
)

# -- Shared CUDA static library (shared by photon_tracer and ppt_tests) --------
add_library(ppt_cuda_objects STATIC ${CUDA_SOURCES})
set_target_properties(ppt_cuda_objects PROPERTIES
    CUDA_ARCHITECTURES "${PPT_CUDA_ARCHS}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(ppt_cuda_objects PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OptiX_INCLUDE_DIR}
)
target_compile_definitions(ppt_cuda_objects PRIVATE PPT_USE_OPTIX=1)
if(MSVC)
    target_compile_options(ppt_cuda_objects PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4505>
        $<$<COMPILE_LANGUAGE:CXX>:/MP>
    )
endif()

# -- Main executable ---------------------------------------------------
add_executable(photon_tracer
    ${CORE_SOURCES}
)

if(MSVC)
    target_compile_options(photon_tracer PRIVATE /MP)
endif()


target_include_directories(photon_tracer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${stb_SOURCE_DIR}
    ${OptiX_INCLUDE_DIR}
)

# Embed DPI-awareness manifest so Windows doesn't bitmap-scale the window.
# The manifest is processed by the OS loader before any DLL init runs,
# which is the only reliable way to prevent CUDA/NVIDIA drivers from
# setting their own (wrong) DPI awareness first.
if(MSVC)
    set(MANIFEST_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/photon_tracer.manifest")
    target_link_options(photon_tracer PRIVATE
        "/MANIFEST:EMBED"
        "/MANIFESTINPUT:${MANIFEST_FILE}"
    )
endif()

find_package(OpenGL REQUIRED)
find_package(OpenMP)

target_link_libraries(photon_tracer PRIVATE
    ppt_cuda_objects
    CUDA::cudart
    glfw
    OpenGL::GL
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(photon_tracer PRIVATE OpenMP::OpenMP_CXX)
endif()

# CUDA architecture (photon_tracer itself has no .cu files, symbols come via ppt_cuda_objects)
set_target_properties(photon_tracer PROPERTIES
    CUDA_ARCHITECTURES "${PPT_CUDA_ARCHS}"
)

# Suppress C4505 from CUDA/stb runtime headers (unreferenced internal functions)
if(MSVC)
    # For C/C++ source files
    set_source_files_properties(${CORE_SOURCES} PROPERTIES
        COMPILE_OPTIONS "/wd4505")
endif()

# OptiX is always enabled
target_compile_definitions(photon_tracer PRIVATE PPT_USE_OPTIX=1)

# -- Compile OptiX device programs to PTX -----------------------------
# Uses a custom command so the PTX output path is predictable and
# config-independent (no Debug/ subdirectory issues with MSVC generators).
set(OPTIX_PTX_DIR "${CMAKE_BINARY_DIR}/ptx")
set(OPTIX_PTX_OUTPUT "${OPTIX_PTX_DIR}/optix_device.ptx")

add_custom_command(
    OUTPUT "${OPTIX_PTX_OUTPUT}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${OPTIX_PTX_DIR}"
    COMMAND ${CMAKE_CUDA_COMPILER}
        -ptx
        --std=c++17
        -arch=compute_75
        --use_fast_math
        -O3
        -DNDEBUG
        -DPPT_USE_OPTIX=1
        "-I${CMAKE_CURRENT_SOURCE_DIR}/src"
        "-I${OptiX_INCLUDE_DIR}"
        -o "${OPTIX_PTX_OUTPUT}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/optix/optix_device.cu"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/optix/optix_device.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/optix/launch_params.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/core/config.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/core/medium.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/core/phase_function.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/core/random.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/core/spectrum.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/core/types.h"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Compiling OptiX device programs (optix_device.cu -> PTX)"
    VERBATIM
)
add_custom_target(optix_ptx ALL DEPENDS "${OPTIX_PTX_OUTPUT}")
add_dependencies(photon_tracer optix_ptx)

# Tell the host code where to find the PTX at runtime
target_compile_definitions(photon_tracer PRIVATE
    PTX_FILE_PATH="${OPTIX_PTX_OUTPUT}"
)

# -- Compile definitions -----------------------------------------------
target_compile_definitions(photon_tracer PRIVATE
    SCENES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/scenes"
)

# -- Compiler flags ----------------------------------------------------
if(MSVC)
    target_compile_options(photon_tracer PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4 /fp:fast>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xcompiler=/W4>
    )
else()
    target_compile_options(photon_tracer PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O2>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xcompiler=-Wall,-Wextra,-O2>
    )
endif()

# -- Unit tests --------------------------------------------------------
option(PPT_BUILD_TESTS "Build unit tests" OFF)
if(PPT_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    # Prevent gtest from overriding our compiler/linker settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    set(TEST_SOURCES
        tests/test_main.cpp
        tests/test_kd_tree.cpp
        tests/test_integration.cpp
        tests/test_surface_filter.cpp
        tests/test_tangential_gather.cpp
        tests/feature_speed_test.cpp
        tests/test_ground_truth.cpp
        tests/test_ground_truth_validation.cpp
        tests/test_medium.cpp
        tests/test_beam_estimator.cpp
        tests/test_per_ray_validation.cpp
        tests/test_pixel_comparison.cpp
        tests/test_speed_tweaks.cpp
        tests/test_caustics.cpp
        tests/test_tri_photon_irradiance.cpp
        src/scene/obj_loader.cpp
        src/renderer/renderer.cpp
        src/optix/optix_renderer.cpp
    )

    add_executable(ppt_tests
        ${TEST_SOURCES}
    )
    set_target_properties(ppt_tests PROPERTIES
        CUDA_ARCHITECTURES "${PPT_CUDA_ARCHS}"
        MSVC_DEBUG_INFORMATION_FORMAT "Embedded"
    )
    target_include_directories(ppt_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${stb_SOURCE_DIR}
        ${OptiX_INCLUDE_DIR}
    )
    target_link_libraries(ppt_tests PRIVATE
        ppt_cuda_objects
        GTest::gtest_main
        CUDA::cudart
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(ppt_tests PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_compile_features(ppt_tests PRIVATE cxx_std_17)
    target_compile_definitions(ppt_tests PRIVATE
        SCENES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/scenes"
        PPT_USE_OPTIX=1
        PTX_FILE_PATH="${OPTIX_PTX_OUTPUT}"
    )
    add_dependencies(ppt_tests optix_ptx)

    if(MSVC)
        target_compile_options(ppt_tests PRIVATE /MP /W4 /fp:fast)
    else()
        target_compile_options(ppt_tests PRIVATE -Wall -Wextra -O2)
    endif()

    include(GoogleTest)
    gtest_discover_tests(ppt_tests)
endif()