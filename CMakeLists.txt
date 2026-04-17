cmake_minimum_required(VERSION 3.24)
cmake_policy(SET CMP0141 NEW)
project(photon_path_tracer LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# -- Find packages (early, so all targets can use them) ----------------
find_package(CUDAToolkit REQUIRED)

# OptiX (MANDATORY) — detect once, use everywhere
if(DEFINED ENV{OptiX_INSTALL_DIR} AND NOT OptiX_INSTALL_DIR)
    set(OptiX_INSTALL_DIR $ENV{OptiX_INSTALL_DIR})
endif()
if(NOT OptiX_INSTALL_DIR)
    file(GLOB _optix_candidates "C:/ProgramData/NVIDIA Corporation/OptiX SDK *")
    if(_optix_candidates)
        list(SORT _optix_candidates)
        list(GET _optix_candidates -1 OptiX_INSTALL_DIR)
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

# =======================================================================
#  V5 Core Foundation (RGB pipeline)
#  Minimal compile-test for the new src/core/ headers.
#  The rest of the old build targets are below, temporarily disabled
#  until migration is complete.
# =======================================================================
option(PPT_V5_CORE_TEST "Build the v5 core header compile test" ON)
if(PPT_V5_CORE_TEST)
    find_package(CUDAToolkit REQUIRED)
    add_executable(core_test tests/integration/core_test.cpp)
    target_include_directories(core_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer
        ${CUDAToolkit_INCLUDE_DIRS}
        ${OptiX_INCLUDE_DIR}
    )
    target_compile_features(core_test PRIVATE cxx_std_17)
    if(MSVC)
        target_compile_options(core_test PRIVATE /W4 /fp:fast)
    else()
        target_compile_options(core_test PRIVATE -Wall -Wextra -O2)
    endif()
endif()
# =======================================================================

# =======================================================================
#  V5 Scene Stage (Phase 2)
#  Scene loading, material definition, camera, PBRT/OBJ loaders,
#  scene analyzer, and scene builder.
# =======================================================================
option(PPT_V5_SCENE_TEST "Build the v5 scene stage compile test" ON)
if(PPT_V5_SCENE_TEST)
    find_package(CUDAToolkit REQUIRED)

    # stb_image (header-only) — for texture loading in scene loaders
    include(FetchContent)
    FetchContent_Declare(stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG        master
    )
    FetchContent_MakeAvailable(stb)

    # tinyexr (header-only) — for EXR envmap loading
    FetchContent_Declare(tinyexr
        GIT_REPOSITORY https://github.com/syoyo/tinyexr.git
        GIT_TAG        v1.0.9
    )
    FetchContent_MakeAvailable(tinyexr)

    add_executable(scene_test
        tests/integration/scene_test.cpp
        src/scene/pbrt/pbrt_parser.cpp
        src/scene/pbrt/pbrt_material_mapper.cpp
        src/scene/pbrt/pbrt_loader.cpp
        src/scene/pbrt/ply_reader.cpp
        src/scene/obj_loader.cpp
        ${tinyexr_SOURCE_DIR}/deps/miniz/miniz.c
    )
    target_include_directories(scene_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer
        ${CUDAToolkit_INCLUDE_DIRS}
        ${OptiX_INCLUDE_DIR}
        ${stb_SOURCE_DIR}
        ${tinyexr_SOURCE_DIR}
        ${tinyexr_SOURCE_DIR}/deps/miniz
    )
    target_compile_features(scene_test PRIVATE cxx_std_17)
    if(MSVC)
        target_compile_options(scene_test PRIVATE /W4 /fp:fast)
    else()
        target_compile_options(scene_test PRIVATE -Wall -Wextra -O2)
    endif()
endif()
# =======================================================================

# =======================================================================
#  V5 Acceleration Stage (Phase 3)
#  OptiX pipeline, GAS/IAS build, hit/miss programs, ray tracing test.
#  Requires: CUDA GPU + OptiX SDK.
# =======================================================================
option(PPT_V5_ACCEL_TEST "Build the v5 acceleration stage test" ON)
if(PPT_V5_ACCEL_TEST)
    find_package(CUDAToolkit REQUIRED)

    # OptiX SDK (auto-detect or user-specified)
    if(DEFINED ENV{OptiX_INSTALL_DIR} AND NOT OptiX_INSTALL_DIR)
        set(OptiX_INSTALL_DIR $ENV{OptiX_INSTALL_DIR})
    endif()
    if(NOT OptiX_INSTALL_DIR)
        file(GLOB _optix_candidates_v5 "C:/ProgramData/NVIDIA Corporation/OptiX SDK *")
        if(_optix_candidates_v5)
            list(SORT _optix_candidates_v5)
            list(GET _optix_candidates_v5 -1 OptiX_INSTALL_DIR)
        endif()
    endif()
    if(NOT OptiX_INSTALL_DIR)
        message(WARNING "OptiX_INSTALL_DIR not set - skipping accel_test")
    else()
        set(OptiX_INCLUDE_V5 "${OptiX_INSTALL_DIR}/include")
        if(NOT EXISTS "${OptiX_INCLUDE_V5}/optix.h")
            message(WARNING "optix.h not found in ${OptiX_INCLUDE_V5} - skipping accel_test")
        else()
            message(STATUS "[v5] OptiX include: ${OptiX_INCLUDE_V5}")

            # -- Compile device programs to PTX ----------------------------
            set(V5_PTX_DIR "${CMAKE_BINARY_DIR}/ptx_v5")
            set(V5_PTX_OUTPUT "${V5_PTX_DIR}/optix_programs.ptx")

            add_custom_command(
                OUTPUT "${V5_PTX_OUTPUT}"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${V5_PTX_DIR}"
                COMMAND ${CMAKE_CUDA_COMPILER}
                    -ptx
                    --std=c++17
                    -arch=compute_75
                    --use_fast_math
                    -O3
                    -DNDEBUG
                    -DPPT_USE_OPTIX=1
                    "-I${CMAKE_CURRENT_SOURCE_DIR}/src"
                    "-I${CMAKE_CURRENT_SOURCE_DIR}/src/renderer"
                    "-I${OptiX_INCLUDE_V5}"
                    -o "${V5_PTX_OUTPUT}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/accel/optix_programs.cu"
                DEPENDS
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/accel/optix_programs.cu"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/accel/optix_utils.cuh"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/accel/launch_params.h"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/types.h"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/config.h"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/random.h"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/hash.h"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/optix_nee.cuh"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/nee_shared.h"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/material/bsdf_shared.h"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/path_tracer.cuh"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/path_state.h"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/russian_roulette.h"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/sample_clamping.h"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/caustic_tracer.cuh"
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                COMMENT "Compiling v5 OptiX programs (optix_programs.cu -> PTX)"
                VERBATIM
            )
            add_custom_target(v5_optix_ptx DEPENDS "${V5_PTX_OUTPUT}")

            # -- stb_image for scene loaders (needed by scene_builder) -----
            include(FetchContent)
            if(NOT stb_POPULATED)
                FetchContent_Declare(stb
                    GIT_REPOSITORY https://github.com/nothings/stb.git
                    GIT_TAG        master
                )
                FetchContent_MakeAvailable(stb)
            endif()

            # -- tinyexr for EXR envmap loading ----------------------------
            if(NOT tinyexr_POPULATED)
                FetchContent_Declare(tinyexr
                    GIT_REPOSITORY https://github.com/syoyo/tinyexr.git
                    GIT_TAG        v1.0.9
                )
                FetchContent_MakeAvailable(tinyexr)
            endif()

            # -- Host-side test executable ---------------------------------
            add_executable(accel_test
                tests/integration/accel_test.cpp
                src/renderer/accel/accel_builder.cpp
            )
            target_include_directories(accel_test PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/src
                ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer
                ${CUDAToolkit_INCLUDE_DIRS}
                ${OptiX_INCLUDE_V5}
                ${stb_SOURCE_DIR}
                ${tinyexr_SOURCE_DIR}
                ${tinyexr_SOURCE_DIR}/deps/miniz
            )
            target_link_libraries(accel_test PRIVATE
                CUDA::cudart
            )
            target_compile_features(accel_test PRIVATE cxx_std_17)
            target_compile_definitions(accel_test PRIVATE
                PPT_USE_OPTIX=1
                "V5_PTX_FILE_PATH=\"${V5_PTX_OUTPUT}\""
            )
            add_dependencies(accel_test v5_optix_ptx)

            if(MSVC)
                target_compile_options(accel_test PRIVATE /W4 /fp:fast)
            else()
                target_compile_options(accel_test PRIVATE -Wall -Wextra -O2)
            endif()
        endif()
    endif()
endif()
# =======================================================================

# =======================================================================
#  V5 Material / BSDF Stage (Phase 4)
#  CPU-only: BSDF evaluate/sample/PDF for all 9 material types.
#  White furnace, reciprocity, PDF consistency, specular bounce tests.
# =======================================================================
option(PPT_V5_MATERIAL_TEST "Build the v5 material / BSDF stage test" ON)
if(PPT_V5_MATERIAL_TEST)
    find_package(CUDAToolkit REQUIRED)
    include(FetchContent)
    if(NOT googletest_POPULATED)
        FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    add_executable(material_test
        tests/integration/material_test.cpp
    )
    target_include_directories(material_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer
        ${CUDAToolkit_INCLUDE_DIRS}
        ${OptiX_INCLUDE_DIR}
    )
    target_link_libraries(material_test PRIVATE GTest::gtest_main)
    target_compile_features(material_test PRIVATE cxx_std_17)
    if(MSVC)
        target_compile_options(material_test PRIVATE /W4 /fp:fast)
    else()
        target_compile_options(material_test PRIVATE -Wall -Wextra -O2)
    endif()
endif()
# =======================================================================

# =======================================================================
#  V5 Direct Lighting / NEE Stage (Phase 5)
#  GPU integration: emissive CDF upload, OptiX NEE shadow rays,
#  one-sample MIS.  Launches __raygen__test_nee on Cornell
#  box scenes and verifies illumination, no NaN/inf.
# =======================================================================
option(PPT_V5_LIGHTING_TEST "Build the v5 direct lighting / NEE stage test" ON)
if(PPT_V5_LIGHTING_TEST)
    find_package(CUDAToolkit REQUIRED)

    # OptiX SDK (auto-detect or user-specified)
    if(DEFINED ENV{OptiX_INSTALL_DIR} AND NOT OptiX_INSTALL_DIR)
        set(OptiX_INSTALL_DIR $ENV{OptiX_INSTALL_DIR})
    endif()
    if(NOT OptiX_INSTALL_DIR)
        file(GLOB _optix_candidates_v5_lt "C:/ProgramData/NVIDIA Corporation/OptiX SDK *")
        if(_optix_candidates_v5_lt)
            list(SORT _optix_candidates_v5_lt)
            list(GET _optix_candidates_v5_lt -1 OptiX_INSTALL_DIR)
        endif()
    endif()
    if(NOT OptiX_INSTALL_DIR)
        message(WARNING "OptiX_INSTALL_DIR not set - skipping lighting_test")
    else()
        set(OptiX_INCLUDE_V5 "${OptiX_INSTALL_DIR}/include")
        if(NOT EXISTS "${OptiX_INCLUDE_V5}/optix.h")
            message(WARNING "optix.h not found in ${OptiX_INCLUDE_V5} - skipping lighting_test")
        else()
            # -- PTX compilation (reuse v5_optix_ptx if already defined) ---
            set(V5_PTX_DIR "${CMAKE_BINARY_DIR}/ptx_v5")
            set(V5_PTX_OUTPUT "${V5_PTX_DIR}/optix_programs.ptx")

            if(NOT TARGET v5_optix_ptx)
                add_custom_command(
                    OUTPUT "${V5_PTX_OUTPUT}"
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${V5_PTX_DIR}"
                    COMMAND ${CMAKE_CUDA_COMPILER}
                        -ptx
                        --std=c++17
                        -arch=compute_75
                        --use_fast_math
                        -O3
                        -DNDEBUG
                        -DPPT_USE_OPTIX=1
                        "-I${CMAKE_CURRENT_SOURCE_DIR}/src"
                        "-I${CMAKE_CURRENT_SOURCE_DIR}/src/renderer"
                        "-I${OptiX_INCLUDE_V5}"
                        -o "${V5_PTX_OUTPUT}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/accel/optix_programs.cu"
                    DEPENDS
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/accel/optix_programs.cu"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/accel/optix_utils.cuh"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/accel/launch_params.h"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/core/types.h"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/core/config.h"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/core/random.h"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/core/hash.h"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/optix_nee.cuh"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/nee_shared.h"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/material/bsdf_shared.h"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/path_tracer.cuh"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/path_state.h"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/russian_roulette.h"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/sample_clamping.h"
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    COMMENT "Compiling v5 OptiX programs (with NEE + integrator + photon) -> PTX"
                    VERBATIM
                )
                add_custom_target(v5_optix_ptx DEPENDS "${V5_PTX_OUTPUT}")
            endif()

            # -- stb_image for scene loaders --------------------------------
            include(FetchContent)
            if(NOT stb_POPULATED)
                FetchContent_Declare(stb
                    GIT_REPOSITORY https://github.com/nothings/stb.git
                    GIT_TAG        master
                )
                FetchContent_MakeAvailable(stb)
            endif()

            # -- tinyexr for EXR envmap loading ----------------------------
            if(NOT tinyexr_POPULATED)
                FetchContent_Declare(tinyexr
                    GIT_REPOSITORY https://github.com/syoyo/tinyexr.git
                    GIT_TAG        v1.0.9
                )
                FetchContent_MakeAvailable(tinyexr)
            endif()

            # -- Host-side test executable ---------------------------------
            add_executable(lighting_test
                tests/integration/lighting_test.cpp
                src/renderer/accel/accel_builder.cpp
            )
            target_include_directories(lighting_test PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/src
                ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer
                ${CUDAToolkit_INCLUDE_DIRS}
                ${OptiX_INCLUDE_V5}
                ${stb_SOURCE_DIR}
                ${tinyexr_SOURCE_DIR}
                ${tinyexr_SOURCE_DIR}/deps/miniz
            )
            target_link_libraries(lighting_test PRIVATE
                CUDA::cudart
            )
            target_compile_features(lighting_test PRIVATE cxx_std_17)
            target_compile_definitions(lighting_test PRIVATE
                PPT_USE_OPTIX=1
                "V5_PTX_FILE_PATH=\"${V5_PTX_OUTPUT}\""
            )
            add_dependencies(lighting_test v5_optix_ptx)

            if(MSVC)
                target_compile_options(lighting_test PRIVATE /W4 /fp:fast)
            else()
                target_compile_options(lighting_test PRIVATE -Wall -Wextra -O2)
            endif()
        endif()
    endif()
endif()
# =======================================================================

# =======================================================================
option(PPT_V5_INTEGRATOR_TEST "Build the v5 path integrator stage test" ON)
if(PPT_V5_INTEGRATOR_TEST)
    find_package(CUDAToolkit REQUIRED)

    if(DEFINED ENV{OptiX_INSTALL_DIR} AND NOT OptiX_INSTALL_DIR)
        set(OptiX_INSTALL_DIR $ENV{OptiX_INSTALL_DIR})
    endif()
    if(NOT OptiX_INSTALL_DIR)
        file(GLOB _optix_candidates_v5_it "C:/ProgramData/NVIDIA Corporation/OptiX SDK *")
        if(_optix_candidates_v5_it)
            list(SORT _optix_candidates_v5_it)
            list(GET _optix_candidates_v5_it -1 OptiX_INSTALL_DIR)
        endif()
    endif()
    if(NOT OptiX_INSTALL_DIR)
        message(WARNING "OptiX_INSTALL_DIR not set - skipping integrator_test")
    else()
        set(OptiX_INCLUDE_V5 "${OptiX_INSTALL_DIR}/include")
        if(NOT EXISTS "${OptiX_INCLUDE_V5}/optix.h")
            message(WARNING "optix.h not found in ${OptiX_INCLUDE_V5} - skipping integrator_test")
        else()
            set(V5_PTX_DIR "${CMAKE_BINARY_DIR}/ptx_v5")
            set(V5_PTX_OUTPUT "${V5_PTX_DIR}/optix_programs.ptx")

            if(NOT TARGET v5_optix_ptx)
                add_custom_command(
                    OUTPUT "${V5_PTX_OUTPUT}"
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${V5_PTX_DIR}"
                    COMMAND ${CMAKE_CUDA_COMPILER}
                        -ptx
                        --std=c++17
                        -arch=compute_75
                        --use_fast_math
                        -O3
                        -DNDEBUG
                        -DPPT_USE_OPTIX=1
                        "-I${CMAKE_CURRENT_SOURCE_DIR}/src"
                        "-I${CMAKE_CURRENT_SOURCE_DIR}/src/renderer"
                        "-I${OptiX_INCLUDE_V5}"
                        -o "${V5_PTX_OUTPUT}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/accel/optix_programs.cu"
                    DEPENDS
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/accel/optix_programs.cu"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/accel/optix_utils.cuh"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/accel/launch_params.h"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/core/types.h"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/core/config.h"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/core/random.h"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/core/hash.h"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/optix_nee.cuh"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/nee_shared.h"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/material/bsdf_shared.h"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/path_tracer.cuh"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/path_state.h"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/russian_roulette.h"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/sample_clamping.h"
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    COMMENT "Compiling v5 OptiX programs -> PTX"
                    VERBATIM
                )
                add_custom_target(v5_optix_ptx DEPENDS "${V5_PTX_OUTPUT}")
            endif()

            include(FetchContent)
            if(NOT stb_POPULATED)
                FetchContent_Declare(stb
                    GIT_REPOSITORY https://github.com/nothings/stb.git
                    GIT_TAG        master
                )
                FetchContent_MakeAvailable(stb)
            endif()

            if(NOT tinyexr_POPULATED)
                FetchContent_Declare(tinyexr
                    GIT_REPOSITORY https://github.com/syoyo/tinyexr.git
                    GIT_TAG        v1.0.9
                )
                FetchContent_MakeAvailable(tinyexr)
            endif()

            add_executable(integrator_test
                tests/integration/integrator_test.cpp
                src/renderer/accel/accel_builder.cpp
            )
            target_include_directories(integrator_test PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/src
                ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer
                ${CUDAToolkit_INCLUDE_DIRS}
                ${OptiX_INCLUDE_V5}
                ${stb_SOURCE_DIR}
                ${tinyexr_SOURCE_DIR}
                ${tinyexr_SOURCE_DIR}/deps/miniz
            )
            target_link_libraries(integrator_test PRIVATE
                CUDA::cudart
            )
            target_compile_features(integrator_test PRIVATE cxx_std_17)
            target_compile_definitions(integrator_test PRIVATE
                PPT_USE_OPTIX=1
                "V5_PTX_FILE_PATH=\"${V5_PTX_OUTPUT}\""
            )
            add_dependencies(integrator_test v5_optix_ptx)

            if(MSVC)
                target_compile_options(integrator_test PRIVATE /W4 /fp:fast)
            else()
                target_compile_options(integrator_test PRIVATE -Wall -Wextra -O2)
            endif()
        endif()
    endif()
endif()
# =======================================================================

# =======================================================================
#  V5 Phase 9: Post-Processing Pipeline Test
#  Tests: tonemap, firefly filter, bloom, full pipeline end-to-end.
#  Requires CUDA GPU.
# =======================================================================
option(PPT_V5_POSTFX_TEST "Build the v5 post-processing pipeline test" ON)
if(PPT_V5_POSTFX_TEST)
    find_package(CUDAToolkit REQUIRED)

    add_library(v5_postfx_cuda STATIC
        src/renderer/postfx/firefly_filter.cu
        src/renderer/postfx/bloom.cu
        src/renderer/postfx/tonemap.cu
    )
    set_target_properties(v5_postfx_cuda PROPERTIES
        CUDA_ARCHITECTURES native
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories(v5_postfx_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer
    )

    add_executable(postfx_test
        tests/integration/postfx_test.cpp
        src/renderer/postfx/postfx_pipeline.cpp
    )
    target_include_directories(postfx_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer
        ${CUDAToolkit_INCLUDE_DIRS}
        ${OptiX_INCLUDE_DIR}
    )
    target_link_libraries(postfx_test PRIVATE
        v5_postfx_cuda
        CUDA::cudart
    )
    target_compile_features(postfx_test PRIVATE cxx_std_17)

    if(MSVC)
        target_compile_options(postfx_test PRIVATE /W4 /fp:fast)
    else()
        target_compile_options(postfx_test PRIVATE -Wall -Wextra -O2)
    endif()
endif()
# =======================================================================

# =======================================================================
#  V5 Phase 10: Render Diagnostics Test
#  Tests: variance tracking, convergence analysis, bottleneck detection.
#  Requires CUDA GPU.
# =======================================================================
option(PPT_V5_DIAGNOSTICS_TEST "Build the v5 render diagnostics test" ON)
if(PPT_V5_DIAGNOSTICS_TEST)
    find_package(CUDAToolkit REQUIRED)

    add_executable(diagnostics_test
        tests/integration/diagnostics_test.cpp
        src/diagnose/variance_tracker.cpp
        src/diagnose/diagnostics.cpp
    )
    target_include_directories(diagnostics_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer
        ${CUDAToolkit_INCLUDE_DIRS}
        ${OptiX_INCLUDE_DIR}
    )
    target_link_libraries(diagnostics_test PRIVATE
        CUDA::cudart
    )
    target_compile_features(diagnostics_test PRIVATE cxx_std_17)
    target_compile_definitions(diagnostics_test PRIVATE PPT_USE_OPTIX=1)

    if(MSVC)
        target_compile_options(diagnostics_test PRIVATE /W4 /fp:fast /wd4505)
    else()
        target_compile_options(diagnostics_test PRIVATE -Wall -Wextra -O2)
    endif()
endif()
# =======================================================================

# =======================================================================
#  V5 Phase 11: Image Oracle Test
#  Tests: image metrics, noise analysis, artifact detection, oracle.
#  Pure CPU — no GPU required.
# =======================================================================
option(PPT_V5_ORACLE_TEST "Build the v5 image oracle test" ON)
if(PPT_V5_ORACLE_TEST)
    add_executable(oracle_test
        tests/integration/oracle_test.cpp
        src/diagnose/image_oracle.cpp
    )
    target_include_directories(oracle_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer
        ${OptiX_INCLUDE_DIR}
    )
    target_compile_features(oracle_test PRIVATE cxx_std_17)

    if(MSVC)
        target_compile_options(oracle_test PRIVATE /W4 /fp:fast)
    else()
        target_compile_options(oracle_test PRIVATE -Wall -Wextra -O2)
    endif()
endif()
# =======================================================================

# =======================================================================
#  v5 TEST HARNESS TEST (Phase 12 — Meta-Skill C)
#  Tests: test framework utilities, statistical tests, convergence
#  regression detection, reference scene construction.
#  Needs CUDAToolkit include dirs for types.h (vector_types.h).
# =======================================================================
option(PPT_V5_HARNESS_TEST "Build the v5 test harness test" ON)
if(PPT_V5_HARNESS_TEST)
    find_package(CUDAToolkit REQUIRED)

    add_executable(harness_test
        tests/integration/test_harness_test.cpp
    )
    target_include_directories(harness_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer
        ${CUDAToolkit_INCLUDE_DIRS}
        ${OptiX_INCLUDE_DIR}
    )
    target_compile_features(harness_test PRIVATE cxx_std_17)

    if(MSVC)
        target_compile_options(harness_test PRIVATE /W4 /fp:fast)
    else()
        target_compile_options(harness_test PRIVATE -Wall -Wextra -O2)
    endif()
endif()
# =======================================================================

# =======================================================================
#  v5 APPLICATION LAYER TEST (Phase 13)
#  Tests: RenderConfig, FrameBuffer, CLI args, font overlay, render log,
#  debug state.  Needs CUDAToolkit include dirs for types.h.
# =======================================================================
option(PPT_V5_APP_TEST "Build the v5 application layer test" OFF)  # TODO: fix after CMake reorder
if(PPT_V5_APP_TEST)
    find_package(CUDAToolkit REQUIRED)

    add_executable(app_test
        tests/integration/app_test.cpp
        src/renderer/app/render_config_json.cpp
        src/renderer/accel/accel_builder.cpp
    )
    target_include_directories(app_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer
        ${CUDAToolkit_INCLUDE_DIRS}
        ${OptiX_INCLUDE_DIR}
    )
    target_compile_features(app_test PRIVATE cxx_std_17)
    target_compile_definitions(app_test PRIVATE PPT_USE_OPTIX=1)

    if(MSVC)
        target_compile_options(app_test PRIVATE /W4 /fp:fast)
    else()
        target_compile_options(app_test PRIVATE -Wall -Wextra -O2)
    endif()
endif()
# =======================================================================

# =======================================================================
#  Phase 5: Test Infrastructure — GTest-based aggregate test targets
#
#  ppt_unit_tests         — CPU-only per-stage unit tests (<5s)
#  ppt_convergence_tests  — Convergence regression infrastructure tests
#  ppt_diagnostics_tests  — Variance/bottleneck/metrics tests
# =======================================================================

# ── ppt_unit_tests ───────────────────────────────────────────────────
option(PPT_UNIT_TESTS "Build ppt_unit_tests (GTest, CPU)" ON)
if(PPT_UNIT_TESTS)
    find_package(CUDAToolkit REQUIRED)
    include(FetchContent)
    if(NOT googletest_POPULATED)
        FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    if(NOT stb_POPULATED)
        FetchContent_Declare(stb
            GIT_REPOSITORY https://github.com/nothings/stb.git
            GIT_TAG        master
        )
        FetchContent_MakeAvailable(stb)
    endif()

    if(NOT tinyexr_POPULATED)
        FetchContent_Declare(tinyexr
            GIT_REPOSITORY https://github.com/syoyo/tinyexr.git
            GIT_TAG        v1.0.9
        )
        FetchContent_MakeAvailable(tinyexr)
    endif()

    add_executable(ppt_unit_tests
        tests/unit/test_scene_loader.cpp
        tests/unit/test_scene_profile.cpp
        tests/unit/test_camera.cpp
        tests/unit/test_material.cpp
        tests/unit/test_lighting.cpp
        tests/unit/test_integrator.cpp
        tests/unit/test_postfx.cpp
        tests/unit/test_acceleration.cpp
        # Source dependencies
        src/scene/pbrt/pbrt_parser.cpp
        src/scene/pbrt/pbrt_material_mapper.cpp
        src/scene/pbrt/pbrt_loader.cpp
        src/scene/pbrt/ply_reader.cpp
        src/scene/obj_loader.cpp
        tests/helpers/stb_tinyexr_impl.cpp
        ${tinyexr_SOURCE_DIR}/deps/miniz/miniz.c
    )
    target_include_directories(ppt_unit_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/helpers
        ${CUDAToolkit_INCLUDE_DIRS}
        ${OptiX_INCLUDE_DIR}
        ${stb_SOURCE_DIR}
        ${tinyexr_SOURCE_DIR}
        ${tinyexr_SOURCE_DIR}/deps/miniz
    )
    target_link_libraries(ppt_unit_tests PRIVATE GTest::gtest_main)
    target_compile_features(ppt_unit_tests PRIVATE cxx_std_17)
    if(MSVC)
        target_compile_options(ppt_unit_tests PRIVATE /W4 /fp:fast)
    else()
        target_compile_options(ppt_unit_tests PRIVATE -Wall -Wextra -O2)
    endif()
endif()

# ── ppt_convergence_tests ────────────────────────────────────────────
option(PPT_CONVERGENCE_TESTS "Build ppt_convergence_tests (GTest)" ON)
if(PPT_CONVERGENCE_TESTS)
    find_package(CUDAToolkit REQUIRED)
    include(FetchContent)
    if(NOT googletest_POPULATED)
        FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    add_executable(ppt_convergence_tests
        tests/convergence/test_convergence_direct.cpp
        tests/convergence/test_convergence_full.cpp
        tests/convergence/test_convergence_caustic.cpp
    )
    target_include_directories(ppt_convergence_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/helpers
        ${CUDAToolkit_INCLUDE_DIRS}
        ${OptiX_INCLUDE_DIR}
    )
    target_link_libraries(ppt_convergence_tests PRIVATE GTest::gtest_main)
    target_compile_features(ppt_convergence_tests PRIVATE cxx_std_17)
    if(MSVC)
        target_compile_options(ppt_convergence_tests PRIVATE /W4 /fp:fast)
    else()
        target_compile_options(ppt_convergence_tests PRIVATE -Wall -Wextra -O2)
    endif()
endif()

# ── ppt_diagnostics_tests ────────────────────────────────────────────
option(PPT_DIAGNOSTICS_TESTS "Build ppt_diagnostics_tests (GTest)" ON)
if(PPT_DIAGNOSTICS_TESTS)
    find_package(CUDAToolkit REQUIRED)
    include(FetchContent)
    if(NOT googletest_POPULATED)
        FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    add_executable(ppt_diagnostics_tests
        tests/diagnostics/test_variance_tracker.cpp
        tests/diagnostics/test_bottleneck_id.cpp
        tests/diagnostics/test_stage_metrics.cpp
        # Source dependencies
        src/diagnose/variance_tracker.cpp
        src/diagnose/diagnostics.cpp
    )
    target_include_directories(ppt_diagnostics_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/helpers
        ${CUDAToolkit_INCLUDE_DIRS}
        ${OptiX_INCLUDE_DIR}
    )
    target_link_libraries(ppt_diagnostics_tests PRIVATE
        GTest::gtest_main
        CUDA::cudart
    )
    target_compile_features(ppt_diagnostics_tests PRIVATE cxx_std_17)
    if(MSVC)
        target_compile_options(ppt_diagnostics_tests PRIVATE /W4 /fp:fast)
    else()
        target_compile_options(ppt_diagnostics_tests PRIVATE -Wall -Wextra -O2)
    endif()
endif()
# =======================================================================

# =======================================================================
#  MAIN EXECUTABLE AND TOOLS
#  The photon_tracer executable, scene editor, CUDA object library,
#  and supporting tool targets.
# =======================================================================
option(PPT_BUILD_LEGACY "Build the main renderer and tool targets" ON)
if(PPT_BUILD_LEGACY)
# -- Clean up stale FetchContent subbuild generator caches -------------
# When CMake regenerates (e.g. after a cmake upgrade or CMakeLists.txt edit),
# subbuild caches previously configured with a different generator (e.g. "Visual
# Studio") cause a fatal mismatch error. Detect and remove them so FetchContent
# re-configures them with the correct (Ninja) generator.
foreach(_dep IN ITEMS glfw stb tinyexr imgui googletest JoltPhysics nlohmann_json)
    set(_subbuild_cache "${CMAKE_BINARY_DIR}/_deps/${_dep}-subbuild/CMakeCache.txt")
    if(EXISTS "${_subbuild_cache}")
        file(STRINGS "${_subbuild_cache}" _gen_lines REGEX "CMAKE_GENERATOR:INTERNAL=")
        foreach(_line IN LISTS _gen_lines)
            if(NOT "${_line}" MATCHES "Ninja")
                message(STATUS "Removing stale ${_dep} subbuild cache (wrong generator)")
                file(REMOVE "${_subbuild_cache}")
                file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/_deps/${_dep}-subbuild/CMakeFiles")
            endif()
        endforeach()
    endif()
endforeach()

# -- Find packages -----------------------------------------------------
# (CUDAToolkit and OptiX already detected at top of file)

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

# tinyexr for HDR/EXR output
FetchContent_Declare(
    tinyexr
    GIT_REPOSITORY https://github.com/syoyo/tinyexr.git
    GIT_TAG        v1.0.9
)
FetchContent_MakeAvailable(tinyexr)

# nlohmann/json for runtime config (header-only)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)

# Dear ImGui for editor UI
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.8-docking
)
FetchContent_MakeAvailable(imgui)

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

# Set globally so CMake's internal try_compile (e.g. FindOpenMP) works.
set(CMAKE_CUDA_ARCHITECTURES "${PPT_CUDA_ARCHS}")

# -- Source files ------------------------------------------------------
set(CORE_SOURCES
    src/renderer/main.cpp
    src/renderer/app/viewer.cpp
    src/renderer/app/render_session.cpp
    src/renderer/app/render_config_json.cpp
    src/renderer/app/stb_tinyexr_impl.cpp
    ${tinyexr_SOURCE_DIR}/deps/miniz/miniz.c
    src/scene/obj_loader.cpp
    src/scene/pbrt/pbrt_parser.cpp
    src/scene/pbrt/ply_reader.cpp
    src/scene/pbrt/pbrt_material_mapper.cpp
    src/scene/pbrt/pbrt_loader.cpp
    src/renderer/accel/accel_builder.cpp
    src/renderer/postfx/postfx_pipeline.cpp
        src/renderer/postfx/optix_denoiser.cpp
    src/diagnose/variance_tracker.cpp
    src/diagnose/diagnostics.cpp
    src/diagnose/image_oracle.cpp
)

set(CUDA_SOURCES
    src/renderer/postfx/tonemap.cu
    src/renderer/postfx/bloom.cu
    src/renderer/postfx/firefly_filter.cu
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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer
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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${stb_SOURCE_DIR}
    ${tinyexr_SOURCE_DIR}
    ${tinyexr_SOURCE_DIR}/deps/miniz
    ${OptiX_INCLUDE_DIR}
)

# Embed DPI-awareness manifest so Windows doesn't bitmap-scale the window.
# The manifest is processed by the OS loader before any DLL init runs,
# which is the only reliable way to prevent CUDA/NVIDIA drivers from
# setting their own (wrong) DPI awareness first.
if(MSVC)
    set(MANIFEST_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/photon_tracer.manifest")
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
    nlohmann_json::nlohmann_json
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

# -- Reuse v5 OptiX PTX from test infrastructure ----------------------
# v5_optix_ptx target is already defined above (compiles src/renderer/accel/optix_programs.cu)
if(NOT TARGET v5_optix_ptx)
    set(V5_PTX_DIR "${CMAKE_BINARY_DIR}/ptx_v5")
    set(V5_PTX_OUTPUT "${V5_PTX_DIR}/optix_programs.ptx")

    add_custom_command(
        OUTPUT "${V5_PTX_OUTPUT}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${V5_PTX_DIR}"
        COMMAND ${CMAKE_CUDA_COMPILER}
            -ptx
            --std=c++17
            -arch=compute_75
            --use_fast_math
            -O3
            -DNDEBUG
            -DPPT_USE_OPTIX=1
            "-I${CMAKE_CURRENT_SOURCE_DIR}/src"
            "-I${CMAKE_CURRENT_SOURCE_DIR}/src/renderer"
            "-I${OptiX_INCLUDE_DIR}"
            -o "${V5_PTX_OUTPUT}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/accel/optix_programs.cu"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/accel/optix_programs.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/accel/optix_utils.cuh"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/accel/launch_params.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/types.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/config.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/random.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/hash.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/optix_nee.cuh"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/nee_shared.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/material/bsdf_shared.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/path_tracer.cuh"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/path_state.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/russian_roulette.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/integrator/sample_clamping.h"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Compiling v5 OptiX programs (optix_programs.cu -> PTX)"
        VERBATIM
    )
    add_custom_target(v5_optix_ptx DEPENDS "${V5_PTX_OUTPUT}")
else()
    # Target already exists, just get the output path
    set(V5_PTX_DIR "${CMAKE_BINARY_DIR}/ptx_v5")
    set(V5_PTX_OUTPUT "${V5_PTX_DIR}/optix_programs.ptx")
endif()
add_dependencies(photon_tracer v5_optix_ptx)

# Tell the host code where to find the PTX at runtime
target_compile_definitions(photon_tracer PRIVATE
    PTX_FILE_PATH="${V5_PTX_OUTPUT}"
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

# -- ppt_analyze — CPU-only scene analysis tool ------------------------
add_executable(ppt_analyze
    src/analyze/analyze_main.cpp
    src/scene/obj_loader.cpp
    src/scene/pbrt/pbrt_parser.cpp
    src/scene/pbrt/ply_reader.cpp
    src/scene/pbrt/pbrt_material_mapper.cpp
    src/scene/pbrt/pbrt_loader.cpp
    src/renderer/app/stb_tinyexr_impl.cpp
    ${tinyexr_SOURCE_DIR}/deps/miniz/miniz.c
)
target_include_directories(ppt_analyze PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer
    ${CUDAToolkit_INCLUDE_DIRS}
    ${stb_SOURCE_DIR}
    ${tinyexr_SOURCE_DIR}
    ${tinyexr_SOURCE_DIR}/deps/miniz
)
target_compile_features(ppt_analyze PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(ppt_analyze PRIVATE /W4 /fp:fast)
else()
    target_compile_options(ppt_analyze PRIVATE -Wall -Wextra -O2)
endif()

# -- ppt_diagnose — CPU-only image quality analysis tool ---------------
add_executable(ppt_diagnose
    src/diagnose/diagnose_main.cpp
    src/diagnose/image_oracle.cpp
    ${tinyexr_SOURCE_DIR}/deps/miniz/miniz.c
)
target_include_directories(ppt_diagnose PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer
    ${CUDAToolkit_INCLUDE_DIRS}
    ${stb_SOURCE_DIR}
    ${tinyexr_SOURCE_DIR}
    ${tinyexr_SOURCE_DIR}/deps/miniz
)
target_compile_features(ppt_diagnose PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(ppt_diagnose PRIVATE /W4 /fp:fast)
else()
    target_compile_options(ppt_diagnose PRIVATE -Wall -Wextra -O2)
endif()

# -- Legacy ppt_tests (v4 era — disabled, use ppt_unit_tests etc.) -----
option(PPT_BUILD_LEGACY_TESTS "Build legacy v4 ppt_tests target" OFF)
if(PPT_BUILD_TESTS AND PPT_BUILD_LEGACY_TESTS)
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
        tests/tinyexr_impl.cpp
        tests/feature_speed_test.cpp
        tests/test_medium.cpp
        tests/test_beam_estimator.cpp
        tests/test_speed_tweaks.cpp
        tests/test_tri_photon_irradiance.cpp
        src/scene/obj_loader.cpp
        src/scene/pbrt/pbrt_parser.cpp
        src/scene/pbrt/ply_reader.cpp
        src/scene/pbrt/pbrt_material_mapper.cpp
        src/scene/pbrt/pbrt_loader.cpp
        src/renderer/accel/accel_builder.cpp
        src/renderer/postfx/postfx_pipeline.cpp
    )

    add_executable(ppt_tests
            src/renderer/postfx/optix_denoiser.cpp
        ${TEST_SOURCES}
    )
    set_target_properties(ppt_tests PROPERTIES
        CUDA_ARCHITECTURES "${PPT_CUDA_ARCHS}"
        MSVC_DEBUG_INFORMATION_FORMAT "Embedded"
    )
    target_include_directories(ppt_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${stb_SOURCE_DIR}
        ${tinyexr_SOURCE_DIR}
        ${tinyexr_SOURCE_DIR}/deps/miniz
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
        V5_PTX_FILE_PATH="${V5_PTX_OUTPUT}"
    )
    add_dependencies(ppt_tests v5_optix_ptx)

    if(MSVC)
        target_compile_options(ppt_tests PRIVATE /MP /W4 /fp:fast)
    else()
        target_compile_options(ppt_tests PRIVATE -Wall -Wextra -O2)
    endif()

    include(GoogleTest)
    gtest_discover_tests(ppt_tests)
endif()

# -- Append PLY Instances Tool (CPU-only, standalone) ----------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/append_ply_instances/main.cpp")
add_executable(append_ply_instances
    tools/append_ply_instances/main.cpp
)
target_compile_features(append_ply_instances PRIVATE cxx_std_17)
target_compile_definitions(append_ply_instances PRIVATE
    SCENES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/scenes"
)
if(MSVC)
    target_compile_options(append_ply_instances PRIVATE /MP /W4 /fp:fast /O2)
else()
    target_compile_options(append_ply_instances PRIVATE -Wall -Wextra -O2)
endif()
endif()

# -- Procedural Scene Generators (CPU-only, standalone) --------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/procedural_scenes/menger_sponge_sphere/main.cpp")
add_executable(gen_menger_spheres
    tools/procedural_scenes/menger_sponge_sphere/main.cpp
)
target_compile_features(gen_menger_spheres PRIVATE cxx_std_17)
target_compile_definitions(gen_menger_spheres PRIVATE
    SCENES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/scenes"
)
if(MSVC)
    target_compile_options(gen_menger_spheres PRIVATE /MP /W4 /fp:fast /O2)
else()
    target_compile_options(gen_menger_spheres PRIVATE -Wall -Wextra -O2)
endif()
endif()

# -- Optional scene tools (CPU-only, standalone / Jolt Physics) ------------
set(GEN_STONES_SRC  "${CMAKE_CURRENT_SOURCE_DIR}/tools/scene_stones/src/gen_stones.cpp")
set(DROP_STONES_SRC "${CMAKE_CURRENT_SOURCE_DIR}/tools/scene_stones/src/drop_stones.cpp")
set(DROP_CUBES_SRC  "${CMAKE_CURRENT_SOURCE_DIR}/tools/bedroom_cubes/src/drop_cubes.cpp")

if(EXISTS "${GEN_STONES_SRC}")
    add_executable(gen_stones
        tools/scene_stones/src/gen_stones.cpp
    )
    target_compile_features(gen_stones PRIVATE cxx_std_17)
    target_compile_definitions(gen_stones PRIVATE
        OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tools/scene_stones/output"
    )
    if(MSVC)
        target_compile_options(gen_stones PRIVATE /MP /W4 /fp:fast /O2)
    else()
        target_compile_options(gen_stones PRIVATE -Wall -Wextra -O2)
    endif()
else()
    message(STATUS "Skipping gen_stones: ${GEN_STONES_SRC} not found")
endif()

if(EXISTS "${DROP_STONES_SRC}" OR EXISTS "${DROP_CUBES_SRC}")
    include(FetchContent)
    set(OVERRIDE_CXX_FLAGS              OFF CACHE BOOL "" FORCE)
    set(GENERATE_DEBUG_SYMBOLS          OFF CACHE BOOL "" FORCE)
    set(INTERPROCEDURAL_OPTIMIZATION    OFF CACHE BOOL "" FORCE)
    set(CPP_EXCEPTIONS_ENABLED          OFF CACHE BOOL "" FORCE)
    set(CPP_RTTI_ENABLED                OFF CACHE BOOL "" FORCE)
    set(ENABLE_ALL_WARNINGS             OFF CACHE BOOL "" FORCE)
    set(USE_STATIC_MSVC_RUNTIME_LIBRARY OFF CACHE BOOL "" FORCE)
    set(FLOATING_POINT_EXCEPTIONS_ENABLED OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        JoltPhysics
        GIT_REPOSITORY https://github.com/jrouwe/JoltPhysics.git
        GIT_TAG        v5.5.0
        SOURCE_SUBDIR  Build
    )
    FetchContent_MakeAvailable(JoltPhysics)
endif()

if(EXISTS "${DROP_STONES_SRC}")
    add_executable(drop_stones
        tools/scene_stones/src/drop_stones.cpp
    )
    target_compile_features(drop_stones PRIVATE cxx_std_17)
    target_include_directories(drop_stones PRIVATE ${JoltPhysics_SOURCE_DIR}/..)
    target_link_libraries(drop_stones PRIVATE Jolt)
    target_compile_definitions(drop_stones PRIVATE
        MESHES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tools/scene_stones/meshes/stones/meshes"
        BOULDER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tools/scene_stones/meshes/boulder/meshes"
        GRASS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tools/scene_stones/meshes/grass/meshes"
        TEX_BOULDER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tools/scene_stones/meshes/boulder/textures"
        TEX_GRASS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tools/scene_stones/meshes/grass/textures"
        TEX_STONES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tools/scene_stones/meshes/stones/textures"
        OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tools/scene_stones/output"
    )
    if(MSVC)
        target_compile_options(drop_stones PRIVATE /MP /W4 /fp:fast /O2)
    else()
        target_compile_options(drop_stones PRIVATE -Wall -Wextra -O2)
    endif()
else()
    message(STATUS "Skipping drop_stones: ${DROP_STONES_SRC} not found")
endif()

if(EXISTS "${DROP_CUBES_SRC}")
    add_executable(drop_cubes
        tools/bedroom_cubes/src/drop_cubes.cpp
    )
    target_compile_features(drop_cubes PRIVATE cxx_std_17)
    target_include_directories(drop_cubes PRIVATE ${JoltPhysics_SOURCE_DIR}/..)
    target_link_libraries(drop_cubes PRIVATE Jolt)
    target_compile_definitions(drop_cubes PRIVATE
        SCENE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/scenes/bedroom"
        CORNELL_OBJ="${CMAKE_CURRENT_SOURCE_DIR}/scenes/cornell_box/cornellbox.obj"
    )
    if(MSVC)
        target_compile_options(drop_cubes PRIVATE /MP /W4 /fp:fast /O2)
    else()
        target_compile_options(drop_cubes PRIVATE -Wall -Wextra -O2)
    endif()
else()
    message(STATUS "Skipping drop_cubes: ${DROP_CUBES_SRC} not found")
endif()

# -- Scene Editor (OptiX + Dear ImGui) ────────────────────────────────
option(PPT_BUILD_EDITOR "Build the PBRTv4 scene editor" OFF)
if(PPT_BUILD_EDITOR)

# ImGui static library with GLFW + OpenGL3 backends
add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_lib PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_lib PUBLIC glfw OpenGL::GL)
if(MSVC)
    target_compile_options(imgui_lib PRIVATE /MP /W0)
endif()

# Editor sources (same scene/renderer stack as photon_tracer, different front-end)
set(EDITOR_SHARED_SOURCES
    src/scene/obj_loader.cpp
    src/scene/camera_prepass.cpp
    src/scene/pbrt/pbrt_parser.cpp
    src/scene/pbrt/ply_reader.cpp
    src/scene/pbrt/pbrt_material_mapper.cpp
    src/scene/pbrt/pbrt_loader.cpp
    src/optix/optix_renderer.cpp
    src/optix/optix_setup.cpp
    src/optix/optix_upload.cpp
    src/optix/optix_denoiser.cpp
    src/renderer/postfx/postfx_pipeline.cpp
)

file(GLOB EDITOR_SOURCES tools/editor/src/*.cpp)

add_executable(scene_editor
    ${EDITOR_SOURCES}
    ${EDITOR_SHARED_SOURCES}
)

set_target_properties(scene_editor PROPERTIES
    CUDA_ARCHITECTURES "${PPT_CUDA_ARCHS}"
)

target_include_directories(scene_editor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${stb_SOURCE_DIR}
    ${tinyexr_SOURCE_DIR}
    ${tinyexr_SOURCE_DIR}/deps/miniz
    ${OptiX_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/editor/src
)

target_link_libraries(scene_editor PRIVATE
    ppt_cuda_objects
    imgui_lib
    CUDA::cudart
    glfw
    OpenGL::GL
)
if(WIN32)
    target_link_libraries(scene_editor PRIVATE comdlg32)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(scene_editor PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_definitions(scene_editor PRIVATE
    PPT_USE_OPTIX=1
    PTX_FILE_PATH="${OPTIX_PTX_OUTPUT}"
    SCENES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/scenes"
)
add_dependencies(scene_editor optix_ptx)

if(MSVC)
    target_compile_options(scene_editor PRIVATE /MP /W4 /fp:fast)
    set(EDITOR_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/photon_tracer.manifest")
    target_link_options(scene_editor PRIVATE
        "/MANIFEST:EMBED"
        "/MANIFESTINPUT:${EDITOR_MANIFEST}"
    )
endif()

# Suppress C4505 from CUDA/stb headers
if(MSVC)
    set_source_files_properties(${EDITOR_SHARED_SOURCES} PROPERTIES
        COMPILE_OPTIONS "/wd4505")
endif()

endif() # PPT_BUILD_EDITOR

endif() # PPT_BUILD_LEGACY
