@echo off
REM -------------------------------------------------------------------
REM  build.bat -- Build the Spectral Photon + Path Tracer (Ninja)
REM -------------------------------------------------------------------
REM
REM  Usage:
REM    build.bat                 Build photon_tracer in Release mode
REM    build.bat rebuild         Clean + rebuild
REM    build.bat test            Build photon_tracer + ppt_tests
REM    build.bat all             Build everything (all archs + tests)
REM
REM -------------------------------------------------------------------

setlocal enabledelayedexpansion

set BUILD_DIR=build
set BUILD_TYPE=Release
set BUILD_TARGET=photon_tracer
set CLEAN_FIRST=
set CMAKE_EXTRA=
set PARALLEL_FLAGS=-j %NUMBER_OF_PROCESSORS%

REM -- Parse arguments -------------------------------------------------
:parse_args
if "%~1"=="" goto :args_done
if /I "%~1"=="rebuild" set CLEAN_FIRST=--clean-first
if /I "%~1"=="rebuild" echo [build.bat] Clean rebuild requested.
if /I "%~1"=="test" set "BUILD_TARGET=photon_tracer ppt_tests"
if /I "%~1"=="test" set "CMAKE_EXTRA=-DPPT_BUILD_TESTS=ON"
if /I "%~1"=="test" echo [build.bat] Building with tests.
if /I "%~1"=="all" set "BUILD_TARGET="
if /I "%~1"=="all" set "CMAKE_EXTRA=-DPPT_BUILD_TESTS=ON -DPPT_CUDA_ARCH_ALL=ON"
if /I "%~1"=="all" echo [build.bat] Full build (all archs + tests).
shift
goto :parse_args
:args_done

REM -- Set up MSVC + Ninja environment --------------------------------
call :find_vcvars
if not defined VCVARS (
    echo [build.bat] ERROR: Could not find vcvarsall.bat for VS 2022.
    exit /b 1
)

REM Only call vcvarsall if cl.exe is not already on PATH
where cl.exe >nul 2>&1
if errorlevel 1 (
    echo [build.bat] Setting up MSVC environment...
    call "!VCVARS!" x64 >nul
)

REM Add VS-bundled Ninja to PATH if needed
where ninja.exe >nul 2>&1
if errorlevel 1 call :find_ninja

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

REM -- Configure (only if needed or if extra options changed) ----------
REM  Also reconfigure if the cache says a non-Ninja generator (e.g. VS)
set "NEED_CONFIGURE="
if not exist "%BUILD_DIR%\build.ninja" set "NEED_CONFIGURE=1"
if not exist "%BUILD_DIR%\CMakeCache.txt" (
    set "NEED_CONFIGURE=1"
) else (
    findstr /C:"CMAKE_GENERATOR:INTERNAL=Ninja" "%BUILD_DIR%\CMakeCache.txt" >nul 2>&1
    if errorlevel 1 (
        echo [build.bat] Stale CMake cache detected ^(wrong generator^). Cleaning...
        del /q "%BUILD_DIR%\CMakeCache.txt" 2>nul
        if exist "%BUILD_DIR%\CMakeFiles" rmdir /s /q "%BUILD_DIR%\CMakeFiles" 2>nul
        set "NEED_CONFIGURE=1"
    )
)

if defined NEED_CONFIGURE (
    echo [build.bat] Configuring ^(%BUILD_TYPE%, Ninja^)...
    cmake -B %BUILD_DIR% -G Ninja -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl %CMAKE_EXTRA%
    if errorlevel 1 goto :error
) else if defined CMAKE_EXTRA (
    REM Re-configure to toggle test/arch options
    cmake -B %BUILD_DIR% %CMAKE_EXTRA%
    if errorlevel 1 goto :error
)

REM -- Build -----------------------------------------------------------
echo [build.bat] Building (%BUILD_TYPE%)...
set "TARGET_FLAGS="
if defined BUILD_TARGET (
    for %%T in (%BUILD_TARGET%) do set "TARGET_FLAGS=!TARGET_FLAGS! --target %%T"
)
cmake --build %BUILD_DIR% %TARGET_FLAGS% %CLEAN_FIRST% %PARALLEL_FLAGS%
if errorlevel 1 goto :error

echo [build.bat] Build successful.
goto :done

:error
echo.
echo [build.bat] ERROR: Build failed!
exit /b 1

:find_vcvars
set "VCVARS="
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
goto :eof

:find_ninja
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
goto :eof

:done
endlocal
