@echo off
REM -------------------------------------------------------------------
REM  run.bat -- Build and run the Spectral Photon + Path Tracer
REM -------------------------------------------------------------------
REM
REM  Usage:
REM    run.bat                   Build & run interactive debug viewer
REM    run.bat --spp 64          Set samples for final render (R key)
REM    run.bat test              Build & run unit tests
REM    run.bat build             Build only (no run)
REM    run.bat clean             Delete build directory
REM    run.bat release           Build & render in Release mode
REM
REM -------------------------------------------------------------------

setlocal enabledelayedexpansion

set BUILD_DIR=build
set BUILD_TYPE=Debug
set TARGET=photon_tracer
set RUN_ARGS=

REM -- Parse first argument -----------------------------------------------
if "%1"=="test" (
    echo [run.bat] Building and running tests...
    cmake -B %BUILD_DIR% -DPPT_BUILD_TESTS=ON
    cmake --build %BUILD_DIR% --target ppt_tests --config Debug
    if errorlevel 1 goto :error
    echo.
    echo [run.bat] Running tests...
    %BUILD_DIR%\Debug\ppt_tests.exe
    goto :done
)

if "%1"=="build" (
    echo [run.bat] Building...
    cmake -B %BUILD_DIR%
    cmake --build %BUILD_DIR% --config Debug
    if errorlevel 1 goto :error
    echo [run.bat] Build successful.
    goto :done
)

if "%1"=="release" (
    set BUILD_TYPE=Release
    shift
    goto :build_and_run
)

if "%1"=="clean" (
    echo [run.bat] Cleaning build directory...
    if exist %BUILD_DIR% rmdir /s /q %BUILD_DIR%
    echo [run.bat] Clean.
    goto :done
)

:build_and_run
REM -- Collect remaining args for the renderer --
set RUN_ARGS=%1 %2 %3 %4 %5 %6 %7 %8 %9

REM -- Configure --
echo [run.bat] Configuring (%BUILD_TYPE%)...
cmake -B %BUILD_DIR%
if errorlevel 1 goto :error

REM -- Build --
echo [run.bat] Building (%BUILD_TYPE%)...
cmake --build %BUILD_DIR% --config %BUILD_TYPE%
if errorlevel 1 goto :error

REM -- Run --
echo.
echo [run.bat] Running renderer...
echo ------------------------------------------------
%BUILD_DIR%\%BUILD_TYPE%\%TARGET%.exe %RUN_ARGS%
echo ------------------------------------------------
goto :done

:error
echo.
echo [run.bat] ERROR: Build failed!
exit /b 1

:done
endlocal
