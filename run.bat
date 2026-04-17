@echo off
REM -------------------------------------------------------------------
REM  run.bat -- Build and run the Spectral Photon + Path Tracer
REM -------------------------------------------------------------------
REM
REM  Usage:
REM    run.bat                   Build & run interactive debug viewer
REM    run.bat --spp 64          Pass args to the renderer
REM    run.bat test              Build & run fast unit tests
REM    run.bat test-all          Build & run ALL tests (slow!)
REM    run.bat build             Build only (no run)
REM    run.bat clean             Delete build directory
REM
REM -------------------------------------------------------------------

setlocal enabledelayedexpansion

set BUILD_DIR=build
set EXE=%BUILD_DIR%\photon_tracer.exe

REM -- Parse first argument -----------------------------------------------
if "%1"=="test" (
    call build.bat test
    if errorlevel 1 goto :error
    echo.
    echo [run.bat] Running fast unit tests...
    %BUILD_DIR%\ppt_tests.exe --gtest_filter=-*Integration*:*PixelComparison*:*GroundTruth*:*PerRay*:*SpeedTest*:*SpeedTweaks*:*CpuGpu*
    goto :done
)

if "%1"=="test-all" (
    call build.bat test
    if errorlevel 1 goto :error
    echo.
    echo [run.bat] Running ALL tests ^(this may take hours^)...
    %BUILD_DIR%\ppt_tests.exe --gtest_print_time=1
    goto :done
)

if "%1"=="build" (
    call build.bat
    if errorlevel 1 goto :error
    goto :done
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

REM -- Build --
call build.bat
if errorlevel 1 goto :error

REM -- Run --
echo.
echo [run.bat] Running renderer...
echo ------------------------------------------------
%EXE% %RUN_ARGS%
set PPT_RET=%errorlevel%
echo ------------------------------------------------
endlocal & exit /b %PPT_RET%

:error
echo.
echo [run.bat] ERROR: Build failed!
exit /b 1

:done
endlocal
