@echo off
REM -------------------------------------------------------------------
REM  build.bat -- Build the Spectral Photon + Path Tracer
REM -------------------------------------------------------------------
REM
REM  Usage:
REM    build.bat                 Build in Debug mode
REM    build.bat release         Build in Release mode
REM    build.bat test            Build unit tests (Debug)
REM
REM -------------------------------------------------------------------

setlocal enabledelayedexpansion

set BUILD_DIR=build
set BUILD_TYPE=Debug
set CMAKE_OPTS=
set BUILD_TARGET=

if /i "%1"=="release" (
    set BUILD_TYPE=Release
)

if /i "%1"=="test" (
    set CMAKE_OPTS=-DPPT_BUILD_TESTS=ON
    set BUILD_TARGET=--target ppt_tests
)

REM -- Configure --
echo [build.bat] Configuring (%BUILD_TYPE%)...
cmake -B %BUILD_DIR% %CMAKE_OPTS%
if errorlevel 1 goto :error

REM -- Build --
echo [build.bat] Building (%BUILD_TYPE%)...
cmake --build %BUILD_DIR% %BUILD_TARGET% --config %BUILD_TYPE%
if errorlevel 1 goto :error

echo [build.bat] Build successful.
goto :done

:error
echo.
echo [build.bat] ERROR: Build failed!
exit /b 1

:done
endlocal
