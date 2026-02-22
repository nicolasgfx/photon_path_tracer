@echo off
REM -------------------------------------------------------------------
REM  build.bat -- Build the Spectral Photon + Path Tracer
REM -------------------------------------------------------------------
REM
REM  Usage:
REM    build.bat                 Build in Release mode (incremental)
REM    build.bat rebuild         Clean rebuild (use after changing config.h)
REM
REM -------------------------------------------------------------------

setlocal enabledelayedexpansion

set BUILD_DIR=build
set BUILD_TYPE=Release
set CMAKE_OPTS=
set BUILD_TARGET=--target photon_tracer
set CLEAN_FIRST=--clean-first

REM -- Configure --
echo [build.bat] Configuring (%BUILD_TYPE%)...
cmake -B %BUILD_DIR% %CMAKE_OPTS%
if errorlevel 1 goto :error

REM -- Build --
echo [build.bat] Building (%BUILD_TYPE%)...
cmake --build %BUILD_DIR% %BUILD_TARGET% %CLEAN_FIRST% --config %BUILD_TYPE%
if errorlevel 1 goto :error

echo [build.bat] Build successful.
goto :done

:error
echo.
echo [build.bat] ERROR: Build failed!
exit /b 1

:done
endlocal
