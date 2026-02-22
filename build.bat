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
set CLEAN_FIRST=
set PARALLEL_FLAGS=-j %NUMBER_OF_PROCESSORS%

REM Check arguments
if /I "%1"=="rebuild" (
    set CLEAN_FIRST=--clean-first
    echo [build.bat] Clean rebuild requested.
)

REM -- Lock Check --
if exist "%BUILD_DIR%\build.lock" (
    echo [build.bat] ERROR: A build is already in progress ^(lock file exists^).
    echo             If you are sure no other build is running, delete "%BUILD_DIR%\build.lock"
    exit /b 1
)

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
echo locked > "%BUILD_DIR%\build.lock"

REM -- Configure --
REM Only configure if CMakeCache doesn't exist or if we want to ensure fresh config options
if not exist "%BUILD_DIR%\CMakeCache.txt" (
    echo [build.bat] Configuring ^(%BUILD_TYPE%^)...
    cmake -B %BUILD_DIR% %CMAKE_OPTS%
    if errorlevel 1 goto :error
)

REM -- Build --
echo [build.bat] Building (%BUILD_TYPE%)...
cmake --build %BUILD_DIR% %BUILD_TARGET% %CLEAN_FIRST% --config %BUILD_TYPE% %PARALLEL_FLAGS%
if errorlevel 1 goto :error

echo [build.bat] Build successful.
del "%BUILD_DIR%\build.lock"
goto :done

:error
echo.
echo [build.bat] ERROR: Build failed!
if exist "%BUILD_DIR%\build.lock" del "%BUILD_DIR%\build.lock"
exit /b 1

:done
endlocal
