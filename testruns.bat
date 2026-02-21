@echo off
REM -------------------------------------------------------------------
REM  testruns.bat -- Build & run all tests, save results to timestamped folder
REM -------------------------------------------------------------------
REM
REM  Usage:
REM    testruns.bat                Run all tests (Release build)
REM    testruns.bat --filter X     Run only matching tests (gtest filter)
REM    testruns.bat --debug        Build & run in Debug mode
REM
REM  Output:  tests\runs\YYYYMMDD_HHMMSS\
REM             summary.txt          Key metrics at the top
REM             full_output.txt      Complete test stdout + stderr
REM             gt_combined.ppm      Ground-truth combined rendering
REM             opt_combined.ppm     Optimized combined rendering
REM             diff_combined.ppm    |GT - Opt| heatmap
REM             gt_nee_direct.ppm    Ground-truth NEE-only
REM             opt_nee_direct.ppm   Optimized NEE-only
REM             diff_nee_direct.ppm  NEE difference heatmap
REM             gt_photon_indirect.ppm   GT photon-only
REM             opt_photon_indirect.ppm  Opt photon-only
REM             diff_photon_indirect.ppm Photon difference heatmap
REM             shadow_mask.ppm      Shadow region mask (blue=shadow)
REM -------------------------------------------------------------------

setlocal enabledelayedexpansion

set BUILD_DIR=build
set BUILD_TYPE=Release
set GTEST_FILTER=*

REM -- Parse arguments ------------------------------------------------
:parse_args
if "%~1"=="" goto :args_done
if "%~1"=="--filter" (
    set GTEST_FILTER=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--debug" (
    set BUILD_TYPE=Debug
    shift
    goto :parse_args
)
shift
goto :parse_args
:args_done

REM -- Timestamp & output folder (locale-independent) -----------------
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do set DT=%%I
set TIMESTAMP=%DT:~0,4%%DT:~4,2%%DT:~6,2%_%DT:~8,2%%DT:~10,2%%DT:~12,2%
set RUN_DIR=tests\runs\%TIMESTAMP%

echo ===================================================================
echo  testruns.bat -- %TIMESTAMP%
echo ===================================================================
echo  Build:  %BUILD_TYPE%
echo  Filter: %GTEST_FILTER%
echo  Output: %RUN_DIR%
echo ===================================================================
echo.

REM -- Create output folder -------------------------------------------
if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"

REM -- Configure & build ----------------------------------------------
echo [testruns] Configuring...
cmake -B %BUILD_DIR% -DPPT_BUILD_TESTS=ON >nul 2>&1
if errorlevel 1 (
    echo [testruns] ERROR: CMake configure failed!
    exit /b 1
)

echo [testruns] Building all targets (%BUILD_TYPE%)...
cmake --build %BUILD_DIR% --config %BUILD_TYPE% >nul 2>&1
if errorlevel 1 (
    echo [testruns] ERROR: Build failed!
    cmake --build %BUILD_DIR% --config %BUILD_TYPE% 2>&1 | findstr /i "error"
    exit /b 1
)
echo [testruns] Build OK.
echo.

REM -- Run tests (images go into the timestamped folder) --------------
set PPT_TEST_OUTPUT_DIR=%CD%\%RUN_DIR%
echo [testruns] Running tests...
echo.

%BUILD_DIR%\%BUILD_TYPE%\ppt_tests.exe --gtest_print_time=1 --gtest_filter=%GTEST_FILTER% > "%RUN_DIR%\full_output.txt" 2>&1
set TEST_EXIT=%ERRORLEVEL%

REM -- Extract key metrics into summary.txt ---------------------------
echo [testruns] Generating summary...

(
    echo ===================================================================
    echo  PHOTON PATH TRACER -- TEST RUN SUMMARY
    echo  %TIMESTAMP%  ^|  Build: %BUILD_TYPE%  ^|  Filter: %GTEST_FILTER%
    echo ===================================================================
    echo.

    REM -- Pass/fail banner
    if %TEST_EXIT% EQU 0 (
        echo  RESULT:  ALL TESTS PASSED
    ) else (
        echo  RESULT:  SOME TESTS FAILED  (exit code %TEST_EXIT%^)
    )
    echo.

    REM -- Overall counts
    echo  ~~~ Test Counts ~~~
    findstr /c:"PASSED" "%RUN_DIR%\full_output.txt" | findstr /c:"test"
    findstr /c:"FAILED" "%RUN_DIR%\full_output.txt" | findstr /c:"test"
    echo.

    REM -- Pixel comparison metrics (most relevant)
    echo  ~~~ Pixel Comparison (128x128, GT vs Optimized) ~~~
    findstr /c:"[NEE-DirectOnly]" "%RUN_DIR%\full_output.txt"
    findstr /c:"[PhotonIndirect]" "%RUN_DIR%\full_output.txt"
    findstr /c:"[Combined   ]" "%RUN_DIR%\full_output.txt"
    findstr /c:"[Global]" "%RUN_DIR%\full_output.txt"
    echo.

    REM -- Shadow & spatial
    echo  ~~~ Shadow and Spatial ~~~
    findstr /c:"[ShadowIndirect]" "%RUN_DIR%\full_output.txt"
    findstr /c:"[Spatial]" "%RUN_DIR%\full_output.txt"
    findstr /c:"[Decomposition]" "%RUN_DIR%\full_output.txt"
    echo.

    REM -- Photon lobe & spectral
    echo  ~~~ Photon Lobe and Spectral ~~~
    findstr /c:"[PhotonLobe]" "%RUN_DIR%\full_output.txt"
    findstr /c:"[SpectralBin]" "%RUN_DIR%\full_output.txt"
    echo.

    REM -- Ground truth comparison (existing tests)
    echo  ~~~ Ground Truth Comparison ~~~
    findstr /c:"[CenterPixel]" "%RUN_DIR%\full_output.txt"
    findstr /c:"[FullImage]" "%RUN_DIR%\full_output.txt"
    findstr /c:"[Variance]" "%RUN_DIR%\full_output.txt"
    findstr /c:"[DirectLighting]" "%RUN_DIR%\full_output.txt"
    findstr /c:"[SingleBounce]" "%RUN_DIR%\full_output.txt"
    findstr /c:"[FullBounce]" "%RUN_DIR%\full_output.txt"
    findstr /c:"[Energy]" "%RUN_DIR%\full_output.txt"
    echo.

    REM -- Per-ray validation
    echo  ~~~ Per-Ray Validation ~~~
    findstr /c:"Combined relErr:" "%RUN_DIR%\full_output.txt"
    findstr /c:"[Decomposition]" "%RUN_DIR%\full_output.txt"
    findstr /c:"[SpectralBins]" "%RUN_DIR%\full_output.txt"
    echo.

    REM -- Images saved
    echo  ~~~ Comparison Images ~~~
    findstr /c:"[SaveImages]" "%RUN_DIR%\full_output.txt"
    echo.

    REM -- List image files
    echo  Files in %RUN_DIR%:
    for %%f in ("%RUN_DIR%\*.ppm") do echo    %%~nxf
    echo.
    echo ===================================================================
    echo  Full output: %RUN_DIR%\full_output.txt
    echo ===================================================================
) > "%RUN_DIR%\summary.txt"

REM -- Print summary to console ---------------------------------------
echo.
echo ===================================================================
type "%RUN_DIR%\summary.txt"
echo.

REM -- Count images ---------------------------------------------------
set IMG_COUNT=0
for %%f in ("%RUN_DIR%\*.ppm") do set /a IMG_COUNT+=1

echo [testruns] Done.  %IMG_COUNT% images saved.
echo [testruns] Results: %RUN_DIR%\
echo.

if %TEST_EXIT% NEQ 0 exit /b %TEST_EXIT%
endlocal
