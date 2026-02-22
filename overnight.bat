@echo off
REM -------------------------------------------------------------------
REM  overnight.bat -- Build & run ALL tests overnight, detailed reports
REM -------------------------------------------------------------------
REM
REM  Usage:
REM    overnight.bat                   Run all tests (Release)
REM    overnight.bat --filter X        Run only matching tests
REM    overnight.bat --debug           Build & run in Debug mode
REM    overnight.bat --no-build        Skip build, run existing binary
REM
REM  Output:  tests\runs\YYYYMMDD_HHMMSS\
REM             summary.txt          Quick pass/fail + key metrics
REM             report.txt           Detailed per-test report with stdout
REM             report.json          Machine-readable JSON results
REM             full_output.txt      Raw console output (stdout+stderr)
REM             *.ppm                Comparison images (from pixel tests)
REM
REM  Designed for unattended overnight runs -- just launch and go to bed.
REM -------------------------------------------------------------------

setlocal enabledelayedexpansion

set BUILD_DIR=build
set BUILD_TYPE=Release
set GTEST_FILTER=*
set DO_BUILD=1

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
if "%~1"=="--no-build" (
    set DO_BUILD=0
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
echo  overnight.bat -- Unattended overnight test run
echo  %TIMESTAMP%
echo ===================================================================
echo  Build:  %BUILD_TYPE%
echo  Filter: %GTEST_FILTER%
echo  Output: %RUN_DIR%
echo ===================================================================
echo.

REM -- Create output folder -------------------------------------------
if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"

REM -- Log start time -------------------------------------------------
echo [overnight] Started at %DATE% %TIME% > "%RUN_DIR%\timing.txt"

REM -- Configure & build (unless --no-build) --------------------------
if %DO_BUILD% EQU 0 goto :skip_build

echo [overnight] Configuring CMake...
cmake -B %BUILD_DIR% -DPPT_BUILD_TESTS=ON > "%RUN_DIR%\build_log.txt" 2>&1
if errorlevel 1 (
    echo [overnight] ERROR: CMake configure failed!
    echo See %RUN_DIR%\build_log.txt for details.
    exit /b 1
)

echo [overnight] Building all targets (%BUILD_TYPE%)...
cmake --build %BUILD_DIR% --config %BUILD_TYPE% >> "%RUN_DIR%\build_log.txt" 2>&1
if errorlevel 1 (
    echo [overnight] ERROR: Build failed!
    echo.
    echo Build errors:
    cmake --build %BUILD_DIR% --config %BUILD_TYPE% 2>&1 | findstr /i "error"
    echo.
    echo See %RUN_DIR%\build_log.txt for full details.
    exit /b 1
)
echo [overnight] Build OK.
echo.

:skip_build

REM -- Verify binary exists -------------------------------------------
if not exist "%BUILD_DIR%\%BUILD_TYPE%\ppt_tests.exe" (
    echo [overnight] ERROR: ppt_tests.exe not found at %BUILD_DIR%\%BUILD_TYPE%\ppt_tests.exe
    exit /b 1
)

REM -- Set environment for test output --------------------------------
set PPT_TEST_OUTPUT_DIR=%CD%\%RUN_DIR%
set PPT_REPORT_DIR=%CD%\%RUN_DIR%

REM -- Run tests ------------------------------------------------------
echo [overnight] Running tests... (this may take several hours)
echo [overnight] Output: %RUN_DIR%
echo [overnight] Filter: %GTEST_FILTER%
echo.

REM Run with --report-dir so the GTest listener writes structured reports,
REM and also capture full stdout+stderr to full_output.txt.
REM GTest XML output is also generated for CI/tooling compatibility.
%BUILD_DIR%\%BUILD_TYPE%\ppt_tests.exe ^
    --gtest_print_time=1 ^
    --gtest_filter=%GTEST_FILTER% ^
    --gtest_output=xml:%RUN_DIR%\gtest_results.xml ^
    --report-dir=%CD%\%RUN_DIR% ^
    > "%RUN_DIR%\full_output.txt" 2>&1

set TEST_EXIT=%ERRORLEVEL%

REM -- Log end time ----------------------------------------------------
echo [overnight] Ended at %DATE% %TIME% >> "%RUN_DIR%\timing.txt"

REM -- Count output files ----------------------------------------------
set IMG_COUNT=0
for %%f in ("%RUN_DIR%\*.ppm") do set /a IMG_COUNT+=1
set REPORT_COUNT=0
for %%f in ("%RUN_DIR%\*.txt") do set /a REPORT_COUNT+=1
set JSON_COUNT=0
for %%f in ("%RUN_DIR%\*.json") do set /a JSON_COUNT+=1

REM -- Print results to console ----------------------------------------
echo.
echo ===================================================================
echo  OVERNIGHT RUN COMPLETE
echo ===================================================================
echo.

if %TEST_EXIT% EQU 0 (
    echo  RESULT:  ALL TESTS PASSED
) else (
    echo  RESULT:  SOME TESTS FAILED  (exit code %TEST_EXIT%)
)
echo.

REM -- Show summary if it was generated --------------------------------
if exist "%RUN_DIR%\summary.txt" (
    type "%RUN_DIR%\summary.txt"
    echo.
) else (
    REM Fall back to extracting key info from full output
    echo  ~~~ Test Counts ~~~
    findstr /c:"PASSED" "%RUN_DIR%\full_output.txt" 2>nul | findstr /c:"test"
    findstr /c:"FAILED" "%RUN_DIR%\full_output.txt" 2>nul | findstr /c:"test"
    echo.
)

echo ===================================================================
echo  Output directory:  %RUN_DIR%
echo  Files: %REPORT_COUNT% reports, %JSON_COUNT% JSON, %IMG_COUNT% images
echo.
echo  Reports:
if exist "%RUN_DIR%\summary.txt"     echo    summary.txt        Quick overview + key metrics
if exist "%RUN_DIR%\report.txt"      echo    report.txt         Detailed per-test results
if exist "%RUN_DIR%\report.json"     echo    report.json        Machine-readable JSON
if exist "%RUN_DIR%\full_output.txt" echo    full_output.txt    Raw test output
if exist "%RUN_DIR%\gtest_results.xml" echo    gtest_results.xml  GTest XML (CI compatible)
if exist "%RUN_DIR%\build_log.txt"   echo    build_log.txt      Build output
if exist "%RUN_DIR%\timing.txt"      echo    timing.txt         Start/end timestamps
echo ===================================================================
echo.

if %TEST_EXIT% NEQ 0 exit /b %TEST_EXIT%
endlocal
