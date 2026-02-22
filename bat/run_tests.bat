@echo off
setlocal EnableDelayedExpansion

set "BAT_DIR=%~dp0"
if "%BAT_DIR:~-1%"=="\" set "BAT_DIR=%BAT_DIR:~0,-1%"
for %%I in ("%BAT_DIR%\..") do set "ROOT_DIR=%%~fI"

echo ==========================================================
echo      TRADINGBOT TEST RUNNER (v6.15.0)
echo ==========================================================
echo.

pushd "%ROOT_DIR%" >nul

python --version
if errorlevel 1 (
    echo [FATAL] Python is not found!
    popd >nul
    exit /b 1
)

echo.
echo [INFO] Running pytest suite...
python -m pytest -q -ra modules\tests
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" (
    echo [FAIL] Tests failed with code %RC%.
) else (
    echo [OK] All tests passed.
)

popd >nul
exit /b %RC%
