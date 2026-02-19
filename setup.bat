@echo off
setlocal EnableDelayedExpansion

:: --- 1. SETUP VARIABLES ---
set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

:: Get Date/Time
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set "YYYY=%datetime:~0,4%"
set "MM=%datetime:~4,2%"
set "DD=%datetime:~6,2%"
set "HH=%datetime:~8,2%"
set "Min=%datetime:~10,2%"
set "SS=%datetime:~12,2%"

:: --- 2. USER INPUT ---
echo ==========================================================
echo      TRADINGBOT SETUP MANAGER (v6.22.1)
echo ==========================================================
echo.
set /p TARGET_VERSION="Enter Version (e.g. v6.22.1): "
if "%TARGET_VERSION%"=="" set TARGET_VERSION=v6.22.1
set /p BUILD_PROFILE="Build profile [LEAN/FULL_AI] (default LEAN): "
if "%BUILD_PROFILE%"=="" set BUILD_PROFILE=LEAN
set "BUILD_PROFILE=%BUILD_PROFILE: =%"
if /I not "%BUILD_PROFILE%"=="LEAN" if /I not "%BUILD_PROFILE%"=="FULL_AI" (
    echo [WARN] Unknown profile "%BUILD_PROFILE%". Falling back to LEAN.
    set BUILD_PROFILE=LEAN
)

:: Define Log File
if not exist "%ROOT_DIR%\logs" mkdir "%ROOT_DIR%\logs"
set "LOG_FILE=%ROOT_DIR%\logs\[%TARGET_VERSION%] [SETUP LOG] [%YYYY%.%MM%.%DD%_%HH%.%Min%.%SS%].txt"

echo.
echo [INFO] Logging started...
echo.

:: --- 3. EXECUTE BUILD (Redirected to Log) ---
call :BUILD_PROCESS > "%LOG_FILE%" 2>&1
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo [FATAL] Build failed with code %RC%.
    echo [LOGS] Log saved to: %LOG_FILE%
    echo [Check the log file for details]
    pause
    exit /b %RC%
)

:: --- 4. FINISH ---
echo.
echo [DONE] Process finished successfully. 
echo [LOGS] Log saved to: %LOG_FILE%
echo [BACKUP] Build artifacts moved to .bak\
pause
exit /b

:: ==========================================================
::               BUILD LOGIC FUNCTION
:: ==========================================================
:BUILD_PROCESS
    echo ==========================================================
    echo      TRADINGBOT BUILD LOG
    echo      Version: %TARGET_VERSION%
    echo      Date: %YYYY%-%MM%-%DD% %HH%:%Min%:%SS%
    echo      Profile: %BUILD_PROFILE%
    echo ==========================================================
    echo.
    
    echo [INFO] Root Directory: %ROOT_DIR%
    
    echo.
    echo [STEP 1] Checking Python...
    python --version
    if errorlevel 1 (
        echo [CRITICAL ERROR] Python is not found!
        exit /b 1
    )

    echo.
    echo [STEP 2] Installing Core Libraries...
    python -m pip install --upgrade "setuptools<82" wheel
    REM Ensure pkg_resources exists (PyInstaller/altgraph dependency)
    python -c "import pkg_resources" >nul 2>&1
    if errorlevel 1 (
        echo [WARN] pkg_resources missing; pinning setuptools to 81.0.0...
        python -m pip install --upgrade --force-reinstall "setuptools==81.0.0"
    )
    if /I "%BUILD_PROFILE%"=="FULL_AI" (
        echo [INFO] Installing FULL_AI dependency set...
        python -m pip install customtkinter alpaca-trade-api pandas_ta pandas requests pyinstaller matplotlib numpy vaderSentiment yfinance pytest scikit-learn scipy
    ) else (
        echo [INFO] Installing LEAN dependency set...
        python -m pip install customtkinter alpaca-trade-api pandas_ta pandas requests pyinstaller matplotlib numpy vaderSentiment yfinance pytest
    )
    python -m pip install jaraco.text jaraco.classes jaraco.context platformdirs packaging more-itertools

    echo.
    echo [STEP 2b] Running Smoke Tests...
    call "%ROOT_DIR%\run_tests.bat"
    if errorlevel 1 (
        echo.
        echo [FATAL ERROR] Test suite failed. Aborting build.
        exit /b 1
    )

    echo.
    if /I "%BUILD_PROFILE%"=="FULL_AI" (
        echo [STEP 2c] FULL_AI profile: enabling sklearn/scipy runtime bundle (no TBB redist).
        set "PYI_AI_COLLECT=--collect-all sklearn --collect-all scipy"
    ) else (
        echo [STEP 2c] LEAN profile: skipping Intel TBB redist and native AI bundles.
        set "PYI_AI_COLLECT="
    )
    set "BINARY_CMD="

    echo.
    echo [STEP 4] Building Application...

    echo [INFO] Strategy: Flat Modules + %BUILD_PROFILE% profile
    echo.

    REM PYINSTALLER COMMAND
    
    python -m PyInstaller ^
        --noconsole ^
        --onefile ^
        --name "TradingBot [%TARGET_VERSION%]" ^
        --distpath "%ROOT_DIR%" ^
        --workpath "%ROOT_DIR%\build\work" ^
        --specpath "%ROOT_DIR%\build" ^
        --clean ^
        --paths "%ROOT_DIR%" ^
        --add-data "%ROOT_DIR%\modules;modules" ^
        !BINARY_CMD! ^
        --collect-all "customtkinter" ^
        --collect-all "vaderSentiment" ^
        !PYI_AI_COLLECT! ^
        --hidden-import "jaraco.text" ^
        --hidden-import "matplotlib" ^
        --hidden-import "modules.tabs.dashboard" ^
        --hidden-import "modules.tabs.inspector" ^
        --hidden-import "modules.tabs.architect" ^
        --hidden-import "modules.tabs.config" ^
        --hidden-import "modules.tabs.candidates" ^
        --hidden-import "modules.popups" ^
        --exclude-module "torch" ^
        --exclude-module "caffe2" ^
        --exclude-module "sklearn.externals.array_api_compat.torch" ^
        --exclude-module "scipy._lib.array_api_compat.torch" ^
        "%ROOT_DIR%\main.py"

    if errorlevel 1 (
        echo.
        echo [FATAL ERROR] PyInstaller Failed.
        exit /b 1
    )

    echo.
    echo [STEP 5] Backup Build Artifacts...
    set "BACKUP_DIR=%ROOT_DIR%\.bak\[%YYYY%.%MM%.%DD%_%HH%.%Min%.%SS%]"
    if not exist "!BACKUP_DIR!" mkdir "!BACKUP_DIR!"
    
    echo [INFO] Moving build folder to: !BACKUP_DIR!
    if exist "%ROOT_DIR%\build" (
        xcopy "%ROOT_DIR%\build" "!BACKUP_DIR!\build" /E /I /H /Y >nul
        rmdir /s /q "%ROOT_DIR%\build"
    )
    
    if exist "%ROOT_DIR%\*.spec" (
        move "%ROOT_DIR%\*.spec" "!BACKUP_DIR!" >nul
    )

    echo.
    echo [STEP 6] Post-build validation...
    if not exist "%ROOT_DIR%\TradingBot [%TARGET_VERSION%].exe" (
        echo [FATAL ERROR] Expected executable missing: TradingBot [%TARGET_VERSION%].exe
        exit /b 1
    )
    if not exist "%ROOT_DIR%\config\config.ini" (
        echo [FATAL ERROR] Expected runtime config missing: config\config.ini
        exit /b 1
    )
    echo [OK] Post-build validation passed.

    echo.
    echo ==========================================================
    echo [SUCCESS] Build Complete.
    echo ==========================================================
    exit /b 0
