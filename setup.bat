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
echo      TRADINGBOT SETUP MANAGER (v5.16.2)
echo ==========================================================
echo.
set /p TARGET_VERSION="Enter Version (e.g. v5.16.2): "
if "%TARGET_VERSION%"=="" set TARGET_VERSION=v5.16.2

:: Define Log File
if not exist "%ROOT_DIR%\logs\_setup" mkdir "%ROOT_DIR%\logs\_setup"
set "LOG_FILE=%ROOT_DIR%\logs\_setup\[%TARGET_VERSION%] [SETUP LOG] [%YYYY%.%MM%.%DD%_%HH%.%Min%.%SS%].txt"

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
    python -m pip install customtkinter alpaca-trade-api pandas_ta pandas requests pyinstaller matplotlib numpy vaderSentiment scikit-learn scipy yfinance pytest
    python -m pip install jaraco.text jaraco.classes jaraco.context platformdirs packaging more-itertools

    echo.
    echo [STEP 2b] Running Smoke Tests...
    call "%ROOT_DIR%\bat\run_tests.bat"
    if errorlevel 1 (
        echo.
        echo [FATAL ERROR] Test suite failed. Aborting build.
        exit /b 1
    )

    echo.
    echo [STEP 2c] Downloading Intel TBB Redist (Windows x64)...
    
    REM v6.23.1 FIX: Force .zip extension so Expand-Archive accepts it
    set "TBB_VER=2022.3.0.380"
    set "TBB_URL=https://www.nuget.org/api/v2/package/inteltbb.redist.win/%TBB_VER%"
    set "DEPS_DIR=%ROOT_DIR%\build\deps"
    set "TBB_ZIP=%DEPS_DIR%\tbb_redist.zip"
    set "TBB_EXTRACT=%DEPS_DIR%\tbb_extracted"
    
    if not exist "%DEPS_DIR%" mkdir "%DEPS_DIR%"
    
    REM Download and Extract using PowerShell (avoid wildcard parsing on [] paths)
    echo [INFO] Downloading TBB %TBB_VER% to %TBB_ZIP%...
    powershell -NoProfile -Command ^
        "$ErrorActionPreference='Stop';" ^
        "$zipPath=[System.IO.Path]::GetFullPath('%TBB_ZIP%');" ^
        "$extractPath=[System.IO.Path]::GetFullPath('%TBB_EXTRACT%');" ^
        "$client=New-Object System.Net.WebClient;" ^
        "$client.DownloadFile('%TBB_URL%',$zipPath);" ^
        "if(!(Test-Path -LiteralPath $zipPath)){throw 'TBB redist download failed: zip not found.'};" ^
        "if(Test-Path -LiteralPath $extractPath){Remove-Item -Recurse -Force -LiteralPath $extractPath};" ^
        "Expand-Archive -LiteralPath $zipPath -DestinationPath $extractPath -Force;"

    if errorlevel 1 (
        echo [FATAL] Intel TBB redist download/extract failed.
        exit /b 1
    )

    if not exist "%TBB_ZIP%" (
        echo [FATAL] Intel TBB redist zip missing after download: %TBB_ZIP%
        exit /b 1
    )
        
    echo.
    echo [STEP 3] Hunting Dependencies...
    echo [INFO] Locating tbb12.dll from extracted redist...
    set "TBB_DLL="

    REM 1) Try the common NuGet layout first:
    set "TBB_CANDIDATE=%TBB_EXTRACT%\runtimes\win-x64\native\tbb12.dll"
    if exist "%TBB_CANDIDATE%" set "TBB_DLL=%TBB_CANDIDATE%"

    REM 2) Fallback: search anywhere under extraction, prefer win-x64 paths
    if not defined TBB_DLL (
      for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command ^
        "Get-ChildItem -Path '%TBB_EXTRACT%' -Recurse -Filter tbb12.dll | Sort-Object @{Expression={$_.FullName -notlike '*win-x64*'}}, FullName | Select-Object -First 1 -ExpandProperty FullName"`) do set "TBB_DLL=%%I"
    )

    if defined TBB_DLL if exist "!TBB_DLL!" (
        echo [OK] Found TBB DLL: !TBB_DLL!
        for %%D in ("!TBB_DLL!") do set "TBB_BIN=%%~dpD"
        
        REM Help PyInstaller resolve dependency at analysis time
        set "PATH=!TBB_BIN!;%PATH%"
        
        REM Bundle into exe root
        set "BINARY_CMD=--add-binary ^"!TBB_DLL!^";."
    ) else (
        echo [FATAL] Could not find tbb12.dll in NuGet redist extraction.
        echo [DEBUG] Looked in: %TBB_EXTRACT%
        set "BINARY_CMD="
        exit /b 1
    )

    echo.
    echo [STEP 4] Building Application...
    echo [INFO] Strategy: Flat Modules + TBB Bundle (Redist) + FULL SciPy Collection
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
        --collect-all "sklearn" ^
        --collect-all "scipy" ^
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
        "%ROOT_DIR%\modules\main.py"

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
    echo ==========================================================
    echo [SUCCESS] Build Complete.
    echo ==========================================================
    exit /b 0
