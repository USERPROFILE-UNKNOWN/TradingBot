@echo off
setlocal enableextensions enabledelayedexpansion

REM ============================================================================
REM TradingBot v6.22.1
REM Forward-only cleanup: remove legacy/compat artifacts + Python bytecode caches
REM ============================================================================
REM Safe to run multiple times.
REM Run from TradingBot\ (same folder as main.py).
REM ============================================================================

cd /d "%~dp0"

echo.
echo [CLEANUP] Removing legacy artifacts...
echo.

REM --- Legacy migration script (root) ---
if exist "migrate_split_db.py" (
  echo   - Deleting migrate_split_db.py
  del /f /q "migrate_split_db.py" >nul 2>&1
) else (
  echo   - migrate_split_db.py not found (OK)
)

REM --- Legacy compatibility shims (tabs_*.py) ---
for %%F in (
  "modules\tabs_architect.py"
  "modules\tabs_candidates.py"
  "modules\tabs_config.py"
  "modules\tabs_dashboard.py"
  "modules\tabs_inspector.py"
) do (
  if exist %%F (
    echo   - Deleting %%~nxF
    del /f /q %%F >nul 2>&1
  ) else (
    echo   - %%~nxF not found (OK)
  )
)

REM --- Legacy engine compatibility shim (name collision with modules\engine\ package) ---
if exist "modules\engine.py" (
  echo   - Deleting modules\engine.py
  del /f /q "modules\engine.py" >nul 2>&1
) else (
  echo   - modules\engine.py not found (OK)
)

REM --- Scaffold-only engine submodules (unused) ---
for %%F in (
  "modules\engine\e4.py"
  "modules\engine\e5.py"
  "modules\engine\orders.py"
  "modules\engine\summary.py"
  "modules\engine\sync.py"
) do (
  if exist %%F (
    echo   - Deleting %%~nxF
    del /f /q %%F >nul 2>&1
  ) else (
    echo   - %%~nxF not found (OK)
  )
)

REM --- Scaffold-only tabs (unused) ---
if exist "modules\tabs\backtest_lab.py" (
  echo   - Deleting modules\tabs\backtest_lab.py
  del /f /q "modules\tabs\backtest_lab.py" >nul 2>&1
) else (
  echo   - modules\tabs\backtest_lab.py not found (OK)
)

echo.
echo [CLEANUP] Removing Python bytecode caches (*.pyc, __pycache__)...
echo.

REM Delete *.pyc files
for /r %%F in (*.pyc) do (
  del /f /q "%%F" >nul 2>&1
)

REM Delete __pycache__ directories
for /d /r %%D in (__pycache__) do (
  rd /s /q "%%D" >nul 2>&1
)

echo.
echo [CLEANUP] Done.
endlocal
