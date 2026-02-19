import sys
import os
import traceback

print(f"python version: {sys.version}")
print(f"working dir: {os.getcwd()}")
try:
    from modules.app_constants import APP_VERSION
    print(f"app version: {APP_VERSION}")
except Exception:
    pass

print("\n--- STEP 1: CHECKING DEPENDENCIES ---")
libs = ['pandas', 'numpy', 'customtkinter', 'alpaca_trade_api', 'vaderSentiment']
for lib in libs:
    try:
        __import__(lib)
        print(f"✅ {lib}: Found")
    except ImportError as e:
        print(f"❌ {lib}: MISSING! ({e})")

print("\n--- STEP 2: CHECKING MODULES ---")

def try_import(module_name):
    print(f"Attempting to import {module_name}...")
    try:
        __import__(module_name)
        print(f"✅ {module_name}: Success")
    except Exception:
        print(f"🔥 CRASH in {module_name}:")
        traceback.print_exc()

try_import('modules.ai')
try_import('modules.sentiment')
try_import('modules.architect')
try_import('modules.strategies')
try_import('modules.engine')
try_import('modules.ui')

print("\n--- DIAGNOSTIC COMPLETE ---")
input("Press Enter to exit...")