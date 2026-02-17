# TradingBot Runbook (v5.17.1 target)

This runbook defines routine operational procedures for paper/live guarded operation.

## 1) Pre-start checks

1. Validate Python and dependency availability.
2. Run test suite using `run_tests.bat` (or `python -m pytest -q -ra modules/tests`).
3. Confirm config files exist and parse correctly (`config/config.ini`, `strategy.ini`, `watchlist.ini`).
4. Verify DB folder and required DB files are present under `db/`.

## 2) Startup procedure

1. Launch from repository root.
2. Observe startup logs for strict validation pass and database mode.
3. Confirm no immediate broker/auth errors if live integrations are enabled.

## 3) Health checks during runtime

- Confirm periodic snapshot logs appear at configured interval.
- Confirm candidate scanning cadence is active when enabled.
- Watch for risk halts and reconcile mismatch logs.

## 4) Safe maintenance

- Run `cleanup_remove_legacy_migrations.bat` after upgrade windows.
- Keep legacy shims removed (forward-only posture).
- Rotate/collect logs from `logs/` and keep summaries under `logs/summaries/`.

## 5) Upgrade checklist (minor release)

1. Confirm `modules/app_constants.py` version and release string.
2. Confirm `config/config.ini` layout remains organized (no auto-added catch-all section).
3. Confirm `modules/config_defaults.py` and `modules/config_io.py` reflect any new config values.
4. Run full test suite and capture output in release notes.

## 6) Rollback plan

- Restore prior executable/config backup set.
- Restore DB snapshots if schema/data migration occurred.
- Re-run tests and startup validation before resuming trading.
