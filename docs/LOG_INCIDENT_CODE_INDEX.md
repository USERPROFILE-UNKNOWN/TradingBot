# Log Incident Code Index (v5.14.2)

This file documents the **incident codes** used in TradingBot logs. Incident codes are stable identifiers (prefixed with `E_`) intended to make troubleshooting and alerting consistent.

| Code | Meaning |
|---|---|
| `E_STARTUP_SANITIZER_READ_FAIL` | Failed reading the config sanitizer log (non-fatal). |
| `E_STARTUP_SANITIZER_WRITE_FAIL` | Failed writing the config sanitizer log (non-fatal). |
| `E_CFG_STRICT_VALIDATION_FAILED` | Strict-mode config validation failed; launch is blocked (expected behavior). |
| `E_DB_SPLIT_CHECK_FAIL` | Error while checking/validating split DB layout at startup. |
| `E_DB_SPLIT_MISSING` | Split DB mode is enabled but one or more expected DB files are missing (auto-created). |
| `E_DB_BACKTEST_SCHEMA_REPAIRED` | Malformed `backtest_results.db` schema detected and automatically repaired. |
| `E_DB_BACKTEST_SCHEMA_REPAIR_FAIL` | Failed attempting to backup/repair malformed `backtest_results.db` (best-effort). |
| `E_DB_ENSURE_INDEXES_FAIL` | DB index creation failed (best-effort; app continues). |
| `E_ENGINE_API_CONNECT_FAIL` | Alpaca API connection failed during engine initialization/boot. |
| `E_ENGINE_CANCEL_PENDING_FAIL` | Engine attempted to cancel pending orders but the operation failed. |

> Note: Many subsystems also emit throttled event identifiers (also `E_*`) via `modules/log_throttle.py`. Those are treated equivalently as incident/event codes and may be expanded in future versions.
