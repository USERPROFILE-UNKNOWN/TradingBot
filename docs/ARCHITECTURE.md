# TradingBot Architecture (v5.17.0 baseline)

This document captures the current production architecture before the v5.17.1 runbook + incident checklist release.

## Runtime entrypoints

- `main.py`: application startup, strict config validation, and UI bootstrap.
- `modules/engine/`: execution runtime (core engine, risk, order lifecycle, broker gateway).
- `modules/tabs/`: UI surfaces (dashboard, config, architect, candidates, inspector).

## Data and storage

- Split DB mode is the default (`db_mode = SPLIT`).
- Current DB files are scoped by domain under `db/`:
  - active trades
  - trade history
  - historical prices
  - metrics
  - decisions
  - backtests

## Configuration model

- Primary user-facing values: `config/config.ini` `[CONFIGURATION]`.
- Secrets and credentials are separated into dedicated files (`keys.ini`, etc.).
- Canonical defaults are declared in `modules/config_defaults.py` and repaired through `modules/config_io.py`.

## Governance and safety envelope

- Agent and governance knobs are centralized in config.
- Forward-only behavior is preferred over compatibility shims.
- Live-impact controls are bounded by exposure and confirmation settings.

## Observability

- Runtime logs are written under `logs/`.
- Health and metrics paths are implemented by `modules/metrics.py` and related startup observability tests.

## v5.17.1 readiness intent

v5.17.1 is intended to add operational documentation and incident handling guardrails without changing trading behavior.
