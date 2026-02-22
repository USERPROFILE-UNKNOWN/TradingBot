# Current program breakdown (v5.20.0)

## 1) What the program is

TradingBot is a desktop (CustomTkinter) trading workstation that combines:

- market-data maintenance and scanning,
- rule/AI-assisted decisioning,
- broker execution via Alpaca,
- governance/agent automation,
- and operational telemetry/reporting.

The startup entrypoint is `main.py`, which loads config, validates it, initializes split SQLite databases, and then launches the UI app (`TradingApp`).

## 2) Runtime architecture at a glance

### Entrypoints and orchestration

- `main.py`
  - resolves app paths/log directories,
  - ensures split config files are present,
  - loads merged runtime config,
  - performs optional strict runtime config validation,
  - initializes `DataManager`,
  - launches `TradingApp`.
- `TradingApp` (`modules/ui.py`) wires up the UI tabs, creates a `TradingEngine` and `AgentMaster`, then allows the operator to start/stop the engine from the UI.

### Core engine (`TradingEngine`)

`TradingEngine` (implemented in `modules/engine/core_impl.py`, re-exported as `modules/engine/core.py`) owns the active scan/execute loop:

- connects to broker,
- starts AI training thread,
- on start emits session telemetry and syncs positions,
- loops while active:
  - kill-switch and watchdog checks,
  - telegram command checks,
  - market regime checks,
  - reconciliation checks,
  - governance checkpoint through `AgentMaster`,
  - market scan and confirmation processing,
  - order lifecycle management,
  - active-position management,
  - periodic snapshot logging,
  - sleeps according to `update_interval_sec` while still handling periodic subtasks.

### Agent layer (`AgentMaster` + `Governance`)

`AgentMaster` runs autonomous maintenance/research/guardrail jobs using an internal scheduler. It supports `OFF`, `ADVISORY`, `PAPER`, and `LIVE` modes.

- In `OFF` and `ADVISORY`, actions are bypassed/non-blocking.
- In `PAPER`/`LIVE`, actions are evaluated by `Governance`.
- In `LIVE`, additional daily limits apply for config tuning/promotion actions.
- Optional scheduled jobs include:
  - DB integrity checks,
  - stale symbol quarantine,
  - auto backfill,
  - daily reporting,
  - research sweep,
  - canary guardrails,
  - candidate validation pump.

`Governance` enforces scope validity, live-change limits, promotion/config-tuning toggles, and risk limits (`agent_live_max_exposure_pct`, canary caps, challenger underperformance guard).

### Data layer (`DataManager` + repositories)

The data layer is split-DB-first:

- `DataManager` resolves `db_dir`, initializes/validates split DB files, sets up runtime connections and locks, and ensures schema.
- Repositories (`TradesRepo`, `HistoryRepo`, `DecisionsRepo`, `BacktestRepo`, `CandidatesRepo`, `AgentRepo`) provide the persistence surface under the manager façade.
- Expected DB domains include historical prices, active trades, trade history, decision logs, and backtest results (plus metrics/agent stores used by runtime modules).

### Configuration model

- Canonical defaults live in `modules/config_defaults.py`.
- Split config layout and load/write logic are in `modules/config_io.py` (used via `modules/utils.py`).
- User-operational knobs live primarily in `config/config.ini` (`[CONFIGURATION]`).
- Secrets/credentials live in `config/keys.ini` (`[KEYS]`).

### Observability and operations

- Runtime logging is configured early in startup and written under `logs/`.
- Run summaries are written under `logs/summaries` when enabled.
- The repository includes a runbook and architecture snapshot docs (`docs/RUNBOOK.md`, `docs/ARCHITECTURE.md`).

## 3) Execution behavior and control plane

### How execution starts/stops today

- The app starts the UI first.
- The engine is started from the **START ENGINE** button (`toggle_engine`): this creates a daemon thread that runs `TradingEngine.run()`.
- Stopping toggles to `engine.stop()`.

### Important control knobs for behavior

High-impact config keys include:

- `paper_trading` (paper vs live credential environment),
- `agent_mode` (`OFF`/`ADVISORY`/`PAPER`/`LIVE`),
- risk/exposure limits (`max_positions`, `max_percent_per_stock`, `max_daily_loss`, `agent_live_max_exposure_pct`),
- loop cadence (`update_interval_sec`),
- reconciliation and watchdog controls,
- agent automation toggles (auto backfill, research, canary, etc.),
- strict startup validation (`strict_config_validation`).

## 4) Full-auto mode setup (practical)

This section describes a **full-auto PAPER mode** (recommended first), then what changes for LIVE.

### Prerequisites

1. Python 3.12+ environment (per `pyproject.toml`).
2. Dependencies installed (at minimum modules imported by `main.py`/engine/UI).
3. Split config files present in `config/` (`config.ini`, `keys.ini`, `strategy.ini`, `watchlist.ini`).
4. `db/` exists and is writable.

### Step A — configure credentials safely

Edit `config/keys.ini`:

- For paper: set `paper_alpaca_key` and `paper_alpaca_secret`.
- For live: set `live_alpaca_key` and `live_alpaca_secret`.
- Keep `paper_base_url` / `live_base_url` as default unless you have a reason to change.

### Step B — set autonomous PAPER mode in `config/config.ini`

Recommended baseline:

- `paper_trading = True`
- `agent_mode = PAPER`
- keep risk conservative:
  - `max_positions = 3` to `5`
  - `max_percent_per_stock <= 0.20`
  - `max_daily_loss` tightly bounded
  - `agent_live_max_exposure_pct` conservative (still used by governance logic)
- keep safeguards enabled:
  - `reconcile_halt_on_mismatch = True`
  - `agent_canary_enabled = True`
  - `agent_hard_halt_supreme = True`
  - `agent_db_integrity_check_enabled = True`
  - `agent_stale_quarantine_enabled = True`
- ensure automation toggles are ON as desired:
  - `agent_auto_backfill_enabled = True`
  - `agent_daily_report_enabled = True`
  - `agent_research_automation_enabled = True`

### Step C — optional strict startup gate (recommended)

Set:

- `strict_config_validation = True`

This forces startup to fail fast on invalid/missing required values instead of continuing with warnings.

### Step D — launch and run

1. Start app from repo root: `python main.py`.
2. Confirm startup logs show config/DB init success.
3. In UI, click **START ENGINE**.
4. Leave app running; the engine loop + agent scheduler now run continuously.

### Step E — verify that full auto is actually active

Check runtime logs for:

- engine started banner,
- recurring scan/snapshot activity,
- periodic maintenance/research log lines,
- no repeated broker auth failures,
- no reconcile hard-halt loops.

### Moving from full-auto PAPER to full-auto LIVE

Only after stable PAPER burn-in:

1. Set `paper_trading = False`.
2. Populate live credentials in `keys.ini`.
3. Set `agent_mode = LIVE`.
4. Tighten canary and exposure guardrails before first run:
   - very low `agent_canary_exposure_pct`,
   - low `agent_max_live_changes_per_day`,
   - keep rollback mode to `PAPER`.
5. Monitor closely on first LIVE sessions.

## 5) Current limitations to be aware of

- The process is UI-hosted (no dedicated headless daemon entrypoint in this repo), so “full auto” currently means “engine + agent running continuously inside the desktop app.”
- Engine run requires broker API connectivity/credentials; if broker init fails, run loop exits early.
- Production safety depends heavily on config quality and conservative limits.
