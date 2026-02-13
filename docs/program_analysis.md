# TradingBot Program Analysis and Improvement Plan

## What the program currently does well

- **Clear module boundaries for major concerns** (engine, database, market data, UI, research).
- **Forward-only split-database strategy** with one SQLite file per logical domain, which is a good base for isolation and recoverability.
- **Defensive config loading** that attempts sanitization and logs repairs at startup.
- **Runtime tuning knobs** in configuration for logging, execution, and safety behavior.

## Key risks and bottlenecks

### 1) Monolithic core components

- `modules/engine/core.py` carries a very large surface area: API connectivity, strategy execution, risk controls, order lifecycle, logging policy, and telemetry.
- `modules/database.py` combines connection lifecycle, schema management, migration repair behavior, and all persistence APIs in one class.

**Why this matters:** hard-to-test behavior, fragile changes, and slower onboarding.

### 2) Error-handling consistency

- There are many broad exception handlers (`except Exception:`) and direct `print(...)` calls across startup and DB/UI paths.
- In practice this can hide failures, create silent data-loss risks, and make incident diagnosis harder.

### 3) Missing quality gates

- The repository currently has no visible automated tests, no static analysis configuration, and minimal project documentation.

**Why this matters:** regressions in trade/risk logic become easier to introduce and harder to detect.

### 4) Config and secrets safety

- Runtime behavior is heavily config-driven, but there is limited schema validation and no strict “fail-fast” mode.
- API-key and environment practices are not documented in the repository.

### 5) Observability and operations readiness

- Logging is partially structured but spread across print-style and callback-style sinks.
- There is no documented health-check profile for data freshness, API availability, or DB write latency.

## Priority roadmap (high impact first)

### Phase 1 (1–2 weeks): Reliability hardening

1. **Introduce centralized logging adapter**
   - Replace ad hoc `print` + broad except paths with consistent `logging` usage and severity levels.
   - Add structured fields for symbol, order_id, strategy, and component.

2. **Add config schema validation**
   - Validate required keys and allowed ranges at startup.
   - Add strict mode for production runs (abort if unsafe/missing values).

3. **Create smoke tests for startup-critical paths**
   - Config load/sanitize, DB split initialization, and “engine boot without broker credentials.”

### Phase 2 (2–4 weeks): Testability and maintainability

1. **Refactor trading engine into services**
   - Extract `BrokerGateway`, `RiskManager`, `OrderLifecycleService`, and `ScanCoordinator` from `TradingEngine`.

2. **Refactor data manager into repositories**
   - Split schema/init concerns from query/write concerns.
   - Introduce typed DTOs for DB I/O boundaries.

3. **Add deterministic tests**
   - Unit tests for risk rules and order-state transitions.
   - Integration tests for SQLite repositories using temp db directories.

### Phase 3 (4+ weeks): Performance and product quality

1. **Profile scan loop and DB hot paths**
   - Measure candidate scan duration, order submit-to-confirm latency, and DB contention.

2. **Define SLOs and health metrics**
   - Time-to-fresh-bar, order failure ratio, and decision-to-execution delay.

3. **Improve developer docs**
   - Architecture map, runbook, and incident triage checklist.

## Concrete low-risk changes to start immediately

- Add `tests/` with at least:
  - config sanitizer test
  - DB split-path resolution test
  - order TTL cancellation policy test
- Add `pyproject.toml` or equivalent for:
  - formatter + linter + type checker settings
- Add a `logging` module wrapper and migrate startup + DB first.

## Suggested target architecture

- `engine/`
  - `coordinator.py` (orchestrates cycle)
  - `broker_gateway.py`
  - `risk_manager.py`
  - `order_lifecycle.py`
- `persistence/`
  - `db_runtime.py` (connections, pragmas)
  - `repositories/*.py`
- `config/`
  - `schema.py`
  - `loader.py`
- `observability/`
  - `logger.py`
  - `metrics.py`

This preserves current behavior while making failures easier to see and logic easier to validate.
