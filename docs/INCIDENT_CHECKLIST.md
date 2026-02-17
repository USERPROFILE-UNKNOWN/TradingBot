# TradingBot Incident Checklist (v5.17.1 target)

Use this checklist for any runtime anomaly, broker/API instability, or data integrity concern.

## A. Immediate containment

- [ ] Set agent mode to `OFF` or reduce to non-live mode as appropriate.
- [ ] Halt new entries if reconcile mismatch or risk anomaly is present.
- [ ] Preserve current logs and do not overwrite incident artifacts.

## B. Classify incident

- [ ] **Execution incident** (order rejects, slippage spike, stuck lifecycle).
- [ ] **Data incident** (missing bars, invalid OHLC, stale symbols).
- [ ] **Config incident** (invalid/missing keys, malformed config).
- [ ] **Infra incident** (filesystem, DB locks, startup failure).

## C. Evidence capture

- [ ] Collect runtime log file(s) from `logs/`.
- [ ] Capture latest summaries from `logs/summaries/`.
- [ ] Record active config snapshot (`config/config.ini`, strategy/watchlist as relevant).
- [ ] Record time window and affected symbols/accounts.

## D. Verification and triage

- [ ] Run test suite (`python -m pytest -q -ra modules/tests`).
- [ ] Validate config integrity and required sections.
- [ ] Verify DB readability and table-level access for impacted domains.
- [ ] Reproduce issue in paper mode when possible.

## E. Recovery

- [ ] Apply minimal fix with clear rollback.
- [ ] Re-test impacted subsystem.
- [ ] Resume in paper/guarded mode first.
- [ ] Re-enable live actions only after stability confirmation.

## F. Post-incident

- [ ] Write incident summary (trigger, blast radius, fix, prevention).
- [ ] Add/adjust regression tests.
- [ ] Update runbook/checklist if any procedural gap was discovered.
