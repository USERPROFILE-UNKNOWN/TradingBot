import configparser
import json

from modules.agent_shadow import AgentShadow


class _DummyDB:
    def __init__(self):
        self.suggestions = []
        self.rationales = []

    def upsert_agent_suggestion(self, **kwargs):
        self.suggestions.append(dict(kwargs))
        return len(self.suggestions)

    def add_agent_rationale(self, suggestion_id, rationale_text, severity="INFO"):
        self.rationales.append((suggestion_id, rationale_text, severity))

    def get_config_value(self, _key):
        return "30"

    def get_agent_shadow_checkpoint(self, scope):
        return getattr(self, "_cp", {}).get(scope)

    def upsert_agent_shadow_checkpoint(self, scope, artifact_path, last_mtime, last_hash):
        if not hasattr(self, "_cp"):
            self._cp = {}
        self._cp[scope] = {
            "scope": scope,
            "artifact_path": artifact_path,
            "last_mtime": last_mtime,
            "last_hash": last_hash,
        }


def _base_cfg(mode="PAPER"):
    c = configparser.ConfigParser()
    c["CONFIGURATION"] = {
        "agent_mode": mode,
        "agent_shadow_include_summaries": "True",
        "agent_shadow_include_backtests": "True",
    }
    return c


def test_shadow_emits_ready_to_apply_proposal_from_summary(tmp_path):
    logs = tmp_path / "logs"
    summaries = logs / "summaries"
    summaries.mkdir(parents=True)

    data = {
        "data_freshness": {"stale_count": 2, "stale_symbols": ["AAPL", "MSFT"]},
        "ai_metrics": {"model_status": "not_enough_data", "roc_auc": 0.5},
    }
    p = summaries / "summary_test.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    db = _DummyDB()
    shadow = AgentShadow(_base_cfg(mode="PAPER"), db, {"logs": str(logs)})
    shadow._scan_summaries(str(summaries))

    assert db.suggestions
    payload = db.suggestions[0]["suggestion_payload"]
    assert payload["kind"] == "proposal"
    assert "target" in payload and "proposed_diff" in payload
    assert "evidence_links" in payload and payload["evidence_links"]
    assert "confidence_score" in payload
    assert payload["required_approval_level"] == "AUTO"


def test_shadow_live_mode_requires_approval_on_proposals(tmp_path):
    logs = tmp_path / "logs"
    bt = logs / "backtest"
    bt.mkdir(parents=True)

    bundle = {
        "schema": {"name": "TradingBot.architect_backtest_bundle", "version": 1},
        "aggregates": {
            "per_variant": {
                "ARCH001": {"profit_sum": 100.0, "symbols": 3, "symbols_positive": 2, "profit_mean": 33.3}
            }
        },
    }
    p = bt / "architect_backtest_bundle_test.json"
    p.write_text(json.dumps(bundle), encoding="utf-8")

    db = _DummyDB()
    shadow = AgentShadow(_base_cfg(mode="LIVE"), db, {"logs": str(logs)})
    shadow._scan_backtests(str(bt))

    assert db.suggestions
    payload = db.suggestions[0]["suggestion_payload"]
    assert payload["kind"] == "proposal"
    assert payload["required_approval_level"] == "REQUIRE_APPROVAL"
    assert payload["risk_impact_estimate"] in {"LOW", "MEDIUM", "HIGH"}


def test_shadow_checkpoint_skips_reprocessing_when_unchanged(tmp_path):
    logs = tmp_path / "logs"
    summaries = logs / "summaries"
    summaries.mkdir(parents=True)

    data = {"data_freshness": {"stale_count": 1, "stale_symbols": ["AAPL"]}}
    p = summaries / "summary_checkpoint.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    db = _DummyDB()
    shadow = AgentShadow(_base_cfg(mode="PAPER"), db, {"logs": str(logs)})

    shadow._scan_once()
    first_count = len(db.suggestions)
    assert first_count >= 1

    shadow._scan_once()
    second_count = len(db.suggestions)
    assert second_count == first_count


def test_shadow_emits_repeated_data_quality_failure_proposals(tmp_path):
    logs = tmp_path / "logs"
    summaries = logs / "summaries"
    summaries.mkdir(parents=True)

    s1 = {
        "data_freshness": {"stale_count": 12, "stale_symbols": ["AAPL"]},
        "crypto": {"stable_set_count": 0, "selected": []},
        "data_quality": {"nan_count": 4},
    }
    s2 = {
        "data_freshness": {"stale_count": 15, "stale_symbols": ["MSFT"]},
        "crypto": {"stable_set_count": 0, "selected": []},
        "data_quality": {"missing_bars": 3},
    }
    (summaries / "summary_rep_1.json").write_text(json.dumps(s1), encoding="utf-8")
    (summaries / "summary_rep_2.json").write_text(json.dumps(s2), encoding="utf-8")

    db = _DummyDB()
    shadow = AgentShadow(_base_cfg(mode="PAPER"), db, {"logs": str(logs)})
    shadow._scan_summaries(str(summaries))

    titles = [str(x.get("title") or "") for x in db.suggestions]
    assert any("empty crypto stable-set" in t.lower() for t in titles)
    assert any("mass quarantine" in t.lower() for t in titles)
    assert any("missing-bars" in t.lower() or "missing bars" in t.lower() for t in titles)
