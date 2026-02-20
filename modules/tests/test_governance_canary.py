import configparser

from modules.governance import Governance


def _cfg() -> configparser.ConfigParser:
    c = configparser.ConfigParser()
    c["CONFIGURATION"] = {
        "agent_max_live_changes_per_day": "8",
        "agent_live_max_exposure_pct": "0.30",
        "agent_canary_enabled": "True",
        "agent_canary_exposure_pct": "0.10",
        "agent_canary_underperform_pct_max": "5.0",
        "agent_promotion_enabled": "True",
        "agent_config_tuning_enabled": "True",
    }
    return c


def test_governance_canary_caps_live_exposure():
    gov = Governance(_cfg())
    ok, _ = gov.approve_action({"scope": "LIVE", "type": "SIZE_CHANGE", "proposed_exposure_pct": 0.11})
    assert ok is False


def test_governance_canary_allows_safe_exposure():
    gov = Governance(_cfg())
    ok, _ = gov.approve_action({"scope": "LIVE", "type": "SIZE_CHANGE", "proposed_exposure_pct": 0.08})
    assert ok is True


def test_governance_canary_blocks_underperforming_challenger():
    gov = Governance(_cfg())
    ok, _ = gov.approve_action(
        {
            "scope": "LIVE",
            "type": "DEPLOY_STRATEGY",
            "proposed_exposure_pct": 0.05,
            "champion_delta_pct": -7.0,
        }
    )
    assert ok is False


def test_governance_blocks_live_promotion_when_disabled():
    c = _cfg()
    c["CONFIGURATION"]["agent_promotion_enabled"] = "False"
    gov = Governance(c)
    ok, reason = gov.approve_action({"scope": "LIVE", "type": "DEPLOY_STRATEGY", "proposed_exposure_pct": 0.05})
    assert ok is False
    assert "Promotion disabled" in reason


def test_governance_blocks_live_config_tuning_when_disabled():
    c = _cfg()
    c["CONFIGURATION"]["agent_config_tuning_enabled"] = "False"
    gov = Governance(c)
    ok, reason = gov.approve_action({"scope": "LIVE", "type": "CONFIG_CHANGE", "proposed_exposure_pct": 0.05})
    assert ok is False
    assert "Config tuning disabled" in reason
