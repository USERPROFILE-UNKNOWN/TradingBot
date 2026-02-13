"""Runtime configuration validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConfigValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _to_float(config, section: str, key: str, fallback: float) -> float:
    try:
        return float(config.get(section, key, fallback=str(fallback)))
    except Exception:
        return fallback


def _to_int(config, section: str, key: str, fallback: int) -> int:
    try:
        return int(float(config.get(section, key, fallback=str(fallback))))
    except Exception:
        return fallback


def validate_runtime_config(config, *, require_credentials: bool = False) -> ConfigValidationReport:
    """Validate key runtime values for safer startup behavior."""
    rep = ConfigValidationReport()

    required_sections = ("KEYS", "CONFIGURATION")
    for sec in required_sections:
        if not config.has_section(sec):
            rep.errors.append(f"Missing required section: {sec}")

    if rep.errors:
        return rep

    # Required credentials fields (warning-only unless strict startup asks for enforcement).
    for key in ("alpaca_key", "alpaca_secret"):
        try:
            val = (config.get("KEYS", key, fallback="") or "").strip()
        except Exception:
            val = ""
        if not val:
            msg = f"KEYS.{key} is empty; live trading API calls may fail"
            if require_credentials:
                rep.errors.append(msg)
            else:
                rep.warnings.append(msg)

    amount_to_trade = _to_float(config, "CONFIGURATION", "amount_to_trade", 2000.0)
    if amount_to_trade <= 0:
        rep.errors.append("CONFIGURATION.amount_to_trade must be > 0")

    max_positions = _to_int(config, "CONFIGURATION", "max_positions", 5)
    if max_positions <= 0:
        rep.errors.append("CONFIGURATION.max_positions must be >= 1")

    update_interval = _to_int(config, "CONFIGURATION", "update_interval_sec", 60)
    if update_interval < 5:
        rep.warnings.append("CONFIGURATION.update_interval_sec is very low (<5); may cause API throttling")

    max_pct = _to_float(config, "CONFIGURATION", "max_percent_per_stock", 0.20)
    if not (0 < max_pct <= 1.0):
        rep.errors.append("CONFIGURATION.max_percent_per_stock must be within (0, 1]")

    daily_loss = _to_float(config, "CONFIGURATION", "max_daily_loss", 100.0)
    if daily_loss <= 0:
        rep.errors.append("CONFIGURATION.max_daily_loss must be > 0")

    ttl = _to_int(config, "CONFIGURATION", "live_entry_ttl_seconds", 120)
    if ttl < 0:
        rep.errors.append("CONFIGURATION.live_entry_ttl_seconds must be >= 0")

    return rep
