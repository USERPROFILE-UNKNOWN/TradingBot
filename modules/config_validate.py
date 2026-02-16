"""Runtime configuration schema validation helpers.

v5.14.3 (Config schema validation + strict mode):
- Required keys, type checks, and range checks for safety-critical knobs.
- Safe defaults are used *only* when strict mode is OFF.
- Secrets (keys/tokens) are never included in error/warning strings.

Used by:
- Startup gate in main.py
- UI save-gates (prevent writing invalid configs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


SENSITIVE_KEYS = {
    "alpaca_key",
    "alpaca_secret",
    "telegram_token",
    "telegram_chat_id",
    "api_key",
    "api_secret",
    "token",
    "secret",
}

_BOOL_TRUE = {"1", "true", "yes", "y", "on"}
_BOOL_FALSE = {"0", "false", "no", "n", "off"}


@dataclass
class ConfigValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


@dataclass(frozen=True)
class FieldSpec:
    section: str
    key: str
    kind: str  # "int" | "float" | "bool" | "str" | "enum"
    default: Any
    required: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed: Optional[Sequence[str]] = None


def _is_sensitive_key(key: str) -> bool:
    k = str(key or "").strip().lower()
    if not k:
        return False
    if k in {s.lower() for s in SENSITIVE_KEYS}:
        return True
    return any(tok in k for tok in ("token", "secret", "key"))


def _default_repr(key: str, value: Any) -> str:
    if _is_sensitive_key(key):
        return "***REDACTED***"
    try:
        return repr(value)
    except Exception:
        return "<unrepr>"


def _parse_bool(raw: Any) -> Optional[bool]:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in _BOOL_TRUE:
        return True
    if s in _BOOL_FALSE:
        return False
    return None


def _coerce(spec: FieldSpec, raw: Any) -> tuple[bool, Any]:
    """Return (ok, value). Never raises."""
    try:
        if spec.kind == "str":
            return True, str(raw)
        if spec.kind == "int":
            return True, int(float(str(raw).strip()))
        if spec.kind == "float":
            return True, float(str(raw).strip())
        if spec.kind == "bool":
            b = _parse_bool(raw)
            if b is None:
                return False, None
            return True, b
        if spec.kind == "enum":
            return True, str(raw).strip().upper()
    except Exception:
        return False, None

    return False, None


def _get_raw(config, section: str, key: str) -> Any:
    try:
        if not config.has_section(section):
            return None
        if not config.has_option(section, key):
            return None
        return config.get(section, key)
    except Exception:
        return None


def _validate_field(config, spec: FieldSpec, rep: ConfigValidationReport, *, strict: bool) -> Any:
    raw = _get_raw(config, spec.section, spec.key)

    # Missing key
    if raw is None:
        if spec.required and strict:
            rep.errors.append(f"Missing required key: {spec.section}.{spec.key}")
        else:
            rep.warnings.append(
                f"Missing {spec.section}.{spec.key}; using default={_default_repr(spec.key, spec.default)}"
            )
        return spec.default

    ok_type, val = _coerce(spec, raw)
    if not ok_type:
        msg = f"{spec.section}.{spec.key} has invalid type; expected {spec.kind}"
        if strict:
            rep.errors.append(msg)
        else:
            rep.warnings.append(msg)
            val = spec.default

    # Enum check
    if spec.kind == "enum" and spec.allowed:
        allowed = [str(x).strip().upper() for x in spec.allowed if str(x).strip()]
        v = str(val).strip().upper()
        if v not in allowed:
            if strict:
                rep.errors.append(f"{spec.section}.{spec.key} must be one of {allowed}")
            else:
                rep.warnings.append(
                    f"{spec.section}.{spec.key} is not one of {allowed}; using default={_default_repr(spec.key, spec.default)}"
                )
                val = str(spec.default).strip().upper()

    # Range checks (numeric only) are always errors (prevent silent risk).
    if spec.kind in ("int", "float"):
        try:
            fval = float(val)
        except Exception:
            fval = None

        if fval is not None:
            if spec.min_value is not None and fval < float(spec.min_value):
                # Keep legacy wordings for core knobs used by tests/logs.
                if spec.section == "CONFIGURATION" and spec.key == "amount_to_trade":
                    rep.errors.append("CONFIGURATION.amount_to_trade must be > 0")
                elif spec.section == "CONFIGURATION" and spec.key == "max_positions":
                    rep.errors.append("CONFIGURATION.max_positions must be >= 1")
                elif spec.section == "CONFIGURATION" and spec.key == "max_daily_loss":
                    rep.errors.append("CONFIGURATION.max_daily_loss must be > 0")
                elif spec.section == "CONFIGURATION" and spec.key == "live_entry_ttl_seconds":
                    rep.errors.append("CONFIGURATION.live_entry_ttl_seconds must be >= 0")
                else:
                    rep.errors.append(f"{spec.section}.{spec.key} must be >= {spec.min_value}")

            if spec.max_value is not None and fval > float(spec.max_value):
                if spec.section == "CONFIGURATION" and spec.key == "max_percent_per_stock":
                    rep.errors.append("CONFIGURATION.max_percent_per_stock must be within (0, 1]")
                else:
                    rep.errors.append(f"{spec.section}.{spec.key} must be <= {spec.max_value}")

    return val


# ---------------------------------------------------------------------------
# Schema Index (v5.14.3)
#
# Key | Type | Required | Default | Constraints / Notes
#
# CONFIGURATION.amount_to_trade | float | yes | 2000.0 | > 0
# CONFIGURATION.max_positions | int | yes | 5 | >= 1
# CONFIGURATION.update_interval_sec | int | yes | 60 | >= 1 (warn if < 5)
# CONFIGURATION.max_percent_per_stock | float | yes | 0.20 | (0, 1]
# CONFIGURATION.max_daily_loss | float | yes | 100.0 | > 0
# CONFIGURATION.live_entry_ttl_seconds | int | yes | 120 | >= 0
#
# CONFIGURATION.agent_mode | enum | no | OFF | OFF|ADVISORY|PAPER|LIVE
# CONFIGURATION.agent_live_max_exposure_pct | float | no | 0.30 | [0, 1]
# CONFIGURATION.agent_max_live_changes_per_day | int | no | 8 | [0, 1000]
# CONFIGURATION.strict_config_validation | bool | no | False | startup gate
#
# CONFIGURATION.log_level | enum | no | INFO | DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL
# CONFIGURATION.log_snapshot_interval_sec | int | no | 300 | [0, 3600]
#
# KEYS.base_url | str | no | https://paper-api.alpaca.markets | must start with http(s) if set
# KEYS.telegram_enabled | bool | no | True |
# KEYS.alpaca_key | str | no | ***REDACTED*** | required only when require_credentials=True
# KEYS.alpaca_secret | str | no | ***REDACTED*** | required only when require_credentials=True
# KEYS.telegram_token | str | no | ***REDACTED*** | warn when telegram_enabled=True
# KEYS.telegram_chat_id | str | no | ***REDACTED*** | warn when telegram_enabled=True
# TRADINGVIEW.enabled | bool | no | False | If True, start webhook listener
# TRADINGVIEW.listen_host | str | no | 127.0.0.1 | Bind address (recommend 127.0.0.1)
# TRADINGVIEW.listen_port | int | no | 5001 | [1, 65535]
# TRADINGVIEW.secret | str | no | ***REDACTED*** | required when enabled in strict mode
# TRADINGVIEW.allowed_signals | str | no | (empty) | optional CSV allow-list (e.g. BUY,SELL)
# TRADINGVIEW.mode | enum | no | ADVISORY | OFF|ADVISORY|PAPER|LIVE
# TRADINGVIEW.candidate_cooldown_minutes | int | no | 5 | minutes; de-dup identical alerts (symbol/timeframe/signal)
# ---------------------------------------------------------------------------


_SCHEMA: list[FieldSpec] = [
    # CONFIGURATION (core safety-critical knobs)
    FieldSpec("CONFIGURATION", "amount_to_trade", "float", 2000.0, required=True, min_value=0.000001),
    FieldSpec("CONFIGURATION", "max_positions", "int", 5, required=True, min_value=1),
    FieldSpec("CONFIGURATION", "update_interval_sec", "int", 60, required=True, min_value=1),
    FieldSpec(
        "CONFIGURATION",
        "max_percent_per_stock",
        "float",
        0.20,
        required=True,
        min_value=0.000001,
        max_value=1.0,
    ),
    FieldSpec("CONFIGURATION", "max_daily_loss", "float", 100.0, required=True, min_value=0.000001),
    FieldSpec("CONFIGURATION", "live_entry_ttl_seconds", "int", 120, required=True, min_value=0),

    # CONFIGURATION (mode / governance)
    FieldSpec("CONFIGURATION", "agent_mode", "enum", "OFF", required=False, allowed=("OFF", "ADVISORY", "PAPER", "LIVE")),
    FieldSpec("CONFIGURATION", "agent_live_max_exposure_pct", "float", 0.30, required=False, min_value=0.0, max_value=1.0),
    FieldSpec("CONFIGURATION", "agent_max_live_changes_per_day", "int", 8, required=False, min_value=0, max_value=1000),
    FieldSpec("CONFIGURATION", "strict_config_validation", "bool", False, required=False),

    # CONFIGURATION (logging)
    FieldSpec(
        "CONFIGURATION",
        "log_level",
        "enum",
        "INFO",
        required=False,
        allowed=("DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"),
    ),
    FieldSpec("CONFIGURATION", "log_snapshot_interval_sec", "int", 300, required=False, min_value=0, max_value=3600),
]


_KEYS_SCHEMA: list[FieldSpec] = [
    FieldSpec("KEYS", "base_url", "str", "https://paper-api.alpaca.markets", required=False),
    FieldSpec("KEYS", "telegram_enabled", "bool", True, required=False),
    FieldSpec("KEYS", "alpaca_key", "str", "", required=False),
    FieldSpec("KEYS", "alpaca_secret", "str", "", required=False),
    FieldSpec("KEYS", "telegram_token", "str", "", required=False),
    FieldSpec("KEYS", "telegram_chat_id", "str", "", required=False),
    FieldSpec("KEYS", "tradingview_secret", "str", "", required=False),
]



_TRADINGVIEW_SCHEMA_ALWAYS: list[FieldSpec] = [
    FieldSpec("TRADINGVIEW", "enabled", "bool", False, required=False),
    FieldSpec("TRADINGVIEW", "mode", "enum", "ADVISORY", required=False, allowed=("OFF", "ADVISORY", "PAPER", "LIVE")),
    FieldSpec("TRADINGVIEW", "candidate_cooldown_minutes", "int", 5, required=False, min_value=0, max_value=1440),
    FieldSpec("TRADINGVIEW", "autovalidation_enabled", "bool", True, required=False),
    FieldSpec("TRADINGVIEW", "autovalidation_cooldown_minutes", "int", 10, required=False, min_value=0, max_value=1440),
    FieldSpec("TRADINGVIEW", "autovalidation_freshness_minutes", "int", 30, required=False, min_value=1, max_value=1440),
    FieldSpec("TRADINGVIEW", "autovalidation_backfill_days", "int", 60, required=False, min_value=1, max_value=365),
    FieldSpec("TRADINGVIEW", "autovalidation_backtest_days", "int", 14, required=False, min_value=1, max_value=90),
    FieldSpec("TRADINGVIEW", "autovalidation_max_strategies", "int", 6, required=False, min_value=1, max_value=20),
    FieldSpec("TRADINGVIEW", "autovalidation_min_trades", "int", 1, required=False, min_value=0, max_value=50),
    FieldSpec("TRADINGVIEW", "autovalidation_max_concurrency", "int", 1, required=False, min_value=1, max_value=4),
]


_TRADINGVIEW_SCHEMA_ENABLED: list[FieldSpec] = [
    FieldSpec("TRADINGVIEW", "listen_host", "str", "127.0.0.1", required=False),
    FieldSpec("TRADINGVIEW", "listen_port", "int", 5001, required=False, min_value=1, max_value=65535),
    FieldSpec("TRADINGVIEW", "allowed_signals", "str", "", required=False),
]

def validate_runtime_config(
    config,
    *,
    strict: bool = False,
    require_credentials: bool = False,
    include_credentials: bool = True,
) -> ConfigValidationReport:
    """Validate config for safer startup and UI write-gates.

    strict:
      - True: missing/bad types become errors
      - False: missing/bad types use safe defaults (warning)

    require_credentials:
      - True: missing alpaca_key/alpaca_secret become errors

    include_credentials:
      - False: skip KEYS validation and credential presence checks (CONFIGURATION-only save)
    """
    rep = ConfigValidationReport()

    # Required sections
    required_sections = ["CONFIGURATION"]
    if include_credentials or require_credentials:
        required_sections.append("KEYS")

    for sec in required_sections:
        try:
            if not config.has_section(sec):
                rep.errors.append(f"Missing required section: {sec}")
        except Exception:
            rep.errors.append(f"Missing required section: {sec}")

    if rep.errors:
        return rep

    # Validate CONFIGURATION fields.
    vals: dict[tuple[str, str], Any] = {}
    for spec in _SCHEMA:
        vals[(spec.section, spec.key)] = _validate_field(config, spec, rep, strict=strict)

    # Advisory warning: very low update interval.
    try:
        upd = int(float(vals.get(("CONFIGURATION", "update_interval_sec"), 60)))
    except Exception:
        upd = 60
    if upd < 5:
        rep.warnings.append("CONFIGURATION.update_interval_sec is very low (<5); may cause API throttling")

    # Optional cross-field check.
    try:
        if config.has_option("CONFIGURATION", "max_open_trades"):
            mot_raw = _get_raw(config, "CONFIGURATION", "max_open_trades")
            if mot_raw is not None:
                try:
                    mot = int(float(mot_raw))
                    mp = int(float(vals.get(("CONFIGURATION", "max_positions"), 5)))
                    if mot < 0:
                        rep.errors.append("CONFIGURATION.max_open_trades must be >= 0")
                    elif mp > 0 and mot > mp:
                        rep.errors.append("CONFIGURATION.max_open_trades must be <= CONFIGURATION.max_positions")
                except Exception:
                    if strict:
                        rep.errors.append("CONFIGURATION.max_open_trades has invalid type; expected int")
                    else:
                        rep.warnings.append("CONFIGURATION.max_open_trades has invalid type; expected int")
    except Exception:
        pass

    # Validate TRADINGVIEW fields (optional integration; only enforced when enabled).
    try:
        if config.has_section("TRADINGVIEW"):
            tv_vals: dict[tuple[str, str], Any] = {}
            for spec in _TRADINGVIEW_SCHEMA_ALWAYS:
                tv_vals[(spec.section, spec.key)] = _validate_field(config, spec, rep, strict=strict)

            enabled = bool(tv_vals.get(("TRADINGVIEW", "enabled"), False))
            mode = str(tv_vals.get(("TRADINGVIEW", "mode"), "ADVISORY")).strip().upper() or "ADVISORY"

            # If integration is enabled, validate the remaining keys (port/secret etc.)
            if enabled and mode != "OFF":
                for spec in _TRADINGVIEW_SCHEMA_ENABLED:
                    tv_vals[(spec.section, spec.key)] = _validate_field(config, spec, rep, strict=strict)

                # Secret is required in strict mode when enabled.
                try:
                    secret = (_get_raw(config, "TRADINGVIEW", "secret") or "").strip()
                except Exception:
                    secret = ""
                if not secret:
                    msg = "TRADINGVIEW.secret is empty while TRADINGVIEW.enabled=True; webhook auth is unsafe"
                    if strict:
                        rep.errors.append(msg)
                    else:
                        rep.warnings.append(msg)
    except Exception:
        pass

    if not include_credentials and not require_credentials:
        return rep

    # Validate KEYS fields (types/enum) (no secret values in messages).
    for spec in _KEYS_SCHEMA:
        _validate_field(config, spec, rep, strict=strict)

    # base_url sanity (not secret)
    try:
        base_url = (_get_raw(config, "KEYS", "base_url") or "").strip()
    except Exception:
        base_url = ""
    if base_url:
        if not (base_url.startswith("http://") or base_url.startswith("https://")):
            if strict:
                rep.errors.append("KEYS.base_url must start with http:// or https://")
            else:
                rep.warnings.append("KEYS.base_url should start with http:// or https://")
    else:
        try:
            if config.has_option("KEYS", "base_url"):
                rep.warnings.append("KEYS.base_url is empty; using paper default")
        except Exception:
            pass

    # Alpaca creds
    for key in ("alpaca_key", "alpaca_secret"):
        try:
            val = (_get_raw(config, "KEYS", key) or "").strip()
        except Exception:
            val = ""
        if not val:
            msg = f"KEYS.{key} is empty; live trading API calls may fail"
            if require_credentials:
                rep.errors.append(msg)
            else:
                rep.warnings.append(msg)

    # Telegram creds when telegram_enabled is true
    try:
        tel_enabled_raw = _get_raw(config, "KEYS", "telegram_enabled")
        tel_enabled = _parse_bool(tel_enabled_raw) if tel_enabled_raw is not None else True
        tel_enabled = True if tel_enabled is None else tel_enabled
    except Exception:
        tel_enabled = True

    if tel_enabled:
        for key in ("telegram_token", "telegram_chat_id"):
            try:
                val = (_get_raw(config, "KEYS", key) or "").strip()
            except Exception:
                val = ""
            if not val:
                rep.warnings.append(f"KEYS.{key} is empty; telegram notifications may not work")

    return rep
