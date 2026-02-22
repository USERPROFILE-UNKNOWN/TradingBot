import os
import logging

from modules.config_validate import validate_runtime_config
from modules.database import DataManager
from modules.logging_utils import configure_logging, get_logger
from modules.startup_paths import migrate_root_db_to_platform, resolve_db_placeholder_path
from modules.ui import TradingApp
from modules.utils import (
    ensure_split_config_layout,
    get_last_config_sanitizer_report,
    get_paths,
    load_split_config,
)


def _is_strict_validation_enabled(config) -> bool:
    try:
        raw = config.get("CONFIGURATION", "strict_config_validation", fallback="False")
    except Exception:
        raw = "False"
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _read_agent_mode(config) -> str:
    try:
        raw = config.get("CONFIGURATION", "agent_mode", fallback="OFF")
    except Exception:
        raw = "OFF"
    return str(raw or "OFF").strip().upper() or "OFF"


def _should_require_credentials_for_validation(config, *, strict_mode: bool) -> bool:
    """Require broker credentials only for strict LIVE startup.

    Strict validation remains a schema/type/range gate for all modes, but blocking
    startup on missing API credentials is reserved for LIVE mode so users can still
    open the app and repair keys in PAPER/ADVISORY/OFF.
    """
    if not strict_mode:
        return False
    return _read_agent_mode(config) == "LIVE"


def main():
    # 1) Setup Paths
    paths = get_paths()
    try:
        os.makedirs(paths["logs"], exist_ok=True)
    except Exception:
        pass
    try:
        os.makedirs(paths["backup"], exist_ok=True)
    except Exception:
        pass

    configure_logging(paths.get("logs"))
    log = logging.LoggerAdapter(get_logger(__name__), {"component": "startup", "mode": "-"})

    # 2) Ensure split config layout (forward-only; creates missing split INIs)
    try:
        ensure_split_config_layout(paths)
    except Exception:
        log.exception("[E_STARTUP_CONFIG_LAYOUT_FAIL] Failed to ensure split config layout")
        raise

    # 3) Load merged runtime config
    try:
        config = load_split_config(paths)
    except Exception:
        log.exception("[E_STARTUP_CONFIG_LOAD_FAIL] Failed to load split config")
        raise

    # Update mode context once config is available
    try:
        log.extra["mode"] = _read_agent_mode(config)
    except Exception:
        pass

    # 3.1) Validate runtime config (strict mode optional)
    strict_mode = _is_strict_validation_enabled(config)
    repv = validate_runtime_config(
        config,
        strict=strict_mode,
        require_credentials=_should_require_credentials_for_validation(config, strict_mode=strict_mode),
    )
    for warning in repv.warnings:
        log.warning("[CONFIG] %s", warning)

    if repv.errors:
        for err in repv.errors:
            log.error("[CONFIG] %s", err)
        if strict_mode:
            log.error("[E_CFG_STRICT_VALIDATION_FAILED] Configuration validation failed in strict mode")
            raise RuntimeError("Configuration validation failed in strict mode")

    # Config sanitizer warning (only if repairs were needed)
    try:
        rep = get_last_config_sanitizer_report()
        if rep:
            msg = (
                f"[CONFIG] ⚠️ Sanitizer repaired config.ini formatting (repairs={rep.get('repairs')}). "
                f"Backup: {rep.get('backup_path', '')}"
            )
            log.warning(msg)
            try:
                with open(
                    os.path.join(paths["logs"], "config_sanitizer.log"),
                    "a",
                    encoding="utf-8",
                ) as lf:
                    lf.write(msg + "\n")
            except Exception:
                log.exception("[E_STARTUP_SANITIZER_WRITE_FAIL] Failed writing config_sanitizer.log")
    except Exception:
        log.exception("[E_STARTUP_SANITIZER_READ_FAIL] Failed to read sanitizer report")

    # 4) Resolve + init Database
    try:
        migrate_root_db_to_platform(paths, log_fn=lambda m: log.info(m))
    except Exception:
        log.exception("[E_DB_PLATFORM_MIGRATION_FAIL] Root->platform DB migration failed")

    db_path = resolve_db_placeholder_path(paths, config)
    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    except Exception:
        pass

    try:
        db_manager = DataManager(db_path, config=config, paths=paths)
    except Exception:
        log.exception("[E_DB_INIT_FAIL] DataManager initialization failed")
        raise

    # 5) Launch GUI
    try:
        app = TradingApp(config, db_manager)
    except Exception:
        log.exception("[E_UI_INIT_FAIL] TradingApp initialization failed")
        raise

    app.mainloop()


if __name__ == "__main__":
    main()
