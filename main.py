import os

from modules.config_validate import validate_runtime_config
from modules.database import DataManager
from modules.logging_utils import configure_logging, get_logger
from modules.startup_paths import resolve_db_placeholder_path
from modules.ui import TradingApp
from modules.utils import (
    ensure_split_config_layout,
    get_last_config_sanitizer_report,
    get_paths,
    load_split_config,
)


def _resolve_db_placeholder_path(paths, config) -> str:
    """Backward-compatible wrapper for startup db placeholder resolution."""
    return resolve_db_placeholder_path(paths, config)


def _is_strict_validation_enabled(config) -> bool:
    try:
        raw = config.get("CONFIGURATION", "strict_config_validation", fallback="False")
    except Exception:
        raw = "False"
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def main():
    # 1) Setup Paths
    paths = get_paths()
    os.makedirs(paths['logs'], exist_ok=True)
    os.makedirs(paths['backup'], exist_ok=True)
    configure_logging(paths.get("logs"))
    log = get_logger(__name__)

    # 2) Ensure split config layout (forward-only; creates missing split INIs)
    ensure_split_config_layout(paths)

    # 3) Load merged runtime config
    config = load_split_config(paths)

    # 3.1) Validate runtime config (strict mode optional)
    repv = validate_runtime_config(config)
    for warning in repv.warnings:
        log.warning("[CONFIG] %s", warning)

    if repv.errors:
        for err in repv.errors:
            log.error("[CONFIG] %s", err)
        if _is_strict_validation_enabled(config):
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
                pass
    except Exception:
        pass

    # 4) Resolve + init Database
    db_path = _resolve_db_placeholder_path(paths, config)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db_manager = DataManager(db_path, config=config, paths=paths)

    # 5) Launch GUI
    app = TradingApp(config, db_manager)
    app.mainloop()


if __name__ == "__main__":
    main()
