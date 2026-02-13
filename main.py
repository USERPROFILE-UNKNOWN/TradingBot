import os
from modules.database import DataManager
from modules.ui import TradingApp
from modules.utils import get_paths, ensure_split_config_layout, load_split_config, get_last_config_sanitizer_report

def _resolve_db_placeholder_path(paths, config) -> str:
    """Resolve a placeholder db_path used only for call-site compatibility.

    v5.14.0 Tier 3: DataManager operates in split-db mode only and uses CONFIGURATION->db_dir
    for live DB files. This helper returns a stable path inside db_dir so callers can pass a
    db_path without depending on legacy market_data.db semantics.
    """
    try:
        raw = (config.get("CONFIGURATION", "db_dir", fallback="") or "").strip()
    except Exception:
        raw = ""

    db_dir = ""
    if raw:
        try:
            raw = os.path.expandvars(os.path.expanduser(raw))
        except Exception:
            pass
        if os.path.isabs(raw):
            db_dir = raw
        else:
            base = paths.get('config_dir') or paths.get('root') or os.getcwd()
            db_dir = os.path.normpath(os.path.join(base, raw))

    if not db_dir:
        db_dir = paths.get('db_dir') or os.path.join(paths.get('root', os.getcwd()), 'db')

    return os.path.join(os.path.normpath(db_dir), 'market_data.db')



def main():
    # 1) Setup Paths
    paths = get_paths()
    os.makedirs(paths['logs'], exist_ok=True)
    os.makedirs(paths['backup'], exist_ok=True)

    # 2) Ensure split config layout (forward-only; creates missing split INIs)
    ensure_split_config_layout(paths)

    # 3) Load merged runtime config
    config = load_split_config(paths)

    # Config sanitizer warning (only if repairs were needed)
    try:
        rep = get_last_config_sanitizer_report()
        if rep:
            msg = (
                f"[CONFIG] ⚠️ Sanitizer repaired config.ini formatting (repairs={rep.get('repairs')}). "
                f"Backup: {rep.get('backup_path', '')}"
            )
            print(msg)
            try:
                with open(os.path.join(paths['logs'], 'config_sanitizer.log'), 'a', encoding='utf-8') as lf:
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
