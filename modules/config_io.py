"""Split-config I/O.

This module owns reading/writing/ensuring the 4-file split INI layout:
  - config/config.ini      [CONFIGURATION]
  - config/watchlist.ini   [WATCHLIST]
  - config/strategy.ini    [STRATEGY_*]
  - config/keys.ini        [KEYS]

It is separated from modules/utils.py to reduce regression risk.

During the v5.x refactor series, modules/utils.py re-exports the public APIs
so existing imports continue to work.
"""

from __future__ import annotations

import configparser
import os
import re
import shutil
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

from .config_defaults import default_split_config

# Phase 4 (v5.14.0): new watchlist layout sections
WATCHLIST_SECTIONS_REQUIRED = [
    "WATCHLIST_FAVORITES_STOCK",
    "WATCHLIST_FAVORITES_CRYPTO",
    "WATCHLIST_ACTIVE_STOCK",
    "WATCHLIST_ACTIVE_CRYPTO",
    "WATCHLIST_ARCHIVE_STOCK",
    "WATCHLIST_ARCHIVE_CRYPTO",
]


def _new_config_parser() -> configparser.ConfigParser:
    """Create a robust ConfigParser.

    - Preserve option name case (optionxform=str)
    - Disable interpolation to avoid '%' parsing errors
    - Allow inline comments (# / ;)
    - Allow duplicates in damaged INI files (last one wins)
    """
    cfg = configparser.ConfigParser(
        interpolation=None,
        inline_comment_prefixes=("#", ";"),
        strict=False,
    )
    cfg.optionxform = str
    return cfg


def _read_ini_with_fallback(cfg: configparser.ConfigParser, path: str) -> str:
    """Read an INI file robustly across common Windows encodings."""
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            cfg.read(path, encoding=enc)
            return enc
        except UnicodeDecodeError:
            continue

    # Last resort: avoid hard crash
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        cfg.read_file(f, source=path)
    return "utf-8+replace"


# Last sanitizer report (set when config.ini formatting is repaired)
_LAST_CONFIG_SANITIZER_REPORT: dict | None = None


def get_last_config_sanitizer_report() -> dict | None:
    """Return the last config.ini sanitizer report (or None if no repairs were needed)."""
    return _LAST_CONFIG_SANITIZER_REPORT


# ---------------------------------------------------------------------------
# Phase 0 guardrail helpers
# ---------------------------------------------------------------------------

_MULTI_ASSIGNMENT_TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\s*=")

def _ini_section_likely_merged_text(text: str, section_name: str = "CONFIGURATION") -> bool:
    """Heuristic detection for the 'merged lines' config corruption.

    This is intentionally conservative: it only flags when a *single* non-comment
    line within the target section appears to contain 2+ INI assignments.
    """
    in_section = False
    target = section_name.strip().lower()

    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("[") and s.endswith("]"):
            in_section = (s[1:-1].strip().lower() == target)
            continue
        if not in_section:
            continue
        if s.startswith("#") or s.startswith(";"):
            continue

        # If a single line contains multiple "key =" tokens, the file has likely lost newlines.
        if len(_MULTI_ASSIGNMENT_TOKEN_RE.findall(raw)) >= 2:
            return True

    return False


def sanitize_configuration_ini_if_needed(paths: Dict[str, str], section_name: str = "CONFIGURATION") -> dict | None:
    """If config.ini appears merged/corrupted, repair it before any further writes.

    Returns the sanitizer report dict if a repair was applied, otherwise None.
    """
    ini_path = paths.get("configuration_ini")
    if not ini_path or not os.path.exists(ini_path):
        return None

    try:
        enc = _detect_text_encoding(ini_path)
        with open(ini_path, "r", encoding=enc, errors="replace") as f:
            text = f.read()
    except Exception:
        return None

    if not _ini_section_likely_merged_text(text, section_name=section_name):
        return None

    # Repair in-place (creates a timestamped backup under TradingBot/backups/config_sanitizer)
    return _sanitize_ini_section_formatting(ini_path, section_name=section_name)

def _sanitize_ini_section_formatting(
    path: str,
    *,
    section_name: str = "CONFIGURATION",
    backups_dir: str | None = None,
) -> dict | None:
    """Repair common INI formatting damage **without** rewriting the file structure.

    Goals:
    - Ensure each key=value pair is on its own line (prevents keys being 'hidden' mid-line).
    - Ensure comment lines are not indented (prevents configparser continuation-line merges).
    - Ensure inline comments have a space before '#' / ';' (readability + safer parsing).

    Returns a report dict if repairs were applied; otherwise None.
    """
    global _LAST_CONFIG_SANITIZER_REPORT

    if not os.path.exists(path):
        _LAST_CONFIG_SANITIZER_REPORT = None
        return None

    enc = _detect_text_encoding(path)
    try:
        with open(path, "r", encoding=enc, errors="replace") as f:
            text = f.read()
    except Exception:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        enc = "utf-8"

    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(True)  # keepends

    # Find section bounds
    sec_pat = re.compile(r"^\s*\[\s*" + re.escape(section_name) + r"\s*\]\s*$", re.IGNORECASE)
    hdr_pat = re.compile(r"^\s*\[[^\]]+\]\s*$")
    start = None
    for i, ln in enumerate(lines):
        if sec_pat.match(ln.strip()):
            start = i
            break
    if start is None:
        _LAST_CONFIG_SANITIZER_REPORT = None
        return None

    end = len(lines)
    for j in range(start + 1, len(lines)):
        s = lines[j].strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith(";"):
            continue
        if hdr_pat.match(s):
            end = j
            break

    def _split_multi_assignments(raw_line: str) -> list[str]:
        # Preserve original line ending
        line_ending = ""
        if raw_line.endswith("\r\n"):
            line_ending = "\r\n"
            core = raw_line[:-2]
        elif raw_line.endswith("\n"):
            line_ending = "\n"
            core = raw_line[:-1]
        else:
            core = raw_line

        # Find all key= occurrences (for non-comment lines)
        matches = list(re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\s*=", core))
        if len(matches) <= 1:
            return [raw_line]

        parts: list[str] = []
        for k, m in enumerate(matches):
            a = m.start()
            b = matches[k + 1].start() if k + 1 < len(matches) else len(core)
            seg = core[a:b].strip()
            if not seg:
                continue
            # Ensure inline comment spacing: 'value#comment' -> 'value #comment'
            seg = re.sub(r"(\S)([#;])", r"\1 \2", seg)
            parts.append(seg + line_ending)

        return parts if parts else [raw_line]

    repairs = 0
    out: list[str] = []
    for idx, ln in enumerate(lines):
        if idx < start + 1 or idx >= end:
            out.append(ln)
            continue

        stripped = ln.lstrip()
        # Normalize indented comment lines (configparser treats them as continuation lines)
        if stripped.startswith("#") or stripped.startswith(";"):
            if ln != stripped:
                repairs += 1
            out.append(stripped if stripped.endswith(("\n", "\r\n")) else stripped + newline)
            continue

        # Skip blank lines
        if not ln.strip():
            out.append(ln)
            continue

        # Repair multi-assignment lines
        new_parts = _split_multi_assignments(ln)
        if len(new_parts) > 1:
            repairs += (len(new_parts) - 1)

        # Normalize inline comment spacing within each part
        for part in new_parts:
            fixed = re.sub(r"(\S)([#;])", r"\1 \2", part)
            if fixed != part:
                repairs += 1
            # Ensure every line has an ending if we're inside the section
            if not fixed.endswith(("\n", "\r\n")):
                fixed += newline
                repairs += 1
            out.append(fixed)

    new_text = "".join(out)

    # Also ensure final newline (prevents accidental append concatenation)
    if not new_text.endswith(("\n", "\r\n")):
        new_text += newline
        repairs += 1

    if new_text == text or repairs == 0:
        _LAST_CONFIG_SANITIZER_REPORT = None
        return None

    # Backup before writing
    try:
        root = backups_dir or os.path.join(os.path.dirname(os.path.dirname(path)), "backups", "config_sanitizer")
        ts = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        bdir = os.path.join(root, ts)
        os.makedirs(bdir, exist_ok=True)
        backup_path = os.path.join(bdir, os.path.basename(path))
        shutil.copy2(path, backup_path)
    except Exception:
        backup_path = ""

    try:
        with open(path, "w", encoding=enc, errors="replace", newline="") as f:
            f.write(new_text)
    except Exception:
        with open(path, "w", encoding="utf-8", errors="replace", newline="") as f:
            f.write(new_text)

    report = {
        "path": path,
        "repairs": repairs,
        "backup_path": backup_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _LAST_CONFIG_SANITIZER_REPORT = report
    return report

def _write_ini(cfg: configparser.ConfigParser, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        cfg.write(f)


def _clone_sections(src: configparser.ConfigParser, section_names: Iterable[str]) -> configparser.ConfigParser:
    dst = _new_config_parser()
    for sec in section_names:
        if src.has_section(sec):
            dst.add_section(sec)
            for k, v in src.items(sec):
                dst.set(sec, k, v)
    return dst


def write_split_config(cfg: configparser.ConfigParser, paths: Dict[str, str]) -> None:
    """Write runtime config into the 4-file split layout."""
    keys_cfg = _clone_sections(cfg, ["KEYS"])
    config_cfg = _clone_sections(cfg, ["CONFIGURATION"])

    watch_secs = [s for s in cfg.sections() if s.upper().startswith("WATCHLIST_")]
    watch_cfg = _clone_sections(cfg, watch_secs)

    strat_secs = [s for s in cfg.sections() if s.upper().startswith("STRATEGY_")]
    strat_cfg = _clone_sections(cfg, strat_secs)

    # Ensure the files exist even if empty sections were missing
    if not keys_cfg.has_section("KEYS"):
        keys_cfg.add_section("KEYS")
    if not config_cfg.has_section("CONFIGURATION"):
        config_cfg.add_section("CONFIGURATION")
    if not watch_secs:
        for s in WATCHLIST_SECTIONS_REQUIRED:
            if not watch_cfg.has_section(s):
                watch_cfg.add_section(s)
    if not strat_secs:
        if not strat_cfg.has_section("STRATEGY_THE_GENERAL"):
            strat_cfg.add_section("STRATEGY_THE_GENERAL")
            strat_cfg.set("STRATEGY_THE_GENERAL", "rsi_buy", "35")

    _write_ini(keys_cfg, paths["keys_ini"])

    # Preserve user's comment formatting in config.ini when possible.
    # If the file doesn't exist (fresh install), create it normally.
    if os.path.exists(paths["configuration_ini"]):
        _update_ini_section_values_preserve_comments(
            paths["configuration_ini"],
            section_name="CONFIGURATION",
            items=list(config_cfg.items("CONFIGURATION")),
        )
    else:
        _write_ini(config_cfg, paths["configuration_ini"])

    _write_ini(watch_cfg, paths["watchlist_ini"])
    _write_ini(strat_cfg, paths["strategy_ini"])


def load_split_config(paths: Dict[str, str]) -> configparser.ConfigParser:
    """Load and merge the 4 split config files into a single runtime ConfigParser."""
    cfg = _new_config_parser()
    for p in (paths["configuration_ini"], paths["watchlist_ini"], paths["strategy_ini"], paths["keys_ini"]):
        if os.path.exists(p):
            if p == paths.get("configuration_ini"):
                try:
                    _sanitize_ini_section_formatting(p, section_name="CONFIGURATION")
                except Exception:
                    pass
            _read_ini_with_fallback(cfg, p)
    return cfg


def _detect_text_encoding(path: str) -> str:
    """Best-effort encoding detection for INI files; preserves existing user files."""
    try:
        with open(path, "rb") as f:
            b = f.read()
        if b.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        try:
            b.decode("utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            return "cp1252"
    except Exception:
        return "utf-8"


def _append_missing_ini_options_preserve_comments(
    path: str,
    *,
    section_name: str,
    defaults_items: List[Tuple[str, str]],
    descriptions: Dict[str, str] | None = None,
    header_title: str = "AUTO-ADDED DEFAULTS",
) -> int:
    """Append missing key=value lines without rewriting the INI (preserve comments)."""
    if not os.path.exists(path):
        return 0

    enc = _detect_text_encoding(path)
    try:
        with open(path, "r", encoding=enc, errors="replace") as f:
            text = f.read()
    except Exception:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

    if f"[{section_name}]" not in text:
        return 0

    missing: List[Tuple[str, str]] = []
    for k, v in defaults_items:
        key = str(k).strip()
        if not key:
            continue
        # Robust presence check: prefer start-of-line, but also detect keys that may be mid-line in a damaged file.
        if re.search(rf"^\s*{re.escape(key)}\s*=", text, flags=re.IGNORECASE | re.MULTILINE):
            continue
        if re.search(rf"\b{re.escape(key)}\s*=", text, flags=re.IGNORECASE):
            continue
        missing.append((key, str(v)))

    if not missing:
        return 0

    lines: List[str] = []
    if not text.endswith("\n"):
        lines.append("\n")

    lines.append("#  ==========================\n")
    lines.append(f"# {header_title}\n")
    lines.append("#  ==========================\n")

    desc_map = descriptions or {}
    for k, v in missing:
        desc = (desc_map.get(k) or desc_map.get(k.lower()) or "").strip()
        lines.append(f"# {desc if desc else 'Auto-added default setting.'}\n")
        lines.append(f"{k} = {v}\n")

    try:
        with open(path, "a", encoding=enc, errors="replace", newline="\n") as f:
            f.writelines(lines)
    except Exception:
        with open(path, "a", encoding="utf-8", errors="replace", newline="\n") as f:
            f.writelines(lines)

    return len(missing)


def _update_ini_section_values_preserve_comments(
    path: str,
    *,
    section_name: str,
    items: List[Tuple[str, str]],
    header_title: str = "AUTO-ADDED DEFAULTS (preserve comments)",
    descriptions: Dict[str, str] | None = None,
) -> Tuple[int, int]:
    """Update existing key=value lines for a section in-place (preserve comments).

    - Keeps all existing comments / blank lines / ordering.
    - Preserves inline comments after the value.
    - Adds missing keys *inside the section* near the end of the section.

    Returns: (updated_count, added_count)
    """
    if not os.path.exists(path):
        return (0, 0)

    enc = _detect_text_encoding(path)
    try:
        with open(path, "r", encoding=enc, errors="replace") as f:
            text = f.read()
    except Exception:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()


    # Phase 0 guardrail: if config.ini already lost newlines ("merged" lines),
    # repair it before attempting any further in-place edits.
    if _ini_section_likely_merged_text(text, section_name=section_name):
        _sanitize_ini_section_formatting(path, section_name=section_name)
        enc = _detect_text_encoding(path)
        with open(path, "r", encoding=enc, errors="replace") as f:
            text = f.read()

    # Preserve newline style if possible
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(True)  # keepends

    # Locate section span
    sec_pat = re.compile(r"^\s*\[\s*" + re.escape(section_name) + r"\s*\]\s*$", re.IGNORECASE)
    hdr_pat = re.compile(r"^\s*\[[^\]]+\]\s*$")

    start = None
    for i, ln in enumerate(lines):
        if sec_pat.match(ln.strip()):
            start = i
            break
    if start is None:
        return (0, 0)

    end = len(lines)
    for j in range(start + 1, len(lines)):
        s = lines[j].strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith(";"):
            continue
        if hdr_pat.match(s):
            end = j
            break

    # Map existing key->line index within section
    key_to_idx: Dict[str, int] = {}
    key_line_pat = re.compile(r"^\s*([^=\s]+)\s*=\s*(.*)$")
    for idx in range(start + 1, end):
        raw = lines[idx]
        s = raw.strip()
        if not s or s.startswith("#") or s.startswith(";"):
            continue
        m = key_line_pat.match(raw)
        if not m:
            continue
        k = m.group(1).strip()
        if k:
            key_to_idx[k.lower()] = idx

    updated = 0
    added_items: List[Tuple[str, str]] = []

    for k, v in items:
        key = str(k).strip()
        if not key:
            continue
        val = str(v)
        idx = key_to_idx.get(key.lower())
        if idx is None:
            added_items.append((key, val))
            continue

        # Preserve inline comment (after # or ;) and preserve original line ending
        raw = lines[idx]
        eol = "\r\n" if raw.endswith("\r\n") else ("\n" if raw.endswith("\n") else "")
        raw_body = raw[:-len(eol)] if eol else raw

        m = re.match(
            r"^(\s*" + re.escape(key) + r"\s*=\s*)([^#;\r\n]*)(.*)$",
            raw_body,
            flags=re.IGNORECASE,
        )
        if m:
            prefix = m.group(1)
            suffix = m.group(3)
            new_line = f"{prefix}{val}{suffix}{eol}"
        else:
            # Fallback: overwrite the whole line (keep file newline style)
            new_line = f"{key} = {val}{eol if eol else newline}"

        if new_line != raw:
            lines[idx] = new_line
            updated += 1

    added = 0
    if added_items:
        # Insert missing keys at end of section (before next header / EOF)
        insert_at = end
        block: List[str] = []
        block.append(f"#  =========================={newline}")
        block.append(f"# {header_title}{newline}")
        block.append(f"#  =========================={newline}")

        desc_map = descriptions or {}
        for k, v in added_items:
            desc = (desc_map.get(k) or desc_map.get(k.lower()) or "").strip()
            block.append(f"# {desc if desc else 'Auto-added default setting.'}{newline}")
            block.append(f"{k} = {v}{newline}")
            added += 1

        lines[insert_at:insert_at] = block

    out_text = "".join(lines)
    if out_text == text:
        return (0, 0)

    try:
        with open(path, "w", encoding=enc, errors="replace", newline="") as f:
            f.write(out_text)
    except Exception:
        with open(path, "w", encoding="utf-8", errors="replace", newline="") as f:
            f.write(out_text)


    # Phase 1 regression check: if the write accidentally produced merged assignments,
    # immediately repair the section to avoid persisting a corrupted config.
    try:
        verify_enc = _detect_text_encoding(path)
        with open(path, "r", encoding=verify_enc, errors="replace") as vf:
            verify_text = vf.read()
        if (
            _ini_section_likely_merged_text(verify_text, section_name=section_name)
            or re.search(r"\)\.[A-Za-z_][A-Za-z0-9_]*\s*=", verify_text)
            or re.search(r"#[^\r\n]*\b[A-Za-z_][A-Za-z0-9_]*\s*=", verify_text)
        ):
            _sanitize_ini_section_formatting(path, section_name=section_name)
    except Exception:
        pass

    return (updated, added)


def write_configuration_only(cfg: configparser.ConfigParser, paths: Dict[str, str]) -> None:
    """Write only TradingBot/config/config.ini (CONFIGURATION section), preserving comments."""
    config_cfg = _clone_sections(cfg, ["CONFIGURATION"])
    if not config_cfg.has_section("CONFIGURATION"):
        config_cfg.add_section("CONFIGURATION")

    out_path = paths["configuration_ini"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # If the file exists, update in-place; otherwise create a fresh file.
    if os.path.exists(out_path):
        _update_ini_section_values_preserve_comments(
            out_path,
            section_name="CONFIGURATION",
            items=list(config_cfg.items("CONFIGURATION")),
        )
    else:
        _write_ini(config_cfg, out_path)



def ensure_split_config_layout(paths: Dict[str, str], *, force_defaults: bool = False) -> None:
    """Ensure the /config folder exists and the split INI layout is present.

    Forward-only policy (v5.14.0 Tier 2):
      - No legacy root-level config.ini migration.
      - No legacy [WATCHLIST] section migration.

    Behavior:
      - Creates missing split files from defaults.
      - Appends missing default keys into existing config.ini / keys.ini (preserving comments).
      - Ensures required WATCHLIST_* sections exist in watchlist.ini.
    """

    os.makedirs(paths["config_dir"], exist_ok=True)

    defaults = default_split_config()

    if force_defaults:
        write_split_config(defaults, paths)
        return

    split_exists = any(
        os.path.exists(p)
        for p in (
            paths["keys_ini"],
            paths["configuration_ini"],
            paths["watchlist_ini"],
            paths["strategy_ini"],
        )
    )

    if split_exists:
        # Create missing split files (do not overwrite existing ones)
        if not os.path.exists(paths["keys_ini"]):
            keys_cfg = _clone_sections(defaults, ["KEYS"])
            if not keys_cfg.has_section("KEYS"):
                keys_cfg.add_section("KEYS")
            _write_ini(keys_cfg, paths["keys_ini"])

        if not os.path.exists(paths["configuration_ini"]):
            cfg_cfg = _clone_sections(defaults, ["CONFIGURATION"])
            if not cfg_cfg.has_section("CONFIGURATION"):
                cfg_cfg.add_section("CONFIGURATION")
            _write_ini(cfg_cfg, paths["configuration_ini"])

        if not os.path.exists(paths["watchlist_ini"]):
            watch_secs = [s for s in defaults.sections() if s.upper().startswith("WATCHLIST_")]
            watch_cfg = _clone_sections(defaults, watch_secs)
            for s in WATCHLIST_SECTIONS_REQUIRED:
                if not watch_cfg.has_section(s):
                    watch_cfg.add_section(s)
            _write_ini(watch_cfg, paths["watchlist_ini"])
        else:
            # Ensure the required sections exist (keeps runtime stable if a user edited/deleted sections)
            try:
                wl = _new_config_parser()
                _read_ini_with_fallback(wl, paths["watchlist_ini"])
                changed = False
                for s in WATCHLIST_SECTIONS_REQUIRED:
                    if not wl.has_section(s):
                        wl.add_section(s)
                        changed = True
                if changed:
                    _write_ini(wl, paths["watchlist_ini"])
            except Exception:
                pass

        if not os.path.exists(paths["strategy_ini"]):
            strat_secs = [s for s in defaults.sections() if s.upper().startswith("STRATEGY_")]
            strat_cfg = _clone_sections(defaults, strat_secs)
            _write_ini(strat_cfg, paths["strategy_ini"])

        # Append NEW safe defaults into existing KEYS/CONFIGURATION without rewriting the file.
        cfg_desc = {
            "db_dir": "Directory containing split DB files (relative paths resolve from TradingBot/config/).",
            "update_db_lookback_days": "How many day(s) of 1-minute candles Update DB should pull.",
            "promotion_enabled": "Enable/disable paper -> live promotion evaluation.",
            "promotion_window_days": "How many days of paper evidence to consider for promotion evaluation.",
            "ai_v2_enabled": "Enable AI v2 pipeline (labels + calibration + walk-forward).",
            "ai_walkforward_folds": "Walk-forward folds for AI v2 (higher = slower).",
            "research_report_rows": "How many rows from backtest_results to include in strategy selection report.",

            # v5.12.7: Config tab + history tracking
            "config_auto_update_enabled": "Enable AI-driven automatic tuning of CONFIGURATION values (currently foundation only).",
            "config_history_enabled": "Record config changes into the database (config_history table).",
            "config_history_max_rows": "Max rows to keep in config_history (old rows pruned when exceeded).",

            # v5.13.1 updateB: Dynamic watchlist + crypto policy
            "watchlist_auto_update_enabled": "Enable periodic auto-update of WATCHLIST based on today's candidates (defaults OFF).",
            "watchlist_auto_update_interval_min": "Auto-update interval in minutes (periodic).",
            "watchlist_auto_update_mode": "Watchlist update mode: ADD (append top candidates) or REPLACE (overwrite non-crypto portion).",
            "watchlist_auto_update_max_add": "When mode=ADD, max number of symbols to add per run (0 = unlimited).",
            "watchlist_auto_update_max_total": "Hard cap on total watchlist size after policy is applied.",
            "watchlist_auto_update_min_score": "Minimum candidate score required to be added to watchlist.",
            "crypto_stable_set_enabled": "Enable crypto stable set selection (liquidity/spread filtered).",
            "crypto_stable_set_replace_existing": "When crypto stable set is enabled, replace existing crypto entries in the watchlist.",
            "crypto_stable_set_lookback_bars": "Lookback bars used to compute crypto liquidity proxy.",
            "crypto_stable_set_min_dollar_volume": "Minimum dollar-volume over lookback window for crypto inclusion.",
            "crypto_stable_set_max_spread_pct": "Maximum allowed spread (%). Falls back to a range-based proxy if quotes are unavailable.",
            "crypto_stable_set_max_assets": "How many crypto symbols to keep in the stable set.",

            # v5.13.2 updateA: Mini-player
            "miniplayer_enabled": "Start the UI in miniplayer mode (Live Log focused).",
            "ui_refresh_ms": "Normal UI refresh cadence for the main UI loop (milliseconds).",
            "miniplayer_ui_refresh_ms": "Miniplayer UI refresh cadence (milliseconds). Slower = lower CPU.",
        }
        keys_desc = {
            "alpaca_key_id": "Alpaca API Key ID.",
            "alpaca_secret_key": "Alpaca API Secret Key.",
            "alpaca_base_url": "Alpaca base URL (paper or live).",
        }

        # Repair any formatting damage (prevents key-lines getting concatenated / hidden)
        try:
            _sanitize_ini_section_formatting(paths["configuration_ini"], section_name="CONFIGURATION")
        except Exception:
            pass

        try:
            _append_missing_ini_options_preserve_comments(
                paths["configuration_ini"],
                section_name="CONFIGURATION",
                defaults_items=list(defaults.items("CONFIGURATION")),
                descriptions=cfg_desc,
                header_title="AUTO-ADDED DEFAULTS (preserve comments)",
            )
        except Exception:
            pass

        try:
            _append_missing_ini_options_preserve_comments(
                paths["keys_ini"],
                section_name="KEYS",
                defaults_items=list(defaults.items("KEYS")),
                descriptions=keys_desc,
                header_title="AUTO-ADDED DEFAULTS (preserve comments)",
            )
        except Exception:
            pass

        return

    # Fresh install: create all split files from defaults
    write_split_config(defaults, paths)
