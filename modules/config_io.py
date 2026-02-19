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
from pathlib import Path
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


def _keys_ini_is_legacy_duplicate_schema(keys_ini_path: str) -> bool:
    """Detect the legacy keys.ini format that repeats base_url/alpaca_key/alpaca_secret
    inside a single [KEYS] section (separated by comment headers like ALPACA_PAPER / ALPACA_LIVE).
    """
    try:
        raw = Path(keys_ini_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    low = raw.lower()
    if ("alpaca_paper" in low or "alpaca_live" in low or "telegram" in low) and low.count("base_url") >= 2:
        return True
    if ("alpaca_paper" in low or "alpaca_live" in low) and low.count("alpaca_key") >= 2:
        return True
    return False


def _parse_keys_ini_loose(keys_ini_path: str) -> dict:
    """Parse keys.ini without losing duplicated keys.

    Supports:
      1) Legacy single-section file with repeated base_url/alpaca_key/alpaca_secret blocks.
      2) Cleaner schema using distinct keys (paper_* / live_*).
      3) Optional explicit sections like [ALPACA_PAPER], [ALPACA_LIVE], [TELEGRAM], [TRADINGVIEW].

    Returns canonical keys in lowercase:
      paper_base_url, paper_alpaca_key, paper_alpaca_secret,
      live_base_url, live_alpaca_key, live_alpaca_secret,
      telegram_token, telegram_chat_id, telegram_enabled,
      tradingview_secret
    """
    try:
        raw = Path(keys_ini_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {}

    out: dict[str, str] = {}
    occ: dict[str, int] = {"base_url": 0, "alpaca_key": 0, "alpaca_secret": 0}
    current: str | None = None

    def _set(k: str, v: str):
        if v is None:
            v = ""
        out[k] = str(v).strip()

    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue

        # Optional INI sections
        if s.startswith("[") and s.endswith("]") and len(s) > 2:
            sect = s[1:-1].strip().lower()
            if "alpaca_paper" in sect or sect in ("paper", "alpaca paper", "alpaca_paper"):
                current = "paper"
            elif "alpaca_live" in sect or sect in ("live", "alpaca live", "alpaca_live"):
                current = "live"
            elif "telegram" in sect:
                current = "telegram"
            elif "tradingview" in sect:
                current = "tradingview"
            elif sect == "keys":
                current = "keys"
            else:
                current = sect
            continue

        # Legacy block headers as comments
        if s.startswith("#") or s.startswith(";"):
            low = s.lower()
            if "alpaca_paper" in low:
                current = "paper"
            elif "alpaca_live" in low:
                current = "live"
            elif "telegram" in low:
                current = "telegram"
            elif "tradingview" in low:
                current = "tradingview"
            continue

        if "=" not in s:
            continue

        k_raw, v_raw = s.split("=", 1)
        k = k_raw.strip().lower()
        v = v_raw.strip()

        if not k:
            continue

        # Clean schema
        if k in (
            "paper_base_url", "paper_alpaca_key", "paper_alpaca_secret",
            "live_base_url", "live_alpaca_key", "live_alpaca_secret",
            "telegram_token", "telegram_chat_id", "telegram_enabled",
            "tradingview_secret",
        ):
            _set(k, v)
            continue

        # Legacy TradingView secret name
        if k == "secret" and current == "tradingview":
            _set("tradingview_secret", v)
            continue

        # Legacy Alpaca keys (duplicated)
        if k in ("base_url", "alpaca_key", "alpaca_secret"):
            if current in ("paper", "live"):
                if k == "base_url":
                    _set(f"{current}_base_url", v)
                elif k == "alpaca_key":
                    _set(f"{current}_alpaca_key", v)
                else:
                    _set(f"{current}_alpaca_secret", v)
            else:
                n = occ.get(k, 0)
                occ[k] = n + 1
                prefix = "paper" if n == 0 else "live"
                if k == "base_url":
                    _set(f"{prefix}_base_url", v)
                elif k == "alpaca_key":
                    _set(f"{prefix}_alpaca_key", v)
                else:
                    _set(f"{prefix}_alpaca_secret", v)
            continue

        if k in ("telegram_token", "telegram_chat_id", "telegram_enabled"):
            _set(k, v)
            continue

    return out


def _apply_keys_to_cfg(cfg: configparser.ConfigParser, keys_map: dict, paper_trading: bool) -> None:
    """Merge keys_map into cfg's [KEYS] section and populate active base_url/alpaca_* fields."""
    if not cfg.has_section("KEYS"):
        cfg.add_section("KEYS")

    # Record environment selection for downstream consumers (e.g., broker gateway defaults).
    cfg.set("KEYS", "paper_trading", "True" if paper_trading else "False")

    canonical = [
        "paper_base_url", "paper_alpaca_key", "paper_alpaca_secret",
        "live_base_url", "live_alpaca_key", "live_alpaca_secret",
        "telegram_token", "telegram_chat_id", "telegram_enabled",
        "tradingview_secret",
    ]
    for k in canonical:
        v = keys_map.get(k)
        cfg.set("KEYS", k, str(v).strip() if v is not None else "")

    prefix = "paper" if paper_trading else "live"

    # Prefer environment-scoped keys, then fall back to legacy single-key layout.
    base_url = (
        keys_map.get(f"{prefix}_base_url")
        or cfg.get("KEYS", f"{prefix}_base_url", fallback="").strip()
        or keys_map.get("base_url")
        or cfg.get("KEYS", "base_url", fallback="").strip()
    )
    alpaca_key = (
        keys_map.get(f"{prefix}_alpaca_key")
        or cfg.get("KEYS", f"{prefix}_alpaca_key", fallback="").strip()
        or keys_map.get("alpaca_key")
        or cfg.get("KEYS", "alpaca_key", fallback="").strip()
    )
    alpaca_secret = (
        keys_map.get(f"{prefix}_alpaca_secret")
        or cfg.get("KEYS", f"{prefix}_alpaca_secret", fallback="").strip()
        or keys_map.get("alpaca_secret")
        or cfg.get("KEYS", "alpaca_secret", fallback="").strip()
    )

    cfg.set("KEYS", "base_url", str(base_url or "").strip())
    cfg.set("KEYS", "alpaca_key", str(alpaca_key or "").strip())
    cfg.set("KEYS", "alpaca_secret", str(alpaca_secret or "").strip())

    # Bridge TradingView secret for callers still reading [TRADINGVIEW].secret
    tv = keys_map.get("tradingview_secret") or cfg.get("KEYS", "tradingview_secret", fallback="").strip()
    if tv:
        if not cfg.has_section("TRADINGVIEW"):
            cfg.add_section("TRADINGVIEW")
        if not cfg.get("TRADINGVIEW", "secret", fallback="").strip():
            cfg.set("TRADINGVIEW", "secret", tv)


def load_split_config(paths: Dict[str, str]) -> configparser.ConfigParser:
    """Load split .ini configs with robust keys.ini parsing.

    keys.ini historically used repeated key names inside [KEYS] (paper/live blocks). Python's
    configparser collapses duplicates, which can make the app think credentials are missing.
    We load keys.ini with a loose parser and then materialize both:
      - canonical paper_* / live_* keys
      - active base_url/alpaca_key/alpaca_secret keys (selected by paper_trading)

    This keeps the rest of the codebase (and older validators/tests) working without requiring
    a destructive rewrite of existing keys.ini files.
    """
    cfg = _new_config_parser()

    configuration_ini = paths.get("configuration_ini", "")
    watchlist_ini = paths.get("watchlist_ini", "")
    strategy_ini = paths.get("strategy_ini", "")
    keys_ini = paths.get("keys_ini", "")

    # Guardrail: repair merged/corrupted CONFIGURATION lines before parsing.
    # This prevents configparser from swallowing extra assignments as part of a value.
    try:
        sanitize_configuration_ini_if_needed(paths, section_name="CONFIGURATION")
    except Exception:
        pass

    # Read config.ini first so we can select paper/live keys.
    _read_ini_with_fallback(cfg, configuration_ini)

    # Keep config.ini human-friendly: TradingView keys are stored in [CONFIGURATION]
    # with comment headers. Hydrate a runtime [TRADINGVIEW] view when absent.
    if not cfg.has_section("TRADINGVIEW"):
        cfg.add_section("TRADINGVIEW")
    for k in (
        "enabled",
        "listen_host",
        "listen_port",
        "secret",
        "allowed_signals",
        "mode",
        "candidate_cooldown_minutes",
        "autovalidation_enabled",
        "autovalidation_cooldown_minutes",
        "autovalidation_freshness_minutes",
        "autovalidation_backfill_days",
        "autovalidation_backtest_days",
        "autovalidation_max_strategies",
        "autovalidation_min_trades",
        "autovalidation_max_concurrency",
    ):
        if not cfg.get("TRADINGVIEW", k, fallback="").strip():
            v = cfg.get("CONFIGURATION", k, fallback="").strip()
            if v != "":
                cfg.set("TRADINGVIEW", k, v)

    _read_ini_with_fallback(cfg, watchlist_ini)
    _read_ini_with_fallback(cfg, strategy_ini)

    # Determine paper/live selection (support older key name 'paper' as fallback)
    try:
        paper_trading = cfg.getboolean("CONFIGURATION", "paper_trading", fallback=cfg.getboolean("CONFIGURATION", "paper", fallback=True))
    except Exception:
        paper_trading = True

    # Load keys.ini without losing duplicates
    if keys_ini and os.path.exists(keys_ini):
        keys_map = _parse_keys_ini_loose(keys_ini)
        _apply_keys_to_cfg(cfg, keys_map, paper_trading)
    else:
        if not cfg.has_section("KEYS"):
            cfg.add_section("KEYS")

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
    section_name: str,
    defaults_items: list[tuple[str, str]],
    descriptions: Dict[str, str] | None = None,
    header_title: str = "DEFAULTS SYNC (preserve comments)",
) -> int:
    """Add missing key/value pairs into the *target section* (not at EOF), preserving comments.

    - Does NOT overwrite existing values.
    - Preserves file structure and user edits.
    - Inserts before the next [SECTION] header.

    Returns the number of keys written.
    """
    if not os.path.exists(path):
        return 0

    section = (section_name or "").strip()
    if not section:
        return 0

    import re as _re

    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines(True)

    sec_re = _re.compile(r"^\s*\[\s*" + _re.escape(section) + r"\s*\]\s*$", _re.IGNORECASE)
    any_sec_re = _re.compile(r"^\s*\[[^\]]+\]\s*$")

    start_idx = None
    for i, ln in enumerate(lines):
        if sec_re.match(ln.strip()):
            start_idx = i
            break
    if start_idx is None:
        return 0

    end_idx = len(lines)
    for j in range(start_idx + 1, len(lines)):
        if any_sec_re.match(lines[j].strip()):
            end_idx = j
            break

    existing = set()
    for ln in lines[start_idx + 1 : end_idx]:
        s = ln.strip()
        if not s or s.startswith('#') or s.startswith(';'):
            continue
        if '=' not in s:
            continue
        k = s.split('=', 1)[0].strip().lower()
        if k:
            existing.add(k)

    required = [(k, v) for (k, v) in (defaults_items or []) if isinstance(k, str)]
    missing = [(k, v) for (k, v) in required if k.strip().lower() not in existing]
    if not missing:
        return 0

    descriptions = descriptions or {}
    insert_lines: list[str] = []
    insert_lines.append("\n")
    insert_lines.append("#  ==========================\n")
    insert_lines.append(f"#  {header_title}\n")
    insert_lines.append("#  ==========================\n")

    for k, v in missing:
        desc = descriptions.get(k, "Auto-added default setting.")
        insert_lines.append(f"# {desc}\n")
        insert_lines.append(f"{k} = {v}\n")
        insert_lines.append("\n")

    new_lines = lines[:end_idx] + insert_lines + lines[end_idx:]
    Path(path).write_text(''.join(new_lines), encoding='utf-8')
    return len(missing)


def _remove_keys_from_ini_section_preserve_comments(
    path: str,
    section_name: str,
    keys_to_remove: list[str],
) -> int:
    """Remove specific key=value lines from a section while preserving comments and other text."""
    if not os.path.exists(path):
        return 0

    section = (section_name or "").strip()
    if not section or not keys_to_remove:
        return 0

    import re as _re

    keys_lc = {k.strip().lower() for k in keys_to_remove if isinstance(k, str) and k.strip()}
    if not keys_lc:
        return 0

    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines(True)

    sec_re = _re.compile(r"^\s*\[\s*" + _re.escape(section) + r"\s*\]\s*$", _re.IGNORECASE)
    any_sec_re = _re.compile(r"^\s*\[[^\]]+\]\s*$")

    start_idx = None
    for i, ln in enumerate(lines):
        if sec_re.match(ln.strip()):
            start_idx = i
            break
    if start_idx is None:
        return 0

    end_idx = len(lines)
    for j in range(start_idx + 1, len(lines)):
        if any_sec_re.match(lines[j].strip()):
            end_idx = j
            break

    removed = 0
    out = []
    out.extend(lines[: start_idx + 1])

    for ln in lines[start_idx + 1 : end_idx]:
        s = ln.strip()
        if s and not s.startswith('#') and not s.startswith(';') and '=' in s:
            k = s.split('=', 1)[0].strip().lower()
            if k in keys_lc:
                removed += 1
                continue
        out.append(ln)

    out.extend(lines[end_idx:])
    if removed:
        Path(path).write_text(''.join(out), encoding='utf-8')
    return removed


def _update_ini_section_values_preserve_comments(
    path: str,
    *,
    section_name: str,
    items: List[Tuple[str, str]],
    header_title: str = "DEFAULTS SYNC (preserve comments)",
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
    _repair_legacy_db_folder_layout(paths)

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
            cfg_cfg = _clone_sections(defaults, ["CONFIGURATION", "TRADINGVIEW"])
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

        # Keep file stable: do not auto-append defaults into existing user INI files.
        cfg_desc = {
            "db_dir": "Directory containing split DB files (relative paths resolve from TradingBot/config/).",
            "update_db_lookback_days": "How many day(s) of 1-minute candles Update DB should pull.",
            "promotion_enabled": "Enable/disable paper -> live promotion evaluation.",
            "promotion_window_days": "How many days of paper evidence to consider for promotion evaluation.",
            "agent_autopilot_enabled": "Start AgentMaster scheduler/services automatically at app startup (explicit opt-in).",
            "agent_autopilot_require_engine_running_for_mutations": "If True, autopilot mutation jobs are blocked while engine is stopped.",
            "agent_candidate_scan_enabled": "Enable scheduled headless candidate scans by AgentMaster.",
            "agent_candidate_scan_interval_minutes": "How often AgentMaster runs headless candidate scans (minutes).",
            "agent_candidate_simulation_enabled": "Enable scheduled candidate simulation/scoring on latest candidates.",
            "agent_candidate_simulation_interval_minutes": "How often AgentMaster runs candidate simulation/scoring (minutes).",
            "agent_candidate_simulation_max_symbols": "Maximum number of latest candidate symbols to rescore per simulation run.",
            "agent_candidate_simulation_run_after_scan": "If True, trigger simulation immediately after candidate_scan_completed event.",
            "agent_watchlist_policy_enabled": "Enable scheduled watchlist policy updates from AgentMaster.",
            "agent_watchlist_policy_interval_minutes": "How often AgentMaster runs watchlist policy update (minutes).",
            "agent_watchlist_policy_max_churn_per_run": "Soft churn budget per policy run (added+removed symbols).",
            "agent_watchlist_policy_run_after_simulation": "If True, trigger watchlist policy update after candidate_simulation_completed.",
            "agent_quick_backtest_enabled": "Enable scheduled quick backtests for top candidate symbols.",
            "agent_quick_backtest_interval_minutes": "How often AgentMaster runs quick backtests (minutes).",
            "agent_quick_backtest_max_symbols": "Maximum number of latest candidate symbols to quick-backtest per run.",
            "agent_quick_backtest_days": "Lookback window in days used for quick backtests.",
            "agent_quick_backtest_max_strategies": "Maximum number of champion strategies evaluated in quick backtests.",
            "agent_quick_backtest_min_trades": "Minimum trade count required for strategy selection in quick backtests.",
            "agent_full_backtest_enabled": "Enable scheduled full backtest sweeps (off-hours recommended).",
            "agent_full_backtest_interval_minutes": "How often AgentMaster attempts full backtest sweeps (minutes).",
            "agent_full_backtest_hour_utc": "UTC hour [0..23] when full backtests are allowed to execute.",
            "agent_shadow_enabled": "Enable AgentShadow artifact scanner/proposal loop (PAPER-first, no direct execution).",
            "agent_shadow_interval_sec": "How often AgentShadow scans artifacts (seconds).",
            "agent_shadow_include_summaries": "If True, AgentShadow scans logs/summaries artifacts.",
            "agent_shadow_include_backtests": "If True, AgentShadow scans logs/backtest artifacts.",
            "agent_stale_quarantine_equity_market_hours_only": "If True, use market-hours-aware stale quarantine logic for equities.",
            "agent_stale_quarantine_equity_market_open_hour_utc": "UTC hour [0..23] when equity stale market-hours gate opens.",
            "agent_stale_quarantine_equity_market_close_hour_utc": "UTC hour [0..23] when equity stale market-hours gate closes.",
            "agent_stale_quarantine_equity_after_hours_threshold_seconds": "Stale threshold for equities outside market-hours.",
            "agent_stale_quarantine_crypto_threshold_seconds": "Stale threshold for crypto symbols (24/7).",
            "agent_stale_quarantine_max_per_day": "Maximum symbols stale-quarantined per day.",
            "agent_stale_quarantine_cooldown_minutes": "Cooldown in minutes after stale quarantine action.",
            "agent_stale_quarantine_warmup_minutes": "Startup grace window before stale-symbol reporting can evaluate symbols (missing history remains uninitialized).",
            "ai_v2_enabled": "Enable AI v2 pipeline (labels + calibration + walk-forward).",
            "ai_walkforward_folds": "Walk-forward folds for AI v2 (higher = slower).",
            "ai_runtime_training_enabled": "Start background AI training thread when TradingEngine initializes (optional AI dependencies required).",
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
            "crypto_stable_set_max_spread_pct": "Maximum allowed bid/ask spread (%) when spread mode is BIDASK.",
            "crypto_stable_set_spread_proxy_mode": "Spread filter mode for crypto stable-set: RANGE_PCT, BIDASK, or OFF.",
            "crypto_stable_set_max_range_pct": "Maximum allowed candle-range proxy (%) when spread mode is RANGE_PCT.",
            "crypto_stable_set_max_assets": "How many crypto symbols to keep in the stable set.",

            # v5.13.2 updateA: Mini-player
            "miniplayer_enabled": "Start the UI in miniplayer mode (Live Log focused).",
            "ui_refresh_ms": "Normal UI refresh cadence for the main UI loop (milliseconds).",
            "miniplayer_ui_refresh_ms": "Miniplayer UI refresh cadence (milliseconds). Slower = lower CPU.",
        }

        tv_desc = {
            "enabled": "Enable TradingView webhook receiver (stdlib HTTP server).",
            "listen_host": "Bind address for the webhook server (recommend 127.0.0.1).",
            "listen_port": "Bind port for the webhook server (1-65535).",
            "secret": "Shared secret required for auth (strict mode requires when enabled).",
            "allowed_signals": "Optional CSV allow-list (e.g. BUY,SELL). Empty = allow all.",
            "mode": "OFF|ADVISORY|PAPER|LIVE (pipeline stage; v5.14.5 persists alerts only).",
        }

        keys_desc = {
            "paper_base_url": "Paper trading Alpaca base URL.",
            "paper_alpaca_key": "Paper trading Alpaca API key.",
            "paper_alpaca_secret": "Paper trading Alpaca API secret.",
            "live_base_url": "Live trading Alpaca base URL.",
            "live_alpaca_key": "Live trading Alpaca API key.",
            "live_alpaca_secret": "Live trading Alpaca API secret.",
            "telegram_token": "Telegram bot token (optional).",
            "telegram_chat_id": "Telegram chat id (optional).",
            "telegram_enabled": "Enable Telegram notifications.",
            "tradingview_secret": "TradingView webhook shared secret (kept in keys.ini).",
        }


        # Repair any formatting damage (prevents key-lines getting concatenated / hidden)
        try:
            _sanitize_ini_section_formatting(paths["configuration_ini"], section_name="CONFIGURATION")
        except Exception:
            pass

        # Do not mutate existing config.ini/keys.ini by appending missing defaults.
        # This preserves user formatting and line count stability across app runs.

        # ---- Hotfix v5.16.2: repair misfiled keys from prior patch runs ----
        # 1) Remove non-secret config keys that were incorrectly injected into keys.ini
        try:
            _remove_keys_from_ini_section_preserve_comments(paths["keys_ini"], "KEYS", ["data_feed"])
        except Exception:
            pass

        # 2) Remove obsolete/legacy keys that were incorrectly appended into the TRADINGVIEW section
        try:
            _remove_keys_from_ini_section_preserve_comments(
                paths["configuration_ini"],
                "TRADINGVIEW",
                ["paper", "allow_live", "webhook_secret"],
            )
        except Exception:
            pass

        # 3) One-way migration: move TRADINGVIEW.secret -> KEYS.tradingview_secret (copy then blank)
        try:
            cfg_now = _new_config_parser()
            _read_ini_with_fallback(cfg_now, paths["configuration_ini"])
            secret = (
                cfg_now.get("TRADINGVIEW", "secret", fallback="")
                or cfg_now.get("CONFIGURATION", "secret", fallback="")
                or ""
            ).strip()

            keys_now = _new_config_parser()
            _read_ini_with_fallback(keys_now, paths["keys_ini"])
            keys_secret = (keys_now.get("KEYS", "tradingview_secret", fallback="") or "").strip()

            if secret and not keys_secret:
                _update_ini_section_values_preserve_comments(
                    paths["keys_ini"],
                    section_name="KEYS",
                    items=[("tradingview_secret", secret)],
                    descriptions={"tradingview_secret": "TradingView webhook shared secret (kept in keys.ini)."},
                    header_title="TRADINGVIEW (secret)",
                )

                _update_ini_section_values_preserve_comments(
                    paths["configuration_ini"],
                    section_name="TRADINGVIEW",
                    items=[("secret", "")],
                    descriptions={"secret": "TradingView webhook secret moved to keys.ini (tradingview_secret)."},
                    header_title="TRADINGVIEW (Webhook Integration)",
                )
        except Exception:
            pass


        return

    # Fresh install: create all split files from defaults
    write_split_config(defaults, paths)


def _repair_legacy_db_folder_layout(paths: Dict[str, str]) -> None:
    """Move mistakenly nested config/db files back to root-level db/.

    Some prior patch flows created DB files under TradingBot/config/db.
    Runtime DB resolution is root-scoped (TradingBot/db), so we auto-heal this
    layout to prevent split-mode startup failures.
    """
    try:
        config_dir = paths.get("config_dir") or ""
        root_dir = paths.get("root") or os.path.dirname(config_dir)
        if not config_dir or not root_dir:
            return

        legacy_dir = os.path.join(config_dir, "db")
        canonical_dir = paths.get("db_dir") or os.path.join(root_dir, "db")
        if not os.path.isdir(legacy_dir):
            return

        os.makedirs(canonical_dir, exist_ok=True)

        moved_any = False
        for name in os.listdir(legacy_dir):
            src = os.path.join(legacy_dir, name)
            dst = os.path.join(canonical_dir, name)
            if not os.path.isfile(src):
                continue
            if os.path.exists(dst):
                continue
            shutil.move(src, dst)
            moved_any = True

        if moved_any:
            try:
                if not os.listdir(legacy_dir):
                    os.rmdir(legacy_dir)
            except Exception:
                pass
    except Exception:
        # Best-effort repair only; never block startup.
        return

def _ensure_ini_section_exists(
    ini_path: str,
    section_name: str,
    defaults_items: List[Tuple[str, str]],
    descriptions: Optional[Dict[str, str]] = None,
    header_title: str = "",
) -> bool:
    """Ensure `ini_path` contains a section header `[section_name]`.

    If the section is missing, append it at the end of the file along with
    `defaults_items` (and optional comment descriptions). Returns True if the
    file was modified.
    """
    p = Path(ini_path)
    if not p.exists():
        return False

    section = section_name

    try:
        enc = _detect_text_encoding(str(p))
    except Exception:
        enc = "utf-8"

    try:
        text = p.read_text(encoding=enc, errors="ignore")
    except Exception:
        text = p.read_text(encoding="utf-8", errors="ignore")

    header_re = re.compile(r"^\s*\[" + re.escape(section) + r"\]\s*$", re.IGNORECASE | re.MULTILINE)
    if header_re.search(text):
        return False

    block: List[str] = []
    if text and not text.endswith("\n"):
        text += "\n"

    block.append("")
    block.append(f"[{section}]")
    title = header_title.strip() or section
    block.append(f"# {title}")

    desc = descriptions or {}
    for k, v in defaults_items:
        d = desc.get(k)
        if d:
            block.append(f"# {d}")
        block.append(f"{k} = {v}")
        block.append("")

    new_text = text + "\n".join(block).rstrip() + "\n"

    try:
        p.write_text(new_text, encoding=enc)
    except Exception:
        p.write_text(new_text, encoding="utf-8")

    return True
