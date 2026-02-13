"""UI tab: Config.

v5.12.7 updateA (foundation):
- View/edit CONFIGURATION values from config.ini
- Backup config.ini before writes
- Log config changes to DB (config_history table)

Notes:
- This tab is intentionally "CONFIGURATION only" (does not edit KEYS/WATCHLIST/STRATEGY).
- "Update Automatically" is a toggle only (foundation); no autonomous tuning is performed here.
"""

from __future__ import annotations

import configparser
import os
import shutil
from datetime import datetime
from typing import Dict, Optional

import customtkinter as ctk
from tkinter import ttk, messagebox

from ..config_io import write_configuration_only


def _as_bool(value: object, default: bool = False) -> bool:
    try:
        s = str(value).strip().lower()
        return s in ("1", "true", "yes", "y", "on")
    except Exception:
        return default


class ConfigTab:
    def __init__(self, parent, db, config, paths: Dict[str, str], log_fn=None):
        self.parent = parent
        self.db = db
        self.config = config
        self.paths = paths or {}
        self._log = log_fn if callable(log_fn) else (lambda _msg: None)

        self._entries: Dict[str, ctk.CTkEntry] = {}
        self._history_box: Optional[ctk.CTkTextbox] = None

        self._var_auto = ctk.BooleanVar(value=False)
        self._var_hist = ctk.BooleanVar(value=True)

        self._max_rows_entry: Optional[ctk.CTkEntry] = None
        self._status: Optional[ctk.CTkLabel] = None

        self.setup_ui()
        self._load_from_config()
        self.refresh_history()

    # --------------------
    # UI
    # --------------------
    def setup_ui(self):
        header = ctk.CTkFrame(self.parent)
        header.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(header, text="Config (CONFIGURATION)", font=("Arial", 16, "bold")).pack(side="left", padx=10)

        # Toggles (write back into entries)
        self._auto_toggle = ctk.CTkSwitch(
            header,
            text="Update Automatically",
            variable=self._var_auto,
            command=self._on_toggle_auto,
        )
        self._auto_toggle.pack(side="left", padx=10)

        self._hist_toggle = ctk.CTkSwitch(
            header,
            text="Record History",
            variable=self._var_hist,
            command=self._on_toggle_history,
        )
        self._hist_toggle.pack(side="left", padx=10)

        ctk.CTkLabel(header, text="Max Rows:").pack(side="left", padx=(10, 4))
        self._max_rows_entry = ctk.CTkEntry(header, width=90)
        self._max_rows_entry.pack(side="left")
        self._max_rows_entry.bind("<FocusOut>", lambda _e: self._on_max_rows_changed())

        ctk.CTkButton(header, text="Reload", width=110, command=self._reload_from_disk).pack(side="right", padx=10)
        ctk.CTkButton(header, text="UPDATE CONFIG", width=150, fg_color="#7B1FA2", command=self._apply_changes).pack(side="right", padx=10)

        self._status = ctk.CTkLabel(self.parent, text="Ready.", font=("Arial", 12))
        self._status.pack(fill="x", padx=10, pady=(0, 6))

        body = ctk.CTkFrame(self.parent)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Left: config editor
        self._editor = ctk.CTkScrollableFrame(body)
        self._editor.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Right: history
        hist_frame = ctk.CTkFrame(body, width=420)
        hist_frame.pack(side="right", fill="y")
        try:
            hist_frame.pack_propagate(False)
        except Exception:
            pass

        hhdr = ctk.CTkFrame(hist_frame)
        hhdr.pack(fill="x", padx=8, pady=8)
        ctk.CTkLabel(hhdr, text="Config History", font=("Arial", 14, "bold")).pack(side="left")
        ctk.CTkButton(hhdr, text="Refresh", width=90, command=self.refresh_history).pack(side="right")

        self._history_box = ctk.CTkTextbox(hist_frame, width=420)
        self._history_box.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        try:
            self._history_box.configure(state="disabled")
        except Exception:
            pass

    # --------------------
    # Data Binding
    # --------------------
    def _ensure_section(self):
        try:
            if not self.config.has_section("CONFIGURATION"):
                self.config.add_section("CONFIGURATION")
        except Exception:
            try:
                self.config["CONFIGURATION"] = {}
            except Exception:
                pass

    def _load_from_config(self):
        """Build rows and populate entries from the in-memory config."""
        self._ensure_section()
        cfg = self.config["CONFIGURATION"]

        # Pull toggle values from config
        self._var_auto.set(_as_bool(cfg.get("config_auto_update_enabled", "False"), False))
        self._var_hist.set(_as_bool(cfg.get("config_history_enabled", "True"), True))
        if self._max_rows_entry is not None:
            self._max_rows_entry.delete(0, "end")
            self._max_rows_entry.insert(0, str(cfg.get("config_history_max_rows", "5000")))

        # Build rows once
        if not self._entries:
            LEGACY_CONFIG_KEYS = {'db_mode', 'db_path', 'db_auto_migrate_on_startup'}

            keys = sorted([k for k in cfg.keys() if k not in LEGACY_CONFIG_KEYS])
            if not keys:
                keys = [
                    "db_dir",
                    "update_db_lookback_days",
                    "config_auto_update_enabled",
                    "config_history_enabled",
                    "config_history_max_rows",
                ]

            for k in keys:
                row = ctk.CTkFrame(self._editor)
                row.pack(fill="x", padx=6, pady=4)
                ctk.CTkLabel(row, text=k, width=260, anchor="w").pack(side="left", padx=(4, 8))
                ent = ctk.CTkEntry(row)
                ent.pack(side="left", fill="x", expand=True, padx=(0, 4))
                self._entries[k] = ent

        # Set values
        for k, ent in self._entries.items():
            try:
                val = cfg.get(k, "")
            except Exception:
                val = ""
            ent.delete(0, "end")
            ent.insert(0, str(val))

        # Keep special values synced
        self._sync_special_entries_from_toggles()

    def _sync_special_entries_from_toggles(self):
        self._set_entry_value("config_auto_update_enabled", "True" if self._var_auto.get() else "False")
        self._set_entry_value("config_history_enabled", "True" if self._var_hist.get() else "False")
        if self._max_rows_entry is not None:
            self._set_entry_value("config_history_max_rows", self._max_rows_entry.get().strip())

    def _set_entry_value(self, key: str, value: str):
        ent = self._entries.get(key)
        if ent is None:
            return
        try:
            ent.delete(0, "end")
            ent.insert(0, str(value))
        except Exception:
            pass

    # --------------------
    # Actions
    # --------------------
    def _set_status(self, text: str):
        try:
            if self._status is not None:
                self._status.configure(text=str(text))
        except Exception:
            pass

    def _on_toggle_auto(self):
        self._sync_special_entries_from_toggles()

    def _on_toggle_history(self):
        self._sync_special_entries_from_toggles()

    def _on_max_rows_changed(self):
        self._sync_special_entries_from_toggles()

    def _reload_from_disk(self):
        """Reload config.ini from disk and update entries (does not touch KEYS/WATCHLIST/STRATEGY)."""
        path = (self.paths or {}).get("configuration_ini")
        if not path or not os.path.exists(path):
            self._set_status("config.ini not found on disk.")
            return

        cp = configparser.ConfigParser()
        cp.optionxform = str
        try:
            cp.read(path, encoding="utf-8")
        except Exception:
            try:
                cp.read(path)
            except Exception:
                self._set_status("Failed to read config.ini.")
                return

        if not cp.has_section("CONFIGURATION"):
            self._set_status("config.ini missing [CONFIGURATION] section.")
            return

        # Update in-memory config section only
        self._ensure_section()
        for k, v in cp.items("CONFIGURATION"):
            try:
                self.config.set("CONFIGURATION", k, v)
            except Exception:
                try:
                    self.config["CONFIGURATION"][k] = v
                except Exception:
                    pass

        self._load_from_config()
        self._set_status("Reloaded config.ini from disk.")

    def _backup_config_ini(self) -> Optional[str]:
        """Backup config.ini to TradingBot/backups/config/<timestamp>/"""
        try:
            src = (self.paths or {}).get("configuration_ini")
            if not src or not os.path.exists(src):
                return None

            backup_root = (self.paths or {}).get("backup")
            if not backup_root:
                return None

            ts = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
            dst_dir = os.path.join(backup_root, "config", ts)
            os.makedirs(dst_dir, exist_ok=True)

            dst = os.path.join(dst_dir, "config.ini")
            shutil.copy2(src, dst)
            return dst_dir
        except Exception:
            return None

    def _apply_changes(self):
        """Apply UI values -> config, backup, write config.ini (preserving comments), log history."""
        self._ensure_section()

        # Sync the special entries first so toggles are captured
        self._sync_special_entries_from_toggles()

        changed = []
        cfg_sec = self.config["CONFIGURATION"]

        for k, ent in self._entries.items():
            new_val = (ent.get() or "").strip()
            old_val = str(cfg_sec.get(k, "")).strip()
            if new_val != old_val:
                changed.append((k, old_val, new_val))
                try:
                    self.config.set("CONFIGURATION", k, new_val)
                except Exception:
                    try:
                        self.config["CONFIGURATION"][k] = new_val
                    except Exception:
                        pass

        if not changed:
            self._set_status("No changes.")
            return

        bdir = self._backup_config_ini()
        if bdir:
            self._log(f"[CONFIG] ✅ Backed up config.ini to: {bdir}")
        else:
            self._log("[CONFIG] ⚠️ Backup failed (continuing).")

        # Persist (preserve # formatting in config.ini)
        try:
            write_configuration_only(self.config, self.paths)
        except Exception as e:
            self._set_status(f"Save failed: {e}")
            return

        # DB history (best-effort)
        try:
            for k, ov, nv in changed:
                try:
                    self.db.log_config_change(cfg_key=k, old_value=ov, new_value=nv, source="manual")
                except Exception:
                    pass
        except Exception:
            pass

        self.refresh_history()
        self._set_status(f"Saved {len(changed)} change(s).")

    def refresh_history(self):
        if self._history_box is None:
            return

        rows = []
        try:
            rows = self.db.get_config_history(limit=200) or []
        except Exception:
            rows = []

        text_lines = []
        for r in rows:
            try:
                ts, key, ov, nv, src = r
            except Exception:
                continue
            text_lines.append(f"{ts} | {key} | {ov} -> {nv} | {src}")

        if not text_lines:
            text_lines = ["(no history yet)"]

        try:
            self._history_box.configure(state="normal")
        except Exception:
            pass
        try:
            self._history_box.delete("1.0", "end")
            self._history_box.insert("1.0", "\n".join(text_lines))
        except Exception:
            pass
        try:
            self._history_box.configure(state="disabled")
        except Exception:
            pass


    # ---------------- Agent Shadow Mode -----------------

    def refresh_agent(self):
        try:
            rows = self.db.get_agent_suggestions(limit=250)
        except Exception as e:
            messagebox.showerror("Agent", f"Failed to load suggestions: {e}")
            return

        for iid in self.agent_tree.get_children():
            self.agent_tree.delete(iid)

        for (sid, created_at, title, suggestion_type, status, artifact_type, artifact_path) in rows:
            self.agent_tree.insert("", "end", iid=str(sid), values=(sid, created_at, title, suggestion_type, status))

        self.agent_details.delete("1.0", "end")

    def _on_agent_select(self, _event=None):
        sel = self.agent_tree.selection()
        if not sel:
            return
        sid = int(sel[0])
        detail = self.db.get_agent_suggestion_detail(sid)
        if not detail:
            return
        (row, rats) = detail
        (sid, created_at, title, suggestion_type, status, artifact_type, artifact_path, suggestion_json, applied_at, applied_by) = row

        try:
            payload = json.loads(suggestion_json or "{}")
        except Exception:
            payload = {}

        lines = []
        lines.append(f"ID: {sid}")
        lines.append(f"Created: {created_at}")
        lines.append(f"Status: {status}")
        if applied_at:
            lines.append(f"Applied At: {applied_at} ({applied_by or ''})")
        lines.append(f"Type: {suggestion_type}")
        lines.append(f"Artifact: {artifact_type} | {artifact_path}")
        lines.append("")
        lines.append(title)
        lines.append("")

        if isinstance(payload, dict) and payload.get("config_changes"):
            lines.append("Proposed config changes:")
            for k, v in payload.get("config_changes", {}).items():
                lines.append(f"  - {k} = {v}")
            lines.append("")

        if rats:
            lines.append("Rationales:")
            for (r_created, rationale, metrics_json) in rats:
                lines.append(f"[{r_created}] {rationale}")
                try:
                    mj = json.loads(metrics_json or "{}")
                    if mj:
                        lines.append(f"  metrics: {mj}")
                except Exception:
                    pass
            lines.append("")

        self.agent_details.delete("1.0", "end")
        self.agent_details.insert("end", "\n".join(lines))

    def _get_selected_agent_id(self):
        sel = self.agent_tree.selection()
        if not sel:
            return None
        try:
            return int(sel[0])
        except Exception:
            return None

    def apply_selected_agent(self):
        sid = self._get_selected_agent_id()
        if sid is None:
            return

        detail = self.db.get_agent_suggestion_detail(sid)
        if not detail:
            return
        row, _rats = detail
        suggestion_json = row[7]
        suggestion_type = row[3]

        if suggestion_type != "config_change":
            messagebox.showinfo("Agent", "This suggestion is not a config change.")
            return

        try:
            payload = json.loads(suggestion_json or "{}")
        except Exception:
            payload = {}

        changes = payload.get("config_changes") or {}
        if not changes:
            messagebox.showinfo("Agent", "No config changes provided in this suggestion.")
            return

        if not messagebox.askyesno("Apply Agent Suggestion", "Apply selected config change(s) now?\n\nThis does NOT place trades."):
            return

        try:
            for key, val in changes.items():
                if "CONFIGURATION" not in self.config:
                    self.config["CONFIGURATION"] = {}
                old_val = self.config["CONFIGURATION"].get(key)
                self.config["CONFIGURATION"][key] = str(val)
                # Log change
                try:
                    self.db.log_config_change(key, str(old_val), str(val), source="agent")
                except Exception:
                    pass

            # Persist
            write_configuration_only(self.config, "Agent suggestion applied", paths=self.paths)

            # Update UI entries
            for key, val in changes.items():
                if key in self.entries:
                    self.entries[key].delete(0, "end")
                    self.entries[key].insert(0, str(val))

            self.db.set_agent_suggestion_status(sid, "APPLIED", applied_by="config_tab")
            self.refresh_agent()
            self.refresh_history()
            messagebox.showinfo("Agent", "Suggestion applied. (Manual approval recorded)")
        except Exception as e:
            messagebox.showerror("Agent", f"Failed to apply suggestion: {e}")

    def ignore_selected_agent(self):
        sid = self._get_selected_agent_id()
        if sid is None:
            return
        try:
            self.db.set_agent_suggestion_status(sid, "IGNORED", applied_by="config_tab")
            self.refresh_agent()
        except Exception as e:
            messagebox.showerror("Agent", f"Failed to ignore: {e}")
