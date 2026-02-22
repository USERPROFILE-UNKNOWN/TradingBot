"""UI tab: Today's Candidates (v5.13.1 Update A)."""

import customtkinter as ctk
import threading
import json
import time
import uuid
import urllib.request
import urllib.error
from tkinter import ttk, messagebox
from datetime import datetime, timezone

from ..research.candidate_scanner import CandidateScanner


class CandidatesTab:
    def __init__(self, parent, engine, db, config):
        self.parent = parent
        self.engine = engine
        self.db = db
        self.config = config

        try:
            self.app = parent.winfo_toplevel()
        except Exception:
            self.app = None

        self._call_ui = getattr(self.app, "call_ui", None)
        if not callable(self._call_ui):
            self._call_ui = lambda func, *a, **k: func(*a, **k)

        self._log = getattr(self.app, "log", print)
        self._scanner = CandidateScanner(self.db, self.config, log=self._log)

        self._selected_symbol = None
        self._last_scan_id = None

        self.setup_ui()
        self.refresh_from_db()

    def setup_ui(self):
        top = ctk.CTkFrame(self.parent)
        top.pack(fill="x", padx=10, pady=10)

        # Use grid for resilience under scaling (prevents button clipping).
        top.grid_columnconfigure(0, weight=1)
        top.grid_columnconfigure(1, weight=0)
        top.grid_columnconfigure(2, weight=0)

        left = ctk.CTkFrame(top, fg_color="transparent")
        left.grid(row=0, column=0, sticky="w", padx=10, pady=8)

        ctk.CTkLabel(left, text="Today's Candidates", font=("Arial", 16, "bold")).grid(row=0, column=0, sticky="w")
        self.lbl_status = ctk.CTkLabel(left, text="Ready", font=("Arial", 12))
        self.lbl_status.grid(row=0, column=1, sticky="w", padx=(15, 0))

        # Candidate Scanner toggle (replaces the old bottom "Tip" line)
        scanner = ctk.CTkFrame(top, fg_color="transparent")
        scanner.grid(row=0, column=1, sticky="e", padx=10, pady=8)
        ctk.CTkLabel(scanner, text="CANDIDATE SCANNER", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="e")

        self._var_scanner_enabled = ctk.BooleanVar(value=self._get_config_bool("candidate_scanner_enabled", default=False))
        self._switch_scanner = ctk.CTkSwitch(scanner, text="", variable=self._var_scanner_enabled, command=self._on_scanner_toggle)
        self._switch_scanner.grid(row=0, column=1, padx=(8, 0))

        self._lbl_scanner_state = ctk.CTkLabel(scanner, text="", font=("Arial", 12))
        self._lbl_scanner_state.grid(row=0, column=2, padx=(6, 0), sticky="w")

        # Buttons: fixed 2-row grid to avoid clipping at common window widths.
        btns = ctk.CTkFrame(top, fg_color="transparent")
        btns.grid(row=0, column=2, sticky="e", padx=10, pady=8)

        self.btn_refresh = ctk.CTkButton(btns, text="REFRESH", command=self.refresh_from_db)
        self.btn_apply_policy = ctk.CTkButton(btns, text="APPLY WATCHLIST POLICY", fg_color="#455A64", command=self.apply_watchlist_policy)
        self.btn_add_watchlist = ctk.CTkButton(btns, text="ADD TO WATCHLIST", fg_color="#00695C", command=self.add_selected_to_watchlist)
        self.btn_backtest = ctk.CTkButton(btns, text="BACKTEST SELECTED", fg_color="#1976D2", command=self.backtest_selected)
        self.btn_fetch = ctk.CTkButton(btns, text="FETCH CANDIDATES", fg_color="#7B1FA2", command=self.fetch_candidates_thread)
        self.btn_apply_policy.grid(row=1, column=1, padx=5, pady=3, sticky="e")
        self.btn_refresh.grid(row=1, column=2, padx=5, pady=3, sticky="e")

        # Table
        container = ctk.CTkFrame(self.parent)
        container.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        cols = ("rank", "symbol", "score", "ret_pct", "dollar_vol", "volatility", "last_price", "bars", "scan_ts")
        self.tree = ttk.Treeview(container, columns=cols, show="headings", selectmode="browse")
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120, anchor="center")

        self.tree.column("rank", width=60, stretch=False)
        self.tree.column("symbol", width=90, stretch=False)
        self.tree.column("scan_ts", width=180, stretch=True)

        vsb = ttk.Scrollbar(container, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        self.tree.bind("<Double-1>", lambda _e: self.backtest_selected())

        # Initialize toggle UI state
        self._update_scanner_ui_state()

    def _get_config_bool(self, key: str, default: bool = False) -> bool:
        try:
            raw = str(self.config.get("CONFIGURATION", key, fallback=str(default))).strip().lower()
            return raw in ("1", "true", "yes", "y", "on")
        except Exception:
            return default

    def _set_config_bool(self, key: str, value: bool) -> None:
        try:
            if "CONFIGURATION" not in self.config:
                self.config.add_section("CONFIGURATION")
        except Exception:
            pass

        try:
            # Back up config first (TradingApp provides this)
            if self.app and hasattr(self.app, "backup_config"):
                self.app.backup_config()
        except Exception:
            pass

        try:
            self.config["CONFIGURATION"][key] = "True" if bool(value) else "False"
        except Exception:
            return

        try:
            if self.app and hasattr(self.app, "write_config"):
                self.app.write_config()
        except Exception:
            pass

    def _update_scanner_ui_state(self):
        enabled = bool(self._var_scanner_enabled.get())
        try:
            self._lbl_scanner_state.configure(text="ON" if enabled else "OFF")
        except Exception:
            pass

        try:
            self.btn_fetch.configure(state="normal" if enabled else "disabled")
        except Exception:
            pass

    def _on_scanner_toggle(self):
        enabled = bool(self._var_scanner_enabled.get())
        self._update_scanner_ui_state()
        self._set_config_bool("candidate_scanner_enabled", enabled)

    def set_status(self, text: str):
        try:
            self.lbl_status.configure(text=text)
        except Exception:
            pass

    def on_select(self, _evt=None):
        try:
            sel = self.tree.selection()
            if not sel:
                self._selected_symbol = None
                return
            row = self.tree.item(sel[0], "values")
            # columns: rank, symbol, ...
            if row and len(row) >= 2:
                self._selected_symbol = str(row[1]).strip().upper()
        except Exception:
            self._selected_symbol = None

    def refresh_from_db(self):
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            today = None

        try:
            df = self.db.get_latest_candidates(scan_date=today, limit=200)
        except Exception:
            df = None

        self.tree.delete(*self.tree.get_children())

        if df is None or getattr(df, "empty", True):
            self.set_status("No candidates in DB for today.")
            return

        try:
            self._last_scan_id = str(df["scan_id"].iloc[0]) if "scan_id" in df.columns else None
        except Exception:
            self._last_scan_id = None

        for i, (_idx, r) in enumerate(df.iterrows(), start=1):
            sym = str(r.get("symbol", "")).upper()
            score = float(r.get("score", 0.0) or 0.0)
            ret = float(r.get("ret_lookback", 0.0) or 0.0)
            dv = float(r.get("dollar_volume", 0.0) or 0.0)
            vol = float(r.get("volatility", 0.0) or 0.0)
            bars = int(r.get("bars", 0) or 0)
            scan_ts = str(r.get("scan_ts", "") or "")

            last_price = ""
            try:
                details = json.loads(r.get("details_json") or "{}")
                last_price = details.get("last_price", "")
            except Exception:
                last_price = ""

            self.tree.insert("", "end", values=(
                i,
                sym,
                round(score, 3),
                round(ret, 3),
                int(dv),
                round(vol, 3),
                last_price,
                bars,
                scan_ts,
            ))

        self.set_status(f"Loaded {len(df)} candidate(s) from DB.")

    def fetch_candidates_thread(self):
        # Gate by config toggle
        try:
            enabled = bool(self._var_scanner_enabled.get())
        except Exception:
            enabled = self._get_config_bool("candidate_scanner_enabled", default=False)

        if not enabled:
            messagebox.showinfo("Candidate Scanner", "candidate_scanner_enabled is OFF. Enable it in Config to fetch new scans.")
            return

        if not messagebox.askyesno("Candidate Scanner", "Fetch and compute today's candidates now?"):
            return

        threading.Thread(target=self.fetch_candidates, daemon=True).start()

    def fetch_candidates(self):
        self._call_ui(self.set_status, "Scanning...")
        try:
            api = getattr(self.engine, "api", None)
        except Exception:
            api = None

        try:
            scan_id, rows = self._scanner.scan_today(api=api)
            self._last_scan_id = scan_id
            if not rows:
                self._call_ui(self.set_status, "Scan complete (no rows).")
            else:
                self._call_ui(self.set_status, f"Scan complete: {len(rows)} candidates.")
        except Exception as e:
            self._call_ui(self.set_status, f"Scan failed: {e}")
            return

        self._call_ui(self.refresh_from_db)
    def apply_watchlist_policy(self):
        """Apply dynamic watchlist policy (manual trigger)."""
        if self.app and hasattr(self.app, "apply_watchlist_policy_now"):
            try:
                self.app.apply_watchlist_policy_now()
                return
            except Exception:
                pass
        messagebox.showwarning("Dynamic Watchlist", "Watchlist policy is unavailable in this build.")


    def add_selected_to_watchlist(self):
        sym = (self._selected_symbol or "").strip().upper()
        if not sym:
            messagebox.showinfo("Watchlist", "Select a candidate first.")
            return

        try:
            from ..watchlist_api import get_watchlist_symbols, add_watchlist_symbol
        except Exception:
            get_watchlist_symbols = None
            add_watchlist_symbol = None

        try:
            existing = set(get_watchlist_symbols(self.config, group="ACTIVE", asset="ALL")) if get_watchlist_symbols else set()
        except Exception:
            existing = set()

        if sym in existing:
            messagebox.showinfo("Watchlist", f"{sym} is already in ACTIVE.")
            return

        if not messagebox.askyesno("Watchlist", f"Add {sym} to ACTIVE watchlist?"):
            return

        try:
            # Back up config first (TradingApp provides this)
            if self.app and hasattr(self.app, "backup_config"):
                self.app.backup_config()

            if add_watchlist_symbol:
                add_watchlist_symbol(self.config, sym, group="ACTIVE")
            else:
                # Fallback (should not occur): write into ACTIVE_STOCK/CRYPTO directly
                sec = "WATCHLIST_ACTIVE_CRYPTO" if "/" in sym else "WATCHLIST_ACTIVE_STOCK"
                try:
                    if not self.config.has_section(sec):
                        self.config.add_section(sec)
                    self.config[sec][sym] = ""
                except Exception:
                    pass

            if self.app and hasattr(self.app, "write_config"):
                self.app.write_config()

            # Best-effort refresh for dropdowns
            if self.app and hasattr(self.app, "refresh_symbol_dropdowns"):
                self.app.refresh_symbol_dropdowns()

            self._log(f"[CANDIDATES] ✅ Added to ACTIVE: {sym}")
            messagebox.showinfo("Watchlist", f"Added {sym} to ACTIVE watchlist.")
        except Exception as e:
            messagebox.showerror("Watchlist", f"Failed to add {sym}: {e}")

