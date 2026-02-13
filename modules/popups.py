import customtkinter as ctk
from tkinter import ttk, messagebox
import threading
import os
from datetime import datetime

# Phase 4 (v5.14.0): canonical watchlist access
from .watchlist_api import get_watchlist_symbols

class StrategyEditor(ctk.CTkToplevel):
    def __init__(self, parent, strategy_name, config_data, on_save):
        super().__init__(parent)
        self.title(f"Edit {strategy_name}")
        self.geometry("400x500")
        self.on_save = on_save
        self.strategy_name = strategy_name
        self.entries = {}
        ctk.CTkLabel(self, text=f"Configuration: {strategy_name}", font=("Arial", 16, "bold")).pack(pady=10)
        self.scroll = ctk.CTkScrollableFrame(self, width=350, height=350)
        self.scroll.pack(pady=5, padx=10)
        for key, value in config_data.items():
            row = ctk.CTkFrame(self.scroll)
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=key, width=150, anchor="w").pack(side="left")
            entry = ctk.CTkEntry(row, width=150)
            entry.insert(0, value)
            entry.pack(side="right")
            self.entries[key] = entry
        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(pady=20, fill="x")
        ctk.CTkButton(btn_frame, text="CANCEL", fg_color="gray", command=self.destroy).pack(side="left", padx=20, expand=True)
        ctk.CTkButton(btn_frame, text="SAVE", fg_color="#00C853", command=self.save).pack(side="right", padx=20, expand=True)
    def save(self):
        new_data = {k: v.get() for k, v in self.entries.items()}
        self.on_save(self.strategy_name, new_data)
        self.destroy()

class DecisionViewer(ctk.CTkToplevel):
    def __init__(self, parent, db_manager):
        super().__init__(parent)
        self.title("The Brain (Decision Logs)")
        self.geometry("900x500")
        self.db = db_manager
        self.tree = ttk.Treeview(self, show="headings", selectmode="browse")
        cols = ("Time", "Symbol", "Action", "RSI", "AI Score", "Reason")
        self.tree["columns"] = cols
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=100)
        self.tree.column("Reason", width=250)
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)
        ctk.CTkButton(self, text="REFRESH", command=self.load_data).pack(pady=5)
        self.load_data()
    def load_data(self):
        self.tree.delete(*self.tree.get_children())
        rows = []
        try:
            if hasattr(self.db, 'get_recent_decisions'):
                rows = self.db.get_recent_decisions(limit=100)
            else:
                cursor = self.db.conn.cursor()
                cursor.execute(
                    "SELECT timestamp, symbol, action, rsi, ai_score, reason "
                    "FROM decision_logs ORDER BY timestamp DESC LIMIT 100"
                )
                rows = cursor.fetchall() or []
        except Exception as e:
            # v5.12.3 updateA: show the failure instead of silently swallowing it
            try:
                self.tree.insert("", "end", values=(datetime.now(), "", "ERROR", "", "", str(e)), tags=("gray",))
            except Exception:
                pass
            try:
                print(f"[E_UI_DECISION_VIEWER] {type(e).__name__}: {e}")
            except Exception:
                pass
            rows = []

        for row in rows:
            try:
                action = row[2] if len(row) > 2 else ""
                tag = "green" if action == "BUY" else "red" if action == "REJECT" else "gray"
                self.tree.insert("", "end", values=row, tags=(tag,))
            except Exception:
                continue

        try:
            self.tree.tag_configure("green", foreground="#00C853")
            self.tree.tag_configure("red", foreground="#FF1744")
            self.tree.tag_configure("gray", foreground="gray")
        except Exception:
            pass

class BacktestOrchestrator(ctk.CTkToplevel):
    """Popup to run orchestrated backtests for queued Architect variants."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Architect Backtest Orchestrator")
        self.geometry("1100x650")

        self._last_result = None
        self._queue_items_ref = []  # reference to app queue list (if provided)
        self._queue_metrics = {}    # vid -> {profit_sum, win_rate_mean, trades_sum}

        # --- Help strip (make it obvious how to use this popup) ---
        help_strip = ctk.CTkFrame(self)
        help_strip.pack(fill="x", padx=10, pady=(10, 0))

        self.help_label = ctk.CTkLabel(
            help_strip,
            text=(
                "How to use: In the Architect tab, click QUEUE on one or more variants. "
                "Then come back here and click RUN ALL to backtest the queued variants. "
                "Results are loaded into the Backtest Lab and can be exported from here."
            ),
            justify="left",
            anchor="w",
        )
        self.help_label.pack(side="left", fill="x", expand=True, padx=10, pady=6)

        self.queue_badge = ctk.CTkLabel(
            help_strip,
            text="Queue: 0 items",
            fg_color="#263238",
            corner_radius=10,
            padx=10,
            pady=4,
        )
        self.queue_badge.pack(side="right", padx=10, pady=6)

        # --- Controls ---
        top = ctk.CTkFrame(self)
        top.pack(fill="x", padx=10, pady=(10, 10))

        ctk.CTkLabel(top, text="Architect Backtest Orchestrator", font=("Arial", 16, "bold")).pack(side="left", padx=10)

        self.universe_var = ctk.StringVar(value="WATCHLIST")
        ctk.CTkOptionMenu(top, variable=self.universe_var, values=["WATCHLIST", "ALL_DB_SYMBOLS"]).pack(side="left", padx=10)

        ctk.CTkLabel(top, text="History bars:").pack(side="left", padx=(20, 5))
        self.history_entry = ctk.CTkEntry(top, width=100)
        self.history_entry.insert(0, "3000")
        self.history_entry.pack(side="left", padx=5)

        self.btn_run = ctk.CTkButton(top, text="RUN ALL", fg_color="#7B1FA2", command=self.run_thread)
        self.btn_run.pack(side="right", padx=5)

        self.btn_export = ctk.CTkButton(top, text="EXPORT LAST RUN", fg_color="#00695C", command=self.export_last_run)
        self.btn_export.pack(side="right", padx=5)
        self.btn_export.configure(state="disabled")

        self.btn_clear = ctk.CTkButton(top, text="CLEAR QUEUE", fg_color="#616161", command=self.clear_queue)
        self.btn_clear.pack(side="right", padx=5)

        self.btn_refresh = ctk.CTkButton(top, text="REFRESH QUEUE", command=self.load_queue)
        self.btn_refresh.pack(side="right", padx=5)

        # --- Queue table ---
        mid = ctk.CTkFrame(self)
        mid.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        cols = ("id", "source", "rsi", "sl", "tp", "ema", "profit", "win_rate")
        self.tree = ttk.Treeview(mid, show="headings", selectmode="browse")
        self.tree["columns"] = cols
        self.tree.heading("id", text="ID")
        self.tree.heading("source", text="SOURCE")
        self.tree.heading("rsi", text="RSI")
        self.tree.heading("sl", text="SL")
        self.tree.heading("tp", text="TP")
        self.tree.heading("ema", text="EMA")
        self.tree.heading("profit", text="FIT PROFIT")
        self.tree.heading("win_rate", text="FIT WR")

        self.tree.column("id", width=90, anchor="center")
        self.tree.column("source", width=120, anchor="center")
        self.tree.column("rsi", width=70, anchor="center")
        self.tree.column("sl", width=90, anchor="center")
        self.tree.column("tp", width=90, anchor="center")
        self.tree.column("ema", width=70, anchor="center")
        self.tree.column("profit", width=110, anchor="center")
        self.tree.column("win_rate", width=110, anchor="center")

        self.tree.pack(fill="both", expand=True, padx=5, pady=5)

        bottom = ctk.CTkFrame(self)
        bottom.pack(fill="x", padx=10, pady=(0, 10))

        self.status = ctk.CTkLabel(bottom, text="Queue loaded.", anchor="w")
        self.status.pack(side="left", fill="x", expand=True, padx=10)

        ctk.CTkButton(bottom, text="OPEN BACKTEST FOLDER", command=self.open_backtest_folder).pack(side="right", padx=10)

        self.load_queue()

    def _log(self, msg: str):
        try:
            if hasattr(self.parent, 'log'):
                self.parent.log(msg)
        except Exception:
            pass
    def load_queue(self):
        """Reload and display the current Architect queue."""
        # Clear table
        self.tree.delete(*self.tree.get_children())

        # Pull queue from the app
        items = []
        try:
            if hasattr(self.parent, 'get_architect_queue'):
                items = self.parent.get_architect_queue() or []
        except Exception:
            items = []

        # Keep a reference for later use (run / clear actions)
        self._queue_items_ref = items

        for it in items:
            try:
                vid = it.get("id", "")
                src = it.get("source_symbol", it.get("source", ""))
                g = it.get("genome") or {}

                rsi = g.get("rsi_buy", g.get("rsi", ""))
                sl = g.get("sl_pct", g.get("sl", ""))
                tp = g.get("tp_pct", g.get("tp", ""))
                ema = g.get("ema_period", g.get("ema", ""))

                profit = it.get("profit", "")
                win_rate = it.get("win_rate", "")

                fmt_profit = f"{profit:.2f}" if isinstance(profit, (int, float)) else (str(profit) if profit != "" else "")
                fmt_wr = f"{win_rate:.2f}" if isinstance(win_rate, (int, float)) else (str(win_rate) if win_rate != "" else "")

                self.tree.insert("", "end", values=(vid, src, rsi, sl, tp, ema, fmt_profit, fmt_wr))
            except Exception:
                continue


    def clear_queue(self):
        try:
            if hasattr(self.parent, 'clear_architect_queue'):
                self.parent.clear_architect_queue()
        except Exception:
            pass
        self.load_queue()

    def _resolve_symbols(self) -> list:
        uni = str(self.universe_var.get() or 'WATCHLIST').strip()
        try:
            if uni == 'ALL_DB_SYMBOLS' and hasattr(self.parent, 'db_manager'):
                return list(self.parent.db_manager.get_all_symbols() or [])
        except Exception:
            pass

        # WATCHLIST (Phase 4): use ACTIVE universe for engine/scanner/policy consistency
        try:
            if hasattr(self.parent, 'config') and self.parent.config is not None:
                syms = get_watchlist_symbols(self.parent.config, group='ACTIVE', asset='ALL')
                return [str(s).strip().upper() for s in (syms or []) if str(s).strip()]
        except Exception:
            pass
        return []

    def _progress(self, msg: str, pct: float):
        try:
            pct = max(0.0, min(1.0, float(pct)))
        except Exception:
            pct = 0.0
        self.status.configure(text=f"{msg} ({int(pct*100)}%)")

    def run_thread(self):
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _run(self):
        try:
            if not hasattr(self.parent, 'get_architect_queue'):
                self.parent.call_ui(messagebox.showwarning, "Orchestrator", "Queue not available on app.")
                return

            queue_items = self.parent.get_architect_queue() or []
            if not queue_items:
                self.parent.call_ui(messagebox.showinfo, "Orchestrator", "Architect queue is empty. Queue variants in the Architect tab first.")
                return

            from .backtest_runner import normalize_variants, run_architect_queue_backtest

            variants = normalize_variants(queue_items)
            symbols = self._resolve_symbols()
            if not symbols:
                self.parent.call_ui(messagebox.showwarning, "Orchestrator", "No symbols found in the selected universe.")
                return

            try:
                history_limit = int(float(self.history_entry.get().strip()))
            except Exception:
                history_limit = 3000

            def cb(m, p):
                self.parent.call_ui(self._progress, m, p)

            result = run_architect_queue_backtest(
                db_manager=self.parent.db_manager,
                config=self.parent.config,
                variants=variants,
                symbols=symbols,
                history_limit=history_limit,
                max_workers=4,
                progress_cb=cb,
                score_key='score',
            )
            self._last_result = result

            # Attach aggregate metrics back onto queued items so the table fills in.
            try:
                per_variant = (result.get('aggregates') or {}).get('per_variant') or {}
                self._queue_metrics = per_variant
                for it in queue_items:
                    vid = str(it.get('id', ''))
                    if vid and vid in per_variant:
                        m = per_variant[vid] or {}
                        if 'profit_sum' in m:
                            it['profit'] = m.get('profit_sum')
                        if 'win_rate_mean' in m:
                            it['win_rate'] = m.get('win_rate_mean')
                        if 'trades_sum' in m:
                            it['trades'] = m.get('trades_sum')
            except Exception:
                pass

            df = result.get('df')
            if df is None:
                self.parent.call_ui(messagebox.showerror, "Orchestrator", "Backtest failed: pandas not available.")
                return

            # Push into Backtest Lab grid for inspection
            try:
                self.parent.backtest_df = df
                self.parent.call_ui(self.parent.populate_backtest_ui, df)
            except Exception:
                pass

            self.parent.call_ui(self._progress, "Orchestrator complete. Results loaded into Backtest Lab.", 1.0)
            try:
                self.parent.call_ui(self.load_queue)
            except Exception:
                pass

        except Exception as e:
            try:
                self.parent.call_ui(messagebox.showerror, "Orchestrator", f"Run failed: {e}")
            except Exception:
                pass

    def export_last_run(self):
        if not self._last_result:
            messagebox.showinfo("Orchestrator", "No orchestrator run to export yet.")
            return

        try:
            from .research.architect_backtest_exporter import export_architect_backtest_bundle
            from .utils import APP_VERSION, APP_RELEASE
        except Exception as e:
            messagebox.showerror("Orchestrator", f"Exporter import failed: {e}")
            return

        try:
            logs_root = (self.parent.paths.get('logs') or '.')
            out_dir = os.path.join(logs_root, 'backtest')
            summ_dir = os.path.join(logs_root, 'summaries')
            res = export_architect_backtest_bundle(
                result=self._last_result,
                config=self.parent.config,
                out_dir=out_dir,
                summaries_dir=summ_dir,
                app_version=APP_VERSION,
                app_release=APP_RELEASE,
                include_csv=True,
            )

            if not res:
                messagebox.showwarning("Orchestrator", "Export failed (no result).")
                return

            parts = [res.get('json'), res.get('csv'), res.get('summary')]
            msg = "Exported\n" + "\n".join([p for p in parts if p])
            messagebox.showinfo("Orchestrator", msg)
        except Exception as e:
            messagebox.showerror("Orchestrator", f"Export failed: {e}")

    def open_backtest_folder(self):
        try:
            folder = os.path.join(self.parent.paths.get('logs') or '.', 'backtest')
            if hasattr(self.parent, 'open_folder'):
                self.parent.open_folder(folder)
            else:
                os.startfile(folder)
        except Exception:
            pass
