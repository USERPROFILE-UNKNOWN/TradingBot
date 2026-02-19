"""UI tab: Inspector.

Moved into modules/tabs/inspector.py in v5.8.0 to reduce regression risk.
"""

import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import os
import threading
import sqlite3
import traceback
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from ..charting.policies import DEFAULT_CHART_POLICY, compute_ylim
from ..charting.candles import draw_candles
from ..charting.overlays import draw_lower_band, draw_ema200, draw_adx
from ..market_data.history_repo import HistoryRepo
from ..market_data.updater import IncrementalUpdater
from ..paths import get_paths
from ..watchlist_api import get_watchlist_symbols


class InspectorTab:
    def __init__(self, parent, engine, db, config):
        self.parent = parent
        self.engine = engine
        self.db = db
        # Canonical history access layer (stabilizes chart data shape)
        self.history_repo = HistoryRepo(self.db)
        self.config = config

        # Irregularities panel state (v5.12.2 Update B)
        self._irr_after_id = None
        self._irr_is_running = False
        self._irr_pending = None
        self._irr_last_result = None
        self._db_update_running = False
        self._last_df = None
        self._last_symbol = None

        # Draw-control (prevents UI freezes from rapid redraw / re-entrant draws)
        self._draw_after_id = None
        self._pending_symbol = None
        self._is_drawing = False
        self._ax2 = None

        # Bottom time mini-chart axis (created in setup_ui)
        self.ax_time = None

        # v5.12.3 updateA: prevent overlapping per-symbol DB update runs
        self._irr_update_running = False

        self.setup_ui()

    # -------------------------
    # Market-hours aware freshness (v5.12.9 updateA)
    # -------------------------
    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Best-effort classification for 24/7 instruments."""
        try:
            s = str(symbol or "").strip().upper()
        except Exception:
            return False
        if not s:
            return False
        # Common crypto/forex formats
        if "/" in s:
            return True
        if "-" in s and s.endswith("USD"):
            return True
        if s.endswith("USD") and len(s) > 3:
            base = s[:-3]
            # conservative allowlist (can be extended later)
            crypto_bases = {
                "BTC", "ETH", "SOL", "ADA", "DOGE", "LTC", "XRP", "BNB",
                "AVAX", "DOT", "LINK", "MATIC", "UNI", "BCH", "ATOM",
                "XLM", "TRX", "ETC", "SHIB", "FIL", "NEAR"
            }
            if base in crypto_bases:
                return True
        return False

    def _equity_market_is_open(self) -> bool | None:
        """Return True/False if known; None if unknown.

        Tries Alpaca clock first. Falls back to a simple America/Chicago schedule.
        """
        # 1) Alpaca clock (best)
        try:
            api = getattr(self.engine, "api", None) or getattr(self.engine, "alpaca", None)
            if api is not None and hasattr(api, "get_clock"):
                clk = api.get_clock()
                is_open = getattr(clk, "is_open", None)
                if is_open is not None:
                    return bool(is_open)
        except Exception:
            pass

        # 2) Time-based heuristic (fallback)
        try:
            tz = ZoneInfo("America/Chicago")
            now_local = datetime.now(tz)
            wd = now_local.weekday()  # 0=Mon
            if wd >= 5:
                return False
            # Regular hours: 08:30–15:00 America/Chicago (matches your summary logs)
            mins = now_local.hour * 60 + now_local.minute
            open_m = 8 * 60 + 30
            close_m = 15 * 60 + 0
            return (open_m <= mins <= close_m)
        except Exception:
            return None

    def _market_freshness_badge(self, symbol: str, age_seconds: float, threshold: float):
        """Return (badge, severity, note) for freshness."""
        try:
            sym = str(symbol or "").strip().upper()
        except Exception:
            sym = ""
        thr = max(1.0, float(threshold or 180.0))

        # Crypto/24-7: treat as always actionable
        if self._is_crypto_symbol(sym):
            if age_seconds > thr * 3.0:
                return "STALE", "ERROR", "24/7 instrument"
            if age_seconds > thr:
                return "stale-ish", "WARN", "24/7 instrument"
            return "OK", "INFO", "24/7 instrument"

        # Equities: stale outside market hours should not panic
        is_open = self._equity_market_is_open()
        if is_open is False:
            # only escalate if it's *extremely* old (e.g., missing days of data)
            if age_seconds > 7 * 24 * 3600:
                return "STALE", "WARN", "market closed (very old last bar)"
            return "CLOSED", "INFO", "market closed"

        # Unknown open/closed -> keep existing behavior
        if is_open is None:
            if age_seconds > thr * 3.0:
                return "STALE", "ERROR", "market hours unknown"
            if age_seconds > thr:
                return "stale-ish", "WARN", "market hours unknown"
            return "OK", "INFO", "market hours unknown"

        # Market open
        if age_seconds > thr * 3.0:
            return "STALE", "ERROR", "market open"
        if age_seconds > thr:
            return "stale-ish", "WARN", "market open"
        return "OK", "INFO", "market open"

    def setup_ui(self):
        self.controls = ctk.CTkFrame(self.parent)
        self.controls.pack(pady=5, fill="x")

        # Phase 4 (v5.14.0): show ACTIVE by default, with an option to view ARCHIVE/FAVORITES.
        self.watchlist_source = ctk.StringVar(value="ACTIVE")
        self.watchlist_source_menu = ctk.CTkOptionMenu(
            self.controls,
            values=["ACTIVE", "ARCHIVE", "FAVORITES", "ALL"],
            variable=self.watchlist_source,
            command=lambda _v=None: self.refresh_symbol_list(),
            width=120,
        )
        self.watchlist_source_menu.pack(side="left", padx=(10, 5))

        symbols = self._build_symbol_list()
        # NOTE: command passes the chosen symbol
        self.symbol_select = ctk.CTkOptionMenu(self.controls, values=symbols, command=self.draw_chart)
        self.symbol_select.pack(side="left", padx=(5, 10))
        self.symbol_select.bind("<MouseWheel>", self.on_scroll)

        self.toggle_frame = ctk.CTkFrame(self.parent, height=40)
        self.toggle_frame.pack(pady=5, fill="x")

        self.var_all = ctk.BooleanVar(value=True)
        self.var_price = ctk.BooleanVar(value=True)
        self.var_lower = ctk.BooleanVar(value=True)
        self.var_ema = ctk.BooleanVar(value=True)
        self.var_adx = ctk.BooleanVar(value=True)

        ctk.CTkCheckBox(self.toggle_frame, text="[ALL]", variable=self.var_all,
                        command=lambda: self.toggle("ALL")).pack(side="left", padx=10)
        ctk.CTkCheckBox(self.toggle_frame, text="Candles", variable=self.var_price,
                        command=lambda: self.toggle("SINGLE")).pack(side="left", padx=10)
        ctk.CTkCheckBox(self.toggle_frame, text="Lower Band", variable=self.var_lower,
                        command=lambda: self.toggle("SINGLE")).pack(side="left", padx=10)
        ctk.CTkCheckBox(self.toggle_frame, text="EMA", variable=self.var_ema,
                        command=lambda: self.toggle("SINGLE")).pack(side="left", padx=10)
        ctk.CTkCheckBox(self.toggle_frame, text="ADX", variable=self.var_adx,
                        command=lambda: self.toggle("SINGLE")).pack(side="left", padx=10)

        # Main area layout: chart + irregularities panel (Update B)
        self.main_container = ctk.CTkFrame(self.parent)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.chart_frame = ctk.CTkFrame(self.main_container)
        self.chart_frame.pack(side="left", fill="both", expand=True)

        self.irr_frame = ctk.CTkFrame(self.main_container, width=360)
        self.irr_frame.pack(side="right", fill="y", padx=(10, 0))
        try:
            self.irr_frame.pack_propagate(False)
        except Exception:
            pass

        # Two-row layout: main chart + bottom time mini-chart
        self.figure, (self.ax, self.ax_time) = plt.subplots(
            2, 1,
            figsize=(6, 4),
            dpi=100,
            sharex=True,
            gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.05}
        )
        self.figure.patch.set_facecolor('#2b2b2b')

        # Main axis styling
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors='white', labelcolor='white')

        # Bottom axis styling (time mini-chart)
        self.ax_time.set_facecolor('#2b2b2b')
        self.ax_time.tick_params(colors='white', labelcolor='white')
        try:
            self.ax_time.tick_params(axis='y', left=False, labelleft=False)
        except Exception:
            pass

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Irregularities panel (right side)
        self._setup_irregularities_panel()

        # Initial draw
        self.draw_chart(self.symbol_select.get())

    def _build_symbol_list(self):
        src = "ACTIVE"
        try:
            src = (self.watchlist_source.get() or "ACTIVE").strip().upper()
        except Exception:
            src = "ACTIVE"

        try:
            if src == "ALL":
                syms = []
                for g in ("FAVORITES", "ACTIVE", "ARCHIVE"):
                    syms.extend(get_watchlist_symbols(self.config, group=g, asset="ALL"))
            else:
                syms = get_watchlist_symbols(self.config, group=src, asset="ALL")
        except Exception:
            syms = []

        out = [str(s).strip().upper() for s in syms if str(s).strip()]
        out = sorted(list({s for s in out if s}))
        return out if out else ["TQQQ"]

    def refresh_symbol_list(self):
        """Rebuild dropdown values based on the watchlist source selector."""
        try:
            new_values = self._build_symbol_list()
            current = None
            try:
                current = str(self.symbol_select.get()).strip().upper()
            except Exception:
                current = None

            try:
                self.symbol_select.configure(values=new_values)
            except Exception:
                # Fallback for older CTkOptionMenu implementations
                try:
                    self.symbol_select._values = new_values
                except Exception:
                    pass

            if current in new_values:
                try:
                    self.symbol_select.set(current)
                except Exception:
                    pass
            else:
                try:
                    self.symbol_select.set(new_values[0])
                except Exception:
                    pass
            try:
                self.draw_chart(self.symbol_select.get())
            except Exception:
                pass
        except Exception:
            pass

    def on_scroll(self, event):
        current_list = self.symbol_select._values
        try:
            current_val = self.symbol_select.get()
            idx = current_list.index(current_val)
            if event.delta > 0:
                new_idx = (idx - 1) % len(current_list)
            else:
                new_idx = (idx + 1) % len(current_list)
            new_val = current_list[new_idx]
            self.symbol_select.set(new_val)
            self.draw_chart(new_val)
        except Exception:
            pass

    def toggle(self, origin):
        if origin == "ALL":
            state = self.var_all.get()
            self.var_price.set(state)
            self.var_lower.set(state)
            self.var_ema.set(state)
            self.var_adx.set(state)
        self.draw_chart(self.symbol_select.get())

    def _format_time_ticks(self, df, max_labels: int = 7):
        """Return (positions, labels) for the bottom time axis.

        Uses df['timestamp'] when available; otherwise falls back to index.
        """
        try:
            n = int(len(df))
        except Exception:
            return [], []
        if n <= 0:
            return [], []

        k = min(int(max_labels), n)
        if k <= 1:
            pos = [0]
        else:
            step = max(1, (n - 1) // (k - 1))
            pos = list(range(0, n, step))
            if pos[-1] != (n - 1):
                pos.append(n - 1)
            if len(pos) > k:
                pos = pos[:k - 1] + [n - 1]

        ts = None
        try:
            ts = df['timestamp']
        except Exception:
            ts = None

        fmt = '%H:%M'
        try:
            if ts is not None and len(ts) >= 2:
                t0 = ts.iloc[0]
                t1 = ts.iloc[-1]
                if getattr(t0, 'date', lambda: None)() != getattr(t1, 'date', lambda: None)():
                    fmt = '%m-%d %H:%M'
                else:
                    try:
                        delta = (t1 - t0)
                        if getattr(delta, 'total_seconds', lambda: 0)() > 12 * 3600:
                            fmt = '%m-%d %H:%M'
                    except Exception:
                        pass
        except Exception:
            pass

        labels = []
        for i in pos:
            if ts is None:
                labels.append(str(i))
                continue
            try:
                t = ts.iloc[int(i)]
                labels.append(t.strftime(fmt))
            except Exception:
                labels.append(str(i))

        return pos, labels

    # -------------------------
    # Debounced draw entrypoint
    # -------------------------
    def draw_chart(self, symbol):
        """
        Public entrypoint (called from UI events). Debounces redraw to prevent
        freezes caused by rapid re-entrant matplotlib draws.
        """
        self._pending_symbol = symbol
        if self._draw_after_id is not None:
            try:
                self.parent.after_cancel(self._draw_after_id)
            except Exception:
                pass
        self._draw_after_id = self.parent.after(120, self._draw_chart_now)

    def _draw_chart_now(self):
        if self._is_drawing:
            return
        self._is_drawing = True

        try:
            symbol = self._pending_symbol or self.symbol_select.get()
            df = self.history_repo.get_history(symbol, limit=120)
            self._last_df = df
            self._last_symbol = symbol

            # Clear axes safely without nuking the figure (prevents widget lock-ups)
            if self._ax2 is not None:
                try:
                    self._ax2.remove()
                except Exception:
                    pass
                self._ax2 = None

            self.ax.cla()
            try:
                self.ax_time.cla()
            except Exception:
                pass

            self.ax.set_facecolor('#2b2b2b')
            self.ax.tick_params(colors='white', labelcolor='white')
            try:
                self.ax_time.set_facecolor('#2b2b2b')
                self.ax_time.tick_params(colors='white', labelcolor='white')
                self.ax_time.tick_params(axis='y', left=False, labelleft=False)
            except Exception:
                pass

            regime = getattr(self.engine, "market_status", "UNKNOWN")
            if regime == "BULL":
                self.ax.patch.set_facecolor('#1e3323')
                try:
                    self.ax_time.patch.set_facecolor('#1e3323')
                except Exception:
                    pass
            elif regime == "BEAR":
                self.ax.patch.set_facecolor('#331e1e')
                try:
                    self.ax_time.patch.set_facecolor('#331e1e')
                except Exception:
                    pass

            if df is None or df.empty:
                self.ax.text(0.5, 0.5, f"No Data for {symbol}", color="white", ha="center", va="center")
                try:
                    self.ax_time.text(0.5, 0.5, "No time series", color="white",
                                      ha="center", va="center", transform=self.ax_time.transAxes)
                except Exception:
                    pass
            else:
                # Determine Y-range for visualization (based on close quantiles)
                y_lo, y_hi = compute_ylim(df, DEFAULT_CHART_POLICY)

                # Price candles
                if self.var_price.get():
                    draw_candles(self.ax, df, y_lo=y_lo, y_hi=y_hi, wick_policy=DEFAULT_CHART_POLICY.wick_policy)
                if self.var_lower.get():
                    draw_lower_band(self.ax, df)

                if self.var_ema.get():
                    draw_ema200(self.ax, df)

                if self.var_adx.get():
                    self._ax2 = draw_adx(self.ax, df)

                # Apply y-limits after plotting
                if y_lo is not None and y_hi is not None:
                    try:
                        self.ax.set_ylim(y_lo, y_hi)
                    except Exception:
                        pass

                self.ax.set_title(f"{symbol} Inspector", color='white')

                # Bottom mini-chart: close vs time + readable time labels
                try:
                    if 'close' in df.columns:
                        self.ax_time.plot(df.index, df['close'], color='#9E9E9E', linewidth=1.0, alpha=0.9)
                except Exception:
                    pass

                try:
                    pos, labels = self._format_time_ticks(df, max_labels=7)
                    if pos and labels:
                        self.ax_time.set_xticks(pos)
                        self.ax_time.set_xticklabels(labels, rotation=0, ha='center', color='white', fontsize=8)
                except Exception:
                    pass

                try:
                    self.ax_time.grid(True, color='#404040', alpha=0.25)
                except Exception:
                    pass

            self.ax.grid(True, color='#404040', alpha=0.3)

            # Keep the top chart clean; show time labels on the bottom axis
            try:
                self.ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            except Exception:
                pass

            # Refresh irregularities panel (non-blocking)
            try:
                self.queue_irregularities_refresh(symbol, df=df)
            except Exception:
                pass

            # draw_idle is less likely to freeze than a blocking draw
            try:
                self.canvas.draw_idle()
            except Exception:
                self.canvas.draw()

        finally:
            self._is_drawing = False
            self._draw_after_id = None

    # -----------------------------
    # Irregularities panel (Update B)
    # -----------------------------
    def _setup_irregularities_panel(self):
        try:
            header = ctk.CTkFrame(self.irr_frame)
            header.pack(fill="x", padx=10, pady=(10, 6))

            self.irr_title = ctk.CTkLabel(header, text="IRREGULARITIES", font=("Arial", 14, "bold"))
            self.irr_title.pack(side="left")

            self.irr_status = ctk.CTkLabel(header, text="idle", font=("Arial", 12))
            self.irr_status.pack(side="right")

            btn_row = ctk.CTkFrame(self.irr_frame)
            btn_row.pack(fill="x", padx=10, pady=(0, 8))

            ctk.CTkButton(
                btn_row, text="Refresh", width=120,
                command=lambda: self.queue_irregularities_refresh(self.symbol_select.get(), immediate=True, deep=False)
            ).pack(side="left", padx=(0, 8))

            ctk.CTkButton(
                btn_row, text="Deep Check", width=120,
                command=lambda: self.queue_irregularities_refresh(self.symbol_select.get(), immediate=True, deep=True)
            ).pack(side="left")

            # v5.12.3 updateA: one-click actions for troubleshooting / recovery.
            action_row = ctk.CTkFrame(self.irr_frame)
            action_row.pack(fill="x", padx=10, pady=(0, 8))

            ctk.CTkButton(
                action_row, text="Open Logs", width=90,
                command=self._open_logs_folder
            ).pack(side="left", padx=(0, 8))

            ctk.CTkButton(
                action_row, text="Latest Summary", width=110,
                command=self._open_latest_summary
            ).pack(side="left", padx=(0, 8))

            ctk.CTkButton(
                action_row, text="Update DB", width=110,
                fg_color="#7B1FA2",
                command=self._update_db_for_current_symbol
            ).pack(side="left")

            self.irr_text = ctk.CTkTextbox(self.irr_frame, wrap="none")
            self.irr_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
            try:
                self.irr_text.configure(font=("Consolas", 11))
            except Exception:
                pass
            # Ensure expected log subfolders exist so the action buttons never report false negatives.
            self._ensure_log_subdirs()

            self._set_irr_text("Ready. Select a symbol or click Refresh.\n")
        except Exception:
            # Never hard-fail the tab for observability UI.
            pass

    def _set_irr_text(self, text: str) -> None:
        try:
            self.irr_text.configure(state="normal")
        except Exception:
            pass
        try:
            self.irr_text.delete("1.0", "end")
            self.irr_text.insert("end", text)
        except Exception:
            pass
        try:
            self.irr_text.configure(state="disabled")
        except Exception:
            pass

    def _ensure_log_subdirs(self):
        """Best-effort creation of expected log subfolders (summaries/research).

        This avoids confusing UI states like "no summaries folder" before the first summary export.
        """
        try:
            paths = get_paths()
            logs_dir = paths.get("logs") or os.path.join(paths.get("root", os.getcwd()), "logs")
            try:
                os.makedirs(logs_dir, exist_ok=True)
            except Exception:
                pass
            for sub in ("summaries", "research"):
                try:
                    os.makedirs(os.path.join(logs_dir, sub), exist_ok=True)
                except Exception:
                    pass
        except Exception:
            pass

    # ---- v5.12.3 updateA: actions ----
    def _open_logs_folder(self):
        try:
            paths = get_paths()
            logs_dir = paths.get("logs") or os.path.join(paths.get("root", os.getcwd()), "logs")
            # Windows-friendly open; best-effort fallbacks
            try:
                os.startfile(logs_dir)  # type: ignore[attr-defined]
                return
            except Exception:
                pass
            try:
                import subprocess
                subprocess.Popen(["explorer", logs_dir])
                return
            except Exception:
                pass
            try:
                import subprocess
                subprocess.Popen(["xdg-open", logs_dir])
            except Exception:
                pass
        except Exception:
            pass

    def _open_latest_summary(self):
        try:
            paths = get_paths()
            logs_dir = paths.get("logs") or os.path.join(paths.get("root", os.getcwd()), "logs")
            summ_dir = os.path.join(logs_dir, "summaries")

            # If the folder doesn't exist yet (e.g., before the first summary export), create it.
            if not os.path.isdir(summ_dir):
                try:
                    os.makedirs(summ_dir, exist_ok=True)
                except Exception:
                    pass

            if not os.path.isdir(summ_dir):
                try:
                    self.irr_status.configure(text="no summaries folder", text_color="orange")
                except Exception:
                    pass
                return

            # Prefer newest by mtime
            candidates = []
            for name in os.listdir(summ_dir):
                if not name.lower().endswith(('.json', '.txt')):
                    continue
                p = os.path.join(summ_dir, name)
                try:
                    candidates.append((os.path.getmtime(p), p))
                except Exception:
                    continue
            if not candidates:
                try:
                    self.irr_status.configure(text="no summaries yet", text_color="orange")
                except Exception:
                    pass
                return
            candidates.sort(key=lambda t: t[0], reverse=True)
            latest = candidates[0][1]
            try:
                os.startfile(latest)  # type: ignore[attr-defined]
                return
            except Exception:
                pass
            try:
                import subprocess
                subprocess.Popen(["explorer", latest])
                return
            except Exception:
                pass
        except Exception:
            pass

    def _update_db_for_current_symbol(self):
        """One-click incremental DB update for the currently selected symbol."""
        if self._db_update_running:
            return

        try:
            sym = str(self.symbol_select.get()).strip().upper()
        except Exception:
            sym = ""

        if not sym:
            return

        # Require a live API client
        api = getattr(self.engine, 'api', None)
        if api is None:
            try:
                self.irr_status.configure(text="API unavailable", text_color="orange")
            except Exception:
                pass
            return

        self._db_update_running = True
        try:
            self.irr_status.configure(text=f"updating DB: {sym}...", text_color="orange")
        except Exception:
            pass

        def worker():
            result = None
            try:
                try:
                    days = int(self.config.get('CONFIGURATION', 'update_db_lookback_days', fallback='60'))
                except Exception:
                    days = 60
                if days < 1:
                    days = 1
                if days > 365:
                    days = 365

                updater = IncrementalUpdater(self.config, self.db)
                result = updater.update_symbol(api, sym, days=days)
            except Exception as e:
                result = {"symbol": sym, "error": f"{type(e).__name__}: {e}"}

            def ui_done():
                try:
                    self._db_update_running = False
                except Exception:
                    pass

                try:
                    if isinstance(result, dict) and result.get('error'):
                        self.irr_status.configure(text="DB update failed", text_color="#D32F2F")
                        extra = f"\n[ERROR] [E_UPDATE_DB_SYMBOL] {result.get('error')}\n"
                    else:
                        fetched = (result or {}).get('fetched', 0)
                        inserted = (result or {}).get('inserted', 0)
                        self.irr_status.configure(text="DB update OK", text_color="#00C853")
                        extra = f"\n[INFO] DB update complete for {sym}: fetched={fetched} inserted={inserted}\n"

                    # Append to panel output without clobbering the latest diagnostics
                    base = ""
                    try:
                        base = (self._irr_last_result or {}).get('text', '')
                    except Exception:
                        base = ""
                    self._set_irr_text((base or "") + extra)
                except Exception:
                    pass

                # Refresh diagnostics after update completes
                try:
                    self.queue_irregularities_refresh(sym, immediate=True, deep=False)
                except Exception:
                    pass

            try:
                self.parent.after(0, ui_done)
            except Exception:
                ui_done()

        try:
            threading.Thread(target=worker, daemon=True).start()
        except Exception:
            self._db_update_running = False

    def queue_irregularities_refresh(self, symbol, df=None, immediate: bool = False, deep: bool = False):
        """Debounced irregularities refresh (never blocks chart drawing)."""
        try:
            sym = str(symbol).strip().upper() if symbol else ""
        except Exception:
            sym = symbol

        self._irr_pending = {
            "symbol": sym,
            "df": df,
            "deep": bool(deep),
        }

        if self._irr_after_id is not None:
            try:
                self.parent.after_cancel(self._irr_after_id)
            except Exception:
                pass
            self._irr_after_id = None

        delay = 10 if immediate else 250
        try:
            self._irr_after_id = self.parent.after(delay, self._start_irregularities_check)
        except Exception:
            # If after isn't available, fall back to immediate (still non-blocking)
            try:
                self._start_irregularities_check()
            except Exception:
                pass

    def _start_irregularities_check(self):
        self._irr_after_id = None
        if self._irr_is_running:
            return

        payload = self._irr_pending or {}
        self._irr_pending = None
        symbol = payload.get("symbol") or ""
        df = payload.get("df", None)
        deep = bool(payload.get("deep", False))

        self._irr_is_running = True
        try:
            if hasattr(self, "irr_status"):
                self.irr_status.configure(text="checking...")
        except Exception:
            pass

        def worker(sym, df_arg, deep_arg):
            try:
                result = self._collect_irregularities(sym, df_arg, deep=deep_arg)
            except Exception:
                result = {
                    "text": "[ERROR] Irregularities check crashed.\n\n" + traceback.format_exc(),
                    "counts": {"ERROR": 1, "WARN": 0, "INFO": 0},
                }

            def ui_update():
                try:
                    self._irr_last_result = result
                    if hasattr(self, "irr_text"):
                        self._set_irr_text(result.get("text", ""))
                    if hasattr(self, "irr_status"):
                        c = result.get("counts", {}) or {}
                        self.irr_status.configure(text=f"E:{c.get('ERROR',0)} W:{c.get('WARN',0)} I:{c.get('INFO',0)}")
                finally:
                    self._irr_is_running = False
                    # If a newer request arrived while we were running, immediately run it.
                    if self._irr_pending:
                        try:
                            self._start_irregularities_check()
                        except Exception:
                            pass

            try:
                self.parent.after(0, ui_update)
            except Exception:
                # Last-resort: update directly (may be unsafe, but avoids silent failure)
                ui_update()

        try:
            threading.Thread(target=worker, args=(symbol, df, deep), daemon=True).start()
        except Exception:
            self._irr_is_running = False

    def _collect_irregularities(self, symbol, df=None, deep: bool = False):
        """Return a formatted irregularities report for Inspector."""
        counts = {"ERROR": 0, "WARN": 0, "INFO": 0}
        lines = []

        def add(level: str, msg: str):
            level = (level or "INFO").upper()
            if level not in counts:
                level = "INFO"
            counts[level] += 1
            lines.append(f"[{level}] {msg}")

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        sym = (str(symbol).strip().upper() if symbol else "").strip()
        if not sym:
            sym = "(none)"
        lines.append(f"[{ts}] Symbol: {sym}")
        lines.append("")

        # Engine snapshot (best-effort, never raises)
        try:
            regime = getattr(self.engine, "market_status", "UNKNOWN")
            api_connected = getattr(self.engine, "api_connected", None)
            mode = getattr(self.engine, "mode", None) or getattr(self.engine, "trade_mode", None)
            add("INFO", f"Engine: market_status={regime} api_connected={api_connected} mode={mode}")
        except Exception:
            add("WARN", "Engine: unable to read status fields.")

        # DB snapshot (quick open test; deep optionally runs PRAGMA quick_check)
        try:
            db_paths = getattr(self.db, "db_paths", None)
            if isinstance(db_paths, dict) and db_paths:
                for key, path in db_paths.items():
                    self._check_db_path(add, key, path, deep=deep)
            else:
                add("INFO", "DB: split paths not exposed; skipping file checks.")
        except Exception:
            add("WARN", "DB: unable to inspect DB paths (non-fatal).")

        # History / data freshness checks
        if df is None:
            try:
                # Prefer the canonical repo so we validate time-shape consistently.
                if symbol:
                    df = self.history_repo.get_history(symbol, limit=250)
            except Exception:
                df = None

        if df is None:
            add("ERROR", "History: no dataframe available for this symbol.")
        else:
            try:
                empty = getattr(df, "empty", False)
            except Exception:
                empty = False

            if empty:
                add("ERROR", "History: dataframe is empty (no bars).")
            else:
                self._check_history_df(add, df)

        # Watchlist freshness snapshot (like run summary)
        try:
            lines.append("")
            lines.append("Watchlist freshness snapshot:")

            # Resolve watchlist symbols (best-effort)
            watch_syms = []
            try:
                # Phase 4: ACTIVE watchlist universe
                watch_syms = get_watchlist_symbols(self.config, group='ACTIVE', asset='ALL')
            except Exception:
                watch_syms = []

            watch_syms = [str(s).strip().upper() for s in (watch_syms or []) if str(s).strip()]
            watch_syms = sorted(set(watch_syms))

            if not watch_syms:
                lines.append("  (no watchlist symbols configured)")
            else:
                ts_map = {}
                try:
                    if hasattr(self.db, 'get_latest_timestamps_for_symbols'):
                        ts_map = self.db.get_latest_timestamps_for_symbols(watch_syms) or {}
                except Exception:
                    ts_map = {}

                now = datetime.now(timezone.utc)
                try:
                    threshold = float(self.config.get("CONFIGURATION", "stale_bar_seconds_threshold", fallback="180"))
                except Exception:
                    threshold = 180.0

                def parse_ts(s: str):
                    if not s:
                        return None
                    txt = str(s).strip()
                    try:
                        dt = datetime.fromisoformat(txt.replace('Z', '+00:00'))
                    except Exception:
                        dt = None
                    if dt is not None:
                        try:
                            if getattr(dt, 'tzinfo', None) is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                        except Exception:
                            pass
                    return dt

                snapshot_lines = []
                for ws in watch_syms:
                    ts_str = ts_map.get(ws)
                    dt = parse_ts(ts_str)
                    if dt is None:
                        snapshot_lines.append(f"  {ws:6}  --  (no data)")
                        continue
                    try:
                        age = (now - dt).total_seconds()
                    except Exception:
                        snapshot_lines.append(f"  {ws:6}  --  (bad timestamp)")
                        continue

                    badge, _sev, _note = self._market_freshness_badge(ws, age, threshold)
                    snapshot_lines.append(f"  {ws:6}  {age:7.0f}s  {badge}")

                # Keep the output bounded and readable
                if len(snapshot_lines) > 40:
                    snapshot_lines = snapshot_lines[:40] + [f"  ... ({len(watch_syms)-40} more)"]
                lines.extend(snapshot_lines)
        except Exception:
            add("WARN", "Watchlist snapshot: unavailable (non-fatal).")

        # Summary footer
        lines.append("")
        lines.append(f"Deep check: {'ON' if deep else 'OFF'}")
        text = "\n".join(lines) + "\n"
        return {"text": text, "counts": counts}

    def _check_db_path(self, add, key: str, path: str, deep: bool = False):
        try:
            p = str(path)
        except Exception:
            p = path

        if not p:
            add("ERROR", f"DB[{key}]: missing path.")
            return

        if not os.path.exists(p):
            add("ERROR", f"DB[{key}]: file not found: {p}")
            return

        # Quick open test
        try:
            uri = f"file:{p}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=2)
            try:
                conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
                add("INFO", f"DB[{key}]: OK (open/read)")
                if deep:
                    try:
                        row = conn.execute("PRAGMA quick_check(1)").fetchone()
                        status = row[0] if row else None
                        if str(status).lower() == "ok":
                            add("INFO", f"DB[{key}]: quick_check OK")
                        else:
                            add("ERROR", f"DB[{key}]: quick_check returned: {status}")
                    except Exception as e:
                        add("WARN", f"DB[{key}]: quick_check failed: {e}")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            add("ERROR", f"DB[{key}]: cannot open read-only: {e}")

    def _check_history_df(self, add, df):
        # Required columns
        required = ["timestamp", "open", "high", "low", "close"]
        cols = list(getattr(df, "columns", []))
        missing = [c for c in required if c not in cols]
        if missing:
            add("ERROR", f"History: missing required columns: {', '.join(missing)}")
            return

        # Basic bar stats
        try:
            n = int(getattr(df, "shape", (0, 0))[0])
        except Exception:
            n = 0
        add("INFO", f"History: bars={n}")

        # Freshness
        try:
            last_ts = df["timestamp"].iloc[-1]
            try:
                last_dt = last_ts.to_pydatetime()
            except Exception:
                last_dt = last_ts
            now = datetime.now(timezone.utc)
            age = (now - last_dt).total_seconds()
            try:
                threshold = float(self.config.get("CONFIGURATION", "stale_bar_seconds_threshold", fallback="180"))
            except Exception:
                threshold = 180.0

            sym = None
            try:
                sym = getattr(self, "_last_symbol", None) or self.symbol_select.get()
            except Exception:
                sym = None
            badge, sev, note = self._market_freshness_badge(sym or "", age, threshold)

            if sev == "ERROR":
                add("ERROR", f"Data freshness: {badge} ({age:.0f}s old; threshold={threshold:.0f}s; {note})")
            elif sev == "WARN":
                add("WARN", f"Data freshness: {badge} ({age:.0f}s old; threshold={threshold:.0f}s; {note})")
            else:
                add("INFO", f"Data freshness: {badge} ({age:.0f}s old; threshold={threshold:.0f}s; {note})")
        except Exception:
            add("WARN", "Data freshness: unable to compute bar age.")

        # Timestamp ordering / gaps
        try:
            ts = df["timestamp"]
            if hasattr(ts, "is_monotonic_increasing") and not ts.is_monotonic_increasing:
                add("WARN", "Timestamps: not strictly increasing (out-of-order bars).")
            try:
                dup = bool(getattr(ts, "duplicated")().any())
            except Exception:
                dup = False
            if dup:
                add("WARN", "Timestamps: duplicates detected (should be collapsed).")

            # Interval heuristics (last ~120 bars)
            tail = df.tail(120)
            d = tail["timestamp"].diff().dt.total_seconds()
            try:
                med = float(d.median())
                mx = float(d.max())
                if mx > max(180.0, med * 3.0):
                    add("WARN", f"Intervals: large gaps detected (median={med:.0f}s max={mx:.0f}s).")
                else:
                    add("INFO", f"Intervals: median={med:.0f}s max={mx:.0f}s.")
            except Exception:
                pass
        except Exception:
            add("WARN", "Timestamps: unable to evaluate ordering/gaps.")

        # OHLC sanity checks (last 50 bars)
        try:
            t = df.tail(50)
            hi_bad = (t["high"] < t[["open", "close"]].max(axis=1)).any()
            lo_bad = (t["low"] > t[["open", "close"]].min(axis=1)).any()
            if hi_bad or lo_bad:
                add("WARN", "OHLC: found bars where high/low do not bound open/close.")
        except Exception:
            pass

        # NaN checks (last 200 bars)
        try:
            t = df.tail(200)
            if t[required].isna().any().any():
                add("WARN", "OHLC: NaN values detected in recent bars.")
        except Exception:
            pass
