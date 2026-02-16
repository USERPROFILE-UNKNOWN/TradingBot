"""UI tab: Architect.

Moved into modules/tabs/architect.py in v5.8.0 to reduce regression risk.
"""

import customtkinter as ctk
import threading
from ..architect import TheArchitect 
from ..watchlist_api import get_watchlist_symbols

class ArchitectTab:
    def __init__(self, parent, db, config):
        self.parent = parent
        self.db = db
        self.config = config
        self.architect = TheArchitect(db, config)
        self.is_running = False 
        # UI thread dispatcher (TradingApp.call_ui)
        try:
            self.app = parent.winfo_toplevel()
        except Exception:
            self.app = None
        self._call_ui = getattr(self.app, "call_ui", None)
        if not callable(self._call_ui):
            # Fallback: if call_ui is unavailable, execute immediately (may be unsafe off-main-thread)
            self._call_ui = lambda func, *a, **k: func(*a, **k)
        self.setup_ui()

    def setup_ui(self):
        # Header Controls
        panel = ctk.CTkFrame(self.parent)
        panel.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(panel, text="Genetic Target:", font=("Arial", 14, "bold")).pack(side="left", padx=10)

        # Phase 4 (v5.14.0): show ACTIVE by default, with an option to view ARCHIVE/FAVORITES.
        self.watchlist_source = ctk.StringVar(value="ACTIVE")
        self.watchlist_source_menu = ctk.CTkOptionMenu(
            panel,
            values=["ACTIVE", "ARCHIVE", "FAVORITES", "ALL"],
            variable=self.watchlist_source,
            command=lambda _v=None: self.refresh_symbol_list(),
            width=120,
        )
        self.watchlist_source_menu.pack(side="left", padx=(0, 10))

        symbols = self._build_symbol_list()

        self.arch_symbol = ctk.CTkOptionMenu(panel, values=symbols, command=self.on_select)
        self.arch_symbol.pack(side="left", padx=10)
        self.arch_symbol.bind("<MouseWheel>", self.on_scroll)
        
        self.btn_optimize = ctk.CTkButton(panel, text="START EVOLUTION", fg_color="#7B1FA2", width=200, command=self.run_thread)
        self.btn_optimize.pack(side="right", padx=10)
        
        # Progress Bar
        self.progress_bar = ctk.CTkProgressBar(self.parent, width=800, height=15)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=5)
        
        self.status = ctk.CTkLabel(self.parent, text="Ready to evolve strategies.", font=("Arial", 12))
        self.status.pack(pady=5)

        # Results Table
        self.results_frame = ctk.CTkScrollableFrame(self.parent, width=1000, height=500)
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=10)

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
        out = sorted(list(dict.fromkeys([s for s in out if s])))
        return out if out else ["TQQQ"]

    def refresh_symbol_list(self):
        try:
            new_values = self._build_symbol_list()
            current = None
            try:
                current = str(self.arch_symbol.get()).strip().upper()
            except Exception:
                current = None

            try:
                self.arch_symbol.configure(values=new_values)
            except Exception:
                try:
                    self.arch_symbol._values = new_values
                except Exception:
                    pass

            if current in new_values:
                try:
                    self.arch_symbol.set(current)
                except Exception:
                    pass
            else:
                try:
                    self.arch_symbol.set(new_values[0])
                except Exception:
                    pass
        except Exception:
            pass

    def on_select(self, choice):
        self.run_thread()

    def on_scroll(self, event):
        if self.is_running: return 
        current_list = self.arch_symbol._values
        try:
            current_val = self.arch_symbol.get()
            idx = current_list.index(current_val)
            if event.delta > 0: new_idx = (idx - 1) % len(current_list)
            else: new_idx = (idx + 1) % len(current_list)
            new_val = current_list[new_idx]
            self.arch_symbol.set(new_val)
            self.run_thread()
        except Exception:
            return

    def run_thread(self):
        if self.is_running: return
        self.is_running = True
        self.btn_optimize.configure(state="disabled", text="EVOLVING...")
        self.progress_bar.set(0)
        
        # Clear previous results
        for widget in self.results_frame.winfo_children(): widget.destroy()
        
        threading.Thread(target=self.run_process, daemon=True).start()

    def run_process(self):
        symbol = self.arch_symbol.get()
        # Persist the symbol used for this result set (used by QUEUE buttons)
        try:
            self._last_symbol = str(symbol).strip().upper()
        except Exception:
            self._last_symbol = symbol
        self.update_status(f"Loading DNA for {symbol}...", 0)
        
        results = self.architect.run_optimization(symbol, self.update_status)
        
        self._call_ui(self.display_results, results)
        self._call_ui(self.reset_ui)

    def update_status(self, text, progress_float):
        # Thread-safe UI update (must not touch Tk/CTk from worker threads)
        self._call_ui(self._update_status_safe, text, progress_float)

    def _update_status_safe(self, text, progress_float):
        self.status.configure(text=text)
        self.progress_bar.set(progress_float)

    def reset_ui(self):
        self.btn_optimize.configure(state="normal", text="START EVOLUTION")
        self.status.configure(text="Evolution Complete. Top survivors listed below.")
        self.is_running = False

    def display_results(self, results):
        for widget in self.results_frame.winfo_children(): widget.destroy()

        # Capture the symbol that produced this result set so QUEUE works reliably.
        try:
            sym_for_queue = str(getattr(self, '_last_symbol', None) or self.arch_symbol.get()).strip().upper()
        except Exception:
            sym_for_queue = getattr(self, '_last_symbol', None) or self.arch_symbol.get()

        if not results:
            ctk.CTkLabel(self.results_frame, text="No profitable strategies found. The market is tough.", font=("Arial", 14)).pack(pady=20)
            return

        h = ctk.CTkFrame(self.results_frame)
        h.pack(fill="x", pady=2)
        cols = ["RANK", "RSI", "STOP LOSS", "TAKE PROFIT", "EMA", "NET PROFIT ($)", "WIN RATE", "QUEUE"]
        for c in cols: ctk.CTkLabel(h, text=c, width=120, font=("Arial", 12, "bold")).pack(side="left")
        
        rank = 1
        for res in results:
            row = ctk.CTkFrame(self.results_frame)
            row.pack(fill="x", pady=2)
            
            profit = res['profit']
            color = "#00C853" if profit > 0 else "#FF1744"
            
            ctk.CTkLabel(row, text=f"#{rank}", width=120).pack(side="left")
            ctk.CTkLabel(row, text=str(res['rsi']), width=120).pack(side="left")
            ctk.CTkLabel(row, text=f"{res['sl']*100:.1f}%", width=120).pack(side="left")
            ctk.CTkLabel(row, text=f"{res['tp']}x", width=120).pack(side="left")
            ctk.CTkLabel(row, text=str(res['ema']), width=120).pack(side="left")
            ctk.CTkLabel(row, text=f"${profit:.2f}", width=120, text_color=color, font=("Arial", 12, "bold")).pack(side="left")
            ctk.CTkLabel(row, text=f"{res['win_rate']:.1f}%", width=120).pack(side="left")
            ctk.CTkButton(row, text="QUEUE", width=100, fg_color="#455A64", command=lambda r=res, sym=sym_for_queue: self.queue_variant(sym, r)).pack(side="left", padx=5)
            rank += 1

    def queue_variant(self, symbol, res):
        """Queue a survivor genome for the Backtest Orchestrator (updateB)."""
        try:
            app = self.app or self.parent.winfo_toplevel()
            if hasattr(app, 'add_architect_variant'):
                ok = app.add_architect_variant(symbol, res)
                if ok:
                    self.status.configure(text=f"Queued variant for orchestration: RSI={res.get('rsi')} SL={res.get('sl')} TP={res.get('tp')} EMA={res.get('ema')}")
                else:
                    self.status.configure(text="Variant already queued (duplicate).")
            else:
                self.status.configure(text="Orchestrator queue API not available.")
        except Exception as e:
            try:
                self.status.configure(text=f"Queue failed: {e}")
            except Exception:
                pass
