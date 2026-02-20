import customtkinter as ctk
import threading
import queue
import sys
import subprocess
import os
import time
import logging
import shutil
import numpy as np 
from tkinter import messagebox, ttk
from datetime import datetime
from .engine import TradingEngine
from .agent_master import AgentMaster
from .backfill import BackfillEngine
from .utils import get_paths, write_split_config, ensure_split_config_layout, APP_VERSION, APP_RELEASE
from .logging_utils import get_logger
from .config_validate import validate_runtime_config

# Refactored Components
from .popups import StrategyEditor, DecisionViewer, BacktestOrchestrator
from .tabs.dashboard import DashboardTab
from .tabs.inspector import InspectorTab
from .tabs.architect import ArchitectTab
from .tabs.config import ConfigTab
from .tabs.candidates import CandidatesTab
from .research.full_backtest_service import run_full_backtest_service
from .research.backtest_exporter import export_backtest_bundle
from .research.watchlist_policy import apply_watchlist_policy

class TradingApp(ctk.CTk):
    def __init__(self, config, db_manager):
        super().__init__()
        self.title(f"TradingBot {APP_VERSION} | {APP_RELEASE}")
        self.geometry("1200x850")
        ctk.set_appearance_mode("dark")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.config = config
        self.db_manager = db_manager
        self.paths = get_paths()

        # Unified structured logging (file/stdout)
        self._py_logger = get_logger(__name__)

        # Release E2: Live Log buffering + file-backed runtime log
        self._log_max_lines = 2000
        self._log_file_enabled = True
        self._runtime_log_path = None
        self._file_log_queue = None
        self._file_log_stop = None
        self._init_log_controls()

        # v5.12.3 updateA: allow DB layer to write diagnostics to the runtime log
        # instead of silently swallowing failures in split mode.
        try:
            if hasattr(self.db_manager, "set_log_callback"):
                self.db_manager.set_log_callback(self.log)
        except Exception:
            pass

        # Hotfix: prevent duplicate AI training threads
        self._ai_retrain_in_progress = False

        # --------------------
        # Thread-safe UI queues
        # --------------------
        # Tk/CTk is not thread-safe: even calling .after(...) from a worker thread
        # can trigger "main thread is not in main loop". We therefore only touch
        # widgets on the main thread and use queues for cross-thread communication.
        self._main_thread_id = threading.get_ident()
        self._log_queue = queue.SimpleQueue()
        self._chart_queue = queue.SimpleQueue()
        self._ui_queue = queue.SimpleQueue()
        self.after(100, self._process_queues)
        self.syncing_selection = False
        
        self.engine = None
        self.agent_master = None
        
        # Data Cache for Sorting
        self.backtest_df = None
        self.sort_descending = True # Toggle state
        

        # v6.8.0: queue of Architect variants for orchestrated backtests
        # Primary store is DB-backed (decision_logs.architect_queue) via DataManager.
        # Keep in-memory list as a compatibility fallback only.
        self._architect_queue = []
        self._architect_queue_counter = 0

        # v5.13.1 updateB: periodic dynamic watchlist policy (defaults OFF)
        self._watchlist_policy_in_progress = False
        self._watchlist_policy_after_id = None

        # v5.13.2 updateA: Mini-player (Live Log)
        # When enabled, the UI throttles heavy redraw loops while the engine continues.
        self._miniplayer_enabled = self._read_miniplayer_start_enabled()
        self._miniplayer_prev_geometry = None

        self.setup_ui()

        # Periodic auto-watchlist tick (interval controlled by config; safe no-op when disabled)
        try:
            self.after(15000, self._watchlist_policy_tick)
        except Exception:
            pass
        self.after(1000, self.update_ui_loop)

    def on_closing(self):
        if self.engine and self.engine.active:
            self.engine.stop()
        try:
            if self.agent_master:
                self.agent_master.shutdown()
        except Exception:
            pass

        # v5.13.1 updateB: cancel periodic watchlist tick (best-effort)
        try:
            if getattr(self, '_watchlist_policy_after_id', None) is not None:
                self.after_cancel(self._watchlist_policy_after_id)
                self._watchlist_policy_after_id = None
        except Exception:
            pass
        # Release E3: write run summary on app close (best-effort)
        try:
            self.export_run_summary(silent=True)
        except Exception:
            pass
        # Release E2: stop runtime file logger (best-effort)
        try:
            if getattr(self, '_file_log_stop', None) is not None:
                self._file_log_stop.set()
            if getattr(self, '_file_log_queue', None) is not None:
                try:
                    self._file_log_queue.put_nowait(None)
                except Exception:
                    pass
        except Exception:
            pass
        self.destroy()
        sys.exit()

    def setup_ui(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # v4.0.2: Improved Grid Styling
        self.style.configure("Treeview", 
                             background="#2b2b2b", 
                             fieldbackground="#2b2b2b", 
                             foreground="white", 
                             rowheight=30, 
                             borderwidth=1, 
                             relief="solid")
        
        self.style.map("Treeview", background=[("selected", "#00C853")])
        self.style.configure("Treeview.Heading", 
                             background="#404040", 
                             foreground="white", 
                             relief="raised", 
                             font=("Arial", 10, "bold"))

        # 1. Top Bar
        self.top_bar = ctk.CTkFrame(self, height=40)
        self.top_bar.pack(fill="x", padx=5, pady=5)
        self.lbl_account = ctk.CTkLabel(self.top_bar, text="Account: Loading...", font=("Arial", 16, "bold"))
        self.lbl_account.pack(side="left", padx=20)
        self.lbl_market_status = ctk.CTkLabel(self.top_bar, text="MARKET: CHECKING...", font=("Arial", 14, "bold"), text_color="gray")
        self.lbl_market_status.pack(side="left", padx=30)

        # Action bar (controls row)
        # Put buttons on their own row so they are always visible
        self.action_bar = ctk.CTkFrame(self, height=45)
        self.action_bar.pack(fill="x", padx=5, pady=(0,5))
        
        self.btn_export = ctk.CTkButton(self.action_bar, text="EXPORT LOG", width=120, command=self.export_log_file, fg_color="#607D8B")
        self.btn_export.pack(side="right", padx=5)

        self.btn_summary = ctk.CTkButton(self.action_bar, text="EXPORT SUMMARY", width=140, command=self.export_run_summary, fg_color="#455A64")
        self.btn_summary.pack(side="right", padx=5)

        self.btn_strategy_report = ctk.CTkButton(self.action_bar, text="STRATEGY REPORT", width=150, command=self.export_strategy_report, fg_color="#37474F")
        self.btn_strategy_report.pack(side="right", padx=5)

        self.btn_retrain_ai = ctk.CTkButton(self.action_bar, text="RETRAIN AI", width=110, command=self.retrain_ai_model, fg_color="#546E7A")
        self.btn_retrain_ai.pack(side="right", padx=5)
        
        self.btn_brain = ctk.CTkButton(self.action_bar, text="VIEW BRAIN", width=100, command=self.open_decision_viewer, fg_color="#7B1FA2")
        self.btn_brain.pack(side="right", padx=5)
        self.update_db_btn = ctk.CTkButton(self.action_bar, text="UPDATE DB", width=100, command=self.open_backfill_window, fg_color="#2196F3")
        self.update_db_btn.pack(side="right", padx=5)
        self.lbl_copyright = ctk.CTkLabel(self.top_bar, text="©Cameron Drake [2026]", font=("Arial", 12), text_color="gray")
        self.lbl_copyright.pack(side="right", padx=20)
        
        # 2. Tabs
        self.tab_view = ctk.CTkTabview(self, width=1050, height=700)
        self.tab_view.pack(pady=10, padx=10, fill="both", expand=True)
        self.tab_log = self.tab_view.add("Live Log")
        self.tab_chart = self.tab_view.add("Dashboard")
        self.tab_candidates = self.tab_view.add("Today's Candidates")
        self.tab_inspect = self.tab_view.add("Inspector")
        self.tab_config = self.tab_view.add("Config")
        self.tab_architect = self.tab_view.add("The Architect") 
        self.tab_backtest = self.tab_view.add("Backtest Lab") 
        self.tab_pos = self.tab_view.add("Active Positions")
        self.tab_hist = self.tab_view.add("Portfolio Manager")
        self.tab_strat = self.tab_view.add("Strategies") 
        self.tab_set = self.tab_view.add("Settings") 

        # 3. Log Box
        self.log_box = ctk.CTkTextbox(self.tab_log, width=1000, height=500)
        self.log_box.pack(pady=10, fill="both", expand=True)
        self.toggle_btn = ctk.CTkButton(self.tab_log, text="START ENGINE", command=self.toggle_engine, fg_color="#00C853", hover_color="#009624", height=40)
        self.toggle_btn.pack(pady=10)

        # v5.13.2 updateA: Mini-player toggle (Live Log only)
        self.miniplayer_frame = ctk.CTkFrame(self.tab_log)
        self.miniplayer_frame.pack(fill="x", padx=10, pady=(0, 10))
        self._miniplayer_var = ctk.BooleanVar(value=bool(getattr(self, "_miniplayer_enabled", False)))
        self.miniplayer_switch = ctk.CTkSwitch(
            self.miniplayer_frame,
            text="MINI-PLAYER (Pause charts/heatmap)",
            variable=self._miniplayer_var,
            command=self.toggle_miniplayer,
        )
        self.miniplayer_switch.pack(side="left", padx=10, pady=8)
        self.miniplayer_status_lbl = ctk.CTkLabel(self.miniplayer_frame, text=self._miniplayer_status_text(), text_color="gray")
        self.miniplayer_status_lbl.pack(side="left", padx=10)

        # 4. Engine
        self.engine = TradingEngine(self.config, self.db_manager, self.log, self.trigger_chart_update)
        self.agent_master = AgentMaster(self.config, self.db_manager, self.log)
        try:
            self.engine.set_agent_master(self.agent_master)
        except Exception:
            pass

        # 5. Logic
        self.dashboard_logic = DashboardTab(self.tab_chart, self.engine, self.db_manager, self.config, metrics_store=getattr(self.agent_master, "metrics", None))
        self.inspector_logic = InspectorTab(self.tab_inspect, self.engine, self.db_manager, self.config)
        self.config_logic = ConfigTab(self.tab_config, self.db_manager, self.config, self.paths, self.log)
        self.architect_logic = ArchitectTab(self.tab_architect, self.db_manager, self.config)
        self.candidates_logic = CandidatesTab(self.tab_candidates, self.engine, self.db_manager, self.config)

        # 6. Remaining
        self.setup_backtest_tab()
        self.setup_positions_tab()
        self.setup_portfolio_tab()
        self.setup_strategies_tab()
        self.setup_settings_tab()

        # Phase 0 guardrail: if config.ini is in a merged/broken state, repair it before any config writes.
        try:
            from .config_io import sanitize_configuration_ini_if_needed, get_last_config_sanitizer_report
            sanitize_configuration_ini_if_needed(self.paths, section_name="CONFIGURATION")
            rep = get_last_config_sanitizer_report()
            if rep and rep.get("repairs_applied"):
                # Small startup warning (Live Log)
                self.log(f"[CONFIG] ⚠ Sanitizer repaired config.ini formatting ({rep.get('repairs_applied')} repairs).")
        except Exception:
            pass

        # Apply miniplayer state after tabs/widgets exist.
        try:
            self._apply_miniplayer_state(bool(getattr(self, "_miniplayer_enabled", False)), initial=True)
        except Exception:
            pass

    # --------------------
    # v5.13.2 updateA: Mini-player (Live Log)
    # --------------------
    def _read_miniplayer_start_enabled(self) -> bool:
        try:
            return str(self.config.get("CONFIGURATION", "miniplayer_enabled", fallback="False")).strip().lower() in ("1", "true", "yes", "y", "on")
        except Exception:
            return False

    def _ui_loop_interval_ms(self) -> int:
        """UI loop cadence (ms). Slows down when miniplayer is enabled."""
        try:
            cfg = self.config["CONFIGURATION"] if "CONFIGURATION" in self.config else {}
            normal = int(float(cfg.get("ui_refresh_ms", 2000)))
            mini = int(float(cfg.get("miniplayer_ui_refresh_ms", 8000)))
        except Exception:
            normal, mini = 2000, 8000

        # Clamp to sane bounds
        try:
            normal = max(250, min(60000, int(normal)))
        except Exception:
            normal = 2000
        try:
            mini = max(250, min(60000, int(mini)))
        except Exception:
            mini = 8000

        if bool(getattr(self, "_miniplayer_enabled", False)):
            return mini
        return normal

    def _miniplayer_status_text(self) -> str:
        try:
            ms = int(self._ui_loop_interval_ms())
            sec = ms / 1000.0
        except Exception:
            sec = 2.0
        state = "ON" if bool(getattr(self, "_miniplayer_enabled", False)) else "OFF"
        return f"Mini-player: {state} | UI refresh: {sec:.1f}s"

    def toggle_miniplayer(self):
        """UI callback (Live Log tab)"""
        try:
            enabled = bool(self._miniplayer_var.get())
        except Exception:
            enabled = bool(getattr(self, "_miniplayer_enabled", False))

        try:
            self._apply_miniplayer_state(enabled, initial=False)
        except Exception:
            pass

    def _apply_miniplayer_state(self, enabled: bool, initial: bool = False):
        enabled = bool(enabled)
        prev = bool(getattr(self, "_miniplayer_enabled", False))
        self._miniplayer_enabled = enabled

        # Keep UI control in sync (best-effort)
        try:
            if hasattr(self, "_miniplayer_var"):
                self._miniplayer_var.set(enabled)
        except Exception:
            pass

        # Persist preference (guardrail: avoid config writes unless the value actually changed)
        try:
            if "CONFIGURATION" in self.config:
                desired = "True" if enabled else "False"
                current = str(self.config["CONFIGURATION"].get("miniplayer_enabled", "")).strip()
                if current.lower() != desired.lower():
                    self.config["CONFIGURATION"]["miniplayer_enabled"] = desired
                    self.write_config()
        except Exception:
            pass

        # Focus the Live Log tab and shrink/restore window geometry.
        try:
            if enabled:
                try:
                    if self._miniplayer_prev_geometry is None:
                        self._miniplayer_prev_geometry = self.geometry()
                except Exception:
                    pass
                try:
                    if hasattr(self, "tab_view"):
                        self.tab_view.set("Live Log")
                except Exception:
                    pass
                try:
                    # Conservative size: keep log readable.
                    self.geometry("980x720")
                except Exception:
                    pass
            else:
                try:
                    if self._miniplayer_prev_geometry:
                        self.geometry(self._miniplayer_prev_geometry)
                except Exception:
                    pass
        except Exception:
            pass

        # Update status label + emit a single log line (not on initial apply)
        try:
            if hasattr(self, "miniplayer_status_lbl"):
                self.miniplayer_status_lbl.configure(text=self._miniplayer_status_text())
        except Exception:
            pass

        if not initial and (enabled != prev):
            try:
                self.log(f"🎛️ [MINIPLAYER] {'ENABLED' if enabled else 'DISABLED'} (UI refresh: {self._ui_loop_interval_ms()/1000:.1f}s)")
            except Exception:
                pass

    def update_ui_loop(self):
        try:
            if self.engine and self.engine.api:
                try:
                    acct = self.engine.api.get_account()
                    equity = float(acct.equity)
                    pl = float(acct.equity) - float(acct.last_equity)
                    color = "#00C853" if pl >= 0 else "#D32F2F"
                    self.lbl_account.configure(text=f"Total Equity: ${equity:,.2f} (Day P/L: ${pl:.2f})", text_color=color)
                    
                    status = self.engine.market_status
                    if status == "BULL":
                        self.lbl_market_status.configure(text="MARKET: BULL 🐂 (TRADING ON)", text_color="#00C853")
                    elif status == "BEAR":
                        self.lbl_market_status.configure(text="MARKET: BEAR 🐻 (BUYS DISABLED)", text_color="#D32F2F")
                    else:
                        self.lbl_market_status.configure(text="MARKET: CHECKING...", text_color="gray")
                except Exception as e:
                    self.lbl_account.configure(text="Account: Connection Lost...", text_color="orange")
                    # Avoid spamming when API is down
                    try:
                        from .log_throttle import log_exception_throttled
                        log_exception_throttled(self.log, "E_ACCT_FETCH", e, key="acct_fetch", throttle_sec=120)
                    except Exception:
                        pass
            mini = bool(getattr(self, "_miniplayer_enabled", False))

            # v5.13.2 updateA: throttle heavy redraw loops when miniplayer is enabled.
            # Keep health widget live even in mini-player so Dashboard does not remain
            # stuck at "Health: loading...".
            if hasattr(self, 'dashboard_logic'):
                if not mini:
                    self.dashboard_logic.update()
                else:
                    try:
                        self.dashboard_logic.update_health_widget()
                    except Exception:
                        pass

            # Positions tab redraw can be expensive; only refresh it while in miniplayer
            # when the user is actually viewing that tab.
            try:
                active_tab = self.tab_view.get() if hasattr(self, 'tab_view') else ""
            except Exception:
                active_tab = ""

            if (not mini) or (active_tab == "Active Positions"):
                self.update_positions_tab()
        except Exception as e:
            try:
                from .log_throttle import log_exception_throttled
                log_exception_throttled(self.log, "E_UI_LOOP", e, key="ui_loop", throttle_sec=60)
            except Exception:
                pass
        try:
            self.after(self._ui_loop_interval_ms(), self.update_ui_loop)
        except Exception:
            # fallback
            self.after(2000, self.update_ui_loop)

    def call_ui(self, func, *args, **kwargs):
        """Run a callable on the UI thread.

        Tk/CTk is not thread-safe. This helper lets worker threads request a UI
        update without directly touching widgets (including via .after()).
        """
        if threading.get_ident() == self._main_thread_id:
            try:
                func(*args, **kwargs)
            except Exception:
                pass
        else:
            try:
                self._ui_queue.put((func, args, kwargs))
            except Exception:
                pass


    # --- Dynamic watchlist policy (v5.13.1 updateB) ---
    def _watchlist_auto_update_interval_ms(self) -> int:
        """Return interval in milliseconds for periodic auto-watchlist runs."""
        try:
            raw = self.config.get("CONFIGURATION", "watchlist_auto_update_interval_min", fallback="30")
            minutes = int(float(str(raw).strip()))
        except Exception:
            minutes = 30

        if minutes < 1:
            minutes = 1
        if minutes > 1440:
            minutes = 1440
        return int(minutes * 60 * 1000)


    def _watchlist_policy_tick(self):
        """Periodic (after()) tick: runs policy in a worker thread when enabled."""
        # Schedule next tick first so periodic behavior stays stable.
        try:
            self._watchlist_policy_after_id = self.after(self._watchlist_auto_update_interval_ms(), self._watchlist_policy_tick)
        except Exception:
            self._watchlist_policy_after_id = None

        try:
            enabled = str(self.config.get("CONFIGURATION", "watchlist_auto_update_enabled", fallback="False")).strip().lower() in ("1","true","yes","y","on")
        except Exception:
            enabled = False

        if not enabled:
            return

        # Prevent overlapping runs.
        if getattr(self, "_watchlist_policy_in_progress", False):
            return

        self._watchlist_policy_in_progress = True
        threading.Thread(target=self._watchlist_policy_worker, kwargs={"manual": False}, daemon=True).start()


    def apply_watchlist_policy_now(self):
        """Manual one-shot apply (runs in a worker thread)."""
        try:
            if not messagebox.askyesno("Dynamic Watchlist", "Apply the watchlist policy now?\n\nThis may add/remove symbols and will create a config backup if changes are made."):
                return
        except Exception:
            pass

        if getattr(self, "_watchlist_policy_in_progress", False):
            try:
                messagebox.showinfo("Dynamic Watchlist", "A watchlist policy run is already in progress.")
            except Exception:
                pass
            return

        self._watchlist_policy_in_progress = True
        threading.Thread(target=self._watchlist_policy_worker, kwargs={"manual": True}, daemon=True).start()


    def _watchlist_policy_worker(self, manual: bool = False):
        try:
            try:
                mode_cfg = str(self.config.get("CONFIGURATION", "watchlist_auto_update_mode", fallback="ADD")).strip().upper()
            except Exception:
                mode_cfg = "ADD"

            source = "watchlist_manual" if manual else "watchlist_auto"

            api = None
            try:
                api = getattr(self.engine, "api", None) if self.engine else None
            except Exception:
                api = None

            # Ensure UI-safe callbacks.
            refresh_cb = lambda: self.call_ui(self.refresh_symbol_dropdowns)

            res = apply_watchlist_policy(
                config=self.config,
                db=self.db_manager,
                paths=self.paths,
                api=api,
                log=self.log,
                source=source,
                backup_cb=self.backup_config,
                write_cb=self.write_config,
                refresh_cb=refresh_cb,
                apply_mode=(mode_cfg if manual else None),
            )

            if res and res.get("changed"):
                try:
                    add_n = len(res.get("added") or [])
                    rem_n = len(res.get("removed") or [])
                    self.log(f"🧭 [WATCHLIST] Policy applied ({res.get('mode')}): +{add_n} -{rem_n} (batch {res.get('batch_id')})")
                except Exception:
                    pass
        except Exception as e:
            try:
                self.log(f"❌ [WATCHLIST] Policy run failed: {e}")
            except Exception:
                pass
        finally:
            try:
                self._watchlist_policy_in_progress = False
            except Exception:
                pass


    # --- Architect queue API (v5.13.0 updateB) ---
    def get_architect_queue(self):
        try:
            if hasattr(self.db_manager, 'architect_queue_list'):
                rows = self.db_manager.architect_queue_list(statuses=['NEW', 'RUNNING', 'DONE', 'FAILED'], limit=1000)
                if isinstance(rows, list):
                    return rows
        except Exception:
            pass
        try:
            return list(self._architect_queue)
        except Exception:
            return []

    def clear_architect_queue(self):
        try:
            if hasattr(self.db_manager, 'architect_queue_clear'):
                self.db_manager.architect_queue_clear()
            self._architect_queue = []
            self._architect_queue_counter = 0
            self.log("[ARCH_QUEUE] Cleared.")
        except Exception:
            pass

    def add_architect_variant(self, source_symbol, variant_dict):
        """Queue a single Architect genome for orchestrated backtests."""
        try:
            try:
                from .backtest_runner import _coerce_genome
            except Exception:
                def _coerce_genome(d):
                    return d

            genome = _coerce_genome(dict(variant_dict or {}))
            key = (genome.get('rsi'), genome.get('sl'), genome.get('tp'), bool(genome.get('ema')))

            try:
                for it in self._architect_queue:
                    g = it.get('genome') or {}
                    k = (g.get('rsi'), g.get('sl'), g.get('tp'), bool(g.get('ema')))
                    if k == key:
                        self.log(f"[ARCH_QUEUE] Skipped duplicate: {key}")
                        return False
            except Exception:
                pass

            item_id = ""
            metrics = {}
            for k in ('profit', 'trades', 'win_rate', 'score'):
                try:
                    if variant_dict and k in variant_dict:
                        metrics[k] = variant_dict.get(k)
                except Exception:
                    pass

            if hasattr(self.db_manager, 'architect_queue_enqueue'):
                item_id = self.db_manager.architect_queue_enqueue(source_symbol, genome, metrics=metrics)

            if not item_id:
                self._architect_queue_counter += 1
                item_id = f"ARCH{self._architect_queue_counter:03d}"

            item = {
                'id': item_id,
                'created_at_utc': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                'source_symbol': str(source_symbol or '').upper(),
                'genome': genome,
            }

            item.update(metrics)

            self._architect_queue.append(item)
            self.log(f"[ARCH_QUEUE] Added {item_id} from {item['source_symbol']} | {genome}")
            # If the Orchestrator popup is open, refresh it immediately
            try:
                pop = getattr(self, '_orchestrator_popup', None)
                if pop is not None and hasattr(pop, 'winfo_exists') and pop.winfo_exists():
                    self.call_ui(pop.load_queue)
            except Exception:
                pass
            return True
        except Exception as e:
            try:
                self.log(f"[ARCH_QUEUE] Add failed: {type(e).__name__}: {e}")
            except Exception:
                pass
            return False

    def open_backtest_orchestrator(self):
        try:
            # Keep a reference so QUEUE actions can refresh the table live.
            self._orchestrator_popup = BacktestOrchestrator(self)
            try:
                self._orchestrator_popup.focus()
            except Exception:
                pass
        except Exception as e:
            try:
                messagebox.showerror("Orchestrator", f"Failed to open orchestrator: {e}")
            except Exception:
                pass

    def _process_queues(self):
        """Drain cross-thread queues and update UI safely on main thread."""
        # 1) Log messages
        while True:
            try:
                msg = self._log_queue.get_nowait()
            except queue.Empty:
                break
            try:
                self._insert_log(msg)
            except Exception:
                pass

        # 2) Inspector chart refresh requests
        while True:
            try:
                sym = self._chart_queue.get_nowait()
            except queue.Empty:
                break
            try:
                # v5.13.2 updateA: miniplayer pauses heavy chart redraws.
                if bool(getattr(self, "_miniplayer_enabled", False)):
                    continue
                if hasattr(self, 'inspector_logic') and self.inspector_logic.symbol_select.get() == sym:
                    self.inspector_logic.draw_chart(sym)
            except Exception:
                pass

        # 3) Arbitrary UI tasks
        while True:
            try:
                func, args, kwargs = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                func(*args, **kwargs)
            except Exception:
                pass

        # re-schedule
        try:
            self.after(100, self._process_queues)
        except Exception:
            pass

    def trigger_chart_update(self, symbol):
        # v5.13.2 updateA: miniplayer pauses chart redraw triggers.
        if bool(getattr(self, "_miniplayer_enabled", False)):
            return
        if not hasattr(self, 'inspector_logic'):
            return
        if threading.get_ident() == self._main_thread_id:
            try:
                if self.inspector_logic.symbol_select.get() == symbol:
                    self.inspector_logic.draw_chart(symbol)
            except Exception:
                pass
        else:
            try:
                self._chart_queue.put(symbol)
            except Exception:
                pass

    def setup_positions_tab(self):
        self.pos_frame = ctk.CTkScrollableFrame(self.tab_pos, width=1000, height=600)
        self.pos_frame.pack(fill="both", expand=True, padx=10, pady=10)
        h = ctk.CTkFrame(self.pos_frame); h.pack(fill="x")
        cols = ["Symbol", "Qty", "Entry Price", "Current Price", "P/L ($)", "Action"]
        for c in cols: ctk.CTkLabel(h, text=c, width=150, font=("Arial", 12, "bold")).pack(side="left")

    def update_positions_tab(self):
        for widget in self.pos_frame.winfo_children():
            if isinstance(widget, ctk.CTkFrame) and widget.winfo_children()[0].cget("text") != "Symbol": widget.destroy()
        trades = self.db_manager.get_active_trades()
        for symbol, data in trades.items():
            row = ctk.CTkFrame(self.pos_frame); row.pack(fill="x", pady=2)
            curr_price = data['entry_price']
            try: 
                if '/' in symbol:
                    curr_price = self.engine.api.get_crypto_bars(symbol, '1Min', limit=1).df.iloc[-1]['close']
                else:
                    q = self.engine.api.get_latest_trade(symbol); curr_price = q.price
            except Exception as e:
                try:
                    from .log_throttle import log_exception_throttled
                    log_exception_throttled(self.log, "E_POS_PRICE_FETCH", e, key=f"pos_price_{symbol}", throttle_sec=300, context={"symbol": symbol})
                except Exception:
                    pass
            pl = (curr_price - data['entry_price']) * data['qty']
            pl_color = "#00C853" if pl >= 0 else "#D32F2F"
            ctk.CTkLabel(row, text=symbol, width=150).pack(side="left")
            ctk.CTkLabel(row, text=str(data['qty']), width=150).pack(side="left")
            ctk.CTkLabel(row, text=f"${data['entry_price']:.2f}", width=150).pack(side="left")
            ctk.CTkLabel(row, text=f"${curr_price:.2f}", width=150).pack(side="left")
            ctk.CTkLabel(row, text=f"${pl:.2f}", width=150, text_color=pl_color).pack(side="left")
            ctk.CTkButton(row, text="CLOSE", width=100, fg_color="#D32F2F", command=lambda s=symbol: self.force_close(s)).pack(side="left", padx=10)

    def setup_portfolio_tab(self):
        self.stats_frame = ctk.CTkFrame(self.tab_hist)
        self.stats_frame.pack(fill="x", padx=10, pady=10)
        self.lbl_total_pl = ctk.CTkLabel(self.stats_frame, text="Total P/L: $0.00", font=("Arial", 18, "bold"), text_color="white")
        self.lbl_total_pl.pack(side="left", padx=20, pady=20)
        self.lbl_win_rate = ctk.CTkLabel(self.stats_frame, text="Win Rate: 0%", font=("Arial", 18, "bold"), text_color="white")
        self.lbl_win_rate.pack(side="left", padx=20)
        self.hist_list = ctk.CTkTextbox(self.tab_hist, width=1000, height=500)
        self.hist_list.pack(fill="both", expand=True, padx=10, pady=10)
        self.refresh_hist_btn = ctk.CTkButton(self.tab_hist, text="Refresh Stats", command=self.update_portfolio_tab)
        self.refresh_hist_btn.pack(pady=10)

    def update_portfolio_tab(self):
        stats = self.db_manager.get_portfolio_stats()
        pl = stats['total_pl']
        color = "#00C853" if pl >= 0 else "#D32F2F"
        self.lbl_total_pl.configure(text=f"Total P/L: ${pl:,.2f}", text_color=color)
        wr = stats['win_rate']
        wr_color = "#00C853" if wr > 50 else "#FFA000"
        self.lbl_win_rate.configure(text=f"Win Rate: {wr:.1f}% ({stats['wins']}W / {stats['losses']}L)", text_color=wr_color)
        self.hist_list.delete("0.0", "end")
        history = self.db_manager.get_recent_history()
        self.hist_list.insert("end", f"{'TIME':<20} | {'SYMBOL':<10} | {'P/L':<10} | {'STRATEGY'}\n")
        self.hist_list.insert("end", "-"*70 + "\n")
        for row in history:
            # row: (symbol, qty, entry_price, exit_price, profit_loss, strategy, entry_time, exit_time)
            try:
                ts = row[7] if len(row) > 7 and row[7] is not None else (row[6] if len(row) > 6 else '')
            except Exception:
                ts = ''
            try:
                pl_val = float(row[4] or 0.0) if len(row) > 4 else 0.0
            except Exception:
                pl_val = 0.0
            strat = row[5] if len(row) > 5 else ''
            line = f"{str(ts):<20} | {str(row[0]):<10} | ${pl_val:<9.2f} | {str(strat)}\n"
            self.hist_list.insert("end", line)

    def setup_backtest_tab(self):
        bt_ctrl = ctk.CTkFrame(self.tab_backtest)
        bt_ctrl.pack(fill="x", padx=10, pady=10)
        ctk.CTkButton(bt_ctrl, text="RUN FULL BACKTEST", width=200, height=40, fg_color="#7B1FA2", command=self.run_full_backtest_thread).pack(side="right", padx=10)
        ctk.CTkButton(bt_ctrl, text="ARCH ORCHESTRATOR", width=200, height=40, fg_color="#455A64", command=self.open_backtest_orchestrator).pack(side="right", padx=10)
        ctk.CTkButton(bt_ctrl, text="EXPORT BACKTEST RESULTS", width=220, height=40, fg_color="#1976D2", command=self.export_backtest_results_thread).pack(side="right", padx=10)
        ctk.CTkLabel(bt_ctrl, text="Backtest Lab (Sorting Enabled)", font=("Arial", 16, "bold")).pack(side="left", padx=10)
        self.bt_container = ctk.CTkFrame(self.tab_backtest)
        self.bt_container.pack(fill="both", expand=True, padx=10, pady=5)
        self.tree_left = ttk.Treeview(self.bt_container, show="headings", selectmode="browse")
        self.tree_left["columns"] = ("symbol")
        self.tree_left.heading("symbol", text="SYMBOL", command=lambda: self.sort_backtest_data("symbol"))
        self.tree_left.column("symbol", width=100, stretch=False, anchor="center")
        self.tree_left.pack(side="left", fill="y")
        self.tree_right = ttk.Treeview(self.bt_container, show="headings", selectmode="browse")
        self.tree_right.pack(side="left", fill="both", expand=True)
        self.vsb = ttk.Scrollbar(self.bt_container, orient="vertical", command=self.sync_scroll_y)
        self.vsb.pack(side="right", fill="y")
        self.hsb = ttk.Scrollbar(self.tab_backtest, orient="horizontal", command=self.tree_right.xview)
        self.hsb.pack(fill="x", padx=10, pady=5)
        self.tree_left.configure(yscrollcommand=None)
        self.tree_right.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)
        
        self.tree_left.bind("<<TreeviewSelect>>", self.on_left_select)
        self.tree_right.bind("<<TreeviewSelect>>", self.on_right_select)
        self.tree_left.bind("<MouseWheel>", self.sync_mousewheel)
        self.tree_right.bind("<MouseWheel>", self.sync_mousewheel)
        
        # v4.0.2: Configure tags for red/green colors
        self.tree_right.tag_configure("win", foreground="#00C853")
        self.tree_right.tag_configure("loss", foreground="#FF1744")
        self.tree_right.tag_configure("neutral", foreground="white")
        
        self.refresh_backtest_ui() 

    def setup_strategies_tab(self):
        self.strat_scroll = ctk.CTkScrollableFrame(self.tab_strat, width=1000, height=600)
        self.strat_scroll.pack(fill="both", expand=True, padx=10, pady=10)
        ctk.CTkButton(self.tab_strat, text="(+) Create Strategy", command=self.create_new_strategy, fg_color="#00C853").pack(pady=10)
        self.load_strategies_tab()

    def setup_settings_tab(self):
        w_frame = ctk.CTkFrame(self.tab_set); w_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(w_frame, text="Wallet & Risk", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=5)
        row1 = ctk.CTkFrame(w_frame, fg_color="transparent"); row1.pack(fill="x", padx=10)
        ctk.CTkLabel(row1, text="Total Capital ($):", width=150).pack(side="left")
        self.ent_cap = ctk.CTkEntry(row1); self.ent_cap.pack(side="left"); self.ent_cap.insert(0, self.config['CONFIGURATION']['amount_to_trade'])
        ctk.CTkLabel(row1, text="Max % Per Stock (0.1-1.0):", width=200).pack(side="left")
        self.ent_pct = ctk.CTkEntry(row1); self.ent_pct.pack(side="left"); self.ent_pct.insert(0, self.config['CONFIGURATION']['max_percent_per_stock'])
        
        row1b = ctk.CTkFrame(w_frame, fg_color="transparent"); row1b.pack(fill="x", padx=10, pady=5)
        self.sw_comp = ctk.CTkSwitch(row1b, text="Enable Auto-Compounding (Use Active Equity)")
        self.sw_comp.pack(side="left")
        if self.config['CONFIGURATION'].get('compounding_enabled', 'False').lower() == 'true': self.sw_comp.select()

        row1c = ctk.CTkFrame(w_frame, fg_color="transparent"); row1c.pack(fill="x", padx=10, pady=5)
        self.sw_ai = ctk.CTkSwitch(row1c, text="Enable AI Position Sizing (The Quant)", text_color="#F57C00")
        self.sw_ai.pack(side="left")
        if self.config['CONFIGURATION'].get('ai_sizing_enabled', 'False').lower() == 'true': self.sw_ai.select()

        row1d = ctk.CTkFrame(w_frame, fg_color="transparent"); row1d.pack(fill="x", padx=10)
        ctk.CTkLabel(row1d, text="Kill Switch ($):", width=150).pack(side="left")
        self.ent_kill = ctk.CTkEntry(row1d); self.ent_kill.pack(side="left"); self.ent_kill.insert(0, self.config['CONFIGURATION'].get('max_daily_loss', '100'))

        agent_frame = ctk.CTkFrame(self.tab_set); agent_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(agent_frame, text="AI Agent Control", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=5)
        row_agent = ctk.CTkFrame(agent_frame, fg_color="transparent"); row_agent.pack(fill="x", padx=10)
        ctk.CTkLabel(row_agent, text="Agent Mode:", width=150).pack(side="left")
        self.opt_agent_mode = ctk.CTkOptionMenu(row_agent, values=["OFF", "ADVISORY", "PAPER", "LIVE"])
        self.opt_agent_mode.pack(side="left")
        self.opt_agent_mode.set(self.config['CONFIGURATION'].get('agent_mode', 'OFF').upper())

        e_frame = ctk.CTkFrame(self.tab_set); e_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(e_frame, text="Engine Speed", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=5)
        row2 = ctk.CTkFrame(e_frame, fg_color="transparent"); row2.pack(fill="x", padx=10)
        ctk.CTkLabel(row2, text="Scan Interval:", width=150).pack(side="left")
        self.opt_speed = ctk.CTkOptionMenu(row2, values=["Fast (30s)", "Normal (60s)", "Slow (5m)"])
        self.opt_speed.pack(side="left")
        curr_speed = self.config['CONFIGURATION'].get('update_interval_sec', '60')
        if curr_speed == '30': self.opt_speed.set("Fast (30s)")
        elif curr_speed == '300': self.opt_speed.set("Slow (5m)")
        else: self.opt_speed.set("Normal (60s)")

        c_frame = ctk.CTkFrame(self.tab_set); c_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(c_frame, text="Credentials", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=5)
        row3 = ctk.CTkFrame(c_frame, fg_color="transparent"); row3.pack(fill="x", padx=10)
        ctk.CTkLabel(row3, text="Alpaca Key:", width=100).pack(side="left")
        self.ent_key = ctk.CTkEntry(row3, show="*"); self.ent_key.pack(side="left", padx=5); self.ent_key.insert(0, self.config['KEYS']['alpaca_key'])
        ctk.CTkLabel(row3, text="Secret:", width=60).pack(side="left")
        self.ent_sec = ctk.CTkEntry(row3, show="*"); self.ent_sec.pack(side="left", padx=5); self.ent_sec.insert(0, self.config['KEYS']['alpaca_secret'])
        ctk.CTkLabel(row3, text="Telegram Token:", width=120).pack(side="left")
        self.ent_tel = ctk.CTkEntry(row3, show="*"); self.ent_tel.pack(side="left", padx=5); self.ent_tel.insert(0, self.config['KEYS']['telegram_token'])
        row4 = ctk.CTkFrame(c_frame, fg_color="transparent"); row4.pack(fill="x", padx=10, pady=5)
        self.sw_tel = ctk.CTkSwitch(row4, text="Enable Telegram Notifications")
        self.sw_tel.pack(side="left")
        if self.config['KEYS'].get('telegram_enabled', 'True') == 'True': self.sw_tel.select()


        d_frame = ctk.CTkFrame(self.tab_set); d_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(d_frame, text="DB Tools", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=5)
        rowd = ctk.CTkFrame(d_frame, fg_color="transparent"); rowd.pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(rowd, text="DB HEALTH", command=self.db_health).pack(side="left", padx=5)
        ctk.CTkButton(rowd, text="REPAIR BACKTEST DB", command=self.db_repair_backtest).pack(side="left", padx=5)
        ctk.CTkButton(rowd, text="OPEN DB FOLDER", command=self.open_db_folder).pack(side="left", padx=5)

        a_frame = ctk.CTkFrame(self.tab_set); a_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkButton(a_frame, text="SAVE SETTINGS", fg_color="#00C853", command=self.save_global_settings).pack(side="right", padx=10, pady=10)
        ctk.CTkButton(a_frame, text="Open Logs Folder", command=self.open_logs).pack(side="left", padx=10, pady=10)
        ctk.CTkButton(a_frame, text="Factory Reset", fg_color="#D32F2F", command=self.factory_reset).pack(side="left", padx=10, pady=10)

    # --- HELPERS ---

    def save_global_settings(self):
        """Save global settings from the Settings tab.

        Save-gate: validate before writing to disk. This is intentionally strict
        on type/range checks even when strict_config_validation is OFF.
        """
        try:
            # Snapshot for revert if validation fails
            old = {}

            def _snap(sec: str, key: str):
                try:
                    old[(sec, key)] = self.config.get(sec, key, fallback=None)
                except Exception:
                    try:
                        old[(sec, key)] = self.config[sec].get(key)
                    except Exception:
                        old[(sec, key)] = None

            # Compute update interval
            spd = self.opt_speed.get()
            val = '30' if "Fast" in spd else '300' if "Slow" in spd else '60'

            keys_to_set = [
                ("CONFIGURATION", "amount_to_trade", self.ent_cap.get()),
                ("CONFIGURATION", "max_percent_per_stock", self.ent_pct.get()),
                ("CONFIGURATION", "max_daily_loss", self.ent_kill.get()),
                ("CONFIGURATION", "compounding_enabled", str(bool(self.sw_comp.get()))),
                ("CONFIGURATION", "ai_sizing_enabled", str(bool(self.sw_ai.get()))),
                ("CONFIGURATION", "agent_mode", str(self.opt_agent_mode.get()).upper()),
                ("CONFIGURATION", "update_interval_sec", val),
                ("KEYS", "alpaca_key", self.ent_key.get()),
                ("KEYS", "alpaca_secret", self.ent_sec.get()),
                ("KEYS", "telegram_token", self.ent_tel.get()),
                ("KEYS", "telegram_enabled", str(bool(self.sw_tel.get()))),
            ]

            for sec, key, _v in keys_to_set:
                _snap(sec, key)

            # Backup current on-disk config prior to mutation
            try:
                self.backup_config()
            except Exception:
                pass

            # Apply changes to in-memory config
            for sec, key, v in keys_to_set:
                try:
                    if not self.config.has_section(sec):
                        self.config.add_section(sec)
                except Exception:
                    pass
                try:
                    self.config[sec][key] = str(v)
                except Exception:
                    try:
                        self.config.set(sec, key, str(v))
                    except Exception:
                        pass

            # If strict validation is enabled, require broker creds at save-time.
            strict_startup = False
            try:
                strict_startup = str(self.config["CONFIGURATION"].get("strict_config_validation", "False")).strip().lower() in (
                    "1", "true", "yes", "y", "on"
                )
            except Exception:
                strict_startup = False

            repv = validate_runtime_config(
                self.config,
                strict=True,
                require_credentials=strict_startup,
                include_credentials=True,
            )

            if repv.errors:
                # Revert in-memory config to previous values
                for (sec, key), prev in old.items():
                    try:
                        if prev is None:
                            try:
                                if self.config.has_option(sec, key):
                                    self.config.remove_option(sec, key)
                            except Exception:
                                pass
                        else:
                            self.config[sec][key] = str(prev)
                    except Exception:
                        pass

                err_lines = "\n".join([f"- {e}" for e in repv.errors])
                try:
                    messagebox.showerror(
                        "Settings Validation Failed",
                        "Settings were not saved. Fix the errors and try again.\n\nErrors:\n" + err_lines,
                    )
                except Exception:
                    pass
                try:
                    self.log(f"❌ Settings not saved: validation failed ({len(repv.errors)} error(s)).")
                except Exception:
                    pass
                return

            # Log warnings (non-blocking)
            if getattr(repv, "warnings", None):
                for w in repv.warnings:
                    try:
                        self.log(f"⚠️ {w}")
                    except Exception:
                        pass

            self.write_config()
            self.engine.reload_strategies()
            if self.agent_master:
                self.agent_master.set_mode(self.config['CONFIGURATION']['agent_mode'])
            self.log("✅ Global Settings Saved & Reloaded.")
        except Exception as e:
            self.log(f"❌ Save Failed: {e}")


    def open_logs(self): 
        subprocess.Popen(f'explorer "{self.paths["logs"]}"')

    def export_backtest_results_thread(self):
        """Export the current backtest snapshot to logs/backtest/."""
        def _worker():
            self.export_backtest_results(silent=False)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()


    def export_backtest_results(self, silent: bool = False):
        """Write a backtest export bundle (JSON + optional CSV).

        Uses the most recently loaded backtest snapshot (self.backtest_df).
        """
        try:
            if self.backtest_df is None or getattr(self.backtest_df, "empty", False):
                if not silent:
                    messagebox.showwarning("Backtest Export", "No backtest results are loaded yet. Run a backtest first.")
                return None

            out_dir = os.path.join(self.paths["logs"], "backtest")
            out = export_backtest_bundle(
                backtest_df=self.backtest_df,
                config=self.config,
                out_dir=out_dir,
                app_version=APP_VERSION,
                app_release=APP_RELEASE,
                include_csv=True,
            )

            if out and not silent:
                msg = f"Backtest export written to:\n{out.get('json')}"
                if out.get('csv'):
                    msg += f"\n\nCSV snapshot:\n{out.get('csv')}"
                messagebox.showinfo("Backtest Export", msg)

                # Convenience: open logs root; user can drill into logs\backtest.
                try:
                    self.open_logs()
                except Exception:
                    pass

            return out
        except Exception as e:
            try:
                self.log(f"❌ Backtest export failed: {e}")
            except Exception:
                pass
            if not silent:
                messagebox.showerror("Backtest Export", f"Export failed:\n{e}")
            return None



    def open_db_folder(self):
        try:
            subprocess.Popen(f'explorer "{self.paths["db_dir"]}"')
        except Exception:
            try:
                subprocess.Popen(f'explorer "{self.paths.get("db_dir")}"')
            except Exception:
                pass

    def db_health(self):
        def _worker():
            try:
                out = self.db_manager.export_db_health_report(out_dir=self.paths["logs"])
            except Exception:
                out = None

            if out:
                try:
                    self.log(f"[DB] ✅ Health report written: {out}")
                except Exception:
                    pass
                try:
                    self._ui_queue.put((messagebox.showinfo, ("DB Health", f"Report written to:\n{out}"), {}))
                except Exception:
                    pass
            else:
                try:
                    self.log("[DB] ⚠️ Health report failed.")
                except Exception:
                    pass
                try:
                    self._ui_queue.put((messagebox.showerror, ("DB Health", "Failed to generate DB health report."), {}))
                except Exception:
                    pass

        threading.Thread(target=_worker, daemon=True).start()

    def db_repair_backtest(self):
        def _worker():
            try:
                ok, msg = self.db_manager.repair_backtest_results_db()
            except Exception as e:
                ok, msg = False, str(e)

            prefix = "✅" if ok else "⚠️"
            try:
                self.log(f"[DB] {prefix} {msg}")
            except Exception:
                pass
            try:
                fn = messagebox.showinfo if ok else messagebox.showerror
                self._ui_queue.put((fn, ("Backtest DB Repair", msg), {}))
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def export_log_file(self):
        try:
            timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
            filename = f"[LIVE LOG] [{timestamp}].txt"
            filepath = os.path.join(self.paths['logs'], filename)

            # Release E2: Prefer exporting the file-backed runtime log if enabled
            rt = getattr(self, "_runtime_log_path", None)
            if getattr(self, "_log_file_enabled", False) and rt and os.path.exists(rt):
                shutil.copy(rt, filepath)
            else:
                content = self.log_box.get("1.0", "end")
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

            messagebox.showinfo("Export Success", f"Log saved to:\n{filename}")
            self.open_logs()
        except Exception as e:
            messagebox.showerror("Export Failed", str(e))


    # --------------------
    # Release E3: Summary export + explicit AI retrain
    # --------------------
    def export_run_summary(self, silent=False):
        """Write an end-of-run JSON summary (same artifact the engine writes on stop).

        If silent=True, no UI popups are shown (used during app close).
        """
        try:
            if not self.engine:
                return None
            out_path = self.engine.export_run_summary(force=True)
            if out_path and (not silent):
                fn = os.path.basename(out_path)
                messagebox.showinfo("Export Success", f"Run summary saved to:\n{fn}")
                self.open_logs()
            return out_path
        except Exception as e:
            if not silent:
                messagebox.showerror("Export Failed", str(e))
            return None

    def export_strategy_report(self, silent: bool = False):
        """Export a strategy selection report (best strategy per symbol snapshot)."""
        try:
            out_path = None
            if self.engine:
                out_path = self.engine.export_strategy_selection_report(force=True)
            if out_path:
                if not silent:
                    messagebox.showinfo("Strategy Report Exported", f"Strategy report saved to:\n{out_path}")
                    self.open_logs()
                return out_path
            else:
                if not silent:
                    messagebox.showwarning("No Backtest Data", "Backtest results are empty.\nRun a full backtest first, then export again.")
                return None
        except Exception as e:
            if not silent:
                messagebox.showerror("Export Failed", str(e))
            return None

    def _start_ai_retrain(self, force: bool = True):
        """Start AI training in a worker thread, with a simple in-progress guard.

        Returns True if a new training thread was started, False otherwise.
        """
        try:
            if not getattr(self, 'engine', None) or not getattr(self.engine, 'ai', None):
                return False
            if getattr(self, '_ai_retrain_in_progress', False):
                return False
            self._ai_retrain_in_progress = True

            def _worker():
                try:
                    # force retrain when requested (default True)
                    self.engine.ai.train_model(force=bool(force))
                except Exception as e:
                    try:
                        self.log(f"🧠 [AI] Training thread crashed: {e}")
                    except Exception:
                        pass
                finally:
                    self._ai_retrain_in_progress = False

            threading.Thread(target=_worker, daemon=True).start()
            return True
        except Exception:
            try:
                self._ai_retrain_in_progress = False
            except Exception:
                pass
            return False



    def retrain_ai_model(self):
        """Explicit AI retrain trigger (runs in a worker thread)."""
        try:
            if not self.engine or not getattr(self.engine, 'ai', None):
                self.log("🧠 [AI] Engine/AI not ready.")
                return

            started = self._start_ai_retrain(force=True)
            if started:
                self.log("🧠 [AI] Retrain started...")
            else:
                self.log("🧠 [AI] Retrain already running...")
        except Exception as e:
            self.log(f"🧠 [AI] Retrain failed to start: {e}")



    def factory_reset(self):
        if messagebox.askyesno("Factory Reset", "Restore default config? This overrides everything!"):
            try:
                self.backup_config()
                ensure_split_config_layout(self.paths, force_defaults=True)
                self.log("⚠️ Factory Reset Complete. Please restart app.")
            except Exception as e:
                self.log(f"❌ Factory Reset Failed: {e}")

    def backup_config(self):
        try:
            ts = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
            # v5.12.7: keep config backups in TradingBot/backups/config/<timestamp>/
            folder = os.path.join(self.paths['backup'], 'config', ts)
            os.makedirs(folder, exist_ok=True)

            # Back up split config files (best-effort)
            for k in ('keys_ini', 'configuration_ini', 'watchlist_ini', 'strategy_ini'):
                fp = self.paths.get(k)
                if fp and os.path.exists(fp):
                    shutil.copy(fp, folder)

            self.log(f"📦 Config backed up to: {folder}")
            return folder
        except Exception as e:
            self.log(f"Backup Failed: {e}")
        return None

    def write_config(self):
        try:
            write_split_config(self.config, self.paths)
        except Exception as e:
            self.log(f"❌ Config Write Failed: {e}")

    def force_close(self, s):

        threading.Thread(target=self.engine.execute_sell, args=(s, 1, 0)).start()

    def delete_strategy(self, n):
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete {n}?"):
            self.backup_config()
            self.config.remove_section(f"STRATEGY_{n}")
            self.write_config()
            self.load_strategies_tab()
            self.log(f"🗑️ Strategy '{n}' deleted.")

    def create_new_strategy(self):
        name = ctk.CTkInputDialog(text="Enter Strategy Name:", title="New Strategy").get_input()
        if name and f"STRATEGY_{name}" not in self.config:
            self.config[f"STRATEGY_{name}"] = {'rsi_buy': '30'}
            self.write_config()
            self.load_strategies_tab()

    def edit_strategy(self, n):
        StrategyEditor(self, n, dict(self.config[f"STRATEGY_{n}"]), self.save_strategy_config)

    def save_strategy_config(self, n, d): 
        for k,v in d.items(): 
            self.config[f"STRATEGY_{n}"][k] = v
        self.write_config()
        self.load_strategies_tab()

    def load_strategies_tab(self):
        for w in self.strat_scroll.winfo_children(): 
            w.destroy()
        for s in self.config.sections(): 
            if s.startswith("STRATEGY_"): 
                self.create_strategy_card(s.replace("STRATEGY_", ""), s)

    def create_strategy_card(self, n, s):
        c = ctk.CTkFrame(self.strat_scroll)
        c.pack(fill="x", pady=5)
        ctk.CTkLabel(c, text=n).pack(side="left", padx=10)
        ctk.CTkButton(c, text="EDIT", width=60, command=lambda: self.edit_strategy(n)).pack(side="right", padx=5)
        ctk.CTkButton(c, text="DELETE", width=60, fg_color="#D32F2F", command=lambda: self.delete_strategy(n)).pack(side="right", padx=5)

    def open_decision_viewer(self): 
        DecisionViewer(self, self.db_manager)

    
    def run_single_symbol_backtest_thread(self, symbol: str):
        sym = (symbol or "").strip().upper()
        if not sym:
            return
        if messagebox.askyesno("Backtest", f"Backtest {sym} across all strategies?"):
            threading.Thread(target=self.run_single_symbol_backtest, args=(sym,), daemon=True).start()

    def run_single_symbol_backtest(self, symbol: str):
        sym = (symbol or "").strip().upper()
        if not sym:
            return

        self.log(f"[CANDIDATES] Starting backtest for {sym}...")
        strategies = [s.replace("STRATEGY_", "") for s in self.config.sections() if s.startswith("STRATEGY_")]

        # Ensure backtest_results schema exists without wiping existing results
        try:
            self.db_manager.ensure_backtest_table(strategies)
        except Exception:
            # Fallback to legacy behavior (may wipe)
            try:
                self.db_manager.rebuild_backtest_table(strategies)
            except Exception:
                pass

        df = self.db_manager.get_history(sym, 5000)
        res = {"symbol": sym}

        best_strat = "None"
        best_profit = -999999.0

        for s in strategies:
            try:
                if df is not None and not df.empty:
                    pl, trades = self.simulate_strategy_numpy(df, s, sym)
                    res[f"PL_{s}"] = round(pl, 2)
                    res[f"Trades_{s}"] = trades
                    if pl > best_profit:
                        best_profit = pl
                        best_strat = s
                else:
                    res[f"PL_{s}"] = 0.0
                    res[f"Trades_{s}"] = 0
            except Exception:
                res[f"PL_{s}"] = 0.0
                res[f"Trades_{s}"] = 0

        res["best_strategy"] = best_strat
        res["best_profit"] = round(best_profit, 2) if best_profit != -999999.0 else 0.0
        res["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            self.db_manager.save_backtest_result(res)
        except Exception:
            pass

        try:
            self.call_ui(self.refresh_backtest_ui)
        except Exception:
            pass

        self.log(f"[CANDIDATES] ✅ Backtest complete for {sym} (best={best_strat}, PL={res['best_profit']}).")

    def refresh_symbol_dropdowns(self):
        """Best-effort refresh of watchlist-driven dropdowns (Inspector/Architect)."""
        arch_refreshed = False
        insp_refreshed = False

        # Prefer tab-level refresh so ACTIVE/ARCHIVE/FAVORITES selection is respected.
        try:
            if hasattr(self, "architect_logic") and hasattr(self.architect_logic, "refresh_symbol_list"):
                self.architect_logic.refresh_symbol_list()
                arch_refreshed = True
        except Exception:
            pass

        try:
            if hasattr(self, "inspector_logic") and hasattr(self.inspector_logic, "refresh_symbol_list"):
                self.inspector_logic.refresh_symbol_list()
                insp_refreshed = True
        except Exception:
            pass

        # Fallback: build ACTIVE list only.
        try:
            from .watchlist_api import get_watchlist_symbols
            syms = [str(s).strip().upper() for s in get_watchlist_symbols(self.config, group="ACTIVE", asset="ALL")]
            syms = sorted(list(dict.fromkeys([s for s in syms if s])))
            if not syms:
                syms = ["TQQQ"]
        except Exception:
            syms = ["TQQQ"]

        # Legacy widget-level fallbacks (only when tab refresh isn't available)
        if not arch_refreshed:
            try:
                if hasattr(self, "architect_logic") and hasattr(self.architect_logic, "arch_symbol"):
                    self.architect_logic.arch_symbol.configure(values=syms)
            except Exception:
                pass

        if not insp_refreshed:
            try:
                if hasattr(self, "inspector_logic") and hasattr(self.inspector_logic, "symbol_select"):
                    self.inspector_logic.symbol_select.configure(values=syms)
            except Exception:
                pass


    def run_full_backtest_thread(self): 
        if messagebox.askyesno("Backtest", "Run?"): 
            threading.Thread(target=self.run_full_backtest, daemon=True).start()

    def run_full_backtest(self):
        self.log("Starting Backtest...")
        run_full_backtest_service(
            self.config,
            self.db_manager,
            simulate_strategy=self.simulate_strategy_numpy,
            log=self.log,
            rebuild_table=True,
            sleep_per_symbol_sec=0.05,
        )
        self.call_ui(self.refresh_backtest_ui)

    def refresh_backtest_ui(self):
        # Update Cache
        self.backtest_df = self.db_manager.get_backtest_data()
        self.populate_backtest_ui(self.backtest_df)

    def populate_backtest_ui(self, df):
        self.tree_left.delete(*self.tree_left.get_children())
        self.tree_right.delete(*self.tree_right.get_children())
        
        if df is None or df.empty: return

        # Dynamic Columns with Sorting Bindings
        cols = list(df.columns)
        if 'symbol' in cols: cols.remove('symbol')
        
        self.tree_right["columns"] = cols
        for c in cols:
            self.tree_right.heading(c, text=c, command=lambda _c=c: self.sort_backtest_data(_c))
            self.tree_right.column(c, width=100)

        for i, row in df.iterrows(): 
            self.tree_left.insert("", "end", values=(row['symbol'],))
            
            # Prepare row values and color tags
            vals = [row[c] for c in cols]
            tags = []
            
            # Apply tags based on column name and value
            for idx, col_name in enumerate(cols):
                if col_name.startswith("PL_"):
                    try:
                        val = float(vals[idx])
                        if val > 0: tags.append("win")
                        elif val < 0: tags.append("loss")
                        else: tags.append("neutral")
                    except Exception:
                        tags.append("neutral")
            
            # Treeview only allows one tag per row easily, so we apply the most dominant one logic
            # OR we just insert the item. Note: Tkinter Treeview applies tags to the whole row, 
            # not individual cells. We can't color just one cell red easily.
            # Workaround: We will use the 'best_profit' to determine row color for now.
            
            row_tag = "neutral"
            if 'best_profit' in row and row['best_profit'] > 0: row_tag = "win"
            elif 'best_profit' in row and row['best_profit'] < 0: row_tag = "loss"
            
            self.tree_right.insert("", "end", values=vals, tags=(row_tag,))

    def sort_backtest_data(self, col_name):
        if self.backtest_df is None or self.backtest_df.empty: return
        
        try:
            # Toggle sort order
            self.sort_descending = not self.sort_descending
            
            # Sort DataFrame
            sorted_df = self.backtest_df.sort_values(by=col_name, ascending=not self.sort_descending)
            
            # Repopulate
            self.populate_backtest_ui(sorted_df)
        except Exception as e:
            self.log(f"❌ Sort Error: {e}")

    def toggle_engine(self):
        if not self.engine.active: 
            threading.Thread(target=self.engine.run, daemon=True).start()
            self.toggle_btn.configure(text="STOP")
        else: 
            self.engine.stop()
            self.toggle_btn.configure(text="START")

    def open_backfill_window(self): 
        threading.Thread(target=self.run_backfill_thread, daemon=True).start()
    def run_backfill_thread(self):
        BackfillEngine(self.config, self.db_manager, self.update_backfill_popup).run()

        # Release E3: retrain AI after a DB update completes (safe default; configurable)
        try:
            cfg = self.config['CONFIGURATION'] if 'CONFIGURATION' in self.config else {}
            retrain = self._as_bool(cfg.get('ai_retrain_after_db_update', 'True'), True)
            if retrain and getattr(self, 'engine', None) and getattr(self.engine, 'ai', None):
                self.log('🧠 [AI] Retraining after DB update...')
                # Force retrain because DB contents may have changed
                started = self._start_ai_retrain(force=True)
                if not started:
                    self.log('🧠 [AI] Retrain already running...')
        except Exception:
            pass

    def update_backfill_popup(self, msg): 
        self.log(f"[DB] {msg}")


    # --------------------
    # Release E2: Live Log controls (bounded buffer + runtime file log)
    # --------------------
    def _as_bool(self, v, default=False):
        try:
            if isinstance(v, bool):
                return v
            return str(v).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        except Exception:
            return bool(default)

    def _as_int(self, v, default=0):
        try:
            return int(float(v))
        except Exception:
            return int(default)

    def _init_log_controls(self):
        cfg = self.config['CONFIGURATION'] if 'CONFIGURATION' in self.config else {}
        try:
            self._log_max_lines = max(0, self._as_int(cfg.get('log_max_lines', 2000), 2000))
        except Exception:
            self._log_max_lines = 2000

        try:
            self._log_file_enabled = self._as_bool(cfg.get('log_file_enabled', True), True)
        except Exception:
            self._log_file_enabled = True

        if not self._log_file_enabled:
            self._runtime_log_path = None
            return

        try:
            prefix = str(cfg.get('log_file_prefix', 'RUNTIME')).strip() or 'RUNTIME'
        except Exception:
            prefix = 'RUNTIME'
        try:
            roll_daily = self._as_bool(cfg.get('log_file_roll_daily', True), True)
        except Exception:
            roll_daily = True

        try:
            day = datetime.now().strftime("%Y.%m.%d")
            if roll_daily:
                filename = f"[{APP_VERSION}] [{prefix}] [{day}].log"
            else:
                filename = f"[{APP_VERSION}] [{prefix}].log"
            self._runtime_log_path = os.path.join(self.paths['logs'], filename)
        except Exception:
            self._runtime_log_path = None
            return

        try:
            os.makedirs(self.paths['logs'], exist_ok=True)
        except Exception:
            pass

        try:
            self._file_log_queue = queue.Queue()
            self._file_log_stop = threading.Event()
            threading.Thread(target=self._file_log_worker, daemon=True).start()
            self._queue_file_log(f"--- START {APP_VERSION} | {APP_RELEASE} ---")
        except Exception:
            self._runtime_log_path = None
            self._log_file_enabled = False

    def _queue_file_log(self, msg):
        if not getattr(self, "_log_file_enabled", False):
            return
        if not getattr(self, "_runtime_log_path", None):
            return
        q = getattr(self, "_file_log_queue", None)
        if q is None:
            return
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = str(msg).replace("\r", " ").replace("\n", " ").strip()
            q.put_nowait(f"{ts} | {line}")
        except Exception:
            pass

    def _file_log_worker(self):
        path = getattr(self, "_runtime_log_path", None)
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except Exception:
            pass

        f = None
        try:
            f = open(path, "a", encoding="utf-8", newline="\n")
        except Exception:
            return

        try:
            while True:
                if getattr(self, "_file_log_stop", None) is not None and self._file_log_stop.is_set():
                    break
                try:
                    item = self._file_log_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if item is None:
                    break
                try:
                    f.write(str(item) + "\n")
                    f.flush()
                except Exception:
                    pass
        finally:
            try:
                f.flush()
                f.close()
            except Exception:
                pass

    def _enforce_log_limit(self):
        max_lines = int(getattr(self, "_log_max_lines", 0) or 0)
        if max_lines <= 0:
            return
        try:
            total_lines = int(str(self.log_box.index("end-1c")).split(".")[0])
            if total_lines <= max_lines:
                return
            extra = total_lines - max_lines
            cutoff = min(total_lines - 1, extra + 200)
            self.log_box.delete("1.0", f"{cutoff}.0")
        except Exception:
            pass

    def log(
        self,
        msg,
        level="INFO",
        category=None,
        symbol=None,
        order_id=None,
        strategy=None,
        component=None,
        mode=None,
        **context,
    ):
        """UI log sink + unified structured logging.

        This method is intentionally permissive in accepted kwargs so engine/DB
        can pass structured context without breaking the UI callback signature.
        """

        # Merge structured context
        try:
            if component is None:
                component = context.get("component")
            if mode is None:
                mode = context.get("mode")
            if symbol is None:
                symbol = context.get("symbol")
            if order_id is None:
                order_id = context.get("order_id")
            if strategy is None:
                strategy = context.get("strategy")
        except Exception:
            pass

        if not component:
            component = "ui"

        if not mode:
            # Best-effort: pull from AgentMaster first, then config
            try:
                am = getattr(self, "agent_master", None)
                if am is not None:
                    am_mode = getattr(am, "mode", None)
                    if am_mode:
                        mode = str(am_mode).strip().upper()
            except Exception:
                pass
            if not mode:
                try:
                    # configparser-like
                    if hasattr(self.config, "get"):
                        mode = str(self.config.get("CONFIGURATION", "agent_mode", fallback="OFF")).strip().upper()
                    else:
                        # dict-like
                        mode = str(self.config.get("CONFIGURATION", {}).get("agent_mode", "OFF")).strip().upper()
                except Exception:
                    mode = "OFF"

        # Build a readable prefix for the live log
        parts = []
        try:
            lvl = str(level or "INFO").upper()
            if lvl == "WARN":
                lvl = "WARNING"
            if lvl and lvl != "INFO":
                parts.append(lvl)
        except Exception:
            pass

        if category:
            parts.append(str(category).upper())
        if symbol:
            parts.append(str(symbol).upper())
        if order_id:
            parts.append(f"OID:{order_id}")
        if strategy:
            parts.append(f"STRAT:{strategy}")

        prefix = " ".join([f"[{p}]" for p in parts])
        final_msg = f"{prefix} {msg}".strip() if prefix else str(msg)

        # Unified structured logging (file/stdout)
        try:
            lvl_txt = str(level or "INFO").upper()
            if lvl_txt == "WARN":
                lvl_txt = "WARNING"
            lvl_no = getattr(logging, lvl_txt, logging.INFO)
            extra = {
                "component": str(component),
                "mode": str(mode),
                "symbol": str(symbol).upper() if symbol else "-",
                "order_id": str(order_id) if order_id else "-",
                "strategy": str(strategy) if strategy else "-",
            }
            self._py_logger.log(lvl_no, str(msg), extra=extra)
        except Exception:
            pass

        # Release E2: always queue file-backed log first (thread-safe)
        try:
            self._queue_file_log(final_msg)
        except Exception:
            pass

        # Thread-safe: do NOT call .after() from worker threads.
        if threading.get_ident() == self._main_thread_id:
            try:
                self._insert_log(final_msg)
            except Exception:
                pass
        else:
            try:
                self._log_queue.put(final_msg)
            except Exception:
                pass

    def _insert_log(self, msg):
        self.log_box.insert("end", f"{msg}\n")
        self.log_box.see("end")
        # Release E2: bound Live Log size to prevent UI slowdowns
        try:
            self._enforce_log_limit()
        except Exception:
            pass

    def sync_scroll_y(self, *args): 
        self.tree_left.yview(*args)
        self.tree_right.yview(*args)

    def sync_mousewheel(self, e): 
        self.tree_left.yview_scroll(-1*(e.delta//120), "units")
        self.tree_right.yview_scroll(-1*(e.delta//120), "units")

    def on_left_select(self, e): 
        if self.syncing_selection: return
        self.syncing_selection = True
        try:
            sel = self.tree_left.selection()
            if sel: 
                self.tree_right.selection_set(sel)
                self.tree_right.see(sel)
        finally:
            self.after_idle(lambda: setattr(self, "syncing_selection", False))

    def on_right_select(self, e):
        if self.syncing_selection: return
        self.syncing_selection = True
        try:
            sel = self.tree_right.selection()
            if sel: 
                self.tree_left.selection_set(sel)
                self.tree_left.see(sel)
        finally:
            self.after_idle(lambda: setattr(self, "syncing_selection", False))
            
    def simulate_strategy_numpy(self, df, strat_name, symbol):
        """
        Release B (engine-aligned backtest simulation)

        Entry:
        - Signal computed on bar close.
        - Entry modeled as a LIMIT order at that close; it fills only if a subsequent bar's LOW trades at/through the limit within TTL bars.

        Exit:
        - Bracket SL/TP modeled using intrabar LOW/HIGH.
        - Trailing stop modeled using close-based highest-close logic (mirrors engine behavior).
        - Optional stagnation exit and max-hold exit.

        Notes:
        - This is intentionally conservative versus instant-fill assumptions, reducing "paper edges" that typically fail live.
        """
        try:
            strat_conf = self.config[f"STRATEGY_{strat_name}"]
        except Exception:
            return 0.0, 0

        # --- Strategy params (safe defaults) ---
        buy_rsi = float(strat_conf.get('rsi_buy', 30))
        trail_pct = float(strat_conf.get('trailing_stop_pct', 0.02))
        stop_loss_pct = float(strat_conf.get('stop_loss', 0.02))
        min_rvol = float(strat_conf.get('min_rvol', 0.0))

        cfg = self.config['CONFIGURATION'] if 'CONFIGURATION' in self.config else {}

        def _cfg_get(key, default):
            try:
                return cfg.get(key, default)
            except Exception:
                return default

        # --- Engine-alignment knobs ---
        entry_ttl = int(float(_cfg_get('backtest_entry_ttl_bars', 2)))
        max_hold = int(float(_cfg_get('backtest_max_hold_bars', 240)))
        use_stagnation = str(_cfg_get('backtest_use_stagnation_exit', 'True')).lower() == 'true'
        stag_minutes = int(float(_cfg_get('backtest_stagnation_minutes', 60)))
        stag_min = float(_cfg_get('backtest_stagnation_min_gain', -0.01))
        stag_max = float(_cfg_get('backtest_stagnation_max_gain', 0.003))
        bar_conflict = str(_cfg_get('backtest_bar_conflict', 'STOP_FIRST')).upper()
        atr_stop_mult = float(_cfg_get('backtest_atr_stop_mult', 2.0))
        atr_take_mult = float(_cfg_get('backtest_atr_take_mult', 3.0))

        # --- Required data (fallbacks where possible) ---
        if df is None or df.empty or 'close' not in df:
            return 0.0, 0

        closes = df['close'].values
        n = len(closes)
        if n < 100:
            return 0.0, 0

        opens = df['open'].values if 'open' in df else closes
        highs = df['high'].values if 'high' in df else closes
        lows = df['low'].values if 'low' in df else closes

        if 'rsi' not in df or 'bb_lower' not in df or 'ema_200' not in df:
            return 0.0, 0

        rsis = df['rsi'].values
        bb_lowers = df['bb_lower'].values
        ema_200s = df['ema_200'].values
        bb_uppers = df['bb_upper'].values if 'bb_upper' in df else bb_lowers
        atrs = df['atr'].values if 'atr' in df else None
        volumes = df['volume'].values if 'volume' in df else None

        # --- Relative volume filter (StrategyOptimizer uses it for scoring) ---
        rvol_ok = np.ones(n, dtype=bool)
        if volumes is not None and min_rvol and min_rvol > 0:
            vol = volumes.astype(float)
            vol_avg = np.full(n, np.nan, dtype=float)
            if n >= 20:
                csum = np.cumsum(vol)
                vol_avg[19:] = (csum[19:] - np.concatenate(([0.0], csum[:-20]))) / 20.0
            rvol = np.zeros(n, dtype=float)
            mask = np.isfinite(vol_avg) & (vol_avg > 0)
            rvol[mask] = vol[mask] / vol_avg[mask]
            rvol_ok = rvol >= min_rvol

        # --- Buy signals aligned with StrategyOptimizer.score_opportunity ---
        if strat_name.upper() == "BREAKOUT":
            buy_signals = (closes > bb_uppers) & (rsis > 50) & (closes > ema_200s) & rvol_ok
        else:
            buy_signals = (closes < bb_lowers) & (rsis < buy_rsi) & (closes > ema_200s) & rvol_ok

        cash = 10000.0
        start_cash = cash
        shares = 0
        trades = 0

        entry_price = 0.0
        pos_start_idx = -1
        highest_close = 0.0
        stop_price = 0.0
        take_price = 0.0

        i = 0
        while i < n:
            if shares > 0:
                # Intrabar bracket hits
                hit_stop = (stop_price > 0) and (lows[i] <= stop_price)
                hit_take = (take_price > 0) and (highs[i] >= take_price)

                if hit_stop and hit_take:
                    exit_price = take_price if bar_conflict == "TP_FIRST" else stop_price
                    cash += shares * exit_price
                    shares = 0
                    i += 1
                    continue
                elif hit_stop:
                    cash += shares * stop_price
                    shares = 0
                    i += 1
                    continue
                elif hit_take:
                    cash += shares * take_price
                    shares = 0
                    i += 1
                    continue

                # Trailing stop (engine uses close/highest close)
                if closes[i] > highest_close:
                    highest_close = closes[i]
                trail_stop = highest_close * (1 - trail_pct)
                if closes[i] < trail_stop:
                    cash += shares * closes[i]
                    shares = 0
                    i += 1
                    continue

                hold_bars = i - pos_start_idx

                # Max-hold exit
                if max_hold and hold_bars >= max_hold:
                    cash += shares * closes[i]
                    shares = 0
                    i += 1
                    continue

                # Stagnation exit (engine default: 60m and -1% < gain < +0.3%)
                if use_stagnation and hold_bars >= stag_minutes:
                    pct_gain = (closes[i] - entry_price) / entry_price if entry_price else 0.0
                    if stag_min < pct_gain < stag_max:
                        cash += shares * closes[i]
                        shares = 0
                        i += 1
                        continue

                i += 1
                continue

            # No position: attempt a limit fill in the next TTL bars
            if not buy_signals[i]:
                i += 1
                continue

            limit_price = float(closes[i])
            filled = False
            start_j = i + 1
            end_j = min(n, i + 1 + max(1, entry_ttl))

            for j in range(start_j, end_j):
                if lows[j] <= limit_price:
                    # Fill at limit (worst-case for buyer)
                    entry_price = limit_price
                    sh = int(cash / entry_price)
                    if sh <= 0:
                        i = j
                        filled = True
                        break

                    shares = sh
                    cash -= shares * entry_price
                    trades += 1

                    pos_start_idx = j
                    highest_close = entry_price

                    # Bracket exits match StrategyOptimizer.calculate_exit_prices defaults
                    atr_val = float(atrs[j]) if atrs is not None and j < len(atrs) and np.isfinite(atrs[j]) else float('nan')
                    if np.isfinite(atr_val) and atr_val > 0:
                        stop_price = entry_price - (atr_val * atr_stop_mult)
                        take_price = entry_price + (atr_val * atr_take_mult)
                    else:
                        stop_price = entry_price * (1 - stop_loss_pct)
                        take_price = entry_price * (1 + stop_loss_pct * 1.5)

                    i = j
                    filled = True
                    break

            if not filled:
                i += 1

        if shares > 0:
            cash += shares * closes[-1]

        return cash - start_cash, trades
