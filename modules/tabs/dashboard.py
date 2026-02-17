"""UI tab: Dashboard.

Moved into modules/tabs/dashboard.py in v5.8.0 to reduce regression risk.
"""

import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DashboardTab:
    def __init__(self, parent, engine, db, config, metrics_store=None):
        self.parent = parent
        self.engine = engine
        self.db = db
        self.config = config
        self.metrics_store = metrics_store
        self.setup_ui()

    def setup_ui(self):
        self.chart_top = ctk.CTkFrame(self.parent)
        self.chart_top.pack(side="top", fill="both", expand=True, padx=5, pady=5)
        self.chart_bottom = ctk.CTkFrame(self.parent, height=200)
        self.chart_bottom.pack(side="bottom", fill="x", padx=5, pady=5)
        
        self.figure, self.ax = plt.subplots(figsize=(6, 3), dpi=100)
        self.figure.patch.set_facecolor('#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors='white', labelcolor='white')
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.chart_top)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        ctk.CTkLabel(self.chart_bottom, text="MARKET PULSE (Prices Updated Live)", font=("Arial", 12, "bold")).pack(pady=5)

        self.health_frame = ctk.CTkFrame(self.chart_bottom)
        self.health_frame.pack(fill="x", padx=5, pady=(0, 5))
        self.health_lbl = ctk.CTkLabel(self.health_frame, text="Health: loading...", font=("Arial", 11))
        self.health_lbl.pack(side="left", padx=8, pady=4)

        self.heatmap_frame = ctk.CTkFrame(self.chart_bottom)
        self.heatmap_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def update(self):
        self.update_equity_chart()
        self.update_health_widget()
        self.update_heatmap()

    def update_equity_chart(self):
        history = self.db.get_recent_history(limit=50)
        history = history[::-1]
        balance = [0]
        dates = ["Start"]
        cumulative = 0
        for row in history:
            cumulative += float(row[4] or 0.0) if len(row) > 4 else 0.0
            balance.append(cumulative)
            dates.append(len(dates))
            
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors='white', labelcolor='white')
        
        color = '#00C853' if cumulative >= 0 else '#FF1744'
        self.ax.plot(dates, balance, color=color, linewidth=2)
        self.ax.fill_between(dates, balance, 0, color=color, alpha=0.1)
        self.ax.set_title(f"Realized P/L Curve (Total: ${cumulative:.2f})", color='white')
        self.ax.grid(True, color='#404040', alpha=0.3)
        self.canvas.draw()

    def update_health_widget(self):
        try:
            store = self.metrics_store
            if store is None:
                self.health_lbl.configure(text="Health: metrics unavailable")
                return
            snap = store.get_health_widget_snapshot(lookback_sec=3600)
            freshness = snap.get("data_freshness_seconds")
            freshness_txt = f"{float(freshness):.0f}s" if freshness is not None else "n/a"
            api_err = int(snap.get("api_error_streak", 0) or 0)
            rej = float(snap.get("order_reject_ratio", 0.0) or 0.0) * 100.0
            lat = snap.get("decision_exec_latency_ms")
            lat_txt = f"{float(lat):.0f}ms" if lat is not None else "n/a"
            sl = snap.get("slippage_bps")
            sl_txt = f"{float(sl):.1f}bps" if sl is not None else "n/a"
            self.health_lbl.configure(
                text=(
                    f"Health | Freshness: {freshness_txt} | API error streak: {api_err} | "
                    f"Reject ratio: {rej:.1f}% | Latency: {lat_txt} | Slippage: {sl_txt}"
                )
            )
        except Exception:
            try:
                self.health_lbl.configure(text="Health: unavailable")
            except Exception:
                pass

    def update_heatmap(self):
        for widget in self.heatmap_frame.winfo_children(): widget.destroy()
        
        try:
            from ..watchlist_api import get_watchlist_symbols
            symbols = list(get_watchlist_symbols(self.config, group="ACTIVE", asset="ALL"))
        except Exception:
            symbols = []
        active_trades = self.db.get_active_trades() or {}
        cols = 5
        for i, sym in enumerate(symbols):
            try:
                color = "#404040" 
                status = "WAITING"
                price = self.db.get_latest_snapshot(sym)
                price_text = f"${price:.2f}" if price is not None else "N/A"
                
                if sym in active_trades:
                    color = "#00C853" 
                    status = "HOLDING"
                elif sym in self.engine.pending_confirmations:
                    color = "#FFA000"
                    status = "PENDING"

                card = ctk.CTkFrame(self.heatmap_frame, fg_color=color)
                card.grid(row=i//cols, column=i%cols, padx=2, pady=2, sticky="nsew")
                ctk.CTkLabel(card, text=sym, font=("Arial", 12, "bold")).pack(pady=2)
                ctk.CTkLabel(card, text=price_text, font=("Arial", 11)).pack(pady=0)
                ctk.CTkLabel(card, text=status, font=("Arial", 9)).pack(pady=0)
            except Exception as e:
                # v5.12.3 updateA: surface UI rendering issues without spamming.
                try:
                    from ..log_throttle import log_exception_throttled
                    log_fn = getattr(self.engine, 'log', None)
                    if not callable(log_fn):
                        # fall back to engine._emit if available
                        emit = getattr(self.engine, '_emit', None)
                        if callable(emit):
                            emit(f"⚠️ [E_DASH_HEATMAP_FAIL] {type(e).__name__} | symbol={sym}", level="WARN", category="UI", throttle_key=f"dash_heatmap_{sym}", throttle_sec=300)
                            continue
                        log_fn = None
                    log_exception_throttled(log_fn, "E_DASH_HEATMAP_FAIL", e, key=f"dash_heatmap_{sym}", throttle_sec=300, context={"symbol": sym})
                except Exception:
                    pass
