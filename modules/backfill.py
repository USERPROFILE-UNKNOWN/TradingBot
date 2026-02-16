import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from .market_data.updater import IncrementalUpdater

class BackfillEngine:
    def __init__(self, config, db, progress_callback):
        self.config = config
        self.db = db
        self.progress = progress_callback 
        self.api = None
        self.updater = IncrementalUpdater(self.config, self.db)
        self.connect_api()

    def connect_api(self):
        try:
            self.api = tradeapi.REST(
                key_id=self.config['KEYS']['alpaca_key'].strip(),
                secret_key=self.config['KEYS']['alpaca_secret'].strip(),
                base_url=self.config['KEYS'].get('base_url', 'https://paper-api.alpaca.markets').strip(),
                api_version='v2'
            )
        except Exception as e:
            # Surface the failure to the UI instead of silently swallowing it.
            self.api = None
            try:
                self.progress(f"❌ [E_BACKFILL_API_CONNECT] {type(e).__name__}: {e}")
            except Exception:
                pass

    def run(self):
        if not self.api:
            self.progress("❌ API Connection Failed.")
            return

        # Phase 4 (v5.14.0): ACTIVE watchlist universe
        try:
            from .watchlist_api import get_watchlist_symbols
            symbols = list(get_watchlist_symbols(self.config, group="ACTIVE", asset="ALL"))
        except Exception:
            symbols = []

        # Configurable lookback for Update DB (days of 1Min candles)
        try:
            lookback_days = int(self.config['CONFIGURATION'].get('update_db_lookback_days', '60'))
        except Exception:
            lookback_days = 60
        if lookback_days < 1:
            lookback_days = 1
        if lookback_days > 365:
            lookback_days = 365

        
        total = len(symbols)
        completed = 0

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self.process_symbol, symbol, lookback_days): symbol for symbol in symbols}
            
            for future in as_completed(futures):
                symbol = futures[future]
                completed += 1
                res = None
                try:
                    res = future.result()
                except Exception as e:
                    res = {'symbol': symbol, 'fetched': 0, 'inserted': 0, 'up_to_date': False, 'error': str(e)}
                fetched = int(res.get('fetched', 0)) if isinstance(res, dict) else 0
                inserted = int(res.get('inserted', 0)) if isinstance(res, dict) else 0
                up_to_date = bool(res.get('up_to_date', inserted == 0)) if isinstance(res, dict) else (inserted == 0)
                err = res.get('error') if isinstance(res, dict) else None
                if err:
                    self.progress(f"⚠️ {symbol} | fetched {fetched} | inserted {inserted} | up_to_date {up_to_date} | error {err} ({completed}/{total})")
                else:
                    self.progress(f"⏳ {symbol} | fetched {fetched} | inserted {inserted} | up_to_date {up_to_date} ({completed}/{total})")
        
        self.progress("✅ Update Complete! You may close this window.")

    def process_symbol(self, symbol, days=60):
        """Delegates per-symbol update to market_data.IncrementalUpdater (v5.4.0).

        This preserves the existing UPDATE DB behavior while moving the logic out of this file
        to reduce blast radius and make future changes safer.

        Returns a dict with fetched/inserted counts for progress display.
        """
        return self.updater.update_symbol(self.api, symbol, days=days)


