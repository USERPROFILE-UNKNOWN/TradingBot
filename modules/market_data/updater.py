"""Incremental DB updater (v5.4.0).

This module centralizes the "UPDATE DB" missing-range optimization logic.

Design goals:
- Keep *all* indicator computation + missing-range fetching logic in one place.
- Make BackfillEngine a thin orchestration/wiring layer (API connection + threading).
- Preserve v5.3.0 behavior: fetch only missing ranges, include warmup for indicators,
  use IEX feed by default for equities.

NOTE: This module intentionally does NOT create an Alpaca API instance.
      The caller must provide a connected api client.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas_ta as ta


class IncrementalUpdater:
    """Compute missing intervals and upsert 1-minute bars + indicators into SQLite."""

    def __init__(self, config: dict, db_manager: Any):
        self.config = config
        self.db = db_manager

    # -----------------------------
    # Public API
    # -----------------------------
    def update_symbol(self, api: Any, symbol: str, days: int = 60) -> Dict[str, Any]:
        """Fetch and store only missing 1-minute bars needed to cover the requested lookback window.

        Returns:
            dict: {symbol, fetched, inserted, up_to_date, error?}
        """
        try:
            end_date = datetime.now(timezone.utc).replace(second=0, microsecond=0)
            window_start = end_date - timedelta(days=days)

            warmup_minutes = self._get_warmup_minutes()
            feed = self._get_equity_feed()

            # Pull existing coverage
            try:
                last_ts = self.db.get_last_timestamp(symbol)
            except Exception:
                last_ts = None
            try:
                first_ts = self.db.get_first_timestamp(symbol)
            except Exception:
                first_ts = None

            intervals = self._compute_missing_intervals(
                window_start=window_start,
                window_end=end_date,
                first_ts=first_ts,
                last_ts=last_ts,
            )

            if not intervals:
                return {'symbol': symbol, 'fetched': 0, 'inserted': 0, 'up_to_date': True}

            total_fetched = 0
            total_inserted = 0

            for interval_start, interval_end in intervals:
                fetch_start = interval_start - timedelta(minutes=warmup_minutes)

                bars = self._fetch_bars(
                    api=api,
                    symbol=symbol,
                    start_dt=fetch_start,
                    end_dt=interval_end,
                    equity_feed=feed,
                )

                if bars is None or bars.empty:
                    continue

                # De-dupe on timestamp index
                bars = bars[~bars.index.duplicated(keep='first')]
                total_fetched += int(len(bars))

                # Need sufficient depth to compute EMA200 and other indicators reliably.
                if len(bars) < 210:
                    continue

                bars = self._add_indicators(bars)

                # Drop warmup NaNs, then keep only rows in the missing interval
                bars.dropna(inplace=True)
                try:
                    bars = bars[bars.index >= interval_start]
                except Exception:
                    pass

                if bars.empty:
                    continue

                bulk_data = self._to_bulk_rows(symbol, bars)

                if bulk_data:
                    inserted = int(self.db.save_bulk_data(symbol, bulk_data) or 0)
                    total_inserted += inserted

            return {
                'symbol': symbol,
                'fetched': int(total_fetched),
                'inserted': int(total_inserted),
                'up_to_date': (total_inserted == 0),
            }

        except Exception as e:
            return {'symbol': symbol, 'fetched': 0, 'inserted': 0, 'up_to_date': False, 'error': str(e)}

    # -----------------------------
    # Internals
    # -----------------------------
    def _get_warmup_minutes(self) -> int:
        try:
            warmup_minutes = int(self.config['CONFIGURATION'].get('indicator_warmup_minutes', '500'))
        except Exception:
            warmup_minutes = 500

        # Keep consistent with v5.3.0 behavior
        if warmup_minutes < 210:
            warmup_minutes = 210
        if warmup_minutes > 5000:
            warmup_minutes = 5000
        return warmup_minutes

    def _get_equity_feed(self) -> str:
        try:
            feed = str(self.config['CONFIGURATION'].get('equity_data_feed', 'iex')).strip().lower()
        except Exception:
            feed = 'iex'
        if feed not in ('iex', 'sip'):
            feed = 'iex'
        return feed

    def _compute_missing_intervals(
        self,
        window_start: datetime,
        window_end: datetime,
        first_ts: Optional[datetime],
        last_ts: Optional[datetime],
    ) -> List[Tuple[datetime, datetime]]:
        """Return list of intervals (inclusive-ish) that need fetching."""
        intervals: List[Tuple[datetime, datetime]] = []

        # If we have no history or timestamps are unusable, fetch the full window.
        if first_ts is None or last_ts is None:
            intervals.append((window_start, window_end))
            return intervals

        # Normalize tz awareness
        try:
            if getattr(last_ts, 'tzinfo', None) is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
        except Exception:
            pass
        try:
            if getattr(first_ts, 'tzinfo', None) is None:
                first_ts = first_ts.replace(tzinfo=timezone.utc)
        except Exception:
            pass

        # If DB doesn't overlap the requested window, fetch the full window.
        try:
            if last_ts < window_start or first_ts > window_end:
                intervals.append((window_start, window_end))
                return intervals

            # Missing earlier portion
            if first_ts > (window_start + timedelta(minutes=1)):
                early_end = first_ts - timedelta(minutes=1)
                if early_end > window_start:
                    intervals.append((window_start, early_end))

            # Missing newest portion
            if last_ts < (window_end - timedelta(minutes=1)):
                late_start = last_ts + timedelta(minutes=1)
                if late_start < window_start:
                    late_start = window_start
                if late_start < window_end:
                    intervals.append((late_start, window_end))

        except Exception:
            intervals.append((window_start, window_end))

        return intervals

    def _to_iso(self, dt: datetime) -> str:
        return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')

    def _fetch_bars(self, api: Any, symbol: str, start_dt: datetime, end_dt: datetime, equity_feed: str):
        start_iso = self._to_iso(start_dt)
        end_iso = self._to_iso(end_dt)

        if '/' in symbol:
            return api.get_crypto_bars(symbol, '1Min', start=start_iso, end=end_iso).df

        # alpaca_trade_api versions differ on whether get_bars accepts feed=
        try:
            return api.get_bars(symbol, '1Min', start=start_iso, end=end_iso, adjustment='raw', feed=equity_feed).df
        except TypeError:
            return api.get_bars(symbol, '1Min', start=start_iso, end=end_iso, adjustment='raw').df

    def _add_indicators(self, bars):
        bb = ta.bbands(bars['close'], length=20, std=2.0)
        rsi = ta.rsi(bars['close'], length=14)
        ema = ta.ema(bars['close'], length=200)
        adx = ta.adx(bars['high'], bars['low'], bars['close'], length=14)
        atr = ta.atr(bars['high'], bars['low'], bars['close'], length=14)

        lower_col = [c for c in bb.columns if c.startswith('BBL')][0]
        upper_col = [c for c in bb.columns if c.startswith('BBU')][0]
        adx_col = [c for c in adx.columns if c.startswith('ADX')][0]

        bars['bb_lower'] = bb[lower_col]
        bars['bb_upper'] = bb[upper_col]
        bars['rsi'] = rsi
        bars['ema_200'] = ema
        bars['adx'] = adx[adx_col]
        bars['atr'] = atr
        return bars

    def _to_bulk_rows(self, symbol: str, bars) -> List[tuple]:
        bulk_data: List[tuple] = []
        sym = symbol.upper()
        for index, row in bars.iterrows():
            ts = index.to_pydatetime()
            bulk_data.append(
                (
                    sym,
                    ts,
                    row['close'],
                    row['open'],
                    row['high'],
                    row['low'],
                    row['volume'],
                    row['rsi'],
                    row['bb_lower'],
                    row['bb_upper'],
                    row['ema_200'],
                    row['adx'],
                    row['atr'],
                )
            )
        return bulk_data
