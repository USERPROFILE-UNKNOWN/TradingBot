import pandas as pd
import numpy as np
import random
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed


class TheArchitect:
    """
    Genetic optimizer for simple mean-reversion entries.

    v4.0.3 upgrades:
      - OHLC-based SL/TP triggering (uses intrabar high/low)
      - Entry fill on next-bar open (more realistic than same-bar close)
      - Optional slippage + per-trade fee modeling
      - Max-hold time exit (prevents long scans / hanging)
      - Optional walk-forward scoring (reduces overfitting to one window)
    """

    def __init__(self, db_manager, config=None):
        self.db = db_manager
        self.config = config

        # DNA Definition: [RSI_BUY, STOP_LOSS, TP_MULTI, EMA_FILTER]
        self.dna_bounds = {
            'rsi_buy': (20, 55),
            'stop_loss': (0.01, 0.10),
            'tp_multi': (1.5, 6.0),
            'ema': [True, False]
        }

        # Backtest / architect knobs (safe defaults)
        self.slippage_bps = self._cfg_float('backtest_slippage_bps', 5.0)          # basis points
        self.fee_per_trade = self._cfg_float('backtest_fee_per_trade', 0.0)        # dollars per round-trip
        self.max_hold_bars = self._cfg_int('architect_max_hold_bars', 200)
        self.min_trades = self._cfg_int('architect_min_trades', 5)

        self.use_walkforward = self._cfg_bool('architect_use_walkforward', False)
        self.walkforward_splits = max(2, self._cfg_int('architect_walkforward_splits', 3))

        self.bar_conflict = (self._cfg_str('architect_bar_conflict', 'STOP_FIRST') or 'STOP_FIRST').upper()
        if self.bar_conflict not in ('STOP_FIRST', 'TP_FIRST'):
            self.bar_conflict = 'STOP_FIRST'

    # -------------------------
    # Config helpers
    # -------------------------
    def _cfg_section(self):
        if self.config is None:
            return None
        return self.config['CONFIGURATION'] if 'CONFIGURATION' in self.config else None

    def _cfg_str(self, key, default):
        sec = self._cfg_section()
        try:
            return str(sec.get(key, default)) if sec is not None else default
        except Exception:
            return default

    def _cfg_float(self, key, default):
        sec = self._cfg_section()
        try:
            return float(sec.get(key, default)) if sec is not None else float(default)
        except Exception:
            return float(default)

    def _cfg_int(self, key, default):
        sec = self._cfg_section()
        try:
            return int(float(sec.get(key, default))) if sec is not None else int(default)
        except Exception:
            return int(default)

    def _cfg_bool(self, key, default):
        sec = self._cfg_section()
        try:
            if sec is None:
                return bool(default)
            return str(sec.get(key, str(default))).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        except Exception:
            return bool(default)

    def _cfg(self, key, default=None):
        """Raw config getter used by simulation-time knobs.

        Release B calls self._cfg(...) throughout _simulate(). In v4.0.4 this
        helper was missing, causing an AttributeError that got swallowed by the
        fitness_function() try/except and resulted in -9999.0 profits for every
        genome.
        """
        sec = self._cfg_section()
        try:
            if sec is None:
                return default
            return sec.get(key, default)
        except Exception:
            return default

    # -------------------------
    # Genetic operators
    # -------------------------
    def generate_random_genome(self):
        return {
            'rsi': random.randint(self.dna_bounds['rsi_buy'][0], self.dna_bounds['rsi_buy'][1]),
            'sl': round(random.uniform(self.dna_bounds['stop_loss'][0], self.dna_bounds['stop_loss'][1]), 3),
            'tp': round(random.uniform(self.dna_bounds['tp_multi'][0], self.dna_bounds['tp_multi'][1]), 1),
            'ema': random.choice(self.dna_bounds['ema'])
        }

    def mutate(self, genome):
        g = copy.deepcopy(genome)
        trait = random.choice(['rsi', 'sl', 'tp', 'ema'])

        if trait == 'rsi':
            g['rsi'] = random.randint(self.dna_bounds['rsi_buy'][0], self.dna_bounds['rsi_buy'][1])
        elif trait == 'sl':
            g['sl'] = round(random.uniform(self.dna_bounds['stop_loss'][0], self.dna_bounds['stop_loss'][1]), 3)
        elif trait == 'tp':
            g['tp'] = round(random.uniform(self.dna_bounds['tp_multi'][0], self.dna_bounds['tp_multi'][1]), 1)
        elif trait == 'ema':
            g['ema'] = not g['ema']
        return g

    def crossover(self, p1, p2):
        child = {}
        child['rsi'] = p1['rsi'] if random.random() > 0.5 else p2['rsi']
        child['sl'] = p1['sl'] if random.random() > 0.5 else p2['sl']
        child['tp'] = p1['tp'] if random.random() > 0.5 else p2['tp']
        child['ema'] = p1['ema'] if random.random() > 0.5 else p2['ema']
        return child

    # -------------------------
    # Simulation core
    # -------------------------
    def _simulate(self, df: pd.DataFrame, genome: dict):
        """
        Returns dict: profit, trades, win_rate (0-100), score.

        Release B alignment notes:
        - Optional engine-style limit-entry fill model (signal on bar i close, fill if a future bar trades down to the limit within TTL).
        - Optional trailing stop and stagnation exits to reduce optimizer/live divergence.
        - Optional ATR-based bracket exits (off by default to preserve genome SL/TP meaning).
        """
        required_cols = {'open', 'high', 'low', 'close', 'rsi', 'bb_lower'}
        if not required_cols.issubset(set(df.columns)):
            return {'profit': -9999.0, 'trades': 0, 'win_rate': 0.0, 'score': -9999.0}

        rsi_thresh = genome['rsi']
        sl_pct = float(genome['sl'])
        tp_mult = float(genome['tp'])
        use_ema = bool(genome['ema'])

        # Build signal mask (vectorized), but execute trade loop sequentially to avoid overlap
        buy_signals = (df['rsi'] < rsi_thresh) & (df['close'] < df['bb_lower'])
        buy_signals = buy_signals & df['rsi'].notna() & df['bb_lower'].notna()

        if use_ema and 'ema_200' in df.columns:
            buy_signals = buy_signals & df['ema_200'].notna() & (df['close'] > df['ema_200'])

        start_balance = 1000.0
        balance = start_balance
        trades = 0
        wins = 0

        slip = max(0.0, float(self.slippage_bps)) / 10000.0

        # Release B tunables
        entry_mode = str(self._cfg('architect_entry_mode', 'LIMIT_CLOSE_TTL')).upper()  # NEXT_OPEN or LIMIT_CLOSE_TTL
        entry_ttl = int(float(self._cfg('architect_entry_ttl_bars', 2)))

        use_trailing = str(self._cfg('architect_use_trailing_stop', 'True')).lower() == 'true'
        trailing_uses_sl = str(self._cfg('architect_trailing_uses_sl', 'True')).lower() == 'true'
        trail_pct = sl_pct if trailing_uses_sl else float(self._cfg('architect_trailing_stop_pct', 0.02))

        use_stagnation = str(self._cfg('architect_use_stagnation_exit', 'True')).lower() == 'true'
        stagnation_bars = int(float(self._cfg('architect_stagnation_bars', 60)))
        stagnation_min = float(self._cfg('architect_stagnation_min_gain', -0.01))
        stagnation_max = float(self._cfg('architect_stagnation_max_gain', 0.003))

        use_atr_exits = str(self._cfg('architect_use_atr_exits', 'False')).lower() == 'true'
        atr_stop_mult = float(self._cfg('architect_atr_stop_mult', 2.0))
        atr_take_mult = float(self._cfg('architect_atr_take_mult', 3.0))

        # arrays for speed
        o = df['open'].to_numpy()
        h = df['high'].to_numpy()
        l = df['low'].to_numpy()
        c = df['close'].to_numpy()
        atr = df['atr'].to_numpy() if 'atr' in df.columns else None
        sig = buy_signals.to_numpy()

        n = len(df)
        if n < 5:
            return {'profit': -999.0, 'trades': 0, 'win_rate': 0.0, 'score': -999.0}

        i = 0
        while i < n - 2:
            if not sig[i]:
                i += 1
                continue

            # --- Entry ---
            entry_idx = None
            entry_fill = None

            if entry_mode == 'LIMIT_CLOSE_TTL':
                limit_price = float(c[i])
                if not np.isfinite(limit_price) or limit_price <= 0:
                    i += 1
                    continue

                # Fill if a future bar trades down to the limit within TTL bars
                j_end_fill = min(n - 1, i + 1 + max(1, entry_ttl))
                for j in range(i + 1, j_end_fill):
                    if np.isfinite(l[j]) and l[j] <= limit_price:
                        entry_idx = j
                        entry_fill = limit_price * (1.0 + slip)  # conservative (worst-case for buyer)
                        break

                if entry_idx is None:
                    i += 1
                    continue

            else:
                # NEXT_OPEN (original behavior)
                entry_idx = i + 1
                entry_open = o[entry_idx]
                if not np.isfinite(entry_open) or entry_open <= 0:
                    i += 1
                    continue
                entry_fill = float(entry_open) * (1.0 + slip)

            # Bracket exits (either genome SL/TP or ATR-based)
            if use_atr_exits and atr is not None and entry_idx < len(atr) and np.isfinite(atr[entry_idx]) and atr[entry_idx] > 0:
                stop_price = entry_fill - float(atr[entry_idx]) * atr_stop_mult
                take_price = entry_fill + float(atr[entry_idx]) * atr_take_mult
            else:
                stop_price = entry_fill * (1.0 - sl_pct)
                take_price = entry_fill * (1.0 + (sl_pct * tp_mult))

            # --- Manage position until exit ---
            exit_idx = None
            exit_fill = None
            win = False

            highest_close = entry_fill

            j_end = min(n - 1, entry_idx + max(1, int(self.max_hold_bars)))
            for j in range(entry_idx, j_end):
                low = l[j]
                high = h[j]
                close = c[j]

                if not (np.isfinite(low) and np.isfinite(high) and np.isfinite(close)):
                    continue

                # Bracket intrabar triggers
                hit_stop = low <= stop_price
                hit_take = high >= take_price

                if hit_stop and hit_take:
                    if self.bar_conflict == 'TP_FIRST':
                        exit_fill = take_price * (1.0 - slip)
                        win = True
                    else:
                        exit_fill = stop_price * (1.0 - slip)
                        win = False
                    exit_idx = j
                    break

                if hit_stop:
                    exit_fill = stop_price * (1.0 - slip)
                    win = False
                    exit_idx = j
                    break

                if hit_take:
                    exit_fill = take_price * (1.0 - slip)
                    win = True
                    exit_idx = j
                    break

                # Trailing stop (engine checks on close)
                if use_trailing:
                    if close > highest_close:
                        highest_close = close
                    trail_stop = highest_close * (1.0 - trail_pct)
                    if close < trail_stop:
                        exit_fill = close * (1.0 - slip)
                        win = exit_fill >= entry_fill
                        exit_idx = j
                        break

                # Stagnation exit (engine-style)
                hold_bars = j - entry_idx
                if use_stagnation and hold_bars >= stagnation_bars:
                    pct_gain = (close - entry_fill) / entry_fill
                    if stagnation_min < pct_gain < stagnation_max:
                        exit_fill = close * (1.0 - slip)
                        win = exit_fill >= entry_fill
                        exit_idx = j
                        break

            # Time exit at close of window if not hit
            if exit_idx is None:
                exit_idx = j_end - 1
                exit_close = float(df['close'].iloc[exit_idx])
                if np.isfinite(exit_close) and exit_close > 0:
                    exit_fill = exit_close * (1.0 - slip)
                else:
                    exit_fill = entry_fill
                win = exit_fill >= entry_fill

            # Apply trade PnL to balance
            ret = (exit_fill - entry_fill) / entry_fill
            balance = balance * (1.0 + ret) - float(self.fee_per_trade)

            trades += 1
            if win:
                wins += 1

            # Safety brakes
            if balance < 100:
                break

            # Prevent overlapping trades: resume scanning after exit
            i = exit_idx + 1

        net_profit = balance - start_balance
        win_rate = (wins / trades) * 100.0 if trades > 0 else 0.0

        # Score: prefer strategies that make money and do so reliably
        score = net_profit * (1.0 + (win_rate / 100.0))
        if trades < self.min_trades:
            score = -500.0

        return {
            'profit': float(net_profit),
            'trades': int(trades),
            'win_rate': float(win_rate),
            'score': float(score)
        }

    def fitness_function(self, df, genome):
        try:
            if df is None or df.empty:
                return {'genome': genome, 'profit': -999.0, 'trades': 0, 'win_rate': 0.0, 'score': -999.0}

            # Optional walk-forward: score by worst segment
            if self.use_walkforward and len(df) > 500:
                splits = self.walkforward_splits
                seg_size = max(200, len(df) // splits)
                segment_scores = []
                segment_profits = []
                segment_trades = 0
                segment_wins = 0
                segment_total_trades = 0

                for k in range(splits):
                    a = k * seg_size
                    b = (k + 1) * seg_size if k < splits - 1 else len(df)
                    seg = df.iloc[a:b]
                    res = self._simulate(seg, genome)
                    segment_scores.append(res['score'])
                    segment_profits.append(res['profit'])
                    segment_total_trades += res['trades']
                    segment_trades += res['trades']
                    segment_wins += int(round(res['trades'] * (res['win_rate'] / 100.0)))

                worst_score = min(segment_scores) if segment_scores else -9999.0
                worst_profit = min(segment_profits) if segment_profits else -9999.0
                wr = (segment_wins / segment_total_trades) * 100.0 if segment_total_trades > 0 else 0.0

                # Penalize if overall activity too low
                if segment_total_trades < self.min_trades:
                    worst_score = -500.0

                return {
                    'genome': genome,
                    'profit': float(worst_profit),           # conservative reporting
                    'trades': int(segment_total_trades),
                    'win_rate': float(wr),
                    'score': float(worst_score)
                }

            # Standard single-window evaluation
            res = self._simulate(df, genome)
            res['genome'] = genome
            return res

        except Exception:
            # Never allow worker thread to die silently
            return {'genome': genome, 'profit': -9999.0, 'trades': 0, 'win_rate': 0.0, 'score': -9999.0}

    # -------------------------
    # Optimization loop
    # -------------------------
    def run_optimization(self, symbol, progress_callback):
        try:
            df = self.db.get_history(symbol, limit=3000)  # Reduced limit for speed
            if df is None or df.empty:
                progress_callback(f"Error: No data for {symbol}", 0)
                return []

            POP_SIZE = 40
            GENERATIONS = 15

            population = [self.generate_random_genome() for _ in range(POP_SIZE)]
            best_of_all_time = None
            results = []

            for gen in range(GENERATIONS):
                results = []

                # Reduced workers to prevent CPU locking on consumer PCs
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {executor.submit(self.fitness_function, df, dna): dna for dna in population}

                    for future in as_completed(futures):
                        try:
                            res = future.result()
                            results.append(res)
                        except Exception:
                            continue

                if not results:
                    continue

                results.sort(key=lambda x: x['score'], reverse=True)
                best_gen = results[0]

                if best_of_all_time is None or best_gen['score'] > best_of_all_time['score']:
                    best_of_all_time = best_gen

                msg = f"Gen {gen+1}/{GENERATIONS}: Best Profit=${best_gen['profit']:.2f} (WR: {best_gen['win_rate']:.1f}%)"
                progress_callback(msg, (gen + 1) / GENERATIONS)

                survivors = [r['genome'] for r in results[:10]]

                next_gen = survivors[:]
                while len(next_gen) < POP_SIZE:
                    p1 = random.choice(survivors)
                    p2 = random.choice(survivors)
                    child = self.crossover(p1, p2)
                    if random.random() < 0.3:
                        child = self.mutate(child)
                    next_gen.append(child)

                population = next_gen

            final_results = []
            seen = set()
            for r in (results or []):
                h = str(r['genome'])
                if h in seen:
                    continue
                final = dict(r['genome'])
                final['profit'] = r['profit']
                final['trades'] = r['trades']
                final['win_rate'] = r['win_rate']
                final_results.append(final)
                seen.add(h)

            return final_results[:15]

        except Exception as e:
            progress_callback(f"Fatal Error: {e}", 0)
            return []
