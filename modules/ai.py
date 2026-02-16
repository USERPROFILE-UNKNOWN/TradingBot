import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


def _safe_clip_proba(p: np.ndarray) -> np.ndarray:
    """Clip probabilities away from 0/1 to keep logit/logloss stable."""
    try:
        return np.clip(p.astype('float64', copy=False), 1e-6, 1.0 - 1e-6)
    except Exception:
        return np.clip(np.asarray(p, dtype='float64'), 1e-6, 1.0 - 1e-6)


def _fit_probability_calibrator(method: str, p_raw: np.ndarray, y_true: np.ndarray):
    """Fit a lightweight probability calibrator without relying on cv='prefit'.

    Newer sklearn versions removed cv='prefit' for CalibratedClassifierCV.
    We therefore fit a tiny calibrator on top of base model probabilities.

    Returns a fitted calibrator object, or None if it cannot be fit.
    """
    try:
        y = np.asarray(y_true, dtype='int64')
        if len(np.unique(y)) < 2:
            return None
    except Exception:
        return None

    p = _safe_clip_proba(np.asarray(p_raw))

    if str(method).lower() == 'isotonic':
        try:
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(p, y)
            return ir
        except Exception:
            return None

    # Default: sigmoid (Platt scaling) on logit(p)
    try:
        x = np.log(p / (1.0 - p)).reshape(-1, 1)
        lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        lr.fit(x, y)
        return lr
    except Exception:
        return None


def _apply_probability_calibrator(calibrator, method: str, p_raw: np.ndarray) -> np.ndarray:
    """Apply a fitted calibrator to raw probs."""
    p = _safe_clip_proba(np.asarray(p_raw))
    if calibrator is None:
        return p
    if str(method).lower() == 'isotonic':
        try:
            return _safe_clip_proba(calibrator.predict(p))
        except Exception:
            return p
    try:
        x = np.log(p / (1.0 - p)).reshape(-1, 1)
        return _safe_clip_proba(calibrator.predict_proba(x)[:, 1])
    except Exception:
        return p


class _CalibratedWrapper:
    """Wrap a base classifier + separate probability calibrator."""
    def __init__(self, base_model, calibrator, method: str):
        self._base = base_model
        self._cal = calibrator
        self._method = str(method).lower()

    def predict_proba(self, X):
        p_raw = self._base.predict_proba(X)[:, 1]
        p = _apply_probability_calibrator(self._cal, self._method, p_raw)
        return np.vstack([1.0 - p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)



class AI_Oracle:
    """AI model that scores trade candidates.

    v4.10.0 (AI v2):
      - Labels: per-symbol (no cross-symbol leakage) with an ATR-based triple-barrier style label.
      - Calibration: probabilistic calibration (sigmoid by default) for meaningful probability outputs.
      - Walk-forward: time-ordered evaluation to avoid random split leakage.

    NOTE: This class is intentionally self-contained to avoid strategy changes.
    """

    def __init__(self, db_manager, config=None, log_func=None):
        # Backwards-compatible signature:
        #   AI_Oracle(db, log_func)
        #   AI_Oracle(db, config, log_func)
        if callable(config) and log_func is None:
            log_func = config
            config = None

        self.db = db_manager
        self.config = config or {}
        self.log = log_func

        self.model = None              # CalibratedClassifierCV
        self.base_model = None         # RandomForestClassifier (for debugging)
        self.is_trained = False
        self.last_train_stats = {}

        # Features that can be computed from a single snapshot (and are present in DB columns)
        self.feature_cols = [
            'rsi',
            'adx',
            'volume_ratio',
            'dist_ema',
            'bb_pos',
            'atr_pct',
        ]

    # -----------------------------
    # Helpers
    # -----------------------------
    def _emit(self, msg: str):
        try:
            if callable(self.log):
                self.log(msg)
        except Exception:
            pass

    def _cfg(self, key: str, default):
        try:
            # config may be a dict-like from split-config loader
            if isinstance(self.config, dict) and 'CONFIGURATION' in self.config:
                raw = self.config['CONFIGURATION'].get(key, None)
            elif isinstance(self.config, dict):
                raw = self.config.get(key, None)
            else:
                raw = None
        except Exception:
            raw = None

        if raw is None:
            return default

        # best-effort casting based on default type
        try:
            if isinstance(default, bool):
                if isinstance(raw, bool):
                    return raw
                s = str(raw).strip().lower()
                return s in ('1', 'true', 'yes', 'y', 'on')
            if isinstance(default, int):
                return int(float(str(raw).strip()))
            if isinstance(default, float):
                return float(str(raw).strip())
            if isinstance(default, str):
                return str(raw).strip()
        except Exception:
            return default

        return raw

    def _safe_bb_pos(self, close, bb_lower, bb_upper):
        try:
            rng = (bb_upper - bb_lower)
            if rng is None or np.isnan(rng) or rng == 0:
                return 0.5
            return float((close - bb_lower) / rng)
        except Exception:
            return 0.5

    def _build_features_and_labels(self, df: pd.DataFrame, symbol: str):
        """Build X/y for a single symbol history (ascending timestamp)."""
        if df is None or df.empty:
            return pd.DataFrame(), pd.Series(dtype='int64')

        # Ensure expected numeric columns exist
        for c in ['close', 'high', 'low', 'volume', 'rsi', 'adx', 'atr', 'bb_lower', 'bb_upper', 'ema_200']:
            if c not in df.columns:
                df[c] = np.nan

        d = df.copy()

        # Feature engineering (kept simple and robust)
        try:
            vol_mean = d['volume'].rolling(20, min_periods=5).mean()
            d['volume_ratio'] = d['volume'] / vol_mean
        except Exception:
            d['volume_ratio'] = np.nan

        try:
            d['dist_ema'] = (d['close'] - d['ema_200']) / d['ema_200']
        except Exception:
            d['dist_ema'] = np.nan

        # BB position
        try:
            rng = (d['bb_upper'] - d['bb_lower']).replace(0, np.nan)
            d['bb_pos'] = (d['close'] - d['bb_lower']) / rng
        except Exception:
            d['bb_pos'] = np.nan

        # ATR percentage of price
        try:
            d['atr_pct'] = d['atr'] / d['close']
        except Exception:
            d['atr_pct'] = np.nan

        # -----------------------------
        # Labeling (AI v2)
        # -----------------------------
        horizon = int(self._cfg('ai_label_horizon_bars', 60))
        horizon = max(5, min(720, horizon))

        tp_atr = float(self._cfg('ai_label_tp_atr_mult', 1.5))
        sl_atr = float(self._cfg('ai_label_sl_atr_mult', 1.0))
        tp_atr = max(0.1, min(10.0, tp_atr))
        sl_atr = max(0.1, min(10.0, sl_atr))

        min_ret = float(self._cfg('ai_label_min_return_pct', 0.001))
        min_ret = max(0.0, min(0.05, min_ret))

        # If we don't have high/low/atr, fall back to simple forward-return label.
        have_barriers = (
            d['high'].notna().any() and
            d['low'].notna().any() and
            d['atr'].notna().any()
        )

        close = d['close'].to_numpy(dtype='float64', copy=False)
        high = d['high'].to_numpy(dtype='float64', copy=False)
        low = d['low'].to_numpy(dtype='float64', copy=False)
        atr = d['atr'].to_numpy(dtype='float64', copy=False)

        y = np.full(len(d), np.nan, dtype='float64')

        n = len(d)
        last_i = n - horizon - 1
        if last_i < 10:
            return pd.DataFrame(), pd.Series(dtype='int64')

        if have_barriers:
            for i in range(0, last_i):
                c0 = close[i]
                a0 = atr[i]
                if not np.isfinite(c0) or c0 <= 0 or not np.isfinite(a0) or a0 <= 0:
                    continue

                up = c0 + tp_atr * a0
                dn = c0 - sl_atr * a0

                hit = np.nan
                # first-hit within horizon
                for j in range(1, horizon + 1):
                    hh = high[i + j]
                    ll = low[i + j]
                    if np.isfinite(hh) and hh >= up:
                        hit = 1.0
                        break
                    if np.isfinite(ll) and ll <= dn:
                        hit = 0.0
                        break

                if np.isnan(hit):
                    # fallback: end-of-horizon direction with minimum return filter
                    fut = close[i + horizon]
                    if np.isfinite(fut) and fut >= c0 * (1.0 + min_ret):
                        hit = 1.0
                    else:
                        hit = 0.0

                y[i] = hit
        else:
            # Directional label by forward return
            for i in range(0, last_i):
                c0 = close[i]
                fut = close[i + horizon]
                if not np.isfinite(c0) or c0 <= 0 or not np.isfinite(fut):
                    continue
                y[i] = 1.0 if fut >= c0 * (1.0 + min_ret) else 0.0

        d['target'] = y

        # Clean up
        X = d[self.feature_cols].replace([np.inf, -np.inf], np.nan)
        y_ser = d['target']

        mask = X.notna().all(axis=1) & y_ser.notna()
        X = X.loc[mask].astype('float64')
        y_ser = y_ser.loc[mask].astype('int64')

        # Hard clamp for bb_pos and volume_ratio to reduce outlier harm
        try:
            if 'bb_pos' in X.columns:
                X['bb_pos'] = X['bb_pos'].clip(-1.0, 2.0)
            if 'volume_ratio' in X.columns:
                X['volume_ratio'] = X['volume_ratio'].clip(0.0, 50.0)
            if 'atr_pct' in X.columns:
                X['atr_pct'] = X['atr_pct'].clip(0.0, 0.2)
        except Exception:
            pass

        return X, y_ser

    # -----------------------------
    # Train / Predict
    # -----------------------------
    def train_model(self, symbols=None, force=False):
        try:
            if (not force) and self.is_trained:
                return
        except Exception:
            pass

        # Defaults chosen to be safe for performance
        limit_per_symbol = int(self._cfg('ai_train_limit_per_symbol', 20000))
        limit_per_symbol = max(500, min(200000, limit_per_symbol))

        min_rows = int(self._cfg('ai_min_train_rows', 2000))
        min_rows = max(200, min(500000, min_rows))

        wf_splits = int(self._cfg('ai_walkforward_splits', 3))
        wf_splits = max(2, min(10, wf_splits))

        cal_method = str(self._cfg('ai_calibration_method', 'sigmoid')).strip().lower()
        if cal_method not in ('sigmoid', 'isotonic'):
            cal_method = 'sigmoid'

        cal_frac = float(self._cfg('ai_calibration_frac', 0.2))
        cal_frac = float(self._cfg('ai_calibration_fraction', cal_frac))
        cal_frac = max(0.1, min(0.4, cal_frac))

        # Determine symbols
        if not symbols:
            syms = []
            try:
                from .watchlist_api import get_watchlist_symbols
                syms = get_watchlist_symbols(self.config, group="ACTIVE", asset="ALL")
            except Exception:
                syms = []

            symbols = sorted(set([str(s).strip().upper() for s in syms if str(s).strip()]))

        if not symbols:
            self._emit("\U0001f9e0 [AI] No symbols provided for training.")
            self.is_trained = False
            return

        X_all = []
        y_all = []
        rows_by_symbol = {}
        barrier_seen = False

        for sym in symbols:
            try:
                df = self.db.get_history(sym, limit=limit_per_symbol)
            except Exception:
                df = pd.DataFrame()

            if df is None or df.empty:
                continue

            try:
                if 'high' in df.columns and 'low' in df.columns and 'atr' in df.columns:
                    if df['high'].notna().any() and df['low'].notna().any() and df['atr'].notna().any():
                        barrier_seen = True
            except Exception:
                pass

            X, y = self._build_features_and_labels(df, sym)
            if X is None or X.empty or y is None or y.empty:
                continue

            rows_by_symbol[str(sym).upper()] = int(len(y))
            X_all.append(X)
            y_all.append(y)

        if not X_all:
            self._emit("\U0001f9e0 [AI] Not enough data to train.")
            self.is_trained = False
            self.last_train_stats = {
                'status': 'no_data',
                'rows': 0,
            }
            return

        X = pd.concat(X_all, axis=0, ignore_index=True)
        y = pd.concat(y_all, axis=0, ignore_index=True)

        # Ensure we have both classes
        try:
            pos_rate = float(y.mean())
        except Exception:
            pos_rate = 0.0

        if len(y) < min_rows or pos_rate in (0.0, 1.0):
            self._emit("\U0001f9e0 [AI] Not enough data to train.")
            self.is_trained = False
            self.last_train_stats = {
                'status': 'insufficient_rows_or_classes',
                'rows': int(len(y)),
                'pos_rate': float(pos_rate),
                'rows_by_symbol': rows_by_symbol,
            }
            return

        # Base model
        base = RandomForestClassifier(
            n_estimators=int(self._cfg('ai_rf_estimators', 200)),
            max_depth=None,
            min_samples_leaf=int(self._cfg('ai_rf_min_leaf', 5)),
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1,
        )

        # Walk-forward evaluation
        tscv = TimeSeriesSplit(n_splits=wf_splits)
        fold_metrics = []

        def _compute_metrics(y_true, p):
            out = {}
            try:
                out['accuracy'] = float(accuracy_score(y_true, (p >= 0.5).astype(int)))
            except Exception:
                pass
            try:
                # AUC requires both classes in the fold
                if len(np.unique(y_true)) > 1:
                    out['auc'] = float(roc_auc_score(y_true, p))
            except Exception:
                pass
            try:
                out['brier'] = float(brier_score_loss(y_true, p))
            except Exception:
                pass
            try:
                # log_loss can be unstable if p is 0/1
                p2 = np.clip(p, 1e-6, 1 - 1e-6)
                out['log_loss'] = float(log_loss(y_true, np.vstack([1 - p2, p2]).T, labels=[0, 1]))
            except Exception:
                pass
            return out

        X_np = X.to_numpy(dtype='float64')
        y_np = y.to_numpy(dtype='int64')

        for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X_np), start=1):
            try:
                if len(train_idx) < 200 or len(test_idx) < 100:
                    continue

                # Time-respecting calibration split from the training window
                cut = int(len(train_idx) * (1.0 - cal_frac))
                if cut < 200 or (len(train_idx) - cut) < 100:
                    continue

                train2 = train_idx[:cut]
                calib = train_idx[cut:]

                base_f = RandomForestClassifier(
                    n_estimators=base.n_estimators,
                    max_depth=base.max_depth,
                    min_samples_leaf=base.min_samples_leaf,
                    class_weight=base.class_weight,
                    random_state=42,
                    n_jobs=-1,
                )

                base_f.fit(X_np[train2], y_np[train2])

                # --- Calibration (sklearn compatibility-safe) ---
                p = None
                try:
                    # Older sklearn: prefit calibration
                    cal = CalibratedClassifierCV(base_f, method=cal_method, cv='prefit')
                    cal.fit(X_np[calib], y_np[calib])
                    p = cal.predict_proba(X_np[test_idx])[:, 1]
                except Exception:
                    # Newer sklearn: no cv='prefit' -> calibrate on probabilities (or fall back to raw probs)
                    try:
                        p_cal = base_f.predict_proba(X_np[calib])[:, 1]
                        calibrator = _fit_probability_calibrator(cal_method, p_cal, y_np[calib])
                        p_raw = base_f.predict_proba(X_np[test_idx])[:, 1]
                        p = _apply_probability_calibrator(calibrator, cal_method, p_raw)
                    except Exception:
                        try:
                            p = base_f.predict_proba(X_np[test_idx])[:, 1]
                        except Exception:
                            p = None

                if p is None:
                    continue
                m = _compute_metrics(y_np[test_idx], p)
                m['fold'] = int(fold_i)
                m['train_rows'] = int(len(train_idx))
                m['test_rows'] = int(len(test_idx))
                fold_metrics.append(m)
            except Exception:
                continue

        # Fit final calibrated model on full dataset using last cal_frac as calibration window
        try:
            n_total = len(X_np)
            cut = int(n_total * (1.0 - cal_frac))
            cut = max(200, min(n_total - 100, cut))

            train2_idx = np.arange(0, cut)
            calib_idx = np.arange(cut, n_total)

            base.fit(X_np[train2_idx], y_np[train2_idx])

            # --- Calibration (sklearn compatibility-safe) ---
            final_model = None
            try:
                cal_model = CalibratedClassifierCV(base, method=cal_method, cv='prefit')
                cal_model.fit(X_np[calib_idx], y_np[calib_idx])
                final_model = cal_model
            except Exception:
                try:
                    p_cal = base.predict_proba(X_np[calib_idx])[:, 1]
                    calibrator = _fit_probability_calibrator(cal_method, p_cal, y_np[calib_idx])
                    if calibrator is not None:
                        final_model = _CalibratedWrapper(base, calibrator, cal_method)
                except Exception:
                    final_model = None

            # If calibration couldn't be fit, fall back to the base model (uncalibrated).
            self.base_model = base
            self.model = final_model if final_model is not None else base
            self.is_trained = True

            # Aggregate fold metrics
            agg = {}
            if fold_metrics:
                for k in ('accuracy', 'auc', 'brier', 'log_loss'):
                    vals = [fm.get(k) for fm in fold_metrics if fm.get(k) is not None]
                    if vals:
                        agg[f'walkforward_{k}_mean'] = float(np.mean(vals))
                        agg[f'walkforward_{k}_std'] = float(np.std(vals))

            # Use last fold as the "holdout" view when available
            holdout = fold_metrics[-1] if fold_metrics else {}

            self.last_train_stats = {
                'status': 'trained',
                'rows': int(len(y_np)),
                'pos_rate': float(pos_rate),
                'symbols': [str(s).upper() for s in symbols],
                'rows_by_symbol': rows_by_symbol,
                'label': {
                    'horizon_bars': int(self._cfg('ai_label_horizon_bars', 60)),
                    'tp_atr_mult': float(self._cfg('ai_label_tp_atr_mult', 1.5)),
                    'sl_atr_mult': float(self._cfg('ai_label_sl_atr_mult', 1.0)),
                    'min_return_pct': float(self._cfg('ai_label_min_return_pct', 0.001)),
                    'method': 'triple_barrier' if barrier_seen else 'forward_return',
                },
                'calibration': {
                    'method': cal_method,
                    'fraction': float(cal_frac),
                },
                'walkforward': {
                    'splits': int(wf_splits),
                    'fold_metrics': fold_metrics,
                    **agg,
                },
                'holdout': holdout,
            }

            # Keep the console line familiar
            acc = holdout.get('accuracy', None)
            if acc is not None:
                self._emit(f"\U0001f9e0 [AI] Model Trained (v2). Holdout Accuracy: {acc*100:.2f}% | Rows: {len(y_np)} | PosRate: {pos_rate*100:.2f}%")
            else:
                self._emit(f"\U0001f9e0 [AI] Model Trained (v2). Rows: {len(y_np)} | PosRate: {pos_rate*100:.2f}%")

        except Exception as e:
            self._emit(f"\U0001f9e0 [AI] Training failed: {e}")
            self.is_trained = False
            self.model = None
            self.base_model = None

    def predict_probability(self, current_bar_data: dict):
        """Return calibrated probability (0..1) for a bullish outcome."""
        try:
            if not self.is_trained or self.model is None:
                return 0.50

            # Build feature row from bar dict (engine provides these)
            close = float(current_bar_data.get('close', np.nan))
            volume = float(current_bar_data.get('volume', np.nan))
            volume_avg = float(current_bar_data.get('volume_avg', np.nan))

            ema_200 = float(current_bar_data.get('ema_200', np.nan))
            bb_lower = float(current_bar_data.get('bb_lower', np.nan))
            bb_upper = float(current_bar_data.get('bb_upper', np.nan))
            atr = float(current_bar_data.get('atr', np.nan))

            rsi = float(current_bar_data.get('rsi', np.nan))
            adx = float(current_bar_data.get('adx', np.nan))

            volume_ratio = np.nan
            try:
                if np.isfinite(volume) and np.isfinite(volume_avg) and volume_avg > 0:
                    volume_ratio = float(volume / volume_avg)
            except Exception:
                volume_ratio = np.nan

            dist_ema = np.nan
            try:
                if np.isfinite(close) and np.isfinite(ema_200) and ema_200 != 0:
                    dist_ema = float((close - ema_200) / ema_200)
            except Exception:
                dist_ema = np.nan

            bb_pos = self._safe_bb_pos(close, bb_lower, bb_upper)

            atr_pct = np.nan
            try:
                if np.isfinite(atr) and np.isfinite(close) and close > 0:
                    atr_pct = float(atr / close)
            except Exception:
                atr_pct = np.nan

            row = {
                'rsi': rsi,
                'adx': adx,
                'volume_ratio': volume_ratio,
                'dist_ema': dist_ema,
                'bb_pos': bb_pos,
                'atr_pct': atr_pct,
            }

            X = pd.DataFrame([row], columns=self.feature_cols).replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0.0)

            prob = float(self.model.predict_proba(X)[0][1])
            if not np.isfinite(prob):
                return 0.50
            return max(0.0, min(1.0, prob))

        except Exception:
            return 0.50