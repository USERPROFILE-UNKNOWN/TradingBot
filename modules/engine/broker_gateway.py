"""BrokerGateway

Centralizes Alpaca connection + retry mechanics.

Design goals (Phase 2 / v5.15.0):
- Keep TradingEngine(core.py) focused on orchestration.
- Keep dependency surface unchanged (alpaca_trade_api already used).
- Preserve behavior: callers still use an API-like object, but core no longer
  owns connect/retry implementation.

NOTE: This gateway intentionally provides a thin surface:
- .connect() builds the REST client using the existing config format
- .retry_api_call(...) mirrors the previous TradingEngine.retry_api_call
- __getattr__ proxies through to the underlying alpaca_trade_api REST client
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import alpaca_trade_api as tradeapi
import requests


class BrokerGateway:
    def __init__(self, config: Any, py_logger, note_api_error: Optional[Callable[..., Any]] = None):
        self.config = config
        self._py_logger = py_logger
        self._note_api_error = note_api_error

        self.base_url: str = ""
        self._api: Optional[tradeapi.REST] = None

    def __bool__(self) -> bool:
        return self._api is not None

    def connect(self) -> tradeapi.REST:
        """Create the Alpaca REST client from config and store it.

        Raises if required fields are missing.
        """
        # v5.15.0+: config is loaded via ConfigParser (config.ini + keys.ini merged).
        # Some tests/stubs may still pass a dict-like config.
        keys: dict = {}
        try:
            import configparser

            if isinstance(self.config, (configparser.ConfigParser, configparser.RawConfigParser)):
                if self.config.has_section("KEYS"):
                    keys = {k: v for k, v in self.config.items("KEYS")}
            elif isinstance(self.config, configparser.SectionProxy):
                keys = {k: v for k, v in self.config.items()}
        except Exception:
            # fall through to dict-like handling
            pass

        if not keys:
            if isinstance(self.config, dict):
                keys = self.config.get("KEYS", {}) or {}
            else:
                try:
                    keys = dict(self.config["KEYS"])
                except Exception:
                    keys = {}

        # Preserve prior behavior: allow booleans or strings.
        paper = keys.get("paper_trading", True)
        if isinstance(paper, str):
            paper = paper.strip().lower() in {"1", "true", "yes", "y", "on"}

        base_url = (keys.get("base_url") or "").strip()
        self.base_url = base_url or (
            "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        )

        api_key = (keys.get("alpaca_key") or "").strip()
        api_secret = (keys.get("alpaca_secret") or "").strip()
        if not api_key or not api_secret:
            raise ValueError(
                "Missing Alpaca API credentials (KEYS.alpaca_key / KEYS.alpaca_secret). Check config/keys.ini."
            )

        self._api = tradeapi.REST(api_key, api_secret, self.base_url, api_version="v2")
        return self._api

    def retry_api_call(self, func: Callable[..., Any], *args: Any, retries: int = 3, delay: float = 2.0, **kwargs: Any):
        """Retry wrapper used by engine for network-ish Alpaca calls.

        This mirrors the previous TradingEngine.retry_api_call implementation.
        """
        last_error = None
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except (requests.exceptions.RequestException, tradeapi.rest.APIError) as e:
                last_error = e
                if self._note_api_error:
                    try:
                        self._note_api_error(str(e))
                    except Exception:
                        # fail-open; error streak tracking must never crash callers
                        pass

                msg = f"[E_ENGINE_API_RETRY] API call failed (attempt {attempt + 1}/{retries}): {e}"
                if self._py_logger:
                    self._py_logger.warning(msg)

                time.sleep(delay)

        if self._py_logger:
            self._py_logger.error("[E_ENGINE_API_RETRY_FAIL] API call failed after retries")

        return None

    def __getattr__(self, name: str):
        """Proxy attribute access to underlying Alpaca REST client."""
        if self._api is None:
            raise AttributeError(name)
        return getattr(self._api, name)
