"""TradingView webhook ingestion (stdlib-only).

Design goals
- Reliable, low-dependency HTTP receiver suitable for PyInstaller EXE builds.
- Respond quickly (<1s) and offload persistence to a background worker.
- Optional shared-secret validation and optional HMAC signature validation.

TradingView webhook requests are typically HTTP POST where the body is the "message"
field from your alert. Best practice is to send JSON so we can extract fields.

Suggested TradingView alert message JSON:
{
  "secret": "<shared-secret>",
  "symbol": "AAPL",
  "exchange": "NASDAQ",
  "timeframe": "15",
  "signal": "BUY",
  "price": 189.12,
  "idempotency_key": "<optional-stable-id>"
}

Auth & signature
- If a shared secret is configured, the server will accept the secret from:
  - Header: X-Webhook-Secret / X-TradingView-Secret
  - JSON field: secret
- Optional HMAC signature validation when signature header exists:
  - Header: X-Webhook-Signature / X-Signature
  - Format accepted: "sha256=<hex>" or "<hex>"
  - Digest: HMAC-SHA256(secret, raw_body_bytes)

This module intentionally does not introduce external dependencies.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import queue
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional, Tuple


HeaderDict = Dict[str, str]
AlertCallback = Callable[[Dict[str, Any], str, str, HeaderDict, str], None]


def _now_epoch_sec() -> int:
    return int(time.time())


def _safe_decode(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except Exception:
        try:
            return b.decode("utf-8", errors="replace")
        except Exception:
            return ""


def _normalize_signature(sig: str) -> str:
    s = (sig or "").strip()
    if not s:
        return ""
    if s.lower().startswith("sha256="):
        s = s.split("=", 1)[1].strip()
    return s


def _hmac_sha256_hex(secret: str, body: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def _read_header(headers, name: str) -> str:
    try:
        v = headers.get(name)
    except Exception:
        v = None
    return str(v or "").strip()


class TradingViewWebhookServer:
    """Threaded HTTP server wrapper.

    Args:
        host: listen host (e.g., 127.0.0.1)
        port: listen port (e.g., 5001)
        secret: shared secret for auth (optional; empty disables auth)
        allowed_signals: optional allow-list of signals (uppercased tokens)
        on_alert: callback invoked async for each accepted request
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        secret: str = "",
        allowed_signals: Optional[list[str]] = None,
        on_alert: Optional[AlertCallback] = None,
        max_body_bytes: int = 256_000,
        path: str = "/tradingview",
        logger=None,
    ):
        self.host = str(host or "127.0.0.1").strip() or "127.0.0.1"
        self.port = int(port)
        self.secret = str(secret or "")
        self.allowed_signals = [s.strip().upper() for s in (allowed_signals or []) if str(s).strip()]
        self.on_alert = on_alert
        self.max_body_bytes = int(max_body_bytes)
        self.path = str(path or "/tradingview")
        self.logger = logger

        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

        # Async work queue to keep handler response-time tiny.
        self._q: queue.Queue[Tuple[Dict[str, Any], str, str, HeaderDict, str]] = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def start(self) -> None:
        if self._httpd is not None:
            return

        server = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args, **_kwargs):
                # Suppress BaseHTTPRequestHandler default stdout logging.
                return

            def do_GET(self):
                if self.path.rstrip("/") != server.path.rstrip("/"):
                    self.send_response(HTTPStatus.NOT_FOUND)
                    self.end_headers()
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok","service":"tradingview_webhook"}')

            def do_POST(self):
                if self.path.rstrip("/") != server.path.rstrip("/"):
                    self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
                    return

                # Read body (bounded)
                try:
                    clen = int(_read_header(self.headers, "Content-Length") or "0")
                except Exception:
                    clen = 0

                if clen <= 0:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "empty_body"})
                    return

                if clen > server.max_body_bytes:
                    self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"ok": False, "error": "body_too_large"})
                    return

                try:
                    body = self.rfile.read(clen)
                except Exception:
                    server._log_exception("[E_TV_HTTP_READ_FAIL] Failed reading request body")
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "read_failed"})
                    return

                raw_text = _safe_decode(body)
                hdrs = {k: str(v) for k, v in (self.headers.items() if self.headers else [])}
                client_ip = str(getattr(self, "client_address", ("", 0))[0] or "")

                # Parse JSON if possible
                payload: Dict[str, Any] = {}
                try:
                    payload = json.loads(raw_text)
                    if not isinstance(payload, dict):
                        payload = {"message": payload}
                except Exception:
                    payload = {"message": raw_text}

                # Auth
                if server.secret:
                    supplied = (
                        _read_header(self.headers, "X-Webhook-Secret")
                        or _read_header(self.headers, "X-TradingView-Secret")
                        or str(payload.get("secret", "") or "").strip()
                    )
                    if not supplied or supplied != server.secret:
                        server._log_warning("[E_TV_AUTH_FAIL] Unauthorized TradingView webhook", client_ip=client_ip)
                        self._send_json(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "unauthorized"})
                        return

                    sig = (
                        _read_header(self.headers, "X-Webhook-Signature")
                        or _read_header(self.headers, "X-Signature")
                    )
                    sig_norm = _normalize_signature(sig)
                    if sig_norm:
                        try:
                            expected = _hmac_sha256_hex(server.secret, body)
                        except Exception:
                            expected = ""
                        if not expected or not hmac.compare_digest(sig_norm, expected):
                            server._log_warning("[E_TV_SIG_FAIL] Invalid TradingView webhook signature", client_ip=client_ip)
                            self._send_json(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "bad_signature"})
                            return

                # Allow-list signal check (if configured)
                try:
                    sig_name = str(payload.get("signal") or payload.get("action") or payload.get("side") or "").strip().upper()
                except Exception:
                    sig_name = ""
                if server.allowed_signals and sig_name and sig_name not in server.allowed_signals:
                    # Acknowledge but mark as ignored for downstream.
                    payload["_ignored"] = True
                    payload["_ignored_reason"] = "signal_not_allowed"

                # Idempotency key
                idem = (
                    str(payload.get("idempotency_key") or payload.get("alert_id") or payload.get("id") or "").strip()
                    or hashlib.sha256(body).hexdigest()
                )

                # Enqueue and ACK immediately
                try:
                    server._q.put_nowait((payload, raw_text, idem, hdrs, client_ip))
                except Exception:
                    server._log_exception("[E_TV_QUEUE_FULL] Failed queueing TradingView webhook")
                    self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"ok": False, "error": "queue_full"})
                    return

                self._send_json(HTTPStatus.OK, {"ok": True})

            def _send_json(self, status: HTTPStatus, obj: dict):
                try:
                    b = json.dumps(obj, separators=(",", ":")).encode("utf-8")
                except Exception:
                    b = b"{}"
                self.send_response(int(status))
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                try:
                    self.wfile.write(b)
                except Exception:
                    return

        self._httpd = ThreadingHTTPServer((self.host, self.port), _Handler)
        self._httpd.daemon_threads = True

        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

        self._log_info(
            "[TV] Webhook listener started",
            host=self.host,
            port=self.port,
            path=self.path,
        )

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._httpd is not None:
                self._httpd.shutdown()
        except Exception:
            pass
        try:
            if self._httpd is not None:
                self._httpd.server_close()
        except Exception:
            pass
        self._httpd = None

        self._log_info("[TV] Webhook listener stopped", host=self.host, port=self.port, path=self.path)

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            except Exception:
                continue

            payload, raw_text, idem, hdrs, client_ip = item

            try:
                if callable(self.on_alert):
                    self.on_alert(payload, raw_text, idem, hdrs, client_ip)
            except Exception:
                self._log_exception("[E_TV_ON_ALERT_FAIL] on_alert callback failed")

    # ---- logging helpers (best-effort) ----

    def _log_info(self, msg: str, **extra):
        if self.logger is None:
            return
        try:
            self.logger.info(msg, extra=extra)
        except Exception:
            try:
                self.logger.info(msg)
            except Exception:
                pass

    def _log_warning(self, msg: str, **extra):
        if self.logger is None:
            return
        try:
            self.logger.warning(msg, extra=extra)
        except Exception:
            try:
                self.logger.warning(msg)
            except Exception:
                pass

    def _log_exception(self, msg: str, **extra):
        if self.logger is None:
            return
        try:
            self.logger.exception(msg, extra=extra)
        except Exception:
            try:
                self.logger.exception(msg)
            except Exception:
                pass
