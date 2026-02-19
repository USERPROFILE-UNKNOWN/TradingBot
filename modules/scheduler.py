"""Simple recurring job scheduler for background AI Agent maintenance tasks."""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict


class JobScheduler:
    def __init__(self, log_callback=None, metrics_store=None):
        self.log = log_callback or (lambda *_args, **_kwargs: None)
        self.metrics = metrics_store
        self._jobs: Dict[str, dict] = {}
        self._stop = threading.Event()
        self._thread = None

    def add_job(self, name: str, interval_sec: int, func: Callable[[], None]) -> None:
        if not name or interval_sec <= 0 or not callable(func):
            return
        next_run = time.time() + int(interval_sec)
        try:
            if self.metrics is not None and hasattr(self.metrics, "get_job_state"):
                st = self.metrics.get_job_state(name) or {}
                cool = st.get("cooldown_until")
                if cool is not None:
                    next_run = max(float(next_run), float(cool))
        except Exception:
            pass

        self._jobs[name] = {
            "interval": int(interval_sec),
            "func": func,
            "next_run": float(next_run),
        }

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            for name, job in list(self._jobs.items()):
                if now < float(job.get("next_run", now + 9999)):
                    continue
                started = time.time()
                status = "ok"
                err = ""
                try:
                    job["func"]()
                except Exception as e:
                    status = "error"
                    err = str(e)
                    try:
                        self.log(f"[SCHEDULER] Job failed ({name}): {e}")
                    except Exception:
                        pass
                finally:
                    dur_ms = int((time.time() - started) * 1000)
                    try:
                        if self.metrics is not None:
                            self.metrics.log_job_run(name, status, dur_ms, {"error": err} if err else {})
                            if hasattr(self.metrics, "update_job_state"):
                                self.metrics.update_job_state(name, status, error=err)
                    except Exception:
                        pass
                    job["next_run"] = time.time() + int(job.get("interval", 60))
            time.sleep(1)
