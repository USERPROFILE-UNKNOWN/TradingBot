"""Simple recurring job scheduler for background AI Agent maintenance tasks."""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict


class JobScheduler:
    def __init__(self, log_callback=None):
        self.log = log_callback or (lambda *_args, **_kwargs: None)
        self._jobs: Dict[str, dict] = {}
        self._stop = threading.Event()
        self._thread = None

    def add_job(self, name: str, interval_sec: int, func: Callable[[], None]) -> None:
        if not name or interval_sec <= 0 or not callable(func):
            return
        self._jobs[name] = {
            "interval": int(interval_sec),
            "func": func,
            "next_run": time.time() + int(interval_sec),
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
                try:
                    job["func"]()
                except Exception as e:
                    try:
                        self.log(f"[SCHEDULER] Job failed ({name}): {e}")
                    except Exception:
                        pass
                finally:
                    job["next_run"] = time.time() + int(job.get("interval", 60))
            time.sleep(1)
