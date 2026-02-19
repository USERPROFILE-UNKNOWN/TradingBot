"""Manual AI training tool (v6.21.3)."""

from __future__ import annotations

import importlib
from typing import Any


def run_manual_training(db_manager: Any, config: Any, log_fn=None, *, force: bool = True):
    """Run one-shot AI training if optional AI dependencies are installed."""
    logger = log_fn or (lambda *_a, **_k: None)
    try:
        ai_mod = importlib.import_module("modules.ai")
        ai_cls = getattr(ai_mod, "AI_Oracle", None)
        if ai_cls is None:
            raise RuntimeError("AI_Oracle class missing")
        oracle = ai_cls(db_manager, config, logger)
        logger("🤖 [AI] Manual training started.")
        stats = oracle.train_model(force=bool(force))
        logger("🤖 [AI] Manual training completed.")
        return {"ok": True, "stats": stats}
    except Exception as e:
        logger(f"🤖 [AI] Manual training unavailable: {e}")
        return {"ok": False, "error": str(e)}
