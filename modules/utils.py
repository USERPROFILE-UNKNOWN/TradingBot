"""Shared utilities and stable convenience imports.

This module centralizes a small set of commonly used helpers to keep imports
consistent across the codebase.
"""

from __future__ import annotations

from .app_constants import APP_RELEASE, APP_VERSION
from .paths import get_paths
from .config_io import (
    ensure_split_config_layout,
    load_split_config,
    write_split_config,
    get_last_config_sanitizer_report,
)
