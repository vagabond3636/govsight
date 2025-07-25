from __future__ import annotations

"""Centralized logging utilities for GovSight.

Why centralize? Historically, the monolithic ``talk.py`` created ad-hoc log
handlers, which led to duplication and occasional missing logs when refactoring.
This helper guarantees that every module requesting a logger gets a consistent
format and rotating file behavior.
"""

import logging
import logging.handlers
import os
import pathlib
from typing import Optional

# Cache created loggers so repeated calls don't duplicate handlers
_LOGGER_CACHE = {}


def get_logger(name: str = "govsight", log_dir: Optional[str] = None) -> logging.Logger:
    """Return a configured :class:`logging.Logger`.

    Parameters
    ----------
    name:
        Logger name (also used in log filename: ``{name}.log``).
    log_dir:
        Directory where log files should be written. Created if missing.

    Behavior
    --------
    * INFO level by default (tweak later via settings).
    * Rotating file handler (5MB x5 backups) + console handler.
    * Reuses cached logger on subsequent calls.
    """
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Guard against double-adding handlers if the interpreter reloads modules
    if logger.handlers:
        return logger

    if log_dir is None:
        log_dir = "logs"
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")

    # Rotating file --------------------------------------------------------
    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )

    # Console --------------------------------------------------------------
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    _LOGGER_CACHE[name] = logger
    return logger