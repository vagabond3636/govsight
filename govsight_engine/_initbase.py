"""GovSight base package metadata and developer notes.

This module holds package-level metadata *separate* from ``__init__.py`` to keep
imports clean and avoid side effects.

R0 refactor goals:
- Wrap legacy monolithic scripts (talk.py, memory_manager.py, config.py).
- Provide stable import points for settings + logging.
- Enable gradual migration of memory + retrieval logic without breaking CLI.
"""

from importlib import metadata as _metadata

try:  # When installed via pip / build backend
    __version__ = _metadata.version("govsight")
except Exception:  # Local src checkout fallback
    __version__ = "0.0.0-dev"