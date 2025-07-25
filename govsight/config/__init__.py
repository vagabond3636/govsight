"""Configuration package init (minimal).

All real work happens in :mod:`govsight.config.settings`.
A supplemental long-form explanation lives in :mod:`govsight.config._initconfig`.

We re-export the key functions/classes here for ergonomic imports:

    from govsight.config import load_settings, Settings

"""

from .settings import Settings, load_settings, settings_from_env  # re-export

__all__ = ["Settings", "load_settings", "settings_from_env"]