from __future__ import annotations

"""Runtime settings loader for GovSight.

This module centralizes all configuration resolution so the rest of the codebase
can consume a *single* validated ``Settings`` object instead of scattering env
reads throughout the code.

**R0 behavior preservation:** We intentionally read from the legacy top-level
``config.py`` (project root) so nothing about your current run changes while we
refactor.
"""

import os
import pathlib
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

# ---------------------------------------------------------------------------
# Settings Dataclass
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Settings:
    """Resolved runtime configuration.

    Attributes mirror existing legacy config globals plus a few new knobs that
    we will use in later refactor phases.
    """

    profile: str = "dev"  # semantic runtime profile label
    openai_api_key: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    pinecone_env: Optional[str] = None
    pinecone_index: str = "gov-index"  # default; may be overridden
    db_path: str = "data/memory.db"    # you confirmed this path
    log_dir: str = "logs"              # you confirmed this path
    auto_web: bool = True              # allow web search fallback
    model: str = "gpt-4o-mini"         # default OpenAI chat model
    temperature: float = 0.2           # sensible, low-variance default

    def ensure_dirs(self) -> None:
        """Create directory parents so downstream code never fails on I/O."""
        pathlib.Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def asdict(self) -> Dict[str, Any]:  # convenience for logging/JSON
        return asdict(self)


# ---------------------------------------------------------------------------
# Legacy Config Introspection
# ---------------------------------------------------------------------------

def _read_legacy_config() -> dict:
    """Import *legacy* top-level ``config.py`` and return its public attributes.

    Used to preserve current behavior during the R0 refactor. If ``config.py``
    is missing or import fails, we just return an empty dict.
    """
    try:
        import config as legacy_cfg  # relative import from project root
    except Exception:  # pragma: no cover - missing legacy config
        return {}

    return {
        k: getattr(legacy_cfg, k)
        for k in dir(legacy_cfg)
        if not k.startswith("__")
    }


# ---------------------------------------------------------------------------
# Settings Builders
# ---------------------------------------------------------------------------

def settings_from_env(profile: str = "dev") -> Settings:
    """Assemble settings from env vars + legacy config + defaults."""
    legacy = _read_legacy_config()
    s = Settings(profile=profile)

    # API keys --------------------------------------------------------------
    s.openai_api_key = os.getenv("OPENAI_API_KEY", legacy.get("OPENAI_API_KEY"))
    s.pinecone_api_key = os.getenv("PINECONE_API_KEY", legacy.get("PINECONE_API_KEY"))
    s.pinecone_env = os.getenv("PINECONE_ENV", legacy.get("PINECONE_ENV"))

    # Pinecone index -------------------------------------------------------
    s.pinecone_index = os.getenv("PINECONE_INDEX", legacy.get("PINECONE_INDEX", s.pinecone_index))

    # DB path --------------------------------------------------------------
    s.db_path = os.getenv("GOVSIGHT_DB_PATH", legacy.get("DB_PATH", s.db_path))

    # Logs -----------------------------------------------------------------
    s.log_dir = os.getenv("GOVSIGHT_LOG_DIR", legacy.get("LOG_DIR", s.log_dir))

    # Model / temperature --------------------------------------------------
    s.model = os.getenv("GOVSIGHT_MODEL", legacy.get("MODEL", s.model))
    try:
        s.temperature = float(os.getenv("GOVSIGHT_TEMP", legacy.get("TEMPERATURE", s.temperature)))
    except Exception:  # leave default
        pass

    # Auto web fallback ----------------------------------------------------
    raw_auto = os.getenv("GOVSIGHT_AUTO_WEB", legacy.get("AUTO_WEB", str(s.auto_web)))
    if isinstance(raw_auto, str):
        s.auto_web = raw_auto.lower() in {"1", "true", "t", "yes", "y"}
    else:
        s.auto_web = bool(raw_auto)

    return s


def load_settings(profile: str = "dev") -> Settings:
    """Public loader: returns a fully-initialized :class:`Settings` object."""
    settings = settings_from_env(profile=profile)
    settings.ensure_dirs()
    return settings