from __future__ import annotations

"""GovSight Chat CLI (R0 wrapper, module invocation).

Usage:
    python -m govsight.cli.chat_cli --profile dev

This module provides a *stable* command-line entrypoint that delegates to your
**legacy** ``talk.py`` engine while we refactor. Nothing about the chat logic
changes yet. Once the new modular engine is ready, we will swap the internals so
this CLI calls the new engine instead of the legacy one.
"""

import argparse
import pathlib
import sys

# ---------------------------------------------------------------------------
# Ensure the project root (where talk.py lives) is importable.
# When launching with ``-m govsight.cli.chat_cli`` Python sets sys.path to the
# package directory, not necessarily your project root, so we manually add it.
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local package imports -----------------------------------------------------
from govsight.config import load_settings  # loads env + legacy config
from govsight.logging_utils import get_logger

# Legacy engine import ------------------------------------------------------
# We import talk.py AFTER adjusting sys.path so the root is visible.
import talk  # type: ignore  # (legacy monolithic engine)


def main(argv=None) -> int:
    """Parse args, load settings, bootstrap logging, and invoke legacy engine."""
    parser = argparse.ArgumentParser(description="GovSight Chat CLI")
    parser.add_argument(
        "--profile", default="dev", help="runtime profile: dev | staging | prod"
    )
    args = parser.parse_args(argv)

    settings = load_settings(profile=args.profile)
    log = get_logger("govsight", log_dir=settings.log_dir)

    log.info("🧠 GovSight Chat CLI (R0) – invoking legacy engine...")

    # TODO (R1+): We'll construct a ChatApp(settings) and run that instead of talk.main()
    try:
        return talk.main()
    except AttributeError:
        # Some older talk.py versions may execute at import; handle gracefully.
        log.warning("Legacy talk.py has no main(); import side effects executed.")
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())