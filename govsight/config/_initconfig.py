"""Extended configuration notes for GovSight.

This file is purely documentation + future expansion hooks. Nothing here is
imported automatically (unless you choose to). It exists because Mike prefers
explicit naming rather than burying docs in multiple ``__init__.py`` files.

### Config Resolution Order (R0)
1. **Environment variables** – take highest precedence.
2. **Legacy `config.py` globals** – fallback values (preserve current behavior).
3. **Defaults in `Settings` dataclass** – last resort.

### Profiles
We expose a ``--profile`` flag (dev|staging|prod) but in R0 all profiles resolve
to the same settings. Later we’ll add profile maps (different DB paths, Pinecone
namespaces, auto_web toggles).

### Directory Guarantees
The loader creates ``data/`` and ``logs/`` if missing so the app never crashes
on first run.

See :mod:`govsight.config.settings` for implementation.
"""