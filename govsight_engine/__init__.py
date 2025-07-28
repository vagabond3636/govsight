"""GovSight package bootstrap (minimal).

*Why this file is tiny:* Python uses ``__init__.py`` to treat this directory as a
package. We intentionally keep logic out of here. For metadata and longer docs,
see :mod:`govsight._initbase`.
"""

from ._initbase import __version__  # re-export version string

__all__ = ["__version__"]