"""GovSight memory subsystem (public API).

Import the :class:`Memory` facade from here::

    from govsight.memory import Memory

This file stays tiny. Implementation lives in :mod:`govsight.memory.memory`.
"""

from .memory import Memory, MemoryError
from .records import SessionRecord, MessageRecord, FactRecord, FileRecord

__all__ = [
    "Memory",
    "MemoryError",
    "SessionRecord",
    "MessageRecord",
    "FactRecord",
    "FileRecord",
]
