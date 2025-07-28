from __future__ import annotations

"""Typed record helpers used by the memory layer.

These dataclasses mirror DB rows but are intentionally lightweight. They help us
pass structured data around instead of raw sqlite tuples.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class SessionRecord:
    id: int
    started_at: str
    profile: str
    notes: Optional[str] = None


@dataclass(slots=True)
class MessageRecord:
    id: int
    session_id: int
    turn_index: int
    role: str            # 'user' | 'assistant' | 'tool'
    content: str
    tokens: Optional[int] = None
    created_at: Optional[str] = None


@dataclass(slots=True)
class FactRecord:
    id: int
    subject_type: str     # 'city','program','person','generic'
    subject_slug: str     # normalized slug, e.g. 'grandview_tx'
    attr: str             # 'mayor','population','deadline'
    value: str
    source: str           # 'user','web','doc','system'
    confidence: float
    status: str           # 'active','superseded','conflict','pending-verify'
    provenance: Optional[str] = None  # JSON serialized
    latest: int = 1
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    @property
    def key(self) -> str:
        """Convenience composite key: ``{attr}:{subject_slug}``."""
        return f"{self.attr}:{self.subject_slug}"


@dataclass(slots=True)
class FileRecord:
    id: int
    path: str
    sha256: str
    mime: str
    embedded: int  # 0/1
    added_at: Optional[str] = None
    meta: Optional[str] = None  # JSON serialized
