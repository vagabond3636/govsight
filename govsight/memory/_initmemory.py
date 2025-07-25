"""Developer notes: GovSight Memory Layer.

Design priorities:
- Append-only *audit* of chats (sessions/messages).
- Structured *facts* store with versioning + provenance.
- Lightweight, stdlib-only (sqlite3) to minimize install friction.
- Transitional compatibility with legacy ``memory_manager.py`` (R1).

Lifecycle of a user correction example
-------------------------------------
User: "No, the mayor is Bill Houston."
 ↓ parse subject=Grandview,TX attr=mayor value=Bill Houston
 ↓ Memory.remember_fact(... source="user", confidence=0.9, status="pending-verify")
 ↓ Mark any previous *latest* fact rows (same subject_slug+attr) as latest=0, status="superseded"
 ↓ Retrieval prefers latest=1 row going forward
Later: verification step upgrades status to "active" if confirmed.

This file is documentation only; no runtime imports required.
"""
