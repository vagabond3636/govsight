from __future__ import annotations

"""
Refactored memory_manager.py – Slim version
Only handles conversation session tracking and fact routing
GPT and embedding logic must move to dedicated modules.
"""

import time
import sqlite3
from typing import Optional

from govsight.config import settings
from govsight.memory.memory import Memory


class MemoryManager:
    def __init__(self):
        self.memory = Memory(settings=settings)

    def start_session(self, session_id: str) -> None:
        self.memory.start_session(session_id)

    def log_message(self, session_id: str, role: str, message: str) -> None:
        self.memory.log_message(session_id, role, message)

    def get_messages(self, session_id: str) -> list:
        return self.memory.get_messages(session_id)

    def save_fact(self, subject: str, attribute: str, value: str,
                  source: Optional[str] = None, confidence: Optional[float] = None) -> None:
        self.memory.store_fact(subject, attribute, value, source, confidence)

    def insert_fact_triples(self, triples: list[dict]) -> None:
        self.memory.insert_fact_triples(triples)

    def search_memory(self, query: str) -> Optional[str]:
        return self.memory.search(query)

    # Placeholder methods for future refactoring
    def summarize_session(self, session_id: str) -> str:
        raise NotImplementedError("Move this to llm/summarization_agent.py")

    def embed_to_vectorstore(self, text: str, namespace: Optional[str] = None):
        raise NotImplementedError("Move this to vector/embedder.py")

    def track_watchlist(self, subject: str):
        raise NotImplementedError("Move this to memory/watchlist.py")
