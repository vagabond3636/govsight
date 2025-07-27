from __future__ import annotations

"""
GovSight LLM Summarization Agent
- Summarizes sessions or document chunks via GPT
- Can be reused by CLI, UI, API, or pipeline
"""

from govsight.llm.openai_wrapper import ask_llm
from govsight.memory.memory import Memory
from govsight.config import settings


def summarize_session(session_id: str) -> str:
    memory = Memory(settings=settings)
    messages = memory.get_messages(session_id)

    context = "\n".join([f"{role}: {msg}" for role, msg in messages])
    prompt = f"""
    Summarize this conversation session into:
    - Main topics discussed
    - Entities mentioned (people, cities, bills, agencies)
    - Actions or issues that should be followed up on

    Conversation:
    {context}

    Return a concise bullet list.
    """

    return ask_llm(prompt)
