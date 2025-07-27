from __future__ import annotations

"""
GovSight Embedding Utility
Embeds text blocks and upserts into Pinecone
"""

from govsight.utils.pinecone_init import get_pinecone_index
from govsight.llm.openai_wrapper import get_embedding


def embed_to_pinecone(text: str, namespace: str = "default", metadata: dict = {}) -> None:
    index = get_pinecone_index()
    vector = get_embedding(text)

    index.upsert(
        vectors=[{
            "id": metadata.get("id", f"auto-{hash(text)}"),
            "values": vector,
            "metadata": metadata
        }],
        namespace=namespace
    )
