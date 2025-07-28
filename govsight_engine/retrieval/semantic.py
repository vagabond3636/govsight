"""
Stub for Pinecone / vector lookup.
"""
from typing import Any, Optional, List, Dict

def semantic_search(
    query: str,
    metadata_filters: Dict[str, Any] = None,
    top_k: int = 3
) -> Optional[List[Dict[str, Any]]]:
    # TODO: embed & query Pinecone here
    return None
