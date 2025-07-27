from govsight.llm.openai_wrapper import get_embedding
from govsight.db.core import upsert_embedding
import os

def embed_and_upsert(text: str, source: str, doc_type: str = "text"):
    vector = get_embedding(text)
    upsert_embedding(text=text, vector=vector, source=source, doc_type=doc_type)
    print(f"✅ Embedded and upserted: [{source}]")

# Example usage
if __name__ == "__main__":
    sample = "Grandview, TX has an estimated population of 1,800 people."
    embed_and_upsert(sample, source="test_input.txt")
