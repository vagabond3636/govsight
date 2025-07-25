# govsight/vector/search.py

import os
from dotenv import load_dotenv
import openai
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Extract keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "govsight-index"  # Customize if needed

# Validate config
if not all([PINECONE_API_KEY, PINECONE_ENV, OPENAI_API_KEY]):
    raise ValueError("Missing required API keys or environment settings in .env")

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Embedding generator
def get_embedding(text: str, model="text-embedding-3-small") -> list[float]:
    try:
        response = openai.Embedding.create(
            input=[text],
            model=model
        )
        return response.data[0]["embedding"]
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return []

# Pinecone semantic search
def search_pinecone(query: str, top_k: int = 5) -> list[dict]:
    try:
        embedding = get_embedding(query)
        if not embedding:
            return []

        index = pc.Index(INDEX_NAME)
        result = index.query(vector=embedding, top_k=top_k, include_metadata=True)
        return result["matches"]
    except Exception as e:
        print(f"[Pinecone Search Error] {e}")
        return []
