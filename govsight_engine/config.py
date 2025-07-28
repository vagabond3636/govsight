import os
from dotenv import load_dotenv
from types import SimpleNamespace

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

settings = SimpleNamespace(
    # API Keys
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
    PINECONE_API_KEY=os.getenv("PINECONE_API_KEY"),
    CONGRESS_API_KEY=os.getenv("CONGRESS_API_KEY"),
    SERPAPI_API_KEY=os.getenv("SERPAPI_API_KEY"),

    # OpenAI Models
    DEFAULT_OPENAI_MODEL=os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o"),
    DEFAULT_EMBED_MODEL=os.getenv("DEFAULT_EMBED_MODEL", "text-embedding-3-small"),

    # Pinecone Configs
    PINECONE_INDEX_NAME=os.getenv("PINECONE_INDEX_NAME", "gov-index"),
    PINECONE_ENV=os.getenv("PINECONE_ENV", "us-west4-gcp"),

    # SQLite DB Path
    db_path=os.path.abspath(os.path.join(BASE_DIR, "..", "data", "memory.db"))
)
