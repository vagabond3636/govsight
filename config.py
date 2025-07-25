import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
CONGRESS_API_KEY = os.getenv("CONGRESS_API_KEY")
PINECONE_INDEX_NAME = "gov-index"
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
DEFAULT_OPENAI_MODEL = os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_EMBED_MODEL  = os.getenv("DEFAULT_EMBED_MODEL", "text-embedding-3-small")