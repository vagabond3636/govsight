import os
from dotenv import load_dotenv

# Load .env file into environment
load_dotenv()

# Project root path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


class Settings:
    def __init__(self):
        self.db_path = os.path.join(ROOT_DIR, "data", "memory.db")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.serpapi_api_key = os.getenv("SERPAPI_API_KEY")
        self.pinecone_index = "gov-index"
        self.debug = True
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4")  # <-- Exposes this attribute


# Exported singleton
settings = Settings()
