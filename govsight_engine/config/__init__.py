import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    def __init__(self):
        self.db_path = os.getenv("DB_PATH", "govsight/data/memory.db")
        self.log_dir = os.getenv("LOG_DIR", "govsight/logs")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
        self.pinecone_env = os.getenv("PINECONE_ENV", "")
        self.pinecone_index = os.getenv("PINECONE_INDEX", "")

def load_settings(profile: str = "dev") -> Settings:
    return Settings()

settings = load_settings()
