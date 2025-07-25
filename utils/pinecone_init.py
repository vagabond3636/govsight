import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load your API keys
load_dotenv()

# Pull credentials from .env
api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENV")  # This is not needed anymore with v3+

# Create Pinecone client
pc = Pinecone(api_key=api_key)

index_name = "gov-index"

# Create index if it doesn't exist
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to index
index = pc.Index(index_name)

print("✅ Pinecone index is ready and connected.")