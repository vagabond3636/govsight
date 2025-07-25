import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load your API key
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "gov-index"

# Delete the existing index (if it exists)
if index_name in [index.name for index in pc.list_indexes()]:
    print(f"🔁 Deleting existing index: {index_name}")
    pc.delete_index(index_name)

# Recreate with correct dimension for OpenAI 'text-embedding-3-small' (1536)
print(f"✅ Creating new index: {index_name}")
pc.create_index(
    name=index_name,
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Match to your Pinecone region
)

print("🎉 Pinecone index recreated with 1536 dimensions.")