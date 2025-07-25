import os
import openai
from dotenv import load_dotenv
from pinecone import Pinecone

# Load your keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to index
index = pc.Index("gov-index")

# Example text
doc_id = "deltona-001"
text = "The Deltona City Commission met on April 21st to discuss public infrastructure, beautification, and the potential for applying to federal grant programs."

try:
    # Get embedding
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
except Exception as e:
    print("🚨 OpenAI Error:", e)
    exit()

# Upsert to Pinecone
try:
    index.upsert(vectors=[{
        "id": doc_id,
        "values": embedding,
        "metadata": {"source": "Deltona Meeting", "topic": "infrastructure"}
    }])
    print("✅ Text embedded and stored in Pinecone.")
except Exception as e:
    print("🚨 Pinecone Error:", e)