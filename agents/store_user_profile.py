import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import hashlib

# Load API keys
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

# Initialize clients
client = OpenAI(api_key=openai_key)
pc = Pinecone(api_key=pinecone_key)
index = pc.Index("gov-index")

# 🧠 Your profile data
profile_text = """
My name is Mike Lane. I am a federal lobbyist focused on helping small and mid-sized cities, counties, and special districts secure funding and federal support from congress or federal agencies.

My core interests include housing, appropriations, disaster relief, rural development, and community resilience. I specialize in understanding congressional activity, grant programs, and agency funding opportunities.

I prefer clear, strategic, direct summaries that speak to city managers, mayors, and congressional staff.

I care most about helping communities in Texas, Florida, Arizona, and California — especially with projects related to infrastructure, emergency response, and economic recovery.
"""

# 🧠 Generate embedding
def generate_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Unique ID for this profile
vector_id = "user_profile_mike"
embedding = generate_embedding(profile_text)

# Metadata
metadata = {
    "source": "user-profile",
    "name": "Mike Lane",
    "role": "federal lobbyist",
    "topics": ["housing", "appropriations", "disaster relief", "rural development", "funding", "advocacy", "infrastructure"],
    "regions": ["TX", "FL", "AZ", "CA"],
    "style": "direct, strategic, clear",
    "audience": "mayors, city managers, congressional staff"
}

# Store in Pinecone
index.upsert([
    {
        "id": vector_id,
        "values": embedding,
        "metadata": metadata
    }
])

print(f"✅ Profile vector '{vector_id}' stored successfully.")
