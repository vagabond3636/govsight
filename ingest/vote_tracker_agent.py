import os
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# Load environment variables
load_dotenv()
congress_key = os.getenv("CONGRESS_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

# Initialize clients
client = OpenAI(api_key=openai_key)
pc = Pinecone(api_key=pinecone_key)
index = pc.Index("gov-index")

# Set headers for Congress.gov API
headers = {"X-API-Key": congress_key}

# Define parameters
CONGRESS = 118
CHAMBER = "house"  # Change to 'senate' for Senate votes
SESSION = 1
NUM_VOTES = 5  # Number of recent votes to fetch

# Fetch recent votes
for roll_call in range(1, NUM_VOTES + 1):
    url = f"https://api.congress.gov/v3/vote/{CONGRESS}/{CHAMBER}/{SESSION}/{roll_call}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"❌ Error fetching vote {roll_call}.")
        continue

    vote = response.json().get("vote", {})
    description = vote.get("description", "No description")
    result = vote.get("result", "No result")
    vote_type = vote.get("voteType", "No type")

    text = f"{description} | Result: {result} | Type: {vote_type}"

    # Generate embedding
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    ).data[0].embedding

    # Upsert into Pinecone
    index.upsert(vectors=[{
        "id": f"vote-{CONGRESS}-{CHAMBER}-{SESSION}-{roll_call}",
        "values": embedding,
        "metadata": {
            "source": "vote",
            "description": description,
            "result": result,
            "chamber": CHAMBER,
            "rollCall": roll_call,
            "congress": CONGRESS,
            "type": vote_type
        }
    }])

    print(f"✅ Embedded vote {roll_call}: {description}")
