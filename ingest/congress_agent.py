import os
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from datetime import datetime

# Load API keys
load_dotenv()
congress_key = os.getenv("CONGRESS_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("gov-index")

base_url = "https://api.congress.gov/v3"
headers = {"X-API-Key": congress_key}

# Get latest 10 bills from current Congress
congress_number = 118
bill_list_url = f"{base_url}/bill/{congress_number}?limit=10"

print("📡 Fetching recent bills from Congress.gov...")
res = requests.get(bill_list_url, headers=headers)

if res.status_code != 200:
    print(f"❌ Failed to fetch bills: {res.status_code} - {res.text}")
    exit()

# Step 1: get bill detail URLs
bill_refs = res.json().get("bills", [])
if not bill_refs:
    print("❌ No bills found.")
    exit()

# Step 2: loop through and fetch details
for ref in bill_refs:
    try:
        detail_url = ref.get("url")
        if not detail_url:
            continue

        detail_res = requests.get(detail_url, headers=headers)
        if detail_res.status_code != 200:
            print(f"⚠️ Could not fetch bill detail: {detail_url}")
            continue

        data = detail_res.json()
        bill = data.get("bill", {})

        bill_id = bill.get("billNumber", "unknown")
        bill_type = bill.get("billType", "unknown")
        title = bill.get("title", "No Title")
        sponsor = bill.get("sponsor", {}).get("fullName", "Unknown")
        latest_action = bill.get("latestAction", {}).get("text", "")
        congress = bill.get("congress", str(congress_number))

        # Get summary
        summary_url = f"{base_url}/bill/{congress}/{bill_type}/{bill_id}/summary"
        summary = "No summary available."
        summary_res = requests.get(summary_url, headers=headers)
        if summary_res.status_code == 200:
            summary_data = summary_res.json()
            summaries = summary_data.get("summaries", [])
            if summaries:
                summary = summaries[0].get("text", summary)

        # Combine text for embedding
        content = f"{title}\n\nSponsor: {sponsor}\n\nLatest Action: {latest_action}\n\nSummary:\n{summary}"

        print(f"🧾 Ingesting: {title}")
        preview = content[:200].replace('\n', ' ')
        print(f"📄 Preview: {preview}...\n")

        embed = client.embeddings.create(
            model="text-embedding-3-small",
            input=content
        )
        embedding = embed.data[0].embedding

        index.upsert(vectors=[{
            "id": f"{congress}_{bill_type}_{bill_id}",
            "values": embedding,
            "metadata": {
                "source": "congress.gov",
                "type": "bill",
                "entity": sponsor,
                "title": title,
                "summary": summary[:1000],
                "bill_id": bill_id,
                "bill_type": bill_type,
                "congress": str(congress),
                "date": datetime.utcnow().isoformat()
            }
        }])

        print("✅ Stored in Pinecone.\n")

    except Exception as e:
        print(f"❌ Error while processing bill: {e}")
