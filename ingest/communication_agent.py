import os
import time
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from scrape_record_playwright import get_record_text_from_communication_page

# === Load API keys ===
load_dotenv()
api_key = os.getenv("CONGRESS_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

headers = {"X-API-Key": api_key}
client = OpenAI(api_key=openai_key)
pc = Pinecone(api_key=pinecone_key)
index = pc.Index("gov-index")

# === Settings ===
CONGRESSES = [118, 117, 116, 115]
ENDPOINTS = {
    "house": "https://api.congress.gov/v3/house-communication/",
    "senate": "https://api.congress.gov/v3/senate-communication/"
}
PER_PAGE = 100
SLEEP = 1

# === Embedding helper ===
def embed_text(text):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    ).data[0].embedding

# === Build URL to comm page ===
def build_comm_url(congress, chamber, comm_type_code, comm_id):
    return f"https://www.congress.gov/{chamber}-communication/{congress}th-congress/{comm_type_code.lower()}/{comm_id}"

# === Main loop ===
for chamber, base_url in ENDPOINTS.items():
    for congress in CONGRESSES:
        print(f"\n📬 Fetching {chamber.title()} communications for Congress {congress}")
        offset = 0
        total_embedded = 0

        while True:
            url = f"{base_url}{congress}?offset={offset}&limit={PER_PAGE}"
            res = requests.get(url, headers=headers, timeout=15)

            if res.status_code != 200:
                print(f"❌ Error {res.status_code} at {url}: {res.text[:200]}")
                break

            key = f"{chamber}Communications"
            records = res.json().get(key, [])
            print(f"📦 Fetched {len(records)} records from offset {offset}")

            if not records:
                break

            for rec in records:
                comm_id = rec.get("number")
                comm_type_obj = rec.get("communicationType", {})
                comm_type = comm_type_obj.get("name", "Unknown")
                comm_type_code = comm_type_obj.get("code", "XX")
                abstract = rec.get("abstract")
                update_date = rec.get("updateDate", "Unknown")
                originator = rec.get("originator", "Unknown")
                comm_url = build_comm_url(congress, chamber, comm_type_code, comm_id)
                vector_id = f"comm-{chamber}-{congress}-{comm_id}"

                # === Skip if already embedded ===
                existing = index.fetch(ids=[vector_id])
                if vector_id in existing.vectors:
                    print(f"⏭️ Skipping already embedded: {vector_id}")
                    continue

                # === Scrape full EC text if needed ===
                if not abstract or abstract.lower() == "no description":
                    print(f"🔍 Scraping full record for EC-{comm_id}...")
                    abstract = get_record_text_from_communication_page(comm_url, comm_id)

                if abstract.startswith("❌"):
                    print(f"⚠️ Skipping embedding for {vector_id} due to missing or invalid text.\n")
                    continue

                text = f"""
{chamber.title()} Communication to Congress
Congress: {congress}
Type: {comm_type}
Date: {update_date}
Originator: {originator}
Abstract: {abstract}
URL: {comm_url}
"""

                # Show preview in console
                print("\n🧾 === COMMUNICATION PREVIEW ===")
                print(f"📌 ID: {vector_id}")
                print(f"🏛 Chamber: {chamber.title()} | Congress: {congress}")
                print(f"📅 Date: {update_date}")
                print(f"📤 Originator: {originator}")
                print(f"📎 Type: {comm_type}")
                print(f"🔗 URL: {comm_url}")
                print("📄 Scraped Text Preview:\n")
                print(text[:1000] + "\n")
                print("🧠 Embedding and sending to Pinecone...\n")

                # === Upsert to Pinecone ===
                try:
                    embedding = embed_text(text)
                    index.upsert([{
                        "id": vector_id,
                        "values": embedding,
                        "metadata": {
                            "source": f"{chamber}-communication",
                            "chamber": chamber,
                            "congress": congress,
                            "type": comm_type,
                            "date": update_date,
                            "originator": str(originator)[:100],
                            "abstract": str(abstract)[:200],
                            "url": comm_url
                        }
                    }])
                    print(f"✅ Embedded and saved: {vector_id}\n")
                    total_embedded += 1
                except Exception as e:
                    print(f"❌ Embed error for {comm_id}: {e}")

            offset += PER_PAGE
            time.sleep(SLEEP)

        print(f"🧾 Total {chamber} communications embedded for Congress {congress}: {total_embedded}")
