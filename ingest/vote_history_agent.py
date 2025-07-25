import os, requests, time, json
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# === Setup ===
load_dotenv()
api_key = os.getenv("CONGRESS_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

print("🔑 API Key prefix:", api_key[:5] if api_key else "None (Check .env)")

headers = {"X-API-Key": api_key}
client = OpenAI(api_key=openai_key)
pc = Pinecone(api_key=pinecone_key)
index = pc.Index("gov-index")

# === Config ===
CONGRESSES = [118, 117]
CHAMBERS = ["house", "senate"]
SESSIONS = [1, 2]
MAX_CONSECUTIVE_MISSES = 10

# === Utilities ===
def get_vote(congress, chamber, session, roll_call):
    url = f"https://api.congress.gov/v3/vote/{congress}/{chamber}/{session}/{roll_call}"
    res = requests.get(url, headers=headers, timeout=10)
    if res.status_code == 200:
        return res.json().get("vote")
    elif res.status_code == 404:
        return None
    else:
        print(f"❌ API error ({res.status_code}): {url}")
        return "error"

def embed_text(text):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    ).data[0].embedding

# === Main ===
processed = []

for congress in CONGRESSES:
    for chamber in CHAMBERS:
        for session in SESSIONS:
            print(f"\n📜 {chamber.title()}, Congress {congress}, Session {session}")
            misses = 0
            roll_call = 1
            while misses < MAX_CONSECUTIVE_MISSES:
                vote = get_vote(congress, chamber, session, roll_call)
                if vote == "error":
                    misses += 1
                    roll_call += 1
                    continue
                elif vote is None:
                    print(f"🔎 No vote found at roll call {roll_call}")
                    misses += 1
                    roll_call += 1
                    continue

                misses = 0  # reset if success
                desc = vote.get("description", "")
                result = vote.get("result", "")
                vote_type = vote.get("voteType", "")
                question = vote.get("question", "")
                positions = vote.get("positions", [])

                vote_id = f"vote-{congress}-{chamber}-{session}-{roll_call}"

                print(f"✅ {vote_id}: {desc[:60]}...")

                full_text = f"{desc}\nQuestion: {question}\nResult: {result}\nType: {vote_type}\n\nVotes:\n"
                full_text += "\n".join([
                    f"{p['member']['fullName']} ({p['member']['party']} - {p['member']['state']}): {p['votePosition']}"
                    for p in positions
                ])

                try:
                    embedding = embed_text(full_text)
                    index.upsert([{
                        "id": vote_id,
                        "values": embedding,
                        "metadata": {
                            "source": "vote",
                            "congress": congress,
                            "chamber": chamber,
                            "session": session,
                            "rollCall": roll_call,
                            "description": desc[:200],
                            "result": result,
                            "type": vote_type,
                            "member_count": len(positions)
                        }
                    }])
                    processed.append(vote_id)
                except Exception as e:
                    print(f"❌ Embedding error at {vote_id}: {e}")

                roll_call += 1
                time.sleep(0.5)

# === Save Log ===
with open("vote_ingestion_log.json", "w") as f:
    json.dump({"processed": processed}, f, indent=2)

print("\n🎉 Finished ingesting vote history.")
