import os
import json
import datetime
from difflib import get_close_matches
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# === Load environment variables ===
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

client = OpenAI(api_key=openai_key)
pc = Pinecone(api_key=pinecone_key)
index = pc.Index("gov-index")

conversation_log = []
current_bill_meta = None

# === Pull all bills ===
def fetch_congress_bills():
    response = index.query(
        vector=[0.0]*1536,
        top_k=1000,
        include_metadata=True,
        filter={"source": "congress.gov"}
    )
    return response["matches"]

# === Try title match ===
def get_title_matches(user_input, matches):
    titles = {}
    for match in matches:
        meta = match["metadata"]
        bill_id = match["id"]
        all_titles = [meta.get("title", ""), meta.get("shortTitle", "")]
        for title in all_titles:
            if title:
                titles[title.lower()] = (bill_id, meta)

    user_input_lower = user_input.lower()
    close = get_close_matches(user_input_lower, titles.keys(), n=1, cutoff=0.6)
    return titles, close

# === Embed and search if title fails ===
def fallback_semantic_match(user_input):
    embedded = client.embeddings.create(
        model="text-embedding-3-small",
        input=user_input
    ).data[0].embedding

    results = index.query(
        vector=embedded,
        top_k=5,
        include_metadata=True,
        filter={"source": "congress.gov"}
    )
    return results["matches"]

# === GPT summary of individual bill ===
def summarize_bill_with_gpt(meta):
    global conversation_log

    prompt = f"""
You are an AI policy assistant for Mike Lane, a federal lobbyist focused on housing, infrastructure, and appropriations.

Mike prefers responses that are:
- Strategic and concise
- Relevant to local governments and funding
- Geared toward city managers and congressional staffers

Summarize the following bill for Mike. Include:
- What it does
- Who sponsored it
- Key provisions
- Current status

Bill title: {meta.get('title', 'N/A')}

Summary:
{meta.get('summary', 'No summary available.')}

Sponsor: {meta.get('sponsor', meta.get('entity', 'N/A'))}
Cosponsors: {', '.join(meta.get('cosponsors', [])) or 'None'}
Latest action: {meta.get('latestAction', 'Unknown')}

Answer:
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=600
    )
    answer = response.choices[0].message.content.strip()
    conversation_log.append({"role": "system", "content": answer})
    return answer

# === Follow-up Q&A ===
def ask_follow_up(question):
    global conversation_log, current_bill_meta
    messages = [
        {"role": "system", "content": f"Bill title: {current_bill_meta.get('title', '')}\nSummary: {current_bill_meta.get('summary', '')}\nSponsor: {current_bill_meta.get('sponsor', '')}\nLatest Action: {current_bill_meta.get('latestAction', '')}"}
    ] + conversation_log + [{"role": "user", "content": question}]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.4,
        max_tokens=600
    )
    answer = response.choices[0].message.content.strip()
    conversation_log.append({"role": "user", "content": question})
    conversation_log.append({"role": "system", "content": answer})
    return answer

# === Save chat ===
def save_session():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bill_summary_{timestamp}.txt"
    with open(filename, "w") as f:
        f.write(f"Bill: {current_bill_meta.get('title', '')}\n\n")
        for exchange in conversation_log:
            role = exchange['role']
            content = exchange['content']
            f.write(f"[{role.upper()}]\n{content}\n\n")
    print(f"💾 Conversation saved to {filename}")

# === Smart trend analysis by year/topic ===
def handle_general_query(question):
    print("📊 Analyzing tagged bills by topic and year...")

    response = index.query(
        vector=[0.0] * 1536,
        top_k=1000,
        include_metadata=True,
        filter={"source": "congress.gov"}
    )

    matches = response["matches"]
    metadata_list = [match["metadata"] for match in matches if "topics" in match["metadata"]]

    # Build structured topic frequency by year
    year_topic_map = {}

    for meta in metadata_list:
        congress = meta.get("congress", "")
        topics = meta.get("topics", [])
        if not topics:
            continue

        # Infer year
        if congress == "116":
            year = "2022"
        elif congress == "117":
            year = "2023"
        elif congress == "118":
            year = "2024"
        else:
            year = "unknown"

        for topic in topics:
            key = (year, topic.lower().strip())
            year_topic_map[key] = year_topic_map.get(key, 0) + 1

    # Build a display string for GPT
    formatted = {}
    for (year, topic), count in year_topic_map.items():
        if year not in formatted:
            formatted[year] = {}
        formatted[year][topic] = formatted[year].get(topic, 0) + count

    trend_summary = ""
    for year in sorted(formatted.keys()):
        trend_summary += f"\n📅 {year}\n"
        for topic, count in sorted(formatted[year].items(), key=lambda x: -x[1])[:10]:
            trend_summary += f"  - {topic}: {count}\n"

    # Prompt GPT for strategic analysis
    prompt = f"""
You are a strategic legislative intelligence analyst working for Mike Lane, a federal lobbyist representing cities and counties.

The following data shows the most frequent bill topics from Congress in each year:

{trend_summary}

Please analyze the data with the following goals:
- Identify which topic categories are growing or shrinking across years
- Group related topics (e.g., stormwater + flooding = resilience)
- Highlight what this suggests about congressional priorities
- Suggest how cities or counties should respond or prepare
- Use bullet points if helpful

Now answer the user's question:

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=700
    )
    return response.choices[0].message.content.strip()

# === Main entry point ===
def main():
    global current_bill_meta, conversation_log

    while True:
        user_input = input("\nAsk about a bill title or type /query, /exit: ")
        if user_input.strip().lower() in ["exit", "/exit"]:
            break

        if user_input.strip().lower().startswith("/query"):
            question = user_input.replace("/query", "").strip()
            print("\n🧠 Querying all indexed bills...\n")
            result = handle_general_query(question)
            print("\n🔎 Insight:\n")
            print(result)
            continue

        matches = fetch_congress_bills()
        titles, close = get_title_matches(user_input, matches)

        if close:
            bill_id, bill_meta = titles[close[0]]
        else:
            print("⚠️ No close title match found. Trying semantic search...\n")
            sem_results = fallback_semantic_match(user_input)
            if not sem_results:
                print("❌ No results found.")
                continue
            bill_meta = sem_results[0]["metadata"]

        current_bill_meta = bill_meta
        conversation_log = []

        print("\n🤖 Generating summary...\n")
        print(summarize_bill_with_gpt(bill_meta))

        while True:
            follow = input("\n💬 Ask a follow-up (or /save, /new, /exit): ")
            if follow.lower() in ["/exit", "exit"]:
                return
            if follow.lower() == "/new":
                break
            if follow.lower() == "/save":
                save_session()
                continue
            print("\n🧠 Response:")
            print(ask_follow_up(follow))

# === Run the tool ===
if __name__ == "__main__":
    main()
