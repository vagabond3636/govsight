import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

# Load clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

st.set_page_config(page_title="GovSight AI", layout="centered")
st.title("🧠 GovSight AI")
st.markdown("_Your municipal intelligence assistant_")

# Session history
if "chat" not in st.session_state:
    st.session_state.chat = []

# Input
query = st.text_input("Ask a question about federal grants, bills, or local news:")

if query:
    st.session_state.chat.append(("You", query))

    # Embed query
    embedded = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    ).data[0].embedding

    # Query Pinecone
    results = index.query(vector=embedded, top_k=5, include_metadata=True)["matches"]
    context = "\n---\n".join([
        f"Title: {m['metadata'].get('title', '')}\nSummary: {m['metadata'].get('summary', '')}" for m in results
    ])

    # Ask GPT
    prompt = f"""
You are helping Mike Lane support municipalities. Based on the context below, answer clearly and briefly.

{context}

Question: {query}
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=600
    )
    answer = response.choices[0].message.content.strip()
    st.session_state.chat.append(("AI", answer))

# Show history
for speaker, msg in st.session_state.chat:
    st.markdown(f"**{speaker}:** {msg}")