# govsight/web_reasoner.py

import os
import logging
import requests
from govsight.llm.openai_wrapper import summarize_web_content
from govsight.config.settings import settings

logger = logging.getLogger(__name__)
SERPAPI_KEY = settings.serpapi_api_key


def serpapi_search(query: str, num_results: int = 5) -> list:
    logger.info(f"[web_reasoner] Querying SerpAPI for: {query}")
    print(f"[web_reasoner] 🔍 Searching with SerpAPI: '{query}'")

    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": num_results
    }

    response = requests.get("https://serpapi.com/search", params=params)
    if response.status_code != 200:
        logger.error(f"[web_reasoner] SerpAPI Error {response.status_code}: {response.text}")
        return []

    data = response.json()
    results = data.get("organic_results", [])

    processed = []
    for result in results[:num_results]:
        processed.append({
            "title": result.get("title"),
            "link": result.get("link"),
            "snippet": result.get("snippet", "")
        })

    print(f"[web_reasoner] ✅ Retrieved {len(processed)} results from SerpAPI")
    return processed


def fetch_full_text(link: str) -> str:
    print(f"[web_reasoner] 🌐 Fetching full content from: {link}")
    try:
        response = requests.get(link, timeout=5)
        if response.status_code == 200:
            return response.text
        else:
            logger.warning(f"[web_reasoner] Failed to fetch {link} - Status: {response.status_code}")
    except Exception as e:
        logger.warning(f"[web_reasoner] Exception fetching {link}: {e}")
    return ""


def web_search_and_summarize(query: str, context: str = None) -> str:
    print(f"[web_reasoner] 🚀 Starting full web search and summarization for: '{query}'")
    search_results = serpapi_search(query)

    if not search_results:
        print("[web_reasoner] ❌ No results found from SerpAPI.")
        return "❌ No results found from SerpAPI."

    summaries = []
    for idx, result in enumerate(search_results):
        url = result["link"]
        print(f"[web_reasoner] 📄 Processing result #{idx+1}: {url}")
        html_content = fetch_full_text(url)
        if html_content:
            full_query = f"{context}. {query}" if context else query
            summary = summarize_web_content(query=full_query, html=html_content, source_url=url)
            if summary:
                summaries.append((url, summary))
                print(f"[web_reasoner] ✅ Summary {idx+1} extracted")
            else:
                print(f"[web_reasoner] ⚠️ No summary extracted from {url}")
        else:
            print(f"[web_reasoner] ⚠️ No content retrieved from {url}")

    if summaries:
        best_summary = pick_best_summary(summaries, query)
        return best_summary
    else:
        print("[web_reasoner] ⚠️ No content could be summarized.")
        return "⚠️ Could not retrieve or summarize content from any of the top search results."


def pick_best_summary(summaries, query):
    # For now, return the first non-empty one. In the future we can use an LLM scoring function.
    print(f"[web_reasoner] 🧠 Picking best summary out of {len(summaries)}")
    for url, text in summaries:
        if query.lower().split()[0] in text.lower():  # crude relevance filter
            return f"From {url}:\n{text}"
    return summaries[0][1]  # fallback to first summary
