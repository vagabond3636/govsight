import requests
from bs4 import BeautifulSoup
from govsight.llm import chat_completion


def query_web_and_summarize(query: str, max_results: int = 5) -> str:
    """
    Perform a Bing search, fetch and scrape top N pages, summarize contents.
    """
    search_url = "https://www.bing.com/search"
    params = {"q": query, "count": max_results}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(search_url, params=params, headers=headers)
        response.raise_for_status()
    except Exception as e:
        return f"[Web Error] Bing search failed: {e}"

    soup = BeautifulSoup(response.text, "html.parser")
    links = [a['href'] for a in soup.select('li.b_algo h2 a') if a['href'].startswith('http')]

    summaries = []
    for link in links[:max_results]:
        try:
            page = requests.get(link, headers=headers, timeout=5)
            page.raise_for_status()
            page_soup = BeautifulSoup(page.text, "html.parser")
            text = page_soup.get_text(separator=' ', strip=True)
            summaries.append(text[:2000])  # truncate long pages for input
        except Exception:
            continue

    if not summaries:
        return "[Web] No valid pages found to summarize."

    merged_content = "\n---\n".join(summaries)
    prompt = f"""
You are a research assistant. Read the content below gathered from search results.
Summarize any useful facts that answer the question: "{query}"

Content:
{merged_content}
"""

    summary = chat_completion(prompt)
    return summary or "[Web] No useful answer generated."
