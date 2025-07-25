import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from govsight.llm.llm import chat_completion  # Make sure this function uses new OpenAI API

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def search_web(query, num_results=5):
    """
    Performs a Bing search for the given query and returns a list of URLs.
    """
    try:
        subscription_key = None  # Replace with actual key if available
        if not subscription_key:
            print("[Search Warning] No Bing API key provided. Returning example URL.")
            return [f"https://example.com/search?q={query.replace(' ', '+')}"]

        url = f"https://api.bing.microsoft.com/v7.0/search?q={query}&count={num_results}"
        headers = {"Ocp-Apim-Subscription-Key": subscription_key}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        results = response.json()
        return [item["url"] for item in results["webPages"]["value"]]
    except Exception as e:
        print(f"[Search Error] {e}")
        return []

def scrape_full_text(url):
    """
    Scrapes and extracts the full readable text content from a webpage.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts and style tags
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        paragraphs = soup.find_all(["p", "li"])
        visible_text = " ".join(p.get_text() for p in paragraphs)
        return visible_text.strip()
    except Exception as e:
        print(f"[Scrape Error: {url}] {e}")
        return ""

def summarize_web_results(query, texts):
    """
    Uses GPT to summarize extracted web text relevant to the query.
    """
    combined_text = "\n\n".join(texts)[:6000]  # Truncate input if necessary
    system_prompt = "You are a research assistant summarizing reliable information from web pages."
    user_prompt = f"Query: {query}\n\nExtracted Content:\n{combined_text}\n\nAnswer the query briefly and clearly based on the above."
    return chat_completion(system_prompt=system_prompt, user_prompt=user_prompt)

def query_web_and_summarize(query):
    """
    Main function to execute web search, scrape, and summarize steps.
    """
    print(f"[web_reasoner] Querying the web for: {query}")
    urls = search_web(query)
    if not urls:
        return "No search results found."

    contents = [scrape_full_text(url) for url in urls if url]
    if not any(contents):
        return "Could not extract usable content from web pages."

    return summarize_web_results(query, contents)
