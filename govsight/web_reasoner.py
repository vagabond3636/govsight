import requests
from bs4 import BeautifulSoup
from govsight.llm.openai_wrapper import summarize_web_content


def query_web_and_summarize(query: str) -> str:
    """
    Performs a web search using Bing and summarizes the most relevant result.
    """
    try:
        print("[web_reasoner] Querying the web for:", query)
        # Placeholder for actual search engine logic. You can replace this with SerpAPI, Bing API, etc.
        search_url = f"https://www.bing.com/search?q={query}"
        response = requests.get(search_url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find and extract the most relevant result (first snippet text)
        first_result = soup.find('li', {'class': 'b_algo'})
        if first_result:
            snippet = first_result.find('p')
            if snippet:
                raw_text = snippet.get_text()
                return summarize_web_content(raw_text)

        return "No relevant information found on the web."

    except Exception as e:
        return f"[web_reasoner] Error during web query: {e}"
