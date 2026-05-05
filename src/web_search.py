"""
Web Search Tool
================
Uses Tavily Search API.
Returns both formatted content AND structured source list.
"""

import os


def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the web using Tavily.
    Returns formatted string with content + URLs embedded.
    """
    api_key = os.getenv("TAVILY_API_KEY", "")

    if not api_key or api_key == "your_tavily_api_key":
        return (
            "[Web search unavailable — add TAVILY_API_KEY to your .env file]\n"
            "Get a free key at https://tavily.com"
        )

    try:
        from tavily import TavilyClient
        client   = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
        )

        results = response.get("results", [])
        if not results:
            return "No web results found."

        formatted = []
        for i, r in enumerate(results, 1):
            title   = r.get("title", "No title")
            url     = r.get("url", "")
            content = r.get("content", "")[:500]
            # URL embedded in content block so learn_from_web also saves it
            formatted.append(f"[Result {i}] {title}\nURL: {url}\n{content}")

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        return f"Web search error: {e}"


def get_sources(web_result: str) -> list[dict]:
    """
    Parse URL and title from the formatted web_result string.
    Returns list of { title, url } dicts for display.
    """
    sources = []
    if not web_result or "[Web search unavailable" in web_result:
        return sources

    blocks = web_result.split("\n\n---\n\n")
    for block in blocks:
        lines = block.strip().splitlines()
        title = ""
        url   = ""
        for line in lines:
            if line.startswith("[Result") and "]" in line:
                title = line.split("]", 1)[-1].strip()
            elif line.startswith("URL:"):
                url = line.replace("URL:", "").strip()
        if url:
            sources.append({"title": title, "url": url})

    return sources


if __name__ == "__main__":
    result = search_web("latest LangGraph features 2025")
    print(result)
    print("\n--- SOURCES ---")
    for s in get_sources(result):
        print(f"  {s['title']} → {s['url']}")