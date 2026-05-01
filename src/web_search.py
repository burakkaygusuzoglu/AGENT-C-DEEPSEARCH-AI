"""
Web Search Tool
================
Uses Tavily Search API — free tier gives 1,000 searches/month.
Get your key at: https://tavily.com
"""

import os


def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the web using Tavily and return formatted results.
    Falls back to a helpful message if API key is missing.
    """
    api_key = os.getenv("TAVILY_API_KEY", "")

    if not api_key or api_key == "your_tavily_key_here":
        return (
            "[Web search unavailable — add TAVILY_API_KEY to your .env file]\n"
            "Get a free key at https://tavily.com"
        )

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)

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
            content = r.get("content", "")[:400]  # truncate
            formatted.append(f"[Result {i}] {title}\n{url}\n{content}")

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        return f"Web search error: {e}"


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = search_web("latest developments in LLM agents 2025")
    print(result)
