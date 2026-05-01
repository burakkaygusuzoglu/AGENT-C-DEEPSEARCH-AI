"""
Deep Research Agent — State & Graph
====================================
LangGraph state machine with 4 nodes and conditional edges:

  [START]
     │
     ▼
  router        ← decides: rag_only | web_only | both | off_topic
     │
  ┌──┼──────────────┐
  ▼  ▼              ▼
 rag  web        off_topic
  └──┬──────────────┘
     ▼
  synthesizer   ← merges findings + appends source URLs
     │
  [END]
"""

from __future__ import annotations

import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from src.logger import logger

load_dotenv()


# ── LLM setup ────────────────────────────────────────────────────────────────
def get_llm() -> ChatOpenAI:
    token = os.getenv("GITHUB_TOKEN", "MISSING")
    return ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=token,
        openai_api_base="https://models.github.ai/inference",
        temperature=0,
    )


# ── Agent State ───────────────────────────────────────────────────────────────
class ResearchState(TypedDict):
    query:        str   # user question
    route:        str   # router decision
    rag_result:   str   # corpus retrieval output
    web_result:   str   # web search output
    sources:      list  # [{title, url}] from web search
    final_answer: str   # synthesized brief


# ── Node 1: Router ────────────────────────────────────────────────────────────
def router_node(state: ResearchState) -> ResearchState:
    llm = get_llm()

    system = """You are a routing agent for a Deep Research system.
Given a user query, decide the best retrieval strategy.

Reply with EXACTLY one of these four words — nothing else:
  rag_only   → query is about AI/ML fundamentals clearly in our knowledge base
  web_only   → query needs ONLY live breaking news with zero relevant prior knowledge
  both       → query needs corpus knowledge AND current web data (use this for most tech questions)
  off_topic  → completely unrelated to AI/technology

IMPORTANT RULES:
- When in doubt always choose "both" — never miss useful corpus knowledge
- "latest", "recent", "2025", "new" → always "both" (corpus may already have it)
- Pure definitional questions ("What is X?") → "rag_only"
- Pure real-time events ("stock price", "live score") → "web_only"

Examples:
  "What is RAG?" → rag_only
  "What is LangGraph and latest updates?" → both
  "Latest AI agent frameworks 2025?" → both
  "Bitcoin price right now?" → web_only
  "Give me a pasta recipe" → off_topic
"""

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=state["query"]),
    ])

    route = response.content.strip().lower()
    if route not in ("rag_only", "web_only", "both", "off_topic"):
        route = "both"

    print(f"\n🔀 Router decision: {route}")
    logger.info("Router decision: %s | query=%r", route, state["query"])
    return {**state, "route": route}


# ── Conditional Edge ──────────────────────────────────────────────────────────
def route_decision(state: ResearchState) -> Literal["rag", "web", "both", "off_topic"]:
    mapping = {
        "rag_only":  "rag",
        "web_only":  "web",
        "both":      "both",
        "off_topic": "off_topic",
    }
    return mapping.get(state["route"], "both")


# ── Node 2a: RAG ──────────────────────────────────────────────────────────────
def rag_node(state: ResearchState) -> ResearchState:
    from src.rag_pipeline import retrieve
    print("📚 RAG node: searching corpus...")
    logger.info("RAG node: retrieving corpus for query=%r", state["query"])
    context = retrieve(state["query"])
    return {**state, "rag_result": context}


# ── Node 2b: Web ──────────────────────────────────────────────────────────────
def web_node(state: ResearchState) -> ResearchState:
    from src.web_search import search_web, get_sources
    from src.rag_pipeline import learn_from_web

    print("🌐 Web node: searching internet...")
    logger.info("Web node: searching Tavily for query=%r", state["query"])
    results = search_web(state["query"])
    sources = get_sources(results)
    learn_from_web(state["query"], results)

    return {**state, "web_result": results, "sources": sources}


# ── Node 2c: Both ─────────────────────────────────────────────────────────────
def both_node(state: ResearchState) -> ResearchState:
    from src.rag_pipeline import retrieve, learn_from_web
    from src.web_search import search_web, get_sources

    print("📚🌐 Both node: corpus + web search...")
    logger.info("Both node: corpus + web for query=%r", state["query"])
    rag_context = retrieve(state["query"])
    web_context = search_web(state["query"])
    sources     = get_sources(web_context)
    learn_from_web(state["query"], web_context)

    return {**state, "rag_result": rag_context, "web_result": web_context, "sources": sources}


# ── Node 2d: Off-topic ────────────────────────────────────────────────────────
def off_topic_node(state: ResearchState) -> ResearchState:
    msg = "⚠️  This query is outside the scope of the Deep Research Agent (AI/Tech domain)."
    logger.info("Off-topic query rejected: %r", state["query"])
    return {**state, "final_answer": msg}


# ── Node 3: Synthesizer ───────────────────────────────────────────────────────
def synthesizer_node(state: ResearchState) -> ResearchState:
    llm = get_llm()

    corpus_section = f"## Corpus Knowledge\n{state['rag_result']}" if state.get("rag_result") else ""
    web_section    = f"## Live Web Results\n{state['web_result']}"  if state.get("web_result")  else ""

    system = """You are a senior research analyst.
Write a clear, structured research brief using the retrieved information.

Format EXACTLY as:
  **Summary**
  (2-3 clear sentences answering the question directly)

  **Key Findings**
  (bullet points — each starts with a bold topic, then explanation)
  (label each finding with [Corpus] or [Web] to show the source)

  **Sources Used**
  (write: Corpus only / Web only / Both corpus and web)

Be specific and informative. Do NOT say "I don't have information" — use what you have.
"""

    user_content = f"""Research question: {state['query']}

{corpus_section}

{web_section}
"""

    print("✍️  Synthesizer: generating research brief...")
    logger.info("Synthesizer: generating brief for query=%r", state["query"])
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_content),
    ])

    # Append web source URLs at the bottom
    sources     = state.get("sources", [])
    source_text = ""
    if sources:
        source_lines = "\n".join(
            f"  [{i+1}] {s['title']}\n      {s['url']}"
            for i, s in enumerate(sources)
        )
        source_text = f"\n\n📎 **Web References**\n{source_lines}"

    return {**state, "final_answer": response.content + source_text}


# ── Build Graph ───────────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(ResearchState)

    g.add_node("router",      router_node)
    g.add_node("rag",         rag_node)
    g.add_node("web",         web_node)
    g.add_node("both",        both_node)
    g.add_node("off_topic",   off_topic_node)
    g.add_node("synthesizer", synthesizer_node)

    g.add_edge(START, "router")

    g.add_conditional_edges(
        "router",
        route_decision,
        {
            "rag":       "rag",
            "web":       "web",
            "both":      "both",
            "off_topic": "off_topic",
        },
    )

    g.add_edge("rag",  "synthesizer")
    g.add_edge("web",  "synthesizer")
    g.add_edge("both", "synthesizer")
    g.add_edge("synthesizer", END)
    g.add_edge("off_topic",   END)

    return g.compile()


if __name__ == "__main__":
    graph = build_graph()
    print("✅ LangGraph compiled!")
    print(f"   Nodes: {list(graph.nodes)}")