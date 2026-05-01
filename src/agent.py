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
  synthesizer   ← merges findings into a final research brief
     │
  [END]
"""

from __future__ import annotations

import logging
import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

_log = logging.getLogger("deep_research")

# ── LLM setup (GitHub Models) ────────────────────────────────────────────────
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
    """Shared state that flows through every node in the graph."""
    query: str                          # Original user question
    route: str                          # Router decision
    rag_result: str                     # Output from RAG node
    web_result: str                     # Output from web search node
    final_answer: str                   # Synthesized research brief


# ── Node 1: Router ────────────────────────────────────────────────────────────
def router_node(state: ResearchState) -> ResearchState:
    """
    Decides how to answer the query:
      - 'rag_only'   → topic is well covered by our document corpus
      - 'web_only'   → topic needs live/recent information only
      - 'both'       → needs corpus context AND live web data
      - 'off_topic'  → completely unrelated, refuse gracefully
    """
    llm = get_llm()

    system = """You are a routing agent for a Deep Research system.
Given a user query, decide the best retrieval strategy.

Reply with EXACTLY one of these four words (nothing else):
  rag_only   → the query is about AI/ML concepts, papers, or foundational knowledge in our corpus
  web_only   → the query needs recent news, live data, or current events
  both       → the query needs both corpus knowledge AND recent web information
  off_topic  → the query is unrelated to AI/technology research

Examples:
  "What is RAG?" → rag_only
  "Who won the Super Bowl today?" → web_only
  "What is GPT-4 and what new features were released this week?" → both
  "Give me a pizza recipe" → off_topic
"""

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=state["query"]),
    ])

    route = response.content.strip().lower()
    if route not in ("rag_only", "web_only", "both", "off_topic"):
        route = "both"  # safe default

    _log.info("Router decision: %s | query=%r", route, state["query"])
    print(f"\n🔀 Router decision: {route}")
    return {**state, "route": route}


# ── Conditional Edge: route_decision ─────────────────────────────────────────
def route_decision(state: ResearchState) -> Literal["rag", "web", "both", "off_topic"]:
    """Maps router output to the next node name."""
    mapping = {
        "rag_only":  "rag",
        "web_only":  "web",
        "both":      "both",
        "off_topic": "off_topic",
    }
    return mapping.get(state["route"], "both")


# ── Node 2a: RAG ──────────────────────────────────────────────────────────────
def rag_node(state: ResearchState) -> ResearchState:
    """Retrieves relevant context from the vector store."""
    from src.rag_pipeline import retrieve

    print("📚 RAG node: searching corpus...")
    _log.info("RAG node triggered | query=%r", state["query"])
    context = retrieve(state["query"])
    return {**state, "rag_result": context}


# ── Node 2b: Web Search ───────────────────────────────────────────────────────
def web_node(state: ResearchState) -> ResearchState:
    """Fetches live results from the web using Tavily."""
    from src.web_search import search_web

    print("🌐 Web node: searching internet...")
    _log.info("Web node triggered | query=%r", state["query"])
    results = search_web(state["query"])
    return {**state, "web_result": results}


# ── Node 2c: Both (RAG + Web) ─────────────────────────────────────────────────
def both_node(state: ResearchState) -> ResearchState:
    """Runs RAG and web search sequentially, populates both fields."""
    from src.rag_pipeline import retrieve
    from src.web_search import search_web

    print("📚🌐 Both node: corpus + web search...")
    _log.info("Both node triggered | query=%r", state["query"])
    rag_context  = retrieve(state["query"])
    web_context  = search_web(state["query"])
    return {**state, "rag_result": rag_context, "web_result": web_context}


# ── Node 2d: Off-topic ────────────────────────────────────────────────────────
def off_topic_node(state: ResearchState) -> ResearchState:
    msg = "⚠️  This query is outside the scope of the Deep Research Agent (AI/Tech domain)."
    return {**state, "final_answer": msg}


# ── Node 3: Synthesizer ───────────────────────────────────────────────────────
def synthesizer_node(state: ResearchState) -> ResearchState:
    """
    Merges RAG context and/or web results into a structured research brief.
    """
    llm = get_llm()

    corpus_section = f"## Corpus Knowledge\n{state['rag_result']}" if state.get("rag_result") else ""
    web_section    = f"## Live Web Results\n{state['web_result']}"  if state.get("web_result")  else ""

    system = """You are a senior research analyst.
Using the retrieved information below, write a clear, structured research brief.

Format your response as:
  **Summary** (2-3 sentences)
  **Key Findings** (bullet points)
  **Sources Used** (corpus / web / both)

Be concise. Cite where each piece of information came from.
If a section has no data, skip it.
"""

    user_content = f"""Research question: {state['query']}

{corpus_section}

{web_section}
"""

    print("✍️  Synthesizer: generating research brief...")
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_content),
    ])

    return {**state, "final_answer": response.content}


# ── Build the Graph ───────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(ResearchState)

    # Add nodes
    g.add_node("router",      router_node)
    g.add_node("rag",         rag_node)
    g.add_node("web",         web_node)
    g.add_node("both",        both_node)
    g.add_node("off_topic",   off_topic_node)
    g.add_node("synthesizer", synthesizer_node)

    # Entry point
    g.add_edge(START, "router")

    # Conditional routing from router → retrieval nodes
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

    # Retrieval nodes → synthesizer
    g.add_edge("rag",  "synthesizer")
    g.add_edge("web",  "synthesizer")
    g.add_edge("both", "synthesizer")

    # Terminal nodes → END
    g.add_edge("synthesizer", END)
    g.add_edge("off_topic",   END)

    return g.compile()


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    graph = build_graph()
    print("✅ LangGraph compiled successfully!")
    print(f"   Nodes: {list(graph.nodes)}")
