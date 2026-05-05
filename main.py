"""
Deep Research Agent: Career & Technology Intelligence Agent
============================================================
Usage:
  python main.py                          # interactive mode
  python main.py "What is RAG?"           # single query mode
  python main.py --ingest                 # rebuild vector store
"""

import sys
import os
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Make sure src/ is importable from root
sys.path.insert(0, str(Path(__file__).parent))

BANNER = """
╔══════════════════════════════════════════════════════════╗
║    🚀  Career & Technology Intelligence Agent  🚀        ║
║    Powered by Deep Research Agent Engine                 ║
║    LangGraph · RAG · Web Search · Career Intelligence   ║
╚══════════════════════════════════════════════════════════╝
Research AI/tech careers, skill gaps, trends & interview prep.
Type your question (or 'quit' to exit)
"""


def run_agent(query: str) -> str:
    from src.agent import build_graph
    from src.logger import save_query_log

    graph = build_graph()
    initial_state = {
        "query":           query,
        "route":           "",
        "intent":          "",
        "route_reason":    "",
        "rag_result":      "",
        "web_result":      "",
        "sources":         [],
        "profile_summary": "",
        "skill_gap":       "",
        "recommendations": [],
        "final_answer":    "",
    }

    print(f"\n{'='*55}")
    print(f"❓ Query: {query}")
    print(f"{'='*55}")

    t0 = time.monotonic()
    try:
        result = graph.invoke(initial_state)
        duration_ms = int((time.monotonic() - t0) * 1000)
        save_query_log(
            query=query,
            route=result.get("route", "unknown"),
            rag_result=result.get("rag_result", ""),
            web_result=result.get("web_result", ""),
            final_answer=result.get("final_answer", ""),
            duration_ms=duration_ms,
            success=True,
        )
    except Exception as exc:
        duration_ms = int((time.monotonic() - t0) * 1000)
        save_query_log(
            query=query,
            route="error",
            duration_ms=duration_ms,
            success=False,
            error=str(exc),
        )
        raise

    print(f"\n{'='*55}")
    print("📋 RESEARCH BRIEF")
    print(f"{'='*55}")
    print(result["final_answer"])
    print(f"{'='*55}\n")

    return result["final_answer"]


def main():
    parser = argparse.ArgumentParser(description="Deep Research Agent")
    parser.add_argument("query", nargs="?", help="Research question")
    parser.add_argument("--ingest", action="store_true", help="Rebuild vector store")
    args = parser.parse_args()

    # Rebuild vector store
    if args.ingest:
        from src.rag_pipeline import ingest
        ingest(force=True)
        return

    # Single query mode
    if args.query:
        run_agent(args.query)
        return

    # Interactive mode
    print(BANNER)
    while True:
        try:
            query = input("🔎 Research: ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye! 👋")
                break
            run_agent(query)
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break


if __name__ == "__main__":
    main()