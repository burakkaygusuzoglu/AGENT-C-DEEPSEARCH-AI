"""
Deep Research Agent — Main Entry Point
========================================
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

# Make sure src/ is importable from root
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

# Logger imported early so all subsequent modules share its handlers
from src.logger import logger, save_query_log  # noqa: E402

BANNER = """
╔══════════════════════════════════════════════════╗
║         🔬  Deep Research Agent  🔬              ║
║   LangGraph + LangChain + RAG + Web Search       ║
╚══════════════════════════════════════════════════╝
Type your research question (or 'quit' to exit)
"""


def run_agent(query: str) -> str:
    from src.agent import build_graph

    graph = build_graph()
    initial_state = {
        "query":        query,
        "route":        "",
        "rag_result":   "",
        "web_result":   "",
        "final_answer": "",
    }

    print(f"\n{'='*55}")
    print(f"❓ Query: {query}")
    print(f"{'='*55}")

    start     = time.time()
    error_msg = None
    result    = dict(initial_state)

    try:
        result = graph.invoke(initial_state)
    except Exception as e:
        error_msg = str(e)
        logger.error("Agent pipeline failed: %s", e, exc_info=True)
        result["final_answer"] = (
            "An error occurred while processing your query. "
            "Please try again or rephrase your question."
        )

    duration_ms = int((time.time() - start) * 1000)

    save_query_log(
        query=query,
        route=result.get("route", "unknown"),
        rag_result=result.get("rag_result", ""),
        web_result=result.get("web_result", ""),
        final_answer=result.get("final_answer", ""),
        duration_ms=duration_ms,
        success=error_msg is None,
        error=error_msg,
    )

    print(f"\n{'='*55}")
    print("📋 RESEARCH BRIEF")
    print(f"{'='*55}")
    print(result["final_answer"])
    print(f"{'='*55}")
    print(f"⏱  Completed in {duration_ms} ms\n")

    return result["final_answer"]


def main():
    parser = argparse.ArgumentParser(description="Deep Research Agent")
    parser.add_argument("query", nargs="?", help="Research question")
    parser.add_argument("--ingest", action="store_true", help="Rebuild vector store")
    args = parser.parse_args()

    # Rebuild vector store
    if args.ingest:
        from src.rag_pipeline import ingest
        try:
            ingest(force=True, verbose=True)
        except Exception as e:
            logger.error("Ingest failed: %s", e, exc_info=True)
            print(f"❌ Ingest failed: {e}")
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
