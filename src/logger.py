"""
Query Logger
============
Central logging for the Deep Research Agent.
- Writes structured JSON per query to logs/
- Writes a running agent.log file for debugging
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

LOGS_DIR = Path(__file__).parent.parent / "logs"
LOG_FILE  = Path(__file__).parent.parent / "agent.log"

# Named logger — handlers attached once, shared across all imports
logger = logging.getLogger("deep_research")

if not logger.handlers:
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False  # don't bubble to root logger


def save_query_log(
    *,
    query: str,
    route: str,
    rag_result: str = "",
    web_result: str = "",
    final_answer: str = "",
    duration_ms: int,
    success: bool,
    error: Optional[str] = None,
) -> Optional[Path]:
    """
    Persist one query record to logs/query-YYYY-MM-DD-HH-mm-ss.json.
    Returns the path written, or None if writing fails.
    """
    try:
        LOGS_DIR.mkdir(exist_ok=True)

        ts = datetime.utcnow()
        record = {
            "timestamp":    ts.isoformat(),
            "query":        query,
            "route":        route,
            "rag_result":   rag_result,
            "web_result":   web_result,
            "final_answer": final_answer,
            "duration_ms":  duration_ms,
            "success":      success,
            "error":        error,
        }

        filename = ts.strftime("query-%Y-%m-%d-%H-%M-%S.json")
        path = LOGS_DIR / filename
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info(
            "Query logged | route=%-10s | %5d ms | success=%s",
            route, duration_ms, success,
        )
        return path

    except Exception as e:
        logger.error("Failed to save query log: %s", e)
        return None