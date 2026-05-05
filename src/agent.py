"""
Career & Technology Intelligence Agent — State & Graph
=======================================================
Powered by the Deep Research Agent engine (LangGraph + RAG + Web Search).

LangGraph state machine:

  [START]
     │
     ▼
  router        ← classifies intent + decides route: rag | web | both | off_topic
     │
  ┌──┼──────────────┐
  ▼  ▼              ▼
 rag  web        off_topic
  └──┬──────────────┘
     ▼
  synthesizer   ← merges findings into career/research brief + appends sources
     │
  [END]
"""

from __future__ import annotations

import json
import os
import re
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from src.logger import logger
from src.scoring import (
    ROLE_DISPLAY_NAMES,
    compute_match_score,
    detect_target_role,
    extract_skills_from_text,
    format_no_skills_section,
    format_score_section,
    get_target_skills,
)

load_dotenv()

CAREER_INTENTS = {"career_plan", "job_research", "skill_gap", "interview_prep", "tech_trend"}


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
    query:           str    # user question
    route:           str    # routing decision: rag_only | web_only | both | off_topic
    intent:          str    # query intent: career_plan | job_research | skill_gap | ...
    route_reason:    str    # one-sentence reason for the routing decision
    rag_result:      str    # corpus retrieval output
    web_result:      str    # web search output
    sources:         list   # [{title, url}] from web search
    profile_summary: str    # user's stated skills/background (if detected in query)
    skill_gap:       str    # identified skill gaps (populated by synthesizer context)
    recommendations: list   # concrete recommended actions (populated by synthesizer)
    final_answer:    str    # synthesized research/career brief


# ── Node 1: Router ────────────────────────────────────────────────────────────
def router_node(state: ResearchState) -> ResearchState:
    llm = get_llm()

    system = """You are the routing intelligence of a Career & Technology Intelligence Agent.
Analyze the user query and return a JSON object with exactly these fields:

{
  "intent": "<intent>",
  "route": "<route>",
  "reason": "<one sentence reason>",
  "profile_summary": "<any skills/background the user explicitly stated, or empty string>"
}

Intent must be exactly one of:
  career_plan      → user wants a learning roadmap, career path, or multi-week plan
  job_research     → user wants internship/job listings, company research, or hiring trends
  skill_gap        → user wants to compare their current skills vs a target role/position
  interview_prep   → user wants technical or behavioral interview preparation
  tech_trend       → user wants analysis of technology/AI/software industry trends
  general_research → general AI/tech/software research that doesn't fit above categories
  off_topic        → completely unrelated to AI, technology, software, or careers

Route must be exactly one of:
  rag_only   → internal corpus is sufficient (established concepts, interview basics, frameworks)
  web_only   → only live data needed (breaking news, live job listings, current events)
  both       → needs internal corpus knowledge AND live web data (most career + trend queries)
  off_topic  → reject query

Routing rules:
  career_plan    → "both" (roadmaps + current market reality)
  job_research   → "both" (corpus context + live listings/companies)
  skill_gap      → "both" (role requirements from corpus + current market from web)
  interview_prep → "rag_only" for fundamentals, "both" if asking about current company practices
  tech_trend     → "both" (corpus foundations + latest web developments)
  general_research → "rag_only" for definitions/concepts, "both" if recency matters
  off_topic      → "off_topic"
  When in doubt  → "both"
"""

    try:
        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=state["query"]),
        ])
        raw = response.content.strip()
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        data = json.loads(json_match.group()) if json_match else {}
    except Exception as exc:
        logger.warning("Router JSON parse failed: %s — falling back to defaults", exc)
        data = {}

    intent          = data.get("intent", "general_research")
    route           = data.get("route", "both")
    reason          = data.get("reason", "")
    profile_summary = data.get("profile_summary", "")

    valid_intents = {"career_plan", "job_research", "skill_gap", "interview_prep",
                     "tech_trend", "general_research", "off_topic"}
    valid_routes  = {"rag_only", "web_only", "both", "off_topic"}

    if intent not in valid_intents:
        intent = "general_research"
    if route not in valid_routes:
        route = "both"

    print(f"\n🎯 Intent:  {intent}")
    print(f"🔀 Route:   {route}")
    if reason:
        print(f"💬 Reason:  {reason}")
    logger.info("Router | intent=%-15s | route=%-10s | query=%r", intent, route, state["query"])

    return {
        **state,
        "route":           route,
        "intent":          intent,
        "route_reason":    reason,
        "profile_summary": profile_summary or state.get("profile_summary", ""),
    }


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
    logger.info("RAG node: query=%r", state["query"])
    context = retrieve(state["query"])
    return {**state, "rag_result": context}


# ── Node 2b: Web ──────────────────────────────────────────────────────────────
def web_node(state: ResearchState) -> ResearchState:
    from src.web_search import search_web, get_sources
    from src.rag_pipeline import learn_from_web

    print("🌐 Web node: searching internet...")
    logger.info("Web node: query=%r", state["query"])
    results = search_web(state["query"])
    sources = get_sources(results)
    learn_from_web(state["query"], results)

    return {**state, "web_result": results, "sources": sources}


# ── Node 2c: Both ─────────────────────────────────────────────────────────────
def both_node(state: ResearchState) -> ResearchState:
    from src.rag_pipeline import retrieve, learn_from_web
    from src.web_search import search_web, get_sources

    print("📚🌐 Both node: corpus + web search...")
    logger.info("Both node: query=%r", state["query"])
    rag_context = retrieve(state["query"])
    web_context = search_web(state["query"])
    sources     = get_sources(web_context)
    learn_from_web(state["query"], web_context)

    return {**state, "rag_result": rag_context, "web_result": web_context, "sources": sources}


# ── Node 2d: Off-topic ────────────────────────────────────────────────────────
def off_topic_node(state: ResearchState) -> ResearchState:
    msg = (
        "⚠️  This query is outside the scope of the Career & Technology Intelligence Agent.\n"
        "This agent specializes in AI/software careers, skill-gap analysis, technology trends, "
        "internship preparation, and related research.\n"
        "Please rephrase your question within the tech/career domain."
    )
    logger.info("Off-topic query rejected: %r", state["query"])
    return {**state, "final_answer": msg}


# ── Node 3: Synthesizer ───────────────────────────────────────────────────────
def synthesizer_node(state: ResearchState) -> ResearchState:
    llm       = get_llm()
    intent    = state.get("intent", "general_research")
    is_career = intent in CAREER_INTENTS

    corpus_section  = f"## Corpus Knowledge\n{state['rag_result']}" if state.get("rag_result") else ""
    web_section     = f"## Live Web Results\n{state['web_result']}"  if state.get("web_result")  else ""
    profile_note    = f"## User Profile / Stated Skills\n{state['profile_summary']}" if state.get("profile_summary") else ""
    sources_used    = (
        "both corpus and web" if corpus_section and web_section
        else ("corpus only" if corpus_section else "web only")
    )

    if is_career:
        system = """You are a senior career advisor and technology intelligence analyst.
Write a structured Career & Technology Intelligence Brief based on the retrieved information.

Use EXACTLY this Markdown format — do not skip or rename any section:

# Career & Technology Intelligence Brief

## 1. Executive Summary
2-3 sentences: what the user wants and your core conclusion.

## 2. Internal Knowledge Findings
Summarize what the internal corpus contributed. Label each insight with [Corpus].
If corpus was not used, write: "Internal corpus was not used for this query."

## 3. Live Web Findings
Summarize what live web search contributed. Label each point with [Web].
If web search was not used, write: "Not required — internal corpus was sufficient."

## 4. Skill Gap / Opportunity Analysis
Be specific about what the user needs to develop or what opportunity exists.
If the user provided their skills, map them directly against the target role requirements.
If no profile was provided, describe the typical gaps for someone targeting this area.

## 5. Recommended Action Plan
Give concrete, numbered steps. For career/roadmap queries include a 7-day quick-start
or a 30-day plan with weekly milestones. Mention specific platforms, tools, or resources.

## 6. Sources
Leave this section blank — web references are appended automatically below.

Rules: Be direct and specific. Never say "I don't have information." Use what you retrieved.
"""
    else:
        system = """You are a senior research analyst.
Write a clear, structured research brief using the retrieved information.

Format:
  **Summary**
  (2-3 sentences answering the question directly)

  **Key Findings**
  (bullets — bold topic, then explanation, labeled [Corpus] or [Web])

  **Sources Used**
  (Corpus only / Web only / Both corpus and web)

Be specific and informative. Do NOT hedge with "I don't have information."
"""

    user_content = f"""Research question: {state['query']}
Intent type: {intent}
Sources available: {sources_used}

{profile_note}

{corpus_section}

{web_section}
"""

    print("✍️  Synthesizer: generating research brief...")
    logger.info("Synthesizer | intent=%s | query=%r", intent, state["query"])
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_content),
    ])

    # ── Match Score injection ─────────────────────────────────────────────────
    # Active for: skill_gap, career_plan, job_research, tech_trend
    SCORE_INTENTS = {"skill_gap", "career_plan", "job_research", "tech_trend"}
    answer = response.content

    if intent in SCORE_INTENTS:
        query_text    = state.get("query", "")
        user_skills   = extract_skills_from_text(query_text)
        target_role   = detect_target_role(query_text)
        target_skills = get_target_skills(target_role)
        role_name     = ROLE_DISPLAY_NAMES.get(target_role, "AI Engineer")

        if user_skills:
            score       = compute_match_score(user_skills, target_skills)
            score_block = format_score_section(score, user_skills, role_name)
            print(f"📊 Match scoring applied  "
                  f"(role={role_name}, score={score['match_score']}%, "
                  f"skills={score['skills_match']}%, demand={score['market_demand']})")
            logger.info(
                "Match score | %d%% | role=%s | skills_match=%d%% | intent=%s",
                score["match_score"], target_role, score["skills_match"], intent,
            )
        else:
            score_block = format_no_skills_section(role_name, target_skills)
            print(f"📊 Match scoring applied  (role={role_name}, no user skills detected)")
            logger.info(
                "Match score | no-skills | role=%s | intent=%s", target_role, intent,
            )

        # Inject before "## 5. Recommended Action Plan" when the LLM used that marker
        marker = "## 5."
        if marker in answer:
            idx    = answer.index(marker)
            answer = answer[:idx] + score_block + answer[idx:]
        else:
            answer = answer + score_block

    # ── Web source citations ──────────────────────────────────────────────────
    sources     = state.get("sources", [])
    source_text = ""
    if sources:
        source_lines = "\n".join(
            f"  [{i+1}] {s['title']}\n      {s['url']}"
            for i, s in enumerate(sources)
        )
        source_text = f"\n\n📎 **Web References**\n{source_lines}"

    return {**state, "final_answer": answer + source_text}


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
