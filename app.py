"""
Career & Technology Intelligence Agent — Streamlit Web Interface
================================================================
Entry point: streamlit run app.py
The CLI (python main.py) remains fully independent and unaffected.
"""

import os
import re
import sys
from pathlib import Path

# Make src/ importable when Streamlit runs from the project root
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Career & Technology Intelligence Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Constants ─────────────────────────────────────────────────────────────────
EXAMPLES: list[str] = [
    "I have Python, FastAPI, SQL, YOLO, OpenCV, LangChain, Docker. Analyze my skill gaps for AI Engineer roles and create a 30-day plan.",
    "I know Python, SQL, Airflow, Spark, Docker, dbt. What gaps do I have for Data Engineer roles?",
    "I have AutoCAD, MATLAB, Python, and thermodynamics knowledge. What gaps do I have to become a mechanical engineer?",
    "What AI internships are currently open at major European tech companies?",
    "check the recent updates, how can i be a machine engineer with the newest informations",
]

ROLES: dict[str, str] = {
    "🤖 AI Engineer":          "LLMs · RAG · agentic AI · PyTorch · deployment",
    "📊 Data Engineer":        "SQL · Spark · Airflow · ETL · cloud pipelines",
    "💻 Software Engineer":    "APIs · frontend · testing · algorithms · DevOps",
    "⚙️ Mechanical Engineer":  "CAD · simulation · materials · digital twin · MATLAB",
    "🧠 ML Engineer":          "PyTorch · training · MLflow · model serving · cloud",
}

INTENT_LABELS: dict[str, str] = {
    "career_plan":      "🗺️ Career Plan",
    "job_research":     "💼 Job Research",
    "skill_gap":        "📊 Skill Gap",
    "interview_prep":   "🎤 Interview Prep",
    "tech_trend":       "📈 Tech Trend",
    "general_research": "🔬 General Research",
    "off_topic":        "⛔ Off-Topic",
}

ROUTE_LABELS: dict[str, str] = {
    "rag_only":  "📚 Corpus Only",
    "web_only":  "🌐 Web Only",
    "both":      "📚🌐 Corpus + Web",
    "off_topic": "⛔ Rejected",
}


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_graph():
    """Build and cache the LangGraph compiled graph (created once per session)."""
    from src.agent import build_graph
    return build_graph()


@st.cache_data(ttl=60, show_spinner=False)
def load_corpus_stats() -> dict:
    """Fetch ChromaDB stats; cached for 60 s so the sidebar stays snappy."""
    try:
        from src.rag_pipeline import corpus_stats
        return corpus_stats()
    except Exception:
        return {}


# ── Helper functions ──────────────────────────────────────────────────────────
def parse_match_score(answer: str) -> dict | None:
    """
    Extract match score fields from the Markdown final answer using regex.
    Returns None if no Match Score section is present (e.g. off-topic, general research).
    """
    score_m = re.search(r"Overall Match Score:\s*(\d+)%", answer)
    if not score_m:
        return None

    role_m   = re.search(r"\*\*Target Role:\*\*\s*([^\n]+)",          answer)
    skills_m = re.search(r"Skills Match\s*\|\s*(\d+)%",               answer)
    exp_m    = re.search(r"Experience Breadth\s*\|\s*(\d+)%",         answer)
    demand_m = re.search(r"Market Demand\s*\|\s*([^\|\n]+)",           answer)

    return {
        "score":      int(score_m.group(1)),
        "role":       role_m.group(1).strip()   if role_m   else "Target Role",
        "skills":     int(skills_m.group(1))    if skills_m else None,
        "experience": int(exp_m.group(1))       if exp_m    else None,
        "demand":     demand_m.group(1).strip() if demand_m else None,
    }


def score_badge(score: int) -> str:
    """Return a colour-coded emoji badge for a numeric match score."""
    if score >= 70:
        return "🟢"
    if score >= 50:
        return "🟡"
    return "🔴"


def invoke_agent(query: str) -> dict:
    """Run the LangGraph agent and return the final state dict."""
    graph = load_graph()
    return graph.invoke({
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
    })


# ── Session state initialisation ──────────────────────────────────────────────
for _key, _default in [
    ("result", None),
    ("error",  None),
    ("ran_query", ""),
    ("last_example", ""),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🚀 Career Agent")
    st.caption("Powered by Deep Research Agent Engine")

    st.divider()

    # ── Supported roles
    st.subheader("Supported Roles")
    for role, desc in ROLES.items():
        st.markdown(f"**{role}**")
        st.caption(desc)

    st.divider()

    # ── Knowledge base stats
    st.subheader("📂 Knowledge Base")
    stats = load_corpus_stats()
    if stats.get("total_chunks"):
        col_a, col_b = st.columns(2)
        col_a.metric("Chunks",    stats["total_chunks"])
        col_b.metric("Web Saved", stats.get("web_chunks", 0))
        st.caption(f"Corpus: {stats.get('corpus_chunks', 0)} chunks from 25+ docs")
    else:
        st.warning(
            "Corpus not indexed.  \nRun once in the terminal:  \n"
            "`python main.py --ingest`"
        )

    st.divider()

    # ── API key status
    st.subheader("⚙️ Configuration")
    gh_ok  = bool(os.getenv("GITHUB_TOKEN"))
    tav_ok = bool(os.getenv("TAVILY_API_KEY"))
    st.markdown(f"{'✅' if gh_ok  else '❌'} GitHub Token (LLM + Embeddings)")
    st.markdown(f"{'✅' if tav_ok else '❌'} Tavily API Key (Web Search)")
    if not gh_ok:
        st.error("GITHUB_TOKEN missing — add it to your `.env` file.")
    elif not tav_ok:
        st.warning("TAVILY_API_KEY missing — web search will be unavailable.")

    st.divider()
    st.caption(
        "💡 **Tip:** Include your skills in the query for a personalised Match Score.\n\n"
        "e.g. *I have Python, Docker, SQL…*"
    )


# ── Main — header ─────────────────────────────────────────────────────────────
st.title("🚀 Career & Technology Intelligence Agent")
st.markdown(
    "*Agentic AI system powered by* **LangGraph · RAG · Web Search · Match Scoring**"
)

if not os.getenv("GITHUB_TOKEN"):
    st.error(
        "**GITHUB_TOKEN is not configured.** The agent cannot call the LLM or embed documents. "
        "Create a `.env` file at the project root with your GitHub token and Tavily API key, "
        "then restart the app."
    )
    st.stop()

st.divider()


# ── Example query selector ────────────────────────────────────────────────────
st.subheader("📋 Example Queries")
st.caption("Select an example to pre-fill the query box, or type your own below.")

example_options = ["— type your own question below —"] + EXAMPLES
chosen_example  = st.selectbox(
    label="example",
    options=example_options,
    label_visibility="collapsed",
    key="example_selector",
)

# When a new example is selected, store it so the text area picks it up
if chosen_example != "— type your own question below —":
    if chosen_example != st.session_state.last_example:
        st.session_state.last_example = chosen_example
        # Clear previous results when loading a new example
        st.session_state.result    = None
        st.session_state.error     = None
        st.session_state.ran_query = ""

query_default = (
    st.session_state.last_example
    if chosen_example != "— type your own question below —"
    else ""
)

st.divider()


# ── Query input ───────────────────────────────────────────────────────────────
st.subheader("🔎 Your Research Question")

query_input = st.text_area(
    label="query",
    label_visibility="collapsed",
    value=query_default,
    height=120,
    placeholder=(
        "Describe your background and what you want to know. "
        "Include your skills for a personalised Match Score.\n\n"
        "e.g.  I have Python, SQL, Docker, LangChain. "
        "What skill gaps do I have for AI Engineer roles?"
    ),
)

btn_col_run, btn_col_clear, btn_col_pad = st.columns([1, 1, 6])
run_clicked   = btn_col_run.button(   "▶ Run Agent", type="primary", use_container_width=True)
clear_clicked = btn_col_clear.button( "✖ Clear",                     use_container_width=True)

if clear_clicked:
    st.session_state.result       = None
    st.session_state.error        = None
    st.session_state.ran_query    = ""
    st.session_state.last_example = ""
    st.rerun()


# ── Agent invocation ──────────────────────────────────────────────────────────
if run_clicked:
    q = query_input.strip()
    if not q:
        st.warning("⚠️ Please enter a research question before running.")
    else:
        st.session_state.result    = None
        st.session_state.error     = None
        st.session_state.ran_query = q

        with st.spinner("🤖 Agent is routing, retrieving, and synthesizing — please wait…"):
            try:
                st.session_state.result = invoke_agent(q)
            except Exception as exc:
                st.session_state.error = str(exc)


# ── Error display ─────────────────────────────────────────────────────────────
if st.session_state.error:
    st.error(f"**Agent error:** {st.session_state.error}")
    with st.expander("🔧 Troubleshooting steps"):
        st.markdown("""
1. Verify your `.env` file contains valid `GITHUB_TOKEN` and `TAVILY_API_KEY`.
2. Run `python main.py --ingest` to build the vector store if you haven't already.
3. Check your internet connection — web-search queries require external access.
4. Try a simpler RAG-only query first:
   *"Explain the STAR method for behavioral interviews."*
5. Check the terminal where Streamlit is running for the full traceback.
        """)


# ── Results ───────────────────────────────────────────────────────────────────
result = st.session_state.result

if result:
    st.divider()
    ran_q = st.session_state.ran_query
    st.success(
        f"✅ Research complete for: *{ran_q[:90]}{'…' if len(ran_q) > 90 else ''}*"
    )

    # ── Agent decision row ────────────────────────────────────────────────────
    st.subheader("🧭 Agent Decision")
    dec_c1, dec_c2, dec_c3 = st.columns([1, 1, 3])

    intent_raw = result.get("intent", "")
    route_raw  = result.get("route",  "")
    reason     = result.get("route_reason", "")

    dec_c1.metric(
        "Intent",
        INTENT_LABELS.get(intent_raw, intent_raw.replace("_", " ").title() or "—"),
    )
    dec_c2.metric(
        "Route",
        ROUTE_LABELS.get(route_raw, route_raw.replace("_", " ").title() or "—"),
    )
    if reason:
        dec_c3.markdown(f"**Routing reason:** {reason}")

    # ── Match Score card ──────────────────────────────────────────────────────
    final_answer = result.get("final_answer", "")
    parsed       = parse_match_score(final_answer)

    if parsed:
        st.divider()
        role_label = parsed.get("role", "Target Role")
        st.subheader(f"🎯 Match Score — {role_label}")

        mc1, mc2, mc3, mc4 = st.columns(4)

        overall = parsed["score"]
        mc1.metric(
            label="Overall Match",
            value=f"{overall}% {score_badge(overall)}",
        )
        if parsed.get("skills") is not None:
            mc2.metric("Skills Match",   f"{parsed['skills']}%")
        if parsed.get("experience") is not None:
            mc3.metric("Experience",     f"{parsed['experience']}%")
        if parsed.get("demand"):
            mc4.metric("Market Demand",  parsed["demand"])

        # Progress bar — visual at-a-glance score indicator
        st.progress(overall / 100)

        if overall >= 70:
            st.success("Strong profile for this role — focus on the quick wins listed in the brief.")
        elif overall >= 50:
            st.warning("Moderate alignment — key gaps exist. Check the Missing Skills section.")
        else:
            st.error("Early-stage alignment — prioritise the Missing Skills and Quick Wins below.")

    # ── Research brief ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Research Brief")
    st.markdown(final_answer)

    # ── Web sources expander ──────────────────────────────────────────────────
    sources = result.get("sources") or []
    if sources:
        st.divider()
        with st.expander(f"🔗 Web Sources ({len(sources)} found)", expanded=False):
            for i, src in enumerate(sources, 1):
                title = src.get("title", "Untitled")
                url   = src.get("url",   "")
                if url:
                    st.markdown(f"**[{i}]** [{title}]({url})")
                else:
                    st.markdown(f"**[{i}]** {title}")

    # ── Run-again nudge ───────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "💡 Refine your query and press **Run Agent** again, "
        "or load a new example from the dropdown above."
    )
