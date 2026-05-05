# GUI Plan — Career & Technology Intelligence Agent

> Future Streamlit web interface for the Career & Technology Intelligence Agent.
> **Status: Planned — not yet implemented.**
> The core agent (LangGraph + RAG + web search + scoring) runs fully via CLI today.

---

## Why Streamlit

Streamlit is the natural choice because:
- The backend is already Python — no API layer required for a local app.
- `graph.stream()` can feed Streamlit's `st.write_stream()` for real-time output.
- The structured `ResearchState` dict maps directly to display cards.
- Zero frontend framework knowledge needed; deployable to Streamlit Community Cloud for free.

---

## Planned Layout

```
┌─────────────────────────────────────────────────────────────┐
│  🚀 Career & Technology Intelligence Agent                  │
│  Powered by Deep Research Agent Engine                      │
├──────────────────────┬──────────────────────────────────────┤
│                      │                                      │
│  QUERY PANEL         │  RESULTS PANEL                       │
│  ─────────────       │  ──────────────────                  │
│  [Text area]         │  📊 Route & Intent Card              │
│                      │  ┌──────────────────────────────┐    │
│  [Run Agent ▶]       │  │ Intent: skill_gap             │    │
│                      │  │ Route:  both                  │    │
│  [Ingest Corpus 🔄]  │  │ Role:   AI Engineer           │    │
│                      │  │ Reason: ...                   │    │
│  ─────────────       │  └──────────────────────────────┘    │
│  CORPUS STATS        │                                      │
│  Total chunks: 183   │  🎯 Match Score Card                 │
│  Corpus:  153        │  ┌──────────────────────────────┐    │
│  Web:      30        │  │ Score: 62% 🟡                 │    │
│                      │  │ Skills Match:    55%          │    │
│                      │  │ Experience:      75%          │    │
│                      │  │ Market Demand:   High 🔥      │    │
│                      │  │ Strengths: python, sql, cv    │    │
│                      │  │ Missing:  agentic ai, deploy  │    │
│                      │  └──────────────────────────────┘    │
│                      │                                      │
│                      │  📋 Research Brief                   │
│                      │  (Streamed Markdown output)          │
│                      │  # Career & Technology Intel...      │
│                      │  ## 1. Executive Summary ...         │
│                      │  ## 2. Internal Knowledge ...        │
│                      │  ...                                 │
│                      │                                      │
│                      │  🔗 Web Sources                      │
│                      │  [1] Title → URL                     │
│                      │  [2] Title → URL                     │
└──────────────────────┴──────────────────────────────────────┘
```

---

## Component Breakdown

### 1. Query Panel (left sidebar or top section)

```python
query = st.text_area(
    "Research Question",
    placeholder="e.g. I have Python, SQL, Docker. What gaps do I have for AI Engineer roles?",
    height=120,
)
run_button    = st.button("▶ Run Agent")
ingest_button = st.button("🔄 Rebuild Corpus Index")
```

### 2. Route & Intent Card

Displayed immediately after the router node fires (before retrieval starts).
Use `graph.stream()` to capture intermediate state.

```python
st.metric("Intent",      state["intent"])
st.metric("Route",       state["route"])
st.metric("Target Role", detected_role)
st.caption(state.get("route_reason", ""))
```

### 3. Match Score Card

Displayed after the synthesizer node completes scoring.
Parse the match score from `state["final_answer"]` or pass it via a side-channel.

```python
col1, col2, col3 = st.columns(3)
col1.metric("Match Score",    f"{score['match_score']}%")
col2.metric("Skills Match",   f"{score['skills_match']}%")
col3.metric("Market Demand",  score["market_demand"].title())

st.progress(score["match_score"] / 100)

with st.expander("Strengths & Gaps"):
    st.write("**Strengths:**", ", ".join(score["strong_skills"]))
    st.write("**Missing:**",   ", ".join(score["missing_skills"]))
    for qw in score["quick_wins"]:
        st.info(qw)
```

### 4. Research Brief Panel

Stream the final answer as Markdown.

```python
st.markdown(state["final_answer"])
```

For real-time streaming:
```python
with st.empty():
    for chunk in graph.stream(initial_state):
        if "final_answer" in chunk.get("synthesizer", {}):
            st.markdown(chunk["synthesizer"]["final_answer"])
```

### 5. Web Sources Panel

```python
if state.get("sources"):
    st.subheader("🔗 Web Sources")
    for i, s in enumerate(state["sources"], 1):
        st.markdown(f"**[{i}]** [{s['title']}]({s['url']})")
```

### 6. Corpus Stats Panel (sidebar)

```python
from src.rag_pipeline import corpus_stats
stats = corpus_stats()
st.sidebar.metric("Total Chunks",  stats["total_chunks"])
st.sidebar.metric("Corpus",        stats["corpus_chunks"])
st.sidebar.metric("Web Learned",   stats["web_chunks"])
```

---

## File to Create

`app.py` in the project root:

```python
import streamlit as st
from src.agent import build_graph
from src.rag_pipeline import ingest, corpus_stats

st.set_page_config(
    page_title="Career & Technology Intelligence Agent",
    page_icon="🚀",
    layout="wide",
)
# ... component code above
```

Run with:
```bash
streamlit run app.py
```

---

## Deployment Options

| Platform | Cost | Notes |
|---|---|---|
| Streamlit Community Cloud | Free | Requires public GitHub repo; secrets via Streamlit UI |
| Hugging Face Spaces | Free | Streamlit SDK supported; good for demo sharing |
| Railway / Render | Free tier | Persistent disk for `.chroma_db/`; better for production |
| Local only | Free | Best for academic submission demo |

---

## Implementation Order (when ready)

1. Create `app.py` with basic query input + run button
2. Wire `build_graph().invoke()` to display `final_answer`
3. Add route/intent card using intermediate state from `graph.stream()`
4. Extract and display Match Score card from score dict
5. Add sidebar corpus stats via `corpus_stats()`
6. Add web sources panel
7. Style with `st.set_page_config` and custom CSS
8. Deploy to Streamlit Community Cloud

**Estimated effort:** 4-6 hours for a functional MVP once the agent is stable.
