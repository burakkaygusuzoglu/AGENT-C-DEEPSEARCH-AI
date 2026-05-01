# Deep Research Agent

Deep Research Agent is a CLI research assistant built with LangGraph, LangChain, Chroma, and Tavily. It can answer questions from the local corpus, the live web, or both, then synthesize the retrieved context into a short research brief.

## What it does

- Routes each query to `rag_only`, `web_only`, `both`, or `off_topic`.
- Searches a local knowledge base stored in `corpus/` with Chroma.
- Pulls live results from Tavily when the question needs current information.
- Combines both sources into a concise final answer.

## Requirements

- Python 3.10+
- A GitHub token with `models:read` access
- A Tavily API key

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GITHUB_TOKEN=your_github_personal_access_token
TAVILY_API_KEY=your_tavily_api_key
```

If you do not have keys yet:

- GitHub token: https://github.com/settings/tokens
- Tavily key: https://tavily.com

## Build the corpus index

Run ingestion once before using the agent:

```bash
python main.py --ingest
```

This creates the local Chroma database in `.chroma_db/`.

## Run the agent

Interactive mode:

```bash
python main.py
```

Single query:

```bash
python main.py "What is retrieval-augmented generation?"
```

Query that uses both corpus and web search:

```bash
python main.py "What is LangGraph and what changed recently?"
```

## How it works

1. `src/agent.py` builds a LangGraph state machine.
2. The router decides whether the query should use the corpus, the web, or both.
3. `src/rag_pipeline.py` loads and retrieves from the Chroma vector store.
4. `src/web_search.py` queries Tavily.
5. The synthesizer merges the retrieved evidence into the final brief.

## Project structure

```text
deep-research-agent/
├── main.py
├── requirements.txt
├── AI_USAGE.md
├── corpus/
├── src/
│   ├── agent.py
│   ├── rag_pipeline.py
│   └── web_search.py
└── .chroma_db/          # generated after ingestion
```

## Adding documents

Drop additional `.txt` files into `corpus/`, then rerun:

```bash
python main.py --ingest
```

## Notes

- `.chroma_db/` is generated locally and should not be committed.
- `logs/` is reserved for runtime output if you add logging later.
- See `AI_USAGE.md` for the project’s AI usage disclosure.
