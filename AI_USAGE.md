# AI Usage Disclosure

This file documents where and how AI assistance was used in this project,
as required by the course academic integrity policy.

## Tools Used

- **Claude (Anthropic)** — Used for initial project scaffolding, code structure suggestions, and debugging help. All code was reviewed, understood, and modified by the author.
- **GitHub Copilot** — Used for inline autocomplete during development.

## What AI Helped With

| Area | AI Role | Human Role |
|---|---|---|
| Project architecture | Suggested LangGraph node structure | Designed final graph, chose routing logic |
| `src/agent.py` | Provided initial StateGraph pattern | Customized nodes, state schema, routing |
| `src/rag_pipeline.py` | Explained Chroma + LangChain patterns | Integrated, tested, tuned chunk size |
| `src/web_search.py` | Suggested Tavily API usage | Integrated, handled errors |
| Corpus documents | Not used | All 15 documents written/curated manually |
| Prompts (system messages) | Drafts provided | All prompts refined through testing |

## What Was NOT AI-Generated

- Final routing logic and conditional edge design
- System prompt refinements after testing
- Corpus document selection and curation
- README architecture diagram
- All debugging and integration work

## Statement

Every line of code in this repository can be explained by the author.
AI tools were used to accelerate development, not to replace understanding.
