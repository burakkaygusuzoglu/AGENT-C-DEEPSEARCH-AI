# AI Usage Disclosure

This file documents where and how AI assistance was used in this project,
as required by the course academic integrity policy.

## Tools Used

- **Claude (Anthropic)** — Used for project scaffolding, code structure suggestions, debugging help, and evolving the agent from a generic research tool into a specialized Career & Technology Intelligence Agent. All code was reviewed, understood, and modified by the author.
- **GitHub Copilot** — Used for inline autocomplete during development.

## What AI Helped With

| Area | AI Role | Human Role |
|---|---|---|
| Project architecture | Suggested LangGraph node structure | Designed final graph, chose routing logic |
| `src/agent.py` | Provided initial StateGraph pattern | Customized nodes, state schema, routing, intent classification |
| `src/rag_pipeline.py` | Explained Chroma + LangChain patterns | Integrated, tested, tuned chunk size and deduplication |
| `src/web_search.py` | Suggested Tavily API usage | Integrated, handled errors, structured source parsing |
| Career corpus documents | Drafted content summaries | Reviewed, curated, extended with domain knowledge |
| Prompts (system messages) | Drafts provided | All prompts refined through testing and iteration |
| README & DEMO_QUERIES.md | Structure suggested | Content written and validated by author |
| Intent routing logic | JSON-based routing approach suggested | Validated correct intent→route mappings, tested edge cases |

## What Was NOT AI-Generated

- Final routing logic and conditional edge design decisions
- System prompt refinements after testing
- Choice of career specialization direction and document topics
- All debugging and integration work
- Testing and validation of the full end-to-end pipeline
- Architecture diagram design

## Statement

Every line of code in this repository can be explained by the author.
AI tools were used to accelerate development, not to replace understanding.
The career specialization direction, corpus topic selection, and routing design
were driven by the author's own domain knowledge and decisions.
