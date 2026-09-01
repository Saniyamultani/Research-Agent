# AI/ML Research Agent Infrastructure

## Run it
```bash
pip install -r requirements.txt

# optional, for real LLM-backed decomposition/claim extraction/synthesis:
pip install anthropic   # or: pip install openai
export ANTHROPIC_API_KEY=sk-...   # or OPENAI_API_KEY=sk-...

# optional, for a stronger web search provider (DuckDuckGo via `ddgs` works with no key):
export TAVILY_API_KEY=tvly-...

uvicorn server:app --reload --port 8000
```

### API keys
Fill in .env with your own keys. Supported configuration slots include Gemini, Groq, Hugging Face, OpenRouter, OpenAI, Anthropic, and Tavily. Never share or commit .env.

Open http://localhost:8000

A SQLite file `research_agent.db` is created next to `server.py` on first run —
that's the agent's persistent memory. Delete it any time to start fresh.

## What's new: database + web search
- **Persistence (`db.py`)** — every run (query, decomposition, claims,
  contradictions, rendered report, metrics) is saved to SQLite, and every
  source the agent ever retrieves is cached in a `sources` table. This
  survives page reloads *and* server restarts — Recent Research, the
  Literature Library, Query Decomposer, Contradiction Engine, and
  Evaluation Metrics views all read straight from the database via REST
  endpoints (`/api/runs`, `/api/library`, `/api/decompositions`,
  `/api/contradictions`, `/api/stats`).
- **Knowledge-base-first retrieval** — before going out to arXiv/the web for
  a sub-question, the agent searches its own cached sources first
  (`agent_pipeline.search_knowledge_base`). If there's already enough
  relevant material, it reuses it (near-instant, no network call); only a
  thin/empty result triggers a live search. This means related follow-up
  questions and repeated topics get faster and progressively richer over
  time instead of re-fetching the same papers.
- **Real web search (`agent_pipeline.fetch_web`)** — retrieval is no longer
  arXiv-only. It also searches the general web (Tavily if `TAVILY_API_KEY`
  is set, otherwise DuckDuckGo via `ddgs`, no key required), tagging results
  as Primary Research / Official Doc / Blog. If neither is installed/
  reachable, it degrades gracefully to arXiv + cached knowledge only.

## Notes
- Works with zero API keys and zero network access out of the box: the LLM
  client (`agent_pipeline.LLMClient`) falls back to a deterministic heuristic
  mode, and retrieval falls back to a small bundled offline arXiv-abstract
  corpus if `arxiv`/`ddgs`/network are unavailable.
- Retrieval ranking is hybrid BM25 (rank-bm25) + TF-IDF cosine similarity as
  a dense-embedding stand-in (`agent_pipeline.VectorIndex`) — swap in
  sentence-transformers + Chroma/FAISS/Qdrant for production by replacing
  that class; the `.search()` interface stays the same.
- SQLite was chosen for zero-setup persistence. For a multi-user production
  deployment, swap the connection helper in `db.py` (`_connect`) for
  Postgres — every other function only talks to that one function.
- The graph is built with LangGraph when installed (`ResearchPipeline._build_graph`);
  the runtime path used by the server calls each stage directly so it can
  stream fine-grained SSE events to the UI as they happen.
- Frontend listens for `step`, `log`, `telemetry`, `source`, `decomposition`,
  `contradiction`, `report`, `metrics`, and `done` SSE events — see `app.js`.
