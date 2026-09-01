# Research Agent

A lightweight AI research assistant for exploring a topic, decomposing a question, retrieving supporting sources, checking for contradictions, and presenting structured results in a browser-based dashboard.

## What this project does

This app lets you:
- submit a research question in natural language
- break it into sub-questions
- search cached sources and live web/arXiv results
- collect and rank source material
- extract claims and identify conflicting evidence
- generate a structured report with small inline citations
- review historical runs, source library, decomposition history, and evaluation metrics

The project is built with:
- FastAPI for the backend API and SSE stream
- a Python research pipeline in `agent_pipeline.py`
- SQLite persistence in `db.py`
- a static web interface in `index.html`, `app.js`, and `styles.css`

## Project structure

- `server.py` — FastAPI app and HTTP/SSE endpoints
- `agent_pipeline.py` — research workflow, retrieval, claim extraction, contradiction detection, and report synthesis
- `db.py` — SQLite database layer for source cache and saved runs
- `index.html` — dashboard shell
- `app.js` — frontend logic for the UI and SSE event handling
- `styles.css` — dashboard styling
- `requirements.txt` — Python dependencies
- `.env.example` — sample environment file for secrets
- `.gitignore` — ignores local secrets and generated files

## Quick start

1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create your local environment file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

4. Add your API keys in `.env`.

5. Start the app:

```bash
uvicorn server:app --reload --port 8000
```

Then open:

```text
http://localhost:8000
```

## Environment variables

The app reads optional keys from `.env` using `python-dotenv`.

Supported keys in this project include:

- `GEMINI_API_KEY`
- `GROQ_API_KEY`
- `HF_TOKEN`
- `OPENROUTER_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `TAVILY_API_KEY`

Only set the providers you plan to use. If no keys are configured, the app falls back to a deterministic heuristic mode and offline retrieval behavior.

## Retrieval and persistence

The project includes several useful behaviors:

- Local SQLite persistence via `research_agent.db`
- cached sources stored in the `sources` table
- saved research runs in the `runs` table
- knowledge-base-first retrieval before external search
- optional web search via Tavily or DuckDuckGo
- fallback behavior when dependencies or network access are unavailable

This means the app can run in a demo or local-only environment without a full production stack.

## API overview

Key endpoints exposed by the app:

- `GET /` — serves the dashboard
- `GET /api/health` — checks backend health
- `GET /api/runs` — recent research runs
- `GET /api/library` — cached sources
- `GET /api/decompositions` — decomposed queries
- `GET /api/contradictions` — contradiction records
- `GET /api/stats` — database summary
- `GET /api/research/stream` — SSE stream for live research progress

## Notes

- `.env` is intentionally ignored and should never be committed.
- `research_agent.db` and related SQLite WAL/SHM files are local state and should be ignored in Git.
- The project is designed for local use and experimentation. For multi-user production deployment, the persistence layer in `db.py` would need to be swapped for a server-backed database.

## License

This project does not currently declare a license file. If you plan to distribute or publish it, add a license before sharing publicly.
