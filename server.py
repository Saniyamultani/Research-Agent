"""
server.py
=========
FastAPI application for the AI/ML Research Agent Infrastructure.

Endpoints
---------
GET  /                          -> serves the dashboard (index.html)
GET  /styles.css, /app.js       -> static assets
GET  /api/research/stream       -> Server-Sent Events stream of a research run
GET  /api/health                -> liveness check

Run
---
    pip install fastapi uvicorn sse-starlette langgraph rank-bm25 scikit-learn arxiv
    # optional, for real LLM calls (falls back to a heuristic mode without these):
    pip install anthropic openai
    export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY=...

    uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from sse_starlette.sse import EventSourceResponse
    HAS_SSE_STARLETTE = True
except ImportError:  # pragma: no cover
    HAS_SSE_STARLETTE = False
    from starlette.responses import StreamingResponse

import agent_pipeline as ap
from agent_pipeline import pipeline
import db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("research-agent")

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="AI/ML Research Agent Infrastructure", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Static frontend
# --------------------------------------------------------------------------

@app.get("/")
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html")


@app.get("/styles.css")
async def styles() -> FileResponse:
    return FileResponse(BASE_DIR / "styles.css", media_type="text/css")


@app.get("/app.js")
async def appjs() -> FileResponse:
    return FileResponse(BASE_DIR / "app.js", media_type="application/javascript")


@app.get("/api/health")
async def health() -> JSONResponse:
    web_search_provider = None
    if ap.HAS_TAVILY and ap.TAVILY_API_KEY:
        web_search_provider = "Tavily"
    elif ap.HAS_DDG:
        web_search_provider = "DuckDuckGo"
    return JSONResponse({
        "status": "ok", "llm_provider": pipeline.llm.provider,
        "web_search_provider": web_search_provider,
    })


@app.get("/api/stats")
async def api_stats() -> JSONResponse:
    row = await asyncio.to_thread(db.stats)
    return JSONResponse(row)


# --------------------------------------------------------------------------
# Persisted research history — backed by SQLite (db.py), survives restarts
# --------------------------------------------------------------------------

@app.get("/api/runs")
async def list_runs(limit: int = 30) -> JSONResponse:
    rows = await asyncio.to_thread(db.get_recent_runs, limit)
    return JSONResponse(rows)


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> JSONResponse:
    row = await asyncio.to_thread(db.get_run, run_id)
    if not row:
        return JSONResponse({"error": "run not found"}, status_code=404)
    return JSONResponse(row)


@app.get("/api/library")
async def library() -> JSONResponse:
    rows = await asyncio.to_thread(db.get_all_sources)
    return JSONResponse(rows)


@app.get("/api/decompositions")
async def decompositions(limit: int = 30) -> JSONResponse:
    rows = await asyncio.to_thread(db.get_recent_runs, limit)
    out = [{
        "query": r["query"], "is_followup": r["is_followup"],
        "sub_questions": r["sub_questions"], "search_queries": r["search_queries"],
        "created_at": r["created_at"],
    } for r in rows]
    return JSONResponse(out)


@app.get("/api/contradictions")
async def contradictions(limit: int = 50) -> JSONResponse:
    rows = await asyncio.to_thread(db.get_recent_runs, limit)
    out = []
    for r in rows:
        for c in r["contradictions"]:
            out.append({**c, "query": r["query"], "created_at": r["created_at"]})
    return JSONResponse(out)


# --------------------------------------------------------------------------
# SSE research stream
# --------------------------------------------------------------------------

async def _sse_event_generator(query: str, mode: str, context: str = ""):
    """Bridges ResearchPipeline.run() (an async generator of {event, data}
    dicts) into the wire format expected by the frontend's EventSource
    listeners (`step`, `log`, `telemetry`, `source`, `decomposition`,
    `contradiction`, `report`, `metrics`, `done`).

    The pipeline also emits an internal `persist` event carrying the full
    serialized run — that one is intercepted here and written to the
    database rather than forwarded to the browser."""
    try:
        async for item in pipeline.run(query=query, mode=mode, context=context):
            if item["event"] == "persist":
                try:
                    await asyncio.to_thread(db.save_run, item["data"])
                except Exception:
                    logger.exception("Failed to persist run to database")
                continue
            yield {
                "event": item["event"],
                "data": json.dumps(item["data"]),
            }
            # tiny yield so the stream flushes incrementally rather than bursting
            await asyncio.sleep(0.05)
    except Exception as exc:  # never let an unhandled error kill the stream silently
        logger.exception("Pipeline error")
        yield {"event": "error", "data": json.dumps({"message": f"Pipeline error: {exc}"})}
        yield {"event": "done", "data": json.dumps({})}


@app.get("/api/research/stream")
async def research_stream(
    query: str = Query(..., min_length=3, description="Natural-language research question"),
    mode: str = Query("compare", description="compare | benchmark | decompose | contradiction"),
    context: str = Query("", description="Optional prior-turn context, for follow-up questions"),
):
    if HAS_SSE_STARLETTE:
        return EventSourceResponse(_sse_event_generator(query, mode, context))

    # Fallback hand-rolled SSE framing if sse-starlette isn't installed.
    async def raw_stream():
        async for evt in _sse_event_generator(query, mode, context):
            yield f"event: {evt['event']}\ndata: {evt['data']}\n\n"

    return StreamingResponse(raw_stream(), media_type="text/event-stream")


# --------------------------------------------------------------------------
# Mount any additional static assets (e.g. /assets/*) if present
# --------------------------------------------------------------------------
assets_dir = BASE_DIR / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
