"""
db.py
=====
Lightweight SQLite persistence layer for the AI/ML Research Agent Infrastructure.

Why SQLite (not Postgres/Mongo/etc.)
-------------------------------------
Zero setup, zero external services, one file on disk — matches this project's
"immediately runnable" philosophy. Swap in Postgres for a multi-user production
deployment by replacing the connection helper below (`_connect`); every other
function only talks to the small API surface defined here.

What's stored
--------------
* `sources` — a growing, deduplicated cache of every paper/page the agent has
  ever retrieved (title, summary, url, type, first/last seen, how many times
  it's been reused). This is the agent's long-term knowledge base: future
  queries search it *before* going back out to arXiv/the web.
* `runs`    — one row per research run (including follow-ups), storing the
  decomposition, claims, contradictions, rendered report, and metrics as
  JSON columns so the whole run can be replayed later without re-querying
  the pipeline.

All functions are plain, synchronous sqlite3 calls — callers from async code
should wrap them in `asyncio.to_thread(...)`.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "research_agent.db"

_lock = threading.Lock()  # serialize writes; SQLite handles concurrent reads fine in WAL mode


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Creates tables if they don't already exist. Safe to call on every startup."""
    with _lock, _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sources (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors_json TEXT,
                summary TEXT,
                url TEXT,
                published TEXT,
                source_type TEXT,
                first_seen_at TEXT,
                last_seen_at TEXT,
                times_used INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                mode TEXT,
                is_followup INTEGER DEFAULT 0,
                context TEXT,
                sub_questions_json TEXT,
                search_queries_json TEXT,
                claims_json TEXT,
                contradictions_json TEXT,
                source_ids_json TEXT,
                report_html TEXT,
                report_markdown TEXT,
                summary_text TEXT,
                metrics_json TEXT,
                hop_count INTEGER,
                created_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
            CREATE INDEX IF NOT EXISTS idx_sources_last_seen ON sources(last_seen_at);
            """
        )
        conn.commit()


# --------------------------------------------------------------------------
# Sources cache — the agent's growing long-term knowledge base
# --------------------------------------------------------------------------

def upsert_source(doc: dict[str, Any]) -> None:
    """Insert a newly-seen source, or bump its last-seen timestamp and reuse
    count if we've already cached it (from an earlier run or a different
    sub-question in this one)."""
    now = _now()
    with _lock, _connect() as conn:
        existing = conn.execute("SELECT id, times_used FROM sources WHERE id = ?", (doc["id"],)).fetchone()
        if existing:
            conn.execute(
                "UPDATE sources SET last_seen_at = ?, times_used = ? WHERE id = ?",
                (now, existing["times_used"] + 1, doc["id"]),
            )
        else:
            conn.execute(
                """INSERT INTO sources (id, title, authors_json, summary, url, published,
                                         source_type, first_seen_at, last_seen_at, times_used)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                (doc["id"], doc.get("title", ""), json.dumps(doc.get("authors", [])),
                 doc.get("summary", ""), doc.get("url", ""), doc.get("published", ""),
                 doc.get("source_type", "Web"), now, now),
            )
        conn.commit()


def get_all_sources() -> list[dict]:
    """The full cached knowledge base — used both to answer future queries
    from cache and to populate the Literature Library view."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM sources ORDER BY last_seen_at DESC").fetchall()
        return [_source_row_to_dict(r) for r in rows]


def _source_row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["authors"] = json.loads(d.pop("authors_json") or "[]")
    return d


# --------------------------------------------------------------------------
# Runs — one row per research run, replayable later without re-querying
# --------------------------------------------------------------------------

def save_run(payload: dict[str, Any]) -> str:
    run_id = payload.get("id") or f"run_{uuid.uuid4().hex[:12]}"
    now = _now()

    source_ids = [d["id"] for d in payload.get("documents", [])]
    for doc in payload.get("documents", []):
        upsert_source(doc)

    with _lock, _connect() as conn:
        conn.execute(
            """INSERT INTO runs (id, query, mode, is_followup, context, sub_questions_json,
                                  search_queries_json, claims_json, contradictions_json,
                                  source_ids_json, report_html, report_markdown, summary_text,
                                  metrics_json, hop_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, payload["query"], payload.get("mode", ""),
                1 if payload.get("is_followup") else 0, payload.get("context", ""),
                json.dumps(payload.get("sub_questions", [])),
                json.dumps(payload.get("search_queries", [])),
                json.dumps(payload.get("claims", [])),
                json.dumps(payload.get("contradictions", [])),
                json.dumps(source_ids),
                payload.get("report_html", ""), payload.get("report_markdown", ""),
                payload.get("summary_text", ""), json.dumps(payload.get("metrics", {})),
                payload.get("hop_count", 0), now,
            ),
        )
        conn.commit()
    return run_id


def get_recent_runs(limit: int = 30) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_run_row_to_dict(r) for r in rows]


def get_run(run_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return _run_row_to_dict(row) if row else None


def _run_row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for key in ("sub_questions_json", "search_queries_json", "claims_json",
                "contradictions_json", "source_ids_json", "metrics_json"):
        new_key = key.replace("_json", "")
        d[new_key] = json.loads(d.pop(key) or ("{}" if key == "metrics_json" else "[]"))
    d["is_followup"] = bool(d["is_followup"])
    return d


def stats() -> dict:
    with _connect() as conn:
        n_runs = conn.execute("SELECT COUNT(*) c FROM runs").fetchone()["c"]
        n_sources = conn.execute("SELECT COUNT(*) c FROM sources").fetchone()["c"]
        return {"runs": n_runs, "sources": n_sources, "db_path": str(DB_PATH)}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
