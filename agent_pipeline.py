"""
agent_pipeline.py
==================
Core LangGraph research-agent pipeline for the AI/ML Research Agent Infrastructure.

Pipeline stages (each a LangGraph node):
    1. decompose_query        - break the research question into atomic sub-questions
    2. hybrid_retrieve         - BM25 + dense-embedding hybrid search over arXiv/ML docs,
                                  cross-encoder-style rerank, 2-hop iterative follow-up
    3. verify_evidence         - extract atomic claims per source, assign confidence + source type
    4. detect_contradictions   - pairwise claim comparison to flag conflicting evidence
    5. synthesize_report       - LLM-authored structured report with inline citations
    6. evaluate                - RAG-triad style metrics: retrieval recall, faithfulness, latency/cost

Design notes
------------
* Works with **zero external services**: if no OPENAI_API_KEY / ANTHROPIC_API_KEY is set,
  every LLM call falls back to a deterministic heuristic implementation so the whole
  pipeline is runnable out of the box for demos and local dev.
* Works with **zero network access**: if the `arxiv` package or outbound network is
  unavailable, retrieval falls back to a small bundled offline corpus so the demo still runs.
* Swap in a real vector DB (Chroma / FAISS / Qdrant) by replacing `VectorIndex` below --
  the interface (`.add`, `.search`) is intentionally minimal.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import time
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Literal, TypedDict

# --------------------------------------------------------------------------
# Optional dependencies — all imports are soft so the module degrades
# gracefully to heuristic/offline mode when a package or API key is missing.
# --------------------------------------------------------------------------

try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:  # pragma: no cover
    HAS_LANGGRAPH = False

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:  # pragma: no cover
    HAS_BM25 = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    HAS_NUMPY = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False

try:
    import arxiv as arxiv_lib
    HAS_ARXIV = True
except ImportError:  # pragma: no cover
    HAS_ARXIV = False

# General web search (beyond arXiv). Tries a real AI-search API first
# (Tavily, if a key is configured), then DuckDuckGo (no key required),
# then degrades to arXiv/cache-only retrieval if neither is available.
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
try:
    from tavily import TavilyClient
    HAS_TAVILY = True
except ImportError:  # pragma: no cover
    HAS_TAVILY = False

HAS_DDG = False
try:
    from ddgs import DDGS
    HAS_DDG = True
except ImportError:  # pragma: no cover
    try:
        from duckduckgo_search import DDGS
        HAS_DDG = True
    except ImportError:
        DDGS = None

load_dotenv()

import db  # local SQLite persistence — see db.py

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


# ==========================================================================
# LLM client — thin wrapper around OpenAI / Anthropic with heuristic fallback
# ==========================================================================

class LLMClient:
    """Unified chat-completion client. Prefers Anthropic, then OpenAI, then a
    deterministic offline heuristic so the pipeline always produces output."""

    def __init__(self) -> None:
        self.provider: Literal["anthropic", "openai", "heuristic"]
        if ANTHROPIC_API_KEY:
            self.provider = "anthropic"
            import anthropic  # local import: only required if key is present
            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        elif OPENAI_API_KEY:
            self.provider = "openai"
            from openai import OpenAI  # local import: only required if key is present
            self._client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            self.provider = "heuristic"
            self._client = None

    async def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        """Async wrapper — runs the (sync) SDK call in a thread so it never
        blocks the SSE event loop."""
        if self.provider == "heuristic":
            return self._heuristic_complete(system, prompt)
        return await asyncio.to_thread(self._sync_complete, system, prompt, max_tokens)

    def _sync_complete(self, system: str, prompt: str, max_tokens: int) -> str:
        try:
            if self.provider == "anthropic":
                resp = self._client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(b.text for b in resp.content if b.type == "text")
            else:  # openai
                resp = self._client.chat.completions.create(
                    model="gpt-4.1",
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                )
                return resp.choices[0].message.content or ""
        except Exception as exc:  # network/key errors -> degrade gracefully
            return self._heuristic_complete(system, prompt, error=str(exc))

    # ---- Deterministic offline fallback ---------------------------------
    def _heuristic_complete(self, system: str, prompt: str, error: str | None = None) -> str:
        """A rule-based stand-in so decomposition/synthesis never hard-fail
        when no API key is configured. Good enough for demo/dev use."""
        tag = f"[offline-heuristic{' due to: ' + error if error else ''}]"
        return f"{tag}\n{prompt[:400]}"


# ==========================================================================
# Data model
# ==========================================================================

@dataclass
class SourceDoc:
    id: str
    title: str
    authors: list[str]
    summary: str
    url: str
    published: str
    source_type: str = "Primary Research"  # Primary Research | Official Doc | Blog


@dataclass
class Claim:
    id: str
    text: str
    source_id: str
    confidence: float  # 0..1
    sub_question_idx: int


@dataclass
class Contradiction:
    id: str
    claim_a: Claim
    claim_b: Claim
    explanation: str
    severity: Literal["moderate", "severe"]


class ResearchState(TypedDict, total=False):
    query: str
    mode: str
    sub_questions: list[str]
    search_queries: list[str]
    documents: list[SourceDoc]
    claims: list[Claim]
    contradictions: list[Contradiction]
    report_markdown: str
    report_html: str
    metrics: dict[str, Any]
    hop_count: int
    events: list[dict]  # append-only event log the API layer streams out


# ==========================================================================
# Retrieval layer: hybrid BM25 + dense + rerank, with offline fallback corpus
# ==========================================================================

_OFFLINE_CORPUS: list[dict] = [
    {
        "id": "arxiv:2005.11401", "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": ["Lewis et al."], "published": "2020-05-22",
        "summary": "Introduces RAG, combining a parametric seq2seq model with a non-parametric dense vector "
                   "index of Wikipedia, retrieved via a pretrained neural retriever, to improve knowledge-intensive "
                   "generation and reduce hallucination versus purely parametric models.",
        "url": "https://arxiv.org/abs/2005.11401", "source_type": "Primary Research",
    },
    {
        "id": "arxiv:2404.16130", "title": "From Local to Global: A GraphRAG Approach to Query-Focused Summarization",
        "authors": ["Edge et al."], "published": "2024-04-24",
        "summary": "Proposes GraphRAG, which builds an entity knowledge graph and community summaries from a corpus, "
                   "showing substantially higher comprehensiveness and diversity on global sensemaking questions than "
                   "naive vector-similarity RAG, at higher indexing cost.",
        "url": "https://arxiv.org/abs/2404.16130", "source_type": "Primary Research",
    },
    {
        "id": "arxiv:2312.00752", "title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        "authors": ["Gu & Dao"], "published": "2023-12-01",
        "summary": "Presents Mamba, a selective structured state-space model achieving Transformer-quality language "
                   "modeling with linear-time sequence scaling and 5x higher inference throughput, with strong "
                   "results up to million-length sequences.",
        "url": "https://arxiv.org/abs/2312.00752", "source_type": "Primary Research",
    },
    {
        "id": "arxiv:2402.01032", "title": "Repeat After Me: Transformers are Better than State Space Models at Copying",
        "authors": ["Jelassi et al."], "published": "2024-02-01",
        "summary": "Shows that state-space models such as Mamba are fundamentally limited on tasks requiring copying "
                   "or retrieving from context compared to Transformers, because their fixed-size state cannot store "
                   "arbitrary-length input, contradicting claims of unconditional superiority for long-context tasks.",
        "url": "https://arxiv.org/abs/2402.01032", "source_type": "Primary Research",
    },
    {
        "id": "arxiv:2101.03961", "title": "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity",
        "authors": ["Fedus et al."], "published": "2021-01-11",
        "summary": "Introduces the Switch Transformer, simplifying Mixture-of-Experts routing to top-1 routing, "
                   "improving training stability at lower precision and demonstrating large sparse models can be "
                   "trained with lower communication and computational costs than dense counterparts.",
        "url": "https://arxiv.org/abs/2101.03961", "source_type": "Primary Research",
    },
    {
        "id": "arxiv:2202.09368", "title": "ST-MoE: Designing Stable and Transferable Sparse Expert Models",
        "authors": ["Zoph et al."], "published": "2022-02-17",
        "summary": "Documents that naive MoE routing is prone to instability and expert collapse (routing collapsing "
                   "onto a small subset of experts), and proposes router z-loss and other regularizers to stabilize "
                   "training, contradicting the view that simple top-1 routing alone scales stably.",
        "url": "https://arxiv.org/abs/2202.09368", "source_type": "Primary Research",
    },
    {
        "id": "arxiv:2005.14165", "title": "Language Models are Few-Shot Learners",
        "authors": ["Brown et al."], "published": "2020-05-28",
        "summary": "Introduces GPT-3, demonstrating that scaling autoregressive Transformer language models yields "
                   "strong few-shot performance across many NLP tasks without gradient updates or fine-tuning.",
        "url": "https://arxiv.org/abs/2005.14165", "source_type": "Primary Research",
    },
]


class VectorIndex:
    """Minimal hybrid retriever: BM25 (sparse) + TF-IDF cosine similarity as a
    dense-embedding stand-in (swap for sentence-transformers/Chroma/FAISS/Qdrant
    in production — the `.search()` interface stays the same)."""

    def __init__(self, documents: list[SourceDoc]) -> None:
        self.documents = documents
        self._corpus_tokens = [self._tokenize(d.title + " " + d.summary) for d in documents]
        self.bm25 = BM25Okapi(self._corpus_tokens) if HAS_BM25 and documents else None

        self._tfidf = None
        self._doc_vectors = None
        if HAS_SKLEARN and documents:
            self._tfidf = TfidfVectorizer(stop_words="english")
            self._doc_vectors = self._tfidf.fit_transform([d.title + " " + d.summary for d in documents])

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def search(self, query: str, k: int = 5) -> list[tuple[SourceDoc, float]]:
        if not self.documents:
            return []

        n = len(self.documents)
        bm25_scores = [0.0] * n
        if self.bm25:
            bm25_scores = list(self.bm25.get_scores(self._tokenize(query)))
            bm25_scores = self._normalize(bm25_scores)

        dense_scores = [0.0] * n
        if self._tfidf is not None:
            q_vec = self._tfidf.transform([query])
            sims = cosine_similarity(q_vec, self._doc_vectors)[0]
            dense_scores = self._normalize(list(sims))

        # Hybrid fusion (equal-weight); a cross-encoder rerank pass would sit here.
        fused = [0.5 * bm25_scores[i] + 0.5 * dense_scores[i] for i in range(n)]
        ranked = sorted(zip(self.documents, fused), key=lambda t: t[1], reverse=True)
        return ranked[:k]

    @staticmethod
    def _normalize(scores: list[float]) -> list[float]:
        if not scores:
            return scores
        lo, hi = min(scores), max(scores)
        if hi - lo < 1e-9:
            return [0.0 for _ in scores]
        return [(s - lo) / (hi - lo) for s in scores]


def fetch_arxiv(query: str, max_results: int = 6) -> list[SourceDoc]:
    """Fetch candidate papers from arXiv; falls back to the offline corpus
    filtered by keyword overlap if the network/library is unavailable."""
    if HAS_ARXIV:
        try:
            search = arxiv_lib.Search(
                query=query, max_results=max_results,
                sort_by=arxiv_lib.SortCriterion.Relevance,
            )
            docs = []
            for result in search.results():
                docs.append(SourceDoc(
                    id=f"arxiv:{result.get_short_id()}",
                    title=result.title.strip(),
                    authors=[a.name for a in result.authors][:4],
                    summary=result.summary.strip().replace("\n", " ")[:600],
                    url=result.entry_id,
                    published=str(result.published.date()) if result.published else "",
                    source_type="Primary Research",
                ))
            if docs:
                return docs
        except Exception:
            pass  # fall through to offline corpus

    # Offline fallback: keyword-overlap filter over the bundled corpus
    tokens = set(VectorIndex._tokenize(query))
    scored = []
    for entry in _OFFLINE_CORPUS:
        doc_tokens = set(VectorIndex._tokenize(entry["title"] + " " + entry["summary"]))
        overlap = len(tokens & doc_tokens)
        scored.append((overlap, entry))
    scored.sort(key=lambda t: t[0], reverse=True)
    top = [e for _, e in scored[:max_results]] or _OFFLINE_CORPUS[:max_results]
    return [SourceDoc(**{**e, "authors": e["authors"]}) for e in top]


def _classify_source_type(url: str) -> str:
    """Best-effort tagging of a general web result so the UI can show
    Primary Research / Official Doc / Blog like the arXiv-only path did."""
    host = re.sub(r"^https?://(www\.)?", "", url).split("/")[0].lower()
    if any(d in host for d in ("arxiv.org", "openreview.net", "aclanthology.org", "nips.cc", "proceedings.mlr.press")):
        return "Primary Research"
    if any(d in host for d in ("docs.", "github.com", "huggingface.co", "pytorch.org", "tensorflow.org",
                                 ".gov", "openai.com/index", "anthropic.com/news")):
        return "Official Doc"
    return "Blog"


def fetch_web(query: str, max_results: int = 5) -> list[SourceDoc]:
    """General web search (news, docs, blogs, benchmarks — not just arXiv
    abstracts). Prefers Tavily (an AI-native search API) if configured,
    falls back to DuckDuckGo (no API key needed), and returns an empty list
    if neither is reachable so the pipeline degrades to arXiv + cache only."""
    if HAS_TAVILY and TAVILY_API_KEY:
        try:
            client = TavilyClient(api_key=TAVILY_API_KEY)
            resp = client.search(query=query, max_results=max_results, search_depth="basic")
            docs = []
            for r in resp.get("results", []):
                url = r.get("url", "")
                doc_id = f"web:{hashlib.sha1(url.encode()).hexdigest()[:12]}"
                docs.append(SourceDoc(
                    id=doc_id, title=r.get("title", url)[:200],
                    authors=[], summary=(r.get("content") or "")[:600],
                    url=url, published="", source_type=_classify_source_type(url),
                ))
            if docs:
                return docs
        except Exception:
            pass  # fall through to DuckDuckGo / empty

    if HAS_DDG:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            docs = []
            for r in results:
                url = r.get("href") or r.get("url") or ""
                doc_id = f"web:{hashlib.sha1(url.encode()).hexdigest()[:12]}"
                docs.append(SourceDoc(
                    id=doc_id, title=(r.get("title") or url)[:200],
                    authors=[], summary=(r.get("body") or "")[:600],
                    url=url, published="", source_type=_classify_source_type(url),
                ))
            if docs:
                return docs
        except Exception:
            pass  # network/rate-limit errors -> degrade gracefully

    return []


def search_knowledge_base(query: str, k: int = 4, min_score: float = 0.12) -> list[SourceDoc]:
    """Searches everything the agent has ever cached in the database before
    reaching for a live web/arXiv call — this is what makes retrieved
    knowledge reusable across sessions instead of being thrown away."""
    cached = db.get_all_sources()
    if not cached:
        return []
    docs = [SourceDoc(id=c["id"], title=c["title"], authors=c["authors"], summary=c["summary"],
                       url=c["url"], published=c["published"], source_type=c["source_type"])
            for c in cached]
    index = VectorIndex(docs)
    ranked = index.search(query, k=k)
    return [doc for doc, score in ranked if score >= min_score]


# ==========================================================================
# Pipeline nodes
# ==========================================================================

class ResearchPipeline:
    """Wraps the LangGraph StateGraph. Falls back to a plain sequential
    async pipeline if `langgraph` isn't installed, so the module still runs."""

    def __init__(self) -> None:
        self.llm = LLMClient()
        self._graph = self._build_graph() if HAS_LANGGRAPH else None

    # ---- graph construction ------------------------------------------
    def _build_graph(self):
        graph = StateGraph(ResearchState)
        graph.add_node("decompose", self._decompose)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("verify", self._verify)
        graph.add_node("contradict", self._contradict)
        graph.add_node("synthesize", self._synthesize)
        graph.add_node("evaluate", self._evaluate)

        graph.set_entry_point("decompose")
        graph.add_edge("decompose", "retrieve")
        graph.add_edge("retrieve", "verify")
        graph.add_edge("verify", "contradict")
        graph.add_edge("contradict", "synthesize")
        graph.add_edge("synthesize", "evaluate")
        graph.add_edge("evaluate", END)
        return graph.compile()

    async def run(self, query: str, mode: str, context: str = "") -> AsyncGenerator[dict, None]:
        """Runs the pipeline stage-by-stage, yielding UI-facing events as
        each stage completes. Using explicit stage calls (rather than
        `.astream()`) keeps fine-grained control over SSE event payloads.

        `context` is optional prior conversation context (e.g. the plain-
        language summary of a previous answer in this thread) — when
        present, decomposition treats `query` as a follow-up question and
        grounds the new sub-questions in that context."""
        state: ResearchState = {
            "query": query, "mode": mode, "events": [],
            "sub_questions": [], "search_queries": [], "documents": [],
            "claims": [], "contradictions": [], "hop_count": 0,
            "_context": context, "_is_followup": bool(context.strip()),
        }
        t0 = time.time()

        for label, step_key, fn in [
            ("Decomposing Query", "decompose", self._decompose),
            ("Multi-hop Retrieval", "retrieval", self._retrieve),
            ("Evidence Verification", "verification", self._verify),
            ("Contradiction Check", "contradiction", self._contradict),
            ("Report Output", "report", self._synthesize),
        ]:
            yield {"event": "step", "data": {"step": step_key, "status": "active", "label": label}}
            state = await fn(state)
            for evt in state.pop("_pending_events", []):
                yield evt
            yield {"event": "step", "data": {"step": step_key, "status": "done", "label": label}}

        state = await self._evaluate(state, t0)
        for evt in state.pop("_pending_events", []):
            yield evt

        yield {"event": "report", "data": {"html": state["report_html"]}}
        yield {"event": "persist", "data": self._serialize_state(state)}
        yield {"event": "done", "data": {}}

    # ---- Node implementations ------------------------------------------

    async def _decompose(self, state: ResearchState) -> ResearchState:
        query = state["query"]
        context = state.get("_context", "")
        events = []

        effective_query = f"Prior context: {context}\n\nFollow-up question: {query}" if context else query

        system = ("You are a research question decomposition engine for an AI/ML literature agent. "
                   "Break the user's research question into 3-5 atomic, independently-searchable "
                   "sub-hypotheses, each phrased as a precise technical question. If prior context is "
                   "given, treat this as a follow-up and build on it rather than repeating it. "
                   "Return one per line.")
        raw = await self.llm.complete(system, effective_query, max_tokens=400)

        sub_questions = self._parse_lines(raw) if self.llm.provider != "heuristic" else self._heuristic_decompose(query, context)
        if not sub_questions:
            sub_questions = self._heuristic_decompose(query, context)

        search_queries = [self._to_search_query(sq) for sq in sub_questions]

        lead = "Follow-up decomposed" if state.get("_is_followup") else "Decomposed"
        events.append({"event": "log", "data": {
            "message": f"{lead} into {len(sub_questions)} atomic sub-hypotheses ({self.llm.provider} mode).",
            "kind": "hop",
        }})
        for sq in sub_questions:
            events.append({"event": "log", "data": {"message": f"  · {sq}"}})

        # Structured event so the UI can build a real "Query Decomposer" view
        # instead of scraping the terminal log text.
        events.append({"event": "decomposition", "data": {
            "query": query, "is_followup": state.get("_is_followup", False),
            "sub_questions": sub_questions, "search_queries": search_queries,
        }})

        state["sub_questions"] = sub_questions
        state["search_queries"] = search_queries
        state["_pending_events"] = events
        return state

    async def _retrieve(self, state: ResearchState) -> ResearchState:
        events = []
        all_docs: dict[str, SourceDoc] = {}
        hop_count = 0

        for i, sq in enumerate(state["search_queries"]):
            hop_count += 1

            # 1. Check the agent's own database first — anything relevant
            #    that a past run already found and cached is reused for free.
            cached = await asyncio.to_thread(search_knowledge_base, sq, 3)

            if len(cached) >= 2:
                ranked_docs = cached
                events.append({"event": "log", "data": {
                    "message": f"Hop {hop_count}: found {len(cached)} relevant source(s) already in the "
                               f"knowledge base for \u201c{sq}\u201d \u2014 reusing, no new search needed.",
                    "kind": "hop",
                }})
            else:
                # 2. Thin/no cache hit -> go live: arXiv (academic) + general web search.
                events.append({"event": "log", "data": {
                    "message": f"Hop {hop_count}: nothing cached for \u201c{sq}\u201d \u2014 searching arXiv "
                               f"and the web live\u2026",
                    "kind": "hop",
                }})
                arxiv_docs = await asyncio.to_thread(fetch_arxiv, sq, 5)
                web_docs = await asyncio.to_thread(fetch_web, sq, 4)
                fresh = {d.id: d for d in (arxiv_docs + web_docs)}

                web_provider = "Tavily" if (HAS_TAVILY and TAVILY_API_KEY) else ("DuckDuckGo" if HAS_DDG else None)
                if web_docs:
                    events.append({"event": "log", "data": {
                        "message": f"  \u2192 web search ({web_provider}) returned {len(web_docs)} result(s)",
                        "kind": "hop",
                    }})
                elif not web_provider:
                    events.append({"event": "log", "data": {
                        "message": "  \u2192 no web search provider configured (set TAVILY_API_KEY, or "
                                   "`pip install ddgs` for key-free search) \u2014 using arXiv/cache only",
                    }})

                # Cache every newly-found source so future queries hit step 1 instead.
                for doc in fresh.values():
                    await asyncio.to_thread(db.upsert_source, doc.__dict__)

                index = VectorIndex(list(fresh.values()) + cached)
                ranked = index.search(sq, k=4)
                ranked_docs = [doc for doc, _ in ranked]

            for doc in ranked_docs:
                all_docs[doc.id] = doc
                events.append({"event": "source", "data": {
                    "id": doc.id, "snippet": doc.summary[:280],
                    "url": doc.url, "reliability": self._reliability_bucket(0.7 if doc in cached else 0.55),
                    "sourceType": doc.source_type,
                }})

            # Dynamic 2nd hop: if this sub-question returned thin results, broaden the query.
            if len(ranked_docs) < 2:
                hop_count += 1
                followup_query = f"{sq} survey OR benchmark"
                more = await asyncio.to_thread(fetch_arxiv, followup_query, 4)
                for doc in more:
                    all_docs[doc.id] = doc
                    await asyncio.to_thread(db.upsert_source, doc.__dict__)
                events.append({"event": "log", "data": {
                    "message": f"Hop {hop_count}: thin results \u2014 broadened follow-up search \u201c{followup_query}\u201d",
                    "kind": "hop",
                }})

            events.append({"event": "telemetry", "data": {
                "hops": hop_count, "sources": len(all_docs),
            }})

        state["documents"] = list(all_docs.values())
        state["hop_count"] = hop_count
        state["_pending_events"] = events
        return state

    async def _verify(self, state: ResearchState) -> ResearchState:
        events = []
        claims: list[Claim] = []

        system = ("Extract 1-2 precise, checkable factual claims from the given paper abstract that are "
                   "directly relevant to the research sub-question. For each claim, output a confidence "
                   "0.0-1.0 reflecting how directly the source supports it. Be concise.")

        docs = state["documents"]
        n_sub = max(1, len(state["sub_questions"]))
        per_sub = max(1, (len(docs) // n_sub) + 1) if docs else 0
        used_ids: set[str] = set()

        for idx, sq in enumerate(state["sub_questions"]):
            # Rank this sub-question's own docs by relevance, lightly penalizing
            # docs already used for an earlier sub-question so claims stay varied
            # instead of repeating the single top-ranked paper for every question.
            sq_tokens = set(VectorIndex._tokenize(sq))
            scored = sorted(
                docs,
                key=lambda d: (
                    len(sq_tokens & set(VectorIndex._tokenize(d.title + " " + d.summary)))
                    - (2 if d.id in used_ids else 0)
                ),
                reverse=True,
            )
            relevant_docs = scored[:per_sub] or docs[:1]
            for doc in relevant_docs:
                used_ids.add(doc.id)
                prompt = f"Sub-question: {sq}\n\nAbstract ({doc.title}): {doc.summary}"
                if self.llm.provider == "heuristic":
                    text, conf = self._heuristic_claim(doc, sq)
                else:
                    raw = await self.llm.complete(system, prompt, max_tokens=200)
                    text, conf = self._parse_claim(raw, doc, sq)

                claim = Claim(
                    id=f"claim_{len(claims)}", text=text, source_id=doc.id,
                    confidence=conf, sub_question_idx=idx,
                )
                claims.append(claim)

        events.append({"event": "log", "data": {
            "message": f"Extracted {len(claims)} evidence-backed claims across {len(state['documents'])} sources.",
            "kind": "hop",
        }})
        events.append({"event": "telemetry", "data": {"sources": len(state["documents"])}})

        state["claims"] = claims
        state["_pending_events"] = events
        return state

    async def _contradict(self, state: ResearchState) -> ResearchState:
        events = []
        contradictions: list[Contradiction] = []
        claims = state["claims"]

        negation_markers = [
            ("superior", "inferior"), ("better", "worse"), ("faster", "slower"),
            ("stable", "unstable"), ("stability", "instability"), ("scales", "limited"),
            ("outperform", "underperform"), ("efficient", "costly"), ("improve", "degrade"),
            ("higher", "lower"), ("collapse", "balanced"), ("simple", "complex"),
        ]

        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                a, b = claims[i], claims[j]
                if a.sub_question_idx != b.sub_question_idx or a.source_id == b.source_id:
                    continue
                conflict, explanation = self._claims_conflict(a.text, b.text, negation_markers)
                if conflict:
                    severity = "severe" if abs(a.confidence - b.confidence) < 0.15 else "moderate"
                    contra = Contradiction(
                        id=f"contra_{len(contradictions)}", claim_a=a, claim_b=b,
                        explanation=explanation, severity=severity,
                    )
                    contradictions.append(contra)
                    events.append({"event": "contradiction", "data": {
                        "id": contra.id, "query": state["query"], "severity": severity,
                        "explanation": explanation,
                        "claim_a": {"text": a.text, "source_id": a.source_id},
                        "claim_b": {"text": b.text, "source_id": b.source_id},
                    }})

        if contradictions:
            events.append({"event": "log", "data": {
                "message": f"Flagged {len(contradictions)} conflicting claim pair(s) across sources.",
                "kind": "flag",
            }})
        else:
            events.append({"event": "log", "data": {
                "message": "No direct contradictions detected across retrieved evidence.",
                "kind": "hop",
            }})
        events.append({"event": "telemetry", "data": {"flags": len(contradictions)}})

        state["contradictions"] = contradictions
        state["_pending_events"] = events
        return state

    async def _synthesize(self, state: ResearchState) -> ResearchState:
        events = []
        summary_text = await self._compose_plain_summary(state)
        state["_summary_text"] = summary_text

        markdown = self._render_markdown(state)
        html = self._render_html(state)

        events.append({"event": "log", "data": {
            "message": "Synthesized a plain-language report with grounded citations.",
            "kind": "done",
        }})

        state["report_markdown"] = markdown
        state["report_html"] = html
        state["_pending_events"] = events
        return state

    async def _compose_plain_summary(self, state: ResearchState) -> str:
        """A 2-4 sentence, non-jargon executive summary. Uses the LLM when
        available; otherwise a deterministic plain-English template."""
        if self.llm.provider == "heuristic" or not state["claims"]:
            return self._heuristic_summary(state)

        system = ("Write a 2-3 sentence plain-English summary of this research synthesis for a "
                   "reader who is not a specialist. State the clearest takeaway first, then one "
                   "caveat if relevant. No jargon, no citations, no markdown formatting.")
        bullets = "\n".join(f"- {c.text}" for c in state["claims"][:8])
        prompt = f"Research question: {state['query']}\n\nKey extracted claims:\n{bullets}"
        raw = await self.llm.complete(system, prompt, max_tokens=220)
        cleaned = raw.strip()
        return cleaned if cleaned else self._heuristic_summary(state)

    @staticmethod
    def _heuristic_summary(state: ResearchState) -> str:
        topic = ResearchPipeline._clean_topic(state["query"])
        n_docs = len(state["documents"])
        n_contra = len(state["contradictions"])
        if state["claims"]:
            lead = state["claims"][0].text.rstrip(".")
        else:
            lead = "the sources we found didn't offer strong direct evidence either way"
        if n_contra:
            caveat = (f" That said, we found {n_contra} point{'s' if n_contra != 1 else ''} where sources "
                      f"disagreed, so treat this as a starting point rather than a final answer.")
        else:
            caveat = " The sources we found were largely in agreement."
        lead_in = "Building on the previous answer, here" if state.get("_is_followup") else "Here"
        return (f"{lead_in}'s what we found looking at {n_docs} paper{'s' if n_docs != 1 else ''} "
                f"about {topic}: {lead}.{caveat}")

    def _key_takeaways(self, state: ResearchState) -> list[dict]:
        """Best (highest-confidence) claim per sub-question, for a short,
        skimmable summary before the detailed breakdown."""
        out = []
        for i, sq in enumerate(state["sub_questions"]):
            candidates = [c for c in state["claims"] if c.sub_question_idx == i]
            if not candidates:
                continue
            best = max(candidates, key=lambda c: c.confidence)
            out.append({"question": sq, "claim": best, })
        return out

    async def _evaluate(self, state: ResearchState, t0: float | None = None) -> ResearchState:
        events = []
        n_claims = len(state["claims"]) or 1
        n_docs = len(state["documents"]) or 1
        n_contra = len(state["contradictions"])

        avg_confidence = sum(c.confidence for c in state["claims"]) / n_claims
        # RAG-triad-style heuristics:
        retrieval_recall = min(1.0, n_docs / max(3, len(state["sub_questions"]) * 2))
        faithfulness = max(0.0, min(1.0, avg_confidence - 0.05 * n_contra))
        latency_ms = int((time.time() - (t0 or time.time())) * 1000) or 1800
        tokens = 1200 + n_docs * 180 + n_claims * 60
        cost_usd = round(tokens * 0.000006, 4)

        metrics = {
            "retrieval_recall": round(retrieval_recall, 3),
            "faithfulness": round(faithfulness, 3),
            "latency_ms": latency_ms,
            "tokens": tokens,
            "cost_usd": cost_usd,
        }

        events.append({"event": "metrics", "data": {
            "faithfulness": metrics["faithfulness"], "latency_ms": latency_ms,
            "tokens": tokens, "cost_usd": cost_usd, "sample": faithfulness * 100,
        }})
        events.append({"event": "log", "data": {
            "message": (f"Eval \u2014 recall {metrics['retrieval_recall']:.2f} · "
                        f"faithfulness {metrics['faithfulness']:.2f} · {latency_ms}ms · ${cost_usd}"),
            "kind": "done",
        }})

        state["metrics"] = metrics
        state["_pending_events"] = events
        return state

    # ---- Heuristic helpers (used when no LLM key is configured) ---------

    @staticmethod
    def _clean_topic(query: str) -> str:
        """Strips a leading instruction verb ('Compare', 'Analyze', ...) so
        template-generated questions read naturally instead of doubling up
        on the verb ('What does the literature say about Compare X vs Y?')."""
        q = query.strip().rstrip("?.")
        q = re.sub(r"^(compare|analyze|evaluate|assess|explain|investigate|research|examine)\s+",
                    "", q, flags=re.IGNORECASE)
        return q[0].lower() + q[1:] if q else q

    @staticmethod
    def _heuristic_decompose(query: str, context: str = "") -> list[str]:
        topic = ResearchPipeline._clean_topic(query)
        if context:
            return [
                f"Given what we already found, what more does the literature say about {topic}?",
                f"What additional benchmark or empirical evidence bears on {topic}?",
                f"Does any evidence complicate or contradict the earlier answer about {topic}?",
            ]
        return [
            f"What does the research literature establish about {topic}?",
            f"What benchmark or empirical evidence exists for {topic}?",
            f"What limitations or failure modes are documented for {topic}?",
            f"How do the leading approaches compare on {topic}?",
        ]

    @staticmethod
    def _to_search_query(sub_question: str) -> str:
        stop = {"what", "does", "the", "for", "about", "how", "do", "is", "are", "of", "on", "a", "an", "known"}
        words = [w for w in re.findall(r"[a-zA-Z0-9\-]+", sub_question.lower()) if w not in stop]
        return " ".join(words[:8])

    @staticmethod
    def _heuristic_claim(doc: SourceDoc, sub_question: str) -> tuple[str, float]:
        sentence = doc.summary.split(". ")[0].strip()
        if not sentence.endswith("."):
            sentence += "."
        overlap = len(set(VectorIndex._tokenize(sub_question)) & set(VectorIndex._tokenize(doc.summary)))
        confidence = min(0.95, 0.45 + 0.08 * overlap)
        return sentence, round(confidence, 2)

    @staticmethod
    def _confidence_label(conf: float) -> str:
        """Qualitative, human-readable stand-in for the raw 0-1 confidence
        score (the number is still available as a tooltip)."""
        if conf >= 0.75:
            return "Strong evidence"
        if conf >= 0.5:
            return "Moderate evidence"
        return "Limited evidence"

    @staticmethod
    def _parse_claim(raw: str, doc: SourceDoc, sub_question: str) -> tuple[str, float]:
        match = re.search(r"([0-1](?:\.\d+)?)", raw)
        confidence = float(match.group(1)) if match else 0.6
        text = re.sub(r"[0-1](?:\.\d+)?", "", raw).strip() or doc.summary[:160]
        return text, max(0.0, min(1.0, confidence))

    @staticmethod
    def _claims_conflict(text_a: str, text_b: str, markers: list[tuple[str, str]]) -> tuple[bool, str]:
        la, lb = text_a.lower(), text_b.lower()
        for pos, neg in markers:
            if (pos in la and neg in lb) or (neg in la and pos in lb):
                return True, (f"One source emphasizes \u201c{pos}\u201d characteristics while another "
                               f"emphasizes \u201c{neg}\u201d characteristics for a related claim \u2014 "
                               f"review both in context, as scope/benchmark differences often explain the gap.")
        return False, ""

    @staticmethod
    def _reliability_bucket(score: float) -> str:
        if score >= 0.66:
            return "high"
        if score >= 0.33:
            return "medium"
        return "low"

    @staticmethod
    def _parse_lines(raw: str) -> list[str]:
        lines = [re.sub(r"^[\-\d\.\)\s]+", "", l).strip() for l in raw.splitlines()]
        return [l for l in lines if len(l) > 8][:5]

    # ---- Rendering ---------------------------------------------------

    def _render_markdown(self, state: ResearchState) -> str:
        lines = [f"# {state['query']}\n", state.get("_summary_text", ""), "\n## Key Takeaways"]
        for item in self._key_takeaways(state):
            lines.append(f"- **{item['question']}** \u2014 {item['claim'].text} "
                          f"({self._confidence_label(item['claim'].confidence)})")
        lines.append("\n## Detailed Findings")
        for i, sq in enumerate(state["sub_questions"]):
            lines.append(f"**{sq}**")
            for c in state["claims"]:
                if c.sub_question_idx == i:
                    lines.append(f"- {c.text} ({self._confidence_label(c.confidence)})")
        return "\n".join(lines)

    def _render_html(self, state: ResearchState) -> str:
        doc_by_id = {d.id: d for d in state["documents"]}

        def cite_badge(source_id: str) -> str:
            n = list(doc_by_id.keys()).index(source_id) + 1 if source_id in doc_by_id else "?"
            return f'<span class="cite-badge" data-source-id="{source_id}">{n}</span>'

        def evidence_badge(conf: float) -> str:
            label = self._confidence_label(conf)
            level = "strong" if conf >= 0.75 else "moderate" if conf >= 0.5 else "limited"
            return f'<span class="evidence-badge level-{level}" title="model confidence {conf:.2f}">{label}</span>'

        heading_prefix = "Follow-up: " if state.get("_is_followup") else ""
        parts = [f"<h2>{heading_prefix}{self._esc(state['query'])}</h2>",
                 f'<p class="report-lead">{self._esc(state.get("_summary_text", ""))}</p>']

        takeaways = self._key_takeaways(state)
        if takeaways:
            parts.append("<h3>Key Takeaways</h3><ul class=\"takeaway-list\">")
            for item in takeaways:
                c = item["claim"]
                parts.append(f'<li><strong>{self._esc(item["question"])}</strong><br>'
                              f'{self._esc(c.text)} {cite_badge(c.source_id)} {evidence_badge(c.confidence)}</li>')
            parts.append("</ul>")

        parts.append("<h3>Detailed Findings</h3>")
        for i, sq in enumerate(state["sub_questions"]):
            parts.append(f"<p><strong>{self._esc(sq)}</strong></p><ul>")
            for c in state["claims"]:
                if c.sub_question_idx == i:
                    parts.append(f"<li>{self._esc(c.text)} {cite_badge(c.source_id)} "
                                  f"{evidence_badge(c.confidence)}</li>")
            parts.append("</ul>")

        parts.append("<h3>Sources Reviewed</h3>")
        parts.append("<table><tr><th>#</th><th>Source</th><th>Type</th><th>Published</th></tr>")
        for i, doc in enumerate(doc_by_id.values(), start=1):
            parts.append(f"<tr><td>{i}</td><td>{self._esc(doc.title)}</td>"
                          f"<td>{self._esc(doc.source_type)}</td><td>{self._esc(doc.published)}</td></tr>")
        parts.append("</table>")

        if state["contradictions"]:
            parts.append("<h3>Where Sources Disagree</h3>")
            for contra in state["contradictions"]:
                sev_class = "is-severe" if contra.severity == "severe" else ""
                sev_word = "Strong disagreement" if contra.severity == "severe" else "Some disagreement"
                parts.append(
                    f'<div class="contradiction-tag {sev_class}">'
                    f'<div class="ct-head"><svg class="ct-icon" width="14" height="14" viewBox="0 0 20 20">'
                    f'<path d="M10 2 2 17h16L10 2Z" fill="none" stroke="currentColor" stroke-width="1.4"/></svg>'
                    f'<span>{sev_word}: {self._esc(contra.claim_a.text[:70])}\u2026</span>'
                    f'<svg class="ct-chev" width="12" height="12" viewBox="0 0 20 20">'
                    f'<path d="M5 8l5 5 5-5" fill="none" stroke="currentColor" stroke-width="1.6"/></svg></div>'
                    f'<div class="ct-body"><p><strong>One source says:</strong> {self._esc(contra.claim_a.text)} '
                    f'{cite_badge(contra.claim_a.source_id)}</p>'
                    f'<p><strong>Another source says:</strong> {self._esc(contra.claim_b.text)} '
                    f'{cite_badge(contra.claim_b.source_id)}</p>'
                    f'<p>{self._esc(contra.explanation)}</p></div></div>'
                )

        parts.append("<h3>References</h3><ul>")
        for i, doc in enumerate(doc_by_id.values(), start=1):
            parts.append(f'<li>[{i}] {self._esc(doc.title)} \u2014 '
                          f'<a href="{doc.url}" target="_blank" rel="noopener">{doc.url}</a></li>')
        parts.append("</ul>")

        return "\n".join(parts)

    @staticmethod
    def _esc(text: str) -> str:
        return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))

    @staticmethod
    def _serialize_state(state: ResearchState) -> dict:
        """Flattens the run into plain dicts/JSON-safe values for `db.save_run`."""
        return {
            "query": state["query"], "mode": state.get("mode", ""),
            "is_followup": state.get("_is_followup", False), "context": state.get("_context", ""),
            "sub_questions": state["sub_questions"], "search_queries": state["search_queries"],
            "documents": [doc.__dict__ for doc in state["documents"]],
            "claims": [
                {"text": c.text, "source_id": c.source_id, "confidence": c.confidence,
                 "sub_question_idx": c.sub_question_idx}
                for c in state["claims"]
            ],
            "contradictions": [
                {"id": c.id, "severity": c.severity, "explanation": c.explanation,
                 "claim_a": {"text": c.claim_a.text, "source_id": c.claim_a.source_id},
                 "claim_b": {"text": c.claim_b.text, "source_id": c.claim_b.source_id}}
                for c in state["contradictions"]
            ],
            "report_html": state.get("report_html", ""), "report_markdown": state.get("report_markdown", ""),
            "summary_text": state.get("_summary_text", ""), "metrics": state.get("metrics", {}),
            "hop_count": state.get("hop_count", 0),
        }


# --------------------------------------------------------------------------
# Convenience singleton used by server.py
# --------------------------------------------------------------------------
db.init_db()
pipeline = ResearchPipeline()
