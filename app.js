/* ==========================================================================
   ResearchAgent AI — app.js
   Handles: query submission + follow-ups, SSE streaming, pipeline stepper,
   telemetry, citation popovers, contradiction accordions, sparkline
   rendering, and the Library / Decomposer / Contradiction / Metrics /
   Settings workspace views.
   ========================================================================== */

const API_BASE = window.RESEARCH_AGENT_API_BASE || ""; // same-origin by default

// ---- DOM refs ----------------------------------------------------------
const homeView = document.getElementById("homeView");
const genericView = document.getElementById("genericView");
const genericTitle = document.getElementById("genericTitle");
const genericEyebrow = document.getElementById("genericEyebrow");
const genericBody = document.getElementById("genericBody");

const heroSection = document.getElementById("heroSection");
const executionSection = document.getElementById("executionSection");
const queryForm = document.getElementById("queryForm");
const queryInput = document.getElementById("queryInput");
const runBtn = document.getElementById("runBtn");
const runQueryTitle = document.getElementById("runQueryTitle");
const newQueryBtn = document.getElementById("newQueryBtn");
const eventLog = document.getElementById("eventLog");
const reportPanel = document.getElementById("reportPanel");
const reportBody = document.getElementById("reportBody");
const pipelineStepper = document.getElementById("pipelineStepper");
const followupBar = document.getElementById("followupBar");
const followupForm = document.getElementById("followupForm");
const followupInput = document.getElementById("followupInput");

const recentList = document.getElementById("recentList");
const recentEmpty = document.getElementById("recentEmpty");

const agentStatusTitle = document.getElementById("agentStatusTitle");
const agentStatusSub = document.getElementById("agentStatusSub");
const telHops = document.getElementById("telHops");
const telSources = document.getElementById("telSources");
const telFlags = document.getElementById("telFlags");
const telProgress = document.getElementById("telProgress");

const metFaithfulness = document.getElementById("metFaithfulness");
const metLatency = document.getElementById("metLatency");
const metTokens = document.getElementById("metTokens");
const metCost = document.getElementById("metCost");
const sparkArea = document.getElementById("sparkArea");
const sparkLine = document.getElementById("sparkLine");

const citationPopover = document.getElementById("citationPopover");
const cpSnippet = document.getElementById("cpSnippet");
const cpReliability = document.getElementById("cpReliability");
const cpSourceType = document.getElementById("cpSourceType");
const cpLink = document.getElementById("cpLink");
const cpClose = document.getElementById("cpClose");

const menuToggle = document.getElementById("menuToggle");
const sidebar = document.getElementById("sidebar");

// ---- Session state --------------------------------------------------------
let activeMode = "compare";
let currentSource = null; // EventSource instance
let latencySamples = [];

// Kept for the whole browser session so citation popovers resolve correctly
// even for reports loaded from history (server is the source of truth for
// aggregate views — see renderLibraryView/renderDecomposerView/etc below).
let sourceRegistry = {};       // doc id -> { snippet, url, reliability, sourceType, title }

let currentRun = null;     // true while a run is in flight (disables the follow-up box)
let lastSummaryText = "";  // plain-language summary of the most recent report, used as follow-up context
let currentViewName = "home";

const STEP_ORDER = ["decompose", "retrieval", "verification", "contradiction", "report"];

// ---- View switching (sidebar nav) -----------------------------------------
document.querySelectorAll(".nav-item[data-view]").forEach((item) => {
  item.addEventListener("click", (e) => {
    e.preventDefault();
    switchView(item.dataset.view);
    sidebar.classList.remove("is-open");
  });
});

function switchView(view) {
  currentViewName = view;
  document.querySelectorAll(".nav-item[data-view]").forEach((el) => {
    el.classList.toggle("is-active", el.dataset.view === view);
  });

  if (view === "home") {
    homeView.hidden = false;
    genericView.hidden = true;
    return;
  }

  homeView.hidden = true;
  genericView.hidden = false;
  renderGenericView(view);
}

function renderGenericView(view) {
  const renderers = {
    library: renderLibraryView,
    decomposer: renderDecomposerView,
    contradiction: renderContradictionView,
    metrics: renderMetricsView,
    settings: renderSettingsView,
  };
  const titles = {
    library: ["Literature Library", "Every source the agent has ever retrieved, stored in its database"],
    decomposer: ["Query Decomposer", "How each question was broken into sub-hypotheses"],
    contradiction: ["Contradiction Engine", "Conflicting evidence flagged across every research run on record"],
    metrics: ["Evaluation Metrics", "RAG-triad style scoring per run"],
    settings: ["Settings", "Agent configuration and stored data"],
  };
  const [title, sub] = titles[view] || ["", ""];
  genericEyebrow.textContent = sub;
  genericTitle.textContent = title;
  genericBody.innerHTML = emptyState("Loading…");
  (renderers[view] || (() => {}))();
}

function emptyState(msg) {
  return `<div class="empty-state">${escapeHtml(msg)}</div>`;
}

async function apiGet(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

async function renderLibraryView() {
  let docs;
  try { docs = await apiGet("/api/library"); }
  catch { genericBody.innerHTML = emptyState("Couldn't reach the backend database."); return; }

  // Keep citation popovers working for reports loaded from history/library.
  docs.forEach((d) => { sourceRegistry[d.id] = { ...sourceRegistry[d.id], ...d, sourceType: d.source_type, snippet: d.summary }; });

  if (!docs.length) {
    genericBody.innerHTML = emptyState("No sources retrieved yet — run a research query on Home to start building the library.");
    return;
  }
  genericBody.innerHTML = `<div class="library-grid">${docs.map((d) => `
    <div class="library-card">
      <span class="library-type">${escapeHtml(d.source_type || "Source")}</span>
      <p class="library-title">${escapeHtml(d.title || d.id)}</p>
      <p class="library-snippet">${escapeHtml((d.summary || "").slice(0, 220))}</p>
      <div class="library-foot">
        <a class="library-link" href="${escapeAttr(d.url || "#")}" target="_blank" rel="noopener">View source ↗</a>
        <span class="library-reuse">used ${d.times_used}×</span>
      </div>
    </div>`).join("")}</div>`;
}

async function renderDecomposerView() {
  let items;
  try { items = await apiGet("/api/decompositions"); }
  catch { genericBody.innerHTML = emptyState("Couldn't reach the backend database."); return; }

  if (!items.length) {
    genericBody.innerHTML = emptyState("No queries decomposed yet — ask a research question on Home to see it broken into sub-hypotheses.");
    return;
  }
  const html = [...items].reverse().map((d) => `
    <div class="decomposer-card">
      <div class="decomposer-head">
        ${d.is_followup ? '<span class="qs-tag">Follow-up</span>' : ""}
        <p class="decomposer-query">${escapeHtml(d.query)}</p>
      </div>
      <ol class="decomposer-list">
        ${d.sub_questions.map((sq, i) => `<li>${escapeHtml(sq)}<span class="decomposer-search">search: “${escapeHtml(d.search_queries[i] || "")}”</span></li>`).join("")}
      </ol>
    </div>`).join("");
  genericBody.innerHTML = `<div class="decomposer-stack">${html}</div>`;
}

async function renderContradictionView() {
  let items;
  try { items = await apiGet("/api/contradictions"); }
  catch { genericBody.innerHTML = emptyState("Couldn't reach the backend database."); return; }

  if (!items.length) {
    genericBody.innerHTML = emptyState("No contradictions flagged yet. When sources disagree on a claim, they’ll show up here across every query you run.");
    return;
  }
  const html = [...items].reverse().map((c) => {
    const sevClass = c.severity === "severe" ? "is-severe" : "";
    const sevWord = c.severity === "severe" ? "Strong disagreement" : "Some disagreement";
    return `
    <div class="contradiction-tag ${sevClass}">
      <div class="ct-head">
        <svg class="ct-icon" width="14" height="14" viewBox="0 0 20 20"><path d="M10 2 2 17h16L10 2Z" fill="none" stroke="currentColor" stroke-width="1.4"/></svg>
        <span>${escapeHtml(sevWord)} — from “${escapeHtml(c.query)}”</span>
        <svg class="ct-chev" width="12" height="12" viewBox="0 0 20 20"><path d="M5 8l5 5 5-5" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>
      </div>
      <div class="ct-body">
        <p><strong>One source says:</strong> ${escapeHtml(c.claim_a.text)}</p>
        <p><strong>Another source says:</strong> ${escapeHtml(c.claim_b.text)}</p>
        <p>${escapeHtml(c.explanation)}</p>
      </div>
    </div>`;
  }).join("");
  genericBody.innerHTML = `<div class="contradiction-stack">${html}</div>`;
  genericBody.querySelectorAll(".contradiction-tag").forEach((tag) => {
    tag.addEventListener("click", () => tag.classList.toggle("is-open"));
  });
}

async function renderMetricsView() {
  let runs;
  try { runs = await apiGet("/api/runs"); }
  catch { genericBody.innerHTML = emptyState("Couldn't reach the backend database."); return; }

  if (!runs.length) {
    genericBody.innerHTML = emptyState("No completed runs yet — metrics appear here once a query finishes.");
    return;
  }
  const rows = runs.map((r) => `
    <tr>
      <td>${escapeHtml(r.is_followup ? "↳ " + r.query : r.query)}</td>
      <td>${r.metrics.faithfulness !== undefined ? Math.round(r.metrics.faithfulness * 100) + "%" : "—"}</td>
      <td>${r.metrics.retrieval_recall !== undefined ? Math.round(r.metrics.retrieval_recall * 100) + "%" : "—"}</td>
      <td>${r.metrics.latency_ms !== undefined ? (r.metrics.latency_ms / 1000).toFixed(1) + "s" : "—"}</td>
      <td>${r.metrics.tokens !== undefined ? formatCompact(r.metrics.tokens) : "—"}</td>
      <td>${r.metrics.cost_usd !== undefined ? "$" + r.metrics.cost_usd.toFixed(3) : "—"}</td>
      <td>${r.contradictions.length}</td>
    </tr>`).join("");
  genericBody.innerHTML = `
    <table class="metrics-table">
      <tr><th>Query</th><th>Faithfulness</th><th>Recall</th><th>Latency</th><th>Tokens</th><th>Est. cost</th><th>Flags</th></tr>
      ${rows}
    </table>`;
}

async function renderSettingsView() {
  let statsData = { runs: "—", sources: "—" };
  try { statsData = await apiGet("/api/stats"); } catch {}

  genericBody.innerHTML = `
    <div class="settings-grid">
      <div class="settings-card">
        <h3 class="panel-title">Model Provider</h3>
        <p class="settings-value" id="settingsProvider">Checking…</p>
        <p class="settings-hint">Set <code>ANTHROPIC_API_KEY</code> or <code>OPENAI_API_KEY</code> on the server to use a real LLM instead of the offline heuristic mode.</p>
      </div>
      <div class="settings-card">
        <h3 class="panel-title">Web Search</h3>
        <p class="settings-value" id="settingsSearch">Checking…</p>
        <p class="settings-hint">Set <code>TAVILY_API_KEY</code> for AI-native search, or install <code>ddgs</code> for key-free DuckDuckGo search.</p>
      </div>
      <div class="settings-card">
        <h3 class="panel-title">Default Research Mode</h3>
        <div class="settings-mode-options" id="settingsModeOptions"></div>
      </div>
      <div class="settings-card">
        <h3 class="panel-title">Database</h3>
        <p class="settings-hint">${statsData.runs} run(s) and ${statsData.sources} cached source(s) stored persistently in SQLite${statsData.db_path ? ` (<code>${escapeHtml(statsData.db_path.split("/").pop())}</code>)` : ""}. This survives page reloads and server restarts.</p>
      </div>
    </div>`;

  const modeOptionsEl = document.getElementById("settingsModeOptions");
  const modes = [
    ["compare", "Compare Architectures"], ["benchmark", "Benchmark Synthesis"],
    ["decompose", "Decompose & Search"], ["contradiction", "Contradiction Scan"],
  ];
  modeOptionsEl.innerHTML = modes.map(([val, label]) => `
    <label class="settings-radio">
      <input type="radio" name="settingsMode" value="${val}" ${activeMode === val ? "checked" : ""}/>
      ${escapeHtml(label)}
    </label>`).join("");
  modeOptionsEl.querySelectorAll("input[name='settingsMode']").forEach((input) => {
    input.addEventListener("change", () => {
      activeMode = input.value;
      document.querySelectorAll(".chip").forEach((c) => c.classList.toggle("is-active", c.dataset.mode === activeMode));
    });
  });

  try {
    const data = await apiGet("/api/health");
    document.getElementById("settingsProvider").textContent =
      data.llm_provider === "heuristic" ? "Offline heuristic mode (no API key configured)" : `Connected — ${data.llm_provider}`;
    const searchEl = document.getElementById("settingsSearch");
    if (data.web_search_provider) {
      searchEl.textContent = `Connected — ${data.web_search_provider}`;
    } else {
      searchEl.textContent = "Not configured (arXiv + cached knowledge base only)";
    }
  } catch {
    document.getElementById("settingsProvider").textContent = "Unable to reach the backend.";
    document.getElementById("settingsSearch").textContent = "Unable to reach the backend.";
  }
}

// ---- Mode chips -----------------------------------------------------------
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    document.querySelectorAll(".chip").forEach((c) => c.classList.remove("is-active"));
    chip.classList.add("is-active");
    activeMode = chip.dataset.mode;
  });
});

// ---- Quickstart cards -----------------------------------------------------
document.querySelectorAll(".quickstart-card").forEach((card) => {
  card.addEventListener("click", () => {
    queryInput.value = card.dataset.query;
    startInvestigation(card.dataset.query);
  });
});

// ---- Sidebar toggle (mobile) -----------------------------------------------
menuToggle?.addEventListener("click", () => sidebar.classList.toggle("is-open"));

// ---- Textarea autosize -------------------------------------------------
queryInput.addEventListener("input", () => {
  queryInput.style.height = "auto";
  queryInput.style.height = Math.min(queryInput.scrollHeight, 160) + "px";
});

// ---- Query submit -----------------------------------------------------
queryForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = queryInput.value.trim();
  if (!q) return;
  startInvestigation(q);
});

// ---- Follow-up submit ----------------------------------------------------
followupForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = followupInput.value.trim();
  if (!q || currentSource) return; // ignore while a run is already in flight
  followupInput.value = "";
  startInvestigation(q, { append: true, context: lastSummaryText });
});

newQueryBtn.addEventListener("click", resetToHero);

function resetToHero() {
  if (currentSource) { currentSource.close(); currentSource = null; }
  executionSection.hidden = true;
  heroSection.hidden = false;
  queryInput.value = "";
  eventLog.innerHTML = "";
  reportPanel.hidden = true;
  reportBody.innerHTML = "";
  followupBar.hidden = true;
  resetPipelineSteps();
  agentStatusTitle.textContent = "Agent Idle";
  agentStatusSub.textContent = "Awaiting a research question";
  telHops.textContent = "0";
  telSources.textContent = "0";
  telFlags.textContent = "0";
  telProgress.style.width = "0%";
  metFaithfulness.textContent = "—";
  metLatency.textContent = "—";
  metTokens.textContent = "—";
  metCost.textContent = "—";
  latencySamples = [];
  lastSummaryText = "";
  drawSparkline();
}

function resetPipelineSteps() {
  pipelineStepper.querySelectorAll(".step").forEach((s) => {
    s.classList.remove("is-active", "is-done");
  });
}

// ---- Main investigation flow -------------------------------------------
function startInvestigation(query, opts = {}) {
  const isFollowup = !!opts.append;

  homeView.hidden = false;
  genericView.hidden = true;
  switchView("home");

  heroSection.hidden = true;
  executionSection.hidden = false;
  if (!isFollowup) {
    runQueryTitle.textContent = query;
    reportBody.innerHTML = "";
  } else {
    runQueryTitle.textContent = `${runQueryTitle.textContent} → ${query}`;
  }
  eventLog.innerHTML = "";
  resetPipelineSteps();

  agentStatusTitle.textContent = "Agent Working";
  agentStatusSub.textContent = isFollowup ? "Following up on your last question…" : "Decomposing your question…";

  runBtn.disabled = true;
  followupForm.querySelector(".followup-btn").disabled = true;

  currentRun = true;

  const params = new URLSearchParams({ query, mode: activeMode });
  if (opts.context) params.set("context", opts.context);
  const url = `${API_BASE}/api/research/stream?${params.toString()}`;

  if (currentSource) currentSource.close();
  const es = new EventSource(url);
  currentSource = es;

  es.addEventListener("step", (evt) => handleStepEvent(JSON.parse(evt.data)));
  es.addEventListener("log", (evt) => handleLogEvent(JSON.parse(evt.data)));
  es.addEventListener("telemetry", (evt) => handleTelemetryEvent(JSON.parse(evt.data)));
  es.addEventListener("source", (evt) => handleSourceEvent(JSON.parse(evt.data)));
  es.addEventListener("decomposition", (evt) => handleDecompositionEvent(JSON.parse(evt.data)));
  es.addEventListener("contradiction", (evt) => handleContradictionEvent(JSON.parse(evt.data)));
  es.addEventListener("report", (evt) => handleReportEvent(JSON.parse(evt.data)));
  es.addEventListener("metrics", (evt) => handleMetricsEvent(JSON.parse(evt.data)));
  es.addEventListener("done", () => {
    agentStatusTitle.textContent = "Investigation Complete";
    agentStatusSub.textContent = "Report ready below — ask a follow-up any time.";
    runBtn.disabled = false;
    followupForm.querySelector(".followup-btn").disabled = false;
    es.close();
    currentSource = null;

    if (currentRun) {
      currentRun = null;
      renderRecentList();
      if (currentViewName !== "home") renderGenericView(currentViewName); // keep aggregate views fresh
    }
  });
  es.addEventListener("error", (evt) => {
    let msg = "Connection lost. The pipeline may still be running.";
    try { if (evt.data) msg = JSON.parse(evt.data).message || msg; } catch (_) {}
    appendLogRow(msg, "type-flag");
    if (es.readyState === EventSource.CLOSED) {
      runBtn.disabled = false;
      followupForm.querySelector(".followup-btn").disabled = false;
      agentStatusTitle.textContent = "Agent Idle";
      agentStatusSub.textContent = "Connection ended";
    }
  });
}

// ---- Event handlers -----------------------------------------------------
function handleStepEvent({ step, status, label }) {
  const idx = STEP_ORDER.indexOf(step);
  if (idx === -1) return;
  const el = pipelineStepper.querySelector(`.step[data-step="${step}"]`);
  if (!el) return;

  if (status === "active") {
    STEP_ORDER.slice(0, idx).forEach((s) => {
      const prev = pipelineStepper.querySelector(`.step[data-step="${s}"]`);
      prev?.classList.remove("is-active");
      prev?.classList.add("is-done");
    });
    el.classList.add("is-active");
    agentStatusSub.textContent = label || step;
    telProgress.style.width = `${((idx + 0.5) / STEP_ORDER.length) * 100}%`;
  } else if (status === "done") {
    el.classList.remove("is-active");
    el.classList.add("is-done");
    telProgress.style.width = `${((idx + 1) / STEP_ORDER.length) * 100}%`;
  }
}

function handleLogEvent({ message, kind }) {
  appendLogRow(message, kind ? `type-${kind}` : "");
}

function appendLogRow(message, cls) {
  const row = document.createElement("div");
  row.className = "log-row";
  const time = new Date().toLocaleTimeString([], { hour12: false });
  row.innerHTML = `<span class="log-time">${time}</span><span class="log-msg ${cls || ""}">${escapeHtml(message)}</span>`;
  eventLog.appendChild(row);
  eventLog.scrollTop = eventLog.scrollHeight;
}

function handleTelemetryEvent({ hops, sources, flags }) {
  if (hops !== undefined) telHops.textContent = hops;
  if (sources !== undefined) telSources.textContent = sources;
  if (flags !== undefined) telFlags.textContent = flags;
}

function handleSourceEvent(source) {
  // Merge into the session-wide registry so citation popovers resolve
  // instantly during a live run (aggregate views pull fresh from the
  // database instead — see renderLibraryView etc.).
  sourceRegistry[source.id] = { ...sourceRegistry[source.id], ...source };
}

function handleDecompositionEvent(_data) {
  // No client-side bookkeeping needed — the Query Decomposer view reads
  // this straight from the database (/api/decompositions) once the run
  // finishes and gets persisted.
}

function handleContradictionEvent(_data) {
  // Same as above — the Contradiction Engine view reads from /api/contradictions.
}

function handleMetricsEvent({ faithfulness, latency_ms, tokens, cost_usd, sample }) {
  if (faithfulness !== undefined) metFaithfulness.textContent = `${Math.round(faithfulness * 100)}%`;
  if (latency_ms !== undefined) metLatency.textContent = `${(latency_ms / 1000).toFixed(1)}s`;
  if (tokens !== undefined) metTokens.textContent = formatCompact(tokens);
  if (cost_usd !== undefined) metCost.textContent = `$${cost_usd.toFixed(3)}`;
  if (sample !== undefined) {
    latencySamples.push(sample);
    if (latencySamples.length > 24) latencySamples.shift();
    drawSparkline();
  }
}

function handleReportEvent({ html }) {
  const block = document.createElement("div");
  block.className = "report-block";
  block.innerHTML = html;
  reportBody.appendChild(block);
  reportPanel.hidden = false;
  followupBar.hidden = false;

  wireCitationBadges(block);
  wireContradictionTags(block);

  const lead = block.querySelector(".report-lead");
  lastSummaryText = lead ? lead.textContent : "";

  block.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---- Recent Research (sidebar) — backed by the database, so it survives reloads ----
async function renderRecentList() {
  let runs;
  try { runs = await apiGet("/api/runs?limit=8"); }
  catch {
    recentList.innerHTML = '<li class="recent-empty">Couldn\u2019t reach the backend database.</li>';
    return;
  }

  if (!runs.length) {
    recentList.innerHTML = '<li class="recent-empty" id="recentEmpty">No research yet — run a query to see it here.</li>';
    return;
  }
  recentList.innerHTML = runs.map((r) => {
    const state = r.contradictions.length > 0 ? "warn" : "";
    return `
    <li class="recent-item" data-run-id="${r.id}">
      <span class="recent-dot" ${state ? `data-state="${state}"` : ""}></span>
      <div class="recent-text">
        <p>${escapeHtml(r.is_followup ? "↳ " + r.query : r.query)}</p>
        <time>${relativeTime(r.created_at)}</time>
      </div>
    </li>`;
  }).join("");

  recentList.querySelectorAll(".recent-item").forEach((el) => {
    el.addEventListener("click", () => reopenRun(el.dataset.runId));
  });
}

async function reopenRun(runId) {
  let run;
  try { run = await apiGet(`/api/runs/${runId}`); }
  catch { return; }

  switchView("home");
  homeView.hidden = false;
  heroSection.hidden = true;
  executionSection.hidden = false;
  runQueryTitle.textContent = run.query;
  agentStatusTitle.textContent = "Viewing Past Run";
  agentStatusSub.textContent = relativeTime(run.created_at);
  eventLog.innerHTML = "";
  resetPipelineSteps();
  pipelineStepper.querySelectorAll(".step").forEach((s) => s.classList.add("is-done"));
  telProgress.style.width = "100%";

  if (run.metrics) {
    metFaithfulness.textContent = run.metrics.faithfulness !== undefined ? `${Math.round(run.metrics.faithfulness * 100)}%` : "—";
    metLatency.textContent = run.metrics.latency_ms !== undefined ? `${(run.metrics.latency_ms / 1000).toFixed(1)}s` : "—";
    metTokens.textContent = run.metrics.tokens !== undefined ? formatCompact(run.metrics.tokens) : "—";
    metCost.textContent = run.metrics.cost_usd !== undefined ? `$${run.metrics.cost_usd.toFixed(3)}` : "—";
  }
  telSources.textContent = run.source_ids.length;
  telFlags.textContent = run.contradictions.length;
  telHops.textContent = run.hop_count;

  reportBody.innerHTML = "";
  const block = document.createElement("div");
  block.className = "report-block";
  block.innerHTML = run.report_html;
  reportBody.appendChild(block);
  reportPanel.hidden = false;
  followupBar.hidden = false;
  lastSummaryText = run.summary_text || "";

  wireCitationBadges(block);
  wireContradictionTags(block);
}

// ---- Citation popovers ----------------------------------------------------
function wireCitationBadges(scopeEl) {
  scopeEl.querySelectorAll(".cite-badge").forEach((badge) => {
    badge.addEventListener("click", (e) => {
      e.stopPropagation();
      const id = badge.dataset.sourceId;
      const src = sourceRegistry[id];
      if (!src) return;
      openCitationPopover(badge, src);
    });
  });
}

function openCitationPopover(anchorEl, src) {
  cpSnippet.textContent = src.snippet || "No snippet available.";
  cpSourceType.textContent = src.sourceType || "Source";
  cpReliability.textContent = `${(src.reliability || "medium").toUpperCase()} CONFIDENCE`;
  cpReliability.dataset.level = src.reliability || "medium";
  cpLink.href = src.url || "#";

  const rect = anchorEl.getBoundingClientRect();
  citationPopover.hidden = false;
  const popRect = citationPopover.getBoundingClientRect();
  let top = rect.bottom + 8 + window.scrollY;
  let left = rect.left + window.scrollX;
  if (left + popRect.width > window.innerWidth - 12) left = window.innerWidth - popRect.width - 12;
  citationPopover.style.top = `${top}px`;
  citationPopover.style.left = `${left}px`;
}

cpClose.addEventListener("click", () => { citationPopover.hidden = true; });
document.addEventListener("click", (e) => {
  if (!citationPopover.hidden && !citationPopover.contains(e.target)) {
    citationPopover.hidden = true;
  }
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") citationPopover.hidden = true;
});

// ---- Contradiction accordions ----------------------------------------------
function wireContradictionTags(scopeEl) {
  scopeEl.querySelectorAll(".contradiction-tag").forEach((tag) => {
    tag.addEventListener("click", () => tag.classList.toggle("is-open"));
  });
}

// ---- Sparkline ------------------------------------------------------------
function drawSparkline() {
  const w = 240, h = 56;
  if (latencySamples.length < 2) {
    sparkLine.setAttribute("d", "");
    sparkArea.setAttribute("d", "");
    return;
  }
  const min = Math.min(...latencySamples);
  const max = Math.max(...latencySamples);
  const range = max - min || 1;
  const step = w / (latencySamples.length - 1);

  const points = latencySamples.map((v, i) => {
    const x = i * step;
    const y = h - ((v - min) / range) * (h - 8) - 4;
    return [x, y];
  });

  const linePath = points.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
  const areaPath = `${linePath} L${w},${h} L0,${h} Z`;

  sparkLine.setAttribute("d", linePath);
  sparkArea.setAttribute("d", areaPath);
}

// ---- Command bar (Ctrl+K) --------------------------------------------------
document.getElementById("commandBarBtn")?.addEventListener("click", () => {
  switchView("home");
  resetToHero();
  queryInput.focus();
});
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    switchView("home");
    resetToHero();
    queryInput.focus();
  }
});

// ---- Utilities --------------------------------------------------------
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

function escapeAttr(str) {
  return (str ?? "").replace(/"/g, "&quot;");
}

function formatCompact(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return String(n);
}

function relativeTime(date) {
  const diffMs = Date.now() - new Date(date).getTime();
  const mins = Math.round(diffMs / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.round(hrs / 24);
  return `${days}d ago`;
}

// ---- Initial render ---------------------------------------------------
renderRecentList();
