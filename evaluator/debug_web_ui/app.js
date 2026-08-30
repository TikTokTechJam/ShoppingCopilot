const $ = (id) => document.getElementById(id);

const esc = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;");

const json = (value) => esc(JSON.stringify(value ?? {}, null, 2));
const score = (value) => value == null ? "N/A" : Number(value).toFixed(3);

function showError(message) {
  $("error").textContent = message;
  $("error").classList.remove("hidden");
}

function clearError() {
  $("error").classList.add("hidden");
}

async function api(path, method = "GET", body = undefined) {
  const options = { method, headers: {} };
  if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}

function canonicalSurface(value) {
  return String(value ?? "").toLowerCase().replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();
}

function similarityFor(similarities, field, value) {
  if (!similarities || typeof similarities !== "object" || Array.isArray(similarities)) return null;
  const wanted = canonicalSurface(value);
  const prefix = `${field}:`;
  const entry = Object.entries(similarities).find(([id]) => id.startsWith(prefix) && canonicalSurface(id.slice(prefix.length)) === wanted);
  if (!entry) return null;
  const numeric = Number(entry[1]);
  return Number.isFinite(numeric) ? numeric : null;
}

function constraintChip(field, item, similarities) {
  const itemText = typeof item === "object" ? JSON.stringify(item) : String(item ?? "");
  const match = similarityFor(similarities, field, itemText);
  const rawTitle = match == null ? itemText : `${itemText} · similarity ${String(match)}`;
  const displayedScore = match == null ? "" : ` <span class="chip-score">${score(match)}</span>`;
  return `<span class="chip" title="${esc(rawTitle)}">${esc(itemText)}${displayedScore}</span>`;
}

function chips(values, similarities = undefined) {
  if (!values || typeof values !== "object" || !Object.keys(values).length) return '<span class="muted">none</span>';
  const scoreLookup = similarities || values.similarities || {};
  const entries = Object.entries(values).filter(([key]) => key !== "similarities");
  if (!entries.length) return '<span class="muted">none</span>';
  return `<div class="constraint-list">${entries.map(([key, value]) => {
    const items = Array.isArray(value) ? value : [value];
    const rendered = items.length ? items.map(item => constraintChip(key, item, scoreLookup)).join("") : '<span class="muted">none</span>';
    return `<div class="constraint-row"><span class="constraint-label">${esc(key)}</span><div class="constraint-values">${rendered}</div></div>`;
  }).join("")}</div>`;
}

function similarityRows(similarities) {
  const entries = Array.isArray(similarities)
    ? similarities.map(item => [item.canonical_id || "—", item.confidence ?? item.similarity])
    : Object.entries(similarities || {});
  if (!entries.length) return '<span class="muted">none</span>';
  return `<div class="similarity-list">${entries.map(([canonicalId, value]) => `
    <div class="similarity-row">
      <code class="similarity-id" title="${esc(canonicalId)}">${esc(canonicalId)}</code>
      <strong title="${esc(value)}">${score(value)}</strong>
    </div>`).join("")}</div>`;
}

function bm25Details(details) {
  if (!details || !details.available) return "";
  const raw = details.raw || {};
  const groups = Object.entries(details.constraints || {});
  const rows = groups.map(([field, item]) => `
    <div class="bm25-query-row">
      <span class="constraint-label">${esc(field)}</span>
      <code title="${esc(item.expression || item.query || "")}">${esc(item.query || "")}</code>
      <span>target #${item.target_rank ?? "MISS"}</span>
      <strong>${item.target_score == null ? "N/A" : score(item.target_score)}</strong>
    </div>`).join("");
  return `<details class="bm25-details"><summary>BM25 query details</summary>
    <div class="bm25-query-row raw"><span class="constraint-label">raw</span>
      <code title="${esc(raw.expression || raw.query || "")}">${esc(raw.query || "")}</code>
      <span>target #${raw.target_rank ?? "MISS"}</span>
      <strong>${raw.target_score == null ? "N/A" : score(raw.target_score)}</strong>
    </div>
    ${rows || '<div class="muted">No constraint queries were active.</div>'}
  </details>`;
}

function renderBanner(data) {
  $("session-id").placeholder = data.evaluator === "local"
    ? "public_0001"
    : "manual400_0095";
  const session = data.session;
  if (!session) {
    $("session-banner").className = "panel empty";
    $("session-banner").textContent = data.interactive_mode
      ? "Interactive mode is active. Choose a target and enter replies in the terminal."
      : "Choose a session to begin.";
    return;
  }
  const target = session.target || {};
  $("session-banner").className = "panel session-banner";
  $("session-banner").innerHTML = `
    <div><span class="eyebrow">SESSION</span><strong>${esc(session.session_id)}</strong>
      <span class="badge">${esc(session.scenario)}</span></div>
    <div>Turn <strong>${session.turn}</strong> / ${session.total_turns}</div>
    <div class="banner-target">Target <code>${esc(target.parent_asin)}</code>${target.title ? `<span class="banner-target-title" title="${esc(target.title)}"> — ${esc(target.title)}</span>` : ""}</div>`;
}

function renderMissSearch(data) {
  const search = data.miss_search;
  const element = $("miss-search-status");
  element.className = "muted search-status";
  if (!search) {
    element.textContent = "Find Next Miss runs unseen sessions and stops on the first non-hit case.";
    return;
  }
  if (search.status === "found") {
    element.className = "search-status found";
    element.textContent = `${search.message} Searched ${search.searched} session${search.searched === 1 ? "" : "s"}; loaded ${search.sample_id}.`;
    return;
  }
  element.className = "search-status exhausted";
  element.textContent = `${search.message} Searched ${search.searched} session${search.searched === 1 ? "" : "s"} (${search.scenario}).`;
}

function renderState(data) {
  const state = data.state;
  if (!state) { $("state").innerHTML = "—"; return; }
  const canonical = data.layer2 || {};
  const productDense = data.product_dense || {};
  const benchmark = data.benchmark || {};
  const metrics = benchmark.metrics || {};
  const evaluatorLabel = data.evaluator === "local"
    ? "Local evaluator score"
    : "Hard evaluator score";
  const interactiveHint = data.interactive_mode
    ? '<div class="muted interactive-hint">Console input is active; this page updates after each reply.</div>'
    : "";
  $("state").innerHTML = `
    ${interactiveHint}
    <div class="kv"><span>Mode</span><b>${esc(state.mode || "—")}</b></div>
    <div class="kv"><span>Last asked</span><b>${esc(state.last_asked || "—")}</b></div>
    <div class="kv"><span>Clarification cycle</span><b>${esc(state.clarification_cycle ?? 1)}</b></div>
    <div class="kv"><span>Ask counts</span><b><code>${esc(json(state.attribute_call_count || {}))}</code></b></div>
    <div class="kv"><span>No preference</span><b>${esc((state.no_preference_attributes || []).join(", ") || "—")}</b></div>
    <div class="kv"><span>Clarification stopped</span><b>${state.clarification_stopped ? "YES" : "NO"}</b></div>
    <h3>Structured constraints</h3>
    <div>${chips(state.constraints)}</div>
    <h3>BGE canonical expansions</h3>
    <div>${chips(state.semantic_constraints)}</div>
    <h3>Canonical similarities</h3>
    <div>${similarityRows(state.semantic_constraints?.similarities || state.semantic_evidence || [])}</div>
    <h3>Excluded recommendations (${(state.excluded || []).length})</h3>
    <details><summary>show IDs</summary><pre>${json(state.excluded || [])}</pre></details>
    <h3>BGE canonical expansion</h3>
    <div class="${canonical.available ? "ok" : "warning"}">${canonical.available ? `Available · ${esc(canonical.model || "BGE")}${canonical.dimension ? ` · ${esc(canonical.dimension)}d` : ""}` : `Unavailable: ${esc(canonical.reason)}`}</div>
    <h3>Browsing product vectors</h3>
    <div class="${productDense.available ? "ok" : "warning"}">${productDense.available ? `Available · ${esc(productDense.model || "Qwen")}${productDense.dimension ? ` · ${esc(productDense.dimension)}d` : ""}${productDense.products ? ` · ${Number(productDense.products).toLocaleString()} products` : ""}` : `Unavailable: ${esc(productDense.reason || "V5 product-card index is unavailable")}`}</div>
    <h3>BM25 lexical search</h3>
    <div class="${data.bm25?.available ? "ok" : "warning"}">${data.bm25?.available ? `Available · ${Number(data.bm25.indexed_products || 0).toLocaleString()} products${data.bm25.build_seconds == null ? "" : ` · ${Number(data.bm25.build_seconds).toFixed(1)}s`}` : `Unavailable: ${esc(data.bm25?.reason || "Initialization failed")}`}</div>
    <h3>${evaluatorLabel}</h3>
    <div class="kv"><span>HitRate@10</span><b>${score(metrics.hit_rate_at_10)}</b></div>
    <div class="kv"><span>MRR</span><b>${score(metrics.mrr)}</b></div>
    <div class="kv"><span>MTTC</span><b>${score(metrics.mttc)}</b></div>
    <div class="kv"><span>Efficiency</span><b>${score(metrics.efficiency)}</b></div>
    <div class="kv"><span>TechnicalScore</span><b>${score(metrics.technical_score)}</b></div>
    <div class="muted">${benchmark.complete ? "Final session score" : "Provisional until session completes"}</div>`;
}

function renderDiagnostics(data) {
  const turns = data.turns || [];
  const turn = turns[turns.length - 1];
  if (!turn) { $("diagnostics").innerHTML = "—"; return; }
  const r = turn.ranking || {};
  const override = turn.override || {};
  const rankFallback = r.eligible ? "MISS" : "INELIGIBLE";
  const status = turn.hit ? `<span class="hit">HIT @ #${r.top10?.find(x => x.target)?.rank || "?"}</span>`
    : turn.pre_override_hit ? '<span class="pre-hit">PRE-OVERRIDE HIT — NOT SCOREABLE</span>'
    : '<span class="miss">MISS</span>';
  const top10 = (r.top10 || []).map(item => `
    <tr class="${item.target ? "target-row" : ""}">
      <td>${item.rank}</td><td><code>${esc(item.parent_asin)}</code></td>
      <td><span class="table-title" title="${esc(item.title || "")}">${esc(item.title || "")}</span></td><td>${score(item.structured_score)}</td>
      <td>${score(item.canonical_score ?? item.semantic_score)}</td><td>${score(item.dense_score)}</td>
      <td>${score(item.bm25_score)}</td><td>${score(item.fusion_score)}</td><td>${score(item.mmr_score)}</td><td>${score(item.final_score)}</td>
    </tr>`).join("");
  $("diagnostics").innerHTML = `
    <div class="status">${status}</div>
    <div class="rank-grid">
      <div><span>Structured</span><b>${r.structured_rank ?? rankFallback}</b></div>
      <div><span>Canonical</span><b>${r.canonical_rank ?? "N/A"}</b></div>
      <div><span>Product dense</span><b>${r.dense_rank ?? "N/A"}</b></div>
      <div><span>BM25</span><b>${r.bm25_rank ?? "N/A"}</b></div>
      <div><span>Hybrid</span><b>${r.hybrid_rank ?? rankFallback}</b></div>
    </div>
    <div class="score-grid">
      <div><span>Structured</span><strong title="${esc(r.structured_score ?? "N/A")}">${score(r.structured_score)}</strong></div>
      <div><span>Canonical BGE</span><strong title="${esc(r.canonical_score ?? r.semantic_score ?? "N/A")}">${score(r.canonical_score ?? r.semantic_score)}</strong></div>
      <div><span>Product dense</span><strong title="${esc(r.dense_score ?? "N/A")}">${score(r.dense_score)}</strong></div>
      <div><span>BM25 lexical</span><strong title="${esc(r.bm25_score ?? "N/A")}">${score(r.bm25_score)}</strong></div>
      <div><span>RRF</span><strong title="${esc(r.fusion_score ?? "N/A")}">${score(r.fusion_score)}</strong></div>
      <div><span>MMR</span><strong title="${esc(r.mmr_score ?? "N/A")}">${score(r.mmr_score)}</strong></div>
      <div><span>Final</span><strong title="${esc(r.final_score ?? "N/A")}">${score(r.final_score)}</strong></div>
    </div>
    <h3>Top 10 (reranker order)</h3>
    <div class="table-wrap"><table class="ranking-table"><thead><tr><th>#</th><th>ASIN</th><th>Title</th><th>Struct.</th><th>Canonical</th><th>Dense</th><th>BM25</th><th>RRF</th><th>MMR</th><th>Final</th></tr></thead><tbody>${top10 || "<tr><td colspan=10>none</td></tr>"}</tbody></table></div>
    ${bm25Details(r.bm25_debug)}
    <h3>Override</h3>
    <div class="${override.detected ? "override" : "muted"}">${override.detected ? `INTENT OVERRIDE: ${esc(override.kind)}` : "No override"}</div>`;
}

function renderTarget(data) {
  const target = data.session?.target || {};
  if (!target.parent_asin) { $("target").innerHTML = "—"; return; }
  $("target").innerHTML = `
    <div class="target-title" title="${esc(target.title || "Untitled")}">${esc(target.title || "Untitled")}</div>
    <div><code>${esc(target.parent_asin)}</code> · ${target.price == null ? "price N/A" : `$${Number(target.price).toFixed(2)}`}</div>
    <h3>Taxonomy</h3><div>${chips({category: target.taxonomy || []})}</div>
    <h3>Canonical facts</h3><div class="target-facts">${chips(target.facts || {})}</div>`;
}

function renderConversation(data) {
  const turns = data.turns || [];
  if (!turns.length) { $("conversation").className = "conversation empty"; $("conversation").textContent = "No turns executed."; return; }
  $("conversation").className = "conversation";
  $("conversation").innerHTML = turns.map(turn => {
    const state = turn.state || {};
    const override = turn.override || {};
    const status = turn.hit ? `<span class="hit">HIT</span>` : turn.pre_override_hit ? '<span class="pre-hit">PRE-OVERRIDE HIT — NOT SCOREABLE</span>' : '<span class="miss">MISS</span>';
    const overrideBox = override.detected ? `<div class="override-box">INTENT OVERRIDE DETECTED · ${esc(override.kind)}<br><small>Old mode: ${esc(override.old_mode || "—")} → New mode: ${esc(override.new_mode || "—")}</small></div>` : "";
    return `<article class="turn-card"><div class="turn-heading"><h3>Turn ${turn.turn}</h3>${status}</div>
      ${overrideBox}<div class="message user"><b>User</b><p>${esc(turn.user_message)}</p></div>
      <div class="message agent"><b>Agent</b><p>${esc(turn.agent?.message || "")}</p><small>Asked: ${esc(turn.agent?.ask_attribute || "—")}</small></div>
      <h4>Structured extracted this turn</h4><div>${chips(state.extracted_this_turn?.structured || {})}</div>
      <h4>BGE canonical expansions this turn</h4><div>${chips(state.extracted_this_turn?.semantic || {}, state.semantic_constraints?.similarities)}</div>
      <h4>Accumulated structured constraints</h4><div>${chips(state.constraints)}</div>
      <h4>Accumulated BGE canonical expansions</h4><div>${chips(state.semantic_constraints || {}, state.semantic_constraints?.similarities)}</div>
      <h4>Retrieval query text</h4><details><summary>show query</summary><p class="query">${esc(state.retrieval_query_text || state.query_text || "")}</p></details>
      <div class="turn-meta">Cycle: ${esc(state.clarification_cycle ?? 1)} · Exclusions: ${(state.exclusions || []).length} · Next asked: ${esc(turn.clarification?.next_asked || "—")}</div></article>`;
  }).join("");
}

let interactivePoll = null;
let lastData = null;
let pendingAction = null;

const actionLabels = {
  random: "New Random Session",
  "find-miss": "Find Next Miss",
  load: "Load",
  next: "Next Turn",
  "run-end": "Run To End",
};

function syncActionButtons() {
  const data = lastData || {};
  const active = !data.interactive_mode && Boolean(data.session) && !data.done;
  const busy = pendingAction !== null;
  $("next").disabled = busy || !active;
  $("run-end").disabled = busy || !active;
  $("random").disabled = busy || Boolean(data.interactive_mode);
  $("find-miss").disabled = busy || Boolean(data.interactive_mode);
  $("load").disabled = busy || Boolean(data.interactive_mode);
}

function setActionBusy(action, message) {
  pendingAction = action;
  Object.keys(actionLabels).forEach((id) => {
    const button = $(id);
    button.disabled = true;
    button.setAttribute("aria-busy", id === action ? "true" : "false");
    if (id === action) button.textContent = message;
  });
  $("action-status").textContent = message;
}

function clearActionBusy() {
  pendingAction = null;
  Object.entries(actionLabels).forEach(([id, label]) => {
    const button = $(id);
    button.textContent = label;
    button.removeAttribute("aria-busy");
  });
  $("action-status").textContent = "";
  syncActionButtons();
}

function updateInteractivePolling(data) {
  if (data.interactive_mode && interactivePoll === null) {
    interactivePoll = setInterval(() => loadState(() => api("/api/state")), 1000);
  } else if (!data.interactive_mode && interactivePoll !== null) {
    clearInterval(interactivePoll);
    interactivePoll = null;
  }
}

function render(data) {
  lastData = data;
  const evaluator = data.evaluator || "hard";
  $("eyebrow").textContent = evaluator === "local"
    ? "LOCAL EVALUATOR TOOL"
    : evaluator === "interactive"
      ? "INTERACTIVE DEBUG TOOL"
      : "HARD EVALUATOR TOOL";
  $("subtitle").textContent = evaluator === "local"
    ? "One real Agent turn at a time — public-set local evaluator mode."
    : evaluator === "interactive"
      ? "One real Agent turn at a time — replies entered in the console."
      : "One real Agent turn at a time — Manual400 benchmark mode.";
  renderBanner(data); renderMissSearch(data); renderState(data); renderDiagnostics(data); renderTarget(data); renderConversation(data);
  syncActionButtons();
  updateInteractivePolling(data);
}

async function loadState(request, action = null, message = "Working…") {
  if (pendingAction !== null) return;
  clearError();
  if (action) setActionBusy(action, message);
  try { render(await request()); } catch (error) { showError(error.message); }
  finally { if (action) clearActionBusy(); }
}

$("random").onclick = () => loadState(() => api("/api/session/random", "POST", {scenario: $("scenario").value}), "random", "Loading session…");
$("find-miss").onclick = () => loadState(() => api("/api/session/find-next-miss", "POST", {scenario: $("scenario").value}), "find-miss", "Searching for a miss…");
$("load").onclick = () => {
  const sessionId = $("session-id").value.trim();
  if (!sessionId) {
    showError("Enter a session ID before selecting Load.");
    return;
  }
  loadState(() => api("/api/session/load", "POST", {session_id: sessionId}), "load", "Loading session…");
};
$("next").onclick = () => loadState(() => api("/api/session/next", "POST"), "next", "Running turn…");
$("run-end").onclick = () => loadState(() => api("/api/session/run-to-end", "POST"), "run-end", "Running session…");

async function initialLoad() {
  try {
    const data = await api("/api/state");
    render(data);
    if (!data.interactive_mode && !data.session) {
      await loadState(() => api("/api/session/random", "POST", {scenario: "ANY"}), "random", "Loading session…");
    }
  } catch (error) {
    showError(error.message);
  }
}

initialLoad();
