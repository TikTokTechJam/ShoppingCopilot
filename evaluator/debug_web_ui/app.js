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

function rrfContribution(rank, constant = 60) {
  const numericRank = Number(rank);
  const numericConstant = Number(constant);
  if (!Number.isFinite(numericRank) || numericRank <= 0 || !Number.isFinite(numericConstant)) return "N/A";
  return (1 / (numericConstant * numericRank)).toFixed(6);
}

function targetPhraseRank(targetRanks, phrase, constant) {
  if (targetRanks == null) return '<span class="muted">target rank unavailable</span>';
  if (!Object.prototype.hasOwnProperty.call(targetRanks, phrase) || targetRanks[phrase] == null) {
    return '<span class="muted">target not in results</span>';
  }
  const rank = Number(targetRanks[phrase]);
  return `<strong class="target-bm25-rank">target #${esc(rank)} <small>RRF ${rrfContribution(rank, constant)}</small></strong>`;
}

function bm25Fusion(debug, targetRanks = null) {
  if (!debug || !debug.bm25_available) {
    return '<span class="muted">BM25 fusion unavailable for this turn.</span>';
  }
  const phrases = Array.isArray(debug.phrases)
    ? debug.phrases
    : (Array.isArray(debug.constraints) ? debug.constraints : []);
  const fused = Array.isArray(debug.top_fused) ? debug.top_fused : [];
  const rankConstant = debug.fusion?.rank_constant ?? 60;
  const rows = phrases.length ? phrases.map(item => {
    const phrase = item.phrase || (item.original_phrases || []).join(", ") || "—";
    const expansions = (item.expansions || []).map(exp =>
      `<span class="chip" title="${esc(exp.attribute || "semantic")} · ${esc(exp.value)} · similarity ${esc(exp.similarity)}">${exp.attribute ? `<span class="chip-source">${esc(exp.attribute)}:</span> ` : ""}${esc(exp.value)} <span class="chip-score">${score(exp.similarity)}</span></span>`
    ).join("") || '<span class="muted">none</span>';
    const topRanks = (item.top_ranks || []).slice(0, 5).map(hit =>
      `${esc(hit.parent_asin)} #${esc(hit.rank)}`
    ).join(" · ");
    return `<div class="bm25-fusion-row">
      <div class="bm25-fusion-label"><strong>Phrase</strong><span title="${esc(phrase)}">${esc(phrase)}</span></div>
      <div class="constraint-values">${expansions}</div>
      <div class="bm25-target-rank">${targetPhraseRank(targetRanks, phrase, rankConstant)}</div>
      <small class="muted">ranks: ${topRanks || "none"}</small>
    </div>`;
  }).join("") : '<span class="muted">No active phrase queries.</span>';
  const fusedRows = fused.map((item, index) => {
    const ranks = Object.entries(item.phrase_ranks || item.constraint_ranks || {})
      .map(([phrase, rank]) => `${esc(phrase)} #${esc(rank)}`).join(" · ");
    return `<div class="bm25-fused-candidate"><span>#${index + 1} <code>${esc(item.parent_asin)}</code></span><strong>${score(item.final_score)}</strong><small>raw #${esc(item.raw_rank ?? "—")} · ${ranks || "no attribute hit"}</small></div>`;
  }).join("");
  return `<div class="bm25-fusion">
    <div class="kv"><span>Raw query</span><code title="${esc(debug.raw_bm25_query || "")}">${esc(debug.raw_bm25_query || "—")}</code></div>
    <div class="kv"><span>Raw results</span><b>${esc(debug.raw_bm25_rank_count ?? 0)}</b></div>
    <div class="kv"><span>Fusion</span><b>1 / (${esc(rankConstant)} × rank)</b></div>
    <div class="bm25-target-help muted">Target rank is shown for each phrase BM25 list; RRF is that phrase's contribution.</div>
    ${rows}
    <h4>Top fused candidates</h4>
    <div class="bm25-fused-list">${fusedRows || '<span class="muted">none</span>'}</div>
  </div>`;
}

function renderBanner(data) {
  const session = data.session;
  if (!session) {
    $("session-banner").className = "panel empty";
    $("session-banner").textContent = "Choose a session to begin.";
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

function renderState(data) {
  const state = data.state;
  if (!state) { $("state").innerHTML = "—"; return; }
  const layer2 = data.layer2 || {};
  const benchmark = data.benchmark || {};
  const metrics = benchmark.metrics || {};
  const turns = data.turns || [];
  const latestTurn = turns.length ? turns[turns.length - 1] : null;
  const targetRanks = latestTurn?.ranking?.target_phrase_bm25_ranks ?? null;
  $("state").innerHTML = `
    <div class="kv"><span>Mode</span><b>${esc(state.mode || "—")}</b></div>
    <div class="kv"><span>Last asked</span><b>${esc(state.last_asked || "—")}</b></div>
    <div class="kv"><span>Clarification cycle</span><b>${esc(state.clarification_cycle ?? 1)}</b></div>
    <div class="kv"><span>Ask counts</span><b><code>${esc(json(state.attribute_call_count || {}))}</code></b></div>
    <div class="kv"><span>No preference</span><b>${esc((state.no_preference_attributes || []).join(", ") || "—")}</b></div>
    <div class="kv"><span>Clarification stopped</span><b>${state.clarification_stopped ? "YES" : "NO"}</b></div>
    <h3>Structured constraints</h3>
    <div>${chips(state.constraints)}</div>
    <h3>Dense semantic constraints</h3>
    <div>${chips(state.semantic_constraints)}</div>
    <h3>Similarities</h3>
    <div>${similarityRows(state.semantic_constraints?.similarities || state.semantic_evidence || [])}</div>
    <h3>Excluded recommendations (${(state.excluded || []).length})</h3>
    <details><summary>show IDs</summary><pre>${json(state.excluded || [])}</pre></details>
    <h3>Layer 2</h3>
    <div class="${layer2.available ? "ok" : "warning"}">${layer2.available ? "Available" : `Unavailable: ${esc(layer2.reason)}`}</div>
    <h3>BM25 lexical search</h3>
    <div class="${data.bm25?.available ? "ok" : "warning"}">${data.bm25?.available ? `Available · ${Number(data.bm25.indexed_products || 0).toLocaleString()} products${data.bm25.build_seconds == null ? "" : ` · ${Number(data.bm25.build_seconds).toFixed(1)}s`}` : `Unavailable: ${esc(data.bm25?.reason || "Initialization failed")}`}</div>
    <h3>BM25 fusion details</h3>
    ${bm25Fusion(state.retrieval_debug, targetRanks)}
    <h3>Hard evaluator score</h3>
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
      <td>${score(item.dense_score)}</td><td>${score(item.bm25_score)}</td><td>${score(item.final_score)}</td>
    </tr>`).join("");
  $("diagnostics").innerHTML = `
    <div class="status">${status}</div>
    <div class="rank-grid">
      <div><span>Structured</span><b>${r.structured_rank ?? rankFallback}</b></div>
      <div><span>Dense</span><b>${r.dense_rank ?? "N/A"}</b></div>
      <div><span>BM25</span><b>${r.bm25_rank ?? "N/A"}</b></div>
      <div><span>Hybrid</span><b>${r.hybrid_rank ?? rankFallback}</b></div>
    </div>
    <div class="score-grid">
      <div><span>Structured</span><strong title="${esc(r.structured_score ?? "N/A")}">${score(r.structured_score)}</strong></div>
      <div><span>Dense semantic</span><strong title="${esc(r.dense_score ?? "N/A")}">${score(r.dense_score)}</strong></div>
      <div><span>BM25 lexical</span><strong title="${esc(r.bm25_score ?? "N/A")}">${score(r.bm25_score)}</strong></div>
      <div><span>Final</span><strong title="${esc(r.final_score ?? "N/A")}">${score(r.final_score)}</strong></div>
    </div>
    <h3>Top 10 (reranker order)</h3>
    <div class="table-wrap"><table class="ranking-table"><thead><tr><th>#</th><th>ASIN</th><th>Title</th><th>Struct.</th><th>Dense</th><th>BM25</th><th>Final</th></tr></thead><tbody>${top10 || "<tr><td colspan=7>none</td></tr>"}</tbody></table></div>
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
      <h4>Semantic extracted this turn</h4><div>${chips(state.extracted_this_turn?.semantic || {}, state.semantic_constraints?.similarities)}</div>
      <h4>Accumulated structured constraints</h4><div>${chips(state.constraints)}</div>
      <h4>Accumulated dense semantic constraints</h4><div>${chips(state.semantic_constraints || {}, state.semantic_constraints?.similarities)}</div>
      <h4>BM25 fusion details</h4><div>${bm25Fusion(state.retrieval_debug, turn.ranking?.target_phrase_bm25_ranks ?? null)}</div>
      <h4>Query text</h4><details><summary>show query</summary><p class="query">${esc(state.query_text || "")}</p></details>
      <div class="turn-meta">Cycle: ${esc(state.clarification_cycle ?? 1)} · Exclusions: ${(state.exclusions || []).length} · Next asked: ${esc(turn.clarification?.next_asked || "—")}</div></article>`;
  }).join("");
}

function render(data) {
  renderBanner(data); renderState(data); renderDiagnostics(data); renderTarget(data); renderConversation(data);
  const active = Boolean(data.session) && !data.done;
  $("next").disabled = !active;
  $("run-end").disabled = !active;
}

async function loadState(request) {
  clearError();
  try { render(await request()); } catch (error) { showError(error.message); }
}

$("random").onclick = () => loadState(() => api("/api/session/random", "POST", {scenario: $("scenario").value}));
$("load").onclick = () => loadState(() => api("/api/session/load", "POST", {session_id: $("session-id").value}));
$("next").onclick = () => loadState(() => api("/api/session/next", "POST"));
$("run-end").onclick = () => loadState(() => api("/api/session/run-to-end", "POST"));

// Select a session on first page load, but do not execute an Agent turn yet.
loadState(() => api("/api/session/random", "POST", {scenario: "ANY"}));
