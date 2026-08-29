const $ = (id) => document.getElementById(id);

const esc = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;");

const json = (value) => esc(JSON.stringify(value ?? {}, null, 2));
const score = (value) => value == null ? "N/A" : Number(value).toFixed(4);

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

function chips(values) {
  if (!values || !Object.keys(values).length) return '<span class="muted">none</span>';
  return Object.entries(values).map(([key, value]) => {
    const rendered = typeof value === "object" ? JSON.stringify(value) : value;
    return `<span class="chip"><b>${esc(key)}</b>: ${esc(rendered)}</span>`;
  }).join(" ");
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
    <div>Target <code>${esc(target.parent_asin)}</code>${target.title ? ` — ${esc(target.title)}` : ""}</div>`;
}

function renderState(data) {
  const state = data.state;
  if (!state) { $("state").innerHTML = "—"; return; }
  const layer2 = data.layer2 || {};
  const benchmark = data.benchmark || {};
  const metrics = benchmark.metrics || {};
  $("state").innerHTML = `
    <div class="kv"><span>Mode</span><b>${esc(state.mode || "—")}</b></div>
    <div class="kv"><span>Last asked</span><b>${esc(state.last_asked || "—")}</b></div>
    <h3>Structured constraints</h3>
    <div>${chips(state.constraints)}</div>
    <h3>Dense semantic constraints</h3>
    <div>${chips(state.semantic_constraints)}</div>
    <h3>Excluded recommendations (${(state.excluded || []).length})</h3>
    <details><summary>show IDs</summary><pre>${json(state.excluded || [])}</pre></details>
    <h3>Layer 2</h3>
    <div class="${layer2.available ? "ok" : "warning"}">${layer2.available ? "Available" : `Unavailable: ${esc(layer2.reason)}`}</div>
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
      <td>${esc(item.title || "")}</td><td>${score(item.structured_score)}</td>
      <td>${score(item.dense_score)}</td><td>${score(item.final_score)}</td>
    </tr>`).join("");
  $("diagnostics").innerHTML = `
    <div class="status">${status}</div>
    <div class="rank-grid">
      <div><span>Structured</span><b>${r.structured_rank ?? rankFallback}</b></div>
      <div><span>Dense</span><b>${r.dense_rank ?? "N/A"}</b></div>
      <div><span>Hybrid</span><b>${r.hybrid_rank ?? rankFallback}</b></div>
    </div>
    <div class="score-line">Structured ${score(r.structured_score)} · Dense semantic ${score(r.dense_score)} · Final ${score(r.final_score)}</div>
    <h3>Top 10 (reranker order)</h3>
    <div class="table-wrap"><table><thead><tr><th>#</th><th>ASIN</th><th>Title</th><th>Struct.</th><th>Dense</th><th>Final</th></tr></thead><tbody>${top10 || "<tr><td colspan=6>none</td></tr>"}</tbody></table></div>
    <h3>Override</h3>
    <div class="${override.detected ? "override" : "muted"}">${override.detected ? `INTENT OVERRIDE: ${esc(override.kind)}` : "No override"}</div>`;
}

function renderTarget(data) {
  const target = data.session?.target || {};
  if (!target.parent_asin) { $("target").innerHTML = "—"; return; }
  $("target").innerHTML = `
    <div class="target-title">${esc(target.title || "Untitled")}</div>
    <div><code>${esc(target.parent_asin)}</code> · ${target.price == null ? "price N/A" : `$${Number(target.price).toFixed(2)}`}</div>
    <h3>Taxonomy</h3><div>${chips({category: target.taxonomy || []})}</div>
    <h3>Canonical facts</h3><div>${chips(target.facts || {})}</div>`;
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
      <h4>Semantic extracted this turn</h4><div>${chips(state.extracted_this_turn?.semantic || {})}</div>
      <h4>Accumulated structured constraints</h4><div>${chips(state.constraints)}</div>
      <h4>Accumulated dense semantic constraints</h4><div>${chips(state.semantic_constraints || {})}</div>
      <h4>Query text</h4><details><summary>show query</summary><p class="query">${esc(state.query_text || "")}</p></details>
      <div class="turn-meta">Exclusions: ${(state.exclusions || []).length} · Next asked: ${esc(turn.clarification?.next_asked || "—")}</div></article>`;
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
