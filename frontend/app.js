/**
 * VIGIL Frontend (L5 presentation layer)
 *
 * Thin client for the FastAPI server (L4). Performs a health check on load
 * and submits validated detections to the /analyze endpoint, rendering the
 * resulting RiskEvent and adjudication metadata.
 */

const API_BASE = (window.VIGIL_API_BASE || "http://localhost:8000").replace(/\/$/, "");

const els = {
  status: () => document.getElementById("backend-status"),
  backend: () => document.getElementById("backend-name"),
  input: () => document.getElementById("detections-input"),
  analyze: () => document.getElementById("analyze-btn"),
  score: () => document.getElementById("risk-score"),
  label: () => document.getElementById("risk-label"),
  card: () => document.getElementById("risk-card"),
  summary: () => document.getElementById("summary"),
  meta: () => document.getElementById("meta"),
  raw: () => document.getElementById("raw"),
};

function setStatus(text, ok) {
  const node = els.status();
  if (!node) return;
  node.textContent = text;
  node.dataset.state = ok ? "ok" : "error";
}

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    setStatus(`online \u00b7 ${data.status || "ok"}`, true);
    if (els.backend()) els.backend().textContent = data.backend || "unknown";
  } catch (err) {
    setStatus(`offline (${err.message})`, false);
    if (els.backend()) els.backend().textContent = "\u2014";
  }
}

function buildPayload() {
  const raw = (els.input() && els.input().value || "").trim();
  if (!raw) {
    return { detections: [], source: "frontend" };
  }
  return JSON.parse(raw);
}

function applyRiskClass(level) {
  const card = els.card();
  if (!card) return;
  card.classList.remove("low", "medium", "high");
  if (level) card.classList.add(level);
}

function riskLevel(score) {
  if (score >= 0.66) return "high";
  if (score >= 0.33) return "medium";
  return "low";
}

function render(data) {
  const event = data.event || data;
  const score = typeof event.risk_score === "number" ? event.risk_score : 0;
  const level = event.risk_label || riskLevel(score);

  if (els.score()) els.score().textContent = score.toFixed(2);
  if (els.label()) els.label().textContent = level.toUpperCase();
  applyRiskClass(level);

  if (els.summary()) {
    els.summary().textContent = event.summary || event.rationale || "No summary provided.";
  }

  const meta = els.meta();
  if (meta) {
    meta.innerHTML = "";
    const items = [
      ["backend", data.backend],
      ["fallback", data.fallback_used ? "yes" : "no"],
      ["sanitized", data.sanitized ? "yes" : "no"],
      ["detections", (event.detections || []).length],
    ];
    for (const [k, v] of items) {
      if (v === undefined || v === null) continue;
      const li = document.createElement("li");
      li.textContent = `${k}: ${v}`;
      meta.appendChild(li);
    }
  }

  if (els.raw()) els.raw().textContent = JSON.stringify(data, null, 2);
}

function renderError(message) {
  if (els.summary()) els.summary().textContent = `Error: ${message}`;
  if (els.raw()) els.raw().textContent = "";
  applyRiskClass(null);
}

async function analyze() {
  const btn = els.analyze();
  let payload;
  try {
    payload = buildPayload();
  } catch (err) {
    renderError(`Invalid JSON input: ${err.message}`);
    return;
  }

  if (btn) {
    btn.disabled = true;
    btn.textContent = "Analyzing...";
  }

  try {
    const res = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }
    render(data);
  } catch (err) {
    renderError(err.message);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Analyze";
    }
  }
}

function init() {
  checkHealth();
  const btn = els.analyze();
  if (btn) btn.addEventListener("click", analyze);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
