const API_URL = "https://organic-space-carnival-x594x6jg76vpf6p5p-7860.app.github.dev";
const ROUNDS = { fast: 0, standard: 3, large: 8 };
const COLORS = { positive: "#22c55e", neutral: "#94a3b8", negative: "#ef4444" };
const LABELS = { positive: "Positif", neutral: "Neutre", negative: "Négatif" };

const $ = (id) => document.getElementById(id);

let predictions = [];
let lastData = null;
let currentFilter = "all";

function showStatus(type, message) {
  const el = $("status");
  el.className = `status ${type}`;
  el.textContent = message;
}

function hideStatus() {
  $("status").className = "status hidden";
}

function setTheme(dark) {
  document.body.classList.toggle("dark", dark);
  $("themeBtn").textContent = dark ? "☀️" : "🌙";
  chrome.storage.local.set({ theme: dark ? "dark" : "light" });
}

function getActiveTab() {
  return new Promise((resolve) => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => resolve(tabs[0]));
  });
}

function askContentScript(tabId, depth) {
  return new Promise((resolve, reject) => {
    chrome.tabs.sendMessage(
      tabId,
      { action: "getComments", scrollRounds: ROUNDS[depth], limit: 1000 },
      (response) => {
        if (chrome.runtime.lastError || !response) {
          reject(new Error("Impossible de lire la page. Recharge l'onglet YouTube (F5) puis réessaie."));
          return;
        }
        resolve(response);
      }
    );
  });
}

async function callApi(comments) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 60000);
  try {
    const response = await fetch(`${API_URL}/predict_batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments }),
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(`Erreur de l'API (code ${response.status}).`);
    }
    return await response.json();
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("L'API met trop de temps à répondre. Réessaie dans un instant.");
    }
    if (error instanceof TypeError) {
      throw new Error("API injoignable. Vérifie que le Codespace est allumé et le port 7860 public.");
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

function renderSummary(stats) {
  const pos = stats.positive_pct;
  const neu = stats.neutral_pct;
  const chart = $("chart");
  chart.style.background = `conic-gradient(
    ${COLORS.positive} 0% ${pos}%,
    ${COLORS.neutral} ${pos}% ${pos + neu}%,
    ${COLORS.negative} ${pos + neu}% 100%)`;
  chart.setAttribute("data-total", stats.total);

  const legend = $("legend");
  legend.innerHTML = "";
  ["positive", "neutral", "negative"].forEach((key) => {
    const row = document.createElement("div");
    row.className = "legend-row";

    const dot = document.createElement("span");
    dot.className = "dot";
    dot.style.background = COLORS[key];

    const name = document.createElement("span");
    name.textContent = LABELS[key];

    const value = document.createElement("span");
    value.className = "value";
    value.textContent = `${stats[key]} (${stats[`${key}_pct`]}%)`;

    row.append(dot, name, value);
    legend.appendChild(row);
  });
}

function renderList() {
  const list = $("list");
  list.innerHTML = "";
  const shown = predictions.filter((p) => currentFilter === "all" || p.sentiment === currentFilter);

  if (shown.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "Aucun commentaire dans cette catégorie.";
    list.appendChild(empty);
    return;
  }

  shown.forEach((p) => {
    const item = document.createElement("div");
    item.className = "item";

    const head = document.createElement("div");
    head.className = "item-head";

    const badge = document.createElement("span");
    badge.className = `badge ${p.sentiment}`;
    badge.textContent = LABELS[p.sentiment];

    const confidence = document.createElement("span");
    confidence.className = "confidence";
    confidence.textContent = `confiance ${Math.round(p.confidence * 100)}%`;

    head.append(badge, confidence);

    const text = document.createElement("p");
    text.textContent = p.text;

    item.append(head, text);
    list.appendChild(item);
  });
}

function renderResults(data) {
  renderSummary(data.stats);
  $("meta").textContent =
    `Confiance moyenne : ${Math.round(data.stats.average_confidence * 100)}% · ` +
    `Traitement API : ${data.processing_time_ms} ms`;
  renderList();
  $("results").classList.remove("hidden");
}

async function analyze() {
  $("results").classList.add("hidden");
  $("analyzeBtn").disabled = true;
  showStatus("loading", "Extraction des commentaires...");

  try {
    const tab = await getActiveTab();
    if (!tab || !tab.url || !tab.url.startsWith("https://www.youtube.com/watch")) {
      throw new Error("Ouvre une vidéo YouTube (youtube.com/watch) puis réessaie.");
    }

    const reply = await askContentScript(tab.id, $("depth").value);
    if (reply.comments.length === 0) {
      throw new Error("Aucun commentaire trouvé. Descends jusqu'aux commentaires et réessaie.");
    }

    showStatus("loading", `Analyse de ${reply.comments.length} commentaires...`);
    lastData = await callApi(reply.comments);
    predictions = lastData.predictions;
    currentFilter = "all";
    document.querySelectorAll("#filters button").forEach((b) => {
      b.classList.toggle("active", b.dataset.filter === "all");
    });

    hideStatus();
    renderResults(lastData);
  } catch (error) {
    showStatus("error", error.message);
  } finally {
    $("analyzeBtn").disabled = false;
  }
}

function buildSummaryText() {
  const s = lastData.stats;
  const lines = [
    "Analyse de sentiment YouTube",
    `Total : ${s.total}`,
    `Positif : ${s.positive} (${s.positive_pct}%)`,
    `Neutre : ${s.neutral} (${s.neutral_pct}%)`,
    `Négatif : ${s.negative} (${s.negative_pct}%)`,
    "",
  ];
  predictions.forEach((p) => {
    lines.push(`[${LABELS[p.sentiment]} ${Math.round(p.confidence * 100)}%] ${p.text}`);
  });
  return lines.join("\n");
}

async function copyResults() {
  await navigator.clipboard.writeText(buildSummaryText());
  $("copyBtn").textContent = "Copié ✓";
  setTimeout(() => ($("copyBtn").textContent = "Copier"), 1500);
}

function exportCsv() {
  const escape = (value) => `"${String(value).replace(/"/g, '""')}"`;
  const rows = ["sentiment,confidence,text"];
  predictions.forEach((p) => rows.push(`${p.sentiment},${p.confidence},${escape(p.text)}`));
  const blob = new Blob(["﻿" + rows.join("\n")], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "sentiment_youtube.csv";
  link.click();
  URL.revokeObjectURL(link.href);
}

$("analyzeBtn").addEventListener("click", analyze);
$("copyBtn").addEventListener("click", copyResults);
$("exportBtn").addEventListener("click", exportCsv);
$("themeBtn").addEventListener("click", () => setTheme(!document.body.classList.contains("dark")));

$("filters").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  currentFilter = button.dataset.filter;
  document.querySelectorAll("#filters button").forEach((b) => b.classList.toggle("active", b === button));
  renderList();
});

chrome.storage.local.get("theme", (data) => setTheme(data.theme === "dark"));
