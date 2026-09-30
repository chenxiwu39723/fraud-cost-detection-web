"use strict";

const $ = (id) => document.getElementById(id);
const fmt = (n) => Math.round(n).toLocaleString("zh-Hant"); // 千分位

let costChart = null;
let rocChart = null;

// 放開滑桿後才送請求，避免連續發送
$("cfp").addEventListener("input", () => { $("cfpValue").textContent = $("cfp").value; });
$("cfp").addEventListener("change", refresh);
$("model").addEventListener("change", refresh);
$("sampleBtn").addEventListener("click", drawSample);

async function refresh() {
  const body = { model: $("model").value, c_fp: Number($("cfp").value) };
  let res;
  try {
    res = await fetch("/api/cost", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (e) {
    return;
  }
  const data = await res.json();
  if (!data.success) { alert(data.error || "請求失敗"); return; }
  renderCards(data);
  renderCostChart(data);
  renderRocChart(data);
  renderConfusion(data);
}

function renderCards(d) {
  $("cardThreshold").textContent = d.best_threshold.toFixed(3);
  $("cardCost").textContent = fmt(d.total_cost);
  const s = d.savings;
  const el = $("cardSavings");
  el.textContent = (s >= 0 ? "+" : "−") + fmt(Math.abs(s));
  el.style.color = s >= 0 ? "var(--good)" : "var(--bad)";
}

function renderCostChart(d) {
  const pts = d.cost_curve.t.map((t, i) => ({ x: t, y: d.cost_curve.cost[i] }));
  const cfg = {
    type: "line",
    data: {
      datasets: [
        { label: "總損失", data: pts, borderColor: "#2563eb", borderWidth: 2,
          pointRadius: 0, tension: 0.1 },
        { label: "t*", data: [
            { x: d.best_threshold, y: Math.min(...d.cost_curve.cost) },
            { x: d.best_threshold, y: Math.max(...d.cost_curve.cost) }],
          borderColor: "#dc2626", borderWidth: 1.5, borderDash: [6, 4],
          pointRadius: 0 },
      ],
    },
    options: {
      scales: {
        x: { type: "linear", title: { display: true, text: "門檻 t" }, min: 0, max: 1 },
        y: { title: { display: true, text: "總損失" } },
      },
      plugins: { legend: { display: true } },
    },
  };
  if (costChart) costChart.destroy();
  costChart = new Chart($("costChart"), cfg);
}

function renderRocChart(d) {
  const pts = d.roc_curve.fpr.map((f, i) => ({ x: f, y: d.roc_curve.tpr[i] }));
  const op = d.roc_curve.operating_point;
  const cfg = {
    type: "line",
    data: {
      datasets: [
        { label: "ROC", data: pts, borderColor: "#2563eb", borderWidth: 2,
          pointRadius: 0, tension: 0.1 },
        { label: "t* 操作點", data: [{ x: op.fpr, y: op.tpr }],
          borderColor: "#dc2626", backgroundColor: "#dc2626",
          pointRadius: 6, showLine: false },
        { label: "隨機", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
          borderColor: "#9ca3af", borderWidth: 1, borderDash: [4, 4], pointRadius: 0 },
      ],
    },
    options: {
      scales: {
        x: { type: "linear", title: { display: true, text: "FPR" }, min: 0, max: 1 },
        y: { title: { display: true, text: "TPR" }, min: 0, max: 1 },
      },
    },
  };
  if (rocChart) rocChart.destroy();
  rocChart = new Chart($("rocChart"), cfg);
}

function renderConfusion(d) {
  const cm = d.confusion_matrix;
  $("cmTN").textContent = fmt(cm.tn);
  $("cmFP").textContent = fmt(cm.fp);
  $("cmFN").textContent = fmt(cm.fn);
  $("cmTP").textContent = fmt(cm.tp);
  $("baseline").textContent =
    `基準比較 — 全部不攔：${fmt(d.cost_block_none)}　全部攔截：${fmt(d.cost_block_all)}　` +
    `t=0.5：${fmt(d.cost_at_0_5)}`;
}

async function drawSample() {
  const params = new URLSearchParams({ model: $("model").value, c_fp: $("cfp").value });
  const res = await fetch("/api/sample?" + params.toString());
  const d = await res.json();
  const box = $("sampleResult");
  box.hidden = false;
  if (!d.success) { box.textContent = d.error || "抽樣失敗"; return; }
  box.innerHTML =
    `<strong>金額：</strong>${fmt(d.amount)}<br>` +
    `<strong>真實標籤：</strong>${d.true_label === 1 ? "盜刷" : "正常"}<br>` +
    `<strong>預測機率：</strong>${d.prob.toFixed(4)}<br>` +
    `<strong>t*（${d.best_threshold.toFixed(3)}）判定：</strong>${d.flagged ? "攔截" : "放行"}<br>` +
    `<strong>結果：</strong>${d.outcome}`;
}

// 首次載入
refresh();
