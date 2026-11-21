"use strict";
const input = document.getElementById("comment");
const form = document.getElementById("predict-form");
const button = document.getElementById("submit");
const panel = document.getElementById("result-panel");
const names = {CLEAN: "Sạch", OFFENSIVE: "Xúc phạm", HATE: "Thù ghét"};
input.addEventListener("input", () => {document.getElementById("counter").textContent = `${input.value.length} / 2000`;});
document.querySelectorAll("[data-example]").forEach(el => el.addEventListener("click", () => {
  input.value = el.dataset.example; input.dispatchEvent(new Event("input")); input.focus();
}));
form.addEventListener("submit", async event => {
  event.preventDefault();
  if (!input.value.trim()) {input.focus(); return;}
  button.disabled = true;
  panel.setAttribute("aria-busy", "true");
  document.getElementById("status").textContent = "Đang phân tích";
  document.getElementById("prediction").hidden = true;
  document.getElementById("empty-state").hidden = true;
  document.getElementById("error").hidden = true;
  try {
    const response = await fetch("/api/predict/", {
      method: "POST", headers: {"Content-Type": "application/json", "X-CSRFToken": form.querySelector("[name=csrfmiddlewaretoken]").value},
      body: JSON.stringify({text: input.value}), signal: AbortSignal.timeout(60000),
    });
    if (!(response.headers.get("content-type") || "").includes("application/json")) throw new Error("Máy chủ không trả về kết quả hợp lệ. Hãy tải lại trang.");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Không thể phân tích.");
    const label = document.getElementById("result-label");
    label.textContent = names[data.label] || data.label;
    label.className = `label-badge ${data.label.toLowerCase()}`;
    const hasScores = typeof data.confidence === "number";
    document.getElementById("confidence").textContent = hasScores ? `${(data.confidence * 100).toFixed(1)}%` : "";
    const explanation = document.getElementById("analysis-explanation");
    explanation.textContent = data.explanation || "";
    explanation.hidden = !data.explanation;
    document.getElementById("distribution-title").hidden = !hasScores;
    const bars = document.getElementById("probabilities");
    bars.replaceChildren();
    Object.entries(hasScores ? names : {}).forEach(([key, value]) => {
      const row = document.createElement("div"); row.className = "prob-row";
      const title = document.createElement("div"); title.className = "prob-title";
      const name = document.createElement("span"); name.textContent = value;
      const score = document.createElement("span"); score.textContent = `${(data.probabilities[key] * 100).toFixed(1)}%`;
      title.append(name, score);
      const bar = document.createElement("div"); bar.className = "bar";
      const fill = document.createElement("div"); fill.className = "bar-fill";
      fill.style.width = `${data.probabilities[key] * 100}%`;
      bar.append(fill); row.append(title, bar); bars.append(row);
    });
    document.getElementById("prediction").hidden = false;
    document.getElementById("status").textContent = "Hoàn tất";
  } catch (error) {
    const box = document.getElementById("error");
    box.textContent = error.name === "TimeoutError" ? "Quá thời gian chờ. Mô hình có thể đang được nạp, hãy thử lại." : error.message;
    box.hidden = false;
    document.getElementById("status").textContent = "Chưa có kết quả";
  } finally {
    button.disabled = false; panel.setAttribute("aria-busy", "false");
  }
});
fetch("/api/health/").then(r => r.json()).then(data => {
  document.getElementById("model-status").textContent = data.model_configured ? (data.backend === "remote" ? `Đã cấu hình API · ${data.model}` : "Đã tìm thấy cấu hình mô hình") : "Chưa cấu hình mô hình phân tích";
}).catch(() => {document.getElementById("model-status").textContent = "Không kết nối được máy chủ";});
