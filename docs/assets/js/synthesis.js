(() => {
  "use strict";
  const $ = selector => document.querySelector(selector);
  const $$ = selector => [...document.querySelectorAll(selector)];
  const math = window.SynthesisMath;
  const vector = values => `[${values.map(x => x.toFixed(3)).join(", ")}]`;
  const farInput = $("#mixFarValue");
  function renderMix() {
    const parsed = Number(farInput.value);
    const value = Number.isFinite(parsed) ? Math.max(-2, Math.min(2, parsed)) : 1;
    farInput.value = String(value);
    const result = math.mix(value);
    $("#mixInputValue").textContent = value.toFixed(2);
    $("#mixVectors").textContent = result.vectors.map((v, i) => `x${i}=[${v.join(",")}]`).join(" · ");
    $("#mixWeights").innerHTML = result.vectors.map((_, i) => `<tr><th scope="row">x${i}${i === 2 ? " · Query" : ""}</th><td>${result.convWeights[i].toFixed(3)}</td><td>${result.attentionWeights[i].toFixed(3)}</td></tr>`).join("");
    $("#mixConvOutput").textContent = vector(result.convOutput);
    $("#mixAttentionOutput").textContent = vector(result.attentionOutput);
    $("#mixExplanation").textContent = `x₀의 Attention 계수는 ${result.attentionWeights[0].toFixed(3)}입니다. 세 칸 합성곱은 x₀와 연결되지 않아 출력 [0.250, 0.750]이 유지됩니다. Attention 계수의 전체 합은 ${result.attentionWeights.reduce((a, b) => a + b, 0).toFixed(3)}입니다.`;
  }
  if (farInput) {
    farInput.addEventListener("input", renderMix);
    $("#mixReset").addEventListener("click", () => {farInput.value = "1"; renderMix();});
    renderMix();
  }
  function renderCurve() {
    const data = math.curves[$("#failureScenario").value];
    const points = values => values.map((value, i) => `${60 + i * 106},${240 - value / 4 * 210}`).join(" ");
    $("#failureTrainLine").setAttribute("points", points(data.train));
    $("#failureValLine").setAttribute("points", points(data.val));
    $("#failureCurveTicks").innerHTML = [0, 1, 2, 3, 4].map(value => `<text x="38" y="${246 - value / 4 * 210}">${value}</text>`).join("") + data.train.map((_, i) => `<text x="${55 + i * 106}" y="261">${i + 1}</text>`).join("");
    $("#failureCurveRows").innerHTML = data.train.map((loss, i) => `<tr><th scope="row">${i + 1}</th><td>${loss.toFixed(2)}</td><td>${data.val[i].toFixed(2)}</td><td>${data.accuracy[i]}%</td></tr>`).join("");
    $("#failureObservation").textContent = `관찰: ${data.observation} 현재 코드의 선택 기준이면 최고 validation accuracy ${Math.max(...data.accuracy)}%인 epoch ${math.bestEpoch(data.accuracy)} 상태를 선택합니다(동률은 먼저 나온 epoch).`;
    $("#failureHypothesis").textContent = data.hypothesis;
    $("#failureAction").textContent = data.action;
  }
  if ($("#failureScenario")) {$("#failureScenario").addEventListener("change", renderCurve); renderCurve();}
  function renderMatrix() {
    const matrix = math.matrices[$("#failureMatrixCase").value];
    const result = math.summarizeMatrix(matrix);
    const percent = value => value === null ? "N/A" : `${value.toFixed(2)}%`;
    $("#failureMatrixRows").innerHTML = matrix.map((row, i) => `<tr><th scope="row">${"ABC"[i]}</th>${row.map((value, j) => `<td${i === j ? ' class="is-diagonal"' : ""}>${value}</td>`).join("")}</tr>`).join("");
    $("#failureMatrixSummary").textContent = `전체 accuracy: ${result.correct}/${result.total} = ${percent(result.accuracy)} · 관측된 클래스의 macro recall: ${percent(result.macroRecall)}`;
    $("#failureClassRecall").textContent = result.recalls.map((recall, i) => `${"ABC"[i]}: support ${result.support[i]}, recall ${percent(recall)}`).join(" · ");
    $("#failureMatrixInterpretation").textContent = result.support.includes(0)
      ? "C는 평가 표본이 없으므로 N/A입니다. macro recall은 표본이 있는 A·B만 평균했습니다. 프로젝트 코드의 0 반환과 구분해 읽으세요."
      : $("#failureMatrixCase").value === "imbalanced"
        ? "모두 A라고 예측해서 90%가 나왔지만 B·C는 하나도 맞히지 못했습니다. 전체 정확도 하나만으로 클래스별 성능을 판단할 수 없습니다."
        : "대각선과 각 행의 표본 수를 함께 봅니다. B→C는 2건, C→B는 3건으로 서로 다른 방향의 혼동입니다.";
  }
  if ($("#failureMatrixCase")) {$("#failureMatrixCase").addEventListener("change", renderMatrix); renderMatrix();}
  $$(".code-notes button").forEach(button => button.addEventListener("click", () => {
    const [start, end] = button.dataset.highlight.split("-").map(Number);
    $$(".line-code [data-line]").forEach(line => line.classList.toggle("is-highlighted", Number(line.dataset.line) >= start && Number(line.dataset.line) <= end));
    $$(".code-notes button").forEach(item => item.classList.toggle("is-active", item === button));
  }));
  $$(".copy-button").forEach(button => button.addEventListener("click", async () => {
    const target = button.dataset.copyTarget ? document.getElementById(button.dataset.copyTarget) : null;
    try {await navigator.clipboard.writeText(button.dataset.copyText || target?.innerText || "");
      const label = button.textContent; button.textContent = "복사됨"; window.setTimeout(() => {button.textContent = label;}, 1200);
    } catch (_) {button.textContent = "복사 실패";}
  }));
  const links = $$(".chapter-sidebar a");
  const observer = new IntersectionObserver(entries => {
    const current = entries.filter(entry => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (current) links.forEach(link => link.classList.toggle("is-current", link.getAttribute("href") === `#${current.target.id}`));
  }, {rootMargin:"-18% 0px -68%", threshold:[0, .2, .55]});
  links.map(link => $(link.getAttribute("href"))).filter(Boolean).forEach(section => observer.observe(section));
})();
