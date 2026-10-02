(() => {
  "use strict";
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const stageChannels = [64, 128, 320, 512];
  const stageHeads = [1, 2, 5, 8];
  const stageDepths = [2, 2, 2, 2];
  const modes = {
    cifar: {size: 32, ratios: [2, 1, 1, 1], label: "32×32 CIFAR-10 조정 설정"},
    naive: {size: 32, ratios: [8, 4, 2, 1], label: "32×32에 224용 비율을 그대로 적용한 비교"},
    paper: {size: 224, ratios: [8, 4, 2, 1], label: "224×224 논문 규모의 설정"},
  };
  let mode = "cifar";
  let stage = 0;

  function drawGrid(selector, exactSide) {
    const grid = $(selector);
    const shownSide = Math.min(exactSide, 8);
    grid.style.setProperty("--pvt-grid-side", String(shownSide));
    grid.replaceChildren(...Array.from({length: shownSide ** 2}, () => document.createElement("i")));
  }

  function render() {
    const selectedMode = modes[mode];
    const sides = [0, 1, 2, 3].map((index) => selectedMode.size / (4 * 2 ** index));
    const qSide = sides[stage];
    const ratio = selectedMode.ratios[stage];
    const kvSide = qSide / ratio;
    const queries = qSide ** 2;
    const keys = kvSide ** 2;
    ["#pvtStageOne", "#pvtStageTwo", "#pvtStageThree", "#pvtStageFour"].forEach((selector, index) => {
      $(selector).textContent = `${sides[index]}×${sides[index]}`;
    });
    $("#pvtModeLabel").textContent = selectedMode.label;
    $("#pvtStageTitle").textContent = `${stage + 1}단계 · ${qSide}×${qSide} 특징 격자`;
    $("#pvtStageDescription").textContent = ratio === 1
      ? `Query와 Key·Value 모두 ${queries}곳에서 만듭니다. 이 단계에서는 SRA 공간 축소를 하지 않습니다.`
      : `Query는 ${queries}곳 모두에서 만들고, Key·Value는 ${ratio}×${ratio} 공간 축소 뒤 ${keys}곳에서 만듭니다.`;
    $("#pvtStageStructure").textContent = `${stageChannels[stage]} / ${stageHeads[stage]} / ${stageDepths[stage]}`;
    $("#pvtStageRatio").textContent = String(ratio);
    $("#pvtQLabel").textContent = `${qSide}×${qSide} = ${queries.toLocaleString("ko-KR")}`;
    $("#pvtKVLabel").textContent = `${kvSide}×${kvSide} = ${keys.toLocaleString("ko-KR")}`;
    $("#pvtQueryCount").textContent = queries.toLocaleString("ko-KR");
    $("#pvtKeyCount").textContent = keys.toLocaleString("ko-KR");
    $("#pvtReducedCells").textContent = (queries * keys).toLocaleString("ko-KR");
    $("#pvtFullCells").textContent = (queries ** 2).toLocaleString("ko-KR");
    const warning = $("#pvtStageWarning");
    warning.classList.toggle("is-warning", keys === 1 && queries > 1);
    warning.textContent = keys === 1 && queries > 1
      ? "Key·Value가 한 곳뿐이라 각 Query의 softmax 가중치는 1입니다. 여러 위치 중 선택하는 Attention이 사라집니다."
      : queries === 1
        ? "마지막 격자는 1×1입니다. 이 입력 크기에서는 공간 위치가 하나만 남습니다."
        : ratio === 1
          ? "공간 축소를 하지 않아 모든 Query가 같은 수의 Key·Value 위치를 비교합니다."
          : `축소 후에도 Key·Value 위치가 ${keys}개 남아 여러 위치를 비교할 수 있습니다.`;
    drawGrid("#pvtQGrid", qSide);
    drawGrid("#pvtKVGrid", kvSide);
    $$('[data-pvt-mode]').forEach((button) => {
      const current = button.dataset.pvtMode === mode;
      button.classList.toggle("is-current", current);
      button.setAttribute("aria-pressed", String(current));
    });
    $$('[data-pvt-stage]').forEach((button) => {
      const current = Number(button.dataset.pvtStage) === stage;
      button.classList.toggle("is-current", current);
      button.setAttribute("aria-pressed", String(current));
    });
  }

  $$('[data-pvt-mode]').forEach((button) => button.addEventListener("click", () => {
    mode = button.dataset.pvtMode;
    render();
  }));
  $$('[data-pvt-stage]').forEach((button) => button.addEventListener("click", () => {
    stage = Number(button.dataset.pvtStage);
    render();
  }));
  render();

  $$('.code-notes button').forEach((button) => button.addEventListener("click", () => {
    const [start, end] = button.dataset.highlight.split("-").map(Number);
    $$('.line-code [data-line]').forEach((line) => {
      const number = Number(line.dataset.line);
      line.classList.toggle("is-highlighted", number >= start && number <= end);
    });
    $$('.code-notes button').forEach((item) => item.classList.toggle("is-active", item === button));
  }));
  $$('.copy-button').forEach((button) => button.addEventListener("click", async () => {
    const target = button.dataset.copyTarget ? document.getElementById(button.dataset.copyTarget) : null;
    const value = button.dataset.copyText || target?.innerText || "";
    try {
      await navigator.clipboard.writeText(value);
      const label = button.textContent;
      button.textContent = "복사됨";
      window.setTimeout(() => { button.textContent = label; }, 1200);
    } catch (_) { button.textContent = "복사 실패"; }
  }));
  $("#checkQuiz")?.addEventListener("click", () => {
    let score = 0;
    $$('.quiz').forEach((quiz) => {
      const choice = $("input:checked", quiz);
      const correct = choice?.value === quiz.dataset.answer;
      quiz.classList.toggle("is-correct", correct);
      quiz.classList.toggle("is-wrong", !correct);
      $(".quiz-feedback", quiz).textContent = !choice ? "답을 선택하세요." : correct ? "맞았습니다." : "다시 확인해 보세요.";
      if (correct) score += 1;
    });
    $("#quizScore").textContent = `3문제 중 ${score}문제를 맞혔습니다.`;
  });

  const links = $$('.chapter-sidebar a');
  const sections = links.map((link) => $(link.getAttribute("href"))).filter(Boolean);
  const observer = new IntersectionObserver((entries) => {
    const current = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!current) return;
    links.forEach((link) => link.classList.toggle("is-current", link.getAttribute("href") === `#${current.target.id}`));
  }, {rootMargin: "-18% 0px -68%", threshold: [0, .2, .55]});
  sections.forEach((section) => observer.observe(section));
})();
