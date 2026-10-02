(() => {
  "use strict";
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const stages = [
    {side: 8, channels: 96, blocks: 3},
    {side: 4, channels: 192, blocks: 3},
    {side: 2, channels: 384, blocks: 9},
    {side: 1, channels: 768, blocks: 3},
  ];
  let stage = 0;
  let kernel = 7;
  let selected = [4, 4];

  function render() {
    const {side, channels, blocks} = stages[stage];
    selected = selected.map((coordinate) => Math.min(coordinate, side - 1));
    const radius = Math.floor(kernel / 2);
    const grid = $("#convnextGrid");
    grid.style.setProperty("--convnext-side", String(side));
    grid.replaceChildren();
    let active = 0;
    for (let row = 0; row < side; row += 1) {
      for (let col = 0; col < side; col += 1) {
        const cell = document.createElement("button");
        cell.type = "button";
        cell.setAttribute("aria-label", `${row + 1}행 ${col + 1}열 출력 위치`);
        const inWindow = Math.abs(row - selected[0]) <= radius && Math.abs(col - selected[1]) <= radius;
        if (inWindow) active += 1;
        cell.classList.toggle("is-covered", inWindow);
        cell.classList.toggle("is-selected", row === selected[0] && col === selected[1]);
        cell.addEventListener("click", () => { selected = [row, col]; render(); });
        grid.append(cell);
      }
    }
    const windowGrid = $("#convnextWindow");
    windowGrid.style.setProperty("--convnext-kernel-side", String(kernel));
    windowGrid.replaceChildren();
    for (let offsetRow = -radius; offsetRow <= radius; offsetRow += 1) {
      for (let offsetCol = -radius; offsetCol <= radius; offsetCol += 1) {
        const inputRow = selected[0] + offsetRow;
        const inputCol = selected[1] + offsetCol;
        const inImage = inputRow >= 0 && inputRow < side && inputCol >= 0 && inputCol < side;
        const cell = document.createElement("i");
        cell.classList.toggle("is-input", inImage);
        cell.classList.toggle("is-center", offsetRow === 0 && offsetCol === 0);
        windowGrid.append(cell);
      }
    }
    $("#convnextStageTitle").textContent = `${stage + 1}단계 · ${side}×${side} 격자`;
    $("#convnextStageInfo").textContent = `채널 ${channels} · 블록 ${blocks}개`;
    $("#convnextKernelLabel").textContent = `${kernel}×${kernel} depthwise 커널`;
    $("#convnextPosition").textContent = `${selected[0] + 1}행 ${selected[1] + 1}열`;
    $("#convnextActive").textContent = String(active);
    $("#convnextWeights").textContent = String(kernel ** 2);
    $("#convnextCoverage").textContent = `${Math.round(active / kernel ** 2 * 100)}%`;
    $("#convnextExplanation").textContent = side === 1
      ? `1×1 격자에서는 ${kernel}×${kernel} 커널의 중심 가중치만 실제 입력 위치와 겹칩니다. 나머지는 0 패딩과 만납니다.`
      : `선택한 출력 위치에서 ${active}개의 실제 입력 칸이 ${kernel}×${kernel} 창에 들어옵니다. 붉은 칸은 실제 특징값, 바깥쪽 창은 0 패딩입니다.`;
    $$('[data-convnext-stage]').forEach((button) => {
      const current = Number(button.dataset.convnextStage) === stage;
      button.classList.toggle("is-current", current);
      button.setAttribute("aria-pressed", String(current));
    });
    $$('[data-convnext-kernel]').forEach((button) => {
      const current = Number(button.dataset.convnextKernel) === kernel;
      button.classList.toggle("is-current", current);
      button.setAttribute("aria-pressed", String(current));
    });
  }

  $$('[data-convnext-stage]').forEach((button) => button.addEventListener("click", () => {
    stage = Number(button.dataset.convnextStage);
    selected = [Math.floor(stages[stage].side / 2), Math.floor(stages[stage].side / 2)];
    render();
  }));
  $$('[data-convnext-kernel]').forEach((button) => button.addEventListener("click", () => {
    kernel = Number(button.dataset.convnextKernel);
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
