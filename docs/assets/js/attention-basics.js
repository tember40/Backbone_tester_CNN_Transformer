(() => {
  "use strict";
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const names = ["A", "B", "C"];
  const vectors = [[1, 0], [0, 1], [1, 1]];
  const dot = (left, right) => left.reduce((total, value, index) => total + value * right[index], 0);
  const softmax = (scores) => {
    const highest = Math.max(...scores);
    const exponents = scores.map((score) => Math.exp(score - highest));
    const total = exponents.reduce((sum, value) => sum + value, 0);
    return exponents.map((value) => value / total);
  };
  const scoreRows = vectors.map((query) => vectors.map((key) => dot(query, key) / Math.sqrt(2)));
  const weightRows = scoreRows.map(softmax);
  let selected = 0;

  function renderAttention() {
    const body = $("#attentionMatrix");
    if (!body) return;
    body.innerHTML = weightRows.map((row, rowIndex) =>
      `<tr class="${rowIndex === selected ? "is-current" : ""}"><th scope="row">${names[rowIndex]}</th>${row.map((weight) =>
        `<td style="background-color:rgba(112,46,54,${(0.06 + weight * 0.68).toFixed(3)})">${weight.toFixed(3)}</td>`
      ).join("")}</tr>`
    ).join("");
    const scores = scoreRows[selected];
    const weights = weightRows[selected];
    const output = [0, 1].map((dimension) =>
      weights.reduce((sum, weight, index) => sum + weight * vectors[index][dimension], 0)
    );
    $("#attentionScores").textContent = `[${scores.map((score) => score.toFixed(2)).join(", ")}]`;
    $("#attentionWeights").textContent = `[${weights.map((weight) => weight.toFixed(3)).join(", ")}]`;
    $("#attentionWeightSum").textContent = `Query ${names[selected]} 행의 합 = ${weights.reduce((sum, value) => sum + value, 0).toFixed(3)}`;
    $("#attentionOutput").textContent = `[${output.map((value) => value.toFixed(3)).join(", ")}]`;
    $("#attentionBreakdown").textContent = weights.map((weight, index) =>
      `${weight.toFixed(3)}×[${vectors[index].join(",")}]`
    ).join(" + ");
    $$("[data-query]").forEach((button) => {
      const current = Number(button.dataset.query) === selected;
      button.classList.toggle("is-current", current);
      button.setAttribute("aria-pressed", String(current));
    });
  }
  $$("[data-query]").forEach((button) => button.addEventListener("click", () => {
    selected = Number(button.dataset.query);
    renderAttention();
  }));
  renderAttention();

  $$(".code-notes button").forEach((button) => button.addEventListener("click", () => {
    const [start, end] = button.dataset.highlight.split("-").map(Number);
    $$(".line-code [data-line]").forEach((line) => {
      const number = Number(line.dataset.line);
      line.classList.toggle("is-highlighted", number >= start && number <= end);
    });
    $$(".code-notes button").forEach((item) => item.classList.toggle("is-active", item === button));
  }));
  $$(".copy-button").forEach((button) => button.addEventListener("click", async () => {
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
    $$(".quiz").forEach((quiz) => {
      const choice = $("input:checked", quiz);
      const correct = choice?.value === quiz.dataset.answer;
      quiz.classList.toggle("is-correct", correct);
      quiz.classList.toggle("is-wrong", !correct);
      $(".quiz-feedback", quiz).textContent = !choice ? "답을 선택하세요." : correct ? "맞았습니다." : "다시 확인해 보세요.";
      if (correct) score += 1;
    });
    $("#quizScore").textContent = `3문제 중 ${score}문제를 맞혔습니다.`;
  });

  const links = $$(".chapter-sidebar a");
  const sections = links.map((link) => $(link.getAttribute("href"))).filter(Boolean);
  const observer = new IntersectionObserver((entries) => {
    const current = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!current) return;
    links.forEach((link) => link.classList.toggle("is-current", link.getAttribute("href") === `#${current.target.id}`));
  }, {rootMargin:"-18% 0px -68%", threshold:[0, .2, .55]});
  sections.forEach((section) => observer.observe(section));
})();
