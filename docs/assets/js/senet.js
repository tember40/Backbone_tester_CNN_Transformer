(() => {
  "use strict";
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const channels = [[1, 3, 5, 7], [2, 2, 4, 4], [8, 4, 0, 0]];
  const channelSelect = $("#seChannel");

  function renderExamples(rebuildPixels = true) {
    if (!channelSelect) return;
    const index = Number(channelSelect.value);
    const values = channels[index];
    if (rebuildPixels) {
      $("#sePixelGrid").innerHTML = values.map((value, position) =>
        `<label><span class="sr-only">${position + 1}번째 위치</span><input type="number" min="0" max="9" step="1" value="${value}" data-position="${position}"></label>`
      ).join("");
    }
    $("#seSqueezeSummary").innerHTML = channels.map((channel, position) => {
      const mean = channel.reduce((sum, value) => sum + value, 0) / channel.length;
      return `<div class="${position === index ? "is-current" : ""}"><span>채널 ${position + 1}</span><strong>${mean.toFixed(2)}</strong></div>`;
    }).join("");
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    $("#seSqueezeEquation").textContent = `채널 ${index + 1}: (${values.join(" + ")}) ÷ 4 = ${mean.toFixed(2)}`;

    const gate = Number($("#seGate" + (index + 1)).value);
    $("#seCurrentGate").textContent = gate.toFixed(2);
    $("#seScaleBefore").innerHTML = values.map((value) => `<span>${value.toFixed(1)}</span>`).join("");
    $("#seScaleAfter").innerHTML = values.map((value) => `<span>${(value * gate).toFixed(2)}</span>`).join("");
    $("#seScaleExplanation").textContent = `채널 ${index + 1}의 모든 위치에 같은 가중치 ${gate.toFixed(2)}를 곱했습니다. 다른 채널에는 각각의 가중치가 적용됩니다.`;
    $$("#seGate1, #seGate2, #seGate3").forEach((slider) => {
      $("#" + slider.id + "Value").textContent = Number(slider.value).toFixed(2);
    });
  }

  channelSelect?.addEventListener("change", renderExamples);
  $("#sePixelGrid")?.addEventListener("input", (event) => {
    const field = event.target.closest("input[data-position]");
    if (!field) return;
    const number = Number(field.value);
    channels[Number(channelSelect.value)][Number(field.dataset.position)] = Number.isFinite(number)
      ? Math.max(0, Math.min(9, Math.round(number))) : 0;
    renderExamples(false);
  });
  $("#sePixelGrid")?.addEventListener("change", () => renderExamples());
  $$("#seGate1, #seGate2, #seGate3").forEach((slider) => slider.addEventListener("input", renderExamples));
  renderExamples();

  function renderReduction() {
    const channelCount = Number($("#seChannels").value);
    const reduction = Number($("#seReduction").value);
    const hidden = Math.floor(channelCount / reduction);
    const parameters = 2 * channelCount * hidden;
    $("#seDimInput").textContent = String(channelCount);
    $("#seDimHidden").textContent = String(hidden);
    $("#seDimOutput").textContent = String(channelCount);
    $("#seParameterExplanation").textContent = `두 Linear 층의 가중치: ${channelCount} × ${hidden} + ${hidden} × ${channelCount} = ${parameters.toLocaleString("ko-KR")}개 (편향 없음)`;
  }
  $$("#seChannels, #seReduction").forEach((control) => control.addEventListener("change", renderReduction));
  if ($("#seChannels")) renderReduction();

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
