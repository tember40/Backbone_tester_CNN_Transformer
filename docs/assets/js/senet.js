(() => {
  "use strict";
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const channels = [[1, 3, 5, 7], [2, 2, 4, 4], [8, 4, 0, 0]];
  const manualGates = [0.25, 0.7, 0.95];
  const channelSelect = $("#seChannel");

  function renderExamples(rebuildPixels = true) {
    if (!channelSelect) return;
    const index = Number(channelSelect.value);
    const values = channels[index];
    if (rebuildPixels) {
      $("#sePixelGrid").innerHTML = values.map((value, position) =>
        `<label><span class="sr-only">채널 ${index + 1}, ${position + 1}번째 위치</span><input type="number" min="-9" max="9" step="1" value="${value}" data-position="${position}"></label>`
      ).join("");
    }
    $("#seSqueezeSummary").innerHTML = channels.map((channel, position) => {
      const mean = channel.reduce((sum, value) => sum + value, 0) / channel.length;
      return `<div class="${position === index ? "is-current" : ""}"><span>채널 ${position + 1}</span><strong>${mean.toFixed(2)}</strong></div>`;
    }).join("");
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    $("#seSqueezeEquation").textContent = `채널 ${index + 1}: (${values.join(" + ")}) ÷ 4 = ${mean.toFixed(2)}`;

    const means = channels.map(channel => channel.reduce((sum, value) => sum + value, 0) / channel.length);
    // Small, fixed-weight SELayer(3, reduction=3), not weights learned on CIFAR-10.
    const beforeRelu = 0.5 * means[0] - 0.5 * means[1] + 0.25 * means[2];
    const hidden = Math.max(0, beforeRelu);
    const logits = [hidden, 0.5 * hidden, -hidden];
    const mlpGates = logits.map(value => 1 / (1 + Math.exp(-value)));
    const automatic = $("#seGateMode").value === "mlp";
    const gates = automatic ? mlpGates : manualGates;
    const gate = gates[index];
    const fixed = value => value.toFixed(4);
    $("#seMLPMeans").textContent = `[${means.map(fixed).join(", ")}]`;
    $("#seMLPHidden").textContent = `${fixed(beforeRelu)} → ReLU → ${fixed(hidden)}`;
    $("#seMLPLogits").textContent = `[${logits.map(fixed).join(", ")}]`;
    $("#seMLPGates").textContent = `[${mlpGates.map(fixed).join(", ")}]`;
    $("#seMLPMessage").textContent = automatic
      ? "고정 MLP 모드: 위 특징맵의 값을 바꾸면 평균 → 은닉값 → gate → 채널 출력이 함께 바뀝니다. 학습은 수행하지 않습니다."
      : "수동 모드: 위 계산표의 MLP 결과 대신 슬라이더 값을 곱합니다. 두 모드의 gate를 구분해 읽으세요.";
    $("#seCurrentGate").textContent = gate.toFixed(2);
    $("#seScaleBefore").innerHTML = values.map((value) => `<span>${value.toFixed(1)}</span>`).join("");
    $("#seScaleAfter").innerHTML = values.map((value) => `<span>${(value * gate).toFixed(2)}</span>`).join("");
    $("#seScaleExplanation").textContent = `10.2절에서 선택한 채널 ${index + 1}: 모든 위치에 같은 가중치 ${gate.toFixed(4)}를 곱했습니다. 다른 채널에는 각각의 가중치가 적용됩니다.`;
    [1, 2, 3].forEach((number) => {
      const slider = $("#seGate" + number);
      slider.disabled = automatic;
      slider.step = automatic ? "0.0001" : "0.05";
      slider.value = String(gates[number - 1]);
      $("#" + slider.id + "Value").textContent = gates[number - 1].toFixed(automatic ? 4 : 2);
    });
  }

  channelSelect?.addEventListener("change", renderExamples);
  $("#sePixelGrid")?.addEventListener("input", (event) => {
    const field = event.target.closest("input[data-position]");
    if (!field) return;
    const number = Number(field.value);
    channels[Number(channelSelect.value)][Number(field.dataset.position)] = Number.isFinite(number)
      ? Math.max(-9, Math.min(9, Math.round(number))) : 0;
    if (field.value !== "" && Number.isFinite(number)) {
      field.value = String(channels[Number(channelSelect.value)][Number(field.dataset.position)]);
    }
    renderExamples(false);
  });
  $("#sePixelGrid")?.addEventListener("change", (event) => {
    const field = event.target.closest("input[data-position]");
    if (!field) return;
    field.value = String(channels[Number(channelSelect.value)][Number(field.dataset.position)]);
    renderExamples(false);
  });
  [1, 2, 3].forEach(number => $("#seGate" + number)?.addEventListener("input", () => {
    manualGates[number - 1] = Number($("#seGate" + number).value);
    renderExamples(false);
  }));
  $("#seGateMode")?.addEventListener("change", () => renderExamples(false));
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
