(() => {
  "use strict";
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const format = (value) => value.toLocaleString("ko-KR");

  function renderCost() {
    const input = Number($("#mobileInputChannels").value);
    const output = Number($("#mobileOutputChannels").value);
    const full = 9 * input * output;
    const split = 9 * input + input * output;
    $("#mobileFullWeights").textContent = format(full);
    $("#mobileSplitWeights").textContent = format(split);
    $("#mobileFullFormula").textContent = `9 × ${input} × ${output}`;
    $("#mobileSplitFormula").textContent = `9 × ${input} + ${input} × ${output}`;
    $("#mobileCostRatio").textContent = `${(full / split).toFixed(2)}배`;
  }
  $$("#mobileInputChannels, #mobileOutputChannels").forEach((control) => control.addEventListener("change", renderCost));
  if ($("#mobileInputChannels")) renderCost();

  function renderBlock() {
    const input = Number($("#mobileBlockInput").value);
    const output = Number($("#mobileBlockOutput").value);
    const expand = Number($("#mobileExpand").value);
    const stride = Number($("#mobileStride").value);
    const hidden = Math.round(input * expand);
    $("#mobileFlowInput").textContent = String(input);
    $("#mobileFlowHidden").textContent = String(hidden);
    $("#mobileFlowDepthwise").textContent = String(hidden);
    $("#mobileFlowOutput").textContent = String(output);
    $("#mobileFlowExpandLabel").textContent = expand === 1 ? "확장 생략" : "1×1 확장";
    $("#mobileExpandNote").textContent = expand === 1 ? "확장 층 생략" : "ReLU6";
    $("#mobileSkipNote").textContent = stride === 1 && input === output
      ? "보폭 1·입출력 채널 일치 → shortcut으로 입력을 더합니다."
      : `보폭 ${stride}·입력 ${input}채널·출력 ${output}채널 → 이 블록은 shortcut을 더하지 않습니다.`;
  }
  $$("#mobileBlockInput, #mobileBlockOutput, #mobileExpand, #mobileStride").forEach((control) => control.addEventListener("change", renderBlock));
  if ($("#mobileBlockInput")) renderBlock();

  const groups = [
    {t:1, c:16, n:1, s:1}, {t:6, c:24, n:2, s:2},
    {t:6, c:32, n:3, s:2}, {t:6, c:64, n:4, s:2},
    {t:6, c:96, n:3, s:1}, {t:6, c:160, n:3, s:2},
    {t:6, c:320, n:1, s:1},
  ];
  const parameters = {"0.5":700490, "0.75":1368234, "1":2236682};
  function makeDivisible(value) {
    let rounded = Math.max(8, Math.floor((value + 4) / 8) * 8);
    if (rounded < 0.9 * value) rounded += 8;
    return rounded;
  }
  function stagesFor(width) {
    let size = 16;
    const stages = [{name:"stem", output:[makeDivisible(32 * width), size, size], stride:2, repeats:1, expand:"해당 없음", pattern:"3×3 Conv → BN → ReLU6"}];
    groups.forEach((group, index) => {
      size = Math.ceil(size / group.s);
      const channels = makeDivisible(group.c * width);
      stages.push({name:`stage ${index + 1}`, output:[channels, size, size], stride:group.s, repeats:group.n, expand:group.t,
        pattern:`t=${group.t}, c=${channels}, n=${group.n}, s=${group.s}`});
    });
    stages.push({name:"final 1×1", output:[1280, size, size], stride:1, repeats:1, expand:"해당 없음", pattern:"1×1 Conv → BN → ReLU6"});
    return stages;
  }
  const stageList = $("#mobileStageList");
  let selectedStage = 1;
  function renderStages() {
    if (!stageList) return;
    const width = $("#mobileWidth").value;
    const stages = stagesFor(Number(width));
    const stage = stages[selectedStage];
    $("#mobileParameters").textContent = format(parameters[width]);
    $("#mobileLastChannels").textContent = "1,280";
    stageList.innerHTML = stages.map((item, index) =>
      `<button type="button" data-index="${index}" class="${index === selectedStage ? "is-current" : ""}" aria-pressed="${index === selectedStage}"><span>${item.name}</span><strong>${item.output[1]}×${item.output[2]}</strong><small>${item.output[0]}채널</small></button>`
    ).join("");
    $("#mobileStageName").textContent = stage.name;
    $("#mobileStagePattern").textContent = stage.pattern;
    $("#mobileStageOutput").textContent = `[${stage.output.join(", ")}]`;
    $("#mobileStageStride").textContent = String(stage.stride);
    $("#mobileStageRepeats").textContent = `${stage.repeats}개`;
    $("#mobileStageExpand").textContent = String(stage.expand);
  }
  $("#mobileWidth")?.addEventListener("change", () => { selectedStage = 1; renderStages(); });
  stageList?.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-index]");
    if (!button) return;
    selectedStage = Number(button.dataset.index);
    renderStages();
  });
  renderStages();

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
