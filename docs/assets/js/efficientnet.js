(() => {
  "use strict";
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const format = (value) => value.toLocaleString("ko-KR");

  function renderScaling() {
    const phi = Number($("#efficientPhi").value);
    const depth = 1.2 ** phi;
    const width = 1.1 ** phi;
    const resolution = 1.15 ** phi;
    const cost = depth * width ** 2 * resolution ** 2;
    $("#efficientPhiValue").textContent = String(phi);
    [["Depth", depth, 1.2 ** 4], ["Width", width, 1.1 ** 4], ["Resolution", resolution, 1.15 ** 4]].forEach(([name, value, maximum]) => {
      $("#efficient" + name).textContent = value.toFixed(2) + "×";
      $("#efficient" + name + "Bar").style.width = (100 * value / maximum).toFixed(1) + "%";
    });
    $("#efficientCost").textContent = `단순 근사 계산량: B0의 ${cost.toFixed(2)}배 (실측 FLOPs가 아닙니다)`;
  }
  $("#efficientPhi")?.addEventListener("input", renderScaling);
  if ($("#efficientPhi")) renderScaling();

  const presets = {
    b0: {width:1, depth:1, parameters:4055730},
    b1: {width:1, depth:1.1, parameters:6574186},
    b2: {width:1.1, depth:1.2, parameters:7794570},
    b3: {width:1.2, depth:1.4, parameters:10669938},
  };
  const groups = [
    {t:1,c:16,n:1,s:1,k:3}, {t:6,c:24,n:2,s:2,k:3},
    {t:6,c:40,n:2,s:2,k:5}, {t:6,c:80,n:3,s:2,k:3},
    {t:6,c:112,n:3,s:1,k:5}, {t:6,c:192,n:4,s:2,k:5},
    {t:6,c:320,n:1,s:1,k:3},
  ];
  function makeDivisible(value) {
    let rounded = Math.max(8, Math.floor((value + 4) / 8) * 8);
    if (rounded < 0.9 * value) rounded += 8;
    return rounded;
  }
  function stagesFor(preset) {
    let size = 16;
    const stages = [{name:"stem", output:[makeDivisible(32 * preset.width),size,size], repeats:1, stride:2, kernel:"3×3", pattern:"3×3 Conv → BN → ReLU6"}];
    groups.forEach((group, index) => {
      size = Math.ceil(size / group.s);
      const channels = makeDivisible(group.c * preset.width);
      const repeats = Math.ceil(group.n * preset.depth);
      stages.push({name:`stage ${index + 1}`, output:[channels,size,size], repeats, stride:group.s, kernel:`${group.k}×${group.k}`,
        pattern:`MBConv${group.t} · k=${group.k} · 기본 ${group.n}회 → 현재 ${repeats}회`});
    });
    stages.push({name:"final 1×1", output:[makeDivisible(1280 * preset.width),size,size], repeats:1, stride:1, kernel:"1×1", pattern:"1×1 Conv → BN → ReLU6"});
    return stages;
  }
  let selectedStage = 1;
  function renderPreset() {
    const list = $("#efficientStageList");
    if (!list) return;
    const preset = presets[$("#efficientPreset").value];
    const stages = stagesFor(preset);
    const stage = stages[selectedStage];
    $("#efficientMultipliers").textContent = `${preset.width.toFixed(1)}× · ${preset.depth.toFixed(1)}×`;
    $("#efficientBlocks").textContent = `${stages.slice(1, -1).reduce((total, item) => total + item.repeats, 0)}개`;
    $("#efficientParameters").textContent = format(preset.parameters);
    list.innerHTML = stages.map((item, index) =>
      `<button type="button" data-index="${index}" class="${index === selectedStage ? "is-current" : ""}" aria-pressed="${index === selectedStage}"><span>${item.name}</span><strong>${item.output[1]}×${item.output[2]}</strong><small>${item.output[0]}채널 · ${item.repeats}회</small></button>`
    ).join("");
    $("#efficientStageName").textContent = stage.name;
    $("#efficientStagePattern").textContent = stage.pattern;
    $("#efficientStageOutput").textContent = `[${stage.output.join(", ")}]`;
    $("#efficientStageRepeats").textContent = `${stage.repeats}개`;
    $("#efficientStageStride").textContent = String(stage.stride);
    $("#efficientStageKernel").textContent = stage.kernel;
  }
  $("#efficientPreset")?.addEventListener("change", () => { selectedStage = 1; renderPreset(); });
  $("#efficientStageList")?.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-index]");
    if (!button) return;
    selectedStage = Number(button.dataset.index);
    renderPreset();
  });
  renderPreset();

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
