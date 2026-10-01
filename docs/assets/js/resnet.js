(() => {
  "use strict";
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const format = (value) => value.toLocaleString("ko-KR");

  function initializeResidualLab() {
    const input = $("#residualInput");
    const change = $("#residualChange");
    if (!input || !change) return;
    function render() {
      const x = Number(input.value);
      const residual = Number(change.value);
      const sum = x + residual;
      const fixed = (value) => value.toFixed(1);
      $("#residualInputValue").textContent = fixed(x);
      $("#residualChangeValue").textContent = fixed(residual);
      $("#residualDiagramInput").textContent = fixed(x);
      $("#residualDiagramChange").textContent = fixed(residual);
      $("#residualDiagramSkip").textContent = fixed(x);
      $("#residualSum").textContent = fixed(sum);
      $("#residualOutput").textContent = fixed(Math.max(0, sum));
      $("#plainOutput").textContent = fixed(Math.max(0, residual));
      $("#residualNote").textContent = residual === 0
        ? "F(x)=0이면 이 실습의 0 이상 입력은 shortcut을 따라 그대로 전달됩니다."
        : sum < 0
          ? "덧셈 결과가 음수라서 마지막 ReLU가 0으로 바꿉니다. shortcut은 무조건 원본 값을 보존하는 스위치가 아닙니다."
          : "주 경로의 변화량에 원래 입력을 더했습니다. 오른쪽의 plain 출력과 비교하세요.";
    }
    input.addEventListener("input", render);
    change.addEventListener("input", render);
    render();
  }

  const variants = {
    resnet18: {label:"ResNet18", blocks:[2, 2, 2, 2], expansion:1, type:"BasicBlock", parameters:11181642},
    resnet34: {label:"ResNet34", blocks:[3, 4, 6, 3], expansion:1, type:"BasicBlock", parameters:21289802},
    resnet50: {label:"ResNet50", blocks:[3, 4, 6, 3], expansion:4, type:"Bottleneck", parameters:23528522},
  };
  const planes = [64, 128, 256, 512];

  function stagesFor(variant) {
    const stages = [{name:"stem", input:[3, 32, 32], output:[64, 8, 8], blocks:0,
      shortcut:"해당 없음", description:"7×7 Conv, stride 2 → BatchNorm → ReLU → 3×3 MaxPool, stride 2"}];
    let channels = 64;
    let size = 8;
    for (let index = 0; index < 4; index += 1) {
      const outputChannels = planes[index] * variant.expansion;
      const input = [channels, size, size];
      if (index > 0) size /= 2;
      const projection = index > 0 || channels !== outputChannels;
      stages.push({name:`layer${index + 1}`, input, output:[outputChannels, size, size],
        blocks:variant.blocks[index], shortcut:projection ? "1×1 Conv + BatchNorm" : "identity (그대로 전달)",
        description:`${variant.type} ${variant.blocks[index]}개 · 첫 블록 stride ${index === 0 ? 1 : 2}`});
      channels = outputChannels;
    }
    return stages;
  }

  function initializeStageLab() {
    const control = $("#resnetVariant");
    const list = $("#resnetStageList");
    if (!control || !list) return;
    let selectedStage = 1;
    function render() {
      const variant = variants[control.value];
      const stages = stagesFor(variant);
      const stage = stages[selectedStage];
      $("#resnetBlockPattern").textContent = variant.blocks.join("–");
      $("#resnetBlockType").textContent = variant.type;
      $("#resnetParameters").textContent = format(variant.parameters);
      list.innerHTML = stages.map((item, index) =>
        `<button type="button" data-index="${index}" class="${index === selectedStage ? "is-current" : ""}" aria-pressed="${index === selectedStage}"><span>${item.name}</span><strong>${item.output[1]}×${item.output[2]}</strong><small>${item.output[0]}채널</small></button>`
      ).join("");
      $("#resnetStageName").textContent = stage.name;
      $("#resnetStageDescription").textContent = stage.description;
      $("#resnetStageInput").textContent = `[${stage.input.join(", ")}]`;
      $("#resnetStageOutput").textContent = `[${stage.output.join(", ")}]`;
      $("#resnetStageBlocks").textContent = stage.blocks ? `${stage.blocks}개` : "해당 없음";
      $("#resnetStageShortcut").textContent = stage.shortcut;
    }
    control.addEventListener("change", () => { selectedStage = 1; render(); });
    list.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-index]");
      if (!button) return;
      selectedStage = Number(button.dataset.index);
      render();
    });
    render();
  }

  const extensions = {
    resnet50: {groups:1, width:64, channels:64, parameters:23528522,
      explanation:"기준 모델입니다. Bottleneck의 3×3 합성곱은 그룹 1개를 사용합니다."},
    resnext50_32x4d: {groups:32, width:4, channels:128, parameters:23000394,
      explanation:"32개 그룹으로 3×3 합성곱을 나눕니다. 내부 채널은 int(64 × 4/64) × 32 = 128개입니다."},
    wide_resnet50_2: {groups:1, width:128, channels:128, parameters:66854730,
      explanation:"그룹은 1개 그대로 두고 Bottleneck 내부 채널을 64개에서 128개로 넓힙니다."},
  };
  function initializeExtensionLab() {
    const control = $("#resnetExtension");
    if (!control) return;
    function render() {
      const selected = extensions[control.value];
      $("#extensionGroups").textContent = String(selected.groups);
      $("#extensionWidth").textContent = String(selected.width);
      $("#extensionChannels").textContent = String(selected.channels);
      $("#extensionParameters").textContent = format(selected.parameters);
      $("#extensionExplanation").textContent = selected.explanation;
    }
    control.addEventListener("change", render);
    render();
  }

  function initializeCodeStudy() {
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
  }

  function initializeQuiz() {
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
  }

  function initializeSectionNavigation() {
    const links = $$(".chapter-sidebar a");
    const sections = links.map((link) => $(link.getAttribute("href"))).filter(Boolean);
    const observer = new IntersectionObserver((entries) => {
      const current = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!current) return;
      links.forEach((link) => link.classList.toggle("is-current", link.getAttribute("href") === `#${current.target.id}`));
    }, {rootMargin:"-18% 0px -68%", threshold:[0, .2, .55]});
    sections.forEach((section) => observer.observe(section));
  }

  initializeResidualLab();
  initializeStageLab();
  initializeExtensionLab();
  initializeCodeStudy();
  initializeQuiz();
  initializeSectionNavigation();
})();
