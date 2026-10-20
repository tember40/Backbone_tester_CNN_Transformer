(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  const STAGES = [
    {name: "입력", type: "image", shape: [3, 32, 32], params: 0, receptive: 1, jump: 1, note: "RGB 이미지 세 채널이 모델로 들어옵니다."},
    {name: "Conv1 + ReLU + LRN", type: "conv", shape: [64, 32, 32], params: 1792, receptive: 3, jump: 1, note: "3×3 필터 64개가 색과 짧은 경계를 찾습니다."},
    {name: "MaxPool1", type: "pool", shape: [64, 16, 16], params: 0, receptive: 4, jump: 2, note: "2×2 최대 풀링으로 공간 크기를 절반으로 줄입니다."},
    {name: "Conv2 + ReLU + LRN", type: "conv", shape: [192, 16, 16], params: 307392, receptive: 12, jump: 2, note: "5×5 필터가 더 넓은 경계 조합을 읽습니다."},
    {name: "MaxPool2", type: "pool", shape: [192, 8, 8], params: 0, receptive: 14, jump: 4, note: "채널 수는 유지하고 위치 수만 8×8로 줄입니다."},
    {name: "Conv3", type: "conv", shape: [384, 8, 8], params: 663936, receptive: 22, jump: 4, note: "가장 많은 384개 채널로 중간 수준의 패턴을 확장합니다."},
    {name: "Conv4", type: "conv", shape: [256, 8, 8], params: 884992, receptive: 30, jump: 4, note: "공간 크기를 유지하면서 특징을 다시 조합합니다."},
    {name: "Conv5", type: "conv", shape: [256, 8, 8], params: 590080, receptive: 38, jump: 4, note: "분류에 가까운 고수준 특징 256개를 만듭니다."},
    {name: "MaxPool3", type: "pool", shape: [256, 4, 4], params: 0, receptive: 42, jump: 8, note: "최종 합성곱 특징을 4×4로 요약합니다."},
    {name: "분류기", type: "classifier", shape: [10], params: 33603594, receptive: 66, jump: null, note: "4×4의 모든 위치를 연결합니다. 한 점수의 이론적 범위는 42 + (4−1)×8 = 66이며, 실제 입력 바깥은 패딩입니다."},
  ];

  function formatCount(value) {
    if (value >= 1000000) return `${(value / 1000000).toFixed(2)}M`;
    if (value >= 1000) return `${(value / 1000).toFixed(1)}K`;
    return value.toLocaleString("ko-KR");
  }

  function shapeText(shape) {
    return shape.length === 3 ? `[${shape.join(", ")}]` : `[${shape[0]}]`;
  }

  function initializeArchitectureExplorer() {
    const slider = $("#alexStage");
    const list = $("#alexStageList");
    if (!slider || !list) return;

    function render() {
      const index = Number(slider.value);
      const stage = STAGES[index];
      $("#alexStageValue").textContent = `${index} / ${STAGES.length - 1}`;
      $("#alexLayerName").textContent = stage.name;
      $("#alexLayerShape").textContent = shapeText(stage.shape);
      $("#alexLayerParams").textContent = formatCount(stage.params);
      $("#alexLayerRF").textContent = `${stage.receptive} × ${stage.receptive}`;
      $("#alexLayerNote").textContent = stage.note;
      const width = stage.shape.length === 3 ? Math.max(18, stage.shape[2] / 32 * 100) : 24;
      const depth = stage.shape.length === 3 ? Math.min(9, Math.max(1, Math.round(stage.shape[0] / 48))) : 3;
      $("#alexFeatureStack").style.setProperty("--alex-width", `${width}%`);
      $("#alexFeatureStack").innerHTML = Array.from({length: depth}, (_, sheetIndex) =>
        `<i style="--alex-sheet:${sheetIndex}"><span>${stage.shape.length === 3 ? `${stage.shape[0]}채널` : "class logits"}</span></i>`
      ).reverse().join("");
      list.innerHTML = STAGES.map((item, stageIndex) =>
        `<button type="button" data-stage="${stageIndex}" aria-pressed="${stageIndex === index}" class="${stageIndex === index ? "is-current" : ""}">` +
        `<span>${stageIndex}</span><strong>${item.name}</strong><small>${shapeText(item.shape)}</small></button>`
      ).join("");
      $$("button", list).forEach((button) => button.addEventListener("click", () => {
        slider.value = button.dataset.stage;
        const restoreFocus = document.activeElement === button;
        render();
        if (restoreFocus) $(`button[data-stage="${slider.value}"]`, list).focus({preventScroll: true});
      }));
    }

    slider.max = STAGES.length - 1;
    slider.addEventListener("input", render);
    render();
  }

  const VERSIONS = {
    original: {
      label: "논문 AlexNet",
      input: "224 × 224 RGB (논문 표기)",
      first: "11×11, stride 4, 96채널",
      firstOutput: "55 × 55 (그림 2 기준)",
      pools: "3×3, stride 2 · 겹치는 풀링",
      final: "256 × 6 × 6",
      classes: "1,000 classes",
      message: "논문의 본문·그림 표기를 비교합니다. 첫 출력 크기를 직접 계산할 때는 아래의 입력·패딩 조건 주의사항도 확인하세요.",
    },
    cifar: {
      label: "프로젝트 AlexNet",
      input: "32 × 32 RGB",
      first: "3×3, stride 1, 64채널",
      firstOutput: "32 × 32",
      pools: "2×2, stride 2 · 세 차례",
      final: "256 × 4 × 4",
      classes: "10 classes",
      message: "작은 CIFAR-10 입력의 공간 정보를 너무 일찍 잃지 않도록 첫 필터와 보폭을 줄였습니다.",
    },
  };

  function initializeAdaptationComparison() {
    const controls = $$("[name='alexVersion']");
    if (!controls.length) return;

    function render() {
      const selected = controls.find((control) => control.checked)?.value || "cifar";
      const version = VERSIONS[selected];
      $("#adaptationLabel").textContent = version.label;
      $("#adaptationInput").textContent = version.input;
      $("#adaptationFirst").textContent = version.first;
      $("#adaptationFirstOutput").textContent = version.firstOutput;
      $("#adaptationPools").textContent = version.pools;
      $("#adaptationFinal").textContent = version.final;
      $("#adaptationClasses").textContent = version.classes;
      $("#adaptationMessage").textContent = version.message;
    }

    controls.forEach((control) => control.addEventListener("change", render));
    render();
  }

  const PARAMETER_GROUPS = [
    {id: "conv1", label: "Conv1", value: 1792},
    {id: "conv2", label: "Conv2", value: 307392},
    {id: "conv3", label: "Conv3", value: 663936},
    {id: "conv4", label: "Conv4", value: 884992},
    {id: "conv5", label: "Conv5", value: 590080},
    {id: "fc1", label: "Linear 1", value: 16781312},
    {id: "fc2", label: "Linear 2", value: 16781312},
    {id: "fc3", label: "Linear 3", value: 40970},
  ];

  function initializeParameterExplorer() {
    const select = $("#parameterLayer");
    const chart = $("#parameterChart");
    if (!select || !chart) return;
    const total = PARAMETER_GROUPS.reduce((sum, item) => sum + item.value, 0);
    select.innerHTML = PARAMETER_GROUPS.map((item) => `<option value="${item.id}">${item.label}</option>`).join("");

    function render() {
      const selected = PARAMETER_GROUPS.find((item) => item.id === select.value) || PARAMETER_GROUPS[0];
      chart.innerHTML = PARAMETER_GROUPS.map((item) => {
        const share = item.value / total * 100;
        return `<button type="button" data-layer="${item.id}" class="${item.id === selected.id ? "is-current" : ""}" ` +
          `style="--parameter-share:${Math.max(1.2, share)}%"><span>${item.label}</span><i></i><strong>${formatCount(item.value)}</strong></button>`;
      }).join("");
      $$("button", chart).forEach((button) => button.addEventListener("click", () => {
        select.value = button.dataset.layer;
        const restoreFocus = document.activeElement === button;
        render();
        if (restoreFocus) $(`button[data-layer="${select.value}"]`, chart).focus({preventScroll: true});
      }));
      const share = selected.value / total * 100;
      $("#parameterSelected").textContent = selected.label;
      $("#parameterCount").textContent = selected.value.toLocaleString("ko-KR");
      $("#parameterShare").textContent = `${share.toFixed(share < 0.01 ? 4 : 2)}%`;
      $("#parameterTotal").textContent = total.toLocaleString("ko-KR");
      $("#parameterInterpretation").textContent = selected.id.startsWith("fc")
        ? "완전연결층은 공간 위치마다 별도 가중치를 사용해 파라미터가 빠르게 늘어납니다."
        : "합성곱층은 같은 필터를 모든 위치에서 공유하므로 특징맵 크기에 비해 파라미터가 적습니다.";
    }

    select.addEventListener("change", render);
    render();
  }

  let dropoutSeed = 1;
  function pseudoRandom(index, seed) {
    const value = Math.sin((index + 1) * 12.9898 + seed * 78.233) * 43758.5453;
    return value - Math.floor(value);
  }

  function initializeDropoutLab() {
    const probability = $("#dropoutProbability");
    const units = $("#dropoutUnits");
    if (!probability || !units) return;
    const mode = $("#dropoutMode");
    const resample = $("#resampleDropout");

    function render() {
      const p = Number(probability.value);
      const keep = 1 - p;
      const training = mode.value === "train";
      const inputs = Array.from({length: 24}, (_, index) => index % 4 + 1);
      const states = inputs.map((_, index) => !training || pseudoRandom(index, dropoutSeed) >= p);
      const active = states.filter(Boolean).length;
      const scale = training ? 1 / keep : 1;
      const outputs = inputs.map((value, index) => states[index] ? value * scale : 0);
      units.innerHTML = outputs.map((value, index) => {
        const status = training ? (states[index] ? "유지" : "제외") : "평가";
        return `<div class="dropout-unit ${states[index] ? "is-active" : "is-dropped"}" ` +
          `data-input="${inputs[index]}" data-output="${value}" ` +
          `aria-label="${index + 1}번: 입력 ${inputs[index]}, 출력 ${value.toFixed(1)}, ${status}">` +
          `<span>${index + 1} · ${status}</span><strong>${inputs[index]} → ${value.toFixed(1)}</strong></div>`;
      }).join("");
      $("#dropoutProbabilityValue").textContent = p.toFixed(1);
      $("#dropoutActive").textContent = `${active} / 24`;
      $("#dropoutScale").textContent = `${scale.toFixed(2)}배`;
      $("#dropoutMessage").textContent = !training
        ? "평가 모드: p 설정과 관계없이 입력을 그대로 전달합니다. 마스크와 학습 중 보정은 적용하지 않습니다."
        : p === 0
          ? "p=0: 마스크로 제외되는 원소가 없고 보정도 1배이므로 입력과 출력이 같습니다."
          : `각 원소를 확률 ${p.toFixed(1)}로 0으로 만듭니다. 이번에는 ${24 - active}개가 제외됐습니다. 남은 값은 ${scale.toFixed(2)}배이며, 매번 정확히 ${(p * 100).toFixed(0)}%가 제외되는 것은 아닙니다.`;
      resample.disabled = !training;
    }

    probability.addEventListener("input", render);
    mode.addEventListener("change", render);
    resample.addEventListener("click", () => { dropoutSeed += 1; render(); });
    render();
  }

  function initializeInnovationReading() {
    const buttons = $$(".innovation-list button");
    const descriptions = {
      relu: ["ReLU", "포화하기 쉬운 sigmoid·tanh보다 양수 구간의 기울기가 단순해 학습 속도를 높였습니다."],
      gpu: ["두 GPU 병렬 학습", "당시 메모리 한계를 넘기 위해 네트워크를 두 GPU에 나누어 대규모 모델을 학습했습니다."],
      augment: ["데이터 증강", "잘라내기와 좌우 반전, RGB 강도 변화를 사용해 같은 이미지에서 다양한 학습 표본을 만들었습니다."],
      dropout: ["Dropout", "완전연결층의 뉴런을 무작위로 꺼 특정 경로에 지나치게 의존하는 과적합을 줄였습니다."],
    };
    if (!buttons.length) return;

    function show(key, activeButton) {
      $("#innovationTitle").textContent = descriptions[key][0];
      $("#innovationDescription").textContent = descriptions[key][1];
      buttons.forEach((button) => button.classList.toggle("is-current", button === activeButton));
    }
    buttons.forEach((button) => button.addEventListener("click", () => show(button.dataset.innovation, button)));
    show(buttons[0].dataset.innovation, buttons[0]);
  }

  function initializeCodeStudy() {
    $$(".code-notes button").forEach((button) => {
      button.addEventListener("click", () => {
        const [start, end] = button.dataset.highlight.split("-").map(Number);
        $$(".line-code [data-line]").forEach((line) => {
          const number = Number(line.dataset.line);
          line.classList.toggle("is-highlighted", number >= start && number <= end);
        });
        $$(".code-notes button").forEach((item) => item.classList.toggle("is-active", item === button));
      });
    });
  }

  function initializeCopyButtons() {
    $$(".copy-button").forEach((button) => {
      button.addEventListener("click", async () => {
        const target = button.dataset.copyTarget ? document.getElementById(button.dataset.copyTarget) : null;
        const text = button.dataset.copyText || target?.innerText || "";
        try {
          await navigator.clipboard.writeText(text);
          const original = button.textContent;
          button.textContent = "복사됨";
          window.setTimeout(() => { button.textContent = original; }, 1200);
        } catch (_) {
          button.textContent = "복사 실패";
        }
      });
    });
  }

  function initializeQuiz() {
    const button = $("#checkQuiz");
    if (!button) return;
    button.addEventListener("click", () => {
      let score = 0;
      $$(".quiz").forEach((quiz) => {
        const selected = $("input:checked", quiz);
        const correct = selected?.value === quiz.dataset.answer;
        quiz.classList.toggle("is-correct", correct);
        quiz.classList.toggle("is-wrong", !correct);
        $(".quiz-feedback", quiz).textContent = !selected ? "답을 선택하세요." : correct ? "맞았습니다." : "구조를 다시 확인해 보세요.";
        if (correct) score += 1;
      });
      $("#quizScore").textContent = `3문제 중 ${score}문제를 맞혔습니다.`;
    });
  }

  function initializeSectionNavigation() {
    const links = $$(".chapter-sidebar a");
    const sections = links.map((link) => $(link.getAttribute("href"))).filter(Boolean);
    const observer = new IntersectionObserver((entries) => {
      const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      links.forEach((link) => link.classList.toggle("is-current", link.getAttribute("href") === `#${visible.target.id}`));
    }, {rootMargin: "-18% 0px -68%", threshold: [0, 0.2, 0.55]});
    sections.forEach((section) => observer.observe(section));
  }

  initializeInnovationReading();
  initializeArchitectureExplorer();
  initializeAdaptationComparison();
  initializeParameterExplorer();
  initializeDropoutLab();
  initializeCodeStudy();
  initializeCopyButtons();
  initializeQuiz();
  initializeSectionNavigation();
})();
