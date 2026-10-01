(() => {
  "use strict";
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  const variants = {
    vgg11: {blocks: [1, 1, 2, 2, 2], parameters: 28144010},
    vgg13: {blocks: [2, 2, 2, 2, 2], parameters: 28328522},
    vgg16: {blocks: [2, 2, 3, 3, 3], parameters: 33638218},
    vgg19: {blocks: [2, 2, 4, 4, 4], parameters: 38947914},
  };
  const widths = [64, 128, 256, 512, 512];
  const format = (value) => value.toLocaleString("ko-KR");

  function initializeKernelLab() {
    const layerControl = $("#smallKernelLayers");
    if (!layerControl) return;
    const channels = 64;
    function render() {
      const layers = Number(layerControl.value);
      const receptive = 1 + 2 * layers;
      const stackedParameters = layers * (3 * 3 * channels * channels + channels);
      const wideParameters = receptive * receptive * channels * channels + channels;
      $("#smallKernelValue").textContent = String(layers);
      $("#kernelReceptive").textContent = `${receptive} × ${receptive}`;
      $("#kernelStackParameters").textContent = format(stackedParameters);
      $("#kernelWideParameters").textContent = format(wideParameters);
      $("#kernelDifference").textContent = `${Math.round((1 - stackedParameters / wideParameters) * 100)}% 적음`;
      $("#kernelStack").innerHTML = Array.from({length: layers}, (_, index) =>
        `<span>3×3 합성곱 + ReLU <small>${index + 1}층</small></span>`
      ).join('<b aria-hidden="true">→</b>');
      $("#kernelFootnote").textContent = layers === 1
        ? "한 층일 때는 비교 대상도 같은 3×3 필터입니다."
        : "입력·출력 채널을 모두 64로 고정한 예시입니다. 중간 ReLU가 추가되므로 두 구조의 함수는 같지 않습니다.";
    }
    layerControl.addEventListener("input", render);
    render();
  }

  function stagesFor(name) {
    const variant = variants[name];
    let channels = 3;
    let size = 32;
    let receptive = 1;
    let jump = 1;
    return variant.blocks.map((count, index) => {
      const out = widths[index];
      let parameters = 0;
      for (let layer = 0; layer < count; layer += 1) {
        parameters += (3 * 3 * channels + 1) * out;
        channels = out;
        receptive += 2 * jump;
      }
      const beforePool = size;
      receptive += jump;
      jump *= 2;
      size /= 2;
      return {number:index + 1, count, channels, beforePool, size, receptive, jump, parameters};
    });
  }

  function initializeDepthLab() {
    const variantControl = $("#vggVariant");
    const blockControl = $("#vggBlock");
    if (!variantControl || !blockControl) return;
    function render() {
      const selected = variantControl.value;
      const variant = variants[selected];
      const stages = stagesFor(selected);
      const index = Number(blockControl.value);
      const stage = stages[index];
      const convCount = variant.blocks.reduce((sum, count) => sum + count, 0);
      $("#vggVariantName").textContent = selected.toUpperCase();
      $("#vggDepth").textContent = `${convCount}개 합성곱 + 3개 선형층`;
      $("#vggParameterTotal").textContent = format(variant.parameters);
      $("#vggBlockValue").textContent = `${index + 1} / 5`;
      $("#vggBlockName").textContent = `블록 ${stage.number}`;
      $("#vggBlockLayers").textContent = `3×3 Conv ${stage.count}회 → 2×2 MaxPool`;
      $("#vggBlockShape").textContent = `[${stage.channels}, ${stage.size}, ${stage.size}]`;
      $("#vggBlockBefore").textContent = `${stage.beforePool}×${stage.beforePool}`;
      $("#vggBlockAfter").textContent = `${stage.size}×${stage.size}`;
      $("#vggBlockRF").textContent = `${stage.receptive}×${stage.receptive}`;
      $("#vggBlockParameters").textContent = format(stage.parameters);
      $("#vggBlockList").innerHTML = stages.map((item, itemIndex) =>
        `<button type="button" data-index="${itemIndex}" class="${itemIndex === index ? "is-current" : ""}"><span>블록 ${item.number}</span><strong>${item.count} × 3×3</strong><small>${item.channels}채널 · ${item.size}×${item.size}</small></button>`
      ).join("");
      $$("#vggBlockList button").forEach((button) => button.addEventListener("click", () => {
        blockControl.value = button.dataset.index;
        render();
      }));
    }
    variantControl.addEventListener("change", () => { blockControl.value = 0; render(); });
    blockControl.addEventListener("input", render);
    render();
  }

  function initializeMemoryLab() {
    const batch = $("#memoryBatch");
    const variant = $("#memoryVariant");
    if (!batch || !variant) return;
    function render() {
      const count = Number(batch.value);
      const weights = variants[variant.value].parameters;
      const weightMiB = weights * 4 / 1048576;
      const firstFeatureMiB = count * 64 * 32 * 32 * 4 / 1048576;
      $("#memoryBatchValue").textContent = String(count);
      $("#memoryWeights").textContent = `${weightMiB.toFixed(1)} MiB`;
      $("#memoryFeature").textContent = `${firstFeatureMiB.toFixed(1)} MiB`;
      $("#memoryNote").textContent = "float32 기준 단일 가중치 복사본과 첫 합성곱 출력만 계산했습니다. 학습에는 기울기·옵티마이저 상태·다른 층의 활성화 메모리가 추가됩니다.";
    }
    batch.addEventListener("input", render);
    variant.addEventListener("change", render);
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

  initializeKernelLab();
  initializeDepthLab();
  initializeMemoryLab();
  initializeCodeStudy();
  initializeQuiz();
  initializeSectionNavigation();
})();
