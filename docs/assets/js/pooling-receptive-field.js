(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  const POOL_INPUT = [
    [1, 3, 2, 4],
    [5, 0, 7, 1],
    [2, 6, 8, 3],
    [4, 1, 5, 9],
  ];

  function formatNumber(value) {
    return Number.isInteger(value) ? String(value) : value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  }

  function pool(matrix, size = 2, stride = 2, mode = "max") {
    const output = [];
    for (let row = 0; row + size <= matrix.length; row += stride) {
      const outputRow = [];
      for (let column = 0; column + size <= matrix[0].length; column += stride) {
        const values = [];
        for (let y = 0; y < size; y += 1) {
          for (let x = 0; x < size; x += 1) values.push(matrix[row + y][column + x]);
        }
        outputRow.push(mode === "max"
          ? Math.max(...values)
          : values.reduce((sum, value) => sum + value, 0) / values.length);
      }
      output.push(outputRow);
    }
    return output;
  }

  function initializePoolingWalkthrough() {
    const modeControl = $("#poolMode");
    const stepControl = $("#poolStep");
    const inputGrid = $("#poolInputGrid");
    const outputGrid = $("#poolOutputGrid");
    if (!modeControl || !stepControl || !inputGrid || !outputGrid) return;

    const positions = [[0, 0], [0, 2], [2, 0], [2, 2]];

    function render() {
      const mode = modeControl.value;
      const step = Number(stepControl.value);
      const [activeRow, activeColumn] = positions[step];
      const pooled = pool(POOL_INPUT, 2, 2, mode);
      const windowValues = POOL_INPUT.slice(activeRow, activeRow + 2)
        .flatMap((row) => row.slice(activeColumn, activeColumn + 2));
      const result = mode === "max"
        ? Math.max(...windowValues)
        : windowValues.reduce((sum, value) => sum + value, 0) / windowValues.length;

      inputGrid.innerHTML = POOL_INPUT.flatMap((row, rowIndex) => row.map((value, columnIndex) => {
        const active = rowIndex >= activeRow && rowIndex < activeRow + 2 && columnIndex >= activeColumn && columnIndex < activeColumn + 2;
        const selected = mode === "max" && active && value === result;
        return `<span class="${active ? "is-window" : ""} ${selected ? "is-maximum" : ""}">${value}</span>`;
      })).join("");
      outputGrid.innerHTML = pooled.flatMap((row, rowIndex) => row.map((value, columnIndex) => {
        const active = rowIndex === Math.floor(activeRow / 2) && columnIndex === Math.floor(activeColumn / 2);
        return `<span class="${active ? "is-output" : ""}">${formatNumber(value)}</span>`;
      })).join("");

      const expression = mode === "max"
        ? `max(${windowValues.join(", ")}) = ${formatNumber(result)}`
        : `(${windowValues.join(" + ")}) ÷ 4 = ${formatNumber(result)}`;
      $("#poolStepValue").textContent = `${step + 1} / 4`;
      $("#poolWindowPosition").textContent = `입력 (${activeRow + 1}, ${activeColumn + 1})에서 시작`;
      $("#poolExpression").textContent = expression;
      $("#poolResultPosition").textContent = `출력 (${Math.floor(activeRow / 2) + 1}, ${Math.floor(activeColumn / 2) + 1})에 기록`;
      $("#poolModeName").textContent = mode === "max" ? "최대 풀링" : "평균 풀링";
    }

    modeControl.addEventListener("change", render);
    stepControl.addEventListener("input", render);
    $("#poolPrev").addEventListener("click", () => {
      stepControl.value = Math.max(0, Number(stepControl.value) - 1);
      render();
    });
    $("#poolNext").addEventListener("click", () => {
      stepControl.value = Math.min(3, Number(stepControl.value) + 1);
      render();
    });
    render();
  }

  function initializeGeometryCalculator() {
    const input = $("#poolInputSize");
    const kernel = $("#poolKernelSize");
    const stride = $("#poolStride");
    if (!input || !kernel || !stride) return;

    function render() {
      const n = Number(input.value);
      const k = Number(kernel.value);
      const s = Number(stride.value);
      const output = Math.floor((n - k) / s) + 1;
      const used = (output - 1) * s + k;
      const leftover = Math.max(0, n - used);
      $("#poolInputSizeValue").textContent = n;
      $("#poolKernelSizeValue").textContent = k;
      $("#poolStrideValue").textContent = s;
      $("#poolGeometryFormula").textContent = `⌊(${n} − ${k}) / ${s}⌋ + 1 = ${output}`;
      $("#poolGeometryShape").textContent = `${n} × ${n} → ${output} × ${output}`;
      $("#poolGeometryWindows").textContent = `${output * output}개 창`;
      $("#poolGeometryRemainder").textContent = leftover === 0
        ? "경계까지 정확히 사용"
        : `오른쪽·아래 ${leftover}px은 창에 포함되지 않음`;
      const cells = Math.min(n, 12);
      $("#poolGeometryGrid").style.setProperty("--geometry-size", cells);
      $("#poolGeometryGrid").innerHTML = Array.from({length: cells * cells}, (_, index) => {
        const row = Math.floor(index / cells);
        const column = index % cells;
        const usedCell = row < used && column < used;
        return `<i class="${usedCell ? "is-used" : ""}"></i>`;
      }).join("");
    }

    [input, kernel, stride].forEach((control) => control.addEventListener("input", render));
    render();
  }

  function createPattern(name, size = 8) {
    return Array.from({length: size}, (_, row) => Array.from({length: size}, (_, column) => {
      if (name === "edge") return column < size / 2 ? 0.15 : 0.9;
      if (name === "spot") return Math.abs(row - 3.5) < 1.5 && Math.abs(column - 3.5) < 1.5 ? 1 : 0.08;
      const deterministicNoise = ((row * 17 + column * 29 + row * column * 7) % 100) / 100;
      return deterministicNoise;
    }));
  }

  function drawMatrix(canvas, matrix) {
    const context = canvas.getContext("2d");
    const cellWidth = canvas.width / matrix[0].length;
    const cellHeight = canvas.height / matrix.length;
    context.clearRect(0, 0, canvas.width, canvas.height);
    matrix.forEach((row, rowIndex) => row.forEach((value, columnIndex) => {
      const lightness = 96 - value * 66;
      context.fillStyle = `hsl(345 55% ${lightness}%)`;
      context.fillRect(columnIndex * cellWidth, rowIndex * cellHeight, cellWidth, cellHeight);
      context.strokeStyle = "rgba(255,255,255,.55)";
      context.strokeRect(columnIndex * cellWidth, rowIndex * cellHeight, cellWidth, cellHeight);
    }));
  }

  function initializeComparisonLab() {
    const patternControl = $("#poolPattern");
    if (!patternControl) return;

    function render() {
      const matrix = createPattern(patternControl.value);
      const maximum = pool(matrix, 2, 2, "max");
      const average = pool(matrix, 2, 2, "average");
      drawMatrix($("#comparisonInput"), matrix);
      drawMatrix($("#comparisonMax"), maximum);
      drawMatrix($("#comparisonAverage"), average);
      const range = (values) => `${Math.min(...values.flat()).toFixed(2)} ~ ${Math.max(...values.flat()).toFixed(2)}`;
      $("#comparisonInputRange").textContent = range(matrix);
      $("#comparisonMaxRange").textContent = range(maximum);
      $("#comparisonAverageRange").textContent = range(average);
      const notes = {
        edge: "최대 풀링은 밝은 영역을 넓게 남기고, 평균 풀링은 경계를 더 부드럽게 만듭니다.",
        spot: "작은 강한 반응은 최대 풀링에서 보존되지만 평균 풀링에서는 주변 값과 섞여 약해집니다.",
        noise: "최대 풀링은 우연히 큰 잡음도 강조할 수 있고, 평균 풀링은 잡음을 완화하지만 선명도도 낮춥니다.",
      };
      $("#comparisonObservation").textContent = notes[patternControl.value];
    }

    patternControl.addEventListener("change", render);
    render();
  }

  const ARCHITECTURES = {
    compact: [
      ["입력", 1, 1], ["Conv 3×3", 3, 1], ["Conv 3×3", 3, 1],
      ["MaxPool 2×2", 2, 2], ["Conv 3×3", 3, 1], ["MaxPool 2×2", 2, 2],
    ],
    alexnet: [
      ["입력", 1, 1], ["Conv 5×5", 5, 1], ["MaxPool 2×2", 2, 2],
      ["Conv 3×3", 3, 1], ["MaxPool 2×2", 2, 2], ["Conv 3×3", 3, 1],
    ],
  };

  function calculateReceptiveFields(layers) {
    let receptiveField = 1;
    let jump = 1;
    return layers.map(([name, kernel, stride], index) => {
      const previousJump = jump;
      if (index > 0) {
        receptiveField += (kernel - 1) * previousJump;
        jump *= stride;
      }
      return {name, kernel, stride, receptiveField, jump, previousJump};
    });
  }

  function initializeReceptiveFieldLab() {
    const architecture = $("#rfArchitecture");
    const stage = $("#rfStage");
    if (!architecture || !stage) return;

    function render() {
      const records = calculateReceptiveFields(ARCHITECTURES[architecture.value]);
      stage.max = records.length - 1;
      stage.value = Math.min(Number(stage.value), records.length - 1);
      const activeIndex = Number(stage.value);
      const active = records[activeIndex];
      $("#rfStageValue").textContent = `${activeIndex} / ${records.length - 1}`;
      $("#rfLayerName").textContent = active.name;
      $("#rfSize").textContent = `${active.receptiveField} × ${active.receptiveField}`;
      $("#rfJump").textContent = `${active.jump}px`;
      $("#rfFormula").textContent = activeIndex === 0
        ? "입력 픽셀의 수용영역은 1 × 1"
        : `${records[activeIndex - 1].receptiveField} + (${active.kernel} − 1) × ${active.previousJump} = ${active.receptiveField}`;
      const percentage = Math.min(94, Math.max(5, active.receptiveField / 32 * 100));
      $("#rfOverlay").style.width = `${percentage}%`;
      $("#rfOverlay").style.height = `${percentage}%`;
      $("#rfStageList").innerHTML = records.map((record, index) =>
        `<button type="button" data-stage="${index}" class="${index === activeIndex ? "is-current" : ""}">` +
        `<span>${index}</span><strong>${record.name}</strong><small>r=${record.receptiveField}, j=${record.jump}</small></button>`
      ).join("");
      $$("#rfStageList button").forEach((button) => button.addEventListener("click", () => {
        stage.value = button.dataset.stage;
        render();
      }));
    }

    architecture.addEventListener("change", () => { stage.value = 0; render(); });
    stage.addEventListener("input", render);
    render();
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
        $(".quiz-feedback", quiz).textContent = !selected ? "답을 선택하세요." : correct ? "맞았습니다." : "다시 계산해 보세요.";
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

  initializePoolingWalkthrough();
  initializeGeometryCalculator();
  initializeComparisonLab();
  initializeReceptiveFieldLab();
  initializeCodeStudy();
  initializeCopyButtons();
  initializeQuiz();
  initializeSectionNavigation();
})();
