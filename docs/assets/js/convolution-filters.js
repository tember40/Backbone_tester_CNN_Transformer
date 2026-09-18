(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  const TRACE_PATTERNS = {
    vertical: Array.from({ length: 5 }, () => [0, 0, 1, 1, 1]),
    horizontal: Array.from({ length: 5 }, (_, row) =>
      Array.from({ length: 5 }, () => (row >= 2 ? 1 : 0))
    ),
    cross: Array.from({ length: 5 }, (_, row) =>
      Array.from({ length: 5 }, (_, column) => (row === 2 || column === 2 ? 1 : 0))
    ),
  };
  const KERNELS = {
    vertical: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
    horizontal: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
    blur: Array.from({ length: 3 }, () => Array(3).fill(1 / 9)),
    sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
  };

  function clean(value, digits = 2) {
    const rounded = Number(value.toFixed(digits));
    return Object.is(rounded, -0) ? 0 : rounded;
  }

  function formatNumber(value) {
    const rounded = clean(value, 2);
    if (Number.isInteger(rounded)) return String(rounded);
    return rounded.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  }

  function convolveValid(image, kernel) {
    const outputHeight = image.length - kernel.length + 1;
    const outputWidth = image[0].length - kernel[0].length + 1;
    return Array.from({ length: outputHeight }, (_, outputRow) =>
      Array.from({ length: outputWidth }, (_, outputColumn) => {
        let sum = 0;
        kernel.forEach((kernelRow, row) => {
          kernelRow.forEach((weight, column) => {
            sum += image[outputRow + row][outputColumn + column] * weight;
          });
        });
        return sum;
      })
    );
  }

  function initializeConvolutionTrace() {
    const imageGrid = $("#convImageGrid");
    const kernelGrid = $("#convKernelGrid");
    const productGrid = $("#convProductGrid");
    const outputGrid = $("#convOutputGrid");
    const stepControl = $("#convStep");
    const kernelControl = $("#convKernel");
    const playButton = $("#convPlay");
    if (!imageGrid || !stepControl) return;

    let patternName = "vertical";
    let timer = null;

    function cellsFor(matrix, className = "") {
      return matrix.flatMap((row, rowIndex) =>
        row.map((value, columnIndex) =>
          `<span class="${className}" data-row="${rowIndex}" data-column="${columnIndex}">${formatNumber(value)}</span>`
        )
      ).join("");
    }

    function stopPlayback() {
      if (timer) window.clearInterval(timer);
      timer = null;
      playButton.textContent = "자동 진행";
    }

    function render() {
      const image = TRACE_PATTERNS[patternName];
      const kernel = KERNELS[kernelControl.value];
      const output = convolveValid(image, kernel);
      const step = Number(stepControl.value);
      const outputRow = Math.floor(step / 3);
      const outputColumn = step % 3;
      const patch = Array.from({ length: 3 }, (_, row) =>
        Array.from({ length: 3 }, (_, column) => image[outputRow + row][outputColumn + column])
      );
      const products = patch.map((row, rowIndex) =>
        row.map((value, columnIndex) => value * kernel[rowIndex][columnIndex])
      );

      imageGrid.innerHTML = cellsFor(image);
      $$('span', imageGrid).forEach((cell) => {
        const row = Number(cell.dataset.row);
        const column = Number(cell.dataset.column);
        cell.classList.toggle(
          "is-patch",
          row >= outputRow && row < outputRow + 3 && column >= outputColumn && column < outputColumn + 3
        );
      });
      kernelGrid.innerHTML = cellsFor(kernel);
      productGrid.innerHTML = cellsFor(products);
      outputGrid.innerHTML = output.flatMap((row, rowIndex) =>
        row.map((value, columnIndex) => {
          const index = rowIndex * 3 + columnIndex;
          const shown = index <= step ? formatNumber(value) : "·";
          const current = index === step ? " is-current" : "";
          return `<span class="${current}" data-row="${rowIndex}" data-column="${columnIndex}">${shown}</span>`;
        })
      ).join("");

      const factors = products.flat().map(formatNumber);
      const result = output[outputRow][outputColumn];
      $("#convStepValue").textContent = `${step + 1} / 9`;
      $("#convEquation").innerHTML =
        `<strong>출력 [${outputRow + 1}, ${outputColumn + 1}]</strong>` +
        `<span>${factors.join(" + ")} = <b>${formatNumber(result)}</b></span>`;
      $$("#convPatterns button").forEach((button) => {
        button.classList.toggle("is-active", button.dataset.pattern === patternName);
      });
    }

    $$("#convPatterns button").forEach((button) => {
      button.addEventListener("click", () => {
        patternName = button.dataset.pattern;
        stepControl.value = 0;
        stopPlayback();
        render();
      });
    });
    kernelControl.addEventListener("change", () => {
      stepControl.value = 0;
      stopPlayback();
      render();
    });
    stepControl.addEventListener("input", () => {
      stopPlayback();
      render();
    });
    playButton.addEventListener("click", () => {
      if (timer) {
        stopPlayback();
        return;
      }
      if (Number(stepControl.value) >= 8) stepControl.value = 0;
      playButton.textContent = "일시정지";
      timer = window.setInterval(() => {
        if (Number(stepControl.value) >= 8) {
          stopPlayback();
          return;
        }
        stepControl.value = Number(stepControl.value) + 1;
        render();
      }, 650);
    });
    render();
  }

  function initializeShapeCalculator() {
    const input = $("#shapeInput");
    const kernel = $("#shapeKernel");
    const padding = $("#shapePadding");
    const stride = $("#shapeStride");
    if (!input || !kernel || !padding || !stride) return;

    function render() {
      const n = Number(input.value);
      const k = Number(kernel.value);
      const p = Number(padding.value);
      const s = Number(stride.value);
      const output = Math.floor((n + 2 * p - k) / s) + 1;
      $("#shapeInputValue").textContent = n;
      $("#shapeInputLabel").textContent = `${n} × ${n}`;
      $("#shapeOutput").textContent = `${output} × ${output}`;
      $("#shapeSubstitution").textContent = `⌊(${n} + 2×${p} − ${k}) / ${s}⌋ + 1 = ${output}`;
      let interpretation = `필터 중심이 한 축에서 ${output}개 위치를 방문합니다.`;
      if (output === n) interpretation = `${k}×${k} 필터와 패딩 ${p}, 보폭 ${s}은 가로·세로 크기를 유지합니다.`;
      if (output < n) interpretation = `출력이 입력보다 작아져 공간 해상도가 ${n}에서 ${output}(으)로 줄어듭니다.`;
      if (output > n) interpretation = `큰 패딩 때문에 출력이 입력보다 ${output - n}칸 커집니다.`;
      $("#shapeInterpretation").textContent = interpretation;
    }

    [input, kernel, padding, stride].forEach((control) => {
      control.addEventListener(control.type === "range" ? "input" : "change", render);
    });
    render();
  }

  function createGalleryPattern(name, size = 16) {
    const image = Array.from({ length: size }, () => Array(size).fill(0.08));
    const paint = (row, column, value = 1) => {
      if (row >= 0 && row < size && column >= 0 && column < size) image[row][column] = value;
    };
    if (name === "vertical" || name === "cross") {
      for (let row = 2; row < size - 2; row += 1) {
        paint(row, 7);
        paint(row, 8);
      }
    }
    if (name === "horizontal" || name === "cross") {
      for (let column = 2; column < size - 2; column += 1) {
        paint(7, column);
        paint(8, column);
      }
    }
    if (name === "diagonal") {
      for (let index = 2; index < size - 2; index += 1) {
        paint(index, index);
        paint(index, index + 1);
      }
    }
    if (name === "box") {
      for (let index = 3; index <= 12; index += 1) {
        paint(3, index);
        paint(12, index);
        paint(index, 3);
        paint(index, 12);
      }
    }
    return image;
  }

  function convolveSame(image, kernel) {
    const size = image.length;
    return Array.from({ length: size }, (_, row) =>
      Array.from({ length: size }, (_, column) => {
        let sum = 0;
        kernel.forEach((kernelRow, kernelY) => {
          kernelRow.forEach((weight, kernelX) => {
            const imageY = row + kernelY - 1;
            const imageX = column + kernelX - 1;
            const pixel = imageY >= 0 && imageY < size && imageX >= 0 && imageX < size
              ? image[imageY][imageX]
              : 0;
            sum += pixel * weight;
          });
        });
        return sum;
      })
    );
  }

  function drawMatrix(canvas, matrix, isInput = false) {
    const context = canvas.getContext("2d");
    const size = matrix.length;
    const cell = canvas.width / size;
    const maximum = Math.max(0.001, ...matrix.flat().map(Math.abs));
    context.clearRect(0, 0, canvas.width, canvas.height);
    matrix.forEach((row, rowIndex) => {
      row.forEach((value, columnIndex) => {
        if (isInput) {
          const lightness = 96 - (value / maximum) * 62;
          context.fillStyle = `hsl(347 66% ${lightness}%)`;
        } else if (value >= 0) {
          const alpha = 0.06 + (Math.abs(value) / maximum) * 0.88;
          context.fillStyle = `rgba(57, 115, 95, ${alpha})`;
        } else {
          const alpha = 0.06 + (Math.abs(value) / maximum) * 0.88;
          context.fillStyle = `rgba(124, 25, 45, ${alpha})`;
        }
        context.fillRect(columnIndex * cell, rowIndex * cell, Math.ceil(cell), Math.ceil(cell));
      });
    });
  }

  function initializeFilterGallery() {
    const patternControl = $("#galleryPattern");
    const inputCanvas = $("#galleryInput");
    if (!patternControl || !inputCanvas) return;
    const observations = {
      vertical: "세로 막대의 양쪽 경계가 세로선 필터에서 가장 강하게 나타납니다.",
      horizontal: "가로 막대의 위아래 경계가 가로선 필터에서 가장 강하게 나타납니다.",
      cross: "십자의 세로 부분과 가로 부분이 서로 다른 필터에서 강조됩니다.",
      diagonal: "수직·수평 필터가 대각선을 짧은 조각들의 변화로 나누어 반응합니다.",
      box: "사각형의 네 변이 방향에 따라 두 경계 필터로 나뉘어 나타납니다.",
    };

    function render() {
      const image = createGalleryPattern(patternControl.value);
      drawMatrix(inputCanvas, image, true);
      $$('canvas[data-filter]').forEach((canvas) => {
        drawMatrix(canvas, convolveSame(image, KERNELS[canvas.dataset.filter]));
      });
      $("#galleryObservation").textContent = observations[patternControl.value];
    }
    patternControl.addEventListener("change", render);
    render();
  }

  function initializeCodeStudy() {
    const code = $("#convCode");
    if (!code) return;
    const lines = $$('[data-line]', code);
    const notes = $$('.code-notes button[data-highlight]');
    notes.forEach((note) => {
      note.addEventListener("click", () => {
        const [start, end] = note.dataset.highlight.split("-").map(Number);
        notes.forEach((item) => item.classList.toggle("is-active", item === note));
        lines.forEach((line) => {
          const number = Number(line.dataset.line);
          line.classList.toggle("is-highlighted", number >= start && number <= end);
        });
      });
    });
    notes[0]?.click();
  }

  function initializeCopyButtons() {
    $$(".copy-button").forEach((button) => {
      button.addEventListener("click", async () => {
        const target = button.dataset.copyTarget && $(`#${button.dataset.copyTarget}`);
        const text = button.dataset.copyText || target?.innerText || "";
        try {
          await navigator.clipboard.writeText(text);
          const previous = button.textContent;
          button.textContent = "복사됨";
          window.setTimeout(() => { button.textContent = previous; }, 1200);
        } catch {
          button.textContent = "복사 실패";
        }
      });
    });
  }

  function initializeQuiz() {
    const checkButton = $("#checkQuiz");
    if (!checkButton) return;
    checkButton.addEventListener("click", () => {
      let correct = 0;
      const quizzes = $$(".quiz");
      quizzes.forEach((quiz) => {
        const selected = $("input:checked", quiz)?.value;
        const isCorrect = selected === quiz.dataset.answer;
        correct += Number(isCorrect);
        quiz.classList.toggle("is-correct", isCorrect);
        quiz.classList.toggle("is-wrong", Boolean(selected) && !isCorrect);
        $(".quiz-feedback", quiz).textContent = !selected
          ? "답을 선택해 주세요."
          : isCorrect
            ? "정답입니다."
            : "다시 생각해 보세요. 필터의 이동과 출력 크기 식을 확인하면 답을 찾을 수 있습니다.";
      });
      $("#quizScore").textContent = `${quizzes.length}문제 중 ${correct}문제를 맞혔습니다.`;
    });
  }

  function initializeSectionNavigation() {
    const links = $$(".chapter-sidebar nav a");
    const sections = links.map((link) => $(link.getAttribute("href"))).filter(Boolean);
    if (!("IntersectionObserver" in window)) return;
    const observer = new IntersectionObserver((entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      links.forEach((link) => {
        link.classList.toggle("is-current", link.getAttribute("href") === `#${visible.target.id}`);
      });
    }, { rootMargin: "-20% 0px -65% 0px", threshold: [0, 0.2, 0.5] });
    sections.forEach((section) => observer.observe(section));
  }

  initializeConvolutionTrace();
  initializeShapeCalculator();
  initializeFilterGallery();
  initializeCodeStudy();
  initializeCopyButtons();
  initializeQuiz();
  initializeSectionNavigation();
})();
