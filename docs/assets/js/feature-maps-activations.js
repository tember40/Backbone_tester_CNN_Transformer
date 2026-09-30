(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  function clean(value, digits = 2) {
    const rounded = Number(value.toFixed(digits));
    return Object.is(rounded, -0) ? 0 : rounded;
  }

  function formatNumber(value) {
    const rounded = clean(value, 2);
    return Number.isInteger(rounded)
      ? String(rounded)
      : rounded.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  }

  function initializeTensorExplorer() {
    const controls = {
      batch: $("#tensorBatch"),
      channels: $("#tensorChannels"),
      height: $("#tensorHeight"),
      width: $("#tensorWidth"),
    };
    if (!controls.batch || !controls.channels) return;

    function render() {
      const n = Number(controls.batch.value);
      const c = Number(controls.channels.value);
      const h = Number(controls.height.value);
      const w = Number(controls.width.value);
      $("#tensorChannelsValue").textContent = c;
      $("#tensorShape").textContent = `[${n}, ${c}, ${h}, ${w}]`;
      $("#tensorCount").textContent =
        `전체 ${(n * c * h * w).toLocaleString("ko-KR")}개 값 · 표본당 ${c}개 특징맵`;
      $("#tensorStack").innerHTML = Array.from({ length: c }, (_, index) =>
        `<div class="tensor-sheet" style="--sheet:${index};--channel-hue:${342 - index * 9}">` +
        `<span>C${index + 1}</span><small>${h}×${w}</small></div>`
      ).reverse().join("");
    }

    Object.values(controls).forEach((control) => {
      control.addEventListener(control.type === "range" ? "input" : "change", render);
    });
    render();
  }

  const RAW_ACTIVATIONS = Array.from({ length: 7 }, (_, row) =>
    Array.from({ length: 7 }, (_, column) => {
      const ridge = 2.4 - Math.abs(column - 3) * 1.05;
      const verticalWave = Math.sin((row + 1) * 1.15) * 0.85;
      return clean(ridge + verticalWave - 1.1, 2);
    })
  );

  function activationColor(value, maximum) {
    const intensity = Math.min(1, Math.abs(value) / Math.max(maximum, 0.001));
    if (value > 0) return `rgba(57, 115, 95, ${0.08 + intensity * 0.82})`;
    if (value < 0) return `rgba(124, 25, 45, ${0.08 + intensity * 0.82})`;
    return "#f1eee9";
  }

  function initializeActivationLab() {
    const functionControl = $("#activationFunction");
    const biasControl = $("#activationBias");
    const rawGrid = $("#rawActivationGrid");
    const afterGrid = $("#afterActivationGrid");
    if (!functionControl || !biasControl || !rawGrid || !afterGrid) return;

    function activate(value) {
      if (functionControl.value === "relu") return Math.max(0, value);
      if (functionControl.value === "leaky") return value >= 0 ? value : value * 0.1;
      return value;
    }

    function describeCell(rawValue, activatedValue) {
      const name = functionControl.value === "relu"
        ? "ReLU"
        : functionControl.value === "leaky"
          ? "Leaky ReLU"
          : "항등 함수";
      $("#activationInterpretation").textContent =
        `선택한 값: ${name}(${formatNumber(rawValue)}) = ${formatNumber(activatedValue)}. ` +
        (rawValue < 0 && functionControl.value === "relu"
          ? "음수 응답이 차단되었습니다."
          : "이 응답은 다음 층으로 전달됩니다.");
    }

    function render() {
      const bias = Number(biasControl.value);
      const raw = RAW_ACTIVATIONS.map((row) => row.map((value) => value + bias));
      const after = raw.map((row) => row.map(activate));
      const rawMaximum = Math.max(...raw.flat().map(Math.abs));
      const afterMaximum = Math.max(...after.flat().map(Math.abs), 0.001);
      const createCells = (matrix, maximum, prefix) => matrix.flatMap((row, rowIndex) =>
        row.map((value, columnIndex) =>
          `<button type="button" data-row="${rowIndex}" data-column="${columnIndex}" ` +
          `data-value="${value}" aria-label="${prefix} ${rowIndex + 1}행 ${columnIndex + 1}열, ${formatNumber(value)}" ` +
          `style="background:${activationColor(value, maximum)}">${formatNumber(value)}</button>`
        )
      ).join("");
      rawGrid.innerHTML = createCells(raw, rawMaximum, "활성화 전");
      afterGrid.innerHTML = createCells(after, afterMaximum, "활성화 후");

      const flat = after.flat();
      const positiveRatio = flat.filter((value) => value > 0).length / flat.length;
      const zeroRatio = flat.filter((value) => Math.abs(value) < 1e-9).length / flat.length;
      const minimum = Math.min(...flat);
      const maximum = Math.max(...flat);
      const metadata = {
        relu: ["ReLU", "max(0, z)", "ReLU는 음수 응답을 0으로 바꾸고 양수 응답은 그대로 유지합니다."],
        leaky: ["Leaky ReLU", "max(0.1z, z)", "Leaky ReLU는 음수 구간을 완전히 지우지 않고 10%만 남깁니다."],
        identity: ["적용 안 함", "a = z", "활성화를 적용하지 않으면 음수와 양수가 모두 그대로 전달됩니다."],
      }[functionControl.value];
      $("#activationName").textContent = metadata[0];
      $("#activationFormula").textContent = metadata[1];
      $("#activationBiasValue").textContent = bias.toFixed(2);
      $("#positiveRatio").textContent = `${(positiveRatio * 100).toFixed(0)}%`;
      $("#zeroRatio").textContent = `${(zeroRatio * 100).toFixed(0)}%`;
      $("#activationRange").textContent = `${formatNumber(minimum)} ~ ${formatNumber(maximum)}`;
      $("#activationInterpretation").textContent = metadata[2];

      $$('button', rawGrid).forEach((button) => {
        button.addEventListener("click", () => {
          const row = Number(button.dataset.row);
          const column = Number(button.dataset.column);
          describeCell(raw[row][column], after[row][column]);
          $$('button', rawGrid).forEach((cell) => cell.classList.toggle("is-selected", cell === button));
          $$('button', afterGrid).forEach((cell) => {
            cell.classList.toggle(
              "is-selected",
              Number(cell.dataset.row) === row && Number(cell.dataset.column) === column
            );
          });
        });
      });
      $$('button', afterGrid).forEach((button) => {
        button.addEventListener("click", () => {
          const row = Number(button.dataset.row);
          const column = Number(button.dataset.column);
          describeCell(raw[row][column], after[row][column]);
          $$('button', afterGrid).forEach((cell) => cell.classList.toggle("is-selected", cell === button));
          $$('button', rawGrid).forEach((cell) => {
            cell.classList.toggle(
              "is-selected",
              Number(cell.dataset.row) === row && Number(cell.dataset.column) === column
            );
          });
        });
      });
    }

    functionControl.addEventListener("change", render);
    biasControl.addEventListener("input", render);
    render();
  }

  const CHANNEL_KERNELS = [
    { name: "세로 경계", matrix: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]] },
    { name: "가로 경계", matrix: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]] },
    { name: "↘ 대각선", matrix: [[-1, -1, 0], [-1, 0, 1], [0, 1, 1]] },
    { name: "↗ 대각선", matrix: [[0, 1, 1], [-1, 0, 1], [-1, -1, 0]] },
    { name: "주변 평균", matrix: Array.from({ length: 3 }, () => Array(3).fill(1 / 9)) },
    { name: "중심 강조", matrix: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] },
  ];

  function createPattern(name, size = 18) {
    const image = Array.from({ length: size }, () => Array(size).fill(0.06));
    const paint = (row, column, value = 1) => {
      if (row >= 0 && row < size && column >= 0 && column < size) image[row][column] = value;
    };
    if (name === "cross") {
      for (let index = 3; index < size - 3; index += 1) {
        paint(8, index); paint(9, index); paint(index, 8); paint(index, 9);
      }
    } else if (name === "box") {
      for (let index = 3; index <= 14; index += 1) {
        paint(3, index); paint(14, index); paint(index, 3); paint(index, 14);
      }
    } else if (name === "diagonal") {
      for (let index = 2; index < size - 2; index += 1) {
        paint(index, index); paint(index, index + 1);
      }
    } else {
      for (let column = 4; column <= 13; column += 1) {
        const roofRow = 8 - Math.floor(Math.abs(column - 8.5));
        paint(roofRow, column); paint(roofRow + 1, column);
      }
      for (let row = 8; row <= 15; row += 1) {
        paint(row, 5); paint(row, 13);
      }
      for (let column = 5; column <= 13; column += 1) paint(15, column);
      for (let row = 11; row <= 15; row += 1) { paint(row, 8); paint(row, 9); }
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
            const y = row + kernelY - 1;
            const x = column + kernelX - 1;
            const pixel = y >= 0 && y < size && x >= 0 && x < size ? image[y][x] : 0;
            sum += pixel * weight;
          });
        });
        return sum;
      })
    );
  }

  function drawMatrix(canvas, matrix, isInput = false) {
    const context = canvas.getContext("2d");
    const cell = canvas.width / matrix.length;
    const maximum = Math.max(0.001, ...matrix.flat().map(Math.abs));
    context.clearRect(0, 0, canvas.width, canvas.height);
    matrix.forEach((row, rowIndex) => {
      row.forEach((value, columnIndex) => {
        if (isInput) {
          const lightness = 96 - (value / maximum) * 62;
          context.fillStyle = `hsl(347 66% ${lightness}%)`;
        } else if (value > 0) {
          context.fillStyle = `rgba(57,115,95,${0.06 + Math.abs(value / maximum) * 0.88})`;
        } else if (value < 0) {
          context.fillStyle = `rgba(124,25,45,${0.06 + Math.abs(value / maximum) * 0.88})`;
        } else {
          context.fillStyle = "#f2efeb";
        }
        context.fillRect(columnIndex * cell, rowIndex * cell, Math.ceil(cell), Math.ceil(cell));
      });
    });
  }

  function initializeChannelLab() {
    const patternControl = $("#channelPattern");
    const activationControl = $("#channelActivation");
    const inputCanvas = $("#channelInput");
    const mapList = $("#channelMapList");
    if (!patternControl || !activationControl || !inputCanvas || !mapList) return;
    const observations = {
      house: "방향별 경계와 중심 영역이 서로 다른 채널로 분리됩니다.",
      cross: "수직과 수평 채널이 십자의 서로 다른 팔에 강하게 반응합니다.",
      box: "사각형의 네 변과 모서리가 방향별 채널로 나뉘어 나타납니다.",
      diagonal: "두 대각선 채널 중 입력 방향과 일치하는 채널의 응답이 더 강합니다.",
    };

    function render() {
      const image = createPattern(patternControl.value);
      const rawMaps = CHANNEL_KERNELS.map((filter) => convolveSame(image, filter.matrix));
      const displayedMaps = activationControl.value === "relu"
        ? rawMaps.map((map) => map.map((row) => row.map((value) => Math.max(0, value))))
        : rawMaps;
      drawMatrix(inputCanvas, image, true);
      mapList.innerHTML = CHANNEL_KERNELS.map((filter, index) =>
        `<figure><figcaption><strong>C${index + 1}</strong><span>${filter.name}</span></figcaption>` +
        `<canvas width="216" height="216" data-channel="${index}" aria-label="${filter.name} 특징맵"></canvas></figure>`
      ).join("");
      $$('canvas[data-channel]', mapList).forEach((canvas) => {
        drawMatrix(canvas, displayedMaps[Number(canvas.dataset.channel)]);
      });
      const flat = displayedMaps.flat(2);
      const zeroRatio = flat.filter((value) => Math.abs(value) < 1e-9).length / flat.length;
      const strengths = displayedMaps.map((map) => map.flat().reduce((sum, value) => sum + Math.abs(value), 0));
      const strongestIndex = strengths.indexOf(Math.max(...strengths));
      $("#channelObservation").textContent = observations[patternControl.value];
      $("#channelSummary").innerHTML =
        `<div><span>출력 shape</span><strong>[1, 6, 18, 18]</strong></div>` +
        `<div><span>가장 강한 채널</span><strong>C${strongestIndex + 1} · ${CHANNEL_KERNELS[strongestIndex].name}</strong></div>` +
        `<div><span>0의 비율</span><strong>${(zeroRatio * 100).toFixed(0)}%</strong></div>`;
    }
    patternControl.addEventListener("change", render);
    activationControl.addEventListener("change", render);
    render();
  }

  function initializeCodeStudy() {
    const code = $("#activationCode");
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
            : "다시 생각해 보세요. 텐서 차원과 ReLU 계산을 확인하면 답을 찾을 수 있습니다.";
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

  initializeTensorExplorer();
  initializeActivationLab();
  initializeChannelLab();
  initializeCodeStudy();
  initializeCopyButtons();
  initializeQuiz();
  initializeSectionNavigation();
})();
