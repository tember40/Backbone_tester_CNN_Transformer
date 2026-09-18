(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const clamp = (value, minimum, maximum) => Math.min(maximum, Math.max(minimum, value));
  const clean = (value, digits = 2) => {
    const rounded = Number(value.toFixed(digits));
    return Object.is(rounded, -0) ? 0 : rounded;
  };
  const signed = (value, digits = 2) => {
    const rounded = clean(value, digits);
    return rounded < 0 ? `−${Math.abs(rounded).toFixed(digits)}` : rounded.toFixed(digits);
  };

  function initializeCalculator() {
    const defaults = { x1: 1, x2: 0.5, w1: 0.8, w2: -0.4, bias: 0.1 };
    const controls = Object.fromEntries(
      Object.keys(defaults).map((id) => [id, $(`#${id}`)])
    );
    if (!controls.x1) return;

    function update() {
      const values = Object.fromEntries(
        Object.entries(controls).map(([id, control]) => [id, Number(control.value)])
      );
      const term1 = values.x1 * values.w1;
      const term2 = values.x2 * values.w2;
      const score = term1 + term2 + values.bias;
      const prediction = score >= 0 ? 1 : 0;

      Object.entries(values).forEach(([id, value]) => {
        const output = $(`#${id}Value`);
        output.textContent = value < 0 ? `−${Math.abs(value).toFixed(1)}` : value.toFixed(1);
      });
      $("#term1").textContent = signed(term1);
      $("#term2").textContent = signed(term2);
      $("#termBias").textContent = signed(values.bias);
      $("#scoreValue").textContent = signed(score);
      $("#calculationLine").textContent =
        `(${values.x1.toFixed(1)} × ${signed(values.w1, 1)}) + ` +
        `(${values.x2.toFixed(1)} × ${signed(values.w2, 1)}) + ${signed(values.bias, 1)}`;

      const markerPosition = clamp(50 + (score / 4) * 50, 2, 98);
      $("#scoreMarker").style.left = `${markerPosition}%`;
      $("#predictionValue").textContent = prediction;
      $("#predictionReason").textContent =
        prediction === 1 ? "z가 0 이상이므로 1입니다." : "z가 0보다 작으므로 0입니다.";
      $("#predictionCard").classList.toggle("is-zero", prediction === 0);
    }

    Object.values(controls).forEach((control) => control.addEventListener("input", update));
    $("#calculatorReset").addEventListener("click", () => {
      Object.entries(defaults).forEach(([id, value]) => { controls[id].value = value; });
      update();
    });
    update();
  }

  function initializeCodeStudy() {
    const code = $("#perceptronCode");
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
        const first = $(`[data-line="${start}"]`, code);
        first?.scrollIntoView({ block: "nearest", behavior: "smooth" });
      });
    });

    notes[0]?.click();
  }

  const FEATURES = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const GATES = {
    AND: [0, 0, 0, 1],
    OR: [0, 1, 1, 1],
    NAND: [1, 1, 1, 0],
    XOR: [0, 1, 1, 0],
  };

  function initializeTrainingLab() {
    const canvas = $("#boundaryCanvas");
    if (!canvas) return;
    const context = canvas.getContext("2d");
    const controls = {
      gate: $("#gateSelect"),
      rate: $("#learningRate"),
      epochs: $("#epochCount"),
      reset: $("#resetTraining"),
      step: $("#stepTraining"),
      run: $("#runTraining"),
    };
    let state;
    let timer = null;

    function reset() {
      if (timer) window.clearInterval(timer);
      timer = null;
      state = {
        weights: [0, 0],
        bias: 0,
        step: 0,
        currentSample: null,
        lastPrediction: null,
        lastTarget: null,
        lastError: null,
        logs: [],
      };
      controls.run.textContent = "자동 학습";
      controls.run.disabled = false;
      render();
    }

    function targets() { return GATES[controls.gate.value]; }
    function score(sample) {
      return sample[0] * state.weights[0] + sample[1] * state.weights[1] + state.bias;
    }
    function predict(sample) { return score(sample) >= 0 ? 1 : 0; }
    function accuracy() {
      const correct = FEATURES.reduce(
        (total, sample, index) => total + Number(predict(sample) === targets()[index]),
        0
      );
      return (correct / FEATURES.length) * 100;
    }
    function totalSteps() { return Number(controls.epochs.value) * FEATURES.length; }

    function trainOneStep() {
      if (state.step >= totalSteps()) return false;
      const sampleIndex = state.step % FEATURES.length;
      const epoch = Math.floor(state.step / FEATURES.length) + 1;
      const sample = FEATURES[sampleIndex];
      const target = targets()[sampleIndex];
      const prediction = predict(sample);
      const error = target - prediction;
      const rate = Number(controls.rate.value);

      state.weights[0] += rate * error * sample[0];
      state.weights[1] += rate * error * sample[1];
      state.bias += rate * error;
      state.currentSample = sampleIndex;
      state.lastPrediction = prediction;
      state.lastTarget = target;
      state.lastError = error;
      state.logs.unshift({
        epoch,
        sample,
        prediction,
        target,
        error,
        weights: [...state.weights],
        bias: state.bias,
      });
      state.logs = state.logs.slice(0, 12);
      state.step += 1;
      render();
      return true;
    }

    function run() {
      if (timer) {
        window.clearInterval(timer);
        timer = null;
        controls.run.textContent = "계속 학습";
        return;
      }
      if (state.step >= totalSteps()) reset();
      controls.run.textContent = "일시정지";
      timer = window.setInterval(() => {
        if (!trainOneStep()) {
          window.clearInterval(timer);
          timer = null;
          controls.run.textContent = "다시 학습";
          renderCompletion();
        }
      }, 430);
    }

    function renderCompletion() {
      const gate = controls.gate.value;
      const finalAccuracy = accuracy();
      const message = gate === "XOR"
        ? `학습을 마쳤지만 정확도는 ${finalAccuracy.toFixed(0)}%입니다. 직선 하나로 XOR을 완전히 분리할 수 없습니다.`
        : `학습을 마쳤습니다. 현재 정확도는 ${finalAccuracy.toFixed(0)}%입니다.`;
      $("#updateExplanation").textContent = message;
    }

    function render() {
      const epoch = state.step === 0 ? 0 : Math.ceil(state.step / FEATURES.length);
      $("#statusEpoch").textContent = `${epoch} / ${controls.epochs.value}`;
      $("#statusSample").textContent = state.currentSample === null
        ? "—"
        : `[${FEATURES[state.currentSample].join(", ")}]`;
      $("#statusPrediction").textContent = state.lastPrediction === null
        ? "—"
        : `${state.lastPrediction} / ${state.lastTarget}`;
      $("#statusError").textContent = state.lastError === null ? "—" : state.lastError;
      $("#statusWeights").textContent = `[${signed(state.weights[0])}, ${signed(state.weights[1])}]`;
      $("#statusBias").textContent = signed(state.bias);
      $("#statusAccuracy").textContent = `${accuracy().toFixed(0)}%`;

      if (state.lastError === null) {
        $("#updateExplanation").textContent = "첫 번째 표본을 확인할 준비가 되었습니다.";
      } else if (state.lastError === 0) {
        $("#updateExplanation").textContent = "예측과 정답이 같습니다. 오차가 0이므로 가중치와 편향은 변하지 않습니다.";
      } else {
        $("#updateExplanation").textContent =
          `오차 ${state.lastError > 0 ? "+" : ""}${state.lastError}가 발생했습니다. ` +
          `학습률 × 오차 × 입력만큼 가중치를 수정했습니다.`;
      }

      renderTruthTable();
      renderLog();
      drawBoundary(context, canvas, state.weights, state.bias, targets(), state.currentSample, predict);
    }

    function renderTruthTable() {
      $("#truthTable").innerHTML = FEATURES.map((sample, index) => {
        const current = index === state.currentSample ? " is-current" : "";
        return `<div class="${current.trim()}">${sample.join(",")} → <strong>${targets()[index]}</strong></div>`;
      }).join("");
    }

    function renderLog() {
      const log = $("#trainingLog");
      if (!state.logs.length) {
        log.innerHTML = '<li class="empty-log">한 단계 또는 자동 학습을 실행하세요.</li>';
        return;
      }
      log.innerHTML = state.logs.map((entry) => {
        const changed = entry.error !== 0 ? " is-update" : "";
        return `<li class="${changed.trim()}">E${entry.epoch} [${entry.sample.join(",")}] ` +
          `ŷ=${entry.prediction}, y=${entry.target}, e=${entry.error} · ` +
          `w=[${clean(entry.weights[0], 1)}, ${clean(entry.weights[1], 1)}], b=${clean(entry.bias, 1)}</li>`;
      }).join("");
    }

    [controls.gate, controls.rate, controls.epochs].forEach((control) => {
      control.addEventListener("change", reset);
    });
    controls.reset.addEventListener("click", reset);
    controls.step.addEventListener("click", () => {
      if (timer) return;
      if (!trainOneStep()) renderCompletion();
    });
    controls.run.addEventListener("click", run);
    reset();
  }

  function drawBoundary(context, canvas, weights, bias, targets, currentSample, predict) {
    const { width, height } = canvas;
    const margin = 54;
    const xMin = -0.25;
    const xMax = 1.25;
    const yMin = -0.25;
    const yMax = 1.25;
    const plotWidth = width - margin * 2;
    const plotHeight = height - margin * 2;
    const toX = (value) => margin + ((value - xMin) / (xMax - xMin)) * plotWidth;
    const toY = (value) => height - margin - ((value - yMin) / (yMax - yMin)) * plotHeight;
    const fromX = (pixel) => xMin + ((pixel - margin) / plotWidth) * (xMax - xMin);
    const fromY = (pixel) => yMin + ((height - margin - pixel) / plotHeight) * (yMax - yMin);

    context.clearRect(0, 0, width, height);
    context.save();
    context.beginPath();
    context.rect(margin, margin, plotWidth, plotHeight);
    context.clip();
    for (let y = margin; y < height - margin; y += 8) {
      for (let x = margin; x < width - margin; x += 8) {
        const prediction = predict([fromX(x), fromY(y)]);
        context.fillStyle = prediction ? "rgba(35,133,109,.075)" : "rgba(193,70,79,.055)";
        context.fillRect(x, y, 8, 8);
      }
    }
    context.restore();

    context.strokeStyle = "#dce2e7";
    context.lineWidth = 1;
    [0, 0.5, 1].forEach((tick) => {
      context.beginPath(); context.moveTo(toX(tick), margin); context.lineTo(toX(tick), height - margin); context.stroke();
      context.beginPath(); context.moveTo(margin, toY(tick)); context.lineTo(width - margin, toY(tick)); context.stroke();
    });

    const [w1, w2] = weights;
    context.strokeStyle = "#172f49";
    context.lineWidth = 4;
    context.beginPath();
    if (Math.abs(w2) > 1e-9) {
      const leftY = -(w1 * xMin + bias) / w2;
      const rightY = -(w1 * xMax + bias) / w2;
      context.moveTo(toX(xMin), toY(leftY));
      context.lineTo(toX(xMax), toY(rightY));
      context.stroke();
      $("#boundaryLabel").textContent = `${signed(w1)}x₁ + ${signed(w2)}x₂ + ${signed(bias)} = 0`;
    } else if (Math.abs(w1) > 1e-9) {
      const x = -bias / w1;
      context.moveTo(toX(x), margin);
      context.lineTo(toX(x), height - margin);
      context.stroke();
      $("#boundaryLabel").textContent = `x₁ = ${signed(x)}`;
    } else {
      $("#boundaryLabel").textContent = "가중치가 모두 0이라 경계 방향이 없습니다";
    }

    FEATURES.forEach((sample, index) => {
      const x = toX(sample[0]);
      const y = toY(sample[1]);
      const correct = predict(sample) === targets[index];
      context.beginPath();
      context.arc(x, y, index === currentSample ? 15 : 12, 0, Math.PI * 2);
      context.fillStyle = targets[index] ? "#23856d" : "#c1464f";
      context.fill();
      context.lineWidth = index === currentSample ? 5 : 3;
      context.strokeStyle = correct ? "#ffffff" : "#f0a13e";
      context.stroke();
      context.fillStyle = "#172033";
      context.font = "bold 15px Segoe UI";
      context.textAlign = "center";
      context.fillText(`(${sample.join(",")})`, x, y - 21);
    });

    context.fillStyle = "#667085";
    context.font = "12px Segoe UI";
    context.textAlign = "center";
    [0, 0.5, 1].forEach((tick) => context.fillText(tick.toFixed(1), toX(tick), height - margin + 25));
    context.textAlign = "right";
    [0, 0.5, 1].forEach((tick) => context.fillText(tick.toFixed(1), margin - 12, toY(tick) + 4));
    context.fillStyle = "#172033";
    context.font = "bold 13px Segoe UI";
    context.textAlign = "right";
    context.fillText("x₁", width - margin, height - 17);
    context.textAlign = "left";
    context.fillText("x₂", 17, margin);
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
        const selected = $('input:checked', quiz)?.value;
        const isCorrect = selected === quiz.dataset.answer;
        correct += Number(isCorrect);
        quiz.classList.toggle("is-correct", isCorrect);
        quiz.classList.toggle("is-wrong", Boolean(selected) && !isCorrect);
        $(".quiz-feedback", quiz).textContent = !selected
          ? "답을 선택해 주세요."
          : isCorrect
            ? "정답입니다."
            : "다시 생각해 보세요. 이 장의 실험 결과를 확인하면 답을 찾을 수 있습니다.";
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

  initializeCalculator();
  initializeCodeStudy();
  initializeTrainingLab();
  initializeCopyButtons();
  initializeQuiz();
  initializeSectionNavigation();
})();
