(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const FEATURES = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const TARGETS = [0, 1, 1, 0];

  function clean(value, digits = 2) {
    const rounded = Number(value.toFixed(digits));
    return Object.is(rounded, -0) ? 0 : rounded;
  }

  function signed(value, digits = 2) {
    const rounded = clean(value, digits);
    return rounded < 0 ? `−${Math.abs(rounded).toFixed(digits)}` : rounded.toFixed(digits);
  }

  function sigmoid(value) {
    return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, value))));
  }

  function initializeForwardTrace() {
    const buttons = $$(".input-choice button[data-input]");
    const activation = $("#activationMode");
    if (!buttons.length || !activation) return;
    let input = [0, 1];

    function render() {
      const [x1, x2] = input;
      const z1 = x1 - x2;
      const z2 = x2 - x1;
      const activate = activation.value === "relu"
        ? (value) => Math.max(0, value)
        : (value) => value;
      const h1 = activate(z1);
      const h2 = activate(z2);
      const score = h1 + h2 - 0.5;
      const prediction = score >= 0 ? 1 : 0;
      const target = x1 === x2 ? 0 : 1;

      $("#traceX1").textContent = x1;
      $("#traceX2").textContent = x2;
      $("#traceZ1").textContent = signed(z1, 1);
      $("#traceZ2").textContent = signed(z2, 1);
      $("#traceH1").textContent = signed(h1, 1);
      $("#traceH2").textContent = signed(h2, 1);
      $("#traceScore").textContent = signed(score, 1);
      $("#tracePrediction").textContent = prediction;
      buttons.forEach((button) => {
        button.classList.toggle("is-active", button.dataset.input === input.join(","));
      });

      const correct = prediction === target;
      const mode = activation.value === "relu" ? "ReLU" : "활성화 없음";
      $("#forwardExplanation").classList.toggle("is-warning", !correct);
      $("#forwardExplanation").innerHTML =
        `<strong>${mode} · XOR 정답 ${target} · 예측 ${prediction}</strong>` +
        `<span>${activation.value === "relu"
          ? "음수인 은닉 값을 0으로 꺾어 서로 다른 두 경우를 별도 특징으로 보존합니다."
          : "h₁+h₂가 항상 0이 되어 네 입력을 구분할 중간 표현이 사라집니다."}</span>`;
    }

    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        input = button.dataset.input.split(",").map(Number);
        render();
      });
    });
    activation.addEventListener("change", render);
    render();
  }

  function initializeTrainingLab() {
    const boundaryCanvas = $("#mlpBoundaryCanvas");
    const lossCanvas = $("#mlpLossCanvas");
    if (!boundaryCanvas || !lossCanvas) return;
    const boundaryContext = boundaryCanvas.getContext("2d");
    const lossContext = lossCanvas.getContext("2d");
    const controls = {
      rate: $("#mlpLearningRate"),
      reset: $("#mlpReset"),
      step: $("#mlpStep"),
      hundred: $("#mlpHundred"),
      auto: $("#mlpAuto"),
    };
    let model;
    let timer = null;

    function seededRandom(seed = 19) {
      let state = seed >>> 0;
      return () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 4294967296;
      };
    }

    function createModel() {
      const random = seededRandom();
      return {
        w1: Array.from({ length: 4 }, () =>
          Array.from({ length: 2 }, () => (random() - 0.5) * 1.6)
        ),
        b1: [0, 0, 0, 0],
        w2: Array.from({ length: 4 }, () => (random() - 0.5) * 1.6),
        b2: 0,
        epoch: 0,
        history: [],
      };
    }

    function forward(sample) {
      const hiddenLinear = model.w1.map((weights, index) =>
        weights[0] * sample[0] + weights[1] * sample[1] + model.b1[index]
      );
      const hidden = hiddenLinear.map(Math.tanh);
      const logit = hidden.reduce(
        (sum, value, index) => sum + value * model.w2[index],
        model.b2
      );
      return { hiddenLinear, hidden, logit, probability: sigmoid(logit) };
    }

    function evaluate() {
      const outputs = FEATURES.map(forward);
      let loss = 0;
      let correct = 0;
      outputs.forEach((output, index) => {
        const probability = Math.min(1 - 1e-7, Math.max(1e-7, output.probability));
        const target = TARGETS[index];
        loss += -(target * Math.log(probability) + (1 - target) * Math.log(1 - probability));
        correct += Number((probability >= 0.5 ? 1 : 0) === target);
      });
      return { outputs, loss: loss / FEATURES.length, accuracy: correct / FEATURES.length };
    }

    function trainOneEpoch() {
      const gradW1 = Array.from({ length: 4 }, () => [0, 0]);
      const gradB1 = [0, 0, 0, 0];
      const gradW2 = [0, 0, 0, 0];
      let gradB2 = 0;

      FEATURES.forEach((sample, sampleIndex) => {
        const output = forward(sample);
        const deltaOutput = (output.probability - TARGETS[sampleIndex]) / FEATURES.length;
        output.hidden.forEach((hiddenValue, hiddenIndex) => {
          gradW2[hiddenIndex] += deltaOutput * hiddenValue;
          const deltaHidden = deltaOutput * model.w2[hiddenIndex] * (1 - hiddenValue ** 2);
          gradW1[hiddenIndex][0] += deltaHidden * sample[0];
          gradW1[hiddenIndex][1] += deltaHidden * sample[1];
          gradB1[hiddenIndex] += deltaHidden;
        });
        gradB2 += deltaOutput;
      });

      const rate = Number(controls.rate.value);
      model.w1.forEach((weights, hiddenIndex) => {
        weights[0] -= rate * gradW1[hiddenIndex][0];
        weights[1] -= rate * gradW1[hiddenIndex][1];
        model.b1[hiddenIndex] -= rate * gradB1[hiddenIndex];
        model.w2[hiddenIndex] -= rate * gradW2[hiddenIndex];
      });
      model.b2 -= rate * gradB2;
      model.epoch += 1;
      const metrics = evaluate();
      model.history.push({ epoch: model.epoch, loss: metrics.loss, accuracy: metrics.accuracy });
      return metrics;
    }

    function reset() {
      if (timer) window.clearInterval(timer);
      timer = null;
      controls.auto.textContent = "자동 학습";
      model = createModel();
      const metrics = evaluate();
      model.history.push({ epoch: 0, loss: metrics.loss, accuracy: metrics.accuracy });
      render(metrics, "무작위 가중치에서 시작했습니다. 1 epoch로 변화 방향을 먼저 확인해 보세요.");
    }

    function trainMany(count) {
      let metrics;
      for (let index = 0; index < count; index += 1) metrics = trainOneEpoch();
      return metrics;
    }

    function toggleAuto() {
      if (timer) {
        window.clearInterval(timer);
        timer = null;
        controls.auto.textContent = "계속 학습";
        return;
      }
      controls.auto.textContent = "일시정지";
      timer = window.setInterval(() => {
        const metrics = trainMany(10);
        render(metrics, "여러 epoch를 연속 실행하며 결정 영역을 갱신하고 있습니다.");
        if (model.epoch >= 2000 || (metrics.accuracy === 1 && metrics.loss < 0.03)) {
          window.clearInterval(timer);
          timer = null;
          controls.auto.textContent = "다시 학습";
          render(metrics, "네 표본을 모두 분류했고 손실도 충분히 낮아졌습니다.");
        }
      }, 45);
    }

    function render(metrics = evaluate(), message = "") {
      $("#mlpEpoch").textContent = model.epoch;
      $("#mlpLoss").textContent = metrics.loss.toFixed(4);
      $("#mlpAccuracy").textContent = `${(metrics.accuracy * 100).toFixed(0)}%`;
      $("#mlpTruthResults").innerHTML = FEATURES.map((sample, index) => {
        const probability = metrics.outputs[index].probability;
        const prediction = probability >= 0.5 ? 1 : 0;
        const correct = prediction === TARGETS[index];
        return `<div class="${correct ? "is-correct" : "is-wrong"}">` +
          `<span>(${sample.join(", ")})</span><strong>${probability.toFixed(2)}</strong>` +
          `<small>예측 ${prediction} / 정답 ${TARGETS[index]}</small></div>`;
      }).join("");
      $("#mlpStatusText").textContent = message || (
        metrics.accuracy === 1
          ? "네 표본을 모두 맞혔습니다. 손실이 더 감소하면 예측 확률의 확신도 커집니다."
          : "아직 오답이 남아 있습니다. epoch를 늘리며 손실과 결정 영역을 함께 확인하세요."
      );
      drawDecisionSurface(boundaryContext, boundaryCanvas);
      drawLossCurve(lossContext, lossCanvas);
    }

    controls.rate.addEventListener("change", reset);
    controls.reset.addEventListener("click", reset);
    controls.step.addEventListener("click", () => render(trainOneEpoch()));
    controls.hundred.addEventListener("click", () => render(trainMany(100)));
    controls.auto.addEventListener("click", toggleAuto);

    function drawDecisionSurface(context, canvas) {
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
      for (let y = margin; y < height - margin; y += 7) {
        for (let x = margin; x < width - margin; x += 7) {
          const probability = forward([fromX(x), fromY(y)]).probability;
          context.fillStyle = probability >= 0.5
            ? `rgba(57,115,95,${0.045 + Math.abs(probability - 0.5) * 0.15})`
            : `rgba(124,25,45,${0.035 + Math.abs(probability - 0.5) * 0.13})`;
          context.fillRect(x, y, 7, 7);
        }
      }
      context.restore();

      context.strokeStyle = "#dfdbd0";
      context.lineWidth = 1;
      [0, 0.5, 1].forEach((tick) => {
        context.beginPath(); context.moveTo(toX(tick), margin); context.lineTo(toX(tick), height - margin); context.stroke();
        context.beginPath(); context.moveTo(margin, toY(tick)); context.lineTo(width - margin, toY(tick)); context.stroke();
      });

      FEATURES.forEach((sample, index) => {
        const output = forward(sample);
        const prediction = output.probability >= 0.5 ? 1 : 0;
        const correct = prediction === TARGETS[index];
        const x = toX(sample[0]);
        const y = toY(sample[1]);
        context.beginPath();
        context.arc(x, y, 15, 0, Math.PI * 2);
        context.fillStyle = TARGETS[index] ? "#39735f" : "#7c192d";
        context.fill();
        context.lineWidth = correct ? 4 : 7;
        context.strokeStyle = correct ? "#ffffff" : "#fbae40";
        context.stroke();
        context.fillStyle = "#2b2524";
        context.font = "bold 17px 'Times New Roman', serif";
        context.textAlign = "center";
        context.fillText(`(${sample.join(",")}) → ${TARGETS[index]}`, x, y - 24);
      });

      context.fillStyle = "#6f625f";
      context.font = "14px 'Times New Roman', serif";
      context.textAlign = "center";
      [0, 0.5, 1].forEach((tick) => context.fillText(tick.toFixed(1), toX(tick), height - margin + 25));
      context.textAlign = "right";
      [0, 0.5, 1].forEach((tick) => context.fillText(tick.toFixed(1), margin - 12, toY(tick) + 4));
    }

    function drawLossCurve(context, canvas) {
      const { width, height } = canvas;
      const margin = { top: 16, right: 15, bottom: 30, left: 42 };
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const history = model.history;
      const maxEpoch = Math.max(1, history[history.length - 1].epoch);
      const maxLoss = Math.max(0.8, ...history.map((point) => point.loss));
      const toX = (epoch) => margin.left + (epoch / maxEpoch) * plotWidth;
      const toY = (loss) => margin.top + (1 - loss / maxLoss) * plotHeight;

      context.clearRect(0, 0, width, height);
      context.strokeStyle = "#dfdbd0";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(margin.left, margin.top);
      context.lineTo(margin.left, height - margin.bottom);
      context.lineTo(width - margin.right, height - margin.bottom);
      context.stroke();
      context.beginPath();
      history.forEach((point, index) => {
        const x = toX(point.epoch);
        const y = toY(point.loss);
        if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
      });
      context.strokeStyle = "#7c192d";
      context.lineWidth = 3;
      context.stroke();
      context.fillStyle = "#6f625f";
      context.font = "13px 'Times New Roman', serif";
      context.fillText("loss", 8, 18);
      context.textAlign = "right";
      context.fillText(`epoch ${maxEpoch}`, width - margin.right, height - 8);
    }

    reset();
  }

  function initializeCodeStudy() {
    const code = $("#mlpCode");
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
            : "다시 생각해 보세요. 순전파와 역전파 흐름을 확인하면 답을 찾을 수 있습니다.";
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

  initializeForwardTrace();
  initializeTrainingLab();
  initializeCodeStudy();
  initializeCopyButtons();
  initializeQuiz();
  initializeSectionNavigation();
})();
