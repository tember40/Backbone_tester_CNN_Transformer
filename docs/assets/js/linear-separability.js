(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const FEATURES = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const GATES = {
    AND: [0, 0, 0, 1],
    OR: [0, 1, 1, 1],
    XOR: [0, 1, 1, 0],
  };

  function formatSigned(value, digits = 2) {
    const rounded = Number(value.toFixed(digits));
    if (Object.is(rounded, -0)) return (0).toFixed(digits);
    return rounded < 0 ? `−${Math.abs(rounded).toFixed(digits)}` : rounded.toFixed(digits);
  }

  function initializeBoundaryLab() {
    const canvas = $("#separabilityCanvas");
    if (!canvas) return;
    const context = canvas.getContext("2d");
    const controls = {
      gate: $("#boundaryGate"),
      angle: $("#boundaryAngle"),
      offset: $("#boundaryOffset"),
      reset: $("#boundaryReset"),
      best: $("#findBestBoundary"),
    };

    function parameters() {
      const degrees = Number(controls.angle.value);
      const radians = degrees * Math.PI / 180;
      return {
        degrees,
        weights: [Math.cos(radians), Math.sin(radians)],
        bias: Number(controls.offset.value),
      };
    }

    function evaluate(weights, bias, targets = GATES[controls.gate.value]) {
      const predictions = FEATURES.map(([x1, x2]) =>
        x1 * weights[0] + x2 * weights[1] + bias >= 0 ? 1 : 0
      );
      const correct = predictions.map((prediction, index) => prediction === targets[index]);
      return {
        predictions,
        correct,
        accuracy: correct.filter(Boolean).length / correct.length,
      };
    }

    function reset() {
      const defaults = controls.gate.value === "AND"
        ? { angle: 45, offset: -1.05 }
        : controls.gate.value === "OR"
          ? { angle: 45, offset: -0.35 }
          : { angle: 45, offset: -0.70 };
      controls.angle.value = defaults.angle;
      controls.offset.value = defaults.offset;
      render();
    }

    function findBest() {
      const targets = GATES[controls.gate.value];
      let best = { accuracy: -1, angle: 0, offset: 0 };
      for (let angle = 0; angle < 360; angle += 1) {
        const radians = angle * Math.PI / 180;
        const weights = [Math.cos(radians), Math.sin(radians)];
        for (let step = 0; step <= 180; step += 1) {
          const offset = -1.8 + step * 0.02;
          const result = evaluate(weights, offset, targets);
          if (result.accuracy > best.accuracy) {
            best = { accuracy: result.accuracy, angle, offset };
          }
          if (best.accuracy === 1) break;
        }
        if (best.accuracy === 1) break;
      }
      controls.angle.value = best.angle;
      controls.offset.value = best.offset.toFixed(2);
      render(true);
    }

    function render(fromSearch = false) {
      const { degrees, weights, bias } = parameters();
      const targets = GATES[controls.gate.value];
      const result = evaluate(weights, bias, targets);

      $("#angleValue").textContent = `${degrees}°`;
      $("#offsetValue").textContent = formatSigned(bias);
      $("#boundaryEquation").textContent =
        `${formatSigned(weights[0])}x₁ + ${formatSigned(weights[1])}x₂ + ${formatSigned(bias)} = 0`;
      $("#boundaryScore").innerHTML =
        `<strong>${(result.accuracy * 100).toFixed(0)}%</strong><span>정확도</span>`;
      $("#pointResults").innerHTML = FEATURES.map((sample, index) => {
        const state = result.correct[index] ? "is-correct" : "is-wrong";
        const label = result.correct[index] ? "정답" : "오답";
        return `<div class="${state}"><span>(${sample.join(", ")})</span>` +
          `<strong>${result.predictions[index]} / ${targets[index]}</strong><small>${label}</small></div>`;
      }).join("");

      const gate = controls.gate.value;
      let explanation;
      if (result.accuracy === 1) {
        explanation = `${gate}의 네 점을 직선 하나로 모두 분리했습니다. 이 문제는 선형 분리 가능합니다.`;
      } else if (gate === "XOR" && result.accuracy === 0.75) {
        explanation = fromSearch
          ? "모든 후보를 탐색해도 최고 정확도는 75%입니다. 네 점 중 적어도 하나는 반드시 오답으로 남습니다."
          : "현재 네 점 중 세 점을 맞혔습니다. 남은 한 점을 맞히도록 직선을 옮기면 다른 점이 오답이 됩니다.";
      } else {
        explanation = "각도는 가중치의 비율을, 위치는 편향을 바꿉니다. 오답 표시가 사라지는 방향으로 조절해 보세요.";
      }
      $("#boundaryExplanation").textContent = explanation;
      drawBoundary(context, canvas, weights, bias, targets, result);
    }

    [controls.angle, controls.offset].forEach((control) =>
      control.addEventListener("input", () => render(false))
    );
    controls.gate.addEventListener("change", reset);
    controls.reset.addEventListener("click", reset);
    controls.best.addEventListener("click", findBest);
    reset();
  }

  function drawBoundary(context, canvas, weights, bias, targets, result) {
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
    const predict = ([x1, x2]) => x1 * weights[0] + x2 * weights[1] + bias >= 0 ? 1 : 0;

    context.clearRect(0, 0, width, height);
    context.save();
    context.beginPath();
    context.rect(margin, margin, plotWidth, plotHeight);
    context.clip();
    for (let y = margin; y < height - margin; y += 8) {
      for (let x = margin; x < width - margin; x += 8) {
        context.fillStyle = predict([fromX(x), fromY(y)])
          ? "rgba(57,115,95,.09)"
          : "rgba(124,25,45,.065)";
        context.fillRect(x, y, 8, 8);
      }
    }
    context.restore();

    context.strokeStyle = "#dfdbd0";
    context.lineWidth = 1;
    [0, 0.5, 1].forEach((tick) => {
      context.beginPath();
      context.moveTo(toX(tick), margin);
      context.lineTo(toX(tick), height - margin);
      context.stroke();
      context.beginPath();
      context.moveTo(margin, toY(tick));
      context.lineTo(width - margin, toY(tick));
      context.stroke();
    });

    context.strokeStyle = "#7c192d";
    context.lineWidth = 4;
    context.beginPath();
    if (Math.abs(weights[1]) > 1e-6) {
      const leftY = -(weights[0] * xMin + bias) / weights[1];
      const rightY = -(weights[0] * xMax + bias) / weights[1];
      context.moveTo(toX(xMin), toY(leftY));
      context.lineTo(toX(xMax), toY(rightY));
    } else {
      const x = -bias / weights[0];
      context.moveTo(toX(x), margin);
      context.lineTo(toX(x), height - margin);
    }
    context.stroke();

    FEATURES.forEach((sample, index) => {
      const x = toX(sample[0]);
      const y = toY(sample[1]);
      context.beginPath();
      context.arc(x, y, 15, 0, Math.PI * 2);
      context.fillStyle = targets[index] ? "#39735f" : "#7c192d";
      context.fill();
      context.lineWidth = result.correct[index] ? 4 : 7;
      context.strokeStyle = result.correct[index] ? "#ffffff" : "#fbae40";
      context.stroke();
      context.fillStyle = "#2b2524";
      context.font = "bold 17px 'Times New Roman', serif";
      context.textAlign = "center";
      context.fillText(`(${sample.join(",")}) → ${targets[index]}`, x, y - 24);
    });

    context.fillStyle = "#6f625f";
    context.font = "14px 'Times New Roman', serif";
    context.textAlign = "center";
    [0, 0.5, 1].forEach((tick) =>
      context.fillText(tick.toFixed(1), toX(tick), height - margin + 25)
    );
    context.textAlign = "right";
    [0, 0.5, 1].forEach((tick) =>
      context.fillText(tick.toFixed(1), margin - 12, toY(tick) + 4)
    );
    context.fillStyle = "#2b2524";
    context.font = "bold 16px 'Times New Roman', serif";
    context.textAlign = "right";
    context.fillText("x₁", width - margin, height - 16);
    context.textAlign = "left";
    context.fillText("x₂", 18, margin);
  }

  function initializeCodeStudy() {
    const code = $("#boundaryCode");
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
            : "다시 생각해 보세요. 위의 수식과 경계 실험을 확인하면 답을 찾을 수 있습니다.";
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

  initializeBoundaryLab();
  initializeCodeStudy();
  initializeCopyButtons();
  initializeQuiz();
  initializeSectionNavigation();
})();
