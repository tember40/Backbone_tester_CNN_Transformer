(() => {
  "use strict";
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const image = $("#vitImage");
  const preview = $("#vitPatchPreview");
  const imageContext = image?.getContext("2d");
  const previewContext = preview?.getContext("2d");
  let patchSize = 4;
  let selectedX = 14;
  let selectedY = 14;

  // A deliberately hand-drawn 32×32 scene, not a CIFAR-10 sample.
  function pixelAt(x, y) {
    if (x >= 23 && x <= 29 && y >= 3 && y <= 9 && (x - 26) ** 2 + (y - 6) ** 2 <= 10) return "#f8cb70";
    if (y >= 5 && y <= 7 && ((x >= 3 && x <= 9) || (x >= 11 && x <= 14))) return "#f4f0e5";
    if (y >= 22 && y <= 31) {
      if (x >= 10 && x <= 22 && y <= 27) {
        if (x >= 15 && x <= 18 && y >= 23) return "#72504e";
        if (x >= 19 && x <= 21 && y >= 20 && y <= 22) return "#6c9bb0";
        return "#e7c39c";
      }
      return (x + y) % 5 === 0 ? "#789a72" : "#83a67c";
    }
    if (y >= 13 && y <= 21 && x >= 10 && x <= 22) {
      const roofEdge = 16 - Math.abs(x - 16);
      if (y >= Math.max(13, roofEdge) && y <= 18) return "#a8525a";
      if (y >= 18) return "#e7c39c";
    }
    return y < 15 ? "#b8d9e6" : "#bad6c4";
  }

  function drawImage() {
    if (!imageContext) return;
    imageContext.clearRect(0, 0, 320, 320);
    for (let y = 0; y < 32; y += 1) {
      for (let x = 0; x < 32; x += 1) {
        imageContext.fillStyle = pixelAt(x, y);
        imageContext.fillRect(x * 10, y * 10, 10, 10);
      }
    }
    imageContext.strokeStyle = "rgba(84,16,31,.65)";
    imageContext.lineWidth = patchSize === 2 ? 1 : 1.4;
    for (let offset = patchSize * 10; offset < 320; offset += patchSize * 10) {
      imageContext.beginPath(); imageContext.moveTo(offset + .5, 0); imageContext.lineTo(offset + .5, 320); imageContext.stroke();
      imageContext.beginPath(); imageContext.moveTo(0, offset + .5); imageContext.lineTo(320, offset + .5); imageContext.stroke();
    }
    const col = Math.floor(selectedX / patchSize);
    const row = Math.floor(selectedY / patchSize);
    imageContext.fillStyle = "rgba(124,25,45,.14)";
    imageContext.fillRect(col * patchSize * 10, row * patchSize * 10, patchSize * 10, patchSize * 10);
    imageContext.strokeStyle = "#54101f";
    imageContext.lineWidth = 3;
    imageContext.strokeRect(col * patchSize * 10 + 1.5, row * patchSize * 10 + 1.5, patchSize * 10 - 3, patchSize * 10 - 3);
  }

  function drawPreview() {
    if (!previewContext) return;
    const col = Math.floor(selectedX / patchSize);
    const row = Math.floor(selectedY / patchSize);
    const pixelWidth = 128 / patchSize;
    previewContext.clearRect(0, 0, 128, 128);
    for (let y = 0; y < patchSize; y += 1) {
      for (let x = 0; x < patchSize; x += 1) {
        previewContext.fillStyle = pixelAt(col * patchSize + x, row * patchSize + y);
        previewContext.fillRect(x * pixelWidth, y * pixelWidth, pixelWidth, pixelWidth);
      }
    }
    if (patchSize <= 8) {
      previewContext.strokeStyle = "rgba(43,37,36,.16)";
      previewContext.lineWidth = 1;
      for (let offset = pixelWidth; offset < 128; offset += pixelWidth) {
        previewContext.beginPath(); previewContext.moveTo(offset + .5, 0); previewContext.lineTo(offset + .5, 128); previewContext.stroke();
        previewContext.beginPath(); previewContext.moveTo(0, offset + .5); previewContext.lineTo(128, offset + .5); previewContext.stroke();
      }
    }
  }

  function renderPatch() {
    const n = (32 / patchSize) ** 2;
    const col = Math.floor(selectedX / patchSize);
    const row = Math.floor(selectedY / patchSize);
    $("#vitPatchCount").textContent = n.toLocaleString("ko-KR");
    $("#vitPatchVector").textContent = (3 * patchSize ** 2).toLocaleString("ko-KR");
    $("#vitTokenCount").textContent = (n + 1).toLocaleString("ko-KR");
    $("#vitAttentionCells").textContent = ((n + 1) ** 2).toLocaleString("ko-KR");
    $("#vitPatchCoordinate").textContent = `${row + 1}행 ${col + 1}열`;
    $("#vitPatchRange").textContent = `x=${col * patchSize}–${(col + 1) * patchSize - 1}, y=${row * patchSize}–${(row + 1) * patchSize - 1}`;
    $("#vitPatchFlatten").textContent = `RGB 픽셀을 펼치면 ${3 * patchSize ** 2}개 숫자 · 선형층에서 192차원으로 투영`;
    $$('[data-patch-size]').forEach((button) => {
      const current = Number(button.dataset.patchSize) === patchSize;
      button.classList.toggle("is-current", current);
      button.setAttribute("aria-pressed", String(current));
    });
    drawImage();
    drawPreview();
  }

  $$('[data-patch-size]').forEach((button) => button.addEventListener("click", () => {
    patchSize = Number(button.dataset.patchSize);
    renderPatch();
  }));
  image?.addEventListener("click", (event) => {
    const bounds = image.getBoundingClientRect();
    selectedX = Math.max(0, Math.min(31, Math.floor((event.clientX - bounds.left) / bounds.width * 32)));
    selectedY = Math.max(0, Math.min(31, Math.floor((event.clientY - bounds.top) / bounds.height * 32)));
    renderPatch();
  });
  image?.addEventListener("keydown", (event) => {
    const movement = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]}[event.key];
    if (!movement) return;
    event.preventDefault();
    selectedX = Math.max(0, Math.min(31, selectedX + movement[0] * patchSize));
    selectedY = Math.max(0, Math.min(31, selectedY + movement[1] * patchSize));
    renderPatch();
  });
  renderPatch();

  $$('[data-pool]').forEach((button) => button.addEventListener("click", () => {
    const pool = button.dataset.pool;
    $$('[data-pool]').forEach((item) => {
      const current = item === button;
      item.classList.toggle("is-current", current);
      item.setAttribute("aria-pressed", String(current));
    });
    $$('.vit-token-strip [data-token]').forEach((token) => token.classList.toggle("is-chosen", pool === "mean" || token.dataset.token === "cls"));
    $("#vitPoolExplanation").innerHTML = pool === "cls"
      ? "[CLS]의 마지막 표현 <code>x[:, 0]</code>만 꺼내 LayerNorm과 선형 분류층에 전달합니다."
      : "<code>x.mean(dim = 1)</code>은 [CLS]를 포함한 65개 토큰 전체를 평균합니다. 출력 표현 길이는 여전히 192입니다.";
  }));

  $$('.code-notes button').forEach((button) => button.addEventListener("click", () => {
    const [start, end] = button.dataset.highlight.split("-").map(Number);
    $$('.line-code [data-line]').forEach((line) => {
      const number = Number(line.dataset.line);
      line.classList.toggle("is-highlighted", number >= start && number <= end);
    });
    $$('.code-notes button').forEach((item) => item.classList.toggle("is-active", item === button));
  }));
  $$('.copy-button').forEach((button) => button.addEventListener("click", async () => {
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
    $$('.quiz').forEach((quiz) => {
      const choice = $("input:checked", quiz);
      const correct = choice?.value === quiz.dataset.answer;
      quiz.classList.toggle("is-correct", correct);
      quiz.classList.toggle("is-wrong", !correct);
      $(".quiz-feedback", quiz).textContent = !choice ? "답을 선택하세요." : correct ? "맞았습니다." : "다시 확인해 보세요.";
      if (correct) score += 1;
    });
    $("#quizScore").textContent = `3문제 중 ${score}문제를 맞혔습니다.`;
  });

  const links = $$('.chapter-sidebar a');
  const sections = links.map((link) => $(link.getAttribute("href"))).filter(Boolean);
  const observer = new IntersectionObserver((entries) => {
    const current = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!current) return;
    links.forEach((link) => link.classList.toggle("is-current", link.getAttribute("href") === `#${current.target.id}`));
  }, {rootMargin: "-18% 0px -68%", threshold: [0, .2, .55]});
  sections.forEach((section) => observer.observe(section));
})();
