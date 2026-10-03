(() => {
  "use strict";

  const cards = [...document.querySelectorAll(".timeline-card")];
  const year = document.getElementById("timelineYear");
  const title = document.getElementById("timelineTitle");
  const description = document.getElementById("timelineDescription");
  const source = document.getElementById("timelineSource");
  const chapter = document.getElementById("timelineChapter");
  if (!cards.length || !year || !title || !description || !source || !chapter) return;

  const detail = year.closest(".timeline-detail");
  let measuredWidth = 0;

  function reserveDetailSpace() {
    const width = detail.getBoundingClientRect().width;
    if (!width) return;
    measuredWidth = width;
    // Reserve the tallest description at the current width, without clipping
    // text or moving the centered hero copy when another item is selected.
    const measure = detail.cloneNode(true);
    measure.removeAttribute("aria-live");
    measure.setAttribute("aria-hidden", "true");
    measure.setAttribute("inert", "");
    measure.querySelectorAll("[id]").forEach((item) => item.removeAttribute("id"));
    Object.assign(measure.style, {
      position: "absolute", visibility: "hidden", pointerEvents: "none",
      top: "0", left: "0", width: `${width}px`, minHeight: "", height: "auto",
    });
    detail.parentElement.append(measure);
    const measureYear = measure.querySelector(".timeline-detail-kicker");
    const measureTitle = measure.querySelector("strong");
    const measureDescription = measure.querySelector("p:not(.timeline-detail-kicker)");
    const measureChapter = measure.querySelector(".timeline-detail-links a:last-child");
    let height = 0;
    for (const card of cards) {
      measureYear.textContent = `${card.dataset.year} · 역사적 전환점`;
      measureTitle.textContent = card.dataset.title;
      measureDescription.textContent = card.dataset.detail;
      measureChapter.hidden = !card.dataset.chapter;
      height = Math.max(height, measure.getBoundingClientRect().height);
    }
    measure.remove();
    detail.style.minHeight = `${Math.ceil(height)}px`;
  }

  reserveDetailSpace();
  document.fonts?.ready.then(reserveDetailSpace);
  if ("ResizeObserver" in window) {
    new ResizeObserver(() => {
      if (detail.getBoundingClientRect().width !== measuredWidth) reserveDetailSpace();
    }).observe(detail);
  } else {
    window.addEventListener("resize", reserveDetailSpace);
  }

  function select(card) {
    for (const item of cards) {
      const active = item === card;
      item.classList.toggle("is-active", active);
      item.closest(".timeline-step")?.classList.toggle("is-active", active);
      if (active) item.setAttribute("aria-current", "step");
      else item.removeAttribute("aria-current");
    }
    year.textContent = `${card.dataset.year} · 역사적 전환점`;
    title.textContent = card.dataset.title;
    description.textContent = card.dataset.detail;
    source.href = card.dataset.source;
    chapter.hidden = !card.dataset.chapter;
    if (card.dataset.chapter) chapter.href = card.dataset.chapter;
  }

  for (const card of cards) {
    card.addEventListener("pointerenter", (event) => {
      if (event.pointerType === "mouse" || event.pointerType === "pen") select(card);
    });
    card.addEventListener("focus", () => select(card));
    card.addEventListener("click", () => select(card));
  }
})();
