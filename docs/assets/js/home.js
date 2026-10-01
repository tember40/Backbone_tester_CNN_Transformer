(() => {
  "use strict";

  const cards = [...document.querySelectorAll(".timeline-card")];
  const year = document.getElementById("timelineYear");
  const title = document.getElementById("timelineTitle");
  const description = document.getElementById("timelineDescription");
  const source = document.getElementById("timelineSource");
  const chapter = document.getElementById("timelineChapter");
  if (!cards.length || !year || !title || !description || !source || !chapter) return;

  function select(card) {
    for (const item of cards) {
      const active = item === card;
      item.classList.toggle("is-active", active);
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
