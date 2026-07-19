/* pbb-style tag filtering: toggle buttons show/hide .pb-pub cards.
   Progressive enhancement — all cards are visible without JS. */
(function () {
  function init() {
    var bar = document.querySelector(".pb-filters");
    if (!bar) return;
    var buttons = bar.querySelectorAll("button[data-tag]");
    var cards = document.querySelectorAll(".pb-pub[data-tags]");
    if (!buttons.length || !cards.length) return;
    buttons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        btn.classList.toggle("active");
        var active = Array.prototype.map.call(
          bar.querySelectorAll("button.active"),
          function (b) { return b.getAttribute("data-tag"); }
        );
        cards.forEach(function (card) {
          var tags = card.getAttribute("data-tags").split(",");
          var show =
            active.length === 0 ||
            active.some(function (t) { return tags.indexOf(t) !== -1; });
          card.classList.toggle("pb-hidden", !show);
        });
      });
    });
  }
  /* document$ re-emits on every instant-navigation page swap;
     plain DOMContentLoaded only fires on the first full load */
  if (typeof document$ !== "undefined") {
    document$.subscribe(init);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
