/* pbb-style tag filtering + theme toggle.
   Progressive enhancement — all cards are visible without JS. */
(function () {
  function initTheme() {
    var btn = document.querySelector(".pb-theme-btn");
    if (!btn) return;
    btn.hidden = false;
    btn.addEventListener("click", function () {
      /* Material's palette radios live in the hidden header; a
         programmatic click still toggles the scheme */
      var input = document.querySelector('input[name="__palette"]:not(:checked)');
      if (input) input.click();
    });
  }

  function initNavMorph() {
    /* pbb.sh-style morph: name the clicked pill pb-title (and unname the
       current title) right before navigation, so the browser's view
       transition slides the pill into the next page's heading */
    document.querySelectorAll(".pb-nav-link").forEach(function (a) {
      a.addEventListener("click", function () {
        var title = document.querySelector(".pb-page-title");
        if (title) title.style.viewTransitionName = "none";
        a.style.viewTransitionName = "pb-title";
      });
    });
  }

  function init() {
    initTheme();
    initNavMorph();
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
