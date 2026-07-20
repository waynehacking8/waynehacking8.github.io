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

  /* pbb.sh-style soft navigation (what Astro's ClientRouter does): fetch
     the next page, swap only .pb-main inside document.startViewTransition,
     and name the clicked pill pb-title so it morphs into the incoming
     heading. The sidebar's DOM is never touched, so it cannot move. */
  var routerArmed = false;
  function initRouter() {
    if (routerArmed) return;
    routerArmed = true;
    document.addEventListener("click", function (e) {
      var a = e.target.closest && e.target.closest("a.pb-nav-link");
      if (!a || e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
      var url = new URL(a.getAttribute("href"), location.href);
      if (url.origin !== location.origin) return;
      if (url.pathname === location.pathname) { e.preventDefault(); return; }
      if (!document.startViewTransition ||
          matchMedia("(prefers-reduced-motion: reduce)").matches) return;
      e.preventDefault();
      fetch(url.href).then(function (r) { return r.text(); }).then(function (html) {
        var doc = new DOMParser().parseFromString(html, "text/html");
        var newMain = doc.querySelector(".pb-main");
        var curMain = document.querySelector(".pb-main");
        if (!newMain || !curMain) { location.href = url.href; return; }
        var title = document.querySelector(".pb-page-title");
        if (title) title.style.viewTransitionName = "none";
        a.style.viewTransitionName = "pb-title";
        var vt = document.startViewTransition(function () {
          curMain.replaceWith(newMain);
          document.title = doc.title;
          history.pushState({}, "", url.href);
          window.scrollTo(0, 0);
        });
        vt.finished.then(init);
      }).catch(function () { location.href = url.href; });
    });
    window.addEventListener("popstate", function () { location.reload(); });
  }

  /* pbb's phone profile strip retracts on scroll-down and returns on
     scroll-up. Bound once to window, so it survives .pb-main swaps. */
  var stripArmed = false;
  function initStrip() {
    if (stripArmed) return;
    stripArmed = true;
    var last = window.scrollY;
    window.addEventListener("scroll", function () {
      var y = window.scrollY;
      /* ignore sub-pixel jitter and the rubber-band region at the top */
      if (Math.abs(y - last) < 6) return;
      var hide = y > last && y > 140;
      document.body.classList.toggle("pb-strip-hidden", hide);
      last = y;
    }, { passive: true });
  }

  function init() {
    initTheme();
    initRouter();
    initStrip();
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
