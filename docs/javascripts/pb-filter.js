/* pbb-style tag filtering + theme toggle.
   Progressive enhancement — all cards are visible without JS. */
(function () {
  var themeArmed = false;
  function updateThemeLabel(btn) {
    var isDark = document.body.getAttribute("data-md-color-scheme") === "slate";
    var label = isDark ? "Switch to light mode" : "Switch to dark mode";
    btn.setAttribute("aria-label", label);
    btn.title = label;
  }
  function initTheme() {
    var btn = document.querySelector(".pb-theme-btn");
    if (!btn) return;
    btn.hidden = false;
    updateThemeLabel(btn);
    /* document$ re-emits on every page swap; without this the handler
       stacks up and an even number of them cancels the toggle out */
    if (themeArmed) return;
    themeArmed = true;
    btn.addEventListener("click", function () {
      /* Material's palette radios live in the hidden header; a programmatic
         click still toggles the scheme. Pick the target by the CURRENT
         scheme rather than by :not(:checked) — with media-query palettes
         neither radio is checked on first paint, so the old selector
         matched index 0 and the very first click re-selected the palette
         that was already active, doing nothing visible. */
      var inputs = document.querySelectorAll('input[name="__palette"]');
      if (inputs.length < 2) return;
      var isDark = document.body.getAttribute("data-md-color-scheme") === "slate";
      inputs[isDark ? 0 : 1].click();
      requestAnimationFrame(function () { updateThemeLabel(btn); });
    });
  }

  /* pbb.sh-style soft navigation (what Astro's ClientRouter does): fetch
     the next page, swap only .pb-main inside document.startViewTransition,
     and name the clicked pill pb-title so it morphs into the incoming
     heading. The sidebar's DOM is never touched, so it cannot move. */
  var routerArmed = false;
  var navigationController = null;
  var pageCache = new Map();

  function getPage(url, signal) {
    var key = url.href;
    if (pageCache.has(key)) return Promise.resolve(pageCache.get(key));
    return fetch(key, { signal: signal, headers: { "X-Requested-With": "fetch" } })
      .then(function (response) {
        if (!response.ok) throw new Error("Navigation failed: " + response.status);
        return response.text();
      })
      .then(function (html) {
        pageCache.set(key, html);
        return html;
      });
  }

  function updateDocument(doc, url) {
    var newMain = doc.querySelector(".pb-main");
    var curMain = document.querySelector(".pb-main");
    if (!newMain || !curMain) return false;
    curMain.replaceWith(newMain);
    document.title = doc.title;
    var canonical = document.querySelector('link[rel="canonical"]');
    if (canonical) canonical.href = url.href;
    var description = doc.querySelector('meta[name="description"]');
    var currentDescription = document.querySelector('meta[name="description"]');
    if (description && currentDescription) currentDescription.content = description.content;
    document.body.classList.remove("pb-strip-hidden");
    window.scrollTo(0, 0);
    return true;
  }

  function navigate(url, trigger, pushState) {
    if (navigationController) navigationController.abort();
    var controller = new AbortController();
    navigationController = controller;
    document.body.classList.add("pb-is-navigating");
    return getPage(url, controller.signal).then(function (html) {
      var doc = new DOMParser().parseFromString(html, "text/html");
      var swap = function () {
        if (!updateDocument(doc, url)) { location.href = url.href; return; }
        if (pushState) history.pushState({}, "", url.href);
      };
      var transition;
      if (document.startViewTransition &&
          !matchMedia("(prefers-reduced-motion: reduce)").matches) {
        var title = document.querySelector(".pb-page-title");
        if (title) title.style.viewTransitionName = "none";
        if (trigger) trigger.style.viewTransitionName = "pb-title";
        transition = document.startViewTransition(swap).finished;
      } else {
        swap();
        transition = Promise.resolve();
      }
      return transition.then(function () {
        init();
        var heading = document.querySelector(".pb-page-title");
        if (heading) {
          heading.setAttribute("tabindex", "-1");
          heading.focus({ preventScroll: true });
        }
      });
    }).catch(function (error) {
      if (error.name !== "AbortError") location.href = url.href;
    }).finally(function () {
      if (navigationController === controller) {
        navigationController = null;
        document.body.classList.remove("pb-is-navigating");
      }
    });
  }

  function initRouter() {
    if (routerArmed) return;
    routerArmed = true;
    document.addEventListener("click", function (e) {
      var a = e.target.closest && e.target.closest("a.pb-nav-link");
      if (!a || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0 ||
          a.target || a.hasAttribute("download")) return;
      var url = new URL(a.getAttribute("href"), location.href);
      if (url.origin !== location.origin) return;
      if (url.pathname === location.pathname) { e.preventDefault(); return; }
      e.preventDefault();
      navigate(url, a, true);
    });
    ["pointerover", "focusin"].forEach(function (eventName) {
      document.addEventListener(eventName, function (e) {
        var a = e.target.closest && e.target.closest("a.pb-nav-link");
        if (!a) return;
        var url = new URL(a.href, location.href);
        if (url.origin === location.origin && !pageCache.has(url.href)) {
          getPage(url).catch(function () {});
        }
      }, { passive: true });
    });
    window.addEventListener("popstate", function () {
      navigate(new URL(location.href), null, false);
    });
  }

  /* pbb's phone profile strip retracts on scroll-down and returns on
     scroll-up. Bound once to window, so it survives .pb-main swaps. */
  var stripArmed = false;
  function initStrip() {
    if (stripArmed) return;
    stripArmed = true;
    var last = window.scrollY;
    var ticking = false;
    window.addEventListener("scroll", function () {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(function () {
        var y = window.scrollY;
        /* ignore sub-pixel jitter and the rubber-band region at the top */
        if (Math.abs(y - last) >= 6) {
          document.body.classList.toggle("pb-strip-hidden", y > last && y > 140);
          last = y;
        }
        ticking = false;
      });
    }, { passive: true });
  }

  function init() {
    initTheme();
    initRouter();
    initStrip();
    document.querySelectorAll("nav.md-code__nav").forEach(function (nav, index) {
      nav.setAttribute("aria-label", "Code block " + (index + 1) + " actions");
    });
    var bar = document.querySelector(".pb-filters");
    if (!bar) return;
    var buttons = bar.querySelectorAll("button[data-tag]");
    var cards = document.querySelectorAll(".pb-pub[data-tags]");
    if (!buttons.length || !cards.length) return;
    buttons.forEach(function (btn) {
      btn.setAttribute("aria-pressed", btn.classList.contains("active"));
      btn.addEventListener("click", function () {
        btn.classList.toggle("active");
        btn.setAttribute("aria-pressed", btn.classList.contains("active"));
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
