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

  /* DOMParser documents do not inherit the destination page's base URL.
     Resolve media before insertion so a relative `assets/...` URL cannot be
     requested under the page being left during a soft navigation. */
  function normalizeMediaUrls(root, url) {
    root.querySelectorAll("[src]").forEach(function (node) {
      var value = node.getAttribute("src");
      if (!value || /^(?:data:|blob:|javascript:)/i.test(value)) return;
      node.setAttribute("src", new URL(value, url).href);
    });
    root.querySelectorAll("video[poster]").forEach(function (node) {
      var value = node.getAttribute("poster");
      if (!value || /^(?:data:|blob:|javascript:)/i.test(value)) return;
      node.setAttribute("poster", new URL(value, url).href);
    });
  }

  function renderMath(root) {
    if (!window.MathJax || typeof window.MathJax.typesetPromise !== "function") return;
    window.MathJax.typesetPromise(root ? [root] : undefined).catch(function (error) {
      console.warn("Math rendering failed", error);
    });
  }

  function updateDocument(doc, url) {
    var newMain = doc.querySelector(".pb-main");
    var curMain = document.querySelector(".pb-main");
    if (!newMain || !curMain) return false;
    disconnectArticleToc();
    normalizeMediaUrls(newMain, url);
    if (window.MathJax && typeof window.MathJax.typesetClear === "function") {
      window.MathJax.typesetClear([curMain]);
    }
    curMain.replaceWith(newMain);
    syncSidebarToc(true);
    document.title = doc.title;
    var canonical = document.querySelector('link[rel="canonical"]');
    if (canonical) canonical.href = url.href;
    var description = doc.querySelector('meta[name="description"]');
    var currentDescription = document.querySelector('meta[name="description"]');
    if (description && currentDescription) currentDescription.content = description.content;
    document.body.classList.remove("pb-strip-hidden");
    var hashId = decodeHash(url.hash);
    var target = hashId && document.getElementById(hashId);
    if (target) target.scrollIntoView({ block: "start" });
    else window.scrollTo(0, 0);
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
      var a = e.target.closest && e.target.closest("a[href]");
      if (a && !a.matches(".pb-nav-link") && !a.closest(".pb-main")) a = null;
      if (!a || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0 ||
          a.target || a.hasAttribute("download")) return;
      var url = new URL(a.getAttribute("href"), location.href);
      if (url.origin !== location.origin) return;
      if (url.pathname === location.pathname && url.search === location.search) {
        if (url.hash) return;
        e.preventDefault();
        return;
      }
      e.preventDefault();
      navigate(url, a.matches(".pb-nav-link") ? a : null, true);
    });
    ["pointerover", "focusin"].forEach(function (eventName) {
      document.addEventListener(eventName, function (e) {
        var a = e.target.closest && e.target.closest("a[href]");
        if (a && !a.matches(".pb-nav-link") && !a.closest(".pb-main")) a = null;
        if (!a) return;
        var url = new URL(a.href, location.href);
        if (url.origin === location.origin && url.pathname !== location.pathname &&
            !pageCache.has(url.href)) {
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
  var articleTocObserver = null;
  var articleTocMedia = window.matchMedia("(min-width: 1024px)");
  var articleTocMediaArmed = false;
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

  function decodeHash(hash) {
    if (!hash) return "";
    try { return decodeURIComponent(hash.slice(1)); }
    catch (error) { return hash.slice(1); }
  }

  function disconnectArticleToc() {
    if (articleTocObserver) {
      articleTocObserver.disconnect();
      articleTocObserver = null;
    }
    document.querySelectorAll('.pb-article-toc a').forEach(function (link) {
      link.classList.remove("pb-toc-active");
      link.removeAttribute("aria-current");
    });
  }

  function syncSidebarToc(clearWhenMissing) {
    var slot = document.querySelector(".pb-sidebar-toc-slot");
    if (!slot) return;
    var template = document.querySelector(".pb-main > .pb-article-toc-template");
    if (template && (clearWhenMissing || !slot.firstElementChild)) {
      slot.replaceChildren(template.content.firstElementChild.cloneNode(true));
    }
    else if (clearWhenMissing) slot.replaceChildren();
  }

  function initArticleToc() {
    disconnectArticleToc();
    syncSidebarToc(false);
    var toc = articleTocMedia.matches
      ? document.querySelector(".pb-sidebar-toc-slot .pb-article-toc--desktop")
      : document.querySelector(".pb-main .pb-article-toc--mobile");
    if (!toc) return;
    var links = toc.querySelectorAll('a[href^="#"]');
    if (!links.length) return;
    var byId = new Map();
    links.forEach(function (link) {
      var id = decodeHash(link.hash);
      if (!byId.has(id)) byId.set(id, []);
      byId.get(id).push(link);
    });
    var headings = Array.from(byId.keys()).map(function (id) {
      return document.getElementById(id);
    }).filter(Boolean);
    var setActive = function (id) {
      links.forEach(function (link) {
        var active = decodeHash(link.hash) === id;
        link.classList.toggle('pb-toc-active', active);
        if (active) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    };
    if (headings[0]) setActive(headings[0].id);
    if (!headings.length || typeof IntersectionObserver === "undefined") return;
    articleTocObserver = new IntersectionObserver(function () {
      var current = headings[0];
      headings.forEach(function (heading) {
        if (heading.getBoundingClientRect().top <= 150) current = heading;
      });
      if (current) setActive(current.id);
    }, { rootMargin: '-120px 0px -65% 0px' });
    headings.forEach(function (heading) { articleTocObserver.observe(heading); });
  }

  function initArticleTocMedia() {
    if (articleTocMediaArmed) return;
    articleTocMediaArmed = true;
    articleTocMedia.addEventListener("change", initArticleToc);
  }

  function init() {
    initTheme();
    initRouter();
    initStrip();
    initArticleTocMedia();
    initArticleToc();
    renderMath(document.querySelector(".pb-main"));
    document.querySelectorAll("nav.md-code__nav").forEach(function (nav, index) {
      nav.setAttribute("aria-label", "Code block " + (index + 1) + " actions");
    });
    document.querySelectorAll(".pb-post-image img").forEach(function (img) {
      if (img.dataset.pbFallbackArmed) return;
      img.dataset.pbFallbackArmed = "true";
      var markFailed = function () { img.parentElement.classList.add("pb-image-failed"); };
      img.addEventListener("error", markFailed, { once: true });
      if (img.complete && !img.naturalWidth) markFailed();
    });
    var bar = document.querySelector(".pb-filters");
    if (!bar) return;
    var buttons = bar.querySelectorAll("button[data-tag]");
    var cards = document.querySelectorAll(".pb-pub[data-tags], .pb-post-card[data-tags]");
    if (!buttons.length || !cards.length) return;
    var requested = new URL(location.href).searchParams.getAll("tag");
    buttons.forEach(function (btn) {
      if (requested.indexOf(btn.getAttribute("data-tag")) !== -1) {
        btn.classList.add("active");
      }
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
            active.every(function (t) { return tags.indexOf(t) !== -1; });
          card.classList.toggle("pb-hidden", !show);
        });
        var url = new URL(location.href);
        url.searchParams.delete("tag");
        active.forEach(function (tag) { url.searchParams.append("tag", tag); });
        history.replaceState(history.state, "", url);
      });
    });
    if (requested.length) {
      cards.forEach(function (card) {
        var tags = card.getAttribute("data-tags").split(",");
        card.classList.toggle("pb-hidden", !requested.every(function (tag) {
          return tags.indexOf(tag) !== -1;
        }));
      });
    }
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
