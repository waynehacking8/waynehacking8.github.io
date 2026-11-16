/* Theme toggle for the home page. The theme is set before paint by an inline
   script in home.html. The choice is saved in Material's own palette key, so
   the blog and Patches pages open in the same scheme. Storage can be blocked,
   so every access is wrapped and the page works without it. */
(function () {
  var root = document.documentElement;
  var btn = document.querySelector(".theme-btn");
  if (!btn) return;
  function show(t) {
    btn.setAttribute("aria-pressed", t === "dark" ? "true" : "false");
  }
  show(root.dataset.theme);
  btn.addEventListener("click", function () {
    var next = root.dataset.theme === "light" ? "dark" : "light";
    root.dataset.theme = next;
    show(next);
    // Same shape Material writes: index into mkdocs.yml theme.palette (0 light, 1 dark).
    var color = next === "dark"
      ? { media: "(prefers-color-scheme: dark)", scheme: "slate", primary: "custom", accent: "custom" }
      : { media: "(prefers-color-scheme: light)", scheme: "default", primary: "custom", accent: "custom" };
    try { localStorage.setItem("/.__palette", JSON.stringify({ index: next === "dark" ? 1 : 0, color: color })); } catch (e) {}
  });
})();
