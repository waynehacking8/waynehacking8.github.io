/* Theme toggle for the home page. The saved choice is applied before paint by
   an inline script in home.html; storage can be blocked, so every access is
   wrapped and the page works without it. */
(function () {
  var root = document.documentElement;
  var btn = document.querySelector(".theme-btn");
  if (!btn) return;
  btn.addEventListener("click", function () {
    var next = root.dataset.theme === "light" ? "dark" : "light";
    root.dataset.theme = next;
    try { localStorage.setItem("wayne-theme", next); } catch (e) {}
  });
})();
