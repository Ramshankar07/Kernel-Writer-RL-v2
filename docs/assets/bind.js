/*
 * bind.js: fills numbers on the page from the JSON result files.
 *
 * Markup:  <span data-k="FILE.json:dotted.path" data-fmt="FMT">fallback</span>
 *   FILE   a file in assets/data/ (numbers.json, extra_numbers.json, phase3.json, mix_report.json)
 *   path   dot-separated keys; array indices are plain numbers (a.b.0.c). Keys that themselves
 *          contain dots (e.g. "speed_reward_std_gt_0.02") resolve too: the longest matching key wins.
 *   FMT    pct0 | pct1 | pct2   fraction -> percent ("0.0806" -> "8.1%")
 *          f0 | f1 | f2 | f3    fixed decimals
 *          int                  rounded integer with thousands separators ("3022" -> "3,022")
 *          x2                   2 decimals + "×"
 *          raw                  value as-is (default)
 *
 * Each file is fetched once. If a fetch fails or a path is missing, the fallback text stays
 * (and the element gets data-bound="miss"); bound elements get data-bound="ok".
 * Also marks the current page's nav link with aria-current="page".
 * Usage: <script src="assets/bind.js" defer></script>
 */
(function () {
  "use strict";

  var script = document.currentScript;
  var dataBase = new URL("data/", script ? script.src : new URL("assets/", location.href));

  function resolve(obj, parts) {
    if (!parts.length) return obj;
    if (obj === null || typeof obj !== "object") return undefined;
    // Try the longest key first so keys containing "." still resolve.
    for (var n = parts.length; n >= 1; n--) {
      var key = parts.slice(0, n).join(".");
      if (Object.prototype.hasOwnProperty.call(obj, key)) {
        var v = resolve(obj[key], parts.slice(n));
        if (v !== undefined) return v;
      }
    }
    return undefined;
  }

  function format(v, fmt) {
    if (v === null || v === undefined) return null;
    if (!fmt || fmt === "raw") return typeof v === "object" ? JSON.stringify(v) : String(v);
    var x = typeof v === "number" ? v : Number(v);
    if (!isFinite(x)) return null;
    var m;
    if ((m = /^pct(\d)$/.exec(fmt))) return (x * 100).toFixed(+m[1]) + "%";
    if ((m = /^f(\d)$/.exec(fmt))) return x.toFixed(+m[1]);
    if (fmt === "int") return Math.round(x).toLocaleString("en-US");
    if (fmt === "x2") return x.toFixed(2) + "×";
    return null; // unknown format: keep fallback
  }

  var cache = {};
  function load(file) {
    if (!cache[file]) {
      cache[file] = fetch(new URL(file, dataBase).href)
        .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
        .catch(function () { return null; });
    }
    return cache[file];
  }

  function bind() {
    var els = document.querySelectorAll("[data-k]");
    Array.prototype.forEach.call(els, function (el) {
      var spec = el.getAttribute("data-k");
      var i = spec.indexOf(":");
      if (i < 1) { el.setAttribute("data-bound", "miss"); return; }
      var file = spec.slice(0, i), path = spec.slice(i + 1);
      load(file).then(function (data) {
        var out = data ? format(resolve(data, path ? path.split(".") : []), el.getAttribute("data-fmt")) : null;
        if (out === null) { el.setAttribute("data-bound", "miss"); return; }
        el.textContent = out;
        el.setAttribute("data-bound", "ok");
      });
    });
  }

  function markNav() {
    var here = location.pathname.split("/").pop() || "index.html";
    var links = document.querySelectorAll(".nav a[href]");
    Array.prototype.forEach.call(links, function (a) {
      if (a.getAttribute("href").split("#")[0] === here) a.setAttribute("aria-current", "page");
    });
  }

  function init() { markNav(); bind(); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
