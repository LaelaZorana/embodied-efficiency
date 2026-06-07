// embodied-efficiency console: theme toggle + the live deploy-compiler.
// The Pareto frontier is static real-L4 data, so the compiler runs entirely in
// the browser: move a slider, re-pick and redraw with no round-trip.
(function () {
  "use strict";

  // ---- Theme (light/dark), persisted, system fallback, no flash ----------
  var root = document.documentElement;
  var THEME_KEY = "embodied-efficiency-theme";
  function applyTheme(t) { t === "dark" ? root.classList.add("dark") : root.classList.remove("dark"); }
  (function initTheme() {
    var saved = localStorage.getItem(THEME_KEY);
    if (saved) applyTheme(saved);
    else applyTheme(window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  })();
  window.toggleTheme = function () {
    var dark = root.classList.toggle("dark");
    localStorage.setItem(THEME_KEY, dark ? "dark" : "light");
    if (window.__eeDraw) window.__eeDraw(); // redraw plot so axis colours follow the theme
  };

  // ---- Deploy-compiler ---------------------------------------------------
  function initCompiler() {
    var dataEl = document.getElementById("configs-data");
    var plotEl = document.getElementById("pareto");
    var resultEl = document.getElementById("pick-result");
    if (!dataEl || !plotEl || !resultEl) return;

    var CONFIGS = JSON.parse(dataEl.textContent);
    var els = {
      lat: document.getElementById("max_lat"),
      mb: document.getElementById("max_mb"),
      rmse: document.getElementById("max_rmse"),
      stale: document.getElementById("max_stale"),
    };
    var vals = {
      lat: document.getElementById("v_lat"),
      mb: document.getElementById("v_mb"),
      rmse: document.getElementById("v_rmse"),
      stale: document.getElementById("v_stale"),
    };
    var objBtns = Array.prototype.slice.call(document.querySelectorAll(".ee-obj"));
    var state = { objective: "latency" };

    // latency slider is log-scaled: it spans the real data range tightly.
    var MS_LO = Math.log10(0.015), MS_HI = Math.log10(13);
    function latFromSlider() { return Math.pow(10, MS_LO + (els.lat.value / 1000) * (MS_HI - MS_LO)); }
    function fmtMs(ms) { return ms >= 1 ? ms.toFixed(2) : ms.toFixed(3); }

    function setObjective(obj) {
      state.objective = obj;
      objBtns.forEach(function (b) {
        var on = b.getAttribute("data-obj") === obj;
        b.classList.toggle("border-signal-500", on);
        b.classList.toggle("bg-signal-50", on);
        b.classList.toggle("text-signal-700", on);
        b.classList.toggle("dark:bg-signal-500/10", on);
        b.classList.toggle("dark:text-signal-300", on);
        b.classList.toggle("border-slate-200", !on);
        b.classList.toggle("text-slate-600", !on);
        b.classList.toggle("dark:border-white/10", !on);
        b.classList.toggle("dark:text-slate-300", !on);
      });
    }

    function budget() {
      return {
        objective: state.objective,
        max_lat: latFromSlider(),
        max_mb: parseFloat(els.mb.value),
        max_rmse: parseFloat(els.rmse.value),
        max_stale: parseInt(els.stale.value, 10),
      };
    }

    function pick(b) {
      var feas = CONFIGS.filter(function (c) {
        return c.ms_per_action <= b.max_lat && c.weight_mb <= b.max_mb &&
               c.rmse <= b.max_rmse && c.staleness <= b.max_stale;
      });
      if (!feas.length) return { chosen: null, feas: feas };
      var key = b.objective === "footprint" ? "weight_mb" : "ms_per_action";
      var chosen = feas.reduce(function (a, c) { return c[key] < a[key] ? c : a; });
      return { chosen: chosen, feas: feas };
    }

    // rMSE -> colour: emerald (low error) through amber to rose (high error).
    function lerp(a, b, t) { return a.map(function (x, i) { return Math.round(x + (b[i] - x) * t); }); }
    function rmseColor(r) {
      var EM = [16, 185, 129], AM = [217, 160, 60], RO = [244, 100, 110];
      var t = Math.max(0, Math.min(1, r / 0.26));
      var c = t < 0.5 ? lerp(EM, AM, t / 0.5) : lerp(AM, RO, (t - 0.5) / 0.5);
      return "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
    }

    function renderResult(res, b) {
      if (!res.chosen) {
        resultEl.innerHTML =
          '<div class="flex items-start gap-3 rounded-xl bg-amber-50 px-4 py-3 text-sm text-amber-800 dark:bg-amber-500/10 dark:text-amber-200">' +
          '<svg class="mt-0.5 h-5 w-5 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 9v4M12 17h.01M10.3 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.7 3.86a2 2 0 0 0-3.4 0z"/></svg>' +
          '<div><div class="font-semibold">No config fits that budget.</div><div class="mt-0.5 opacity-80">Loosen a slider, the fidelity cap is usually the tight one.</div></div></div>';
        return;
      }
      var c = res.chosen;
      var primary = b.objective === "footprint"
        ? { big: c.weight_mb + " MB", label: "of weights" }
        : { big: fmtMs(c.ms_per_action), label: "ms / action" };
      resultEl.innerHTML =
        '<div class="ee-fade">' +
          '<div class="flex items-center justify-between">' +
            '<span class="ee-chip bg-signal-50 text-signal-700 ring-1 ring-inset ring-signal-200/70 dark:bg-signal-500/10 dark:text-signal-300 dark:ring-signal-400/20">' +
              '<span class="ee-light ee-light--go"></span> Picked config</span>' +
            '<span class="ee-mono text-xs text-slate-400">' + res.feas.length + ' of ' + CONFIGS.length + ' fit</span>' +
          '</div>' +
          '<div class="mt-4 flex items-end gap-4">' +
            '<div><div class="ee-mono text-4xl font-bold tracking-tight text-slate-900 dark:text-slate-50">' + primary.big + '</div>' +
            '<div class="text-xs font-medium text-slate-400">' + primary.label + '</div></div>' +
            '<div class="mb-1 flex flex-wrap gap-1.5">' +
              '<span class="ee-chip bg-slate-100 text-slate-700 dark:bg-white/10 dark:text-slate-200">' + c.precision + '</span>' +
              '<span class="ee-chip bg-slate-100 text-slate-700 dark:bg-white/10 dark:text-slate-200">' + c.steps + ' flow steps</span>' +
              '<span class="ee-chip bg-slate-100 text-slate-700 dark:bg-white/10 dark:text-slate-200">chunk ' + c.exec_horizon + '</span>' +
            '</div>' +
          '</div>' +
          '<div class="mt-4 grid grid-cols-3 gap-3 text-center">' +
            statCell(fmtMs(c.ms_per_action), "ms / action") +
            statCell(c.weight_mb + " MB", "footprint") +
            statCell(c.rmse.toFixed(3), "action rMSE") +
          '</div>' +
          '<div class="mt-3 flex items-center justify-between rounded-xl ee-panel px-4 py-2.5 text-sm">' +
            '<span class="text-slate-500 dark:text-slate-400">staleness of the last chunked action</span>' +
            '<span class="ee-mono font-semibold text-slate-700 dark:text-slate-200">' + c.staleness + ' steps</span>' +
          '</div>' +
        '</div>';
    }
    function statCell(big, label) {
      return '<div class="rounded-xl ee-panel px-3 py-2.5">' +
        '<div class="ee-mono text-lg font-bold text-slate-800 dark:text-slate-100">' + big + '</div>' +
        '<div class="text-[11px] text-slate-400">' + label + '</div></div>';
    }

    // ---- SVG Pareto scatter ----
    function draw(res) {
      var W = 480, H = 300, m = { l: 50, r: 14, t: 14, b: 40 };
      var iw = W - m.l - m.r, ih = H - m.t - m.b;
      var dark = root.classList.contains("dark");
      var axis = dark ? "rgb(100 116 139)" : "rgb(148 163 184)";
      var grid = dark ? "rgb(148 163 184 / 0.12)" : "rgb(15 23 42 / 0.07)";
      var label = dark ? "rgb(148 163 184)" : "rgb(100 116 139)";

      var xs = CONFIGS.map(function (c) { return Math.log10(c.ms_per_action); });
      var xLo = Math.min.apply(null, xs), xHi = Math.max.apply(null, xs);
      xLo -= 0.12; xHi += 0.12;
      var yLo = 10, yHi = 54;
      function X(ms) { return m.l + (Math.log10(ms) - xLo) / (xHi - xLo) * iw; }
      function Y(mb) { return m.t + (1 - (mb - yLo) / (yHi - yLo)) * ih; }

      var feasSet = new Set(res.feas);
      var svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" class="w-full" role="img" aria-label="Pareto frontier of deploy configs: latency versus footprint, coloured by action error">';

      // x grid + ticks at decade-ish marks
      var xticks = [0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 12];
      xticks.forEach(function (t) {
        if (Math.log10(t) < xLo || Math.log10(t) > xHi) return;
        var x = X(t);
        svg += line(x, m.t, x, m.t + ih, grid, 1);
        svg += text(x, H - m.b + 16, t, label, "middle");
      });
      // y grid + ticks
      [13.7, 26.4, 51].forEach(function (mb) {
        var y = Y(mb);
        svg += line(m.l, y, m.l + iw, y, grid, 1);
        svg += text(m.l - 8, y + 3, mb, label, "end");
      });
      // axis lines
      svg += line(m.l, m.t + ih, m.l + iw, m.t + ih, axis, 1.2);
      svg += line(m.l, m.t, m.l, m.t + ih, axis, 1.2);
      // axis labels
      svg += '<text x="' + (m.l + iw / 2) + '" y="' + (H - 4) + '" fill="' + label + '" font-size="11" text-anchor="middle" font-weight="600">ms / action  (log scale, lower is faster)</text>';
      svg += '<text x="14" y="' + (m.t + ih / 2) + '" fill="' + label + '" font-size="11" text-anchor="middle" font-weight="600" transform="rotate(-90 14 ' + (m.t + ih / 2) + ')">weight (MB)</text>';

      // points (infeasible faded, feasible solid), picked drawn last with a ring
      CONFIGS.forEach(function (c) {
        if (c === res.chosen) return;
        var feasible = feasSet.has(c);
        svg += '<circle class="ee-dot" cx="' + X(c.ms_per_action).toFixed(1) + '" cy="' + Y(c.weight_mb).toFixed(1) +
          '" r="5" fill="' + rmseColor(c.rmse) + '" opacity="' + (feasible ? 0.92 : 0.22) + '">' +
          '<title>' + c.precision + ', ' + c.steps + ' steps, chunk ' + c.exec_horizon + ' — ' + fmtMs(c.ms_per_action) + ' ms/action, ' + c.weight_mb + ' MB, rMSE ' + c.rmse.toFixed(3) + '</title></circle>';
      });
      if (res.chosen) {
        var cx = X(res.chosen.ms_per_action), cy = Y(res.chosen.weight_mb);
        svg += '<circle cx="' + cx.toFixed(1) + '" cy="' + cy.toFixed(1) + '" r="10" fill="none" stroke="rgb(79 70 229)" stroke-width="2.5"/>';
        svg += '<circle cx="' + cx.toFixed(1) + '" cy="' + cy.toFixed(1) + '" r="5" fill="' + rmseColor(res.chosen.rmse) + '"/>';
      }
      svg += "</svg>";
      plotEl.innerHTML = svg;
    }
    function line(x1, y1, x2, y2, stroke, w) {
      return '<line x1="' + x1.toFixed(1) + '" y1="' + y1.toFixed(1) + '" x2="' + x2.toFixed(1) + '" y2="' + y2.toFixed(1) + '" stroke="' + stroke + '" stroke-width="' + w + '"/>';
    }
    function text(x, y, t, fill, anchor) {
      return '<text x="' + x.toFixed(1) + '" y="' + y.toFixed(1) + '" fill="' + fill + '" font-size="10" text-anchor="' + anchor + '" font-family="ui-monospace, monospace">' + t + '</text>';
    }

    function update() {
      var b = budget();
      vals.lat.textContent = fmtMs(b.max_lat);
      vals.mb.textContent = (b.max_mb % 1 === 0 ? b.max_mb : b.max_mb.toFixed(1));
      vals.rmse.textContent = b.max_rmse.toFixed(3);
      vals.stale.textContent = b.max_stale;
      var res = pick(b);
      renderResult(res, b);
      draw(res);
    }
    window.__eeDraw = function () { update(); };

    objBtns.forEach(function (b) { b.addEventListener("click", function () { setObjective(b.getAttribute("data-obj")); update(); }); });
    Object.keys(els).forEach(function (k) { els[k].addEventListener("input", update); });
    var reset = document.getElementById("reset-budget");
    if (reset) reset.addEventListener("click", function () {
      els.lat.value = 1000; els.mb.value = 51; els.rmse.value = 0.05; els.stale.value = 49;
      setObjective("latency"); update();
    });

    setObjective("latency");
    update();
  }

  // ---- Supervisor scenario rows: reflect the checked radio ---------------
  function initScenarioStyling() {
    var labels = Array.prototype.slice.call(document.querySelectorAll(".ee-scen"));
    if (!labels.length) return;
    function refresh() {
      labels.forEach(function (l) {
        var input = l.querySelector("input[type=radio]");
        var on = input && input.checked;
        l.classList.toggle("border-signal-400", on);
        l.classList.toggle("bg-signal-50/60", on);
        l.classList.toggle("dark:border-signal-400/40", on);
        l.classList.toggle("dark:bg-signal-500/10", on);
      });
    }
    labels.forEach(function (l) { l.addEventListener("change", refresh); });
    refresh();
  }

  document.addEventListener("DOMContentLoaded", function () {
    initCompiler();
    initScenarioStyling();
  });
})();
