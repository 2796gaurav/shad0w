// shad0w site behaviour: theme toggle, mobile menu, copy buttons, tabs, scroll reveal, count-up numbers.
(function () {
  const root = document.documentElement;
  try { const t = localStorage.getItem("shad0w-theme"); if (t) root.dataset.theme = t; } catch (e) {}

  document.addEventListener("DOMContentLoaded", () => {
    const tb = document.querySelector("[data-theme-toggle]");
    if (tb) tb.addEventListener("click", () => {
      const dark = root.dataset.theme ? root.dataset.theme === "dark" : !matchMedia("(prefers-color-scheme: light)").matches;
      root.dataset.theme = dark ? "light" : "dark";
      try { localStorage.setItem("shad0w-theme", root.dataset.theme); } catch (e) {}
    });
    const mb = document.querySelector("[data-menu]");
    if (mb) mb.addEventListener("click", () => document.querySelector(".nav .links").classList.toggle("open"));

    // copy buttons: on .install and on every code block
    document.querySelectorAll(".prose pre, .tabs pre").forEach((pre) => {
      if (pre.parentElement.querySelector(":scope > .copy")) return;
      const b = document.createElement("button");
      b.className = "copy"; b.textContent = "copy"; b.type = "button";
      b.style.cssText = "position:absolute;top:10px;right:10px";
      pre.parentElement.style.position = "relative";
      b.dataset.copyFrom = "pre";
      pre.parentElement.appendChild(b);
    });
    document.addEventListener("click", async (e) => {
      const b = e.target.closest(".copy");
      if (!b) return;
      const text = b.dataset.copy || (b.parentElement.querySelector("pre") || {}).innerText || "";
      try { await navigator.clipboard.writeText(text.trim()); b.textContent = "copied"; } catch (err) { b.textContent = "select + copy"; }
      setTimeout(() => (b.textContent = "copy"), 1400);
    });

    // tabs
    document.querySelectorAll(".tabs").forEach((tabs) => {
      const btns = tabs.querySelectorAll(".tabbar button"), panels = tabs.querySelectorAll(".tabpanel");
      btns.forEach((btn, i) => btn.addEventListener("click", () => {
        btns.forEach((x) => x.setAttribute("aria-selected", "false"));
        panels.forEach((p) => p.classList.remove("on"));
        btn.setAttribute("aria-selected", "true"); panels[i].classList.add("on");
      }));
    });

    // reveal + count-up
    const countUp = (el) => {
      const to = parseFloat(el.dataset.count), dec = (el.dataset.count.split(".")[1] || "").length;
      const pre = el.dataset.prefix || "", suf = el.dataset.suffix || "", t0 = performance.now(), dur = 1400;
      const step = (t) => {
        const k = Math.min(1, (t - t0) / dur), v = to * (1 - Math.pow(1 - k, 3));
        el.textContent = pre + v.toFixed(dec) + suf;
        if (k < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    };
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((entries) => entries.forEach((en) => {
        if (!en.isIntersecting) return;
        en.target.classList.add("in");
        en.target.querySelectorAll("[data-count]").forEach(countUp);
        if (en.target.dataset.count) countUp(en.target);
        io.unobserve(en.target);
      }), { threshold: 0.15 });
      document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
    } else document.querySelectorAll(".reveal").forEach((el) => el.classList.add("in"));
  });
})();
