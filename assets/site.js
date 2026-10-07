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

    // tiny syntax highlighter for code[data-lang] (py, js, sh, c): comments, strings, keywords, numbers, calls
    const KW = { py: "import|from|def|return|async|await|for|in|if|else|elif|with|as|lambda|class|None|True|False|print|not|and|or",
                 js: "import|from|const|let|await|async|return|export|default|function|new|if|else|for|of|typeof|null|true|false",
                 c: "include|int|float|const|char|return|if|else|void|struct|double|unsigned",
                 sh: "pip|npm|shad0w|curl|cp|cc|open|python|install|export" };
    document.querySelectorAll("code[data-lang]").forEach((el) => {
      const lang = el.dataset.lang, src = el.textContent;
      const kw = new RegExp(`^(?:${KW[lang] || KW.py})\\b`);
      let out = "", i = 0;
      const esc = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      while (i < src.length) {
        const rest = src.slice(i);
        let m;
        if ((m = rest.match(lang === "js" || lang === "c" ? /^\/\/[^\n]*|^\/\*[\s\S]*?\*\// : /^#[^\n]*/))) out += `<span class="tok-c">${esc(m[0])}</span>`;
        else if ((m = rest.match(/^("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'|`[^`]*`)/))) out += `<span class="tok-s">${esc(m[0])}</span>`;
        else if (/[A-Za-z_]/.test(rest[0]) && !/[A-Za-z0-9_.]/.test(src[i - 1] || "") && (m = rest.match(kw))) out += `<span class="tok-k">${m[0]}</span>`;
        else if ((m = rest.match(/^[A-Za-z_][A-Za-z0-9_]*(?=\()/))) out += `<span class="tok-f">${m[0]}</span>`;
        else if ((m = rest.match(/^\d[\d_.,]*\b/)) && !/[A-Za-z_]/.test(src[i - 1] || "")) out += `<span class="tok-n">${m[0]}</span>`;
        else if ((m = rest.match(/^[A-Za-z_][A-Za-z0-9_]*/))) out += esc(m[0]);
        else { out += esc(rest[0]); i += 1; continue; }
        i += m[0].length;
      }
      el.innerHTML = out;
    });

    // copy buttons: on .install and on every code block
    document.querySelectorAll(".prose pre, .tabs pre").forEach((pre) => {
      let box = pre.parentElement;
      if (box.querySelector(":scope > .copy")) return;
      if (!box.classList.contains("codehilite")) {
        box = document.createElement("div");
        pre.replaceWith(box);
        box.appendChild(pre);
      }
      box.style.position = "relative";
      const b = document.createElement("button");
      b.className = "copy"; b.textContent = "copy"; b.type = "button";
      b.style.cssText = "position:absolute;top:8px;right:8px";
      box.appendChild(b);
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
