// shad0w site behaviour (dark only): mobile menu, syntax colours, copy buttons, tabs, reveals, spotlight, number tickers.
(function () {
  const root = document.documentElement;
  root.classList.add("js");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

  document.addEventListener("DOMContentLoaded", () => {
    const mb = document.querySelector("[data-menu]");
    if (mb) mb.addEventListener("click", () => {
      const l = document.querySelector(".nav .links"); l.classList.toggle("open");
      mb.setAttribute("aria-expanded", l.classList.contains("open"));
    });

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

    // copy buttons on every prose / tab code block (code windows bring their own)
    document.querySelectorAll(".prose pre, .tabs pre").forEach((pre) => {
      if (pre.closest(".codewin")) return;
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
      const scope = b.closest(".codewin") || b.parentElement;
      const text = b.dataset.copy || (scope.querySelector("pre") || {}).innerText || "";
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

    // spotlight: cards light up under the pointer
    document.addEventListener("pointermove", (e) => {
      const c = e.target.closest && e.target.closest(".spot");
      if (!c) return;
      const r = c.getBoundingClientRect();
      c.style.setProperty("--mx", (e.clientX - r.left) + "px"); c.style.setProperty("--my", (e.clientY - r.top) + "px");
    }, { passive: true });

    // number tickers (900 ms, ease-out). Elements: data-count="61" data-suffix="%"
    const countUp = (el) => {
      if (el.dataset.counted) return; el.dataset.counted = 1;
      const to = parseFloat(el.dataset.count), dec = (el.dataset.count.split(".")[1] || "").length;
      const pre = el.dataset.prefix || "", suf = el.dataset.suffix || "";
      if (reduce || isNaN(to)) { el.textContent = pre + (isNaN(to) ? el.dataset.count : to.toFixed(dec)) + suf; return; }
      const t0 = performance.now(), dur = 900;
      const step = (t) => {
        const k = Math.min(1, (t - t0) / dur), v = to * (1 - Math.pow(1 - k, 4));
        el.textContent = pre + v.toFixed(dec) + suf;
        if (k < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    };

    // reveals: once, at 15% visible; staggered 60 ms inside a group (cap 8). Anything already on screen is shown at once,
    // and everything is shown after 1.5 s, so nothing stays hidden in screenshots, print or slow devices.
    const items = [...document.querySelectorAll(".reveal")];
    const show = (el) => {
      if (el.classList.contains("in")) return;
      const sibs = el.parentElement ? [...el.parentElement.children].filter((x) => x.classList.contains("reveal")) : [];
      const idx = Math.min(7, Math.max(0, sibs.indexOf(el)));
      if (!reduce && idx) el.style.transitionDelay = (idx * 60) + "ms";
      el.classList.add("in");
      el.querySelectorAll("[data-count]").forEach(countUp);
      if (el.dataset.count) countUp(el);
    };
    const onScreen = (el) => { const r = el.getBoundingClientRect(); return r.top < innerHeight && r.bottom > 0; };
    if (reduce || !("IntersectionObserver" in window)) items.forEach(show);
    else {
      items.filter(onScreen).forEach(show);
      const io = new IntersectionObserver((entries) => entries.forEach((en) => {
        if (!en.isIntersecting) return;
        show(en.target); io.unobserve(en.target);
      }), { threshold: 0.15 });
      items.forEach((el) => io.observe(el));
      setTimeout(() => items.forEach(show), 1500);
    }
    document.querySelectorAll("[data-count]:not(.reveal [data-count])").forEach((el) => setTimeout(() => countUp(el), 200));
  });
})();
