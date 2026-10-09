// The "type anything" decision panel, shared by the landing page and the playground.
// mountTry(root, { bundle, q, msgs: [[text, llmLabel], ...], getThr, examples, onRender })
// Shows shad0w's top choice and confidence, the next choices, the certified line, why it answers or defers, µs per
// decision, and a searchable list of every intent (clicking one fills a real message your LLM gave that intent).
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pretty = (k) => k.replaceAll("_", " ");
export const usLabel = (v) => !isFinite(v) ? "< 50 µs" : v < 10 ? v.toFixed(1) + " µs" : v < 1000 ? v.toFixed(0) + " µs" : (v / 1000) < 100 ? (v / 1000).toFixed(1) + " ms" : (v / 1000).toFixed(0) + " ms";

export function mountTry(root, opts) {
  const { bundle, q = "intent", msgs = [], getThr, examples = [], onRender } = opts;
  const options = bundle.manifest.questions[q].options;
  const byLabel = {};
  msgs.forEach(([t, lab]) => { if (lab && !byLabel[lab] && t.length < 90) byLabel[lab] = t; });
  root.innerHTML = `
  <div class="tryui">
    <div class="try-in">
      <label class="tlabel" for="${root.id}-t">A customer message</label>
      <textarea id="${root.id}-t" spellcheck="false">${esc(examples[0] || "my card got stolen at the airport, please block it")}</textarea>
      <div class="tchips">${examples.map((e) => `<button type="button" class="chip" data-ex="${esc(e)}">${esc(e)}</button>`).join("")}</div>
      <details class="intents">
        <summary>Browse all ${options.length} intents <span>click one to try a real message</span></summary>
        <input type="search" placeholder="filter intents, e.g. card, refund, top up" aria-label="Filter intents">
        <div class="ilist">${options.slice().sort().map((o) => `<button type="button" class="ichip${byLabel[o] ? "" : " noex"}" data-i="${esc(o)}">${esc(pretty(o))}</button>`).join("")}</div>
      </details>
    </div>
    <div class="try-out">
      <div class="tv"><span class="badge" data-r="badge">loading…</span></div>
      <div class="top"><b data-r="ans">–</b><span data-r="conf">–</span></div>
      <div class="tgauge"><div class="fill" data-r="fill"></div><div class="thr" data-r="thr"><span>certified line</span></div></div>
      <p class="why" data-r="why">&nbsp;</p>
      <div class="alts" data-r="alts"></div>
      <div class="tk"><div><b data-r="us">–</b><span>per decision, in this tab</span></div><div><b data-r="line">–</b><span>shad0w answers above this confidence</span></div></div>
      <div data-r="extra"></div>
    </div>
  </div>`;
  const $ = (r) => root.querySelector(`[data-r="${r}"]`);
  const ta = root.querySelector("textarea");
  const time = (text) => {  // browsers round performance.now(): time a batch, take the median of 5
    const ts = [];
    for (let b = 0; b < 5; b++) { const t0 = performance.now(); for (let i = 0; i < 200; i++) bundle.decide(text, { questions: [q] }); ts.push((performance.now() - t0) * 1000 / 200); }
    const us = ts.sort((a, b) => a - b)[2];
    return us > 0.2 ? us : NaN;  // a frozen or very coarse clock: the caller shows a dash instead of 0.0
  };
  function render() {
    const text = ta.value.trim() || " ";
    const a = bundle.decide(text, { questions: [q] }).answers[q];
    const thr = getThr ? getThr() : (bundle.manifest.questions[q].threshold ?? 1);
    const ok = a.confidence >= thr;
    const pc = (v) => (100 * v).toFixed(1) + "%";
    $("badge").className = "badge " + (ok ? "ok" : "defer");
    $("badge").textContent = ok ? "shad0w answers" : "sent to your LLM";
    $("ans").textContent = pretty(a.choice);
    $("conf").textContent = pc(a.confidence) + " sure"; $("conf").classList.toggle("defer", !ok); $("alts").classList.toggle("defer", !ok);
    $("fill").style.width = pc(a.confidence); $("fill").classList.toggle("ok", ok);
    $("thr").style.left = pc(Math.min(1, thr));
    $("why").innerHTML = ok
      ? `shad0w is <b>${pc(a.confidence)}</b> sure, above its certified line of ${pc(thr)}, so it answers right here and your LLM is never called.`
      : `shad0w is only <b>${pc(a.confidence)}</b> sure, below its certified line of ${pc(thr)}, so this message goes to your LLM. Its answer is logged and shad0w learns from it.`;
    const top = Object.entries(a.probabilities).sort((x, y) => y[1] - x[1]).slice(0, 5);
    $("alts").innerHTML = top.map(([k, v], i) => `<div class="alt${i ? "" : " first"}"><span class="n" title="${esc(k)}">${esc(pretty(k))}</span><span class="tr"><i style="width:${(100 * v).toFixed(1)}%"></i></span><span class="v">${pc(v)}</span></div>`).join("");
    $("us").textContent = usLabel(time(text));
    $("line").textContent = pc(thr);
    root.querySelectorAll(".ichip").forEach((c) => c.classList.toggle("on", c.dataset.i === a.choice));
    onRender && onRender(a, ok);
  }
  ta.addEventListener("input", render);
  root.querySelector(".tchips").addEventListener("click", (e) => { const b = e.target.closest("[data-ex]"); if (b) { ta.value = b.dataset.ex; render(); } });
  const list = root.querySelector(".ilist");
  root.querySelector(".intents input").addEventListener("input", (e) => {
    const f = e.target.value.toLowerCase().replaceAll("_", " ");
    list.querySelectorAll(".ichip").forEach((c) => (c.hidden = f && !c.textContent.includes(f)));
  });
  list.addEventListener("click", (e) => {
    const b = e.target.closest(".ichip"); if (!b) return;
    const ex = byLabel[b.dataset.i];
    if (ex) { ta.value = ex; render(); ta.scrollIntoView({ block: "nearest", behavior: "smooth" }); }
    else { list.querySelectorAll(".ichip").forEach((c) => c.classList.toggle("on", c === b)); }
  });
  render();
  return { render, extra: $("extra"), textarea: ta };
}
