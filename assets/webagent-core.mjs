// The web-agent playground's engine: the shop app (real HTML pages), what the agent reads from a page's HTML, a
// scripted stand-in for the LLM, and one agent step with the two safety checks. Mirrors arena/webagent (Python);
// describe() must produce the exact text the table was trained on (checked against Python, page by page).

export const PRODUCTS = {
  mug: ["Ceramic mug", ["white", "blue", "black"]], lamp: ["Desk lamp", ["brass", "white", "black"]],
  tee: ["Cotton t-shirt", ["S", "M", "L", "XL"]], hoodie: ["Fleece hoodie", ["S", "M", "L"]],
  shoes: ["Running shoes", ["40", "41", "42", "43", "44"]], bottle: ["Water bottle", ["500 ml", "750 ml", "1 l"]],
  backpack: ["Laptop backpack", ["grey", "black", "navy"]], headphones: ["Wireless headphones", ["black", "silver"]],
  notebook: ["Paper notebook", ["A5", "A4"]], plant: ["Potted plant", ["small", "large"]],
  chair: ["Office chair", ["black", "grey"]], kettle: ["Electric kettle", ["steel", "white"]],
};
const CITIES = ["Pune", "Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Hyderabad", "Jaipur"];
export const ACTIONS = ["click:accept-cookies", "click:reject-cookies", "type:q=product", "click:search",
  ...Object.keys(PRODUCTS).map((s) => `click:p-${s}`),
  "select:variant=variant", "type:qty=qty", "click:add", "click:nav-cart", "click:checkout", "type:email=email",
  "type:password=password", "click:login", "type:city=city", "click:place-order", "click:retry", "done", "fail:out-of-stock",
  "click:close-newsletter", "click:terms"];
export const RISKY = new Set(["click:place-order"]);
export const MAX_STEPS = 25;

export function rng(seed) {   // mulberry32
  let a = (seed * 2654435761) >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const pick = (r, xs) => xs[Math.floor(r() * xs.length)];
function sample(r, xs, k) { const a = xs.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a.slice(0, k); }
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#x27;");

function page(title, body, banner) {
  const cookie = banner ? '<div class="cookie" role="dialog" aria-label="Cookie consent"><p>We use cookies to improve your visit.</p>'
    + '<button id="accept-cookies">Accept all</button><button id="reject-cookies">Reject non-essential</button></div>' : "";
  return `<!doctype html><html><head><title>${esc(title)} | Bazaar</title></head><body>`
    + '<nav><a id="nav-home" href="/">Bazaar</a> <a id="nav-cart" href="/cart">Cart</a> <a id="nav-account" href="/login">Account</a></nav>'
    + `${cookie}<main>${body}</main><footer>© Bazaar</footer></body></html>`;
}

export class Shop {
  constructor(seed, shift = false) {
    const r = this.r = rng(seed);
    this.banner = r() < .5; this.loginWall = r() < .5; this.decline = r() < .2;
    this.oos = Object.fromEntries(Object.entries(PRODUCTS).map(([s, [, vs]]) => [s, new Set(vs.filter(() => r() < .15))]));
    this.path = "/"; this.cart = []; this.loggedIn = false; this.placed = false; this.declinedOnce = false;
    this.form = {}; this.msg = ""; this.query = ""; this.product = null;
    const r2 = rng(seed + 777);   // shift: a newsletter pop-up after the first search, a terms box before ordering
    this.shift = shift; this.newsletter = false; this.terms = false; this.newsletterDue = shift && r2() < .6;
  }
  v(el, kind = "text") { const x = this.form[el] || ""; return x ? ` value="${kind === "password" ? "•".repeat(x.length) : esc(x)}"` : ""; }
  html() {
    const h = this._html();
    return this.newsletter ? h.replace("<main>", '<div class="modal" role="dialog" aria-label="Newsletter"><p>Get 10% off your first order! Join our newsletter.</p>'
      + '<button id="close-newsletter">No thanks</button></div><main>') : h;
  }
  _html() {
    const p = this.path, b = this.banner, note = this.msg ? `<p class="alert">${esc(this.msg)}</p>` : "";
    if (p === "/") return page("Home", `<h1>Welcome to Bazaar</h1><form><label for="q">Search products</label><input id="q" name="q" type="search"${this.v("q")}>`
      + '<button id="search">Search</button></form><section><h2>Popular</h2><p>Mugs, lamps, shoes and more.</p></section>', b);
    if (p === "/search") {
      let hits = Object.keys(PRODUCTS).filter((s) => { const n = PRODUCTS[s][0].toLowerCase(), q = this.query.toLowerCase(); return this.query && (n.includes(q) || q.includes(n.split(" ").pop())); });
      if (hits.length) hits = hits.concat(sample(this.r, Object.keys(PRODUCTS).filter((x) => !hits.includes(x)), 3));
      const items = sample(this.r, hits, hits.length).map((s) => `<li><a id="p-${s}" href="/p/${s}">${esc(PRODUCTS[s][0])}</a></li>`).join("");
      const body = `<h1>Results for "${esc(this.query)}"</h1>` + (hits.length ? `<ul>${items}</ul>` : '<p class="empty">No products match your search.</p>');
      return page("Search", body + `<form><input id="q" name="q" type="search"${this.v("q")}><button id="search">Search again</button></form>`, b);
    }
    if (p.startsWith("/p/")) {
      const s = this.product, [n, vs] = PRODUCTS[s];
      const opts = vs.map((v) => `<option value="${esc(v)}"${this.oos[s].has(v) ? " disabled" : ""}${this.form.variant === v ? " selected" : ""}>${esc(v)}${this.oos[s].has(v) ? " (out of stock)" : ""}</option>`).join("");
      return page(n, `<h1>${esc(n)}</h1>${note}<label for='variant'>Variant</label><select id='variant'>${opts}</select>`
        + `<label for='qty'>Quantity</label><input id='qty' type='number'${this.v("qty") || " value='1'"}><button id='add'>Add to cart</button>`, b);
    }
    if (p === "/cart") {
      const rows = this.cart.map(([s, v, q]) => `<li>${esc(PRODUCTS[s][0])}, ${esc(v)} × ${q}</li>`).join("");
      return page("Cart", "<h1>Your cart</h1>" + (this.cart.length ? `<ul>${rows}</ul><button id='checkout'>Checkout</button>` : "<p class='empty'>Your cart is empty.</p>"), b);
    }
    if (p === "/login") return page("Sign in", `<h1>Sign in to continue</h1>${note}<label for='email'>Email</label><input id='email' type='email'${this.v("email")}>`
      + `<label for='password'>Password</label><input id='password' type='password'${this.v("password", "password")}><button id='login'>Sign in</button>`, b);
    if (p === "/checkout") return page("Checkout", `<h1>Shipping</h1>${note}<label for='city'>City</label><input id='city' type='text'${this.v("city")}>`
      + (this.shift ? `<input id='terms' type='checkbox'${this.terms ? " checked" : ""}><label for='terms'>I agree to the terms of sale</label>` : "")
      + "<button id='place-order'>Place order</button>", b);
    if (p === "/declined") return page("Payment", "<h1>Payment declined</h1><p class='alert'>Your bank declined the payment. You can try again.</p><button id='retry'>Try again</button>", b);
    if (p === "/done") return page("Thank you", `<h1>Order placed</h1><p>Thank you! Your order number is 104${10 + Math.floor(this.r() * 90)}.</p>`, b);
    return page("Not found", "<h1>Page not found</h1>", b);
  }
  act({ action: kind, id: el, value: val }) {
    this.msg = "";
    if (this.newsletter && el !== "close-newsletter") return this.html();
    if (el === "close-newsletter") { this.newsletter = false; return this.html(); }
    if (el === "terms" && this.shift) { this.terms = !this.terms; return this.html(); }
    if (this.banner && el !== "accept-cookies" && el !== "reject-cookies") return this.html();
    if (el === "accept-cookies" || el === "reject-cookies") this.banner = false;
    else if (kind === "type" || kind === "select") this.form[el] = val || "";
    else if (el === "search") { this.query = this.form.q || ""; this.path = "/search"; if (this.newsletterDue) { this.newsletter = true; this.newsletterDue = false; } }
    else if (el && el.startsWith("p-") && PRODUCTS[el.slice(2)]) { this.product = el.slice(2); this.path = `/p/${this.product}`; this.form = {}; }
    else if (el === "add" && this.product) {
      const v = this.form.variant || PRODUCTS[this.product][1][0];
      if (this.oos[this.product].has(v) || !PRODUCTS[this.product][1].includes(v)) this.msg = "That variant is out of stock.";
      else { this.cart.push([this.product, v, parseInt(this.form.qty || "1", 10)]); this.path = "/cart"; }
    } else if (el === "nav-cart") this.path = "/cart";
    else if (el === "nav-home") this.path = "/";
    else if (el === "nav-account") this.path = "/login";
    else if (el === "checkout" && this.cart.length) this.path = this.loginWall && !this.loggedIn ? "/login" : "/checkout";
    else if (el === "login") {
      if ((this.form.email || "").includes("@") && this.form.password) { this.loggedIn = true; this.path = this.cart.length ? "/checkout" : "/"; }
      else this.msg = "Enter your email and password.";
    } else if (el === "place-order") {
      if (!this.form.city) this.msg = "Enter a city.";
      else if (this.shift && !this.terms) this.msg = "Please agree to the terms of sale.";
      else if (this.decline && !this.declinedOnce) { this.declinedOnce = true; this.path = "/declined"; }
      else { this.placed = true; this.path = "/done"; }
    } else if (el === "retry") this.path = "/checkout";
    return this.html();
  }
}

export function makeTask(seed) {
  const r = rng(10000 + seed), s = pick(r, Object.keys(PRODUCTS)), [n, vs] = PRODUCTS[s];
  const t = { product: n, slug: s, variant: pick(r, vs), qty: pick(r, [1, 1, 2, 3]), city: pick(r, CITIES),
    email: `user${100 + Math.floor(r() * 900)}@example.com`, password: "test-pass-" + (1000 + Math.floor(r() * 9000)) };
  t.text = `Buy ${t.qty} × ${n} (${t.variant}) and ship to ${t.city}.`;
  return t;
}

// ---------------------------------------------------------------- what the agent reads
function parse(html) {
  const d = new DOMParser().parseFromString(html, "text/html");
  const dialogs = [...d.querySelectorAll("[role=dialog]")];
  const els = [...d.querySelectorAll("a[id], button[id], input[id], select[id]")];
  return { d, els, dialog: dialogs.length ? (dialogs.at(-1).getAttribute("aria-label") || "Cookie consent") : null };
}
const pyRepr = (s) => (s.includes("'") && !s.includes('"') ? `"${s}"` : `'${s.replace(/\\/g, "\\\\").replace(/'/g, "\\'")}'`);

export function elementIds(html) {
  const { els, dialog } = parse(html), ids = new Set(els.map((e) => e.id));
  if (dialog) ids.add("dialog:" + dialog);
  return ids;
}

/** The page in plain words, plus the actions possible on it (exactly the text the table was trained on). */
export function describe(html, task, history) {
  const { d, els, dialog } = parse(html);
  const txt = (e) => (e ? e.textContent.trim() : "");
  const lines = [`Task: buy ${task.qty} x ${task.product} (${task.variant}), ship to ${task.city}, account ${task.email}.`,
    `Page: ${txt(d.querySelector("title"))}. Heading: ${txt(d.querySelector("h1"))}.`];
  if (dialog) lines.push(`A ${dialog.toLowerCase()} dialog covers the page.`);
  const alerts = [...d.querySelectorAll("[class]")].filter((e) => /alert|empty/.test(e.getAttribute("class"))).map(txt).filter(Boolean);
  if (alerts.length) lines.push("Messages: " + alerts.join(" / ") + ".");
  const slot = { q: task.product, qty: String(task.qty), email: task.email, city: task.city, variant: task.variant };
  for (const e of els) {
    const tag = e.tagName.toLowerCase();
    if (tag === "input" && (e.getAttribute("type") || "text") === "checkbox") lines.push(`Checkbox ${e.id}: ${e.hasAttribute("checked") ? "ticked" : "not ticked"}.`);
    else if (tag === "input") {
      const v = e.getAttribute("value") || "", type = e.getAttribute("type") || "text";
      let st = !v ? "empty" : type === "password" ? "filled (hidden)" : v === slot[e.id] ? "matches the task" : `has ${pyRepr(v)}, not the task's value`;
      if (e.id === "qty" && v === "1" && task.qty !== 1) st = "has 1, not the task's value";
      lines.push(`Field ${e.id}: ${st}.`);
    } else if (tag === "select") {
      const opts = [...e.querySelectorAll("option")].map((o) => ({ v: o.getAttribute("value") || "", sel: o.hasAttribute("selected"), off: o.hasAttribute("disabled") }));
      const sel = opts.find((o) => o.sel)?.v, mine = opts.find((o) => o.v === task.variant);
      lines.push(`Variant list: task variant ${task.variant} is ${mine && mine.off ? "out of stock" : mine ? "available" : "not offered"}; `
        + `selected: ${sel === task.variant ? "the task variant" : (sel || "none")}.`);
    } else if (tag === "a" && e.id.startsWith("p-")) lines.push(`Product link ${e.id}: ${txt(e)}${txt(e) === task.product ? " (the task product)" : ""}.`);
    else if (tag === "button") lines.push(`Button ${e.id}: ${txt(e)}.`);
  }
  lines.push("Recent actions: " + (history.length ? history.slice(-4).join(", ") : "none") + ".");
  const ids = new Set(els.map((e) => e.id));
  const ok = new Set(ACTIONS.filter((a) => a === "done" || a === "fail:out-of-stock" || ids.has(a.split(/[:=]/)[1])));
  return { text: lines.join("\n"), ok };
}

/** The page's state without history or dynamic content (link lists change on every render). */
export const signature = (text) => text.split("\n").filter((l) => !l.startsWith("Recent actions") && !l.startsWith("Product link")).sort().join("\n");

/** A scripted agent that knows the shop. On this page it stands in for the LLM (in our tests the real LLM chose the same). */
export function scripted(app, task) {
  if (app.newsletter) return "click:close-newsletter";
  if (app.banner) return "click:accept-cookies";
  const p = app.path;
  if (p === "/") return app.form.q === task.product ? "click:search" : "type:q=product";
  if (p === "/search") return `click:p-${task.slug}`;
  if (p.startsWith("/p/")) {
    if (app.oos[task.slug].has(task.variant)) return "fail:out-of-stock";
    if (app.form.variant !== task.variant) return "select:variant=variant";
    if (task.qty !== 1 && app.form.qty !== String(task.qty)) return "type:qty=qty";
    return "click:add";
  }
  if (p === "/cart") return "click:checkout";
  if (p === "/login") return app.form.email !== task.email ? "type:email=email" : !app.form.password ? "type:password=password" : "click:login";
  if (p === "/checkout") return app.form.city !== task.city ? "type:city=city" : app.shift && !app.terms ? "click:terms" : "click:place-order";
  if (p === "/declined") return "click:retry";
  return "done";
}

export function doAction(app, action, task) {
  if (action === "done" || action === "fail:out-of-stock") return "";
  const [kind, rest] = [action.slice(0, action.indexOf(":")), action.slice(action.indexOf(":") + 1)];
  const [el, slotName] = rest.split("=");
  const val = { product: task.product, variant: task.variant, qty: String(task.qty), email: task.email, password: task.password, city: task.city }[slotName];
  return app.act({ action: kind, id: el, value: val });
}

export function success(app, task, final) {
  if (app.oos[task.slug].has(task.variant)) return final === "fail:out-of-stock" && !app.placed;
  const c = app.cart;
  return final === "done" && app.placed && c.length === 1 && c[0][0] === task.slug && c[0][1] === task.variant && c[0][2] === task.qty
    && app.form.city === task.city && (!app.loginWall || app.loggedIn);
}

/** An errand, one step at a time. opts: {shift, novel, progress, risky, known:Set}. */
export class Episode {
  constructor(bundle, seed, opts) {
    this.b = bundle; this.o = opts; this.seed = seed; this.app = new Shop(seed, opts.shift); this.task = makeTask(seed);
    this.hist = []; this.html = this.app.html(); this.steps = 0; this.served = 0; this.llm = 0; this.final = null; this.stuck = false; this.tableUs = 0;
  }
  get over() { return this.final !== null || this.steps >= MAX_STEPS; }
  /** Decide the next action on the current page: -> {text, action, by: "table"|"llm", reason, a (table answer)} */
  decide() {
    const { text, ok } = describe(this.html, this.task, this.hist);
    const t0 = performance.now(), a = this.b.decide(text, { probabilities: false }).answers.action; this.tableUs += (performance.now() - t0) * 1000;
    let reason = null;
    const novelIds = this.o.novel ? [...elementIds(this.html)].filter((x) => !this.o.known.has(x)) : [];
    if (this.stuck) reason = "no progress: the last table action changed nothing";
    else if (novelIds.length) reason = `new on this page: ${novelIds.map((x) => x.replace("dialog:", "") + (x.startsWith("dialog:") ? " pop-up" : "")).join(", ")}`;
    else if (!a.certified) reason = "not certified";
    else if (!ok.has(a.choice)) reason = "not possible on this page";
    else if (this.o.risky && RISKY.has(a.choice)) reason = "irreversible: always asks the LLM";
    const by = reason ? "llm" : "table", action = reason ? scripted(this.app, this.task) : a.choice;
    return { text, action, by, reason, a };
  }
  /** Execute a decision. */
  apply(dec) {
    this.steps++; this.hist.push(dec.action);
    if (dec.by === "table") this.served++; else this.llm++;
    if (dec.action === "done" || dec.action === "fail:out-of-stock") { this.final = dec.action; return; }
    const next = doAction(this.app, dec.action, this.task);
    this.stuck = !!this.o.progress && dec.by === "table" && signature(describe(next, this.task, this.hist).text) === signature(dec.text);
    this.html = next;
  }
  get ok() { return success(this.app, this.task, this.final); }
}
