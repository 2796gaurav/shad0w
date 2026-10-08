# shad0w for JavaScript

Your app asks an LLM the same kinds of small questions all day: which intent, which team, spam or not. **shad0w** learns the answers and serves them from a tiny table in **~10 µs**, with a certified bound on how often it disagrees with your LLM. Anything the table is unsure about still goes to your LLM.

Runs in Node 18+, browsers, Cloudflare Workers, Deno and Bun. Zero dependencies. ESM, CommonJS and TypeScript types are included.

```bash
npm i shad0wllm
```

## 1 · Put it in front of your LLM

```js
import { decision } from "shad0wllm";

const intent = await decision("intent", {
  options: { refund: "wants money back", lost_card: "card lost or stolen", balance: "asks about balance" },
  llm: "openai/gpt-6-luna",          // or anthropic/…, gemini/…, groq/…, ollama/llama3.1, or baseURL for any OpenAI-compatible server
  bundle: "shad0w/intent/bundle",      // a trained table, if you have one yet (path in Node, URL in browsers/workers)
  log: "shad0w/intent/log.jsonl",      // where the LLM's answers go (a file path in Node, or a function / stream anywhere)
});

const d = await intent.decide("my card was stolen");
// { answer: "lost_card", source: "table", confidence: 0.99, certified: true, flag: null, latencyUs: 9.8 }
```

With no bundle yet, every call goes to your LLM and the answer is logged. That is your training data.

## 2 · Train the table (once you have ~1,000 answers)

Training runs in Python:

```bash
pip install "shad0wllm[compile]"
shad0w train --log shad0w/intent/log.jsonl --out shad0w/intent/bundle
```

## 3 · Serve it anywhere

```js
import { Bundle } from "shad0wllm";
const bundle = await Bundle.load("shad0w/intent/bundle");     // Node: a folder. Browser / Worker: a URL
const a = bundle.decide("my card was stolen").answers.intent;
const answer = a.certified ? a.choice : await askMyLLM(text);  // certified: serve it; otherwise ask your LLM
```

**Cloudflare Worker**

```js
import { decision } from "shad0wllm";
let intent;
export default {
  async fetch(req, env) {
    intent ??= await decision("intent", { llm: "openai/gpt-6-luna", apiKey: env.OPENAI_API_KEY, bundle: env.BUNDLE_URL,
                                          log: (row) => env.LOGS.send(row) });   // e.g. a Queue
    const { text } = await req.json();
    return Response.json(await intent.decide(text));
  },
};
```

## Middleware and decision models (Vercel AI SDK)

```js
import { wrapLanguageModel, experimental_decide } from "ai";
import { openai } from "@ai-sdk/openai";
import { decision, shad0wMiddleware, decisionModel } from "shad0wllm";

const intent = await decision("intent", { options: ["refund", "lost_card", "other"], bundle: "shad0w/intent/bundle", log: "shad0w/intent/log.jsonl" });

// 1. generateText / generateObject: certified answers skip the model entirely; the model's answers are logged for training.
const model = wrapLanguageModel({ model: openai("gpt-6-luna"), middleware: shad0wMiddleware(intent, { specificationVersion: "v3" }) });

// 2. experimental_decide: the table answers the questions it certifies, gpt-6-luna answers the rest (and teaches the table).
const result = await experimental_decide({
  model: decisionModel(intent, { fallback: openai.decisionModel("gpt-6-luna") }),
  state: "my card was stolen",
  questions: { intent: { type: "choice", instructions: "What does the customer want?", criteria: { refund: null, lost_card: null, other: null } } },
});
```

`specificationVersion` must match the installed `ai` major (ai 5 → `"v2"`, ai 6 → `"v3"`, ai 7 → `"v4"`). Decision-model servers can also be the teacher: `llm: "systemone/kev-0.8b"` with `baseURL` (Jev, Kev, Laya, Ollaya, llama.cpp) or `llm: "openai-decisions/gpt-6-luna"`.

## API

- `decision(name?, { options, llm, bundle, log, fallback, rename, onNewOption, auditRate, onDecision, baseURL, apiKey, timeout })` → a `Shadow`. `name` defaults to `"decision"`; it names the question in the bundle and the log. `llm` is `"provider/model"`, `"systemone/<model>"`, `"openai-decisions/<model>"` or your own `async (text) => answer`. A bundle path that does not exist yet starts log-only; a corrupt bundle throws.
- `new Shadow(bundle, { teacher, question, options, log, fallback, rename, onNewOption, auditRate, onDecision })`:
  - `await shadow.decide(text)` returns `{ answer, source, confidence, certified, flag, latencyUs, question, threshold }`;
  - `await shadow.peek(text)` returns the table's decision when it would be served, else `null` (never calls the teacher);
  - `await shadow.record(text, answer)` logs an answer you obtained yourself;
  - `shadow.explain(text)` returns `{ answer, confidence, threshold, certified, flag, why, top }` without calling your model;
  - `shadow.stats()` returns offload and spot-check counts, plus `optionsAdded` / `optionsRemoved`.
- **Changing options.** Give `options` with a bundle: options the bundle never learned make every decision go to the teacher (`flag: "options_changed"`) until you retrain (`onNewOption: "serve"` to keep serving the known ones); removed options are never served (`"option_removed"`); `rename: { old: "new" }` applies without retraining.
- **No LLM.** Without a teacher, `fallback` (a value or `(text) => answer`) answers what the table is unsure about, with `source: "fallback"`; it is never logged.
- `openaiTeacher({ options, model, baseURL, apiKey, timeout })` → `async (text) => option`. It works with any OpenAI-compatible chat API, uses structured outputs with a plain-text fallback, retries, and times out after 30 s by default.
- `systemoneTeacher({ options, model, baseURL })` and `decisionsTeacher({ options, model, apiKey })` ask a System One server or the OpenAI Decisions API instead of a chat model.
- `shad0wMiddleware(shadow, { specificationVersion, format })` and `decisionModel(shadow | { [question]: shadow }, { fallback })` for the Vercel AI SDK (above).
- `Bundle.load(dirOrUrl)`; `bundle.decide(text)` → `{ answers: { [question]: { choice | answer, confidence, certified, flag, probabilities } } }`. `bundle.decide(text, { probabilities: false })` is the fast path: same answers and confidences, without the per-option map.
- `new Table(arrayBuffer, labels).decide(text)` works on a single `.s0` file.

Decisions match the Python and C runtimes exactly; the test suite checks it.

Docs: https://2796gaurav.github.io/shad0w/docs/javascript.html · Source: https://github.com/2796gaurav/shad0w · Apache-2.0
