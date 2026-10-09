---
title: JavaScript & TypeScript
description: Run shad0w in Node, browsers, Cloudflare Workers, Deno and Bun - decision(), Shadow, decideMany, configure, teachers and Bundle.
---
# JavaScript & TypeScript

Serve a trained shad0w table from JavaScript, in front of your LLM, in Node, the browser or at the edge. Training stays in Python; serving runs anywhere.

```bash
npm i shad0wllm
```

Zero dependencies, ESM and CommonJS, TypeScript types included. It runs in Node 18+, browsers, Cloudflare Workers, Deno and Bun. Decisions are identical to Python and C on the same table.

## TL;DR

```js
import { decision } from "shad0wllm";

const intent = await decision("intent", {
  options: { refund: "wants money back", lost_card: "card lost or stolen", balance: "asks about balance", other: "anything else" },
  llm: "openai/gpt-6-luna",
  apiKey: process.env.OPENAI_API_KEY,
  bundle: "shad0w/intent/bundle",          // a trained table: a folder in Node, a URL in browsers and workers
  log: "shad0w/intent/log.jsonl",          // where your LLM's answers go (the training data)
});

const d = await intent.decide("my card was stolen");
// { answer: "lost_card", source: "table", confidence: 0.99, certified: true, flag: null,
//   why: "certified: answered by the table", threshold: 0.91, question: "intent", latencyUs: 9.8 }
```

The table (shad0w's small learned model) answers when it is **certified**: tested on answers it never trained on, and allowed to answer only where it disagrees with your LLM on at most α of them (α = 5% by default). Otherwise your LLM answers and the answer goes to `log`. A bundle path that does not exist yet is fine: every call goes to your LLM until you train one.

## What a decision contains

| Field | Type | Meaning |
|---|---|---|
| `answer` | `string` or `boolean` | the option (a `boolean` for yes/no questions) |
| `source` | `"table"`, `"teacher"` or `"fallback"` | who answered: the table, your LLM (logged), or your `fallback` (not logged) |
| `certified` | `boolean` | the answer came from the table and is covered by the certificate |
| `confidence` | `number` or `null` | the table's confidence in its top option; `null` without a table |
| `threshold` | `number` or `null` | the table's certified threshold |
| `flag` | `string` or `null` | why the table did not answer: `no_bundle`, `low_confidence`, `uncalibrated`, `options_changed`, `option_removed` ([all flags](python.html#flags)) |
| `why` | `string` | the flag in plain words (same wording as Python) |
| `latencyUs` | `number` | the whole call in microseconds, including any LLM call |
| `question` | `string` | the decision's name |

## Keys {#keys}

`llm` takes the same `"provider/model"` names as Python (`anthropic/…`, `groq/…`, `ollama/llama3.1`, …; see [Connect your LLM](connect-your-llm.html)), or a model name plus `baseURL` for any OpenAI-compatible server.

The key, first match wins:

1. `apiKey`: a string, or a function `() => key` (sync or async) called on every request, so rotation just works;
2. `apiKeyEnv`: the *name* of an environment variable (Node);
3. `configure({ apiKey })`, then `configure({ apiKeyEnv })`;
4. the provider's usual variable, e.g. `OPENAI_API_KEY` (Node).

```js
import { configure, decision } from "shad0wllm";

configure({ llm: "openai/gpt-6-luna", apiKeyEnv: "SUPPORT_BOT_OPENAI_KEY" });   // once, for the whole process
const intent = await decision("intent", { options: ["refund", "lost_card", "other"] });
console.log(String(intent));   // Shadow("intent", options=3, logging (no table yet), llm=openai/gpt-6-luna, key: set via SUPPORT_BOT_OPENAI_KEY (sk-…3f9a))
configure();                   // no argument: clear the defaults
```

The key stays in a closure. It is not a property of the teacher and never appears in errors or logs; `String(intent)` shows it masked. `apiKey`, `apiKeyEnv` or `baseURL` with a function `llm`, or a mistyped option (`apikey` → `did you mean "apiKey"?`), throw a `TypeError`.

## Train (Python, once)

```bash
pip install "shad0wllm[compile]"
shad0w train --log shad0w/intent/log.jsonl --out shad0w/intent/bundle
```

The `log` rows have the same format as Python's, so `shad0w train` reads logs written by either language. Ship the bundle folder (`manifest.json` plus one `.s0` file per question) with your app, or put it on a CDN.

## Serve

**Node**: as in the TL;DR. A file path for `log` appends JSON lines.

**Cloudflare Worker**: certified answers never leave the edge.

```js
import { decision } from "shad0wllm";
let intent;
export default {
  async fetch(req, env) {
    intent ??= await decision("intent", {
      llm: "openai/gpt-6-luna", apiKey: env.OPENAI_API_KEY,
      bundle: env.BUNDLE_URL,                          // e.g. a static-assets or R2 URL
      log: (row) => env.LOG_QUEUE.send(row),           // collect LLM answers for the next training run
    });
    const { text } = await req.json();
    return Response.json(await intent.decide(text));
  },
};
```

With a bundle and no `options`, the options come from the bundle.

**Browser**: decide as the user types, with no server and no LLM.

```js
import { Bundle } from "https://cdn.jsdelivr.net/npm/shad0wllm@0.3.2/index.mjs";
const bundle = await Bundle.load("/models/intent");
const a = bundle.decide(input.value).answers.intent;   // { choice, confidence, certified, flag, probabilities }
if (a.certified) show(a.choice);                       // otherwise ask your server
```

## Batches

```js
const results = await intent.decideMany(texts, { concurrency: 8 });   // decisions, in the same order as texts
```

The table answers first; only the texts it defers go to your LLM, at most `concurrency` at a time.

## Vercel AI SDK

```js
import { wrapLanguageModel, experimental_decide } from "ai";
import { openai } from "@ai-sdk/openai";
import { decision, shad0wMiddleware, decisionModel } from "shad0wllm";

const intent = await decision("intent", {
  options: ["refund", "lost_card", "other"], bundle: "shad0w/intent/bundle", log: "shad0w/intent/log.jsonl",
});

// generateText / generateObject: certified answers skip the model; the model's answers are logged for training.
const model = wrapLanguageModel({ model: openai("gpt-6-luna"), middleware: shad0wMiddleware(intent, { specificationVersion: "v3" }) });

// experimental_decide: the table answers the questions it certifies, the fallback decision model answers the rest.
const result = await experimental_decide({
  model: decisionModel(intent, { fallback: openai.decisionModel("gpt-6-luna") }),
  state: "my card was stolen",
  questions: { intent: { type: "choice", instructions: "What does the customer want?", criteria: { refund: null, lost_card: null, other: null } } },
});
```

| Option | Default | Meaning |
|---|---|---|
| `shad0wMiddleware(shadow, { specificationVersion })` | `"v3"` | must match the installed `ai` major: ai 5 → `"v2"`, ai 6 → `"v3"`, ai 7 → `"v4"` |
| `shad0wMiddleware(shadow, { format })` | the bare option | shapes the reply text of a table answer; for enum output through `generateObject`, pass `(a) => JSON.stringify({ result: a })` |
| `shad0wMiddleware(shadow, { options })` | the Shadow's options | the options to find in the model's replies |
| `decisionModel(shadows, { fallback })` | none | another decision model for the questions the tables do not certify. Without it, those come back as `{ type: "refusal" }` with a warning. |

`decisionModel` takes one `Shadow` or `{ [question]: Shadow }`. More recipes: [Frameworks](frameworks.html#vercel-ai-sdk).

## Decision models as the teacher {#decision-models}

```js
const intent = await decision("intent", {
  options: { lost_card: "card lost or stolen", refund: "wants money back", other: "anything else" },
  llm: "systemone/kev", baseURL: "http://127.0.0.1:8081",   // any /v1/systemone server: Jev, Kev, Laya, llama.cpp
  // llm: "openai-decisions/gpt-6-luna",                     // the OpenAI Decisions API (reads OPENAI_API_KEY)
});
```

See [Decisions API & System One](decisions-api.html) for running an open decision model locally.

## API

### `decision(name?, opts)` → `Promise<Shadow>`

`name` defaults to `"decision"`; `decision(opts)` with `opts.name` works too.

| Option | Type | Default | What it does |
|---|---|---|---|
| `options` | `string[]` or `{ name: description }` | the bundle's | the possible answers; needed with a `"provider/model"` `llm` until a bundle exists |
| `llm` | `string` or `async (text) => answer` | `configure({ llm })` | `"provider/model"` or your own function |
| `apiKey`, `apiKeyEnv` | | see [Keys](#keys) | the key, or the name of the variable that holds it |
| `baseURL` | `string` | `configure({ baseURL })`, then `SHAD0W_BASE_URL`, then the provider's | an OpenAI-compatible server |
| `bundle` | path, URL or `Bundle` | none | the trained table. Missing = log-only; corrupt or unsupported = throws. |
| `log` | path (Node), `(row) => void` or a stream | none | where your LLM's answers go. Without it nothing is logged. |
| `fallback` | value or `(text) => answer` | none | the answer when the table defers and there is no `llm` (`source: "fallback"`, never logged) |
| `auditRate` | `number` | `0.01` | share of answers served by the table that are re-asked to the LLM in the background (spot checks) |
| `onDecision` | `(d, text) => void` | none | called after every decision |
| `rename` | `{ old: new }` | `{}` | rename labels without retraining ([Changing your options](options.html)) |
| `onNewOption` | `"defer"` or `"serve"` | `"defer"` | options added since training: `defer` sends everything to the LLM until you retrain |
| `system`, `temperature`, `maxTokens`, `retries`, `headers`, `structured` | | built from options, `0`, `50`, `2`, `{}`, `true` | the LLM request, as in [Python](connect-your-llm.html#llm-settings) |
| `timeout` | `number` (ms) | `30000` | per attempt |
| `fetch` | `typeof fetch` | `globalThis.fetch` | a custom fetch |

### `Shadow`

`new Shadow(bundle | null, { teacher, question, options, log, fallback, rename, onNewOption, auditRate, onDecision })` builds one directly. With `options` and a bundle, options the bundle never learned send every decision to the teacher until you retrain, and removed ones are never served.

| Method | Returns | What it does |
|---|---|---|
| `await s.decide(text, { teacher })` | decision | the table's answer when certified, otherwise the teacher's (logged) |
| `await s.decideMany(texts, { concurrency: 8 })` | decisions | a batch, in order ([above](#batches)) |
| `await s.peek(text)` | decision or `null` | the table's decision if it would serve it; never calls the teacher |
| `await s.record(text, answer)` | decision | log an answer you got from your model yourself |
| `s.explain(text)` | `{ text, answer, confidence, threshold, certified, flag, why, top }` | what the table thinks, without calling your model |
| `s.stats()` | `{ table, teacher, fallback, offload, audits, auditDisagreements, auditDisagreement, optionsAdded, optionsRemoved }` | counters |
| `s.options()`, `s.threshold()` | | the question's options and certified threshold |
| `String(s)` | `string` | question, options, state, LLM and where its key comes from (masked) |

### Lower level

| Export | What it does |
|---|---|
| `configure({ llm, apiKey, apiKeyEnv, baseURL })` | process-wide defaults; `configure()` clears them; returns them with the key masked |
| `openaiTeacher({ options, model, baseURL, apiKey, apiKeyEnv, ... })` | `async (text) => option` for any OpenAI-compatible chat model: structured output with a plain-text fallback, retries with backoff, every reply mapped to exactly one option (or an error) |
| `systemoneTeacher({ options, model, baseURL, apiKey })` | the same, asking a System One server |
| `decisionsTeacher({ options, model, baseURL, apiKey, apiKeyEnv })` | the same, asking the OpenAI Decisions API |
| `Bundle.load(dirOrUrl)` | load a trained table |
| `bundle.decide(text, { questions, probabilities })` | `{ answers: { [name]: answer } }`, where each answer has `choice` (or `answer` for yes/no), `confidence`, `certified`, `flag` and `probabilities` (or `probability`). `probabilities: false` is the fast path: the same answer, confidence and certified flag without the per-option map. |
| `new Table(arrayBuffer, labels).decide(text)` | one `.s0` file; `{ index, choice, confidence, probabilities }` |
| `matchOption(reply, options)` | the reply-to-option mapping the teachers use |
| `explainFlag(flag)`, `FLAG_WORDS` | the reason behind a flag, in plain words |
| `maskKey(key)` | `"sk-…3f9a"` |

## What is Python-only

Training, the rollout settings (`mode`, `canary`, `never_serve`, `min_confidence`), the drift guard and exposed mode are Python-only. A table trained and certified in Python serves the same answers in JavaScript; its `certified` flag is based on the threshold alone.
