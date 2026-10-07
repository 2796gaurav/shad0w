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
  llm: "openai/gpt-4o-mini",          // or anthropic/…, gemini/…, groq/…, ollama/llama3.1, or baseURL for any OpenAI-compatible server
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
    intent ??= await decision("intent", { llm: "openai/gpt-4o-mini", apiKey: env.OPENAI_API_KEY, bundle: env.BUNDLE_URL,
                                          log: (row) => env.LOGS.send(row) });   // e.g. a Queue
    const { text } = await req.json();
    return Response.json(await intent.decide(text));
  },
};
```

## API

- `decision(name, { options, llm, bundle, log, auditRate, onDecision, baseURL, apiKey })` → a `Shadow`.
- `new Shadow(bundle, { teacher, question, log, auditRate, onDecision })`:
  - `await shadow.decide(text)` returns `{ answer, source, confidence, certified, flag, latencyUs }`;
  - `shadow.stats()` returns offload and spot-check counts.
- `openaiTeacher({ options, model, baseURL, apiKey })` → `async (text) => option`. It works with any OpenAI-compatible chat API, uses structured outputs with a plain-text fallback, and retries.
- `Bundle.load(dirOrUrl)`; `bundle.decide(text)` → `{ answers: { [question]: { choice | answer, confidence, certified, flag, probabilities } } }`.
- `new Table(arrayBuffer, labels).decide(text)` works on a single `.s0` file.

Decisions match the Python and C runtimes exactly; the test suite checks it.

Docs: https://2796gaurav.github.io/shad0w/docs/javascript.html · Source: https://github.com/2796gaurav/shad0w · Apache-2.0
