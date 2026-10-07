# shad0w for JavaScript

Run [shad0w](https://github.com/2796gaurav/shad0w) tables in Node, browsers, Deno, Bun and edge workers. A table is a decision compiled from your LLM's own answers, plus a certificate of how often it disagrees with that LLM. Zero dependencies, ~7–11 µs per decision, and the same decisions as the Python and C runtimes (checked by the test suite).

```bash
npm i shad0w
```

```js
import { Bundle } from "shad0w";

const bundle = await Bundle.load("bundle/");          // Node: a directory. Browser/Worker: a URL.
const { answers } = bundle.decide("my card was stolen yesterday");
const a = answers.intent;

const intent = a.certified ? a.choice : await askMyLLM(text);   // certified: serve it; otherwise ask your model
```

Each answer has `choice` (or `answer` for yes/no questions), `confidence`, `probabilities`, `certified` and `flag` (`"low_confidence"` or `"uncalibrated"`). The certified threshold comes from the bundle's `manifest.json`. Compile and certify bundles with the Python package: `pip install "shad0w[compile]"`, then `shad0w shadow ...`.

**Cloudflare Worker**

```js
import { Bundle } from "shad0w";
let bundle;
export default {
  async fetch(req, env) {
    bundle ??= await Bundle.load(env.BUNDLE_URL);      // e.g. an R2 or static-assets URL
    const { text } = await req.json();
    return Response.json(bundle.decide(text));
  },
};
```

Lower level: `new Table(arrayBuffer, labels).decide(text)` returns `{index, choice, confidence, probabilities}` for a single `.s0` file.

Docs: https://2796gaurav.github.io/shad0w · License: Apache-2.0
