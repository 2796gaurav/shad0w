---
title: C & other languages
description: Decide in about a microsecond from C, C++, Go, Rust, Swift or anything with an FFI.
---
# C & other languages

Load a trained shad0w table from C and decide in about 1.0 µs. The core is one file, `reflex.c`, plus a header, `shad0w.h`. It needs only libc and libm. Use it from C, C++, or any language with a C FFI.

This page gives you the two files, a complete program, and the five functions. Training still happens in Python.

## TL;DR

```bash
D=$(python -c "import shad0w, os; print(os.path.dirname(shad0w.__file__))")/_native
cp $D/shad0w.h $D/reflex.c .
cc -O2 app.c reflex.c -lm -o app        # app.c below
```

## 1. Get the two files

They ship inside the Python package, in `shad0w/_native/`. The commands above copy them next to your code.

## 2. Read the manifest

A trained bundle is a folder: `manifest.json` plus one `<question>.s0` file per question. The C core reads the `.s0` file. Two values come from `manifest.json`, under `questions.<name>`:

| Manifest field | What it is | Used for |
|---|---|---|
| `options` | the option names, in table order | turning the index `k` into a name |
| `threshold` | the certified confidence threshold | the answer is certified when `probs[k] >= threshold`. `null` means the table certifies nothing: always ask your LLM. |
| `calibrated` | `false` when the table has no certificate | treat every answer as uncertified |
| `type` | `"choice"` or `"yesno"` | for yes/no, options are `["no", "yes"]`: `k == 1` means yes |

## 3. Decide

```c
#include <stdio.h>
#include <string.h>
#include "shad0w.h"

int main(void) {
  S0Reflex *m = s0_load("bundle/intent.s0");                               /* NULL if missing or corrupt */
  if (!m) return 1;
  const char *options[] = {"refund", "lost_card", "balance", "other"};     /* manifest.json "options", same order */
  const float threshold = 0.911f;                                          /* manifest.json "threshold" */

  const char *text = "my card was stolen";
  float probs[S0_MAX_CLASSES], radius;
  int k = s0_decide(m, text, (int)strlen(text), probs, &radius);
  if (probs[k] >= threshold) printf("%s (certified, %.3f)\n", options[k], probs[k]);
  else printf("ask the LLM (table guessed %s, %.3f)\n", options[k], probs[k]);

  s0_free(m);
  return 0;
}
```

```bash
cc -O2 app.c reflex.c -lm -o app && ./app
```

```text
lost_card (certified, 0.993)
```

Text is UTF-8; pass its length in bytes. The answers are identical to Python's and JavaScript's on the same table.

## Threads

A loaded table is read-only and can be shared across threads. Give each thread its own `probs` buffer.

## Functions

| Function | What it does |
|---|---|
| `S0Reflex *s0_load(const char *path)` | load a `.s0` table; `NULL` on any error (missing file, wrong version, truncated, unsorted keys) |
| `int s0_decide(m, text, len, probs, &radius)` | returns the index of the best option; writes every option's probability to `probs` (`s0_num_classes(m)` floats) and the robustness radius (how many inserted words it takes to flip the answer; `INFINITY` when none can) |
| `uint32_t s0_num_classes(m)` | the number of options |
| `double s0_bench(m, texts, lens, n, reps, per_call_ns)` | mean nanoseconds per decision over `reps` passes; `per_call_ns` may be `NULL` |
| `void s0_free(m)` | release the table |

`S0_MAX_CLASSES` (1024) is the largest number of options a table can have.

The C core does not apply the drift guard or the rollout settings (`mode`, `canary`, `never_serve`); those live in Python. For them, use the [HTTP server](http.html) or the [proxy](proxy.html).

## Other languages

| Language | How |
|---|---|
| Go | cgo: `// #include "shad0w.h"` and build `reflex.c` with the package |
| Rust | `bindgen`, or a five-line `extern "C"` block |
| Swift / Objective-C | add both files to the target |
| Java / Kotlin | JNI or Panama |
| Anything else | the [HTTP server](http.html), or the [proxy](proxy.html) for OpenAI-compatible clients |
