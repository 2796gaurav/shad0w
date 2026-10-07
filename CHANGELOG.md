# Changelog

## 0.1.0

First public release.

- **Shadow mode.** `shad0w shadow` compiles a table from your model's logged answers and certifies how often it disagrees with that model (Learn-then-Test, Clopper–Pearson, fixed-sequence testing). `shad0w certify` re-certifies on fresh traffic, and `shad0w report` prints the certificate.
- **`shad0w.Shadow` wrapper and `@shad0w.cascade` decorator.** Put the table in front of your existing model call in one line. The wrapper has a log-only start (no bundle yet), logs deferred answers for the next compile, and runs live spot-check audits with a Clopper–Pearson bound in `stats()`.
- **Robust certificate start.** The test sequence starts where a run with disagreement at half of α can still certify. One unlucky disagreement among the most confident answers no longer certifies nothing. A simulation test checks that the violation rate stays ≤ δ.
- **Three runtimes, identical decisions.** Python (numpy only), a C core shipped inside the wheels, and a dependency-free JavaScript package (`npm i shad0w`, CJS + ESM + types, with `Bundle.load` applying the certified threshold).
- **Strict table loading** in all three runtimes. Unknown versions, impossible shapes, unsorted keys and truncated or padded files are refused.
- Typed answers with confidence, an exact robustness radius, `certified` and `flag`. Also a label-free drift guard and anytime-valid auditors.
- Human-label mode (`shad0w compile --data`, `shad0w calibrate`) and label-free mode (`shad0w compile --unlabeled`).
- Reference HTTP server: `POST /v1/decide` (alias `/v1/systemone`) and `GET /v1/health`.
- `shad0w init` and `shad0w --version`.
