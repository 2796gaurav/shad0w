# Changelog

## 0.1.1

- **Certificate holds up under LLM-like teachers.** Teachers that are wrong a few percent of the time regardless of the input (sampling noise) could make the strict-to-lenient fixed-sequence test stop early. A 97.6%-agreeing teacher then certified 0%. The default procedure (`auto`) now runs fixed-sequence and a Bonferroni test over 12 pre-registered coverage levels, each at δ/2, and keeps the more lenient threshold. That is still valid at δ by the union bound. In simulation, offload under flat teacher noise rose from 39% to ~100%; where disagreement rises as confidence falls, ~98% of the fixed-sequence offload was kept. The violation rate stayed within δ for every procedure. The end-to-end case above now certifies 100% with 2.3% realised disagreement on fresh traffic (bound 5%).
- `SelectiveRiskController(alpha, delta, procedure="auto" | "fixed-sequence" | "bonferroni")`. The certificate records which procedure produced the threshold.
- Release pipeline: PyPI and npm both publish through Trusted Publishing (OIDC). No tokens are stored.

## 0.1.0

First public release.

- **Shadow mode.** `shad0w shadow` compiles a table from your model's logged answers and certifies how often it disagrees with that model (Learn-then-Test, Clopper–Pearson, fixed-sequence testing). `shad0w certify` re-certifies on fresh traffic, and `shad0w report` prints the certificate.
- **`shad0w.Shadow` wrapper and `@shad0w.cascade` decorator.** Put the table in front of your existing model call in one line. The wrapper has a log-only start (no bundle yet), logs deferred answers for the next compile, and runs live spot-check audits with a Clopper–Pearson bound in `stats()`.
- **Robust certificate start.** The test sequence starts where a run with disagreement at half of α can still certify. One unlucky disagreement among the most confident answers no longer certifies nothing. A simulation test checks that the violation rate stays ≤ δ.
- **Three runtimes, identical decisions.** Python (numpy only), a C core shipped inside the wheels, and a dependency-free JavaScript package (`npm i shad0wllm`, CJS + ESM + types, with `Bundle.load` applying the certified threshold).
- **Strict table loading** in all three runtimes. Unknown versions, impossible shapes, unsorted keys and truncated or padded files are refused.
- Typed answers with confidence, an exact robustness radius, `certified` and `flag`. Also a label-free drift guard and anytime-valid auditors.
- Human-label mode (`shad0w compile --data`, `shad0w calibrate`) and label-free mode (`shad0w compile --unlabeled`).
- Fast Python path: `probabilities=False` skips the per-option dict (about 8 µs per call through the C core, versus about 15 µs with it).
- Reference HTTP server: `POST /v1/decide` (alias `/v1/systemone`) and `GET /v1/health`.
- `shad0w init` and `shad0w --version`.
