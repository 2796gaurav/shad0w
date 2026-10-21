# Decision Bench

A reproducible benchmark for fixed-option decisions (intent, topic, sentiment, spam, prompt injection) on a CPU only.
It puts shad0w next to the open decision models: Julia-1, Laya, Kev-0.8B, Decider-0.8B and GLiNER2.5-Decide. Every
model answers the same rows and goes through the same certifier.

Open `bench/site/index.html` in a browser for the full results: a chart and table per task.

## Results

<!-- bench:start -->
Measured on Apple M5 Pro, 2026-10-19. Decision models run zero-shot on the CPU only with 5 threads; shad0w runs on one thread and learns from each task's training split. Accuracy is against the true labels; certified = share of test rows served under the same 5% Learn-then-Test certificate.

| task | shad0w: accuracy / certified / p50 | most accurate decision model | fastest decision model | shad0w in front of the most accurate |
|---|---|---|---|---|
| ag_news | 91.2% / 66% / 10.1 µs | Laya: 93.8%, 124.4 ms | Julia-1: 82.8%, 50.1 ms | 94.2%, 2.9× faster |
| banking77 | 94.0% / 89% / 6.0 µs | Kev-0.8B: 83.3%, 2.6 s (150 rows) | Julia-1: 62.8%, 197.1 ms | 91.3%, 7.8× faster |
| clinc150 | 84.6% / 86% / 6.5 µs | Decider-0.8B: 83.3%, 5.6 s (150 rows) | Julia-1: 27.4%, 314.1 ms | 91.3%, 7.5× faster |
| prompt_injection | 98.8% / 100% / 2.4 µs | GLiNER2.5-Decide: 94.6%, 112.6 ms | Julia-1: 34.4%, 14.3 ms | 98.8%, 28,988× faster |
| sms_spam | 97.8% / 100% / 1.9 µs | Decider-0.8B: 95.3%, 3.6 s (150 rows) | Julia-1: 66.4%, 15.9 ms | 97.3%, 1,704,195× faster |
| sst2 | 83.2% / 62% / 3.0 µs | Laya: 91.6%, 49.2 ms | Julia-1: 76.8%, 15.5 ms | 92.4%, 2.5× faster |
<!-- bench:end -->

## What is measured

| | |
|---|---|
| Tasks | BANKING77 (77 intents), CLINC150 (150 intents + out-of-scope), SST-2, AG News, SMS spam, prompt injection (`bench/tasks.py`) |
| Rows | 500 test rows per task: a fixed random sample (seed 0) of the dataset's test split. 500 calibration rows: a fixed sample of its train split, held out from training. A decision model that takes over 1 s per decision on a task (timed on its first 5 rows) answers the first 150 of each. |
| Accuracy | against the dataset's true labels, with a 95% Wilson interval |
| Time per decision | wall time of one call with one input (batch size 1), from Python, after 20 warm-up calls. shad0w runs in-process on one thread. Decision models run on the CPU only (llama.cpp `llama-server --device none`, or their own Python package behind `py_server.py`), over HTTP on localhost, with the thread count in `results/hardware.json`; their single-thread time is measured separately on 50 rows. |
| Certified coverage | the same certifier for every system: Learn-then-Test with a Clopper–Pearson bound (`shad0w.reliability.SelectiveRiskController`, α = 5%, δ = 10%) picks a confidence threshold on the calibration rows. Coverage is the share of test rows at or above it; served error is the error on those rows. |
| Cascade | shad0w (`student="auto"`, the library default) answers what it certifies; the decision model answers the rest. |

**Systems**

- **shad0w**: `shad0w-auto` (the default), `shad0w-linear`, `shad0w-deep`. Each learns from the train split and is served
  by the C core.
- **Decision models, zero-shot, CPU only**:
  - Julia-1 144M, Laya 421M and Kev-0.8B, as GGUF builds from `ggml-org` on Hugging Face, run with llama.cpp.
  - Decider-0.8B (`Mapika/decider-0.8b`) and GLiNER2.5-Decide (`fastino/GLiNER2.5-Decide`), run by their own Python
    packages behind `bench/py_server.py`, which speaks the same `/v1/systemone` request shape.
  - Decider-0.8B runs on its package's reference PyTorch kernels on a CPU (the optimised kernels need a GPU), about
    5 s per decision. Its single-thread time was not measured.
  - Laya-multilingual, Laya-typed-decisions and d1-omni-600M did not load in this llama.cpp build, so they are left out.
  - Classic CPU classifiers (TF-IDF, potion-8M, MiniLM, bge-small, each with logistic regression) are available with
    `--systems tfidf-logreg,potion8m-logreg,minilm-logreg,bge-small-logreg`, but are not part of the default run. The 4B–9B models (Kev-4B, Lev, Clef-flash) take seconds per decision on a CPU. They are not in
  the default run, but `--systems kev-4b,lev,clef-flash` adds them, on the first 150 test rows and uncertified. Every model gets the same
  neutral instruction and each option's short description. The instruction was chosen on calibration rows, before any
  test row was run.

## What it does not show

- **Different training.** shad0w is trained on each task; the decision models are not. A decision
  model fine-tuned on the task would score higher than its zero-shot result here.
- **Labels as the teacher.** Here the teacher is the dataset's labels. In production shad0w learns from your LLM's
  answers, and its certificate is against that LLM, not the truth.
- **One machine, one run, batch size 1.** Decision-model times include localhost HTTP. They would be lower in-process and
  much lower on a GPU.

## Reproduce

```bash
pip install -e ".[compile]"                       # plus llama.cpp's llama-server on PATH
# Decider-0.8B and GLiNER2.5-Decide need Python >= 3.11 in a separate environment:
uv venv --python 3.12 .dm && VIRTUAL_ENV=.dm uv pip install "git+https://github.com/Mapika/decider" "gliner2[local]" torch
export SHAD0W_BENCH_DM_PYTHON=$PWD/.dm/bin/python
python bench/run.py      # every system (about two hours on a laptop CPU)
python bench/score.py    # predictions -> results/results.json, results/SUMMARY.md, the tables in both READMEs
python bench/render.py   # results.json -> site/index.html
```

- **Partial runs:** `--tasks sst2,banking77` and `--systems shad0w-auto,julia-1` limit a run to some tasks or systems.
- **Threads:** `--dm-threads N` changes the decision models' threads.
- **Re-running:** existing predictions are reused, so delete a file in `results/preds/` to re-run it.

## Check the numbers without running anything

Every answer of every system is saved in `results/preds/<task>/<system>.jsonl.gz`, one JSON line per row:

```json
{"split": "test", "i": 17, "y": 3, "pred": 3, "conf": 0.9981, "us": 2.41}
```

`python bench/score.py` recomputes every number from these files with no model, no network and no GPU.
`results/hardware.json` records the CPU, core counts, OS, library versions and date of the run.
